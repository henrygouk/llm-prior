import os
os.environ['OPENBLAS_NUM_THREADS'] = '1' 
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import argparse
from data import load_arff
from llm import DirectLLMSampler
import numpy as np
import sys
import pickle
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_halving_search_cv #noqa
from sklearn.model_selection import HalvingRandomSearchCV
from scipy.stats import uniform

from supervised import BNNClassifier, BLRClassifier

def load_data(args):
    meta_data, X, y = load_arff(args.data_path)

    if args.prior_samples == 0:
        K_X = None
        K_py = None
    else:
        if args.prior_cache is not None and os.path.exists(args.prior_cache):
            with open(args.prior_cache, "rb") as f:
                K_X, K_py = pickle.load(f)
                K_X = K_X[:args.prior_samples]
                K_py = K_py[:args.prior_samples]
        else:
            if args.llm_sampler == "direct":
                sampler = DirectLLMSampler(meta_data, args.llm, args.base_url, use_random_features=args.use_rand_features) #DirectLLMSampler(client, args.llm, meta_data)
            else:
                raise ValueError(f"Unknown LLM sampler: {args.llm_sampler}")

            K_X, K_py = sampler.sample(args.prior_samples)

            if args.prior_cache is not None:
                with open(args.prior_cache, "wb") as f:
                    pickle.dump((K_X, K_py), f)

    return meta_data, X, y, K_X, K_py

def create_model(meta_data, args):
    if args.model == "blr":
        return BLRClassifier(
            tau=(args.tau_min, args.tau_max),
            gamma=(args.gamma_min, args.gamma_max),
            delta=(args.delta_min, args.delta_max),
            nominal_features=[(i, len(f.values)) for i, f in enumerate(meta_data.features) if f.dtype == "str"],
            n_classes=len(meta_data.target.values)
        )
    elif args.model == "bnn":
        return BNNClassifier(
            tau=(args.tau_min, args.tau_max),
            gamma=(args.gamma_min, args.gamma_max),
            delta=(args.delta_min, args.delta_max),
            nominal_features=[(i, len(f.values)) for i, f in enumerate(meta_data.features) if f.dtype == "str"],
            n_classes=len(meta_data.target.values)
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

def evaluate_repeated_holdout(X, y, K_X, K_py, base_model, args):
    rng = np.random.default_rng(args.seed)
    rep_size = args.ho_max_test_size + max(args.samples)

    if not args.no_header:
        print("rep,num_prior,num_train,roc_auc")

    for i in range(args.ho_reps):
        X_rep, _, y_rep, _ = train_test_split(X, y, train_size=rep_size, stratify=y, random_state=rng.integers(0, 2**32))
        for k in args.samples:
            if k == 0:
                X_train, y_train = np.zeros((0, X_rep.shape[1])), np.zeros(0)
                X_test, y_test = X_rep[:args.ho_max_test_size], y_rep[:args.ho_max_test_size]
            else:
                X_train, X_test, y_train, y_test = train_test_split(X_rep, y_rep, train_size=k, stratify=y_rep, random_state=rng.integers(0, 2**32))

            if k > base_model.n_classes * 4:
                model = HalvingRandomSearchCV(
                    base_model,
                    param_distributions={
                        "tau": uniform(args.tau_min, args.tau_max - args.tau_min),
                        "gamma": uniform(args.gamma_min, args.gamma_max - args.gamma_min),
                        "delta": uniform(args.delta_min, args.delta_max - args.delta_min)
                    },
                    resource="n_iter",
                    max_resources=2000,
                    min_resources=125,
                    random_state=rng.integers(0, 2**32),
                    cv=4,
                    factor=2,
                    n_jobs=1
                )
            else:
                model = base_model

            try:
                if K_X is not None:
                    model.fit(X_train, y_train, K_X, K_py, progress=args.progress)
                else:
                    model.fit(X_train, y_train, progress=args.progress)

                auc = model.score(X_test, y_test)
                print(f"{i},{args.prior_samples},{k},{auc}")
            except Exception as e:
                # Print to stderr
                print(e, file=sys.stderr)
                print(f"{i},{args.prior_samples},{k},nan")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--llm", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--prior-cache", type=str, default=None)
    parser.add_argument("--prior-samples", type=int, default=0)
    parser.add_argument("--llm-sampler", type=str, default="direct")
    parser.add_argument("--samples", nargs="+", type=int, default=[4, 8, 16, 32, 64, 128])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--no-header", action="store_true")
    parser.add_argument("--use-rand-features", action="store_true")

    parser.add_argument("--eval-method", choices=["holdout"], required=True)
    # Options for holdout
    parser.add_argument("--ho-reps", type=int, default=50)
    parser.add_argument("--ho-max-test-size", type=int, default=500)

    parser.add_argument("--model", choices=["blr", "bnn"], required=True)
    parser.add_argument("--tau-min", type=float, default=0.5)
    parser.add_argument("--tau-max", type=float, default=3.0)
    parser.add_argument("--gamma-min", type=float, default=0.5)
    parser.add_argument("--gamma-max", type=float, default=5.0)
    parser.add_argument("--delta-min", type=float, default=0.0)
    parser.add_argument("--delta-max", type=float, default=5.0)

    args = parser.parse_args()

    # Load th data
    meta_data, X, y, K_X, K_py = load_data(args)

    # Create the model
    model = create_model(meta_data, args)

    # Evaluate the model
    if args.eval_method == "holdout":
        evaluate_repeated_holdout(X, y, K_X, K_py, model, args)
    else:
        raise ValueError(f"Unknown evaluation method: {args.eval_method}")

if __name__ == "__main__":
    main()
