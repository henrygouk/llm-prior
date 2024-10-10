import argparse
from data import load_arff
from linear import BayesLogisticRegression
from bart import BART
from llm import DirectLLMSampler
import numpy as np
from openai import OpenAI
import os
import pickle
from scipy.stats import uniform
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split, ParameterSampler
from sklearn.preprocessing import StandardScaler

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
                sampler = DirectLLMSampler(meta_data, args.llm, args.base_url) #DirectLLMSampler(client, args.llm, meta_data)
            else:
                raise ValueError(f"Unknown LLM sampler: {args.llm_sampler}")

            K_X, K_py = sampler.sample(args.prior_samples)

            if args.prior_cache is not None:
                with open(args.prior_cache, "wb") as f:
                    pickle.dump((K_X, K_py), f)

    return meta_data, X, y, K_X, K_py

def create_model(meta_data, args):
    if args.model == "blr":
        return BayesLogisticRegression(
            tau=(args.blr_tau_min, args.blr_tau_max),
            gamma=(args.blr_gamma_min, args.blr_gamma_max),
            delta=(args.blr_delta_min, args.blr_delta_max),
            hpo_iter=args.blr_hpo_iter,
            num_classes=len(meta_data.target.values),
            nominal_features=[(i, len(f.values)) for i, f in enumerate(meta_data.features) if f.dtype == "str"],
        )
    elif args.model == "bart":
        return BART(
            n_trees=args.bart_n_trees,
            gamma=(0.5, 5.0),
            delta=(0.0, 5.0),
            hpo_iter=args.bart_hpo_iter,
            num_classes=len(meta_data.target.values),
            nominal_features=[(i, len(f.values)) for i, f in enumerate(meta_data.features) if f.dtype == "str"],
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

# Repeated holdout will give a better picture of true performance than CV in the few-shot setting
def evaluate_repeated_cv(X, y, K_X, K_py, model, args):
    num_classes = len(np.unique(y))
    print("rep,fold,num_train,roc_auc")

    rskf = RepeatedStratifiedKFold(n_splits=args.cv_folds, n_repeats=args.cv_reps, random_state=args.seed)

    for split_id, (train_index, test_index) in enumerate(rskf.split(X, y)):
        rep = int(split_id / args.cv_folds)
        fold = split_id % args.cv_folds

        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        for k in args.samples:
            if k == X_train.shape[0]:
                X_train_k = X_train
                y_train_k = y_train
            elif k == 0:
                X_train_k, y_train_k = np.zeros((0, X_train.shape[1])), np.zeros(0)
            elif k < X_train.shape[0] and num_classes <= k:
                X_train_k, _, y_train_k, _ = train_test_split(X_train, y_train, train_size=k, stratify=y_train, random_state=args.seed)
            else:
                raise ValueError("--samples cannot contain a value greater than the the number of classes or the number of training samples avialable during cross validation.")

            if args.prior_samples > 0:
                model.fit(X_train_k, y_train_k, K_X, K_py)
            else:
                model.fit(X_train_k, y_train_k)

            auc = compute_auc(model, X_test, y_test)
            print(f"{rep},{fold},{X_train_k.shape[0]},{auc}")

def evaluate_repeated_holdout(X, y, K_X, K_py, model, args):
    rng = np.random.default_rng(args.seed)

    print("rep,num_train,roc_auc")

    for i in range(args.ho_reps):
        for k in args.samples:
            if k == 0:
                X_train, y_train = np.zeros((0, X.shape[1])), np.zeros(0)
                X_test, y_test = X, y
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=k, stratify=y, random_state=rng.integers(0, 2**32))

            if K_X is not None:
                model.fit(X_train, y_train, K_X, K_py)
            else:
                model.fit(X_train, y_train)

            auc = compute_auc(model, X_test, y_test)
            print(f"{i},{k},{auc}")

def compute_auc(model, X, y):
    num_classes = len(np.unique(y))

    if num_classes == 2:
        return roc_auc_score(y, model.predict_proba(X)[:, 1])
    else:
        return roc_auc_score(y, model.predict_proba(X), multi_class="ovr")

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

    parser.add_argument("--eval-method", choices=["crossval", "holdout"], required=True)
    # Options for crossval
    parser.add_argument("--cv-folds", type=int, default=10)
    parser.add_argument("--cv-reps", type=int, default=5)
    # Options for holdout
    parser.add_argument("--ho-reps", type=int, default=10)

    parser.add_argument("--model", choices=["blr", "bart"], required=True)
    # Options for blr
    parser.add_argument("--blr-hpo-iter", type=int, default=20)
    parser.add_argument("--blr-tau-min", type=float, default=0.5)
    parser.add_argument("--blr-tau-max", type=float, default=5.0)
    parser.add_argument("--blr-gamma-min", type=float, default=0.5)
    parser.add_argument("--blr-gamma-max", type=float, default=5.0)
    parser.add_argument("--blr-delta-min", type=float, default=0.0)
    parser.add_argument("--blr-delta-max", type=float, default=5.0)
    # Options for bart
    parser.add_argument("--bart-hpo-iter", type=int, default=10)
    parser.add_argument("--bart-n-trees", type=int, default=50)

    args = parser.parse_args()

    # Load th data
    meta_data, X, y, K_X, K_py = load_data(args)

    # Create the model
    model = create_model(meta_data, args)

    # Evaluate the model
    if args.eval_method == "crossval":
        evaluate_repeated_cv(X, y, K_X, K_py, model, args)
    elif args.eval_method == "holdout":
        evaluate_repeated_holdout(X, y, K_X, K_py, model, args)
    else:
        raise ValueError(f"Unknown evaluation method: {args.eval_method}")

if __name__ == "__main__":
    main()
