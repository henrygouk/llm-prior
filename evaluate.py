import argparse
from data import load_arff
from linear import BayesLogisticRegression
from llm import LLMSampler
import numpy as np
from openai import OpenAI
import os
import pickle
from sklearn.preprocessing import StandardScaler
import sys

def main():
    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--prior-cache", type=str, default=None)
    parser.add_argument("--prior-samples", type=int, default=0)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-reps", type=int, default=5)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--samples", nargs="+", type=int, default=[5, 10, 20, 40, 80, 120])
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Load the specified ARFF file
    meta_data, X, y = load_arff(args.data_path)
    num_classes = len(np.unique(y))

    # Get prior knowledge (in the form of synthetic data) from the LLM
    if args.prior_samples > 0:
        if args.prior_cache is not None and os.path.exists(args.prior_cache):
            with open(args.prior_cache, "rb") as f:
                K_X, K_y = pickle.load(f)
                K_X = K_X[:args.prior_samples]
                K_y = K_y[:args.prior_samples]
        else:
            client = OpenAI(api_key="none", base_url=args.base_url)
            sampler = LLMSampler(client, args.model, meta_data)
            K_X, K_y = sampler.sample(args.prior_samples)

            if args.prior_cache is not None:
                with open(args.prior_cache, "wb") as f:
                    pickle.dump((K_X, K_y), f)

        K_X = StandardScaler().fit_transform(K_X)

    model = BayesLogisticRegression(tau=args.tau, gamma=args.gamma, delta=args.delta, seed=args.seed)

    print("rep,fold,num_train,accuracy")

    # Use sklearn to generate repeated cross validation folds
    from sklearn.model_selection import RepeatedStratifiedKFold
    rskf = RepeatedStratifiedKFold(n_splits=args.cv_folds, n_repeats=args.cv_reps, random_state=args.seed)

    for split_id, (train_index, test_index) in enumerate(rskf.split(X, y)):
        rep = int(split_id / args.cv_folds)
        fold = split_id % args.cv_folds

        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        for k in args.samples:
            # Get k random samples
            indices = np.random.permutation(X_train.shape[0])
            X_train_k = X_train[indices[:k]]
            y_train_k = y_train[indices[:k]]

            scaler = StandardScaler()
            X_train_k = scaler.fit_transform(X_train_k)
            X_test_k = scaler.transform(X_test)

            # Train the model
            if args.prior_samples > 0:
                model.fit(X_train_k, y_train_k, K_X, K_y, num_classes=num_classes)
            else:
                model.fit(X_train_k, y_train_k, num_classes=num_classes)

            # Evaluate the model
            accuracy = model.score(X_test_k, y_test)
            print(f"{rep},{fold},{X_train_k.shape[0]},{accuracy}")

if __name__ == "__main__":
    main()
