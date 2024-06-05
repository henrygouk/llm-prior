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

    args = parser.parse_args()

    # Load the specified ARFF file
    meta_data, X, y = load_arff(args.data_path)
    num_classes = len(np.unique(y))

    # Get prior knowledge (in the form of synthetic data) from the LLM
    if args.prior_samples > 0:
        if args.prior_cache is not None and os.path.exists(args.prior_cache):
            with open(args.prior_cache, "rb") as f:
                K_X, K_y = pickle.load(f)
        else:
            client = OpenAI(api_key="none", base_url=args.base_url)
            sampler = LLMSampler(client, args.model, meta_data)
            K_X, K_y = sampler.sample(args.prior_samples)

            if args.prior_cache is not None:
                with open(args.prior_cache, "wb") as f:
                    pickle.dump((K_X, K_y), f)

        K_X = StandardScaler().fit_transform(K_X)


    model = BayesLogisticRegression()

    print("rep,fold,num_train,accuracy")

    for rep in range(args.cv_reps):
        ## Evaluate the accuracy of the model with cross validation
        # Start by randomising the order of X, y
        indices = np.random.permutation(X.shape[0])
        X = X[indices]
        y = y[indices]

        for i in range(args.cv_folds):
            # Split the data into training and test sets
            start = i * X.shape[0] // args.cv_folds
            end = (i + 1) * X.shape[0] // args.cv_folds
            X_train = np.concatenate((X[:start], X[end:]))
            y_train = np.concatenate((y[:start], y[end:]))
            X_test = X[start:end]
            y_test = y[start:end]

            for k in [5, 10, 20, 40, 80]:
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
                print(f"{rep},{i},{k},{accuracy}")

if __name__ == "__main__":
    main()
