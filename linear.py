import numpy as np
import pymc as pm
import scipy.stats

class BayesLogisticRegression:

    def __init__(self, tau: float = 1.0, gamma: float = 0.5, delta: float = 1.0, seed: int = 42):
        self.tau = tau
        self.gamma = gamma
        self.delta = delta
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray, K_X: np.ndarray | None = None, K_py: np.ndarray | None = None, progressbar: bool = False, num_classes: int | None = None):
        self.num_trials = X.shape[0]
        self.num_features = X.shape[1]

        if num_classes is None:
            self.num_classes = len(np.unique(y))
        else:
            self.num_classes = num_classes

        coords = {
            "trials": np.arange(self.num_trials),
            "features": np.arange(self.num_features),
            "classes": np.arange(self.num_classes)
        }

        if K_X is not None and K_py is not None:
            coords["prior_trials"] = np.arange(K_X.shape[0])

        with pm.Model(coords=coords) as generative_model:
            beta = pm.Normal("beta", mu=0, tau=self.tau, dims=["features", "classes"])
            alpha = pm.Normal("alpha", mu=0, tau=self.tau, dims="classes")

            X_sym = pm.Data("X", X, dims=["trials", "features"])
            logits = X_sym @ beta + alpha
            y_sym = pm.Categorical("y", logit_p=logits, dims="trials")

            if K_X is not None and K_py is not None:
                K_X_sym = pm.Data("K_X", K_X, dims=["prior_trials", "features"])
                K_logits = K_X_sym @ beta + alpha
                K_py_sym = pm.Dirichlet("K_py", a=self.gamma + self.delta * pm.math.softmax(K_logits, axis=1), dims=["prior_trials", "classes"])

        if K_X is not None and K_py is not None:
            with pm.observe(generative_model, {"y": y, "K_py": K_py}) as inference_model:
                idata = pm.sample(random_seed=self.seed, progressbar=progressbar)
        else:
            with pm.observe(generative_model, {"y": y}) as inference_model:
                idata = pm.sample(random_seed=self.seed, progressbar=progressbar)

        self.inference_model = inference_model
        self.idata = idata
        self.coords = coords

    def predict(self, X: np.ndarray, progressbar: bool = False) -> np.ndarray:
        # Sample from the posterior predictive
        with self.inference_model:
            pm.set_data({"X": X}, coords=self.coords | {
                "trials": np.arange(0, X.shape[0]),
            })
            pps = pm.sample_posterior_predictive(
                self.idata,
                predictions=True,
                extend_inferencedata=True,
                random_seed=self.seed,
                progressbar=progressbar
            )
            predictions = pps.predictions["y"]
            votes = predictions.to_numpy().reshape((-1, X.shape[0])).T
            return scipy.stats.mode(votes, axis=1).mode

    def score(self, X: np.ndarray, y: np.ndarray, progressbar: bool = False) -> float:
        y_hat = self.predict(X, progressbar=progressbar)
        return np.mean(y_hat == y)

def test_bayes_logistic_regression():
    X = np.random.randn(100, 2)
    beta = np.array([[1.0, -1.0], [-1.0, 1.0]])
    alpha = np.array([0.0, 0.0])
    y = np.argmax(np.dot(X, beta) + alpha, axis=1)

    blr = BayesLogisticRegression()
    blr.fit(X, y)
    print(blr.predict(X))
    print(y)

if __name__ == "__main__":
    test_bayes_logistic_regression()
