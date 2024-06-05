import numpy as np
import pymc as pm
import scipy.stats

class BayesLogisticRegression:

    def __init__(self, tau: float = 1.0, gamma: float = 0.5, delta: float = 1.0):
        self.tau = tau
        self.gamma = gamma
        self.delta = delta

    def fit(self, X: np.ndarray, y: np.ndarray, K_X: np.ndarray | None = None, K_y: np.ndarray | None = None, progressbar: bool = False, num_classes: int = None):
        self.model = pm.Model()

        with self.model:
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

            self.model = pm.Model(coords=coords)

            with self.model:
                X_sym = pm.Data("X", X, dims=("trials", "features"))
                y_sym = pm.Data("y", y, dims=("trials"))

                beta = pm.Normal("beta", mu=0, tau=self.tau, shape=(self.num_features, self.num_classes))
                alpha = pm.Normal("alpha", mu=0, tau=self.tau, shape=self.num_classes)

                logits = pm.math.dot(X_sym, beta) + alpha
                y_obs = pm.Categorical("y_obs", p=pm.math.softmax(logits, axis=1), observed=y_sym)

                if K_X is not None and K_y is not None:
                    K_logits = pm.math.dot(K_X, beta) + alpha
                    a = self.gamma + self.delta * pm.math.softmax(K_logits, axis=1)
                    K_y_obs = pm.Dirichlet("K_y_obs", a=a, observed=K_y)

                self.trace = pm.sample(1000, tune=1000, progressbar=progressbar)

    def predict(self, X: np.ndarray, progressbar: bool = False) -> np.ndarray:
        # Sample from the posterior predictive
        with self.model:
            pm.set_data({"X": X, "y": np.zeros(X.shape[0], dtype=int)}, coords={
                "trials": np.arange(self.num_trials, self.num_trials + X.shape[0]),
                "features": np.arange(X.shape[1])
            })

            post_pred = pm.sample_posterior_predictive(self.trace, predictions=True, progressbar=progressbar).predictions["y_obs"]
            votes = post_pred.to_numpy().reshape((-1, X.shape[0])).T
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
