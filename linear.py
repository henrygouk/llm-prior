from typing import Tuple
import arviz as az
import numpy as np
import pymc as pm
import scipy.stats
from pymc.model.fgraph import clone_model
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
import sys

class BayesLogisticRegression(BaseEstimator, ClassifierMixin):

    def __init__(self,
                 tau: float | Tuple[float, float] = (0.5, 5.0),
                 gamma: float | Tuple[float, float] =(0.5, 5.0),
                 delta: float | Tuple[float, float] = (0.0, 5.0),
                 hpo_iter: int = 20,
                 use_dirichlet: bool = True,
                 num_classes: int | None = None,
                 nominal_features: list[Tuple[int, int]] = [],
                 seed: int = 42,
                 chains: int = 4,
                 tune: int = 1000,
                 draws: int = 1000,
                 cores: int | None = None):
        self.tau = tau
        self.gamma = gamma
        self.delta = delta
        self.hpo_iter = hpo_iter
        self.use_dirichlet = use_dirichlet
        self.num_classes = num_classes
        self.nominal_features = nominal_features
        self.seed = seed
        self.chains = chains
        self.tune = tune
        self.draws = draws
        self.cores = cores

    def _get_coords(self, X: np.ndarray, y: np.ndarray, K_X: np.ndarray | None = None, K_py: np.ndarray | None = None, num_classes: int | None = None) -> dict:
        if X.shape[0] == 0 and (K_X is None or (K_X is not None and K_X.shape[0] == 0)):
            raise ValueError("Both X and K_X cannot be empty.")

        num_trials = X.shape[0]
        num_features = X.shape[1] if X.shape[0] > 0 else K_X.shape[1]

        if num_classes is None:
            num_classes = len(np.unique(y))
        else:
            num_classes = num_classes

        coords = {
            "features": np.arange(num_features),
            "classes": np.arange(num_classes)
        }

        if X.shape[0] > 0:
            coords["trials"] = np.arange(num_trials)

        return coords

    def _pymc_model(self,
                    X: np.ndarray,
                    y: np.ndarray,
                    K_X: np.ndarray | None,
                    K_py: np.ndarray | None,
                    num_classes: int | None,
                    tau: float | Tuple[float, float],
                    gamma: float | Tuple[float, float],
                    delta: float | Tuple[float, float]
                ) -> pm.Model:
        coords = self._get_coords(X, y, K_X, K_py, num_classes)

        if K_X is not None and K_py is not None:
            coords["prior_trials"] = np.arange(K_X.shape[0])
 
        with pm.Model(coords=coords) as generative_model:
            if isinstance(tau, tuple):
                tau = pm.Uniform("tau", lower=tau[0], upper=tau[1])

            beta = pm.Normal("beta", mu=0, tau=tau, dims=["features", "classes"])
            alpha = pm.Normal("alpha", mu=0, tau=tau, dims="classes")

            if X.shape[0] > 0:
                X_sym = pm.Data("X", X, dims=["trials", "features"])
                logits = X_sym @ beta + alpha
                y_sym = pm.Categorical("y", logit_p=logits, dims="trials")

            if K_X is not None and K_py is not None:
                if isinstance(gamma, tuple):
                    gamma = pm.Uniform("gamma", lower=gamma[0], upper=gamma[1])

                if isinstance(delta, tuple):
                    delta = pm.Uniform("delta", lower=delta[0], upper=delta[1])

                K_X_sym = pm.Data("K_X", K_X, dims=["prior_trials", "features"])
                K_logits = K_X_sym @ beta + alpha

                if self.use_dirichlet:
                    K_py_sym = pm.Dirichlet("K_py", a=gamma + delta * pm.math.softmax(K_logits, axis=1), dims=["prior_trials", "classes"])
                else:
                    K_py_sym = pm.Categorical("K_py", logit_p=K_logits, dims="prior_trials")

        if K_X is not None and K_py is not None:
            if self.use_dirichlet:
                generative_model = pm.observe(generative_model, {"K_py": K_py})
            else:
                generative_model = pm.observe(generative_model, {"K_py": K_py.argmax(axis=1)})

        if X.shape[0] > 0:
            generative_model = pm.observe(generative_model, {"y": y})

        return generative_model

    def _one_hot(self, X):
        for i, c in self.nominal_features:
            X = np.hstack((X, np.eye(c)[X[:, i].astype(int)]))

        return np.delete(X, [i for i, _ in self.nominal_features], axis=1)

    def fit(self, X: np.ndarray, y: np.ndarray, K_X: np.ndarray | None = None, K_py: np.ndarray | None = None, progressbar: bool = False) -> BaseEstimator:
        num_classes = self.num_classes
        coords = self._get_coords(X, y, K_X, K_py, num_classes)

        # One-hot encode nominal features while leaving numeric features unchanged
        if X.shape[0] > 0:
            X = self._one_hot(X)
        if K_X is not None:
            K_X = self._one_hot(K_X)

        self.scaler = StandardScaler()

        if X.shape[0] > 0:
            self.scaler.partial_fit(X)

        if K_X is not None:
            self.scaler.partial_fit(K_X)
            K_X = self.scaler.transform(K_X)

        if X.shape[0] > 0:
            X = self.scaler.transform(X)

        best_loo = float("-inf")
        best_model = None
        best_idata = None

        if X.shape[0] == 0:
            best_model = self._pymc_model(X, y, K_X, K_py, num_classes, self.tau, self.gamma, self.delta)

            with best_model:
                best_idata = pm.sample(
                    random_seed=self.seed,
                    progressbar=progressbar,
                    chains=self.chains,
                    tune=self.tune,
                    draws=self.draws,
                    cores=self.cores
                )
        else:
            # Get a seeded rng
            rng = np.random.default_rng(self.seed)

            for i in range(self.hpo_iter):
                tau = rng.uniform(self.tau[0], self.tau[1]) if isinstance(self.tau, tuple) else self.tau
                gamma = rng.uniform(self.gamma[0], self.gamma[1]) if isinstance(self.gamma, tuple) else self.gamma
                delta = rng.uniform(self.delta[0], self.delta[1]) if isinstance(self.delta, tuple) else self.delta

                print(f"Hyperparameter iteration {i+1}/{self.hpo_iter} with tau={tau}, gamma={gamma}, delta={delta}", file=sys.stderr)

                inference_model = self._pymc_model(X, y, K_X, K_py, num_classes, tau, gamma, delta)

                with inference_model:
                    idata = pm.sample(
                        random_seed=self.seed,
                        progressbar=progressbar,
                        chains=self.chains,
                        tune=self.tune,
                        draws=self.draws,
                        cores=self.cores,
                        idata_kwargs={"log_likelihood": True}
                    )

                    loo = self._loo(idata)

                    print(f"LOO: {loo}", file=sys.stderr)

                    if loo > best_loo:
                        best_loo = loo
                        best_model = clone_model(inference_model)
                        best_idata = idata

        self.inference_model_ = best_model
        self.idata_ = best_idata
        self.coords_ = coords

        if num_classes is not None:
            self.classes_ = np.arange(num_classes)
        elif y.shape[0] > 0:
            self.classes_ = np.unique(y)
        else:
            self.classes_ = np.arange(K_py.shape[1])

        return self

    def predict(self, X: np.ndarray, progressbar: bool = False) -> np.ndarray:
        if self.idata_ is None:
            raise ValueError("Model must be fit before making predictions.")

        votes = self._predict_impl(X, progressbar)
        return scipy.stats.mode(votes, axis=1).mode

    def predict_proba(self, X: np.ndarray, progressbar: bool = False) -> np.ndarray:
        if self.idata_ is None:
            raise ValueError("Model must be fit before making predictions.")

        votes = self._predict_impl(X, progressbar)
        probs = np.zeros((X.shape[0], self.classes_.shape[0]))

        for i in range(X.shape[0]):
            for j in range(self.classes_.shape[0]):
                probs[i, j] = np.mean(votes[:, i] == j)

        return probs

    def _predict_impl(self, X: np.ndarray, progressbar: bool = False) -> np.ndarray:
        X = self._one_hot(X)
        X = self.scaler.transform(X)
        beta = self.idata_.posterior["beta"].to_numpy().reshape((-1, X.shape[1], self.classes_.shape[0]))
        alpha = self.idata_.posterior["alpha"].to_numpy().reshape((-1, 1, self.classes_.shape[0]))
        logits = np.matmul(X, beta) + alpha
        preds = np.argmax(logits, axis=2)
        return preds

    def _loo(self, idata) -> float:
        return az.loo(idata, var_name="y").loo_i.mean().item()

