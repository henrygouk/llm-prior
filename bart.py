import numpy as np
from numpy.core.fromnumeric import argmax
import pymc as pm
from pymc.backends.arviz import coords_and_dims_for_inferencedata
import pymc_bart as pmb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from pymc.model.fgraph import clone_model
from typing import Tuple
import arviz as az
import sys

class BART(BaseEstimator, ClassifierMixin):

    def __init__(self,
                 n_trees: int = 50,
                 gamma: float | Tuple[float, float] = (0.5, 5.0),
                 delta: float | Tuple[float, float] = (0.0, 5.0),
                 hpo_iter: int = 10,
                 num_classes: int | None = None,
                 nominal_features: list[Tuple[int, int]] = [],
                 use_dirichlet: bool = False,
                 seed: int = 42,
                 chains: int = 4,
                 tune: int = 1000,
                 draws: int = 1000,
                 cores: int | None = None):
        self.n_trees = n_trees
        self.gamma = gamma
        self.delta = delta
        self.hpo_iter = hpo_iter
        self.num_classes = num_classes
        self.nominal_features = nominal_features
        self.seed = seed
        self.chains = chains
        self.tune = tune
        self.draws = draws
        self.cores = cores
        self.use_dirichlet = use_dirichlet

    def _get_coords(self, X: np.ndarray, y: np.ndarray, K_X: np.ndarray | None = None, K_py: np.ndarray | None = None, num_classes: int | None = None) -> dict:
        if X.shape[0] == 0 and (K_X is None or (K_X is not None and K_X.shape[0] == 0)):
            raise ValueError("Both X and K_X cannot be empty.")

        num_trials = X.shape[0] if K_X is None else K_X.shape[0] + X.shape[0]
        num_features = X.shape[1] if X.shape[0] > 0 else K_X.shape[1]

        if num_classes is None:
            num_classes = len(np.unique(y))
        else:
            num_classes = num_classes

        coords = {
            "features": np.arange(num_features),
            "classes": np.arange(num_classes)
        }

        coords["trials"] = np.arange(num_trials)

        return coords

    def _one_hot(self, X):
        for i, c in self.nominal_features:
            X = np.hstack((X, np.eye(c)[X[:, i].astype(int)]))

        return np.delete(X, [i for i, _ in self.nominal_features], axis=1)

    def _pymc_model(self, X: np.ndarray, y: np.ndarray, K_X: np.ndarray | None, K_py: np.ndarray | None, num_classes: int | None, gamma: Tuple[float, float] | float, delta: Tuple[float, float] | float) -> pm.Model:
        coords = self._get_coords(X, y, K_X, K_py, num_classes)

        with pm.Model(coords=coords) as model:
            if K_X is not None and X.shape[0] > 0:
                X_all = np.concatenate((X, K_X), axis=0)
                y_all = np.concatenate((y, K_py.argmax(axis=1)), axis=0)
                mask = np.concatenate((np.ones(X.shape[0]), np.zeros(K_X.shape[0])), axis=0)
            elif K_X is not None:
                X_all = K_X
                y_all = K_py.argmax(axis=1)
                mask = np.zeros(K_X.shape[0])
            else:
                X_all = X
                y_all = y
                mask = np.ones(X.shape[0])

            X_sym = pm.Data("X", X_all, dims=["trials", "features"])
            mask_sym = pm.Data("mask", mask, dims=["trials"])

            z = pmb.BART("z", X_sym, y_all, m=self.n_trees)
            p_y = pm.Deterministic("p", pm.math.sigmoid(z))

            if self.use_dirichlet:
                if isinstance(gamma, tuple):
                    gamma = pm.Uniform("gamma", lower=gamma[0], upper=gamma[1])

                if isinstance(delta, tuple):
                    delta = pm.Uniform("delta", lower=delta[0], upper=delta[1])

                y_sym = pm.Bernoulli("y", p=p_y, observed=y_all, dims=["trials"]) * mask_sym
                p_y_mat = pm.math.stack((1 - p_y, p_y), axis=-1)
                y_k_sym = pm.Dirichlet("y_k", a=gamma + delta * p_y_mat, dims=["trials", "classes"]) * pm.math.stack((1 - mask_sym, 1 - mask_sym), axis=-1)
            else:
                y_sym = pm.Bernoulli("y", p=p_y, observed=y_all, dims=["trials"])

        return model

    def fit(self, X: np.ndarray, y: np.ndarray, K_X: np.ndarray | None = None, K_py: np.ndarray | None = None, progressbar: bool = False) -> BaseEstimator:
        # One-hot encode nominal features while leaving numeric features unchanged
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

        num_classes = self.num_classes
        coords = self._get_coords(X, y, K_X, K_py, num_classes)

        best_loo = float("-inf")
        best_model = None
        best_idata = None

        if self.hpo_iter == 0 or X.shape[0] == 0 or self.use_dirichlet == False:
            best_model = self._pymc_model(X, y, K_X, K_py, num_classes, self.gamma, self.delta)

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
                gamma = rng.uniform(self.gamma[0], self.gamma[1]) if isinstance(self.gamma, tuple) else self.gamma
                delta = rng.uniform(self.delta[0], self.delta[1]) if isinstance(self.delta, tuple) else self.delta

                print(f"Hyperparameter iteration {i+1}/{self.hpo_iter} with gamma={gamma}, delta={delta}", file=sys.stderr)

                inference_model = self._pymc_model(X, y, K_X, K_py, num_classes, gamma, delta)

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

    def predict_proba(self, X: np.ndarray, progressbar: bool = False) -> np.ndarray:
        if self.idata_ is None:
            raise ValueError("Model must be fit before making predictions.")

        probs = self._predict_impl(X, progressbar).mean(axis=(0, 1))
        probs = np.column_stack([1 - probs, probs])
        return probs

    def _predict_impl(self, X: np.ndarray, progressbar: bool = False) -> np.ndarray:
        if X.shape[0] > 0:
            X = self._one_hot(X)
            X = self.scaler.transform(X)

        with self.inference_model_:
            pm.set_data({"X": X, "mask": np.ones(X.shape[0])}, coords={
                "features": np.arange(X.shape[1]),
                "trials": np.arange(X.shape[0]),
                "classes": self.classes_
            })
            ppc = pm.sample_posterior_predictive(self.idata_, progressbar=progressbar, predictions=True, random_seed=self.seed)

        preds = ppc.predictions["y"].to_numpy()
        return preds

    def _loo(self, idata) -> float:
        return az.loo(idata, var_name="y").loo_i.mean().item()
