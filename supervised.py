from abc import ABC, abstractmethod
import math
import numpy as np
import pymc as pm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

class SciKitPyMC(ABC, BaseEstimator, ClassifierMixin):
    def __init__(self, gamma=(0.5, 5.0), delta=(0.0, 5.0), use_uncertainty=True, n_iter=2_000, nominal_features=[], n_classes=2, nuts_sampler="pymc", K_X=None, K_y=None):
        self.nominal_features = nominal_features
        self.gamma = gamma
        self.delta = delta
        self.use_uncertainty = use_uncertainty
        self.n_iter = n_iter
        self.n_classes = n_classes
        self.classes_ = np.arange(n_classes)
        self.nuts_sampler = nuts_sampler
        self.K_X = K_X
        self.K_y = K_y

    def get_params(self, deep=True):
        return {
            "gamma": self.gamma,
            "delta": self.delta,
            "use_uncertainty": self.use_uncertainty,
            "n_iter": self.n_iter,
            "nominal_features": self.nominal_features,
            "n_classes": self.n_classes
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def _get_coords(self, X, y, K_X, k_y):
        coords = {
            "classes": np.arange(len(self.classes_))
        }

        if X.shape[0] > 0:
            coords["trials"] = np.arange(X.shape[0])
            coords["features"] = np.arange(X.shape[1])

        if K_X is not None:
            coords["prior_trials"] = np.arange(K_X.shape[0])
            coords["features"] = np.arange(K_X.shape[1])

        return coords

    def _one_hot(self, X):
        for i, c in self.nominal_features:
            X = np.hstack((X, np.eye(c)[X[:, i].astype(int)]))

        return np.delete(X, [i for i, _ in self.nominal_features], axis=1)

    def _fit_scaler(self, X, K_X):
        scaler = StandardScaler()

        if X.shape[0] > 0:
            scaler.partial_fit(X)

        if K_X is not None:
            scaler.partial_fit(K_X)

        return scaler

    @abstractmethod
    def _create_pymc_model(self, X, y, K_X, K_y):
        pass

    def fit(self, X, y, progress=False):
        K_X = self.K_X
        K_y = self.K_y
        if (K_X is None) != (K_y is None):
            raise ValueError("K_X and K_y must be both None or both not None")
    
        if X.shape[0] > 0:
            X = self._one_hot(X)

        if K_X is not None:
            K_X = self._one_hot(K_X)

        self.scaler_ = self._fit_scaler(X, K_X)

        if X.shape[0] > 0:
            X = self.scaler_.transform(X)
        
        if K_X is not None:
            K_X = self.scaler_.transform(K_X)

        self.pymc_model_ = self._create_pymc_model(X, y, K_X, K_y)

        with self.pymc_model_:
            self.idata_ = pm.sample(self.n_iter, tune=self.n_iter, progressbar=progress, nuts_sampler=self.nuts_sampler)

    def predict(self, X, progress=False):
        probs = self.predict_proba(X, progress)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X, progress=False):
        if self.pymc_model_ is None:
            raise ValueError("Model has not been trained yet")

        X = self._one_hot(X)
        X = self.scaler_.transform(X)

        if "trials" not in self.pymc_model_.coords:
            return self._predict_proba_zero_shot(X, progress)

        with self.pymc_model_:
            pm.set_data({"X_data": X, "y_data": np.zeros((X.shape[0]), dtype="int32")}, coords={"trials": np.arange(X.shape[0])})
            ppc = pm.sample_posterior_predictive(self.idata_.posterior, var_names=["y"], progressbar=progress)
        
        y_hat = ppc.posterior_predictive["y"].to_numpy().reshape(-1, X.shape[0])
        y_hat = np.apply_along_axis(lambda x: np.bincount(x, minlength=len(self.classes_)), axis=0, arr=y_hat)
        y_hat = y_hat / y_hat.sum(axis=0)
        return y_hat.transpose()

    def _predict_proba_zero_shot(self, X, progress=False):
        if "prior_trials" not in self.pymc_model_.coords:
            raise ValueError("Model has not been trained with prior samples")

        with self.pymc_model_:
            dummy_y = np.zeros((X.shape[0], len(self.classes_))) if self.use_uncertainty else np.zeros(X.shape[0], dtype="int32")
            pm.set_data({"K_X_data": X, "K_y_data": dummy_y}, coords={"prior_trials": np.arange(X.shape[0])})
            ppc = pm.sample_posterior_predictive(self.idata_.posterior, var_names=["K_y"], progressbar=progress)
        
        y_hat = ppc.posterior_predictive["K_y"].to_numpy()

        if self.use_uncertainty:
            return y_hat.mean(axis=(0, 1))
        else:
            y_hat = y_hat.reshape(-1, X.shape[0])
            y_hat = np.apply_along_axis(lambda x: np.bincount(x, minlength=len(self.classes_)), axis=0, arr=y_hat)
            y_hat = y_hat / y_hat.sum(axis=0)
            return y_hat.transpose()

    def score(self, X, y, progress=False):
        if len(self.classes_) == 2:
            return roc_auc_score(y, self.predict_proba(X, progress=progress)[:, 1])
        else:
            return roc_auc_score(y, self.predict_proba(X, progress=progress), multi_class="ovr")


class BNNClassifier(SciKitPyMC):
    def __init__(self, hidden_size=100, tau=(0.5, 2.0), **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.tau = tau

    def get_params(self, deep=True):
        return {
            **super().get_params(),
            "tau": self.tau,
            "hidden_size": self.hidden_size
        }

    def set_params(self, **params):
        super().set_params(**params)
        return self

    def _create_pymc_model(self, X, y, K_X, K_y):
        coords = self._get_coords(X, y, K_X, K_y)
        coords["hidden"] = np.arange(self.hidden_size)

        with pm.Model(coords=coords) as nn:
            if isinstance(self.tau, tuple):
                tau = pm.Uniform("tau", lower=self.tau[0], upper=self.tau[1])
            else:
                tau = self.tau

            w_1 = pm.Normal("w_1", 0.0, tau, dims=("features", "hidden"))
            b_1 = pm.Normal("b_1", 0.0, tau, dims="hidden")

            w_2 = pm.Normal("w_2", 0.0, tau, dims=("hidden", "classes"))
            b_2 = pm.Normal("b_2", 0.0, tau, dims="classes")

            if X.shape[0] > 0:
                X_data = pm.Data("X_data", X, dims=("trials", "features"))
                y_data = pm.Data("y_data", y, dims=("trials"))

                z_1 = pm.math.dot(X_data, w_1) + b_1
                a_1 = pm.math.tanh(z_1)
                
                z_2 = pm.math.dot(a_1, w_2) + b_2
                a_2 = pm.math.softmax(z_2, axis=1)

                y_sym = pm.Categorical("y", a_2, observed=y_data, total_size=X.shape[0])

            if K_X is not None:
                if isinstance(self.gamma, tuple):
                    gamma = pm.Uniform("gamma", lower=self.gamma[0], upper=self.gamma[1])
                else:
                    gamma = self.gamma

                if isinstance(self.delta, tuple):
                    delta = pm.Uniform("delta", lower=self.delta[0], upper=self.delta[1])
                else:
                    delta = self.delta

                K_X_data = pm.Data("K_X_data", K_X, dims=("prior_trials", "features"))

                zk_1 = pm.math.dot(K_X_data, w_1) + b_1
                ak_1 = pm.math.tanh(zk_1)

                zk_2 = pm.math.dot(ak_1, w_2) + b_2
                ak_2 = pm.math.softmax(zk_2, axis=1)

                if self.use_uncertainty:
                    K_y_data = pm.Data("K_y_data", K_y, dims=("prior_trials", "classes"))
                    K_y = pm.Dirichlet("K_y", a=gamma + delta * ak_2, observed=K_y_data, total_size=K_y.shape[0])
                else:
                    K_y_data = pm.Data("K_y_data", K_y, dims=("prior_trials"))
                    K_y = pm.Categorical("K_y", ak_2, observed=K_y_data, total_size=K_y.shape[0])

        return nn

class BLRClassifier(SciKitPyMC):
    def __init__(self, tau=(0.5, 2.0), **kwargs):
        super().__init__(**kwargs)
        self.tau = tau

    def get_params(self, deep=True):
        return {
            **super().get_params(),
            "tau": self.tau
        }

    def set_params(self, **params):
        super().set_params(**params)
        return self
    
    def _create_pymc_model(self, X, y, K_X, K_y):
        coords = self._get_coords(X, y, K_X, K_y)

        with pm.Model(coords=coords) as blr:
            if isinstance(self.tau, tuple):
                tau = pm.Uniform("tau", lower=self.tau[0], upper=self.tau[1])
            else:
                tau = self.tau

            beta = pm.Normal("beta", 0.0, tau, dims=("features", "classes"))
            alpha = pm.Normal("alpha", 0.0, tau, dims="classes")

            if X.shape[0] > 0:
                X_data = pm.Data("X_data", X, dims=("trials", "features"))
                y_data = pm.Data("y_data", y, dims=("trials"))

                logits = pm.math.dot(X_data, beta) + alpha
                probs = pm.math.softmax(logits, axis=1)

                y_sym = pm.Categorical("y", probs, observed=y_data, total_size=X.shape[0])

            if K_X is not None:
                if isinstance(self.gamma, tuple):
                    gamma = pm.Uniform("gamma", lower=self.gamma[0], upper=self.gamma[1])
                else:
                    gamma = self.gamma

                if isinstance(self.delta, tuple):
                    delta = pm.Uniform("delta", lower=self.delta[0], upper=self.delta[1])
                else:
                    delta = self.delta

                K_X_data = pm.Data("K_X_data", K_X, dims=("prior_trials", "features"))
                K_logits = pm.math.dot(K_X_data, beta) + alpha
                K_probs = pm.math.softmax(K_logits, axis=1)

                if self.use_uncertainty:
                    K_y_data = pm.Data("K_y_data", K_y, dims=("prior_trials", "classes"))
                    K_y = pm.Dirichlet("K_y", a=gamma + delta * K_probs, observed=K_y_data, total_size=K_y.shape[0])
                else:
                    K_y_data = pm.Data("K_y_data", K_y, dims=("prior_trials"))
                    K_y = pm.Categorical("K_y", K_probs, observed=K_y_data, total_size=K_y.shape[0])

        return blr


def test_bnn_classifier(X_train, y_train, K_X, K_y, X_test, y_test):
    bnn = BNNClassifier(tau=1.0, K_X=K_X, K_y=K_y)
    bnn.fit(X_train, y_train, progress=True)
    y_hat = bnn.predict(X_test, progress=True)
    print((y_hat == y_test.astype(int)).mean())

def test_blr_classifier(X_train, y_train, K_X, K_y, X_test, y_test):
    blr = BLRClassifier(tau=1.0, K_X=K_X, K_y=K_y)
    blr.fit(X_train, y_train, progress=True)
    y_hat = blr.predict(X_test, progress=True)
    print((y_hat == y_test.astype(int)).mean())

if __name__ == "__main__":
    X = np.random.randn(100, 4)
    y_mask = X[:, 0] + X[:, 1] + X[:, 2] + X[:, 3] > 0
    y = np.zeros(100)
    y[y_mask] = 1

    X_train = X[0:25]
    y_train = y[0:25]
    K_X = X[25:40]
    K_y = np.column_stack((0.95 - 0.9 * y[25:40], 0.05 + 0.9 * y[25:40]))
    X_test = X[40:]
    y_test = y[40:]

    test_bnn_classifier(X_train, y_train, K_X, K_y, X_test, y_test)
    test_blr_classifier(X_train, y_train, K_X, K_y, X_test, y_test)
