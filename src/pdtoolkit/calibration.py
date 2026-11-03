"""Calibration learned on a later, disjoint window; never on final test labels."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression


def probability_array(p):
    result = np.asarray(p, dtype=float)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise ValueError("Probabilities must be a finite, non-empty one-dimensional array")
    if ((result < 0) | (result > 1)).any():
        raise ValueError("Probabilities must lie in [0, 1]")
    return result


def fit_logistic_curve(y, p, monotone=False):
    p = probability_array(p)
    y = np.asarray(y)
    if y.shape != p.shape or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("Curve fitting requires aligned binary outcomes with both classes")
    z = logit(np.clip(p, 1e-8, 1 - 1e-8))

    def objective(beta):
        eta = beta[0] + beta[1] * z
        residual = expit(eta) - y
        return np.mean(np.logaddexp(0, eta) - y * eta), np.array(
            [residual.mean(), (residual * z).mean()]
        )

    result = minimize(
        objective,
        [0.0, 1.0],
        jac=True,
        method="L-BFGS-B",
        bounds=[(None, None), (0, None)] if monotone else None,
    )
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError("Calibration optimisation did not converge")
    return result.x


@dataclass
class Calibrator:
    method: str
    estimator: object = None
    parameters: dict | None = None

    def fit(self, p, y):
        p = probability_array(p)
        if self.method == "sigmoid":
            self.estimator = fit_logistic_curve(y, p, monotone=True)
            self.parameters = dict(zip(["intercept", "slope"], self.estimator.tolist()))
        elif self.method == "isotonic":
            y = np.asarray(y)
            if y.shape != p.shape or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
                raise ValueError("Isotonic calibration requires aligned binary outcomes")
            self.estimator = IsotonicRegression(out_of_bounds="clip").fit(p, y)
            self.parameters = {"n_thresholds": len(self.estimator.X_thresholds_)}
        elif self.method == "none":
            self.parameters = {}
        else:
            raise ValueError("Unsupported calibration method")
        return self

    def predict(self, p):
        p = probability_array(p)
        if self.parameters is None:
            raise ValueError("Calibrator has not been fitted")
        if self.method == "sigmoid":
            return expit(self.estimator[0] + self.estimator[1] * logit(np.clip(p, 1e-8, 1 - 1e-8)))
        if self.method == "isotonic":
            return self.estimator.predict(p)
        return p.copy()
