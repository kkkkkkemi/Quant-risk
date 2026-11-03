"""Validated binary metrics and uncertainty conditional on the synthetic portfolio."""

import numpy as np
import pandas as pd
from scipy.special import ndtri
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .calibration import fit_logistic_curve, probability_array


def aligned(y, p):
    p = probability_array(p)
    y = np.asarray(y)
    if y.shape != p.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("Outcomes and probabilities must be aligned, finite and binary")
    return y.astype(int), p


def wilson_interval(defaults, n, confidence=0.95):
    if n <= 0 or not 0 <= defaults <= n or not 0 < confidence < 1:
        raise ValueError("Invalid count or confidence")
    z = ndtri((1 + confidence) / 2)
    rate = defaults / n
    scale = 1 + z * z / n
    centre = (rate + z * z / (2 * n)) / scale
    radius = z * np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / scale
    return [
        0.0 if defaults == 0 else float(max(0, centre - radius)),
        1.0 if defaults == n else float(min(1, centre + radius)),
    ]


def point_metrics(y, p):
    y, p = aligned(y, p)
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None
    return {
        "roc_auc": auc,
        "gini": None if auc is None else 2 * auc - 1,
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "mean_pd": float(p.mean()),
        "observed_rate": float(y.mean()),
        "pd_gap": float(y.mean() - p.mean()),
    }


def evaluate(y, p, samples=150, seed=42, confidence=0.95):
    y, p = aligned(y, p)
    result = point_metrics(y, p)
    result.update(
        n=len(y),
        defaults=int(y.sum()),
        observed_rate_ci=wilson_interval(int(y.sum()), len(y), confidence),
    )
    # Independent Bernoulli outcomes conditional on all borrower and macro covariates.
    z = ndtri((1 + confidence) / 2)
    radius = float(z * np.sqrt(np.sum(p * (1 - p))) / len(y))
    result["pd_gap_null_interval"] = [-radius, radius]
    result["calibration_intercept"] = None
    result["calibration_slope"] = None
    if len(np.unique(y)) == 2:
        try:
            intercept, slope = fit_logistic_curve(y, p)
            result.update(calibration_intercept=float(intercept), calibration_slope=float(slope))
        except ValueError:
            pass
    rng = np.random.default_rng(seed)
    boot = {key: [] for key in ["roc_auc", "gini", "brier", "log_loss", "pd_gap"]}
    for _ in range(samples):
        idx = rng.integers(0, len(y), len(y))
        scores = point_metrics(y[idx], p[idx])
        for key in boot:
            if scores[key] is not None:
                boot[key].append(scores[key])
    alpha = (1 - confidence) / 2
    result["confidence_intervals"] = {
        key: np.quantile(value, [alpha, 1 - alpha]).tolist() if value else None
        for key, value in boot.items()
    }
    result["bootstrap_valid_auc_replicates"] = len(boot["roc_auc"])
    return result


def calibration_table(y, p, bins=8, confidence=0.95):
    y, p = aligned(y, p)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    group = np.searchsorted(edges[1:-1], p, side="right")
    rows = []
    for bin_id in np.unique(group):
        mask = group == bin_id
        n, k = int(mask.sum()), int(y[mask].sum())
        lo, hi = wilson_interval(k, n, confidence)
        rows.append(
            {
                "bin": int(bin_id),
                "n": n,
                "defaults": k,
                "mean_pd": float(p[mask].mean()),
                "observed_rate": k / n,
                "observed_low": lo,
                "observed_high": hi,
            }
        )
    return pd.DataFrame(rows)


def paired_comparison(y, baseline, challenger, samples=150, seed=42, confidence=0.95):
    y, baseline = aligned(y, baseline)
    _, challenger = aligned(y, challenger)
    base, other = point_metrics(y, baseline), point_metrics(y, challenger)
    rng = np.random.default_rng(seed)
    keys = ["roc_auc", "brier", "log_loss"]
    differences = {key: [] for key in keys}
    for _ in range(samples):
        idx = rng.integers(0, len(y), len(y))
        a, b = point_metrics(y[idx], baseline[idx]), point_metrics(y[idx], challenger[idx])
        for key in keys:
            if a[key] is not None and b[key] is not None:
                differences[key].append(b[key] - a[key])
    alpha = (1 - confidence) / 2
    return {
        key: {
            "challenger_minus_baseline": None if base[key] is None else other[key] - base[key],
            "ci": np.quantile(differences[key], [alpha, 1 - alpha]).tolist()
            if differences[key]
            else None,
        }
        for key in keys
    }
