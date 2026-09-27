"""Subpopulation metrics, reference-based drift, and explicit review findings."""

import numpy as np
import pandas as pd

from .config import Config
from .data import EXPECTED_SIGNS, FEATURES, NUMERIC
from .metrics import evaluate


def psi(reference, current, bins=10):
    reference, current = pd.Series(reference), pd.Series(current)
    if reference.empty or current.empty:
        raise ValueError("PSI requires non-empty reference and current samples")
    if pd.api.types.is_numeric_dtype(reference):
        values = reference.dropna().to_numpy(dtype=float)
        edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1))) if len(values) else []
        if len(edges) == 1:
            interior = np.array([np.nextafter(edges[0], -np.inf), np.nextafter(edges[0], np.inf)])
        else:
            interior = np.asarray(edges)[1:-1]

        def counts(series):
            valid = series.dropna().to_numpy(dtype=float)
            groups = np.searchsorted(interior, valid, side="right")
            return np.append(np.bincount(groups, minlength=len(interior) + 1), series.isna().sum())

        a, b = counts(reference), counts(current)
    else:
        levels = sorted(set(reference.fillna("MISSING")) | set(current.fillna("MISSING")))
        a = reference.fillna("MISSING").value_counts().reindex(levels, fill_value=0).to_numpy()
        b = current.fillna("MISSING").value_counts().reindex(levels, fill_value=0).to_numpy()
    # Jeffreys-style smoothing keeps empty bins finite; bins always come from training.
    a, b = (a + 0.5) / (a.sum() + 0.5 * len(a)), (b + 0.5) / (b.sum() + 0.5 * len(b))
    return float(np.sum((b - a) * np.log(b / a)))


def validate_predictions(
    predictions: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: Config,
    coefficients: pd.DataFrame,
):
    rows, findings = [], []
    v = config.validation
    for (model, variant), subset in predictions.groupby(["model", "variant"], sort=True):
        populations = [("overall", "all", subset)]
        populations += [
            ("period", str(year), group) for year, group in subset.groupby("observation_year")
        ]
        populations += [("segment", segment, group) for segment, group in subset.groupby("segment")]
        for population, label, group in populations:
            scores = evaluate(
                group.default_12m,
                group.predicted_pd,
                v.bootstrap_samples,
                config.data.seed,
                v.confidence,
            )
            rows.append(
                {
                    "model": model,
                    "variant": variant,
                    "population": population,
                    "label": label,
                    **scores,
                }
            )
            if scores["defaults"] < v.min_defaults:
                findings.append(
                    {
                        "severity": "LIMITATION",
                        "model": model,
                        "variant": variant,
                        "population": f"{population}:{label}",
                        "code": "LOW_DEFAULT_COUNT",
                        "message": f"Only {scores['defaults']} defaults; inference is weak.",
                        "action": "Pool more mature cohorts; use benchmarking and judgement.",
                    }
                )
            if variant != "calibrated":
                continue
            interval = scores["pd_gap_null_interval"]
            gap = scores["pd_gap"]
            if abs(gap) > v.material_pd_gap and not interval[0] <= gap <= interval[1]:
                direction = "underestimation" if gap > 0 else "overestimation"
                findings.append(
                    {
                        "severity": "HIGH",
                        "model": model,
                        "variant": variant,
                        "population": f"{population}:{label}",
                        "code": "CALIBRATION_BIAS",
                        "message": f"Material PD {direction}: observed minus predicted = {gap:.3%}.",
                        "action": "Restrict use in this population; validate recalibration on new data.",
                    }
                )
            if scores["roc_auc"] is not None and scores["roc_auc"] < v.auc_floor:
                findings.append(
                    {
                        "severity": "MEDIUM",
                        "model": model,
                        "variant": variant,
                        "population": f"{population}:{label}",
                        "code": "WEAK_DISCRIMINATION",
                        "message": f"AUC {scores['roc_auc']:.3f} is below the illustrative threshold.",
                        "action": "Investigate specification, segmentation and additional predictors.",
                    }
                )
    drift = {feature: psi(train[feature], test[feature]) for feature in FEATURES}
    for feature, value in drift.items():
        if value > v.psi_warning:
            findings.append(
                {
                    "severity": "MEDIUM",
                    "model": "portfolio",
                    "variant": "all",
                    "population": "test",
                    "code": "POPULATION_SHIFT",
                    "message": f"{feature}: training-reference PSI {value:.3f}.",
                    "action": "Review representativeness; separate mix changes from risk changes.",
                }
            )
    if not coefficients.empty:
        overall = coefficients.loc[coefficients.window == "all_train"]
        for row in overall.itertuples():
            if (
                row.feature in EXPECTED_SIGNS
                and np.sign(row.coefficient) != EXPECTED_SIGNS[row.feature]
            ):
                findings.append(
                    {
                        "severity": "MEDIUM",
                        "model": "logistic",
                        "variant": "raw",
                        "population": "training",
                        "code": "VARIABLE_EFFECT",
                        "message": f"{row.feature} fitted sign differs from the generating mechanism.",
                        "action": "Review correlated covariates, sample size and omitted interactions.",
                    }
                )
    if config.data.scenario == "omitted_variable":
        findings.append(
            {
                "severity": "MEDIUM",
                "model": "all",
                "variant": "all",
                "population": "all",
                "code": "OMITTED_VARIABLE",
                "message": "Delinquency is deliberately excluded despite a direct risk effect.",
                "action": "Restore the variable and compare on a fresh validation cohort.",
            }
        )
    high = any(f["severity"] == "HIGH" for f in findings)
    conclusion = (
        "Restrict use pending remediation"
        if high
        else "Further review required"
        if findings
        else "No material issue under configured checks"
    )
    return {
        "metrics": rows,
        "drift_psi": drift,
        "findings": findings,
        "conclusion": conclusion,
        "thresholds": vars(v),
        "scope": "Synthetic exercise; not production or regulatory approval",
    }


def local_effects(models, train):
    # Train-only perturbations: association/sensitivity, never a causal estimate.
    rows = []
    sample = train.head(500).copy()
    for model in models:
        baseline = model.predict(sample)[1]
        for feature in NUMERIC:
            changed = sample.copy()
            increment = float(train[feature].std()) * 0.25
            changed[feature] = changed[feature] + increment
            if feature in ("debt_to_income", "credit_utilisation"):
                changed[feature] = changed[feature].clip(0, 1)
            delta = float(np.mean(model.predict(changed)[1] - baseline))
            rows.append(
                {
                    "model": model.name,
                    "feature": feature,
                    "increment": increment,
                    "mean_pd_change": delta,
                    "expected_sign": EXPECTED_SIGNS[feature],
                }
            )
    return pd.DataFrame(rows)
