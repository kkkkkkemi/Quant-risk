"""Fail closed on invalid targets, timing, features and experiment boundaries."""

import numpy as np
import pandas as pd

from .config import Config
from .data import CATEGORICAL, FEATURES, NUMERIC, SyntheticData


class IntegrityError(ValueError):
    """A data or workflow integrity check failed."""


def check_inputs(data: SyntheticData, config: Config) -> dict:
    frame = data.observations
    required = set(
        FEATURES + ["borrower_id", "observation_date", "outcome_end", "default_date", "default_12m"]
    )
    if required - set(frame):
        raise IntegrityError(f"Missing columns: {required - set(frame)}")
    if set(frame) - required:
        raise IntegrityError(f"Unapproved feature / leakage columns: {set(frame) - required}")
    if frame.empty or frame.borrower_id.isna().any() or frame.borrower_id.duplicated().any():
        raise IntegrityError("Empty data, missing identifier or duplicate borrower records")
    if frame[["observation_date", "outcome_end"]].isna().any().any():
        raise IntegrityError("Missing timing information")
    if not frame.default_12m.isin([0, 1]).all():
        raise IntegrityError("Target must contain binary realised outcomes")
    expected_end = frame.observation_date.map(lambda x: x + pd.DateOffset(months=12))
    if (frame.outcome_end > pd.Timestamp(config.data.as_of)).any():
        raise IntegrityError("Immature twelve-month outcomes at the data as-of date")
    if not frame.outcome_end.equals(expected_end.rename("outcome_end")):
        raise IntegrityError("Outcome horizon must be exactly twelve calendar months")
    defaults = frame.default_12m == 1
    if (
        frame.loc[defaults, "default_date"].isna().any()
        or frame.loc[~defaults, "default_date"].notna().any()
        or (frame.loc[defaults, "default_date"] <= frame.loc[defaults, "observation_date"]).any()
        or (frame.loc[defaults, "default_date"] > frame.loc[defaults, "outcome_end"]).any()
    ):
        raise IntegrityError("Default event timing disagrees with the twelve-month target")
    availability = data.feature_available_at.set_index("borrower_id")
    if availability.index.duplicated().any() or set(availability.index) != set(frame.borrower_id):
        raise IntegrityError("Availability metadata identifiers do not match observations")
    for feature in FEATURES:
        if feature not in availability:
            raise IntegrityError(f"Missing availability metadata for {feature}")
        dates = availability.loc[frame.borrower_id, feature].reset_index(drop=True)
        if dates.isna().any() or (dates > frame.observation_date.reset_index(drop=True)).any():
            raise IntegrityError(f"Feature {feature} unavailable at prediction time")
    for col in NUMERIC:
        if not pd.api.types.is_numeric_dtype(frame[col]):
            raise IntegrityError(f"Non-numeric values in {col}")
        if np.isinf(frame[col].to_numpy(dtype=float)).any():
            raise IntegrityError(f"Infinite values in {col}")
        if frame[col].isna().all():
            raise IntegrityError(f"Entirely missing feature {col}")
    for col in ["income", "loan_amount"]:
        if (frame[col].dropna() <= 0).any():
            raise IntegrityError(f"{col} must be positive")
    for col in ["debt_to_income", "credit_utilisation"]:
        if not frame[col].dropna().between(0, 1).all():
            raise IntegrityError(f"{col} must be in [0, 1]")
    delinquency = frame.previous_delinquency.dropna()
    if (delinquency < 0).any() or (delinquency % 1 != 0).any():
        raise IntegrityError("Delinquency counts must be non-negative integers")
    for col in CATEGORICAL:
        if (
            frame[col].isna().any()
            or not frame[col].isin(["prime", "standard", "vulnerable"]).all()
        ):
            raise IntegrityError("Unknown or missing segment")
    for feature in NUMERIC:
        observed = frame[feature].dropna()
        if observed.nunique() <= 2 and len(observed) > 0:
            target = frame.loc[observed.index, "default_12m"]
            if np.array_equal(observed, target) or np.array_equal(observed, 1 - target):
                raise IntegrityError(f"Perfect target proxy in feature {feature}")
    return {
        "status": "passed",
        "n_records": len(frame),
        "unique_borrowers": len(frame),
        "missing_rates": frame[FEATURES].isna().mean().to_dict(),
        "max_outcome_end": str(frame.outcome_end.max().date()),
        "as_of": config.data.as_of,
    }


def chronological_split(frame: pd.DataFrame, config: Config) -> dict[str, pd.DataFrame]:
    config.validate()
    stages = {
        name: frame.loc[frame.observation_date.dt.year.isin(years)].copy()
        for name, years in vars(config.splits).items()
    }
    for name, stage in stages.items():
        expected_years = set(getattr(config.splits, name))
        if stage.empty or set(stage.observation_date.dt.year) != expected_years:
            raise IntegrityError(f"Missing observations for {name} split years")
        if name != "test" and stage.default_12m.nunique() != 2:
            raise IntegrityError(f"{name} split requires defaults and non-defaults")
    values = list(stages.values())
    for before, after in zip(values, values[1:]):
        if before.outcome_end.max() >= after.observation_date.min():
            raise IntegrityError(
                "Labels from an earlier stage are not mature before the next stage"
            )
        if set(before.borrower_id) & set(after.borrower_id):
            raise IntegrityError("Borrower overlap between time splits")
    return stages
