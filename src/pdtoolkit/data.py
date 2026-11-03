"""Generate observable snapshots and a separately held synthetic oracle."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit

from .config import Config

NUMERIC = [
    "income",
    "debt_to_income",
    "credit_utilisation",
    "previous_delinquency",
    "loan_amount",
    "unemployment",
    "gdp_growth",
]
CATEGORICAL = ["segment"]
FEATURES = NUMERIC + CATEGORICAL
EXPECTED_SIGNS = {
    "income": -1,
    "debt_to_income": 1,
    "credit_utilisation": 1,
    "previous_delinquency": 1,
    "loan_amount": 1,
    "unemployment": 1,
    "gdp_growth": -1,
}


@dataclass
class SyntheticData:
    observations: pd.DataFrame
    oracle: pd.DataFrame
    feature_available_at: pd.DataFrame


def generate(config: Config) -> SyntheticData:
    config.validate()
    d = config.data
    frames, truths, availability = [], [], []
    for year in range(d.start_year, d.end_year + 1):
        # Per-year streams keep all development data identical across stress scenarios.
        rng = np.random.default_rng(np.random.SeedSequence([d.seed, year]))
        n = d.n_per_year
        stress = year >= d.stress_start_year
        weights = [0.62, 0.25, 0.13]
        if stress and d.scenario == "composition_shift":
            weights = [0.15, 0.35, 0.50]
        segment = rng.choice(["prime", "standard", "vulnerable"], n, p=weights)
        seg_risk = np.select([segment == "standard", segment == "vulnerable"], [0.5, 1.1])
        latent = rng.normal(size=n)
        income = np.exp(10.85 - 0.22 * latent - 0.17 * seg_risk + rng.normal(0, 0.38, n))
        dti = expit(-0.9 + 0.7 * latent + 0.35 * seg_risk + rng.normal(0, 0.5, n))
        util = expit(-0.6 + 0.65 * latent + 0.4 * seg_risk + rng.normal(0, 0.65, n))
        delinquency = rng.poisson(np.exp(-1.2 + 0.45 * latent + 0.35 * seg_risk))
        loan = income * np.exp(-0.8 + 0.7 * dti + rng.normal(0, 0.4, n))
        cycle = np.sin((year - d.start_year) * 0.8)
        unemployment = 5.5 + 1.1 * cycle
        gdp = 1.6 - 0.8 * cycle
        if stress and d.scenario == "downturn":
            unemployment += 4.5
            gdp -= 4.0
            dti = np.clip(dti + 0.12, 0, 1)
            util = np.clip(util + 0.18, 0, 1)
        eta = (
            d.intercept
            - 0.45 * np.log(income / 50000)
            + 2.1 * (dti - 0.3)
            + 1.4 * (util - 0.35)
            + 0.55 * delinquency
            + 0.2 * np.log(loan / 25000)
            + 0.24 * (unemployment - 5.5)
            - 0.16 * (gdp - 1.6)
            + seg_risk
            + 1.8 * np.maximum(util - 0.65, 0) ** 2
            + 0.8 * dti * (delinquency > 0)
        )
        if stress and d.scenario == "mechanism_shift":
            eta += 1.7 + 1.1 * (segment == "vulnerable") + 1.5 * (util > 0.6)
        p = expit(eta)
        target = rng.binomial(1, p)
        observed = pd.Timestamp(year=year, month=1, day=1)
        horizon = observed + pd.DateOffset(months=12)
        frame = pd.DataFrame(
            {
                "borrower_id": [f"B{year}-{i:06d}" for i in range(n)],
                "observation_date": observed,
                "outcome_end": horizon,
                "income": income,
                "debt_to_income": dti,
                "credit_utilisation": util,
                "previous_delinquency": delinquency,
                "loan_amount": loan,
                "unemployment": unemployment,
                "gdp_growth": gdp,
                "segment": segment,
                "default_12m": target,
            }
        )
        days = rng.integers(1, (horizon - observed).days + 1, n)
        frame["default_date"] = pd.to_datetime(
            np.where(target == 1, observed + pd.to_timedelta(days, unit="D"), pd.NaT)
        )
        for col in ["income", "debt_to_income", "credit_utilisation", "loan_amount"]:
            mask = rng.random(n) < d.missing_rate * (1 + 0.3 * (delinquency > 0))
            frame.loc[mask, col] = np.nan
        avail = pd.DataFrame({"borrower_id": frame.borrower_id})
        for feature in FEATURES:
            avail[feature] = observed
        if d.scenario == "leakage":
            frame["collections_after_default"] = target
            avail["collections_after_default"] = horizon
        if d.scenario == "unavailable_feature":
            avail["income"] = observed + pd.Timedelta(days=1)
        if stress and d.scenario == "immature":
            frame["outcome_end"] = pd.Timestamp(d.as_of) + pd.Timedelta(days=30)
        frames.append(frame)
        truths.append(pd.DataFrame({"borrower_id": frame.borrower_id, "true_pd": p}))
        availability.append(avail)
    observations = pd.concat(frames, ignore_index=True)
    if d.scenario == "duplicate":
        observations = pd.concat([observations, observations.iloc[[0]]], ignore_index=True)
    return SyntheticData(
        observations,
        pd.concat(truths, ignore_index=True),
        pd.concat(availability, ignore_index=True),
    )
