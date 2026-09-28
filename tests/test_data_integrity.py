import copy

import numpy as np
import pandas as pd
import pytest

from pdtoolkit.checks import IntegrityError, check_inputs, chronological_split
from pdtoolkit.data import FEATURES, generate


def test_generator_reproducible_and_oracle_is_separate(config):
    a, b = generate(config), generate(config)
    pd.testing.assert_frame_equal(a.observations, b.observations)
    assert "true_pd" not in a.observations
    assert "true_pd" not in FEATURES
    assert a.oracle.true_pd.between(0, 1).all()
    assert 0 < a.observations.default_12m.mean() < 0.15


def test_stress_changes_only_final_periods(config):
    baseline = generate(config)
    config.data.scenario = "mechanism_shift"
    stressed = generate(config)
    past = baseline.observations.observation_date.dt.year < config.data.stress_start_year
    pd.testing.assert_frame_equal(baseline.observations.loc[past], stressed.observations.loc[past])
    assert (
        stressed.observations.loc[~past, "default_12m"].mean()
        > baseline.observations.loc[~past, "default_12m"].mean()
    )


@pytest.mark.parametrize(
    "scenario, message",
    [
        ("leakage", "leakage"),
        ("immature", "Immature"),
        ("duplicate", "duplicate"),
        ("unavailable_feature", "unavailable"),
    ],
)
def test_deliberate_defects_rejected(config, scenario, message):
    config.data.scenario = scenario
    with pytest.raises(IntegrityError, match=message):
        check_inputs(generate(config), config)


def test_valid_integrity_and_mature_splits(config, data):
    assert check_inputs(data, config)["status"] == "passed"
    stages = chronological_split(data.observations, config)
    for left, right in zip(list(stages.values()), list(stages.values())[1:]):
        assert left.outcome_end.max() < right.observation_date.min()
        assert not set(left.borrower_id) & set(right.borrower_id)


@pytest.mark.parametrize(
    "defect", ["target", "infinity", "horizon", "event", "all_missing", "proxy", "negative_income"]
)
def test_mutated_data_rejected(config, data, defect):
    frame = data.observations
    if defect == "target":
        frame.loc[0, "default_12m"] = 2
    elif defect == "infinity":
        frame.loc[0, "income"] = np.inf
    elif defect == "horizon":
        frame.loc[0, "outcome_end"] += pd.Timedelta(days=1)
    elif defect == "event":
        idx = frame.index[frame.default_12m == 1][0]
        frame.loc[idx, "default_date"] = frame.loc[idx, "observation_date"]
    elif defect == "all_missing":
        frame["income"] = np.nan
    elif defect == "proxy":
        frame["previous_delinquency"] = frame.default_12m
    else:
        frame.loc[0, "income"] = -100
    with pytest.raises(IntegrityError):
        check_inputs(data, config)


def test_oracle_cannot_enter_feature_allowlist(config, data):
    poisoned = copy.deepcopy(data)
    poisoned.observations["true_pd"] = poisoned.oracle.true_pd
    with pytest.raises(IntegrityError, match="Unapproved"):
        check_inputs(poisoned, config)


def test_missing_split_year_is_rejected(config, data):
    frame = data.observations.loc[data.observations.observation_date.dt.year != 2019]
    with pytest.raises(IntegrityError, match="Missing observations"):
        chronological_split(frame, config)


def test_training_feature_without_observed_values_is_rejected(config, data):
    train_years = data.observations.observation_date.dt.year.isin(config.splits.train)
    data.observations.loc[train_years, "income"] = np.nan
    assert check_inputs(data, config)["status"] == "passed"
    with pytest.raises(IntegrityError, match="Entirely missing training features.*income"):
        chronological_split(data.observations, config)
