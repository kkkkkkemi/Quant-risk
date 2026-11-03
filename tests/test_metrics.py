import numpy as np
import pytest
from sklearn.metrics import brier_score_loss, log_loss

from pdtoolkit.calibration import Calibrator
from pdtoolkit.metrics import (
    calibration_table,
    evaluate,
    paired_comparison,
    point_metrics,
    wilson_interval,
)
from pdtoolkit.validation import psi


def test_metric_known_values():
    y, p = [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]
    result = point_metrics(y, p)
    assert result["roc_auc"] == 1
    assert result["gini"] == 1
    assert result["brier"] == pytest.approx(0.025)
    assert result["brier"] == pytest.approx(brier_score_loss(y, p))
    assert result["log_loss"] == pytest.approx(log_loss(y, p))
    assert result["pd_gap"] == pytest.approx(0)


@pytest.mark.parametrize(
    "y,p",
    [
        ([0, 1], [0.1]),
        ([0, 2], [0.1, 0.9]),
        ([0, 1], [-0.1, 0.9]),
        ([0, 1], [0.1, 1.1]),
        ([0, 1], [np.nan, 0.9]),
        ([], []),
    ],
)
def test_metric_invalid_inputs(y, p):
    with pytest.raises(ValueError):
        point_metrics(y, p)


def test_single_class_metrics_and_uncertainty_are_explicit():
    result = evaluate(np.zeros(25), np.full(25, 0.05), samples=20)
    assert result["roc_auc"] is None
    assert result["confidence_intervals"]["roc_auc"] is None
    assert result["defaults"] == 0
    assert result["observed_rate_ci"][1] > 0
    assert result["brier"] > 0


def test_wilson_bounds_and_invalid_counts():
    assert wilson_interval(0, 100)[0] == 0
    assert wilson_interval(100, 100)[1] == 1
    with pytest.raises(ValueError):
        wilson_interval(2, 1)


def test_constant_predictions_have_one_calibration_bin():
    result = calibration_table([0, 0, 1, 0], [0.2] * 4)
    assert len(result) == 1
    assert result.n.sum() == 4


def test_paired_identical_models_have_zero_difference():
    result = paired_comparison([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], [0.1, 0.2, 0.8, 0.9], samples=20)
    for values in result.values():
        assert values["challenger_minus_baseline"] == 0
        assert values["ci"] == [0, 0]


@pytest.mark.parametrize("method", ["sigmoid", "isotonic", "none"])
def test_calibrators_bounded_and_monotone(method):
    y = np.array([0] * 70 + [1] * 30)
    p = np.linspace(0.001, 0.99, 100)
    model = Calibrator(method).fit(p, y)
    adjusted = model.predict(np.array([0, 0.2, 0.8, 1]))
    assert ((adjusted >= 0) & (adjusted <= 1)).all()
    assert (np.diff(adjusted) >= 0).all()


def test_unfitted_calibrator_rejected():
    with pytest.raises(ValueError, match="not been fitted"):
        Calibrator("sigmoid").predict([0.5])


def test_psi_detects_mix_missingness_and_constant_macro_shift():
    assert psi(np.arange(100), np.arange(100)) == 0
    assert psi([5.0] * 100, [9.0] * 100) > 0.2
    assert psi(["prime"] * 100, ["vulnerable"] * 100) > 0.2
    assert psi(np.arange(100), [np.nan] * 100) > 0.2
