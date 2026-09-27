import copy
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from pypdf import PdfReader
from threadpoolctl import threadpool_limits

from pdtoolkit.checks import IntegrityError, chronological_split
from pdtoolkit.cli import main
from pdtoolkit.config import Config, load_config
from pdtoolkit.data import FEATURES, generate
from pdtoolkit.models import train_models
from pdtoolkit.pipeline import run_experiment, verify_bundle
from pdtoolkit.preprocessing import make_preprocessor
from pdtoolkit.validation import validate_predictions


def test_preprocessing_isolation(config, data):
    stages = chronological_split(data.observations, config)
    preprocessor = make_preprocessor().fit(stages["train"][FEATURES])
    medians = preprocessor.named_transformers_["numeric"].named_steps["impute"].statistics_.copy()
    contaminated = stages["test"][FEATURES].copy()
    contaminated["income"] = 1e12
    preprocessor.transform(contaminated)
    np.testing.assert_array_equal(
        medians, preprocessor.named_transformers_["numeric"].named_steps["impute"].statistics_
    )
    assert medians[0] == pytest.approx(stages["train"].income.median())


def test_training_has_no_access_to_test_outcomes_or_oracle(config, data):
    stages = chronological_split(data.observations, config)
    modified = copy.deepcopy(stages)
    modified["test"]["default_12m"] = 1 - modified["test"].default_12m
    modified["test"]["income"] = 1e12
    with threadpool_limits(limits=1):
        a, _, _ = train_models(stages, config)
        b, _, _ = train_models(modified, config)
        for left, right in zip(a, b):
            np.testing.assert_allclose(
                left.predict(stages["test"])[1], right.predict(stages["test"])[1], atol=1e-12
            )
            assert left.selected_parameters == right.selected_parameters


def test_validation_detects_calibration_error_and_weak_model(config, data):
    stages = chronological_split(data.observations, config)
    test = stages["test"]
    frame = test[["borrower_id", "segment", "default_12m"]].copy()
    frame["observation_year"] = test.observation_date.dt.year
    frame["model"], frame["variant"], frame["predicted_pd"] = "bad_model", "calibrated", 0.001
    result = validate_predictions(frame, stages["train"], test, config, pd.DataFrame())
    assert "CALIBRATION_BIAS" in {f["code"] for f in result["findings"]}
    assert "WEAK_DISCRIMINATION" in {f["code"] for f in result["findings"]}
    assert result["conclusion"] == "Restrict use pending remediation"


def test_end_to_end_report_evidence_reproducibility_and_saved_models(config, tmp_path):
    with threadpool_limits(limits=1):
        run_experiment(config, tmp_path / "first")
        run_experiment(config, tmp_path / "second")
    first, second = tmp_path / "first", tmp_path / "second"
    pd.testing.assert_frame_equal(
        pd.read_csv(first / "predictions.csv"), pd.read_csv(second / "predictions.csv")
    )
    assert json.loads((first / "validation.json").read_text()) == json.loads(
        (second / "validation.json").read_text()
    )
    assert verify_bundle(first)["status"] == "passed"
    reader = PdfReader(first / "report.pdf")
    assert len(reader.pages) >= 7
    assert "SYNTHETIC DATA" in reader.pages[0].extract_text()
    model = joblib.load(first / "models" / "logistic.joblib")
    test = chronological_split(generate(config).observations, config)["test"]
    assert np.isfinite(model.predict(test)[1]).all()
    assert (first / "evidence.zip").is_file()
    with (first / "predictions.csv").open("a") as stream:
        stream.write("tampered\n")
    assert verify_bundle(first)["failures"] == ["predictions.csv"]


def test_output_evidence_not_overwritten(config, tmp_path):
    (tmp_path / "existing.txt").write_text("keep")
    with pytest.raises(ValueError, match="not empty"):
        run_experiment(config, tmp_path)
    assert (tmp_path / "existing.txt").read_text() == "keep"


def test_integrity_failure_creates_evidence(config, tmp_path):
    config.data.scenario = "leakage"
    with pytest.raises(IntegrityError):
        run_experiment(config, tmp_path)
    assert json.loads((tmp_path / "integrity_failure.json").read_text())["status"] == "blocked"
    assert not (tmp_path / "report.pdf").exists()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: setattr(c.data, "scenario", "wrong"),
        lambda c: setattr(c.splits, "tune", [2018]),
        lambda c: setattr(c.data, "n_per_year", 0),
        lambda c: setattr(c.models, "calibration", "invalid"),
        lambda c: setattr(c.models, "names", ["unknown"]),
        lambda c: setattr(c.validation, "confidence", 1.0),
    ],
)
def test_invalid_config_rejected(mutate):
    config = Config()
    mutate(config)
    with pytest.raises(ValueError):
        config.validate()


def test_unknown_yaml_keys_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("data:\n  misspelled_seed: 42\n")
    with pytest.raises(ValueError, match="Unknown"):
        load_config(path)


@pytest.mark.parametrize("setting", ["n_per_year: text", "seed: true", "missing_rate: null"])
def test_wrong_yaml_types_rejected(tmp_path, setting):
    path = tmp_path / "bad-types.yaml"
    path.write_text(f"data:\n  {setting}\n")
    with pytest.raises(ValueError, match="Invalid type"):
        load_config(path)


def test_cli_config_and_expected_defect_suite(tmp_path):
    path = tmp_path / "config.yaml"
    assert main(["init-config", str(path)]) == 0
    assert main(["init-config", str(path)]) == 2
    assert (
        main(
            [
                "suite",
                "--config",
                str(path),
                "--output",
                str(tmp_path / "suite"),
                "--scenarios",
                "leakage",
                "duplicate",
                "unavailable_feature",
                "immature",
            ]
        )
        == 0
    )
