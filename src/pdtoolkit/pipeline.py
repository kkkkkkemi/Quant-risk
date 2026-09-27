"""Orchestrate auditable development, immutable test evaluation and evidence export."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

from . import __version__
from .checks import IntegrityError, check_inputs, chronological_split
from .config import Config
from .data import generate
from .metrics import calibration_table, paired_comparison, point_metrics
from .models import introduce_calibration_error, train_models
from .reporting import create_report
from .validation import local_effects, validate_predictions


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_fingerprint():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git_state():
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
            ).strip()
        )
        return revision, dirty
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None, None


def dependency_versions():
    return {
        dist.metadata["Name"]: dist.version
        for dist in importlib.metadata.distributions()
        if dist.metadata["Name"]
    }


def verify_bundle(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    failures = []
    for name, expected in manifest["artifacts"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or sha256(path) != expected:
            failures.append(name)
    return {
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "checked_files": len(manifest["artifacts"]),
    }


def run_experiment(config: Config, output: str | Path):
    config.validate()
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            "Output directory is not empty; choose a new directory to preserve evidence"
        )
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    (output / "config.yaml").write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
    )
    data = generate(config)
    try:
        integrity = check_inputs(data, config)
        stages = chronological_split(data.observations, config)
    except IntegrityError as exc:
        write_json(
            output / "integrity_failure.json",
            {
                "status": "blocked",
                "scenario": config.data.scenario,
                "reason": str(exc),
                "action": "Correct data integrity before training or validation.",
            },
        )
        raise
    data.observations.to_csv(output / "observations.csv", index=False)
    data.oracle.to_csv(output / "oracle.csv", index=False)
    data.feature_available_at.to_csv(output / "feature_availability.csv", index=False)
    write_json(output / "integrity.json", integrity)
    # Fixed numerical threading makes small demonstrations quick and more reproducible.
    with threadpool_limits(limits=1):
        models, tuning, coefficients = train_models(stages, config)
        test = stages["test"]
        predictions = []
        for model in models:
            raw, calibrated = model.predict(test)
            for variant, p in [("raw", raw), ("calibrated", calibrated)]:
                p = introduce_calibration_error(p, config)
                frame = test[["borrower_id", "segment", "default_12m"]].copy()
                frame["observation_year"] = test.observation_date.dt.year
                frame["model"], frame["variant"], frame["predicted_pd"] = model.name, variant, p
                predictions.append(frame)
        predictions = pd.concat(predictions, ignore_index=True)
        validation = validate_predictions(predictions, stages["train"], test, config, coefficients)
        effects = local_effects(models, stages["train"])
        comparisons = {}
        if {model.name for model in models} == {"logistic", "gradient_boosting"}:
            for variant in ("raw", "calibrated"):
                pivot = (
                    predictions.loc[predictions.variant == variant]
                    .pivot(index="borrower_id", columns="model", values="predicted_pd")
                    .reindex(test.borrower_id)
                )
                comparisons[variant] = paired_comparison(
                    test.default_12m,
                    pivot.logistic,
                    pivot.gradient_boosting,
                    config.validation.bootstrap_samples,
                    config.data.seed,
                    config.validation.confidence,
                )
    # Oracle is first consulted here, after model fitting and calibration have completed.
    oracle = data.oracle.set_index("borrower_id").loc[test.borrower_id, "true_pd"].to_numpy()
    benchmark = {
        "mean_true_pd": float(oracle.mean()),
        "oracle_brier": point_metrics(test.default_12m, oracle)["brier"],
        "pd_mse": {},
    }
    for (model, variant), group in predictions.groupby(["model", "variant"]):
        truth = data.oracle.set_index("borrower_id").loc[group.borrower_id, "true_pd"].to_numpy()
        benchmark["pd_mse"][f"{model}/{variant}"] = float(
            np.mean((group.predicted_pd.to_numpy() - truth) ** 2)
        )
    revision, dirty = git_state()
    metadata = {
        "toolkit_version": __version__,
        "started_utc": started,
        "python": sys.version,
        "platform": platform.platform(),
        "git_revision": revision,
        "git_dirty": dirty,
        "source_sha256": source_fingerprint(),
        "data_sha256": sha256(output / "observations.csv"),
        "dependency_versions": dependency_versions(),
        "configuration": config.to_dict(),
        "command": sys.argv,
        "numerical_threads": 1,
        "splits": {
            name: {
                "n": len(stage),
                "defaults": int(stage.default_12m.sum()),
                "first_observation": str(stage.observation_date.min().date()),
                "latest_outcome_end": str(stage.outcome_end.max().date()),
            }
            for name, stage in stages.items()
        },
        "models": {
            model.name: {
                "selected_parameters": model.selected_parameters,
                "pipeline_parameters": {k: repr(v) for k, v in model.pipeline.get_params().items()},
                "calibration": model.calibrator.parameters,
            }
            for model in models
        },
        "oracle_benchmark": benchmark,
    }
    predictions.to_csv(output / "predictions.csv", index=False)
    coefficients.to_csv(output / "coefficients.csv", index=False)
    effects.to_csv(output / "sensitivities.csv", index=False)
    pd.json_normalize(validation["metrics"]).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(
        validation["findings"],
        columns=["severity", "model", "variant", "population", "code", "message", "action"],
    ).to_csv(output / "findings.csv", index=False)
    bins = []
    for (model, variant), group in predictions.groupby(["model", "variant"]):
        table = calibration_table(
            group.default_12m,
            group.predicted_pd,
            config.validation.calibration_bins,
            config.validation.confidence,
        )
        table["model"], table["variant"] = model, variant
        bins.append(table)
    pd.concat(bins, ignore_index=True).to_csv(output / "calibration_bins.csv", index=False)
    write_json(output / "tuning.json", tuning)
    write_json(output / "comparisons.json", comparisons)
    write_json(output / "validation.json", validation)
    (output / "models").mkdir()
    for model in models:
        joblib.dump(model, output / "models" / f"{model.name}.joblib")
    create_report(
        output / "report.pdf",
        config,
        validation,
        metadata,
        predictions,
        coefficients,
        comparisons,
        effects,
    )
    metadata["elapsed_seconds"] = time.perf_counter() - start
    write_json(output / "metadata.json", metadata)
    files = sorted(path for path in output.rglob("*") if path.is_file())
    write_json(
        output / "manifest.json",
        {
            "algorithm": "sha256",
            "artifacts": {path.relative_to(output).as_posix(): sha256(path) for path in files},
        },
    )
    with zipfile.ZipFile(output / "evidence.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files + [output / "manifest.json"]:
            archive.write(path, path.relative_to(output).as_posix())
    return {
        "scenario": config.data.scenario,
        "conclusion": validation["conclusion"],
        "output": str(output.resolve()),
        "report": str((output / "report.pdf").resolve()),
        "findings": len(validation["findings"]),
    }
