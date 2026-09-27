"""Strict, serialisable configuration; unknown fields fail instead of being ignored."""

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import get_args, get_origin

import numpy as np
import pandas as pd
import yaml

SCENARIOS = (
    "baseline",
    "downturn",
    "composition_shift",
    "mechanism_shift",
    "calibration_error",
    "omitted_variable",
    "leakage",
    "immature",
    "duplicate",
    "unavailable_feature",
)
MODELS = ("logistic", "gradient_boosting")


@dataclass
class DataConfig:
    seed: int = 42
    n_per_year: int = 1800
    start_year: int = 2013
    end_year: int = 2024
    as_of: str = "2026-01-01"
    missing_rate: float = 0.04
    intercept: float = -4.5
    scenario: str = "baseline"
    stress_start_year: int = 2023


@dataclass
class SplitConfig:
    train: list[int] = field(default_factory=lambda: list(range(2013, 2018)))
    tune: list[int] = field(default_factory=lambda: [2019])
    calibrate: list[int] = field(default_factory=lambda: [2021])
    test: list[int] = field(default_factory=lambda: [2023, 2024])


@dataclass
class ModelConfig:
    names: list[str] = field(default_factory=lambda: list(MODELS))
    calibration: str = "sigmoid"
    logistic_c: list[float] = field(default_factory=lambda: [0.1, 1.0, 10.0])
    boosting_leaves: list[int] = field(default_factory=lambda: [7, 15])
    boosting_iterations: int = 140
    calibration_error_logit: float = -1.5


@dataclass
class ValidationConfig:
    bootstrap_samples: int = 150
    confidence: float = 0.95
    calibration_bins: int = 8
    min_defaults: int = 20
    auc_floor: float = 0.65
    psi_warning: float = 0.2
    material_pd_gap: float = 0.01


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    def to_dict(self):
        return asdict(self)

    def validate(self):
        # Validate YAML types before arithmetic so malformed input fails with a useful error.
        for section in (self.data, self.splits, self.models, self.validation):
            for spec in fields(section):
                value = getattr(section, spec.name)
                expected = spec.type
                if get_origin(expected) is list:
                    item_type = get_args(expected)[0]
                    valid = isinstance(value, list) and all(
                        type(item) is item_type or (item_type is float and type(item) is int)
                        for item in value
                    )
                else:
                    valid = type(value) is expected or (expected is float and type(value) is int)
                if not valid:
                    raise ValueError(f"Invalid type for {type(section).__name__}.{spec.name}")
        d, s, m, v = self.data, self.splits, self.models, self.validation
        if d.scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario {d.scenario!r}; choose from {SCENARIOS}")
        if not isinstance(d.seed, int) or d.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(d.n_per_year, int) or d.n_per_year < 100:
            raise ValueError("n_per_year must be an integer >= 100")
        if not 2000 <= d.start_year <= d.end_year <= 2100:
            raise ValueError("Years must be ordered and within 2000..2100")
        if not 0 <= d.missing_rate < 0.5 or not np.isfinite(d.intercept):
            raise ValueError("Invalid missing_rate or intercept")
        if pd.isna(pd.Timestamp(d.as_of)):
            raise ValueError("Invalid as_of date")
        stages = [s.train, s.tune, s.calibrate, s.test]
        if any(not x or sorted(set(x)) != x for x in stages):
            raise ValueError("Every time split must contain sorted, unique years")
        for earlier, later in zip(stages, stages[1:]):
            if max(earlier) + 1 >= min(later):
                raise ValueError("Time splits require an intervening year for outcome maturity")
        used = [year for stage in stages for year in stage]
        if min(used) < d.start_year or max(used) > d.end_year:
            raise ValueError("Split years fall outside generated data")
        if d.stress_start_year < min(s.test):
            raise ValueError("Stress may start only in the final test window")
        if not m.names or len(set(m.names)) != len(m.names) or set(m.names) - set(MODELS):
            raise ValueError("Choose unique supported model names")
        if m.calibration not in ("sigmoid", "isotonic", "none"):
            raise ValueError("calibration must be sigmoid, isotonic or none")
        if not m.logistic_c or any(not np.isfinite(c) or c <= 0 for c in m.logistic_c):
            raise ValueError("logistic_c values must be positive and finite")
        if not m.boosting_leaves or any(x < 2 for x in m.boosting_leaves):
            raise ValueError("boosting_leaves values must be >= 2")
        if m.boosting_iterations < 1 or not np.isfinite(m.calibration_error_logit):
            raise ValueError("Invalid model parameters")
        if v.bootstrap_samples < 20 or not 0.5 < v.confidence < 1:
            raise ValueError("Use >=20 bootstrap samples and confidence in (0.5, 1)")
        if v.calibration_bins < 2 or v.min_defaults < 1 or not 0.5 <= v.auc_floor <= 1:
            raise ValueError("Invalid validation thresholds")
        if v.psi_warning <= 0 or v.material_pd_gap <= 0:
            raise ValueError("Materiality thresholds must be positive")
        return self


def load_config(path: str | Path | None = None) -> Config:
    raw = {} if path is None else yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a YAML mapping")
    classes = {
        "data": DataConfig,
        "splits": SplitConfig,
        "models": ModelConfig,
        "validation": ValidationConfig,
    }
    if set(raw) - set(classes):
        raise ValueError(f"Unknown config sections: {set(raw) - set(classes)}")
    sections = {}
    for name, cls in classes.items():
        value = raw.get(name, {})
        if not isinstance(value, dict) or set(value) - {f.name for f in fields(cls)}:
            raise ValueError(f"Unknown or invalid fields in {name}")
        sections[name] = cls(**value)
    return Config(**sections).validate()
