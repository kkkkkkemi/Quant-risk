"""Train-only pipelines; tune-only candidate choice; calibration-only adjustment."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline

from .calibration import Calibrator
from .config import Config
from .data import FEATURES
from .preprocessing import make_preprocessor


@dataclass
class FittedModel:
    name: str
    pipeline: Pipeline
    calibrator: Calibrator
    selected_parameters: dict

    def predict(self, observations: pd.DataFrame):
        raw = self.pipeline.predict_proba(observations[FEATURES])[:, 1]
        return raw, self.calibrator.predict(raw)


def build_pipeline(name, config, parameters):
    if name == "logistic":
        estimator = LogisticRegression(
            C=parameters["C"], max_iter=2000, random_state=config.data.seed
        )
    elif name == "gradient_boosting":
        estimator = HistGradientBoostingClassifier(
            max_leaf_nodes=parameters["max_leaf_nodes"],
            max_iter=config.models.boosting_iterations,
            learning_rate=0.06,
            l2_regularization=2.0,
            early_stopping=False,
            random_state=config.data.seed,
        )
    else:
        raise ValueError(f"Unsupported model {name}")
    return Pipeline(
        [
            ("preprocess", make_preprocessor(config.data.scenario == "omitted_variable")),
            ("model", estimator),
        ]
    )


def train_models(stages: dict, config: Config):
    train, tune, cal = (stages[key] for key in ("train", "tune", "calibrate"))
    fitted, tuning, coefficients = [], [], []
    for name in config.models.names:
        candidates = (
            [{"C": c} for c in config.models.logistic_c]
            if name == "logistic"
            else [{"max_leaf_nodes": x} for x in config.models.boosting_leaves]
        )
        best, best_loss, best_parameters = None, float("inf"), None
        for parameters in candidates:
            pipeline = build_pipeline(name, config, parameters)
            pipeline.fit(train[FEATURES], train.default_12m)
            tune_pd = pipeline.predict_proba(tune[FEATURES])[:, 1]
            loss = float(log_loss(tune.default_12m, tune_pd, labels=[0, 1]))
            tuning.append({"model": name, "parameters": parameters, "tune_log_loss": loss})
            if loss < best_loss:
                best, best_loss, best_parameters = pipeline, loss, parameters
        p = best.predict_proba(cal[FEATURES])[:, 1]
        calibrator = Calibrator(config.models.calibration).fit(p, cal.default_12m)
        fitted.append(FittedModel(name, best, calibrator, best_parameters))
        if name == "logistic":
            for label, subset in [("all_train", train)] + [
                (str(year), train.loc[train.observation_date.dt.year == year])
                for year in config.splits.train
            ]:
                if subset.default_12m.nunique() < 2:
                    continue
                model = (
                    best if label == "all_train" else build_pipeline(name, config, best_parameters)
                )
                if label != "all_train":
                    model.fit(subset[FEATURES], subset.default_12m)
                feature_names = model.named_steps["preprocess"].get_feature_names_out()
                values = model.named_steps["model"].coef_[0]
                coefficients.extend(
                    {"window": label, "feature": f, "coefficient": float(c)}
                    for f, c in zip(feature_names, values)
                )
    return fitted, tuning, pd.DataFrame(coefficients, columns=["window", "feature", "coefficient"])


def introduce_calibration_error(p, config):
    if config.data.scenario == "calibration_error":
        return expit(logit(np.clip(p, 1e-8, 1 - 1e-8)) + config.models.calibration_error_logit)
    return np.asarray(p).copy()
