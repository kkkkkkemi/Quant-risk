"""All imputation, scaling and categories are learned exclusively from training data."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL, NUMERIC


def make_preprocessor(omit_delinquency: bool = False) -> ColumnTransformer:
    numeric = [c for c in NUMERIC if not (omit_delinquency and c == "previous_delinquency")]
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "segment",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first"),
                CATEGORICAL,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
