"""Leakage-safe preprocessing pipeline (best-practices §06, §07).

Every learned step (imputation values, scaling, encodings) sits inside a
ColumnTransformer, so calling .fit() on training rows is the only way any
statistic is learned and .transform() is all that touches test rows.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, TargetEncoder

from . import config


def _as_object(X):
    """sklearn imputers expect np.nan, not pd.NA, in categorical columns."""
    X = pd.DataFrame(X).astype(object)
    return X.where(X.notna(), np.nan)


def build_preprocessor(groups: dict[str, list[str]], scale: bool = True) -> ColumnTransformer:
    numeric_steps = [SimpleImputer(strategy="median", add_indicator=True)]
    if scale:
        numeric_steps.append(StandardScaler())
    to_obj = FunctionTransformer(_as_object, feature_names_out="one-to-one")
    transformers = [
        ("num", make_pipeline(*numeric_steps), groups["numeric"]),
        # Unknown Y/N is often informative (field not filled in), so keep an indicator.
        ("bool", SimpleImputer(strategy="most_frequent", add_indicator=True), groups["boolean"]),
        (
            "low_card",
            make_pipeline(
                to_obj,
                SimpleImputer(strategy="constant", fill_value="Missing"),
                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20, sparse_output=False),
            ),
            groups["low_card"],
        ),
        (
            # TargetEncoder cross-fits during fit_transform, so a row's encoding
            # never sees its own label; categories unseen in train get the global mean.
            "high_card",
            make_pipeline(
                to_obj,
                SimpleImputer(strategy="constant", fill_value="Missing"),
                TargetEncoder(
                    target_type="continuous",
                    cv=KFold(5, shuffle=True, random_state=config.RANDOM_SEED),
                ),
            ),
            groups["high_card"],
        ),
    ]
    transformers = [t for t in transformers if t[2]]
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)


def build_model_pipeline(groups: dict[str, list[str]], regressor, scale: bool = True, log_target: bool = True):
    """Preprocessor + regressor, optionally trained on log(ClosePrice)."""
    pipe = Pipeline([("pre", build_preprocessor(groups, scale=scale)), ("model", regressor)])
    if log_target:
        return TransformedTargetRegressor(regressor=pipe, func=np.log, inverse_func=np.exp)
    return pipe


def feature_frame(df: pd.DataFrame, groups: dict[str, list[str]]) -> pd.DataFrame:
    cols = [c for cols in groups.values() for c in cols]
    return df[cols]
