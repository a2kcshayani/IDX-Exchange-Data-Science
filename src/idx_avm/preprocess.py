"""Leakage-safe preprocessing pipeline (best-practices §06, §07).

Every learned step (imputation values, scaling, encodings) sits inside a
ColumnTransformer, so calling .fit() on training rows is the only way any
statistic is learned and .transform() is all that touches test rows.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
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


class QuantileClipper(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Clip each numeric column to percentiles learned on the training rows.

    The raw MLS fields contain entry errors (e.g. a 24.9-billion sq ft lot,
    500 garage spaces) that a linear model extrapolates into absurd prices.
    Bounds are fit on train only and frozen, like every other learned step.
    """

    def __init__(self, lower: float = config.PRICE_LOWER_Q, upper: float = config.PRICE_UPPER_Q):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.lower_ = np.nanquantile(X, self.lower, axis=0)
        self.upper_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)


class CompPPSFEncoder(TransformerMixin, BaseEstimator):
    """Comp-based neighborhood price per sq ft, learned from training sales only.

    Input columns: [PostalCode, LivingArea]; y is log(ClosePrice) (the model's
    target space). For each ZIP it learns the median log price-per-sq-ft of
    prior (training-window) sales, shrunk toward the overall median when a ZIP
    has few sales. Outputs two features: log comp PPSF, and log comp PPSF +
    log(LivingArea), i.e. a comps-implied log value for the home.

    Like TargetEncoder, fit_transform cross-fits (K folds) so a training row's
    feature never includes its own sale price; transform() on new rows uses
    statistics from all training rows. ClosePrice itself is never an input.
    """

    def __init__(self, smoothing: float = 20.0, n_splits: int = 5, random_state: int = config.RANDOM_SEED):
        self.smoothing = smoothing
        self.n_splits = n_splits
        self.random_state = random_state

    @staticmethod
    def _split(X):
        X = pd.DataFrame(X)
        zips = X.iloc[:, 0].astype(object).where(X.iloc[:, 0].notna(), "Missing").astype(str).to_numpy()
        sqft = pd.to_numeric(X.iloc[:, 1], errors="coerce").to_numpy(dtype=float)
        return zips, sqft

    def _stats(self, zips, sqft, y):
        ok = np.isfinite(sqft) & (sqft > 0)
        lp = pd.Series(np.asarray(y, dtype=float)[ok] - np.log(sqft[ok]))
        g = lp.groupby(zips[ok]).agg(["median", "size"])
        prior = float(lp.median())
        shrunk = (g["size"] * g["median"] + self.smoothing * prior) / (g["size"] + self.smoothing)
        return shrunk.to_dict(), prior

    def _apply(self, zips, sqft, table, prior):
        log_ppsf = np.array([table.get(z, prior) for z in zips], dtype=float)
        s = np.where(np.isfinite(sqft) & (sqft > 0), sqft, self.sqft_median_)
        s = np.clip(s, self.sqft_lo_, self.sqft_hi_)
        return np.column_stack([log_ppsf, log_ppsf + np.log(s)])

    def fit(self, X, y):
        zips, sqft = self._split(X)
        valid = sqft[np.isfinite(sqft) & (sqft > 0)]
        self.sqft_median_ = float(np.median(valid))
        self.sqft_lo_, self.sqft_hi_ = np.quantile(valid, [config.PRICE_LOWER_Q, config.PRICE_UPPER_Q])
        self.table_, self.prior_ = self._stats(zips, sqft, y)
        self.n_features_in_ = 2
        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        zips, sqft = self._split(X)
        y = np.asarray(y, dtype=float)
        out = np.empty((len(zips), 2))
        folds = KFold(self.n_splits, shuffle=True, random_state=self.random_state)
        for fit_idx, enc_idx in folds.split(zips):
            table, prior = self._stats(zips[fit_idx], sqft[fit_idx], y[fit_idx])
            out[enc_idx] = self._apply(zips[enc_idx], sqft[enc_idx], table, prior)
        return out

    def transform(self, X):
        zips, sqft = self._split(X)
        return self._apply(zips, sqft, self.table_, self.prior_)

    def get_feature_names_out(self, input_features=None):
        return np.array(["comp_log_ppsf", "comp_log_value"], dtype=object)


def build_preprocessor(groups: dict[str, list[str]], scale: bool = True, clip_features: bool = True) -> ColumnTransformer:
    numeric_steps = [QuantileClipper()] if clip_features else []
    numeric_steps.append(SimpleImputer(strategy="median", add_indicator=True))
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
    if groups.get("comp"):
        transformers.append(("comp", CompPPSFEncoder(), groups["comp"]))
    transformers = [t for t in transformers if t[2]]
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)


def build_model_pipeline(
    groups: dict[str, list[str]], regressor, scale: bool = True, log_target: bool = True, clip_features: bool = True
):
    """Preprocessor + regressor, optionally trained on log(ClosePrice)."""
    pipe = Pipeline([("pre", build_preprocessor(groups, scale=scale, clip_features=clip_features)), ("model", regressor)])
    if log_target:
        return TransformedTargetRegressor(regressor=pipe, func=np.log, inverse_func=np.exp)
    return pipe


def feature_list(groups: dict[str, list[str]]) -> list[str]:
    """Unique raw input columns, in order (a column can feed more than one group)."""
    return list(dict.fromkeys(c for cols in groups.values() for c in cols))


def feature_frame(df: pd.DataFrame, groups: dict[str, list[str]]) -> pd.DataFrame:
    return df[feature_list(groups)]
