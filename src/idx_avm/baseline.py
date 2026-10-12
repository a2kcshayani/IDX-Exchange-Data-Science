"""Week 4: Linear Regression baseline, evaluated on the held-out test month."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from . import config
from .features import feature_columns
from .metrics import metrics_by_price_band, regression_metrics
from .preprocess import build_model_pipeline, feature_frame
from .split import apply_outlier_filter, window_split


@dataclass
class BaselineResult:
    model: object
    groups: dict[str, list[str]]
    metrics: dict
    test: pd.DataFrame          # test rows with a "Predicted" column
    train_months: list[pd.Period]
    test_month: pd.Period
    price_thresholds: dict


def fit_linear(train: pd.DataFrame, clip_features: bool = True):
    groups = feature_columns(train)
    model = build_model_pipeline(groups, LinearRegression(), clip_features=clip_features)
    model.fit(feature_frame(train, groups), train[config.TARGET])
    return model, groups


def run_baseline(df: pd.DataFrame, test_month: pd.Period, n_train_months: int, clip_features: bool = True) -> BaselineResult:
    split, _, stats = apply_outlier_filter(window_split(df, test_month, n_train_months))
    model, groups = fit_linear(split.train, clip_features)
    test = split.test.copy()
    test["Predicted"] = model.predict(feature_frame(test, groups))
    return BaselineResult(
        model=model,
        groups=groups,
        metrics=regression_metrics(test[config.TARGET], test["Predicted"]),
        test=test,
        train_months=split.train_months,
        test_month=test_month,
        price_thresholds=stats,
    )


def breakdowns(test: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Error by price quintile and by county (counties with >= 100 test sales)."""
    by_band = metrics_by_price_band(test[config.TARGET], test["Predicted"])
    by_band.insert(0, "price_range", test.groupby(
        pd.qcut(test[config.TARGET], 5, labels=by_band.index), observed=True
    )[config.TARGET].agg(lambda s: f"${s.min():,.0f} - ${s.max():,.0f}"))
    counties = test["CountyOrParish"].value_counts()
    big = counties[counties >= 100].index
    by_county = (
        test[test["CountyOrParish"].isin(big)]
        .groupby("CountyOrParish")
        .apply(lambda g: pd.Series(regression_metrics(g[config.TARGET], g["Predicted"])), include_groups=False)
        .sort_values("n", ascending=False)
    )
    return {"by_price_band": by_band, "by_county": by_county}


def residual_summary(test: pd.DataFrame) -> dict:
    ape = (test["Predicted"] - test[config.TARGET]).abs() / test[config.TARGET]
    return {
        "within_5pct": float((ape <= 0.05).mean() * 100),
        "within_10pct": float((ape <= 0.10).mean() * 100),
        "within_20pct": float((ape <= 0.20).mean() * 100),
        "median_signed_error_pct": float(np.median((test["Predicted"] - test[config.TARGET]) / test[config.TARGET]) * 100),
    }
