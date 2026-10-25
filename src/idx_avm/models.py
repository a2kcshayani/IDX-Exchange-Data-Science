"""Weeks 5-6: model zoo, validation-month tuning, and test-month evaluation.

Every model shares the same leakage-safe preprocessing pipeline and is trained
on log(ClosePrice). Hyperparameters are chosen on the validation month (the
month before the test month, trained on the X months before *it*); the test
month is scored once with the chosen settings.
"""
import time
from dataclasses import dataclass, field
from itertools import product


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from . import config
from .features import feature_columns
from .metrics import regression_metrics
from .preprocess import build_model_pipeline, feature_frame
from .split import apply_outlier_filter, window_split

# Small, explainable grids. Trees don't need scaling, but sharing one pipeline keeps comparisons fair.
MODEL_GRIDS = {
    "Linear Regression": (LinearRegression, [{}]),
    "Decision Tree": (
        DecisionTreeRegressor,
        [{"max_depth": d, "min_samples_leaf": leaf} for d, leaf in product([10, 15, 20], [5, 20])],
    ),
    "Random Forest": (
        RandomForestRegressor,
        [{"n_estimators": 150, "min_samples_leaf": leaf, "max_features": mf, "n_jobs": -1}
         for leaf, mf in product([2, 5], [0.33, 0.5])],
    ),
}


def make_regressor(name: str, params: dict):
    cls, _ = MODEL_GRIDS[name]
    if cls is LinearRegression:
        return cls(**params)
    return cls(random_state=config.RANDOM_SEED, **params)


@dataclass
class FitResult:
    model_name: str
    params: dict
    feature_set: str
    test_month: str
    metrics: dict
    fit_seconds: float
    model: object = field(repr=False, default=None)
    groups: dict = field(repr=False, default=None)
    test: pd.DataFrame = field(repr=False, default=None)


def fit_and_score(df, test_month, n_train_months, model_name, params, flags, feature_set="base") -> FitResult:
    split, _, _ = apply_outlier_filter(window_split(df, test_month, n_train_months))
    groups = feature_columns(split.train, **flags)
    model = build_model_pipeline(groups, make_regressor(model_name, params))
    start = time.perf_counter()
    model.fit(feature_frame(split.train, groups), split.train[config.TARGET])
    seconds = time.perf_counter() - start
    test = split.test.copy()
    test["Predicted"] = model.predict(feature_frame(test, groups))
    return FitResult(
        model_name, params, feature_set, str(test_month),
        regression_metrics(test[config.TARGET], test["Predicted"]), seconds, model, groups, test,
    )


def tune_on_validation(df, test_month, n_train_months, model_name, flags, log=print) -> tuple[dict, pd.DataFrame]:
    """Pick hyperparameters by validation-month MdAPE. Returns (best params, all trials)."""
    val_month = test_month - 1
    rows = []
    for params in MODEL_GRIDS[model_name][1]:
        r = fit_and_score(df, val_month, n_train_months, model_name, params, flags)
        rows.append({"model": model_name, "params": _fmt(params), "val_month": str(val_month),
                     **r.metrics, "fit_seconds": r.fit_seconds})
        log(f"  {model_name:17s} {_fmt(params):55s} val MdAPE {r.metrics['MdAPE']:.2f}%")
    trials = pd.DataFrame(rows)
    best = MODEL_GRIDS[model_name][1][int(trials["MdAPE"].idxmin())]
    return best, trials


def feature_importance(result: FitResult, top: int = 15) -> pd.DataFrame:
    """Impurity importances for tree models, mapped back to raw input names."""
    pipe = result.model.regressor_
    reg = pipe.named_steps["model"]
    if not hasattr(reg, "feature_importances_"):
        return pd.DataFrame()
    names = pipe.named_steps["pre"].get_feature_names_out()
    imp = pd.Series(reg.feature_importances_, index=names)
    # "num__LivingArea" -> "LivingArea"; indicator columns fold into their source feature.
    raw = imp.index.str.split("__", n=1).str[1].str.replace(r"^missingindicator_", "", regex=True)
    raw = raw.str.replace(r"^(CountyOrParish)_.*$", r"\1", regex=True)
    grouped = imp.groupby(raw).sum().sort_values(ascending=False)
    return grouped.head(top).rename("importance").reset_index().rename(columns={"index": "feature"})


def _fmt(params: dict) -> str:
    shown = {k: v for k, v in params.items() if k != "n_jobs"}
    return ", ".join(f"{k}={v}" for k, v in shown.items()) or "defaults"



