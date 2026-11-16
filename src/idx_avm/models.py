"""Weeks 5-7: model zoo, validation-month tuning, and test-month evaluation.

Every model shares the same leakage-safe preprocessing pipeline and is trained
on log(ClosePrice). Hyperparameters are chosen on the validation month (the
month before the test month, trained on the X months before *it*); the test
month is scored once with the chosen settings.
"""
import time
from dataclasses import dataclass, field
from itertools import product

import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from . import config
from .features import feature_columns
from .metrics import regression_metrics
from .preprocess import build_model_pipeline, feature_frame, feature_list
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
    # Week 7. scikit-learn's histogram gradient boosting (same algorithm family as LightGBM; ships
    # with scikit-learn, so no OpenMP install is needed). Early stopping is off so the number of
    # trees is a tuned setting, not a random internal split.
    "Gradient Boosting": (
        HistGradientBoostingRegressor,
        [{"max_iter": n, "learning_rate": lr, "max_depth": d, "max_leaf_nodes": 127,
          "min_samples_leaf": 20, "early_stopping": False}
         for d, lr, n in product([6, 10, None], [0.05, 0.1], [500, 1000])],
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


# Validation MdAPE differences smaller than this are treated as noise (the selected model's
# month-to-month backtest spread is ~0.2 pts), per best-practices §09.
NOISE_TOLERANCE_PTS = 0.10
MODEL_COMPLEXITY = {name: i for i, name in enumerate(MODEL_GRIDS)}


def select_model(results: pd.DataFrame, feature_set_order: list[str], tol: float = NOISE_TOLERANCE_PTS) -> pd.Series:
    """Pick the simplest configuration whose validation MdAPE is within `tol` of the best.

    "Simplest" = earliest feature set (fewest additions), then simplest model. An addition
    only earns its place if it beats the noise; ties go to the simpler pipeline.
    """
    best = results["val_MdAPE"].min()
    near = results[results["val_MdAPE"] <= best + tol].copy()
    near["_fs"] = near["feature_set"].map({f: i for i, f in enumerate(feature_set_order)})
    near["_m"] = near["model"].map(MODEL_COMPLEXITY)
    return near.sort_values(["_fs", "_m", "val_MdAPE"]).iloc[0].drop(["_fs", "_m"])


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


def params_from_label(model_name: str, label: str) -> dict:
    """Map a saved trial label (e.g. from week5_tuning_trials.csv) back to its grid params."""
    for params in MODEL_GRIDS[model_name][1]:
        if _fmt(params) == label:
            return params
    raise KeyError(f"No {model_name} grid entry matches '{label}'")


def _mdape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, float)
    return float(np.median(np.abs(np.asarray(y_pred, float) - y_true) / y_true) * 100)


def permutation_mdape_importance(result: FitResult, n_rows: int = 5000, n_repeats: int = 3) -> pd.DataFrame:
    """How much test MdAPE rises (in points) when each raw input is shuffled. Works for any model."""
    test = result.test.sample(min(n_rows, len(result.test)), random_state=config.RANDOM_SEED)
    X = feature_frame(test, result.groups)
    scorer = make_scorer(_mdape, greater_is_better=False)
    r = permutation_importance(result.model, X, test[config.TARGET], scoring=scorer,
                               n_repeats=n_repeats, random_state=config.RANDOM_SEED)
    return (pd.DataFrame({"feature": feature_list(result.groups), "mdape_increase_pts": r.importances_mean,
                          "std": r.importances_std})
            .sort_values("mdape_increase_pts", ascending=False).reset_index(drop=True))


def _fmt(params: dict) -> str:
    shown = {k: v for k, v in params.items() if k not in ("n_jobs", "early_stopping")}
    return ", ".join(f"{k}={v}" for k, v in shown.items()) or "defaults"
