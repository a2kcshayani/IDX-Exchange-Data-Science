"""Tune the training-window length X (task doc, Week 3).

X is chosen without touching the test month: for each candidate X we train on
the X months before a validation month and score that month, repeating over
several validation origins (rolling-origin, best-practices §01) and averaging.
"""
import pandas as pd
from sklearn.linear_model import Ridge

from . import config
from .features import feature_columns
from .metrics import regression_metrics
from .preprocess import build_model_pipeline, feature_frame
from .split import apply_outlier_filter, window_split


def default_regressor():
    # Near-OLS linear model; tiny ridge penalty just keeps one-hot columns stable.
    return Ridge(alpha=1.0)


def evaluate_window(df: pd.DataFrame, val_month: pd.Period, n_train_months: int, regressor_factory=default_regressor) -> dict:
    split, _, _ = apply_outlier_filter(window_split(df, val_month, n_train_months))
    groups = feature_columns(split.train)
    model = build_model_pipeline(groups, regressor_factory())
    model.fit(feature_frame(split.train, groups), split.train[config.TARGET])
    preds = model.predict(feature_frame(split.test, groups))
    return {
        "val_month": str(val_month),
        "X": n_train_months,
        "n_train": len(split.train),
        **regression_metrics(split.test[config.TARGET], preds),
    }


def tune_training_window(
    df: pd.DataFrame,
    months: list[pd.Period],
    candidates: list[int],
    n_origins: int = 3,
    regressor_factory=default_regressor,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score each X at the n_origins months before the test month (months[-1]).

    Returns (per-origin results, summary averaged over origins sorted by MdAPE).
    Candidates that do not fit in the available history at an origin are skipped.
    """
    test_month = months[-1]
    origins = [test_month - k for k in range(n_origins, 0, -1)]
    first = months[0]
    rows = []
    for origin in origins:
        available = (origin - first).n
        for x in candidates:
            if x <= available:
                rows.append(evaluate_window(df, origin, x, regressor_factory))
    results = pd.DataFrame(rows)
    # Only compare X values that were scored at every origin, so averages are like-for-like.
    counts = results.groupby("X")["val_month"].nunique()
    full = counts[counts == counts.max()].index
    summary = (
        results[results["X"].isin(full)]
        .groupby("X")[["MdAPE", "MAPE", "R2", "MAE", "n_train"]]
        .mean()
        .sort_values("MdAPE")
    )
    return results, summary
