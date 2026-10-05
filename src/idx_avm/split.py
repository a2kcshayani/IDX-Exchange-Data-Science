"""Chronological splitting and train-only outlier thresholds (best-practices §01, §03).

Layout for a test month T and training window of X months:

    [ T-X-1 ... T-2 ] [ T-1 ]   [ T ]
      train (tuning)    valid     test   <- choose X / hyperparameters on valid
    [ T-X ...   T-1 ]           [ T ]
      train (final)               test   <- refit on the X months right before T

The test month is touched once, at the end.
"""
from dataclasses import dataclass

import pandas as pd

from . import config


def add_close_month(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["CloseMonth"] = df[config.DATE_COL].dt.to_period("M")
    return df


def complete_months(df: pd.DataFrame) -> list[pd.Period]:
    """Sorted months present in the data, dropping the last one if it is partial.

    A month counts as partial when the latest CloseDate falls before its final
    day minus 3 days (extract taken mid-month).
    """
    months = sorted(df["CloseMonth"].unique())
    last = months[-1]
    max_date = df[config.DATE_COL].max()
    if max_date < last.end_time.normalize() - pd.Timedelta(days=3):
        months = months[:-1]
    return months


@dataclass
class Split:
    train: pd.DataFrame
    test: pd.DataFrame
    train_months: list[pd.Period]
    test_month: pd.Period


def window_split(df: pd.DataFrame, test_month: pd.Period, n_train_months: int) -> Split:
    """Train on the n months immediately preceding test_month; test on test_month."""
    train_months = [test_month - i for i in range(n_train_months, 0, -1)]
    train = df[df["CloseMonth"].isin(train_months)].copy()
    test = df[df["CloseMonth"] == test_month].copy()
    if train.empty or test.empty:
        raise ValueError(f"Empty split for test={test_month}, X={n_train_months}")
    return Split(train, test, train_months, test_month)


@dataclass
class PriceThresholds:
    lower: float
    upper: float

    @classmethod
    def fit(cls, train: pd.DataFrame) -> "PriceThresholds":
        q = train[config.TARGET].quantile([config.PRICE_LOWER_Q, config.PRICE_UPPER_Q])
        return cls(lower=float(q.iloc[0]), upper=float(q.iloc[1]))

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df[config.TARGET].between(self.lower, self.upper)].copy()


def apply_outlier_filter(split: Split) -> tuple[Split, PriceThresholds, dict]:
    """Learn 0.5/99.5 pct ClosePrice cutoffs on train; apply the frozen cutoffs to both sets."""
    thresholds = PriceThresholds.fit(split.train)
    train, test = thresholds.apply(split.train), thresholds.apply(split.test)
    stats = {
        "lower": thresholds.lower,
        "upper": thresholds.upper,
        "train_removed": len(split.train) - len(train),
        "test_removed": len(split.test) - len(test),
    }
    return Split(train, test, split.train_months, split.test_month), thresholds, stats
