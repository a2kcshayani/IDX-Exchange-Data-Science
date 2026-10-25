"""Shared raw -> cleaned dataset steps used by the Week 3 and Week 4 scripts."""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import config
from .clean import CleaningLog, clean
from .features import add_base_features, add_ratio_features
from .load import filter_single_family, load_raw
from .split import add_close_month, complete_months


@dataclass
class Dataset:
    df: pd.DataFrame            # cleaned single-family rows in complete months
    months: list[pd.Period]     # complete months, oldest first; months[-1] is the test month
    log: CleaningLog
    source_files: list[str]
    n_raw: int
    n_single_family: int


def prepare_dataset(raw_dir: Path = config.RAW_DIR, week6_features: bool = False) -> Dataset:
    """week6_features adds the row-wise ratios and school-district spatial join.

    Both are fixed per row (no statistics learned across rows), so adding them
    before the split is leakage-safe; models only use them when feature_columns()
    is called with ratios=True / school=True.
    """
    raw = load_raw(raw_dir)
    sfr = filter_single_family(raw)
    df, log = clean(sfr)
    df = add_close_month(add_base_features(df))
    if week6_features:
        from .geo import assign_school_districts

        df = assign_school_districts(add_ratio_features(df))
    months = complete_months(df)
    df = df[df["CloseMonth"].isin(months)].reset_index(drop=True)
    return Dataset(df, months, log, sorted(raw["SourceFile"].unique()), len(raw), len(sfr))
