"""Shared raw -> cleaned dataset steps used by the Week 3 and Week 4 scripts."""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import config
from .clean import CleaningLog, clean
from .features import add_base_features
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


def prepare_dataset(raw_dir: Path = config.RAW_DIR) -> Dataset:
    raw = load_raw(raw_dir)
    sfr = filter_single_family(raw)
    df, log = clean(sfr)
    df = add_close_month(add_base_features(df))
    months = complete_months(df)
    df = df[df["CloseMonth"].isin(months)].reset_index(drop=True)
    return Dataset(df, months, log, sorted(raw["SourceFile"].unique()), len(raw), len(sfr))
