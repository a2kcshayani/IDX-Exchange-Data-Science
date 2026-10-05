"""Row-level data-quality cleaning (best-practices §02).

Every step is recorded in a CleaningLog so the row count and percentage removed
at each stage can be reported. These rules use no learned statistics, so they
are safe to apply before the train/test split. Anything learned from data
(outlier percentiles, imputation values) lives in split.py / preprocess.py and
is fit on the training set only.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config


@dataclass
class CleaningLog:
    steps: list[dict] = field(default_factory=list)

    def record(self, step: str, before: int, after: int, note: str = "") -> None:
        removed = before - after
        self.steps.append(
            {
                "step": step,
                "rows_before": before,
                "rows_after": after,
                "removed": removed,
                "pct_removed": round(100 * removed / before, 3) if before else 0.0,
                "note": note,
            }
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)


def _drop(df: pd.DataFrame, mask: pd.Series, log: CleaningLog, step: str, note: str = "") -> pd.DataFrame:
    before = len(df)
    out = df.loc[~mask].copy()
    log.record(step, before, len(out), note)
    return out


def _nullify(df: pd.DataFrame, col: str, mask: pd.Series, log: CleaningLog, step: str) -> pd.DataFrame:
    """Set implausible feature values to NaN (later imputed) rather than dropping the row."""
    n = int(mask.sum())
    df.loc[mask, col] = np.nan
    log.record(step, len(df), len(df), f"{n} values set to NaN in {col}")
    return df


def clean(df: pd.DataFrame, log: CleaningLog | None = None) -> tuple[pd.DataFrame, CleaningLog]:
    log = log or CleaningLog()
    log.record("start (single-family residential)", len(df), len(df))

    # Target validity. The target is never imputed.
    df = _drop(df, df[config.TARGET].isna(), log, "missing ClosePrice")
    df = _drop(df, df[config.TARGET] <= 0, log, "ClosePrice <= 0")
    df = _drop(df, df[config.DATE_COL].isna(), log, "missing CloseDate")

    # Duplicates: exact rows, then the same listing appearing in multiple monthly files.
    df = _drop(df, df.duplicated(), log, "exact duplicate rows")
    if "ListingKey" in df.columns:
        sort_col = "ModificationTimestamp" if "ModificationTimestamp" in df.columns else config.DATE_COL
        df = df.sort_values(sort_col)
        df = _drop(df, df.duplicated("ListingKey", keep="last"), log, "duplicate ListingKey", "kept latest record")
    if "UnparsedAddress" in df.columns:
        dup = df["UnparsedAddress"].notna() & df.duplicated(["UnparsedAddress", config.DATE_COL], keep="last")
        df = _drop(df, dup, log, "duplicate address + CloseDate")

    # Logical impossibilities.
    if "ListingContractDate" in df.columns:
        bad = df["ListingContractDate"].notna() & (df[config.DATE_COL] < df["ListingContractDate"])
        df = _drop(df, bad, log, "CloseDate before ListingContractDate", "date used for QA only, never as a feature")
    if "LivingArea" in df.columns:
        df = _drop(df, df["LivingArea"] <= 0, log, "LivingArea <= 0")
        # < 70 sq ft per bedroom is not a habitable layout -> data-entry error.
        if "BedroomsTotal" in df.columns:
            cramped = (df["BedroomsTotal"] > 0) & (df["LivingArea"] / df["BedroomsTotal"] < 70)
            df = _drop(df, cramped, log, "LivingArea inconsistent with bedrooms (<70 sqft/bed)")
    for col, cap in (("BedroomsTotal", 20), ("BathroomsTotalInteger", 20)):
        if col in df.columns:
            df = _drop(df, (df[col] < 0) | (df[col] > cap), log, f"{col} outside [0, {cap}]")

    # Implausible feature values -> NaN so the imputer handles them.
    close_year = df[config.DATE_COL].dt.year
    if "YearBuilt" in df.columns:
        bad = (df["YearBuilt"] < 1850) | (df["YearBuilt"] > close_year + 1)
        df = _nullify(df, "YearBuilt", bad, log, "YearBuilt implausible")
    if "LotSizeSquareFeet" in df.columns:
        if "LotSizeAcres" in df.columns:
            fill = df["LotSizeSquareFeet"].isna() & df["LotSizeAcres"].notna()
            df.loc[fill, "LotSizeSquareFeet"] = df.loc[fill, "LotSizeAcres"] * 43_560
            log.record("LotSizeSquareFeet backfilled from LotSizeAcres", len(df), len(df), f"{int(fill.sum())} values")
        df = _nullify(df, "LotSizeSquareFeet", df["LotSizeSquareFeet"] <= 0, log, "LotSizeSquareFeet <= 0")
    if {"Latitude", "Longitude"} <= set(df.columns):
        outside = ~df["Latitude"].between(*config.CA_LAT) | ~df["Longitude"].between(*config.CA_LON)
        outside &= df["Latitude"].notna() | df["Longitude"].notna()
        for col in ("Latitude", "Longitude"):
            df.loc[outside, col] = np.nan
        log.record("coordinates outside California", len(df), len(df), f"{int(outside.sum())} rows set to NaN")

    return df.reset_index(drop=True), log
