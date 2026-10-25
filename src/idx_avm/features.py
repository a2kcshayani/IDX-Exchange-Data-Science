"""Feature selection, row-wise feature construction, and the leakage audit.

Only row-wise transforms live here (no statistics learned across rows), so
these are safe to apply before splitting. Learned transforms belong in the
sklearn pipeline in preprocess.py.
"""
import numpy as np
import pandas as pd

from . import config

_TRUE = {"true", "t", "yes", "y", "1"}
_FALSE = {"false", "f", "no", "n", "0"}


def _to_bool(s: pd.Series) -> pd.Series:
    """Map Y/N-style values to 1.0/0.0, leaving unknowns as NaN."""
    lowered = s.astype("string").str.strip().str.lower()
    return lowered.map(lambda v: 1.0 if v in _TRUE else 0.0 if v in _FALSE else np.nan).astype(float)


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer row-wise features. Valuation date = CloseDate."""
    df = df.copy()
    close = df[config.DATE_COL]
    if "YearBuilt" in df.columns:
        df["PropertyAge"] = close.dt.year - df["YearBuilt"]
    month = close.dt.month
    df["CloseMonthSin"] = np.sin(2 * np.pi * month / 12)
    df["CloseMonthCos"] = np.cos(2 * np.pi * month / 12)
    for col in config.BOOLEAN_FEATURES:
        if col in df.columns:
            df[col] = _to_bool(df[col])
    if "PostalCode" in df.columns:
        df["PostalCode"] = df["PostalCode"].astype("string").str.extract(r"(\d{5})", expand=False)
    for col in ("CountyOrParish", "City"):
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip().str.title()
    return df


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Week 6 row-wise ratios. Zero/missing denominators give NaN (imputed later)."""
    df = df.copy()
    baths = df["BathroomsTotalInteger"].where(df["BathroomsTotalInteger"] > 0)
    df["BedBathRatio"] = df["BedroomsTotal"] / baths
    sqft = df["LivingArea"].where(df["LivingArea"] > 0)
    df["LotToLivingRatio"] = df["LotSizeSquareFeet"] / sqft
    return df


def feature_columns(
    df: pd.DataFrame, ratios: bool = False, school: bool = False, comps: bool = False
) -> dict[str, list[str]]:
    """Return the whitelisted feature groups that actually exist in df.

    The flags switch on the Week 6 additions so feature sets can be compared:
    ratios (bed/bath, lot/living), school (district spatial join), and comps
    (train-fit ZIP price-per-sq-ft from prior sales, see CompPPSFEncoder).
    """
    # Columns that are entirely empty carry no information (and break median imputation).
    present = {c for c in df.columns if df[c].notna().any()}
    numeric = config.NUMERIC_FEATURES + config.ENGINEERED_NUMERIC + (config.RATIO_FEATURES if ratios else [])
    high_card = config.HIGH_CARD_CATEGORICAL + (config.SCHOOL_FEATURES if school else [])
    groups = {
        "numeric": [c for c in numeric if c in present],
        "boolean": [c for c in config.BOOLEAN_FEATURES if c in present],
        "low_card": [c for c in config.LOW_CARD_CATEGORICAL if c in present],
        "high_card": [c for c in high_card if c in present],
    }
    if comps:
        groups["comp"] = [c for c in config.COMP_INPUTS if c in present]
    audit_features([c for cols in groups.values() for c in cols])
    return groups


def is_leaky(name: str) -> bool:
    lowered = name.lower()
    return name in config.LEAKAGE_COLUMNS or any(p in lowered for p in config.LEAKAGE_PATTERNS)


def audit_features(features: list[str]) -> None:
    """Raise if any model input is a listing/sale-process field (structural leakage guard)."""
    leaky = [f for f in features if is_leaky(f) or f == config.TARGET]
    if leaky:
        raise ValueError(f"Leakage audit failed; these cannot be model inputs: {leaky}")


def leakage_report(df: pd.DataFrame) -> pd.DataFrame:
    """List every raw column flagged as leaky, with its correlation to ClosePrice where numeric."""
    rows = []
    for col in df.columns:
        if col == config.TARGET or not is_leaky(col):
            continue
        corr = np.nan
        if pd.api.types.is_numeric_dtype(df[col]):
            # Spearman: robust to the raw data-entry outliers that swamp Pearson correlation.
            corr = df[[col, config.TARGET]].corr(method="spearman").iloc[0, 1]
        rows.append({"column": col, "spearman_corr_with_ClosePrice": corr, "verdict": "exclude"})
    return pd.DataFrame(rows)
