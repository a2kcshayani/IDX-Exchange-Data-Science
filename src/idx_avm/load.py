"""Load the monthly CRMLSSold CSV extracts into one DataFrame."""
from pathlib import Path

import pandas as pd

from . import config

DATE_COLUMNS = ["CloseDate", "ListingContractDate", "PurchaseContractDate", "ModificationTimestamp"]


def list_raw_files(raw_dir: Path = config.RAW_DIR) -> list[Path]:
    files = sorted(Path(raw_dir).glob(config.RAW_GLOB))
    if not files:
        raise FileNotFoundError(
            f"No {config.RAW_GLOB} files in {raw_dir}. Download them from the FTP "
            "server (raw/California) with FileZilla and place them there."
        )
    return files


def load_raw(raw_dir: Path = config.RAW_DIR) -> pd.DataFrame:
    """Concatenate every CRMLSSold*.csv, tagging each row with its source file."""
    frames = []
    for path in list_raw_files(raw_dir):
        df = pd.read_csv(path, low_memory=False)
        df["SourceFile"] = path.name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def filter_single_family(df: pd.DataFrame) -> pd.DataFrame:
    """Keep PropertyType == Residential and PropertySubType == SingleFamilyResidence."""
    mask = (df["PropertyType"] == config.PROPERTY_TYPE) & (
        df["PropertySubType"] == config.PROPERTY_SUBTYPE
    )
    return df.loc[mask].copy()
