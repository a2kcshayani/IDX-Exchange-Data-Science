"""Project-wide constants: paths, column groups, and the feature/leakage policy.

The modeling goal is to predict ClosePrice for *any* California single-family
address at the time of prediction, whether or not it is on the market. Every
feature must therefore be something knowable for an off-market home.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Override with IDX_RAW_DIR / IDX_PROCESSED_DIR to point at data stored elsewhere.
RAW_DIR = Path(os.environ.get("IDX_RAW_DIR", PROJECT_ROOT / "data" / "raw"))
PROCESSED_DIR = Path(os.environ.get("IDX_PROCESSED_DIR", PROJECT_ROOT / "data" / "processed"))
RAW_GLOB = "CRMLSSold*.csv"

RANDOM_SEED = 42

TARGET = "ClosePrice"
DATE_COL = "CloseDate"

# Task-doc scope filter.
PROPERTY_TYPE = "Residential"
PROPERTY_SUBTYPE = "SingleFamilyResidence"

# Outlier policy (best-practices §03): percentiles learned on train only.
PRICE_LOWER_Q = 0.005
PRICE_UPPER_Q = 0.995

# Rough California bounding box for coordinate sanity checks.
CA_LAT = (32.5, 42.1)
CA_LON = (-124.5, -114.1)

# ---------------------------------------------------------------------------
# Leakage policy. These exist only because a home was listed or sold, so they
# are unavailable for an off-market address. They may be read for data-quality
# checks (e.g. CloseDate before ListingContractDate) but never used as inputs.
# ---------------------------------------------------------------------------
LEAKAGE_COLUMNS = {
    "ListPrice",
    "OriginalListPrice",
    "PreviousListPrice",
    "ListPriceLow",
    "DaysOnMarket",
    "CumulativeDaysOnMarket",
    "PurchaseContractDate",
    "ListingContractDate",
    "OnMarketDate",
    "OffMarketDate",
    "PriceChangeTimestamp",
    "ConcessionsAmount",
    "BuyerAgencyCompensation",
}
# Name patterns that indicate listing/sale-process or post-close information.
LEAKAGE_PATTERNS = (
    "listprice",
    "daysonmarket",
    "contractdate",
    "closeprice",
    "closedate",
    "listagent",
    "listoffice",
    "buyeragent",
    "buyeroffice",
    "concession",
    "pricechange",
    "ppsf",
    "pricepersq",
)

# ---------------------------------------------------------------------------
# Candidate model inputs (whitelist). Only columns present in the data are used.
# All are property characteristics or location, known for any address.
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "LivingArea",
    "BedroomsTotal",
    "BathroomsTotalInteger",
    "LotSizeSquareFeet",
    "Latitude",
    "Longitude",
    "GarageSpaces",
    "ParkingTotal",
    "Stories",
    "FireplacesTotal",
]
BOOLEAN_FEATURES = [
    "PoolPrivateYN",
    "ViewYN",
    "FireplaceYN",
    "AttachedGarageYN",
    "NewConstructionYN",
]
LOW_CARD_CATEGORICAL = ["CountyOrParish"]
HIGH_CARD_CATEGORICAL = ["City", "PostalCode"]

# Engineered in features.add_base_features().
ENGINEERED_NUMERIC = ["PropertyAge", "CloseMonthSin", "CloseMonthCos"]

# Week 6 feature additions (see features.feature_columns flags).
RATIO_FEATURES = ["BedBathRatio", "LotToLivingRatio"]          # features.add_ratio_features()
SCHOOL_FEATURES = ["SchoolDistrictK8", "SchoolDistrictHigh"]   # geo.assign_school_districts()
COMP_INPUTS = ["PostalCode", "LivingArea"]                     # preprocess.CompPPSFEncoder

# Columns used for de-duplication / identity, never as features.
ID_COLUMNS = ["ListingKey", "ListingId", "UnparsedAddress", "ParcelNumber"]
