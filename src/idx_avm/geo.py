"""Week 6: school-district layer via spatial join (CA School District Areas 2024-25).

Each home's coordinates are matched to the district polygons published by the
CA Department of Education. A boundary is a fixed property of the location, so
it exists for any address (on or off market) and uses no sale information.

California has three district types: Unified (K-12), Elementary (K-8) and High
(9-12). A home is in either one Unified district, or one Elementary plus one
High district, so we derive two columns:
  SchoolDistrictK8   - Unified or Elementary district name
  SchoolDistrictHigh - Unified or High district name

Only district identity and type are used. The file's demographic fields
(race/ethnicity shares, etc.) are deliberately excluded: using neighborhood
demographic composition to price homes raises fair-housing concerns.
"""
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from . import config

DISTRICTS_PATH = config.PROJECT_ROOT / "data" / "external" / "school_districts_2425.geojson"
DISTRICTS_URL = (
    "https://gis.data.ca.gov/api/download/v1/items/b0e3b936426a47ce9d9a2e77e2bb86cc/geojson?layers=0"
)
SCHOOL_COLUMNS = ["SchoolDistrictK8", "SchoolDistrictHigh"]


@lru_cache(maxsize=1)
def load_districts(path: Path = DISTRICTS_PATH) -> gpd.GeoDataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"School district boundaries not found at {path}. Download from {DISTRICTS_URL}")
    g = gpd.read_file(path, columns=["DistrictName", "DistrictType", "CountyName", "geometry"])
    # Names repeat across counties (e.g. several "Washington Unified"), so qualify with county.
    g["District"] = g["DistrictName"].str.strip() + " (" + g["CountyName"].str.strip() + ")"
    return g[["District", "DistrictType", "geometry"]].to_crs("EPSG:4326")


def assign_school_districts(df: pd.DataFrame, districts: gpd.GeoDataFrame | None = None) -> pd.DataFrame:
    """Add SchoolDistrictK8 / SchoolDistrictHigh from Latitude/Longitude (NaN when unmatched)."""
    districts = load_districts() if districts is None else districts
    out = df.copy()
    has_xy = out["Latitude"].notna() & out["Longitude"].notna()
    pts = gpd.GeoDataFrame(
        {"row": out.index[has_xy]},
        geometry=gpd.points_from_xy(out.loc[has_xy, "Longitude"], out.loc[has_xy, "Latitude"]),
        crs="EPSG:4326",
    )
    hits = gpd.sjoin(pts, districts, how="inner", predicate="within")
    # A point on a shared border can match two polygons of the same type; keep the first.
    hits = hits.drop_duplicates(["row", "DistrictType"])
    by_type = hits.pivot(index="row", columns="DistrictType", values="District")
    for col in ("Unified", "Elementary", "High"):
        if col not in by_type:
            by_type[col] = np.nan
    out["SchoolDistrictK8"] = by_type["Unified"].fillna(by_type["Elementary"]).reindex(out.index)
    out["SchoolDistrictHigh"] = by_type["Unified"].fillna(by_type["High"]).reindex(out.index)
    return out


def district_for_point(lat: float, lon: float) -> dict[str, str | None]:
    """Look up the districts for one coordinate (used by the app's price estimator)."""
    row = assign_school_districts(pd.DataFrame({"Latitude": [lat], "Longitude": [lon]}))
    return {c: (None if pd.isna(row[c].iloc[0]) else row[c].iloc[0]) for c in SCHOOL_COLUMNS}
