"""Generate small CRMLSSold-shaped CSVs so the pipeline can be tested without real data."""
from pathlib import Path

import numpy as np
import pandas as pd

CITIES = [("Los Angeles", "Los Angeles", "900"), ("Irvine", "Orange", "926"), ("Riverside", "Riverside", "925"),
          ("San Diego", "San Diego", "921"), ("Pasadena", "Los Angeles", "911")]


def make_month(year: int, month: int, n: int, rng: np.random.Generator) -> pd.DataFrame:
    city_idx = rng.integers(0, len(CITIES), n)
    city = [CITIES[i] for i in city_idx]
    sqft = rng.lognormal(7.5, 0.35, n).round()
    beds = np.clip((sqft / 600).round() + rng.integers(-1, 2, n), 1, 8)
    year_built = rng.integers(1920, 2024, n)
    loc_mult = np.array([1.6, 1.8, 0.9, 1.4, 1.5])[city_idx]
    price = (sqft * 350 * loc_mult * rng.lognormal(0, 0.15, n)).round(-3)
    close_day = rng.integers(1, 29, n)
    close = pd.to_datetime(dict(year=year, month=month, day=close_day))
    list_price = (price * rng.normal(1.02, 0.03, n)).round(-3)
    df = pd.DataFrame({
        "ListingKey": [f"K{year}{month:02d}{i:05d}" for i in range(n)],
        "PropertyType": "Residential",
        "PropertySubType": rng.choice(["SingleFamilyResidence", "Condominium"], n, p=[0.85, 0.15]),
        "ClosePrice": price,
        "ListPrice": list_price,
        "OriginalListPrice": list_price * 1.03,
        "DaysOnMarket": rng.integers(1, 120, n),
        "CloseDate": close.dt.strftime("%Y-%m-%d"),
        "ListingContractDate": (close - pd.to_timedelta(rng.integers(10, 120, n), unit="D")).dt.strftime("%Y-%m-%d"),
        "LivingArea": sqft,
        "BedroomsTotal": beds,
        "BathroomsTotalInteger": np.clip(beds - rng.integers(0, 2, n), 1, 6),
        "LotSizeSquareFeet": np.where(rng.random(n) < 0.1, np.nan, rng.lognormal(8.7, 0.5, n).round()),
        "YearBuilt": year_built,
        "Latitude": 34 + rng.normal(0, 0.5, n),
        "Longitude": -118 + rng.normal(0, 0.5, n),
        "City": [c[0] for c in city],
        "CountyOrParish": [c[1] for c in city],
        "PostalCode": [f"{c[2]}{rng.integers(10, 99)}" for c in city],
        "GarageSpaces": rng.integers(0, 4, n),
        "PoolPrivateYN": rng.choice(["True", "False", None], n),
        "ViewYN": rng.choice(["True", "False"], n),
    })
    # Inject data-quality problems the cleaner must catch.
    df.loc[df.index[:3], "ClosePrice"] = 0
    df.loc[df.index[3:5], "LivingArea"] = 0
    df.loc[df.index[5], "YearBuilt"] = 3020
    return pd.concat([df, df.iloc[[10, 11]]], ignore_index=True)  # duplicates


def write_synthetic(out_dir: Path, months: int = 12, n: int = 400, seed: int = 0) -> list[Path]:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for k in range(months):
        y, m = 2025 + (k // 12), k % 12 + 1
        p = out_dir / f"CRMLSSold{y}{m:02d}.csv"
        make_month(y, m, n, rng).to_csv(p, index=False)
        paths.append(p)
    return paths
