"""List every model feature: what it is, where it came from, and which feature set uses it.

Reads the cleaned training split so the list reflects the columns that actually exist in
the data (whitelisted columns missing from an extract are reported, not listed as used).

Usage:
    python scripts/list_features.py            # print to the terminal
    python scripts/list_features.py --markdown # also write docs/features.md
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idx_avm import config  # noqa: E402
from idx_avm.features import add_ratio_features, feature_columns  # noqa: E402
from idx_avm.load import load_processed  # noqa: E402

# feature -> (description, how it's built)
DESCRIPTIONS = {
    "LivingArea": ("Finished living space (sq ft)", "MLS field; clipped to train 0.5–99.5 pct"),
    "BedroomsTotal": ("Number of bedrooms", "MLS field"),
    "BathroomsTotalInteger": ("Number of bathrooms", "MLS field"),
    "LotSizeSquareFeet": ("Lot size (sq ft)", "MLS field; backfilled from LotSizeAcres × 43,560"),
    "Latitude": ("Latitude", "MLS geocode; NaN if outside California"),
    "Longitude": ("Longitude", "MLS geocode; NaN if outside California"),
    "GarageSpaces": ("Garage spaces", "MLS field"),
    "ParkingTotal": ("Total parking spaces", "MLS field"),
    "Stories": ("Number of floors", "MLS field"),
    "FireplacesTotal": ("Number of fireplaces", "MLS field"),
    "PropertyAge": ("Age at sale (years)", "Sale year − YearBuilt"),
    "CloseMonthSin": ("Season, sine part", "sin(2π · month / 12), so Dec and Jan sit next to each other"),
    "CloseMonthCos": ("Season, cosine part", "cos(2π · month / 12)"),
    "PoolPrivateYN": ("Has a private pool", "Y/N → 1/0, plus a missing flag"),
    "ViewYN": ("Has a view", "Y/N → 1/0, plus a missing flag"),
    "FireplaceYN": ("Has a fireplace", "Y/N → 1/0, plus a missing flag"),
    "AttachedGarageYN": ("Garage is attached", "Y/N → 1/0, plus a missing flag"),
    "NewConstructionYN": ("Newly built", "Y/N → 1/0, plus a missing flag"),
    "CountyOrParish": ("County", "One-hot encoded (rare counties grouped)"),
    "City": ("City", "Target-encoded, 5-fold cross-fitted"),
    "PostalCode": ("ZIP code", "Target-encoded, 5-fold cross-fitted"),
    "BedBathRatio": ("Bedrooms ÷ bathrooms", "Week 6 ratio; NaN when bathrooms = 0"),
    "LotToLivingRatio": ("Lot size ÷ living area", "Week 6 ratio; how much yard per sq ft of house"),
    "SchoolDistrictK8": ("K-8 school district", "Week 6 spatial join to CA School District Areas 2024-25; target-encoded"),
    "SchoolDistrictHigh": ("High-school district", "Week 6 spatial join to CA School District Areas 2024-25; target-encoded"),
    "comp_log_ppsf": ("Neighborhood $/sq ft from prior sales", "Week 6 comps: ZIP median of training-window sales, cross-fitted"),
    "comp_log_value": ("Comps-implied home value", "Week 6 comps: neighborhood $/sq ft × this home's sq ft"),
}

EXCLUDED = [
    ("ListPrice, OriginalListPrice", "Only exist for listed homes; 0.99 correlation with sale price (leaks the answer)"),
    ("DaysOnMarket, CumulativeDaysOnMarket", "Only exist after a home is listed"),
    ("ListingContractDate, PurchaseContractDate", "Sale-process dates; with CloseDate they rebuild days on market"),
    ("Anything derived from the above", "e.g. sale-to-list ratio, list price per sq ft"),
    ("PPSF computed from the home's own ClosePrice", "Uses the target; replaced by comps from prior sales"),
    ("Agent / office / brokerage fields", "Reflect pricing strategy, not the home itself"),
    ("TaxAnnualAmount", "Can be reassessed after a sale; timing not verified"),
    ("School-district demographics (race/ethnicity shares)", "Fair-housing risk; only district identity is used"),
]

SETS = [
    ("Base (Week 4)", {}),
    ("+ Ratios", {"ratios": True}),
    ("+ School districts", {"ratios": True, "school": True}),
    ("+ Comp PPSF", {"ratios": True, "school": True, "comps": True}),
]


def raw_features(groups: dict) -> list[str]:
    cols = [c for k, cs in groups.items() if k != "comp" for c in cs]
    if groups.get("comp"):
        cols += ["comp_log_ppsf", "comp_log_value"]
    return list(dict.fromkeys(cols))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown", action="store_true", help="also write docs/features.md")
    args = parser.parse_args()

    train = add_ratio_features(load_processed("train"))
    # Spatial join only for the column list (fast); fall back to assuming it exists.
    try:
        from idx_avm.geo import assign_school_districts

        train = assign_school_districts(train)
    except FileNotFoundError:
        train["SchoolDistrictK8"] = train["SchoolDistrictHigh"] = "unknown"

    first_set, rows = {}, []
    for set_name, flags in SETS:
        for f in raw_features(feature_columns(train, **flags)):
            first_set.setdefault(f, set_name)
    best = {}
    best_path = config.PROCESSED_DIR / "best_model_metrics.json"
    if best_path.exists():
        best = json.loads(best_path.read_text())
    selected = dict(SETS).get(best.get("selected_feature_set"), None)
    selected_cols = set(raw_features(feature_columns(train, **selected))) if selected is not None else set()

    for f, set_name in first_set.items():
        desc, how = DESCRIPTIONS.get(f, (f, ""))
        rows.append((set_name, f, desc, how, "yes" if f in selected_cols else "no"))

    whitelist = (config.NUMERIC_FEATURES + config.BOOLEAN_FEATURES + config.LOW_CARD_CATEGORICAL
                 + config.HIGH_CARD_CATEGORICAL)
    absent = [c for c in whitelist if c not in first_set]

    # ---------- terminal output
    model_label = f"{best.get('selected_model')} ({best.get('selected_feature_set')})" if best else "n/a"
    print(f"Model features: {len(rows)} total | selected model: {model_label}\n")
    current = None
    for set_name, f, desc, how, used in rows:
        if set_name != current:
            current = set_name
            print(f"== {set_name}")
        mark = "✓" if used == "yes" else " "
        print(f"  {mark} {f:24s} {desc:40s} {how}")
    print("\n  ✓ = used by the selected model")
    if absent:
        print(f"\nWhitelisted but empty or missing in this extract (not used): {', '.join(absent)}")
    print("\n== Deliberately excluded")
    for name, why in EXCLUDED:
        print(f"  ✗ {name:52s} {why}")

    # ---------- markdown output
    if args.markdown:
        lines = ["# Model Features", "",
                 f"Generated by `scripts/list_features.py`. Selected model: **{model_label}**.", "",
                 "| Added in | Feature | What it is | How it's built | In selected model |",
                 "|---|---|---|---|---|"]
        lines += [f"| {s} | `{f}` | {d} | {h} | {'✓' if u == 'yes' else ''} |" for s, f, d, h, u in rows]
        if absent:
            lines += ["", f"Whitelisted but empty or missing in this extract (not used): {', '.join(f'`{a}`' for a in absent)}"]
        lines += ["", "## Deliberately excluded", "", "| Field | Why |", "|---|---|"]
        lines += [f"| {n} | {w} |" for n, w in EXCLUDED]
        out = config.PROJECT_ROOT / "docs" / "features.md"
        out.write_text("\n".join(lines) + "\n")
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
