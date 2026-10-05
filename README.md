# IDX Exchange — Data Science: California Close Price Prediction

An automated valuation model (AVM) that predicts **`ClosePrice` for any California single-family address
at the time of prediction**, whether or not the home is on the market.

Because the model has to value off-market homes, it only uses information that exists for **any**
address: property characteristics (living area, beds, baths, lot size, age), location (coordinates,
county, city, ZIP), and later, comp-based neighborhood prices built from *prior* sales. Fields that exist
only because a home was listed or sold (`ListPrice`, `OriginalListPrice`, `DaysOnMarket`,
`PurchaseContractDate`, `ListingContractDate`, and anything derived from them) are never model inputs.
See [docs/data_dictionary.md](docs/data_dictionary.md).

> This repo contains code only. Raw and processed data (`data/`, `*.csv`) are git-ignored and must not be committed.

## Status

| Week | Task | Deliverable | Status |
|---|---|---|---|
| 1 | Orientation & setup | [docs/data_dictionary.md](docs/data_dictionary.md), environment | ✅ code done; confirm column notes against the Trestle metadata PDF |
| 2 | Data exploration | [notebooks/01_exploration.ipynb](notebooks/01_exploration.ipynb) | ✅ notebook built; run on real data and fill in findings |
| 3 | Preprocessing + train/test split | [notebooks/02_preprocessing.ipynb](notebooks/02_preprocessing.ipynb), `data/processed/*_clean.csv` | ✅ notebook built; run on real data |
| 4+ | Baseline, model comparison, features, boosting, evaluation, app | — | upcoming |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.14 and the versions pinned in `requirements.txt`.

### Data

1. Connect to the IDX FTP server with FileZilla (credentials come from your onboarding docs; they are
   confidential, so never put them in this repo). The FTP server is **download only**. Never upload or delete anything there.
2. From `raw/California`, download the `CRMLSSold*.csv` files (at least 6 months; more months let the
   training-window search try longer windows) into `data/raw/`.
3. Optionally download `resources/Trestle Property MetaData.pdf` for column definitions.

Record which files you used (the data snapshot). `split_metadata.json` lists them automatically.

## Run

```bash
# Week 3 pipeline end to end (cleaning -> window tuning -> split -> cleaned CSVs)
python scripts/run_preprocessing.py

# Or step through it interactively
jupyter notebook notebooks/
```

Outputs in `data/processed/`:

| File | Contents |
|---|---|
| `train_clean.csv`, `test_clean.csv` | Cleaned, split rows with whitelisted features + target |
| `cleaning_log.csv` | Rows removed at each cleaning step (count and %) |
| `leakage_audit.csv` | Raw columns flagged as leaky and their correlation with ClosePrice |
| `window_tuning_by_origin.csv`, `window_tuning_summary.csv` | Validation scores for each training-window length X |
| `split_metadata.json` | Source files, train/test months, chosen X, outlier cutoffs, feature groups |

Tests (these use synthetic CRMLS-shaped data, so they run without the real files):

```bash
pytest -q
```

## Methodology (Weeks 1–3)

**Scope.** `PropertyType == "Residential"` and `PropertySubType == "SingleFamilyResidence"`.

**Cleaning** ([src/idx_avm/clean.py](src/idx_avm/clean.py)). Every step logs rows before and after:
- Drop rows with missing or ≤ 0 `ClosePrice`, or missing `CloseDate`. The target is never imputed.
- Drop exact duplicates, a `ListingKey` repeated across monthly files (keeping the latest), and repeated address + close date.
- Drop logical impossibilities: close before listing date, living area ≤ 0, < 70 sq ft per bedroom, bed or bath counts outside 0–20.
- Set implausible feature values to NaN (later imputed), but keep the sale: YearBuilt < 1850 or after the sale year, lot ≤ 0, coordinates outside California. Lot size is backfilled from acres when the square-foot value is missing.

**Features** ([src/idx_avm/features.py](src/idx_avm/features.py)). These come from an explicit whitelist of property, location, and
temporal fields: `PropertyAge` at sale, and sale month encoded as sine/cosine. `audit_features` raises an error if a
listing or sale-process column is ever passed to the model.

**Chronological split** ([src/idx_avm/split.py](src/idx_avm/split.py)).
- Test is the most recent *complete* month. A trailing partial month is dropped.
- Train is the **X** months immediately before the test month. Nothing is shuffled.

**Choosing X** ([src/idx_avm/window.py](src/idx_avm/window.py)). For each candidate X, the same pipeline with a linear baseline is trained on
the X months before a *validation* month and scored by MdAPE. This repeats over the 3 months before the
test month (rolling origin), and the X with the lowest mean MdAPE wins. The test month is never used for this choice.
X will be re-checked when stronger models arrive in Weeks 5–7.

**Outliers.** The 0.5th and 99.5th percentiles of `ClosePrice` are learned on the training set only. Those same
frozen cutoffs are then applied to the test set.

**Preprocessing pipeline** ([src/idx_avm/preprocess.py](src/idx_avm/preprocess.py)). This is a scikit-learn `ColumnTransformer`, fit on train only:
- numeric: median imputation + missing-indicator flags + standardization
- Y/N flags: mode imputation + missing indicators
- county: one-hot encoding (rare levels grouped)
- city / ZIP: `TargetEncoder` with 5-fold cross-fitting, so a row's encoding never sees its own label
- target: modeled as `log(ClosePrice)` via `TransformedTargetRegressor`

**Metrics** ([src/idx_avm/metrics.py](src/idx_avm/metrics.py)). MdAPE is the headline metric, reported with MAPE, R², MAE, and RMSE, plus a breakdown by price band.

## Layout

```
src/idx_avm/        config, load, clean, features, split, preprocess, window, metrics
scripts/            run_preprocessing.py
notebooks/          01_exploration.ipynb, 02_preprocessing.ipynb
docs/               data_dictionary.md
tests/              synthetic data generator + pipeline tests
data/               (git-ignored) raw/ and processed/
```

## Assumptions and limitations

- Column names follow the RESO / Trestle standard. Whitelisted columns that are missing from an extract are skipped and reported, not treated as errors.
- `CloseDate` is the valuation date, so property age and seasonality are computed as of the sale.
- `ListingContractDate` is read for one data-quality check only and never becomes a feature.
- A month is treated as partial if the latest close date falls more than 3 days before the end of the month.
