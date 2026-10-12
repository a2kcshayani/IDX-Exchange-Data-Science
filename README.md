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
| 4 | Linear Regression baseline | [notebooks/03_baseline_model.ipynb](notebooks/03_baseline_model.ipynb), [scripts/run_baseline.py](scripts/run_baseline.py) | ✅ run on real data, results below |
| — | Presentation app (Streamlit) | [app/app.py](app/app.py) | ✅ |
| 5+ | Tree models, feature engineering, boosting, evaluation | — | upcoming |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.14 and the versions pinned in `requirements.txt`.

The pipeline reads monthly `CRMLSSold*.csv` files from `data/raw/` (git-ignored). `split_metadata.json` records which files were used.

## Run

```bash
# Week 3 pipeline end to end (cleaning -> window tuning -> split -> cleaned CSVs)
python scripts/run_preprocessing.py

# Week 4 baseline: train, evaluate, backtest, save model to models/ (git-ignored)
python scripts/run_baseline.py

# Presentation app (reads data/processed/ and models/)
streamlit run app/app.py

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
| `baseline_metrics.json`, `baseline_*.csv` | Week 4 test metrics, before/after comparison, backtest, error by price band and county |

Tests (these use synthetic CRMLS-shaped data, so they run without the real files):

```bash
pytest -q
```

## Week 4 results: Linear Regression baseline

Linear Regression on `log(ClosePrice)`. It trains on the 18 months Oct 2024 – Mar 2026 (189,584 sales) and is tested once on
**April 2026** (12,337 sales it never saw).

| Metric | Before clipping fix | Baseline |
|---|---|---|
| **MdAPE** (headline) | 11.24% | **11.11%** |
| MAPE | 15.79% | 15.57% |
| R² | 0.685 | **0.811** |
| MAE | $219,672 | $212,780 |
| RMSE | $552,984 | $428,721 |

- **Fix found in Week 4:** raw MLS fields contain entry errors, such as a 24.9-billion sq ft lot, 500 garage spaces, and −143 parking
  spaces. A linear model extrapolates these into absurd prices (one prediction was $38.8M). Clipping each numeric feature to its
  0.5th/99.5th percentile **learned on the training set** (`QuantileClipper`) raised R² from 0.685 to 0.811.
- **Stable over time:** in the rolling-origin backtest, MdAPE was 11.12% / 11.11% / 11.11% for test months Feb / Mar / Apr 2026.
- **Accuracy bands:** 24.4% of homes are within 5% of the sale price, 45.7% within 10%, and 74.4% within 20%. The median signed error is +0.1%, so the model is essentially unbiased.
- **Weakness:** the model is least accurate on the top price quintile ($1.67M+), with MdAPE 16.4%, versus 9–10% for $575K–$1.1M homes. A linear model
  can't capture how location and size interact. Tree models come next, in Week 5.
- **Leakage check:** `ListPrice` has a Spearman correlation of 0.993 with `ClosePrice`, and `OriginalListPrice` 0.986. That's why
  they are excluded: they'd inflate accuracy, and they don't exist for off-market homes.

## Presentation app

`streamlit run app/app.py` opens a 6-page app:

1. Overview
2. Data & cleaning: a data funnel, the cleaning log, and EDA charts
3. Train/test design: the time split, window tuning, and the leakage audit
4. Baseline results: metrics, before/after comparison, backtest, and error by price band and county
5. Price estimator: enter a home's details to get a predicted price with its typical range
6. What's next

It reads only local, git-ignored outputs, so run the two scripts above first.

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
- numeric: train-fit percentile clipping + median imputation + missing-indicator flags + standardization
- Y/N flags: mode imputation + missing indicators
- county: one-hot encoding (rare levels grouped)
- city / ZIP: `TargetEncoder` with 5-fold cross-fitting, so a row's encoding never sees its own label
- target: modeled as `log(ClosePrice)` via `TransformedTargetRegressor`

**Metrics** ([src/idx_avm/metrics.py](src/idx_avm/metrics.py)). MdAPE is the headline metric, reported with MAPE, R², MAE, and RMSE, plus a breakdown by price band.

## Layout

```
src/idx_avm/        config, load, clean, features, split, preprocess, window, metrics, pipeline, baseline
scripts/            run_preprocessing.py, run_baseline.py
app/                app.py (Streamlit presentation + price estimator)
notebooks/          01_exploration.ipynb, 02_preprocessing.ipynb, 03_baseline_model.ipynb
docs/               data_dictionary.md
tests/              synthetic data generator + pipeline tests
data/, models/      (git-ignored) raw data, processed outputs, saved model
```

## Assumptions and limitations

- Column names follow the RESO / Trestle standard. Whitelisted columns that are missing from an extract are skipped and reported, not treated as errors.
- `CloseDate` is the valuation date, so property age and seasonality are computed as of the sale.
- `ListingContractDate` is read for one data-quality check only and never becomes a feature.
- A month is treated as partial if the latest close date falls more than 3 days before the end of the month.
