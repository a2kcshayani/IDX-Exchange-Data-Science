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
| 5 | Decision Tree + Random Forest vs. baseline | [notebooks/04_model_comparison.ipynb](notebooks/04_model_comparison.ipynb), [scripts/run_models.py](scripts/run_models.py) | ✅ run on real data, results below |
| 6 | Feature engineering incl. school-district spatial join | [notebooks/04b_feature_engineering.ipynb](notebooks/04b_feature_engineering.ipynb), [src/idx_avm/geo.py](src/idx_avm/geo.py) | ✅ run on real data, results below |
| 7 | Gradient boosting + light tuning | [notebooks/05_advanced_models.ipynb](notebooks/05_advanced_models.ipynb), [scripts/run_advanced_models.py](scripts/run_advanced_models.py) | ✅ run on real data, results below |
| 8 | Evaluation expansion (MAPE, MdAPE, price bands) | [notebooks/06_evaluation.ipynb](notebooks/06_evaluation.ipynb), [scripts/run_evaluation.py](scripts/run_evaluation.py), `metrics_summary.csv` | ✅ run on real data, results below |
| 9–12 | App polish, documentation, presentation | — | upcoming |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.14 and the versions pinned in `requirements.txt`.

The pipeline reads monthly `CRMLSSold*.csv` files from `data/raw/` (git-ignored). `split_metadata.json` records which files were used.

Week 6 also needs the public [CA School District Areas 2024-25](https://data.ca.gov/dataset/california-school-district-areas-2024-25) boundaries (CA Dept. of Education, ~45 MB GeoJSON):

```bash
mkdir -p data/external && curl -L -o data/external/school_districts_2425.geojson \
  "https://gis.data.ca.gov/api/download/v1/items/b0e3b936426a47ce9d9a2e77e2bb86cc/geojson?layers=0"
```

## Run

```bash
# Week 3 pipeline end to end (cleaning -> window tuning -> split -> cleaned CSVs)
python scripts/run_preprocessing.py

# Week 4 baseline: train, evaluate, backtest, save model to models/ (git-ignored)
python scripts/run_baseline.py

# Weeks 5-6: tune tree models, compare feature sets, save the selected model (~10 min)
python scripts/run_models.py

# Week 7: tune gradient boosting, compare on every feature set, re-select and save the best model (~15 min)
python scripts/run_advanced_models.py

# Week 8: score the finalists side by side; writes metrics_summary.csv and error breakdowns (~5 min)
python scripts/run_evaluation.py

# Presentation app: one command that builds anything missing, then launches on localhost
bash scripts/run_dashboard.sh

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
| `week5_*.csv`, `week6_feature_sets.csv` | Tuning trials, model comparison, every model × feature set (validation and test) |
| `week6_selection.json`, `rf_feature_importance.csv` | The Week 6 decision and Random Forest importances |
| `week7_tuning_trials.csv`, `week7_gb_feature_sets.csv`, `week7_model_comparison.csv` | Gradient boosting tuning, feature sets, and the side-by-side table |
| `best_model_*.csv`, `best_model_metrics.json`, `best_model_permutation_importance.csv` | Current selected model's metrics, backtest, error breakdowns, importances |
| `tuned_params.json` | Validation-tuned settings for every model |
| **`metrics_summary.csv`** | Week 8 deliverable: one row per finalist with R², MAPE, MdAPE, MAE, RMSE, accuracy bands, bias, backtest range |
| `eval_segments.csv`, `eval_backtest.csv`, `eval_significance.csv`, `eval_insights.json` | Week 8 error by price band / county / size / age, 3-month backtest, bootstrap test, findings |

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

## Week 5 results: tree models vs. the baseline

Same features, same pipeline, same log target. Settings tuned on the **validation month (Mar 2026)**, then scored once on
the **test month (Apr 2026)**, 12,337 sales.

| Model | Chosen settings | MdAPE | MAPE | R² | MAE | Train time |
|---|---|---|---|---|---|---|
| Linear Regression (baseline) | — | 11.11% | 15.57% | 0.811 | $212,780 | 1 s |
| Decision Tree | depth 20, min leaf 20 | 9.55% | 14.07% | 0.847 | $192,747 | 3 s |
| **Random Forest** | 150 trees, min leaf 2, 33% of features per split | **7.78%** | **11.65%** | **0.885** | **$161,694** | 14 s |

- The Random Forest cuts typical error by **3.3 points (30%)** versus the baseline. That margin is far larger than the
  month-to-month noise seen in the backtests (about 0.2 points).
- Trees win because they learn interactions on their own, like "big house × expensive ZIP", which a linear model can't.
- A single Decision Tree is jumpy (high variance). Averaging 150 de-correlated trees fixes most of that.

## Week 6 results: feature engineering

Six cumulative feature sets, each re-trained with all three models (test-month MdAPE, with R² in brackets):

| Feature set | Linear Regression | Decision Tree | Random Forest |
|---|---|---|---|
| Base (Week 4) | 11.11% (0.811) | 9.55% (0.847) | 7.78% (0.885) |
| + Bed/bath and lot/living ratios | 11.12% (0.811) | 9.63% (0.847) | 7.82% (0.882) |
| + School districts (spatial join) | 11.14% (0.813) | 9.70% (0.851) | 7.81% (0.885) |
| + Distance to nearest job center | 11.03% (0.813) | 9.65% (0.848) | 7.70% (0.888) |
| + ZIP comps (ZIP $/sq ft, prior sales) | 9.87% (0.843) | 9.54% (0.849) | 7.76% (0.888) |
| + Nearby comps (10 closest prior sales) | **9.61% (0.871)** | 9.54% (0.870) | 7.80% (**0.902**) |

- **School districts:** a spatial join of each home's coordinates against the CA School District Areas 2024-25 polygons
  matched over 99% of homes (99.3% of the test month; the rest have no coordinates) to 469 K-8 and 321 high-school districts.
- **Comps, done leakage-safe:** both comp features use only *prior* sales in the training window, are fit inside the
  pipeline, and are cross-fitted so no home ever sees its own price. This replaces PPSF computed from ClosePrice. The
  nearby version follows the IDX memo's "comp-based PPSF from nearby sales": the 10 closest prior sales by distance
  (`NearbyCompsEncoder`), instead of a ZIP-wide median (`CompPPSFEncoder`).
- **What helped:** nearby comps are the strongest addition. They cut Linear Regression error from 11.11% to 9.61% (R² 0.811 → 0.871)
  and give the Random Forest the best R² of any configuration (0.902), meaning fewer large misses.
  The Random Forest's *typical* error doesn't move: it was already learning neighborhood price levels from location, and the
  comps now carry most of its importance (nearby comps value 38%, ZIP comps value 21%).
- **Selection rule (best practices §09):** an addition has to beat noise to be kept. Validation-MdAPE differences under 0.10
  points (about half the month-to-month backtest spread) count as ties, and ties go to the simpler setup. The lowest
  validation error was Random Forest + job-center distance (7.64%), only 0.08 points ahead of the base Random Forest (7.72%), so
  the **selected model is the Random Forest on the Week 4 base features**.
- **Selected model on the test month:** **7.78% MdAPE, R² 0.885, MAE $161,694**. Backtest: 7.99% / 7.72% / 7.78% for Feb / Mar / Apr 2026.
- **Accuracy bands:** 34.9% of homes within 5% of the sale price (baseline: 24.4%), **59.4% within 10%** (45.7%), 84.3% within 20% (74.4%).
- **Luxury homes improved the most:** MdAPE on $1.67M+ homes fell from 16.4% to 12.1%, and on $575K–$800K homes from 9.1% to 6.0%.
- **Excluded on purpose:** the district file's demographic fields (race/ethnicity shares). Pricing homes on neighborhood
  demographic composition raises fair-housing concerns, so only district identity is used.

Full list of features, how each is built, and what's excluded: [docs/features.md](docs/features.md).

## Week 7 results: gradient boosting

`HistGradientBoostingRegressor` (scikit-learn's histogram gradient boosting, the LightGBM algorithm family). LightGBM and
XGBoost were tried first. Both need a system OpenMP library on macOS (`brew install libomp`), so the built-in version
keeps the repo runnable with only `pip install -r requirements.txt`.

**Light tuning on the validation month (Mar 2026):** 12 settings, depth {6, 10, unlimited} × learning rate {0.05, 0.1} ×
trees {500, 1000}, with up to 127 leaves per tree. Validation MdAPE ranged from 8.20% to 7.35%. More trees, a higher learning rate and deeper
trees all helped. The winner was **1,000 trees, learning rate 0.1, no depth limit**.

Baseline and advanced models, same table (base features, test month Apr 2026, 12,337 sales):

| Model | MdAPE | MAPE | R² | MAE | RMSE | vs. baseline |
|---|---|---|---|---|---|---|
| Linear Regression (baseline) | 11.11% | 15.57% | 0.811 | $212,780 | $428,721 | — |
| Decision Tree | 9.55% | 14.07% | 0.847 | $192,747 | $385,207 | −14% |
| Random Forest | 7.78% | 11.65% | 0.885 | $161,694 | $334,836 | −30% |
| **Gradient Boosting** | **7.50%** | **11.13%** | **0.904** | **$150,331** | **$305,441** | **−33%** |

Gradient boosting on the Week 6 feature sets (test MdAPE, R²): base 7.50% (0.904), + ratios 7.52%, + school districts 7.55%,
+ job-center distance 7.43%, + ZIP comps 7.39%, + nearby comps **7.37% (0.915)**, the best R² of any configuration.

- **Selection** (same rule as Week 6, over every model × feature set from Weeks 5–7): the lowest validation MdAPE was gradient
  boosting + job-center distance (7.32%), only 0.03 points ahead of gradient boosting on the base features (7.35%), so the
  **selected model is gradient boosting on the Week 4 base features**. It beats the Week 6 Random Forest by 0.37 validation points,
  well past the 0.10 noise threshold.
- **Backtest:** 7.66% / 7.35% / 7.50% MdAPE for Feb / Mar / Apr 2026 (Random Forest: 7.99% / 7.72% / 7.78%).
- **Smaller and faster:** the saved model is **6.6 MB instead of 349 MB**, so the dashboard estimator loads almost instantly.
  Training takes about a minute, versus about 8 seconds for the Random Forest.
- **What it relies on** (permutation importance, i.e. how much test MdAPE rises when one input is shuffled): longitude +17.0 pts,
  ZIP +16.7, latitude +14.7, living area +8.4, city +5.7, lot size +2.4. Location dominates, then size.
- The winning settings sit at the edge of the grid (most trees, highest learning rate), so a slightly larger search may help.

## Week 8 results: evaluation expansion

All finalists scored the same way on the same 12,337 April 2026 homes (`metrics_summary.csv`):

| Model | MdAPE | MAPE | R² | MAE | Within 10% | Median bias | Backtest MdAPE |
|---|---|---|---|---|---|---|---|
| Linear Regression (baseline) | 11.11% | 15.57% | 0.811 | $212,780 | 45.7% | +0.1% | 11.11–11.12% |
| Random Forest | 7.78% | 11.65% | 0.885 | $161,694 | 59.4% | −1.0% | 7.72–7.99% |
| **Gradient Boosting (selected)** | **7.50%** | **11.13%** | **0.904** | **$150,331** | **61.0%** | **−0.2%** | **7.35–7.66%** |
| Gradient Boosting + all features | 7.37% | 10.91% | 0.915 | $145,339 | 62.0% | −0.4% | 7.32–7.49% |

**Which price bands perform better** (MdAPE by sale-price quintile, baseline → selected model):

| Band | Baseline | Selected | Within 10% (selected) |
|---|---|---|---|
| Q1: $190K–$575K | 10.4% | 7.1% | 61.8% |
| Q2: $575K–$800K | 9.1% | **5.9%** | 71.5% |
| Q3: $802K–$1.10M | 9.9% | 6.5% | 68.4% |
| Q4: $1.10M–$1.67M | 11.6% | 8.4% | 56.0% |
| Q5: $1.67M–$8.4M | 16.4% | **10.8%** | 47.2% |

- **Mid-priced homes are the sweet spot.** $575K–$1.1M homes have the most comparable sales and the lowest error. Luxury homes are still
  the hardest (10.8%), but that's also where gradient boosting gained the most over the baseline (5.5 points).
- **MdAPE vs. MAPE:** mean error (11.1%) is well above median error (7.5%) because a minority of large misses pulls the average up.
  That's why MdAPE is the headline and RMSE / R² are the tail-risk view.
- **Bias:** grouped by sale price, the cheapest homes look over-estimated (+2.2%) and the most expensive under-estimated (−4.5%).
  Grouped by the model's own estimate, which is what a user actually sees, the bias is −0.2% to −0.4% at both ends. So the
  sale-price pattern is mostly regression to the middle, not a fixed bias. The Random Forest's overall −1.0% bias is gone (−0.2%).
- **Geography:** lowest error in the Inland Empire (San Bernardino 6.1%, Riverside 6.2%). Highest in the Bay Area (San Mateo
  11.7%, Alameda 11.1%), the one place the Random Forest did better (10.8% and 10.0%). Adding the nearby-comps features brings those
  to 10.0% and 9.8%.
- **Size and age:** error rises for very large homes (3,500+ sq ft: 10.7%) and older homes (70+ years: 9.4% vs. 5.2% for homes under 10 years old).
  Unusual and older homes have fewer true comparables.
- **Is it bigger than noise?** A paired bootstrap on the same test homes (2,000 resamples) puts the gain over the baseline at
  −3.62 points (95% CI −3.88 to −3.36) and over the Random Forest at −0.28 points (−0.44 to −0.11). Both intervals exclude zero.
  Gradient boosting + all features vs. the selected base version is +0.12 points (−0.02 to +0.32), which is within noise. That's consistent
  with the validation-based selection rule.

`metrics_summary.csv` and the other outputs are aggregate tables but stay git-ignored, per the repo's no-CSV rule.
Running `scripts/run_evaluation.py` recreates them.

## Presentation app

`bash scripts/run_dashboard.sh` (or `streamlit run app/app.py`) opens a 10-page app:

1. Overview
2. Data & cleaning: a data funnel, the cleaning log, and EDA charts
3. Train/test design: the time split, window tuning, and the leakage audit
4. Baseline results: metrics, before/after comparison, backtest, and error by price band and county
5. Model comparison (Week 5): tree models vs. baseline, tuning trials, strengths and weaknesses
6. Feature engineering (Week 6): every feature set × model, old vs. new table, importances, school-district prices
7. Gradient boosting (Week 7): the side-by-side table, tuning, feature sets, selection, backtest, permutation importance
8. Evaluation (Week 8): metrics_summary, findings, error by price band / county / size / age, bias, bootstrap significance
9. Price estimator: the selected model, with distance and school-district lookup from the ZIP code
10. What's next

It reads only local, git-ignored outputs. The launcher builds any that are missing.

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
X was chosen with the linear baseline; re-checking it with the final model is an open item.

**Outliers.** The 0.5th and 99.5th percentiles of `ClosePrice` are learned on the training set only. Those same
frozen cutoffs are then applied to the test set.

**Preprocessing pipeline** ([src/idx_avm/preprocess.py](src/idx_avm/preprocess.py)). This is a scikit-learn `ColumnTransformer`, fit on train only:
- numeric: train-fit percentile clipping + median imputation + missing-indicator flags + standardization
- Y/N flags: mode imputation + missing indicators
- county: one-hot encoding (rare levels grouped)
- city / ZIP: `TargetEncoder` with 5-fold cross-fitting, so a row's encoding never sees its own label
- target: modeled as `log(ClosePrice)` via `TransformedTargetRegressor`

**Metrics** ([src/idx_avm/metrics.py](src/idx_avm/metrics.py), [src/idx_avm/evaluation.py](src/idx_avm/evaluation.py)). MdAPE is the headline metric, reported with MAPE, R², MAE, and RMSE, accuracy bands (within 5/10/20%), median signed error, and breakdowns by price band, county, living area and home age. Model gaps are checked with a paired bootstrap on the same test homes.

## Layout

```
src/idx_avm/        config, load, clean, features, split, preprocess, window, metrics, pipeline, baseline, models,
                    geo, evaluation, artifacts
scripts/            run_preprocessing.py, run_baseline.py, run_models.py, run_advanced_models.py, run_evaluation.py,
                    run_dashboard.sh, list_features.py
app/                app.py (Streamlit presentation + price estimator)
notebooks/          01_exploration … 03_baseline_model, 04_model_comparison, 04b_feature_engineering,
                    05_advanced_models, 06_evaluation
docs/               data_dictionary.md, features.md (generated by scripts/list_features.py), progress_notes.md
tests/              synthetic data generator + pipeline tests
data/, models/      (git-ignored) raw data, processed outputs, saved model
```

## Assumptions and limitations

- Column names follow the RESO / Trestle standard. Whitelisted columns that are missing from an extract are skipped and reported, not treated as errors.
- `CloseDate` is the valuation date, so property age and seasonality are computed as of the sale.
- `ListingContractDate` is read for one data-quality check only and never becomes a feature.
- A month is treated as partial if the latest close date falls more than 3 days before the end of the month.
