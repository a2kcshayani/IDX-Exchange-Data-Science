# Progress Notes (Weeks 1–6)

_Last updated after the Week 6 completion pass (nearby comps, job-center distance, noise-aware model selection)._

A running list of what changed, what looked wrong and how it was handled, and what's still open.
Numbers are from the April 2026 test month (12,337 sales) unless noted.

## Run the dashboard

```bash
cd ~/claude_cowork/IDX-Exchange-Data-Science
bash scripts/run_dashboard.sh
```

The script checks the environment and inputs, runs any missing pipeline step, picks a free port starting
at 8501, and opens the browser. `--check` only reports status; `--rebuild` re-runs every step.
Stop it with Ctrl+C. The app is bound to `localhost`, so only this computer can open it.

## Updates since the Week 4 baseline

| | Week 4 baseline | Weeks 5–6 selected model |
|---|---|---|
| Model | Linear Regression | Random Forest (Week 4 base features) |
| MdAPE (headline) | 11.11% | **7.78%** |
| R² | 0.811 | **0.885** |
| MAE | $212,780 | $161,694 |
| Within 10% of sale price | 45.7% | **59.4%** |
| Luxury homes ($1.67M+) MdAPE | 16.4% | **12.1%** |

- Week 5: the Random Forest beats the baseline by 3.3 MdAPE points (30%). The Decision Tree is in between, at 9.55%.
- Week 6, six cumulative feature sets: ratios, school districts (spatial join), job-center distance, ZIP comps, and
  nearby comps (10 closest prior sales). Nearby comps are the strongest: Linear Regression 11.11% → 9.61%, and the Random Forest
  reaches the best R², 0.902. Typical error for tree models is unchanged within noise.
- Selection rule: an addition must beat the best by more than 0.10 validation points, or the simpler setup wins.
  So the selected model is the base Random Forest. The runner-up, + job-center distance, was 0.08 points better: a tie.
- Dashboard: Model Comparison and Feature Engineering pages. The estimator uses the selected model.

## Issues found and how they were handled

| Found | Impact | Handling |
|---|---|---|
| Raw MLS entry errors (24.9-billion sq ft lot, 500 garage spaces, −143 parking spaces) | Linear predictions up to $38.8M; R² 0.685 | Train-fit percentile clipping (`QuantileClipper`), R² 0.685 → 0.811 |
| ZIP codes read back from CSV as floats ("91762.0") | Silently broke ZIP encoding; MdAPE looked like 26.7% | `load_processed()` forces 5-digit string ZIPs |
| Leakage audit showed ListPrice correlation as 0.25 | Understated the leak; outliers swamp Pearson correlation | Switched to Spearman: ListPrice 0.993, OriginalListPrice 0.986 |
| `FireplacesTotal` empty in every training row | Broke median imputation | Empty columns are skipped automatically |
| School district file includes race/ethnicity shares | Fair-housing risk if used for pricing | Excluded; only district identity is used |
| Streamlit listened on the whole network by default | Anyone on the same Wi-Fi could view IDX data summaries | Bound to `localhost` in `.streamlit/config.toml` and the launch script |

## Things to look into

1. **Comps raise R² but not typical error for the Random Forest** (R² 0.885 → 0.902, MdAPE flat). They mostly fix big misses.
   If IDX cares about tail risk (lending exposure), consider choosing on RMSE/R² as well as MdAPE. Next ideas: weight
   recent sales more, and compare like-for-like homes (similar size and age) instead of just the nearest.
2. **Selection is sensitive to the tie rule.** With a 0.10-point tolerance the base Random Forest wins. Without it, + job-center
   distance would win (test MdAPE 7.70% vs. 7.78%). Re-check after Week 7, ideally on May 2026 data.
3. **Luxury homes are still the weakest band** (12.1% MdAPE versus 6–7% for mid-priced homes). Trees can't predict
   above the highest training price, so check how the $5M+ tail behaves.
4. **Small signed bias:** the median prediction is 1.0% below the sale price. Check whether this is the market rising
   between the training window and the test month (an argument for a time-trend feature).
5. **Model file size:** the saved Random Forest is 353 MB, so the estimator's first load takes about 15 seconds. Week 9: fewer or shallower
   trees, or gradient boosting (much smaller).
6. **The test month has been viewed many times.** Every decision so far was made on validation months, but after Week 7, add
   May 2026 data when it's released and score the final model on it fresh.
7. **Notebooks have no saved results.** Run 01–04b in Jupyter and fill in the "Findings" cells before presenting.
8. **Confirm with IDX** that publishing aggregate metrics in the public README is fine. No row-level data is in the repo.

## Next up (Week 7)

Gradient boosting (LightGBM or XGBoost), with light tuning of depth, learning rate and number of trees on the validation month,
compared against the Random Forest in the same table.
