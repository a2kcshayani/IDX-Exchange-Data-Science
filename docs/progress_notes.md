# Progress Notes (Weeks 1–6)

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
| Model | Linear Regression | Random Forest + school districts |
| MdAPE (headline) | 11.11% | **7.81%** |
| R² | 0.811 | **0.885** |
| MAE | $212,780 | $161,137 |
| Within 10% of sale price | 45.7% | **59.8%** |
| Luxury homes ($1.67M+) MdAPE | 16.4% | **11.8%** |

- Week 5: the Random Forest beats the baseline by 3.3 MdAPE points (30%). The Decision Tree is in between, at 9.55%.
- Week 6: comps from prior sales help the linear model (11.11% → 9.88%). For tree models, every Week 6 feature change is within noise.
- Dashboard: added Model Comparison and Feature Engineering pages. The estimator uses the selected Random Forest and looks up school districts automatically.

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

1. **Week 6 features barely help the Random Forest.** It already learns neighborhood price levels from ZIP and city.
   Better comps could change that: nearest-neighbor comps by distance instead of ZIP, and recent sales weighted more heavily.
2. **Selected model is a near-tie.** RF + school districts beat the base RF by only 0.03 points on validation. Prefer the
   simpler model when results tie, and check it again after Week 7.
3. **Luxury homes are still the weakest band** (11.8% MdAPE versus 6–7% for mid-priced homes). Trees can't predict
   above the highest training price, so check how the $5M+ tail behaves.
4. **Small signed bias:** the median prediction is 1.1% below the sale price. Check whether this is the market rising
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
