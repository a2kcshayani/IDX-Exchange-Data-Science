# Progress Notes (Weeks 1–8)

_Last updated after Weeks 7–8 (gradient boosting, evaluation expansion)._

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

| | Week 4 baseline | Week 6 pick | Week 7 selected model |
|---|---|---|---|
| Model | Linear Regression | Random Forest | **Gradient Boosting** (base features) |
| MdAPE (headline) | 11.11% | 7.78% | **7.50%** |
| R² | 0.811 | 0.885 | **0.904** |
| MAE | $212,780 | $161,694 | **$150,331** |
| Within 10% of sale price | 45.7% | 59.4% | **61.0%** |
| Luxury homes ($1.67M+) MdAPE | 16.4% | 12.1% | **10.8%** |
| Median bias | +0.1% | −1.0% | **−0.2%** |
| Saved model | small | 349 MB | **6.6 MB** |

- Week 5: Random Forest beats the baseline by 30%. Decision Tree is in between, at 9.55%.
- Week 6: nearby comps are the strongest added feature. Ties go to the simpler setup (0.10-point noise rule).
- Week 7: gradient boosting (scikit-learn `HistGradientBoostingRegressor`), 12 settings tuned on the validation month.
  It beats the Random Forest by 0.37 validation points, so it's selected. Backtest: 7.66% / 7.35% / 7.50%.
- Week 8: `metrics_summary.csv`, error by price band, county, size and age, and a paired bootstrap. The gain over the Random Forest
  (−0.28 points, 95% CI −0.44 to −0.11) is bigger than noise.
- Dashboard: new Gradient Boosting and Evaluation pages. The estimator now loads in about a second.

## Issues found and how they were handled

| Found | Impact | Handling |
|---|---|---|
| Raw MLS entry errors (24.9-billion sq ft lot, 500 garage spaces, −143 parking spaces) | Linear predictions up to $38.8M; R² 0.685 | Train-fit percentile clipping (`QuantileClipper`), R² 0.685 → 0.811 |
| ZIP codes read back from CSV as floats ("91762.0") | Silently broke ZIP encoding; MdAPE looked like 26.7% | `load_processed()` forces 5-digit string ZIPs |
| Leakage audit showed ListPrice correlation as 0.25 | Understated the leak; outliers swamp Pearson correlation | Switched to Spearman: ListPrice 0.993, OriginalListPrice 0.986 |
| `FireplacesTotal` empty in every training row | Broke median imputation | Empty columns are skipped automatically |
| School district file includes race/ethnicity shares | Fair-housing risk if used for pricing | Excluded; only district identity is used |
| LightGBM / XGBoost wouldn't load (macOS needs a system OpenMP library) | Week 7 blocked on extra system setup | Used scikit-learn's histogram gradient boosting (same algorithm family), no extra installs |
| Streamlit listened on the whole network by default | Anyone on the same Wi-Fi could view IDX data summaries | Bound to `localhost` in `.streamlit/config.toml` and the launch script |

## Things to look into

1. **Bay Area is the weak spot.** San Mateo (11.7%) and Alameda (11.1%) have the highest error, and are the one place the Random Forest
   beat gradient boosting. Adding nearby comps brings them to 10.0% / 9.8%. Worth a closer look: dense, expensive, fast-moving markets.
2. **Typical error vs. big misses.** Gradient boosting + all features has the best R² (0.915) and MAPE, but its MdAPE edge over the
   base version (0.12 points) is within noise, so the simpler model is selected. If IDX cares more about tail risk, switch to it.
3. **Luxury, very large and older homes** are still the hardest (10.8%, 10.7%, 9.4% MdAPE). There are fewer true comparables. Next ideas:
   like-for-like comps (similar size and age), and weighting recent sales more.
4. **Tuning sits at the edge of the grid** (most trees, highest learning rate). Try 1,500–2,000 trees, or a lower rate with more trees.
5. **The training window (18 months) was chosen with the linear baseline.** Re-check it with gradient boosting.
6. **The test month has been viewed many times.** Score the final model on May 2026 or later data if it's
   on the FTP: download the new `CRMLSSold*.csv` files into `data/raw/` and run `bash scripts/run_dashboard.sh --rebuild`.
7. **Notebooks have no saved results.** Run 01–06 in Jupyter and fill in the "Takeaways" cells before presenting.
8. **Confirm with IDX** that publishing aggregate metrics in the public README is fine. No row-level data is in the repo.

## Next up (Weeks 9–10)

Week 9's Streamlit estimator already exists (Price Estimator page). Remaining: polish, then the Week 10 README pass
(dataset source, preprocessing, models tested, best results, how to re-run).
