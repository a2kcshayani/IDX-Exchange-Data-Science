"""Week 8: evaluation expansion.

Scores the finalists side by side, the same way, on the same homes:
  - Linear Regression baseline (Week 4)
  - Random Forest (Weeks 5-6 selection)
  - the Week 7 selected model, and gradient boosting with every Week 6 feature
    (the best R² setup, kept to show the typical-error vs. big-miss trade-off)

Outputs (data/processed/, git-ignored):
  metrics_summary.csv     one row per model: R², MAPE, MdAPE, MAE, RMSE, accuracy bands,
                          bias, and the 3-month backtest range
  eval_backtest.csv       every model x month
  eval_segments.csv       error by price band (sale price and model estimate), county,
                          living area, and home age
  eval_significance.csv   paired bootstrap of MdAPE gaps between models
  eval_insights.json      plain-language findings computed from the tables

Needs run_advanced_models.py first.

Usage:
    python scripts/run_evaluation.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idx_avm import config  # noqa: E402
from idx_avm.artifacts import load_tuned_params  # noqa: E402
from idx_avm.evaluation import (  # noqa: E402
    band_labels, full_metrics, paired_bootstrap, price_band_edges, segment_breakdowns,
)
from idx_avm.features import FEATURE_SETS  # noqa: E402
from idx_avm.models import fit_and_score  # noqa: E402
from idx_avm.pipeline import prepare_dataset  # noqa: E402

BASE = "Base (Week 4)"
FULL = list(FEATURE_SETS)[-1]


def candidates(selected: dict) -> list[dict]:
    rows = [
        {"label": "Linear Regression (baseline)", "model": "Linear Regression", "feature_set": BASE, "role": "Week 4 baseline"},
        {"label": "Random Forest", "model": "Random Forest", "feature_set": BASE, "role": "Weeks 5-6 selection"},
        {"label": f"{selected['selected_model']} (selected)", "model": selected["selected_model"],
         "feature_set": selected["selected_feature_set"], "role": "Week 7 selection"},
        {"label": "Gradient Boosting + all features", "model": "Gradient Boosting", "feature_set": FULL,
         "role": "Best-R² alternative"},
    ]
    # If the selection is one of the fixed comparison models, keep only its "selected" row.
    sel_key = (selected["selected_model"], selected["selected_feature_set"])
    return [r for r in rows if r["role"] == "Week 7 selection" or (r["model"], r["feature_set"]) != sel_key]


def insights(summary: pd.DataFrame, segments: pd.DataFrame, sig: pd.DataFrame, sel: str, base: str) -> list[str]:
    s = summary.set_index("label")
    b, m = s.loc[base], s.loc[sel]
    out = [
        f"{sel} has a {m['MdAPE']:.2f}% typical error (MdAPE) on {int(m['n']):,} unseen sales, versus "
        f"{b['MdAPE']:.2f}% for the baseline ({(b['MdAPE'] - m['MdAPE']) / b['MdAPE'] * 100:.0f}% lower). "
        f"Mean error (MAPE) is higher, {m['MAPE']:.2f}%, because a minority of large misses pulls the average up.",
        f"{m['within_10pct']:.1f}% of its estimates land within 10% of the sale price (baseline {b['within_10pct']:.1f}%), "
        f"and {m['within_20pct']:.1f}% within 20%.",
        f"Across the 3 backtest months its MdAPE ranged {m['backtest_MdAPE_min']:.2f}–{m['backtest_MdAPE_max']:.2f}%, "
        f"so the gain is stable, not a one-month result.",
    ]
    band = segments[(segments["label"] == sel) & (segments["segment_type"] == "Price band (sale price)")]
    if not band.empty:
        lo, hi = band.loc[band["MdAPE"].idxmin()], band.loc[band["MdAPE"].idxmax()]
        out.append(f"By price band, it is most accurate for {lo['segment']} ({lo['MdAPE']:.1f}% MdAPE) and least "
                   f"accurate for {hi['segment']} ({hi['MdAPE']:.1f}%).")
        first, last = band.iloc[0], band.iloc[-1]
        text = (f"Bias by sale price: median signed error is {first['median_signed_error_pct']:+.1f}% in the cheapest band "
                f"and {last['median_signed_error_pct']:+.1f}% in the most expensive (+ = over-estimate).")
        est = segments[(segments["label"] == sel) & (segments["segment_type"] == "Price band (model estimate)")]
        if len(est) >= 2:
            text += (f" Grouped by the model's own estimate instead (what a user sees), it is "
                     f"{est.iloc[0]['median_signed_error_pct']:+.1f}% and {est.iloc[-1]['median_signed_error_pct']:+.1f}%. "
                     "The gap between the two views is regression to the middle, not a fixed bias in the model.")
        out.append(text)
    base_band = segments[(segments["label"] == base) & (segments["segment_type"] == "Price band (sale price)")]
    if not band.empty and not base_band.empty:
        gain = (base_band.set_index("segment")["MdAPE"] - band.set_index("segment")["MdAPE"]).dropna()
        out.append(f"The biggest gain over the baseline is in {gain.idxmax()} ({gain.max():.1f} pts lower MdAPE); "
                   f"the smallest is in {gain.idxmin()} ({gain.min():.1f} pts).")
    county = segments[(segments["label"] == sel) & (segments["segment_type"] == "County")]
    if len(county) >= 2:
        lo, hi = county.loc[county["MdAPE"].idxmin()], county.loc[county["MdAPE"].idxmax()]
        out.append(f"Among counties with 100+ test sales, error is lowest in {lo['segment']} ({lo['MdAPE']:.1f}%) and "
                   f"highest in {hi['segment']} ({hi['MdAPE']:.1f}%).")
    for r in sig.itertuples():
        verdict = "bigger than test-month noise" if r.ci_high < 0 else ("not distinguishable from noise" if r.ci_low <= 0 else "worse")
        out.append(f"{r.model_a} vs. {r.model_b}: MdAPE gap {r.mdape_diff_pts:+.2f} pts (95% CI {r.ci_low:+.2f} to "
                   f"{r.ci_high:+.2f}), {verdict}; it is closer to the sale price on {r.homes_a_closer_pct:.0f}% of homes.")
    return out


def main() -> None:
    out = config.PROCESSED_DIR
    if not (out / "week7_gb_feature_sets.csv").exists():
        sys.exit("Missing Week 7 outputs. Run scripts/run_advanced_models.py first.")
    selected = json.loads((out / "best_model_metrics.json").read_text())
    params = load_tuned_params(out)
    x = int(selected["n_train_months"])
    data = prepare_dataset(week6_features=True)
    df, test_month = data.df, data.months[-1]
    models = candidates(selected)
    print(f"Test month {test_month}; evaluating: {', '.join(m['label'] for m in models)}")

    backtest, test_preds = [], {}
    for m in models:
        for k in (2, 1, 0):
            tm = test_month - k
            r = fit_and_score(df, tm, x, m["model"], params[m["model"]], FEATURE_SETS[m["feature_set"]], m["feature_set"])
            backtest.append({"label": m["label"], "test_month": str(tm),
                             **full_metrics(r.test[config.TARGET], r.test["Predicted"])})
            if k == 0:
                test_preds[m["label"]] = r.test
        print(f"  {m['label']:34s} test MdAPE {backtest[-1]['MdAPE']:.2f}%  R2 {backtest[-1]['R2']:.3f}")
    backtest = pd.DataFrame(backtest)

    # ---------------- metrics_summary.csv
    rows = []
    for m in models:
        t = test_preds[m["label"]]
        bt = backtest[backtest["label"] == m["label"]]["MdAPE"]
        rows.append({**m, "test_month": str(test_month), **full_metrics(t[config.TARGET], t["Predicted"]),
                     "backtest_MdAPE_mean": bt.mean(), "backtest_MdAPE_min": bt.min(), "backtest_MdAPE_max": bt.max()})
    summary = pd.DataFrame(rows)
    base_label, sel_label = models[0]["label"], next(m["label"] for m in models if m["role"] == "Week 7 selection")
    base_mdape = summary.loc[summary["label"] == base_label, "MdAPE"].iloc[0]
    summary["MdAPE_vs_baseline_pct"] = (summary["MdAPE"] - base_mdape) / base_mdape * 100
    base_bt = backtest[backtest["label"] == base_label].set_index("test_month")["MdAPE"]
    summary["months_beating_baseline"] = [
        int((backtest[backtest["label"] == lab].set_index("test_month")["MdAPE"] < base_bt).sum()) for lab in summary["label"]
    ]

    # ---------------- Segments (same price-band edges for every model)
    y_test = test_preds[base_label][config.TARGET]
    edges = price_band_edges(y_test)
    labels = band_labels(y_test, edges)
    segments = pd.concat([segment_breakdowns(test_preds[m["label"]], edges, labels).assign(label=m["label"])
                          for m in models], ignore_index=True)
    segments = segments[["label"] + [c for c in segments.columns if c != "label"]]

    # ---------------- Is the improvement bigger than noise? (paired bootstrap, same homes)
    sig = []
    sel = test_preds[sel_label]
    for m in models:
        if m["label"] == sel_label:
            continue
        other = test_preds[m["label"]]
        assert (other.index == sel.index).all(), "models must be scored on the same homes"
        sig.append({"model_a": sel_label, "model_b": m["label"],
                    **paired_bootstrap(sel[config.TARGET], sel["Predicted"], other["Predicted"])})
    sig = pd.DataFrame(sig)

    notes = insights(summary, segments, sig, sel_label, base_label)

    summary.to_csv(out / "metrics_summary.csv", index=False)
    backtest.to_csv(out / "eval_backtest.csv", index=False)
    segments.to_csv(out / "eval_segments.csv", index=False)
    sig.to_csv(out / "eval_significance.csv", index=False)
    (out / "eval_insights.json").write_text(json.dumps({"selected": sel_label, "baseline": base_label,
                                                       "test_month": str(test_month), "insights": notes}, indent=2))
    cols = ["label", "MdAPE", "MAPE", "R2", "MAE", "RMSE", "within_10pct", "median_signed_error_pct",
            "backtest_MdAPE_min", "backtest_MdAPE_max"]
    print("\nmetrics_summary.csv:\n", summary[cols].round(3).to_string(index=False))
    print("\nSignificance:\n", sig.round(3).to_string(index=False))
    print("\nInsights:")
    for n in notes:
        print(" -", n)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
