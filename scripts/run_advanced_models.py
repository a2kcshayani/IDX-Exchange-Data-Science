"""Week 7: gradient boosting vs. the Week 4-6 models.

1. Tune gradient boosting (depth, learning rate, number of trees) on the validation
   month, base features, the same protocol as Week 5.
2. Re-train it on all six Week 6 feature sets (validation + test month).
3. Re-run selection over every model x feature set (Weeks 5-7) on validation MdAPE
   with the same noise rule; the test month is reported, never used to choose.
4. Backtest the selected model on 3 months and save it as models/best_model.joblib.

Needs the Weeks 5-6 outputs (run scripts/run_models.py first).

Usage:
    python scripts/run_advanced_models.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idx_avm import config  # noqa: E402
from idx_avm.artifacts import load_tuned_params, save_best_model, save_tuned_params  # noqa: E402
from idx_avm.features import FEATURE_SETS  # noqa: E402
from idx_avm.models import (  # noqa: E402
    NOISE_TOLERANCE_PTS, _fmt, fit_and_score, params_from_label, permutation_mdape_importance, select_model,
    tune_on_validation,
)
from idx_avm.pipeline import prepare_dataset  # noqa: E402

GB = "Gradient Boosting"


def week5_params(out: Path) -> dict[str, dict]:
    """Tuned Week 5 settings: tuned_params.json if present, else recovered from the trial log."""
    if (out / "tuned_params.json").exists():
        params = load_tuned_params(out)
        if all(m in params for m in ("Linear Regression", "Decision Tree", "Random Forest")):
            return params
    trials = pd.read_csv(out / "week5_tuning_trials.csv")
    best = trials.loc[trials.groupby("model")["MdAPE"].idxmin()]
    params = {r.model: params_from_label(r.model, r.params) for r in best.itertuples()}
    save_tuned_params(params, out)
    return params


def main() -> None:
    out = config.PROCESSED_DIR
    for name in ("week6_feature_sets.csv", "week5_model_comparison.csv", "week5_tuning_trials.csv"):
        if not (out / name).exists():
            sys.exit(f"Missing {name}. Run scripts/run_models.py first.")
    x = int(json.loads((out / "split_metadata.json").read_text())["chosen_window_months"])
    data = prepare_dataset(week6_features=True)
    df, test_month = data.df, data.months[-1]
    val_month = test_month - 1
    print(f"Test month {test_month}, validation month {val_month}, training window {x} months")

    # ---------------- 1. Tune on the validation month (base features, as in Week 5)
    print(f"\nTuning {GB} on {val_month}")
    gb_params, trials = tune_on_validation(df, test_month, x, GB, flags={})
    best_trial = trials.loc[trials["MdAPE"].idxmin()]
    params = {**week5_params(out), GB: gb_params}
    save_tuned_params({GB: gb_params}, out)

    # ---------------- 2. Gradient boosting on every feature set
    print(f"\n{GB} on the Week 6 feature sets")
    rows, fitted = [], {}
    for set_name, flags in FEATURE_SETS.items():
        val = fit_and_score(df, val_month, x, GB, gb_params, flags, set_name)
        test = fit_and_score(df, test_month, x, GB, gb_params, flags, set_name)
        fitted[set_name] = test
        rows.append({"feature_set": set_name, "model": GB, "val_MdAPE": val.metrics["MdAPE"],
                     **{f"test_{k}": v for k, v in test.metrics.items() if k != "n"},
                     "fit_seconds": test.fit_seconds})
        print(f"  {set_name:22s} val MdAPE {val.metrics['MdAPE']:.2f}%  test MdAPE {test.metrics['MdAPE']:.2f}%  "
              f"test R2 {test.metrics['R2']:.3f}  ({test.fit_seconds:.0f}s)")
    gb_sets = pd.DataFrame(rows)

    # ---------------- 3. Selection across Weeks 5-7 (validation only)
    all_results = pd.concat([pd.read_csv(out / "week6_feature_sets.csv"), gb_sets.drop(columns="fit_seconds")],
                            ignore_index=True)
    lowest = all_results.loc[all_results["val_MdAPE"].idxmin()]
    best_row = select_model(all_results, list(FEATURE_SETS))
    best_key = (best_row["model"], best_row["feature_set"])
    print(f"\nLowest validation MdAPE: {lowest['model']} with '{lowest['feature_set']}' ({lowest['val_MdAPE']:.2f}%)")
    print(f"Selected (simplest within {NOISE_TOLERANCE_PTS} pts): {best_key[0]} with '{best_key[1]}' "
          f"({best_row['val_MdAPE']:.2f}%)")
    flags = FEATURE_SETS[best_key[1]]
    best = (fitted[best_key[1]] if best_key[0] == GB
            else fit_and_score(df, test_month, x, best_key[0], params[best_key[0]], flags, best_key[1]))

    # ---------------- 4. Backtest + side-by-side table (base features, plus the selected config)
    backtest = []
    for k in (2, 1, 0):
        tm = test_month - k
        r = best if k == 0 else fit_and_score(df, tm, x, best_key[0], params[best_key[0]], flags, best_key[1])
        backtest.append({"test_month": str(tm), **r.metrics})
    backtest = pd.DataFrame(backtest)
    print("\nBacktest of selected model:\n", backtest.round(3).to_string(index=False))

    week5 = pd.read_csv(out / "week5_model_comparison.csv")
    gb_base = fitted["Base (Week 4)"]
    comparison = pd.concat([week5.drop(columns=["MdAPE_vs_baseline_pts"], errors="ignore"), pd.DataFrame([{
        "model": GB, "params": best_trial["params"], "val_MdAPE": best_trial["MdAPE"],
        **gb_base.metrics, "fit_seconds": gb_base.fit_seconds}])], ignore_index=True)
    base_mdape = comparison.loc[comparison["model"] == "Linear Regression", "MdAPE"].iloc[0]
    comparison["MdAPE_vs_baseline_pts"] = comparison["MdAPE"] - base_mdape
    comparison["MdAPE_vs_baseline_pct"] = comparison["MdAPE_vs_baseline_pts"] / base_mdape * 100
    print("\nSide by side (base features, test month):\n",
          comparison.drop(columns="params").round(3).to_string(index=False))

    importance = permutation_mdape_importance(best)
    print("\nTop inputs (MdAPE rise when shuffled):\n", importance.head(10).round(3).to_string(index=False))

    # ---------------- Save (all git-ignored)
    trials.to_csv(out / "week7_tuning_trials.csv", index=False)
    gb_sets.to_csv(out / "week7_gb_feature_sets.csv", index=False)
    comparison.to_csv(out / "week7_model_comparison.csv", index=False)
    importance.to_csv(out / "best_model_permutation_importance.csv", index=False)
    week6_sel = json.loads((out / "week6_selection.json").read_text()) if (out / "week6_selection.json").exists() else {}
    summary = {
        "stage": "Week 7",
        "test_month": str(test_month), "validation_month": str(val_month), "n_train_months": x,
        "selected_model": best_key[0], "selected_feature_set": best_key[1],
        "selection_rule": f"simplest configuration within {NOISE_TOLERANCE_PTS} pts of the lowest validation MdAPE",
        "lowest_val": {"model": lowest["model"], "feature_set": lowest["feature_set"], "val_MdAPE": lowest["val_MdAPE"]},
        "selected_val_MdAPE": best_row["val_MdAPE"],
        "feature_set_order": list(FEATURE_SETS),
        "selected_params": {k: v for k, v in params[best_key[0]].items() if k != "n_jobs"},
        "selected_params_label": _fmt(params[best_key[0]]),
        "previous_selection": {k: week6_sel.get(k) for k in ("selected_model", "selected_feature_set",
                                                             "selected_val_MdAPE", "model_file_mb")},
        "baseline_test_metrics": week5.loc[week5["model"] == "Linear Regression"].iloc[0].to_dict(),
    }
    summary = save_best_model(best, backtest, summary, df, test_month, x)
    print(f"\nSaved results to {out} and model to models/best_model.joblib ({summary['model_file_mb']} MB)")


if __name__ == "__main__":
    main()
