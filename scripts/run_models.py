"""Weeks 5-6: model comparison and feature engineering.

Week 5: tune Decision Tree and Random Forest on the validation month, then
        compare test-month metrics against the Linear Regression baseline.
Week 6: re-train every model on six cumulative feature sets
        (base -> +ratios -> +school districts -> +job-center distance
        -> +ZIP comps -> +nearby comps) and compare.

Model / feature-set selection uses the validation month only; the test month
is reported for every configuration but never used to choose. Differences
within NOISE_TOLERANCE_PTS go to the simpler configuration (models.select_model).

Usage:
    python scripts/run_models.py
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idx_avm import config  # noqa: E402
from idx_avm.baseline import breakdowns, residual_summary  # noqa: E402
from idx_avm.models import (  # noqa: E402
    MODEL_GRIDS, NOISE_TOLERANCE_PTS, feature_importance, fit_and_score, select_model, tune_on_validation,
)
from idx_avm.pipeline import prepare_dataset  # noqa: E402
from idx_avm.preprocess import feature_list  # noqa: E402

MODELS_DIR = config.PROJECT_ROOT / "models"

# Cumulative: each set adds to the one before it. Order = complexity, used for tie-breaking.
FEATURE_SETS = {
    "Base (Week 4)": {},
    "+ Ratios": {"ratios": True},
    "+ School districts": {"ratios": True, "school": True},
    "+ Job-center distance": {"ratios": True, "school": True, "distance": True},
    "+ ZIP comps": {"ratios": True, "school": True, "distance": True, "comps": True},
    "+ Nearby comps": {"ratios": True, "school": True, "distance": True, "comps": True, "nearby": True},
}


def main() -> None:
    out = config.PROCESSED_DIR
    x = int(json.loads((out / "split_metadata.json").read_text())["chosen_window_months"])
    data = prepare_dataset(week6_features=True)
    df, test_month = data.df, data.months[-1]
    val_month = test_month - 1
    print(f"Test month {test_month}, validation month {val_month}, training window {x} months")

    # ---------------- Week 5: tune on validation, score once on test (base features)
    print("\nWeek 5: tuning on the validation month")
    best_params, trials, week5 = {}, [], []
    for name in MODEL_GRIDS:
        params, t = tune_on_validation(df, test_month, x, name, flags={})
        best_params[name] = params
        trials.append(t)
        r = fit_and_score(df, test_month, x, name, params, flags={}, feature_set="Base (Week 4)")
        week5.append({"model": name, "params": t.loc[t["MdAPE"].idxmin(), "params"],
                      "val_MdAPE": t["MdAPE"].min(), **r.metrics, "fit_seconds": r.fit_seconds})
    week5 = pd.DataFrame(week5)
    base_mdape = week5.loc[week5["model"] == "Linear Regression", "MdAPE"].iloc[0]
    week5["MdAPE_vs_baseline_pts"] = week5["MdAPE"] - base_mdape
    print("\nWeek 5 test-month results:\n", week5.drop(columns="params").round(3).to_string(index=False))

    # ---------------- Week 6: every model x cumulative feature set (validation + test)
    print("\nWeek 6: feature sets")
    rows, fitted = [], {}
    for set_name, flags in FEATURE_SETS.items():
        for name in MODEL_GRIDS:
            val = fit_and_score(df, val_month, x, name, best_params[name], flags, set_name)
            test = fit_and_score(df, test_month, x, name, best_params[name], flags, set_name)
            fitted[(name, set_name)] = test
            rows.append({"feature_set": set_name, "model": name, "val_MdAPE": val.metrics["MdAPE"],
                         **{f"test_{k}": v for k, v in test.metrics.items() if k != "n"}})
            print(f"  {set_name:20s} {name:17s} val MdAPE {val.metrics['MdAPE']:.2f}%  "
                  f"test MdAPE {test.metrics['MdAPE']:.2f}%  test R2 {test.metrics['R2']:.3f}")
    week6 = pd.DataFrame(rows)

    # ---------------- Select by validation, then backtest the winner
    lowest = week6.loc[week6["val_MdAPE"].idxmin()]
    best_row = select_model(week6, list(FEATURE_SETS))
    best_key = (best_row["model"], best_row["feature_set"])
    best = fitted[best_key]
    print(f"\nLowest validation MdAPE: {lowest['model']} with '{lowest['feature_set']}' ({lowest['val_MdAPE']:.2f}%)")
    print(f"Selected (simplest within {NOISE_TOLERANCE_PTS} pts): {best_key[0]} with '{best_key[1]}' "
          f"({best_row['val_MdAPE']:.2f}%)")

    backtest = []
    for k in (2, 1, 0):
        tm = test_month - k
        r = best if k == 0 else fit_and_score(df, tm, x, best_key[0], best_params[best_key[0]],
                                              FEATURE_SETS[best_key[1]], best_key[1])
        backtest.append({"test_month": str(tm), **r.metrics})
    backtest = pd.DataFrame(backtest)
    print("\nBacktest of selected model:\n", backtest.round(3).to_string(index=False))

    # Importances from the Random Forest on the full feature set (most informative view of Week 6).
    importances = feature_importance(fitted[("Random Forest", list(FEATURE_SETS)[-1])])
    parts = breakdowns(best.test)
    resid = residual_summary(best.test)

    # ---------------- Save artifacts (all git-ignored)
    pd.concat(trials).to_csv(out / "week5_tuning_trials.csv", index=False)
    week5.to_csv(out / "week5_model_comparison.csv", index=False)
    week6.to_csv(out / "week6_feature_sets.csv", index=False)
    backtest.to_csv(out / "best_model_backtest.csv", index=False)
    importances.to_csv(out / "rf_feature_importance.csv", index=False)
    parts["by_price_band"].to_csv(out / "best_model_by_price_band.csv")
    parts["by_county"].to_csv(out / "best_model_by_county.csv")
    pred_cols = [config.TARGET, "Predicted", "CountyOrParish", "City", "LivingArea"]
    best.test[pred_cols].to_csv(out / "best_model_test_predictions.csv", index=False)
    summary = {
        "test_month": str(test_month), "validation_month": str(val_month), "n_train_months": x,
        "selected_model": best_key[0], "selected_feature_set": best_key[1],
        "selection_rule": f"simplest configuration within {NOISE_TOLERANCE_PTS} pts of the lowest validation MdAPE",
        "lowest_val": {"model": lowest["model"], "feature_set": lowest["feature_set"], "val_MdAPE": lowest["val_MdAPE"]},
        "selected_val_MdAPE": best_row["val_MdAPE"],
        "feature_set_order": list(FEATURE_SETS),
        "selected_params": {k: v for k, v in best_params[best_key[0]].items() if k != "n_jobs"},
        "test_metrics": best.metrics, "accuracy_bands": resid,
        "baseline_test_metrics": week5.loc[week5["model"] == "Linear Regression"].iloc[0].to_dict(),
    }
    (out / "best_model_metrics.json").write_text(json.dumps(summary, indent=2, default=str))

    train = df[df["CloseMonth"].isin([test_month - i for i in range(x, 0, -1)])]
    bundle = {
        "model": best.model,
        "groups": best.groups,
        "features": feature_list(best.groups),
        "model_name": best_key[0],
        "feature_set": best_key[1],
        "defaults": train[best.groups["numeric"]].median(numeric_only=True).to_dict(),
        "options": {
            "CountyOrParish": sorted(train["CountyOrParish"].dropna().unique().tolist()),
            "city_by_county": train.groupby("CountyOrParish")["City"].agg(lambda s: sorted(s.dropna().unique())).to_dict(),
            "zip_by_city": train.groupby("City")["PostalCode"].agg(lambda s: sorted(s.dropna().unique())).to_dict(),
            "latlon_by_zip": train.groupby("PostalCode")[["Latitude", "Longitude"]].median().round(5).to_dict("index"),
        },
        "test_month": str(test_month),
        "metrics": best.metrics,
    }
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle, MODELS_DIR / "best_model.joblib", compress=3)
    print(f"\nSaved results to {out} and model to {MODELS_DIR / 'best_model.joblib'}")


if __name__ == "__main__":
    main()
