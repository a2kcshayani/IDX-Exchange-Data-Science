"""Week 4: train and evaluate the Linear Regression baseline.

Reads the training-window length X chosen in Week 3 (data/processed/split_metadata.json),
trains on the X months before the test month, and evaluates once on the test month.
Also re-runs the same pipeline at earlier cutoffs (rolling-origin backtest) and records
the effect of the train-only feature clipping fix.

Usage:
    python scripts/run_baseline.py
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idx_avm import config  # noqa: E402
from idx_avm.baseline import breakdowns, residual_summary, run_baseline  # noqa: E402
from idx_avm.pipeline import prepare_dataset  # noqa: E402

MODELS_DIR = config.PROJECT_ROOT / "models"


def main() -> None:
    out = config.PROCESSED_DIR
    meta = json.loads((out / "split_metadata.json").read_text())
    x = int(meta["chosen_window_months"])

    data = prepare_dataset()
    test_month = data.months[-1]
    print(f"Test month {test_month}, training window X = {x} months")

    # 1) The baseline, plus the same model without the feature-clipping fix for comparison.
    result = run_baseline(data.df, test_month, x, clip_features=True)
    unclipped = run_baseline(data.df, test_month, x, clip_features=False)
    comparison = pd.DataFrame(
        [
            {"variant": "Linear Regression, raw numeric features", **unclipped.metrics},
            {"variant": "Linear Regression, train-fit feature clipping (baseline)", **result.metrics},
        ]
    )
    print("\nTest-month results:\n", comparison.round(3).to_string(index=False))

    # 2) Rolling-origin backtest: same X, two earlier test months.
    backtest = []
    for k in (2, 1, 0):
        tm = test_month - k
        r = result if k == 0 else run_baseline(data.df, tm, x)
        backtest.append({"test_month": str(tm), **r.metrics})
    backtest = pd.DataFrame(backtest)
    print("\nRolling-origin backtest:\n", backtest.round(3).to_string(index=False))

    # 3) Error breakdowns and artifacts.
    parts = breakdowns(result.test)
    resid = residual_summary(result.test)
    print("\nBy price band:\n", parts["by_price_band"].round(2).to_string())
    print("\nAccuracy bands:", {k: round(v, 1) for k, v in resid.items()})

    comparison.to_csv(out / "baseline_comparison.csv", index=False)
    backtest.to_csv(out / "baseline_backtest.csv", index=False)
    parts["by_price_band"].to_csv(out / "baseline_by_price_band.csv")
    parts["by_county"].to_csv(out / "baseline_by_county.csv")
    pred_cols = [config.TARGET, "Predicted", "CountyOrParish", "City", "LivingArea", "BedroomsTotal", "BathroomsTotalInteger"]
    result.test[pred_cols].to_csv(out / "baseline_test_predictions.csv", index=False)

    summary = {
        "model": "LinearRegression on log(ClosePrice)",
        "test_month": str(test_month),
        "train_months": [str(m) for m in result.train_months],
        "n_train_months": x,
        "test_metrics": result.metrics,
        "unclipped_test_metrics": unclipped.metrics,
        "accuracy_bands": resid,
        "price_thresholds": result.price_thresholds,
        "feature_groups": result.groups,
        "pipeline_counts": {"raw": data.n_raw, "single_family": data.n_single_family, "cleaned": len(data.df)},
    }
    (out / "baseline_metrics.json").write_text(json.dumps(summary, indent=2, default=str))

    # Everything the app needs to make a prediction with the exact training-time preprocessing.
    train = data.df[data.df["CloseMonth"].isin(result.train_months)]
    features = [c for cols in result.groups.values() for c in cols]
    bundle = {
        "model": result.model,
        "groups": result.groups,
        "features": features,
        "defaults": train[result.groups["numeric"]].median(numeric_only=True).to_dict(),
        "options": {
            "CountyOrParish": sorted(train["CountyOrParish"].dropna().unique().tolist()),
            "city_by_county": train.groupby("CountyOrParish")["City"].agg(lambda s: sorted(s.dropna().unique())).to_dict(),
            "zip_by_city": train.groupby("City")["PostalCode"].agg(lambda s: sorted(s.dropna().unique())).to_dict(),
            "latlon_by_zip": train.groupby("PostalCode")[["Latitude", "Longitude"]].median().round(5).to_dict("index"),
        },
        "test_month": str(test_month),
        "metrics": result.metrics,
    }
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle, MODELS_DIR / "baseline_linear.joblib")
    print(f"\nSaved model bundle to {MODELS_DIR / 'baseline_linear.joblib'} and results to {out}")


if __name__ == "__main__":
    main()
