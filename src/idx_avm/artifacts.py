"""Save the currently selected model and its results (shared by the Week 5-6 and Week 7 scripts).

Everything written here is git-ignored: data/processed/*.csv|json and models/*.joblib.
"""
import json
from pathlib import Path

import joblib
import pandas as pd

from . import config
from .baseline import breakdowns
from .evaluation import accuracy_bands
from .models import FitResult
from .preprocess import feature_list

MODELS_DIR = config.PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"


def build_bundle(best: FitResult, df: pd.DataFrame, test_month, n_train_months: int) -> dict:
    """Model plus everything the Streamlit estimator needs to build a valid input row."""
    train = df[df["CloseMonth"].isin([test_month - i for i in range(n_train_months, 0, -1)])]
    return {
        "model": best.model,
        "groups": best.groups,
        "features": feature_list(best.groups),
        "model_name": best.model_name,
        "feature_set": best.feature_set,
        "params": {k: v for k, v in best.params.items() if k != "n_jobs"},
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


def save_best_model(best: FitResult, backtest: pd.DataFrame, summary: dict, df: pd.DataFrame,
                    test_month, n_train_months: int, out: Path = config.PROCESSED_DIR) -> dict:
    """Write best_model_* results and models/best_model.joblib. Returns the summary as saved."""
    parts = breakdowns(best.test)
    backtest.to_csv(out / "best_model_backtest.csv", index=False)
    parts["by_price_band"].to_csv(out / "best_model_by_price_band.csv")
    parts["by_county"].to_csv(out / "best_model_by_county.csv")
    pred_cols = [config.TARGET, "Predicted", "CountyOrParish", "City", "LivingArea"]
    best.test[pred_cols].to_csv(out / "best_model_test_predictions.csv", index=False)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(build_bundle(best, df, test_month, n_train_months), BEST_MODEL_PATH, compress=3)
    summary = {**summary, "test_metrics": best.metrics,
               "accuracy_bands": accuracy_bands(best.test[config.TARGET], best.test["Predicted"]),
               "model_file_mb": round(BEST_MODEL_PATH.stat().st_size / 1e6, 1)}
    (out / "best_model_metrics.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def save_tuned_params(updates: dict[str, dict], out: Path = config.PROCESSED_DIR) -> dict:
    """Merge validation-tuned hyperparameters per model into tuned_params.json."""
    path = out / "tuned_params.json"
    current = json.loads(path.read_text()) if path.exists() else {}
    current.update({k: {p: v for p, v in params.items() if p != "n_jobs"} for k, params in updates.items()})
    path.write_text(json.dumps(current, indent=2, default=str))
    return current


def load_tuned_params(out: Path = config.PROCESSED_DIR) -> dict[str, dict]:
    """Tuned params with n_jobs restored for models that accept it."""
    params = json.loads((out / "tuned_params.json").read_text())
    for name, p in params.items():
        if name in ("Random Forest",):
            p["n_jobs"] = -1
    return params
