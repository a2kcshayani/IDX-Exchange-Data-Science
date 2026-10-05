"""Evaluation metrics (best-practices §08). MdAPE is the headline figure."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ape = np.abs(y_pred - y_true) / y_true
    return {
        "MdAPE": float(np.median(ape) * 100),
        "MAPE": float(np.mean(ape) * 100),
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "n": int(len(y_true)),
    }


def metrics_by_price_band(y_true, y_pred, n_bands: int = 5) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y_true, float), "p": np.asarray(y_pred, float)})
    df["band"] = pd.qcut(df["y"], n_bands, labels=[f"Q{i + 1}" for i in range(n_bands)])
    return df.groupby("band", observed=True).apply(
        lambda g: pd.Series(regression_metrics(g["y"], g["p"])), include_groups=False
    )
