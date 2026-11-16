"""Week 8: evaluation beyond R² (best-practices §08-09).

Accuracy bands, error by segment (price band, county, size, age), and a paired
bootstrap that says whether one model's MdAPE edge over another is bigger than
test-month sampling noise.
"""
import numpy as np
import pandas as pd

from . import config
from .metrics import regression_metrics

SIZE_BINS = [0, 1200, 1800, 2500, 3500, np.inf]
SIZE_LABELS = ["< 1,200 sq ft", "1,200–1,800", "1,800–2,500", "2,500–3,500", "3,500+ sq ft"]
AGE_BINS = [-np.inf, 10, 30, 50, 70, np.inf]
AGE_LABELS = ["< 10 yrs", "10–30 yrs", "30–50 yrs", "50–70 yrs", "70+ yrs"]


def accuracy_bands(y_true, y_pred) -> dict[str, float]:
    """Share of homes within 5/10/20% of the sale price, plus the median signed error (bias)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    signed = (y_pred - y_true) / y_true
    ape = np.abs(signed)
    return {
        "within_5pct": float((ape <= 0.05).mean() * 100),
        "within_10pct": float((ape <= 0.10).mean() * 100),
        "within_20pct": float((ape <= 0.20).mean() * 100),
        "median_signed_error_pct": float(np.median(signed) * 100),
    }


def full_metrics(y_true, y_pred) -> dict[str, float]:
    return {**regression_metrics(y_true, y_pred), **accuracy_bands(y_true, y_pred)}


def price_band_edges(prices, n_bands: int = 5) -> np.ndarray:
    """Quantile edges from one set of prices, so every model is scored on the same bands."""
    edges = np.quantile(np.asarray(prices, float), np.linspace(0, 1, n_bands + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def band_labels(prices, edges) -> list[str]:
    s = pd.Series(np.asarray(prices, float))
    cut = pd.cut(s, edges, labels=False, include_lowest=True)
    out = []
    for i in range(len(edges) - 1):
        part = s[cut == i]
        out.append(f"Q{i + 1}: ${part.min() / 1e3:,.0f}K–${part.max() / 1e3:,.0f}K" if len(part) else f"Q{i + 1}")
    return out


def segment_table(test: pd.DataFrame, segment: pd.Series, min_n: int = 100) -> pd.DataFrame:
    """Metrics for each value of `segment` with at least min_n test sales."""
    rows = []
    for name, g in test.groupby(segment, observed=True, sort=False):
        if len(g) >= min_n:
            rows.append({"segment": str(name), **full_metrics(g[config.TARGET], g["Predicted"])})
    return pd.DataFrame(rows)


def segment_breakdowns(test: pd.DataFrame, edges: np.ndarray, actual_labels: list[str]) -> pd.DataFrame:
    """Long table: segment_type, segment, metrics. Price bands use the shared actual-price edges."""
    parts = {
        "Price band (sale price)": pd.cut(test[config.TARGET], edges, labels=actual_labels, include_lowest=True),
        # What a user sees at prediction time: bands of the model's own estimate.
        "Price band (model estimate)": pd.cut(test["Predicted"], edges, labels=actual_labels, include_lowest=True),
        "County": test["CountyOrParish"],
        "Living area": pd.cut(test["LivingArea"], SIZE_BINS, labels=SIZE_LABELS, right=False),
        "Home age": pd.cut(test["PropertyAge"], AGE_BINS, labels=AGE_LABELS, right=False),
    }
    order = {"Price band (sale price)": actual_labels, "Price band (model estimate)": actual_labels,
             "Living area": SIZE_LABELS, "Home age": AGE_LABELS}
    out = []
    for seg_type, seg in parts.items():
        t = segment_table(test, seg, min_n=100 if seg_type == "County" else 30)
        if t.empty:
            continue
        if seg_type in order:
            t["_o"] = t["segment"].map({v: i for i, v in enumerate(order[seg_type])})
            t = t.sort_values("_o").drop(columns="_o")
        else:
            t = t.sort_values("n", ascending=False)
        t.insert(0, "segment_type", seg_type)
        out.append(t)
    return pd.concat(out, ignore_index=True)


def paired_bootstrap(y_true, pred_a, pred_b, n_boot: int = 2000, seed: int = config.RANDOM_SEED) -> dict:
    """MdAPE(a) − MdAPE(b) on the same homes, resampled with replacement.

    Negative = model a is better. If the 95% interval excludes 0, the gap is bigger than
    what re-drawing the test month's homes would produce by chance.
    """
    y = np.asarray(y_true, float)
    ape_a = np.abs(np.asarray(pred_a, float) - y) / y * 100
    ape_b = np.abs(np.asarray(pred_b, float) - y) / y * 100
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        diffs[i] = np.median(ape_a[idx]) - np.median(ape_b[idx])
    return {
        "mdape_diff_pts": float(np.median(ape_a) - np.median(ape_b)),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "share_of_resamples_a_better": float((diffs < 0).mean() * 100),
        # Per-home view: how often model a's error is smaller than model b's.
        "homes_a_closer_pct": float((ape_a < ape_b).mean() * 100),
    }
