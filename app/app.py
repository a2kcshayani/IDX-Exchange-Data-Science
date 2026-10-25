"""Streamlit app presenting the California close-price AVM project.

Run from the repo root after the pipeline scripts:
    python scripts/run_preprocessing.py
    python scripts/run_baseline.py
    streamlit run app/app.py

Reads only local, git-ignored artifacts (data/processed/, models/).
"""
import json
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from idx_avm import config  # noqa: E402  (also needed to unpickle QuantileClipper)
from idx_avm.load import load_processed  # noqa: E402

PROCESSED = config.PROCESSED_DIR
BASELINE_PATH = ROOT / "models" / "baseline_linear.joblib"
BEST_PATH = ROOT / "models" / "best_model.joblib"
MODEL_PATH = BEST_PATH if BEST_PATH.exists() else BASELINE_PATH
FEATURE_LABELS = {
    "comp_log_value": "Comps value (ZIP $/sq ft × size)", "comp_log_ppsf": "Comps $/sq ft (ZIP, prior sales)",
    "PostalCode": "ZIP code", "City": "City", "CountyOrParish": "County", "LivingArea": "Living area",
    "SchoolDistrictK8": "School district (K-8)", "SchoolDistrictHigh": "School district (high school)",
    "BathroomsTotalInteger": "Bathrooms", "BedroomsTotal": "Bedrooms", "LotSizeSquareFeet": "Lot size",
    "LotToLivingRatio": "Lot / living area", "BedBathRatio": "Bed / bath ratio", "PropertyAge": "Age",
    "Latitude": "Latitude", "Longitude": "Longitude", "GarageSpaces": "Garage spaces", "ParkingTotal": "Parking",
    "Stories": "Stories", "PoolPrivateYN": "Pool", "ViewYN": "View", "CloseMonthSin": "Season (sin)",
    "CloseMonthCos": "Season (cos)",
}
MODEL_COLORS = {"Linear Regression": "#94a3b8", "Decision Tree": "#60a5fa", "Random Forest": "#2563eb"}
ACCENT = "#2563eb"

st.set_page_config(page_title="California Home Price Model", page_icon="🏠", layout="wide")


# --------------------------------------------------------------------------- data access
def _need(*paths: Path) -> bool:
    missing = [p for p in paths if not p.exists()]
    if missing:
        st.warning(
            "Missing pipeline outputs: " + ", ".join(p.name for p in missing)
            + ".\n\nRun `python scripts/run_preprocessing.py` then `python scripts/run_baseline.py`."
        )
        return False
    return True


def _mtime(name: str) -> float:
    """Part of each cache key, so re-running the pipeline invalidates cached results."""
    return (PROCESSED / name).stat().st_mtime


def read_csv(name: str) -> pd.DataFrame:
    return _read_csv(name, _mtime(name))


@st.cache_data
def _read_csv(name: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(PROCESSED / name)


def read_json(name: str) -> dict:
    return _read_json(name, _mtime(name))


@st.cache_data
def _read_json(name: str, mtime: float) -> dict:
    return json.loads((PROCESSED / name).read_text())


def cleaned_rows() -> pd.DataFrame:
    return _cleaned_rows(_mtime("train_clean.csv"), _mtime("test_clean.csv"))


@st.cache_data
def _cleaned_rows(train_mtime: float, test_mtime: float) -> pd.DataFrame:
    train, test = load_processed("train"), load_processed("test")
    train["Split"], test["Split"] = "Train", "Test"
    df = pd.concat([train, test], ignore_index=True)
    df["CloseMonth"] = df["CloseMonth"].astype(str)
    return df


def load_bundle() -> dict:
    return _load_bundle(MODEL_PATH.stat().st_mtime)


@st.cache_resource
def _load_bundle(mtime: float) -> dict:
    return joblib.load(MODEL_PATH)


def money(x: float) -> str:
    return f"${x:,.0f}"


# --------------------------------------------------------------------------- pages
def page_overview() -> None:
    st.title("Predicting California Home Sale Prices")
    st.markdown(
        "**IDX Exchange · Data Science Internship.** This is an automated valuation model (AVM) that estimates the "
        "**close price of any California single-family home**, whether it's on the market or not."
    )
    if not _need(PROCESSED / "baseline_metrics.json", PROCESSED / "split_metadata.json"):
        return
    m = read_json("baseline_metrics.json")
    meta = read_json("split_metadata.json")
    counts, tm = m["pipeline_counts"], m["test_metrics"]

    best = read_json("best_model_metrics.json") if (PROCESSED / "best_model_metrics.json").exists() else None
    c = st.columns(4)
    c[0].metric("MLS sales records", f"{counts['raw']:,}")
    c[1].metric("Single-family sales modeled", f"{counts['cleaned']:,}")
    if best:
        bm = best["test_metrics"]
        c[2].metric(f"Typical error · {best['selected_model']}", f"{bm['MdAPE']:.1f}%",
                    delta=f"{bm['MdAPE'] - tm['MdAPE']:.1f} pts vs. baseline", delta_color="inverse")
        c[3].metric("R² on unseen month", f"{bm['R2']:.3f}", delta=f"+{bm['R2'] - tm['R2']:.3f} vs. baseline")
    else:
        c[2].metric("Typical error (MdAPE)", f"{tm['MdAPE']:.1f}%")
        c[3].metric("R² on unseen month", f"{tm['R2']:.3f}",
                    delta=f"+{tm['R2'] - m['unclipped_test_metrics']['R2']:.3f} vs. before fix")

    st.subheader("The rule that shapes everything")
    st.markdown(
        "The model has to value homes that **aren't for sale**, so it only uses facts that exist for *any* address: "
        "size, bedrooms, bathrooms, lot, age, and location. Listing-only fields like **list price** and "
        "**days on market** are banned. They'd make the scores look better, but the model couldn't value an "
        "off-market home."
    )

    st.subheader("How it works")
    steps = pd.DataFrame(
        [
            ("1. Load", f"{len(meta['source_files'])} monthly CRMLS files, Jan 2024 – Apr 2026"),
            ("2. Clean", "Remove invalid prices, duplicates and impossible records, with a logged count at each step"),
            ("3. Split by time", f"Train on {len(meta['train_months'])} months, test on {meta['test_month']} (never shuffled)"),
            ("4. Preprocess", "Clip outliers, impute, encode, scale. All fit on training data only"),
            ("5. Model", "Linear Regression baseline (Week 4), then Decision Tree and Random Forest (Week 5)"),
            ("6. Features", "Bed/bath ratios, school districts via spatial join, comps from prior sales (Week 6)"),
            ("7. Evaluate", "MdAPE (headline), MAPE, R², MAE, plus a backtest on 3 months"),
        ],
        columns=["Step", "What happens"],
    )
    st.dataframe(steps, hide_index=True, width="stretch")
    st.caption("Built with Python · pandas · NumPy · scikit-learn · Streamlit · Plotly")


def page_data() -> None:
    st.title("Data & Cleaning")
    if not _need(PROCESSED / "cleaning_log.csv", PROCESSED / "train_clean.csv"):
        return
    m = read_json("baseline_metrics.json") if (PROCESSED / "baseline_metrics.json").exists() else None
    log = read_csv("cleaning_log.csv")

    if m:
        counts = m["pipeline_counts"]
        funnel = go.Figure(go.Funnel(
            y=["All MLS sales", "Single-family residential", "After cleaning"],
            x=[counts["raw"], counts["single_family"], counts["cleaned"]],
            marker_color=[ACCENT, "#60a5fa", "#93c5fd"],
            texttemplate="%{value:,} (%{percentInitial:.0%})",
        ))
        funnel.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(funnel, width="stretch")

    st.subheader("Cleaning log")
    st.caption("Every rule records how many rows it removed. Rows aren't dropped silently.")
    shown = log[(log["removed"] > 0) | log["note"].fillna("").str.contains("NaN|values")]
    st.dataframe(
        shown[["step", "removed", "pct_removed", "note"]].fillna("").rename(
            columns={"step": "Rule", "removed": "Rows removed", "pct_removed": "% removed", "note": "Note"}
        ),
        hide_index=True, width="stretch",
    )

    df = cleaned_rows()
    st.subheader("What the data looks like")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x=np.log10(df[config.TARGET]), nbins=80, color_discrete_sequence=[ACCENT])
        fig.update_layout(xaxis_title="log10(sale price)", yaxis_title="sales", height=340, margin=dict(t=30))
        fig.update_xaxes(tickvals=[5.5, 6, 6.5], ticktext=["$316K", "$1M", "$3.2M"])
        st.markdown("**Sale prices are heavily right-skewed**, so the model predicts log(price)")
        st.plotly_chart(fig, width="stretch")
    with c2:
        monthly = df.groupby("CloseMonth")[config.TARGET].agg(sales="size", median="median").reset_index()
        fig = go.Figure()
        fig.add_bar(x=monthly["CloseMonth"], y=monthly["sales"], name="Sales", marker_color="#cbd5e1", yaxis="y2")
        fig.add_scatter(x=monthly["CloseMonth"], y=monthly["median"], name="Median price", line=dict(color=ACCENT, width=3))
        fig.update_layout(
            height=340, margin=dict(t=30), legend=dict(orientation="h", y=1.1),
            yaxis=dict(title="Median price", tickprefix="$"),
            yaxis2=dict(title="Sales", overlaying="y", side="right", showgrid=False),
        )
        st.markdown("**Market level drifts month to month**, which is why we split by time")
        st.plotly_chart(fig, width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        county = (df.groupby("CountyOrParish")[config.TARGET].agg(sales="size", median="median")
                  .query("sales >= 500").sort_values("median").tail(15).reset_index())
        fig = px.bar(county, x="median", y="CountyOrParish", orientation="h", color_discrete_sequence=[ACCENT])
        fig.update_layout(height=420, xaxis_title="Median sale price", yaxis_title="", margin=dict(t=30))
        fig.update_xaxes(tickprefix="$")
        st.markdown("**Location dominates price.** County medians differ by multiples")
        st.plotly_chart(fig, width="stretch")
    with c4:
        sample = df.sample(min(6000, len(df)), random_state=config.RANDOM_SEED)
        fig = px.scatter(sample, x="LivingArea", y=config.TARGET, opacity=0.25, log_x=True, log_y=True,
                         color_discrete_sequence=[ACCENT])
        fig.update_traces(marker_size=4)
        fig.update_layout(height=420, xaxis_title="Living area (sq ft)", yaxis_title="Sale price", margin=dict(t=30))
        st.markdown("**Bigger homes cost more.** Size is the strongest single feature")
        st.plotly_chart(fig, width="stretch")


def page_design() -> None:
    st.title("Train/Test Design & Leakage")
    if not _need(PROCESSED / "split_metadata.json", PROCESSED / "window_tuning_summary.csv"):
        return
    meta = read_json("split_metadata.json")

    st.subheader("Split by time, the way the model is actually used")
    st.markdown(
        f"Production models predict the future from the past, so the test set is the **most recent month "
        f"({meta['test_month']})** and the model trains only on the months before it. Nothing is shuffled."
    )
    months = pd.period_range(meta["train_months"][0], meta["test_month"], freq="M").astype(str)
    timeline = pd.DataFrame({"month": months, "role": ["Train"] * (len(months) - 1) + ["Test"], "y": 1})
    fig = px.bar(timeline, x="month", y="y", color="role", color_discrete_map={"Train": "#93c5fd", "Test": "#f97316"})
    fig.update_layout(height=170, yaxis_visible=False, xaxis_title="", margin=dict(t=10, b=10), bargap=0.08,
                      legend=dict(orientation="h", y=1.25, title=""))
    st.plotly_chart(fig, width="stretch")

    st.subheader("How many months of history to train on?")
    st.markdown(
        "More months means more examples, but older sales reflect older prices. Each window length was scored on "
        "**3 validation months before the test month**, never on the test month itself."
    )
    tune = read_csv("window_tuning_summary.csv").sort_values("X")
    best = int(meta["chosen_window_months"])
    fig = px.line(tune, x="X", y="MdAPE", markers=True, color_discrete_sequence=[ACCENT])
    fig.add_vline(x=best, line_dash="dash", line_color="#f97316", annotation_text=f"chosen: {best} months")
    fig.update_layout(height=340, xaxis_title="Training window (months)", yaxis_title="Validation MdAPE (%)")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Leakage audit")
    st.markdown(
        "These columns exist only because a home was listed or sold. Their correlation with sale price shows "
        "why they'd inflate accuracy. **None of them are model inputs**, and the code raises an error if one slips in."
    )
    if (PROCESSED / "leakage_audit.csv").exists():
        audit = read_csv("leakage_audit.csv")
        audit = audit.dropna(subset=["spearman_corr_with_ClosePrice"]).sort_values("spearman_corr_with_ClosePrice", ascending=False)
        st.dataframe(
            audit.rename(columns={"column": "Column", "spearman_corr_with_ClosePrice": "Correlation with sale price", "verdict": "Verdict"}),
            hide_index=True, width="stretch",
        )
    st.markdown("**Model inputs:** " + ", ".join(f"`{c}`" for g in meta["feature_groups"].values() for c in g))


def page_results() -> None:
    st.title("Baseline Model Results (Week 4)")
    if not _need(PROCESSED / "baseline_metrics.json", PROCESSED / "baseline_test_predictions.csv"):
        return
    m = read_json("baseline_metrics.json")
    tm, raw, acc = m["test_metrics"], m["unclipped_test_metrics"], m["accuracy_bands"]
    st.markdown(
        f"**Linear Regression on log(price)**, trained on {m['n_train_months']} months and tested on "
        f"**{m['test_month']}**: {tm['n']:,} sales the model never saw."
    )
    c = st.columns(5)
    c[0].metric("MdAPE (headline)", f"{tm['MdAPE']:.2f}%")
    c[1].metric("MAPE", f"{tm['MAPE']:.2f}%")
    c[2].metric("R²", f"{tm['R2']:.3f}")
    c[3].metric("MAE", money(tm["MAE"]))
    c[4].metric("Within 10% of sale price", f"{acc['within_10pct']:.1f}%")

    st.subheader("What improved: train-only outlier clipping")
    st.markdown(
        "Raw MLS fields had entry errors, like a **24.9-billion sq ft lot** and **500 garage spaces**. A linear "
        "model turns those into absurd predictions. Clipping each feature to limits learned from the training data "
        "fixed it:"
    )
    comp = pd.DataFrame(
        {"Before (raw features)": raw, "After (clipped)": tm}
    ).loc[["MdAPE", "MAPE", "R2", "MAE", "RMSE"]]
    comp["Change"] = comp["After (clipped)"] - comp["Before (raw features)"]
    st.dataframe(comp.style.format("{:,.3f}"), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Stable across months (rolling-origin backtest)**")
        bt = read_csv("baseline_backtest.csv")
        fig = px.bar(bt, x="test_month", y="MdAPE", text=bt["MdAPE"].round(2), color_discrete_sequence=[ACCENT])
        fig.update_layout(height=320, yaxis_title="MdAPE (%)", xaxis_title="Test month", yaxis_range=[0, bt["MdAPE"].max() * 1.3])
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown("**Error by price range.** Luxury homes are hardest")
        band = read_csv("baseline_by_price_band.csv")
        fig = px.bar(band, x="price_range", y="MdAPE", text=band["MdAPE"].round(1), color_discrete_sequence=[ACCENT])
        fig.update_layout(height=320, yaxis_title="MdAPE (%)", xaxis_title="")
        st.plotly_chart(fig, width="stretch")

    pred = read_csv("baseline_test_predictions.csv")
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Predicted vs actual**")
        s = pred.sample(min(5000, len(pred)), random_state=config.RANDOM_SEED)
        fig = px.scatter(s, x=config.TARGET, y="Predicted", opacity=0.3, log_x=True, log_y=True,
                         color_discrete_sequence=[ACCENT])
        fig.update_traces(marker_size=4)
        lo, hi = pred[config.TARGET].min(), pred[config.TARGET].max()
        fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="#111", dash="dash"), name="perfect")
        fig.update_layout(height=380, xaxis_title="Actual sale price", yaxis_title="Predicted", showlegend=False)
        st.plotly_chart(fig, width="stretch")
    with c4:
        st.markdown("**How close are predictions?**")
        bands = pd.DataFrame({"Within": ["5%", "10%", "20%"],
                              "Share of homes": [acc["within_5pct"], acc["within_10pct"], acc["within_20pct"]]})
        fig = px.bar(bands, x="Within", y="Share of homes", text=bands["Share of homes"].round(1).astype(str) + "%",
                     color_discrete_sequence=[ACCENT])
        fig.update_layout(height=380, yaxis_range=[0, 100], yaxis_title="% of test homes")
        st.plotly_chart(fig, width="stretch")

    if (PROCESSED / "baseline_by_county.csv").exists():
        st.markdown("**Error by county** (counties with 100+ test sales)")
        county = read_csv("baseline_by_county.csv")
        st.dataframe(
            county[["CountyOrParish", "n", "MdAPE", "MAPE", "MAE"]].rename(columns={"CountyOrParish": "County", "n": "Sales"})
            .style.format({"MdAPE": "{:.1f}%", "MAPE": "{:.1f}%", "MAE": "${:,.0f}", "Sales": "{:,.0f}"}),
            hide_index=True, width="stretch",
        )


def page_models() -> None:
    st.title("Model Comparison (Week 5)")
    if not _need(PROCESSED / "week5_model_comparison.csv", PROCESSED / "week5_tuning_trials.csv"):
        return
    comp = read_csv("week5_model_comparison.csv")
    st.markdown(
        "Three models on the **same features and the same pipeline**. Each model's settings were tuned on the "
        "**validation month**, and the **test month** was scored once at the end, so the comparison is fair."
    )
    base = comp.set_index("model").loc["Linear Regression"]
    cols = st.columns(len(comp))
    for col, (_, r) in zip(cols, comp.iterrows()):
        delta = None if r["model"] == "Linear Regression" else f"{r['MdAPE'] - base['MdAPE']:.2f} pts vs. baseline"
        col.metric(r["model"], f"{r['MdAPE']:.2f}% MdAPE", delta=delta, delta_color="inverse")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Typical error (lower is better)**")
        fig = px.bar(comp, x="model", y="MdAPE", color="model", color_discrete_map=MODEL_COLORS,
                     text=comp["MdAPE"].round(2).astype(str) + "%")
        fig.update_layout(height=340, showlegend=False, xaxis_title="", yaxis_title="Test MdAPE (%)")
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown("**Variance explained (higher is better)**")
        fig = px.bar(comp, x="model", y="R2", color="model", color_discrete_map=MODEL_COLORS,
                     text=comp["R2"].round(3))
        fig.update_layout(height=340, showlegend=False, xaxis_title="", yaxis_title="Test R²", yaxis_range=[0, 1])
        st.plotly_chart(fig, width="stretch")

    st.subheader("Full test-month results")
    show = comp[["model", "params", "MdAPE", "MAPE", "R2", "MAE", "RMSE", "fit_seconds"]].rename(
        columns={"model": "Model", "params": "Chosen settings", "fit_seconds": "Train time (s)"})
    st.dataframe(show.style.format({"MdAPE": "{:.2f}%", "MAPE": "{:.2f}%", "R2": "{:.3f}", "MAE": "${:,.0f}",
                                    "RMSE": "${:,.0f}", "Train time (s)": "{:.1f}"}),
                 hide_index=True, width="stretch")

    st.subheader("How the settings were chosen")
    trials = read_csv("week5_tuning_trials.csv")
    trials = trials[trials["model"] != "Linear Regression"].copy()
    short = {"max_depth": "depth", "min_samples_leaf": "min leaf", "max_features": "features/split", "n_estimators": "trees"}
    for k, v in short.items():
        trials["params"] = trials["params"].str.replace(k, v)
    trials["setting"] = trials["model"] + " · " + trials["params"]
    best_idx = trials.groupby("model")["MdAPE"].idxmin()
    trials["chosen"] = np.where(trials.index.isin(best_idx), "Chosen", "Tried")
    trials["label"] = trials["MdAPE"].round(2).astype(str) + "%"
    trials = trials.sort_values(["model", "MdAPE"], ascending=[True, False])
    fig = px.bar(trials, x="MdAPE", y="setting", color="chosen", orientation="h", text="label",
                 color_discrete_map={"Chosen": ACCENT, "Tried": "#cbd5e1"})
    fig.update_yaxes(categoryorder="array", categoryarray=trials["setting"].tolist())
    fig.update_layout(height=430, yaxis_title="", legend_title="",
                      xaxis_title=f"Validation MdAPE (%) · trained on the 18 months before {trials['val_month'].iloc[0]}")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Strengths and weaknesses")
    st.markdown(
        """
| Model | Strengths | Weaknesses |
|---|---|---|
| **Linear Regression** | Fast, interpretable, stable across months | One straight-line effect per feature; misses how location and size interact |
| **Decision Tree** | Learns non-linear effects and interactions on its own | Jumpy, high-variance predictions; overfits without depth limits |
| **Random Forest** | Averages hundreds of trees, so much lower variance and the best accuracy here | Slower, larger, harder to explain; can't predict outside the price range it trained on |
"""
    )


def page_features() -> None:
    st.title("Feature Engineering (Week 6)")
    if not _need(PROCESSED / "week6_feature_sets.csv"):
        return
    fs = read_csv("week6_feature_sets.csv")
    order = list(dict.fromkeys(fs["feature_set"]))
    st.markdown(
        """
Each feature set **adds** to the one before it, and every model is re-trained on each set.

| Set | Adds | Allowed because |
|---|---|---|
| **Base** | size, beds, baths, lot, age, coordinates, county / city / ZIP | property facts and location |
| **+ Ratios** | bed/bath ratio, lot-to-living-area ratio | arithmetic on property facts |
| **+ School districts** | K-8 and high-school district from a **spatial join** against CA School District Areas 2024-25 | boundaries are fixed by location |
| **+ Comp PPSF** | ZIP price per sq ft from **prior sales in the training window**, plus the implied comps value | learned on training sales only; never uses the home's own price |
"""
    )

    st.subheader("Typical error by feature set")
    fig = px.line(fs, x="feature_set", y="test_MdAPE", color="model", markers=True, color_discrete_map=MODEL_COLORS,
                  category_orders={"feature_set": order})
    fig.update_layout(height=380, xaxis_title="", yaxis_title="Test MdAPE (%)", legend_title="")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Old vs. new feature set")
    old = fs[fs["feature_set"] == order[0]].set_index("model")
    new = fs[fs["feature_set"] == order[-1]].set_index("model")
    table = pd.DataFrame({
        "MdAPE, base": old["test_MdAPE"], "MdAPE, all features": new["test_MdAPE"],
        "Change (pts)": new["test_MdAPE"] - old["test_MdAPE"],
        "R², base": old["test_R2"], "R², all features": new["test_R2"], "Change in R²": new["test_R2"] - old["test_R2"],
    })
    st.dataframe(table.style.format("{:.3f}").format("{:.2f}", subset=["MdAPE, base", "MdAPE, all features", "Change (pts)"]),
                 width="stretch")

    with st.expander("Every model × feature set (validation and test)"):
        st.dataframe(fs.style.format({c: "{:.3f}" for c in fs.columns if c.startswith(("val_", "test_"))}),
                     hide_index=True, width="stretch")

    if (PROCESSED / "best_model_metrics.json").exists():
        best = read_json("best_model_metrics.json")
        st.success(
            f"**Selected model: {best['selected_model']} with '{best['selected_feature_set']}'**, chosen by validation-month "
            f"MdAPE, not by the test month. Test MdAPE **{best['test_metrics']['MdAPE']:.2f}%**, R² "
            f"**{best['test_metrics']['R2']:.3f}**, versus the baseline's "
            f"{best['baseline_test_metrics']['MdAPE']:.2f}% and {best['baseline_test_metrics']['R2']:.3f}."
        )
        same = fs[fs["model"] == best["selected_model"]].set_index("feature_set")["val_MdAPE"]
        gap = same[best["selected_feature_set"]] - same[order[0]]
        if abs(gap) < 0.1:
            st.caption(f"Honest read: on validation this is only {abs(gap):.2f} pts better than the same model on base "
                       "features, which is effectively a tie. The new features mainly help the linear model.")

    if (PROCESSED / "rf_feature_importance.csv").exists():
        st.subheader("What the Random Forest relies on (all features)")
        imp = read_csv("rf_feature_importance.csv")
        imp["feature"] = imp["feature"].map(FEATURE_LABELS).fillna(imp["feature"])
        fig = px.bar(imp.sort_values("importance"), x="importance", y="feature", orientation="h",
                     color_discrete_sequence=[ACCENT])
        fig.update_layout(height=460, xaxis_title="Share of total importance", yaxis_title="")
        st.plotly_chart(fig, width="stretch")
        st.caption("The comps value and ZIP/city encodings carry most of the signal. That's also why the new features "
                   "barely move the Random Forest: it was already learning neighborhood price levels from ZIP and city.")

    st.subheader("The school-district layer")
    st.caption("Median sale price by high-school district (districts with 300+ sales in the train/test window)")
    df = school_rows()
    med = (df.groupby("SchoolDistrictHigh")[config.TARGET].agg(sales="size", median="median")
           .query("sales >= 300").sort_values("median"))
    pick = pd.concat([med.head(8), med.tail(8)]).reset_index()
    fig = px.bar(pick, x="median", y="SchoolDistrictHigh", orientation="h", color_discrete_sequence=[ACCENT])
    fig.update_layout(height=520, xaxis_title="Median sale price", yaxis_title="")
    fig.update_xaxes(tickprefix="$")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"{df['SchoolDistrictHigh'].notna().mean():.1%} of homes matched to a district. "
               "Demographic fields in the district file are deliberately not used, to avoid fair-housing issues.")


def school_rows() -> pd.DataFrame:
    return _school_rows(_mtime("train_clean.csv"), _mtime("test_clean.csv"))


@st.cache_data
def _school_rows(train_mtime: float, test_mtime: float) -> pd.DataFrame:
    from idx_avm.geo import assign_school_districts

    return assign_school_districts(_cleaned_rows(train_mtime, test_mtime))


def page_estimator() -> None:
    st.title("Price Estimator")
    if not _need(MODEL_PATH):
        return
    b = load_bundle()
    opts, d = b["options"], b["defaults"]
    label = f"{b['model_name']} ({b['feature_set']})" if "model_name" in b else "Week 4 Linear Regression baseline"
    st.caption(f"Uses the saved {label}. Its typical error on unseen homes is ±{b['metrics']['MdAPE']:.1f}% (MdAPE).")

    # Location sits outside the form so City/ZIP lists refresh as soon as County/City change.
    c1, c2, c3 = st.columns(3)
    counties = opts["CountyOrParish"]
    county = c1.selectbox("County", counties, index=counties.index("Los Angeles") if "Los Angeles" in counties else 0)
    city = c2.selectbox("City", opts["city_by_county"].get(county, []))
    zips = opts["zip_by_city"].get(city, [])
    zip_code = c3.selectbox("ZIP code", zips) if zips else None

    with st.form("estimate"):
        c4, c5, c6, c7 = st.columns(4)
        sqft = c4.number_input("Living area (sq ft)", 300, 15000, int(d.get("LivingArea", 1800)), step=50)
        beds = c5.number_input("Bedrooms", 0, 12, int(d.get("BedroomsTotal", 3)))
        baths = c6.number_input("Bathrooms", 0, 12, int(d.get("BathroomsTotalInteger", 2)))
        lot = c7.number_input("Lot size (sq ft)", 0, 500000, int(d.get("LotSizeSquareFeet", 7000)), step=100)

        c8, c9, c10, c11 = st.columns(4)
        year_built = c8.number_input("Year built", 1850, date.today().year, date.today().year - int(d.get("PropertyAge", 50)))
        garage = c9.number_input("Garage spaces", 0, 10, int(d.get("GarageSpaces", 2)))
        stories = c10.selectbox("Stories", [1, 2, 3], index=0)
        pool = c11.checkbox("Private pool")
        view = c11.checkbox("View")
        submitted = st.form_submit_button("Estimate price", type="primary")

    if not submitted:
        st.info("Pick a location and home details, then click **Estimate price**.")
        return

    ll = opts["latlon_by_zip"].get(zip_code, {}) if zip_code else {}
    today = date.today()
    row = {
        "LivingArea": sqft, "BedroomsTotal": beds, "BathroomsTotalInteger": baths, "LotSizeSquareFeet": lot or np.nan,
        "Latitude": ll.get("Latitude", np.nan), "Longitude": ll.get("Longitude", np.nan),
        "GarageSpaces": garage, "ParkingTotal": garage, "Stories": stories, "FireplacesTotal": np.nan,
        "PropertyAge": today.year - year_built,
        "CloseMonthSin": np.sin(2 * np.pi * today.month / 12), "CloseMonthCos": np.cos(2 * np.pi * today.month / 12),
        "PoolPrivateYN": float(pool), "ViewYN": float(view), "FireplaceYN": np.nan,
        "AttachedGarageYN": np.nan, "NewConstructionYN": float(year_built >= today.year - 1),
        "CountyOrParish": county, "City": city, "PostalCode": zip_code,
    }
    # Week 6 features: ratios from the inputs, school districts from the ZIP's coordinates.
    row["BedBathRatio"] = beds / baths if baths else np.nan
    row["LotToLivingRatio"] = (lot / sqft) if lot else np.nan
    schools = {"SchoolDistrictK8": None, "SchoolDistrictHigh": None}
    if any(c in b["features"] for c in schools) and ll:
        from idx_avm.geo import district_for_point

        schools = district_for_point(ll["Latitude"], ll["Longitude"])
    row.update(schools)
    X = pd.DataFrame([row])
    missing = [f for f in b["features"] if f not in X.columns]
    assert not missing, f"App inputs don't match the model's schema: {missing}"
    X = X[b["features"]]
    for col in ("CountyOrParish", "City", "PostalCode", "SchoolDistrictK8", "SchoolDistrictHigh"):
        if col in X.columns:
            X[col] = X[col].astype("string")
    price = float(b["model"].predict(X)[0])
    err = b["metrics"]["MdAPE"] / 100

    c1, c2 = st.columns([1, 2])
    c1.metric("Estimated sale price", money(price))
    # Escape "$" so Streamlit markdown doesn't read the range as LaTeX.
    c1.caption(f"Typical range: {money(price * (1 - err))} – {money(price * (1 + err))}".replace("$", "\\$"))
    c2.markdown(
        f"A **{beds} bed / {baths} bath, {sqft:,} sq ft** home in **{city}, {zip_code}** ({county} County). "
        "List price and days on market aren't used, so this works for homes that aren't for sale."
    )
    if schools.get("SchoolDistrictHigh"):
        c2.caption(f"School districts (spatial join): {schools['SchoolDistrictK8']} · {schools['SchoolDistrictHigh']}")


def page_next() -> None:
    st.title("What's Next")
    st.markdown(
        """
| Week | Plan |
|---|---|
| ✅ 1–4 | Setup, exploration, leakage-safe preprocessing, Linear Regression baseline |
| ✅ 5 | Decision Tree and Random Forest vs. the baseline |
| ✅ 6 | Feature engineering: ratios, school-district spatial join, comps from prior sales |
| 7 | Gradient boosting (XGBoost / LightGBM) with tuning on validation months |
| 8 | Full error analysis: MAPE / MdAPE by price band and geography |
| 9–12 | App polish, documentation, final presentation |

**Where the models still struggle:** the most expensive homes (top 20% of prices) have the highest error.
Gradient boosting and stronger comp features are the next things to try there.
"""
    )

PAGES = {
    "Overview": page_overview,
    "Data & Cleaning": page_data,
    "Train/Test Design": page_design,
    "Baseline Results": page_results,
    "Model Comparison": page_models,
    "Feature Engineering": page_features,
    "Price Estimator": page_estimator,
    "What's Next": page_next,
}
with st.sidebar:
    st.markdown("### 🏠 CA Home Price AVM")
    choice = st.radio("Section", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.caption("IDX Exchange · Data Science Internship\n\n[GitHub repo](https://github.com/a2kcshayani/IDX-Exchange-Data-Science)")
PAGES[choice]()
