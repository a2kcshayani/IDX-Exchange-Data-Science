import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from idx_avm.clean import clean  # noqa: E402
from idx_avm.features import add_base_features, audit_features, feature_columns  # noqa: E402
from idx_avm.load import filter_single_family, load_raw  # noqa: E402
from idx_avm.split import add_close_month, apply_outlier_filter, complete_months, window_split  # noqa: E402
from idx_avm.window import tune_training_window  # noqa: E402
from synthetic import write_synthetic  # noqa: E402


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    write_synthetic(raw, months=8)
    df = filter_single_family(load_raw(raw))
    df, log = clean(df)
    return add_close_month(add_base_features(df)), log


def test_cleaning_removes_bad_rows(data):
    df, log = data
    assert (df["ClosePrice"] > 0).all()
    assert (df["LivingArea"] > 0).all()
    assert not df["ListingKey"].duplicated().any()
    assert df["YearBuilt"].max() <= 2026
    assert log.to_frame()["removed"].sum() > 0


def test_no_leaky_features(data):
    df, _ = data
    features = [c for g in feature_columns(df).values() for c in g]
    for leaky in ("ListPrice", "OriginalListPrice", "DaysOnMarket", "ListingContractDate", "ClosePrice"):
        assert leaky not in features
    with pytest.raises(ValueError):
        audit_features(["LivingArea", "ListPrice"])


def test_chronological_split_and_frozen_thresholds(data):
    df, _ = data
    months = complete_months(df)
    split, thr, _ = apply_outlier_filter(window_split(df, months[-1], 3))
    assert split.train["CloseDate"].max() < split.test["CloseDate"].min()
    assert split.test["ClosePrice"].between(thr.lower, thr.upper).all()
    train_raw = window_split(df, months[-1], 3).train
    assert thr.lower == pytest.approx(train_raw["ClosePrice"].quantile(0.005))


def test_window_tuning_never_uses_test_month(data):
    df, _ = data
    months = complete_months(df)
    results, summary = tune_training_window(df, months, [1, 2, 3], n_origins=2)
    assert str(months[-1]) not in set(results["val_month"])
    assert summary["MdAPE"].notna().all()


def test_quantile_clipper_learns_bounds_from_train_only():
    import numpy as np
    from idx_avm.preprocess import QuantileClipper

    train = np.array([[float(v)] for v in range(1, 1001)])
    clipper = QuantileClipper().fit(train)
    out = clipper.transform(np.array([[-1e9], [500.0], [2.49e10]]))
    assert out[0, 0] == clipper.lower_[0] and out[2, 0] == clipper.upper_[0]
    assert out[1, 0] == 500.0


def test_baseline_runs_and_beats_naive_median(data):
    import numpy as np
    from idx_avm.baseline import run_baseline

    df, _ = data
    months = complete_months(df)
    result = run_baseline(df, months[-1], 3)
    test = result.test
    naive = np.median(np.abs(test["ClosePrice"].median() - test["ClosePrice"]) / test["ClosePrice"]) * 100
    assert result.metrics["MdAPE"] < naive
    assert {"MdAPE", "MAPE", "R2", "MAE", "RMSE"} <= result.metrics.keys()


def test_comp_ppsf_encoder_cross_fits_and_handles_unseen_zips():
    import numpy as np
    import pandas as pd
    from idx_avm.preprocess import CompPPSFEncoder

    X = pd.DataFrame({"PostalCode": ["90001"] * 50 + ["94301"] * 50, "LivingArea": [1000.0] * 100})
    y = np.log(np.r_[np.full(50, 500_000.0), np.full(50, 2_000_000.0)])
    enc = CompPPSFEncoder(smoothing=0)
    train_out = enc.fit_transform(X, y)
    # Expensive ZIP gets a higher comp PPSF than the cheap one.
    assert train_out[50:, 0].mean() > train_out[:50, 0].mean()
    new = enc.transform(pd.DataFrame({"PostalCode": ["90001", "00000"], "LivingArea": [2000.0, np.nan]}))
    assert np.isclose(new[0, 0], np.log(500.0))
    assert np.isclose(new[1, 0], enc.prior_)          # unseen ZIP -> overall median
    # Square footage is clipped to the training range (all 1,000 sq ft here) before computing comp value.
    assert np.isclose(new[0, 1], np.log(500.0) + np.log(1000.0))


def test_week6_feature_flags_add_columns(data):
    from idx_avm.features import add_ratio_features, feature_columns

    df, _ = data
    df = add_ratio_features(df)
    df["SchoolDistrictK8"], df["SchoolDistrictHigh"] = "A Unified (X)", "A Unified (X)"
    base = feature_columns(df)
    full = feature_columns(df, ratios=True, school=True, comps=True)
    assert "BedBathRatio" in full["numeric"] and "BedBathRatio" not in base["numeric"]
    assert "SchoolDistrictK8" in full["high_card"] and "comp" in full and "comp" not in base


def test_school_district_join_prefers_unified_then_level():
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box
    from idx_avm.geo import assign_school_districts

    districts = gpd.GeoDataFrame(
        {"District": ["U (C)", "E (C)", "H (C)"], "DistrictType": ["Unified", "Elementary", "High"]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(1, 0, 2, 1)], crs="EPSG:4326",
    )
    homes = pd.DataFrame({"Latitude": [0.5, 0.5, None], "Longitude": [0.5, 1.5, None]})
    out = assign_school_districts(homes, districts)
    assert out["SchoolDistrictK8"].tolist()[:2] == ["U (C)", "E (C)"]
    assert out["SchoolDistrictHigh"].tolist()[:2] == ["U (C)", "H (C)"]
    assert out[["SchoolDistrictK8", "SchoolDistrictHigh"]].iloc[2].isna().all()


def test_nearby_comps_use_closest_sales_and_never_a_rows_own_price():
    import numpy as np
    import pandas as pd
    from idx_avm.preprocess import NearbyCompsEncoder

    # Two clusters ~50 km apart: a cheap one ($300/sq ft) and an expensive one ($1,200/sq ft).
    rng = np.random.default_rng(0)
    lat = np.r_[34.0 + rng.normal(0, 0.01, 60), 34.45 + rng.normal(0, 0.01, 60)]
    lon = np.r_[-118.0 + rng.normal(0, 0.01, 60), -118.0 + rng.normal(0, 0.01, 60)]
    X = pd.DataFrame({"Latitude": lat, "Longitude": lon, "LivingArea": 1500.0})
    y = np.log(np.r_[np.full(60, 300.0), np.full(60, 1200.0)] * 1500.0)
    enc = NearbyCompsEncoder(k=5)
    train_out = enc.fit_transform(X, y)
    assert np.allclose(train_out[:60, 0], np.log(300.0)) and np.allclose(train_out[60:, 0], np.log(1200.0))
    # A unique price can't come back through cross-fitting: its comps exclude it.
    y_unique = y.copy()
    y_unique[0] = np.log(99_999.0 * 1500.0)
    assert not np.isclose(NearbyCompsEncoder(k=1).fit_transform(X, y_unique)[0, 0], np.log(99_999.0))
    new = enc.transform(pd.DataFrame({"Latitude": [34.45, np.nan], "Longitude": [-118.0, np.nan], "LivingArea": [1500.0, 1500.0]}))
    assert np.isclose(new[0, 0], np.log(1200.0))
    assert np.isclose(new[1, 0], enc.prior_) and new[0, 2] < 5        # missing coords -> overall median; dist in km


def test_job_center_distance_is_zero_downtown_and_grows_inland():
    import pandas as pd
    from idx_avm.features import add_distance_features

    out = add_distance_features(pd.DataFrame({"Latitude": [34.0522, 36.7378], "Longitude": [-118.2437, -119.7871]}))
    assert out["DistToJobCenterKm"].iloc[0] < 0.01            # downtown LA
    assert 150 < out["DistToJobCenterKm"].iloc[1] < 300       # Fresno: far from every center


def test_select_model_prefers_simpler_config_within_noise():
    import pandas as pd
    from idx_avm.models import select_model

    results = pd.DataFrame({
        "feature_set": ["Base", "Base", "+ More", "+ More"],
        "model": ["Linear Regression", "Random Forest", "Linear Regression", "Random Forest"],
        "val_MdAPE": [11.0, 7.75, 10.0, 7.70],
    })
    assert select_model(results, ["Base", "+ More"], tol=0.10)["feature_set"] == "Base"      # 0.05 gain = noise
    assert select_model(results, ["Base", "+ More"], tol=0.01)["feature_set"] == "+ More"    # real gain wins


def test_gradient_boosting_fits_in_shared_pipeline_and_beats_naive(data):
    import numpy as np
    from idx_avm.models import MODEL_GRIDS, fit_and_score, permutation_mdape_importance

    df, _ = data
    months = complete_months(df)
    params = {**MODEL_GRIDS["Gradient Boosting"][1][0], "max_iter": 50}
    r = fit_and_score(df, months[-1], 3, "Gradient Boosting", params, flags={})
    test = r.test
    naive = np.median(np.abs(test["ClosePrice"].median() - test["ClosePrice"]) / test["ClosePrice"]) * 100
    assert r.metrics["MdAPE"] < naive
    imp = permutation_mdape_importance(r, n_rows=300, n_repeats=1)
    assert {"feature", "mdape_increase_pts"} <= set(imp.columns) and len(imp) > 5


def test_gradient_boosting_is_more_complex_than_random_forest_for_ties():
    import pandas as pd
    from idx_avm.models import select_model

    results = pd.DataFrame({"feature_set": ["Base", "Base"], "model": ["Random Forest", "Gradient Boosting"],
                            "val_MdAPE": [7.70, 7.65]})
    assert select_model(results, ["Base"], tol=0.10)["model"] == "Random Forest"        # within noise
    assert select_model(results, ["Base"], tol=0.01)["model"] == "Gradient Boosting"    # clear win


def test_params_round_trip_from_trial_labels():
    from idx_avm.models import MODEL_GRIDS, _fmt, params_from_label

    for name, (_, grid) in MODEL_GRIDS.items():
        for p in grid:
            assert params_from_label(name, _fmt(p)) == p


def test_paired_bootstrap_detects_a_better_model_and_not_a_tie():
    import numpy as np
    from idx_avm.evaluation import paired_bootstrap

    rng = np.random.default_rng(1)
    y = rng.uniform(3e5, 2e6, 4000)
    good = y * (1 + rng.normal(0, 0.05, y.size))
    bad = y * (1 + rng.normal(0, 0.10, y.size))
    r = paired_bootstrap(y, good, bad, n_boot=300)
    assert r["mdape_diff_pts"] < 0 and r["ci_high"] < 0
    tie = paired_bootstrap(y, good, good * (1 + rng.normal(0, 1e-4, y.size)), n_boot=300)
    assert tie["ci_low"] <= 0 <= tie["ci_high"]


def test_segment_breakdowns_use_shared_price_bands():
    import numpy as np
    import pandas as pd
    from idx_avm.evaluation import accuracy_bands, band_labels, price_band_edges, segment_breakdowns

    rng = np.random.default_rng(2)
    n = 1000
    y = rng.uniform(2e5, 3e6, n)
    test = pd.DataFrame({"ClosePrice": y, "Predicted": y * 1.05, "CountyOrParish": "Orange",
                         "LivingArea": rng.uniform(800, 5000, n), "PropertyAge": rng.uniform(0, 90, n)})
    edges = price_band_edges(y)
    seg = segment_breakdowns(test, edges, band_labels(y, edges))
    band = seg[seg["segment_type"] == "Price band (sale price)"]
    assert len(band) == 5 and band["n"].sum() == n
    assert np.allclose(band["median_signed_error_pct"], 5.0)
    assert accuracy_bands(y, y * 1.04)["within_5pct"] == 100.0
