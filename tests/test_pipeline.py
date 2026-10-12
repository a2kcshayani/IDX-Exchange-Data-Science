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
