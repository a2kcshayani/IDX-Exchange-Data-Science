"""Week 3 end-to-end preprocessing: raw CSVs -> cleaned, split CSVs + logs.

Usage:
    python scripts/run_preprocessing.py [--raw-dir data/raw] [--candidates 1 2 3 4 5 6 9 12]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idx_avm import config  # noqa: E402
from idx_avm.clean import clean  # noqa: E402
from idx_avm.features import add_base_features, feature_columns, leakage_report  # noqa: E402
from idx_avm.load import filter_single_family, load_raw  # noqa: E402
from idx_avm.split import add_close_month, apply_outlier_filter, complete_months, window_split  # noqa: E402
from idx_avm.window import tune_training_window  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=config.PROCESSED_DIR)
    parser.add_argument("--candidates", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 9, 12])
    parser.add_argument("--origins", type=int, default=3)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw(args.raw_dir)
    files = sorted(raw["SourceFile"].unique())
    print(f"Loaded {len(raw):,} rows from {len(files)} files")
    sfr = filter_single_family(raw)
    print(f"Single-family residential: {len(sfr):,} rows")

    leakage_report(sfr).to_csv(args.out_dir / "leakage_audit.csv", index=False)
    df, log = clean(sfr)
    log.to_frame().to_csv(args.out_dir / "cleaning_log.csv", index=False)
    print(log.to_frame()[["step", "removed", "pct_removed"]].to_string(index=False))

    df = add_close_month(add_base_features(df))
    months = complete_months(df)
    df = df[df["CloseMonth"].isin(months)]
    print(f"Complete months: {months[0]} .. {months[-1]} ({len(months)})")

    results, summary = tune_training_window(df, months, args.candidates, n_origins=args.origins)
    results.to_csv(args.out_dir / "window_tuning_by_origin.csv", index=False)
    summary.to_csv(args.out_dir / "window_tuning_summary.csv")
    best_x = int(summary.index[0])
    print(f"\nTraining-window tuning (validation months only, test untouched):\n{summary.round(3)}")
    print(f"Chosen X = {best_x} months")

    split, thresholds, stats = apply_outlier_filter(window_split(df, months[-1], best_x))
    keep = ["ListingKey"] if "ListingKey" in df.columns else []
    groups = feature_columns(split.train)
    cols = keep + [c for g in groups.values() for c in g] + [config.DATE_COL, "CloseMonth", config.TARGET]
    split.train[cols].to_csv(args.out_dir / "train_clean.csv", index=False)
    split.test[cols].to_csv(args.out_dir / "test_clean.csv", index=False)

    meta = {
        "source_files": files,
        "train_months": [str(m) for m in split.train_months],
        "test_month": str(split.test_month),
        "chosen_window_months": best_x,
        "outlier_thresholds": stats,
        "n_train": len(split.train),
        "n_test": len(split.test),
        "feature_groups": groups,
    }
    (args.out_dir / "split_metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"\nWrote cleaned train/test CSVs and logs to {args.out_dir}")
    print(json.dumps({k: meta[k] for k in ("train_months", "test_month", "n_train", "n_test")}, indent=2))


if __name__ == "__main__":
    main()
