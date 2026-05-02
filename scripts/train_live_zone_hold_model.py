from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ml_zone_hold_filter import FEATURE_COLUMNS, classifier_metrics, fit_sklearn_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the live zone-hold model from a frozen research dataset.")
    parser.add_argument("--dataset", type=Path, default=Path("scripts/zone_hold_dataset_majors20_1h_2022_2026.csv"))
    parser.add_argument("--model-out", type=Path, default=Path("scripts/zone_hold_model_majors20_1h_live.joblib"))
    parser.add_argument("--model", choices=["sklearn_rf", "sklearn_hgb"], default="sklearn_rf")
    parser.add_argument("--zone-tf", default="1h")
    parser.add_argument("--start")
    parser.add_argument("--end")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        raise SystemExit(f"Dataset not found: {args.dataset}")

    dataset = pd.read_csv(args.dataset)
    dataset["time"] = pd.to_datetime(dataset["time"], utc=True)
    dataset = dataset.sort_values(["time", "symbol"]).reset_index(drop=True)
    if args.start:
        dataset = dataset[dataset["time"] >= pd.Timestamp(args.start, tz="UTC")]
    if args.end:
        dataset = dataset[dataset["time"] < pd.Timestamp(args.end, tz="UTC")]
    dataset = dataset.dropna(subset=["hold_label"]).copy()

    missing = [column for column in FEATURE_COLUMNS if column not in dataset.columns]
    if missing:
        raise SystemExit(f"Dataset is missing feature columns: {missing}")
    if dataset["hold_label"].nunique() < 2:
        raise SystemExit("Training set only has one label class.")

    model = fit_sklearn_model(dataset, FEATURE_COLUMNS, args.model)
    dataset["hold_prob"] = model.predict_proba(dataset[FEATURE_COLUMNS].astype(float))[:, 1]

    payload = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "model_kind": args.model,
        "zone_tf": args.zone_tf,
        "source_dataset": str(args.dataset),
        "train_start": dataset["time"].min().isoformat(),
        "train_end": dataset["time"].max().isoformat(),
        "rows": int(len(dataset)),
        "symbols": sorted(str(symbol) for symbol in dataset["symbol"].dropna().unique()),
    }
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, args.model_out)

    print(f"Model saved to {args.model_out}")
    print(f"Rows: {len(dataset)}")
    print(f"Train range: {payload['train_start']} -> {payload['train_end']}")
    print(f"Symbols: {len(payload['symbols'])}")
    print(pd.DataFrame([classifier_metrics(dataset)]).to_string(index=False))


if __name__ == "__main__":
    main()
