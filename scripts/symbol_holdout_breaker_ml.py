from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ml_breaker_continuation_filter import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_model,
    classifier_metrics,
    enrich,
    trade_metrics,
)
from scripts.walk_forward_breaker_ml import parse_folds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leave-one-symbol-out walk-forward test for breaker ML.")
    parser.add_argument(
        "--trades",
        type=Path,
        default=Path("scripts/breaker_continuation_majors10_1h_fvg_print15_retest72_2022_2026.csv"),
    )
    parser.add_argument("--fee-bps-side", type=float, default=5.0)
    parser.add_argument("--model", choices=["rf", "hgb"], default="rf")
    parser.add_argument(
        "--folds",
        default="2023-04-20:2024-04-20,2024-04-20:2025-04-20,2025-04-20:2026-04-20",
    )
    parser.add_argument("--out-prefix", type=Path, default=Path("scripts/symbol_holdout_breaker_ml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = enrich(pd.read_csv(args.trades), args.fee_bps_side).sort_values("entry_time").reset_index(drop=True)
    symbols = sorted(dataset["symbol"].astype(str).unique())
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    folds = parse_folds(args.folds)

    scored_frames: list[pd.DataFrame] = []
    rows: list[dict[str, Any]] = []

    for fold_index, (test_start, test_end) in enumerate(folds, start=1):
        for symbol in symbols:
            train = dataset[(dataset["entry_time"] < test_start) & (dataset["symbol"].astype(str) != symbol)].copy()
            test = dataset[
                (dataset["entry_time"] >= test_start)
                & (dataset["entry_time"] < test_end)
                & (dataset["symbol"].astype(str) == symbol)
            ].copy()
            if train.empty or test.empty:
                continue
            if train["label"].nunique() < 2:
                continue

            model = build_model(args.model)
            model.fit(train[features], train["label"].astype(int))
            test["breaker_prob"] = model.predict_proba(test[features])[:, 1]
            test["fold"] = fold_index
            test["heldout_symbol"] = symbol
            test["fold_train_end"] = test_start
            test["fold_test_end"] = test_end
            scored_frames.append(test)

            row: dict[str, Any] = {
                "fold": fold_index,
                "heldout_symbol": symbol,
                "train_end": test_start.isoformat(),
                "test_end": test_end.isoformat(),
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
            }
            row.update({f"clf_{key}": value for key, value in classifier_metrics(test).items()})
            row.update({f"all_{key}": value for key, value in trade_metrics(test).items()})
            rows.append(row)

    summary = pd.DataFrame(rows)
    scored = pd.concat(scored_frames, ignore_index=True) if scored_frames else pd.DataFrame()
    if not scored.empty:
        aggregate: dict[str, Any] = {
            "fold": "aggregate",
            "heldout_symbol": "ALL",
            "train_end": "",
            "test_end": "",
            "train_rows": int(summary["train_rows"].sum()) if not summary.empty else 0,
            "test_rows": int(len(scored)),
        }
        aggregate.update({f"clf_{key}": value for key, value in classifier_metrics(scored).items()})
        aggregate.update({f"all_{key}": value for key, value in trade_metrics(scored).items()})
        summary = pd.concat([summary, pd.DataFrame([aggregate])], ignore_index=True)

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_prefix.with_name(args.out_prefix.name + "_summary.csv")
    scored_path = args.out_prefix.with_name(args.out_prefix.name + "_scored.csv")
    summary.to_csv(summary_path, index=False)
    scored.to_csv(scored_path, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved summary to {summary_path}")
    print(f"Saved scored rows to {scored_path}")


if __name__ == "__main__":
    main()
