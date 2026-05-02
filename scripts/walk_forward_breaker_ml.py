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


def parse_folds(value: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    folds: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        start, end = [part.strip() for part in item.split(":", 1)]
        folds.append((pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")))
    return folds


def apply_gate(frame: pd.DataFrame, min_prob: float, direction: str, min_risk_pct: float) -> pd.DataFrame:
    out = frame[pd.to_numeric(frame["breaker_prob"], errors="coerce") >= min_prob].copy()
    if direction != "ALL":
        out = out[out["direction"].astype(str) == direction].copy()
    if min_risk_pct > 0:
        out = out[pd.to_numeric(out["risk_pct"], errors="coerce") >= min_risk_pct].copy()
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leak-clean walk-forward test for breaker ML gates.")
    parser.add_argument(
        "--trades",
        type=Path,
        default=Path("scripts/breaker_continuation_majors10_1h_fvg_print15_retest72_2022_2026.csv"),
    )
    parser.add_argument("--fee-bps-side", type=float, default=5.0)
    parser.add_argument("--model", choices=["rf", "hgb"], default="rf")
    parser.add_argument("--min-prob", type=float, default=0.50)
    parser.add_argument("--direction", choices=["ALL", "long", "short"], default="short")
    parser.add_argument("--min-risk-pct", type=float, default=2.0)
    parser.add_argument(
        "--folds",
        default="2023-04-20:2024-04-20,2024-04-20:2025-04-20,2025-04-20:2026-04-20",
        help="Comma-separated test windows, each as start:end. Training uses all rows before start.",
    )
    parser.add_argument("--out-prefix", type=Path, default=Path("scripts/wf_breaker_ml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folds = parse_folds(args.folds)
    dataset = enrich(pd.read_csv(args.trades), args.fee_bps_side).sort_values("entry_time").reset_index(drop=True)
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    rows: list[dict[str, Any]] = []
    selected_frames: list[pd.DataFrame] = []
    scored_frames: list[pd.DataFrame] = []

    for fold_index, (test_start, test_end) in enumerate(folds, start=1):
        train = dataset[dataset["entry_time"] < test_start].copy()
        test = dataset[(dataset["entry_time"] >= test_start) & (dataset["entry_time"] < test_end)].copy()
        if train.empty or test.empty:
            continue
        if train["label"].nunique() < 2:
            raise RuntimeError(f"Fold {fold_index} training labels contain only one class.")

        model = build_model(args.model)
        model.fit(train[features], train["label"].astype(int))
        test["breaker_prob"] = model.predict_proba(test[features])[:, 1]
        test["fold"] = fold_index
        test["fold_train_end"] = test_start
        test["fold_test_end"] = test_end
        scored_frames.append(test)

        selected = apply_gate(test, args.min_prob, args.direction, args.min_risk_pct)
        selected_frames.append(selected)

        row: dict[str, Any] = {
            "fold": fold_index,
            "train_end": test_start.isoformat(),
            "test_end": test_end.isoformat(),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "gate_min_prob": args.min_prob,
            "gate_direction": args.direction,
            "gate_min_risk_pct": args.min_risk_pct,
        }
        row.update({f"clf_{key}": value for key, value in classifier_metrics(test).items()})
        row.update({f"all_{key}": value for key, value in trade_metrics(test).items()})
        row.update({f"selected_{key}": value for key, value in trade_metrics(selected).items()})
        rows.append(row)

    summary = pd.DataFrame(rows)
    selected_all = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    scored_all = pd.concat(scored_frames, ignore_index=True) if scored_frames else pd.DataFrame()

    if not selected_all.empty:
        aggregate: dict[str, Any] = {
            "fold": "aggregate",
            "train_end": "",
            "test_end": "",
            "train_rows": int(sum(row["train_rows"] for row in rows)),
            "test_rows": int(len(scored_all)),
            "gate_min_prob": args.min_prob,
            "gate_direction": args.direction,
            "gate_min_risk_pct": args.min_risk_pct,
        }
        aggregate.update({f"clf_{key}": value for key, value in classifier_metrics(scored_all).items()})
        aggregate.update({f"all_{key}": value for key, value in trade_metrics(scored_all).items()})
        aggregate.update({f"selected_{key}": value for key, value in trade_metrics(selected_all).items()})
        summary = pd.concat([summary, pd.DataFrame([aggregate])], ignore_index=True)

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_prefix.with_name(args.out_prefix.name + "_summary.csv")
    selected_path = args.out_prefix.with_name(args.out_prefix.name + "_selected.csv")
    scored_path = args.out_prefix.with_name(args.out_prefix.name + "_scored.csv")
    summary.to_csv(summary_path, index=False)
    selected_all.to_csv(selected_path, index=False)
    scored_all.to_csv(scored_path, index=False)

    print(summary.to_string(index=False))
    print(f"\nSaved summary to {summary_path}")
    print(f"Saved selected trades to {selected_path}")
    print(f"Saved scored fold rows to {scored_path}")


if __name__ == "__main__":
    main()
