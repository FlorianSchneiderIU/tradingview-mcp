from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.deep_trade_edge_study import enrich, metrics


def parse_float_list(value: str) -> list[float]:
    out: list[float] = []
    for item in value.split(","):
        item = item.strip()
        if item:
            out.append(float(item))
    return out


def split_metrics(frame: pd.DataFrame, train_end: pd.Timestamp, validation_end: pd.Timestamp, test_end: pd.Timestamp) -> dict[str, Any]:
    windows = {
        "train": frame[frame["entry_time"] < train_end],
        "validation": frame[(frame["entry_time"] >= train_end) & (frame["entry_time"] < validation_end)],
        "oos": frame[(frame["entry_time"] >= validation_end) & (frame["entry_time"] < test_end)],
    }
    row: dict[str, Any] = {}
    for name, window in windows.items():
        row.update({f"{name}_{key}": value for key, value in metrics(window, "r_net_cost").items()})
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid-test simple trade gates across train/validation/OOS.")
    parser.add_argument("file", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--fee-bps-side", type=float, default=5.0)
    parser.add_argument("--train-end", default="2024-04-20")
    parser.add_argument("--validation-end", default="2025-04-20")
    parser.add_argument("--test-end", default="2026-04-20")
    parser.add_argument("--prob-column")
    parser.add_argument("--prob-values", default="0")
    parser.add_argument("--min-risk-pct-values", default="0,0.5,0.75,1.0,1.25,1.5")
    parser.add_argument("--symbols", default="ALL,BTCUSDT,ETHUSDT,SOLUSDT")
    parser.add_argument("--directions", default="ALL,long,short")
    parser.add_argument("--min-train", type=int, default=20)
    parser.add_argument("--min-validation", type=int, default=10)
    parser.add_argument("--min-oos", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("scripts/trade_gate_grid.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_end = pd.Timestamp(args.train_end, tz="UTC")
    validation_end = pd.Timestamp(args.validation_end, tz="UTC")
    test_end = pd.Timestamp(args.test_end, tz="UTC")
    frame = enrich(pd.read_csv(args.file), args.fee_bps_side)
    prob_values = parse_float_list(args.prob_values)
    min_risk_values = parse_float_list(args.min_risk_pct_values)
    symbols = [item.strip() for item in args.symbols.split(",") if item.strip()]
    directions = [item.strip() for item in args.directions.split(",") if item.strip()]

    rows: list[dict[str, Any]] = []
    for prob, min_risk, symbol, direction in product(prob_values, min_risk_values, symbols, directions):
        selected = frame.copy()
        if args.prob_column and args.prob_column in selected.columns:
            selected = selected[pd.to_numeric(selected[args.prob_column], errors="coerce") >= prob]
        if min_risk > 0:
            selected = selected[selected["risk_pct"] >= min_risk]
        if symbol != "ALL":
            selected = selected[selected["symbol"].astype(str) == symbol]
        if direction != "ALL":
            selected = selected[selected["direction"].astype(str) == direction]
        row = {
            "label": args.label,
            "prob_column": args.prob_column or "",
            "min_prob": prob if args.prob_column else "",
            "min_risk_pct": min_risk,
            "symbol": symbol,
            "direction": direction,
            **split_metrics(selected, train_end, validation_end, test_end),
        }
        if row["train_trades"] < args.min_train or row["validation_trades"] < args.min_validation or row["oos_trades"] < args.min_oos:
            continue
        row["stable_positive"] = row["train_net_r"] > 0 and row["validation_net_r"] > 0 and row["oos_net_r"] > 0
        row["score"] = (
            float(row["train_net_r"]) * 0.25
            + float(row["validation_net_r"]) * 0.35
            + float(row["oos_net_r"]) * 0.40
            + (float(row["train_profit_factor"]) + float(row["validation_profit_factor"]) + float(row["oos_profit_factor"]) - 3.0) * 3.0
        )
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["stable_positive", "score", "oos_net_r"], ascending=[False, False, False])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.head(30).to_string(index=False))
    print(f"\nSaved {len(out)} grid rows to {args.out}")


if __name__ == "__main__":
    main()
