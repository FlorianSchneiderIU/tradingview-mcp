from __future__ import annotations

import argparse
import math
import sys
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ml_breaker_continuation_filter import trade_metrics


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def apply_gate(frame: pd.DataFrame, min_prob: float, direction: str, min_risk_pct: float, symbol: str) -> pd.DataFrame:
    out = frame[pd.to_numeric(frame["breaker_prob"], errors="coerce") >= min_prob].copy()
    if direction != "ALL":
        out = out[out["direction"].astype(str) == direction].copy()
    if symbol != "ALL":
        out = out[out["symbol"].astype(str) == symbol].copy()
    if min_risk_pct > 0:
        out = out[pd.to_numeric(out["risk_pct"], errors="coerce") >= min_risk_pct].copy()
    return out


def finite_pf(value: float) -> float:
    if math.isfinite(float(value)):
        return float(value)
    return 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid-test gates on walk-forward scored breaker rows.")
    parser.add_argument("scored", type=Path)
    parser.add_argument("--prob-values", default="0.45,0.50,0.525,0.55,0.575,0.60,0.625,0.65,0.70")
    parser.add_argument("--min-risk-pct-values", default="0,0.5,1.0,1.5,2.0,2.5,3.0")
    parser.add_argument("--directions", default="ALL,long,short")
    parser.add_argument("--symbols", default="AUTO", help="AUTO tests ALL plus each symbol in the scored file.")
    parser.add_argument("--min-fold-trades", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("scripts/wf_gate_grid.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.scored)
    for column in ["entry_time", "exit_time"]:
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")

    prob_values = parse_float_list(args.prob_values)
    risk_values = parse_float_list(args.min_risk_pct_values)
    directions = [item.strip() for item in args.directions.split(",") if item.strip()]
    if args.symbols == "AUTO":
        symbols = ["ALL", *sorted(frame["symbol"].astype(str).unique())]
    else:
        symbols = [item.strip() for item in args.symbols.split(",") if item.strip()]

    rows: list[dict[str, Any]] = []
    folds = sorted(frame["fold"].dropna().unique())
    for min_prob, min_risk, direction, symbol in product(prob_values, risk_values, directions, symbols):
        selected = apply_gate(frame, min_prob, direction, min_risk, symbol)
        if selected.empty:
            continue

        row: dict[str, Any] = {
            "min_prob": min_prob,
            "min_risk_pct": min_risk,
            "direction": direction,
            "symbol": symbol,
        }
        fold_net: list[float] = []
        fold_pf: list[float] = []
        fold_counts: list[int] = []
        skip = False
        for fold in folds:
            fold_frame = selected[selected["fold"] == fold]
            stats = trade_metrics(fold_frame)
            if stats["trades"] < args.min_fold_trades:
                skip = True
                break
            prefix = f"fold{int(fold)}"
            row.update({f"{prefix}_{key}": value for key, value in stats.items()})
            fold_net.append(float(stats["net_r"]))
            fold_pf.append(finite_pf(float(stats["profit_factor"])))
            fold_counts.append(int(stats["trades"]))
        if skip:
            continue

        aggregate = trade_metrics(selected)
        row.update({f"aggregate_{key}": value for key, value in aggregate.items()})
        row["folds_positive"] = all(value > 0 for value in fold_net)
        row["worst_fold_net_r"] = round(min(fold_net), 3)
        row["worst_fold_pf"] = round(min(fold_pf), 3)
        row["min_fold_trades"] = min(fold_counts)
        row["score"] = round(
            float(aggregate["net_r"]) * 0.35
            + min(fold_net) * 0.45
            + sum(fold_net) / len(fold_net) * 0.20
            + (sum(fold_pf) / len(fold_pf) - 1.0) * 8.0,
            3,
        )
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["folds_positive", "score", "aggregate_net_r"], ascending=[False, False, False])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.head(40).to_string(index=False))
    print(f"\nSaved {len(out)} rows to {args.out}")


if __name__ == "__main__":
    main()
