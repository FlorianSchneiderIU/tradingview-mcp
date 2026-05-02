from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


def profit_factor(rs: pd.Series) -> float:
    wins = rs[rs > 0]
    losses = rs[rs <= 0]
    gross_loss = abs(float(losses.sum()))
    if gross_loss == 0:
        return float("inf") if len(wins) else 0.0
    return float(wins.sum()) / gross_loss


def max_drawdown(values: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for value in values:
        equity += float(value)
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)
    return max_dd


def metrics(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    if frame.empty:
        return {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "net_r": 0.0, "avg_r": 0.0, "max_dd_r": 0.0}
    rs = frame.sort_values("exit_time")[column].astype(float)
    return {
        "trades": int(len(rs)),
        "win_rate": round(100.0 * float((rs > 0).mean()), 2),
        "profit_factor": round(profit_factor(rs), 3),
        "net_r": round(float(rs.sum()), 3),
        "avg_r": round(float(rs.mean()), 4),
        "max_dd_r": round(max_drawdown(rs.to_list()), 3),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cost-adjust breaker continuation trade CSVs.")
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--split", default="2025-04-20")
    parser.add_argument("--end", default="2026-04-20")
    parser.add_argument("--fee-bps-side", type=float, default=5.0)
    parser.add_argument("--out", type=Path, default=Path("scripts/breaker_continuation_cost_summary.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = pd.Timestamp(args.split, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    rows: list[dict[str, Any]] = []

    for path in args.files:
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True)
        frame["exit_time"] = pd.to_datetime(frame["exit_time"], utc=True)
        risk = (frame["entry_price"].astype(float) - frame["stop_price"].astype(float)).abs()
        cost_r = (frame["entry_price"].abs() + frame["exit_price"].abs()) * args.fee_bps_side / 10000.0 / risk
        frame["r_net_cost"] = frame["r_multiple"].astype(float) - cost_r
        oos = frame[(frame["entry_time"] >= split) & (frame["entry_time"] < end)].copy()

        label = path.stem.replace("breaker_continuation_core3_", "")
        rows.append({"variant": label, "basis": "raw", **metrics(oos, "r_multiple")})
        rows.append({ "variant": label, "basis": f"net_{args.fee_bps_side:g}bps_side", **metrics(oos, "r_net_cost")})

    summary = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved summary to {args.out}")


if __name__ == "__main__":
    main()
