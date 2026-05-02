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


def parse_symbol_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def finite_pf(value: float) -> float:
    if math.isfinite(float(value)):
        return float(value)
    return 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid-test explicit breaker context gates.")
    parser.add_argument("scored_file", type=Path)
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT")
    parser.add_argument("--direction", default="long")
    parser.add_argument("--min-risk-pct", type=float, default=1.0)
    parser.add_argument("--min-prob-values", default="0,0.5,0.55")
    parser.add_argument("--min-reject-values", default="0,0.75,0.8,0.85")
    parser.add_argument("--max-confirm-delay-values", default="999,1.0,0.75,0.5")
    parser.add_argument("--max-confirm-close-pos-values", default="999,0.9,0.85,0.8")
    parser.add_argument("--max-entry-extension-values", default="999,1.0,0.75,0.5")
    parser.add_argument("--min-fold-trades", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("scripts/breaker_context_gate_grid.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.scored_file)
    symbols = parse_symbol_list(args.symbols)

    base = frame[frame["symbol"].astype(str).isin(symbols)].copy()
    if args.direction != "ALL":
        base = base[base["direction"].astype(str) == args.direction].copy()
    if args.min_risk_pct > 0:
        base = base[pd.to_numeric(base["risk_pct"], errors="coerce") >= args.min_risk_pct].copy()

    rows: list[dict[str, Any]] = []
    min_probs = parse_float_list(args.min_prob_values)
    min_rejects = parse_float_list(args.min_reject_values)
    max_delays = parse_float_list(args.max_confirm_delay_values)
    max_close_pos = parse_float_list(args.max_confirm_close_pos_values)
    max_extensions = parse_float_list(args.max_entry_extension_values)

    for min_prob, min_reject, max_delay, max_close, max_ext in product(
        min_probs, min_rejects, max_delays, max_close_pos, max_extensions
    ):
        selected = base.copy()
        if min_prob > 0:
            selected = selected[pd.to_numeric(selected["breaker_prob"], errors="coerce") >= min_prob]
        if min_reject > 0:
            selected = selected[pd.to_numeric(selected["retest_reject_pos"], errors="coerce") >= min_reject]
        if max_delay < 999:
            selected = selected[pd.to_numeric(selected["confirm_delay_hours"], errors="coerce") <= max_delay]
        if max_close < 999:
            selected = selected[pd.to_numeric(selected["confirm_close_pos_dir"], errors="coerce") <= max_close]
        if max_ext < 999:
            selected = selected[pd.to_numeric(selected["entry_extension_r"], errors="coerce") <= max_ext]
        if selected.empty:
            continue

        row: dict[str, Any] = {
            "symbols": " ".join(symbols),
            "min_prob": min_prob,
            "min_reject_pos": min_reject,
            "max_confirm_delay_h": max_delay,
            "max_confirm_close_pos_dir": max_close,
            "max_entry_extension_r": max_ext,
        }
        fold_net: list[float] = []
        fold_pf: list[float] = []
        fold_trades: list[int] = []
        stable = True
        for fold in [1, 2, 3]:
            fold_frame = selected[selected["fold"] == fold].copy()
            stats = trade_metrics(fold_frame)
            row.update({f"fold{fold}_{key}": value for key, value in stats.items()})
            fold_net.append(float(stats["net_r"]))
            fold_pf.append(finite_pf(float(stats["profit_factor"])))
            fold_trades.append(int(stats["trades"]))
            if int(stats["trades"]) < args.min_fold_trades or float(stats["net_r"]) <= 0 or float(stats["profit_factor"]) <= 1.0:
                stable = False

        aggregate = trade_metrics(selected)
        row.update({f"aggregate_{key}": value for key, value in aggregate.items()})
        row["stable_positive"] = stable
        row["min_fold_trades"] = min(fold_trades)
        row["worst_fold_net_r"] = round(min(fold_net), 3)
        row["worst_fold_pf"] = round(min(fold_pf), 3)
        row["score"] = round(
            float(aggregate["net_r"]) * 0.35
            + min(fold_net) * 0.45
            + (sum(fold_pf) / len(fold_pf) - 1.0) * 8.0
            + (sum(fold_net) / len(fold_net)) * 0.20,
            3,
        )
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["stable_positive", "score", "aggregate_net_r"], ascending=[False, False, False])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.head(40).to_string(index=False))
    print(f"\nSaved {len(out)} rows to {args.out}")


if __name__ == "__main__":
    main()
