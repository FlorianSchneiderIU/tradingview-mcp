from __future__ import annotations

import argparse
import math
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backtest_turtle_soup import Config, normalize_timeframe, parse_utc_datetime, run_backtest
from scripts.crypto_symbol_sets import SYMBOL_SETS, expand_symbol_args
from scripts.sweep_turtle_soup_oos import ensure_cache


def trade_rows(symbol: str, df: pd.DataFrame, cfg: Config) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trade in run_backtest(df, cfg):
        row = asdict(trade)
        row["symbol"] = symbol
        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Turtle Soup backtest trades for research.")
    parser.add_argument("--symbols", nargs="+", default=[])
    parser.add_argument("--symbol-set", choices=["none", *SYMBOL_SETS.keys()], default="core3")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--start", default="2022-04-20")
    parser.add_argument("--end", default="2026-04-20")
    parser.add_argument("--warmup-start", default="2021-09-01")
    parser.add_argument("--cache-dir", type=Path, default=Path("scripts/.cache"))
    parser.add_argument("--output", type=Path, default=Path("scripts/turtle_soup_trades.csv"))
    parser.add_argument("--structure-tf", default="15m")
    parser.add_argument("--entry-mode", choices=["zone_retest", "retest_close", "limit_mid"], default="zone_retest")
    parser.add_argument("--tf1", default="1h")
    parser.add_argument("--tf2", default="1d")
    parser.add_argument("--use-tf2", action="store_true")
    parser.add_argument("--block-dead-zone", action="store_true")
    parser.add_argument("--max-structure-bars-to-choch", type=int, default=32)
    parser.add_argument("--htf-left", type=int, default=5)
    parser.add_argument("--htf-right", type=int, default=5)
    parser.add_argument("--htf-ob-search-bars", type=int, default=50)
    parser.add_argument("--max-zone-scan", type=int, default=250)
    parser.add_argument("--min-entry-risk-pct", type=float, default=0.0)
    parser.add_argument("--max-entry-risk-pct", type=float, default=math.inf)
    parser.add_argument("--zone-hold-model", type=Path)
    parser.add_argument("--zone-hold-min-prob", type=float, default=0.0)
    parser.add_argument("--zone-hold-filter-tf", default="1h")
    parser.add_argument("--reject-unscored-zone-hold", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols = expand_symbol_args(args.symbols, args.symbol_set)
    interval = normalize_timeframe(args.interval)
    warmup = parse_utc_datetime(args.warmup_start)
    start = parse_utc_datetime(args.start)
    end = parse_utc_datetime(args.end)
    if warmup > start:
        raise SystemExit("--warmup-start must be at or before --start.")

    cfg = Config(
        exec_tf=interval,
        structure_tf=normalize_timeframe(args.structure_tf),
        entry_mode=args.entry_mode,
        tf1=normalize_timeframe(args.tf1),
        tf2=normalize_timeframe(args.tf2),
        use_tf1=True,
        use_tf2=args.use_tf2,
        block_dead_zone=args.block_dead_zone,
        max_structure_bars_to_choch=args.max_structure_bars_to_choch,
        htf_left=args.htf_left,
        htf_right=args.htf_right,
        htf_ob_search_bars=args.htf_ob_search_bars,
        max_zone_scan=args.max_zone_scan,
        min_entry_risk_pct=args.min_entry_risk_pct,
        max_entry_risk_pct=args.max_entry_risk_pct,
        zone_hold_model_path=str(args.zone_hold_model) if args.zone_hold_model else None,
        zone_hold_min_prob=args.zone_hold_min_prob,
        zone_hold_filter_tf=normalize_timeframe(args.zone_hold_filter_tf),
        zone_hold_reject_unscored=args.reject_unscored_zone_hold,
    )

    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        cache = ensure_cache(symbol, interval, warmup, end, args.cache_dir)
        df = pd.read_pickle(cache)
        df = df[(df["open_time"] >= pd.Timestamp(warmup)) & (df["open_time"] < pd.Timestamp(end))].copy()
        rows = trade_rows(symbol.split(":")[-1].upper(), df, cfg)
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True)
            frame = frame[(frame["entry_time"] >= pd.Timestamp(start)) & (frame["entry_time"] < pd.Timestamp(end))]
        frames.append(frame)
        print(f"{symbol}: exported {len(frame)} trades")

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result)} trades to {args.output}")


if __name__ == "__main__":
    main()
