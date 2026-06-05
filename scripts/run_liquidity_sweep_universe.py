from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_btc_liquidity_sweeps import (
    TF_SETTINGS,
    add_candidate_flags,
    add_ideal_excursions,
    detect_sweeps,
    enrich,
)
from scripts.backtest_btc_liquidity_sweep_preference import (
    build_backtest,
    one_trade_at_a_time,
    summarize_trades,
    yearly_summary,
)
from scripts.backtest_wolfe_wave import ensure_ohlcv_frame, load_ohlcv_csv, resample_ohlc


DEFAULT_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "BNBUSDT",
    "1000PEPEUSDT",
    "DOGEUSDT",
    "AVAXUSDT",
    "LINKUSDT",
]


def parse_symbols(raw: str) -> list[str]:
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


def input_path(data_dir: Path, symbol: str) -> Path:
    return data_dir / f"{symbol.lower()}_5m_bybit.csv"


def symbol_output_dir(output_dir: Path, symbol: str) -> Path:
    return output_dir / symbol.lower()


def build_events(base: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    all_events: list[pd.DataFrame] = []
    for timeframe in [item.strip() for item in args.timeframes.split(",") if item.strip()]:
        frame = enrich(base if timeframe == "5m" else resample_ohlc(base, timeframe))
        settings = TF_SETTINGS[timeframe]
        events = detect_sweeps(
            frame,
            timeframe=timeframe,
            pivot_window=int(settings["pivot_window"]),
            max_age=int(settings["max_age"]),
            horizon=int(settings["horizon"]),
            min_reclaim_pos=args.min_reclaim_pos,
            min_sweep_depth_atr=args.min_sweep_depth_atr,
            stop_buffer_atr=args.stop_buffer_atr,
            target_rr=args.target_rr,
            max_scan=args.max_scan,
        )
        all_events.append(events)
    events = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    events = add_ideal_excursions(events, base)
    events, thresholds = add_candidate_flags(events)
    events.attrs["candidate_thresholds"] = thresholds
    return events


def write_backtest_outputs(events: pd.DataFrame, base: pd.DataFrame, out_dir: Path, symbol: str) -> pd.DataFrame:
    trades = build_backtest(events, base)
    trades.insert(0, "symbol", symbol)
    trades.to_csv(out_dir / "preferred_backtest_trades.csv", index=False)

    no_overlap_parts = []
    for _, group in trades.groupby(["candidate_filter", "config", "stop_buffer_atr"], dropna=False):
        no_overlap_parts.append(one_trade_at_a_time(group))
    no_overlap = pd.concat(no_overlap_parts, ignore_index=True) if no_overlap_parts else pd.DataFrame()
    no_overlap.to_csv(out_dir / "preferred_backtest_trades_one_at_a_time.csv", index=False)

    summary = pd.concat(
        [
            summarize_trades(trades, "all_signals"),
            summarize_trades(no_overlap, "one_trade_at_a_time"),
        ],
        ignore_index=True,
    )
    if not summary.empty:
        summary.insert(0, "symbol", symbol)
    summary.to_csv(out_dir / "preferred_backtest_summary.csv", index=False)

    yearly = pd.concat(
        [
            yearly_summary(trades, "all_signals"),
            yearly_summary(no_overlap, "one_trade_at_a_time"),
        ],
        ignore_index=True,
    )
    if not yearly.empty:
        yearly.insert(0, "symbol", symbol)
    yearly.to_csv(out_dir / "preferred_backtest_by_year.csv", index=False)
    return summary


def run_symbol(symbol: str, args: argparse.Namespace) -> pd.DataFrame:
    source = input_path(args.data_dir, symbol)
    if not source.exists():
        raise FileNotFoundError(f"Missing candle file for {symbol}: {source}")
    out_dir = symbol_output_dir(args.output_dir, symbol)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = ensure_ohlcv_frame(load_ohlcv_csv(source))
    events = build_events(base, args)
    events.insert(0, "symbol", symbol)
    events.to_csv(out_dir / "liquidity_sweep_events.csv", index=False)
    events[events["candidate_4h_session_trend"]].to_csv(out_dir / "filtered_4h_session_trend_candidates.csv", index=False)
    events[events["candidate_4h_session_trend_bounded"]].to_csv(out_dir / "filtered_4h_session_trend_bounded_candidates.csv", index=False)
    pd.DataFrame([events.attrs.get("candidate_thresholds", {})]).to_csv(out_dir / "candidate_filter_thresholds.csv", index=False)

    summary = write_backtest_outputs(events, base, out_dir, symbol)
    print(f"{symbol}: events={len(events)} wrote {out_dir}", flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the preferred liquidity sweep research package across a symbol universe.")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--data-dir", type=Path, default=Path("scripts/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("scripts/liquidity_sweep_universe"))
    parser.add_argument("--timeframes", default="4h")
    parser.add_argument("--min-reclaim-pos", type=float, default=0.55)
    parser.add_argument("--min-sweep-depth-atr", type=float, default=0.02)
    parser.add_argument("--stop-buffer-atr", type=float, default=0.15)
    parser.add_argument("--target-rr", type=float, default=2.0)
    parser.add_argument("--max-scan", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [run_symbol(symbol, args) for symbol in parse_symbols(args.symbols)]
    combined = pd.concat([summary for summary in summaries if not summary.empty], ignore_index=True)
    combined.to_csv(args.output_dir / "preferred_backtest_summary_all_symbols.csv", index=False)
    print(f"Wrote {args.output_dir / 'preferred_backtest_summary_all_symbols.csv'}", flush=True)


if __name__ == "__main__":
    main()
