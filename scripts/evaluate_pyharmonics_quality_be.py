from __future__ import annotations

import argparse
import json
import math
import pickle
import shlex
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest_pyharmonics_strategy import (  # noqa: E402
    HarmonicConfig,
    add_lowpass_scores,
    candle_pattern_summary,
    config_metrics,
    event_filter_matches,
    parse_csv_values,
    prepare_harmonic_frame,
    run_backtest,
)
from scripts.backtest_wolfe_wave import ensure_ohlcv_frame, normalize_timeframe  # noqa: E402
from scripts.run_pyharmonics_top100_lowpass import family_filter, rank_score  # noqa: E402
from scripts.tune_wolfe_wave_universe import split_bounds  # noqa: E402


DEFAULT_RUN_DIR = Path("scripts/pyharmonics_focus_filters_v2_inj_link_ltc_15m_abcd_20260603_141333")


def parse_command_args(run_dir: Path) -> dict[str, str]:
    command_path = run_dir / "command.txt"
    if not command_path.exists():
        return {}
    tokens = shlex.split(command_path.read_text(encoding="utf-8"), posix=False)
    out: dict[str, str] = {}
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token.startswith("--"):
            key = token[2:]
            value = "true"
            if idx + 1 < len(tokens) and not tokens[idx + 1].startswith("--"):
                value = tokens[idx + 1]
                idx += 1
            out[key] = value
        idx += 1
    return out


def load_symbol_frame(cache_dir: Path, symbol: str, timeframe: str) -> pd.DataFrame:
    path = cache_dir / f"{symbol.lower()}_{timeframe}_bybit.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing cached OHLCV for {symbol}: {path}")
    return ensure_ohlcv_frame(pd.read_csv(path))


def load_events(run_dir: Path, symbol: str) -> list[Any]:
    cache_dir = run_dir / "per_symbol" / "event_cache"
    events: list[Any] = []
    for path in sorted(cache_dir.glob(f"{symbol.lower()}_*.pkl")):
        events.extend(pickle.loads(path.read_bytes()))
    if not events:
        raise FileNotFoundError(f"Missing event cache for {symbol} in {cache_dir}")
    return events


def selected_configs(run_dir: Path, symbols: set[str] | None = None) -> list[tuple[str, HarmonicConfig]]:
    path = run_dir / "candidate_retest.csv"
    table = pd.read_csv(path)
    out: list[tuple[str, HarmonicConfig]] = []
    for _, row in table.iterrows():
        symbol = str(row["symbol"]).upper()
        if symbols and symbol not in symbols:
            continue
        payload = json.loads(str(row["selected_config_json"]))
        out.append((symbol, HarmonicConfig(**payload)))
    return out


def exit_counts(trades: pd.DataFrame) -> dict[str, int]:
    if trades.empty or "exit_reason" not in trades.columns:
        return {"targets": 0, "stops": 0, "breakevens": 0, "timeouts": 0}
    reasons = trades["exit_reason"].astype(str)
    return {
        "targets": int(reasons.str.contains("target", case=False, na=False).sum()),
        "stops": int(reasons.str.contains("stop", case=False, na=False).sum()),
        "breakevens": int(reasons.str.contains("breakeven", case=False, na=False).sum()),
        "timeouts": int(reasons.eq("timeout").sum()),
    }


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Path]]:
    command_args = parse_command_args(args.run_dir)
    cache_dir = args.cache_dir or Path(command_args.get("cache-dir", ""))
    if not str(cache_dir):
        raise SystemExit("Pass --cache-dir or use a run directory with command.txt containing --cache-dir.")
    exec_tf = normalize_timeframe(args.exec_tf or command_args.get("exec-tf", "15m"))
    validation_days = int(args.validation_days or command_args.get("validation-days", 365))
    oos_days = int(args.oos_days or command_args.get("oos-days", 365))
    qualities = parse_csv_values(args.quality_scores, float)
    breakevens = parse_csv_values(args.breakeven_triggers, float)
    entry_modes = parse_csv_values(args.entry_modes, str)
    time_filters = parse_csv_values(args.time_filters, str)
    htf_filters = parse_csv_values(args.htf_filters, str)
    htf_stretch_atrs = parse_csv_values(args.htf_stretch_atrs, float)
    htf_rsi_extremes = parse_csv_values(args.htf_rsi_extremes, float)
    symbols = {item.strip().upper() for item in parse_csv_values(args.symbols, str)} if args.symbols else None
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    best_trade_paths: dict[str, Path] = {}
    for symbol, base_cfg in selected_configs(args.run_dir, symbols):
        frame = load_symbol_frame(cache_dir, symbol, exec_tf)
        frame = prepare_harmonic_frame(frame, base_cfg)
        train_end, validation_end = split_bounds(frame, validation_days=validation_days, oos_days=oos_days)
        families = family_filter(base_cfg.family)
        base_events = [
            event
            for event in load_events(args.run_dir, symbol)
            if (not families or event.family in families) and event_filter_matches(event, base_cfg)
        ]
        symbol_rows: list[dict[str, Any]] = []
        trades_by_key: dict[tuple[float, float], pd.DataFrame] = {}

        for quality in qualities:
            for breakeven in breakevens:
                for entry_mode in entry_modes:
                    for time_filter in time_filters:
                        for htf_filter in htf_filters:
                            for htf_stretch_atr in htf_stretch_atrs:
                                for htf_rsi_extreme in htf_rsi_extremes:
                                    cfg = replace(
                                        base_cfg,
                                        min_harmonic_quality_score=float(quality),
                                        breakeven_trigger_r=float(breakeven),
                                        entry_mode=str(entry_mode).strip().lower(),
                                        time_filter=str(time_filter).strip().lower(),
                                        htf_filter=str(htf_filter).strip().lower(),
                                        htf_stretch_atr=float(htf_stretch_atr),
                                        htf_rsi_extreme=float(htf_rsi_extreme),
                                    )
                                    trades = run_backtest(frame, cfg, symbol=symbol, precomputed_events=base_events)
                                    metrics = config_metrics(trades, train_end=train_end, validation_end=validation_end)
                                    counts = exit_counts(trades)
                                    row = {
                                        "symbol": symbol,
                                        **asdict(cfg),
                                        "pattern_events": len(base_events),
                                        **metrics,
                                        **counts,
                                        "avg_harmonic_quality_score": finite(trades["harmonic_quality_score"].mean()) if "harmonic_quality_score" in trades else 0.0,
                                        "median_harmonic_quality_score": finite(trades["harmonic_quality_score"].median()) if "harmonic_quality_score" in trades else 0.0,
                                    }
                                    symbol_rows.append(row)
                                    trades_by_key[
                                        (
                                            float(quality),
                                            float(breakeven),
                                            str(entry_mode).strip().lower(),
                                            str(time_filter).strip().lower(),
                                            str(htf_filter).strip().lower(),
                                            float(htf_stretch_atr),
                                            float(htf_rsi_extreme),
                                        )
                                    ] = trades.copy()

        symbol_table = add_lowpass_scores(pd.DataFrame(symbol_rows), radius=float(args.lowpass_radius), min_neighbors=int(args.lowpass_min_neighbors))
        symbol_table["rank_score"] = symbol_table.apply(rank_score, axis=1)
        symbol_table = symbol_table.sort_values(
            ["lowpass_robust_score", "rank_score", "oos_net_r", "validation_net_r"],
            ascending=[False, False, False, False],
            na_position="last",
        ).reset_index(drop=True)
        rows.extend(symbol_table.to_dict("records"))
        symbol_table.to_csv(args.output_dir / f"{symbol.lower()}_quality_be_tuning.csv", index=False)
        if not symbol_table.empty:
            top = symbol_table.iloc[0]
            best_key = (
                float(top["min_harmonic_quality_score"]),
                float(top["breakeven_trigger_r"]),
                str(top["entry_mode"]).strip().lower(),
                str(top["time_filter"]).strip().lower(),
                str(top["htf_filter"]).strip().lower(),
                float(top["htf_stretch_atr"]),
                float(top["htf_rsi_extreme"]),
            )
            best_trades = trades_by_key.get(best_key, pd.DataFrame())
            best_path = args.output_dir / f"{symbol.lower()}_quality_be_best_trades.csv"
            best_trades.to_csv(best_path, index=False)
            candle_pattern_summary(best_trades).to_csv(args.output_dir / f"{symbol.lower()}_quality_be_candle_patterns.csv", index=False)
            best_trade_paths[symbol] = best_path

    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "quality_be_summary.csv", index=False)
    return table, best_trade_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Targeted pyharmonics harmonic-quality and breakeven experiment.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("scripts/pyharmonics_quality_be_targeted"))
    parser.add_argument("--symbols", default="")
    parser.add_argument("--exec-tf", default="")
    parser.add_argument("--validation-days", type=int, default=0)
    parser.add_argument("--oos-days", type=int, default=0)
    parser.add_argument("--quality-scores", default="0,45,55,65,75")
    parser.add_argument("--breakeven-triggers", default="0,0.75,1.0,1.25")
    parser.add_argument("--entry-modes", default="next_open")
    parser.add_argument("--time-filters", default="all")
    parser.add_argument("--htf-filters", default="none")
    parser.add_argument("--htf-stretch-atrs", default="0.75")
    parser.add_argument("--htf-rsi-extremes", default="55")
    parser.add_argument("--lowpass-radius", type=float, default=1.60)
    parser.add_argument("--lowpass-min-neighbors", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table, best_trade_paths = run(args)
    if table.empty:
        print("No rows produced.")
        return
    cols = [
        "symbol",
        "min_harmonic_quality_score",
        "breakeven_trigger_r",
        "entry_mode",
        "time_filter",
        "htf_filter",
        "htf_stretch_atr",
        "htf_rsi_extreme",
        "all_trades",
        "validation_trades",
        "oos_trades",
        "all_net_r",
        "validation_net_r",
        "oos_net_r",
        "all_avg_r",
        "all_max_dd_r",
        "targets",
        "stops",
        "breakevens",
        "timeouts",
        "avg_harmonic_quality_score",
        "lowpass_robust_score",
        "rank_score",
    ]
    print(table[[col for col in cols if col in table.columns]].head(20).to_string(index=False))
    print(f"Saved summary: {args.output_dir / 'quality_be_summary.csv'}")
    for symbol, path in best_trade_paths.items():
        print(f"Saved best trades {symbol}: {path}")


if __name__ == "__main__":
    main()
