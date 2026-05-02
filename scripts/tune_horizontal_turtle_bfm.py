from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backtest_turtle_soup import build_confirmed_pivots, normalize_binance_spot_symbol
from scripts.channel_state_research.backtest import strategy_metrics
from scripts.channel_state_research.data import load_base_candles, prepare_timeframe_bars
from scripts.channel_state_research.production import load_production_config
from scripts.plot_zone_channel_history import build_bfm_magic_lines, parse_timeframes
from scripts.tune_bfm_support_resistance import LineBundle, Projection, project_lines_to_execution_frame
from scripts.tune_bfm_turtle_soup import (
    OPTIMIZED_BFM_TF_SETS,
    format_sets,
    label_and_schedule,
    parse_float_list,
    parse_int_list,
    parse_str_list,
    parse_tf_sets,
    score_metrics,
    tf_name,
    unique_preserve_order,
)


@dataclass(frozen=True)
class HorizontalLevel:
    level_id: str
    side: str
    timeframe: str
    value: float
    pivot_time: pd.Timestamp
    confirm_time: pd.Timestamp
    pivot_index: int
    confirm_index: int
    rank: int


@dataclass(frozen=True)
class HorizontalTurtleSpec:
    entry_strategy: str
    max_channel_level_gap_atr: float
    min_reclaim_pos: float
    target_rr: float
    stop_buffer_atr: float
    min_sweep_depth_atr: float
    max_sweep_depth_atr: float
    min_level_age_bars: int
    entry_window_bars: int
    confirm_window_bars: int
    min_risk_atr: float
    max_risk_atr: float
    max_hold_bars: int


BFM_SCALP_TF_SETS = (
    "5m=144:96,115:77,92:61,74:49;"
    "15m=120:80,96:64,77:51,62:41;"
    f"{OPTIMIZED_BFM_TF_SETS}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune original-style Turtle Soup sweeps of horizontal HTF support/resistance, "
            "using BFM channels as confluence features rather than as the entry trigger."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("scripts/channel_15m_broad_v2_full5y_config.json"))
    parser.add_argument("--symbols", default="BTCUSDT")
    parser.add_argument("--start", default="2021-09-01")
    parser.add_argument("--end", default="2026-04-20")
    parser.add_argument("--cache-dir", type=Path, default=Path("scripts/.cache"))
    parser.add_argument("--exec-timeframe", default="5m")
    parser.add_argument("--horizontal-timeframes", default="4h,1d")
    parser.add_argument(
        "--horizontal-pivots",
        default="4h=3:3;1d=2:2",
        help="Per-timeframe pivot left:right definitions, e.g. '1h=5:5;4h=3:3;1d=2:2'.",
    )
    parser.add_argument("--line-timeframes", default="5m,15m,1h,4h,1d")
    parser.add_argument("--bfm-tf-sets", default=BFM_SCALP_TF_SETS)
    parser.add_argument("--bfm-invalidation", choices=["wick", "close", "none"], default="wick")
    parser.add_argument("--bfm-max-extension-bars", type=int, default=300)
    parser.add_argument("--directions", default="long,short")
    parser.add_argument("--entry-strategies", default="reclaim_next_open,level_retest,break_confirm")
    parser.add_argument("--max-channel-level-gap-atrs", default="0.5,1.0,1.5,999")
    parser.add_argument("--min-reclaim-positions", default="0.45,0.6")
    parser.add_argument("--target-rrs", default="1.0,1.5,2.0")
    parser.add_argument("--stop-buffer-atrs", default="0.05,0.10")
    parser.add_argument("--min-sweep-depth-atrs", default="0.02,0.08")
    parser.add_argument("--max-sweep-depth-atrs", default="2.5")
    parser.add_argument("--min-level-age-bars", default="12,48")
    parser.add_argument("--entry-window-bars", default="12")
    parser.add_argument("--confirm-window-bars", default="12")
    parser.add_argument("--min-risk-atrs", default="0.10")
    parser.add_argument("--max-risk-atrs", default="6.0")
    parser.add_argument("--max-hold-bars", default="96,192")
    parser.add_argument("--level-cluster-atrs", type=float, default=0.25)
    parser.add_argument("--max-level-scan", type=int, default=400)
    parser.add_argument("--fee-bps-side", type=float, default=None)
    parser.add_argument("--slippage-bps-side", type=float, default=None)
    parser.add_argument("--risk-fraction", type=float, default=None)
    parser.add_argument("--min-trades-for-score", type=int, default=100)
    parser.add_argument("--max-configs", type=int, default=0)
    parser.add_argument("--output-prefix", type=Path, default=Path("scripts/horizontal_turtle_bfm"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_production_config(args.config)
    horizontal_timeframes = parse_timeframes(args.horizontal_timeframes, "4h")
    line_timeframes = parse_timeframes(args.line_timeframes, "1h")
    bfm_sets_by_tf = parse_tf_sets(args.bfm_tf_sets, line_timeframes)
    horizontal_pivots = parse_horizontal_pivots(args.horizontal_pivots, horizontal_timeframes)
    fee_bps_side = float(config.fee_bps_side if args.fee_bps_side is None else args.fee_bps_side)
    slippage_bps_side = float(config.slippage_bps_side if args.slippage_bps_side is None else args.slippage_bps_side)
    risk_fraction = float(config.risk.risk_fraction if args.risk_fraction is None else args.risk_fraction)
    specs = build_specs(args)
    if args.max_configs > 0:
        specs = specs[: int(args.max_configs)]

    all_summary_rows: list[dict[str, Any]] = []
    global_best_rank = (False, -float("inf"), -float("inf"), -float("inf"))
    global_best_trades = pd.DataFrame()
    global_best_candidates = pd.DataFrame()
    global_best_config: dict[str, Any] = {}

    for raw_symbol in parse_str_list(args.symbols):
        symbol = normalize_binance_spot_symbol(raw_symbol)
        print(f"\nLoading {symbol} {args.exec_timeframe} data {args.start} -> {args.end}")
        base = load_base_candles(symbol, args.start, args.end, cache_dir=args.cache_dir, interval="5m")
        all_timeframes = unique_preserve_order([args.exec_timeframe, *horizontal_timeframes, *line_timeframes])
        bars_by_tf = {
            timeframe: prepare_timeframe_bars(base, timeframe, atr_length=config.atr_length)
            for timeframe in all_timeframes
        }
        exec_bars = bars_by_tf[args.exec_timeframe].reset_index(drop=True)
        levels = build_horizontal_levels(bars_by_tf, horizontal_pivots)
        print(
            f"  horizontal levels: {sum(level.side == 'support' for level in levels):,} support / "
            f"{sum(level.side == 'resistance' for level in levels):,} resistance"
        )

        bundles: dict[str, LineBundle] = {}
        for timeframe in line_timeframes:
            lines, pivots = build_bfm_magic_lines(
                bars_by_tf[timeframe],
                bfm_sets_by_tf[timeframe],
                invalidation=args.bfm_invalidation,
                max_extension_bars=args.bfm_max_extension_bars,
            )
            bundles[timeframe] = LineBundle(
                timeframe=timeframe,
                scale=1.0,
                sets=tuple(bfm_sets_by_tf[timeframe]),
                bars=bars_by_tf[timeframe],
                lines=tuple(lines),
                pivots_count=len(pivots),
            )
            print(f"  {timeframe} BFM: {len(pivots):,} pivots, {len(lines):,} lines")

        projection = project_lines_to_execution_frame(exec_bars, bundles)
        allowed_directions = set(parse_str_list(args.directions))
        print(f"  evaluating {len(specs):,} horizontal Turtle/BFM variants")

        symbol_best_rank = (False, -float("inf"), -float("inf"), -float("inf"))
        symbol_best_trades = pd.DataFrame()
        symbol_best_candidates = pd.DataFrame()
        symbol_best_config: dict[str, Any] = {}
        for spec_index, spec in enumerate(specs, start=1):
            candidates = build_candidates(
                exec_bars=exec_bars,
                levels=levels,
                projection=projection,
                line_timeframes=line_timeframes,
                spec=spec,
                symbol=symbol,
                allowed_directions=allowed_directions,
                level_cluster_atrs=float(args.level_cluster_atrs),
                max_level_scan=int(args.max_level_scan),
            )
            trades = label_and_schedule(
                candidates,
                exec_bars,
                fee_bps_side=fee_bps_side,
                slippage_bps_side=slippage_bps_side,
                risk_fraction=risk_fraction,
            )
            metrics = strategy_metrics(trades)
            score = score_metrics(metrics, int(args.min_trades_for_score))
            row = spec_row(spec)
            row.update(
                {
                    "symbol": symbol,
                    "config_index": spec_index,
                    "score": score,
                    "raw_candidates": float(len(candidates)),
                    "exec_timeframe": args.exec_timeframe,
                    "horizontal_timeframes": ",".join(horizontal_timeframes),
                    "line_timeframes": ",".join(line_timeframes),
                    "fee_bps_side": fee_bps_side,
                    "slippage_bps_side": slippage_bps_side,
                    "risk_fraction": risk_fraction,
                }
            )
            row.update(metrics)
            all_summary_rows.append(row)
            rank = (
                math.isfinite(score),
                float(score),
                float(metrics.get("total_return", 0.0)),
                float(metrics.get("net_r", 0.0)),
            )
            if rank > symbol_best_rank:
                symbol_best_rank = rank
                symbol_best_trades = trades
                symbol_best_candidates = candidates
                symbol_best_config = row
            if rank > global_best_rank:
                global_best_rank = rank
                global_best_trades = trades
                global_best_candidates = candidates
                global_best_config = row
            if spec_index == 1 or spec_index % 25 == 0 or spec_index == len(specs):
                print(
                    f"    [{spec_index}/{len(specs)}] best score {symbol_best_rank[1]:.4f}; "
                    f"latest {int(metrics['trades'])} trades, PF {metrics['profit_factor']:.2f}, "
                    f"netR {metrics['net_r']:.2f}"
                )

        write_symbol_outputs(args.output_prefix, symbol, symbol_best_config, symbol_best_candidates, symbol_best_trades)

    summary = pd.DataFrame(all_summary_rows).sort_values(["score", "total_return", "net_r"], ascending=[False, False, False])
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_prefix.with_name(f"{args.output_prefix.name}_summary.csv")
    best_trades_path = args.output_prefix.with_name(f"{args.output_prefix.name}_best_trades.csv")
    best_candidates_path = args.output_prefix.with_name(f"{args.output_prefix.name}_best_candidates.csv")
    best_config_path = args.output_prefix.with_name(f"{args.output_prefix.name}_best_config.json")
    summary.to_csv(summary_path, index=False)
    global_best_trades.to_csv(best_trades_path, index=False)
    global_best_candidates.to_csv(best_candidates_path, index=False)
    best_config_path.write_text(
        json.dumps(
            {
                **global_best_config,
                "horizontal_pivots": {tf: f"{left}:{right}" for tf, (left, right) in horizontal_pivots.items()},
                "bfm_tf_sets": {tf: format_sets(sets) for tf, sets in bfm_sets_by_tf.items()},
            },
            default=str,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print("\nBest horizontal Turtle/BFM config")
    for key in [
        "symbol",
        "entry_strategy",
        "max_channel_level_gap_atr",
        "min_reclaim_pos",
        "target_rr",
        "stop_buffer_atr",
        "min_sweep_depth_atr",
        "max_sweep_depth_atr",
        "min_level_age_bars",
        "max_hold_bars",
        "trades",
        "hit_rate",
        "profit_factor",
        "net_r",
        "total_return",
        "score",
    ]:
        print(f"  {key}: {global_best_config.get(key)}")
    print(f"\nWrote {summary_path}")
    print(f"Wrote {best_trades_path}")
    print(f"Wrote {best_candidates_path}")
    print(f"Wrote {best_config_path}")


def write_symbol_outputs(
    output_prefix: Path,
    symbol: str,
    config_row: dict[str, Any],
    candidates: pd.DataFrame,
    trades: pd.DataFrame,
) -> None:
    safe_symbol = symbol.lower()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    trades_path = output_prefix.with_name(f"{output_prefix.name}_{safe_symbol}_best_trades.csv")
    candidates_path = output_prefix.with_name(f"{output_prefix.name}_{safe_symbol}_best_candidates.csv")
    config_path = output_prefix.with_name(f"{output_prefix.name}_{safe_symbol}_best_config.json")
    trades.to_csv(trades_path, index=False)
    candidates.to_csv(candidates_path, index=False)
    config_path.write_text(json.dumps(config_row, default=str, indent=2, sort_keys=True), encoding="utf-8")
    print(f"  wrote {trades_path} ({len(trades):,} scheduled trades)")


def parse_horizontal_pivots(raw: str, timeframes: list[str]) -> dict[str, tuple[int, int]]:
    values: dict[str, tuple[int, int]] = {}
    for item in str(raw).split(";"):
        item = item.strip()
        if not item:
            continue
        key, value = item.split("=", 1)
        left, right = value.split(":", 1)
        values[key.strip()] = (int(left), int(right))
    for timeframe in timeframes:
        values.setdefault(timeframe, (3, 3))
    return values


def build_specs(args: argparse.Namespace) -> list[HorizontalTurtleSpec]:
    specs: list[HorizontalTurtleSpec] = []
    entry_windows = parse_int_list(args.entry_window_bars)
    confirm_windows = parse_int_list(args.confirm_window_bars)
    for strategy in parse_str_list(args.entry_strategies):
        for gap in parse_float_list(args.max_channel_level_gap_atrs):
            for reclaim in parse_float_list(args.min_reclaim_positions):
                for rr in parse_float_list(args.target_rrs):
                    for stop_buffer in parse_float_list(args.stop_buffer_atrs):
                        for min_depth in parse_float_list(args.min_sweep_depth_atrs):
                            for max_depth in parse_float_list(args.max_sweep_depth_atrs):
                                for age in parse_int_list(args.min_level_age_bars):
                                    for entry_window in entry_windows:
                                        for confirm_window in confirm_windows:
                                            for min_risk in parse_float_list(args.min_risk_atrs):
                                                for max_risk in parse_float_list(args.max_risk_atrs):
                                                    for hold in parse_int_list(args.max_hold_bars):
                                                        if min_risk > max_risk:
                                                            continue
                                                        if strategy != "level_retest" and entry_window != entry_windows[0]:
                                                            continue
                                                        if strategy != "break_confirm" and confirm_window != confirm_windows[0]:
                                                            continue
                                                        specs.append(
                                                            HorizontalTurtleSpec(
                                                                entry_strategy=strategy,
                                                                max_channel_level_gap_atr=float(gap),
                                                                min_reclaim_pos=float(reclaim),
                                                                target_rr=float(rr),
                                                                stop_buffer_atr=float(stop_buffer),
                                                                min_sweep_depth_atr=float(min_depth),
                                                                max_sweep_depth_atr=float(max_depth),
                                                                min_level_age_bars=int(age),
                                                                entry_window_bars=int(entry_window),
                                                                confirm_window_bars=int(confirm_window),
                                                                min_risk_atr=float(min_risk),
                                                                max_risk_atr=float(max_risk),
                                                                max_hold_bars=int(hold),
                                                            )
                                                        )
    return specs


def build_horizontal_levels(
    bars_by_tf: dict[str, pd.DataFrame],
    pivots_by_tf: dict[str, tuple[int, int]],
) -> list[HorizontalLevel]:
    levels: list[HorizontalLevel] = []
    tf_rank = {timeframe: rank for rank, timeframe in enumerate(reversed(list(pivots_by_tf.keys())), start=1)}
    for timeframe, (left, right) in pivots_by_tf.items():
        bars = bars_by_tf[timeframe].reset_index(drop=True)
        close_times = pd.to_datetime(bars["close_time"], utc=True, errors="coerce")
        for side, column, mode in [("resistance", "high", "high"), ("support", "low", "low")]:
            for count, item in enumerate(build_confirmed_pivots(bars[column], left, right, mode), start=1):
                pivot_index = int(item["pivot_index"])
                confirm_index = pivot_index + int(right)
                if confirm_index >= len(bars):
                    continue
                levels.append(
                    HorizontalLevel(
                        level_id=f"{timeframe}-{side}-{count}",
                        side=side,
                        timeframe=timeframe,
                        value=float(item["value"]),
                        pivot_time=pd.Timestamp(close_times.iloc[pivot_index]).tz_convert("UTC"),
                        confirm_time=pd.Timestamp(close_times.iloc[confirm_index]).tz_convert("UTC"),
                        pivot_index=pivot_index,
                        confirm_index=confirm_index,
                        rank=int(tf_rank.get(timeframe, 0)),
                    )
                )
    return sorted(levels, key=lambda level: (level.confirm_time, level.timeframe, level.side, level.value))


def build_candidates(
    *,
    exec_bars: pd.DataFrame,
    levels: list[HorizontalLevel],
    projection: Projection,
    line_timeframes: list[str],
    spec: HorizontalTurtleSpec,
    symbol: str,
    allowed_directions: set[str],
    level_cluster_atrs: float,
    max_level_scan: int,
) -> pd.DataFrame:
    opens = pd.to_numeric(exec_bars["open"], errors="coerce").to_numpy(dtype=float)
    highs = pd.to_numeric(exec_bars["high"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(exec_bars["low"], errors="coerce").to_numpy(dtype=float)
    closes = pd.to_numeric(exec_bars["close"], errors="coerce").to_numpy(dtype=float)
    atrs = pd.to_numeric(exec_bars["atr"], errors="coerce").to_numpy(dtype=float)
    times = pd.to_datetime(exec_bars["close_time"], utc=True, errors="coerce").reset_index(drop=True)
    time_values = times.to_numpy(dtype="datetime64[ns]").astype("int64")
    level_confirm_indices = {
        level.level_id: int(np.searchsorted(time_values, int(level.confirm_time.tz_convert("UTC").value), side="left"))
        for level in levels
    }
    supports = [level for level in levels if level.side == "support"]
    resistances = [level for level in levels if level.side == "resistance"]
    support_ptr = 0
    resistance_ptr = 0
    active_supports: list[HorizontalLevel] = []
    active_resistances: list[HorizontalLevel] = []
    rows: list[dict[str, Any]] = []
    n = len(exec_bars)
    ranges = highs - lows
    long_reclaim = np.divide(closes - lows, ranges, out=np.zeros(n, dtype=float), where=ranges > 0.0)
    short_reclaim = np.divide(highs - closes, ranges, out=np.zeros(n, dtype=float), where=ranges > 0.0)

    for index in range(n - 1):
        now = pd.Timestamp(times.iloc[index]).tz_convert("UTC")
        while support_ptr < len(supports) and supports[support_ptr].confirm_time <= now:
            active_supports.append(supports[support_ptr])
            support_ptr += 1
        while resistance_ptr < len(resistances) and resistances[resistance_ptr].confirm_time <= now:
            active_resistances.append(resistances[resistance_ptr])
            resistance_ptr += 1
        atr = float(atrs[index])
        if not math.isfinite(atr) or atr <= 0.0:
            continue
        if "long" in allowed_directions:
            active_supports, long_candidates = scan_side(
                side="support",
                active_levels=active_supports,
                index=index,
                opens=opens,
                highs=highs,
                lows=lows,
                closes=closes,
                atrs=atrs,
                times=times,
                projection=projection,
                line_timeframes=line_timeframes,
                spec=spec,
                symbol=symbol,
                reclaim_pos=float(long_reclaim[index]),
                level_confirm_indices=level_confirm_indices,
                level_cluster_atrs=float(level_cluster_atrs),
                max_level_scan=max_level_scan,
            )
            rows.extend(long_candidates)
        if "short" in allowed_directions:
            active_resistances, short_candidates = scan_side(
                side="resistance",
                active_levels=active_resistances,
                index=index,
                opens=opens,
                highs=highs,
                lows=lows,
                closes=closes,
                atrs=atrs,
                times=times,
                projection=projection,
                line_timeframes=line_timeframes,
                spec=spec,
                symbol=symbol,
                reclaim_pos=float(short_reclaim[index]),
                level_confirm_indices=level_confirm_indices,
                level_cluster_atrs=float(level_cluster_atrs),
                max_level_scan=max_level_scan,
            )
            rows.extend(short_candidates)
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(["entry_index", "channel_gap_atr", "horizontal_rank", "risk_atr"], ascending=[True, True, False, True])
        .reset_index(drop=True)
    )


def scan_side(
    *,
    side: str,
    active_levels: list[HorizontalLevel],
    index: int,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    atrs: np.ndarray,
    times: pd.Series,
    projection: Projection,
    line_timeframes: list[str],
    spec: HorizontalTurtleSpec,
    symbol: str,
    reclaim_pos: float,
    level_confirm_indices: dict[str, int],
    level_cluster_atrs: float,
    max_level_scan: int,
) -> tuple[list[HorizontalLevel], list[dict[str, Any]]]:
    atr = float(atrs[index])
    kept: list[HorizontalLevel] = []
    swept: list[HorizontalLevel] = []
    scan_start = max(0, len(active_levels) - max(1, int(max_level_scan)))
    scan_ids = {level.level_id for level in active_levels[scan_start:]}
    for level in active_levels:
        age = index - int(level_confirm_indices.get(level.level_id, index))
        if level.side == "support":
            invalid = closes[index] < level.value
            event = (
                level.level_id in scan_ids
                and age >= spec.min_level_age_bars
                and lows[index] < level.value
                and closes[index] > level.value
            )
            depth = (level.value - lows[index]) / atr if atr > 0.0 else math.nan
        else:
            invalid = closes[index] > level.value
            event = (
                level.level_id in scan_ids
                and age >= spec.min_level_age_bars
                and highs[index] > level.value
                and closes[index] < level.value
            )
            depth = (highs[index] - level.value) / atr if atr > 0.0 else math.nan
        if event and reclaim_pos >= spec.min_reclaim_pos and spec.min_sweep_depth_atr <= depth <= spec.max_sweep_depth_atr:
            swept.append(level)
            continue
        if not invalid:
            kept.append(level)

    candidates: list[dict[str, Any]] = []
    for level in swept:
        candidate = make_candidate(
            level=level,
            index=index,
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
            atrs=atrs,
            times=times,
            projection=projection,
            line_timeframes=line_timeframes,
            spec=spec,
            symbol=symbol,
            reclaim_pos=reclaim_pos,
            level_confirm_indices=level_confirm_indices,
        )
        if candidate is not None:
            candidates.append(candidate)
    if not candidates:
        return kept, []

    candidates.sort(key=lambda row: (float(row["channel_gap_atr"]), -float(row["horizontal_rank"]), float(row["risk_atr"])))
    selected = candidates[0]
    selected_level = float(selected["liquidity_level"])
    cluster_width = float(level_cluster_atrs) * atr
    kept = [level for level in kept if abs(float(level.value) - selected_level) > cluster_width]
    return kept, [selected]


def make_candidate(
    *,
    level: HorizontalLevel,
    index: int,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    atrs: np.ndarray,
    times: pd.Series,
    projection: Projection,
    line_timeframes: list[str],
    spec: HorizontalTurtleSpec,
    symbol: str,
    reclaim_pos: float,
    level_confirm_indices: dict[str, int],
) -> dict[str, Any] | None:
    del closes
    direction = "long" if level.side == "support" else "short"
    atr = float(atrs[index])
    if not math.isfinite(atr) or atr <= 0.0:
        return None
    penetration_price = float(lows[index] if direction == "long" else highs[index])
    sweep_depth_atr = (level.value - penetration_price) / atr if direction == "long" else (penetration_price - level.value) / atr
    channel = channel_confluence_for_level(
        direction=direction,
        level=float(level.value),
        index=index,
        projection=projection,
        line_timeframes=line_timeframes,
        atr=atr,
    )
    if channel is None:
        if spec.max_channel_level_gap_atr < 999.0:
            return None
        channel = {"value": math.nan, "gap_atr": 999.0, "tf": "", "set": -1, "source": "none"}
    if float(channel["gap_atr"]) > spec.max_channel_level_gap_atr:
        return None

    if direction == "long":
        stop_price = penetration_price - spec.stop_buffer_atr * atr
    else:
        stop_price = penetration_price + spec.stop_buffer_atr * atr

    entry = resolve_entry(
        direction=direction,
        strategy=spec.entry_strategy,
        event_index=index,
        level=float(level.value),
        stop_price=stop_price,
        entry_window_bars=spec.entry_window_bars,
        confirm_window_bars=spec.confirm_window_bars,
        opens=opens,
        highs=highs,
        lows=lows,
    )
    if entry is None:
        return None
    entry_index, entry_price, signal_index = entry
    risk = entry_price - stop_price if direction == "long" else stop_price - entry_price
    risk_atr = risk / atr
    if not math.isfinite(risk) or risk <= 0.0 or risk_atr < spec.min_risk_atr or risk_atr > spec.max_risk_atr:
        return None
    target_price = entry_price + spec.target_rr * risk if direction == "long" else entry_price - spec.target_rr * risk
    confirm_index = int(level_confirm_indices.get(level.level_id, index))
    return {
        "symbol": symbol,
        "entry_strategy": spec.entry_strategy,
        "trigger_family": "horizontal_turtle",
        "event_index": int(index),
        "signal_index": int(signal_index),
        "entry_index": int(entry_index),
        "event_time": pd.Timestamp(times.iloc[index]).tz_convert("UTC"),
        "entry_time": pd.Timestamp(times.iloc[entry_index]).tz_convert("UTC"),
        "direction": direction,
        "entry_price": float(entry_price),
        "stop_price": float(stop_price),
        "target_price": float(target_price),
        "risk_abs": float(risk),
        "target_rr_planned": float(spec.target_rr),
        "max_hold_bars": int(spec.max_hold_bars),
        "liquidity_level": float(level.value),
        "horizontal_level": float(level.value),
        "horizontal_tf": level.timeframe,
        "horizontal_side": level.side,
        "horizontal_rank": int(level.rank),
        "horizontal_pivot_time": level.pivot_time,
        "horizontal_confirm_time": level.confirm_time,
        "level_age_bars": int(index - confirm_index),
        "channel_value": float(channel["value"]),
        "channel_gap_atr": float(channel["gap_atr"]),
        "channel_tf": str(channel["tf"]),
        "channel_set": int(channel["set"]),
        "channel_source": str(channel["source"]),
        "reclaim_pos": float(reclaim_pos),
        "sweep_depth_atr": float(sweep_depth_atr),
        "entry_delay_bars": int(entry_index - index),
        "lookback_bars": int(spec.min_level_age_bars),
        "trend_filter": "none",
        "min_risk_atr": float(spec.min_risk_atr),
        "max_risk_atr": float(spec.max_risk_atr),
        "stop_lookback_bars": 0,
        "penetration_price": float(penetration_price),
        "stop_anchor_price": float(penetration_price),
        "risk_atr": float(risk_atr),
    }


def channel_confluence_for_level(
    *,
    direction: str,
    level: float,
    index: int,
    projection: Projection,
    line_timeframes: list[str],
    atr: float,
) -> dict[str, Any] | None:
    options: list[dict[str, Any]] = []
    if direction == "long":
        append_channel_option(
            options,
            value=projection.support_touch_value[index],
            tf_index=projection.support_touch_tf[index],
            set_number=projection.support_touch_set[index],
            source="support_touch",
            level=level,
            atr=atr,
            line_timeframes=line_timeframes,
        )
        append_channel_option(
            options,
            value=projection.nearest_support_below[index],
            tf_index=projection.nearest_support_below_tf[index],
            set_number=projection.nearest_support_below_set[index],
            source="nearest_support_below",
            level=level,
            atr=atr,
            line_timeframes=line_timeframes,
        )
    else:
        append_channel_option(
            options,
            value=projection.resistance_touch_value[index],
            tf_index=projection.resistance_touch_tf[index],
            set_number=projection.resistance_touch_set[index],
            source="resistance_touch",
            level=level,
            atr=atr,
            line_timeframes=line_timeframes,
        )
        append_channel_option(
            options,
            value=projection.nearest_resistance_above[index],
            tf_index=projection.nearest_resistance_above_tf[index],
            set_number=projection.nearest_resistance_above_set[index],
            source="nearest_resistance_above",
            level=level,
            atr=atr,
            line_timeframes=line_timeframes,
        )
    if not options:
        return None
    options.sort(key=lambda item: float(item["gap_atr"]))
    return options[0]


def append_channel_option(
    options: list[dict[str, Any]],
    *,
    value: float,
    tf_index: int,
    set_number: int,
    source: str,
    level: float,
    atr: float,
    line_timeframes: list[str],
) -> None:
    value = float(value)
    if not math.isfinite(value) or atr <= 0.0:
        return
    options.append(
        {
            "value": value,
            "gap_atr": abs(value - float(level)) / float(atr),
            "tf": tf_name(line_timeframes, int(tf_index)),
            "set": int(set_number),
            "source": source,
        }
    )


def resolve_entry(
    *,
    direction: str,
    strategy: str,
    event_index: int,
    level: float,
    stop_price: float,
    entry_window_bars: int,
    confirm_window_bars: int,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
) -> tuple[int, float, int] | None:
    if strategy == "reclaim_next_open":
        entry_index = event_index + 1
        if entry_index >= len(opens):
            return None
        return entry_index, float(opens[entry_index]), event_index
    if strategy == "level_retest":
        final = min(len(opens) - 1, event_index + int(entry_window_bars))
        for idx in range(event_index + 1, final + 1):
            if direction == "long":
                if lows[idx] <= stop_price:
                    return None
                if opens[idx] <= level:
                    return idx, float(opens[idx]), idx
                if lows[idx] <= level <= highs[idx]:
                    return idx, float(level), idx
            else:
                if highs[idx] >= stop_price:
                    return None
                if opens[idx] >= level:
                    return idx, float(opens[idx]), idx
                if lows[idx] <= level <= highs[idx]:
                    return idx, float(level), idx
        return None
    if strategy == "break_confirm":
        final = min(len(opens) - 2, event_index + int(confirm_window_bars))
        trigger = float(highs[event_index] if direction == "long" else lows[event_index])
        for idx in range(event_index + 1, final + 1):
            if direction == "long":
                if lows[idx] <= stop_price:
                    return None
                if highs[idx] > trigger:
                    return idx + 1, float(opens[idx + 1]), idx
            else:
                if highs[idx] >= stop_price:
                    return None
                if lows[idx] < trigger:
                    return idx + 1, float(opens[idx + 1]), idx
        return None
    raise ValueError(f"Unsupported entry strategy {strategy!r}")


def spec_row(spec: HorizontalTurtleSpec) -> dict[str, Any]:
    return {
        "entry_strategy": spec.entry_strategy,
        "max_channel_level_gap_atr": spec.max_channel_level_gap_atr,
        "min_reclaim_pos": spec.min_reclaim_pos,
        "target_rr": spec.target_rr,
        "stop_buffer_atr": spec.stop_buffer_atr,
        "min_sweep_depth_atr": spec.min_sweep_depth_atr,
        "max_sweep_depth_atr": spec.max_sweep_depth_atr,
        "min_level_age_bars": spec.min_level_age_bars,
        "entry_window_bars": spec.entry_window_bars,
        "confirm_window_bars": spec.confirm_window_bars,
        "min_risk_atr": spec.min_risk_atr,
        "max_risk_atr": spec.max_risk_atr,
        "max_hold_bars": spec.max_hold_bars,
    }


if __name__ == "__main__":
    main()
