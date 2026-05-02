from __future__ import annotations

import argparse
import itertools
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

from scripts.channel_state_research.backtest import strategy_metrics
from scripts.channel_state_research.data import load_base_candles, prepare_timeframe_bars
from scripts.channel_state_research.labels import high_before_low
from scripts.channel_state_research.production import ZoneChannelProductionConfig, load_production_config
from scripts.plot_zone_channel_history import (
    BfmTrendline,
    build_bfm_magic_lines,
    parse_bfm_sets,
    parse_timeframes,
)


BASE_BFM_SETS = "300:200,240:160,192:128,154:102"


@dataclass(frozen=True)
class LineBundle:
    timeframe: str
    scale: float
    sets: tuple[tuple[int, int], ...]
    bars: pd.DataFrame
    lines: tuple[BfmTrendline, ...]
    pivots_count: int


@dataclass(frozen=True)
class Projection:
    support_touch_value: np.ndarray
    support_touch_gap: np.ndarray
    support_touch_tf: np.ndarray
    support_touch_set: np.ndarray
    resistance_touch_value: np.ndarray
    resistance_touch_gap: np.ndarray
    resistance_touch_tf: np.ndarray
    resistance_touch_set: np.ndarray
    nearest_support_below: np.ndarray
    nearest_support_below_tf: np.ndarray
    nearest_support_below_set: np.ndarray
    nearest_resistance_above: np.ndarray
    nearest_resistance_above_tf: np.ndarray
    nearest_resistance_above_set: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune BFM Magic Trendline pivot parameters for a causal support/resistance "
            "rejection strategy."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("scripts/channel_15m_broad_v2_full5y_config.json"))
    parser.add_argument("--start", default="2021-04-30")
    parser.add_argument("--end", default="2026-04-30")
    parser.add_argument("--cache-dir", type=Path, default=Path("scripts/.cache"))
    parser.add_argument("--exec-timeframe", default=None, help="Execution/decision timeframe. Defaults to config decision_timeframe.")
    parser.add_argument(
        "--line-timeframes",
        default="1h,4h,1d",
        help="Comma-separated BFM line timeframes to combine, e.g. 1h,4h,1d.",
    )
    parser.add_argument("--base-sets", default=BASE_BFM_SETS, help="Base left:right sets to scale.")
    parser.add_argument(
        "--scale-grid",
        default="0.6,0.8,1.0",
        help="Default scale grid for every line timeframe.",
    )
    parser.add_argument(
        "--tf-scale-grid",
        default=None,
        help="Optional per-timeframe grid, e.g. '1h=0.6,0.8,1.0;4h=0.75,1.0;1d=0.5,0.75'.",
    )
    parser.add_argument("--max-configs", type=int, default=0, help="Optional cap for quick smoke runs.")
    parser.add_argument("--bfm-invalidation", choices=["wick", "close", "none"], default="wick")
    parser.add_argument("--bfm-max-extension-bars", type=int, default=300)
    parser.add_argument("--touch-epsilon-atr", type=float, default=None)
    parser.add_argument("--stop-buffer-atr", type=float, default=None)
    parser.add_argument("--target-buffer-atr", type=float, default=None)
    parser.add_argument("--min-reclaim-pos", type=float, default=None)
    parser.add_argument("--touch-lookback-bars", type=int, default=None)
    parser.add_argument("--horizon-bars", type=int, default=None)
    parser.add_argument("--min-target-rr", type=float, default=0.35)
    parser.add_argument("--max-target-rr", type=float, default=5.0)
    parser.add_argument("--fallback-rr", type=float, default=0.0, help="Use this RR target if no opposite BFM line exists.")
    parser.add_argument("--objective", choices=["total_return", "net_r", "calmar", "sharpe", "profit_factor"], default="total_return")
    parser.add_argument("--min-trades-for-score", type=int, default=20)
    parser.add_argument("--output-prefix", type=Path, default=Path("scripts/bfm_sr_tuning"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_production_config(args.config)
    exec_timeframe = args.exec_timeframe or config.decision_timeframe
    line_timeframes = parse_timeframes(args.line_timeframes, "1h")
    pivot_template = tuple(parse_bfm_sets(args.base_sets))
    scale_grid_by_tf = parse_scale_grid_by_timeframe(args, line_timeframes)

    all_timeframes = unique_preserve_order([exec_timeframe, *line_timeframes])
    print(
        f"Loading {config.symbol} {config.base_interval} data {args.start} -> {args.end}; "
        f"execution {exec_timeframe}; lines {','.join(line_timeframes)}"
    )
    base = load_base_candles(
        config.symbol,
        args.start,
        args.end,
        cache_dir=args.cache_dir,
        interval=config.base_interval,
    )
    bars_by_tf = {
        timeframe: prepare_timeframe_bars(base, timeframe, atr_length=config.atr_length)
        for timeframe in all_timeframes
    }
    exec_bars = bars_by_tf[exec_timeframe].reset_index(drop=True)

    touch_epsilon_atr = float(config.touch_epsilon_atr if args.touch_epsilon_atr is None else args.touch_epsilon_atr)
    stop_buffer_atr = float(config.stop_buffer_atr if args.stop_buffer_atr is None else args.stop_buffer_atr)
    target_buffer_atr = float(config.target_buffer_atr if args.target_buffer_atr is None else args.target_buffer_atr)
    min_reclaim_pos = float(config.min_reclaim_pos if args.min_reclaim_pos is None else args.min_reclaim_pos)
    touch_lookback_bars = int(config.touch_lookback_bars if args.touch_lookback_bars is None else args.touch_lookback_bars)
    horizon_bars = int(config.label_horizon_bars if args.horizon_bars is None else args.horizon_bars)

    bundle_cache: dict[tuple[str, float], LineBundle] = {}
    for timeframe in line_timeframes:
        for scale in scale_grid_by_tf[timeframe]:
            key = (timeframe, scale)
            bundle_cache[key] = build_line_bundle(
                timeframe=timeframe,
                scale=scale,
                bars=bars_by_tf[timeframe],
                pivot_template=pivot_template,
                invalidation=args.bfm_invalidation,
                max_extension_bars=args.bfm_max_extension_bars,
            )
            bundle = bundle_cache[key]
            print(
                f"Built {timeframe} scale {scale:g}: {format_sets(bundle.sets)} | "
                f"{len(bundle.lines):,} lines, {bundle.pivots_count:,} pivots"
            )

    combos = list(itertools.product(*(scale_grid_by_tf[timeframe] for timeframe in line_timeframes)))
    if args.max_configs > 0:
        combos = combos[: int(args.max_configs)]
    print(f"Evaluating {len(combos):,} parameter combinations")

    summaries: list[dict[str, Any]] = []
    best_trades = pd.DataFrame()
    best_projection_trades = pd.DataFrame()
    best_score = -float("inf")
    best_config: dict[str, Any] | None = None

    for config_index, combo in enumerate(combos, start=1):
        selected = {
            timeframe: bundle_cache[(timeframe, scale)]
            for timeframe, scale in zip(line_timeframes, combo, strict=True)
        }
        projection = project_lines_to_execution_frame(exec_bars, selected)
        candidates = build_signal_candidates(
            exec_bars=exec_bars,
            projection=projection,
            line_timeframes=line_timeframes,
            symbol=config.symbol,
            touch_epsilon_atr=touch_epsilon_atr,
            stop_buffer_atr=stop_buffer_atr,
            target_buffer_atr=target_buffer_atr,
            min_reclaim_pos=min_reclaim_pos,
            touch_lookback_bars=touch_lookback_bars,
            min_target_rr=float(args.min_target_rr),
            max_target_rr=float(args.max_target_rr),
            fallback_rr=float(args.fallback_rr),
        )
        trades = label_and_schedule_trades(
            candidates=candidates,
            bars=exec_bars,
            horizon_bars=horizon_bars,
            fee_bps_side=float(config.fee_bps_side),
            slippage_bps_side=float(config.slippage_bps_side),
            risk_fraction=float(config.risk.risk_fraction),
            one_trade_at_a_time=bool(config.risk.one_trade_at_a_time),
        )
        metrics = strategy_metrics(trades)
        score = objective_score(metrics, args.objective, int(args.min_trades_for_score))
        row: dict[str, Any] = {
            "config_index": config_index,
            "score": score,
            "objective": args.objective,
            "exec_timeframe": exec_timeframe,
            "line_timeframes": ",".join(line_timeframes),
            "touch_epsilon_atr": touch_epsilon_atr,
            "stop_buffer_atr": stop_buffer_atr,
            "target_buffer_atr": target_buffer_atr,
            "min_reclaim_pos": min_reclaim_pos,
            "touch_lookback_bars": touch_lookback_bars,
            "horizon_bars": horizon_bars,
            "min_target_rr": float(args.min_target_rr),
            "max_target_rr": float(args.max_target_rr),
            "fallback_rr": float(args.fallback_rr),
            "raw_candidates": float(len(candidates)),
        }
        for timeframe, scale in zip(line_timeframes, combo, strict=True):
            row[f"{timeframe}_scale"] = float(scale)
            row[f"{timeframe}_sets"] = format_sets(selected[timeframe].sets)
            row[f"{timeframe}_lines"] = float(len(selected[timeframe].lines))
        row.update(metrics)
        summaries.append(row)
        if best_config is None or score > best_score:
            best_score = score
            best_config = row
            best_trades = trades
            best_projection_trades = candidates
        if config_index == 1 or config_index % 10 == 0 or config_index == len(combos):
            print(
                f"[{config_index}/{len(combos)}] best score {best_score:.4f}; "
                f"latest trades {int(metrics['trades'])}, return {metrics['total_return']:.2%}, "
                f"netR {metrics['net_r']:.2f}"
            )

    summary = pd.DataFrame(summaries).sort_values(["score", "total_return", "net_r"], ascending=[False, False, False])
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_prefix.with_name(f"{args.output_prefix.name}_summary.csv")
    trades_path = args.output_prefix.with_name(f"{args.output_prefix.name}_best_trades.csv")
    candidates_path = args.output_prefix.with_name(f"{args.output_prefix.name}_best_candidates.csv")
    config_path = args.output_prefix.with_name(f"{args.output_prefix.name}_best_config.json")
    summary.to_csv(summary_path, index=False)
    best_trades.to_csv(trades_path, index=False)
    best_projection_trades.to_csv(candidates_path, index=False)
    config_path.write_text(json.dumps(best_config or {}, indent=2, sort_keys=True, default=str), encoding="utf-8")

    if best_config is not None:
        print("\nBest configuration")
        for key in ["score", "trades", "total_return", "max_drawdown", "hit_rate", "profit_factor", "net_r", "calmar", "sharpe"]:
            print(f"  {key}: {best_config.get(key)}")
        for timeframe in line_timeframes:
            print(f"  {timeframe}: scale {best_config[f'{timeframe}_scale']}, sets {best_config[f'{timeframe}_sets']}")
    print(f"\nWrote {summary_path}")
    print(f"Wrote {trades_path}")
    print(f"Wrote {candidates_path}")
    print(f"Wrote {config_path}")


def unique_preserve_order(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def parse_scale_grid_by_timeframe(args: argparse.Namespace, timeframes: list[str]) -> dict[str, list[float]]:
    default_grid = parse_float_list(args.scale_grid)
    out = {timeframe: list(default_grid) for timeframe in timeframes}
    if args.tf_scale_grid is None or not str(args.tf_scale_grid).strip():
        return out
    for chunk in str(args.tf_scale_grid).split(";"):
        text = chunk.strip()
        if not text:
            continue
        if "=" not in text:
            raise ValueError(f"Invalid --tf-scale-grid chunk {text!r}; expected timeframe=v1,v2.")
        timeframe, raw_values = text.split("=", 1)
        timeframe = timeframe.strip()
        if timeframe not in out:
            raise ValueError(f"--tf-scale-grid references {timeframe!r}, not in --line-timeframes.")
        out[timeframe] = parse_float_list(raw_values)
    return out


def parse_float_list(raw: str) -> list[float]:
    values: list[float] = []
    for chunk in str(raw).split(","):
        text = chunk.strip()
        if not text:
            continue
        value = float(text)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"Scale values must be positive finite numbers, got {text!r}.")
        values.append(value)
    if not values:
        raise ValueError("At least one scale value is required.")
    return values


def build_line_bundle(
    *,
    timeframe: str,
    scale: float,
    bars: pd.DataFrame,
    pivot_template: tuple[tuple[int, int], ...],
    invalidation: str,
    max_extension_bars: int,
) -> LineBundle:
    scaled = tuple(scale_sets(pivot_template, scale))
    lines, pivots = build_bfm_magic_lines(
        bars,
        list(scaled),
        invalidation=invalidation,
        max_extension_bars=max_extension_bars,
    )
    return LineBundle(
        timeframe=timeframe,
        scale=float(scale),
        sets=scaled,
        bars=bars,
        lines=tuple(lines),
        pivots_count=len(pivots),
    )


def scale_sets(template: tuple[tuple[int, int], ...], scale: float) -> tuple[tuple[int, int], ...]:
    return tuple(
        (
            max(1, int(round(leftbars * scale))),
            max(1, int(round(rightbars * scale))),
        )
        for leftbars, rightbars in template
    )


def format_sets(sets: tuple[tuple[int, int], ...]) -> str:
    return ",".join(f"{left}:{right}" for left, right in sets)


def project_lines_to_execution_frame(exec_bars: pd.DataFrame, bundles: dict[str, LineBundle]) -> Projection:
    n = len(exec_bars)
    exec_times_ns = timestamp_ns(exec_bars["close_time"])
    exec_highs = pd.to_numeric(exec_bars["high"], errors="coerce").to_numpy(dtype=float)
    exec_lows = pd.to_numeric(exec_bars["low"], errors="coerce").to_numpy(dtype=float)
    exec_closes = pd.to_numeric(exec_bars["close"], errors="coerce").to_numpy(dtype=float)

    support_touch_value = np.full(n, np.nan, dtype=float)
    support_touch_gap = np.full(n, np.inf, dtype=float)
    support_touch_tf = np.full(n, -1, dtype=np.int16)
    support_touch_set = np.full(n, -1, dtype=np.int16)
    resistance_touch_value = np.full(n, np.nan, dtype=float)
    resistance_touch_gap = np.full(n, np.inf, dtype=float)
    resistance_touch_tf = np.full(n, -1, dtype=np.int16)
    resistance_touch_set = np.full(n, -1, dtype=np.int16)
    nearest_support_below = np.full(n, np.nan, dtype=float)
    nearest_support_below_dist = np.full(n, np.inf, dtype=float)
    nearest_support_below_tf = np.full(n, -1, dtype=np.int16)
    nearest_support_below_set = np.full(n, -1, dtype=np.int16)
    nearest_resistance_above = np.full(n, np.nan, dtype=float)
    nearest_resistance_above_dist = np.full(n, np.inf, dtype=float)
    nearest_resistance_above_tf = np.full(n, -1, dtype=np.int16)
    nearest_resistance_above_set = np.full(n, -1, dtype=np.int16)

    for tf_index, (timeframe, bundle) in enumerate(bundles.items()):
        source_times_ns = timestamp_ns(bundle.bars["close_time"])
        source_indices = pd.to_numeric(bundle.bars["bar_index"], errors="coerce").to_numpy(dtype=float)
        if len(source_times_ns) == 0:
            continue
        for line in bundle.lines:
            active_start_ns = int(pd.Timestamp(line.end_pivot.confirm_time).tz_convert("UTC").value)
            if line.line_end_index < 0 or line.line_end_index >= len(source_times_ns):
                continue
            active_end_ns = int(source_times_ns[line.line_end_index])
            if active_end_ns < active_start_ns:
                continue
            lo = int(np.searchsorted(exec_times_ns, active_start_ns, side="left"))
            hi = int(np.searchsorted(exec_times_ns, active_end_ns, side="right"))
            if lo >= hi:
                continue
            source_x = np.interp(exec_times_ns[lo:hi].astype(float), source_times_ns.astype(float), source_indices)
            values = line.slope * source_x + line.intercept

            if line.side == "support":
                touch_gap = np.abs(exec_lows[lo:hi] - values)
                view_gap = support_touch_gap[lo:hi]
                view_value = support_touch_value[lo:hi]
                view_tf = support_touch_tf[lo:hi]
                view_set = support_touch_set[lo:hi]
                mask = np.isfinite(values) & (touch_gap < view_gap)
                if np.any(mask):
                    view_gap[mask] = touch_gap[mask]
                    view_value[mask] = values[mask]
                    view_tf[mask] = tf_index
                    view_set[mask] = line.set_number

                dist = exec_closes[lo:hi] - values
                view_dist = nearest_support_below_dist[lo:hi]
                view_value = nearest_support_below[lo:hi]
                view_tf = nearest_support_below_tf[lo:hi]
                view_set = nearest_support_below_set[lo:hi]
                mask = np.isfinite(values) & (dist > 0.0) & (dist < view_dist)
                if np.any(mask):
                    view_dist[mask] = dist[mask]
                    view_value[mask] = values[mask]
                    view_tf[mask] = tf_index
                    view_set[mask] = line.set_number
            else:
                touch_gap = np.abs(exec_highs[lo:hi] - values)
                view_gap = resistance_touch_gap[lo:hi]
                view_value = resistance_touch_value[lo:hi]
                view_tf = resistance_touch_tf[lo:hi]
                view_set = resistance_touch_set[lo:hi]
                mask = np.isfinite(values) & (touch_gap < view_gap)
                if np.any(mask):
                    view_gap[mask] = touch_gap[mask]
                    view_value[mask] = values[mask]
                    view_tf[mask] = tf_index
                    view_set[mask] = line.set_number

                dist = values - exec_closes[lo:hi]
                view_dist = nearest_resistance_above_dist[lo:hi]
                view_value = nearest_resistance_above[lo:hi]
                view_tf = nearest_resistance_above_tf[lo:hi]
                view_set = nearest_resistance_above_set[lo:hi]
                mask = np.isfinite(values) & (dist > 0.0) & (dist < view_dist)
                if np.any(mask):
                    view_dist[mask] = dist[mask]
                    view_value[mask] = values[mask]
                    view_tf[mask] = tf_index
                    view_set[mask] = line.set_number

    return Projection(
        support_touch_value=support_touch_value,
        support_touch_gap=support_touch_gap,
        support_touch_tf=support_touch_tf,
        support_touch_set=support_touch_set,
        resistance_touch_value=resistance_touch_value,
        resistance_touch_gap=resistance_touch_gap,
        resistance_touch_tf=resistance_touch_tf,
        resistance_touch_set=resistance_touch_set,
        nearest_support_below=nearest_support_below,
        nearest_support_below_tf=nearest_support_below_tf,
        nearest_support_below_set=nearest_support_below_set,
        nearest_resistance_above=nearest_resistance_above,
        nearest_resistance_above_tf=nearest_resistance_above_tf,
        nearest_resistance_above_set=nearest_resistance_above_set,
    )


def timestamp_ns(values: pd.Series) -> np.ndarray:
    # Pick nanoseconds explicitly; cached Binance frames in this repo can carry
    # microsecond dtypes, and raw astype("int64") would then silently mix units
    # with pd.Timestamp.value.
    return pd.to_datetime(values, utc=True, errors="coerce").to_numpy(dtype="datetime64[ns]").astype("int64")


def build_signal_candidates(
    *,
    exec_bars: pd.DataFrame,
    projection: Projection,
    line_timeframes: list[str],
    symbol: str,
    touch_epsilon_atr: float,
    stop_buffer_atr: float,
    target_buffer_atr: float,
    min_reclaim_pos: float,
    touch_lookback_bars: int,
    min_target_rr: float,
    max_target_rr: float,
    fallback_rr: float,
) -> pd.DataFrame:
    highs = pd.to_numeric(exec_bars["high"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(exec_bars["low"], errors="coerce").to_numpy(dtype=float)
    closes = pd.to_numeric(exec_bars["close"], errors="coerce").to_numpy(dtype=float)
    atrs = pd.to_numeric(exec_bars["atr"], errors="coerce").to_numpy(dtype=float)
    times = pd.to_datetime(exec_bars["close_time"], utc=True, errors="coerce")
    rows: list[dict[str, Any]] = []

    ranges = highs - lows
    long_reclaim_pos = np.divide(closes - lows, ranges, out=np.zeros_like(closes), where=ranges > 0.0)
    short_reclaim_pos = np.divide(highs - closes, ranges, out=np.zeros_like(closes), where=ranges > 0.0)
    touch_budget = touch_epsilon_atr * atrs

    touch_lookback_bars = max(0, int(touch_lookback_bars))
    recent_support: dict[str, Any] | None = None
    recent_resistance: dict[str, Any] | None = None

    for index in range(len(exec_bars)):
        if np.isfinite(atrs[index]) and atrs[index] > 0.0:
            if np.isfinite(projection.support_touch_value[index]) and projection.support_touch_gap[index] <= touch_budget[index]:
                recent_support = {
                    "index": index,
                    "value": float(projection.support_touch_value[index]),
                    "event_low": float(lows[index]),
                    "gap_atr": float(projection.support_touch_gap[index] / atrs[index]),
                    "tf": tf_name(line_timeframes, int(projection.support_touch_tf[index])),
                    "set": int(projection.support_touch_set[index]),
                }
            if np.isfinite(projection.resistance_touch_value[index]) and projection.resistance_touch_gap[index] <= touch_budget[index]:
                recent_resistance = {
                    "index": index,
                    "value": float(projection.resistance_touch_value[index]),
                    "event_high": float(highs[index]),
                    "gap_atr": float(projection.resistance_touch_gap[index] / atrs[index]),
                    "tf": tf_name(line_timeframes, int(projection.resistance_touch_tf[index])),
                    "set": int(projection.resistance_touch_set[index]),
                }

        if (
            recent_support is not None
            and index - int(recent_support["index"]) <= touch_lookback_bars
            and np.isfinite(projection.nearest_resistance_above[index])
            and np.isfinite(atrs[index])
            and atrs[index] > 0.0
            and closes[index] > float(recent_support["value"])
            and long_reclaim_pos[index] >= min_reclaim_pos
        ):
            append_long_candidate(
                rows=rows,
                symbol=symbol,
                index=index,
                event_time=times.iloc[index],
                entry=closes[index],
                stop_anchor=min(float(recent_support["value"]), float(recent_support["event_low"])),
                target_anchor=projection.nearest_resistance_above[index],
                atr=atrs[index],
                target_buffer_atr=target_buffer_atr,
                stop_buffer_atr=stop_buffer_atr,
                touched_line=float(recent_support["value"]),
                touch_gap_atr=float(recent_support["gap_atr"]),
                touch_tf=str(recent_support["tf"]),
                touch_set=int(recent_support["set"]),
                touch_delay_bars=int(index - int(recent_support["index"])),
                target_tf=tf_name(line_timeframes, int(projection.nearest_resistance_above_tf[index])),
                target_set=int(projection.nearest_resistance_above_set[index]),
                reclaim_pos=long_reclaim_pos[index],
                reaction_range_atr=ranges[index] / atrs[index],
                min_target_rr=min_target_rr,
                max_target_rr=max_target_rr,
            )

        if (
            recent_resistance is not None
            and index - int(recent_resistance["index"]) <= touch_lookback_bars
            and np.isfinite(projection.nearest_support_below[index])
            and np.isfinite(atrs[index])
            and atrs[index] > 0.0
            and closes[index] < float(recent_resistance["value"])
            and short_reclaim_pos[index] >= min_reclaim_pos
        ):
            append_short_candidate(
                rows=rows,
                symbol=symbol,
                index=index,
                event_time=times.iloc[index],
                entry=closes[index],
                stop_anchor=max(float(recent_resistance["value"]), float(recent_resistance["event_high"])),
                target_anchor=projection.nearest_support_below[index],
                atr=atrs[index],
                target_buffer_atr=target_buffer_atr,
                stop_buffer_atr=stop_buffer_atr,
                touched_line=float(recent_resistance["value"]),
                touch_gap_atr=float(recent_resistance["gap_atr"]),
                touch_tf=str(recent_resistance["tf"]),
                touch_set=int(recent_resistance["set"]),
                touch_delay_bars=int(index - int(recent_resistance["index"])),
                target_tf=tf_name(line_timeframes, int(projection.nearest_support_below_tf[index])),
                target_set=int(projection.nearest_support_below_set[index]),
                reclaim_pos=short_reclaim_pos[index],
                reaction_range_atr=ranges[index] / atrs[index],
                min_target_rr=min_target_rr,
                max_target_rr=max_target_rr,
            )

    if fallback_rr <= 0.0:
        if not rows:
            return pd.DataFrame()
        out = pd.DataFrame(rows).sort_values(["signal_index", "touch_gap_atr", "target_rr_planned"], ascending=[True, True, False])
        return out.reset_index(drop=True)

    # Optional fallback targets are kept separate from the main opposite-line
    # strategy so a normal run remains a clean support/resistance test.
    exact_long_mask = (
        np.isfinite(projection.support_touch_value)
        & np.isfinite(atrs)
        & (atrs > 0.0)
        & (projection.support_touch_gap <= touch_budget)
        & (closes > projection.support_touch_value)
        & (long_reclaim_pos >= min_reclaim_pos)
    )
    exact_short_mask = (
        np.isfinite(projection.resistance_touch_value)
        & np.isfinite(atrs)
        & (atrs > 0.0)
        & (projection.resistance_touch_gap <= touch_budget)
        & (closes < projection.resistance_touch_value)
        & (short_reclaim_pos >= min_reclaim_pos)
    )

    for index in np.where(exact_long_mask & ~np.isfinite(projection.nearest_resistance_above))[0]:
        entry = closes[index]
        support = projection.support_touch_value[index]
        stop = min(support, lows[index]) - stop_buffer_atr * atrs[index]
        risk = entry - stop
        target = entry + fallback_rr * risk
        maybe_append_candidate(
            rows,
            symbol=symbol,
            index=int(index),
            event_time=times.iloc[index],
            direction="long",
            entry=entry,
            stop=stop,
            target=target,
            atr=atrs[index],
            touched_line=support,
            touch_gap=projection.support_touch_gap[index],
            touch_tf=tf_name(line_timeframes, int(projection.support_touch_tf[index])),
            touch_set=int(projection.support_touch_set[index]),
            target_tf="",
            target_set=-1,
            target_source="fallback_rr",
            touch_delay_bars=0,
            reclaim_pos=long_reclaim_pos[index],
            reaction_range_atr=ranges[index] / atrs[index],
            min_target_rr=min_target_rr,
            max_target_rr=max_target_rr,
        )

    for index in np.where(exact_short_mask & ~np.isfinite(projection.nearest_support_below))[0]:
        entry = closes[index]
        resistance = projection.resistance_touch_value[index]
        stop = max(resistance, highs[index]) + stop_buffer_atr * atrs[index]
        risk = stop - entry
        target = entry - fallback_rr * risk
        maybe_append_candidate(
            rows,
            symbol=symbol,
            index=int(index),
            event_time=times.iloc[index],
            direction="short",
            entry=entry,
            stop=stop,
            target=target,
            atr=atrs[index],
            touched_line=resistance,
            touch_gap=projection.resistance_touch_gap[index],
            touch_tf=tf_name(line_timeframes, int(projection.resistance_touch_tf[index])),
            touch_set=int(projection.resistance_touch_set[index]),
            target_tf="",
            target_set=-1,
            target_source="fallback_rr",
            touch_delay_bars=0,
            reclaim_pos=short_reclaim_pos[index],
            reaction_range_atr=ranges[index] / atrs[index],
            min_target_rr=min_target_rr,
            max_target_rr=max_target_rr,
        )

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values(["signal_index", "touch_gap_atr", "target_rr_planned"], ascending=[True, True, False])
    return out.reset_index(drop=True)


def append_long_candidate(
    *,
    rows: list[dict[str, Any]],
    symbol: str,
    index: int,
    event_time: pd.Timestamp,
    entry: float,
    stop_anchor: float,
    target_anchor: float,
    atr: float,
    target_buffer_atr: float,
    stop_buffer_atr: float,
    touched_line: float,
    touch_gap_atr: float,
    touch_tf: str,
    touch_set: int,
    touch_delay_bars: int,
    target_tf: str,
    target_set: int,
    reclaim_pos: float,
    reaction_range_atr: float,
    min_target_rr: float,
    max_target_rr: float,
) -> None:
    maybe_append_candidate(
        rows,
        symbol=symbol,
        index=index,
        event_time=event_time,
        direction="long",
        entry=entry,
        stop=stop_anchor - stop_buffer_atr * atr,
        target=target_anchor - target_buffer_atr * atr,
        atr=atr,
        touched_line=touched_line,
        touch_gap=touch_gap_atr * atr,
        touch_tf=touch_tf,
        touch_set=touch_set,
        target_tf=target_tf,
        target_set=target_set,
        target_source="opposite_bfm",
        touch_delay_bars=touch_delay_bars,
        reclaim_pos=reclaim_pos,
        reaction_range_atr=reaction_range_atr,
        min_target_rr=min_target_rr,
        max_target_rr=max_target_rr,
    )


def append_short_candidate(
    *,
    rows: list[dict[str, Any]],
    symbol: str,
    index: int,
    event_time: pd.Timestamp,
    entry: float,
    stop_anchor: float,
    target_anchor: float,
    atr: float,
    target_buffer_atr: float,
    stop_buffer_atr: float,
    touched_line: float,
    touch_gap_atr: float,
    touch_tf: str,
    touch_set: int,
    touch_delay_bars: int,
    target_tf: str,
    target_set: int,
    reclaim_pos: float,
    reaction_range_atr: float,
    min_target_rr: float,
    max_target_rr: float,
) -> None:
    maybe_append_candidate(
        rows,
        symbol=symbol,
        index=index,
        event_time=event_time,
        direction="short",
        entry=entry,
        stop=stop_anchor + stop_buffer_atr * atr,
        target=target_anchor + target_buffer_atr * atr,
        atr=atr,
        touched_line=touched_line,
        touch_gap=touch_gap_atr * atr,
        touch_tf=touch_tf,
        touch_set=touch_set,
        target_tf=target_tf,
        target_set=target_set,
        target_source="opposite_bfm",
        touch_delay_bars=touch_delay_bars,
        reclaim_pos=reclaim_pos,
        reaction_range_atr=reaction_range_atr,
        min_target_rr=min_target_rr,
        max_target_rr=max_target_rr,
    )


def maybe_append_candidate(
    rows: list[dict[str, Any]],
    *,
    symbol: str,
    index: int,
    event_time: pd.Timestamp,
    direction: str,
    entry: float,
    stop: float,
    target: float,
    atr: float,
    touched_line: float,
    touch_gap: float,
    touch_tf: str,
    touch_set: int,
    target_tf: str,
    target_set: int,
    target_source: str,
    touch_delay_bars: int,
    reclaim_pos: float,
    reaction_range_atr: float,
    min_target_rr: float,
    max_target_rr: float,
) -> None:
    if not all(math.isfinite(float(value)) for value in [entry, stop, target, atr, touched_line, touch_gap]):
        return
    if atr <= 0.0:
        return
    if direction == "long":
        if not (stop < entry < target):
            return
        risk_abs = entry - stop
        target_distance = target - entry
    else:
        if not (target < entry < stop):
            return
        risk_abs = stop - entry
        target_distance = entry - target
    if risk_abs <= 0.0:
        return
    target_rr = target_distance / risk_abs
    if not math.isfinite(target_rr) or target_rr < min_target_rr or target_rr > max_target_rr:
        return
    rows.append(
        {
            "symbol": symbol,
            "signal_index": index,
            "event_time": pd.Timestamp(event_time).tz_convert("UTC"),
            "direction": direction,
            "entry_price": float(entry),
            "stop_price": float(stop),
            "target_price": float(target),
            "risk_abs": float(risk_abs),
            "target_rr_planned": float(target_rr),
            "touched_line": float(touched_line),
            "touch_gap_atr": float(touch_gap / atr),
            "touch_tf": touch_tf,
            "touch_set": touch_set,
            "target_tf": target_tf,
            "target_set": target_set,
            "target_source": target_source,
            "touch_delay_bars": int(touch_delay_bars),
            "reclaim_pos": float(reclaim_pos),
            "reaction_range_atr": float(reaction_range_atr),
        }
    )


def tf_name(timeframes: list[str], index: int) -> str:
    if index < 0 or index >= len(timeframes):
        return ""
    return timeframes[index]


def label_and_schedule_trades(
    *,
    candidates: pd.DataFrame,
    bars: pd.DataFrame,
    horizon_bars: int,
    fee_bps_side: float,
    slippage_bps_side: float,
    risk_fraction: float,
    one_trade_at_a_time: bool,
) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    opens = pd.to_numeric(bars["open"], errors="coerce").to_numpy(dtype=float)
    highs = pd.to_numeric(bars["high"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float)
    closes = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float)
    close_times = pd.to_datetime(bars["close_time"], utc=True, errors="coerce").to_list()

    rows: list[dict[str, Any]] = []
    active_until_index = -1
    ordered = candidates.sort_values(["signal_index", "touch_gap_atr", "target_rr_planned"], ascending=[True, True, False])
    for _, candidate in ordered.iterrows():
        signal_index = int(candidate["signal_index"])
        if one_trade_at_a_time and signal_index <= active_until_index:
            continue
        outcome = label_trade_outcome(
            direction=str(candidate["direction"]),
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
            close_times=close_times,
            signal_index=signal_index,
            entry_price=float(candidate["entry_price"]),
            stop_price=float(candidate["stop_price"]),
            target_price=float(candidate["target_price"]),
            horizon_bars=horizon_bars,
        )
        if outcome is None:
            continue
        risk_abs = float(candidate["risk_abs"])
        cost_r = ((2.0 * fee_bps_side) + (2.0 * slippage_bps_side)) / 10_000.0 * float(candidate["entry_price"]) / risk_abs
        gross_r = float(outcome["future_r"])
        net_r = gross_r - cost_r
        row = candidate.to_dict()
        row.update(
            {
                "entry_time": outcome["entry_time"],
                "exit_time": outcome["exit_time"],
                "exit_price": outcome["exit_price"],
                "exit_reason": outcome["outcome"],
                "hold_bars": int(outcome["bars_to_outcome"]),
                "mfe_r": float(outcome["mfe_r"]),
                "mae_r": float(outcome["mae_r"]),
                "cost_r": float(cost_r),
                "r_multiple_gross": float(gross_r),
                "r_multiple_net": float(net_r),
                "return_pct": float(risk_fraction * net_r),
            }
        )
        rows.append(row)
        if one_trade_at_a_time:
            active_until_index = max(active_until_index, int(outcome["exit_index"]))

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).reset_index(drop=True)


def label_trade_outcome(
    *,
    direction: str,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    close_times: list[pd.Timestamp],
    signal_index: int,
    entry_price: float,
    stop_price: float,
    target_price: float,
    horizon_bars: int,
) -> dict[str, Any] | None:
    risk = abs(entry_price - stop_price)
    if not math.isfinite(risk) or risk <= 0.0:
        return None
    target_r = abs(target_price - entry_price) / risk
    if not math.isfinite(target_r) or target_r <= 0.0:
        return None
    start_index = signal_index + 1
    final_index = min(len(closes) - 1, signal_index + int(horizon_bars))
    if start_index > final_index:
        return None

    entry_time = pd.Timestamp(close_times[signal_index]).tz_convert("UTC")
    mfe_r = 0.0
    mae_r = 0.0
    last_close_r = 0.0

    for cursor in range(start_index, final_index + 1):
        if direction == "long":
            mfe_r = max(mfe_r, (highs[cursor] - entry_price) / risk)
            mae_r = max(mae_r, (entry_price - lows[cursor]) / risk)
            target_hit = highs[cursor] >= target_price
            stop_hit = lows[cursor] <= stop_price
            if target_hit and stop_hit:
                target_first = high_before_low(opens[cursor], highs[cursor], lows[cursor])
                return realized_outcome(
                    target_first=target_first,
                    target_r=target_r,
                    cursor=cursor,
                    start_index=start_index,
                    close_times=close_times,
                    target_price=target_price,
                    stop_price=stop_price,
                    mfe_r=mfe_r,
                    mae_r=mae_r,
                )
            if target_hit:
                return hit_outcome("target", target_r, cursor, start_index, close_times, target_price, mfe_r, mae_r)
            if stop_hit:
                return hit_outcome("stop", -1.0, cursor, start_index, close_times, stop_price, mfe_r, mae_r)
            last_close_r = (closes[cursor] - entry_price) / risk
        else:
            mfe_r = max(mfe_r, (entry_price - lows[cursor]) / risk)
            mae_r = max(mae_r, (highs[cursor] - entry_price) / risk)
            target_hit = lows[cursor] <= target_price
            stop_hit = highs[cursor] >= stop_price
            if target_hit and stop_hit:
                target_first = not high_before_low(opens[cursor], highs[cursor], lows[cursor])
                return realized_outcome(
                    target_first=target_first,
                    target_r=target_r,
                    cursor=cursor,
                    start_index=start_index,
                    close_times=close_times,
                    target_price=target_price,
                    stop_price=stop_price,
                    mfe_r=mfe_r,
                    mae_r=mae_r,
                )
            if target_hit:
                return hit_outcome("target", target_r, cursor, start_index, close_times, target_price, mfe_r, mae_r)
            if stop_hit:
                return hit_outcome("stop", -1.0, cursor, start_index, close_times, stop_price, mfe_r, mae_r)
            last_close_r = (entry_price - closes[cursor]) / risk

    clipped_r = max(-1.0, min(float(target_r), float(last_close_r)))
    exit_price = closes[final_index]
    return {
        "future_r": float(clipped_r),
        "outcome": "timeout",
        "entry_time": entry_time,
        "bars_to_outcome": int(final_index - start_index + 1),
        "exit_time": pd.Timestamp(close_times[final_index]).tz_convert("UTC"),
        "exit_price": float(exit_price),
        "exit_index": int(final_index),
        "mfe_r": float(mfe_r),
        "mae_r": float(mae_r),
    }


def realized_outcome(
    *,
    target_first: bool,
    target_r: float,
    cursor: int,
    start_index: int,
    close_times: list[pd.Timestamp],
    target_price: float,
    stop_price: float,
    mfe_r: float,
    mae_r: float,
) -> dict[str, Any]:
    return hit_outcome(
        "target_same_bar" if target_first else "stop_same_bar",
        target_r if target_first else -1.0,
        cursor,
        start_index,
        close_times,
        target_price if target_first else stop_price,
        mfe_r,
        mae_r,
    )


def hit_outcome(
    reason: str,
    future_r: float,
    cursor: int,
    start_index: int,
    close_times: list[pd.Timestamp],
    exit_price: float,
    mfe_r: float,
    mae_r: float,
) -> dict[str, Any]:
    return {
        "future_r": float(future_r),
        "outcome": reason,
        "entry_time": pd.Timestamp(close_times[start_index - 1]).tz_convert("UTC"),
        "bars_to_outcome": int(cursor - start_index + 1),
        "exit_time": pd.Timestamp(close_times[cursor]).tz_convert("UTC"),
        "exit_price": float(exit_price),
        "exit_index": int(cursor),
        "mfe_r": float(mfe_r),
        "mae_r": float(mae_r),
    }


def objective_score(metrics: dict[str, float], objective: str, min_trades: int) -> float:
    if float(metrics.get("trades", 0.0)) < float(min_trades):
        return -float("inf")
    value = float(metrics.get(objective, 0.0))
    if objective == "profit_factor" and math.isinf(value):
        return 1_000.0
    if not math.isfinite(value):
        return -float("inf")
    return value


if __name__ == "__main__":
    main()
