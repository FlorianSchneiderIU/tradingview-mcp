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
from scripts.channel_state_research.production import load_production_config
from scripts.plot_zone_channel_history import build_bfm_magic_lines, parse_bfm_sets, parse_timeframes
from scripts.tune_bfm_support_resistance import LineBundle, Projection, project_lines_to_execution_frame


OPTIMIZED_BFM_TF_SETS = (
    "1h=330:220,264:176,211:141,169:112;"
    "4h=180:120,144:96,115:77,92:61;"
    "1d=105:70,84:56,67:45,54:36"
)


@dataclass(frozen=True)
class StrategySpec:
    entry_strategy: str
    lookback_bars: int
    channel_gap_atr: float
    min_reclaim_pos: float
    target_rr: float
    stop_buffer_atr: float
    min_sweep_depth_atr: float
    entry_window_bars: int
    confirm_window_bars: int
    max_hold_bars: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune Turtle Soup style entries using optimized BFM trendline channels as "
            "support/resistance features."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("scripts/channel_15m_broad_v2_full5y_config.json"))
    parser.add_argument("--start", default="2021-04-30")
    parser.add_argument("--end", default="2026-04-30")
    parser.add_argument("--cache-dir", type=Path, default=Path("scripts/.cache"))
    parser.add_argument("--exec-timeframe", default=None, help="Defaults to config decision_timeframe.")
    parser.add_argument("--line-timeframes", default="1h,4h,1d")
    parser.add_argument("--bfm-tf-sets", default=OPTIMIZED_BFM_TF_SETS)
    parser.add_argument("--bfm-invalidation", choices=["wick", "close", "none"], default="wick")
    parser.add_argument("--bfm-max-extension-bars", type=int, default=300)
    parser.add_argument("--entry-strategies", default="sweep_market,sweep_retest,sweep_momentum,line_reclaim")
    parser.add_argument("--directions", default="long,short", help="Comma-separated directions to allow: long,short.")
    parser.add_argument("--lookbacks", default="24,48,96,192", help="Execution bars for prior-liquidity sweeps.")
    parser.add_argument("--channel-gap-atrs", default="0.25,0.5,0.8")
    parser.add_argument("--min-reclaim-positions", default="0.45,0.6")
    parser.add_argument("--target-rrs", default="1.0,1.5,2.0")
    parser.add_argument("--stop-buffer-atrs", default="0.1,0.2")
    parser.add_argument("--min-sweep-depth-atrs", default="0.0,0.1")
    parser.add_argument("--entry-window-bars", default="8,16", help="Limit-retest validity windows.")
    parser.add_argument("--confirm-window-bars", default="8,16", help="Momentum confirmation windows.")
    parser.add_argument("--max-hold-bars", default="32,64,96")
    parser.add_argument("--fee-bps-side", type=float, default=None)
    parser.add_argument("--slippage-bps-side", type=float, default=None)
    parser.add_argument("--risk-fraction", type=float, default=None)
    parser.add_argument("--min-trades-for-score", type=int, default=50)
    parser.add_argument("--max-configs", type=int, default=0)
    parser.add_argument("--output-prefix", type=Path, default=Path("scripts/bfm_turtle_soup_full5y"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_production_config(args.config)
    exec_timeframe = args.exec_timeframe or config.decision_timeframe
    line_timeframes = parse_timeframes(args.line_timeframes, "1h")
    bfm_sets_by_tf = parse_tf_sets(args.bfm_tf_sets, line_timeframes)
    fee_bps_side = float(config.fee_bps_side if args.fee_bps_side is None else args.fee_bps_side)
    slippage_bps_side = float(config.slippage_bps_side if args.slippage_bps_side is None else args.slippage_bps_side)
    risk_fraction = float(config.risk.risk_fraction if args.risk_fraction is None else args.risk_fraction)

    print(
        f"Loading {config.symbol} {config.base_interval} data {args.start} -> {args.end}; "
        f"execution {exec_timeframe}; optimized BFM lines {','.join(line_timeframes)}"
    )
    base = load_base_candles(
        config.symbol,
        args.start,
        args.end,
        cache_dir=args.cache_dir,
        interval=config.base_interval,
    )
    all_timeframes = unique_preserve_order([exec_timeframe, *line_timeframes])
    bars_by_tf = {
        timeframe: prepare_timeframe_bars(base, timeframe, atr_length=config.atr_length)
        for timeframe in all_timeframes
    }
    exec_bars = bars_by_tf[exec_timeframe].reset_index(drop=True)
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
        print(f"{timeframe}: {len(pivots):,} pivots, {len(lines):,} lines, sets {format_sets(bfm_sets_by_tf[timeframe])}")

    projection = project_lines_to_execution_frame(exec_bars, bundles)
    specs = build_strategy_specs(args)
    if args.max_configs > 0:
        specs = specs[: int(args.max_configs)]
    print(f"Evaluating {len(specs):,} turtle/channel strategy variants")

    summaries: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_spec: StrategySpec | None = None
    best_trades = pd.DataFrame()
    best_candidates = pd.DataFrame()

    for index, spec in enumerate(specs, start=1):
        candidates = build_candidates(
            exec_bars=exec_bars,
            projection=projection,
            line_timeframes=line_timeframes,
            spec=spec,
            symbol=config.symbol,
            allowed_directions=set(parse_str_list(args.directions)),
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
                "config_index": index,
                "score": score,
                "raw_candidates": float(len(candidates)),
                "exec_timeframe": exec_timeframe,
                "line_timeframes": ",".join(line_timeframes),
                "fee_bps_side": fee_bps_side,
                "slippage_bps_side": slippage_bps_side,
                "risk_fraction": risk_fraction,
            }
        )
        row.update(metrics)
        summaries.append(row)
        if best_spec is None or score > best_score:
            best_score = score
            best_spec = spec
            best_trades = trades
            best_candidates = candidates
        if index == 1 or index % 50 == 0 or index == len(specs):
            print(
                f"[{index}/{len(specs)}] best {best_score:.4f}; latest "
                f"{int(metrics['trades'])} trades, return {metrics['total_return']:.2%}, "
                f"PF {metrics['profit_factor']:.2f}, netR {metrics['net_r']:.2f}"
            )

    summary = pd.DataFrame(summaries).sort_values(["score", "total_return", "net_r"], ascending=[False, False, False])
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_prefix.with_name(f"{args.output_prefix.name}_summary.csv")
    trades_path = args.output_prefix.with_name(f"{args.output_prefix.name}_best_trades.csv")
    candidates_path = args.output_prefix.with_name(f"{args.output_prefix.name}_best_candidates.csv")
    config_path = args.output_prefix.with_name(f"{args.output_prefix.name}_best_config.json")
    summary.to_csv(summary_path, index=False)
    best_trades.to_csv(trades_path, index=False)
    best_candidates.to_csv(candidates_path, index=False)
    best_payload = summary.iloc[0].to_dict() if not summary.empty else {}
    best_payload["bfm_tf_sets"] = {tf: format_sets(sets) for tf, sets in bfm_sets_by_tf.items()}
    config_path.write_text(json.dumps(best_payload, indent=2, sort_keys=True, default=str), encoding="utf-8")

    print("\nBest strategy")
    if not summary.empty:
        best = summary.iloc[0].to_dict()
        for key in [
            "entry_strategy",
            "lookback_bars",
            "channel_gap_atr",
            "min_reclaim_pos",
            "target_rr",
            "stop_buffer_atr",
            "min_sweep_depth_atr",
            "max_hold_bars",
            "trades",
            "total_return",
            "max_drawdown",
            "hit_rate",
            "profit_factor",
            "net_r",
            "sharpe",
        ]:
            print(f"  {key}: {best.get(key)}")
    print(f"\nWrote {summary_path}")
    print(f"Wrote {trades_path}")
    print(f"Wrote {candidates_path}")
    print(f"Wrote {config_path}")


def parse_tf_sets(raw: str, timeframes: list[str]) -> dict[str, list[tuple[int, int]]]:
    out: dict[str, list[tuple[int, int]]] = {}
    for chunk in str(raw).split(";"):
        text = chunk.strip()
        if not text:
            continue
        if "=" not in text:
            raise ValueError(f"Invalid timeframe set chunk {text!r}")
        timeframe, raw_sets = text.split("=", 1)
        out[timeframe.strip()] = parse_bfm_sets(raw_sets)
    missing = [timeframe for timeframe in timeframes if timeframe not in out]
    if missing:
        raise ValueError(f"Missing --bfm-tf-sets for {', '.join(missing)}")
    return out


def build_strategy_specs(args: argparse.Namespace) -> list[StrategySpec]:
    strategies = [item.strip() for item in str(args.entry_strategies).split(",") if item.strip()]
    lookbacks = parse_int_list(args.lookbacks)
    channel_gaps = parse_float_list(args.channel_gap_atrs)
    reclaims = parse_float_list(args.min_reclaim_positions)
    target_rrs = parse_float_list(args.target_rrs)
    stop_buffers = parse_float_list(args.stop_buffer_atrs)
    sweep_depths = parse_float_list(args.min_sweep_depth_atrs)
    entry_windows = parse_int_list(args.entry_window_bars)
    confirm_windows = parse_int_list(args.confirm_window_bars)
    hold_windows = parse_int_list(args.max_hold_bars)
    specs: list[StrategySpec] = []
    for entry_strategy, lookback, gap, reclaim, rr, stop_buffer, depth, entry_window, confirm_window, hold in itertools.product(
        strategies,
        lookbacks,
        channel_gaps,
        reclaims,
        target_rrs,
        stop_buffers,
        sweep_depths,
        entry_windows,
        confirm_windows,
        hold_windows,
    ):
        if entry_strategy == "line_reclaim" and lookback != lookbacks[0]:
            continue
        if entry_strategy != "sweep_retest" and entry_window != entry_windows[0]:
            continue
        if entry_strategy != "sweep_momentum" and confirm_window != confirm_windows[0]:
            continue
        specs.append(
            StrategySpec(
                entry_strategy=entry_strategy,
                lookback_bars=int(lookback),
                channel_gap_atr=float(gap),
                min_reclaim_pos=float(reclaim),
                target_rr=float(rr),
                stop_buffer_atr=float(stop_buffer),
                min_sweep_depth_atr=float(depth),
                entry_window_bars=int(entry_window),
                confirm_window_bars=int(confirm_window),
                max_hold_bars=int(hold),
            )
        )
    return specs


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in str(raw).split(",") if item.strip()]


def parse_str_list(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def unique_preserve_order(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def build_candidates(
    *,
    exec_bars: pd.DataFrame,
    projection: Projection,
    line_timeframes: list[str],
    spec: StrategySpec,
    symbol: str,
    allowed_directions: set[str],
) -> pd.DataFrame:
    opens = pd.to_numeric(exec_bars["open"], errors="coerce").to_numpy(dtype=float)
    highs = pd.to_numeric(exec_bars["high"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(exec_bars["low"], errors="coerce").to_numpy(dtype=float)
    closes = pd.to_numeric(exec_bars["close"], errors="coerce").to_numpy(dtype=float)
    atrs = pd.to_numeric(exec_bars["atr"], errors="coerce").to_numpy(dtype=float)
    times = pd.to_datetime(exec_bars["close_time"], utc=True, errors="coerce")
    n = len(exec_bars)
    rows: list[dict[str, Any]] = []

    ranges = highs - lows
    long_reclaim = np.divide(closes - lows, ranges, out=np.zeros_like(closes), where=ranges > 0.0)
    short_reclaim = np.divide(highs - closes, ranges, out=np.zeros_like(closes), where=ranges > 0.0)
    support_gap_atr = np.divide(projection.support_touch_gap, atrs, out=np.full(n, np.inf), where=atrs > 0.0)
    resistance_gap_atr = np.divide(projection.resistance_touch_gap, atrs, out=np.full(n, np.inf), where=atrs > 0.0)
    channel_long_ok = np.isfinite(projection.support_touch_value) & (support_gap_atr <= spec.channel_gap_atr)
    channel_short_ok = np.isfinite(projection.resistance_touch_value) & (resistance_gap_atr <= spec.channel_gap_atr)

    if spec.entry_strategy == "line_reclaim":
        long_events = (
            channel_long_ok
            & (lows < projection.support_touch_value)
            & (closes > projection.support_touch_value)
            & (long_reclaim >= spec.min_reclaim_pos)
        )
        short_events = (
            channel_short_ok
            & (highs > projection.resistance_touch_value)
            & (closes < projection.resistance_touch_value)
            & (short_reclaim >= spec.min_reclaim_pos)
        )
        if "long" in allowed_directions:
            for event_index in np.where(long_events)[0]:
                append_market_candidate(
                    rows,
                    symbol=symbol,
                    event_index=int(event_index),
                    entry_index=int(event_index) + 1,
                    times=times,
                    opens=opens,
                    highs=highs,
                    lows=lows,
                    closes=closes,
                    atrs=atrs,
                    direction="long",
                    strategy=spec.entry_strategy,
                    stop_anchor=lows[event_index],
                    target_rr=spec.target_rr,
                    stop_buffer_atr=spec.stop_buffer_atr,
                    max_hold_bars=spec.max_hold_bars,
                    liquidity_level=projection.support_touch_value[event_index],
                    channel_value=projection.support_touch_value[event_index],
                    channel_gap_atr=support_gap_atr[event_index],
                    channel_tf=tf_name(line_timeframes, int(projection.support_touch_tf[event_index])),
                    channel_set=int(projection.support_touch_set[event_index]),
                    reclaim_pos=long_reclaim[event_index],
                    sweep_depth_atr=max(0.0, (projection.support_touch_value[event_index] - lows[event_index]) / atrs[event_index]) if atrs[event_index] > 0 else math.nan,
                )
        if "short" in allowed_directions:
            for event_index in np.where(short_events)[0]:
                append_market_candidate(
                    rows,
                    symbol=symbol,
                    event_index=int(event_index),
                    entry_index=int(event_index) + 1,
                    times=times,
                    opens=opens,
                    highs=highs,
                    lows=lows,
                    closes=closes,
                    atrs=atrs,
                    direction="short",
                    strategy=spec.entry_strategy,
                    stop_anchor=highs[event_index],
                    target_rr=spec.target_rr,
                    stop_buffer_atr=spec.stop_buffer_atr,
                    max_hold_bars=spec.max_hold_bars,
                    liquidity_level=projection.resistance_touch_value[event_index],
                    channel_value=projection.resistance_touch_value[event_index],
                    channel_gap_atr=resistance_gap_atr[event_index],
                    channel_tf=tf_name(line_timeframes, int(projection.resistance_touch_tf[event_index])),
                    channel_set=int(projection.resistance_touch_set[event_index]),
                    reclaim_pos=short_reclaim[event_index],
                    sweep_depth_atr=max(0.0, (highs[event_index] - projection.resistance_touch_value[event_index]) / atrs[event_index]) if atrs[event_index] > 0 else math.nan,
                )
        return frame_candidates(rows)

    lookback = int(spec.lookback_bars)
    prior_low = pd.Series(lows).rolling(lookback, min_periods=lookback).min().shift(1).to_numpy(dtype=float)
    prior_high = pd.Series(highs).rolling(lookback, min_periods=lookback).max().shift(1).to_numpy(dtype=float)
    long_depth = np.divide(prior_low - lows, atrs, out=np.full(n, np.nan), where=atrs > 0.0)
    short_depth = np.divide(highs - prior_high, atrs, out=np.full(n, np.nan), where=atrs > 0.0)
    long_events = (
        np.isfinite(prior_low)
        & channel_long_ok
        & (long_depth >= spec.min_sweep_depth_atr)
        & (lows < prior_low)
        & (closes > prior_low)
        & (long_reclaim >= spec.min_reclaim_pos)
    )
    short_events = (
        np.isfinite(prior_high)
        & channel_short_ok
        & (short_depth >= spec.min_sweep_depth_atr)
        & (highs > prior_high)
        & (closes < prior_high)
        & (short_reclaim >= spec.min_reclaim_pos)
    )

    if "long" in allowed_directions:
        for event_index in np.where(long_events)[0]:
            append_entry_for_strategy(
                rows,
                spec=spec,
                symbol=symbol,
                event_index=int(event_index),
                times=times,
                opens=opens,
                highs=highs,
                lows=lows,
                closes=closes,
                atrs=atrs,
                direction="long",
                liquidity_level=prior_low[event_index],
                stop_anchor=lows[event_index],
                channel_value=projection.support_touch_value[event_index],
                channel_gap_atr=support_gap_atr[event_index],
                channel_tf=tf_name(line_timeframes, int(projection.support_touch_tf[event_index])),
                channel_set=int(projection.support_touch_set[event_index]),
                reclaim_pos=long_reclaim[event_index],
                sweep_depth_atr=long_depth[event_index],
            )
    if "short" in allowed_directions:
        for event_index in np.where(short_events)[0]:
            append_entry_for_strategy(
                rows,
                spec=spec,
                symbol=symbol,
                event_index=int(event_index),
                times=times,
                opens=opens,
                highs=highs,
                lows=lows,
                closes=closes,
                atrs=atrs,
                direction="short",
                liquidity_level=prior_high[event_index],
                stop_anchor=highs[event_index],
                channel_value=projection.resistance_touch_value[event_index],
                channel_gap_atr=resistance_gap_atr[event_index],
                channel_tf=tf_name(line_timeframes, int(projection.resistance_touch_tf[event_index])),
                channel_set=int(projection.resistance_touch_set[event_index]),
                reclaim_pos=short_reclaim[event_index],
                sweep_depth_atr=short_depth[event_index],
            )
    return frame_candidates(rows)


def append_entry_for_strategy(
    rows: list[dict[str, Any]],
    *,
    spec: StrategySpec,
    symbol: str,
    event_index: int,
    times: pd.Series,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    atrs: np.ndarray,
    direction: str,
    liquidity_level: float,
    stop_anchor: float,
    channel_value: float,
    channel_gap_atr: float,
    channel_tf: str,
    channel_set: int,
    reclaim_pos: float,
    sweep_depth_atr: float,
) -> None:
    if spec.entry_strategy == "sweep_market":
        append_market_candidate(
            rows,
            symbol=symbol,
            event_index=event_index,
            entry_index=event_index + 1,
            times=times,
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
            atrs=atrs,
            direction=direction,
            strategy=spec.entry_strategy,
            stop_anchor=stop_anchor,
            target_rr=spec.target_rr,
            stop_buffer_atr=spec.stop_buffer_atr,
            max_hold_bars=spec.max_hold_bars,
            liquidity_level=liquidity_level,
            channel_value=channel_value,
            channel_gap_atr=channel_gap_atr,
            channel_tf=channel_tf,
            channel_set=channel_set,
            reclaim_pos=reclaim_pos,
            sweep_depth_atr=sweep_depth_atr,
        )
        return
    if spec.entry_strategy == "sweep_retest":
        fill = find_limit_fill(
            direction=direction,
            limit_price=float(liquidity_level),
            start_index=event_index + 1,
            end_index=event_index + int(spec.entry_window_bars),
            highs=highs,
            lows=lows,
            opens=opens,
        )
        if fill is None:
            return
        entry_index, entry_price = fill
        append_candidate(
            rows,
            symbol=symbol,
            event_index=event_index,
            entry_index=entry_index,
            event_time=times.iloc[event_index],
            entry_time=times.iloc[entry_index],
            direction=direction,
            strategy=spec.entry_strategy,
            entry_price=entry_price,
            stop_anchor=stop_anchor,
            target_rr=spec.target_rr,
            stop_buffer_atr=spec.stop_buffer_atr,
            max_hold_bars=spec.max_hold_bars,
            atr=atrs[event_index],
            liquidity_level=liquidity_level,
            channel_value=channel_value,
            channel_gap_atr=channel_gap_atr,
            channel_tf=channel_tf,
            channel_set=channel_set,
            reclaim_pos=reclaim_pos,
            sweep_depth_atr=sweep_depth_atr,
            entry_delay_bars=entry_index - event_index,
        )
        return
    if spec.entry_strategy == "sweep_momentum":
        confirm = find_momentum_confirm(
            direction=direction,
            trigger_high=highs[event_index],
            trigger_low=lows[event_index],
            start_index=event_index + 1,
            end_index=event_index + int(spec.confirm_window_bars),
            closes=closes,
        )
        if confirm is None:
            return
        append_market_candidate(
            rows,
            symbol=symbol,
            event_index=event_index,
            entry_index=confirm + 1,
            times=times,
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
            atrs=atrs,
            direction=direction,
            strategy=spec.entry_strategy,
            stop_anchor=stop_anchor,
            target_rr=spec.target_rr,
            stop_buffer_atr=spec.stop_buffer_atr,
            max_hold_bars=spec.max_hold_bars,
            liquidity_level=liquidity_level,
            channel_value=channel_value,
            channel_gap_atr=channel_gap_atr,
            channel_tf=channel_tf,
            channel_set=channel_set,
            reclaim_pos=reclaim_pos,
            sweep_depth_atr=sweep_depth_atr,
            signal_index=confirm,
        )


def append_market_candidate(
    rows: list[dict[str, Any]],
    *,
    symbol: str,
    event_index: int,
    entry_index: int,
    times: pd.Series,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    atrs: np.ndarray,
    direction: str,
    strategy: str,
    stop_anchor: float,
    target_rr: float,
    stop_buffer_atr: float,
    max_hold_bars: int,
    liquidity_level: float,
    channel_value: float,
    channel_gap_atr: float,
    channel_tf: str,
    channel_set: int,
    reclaim_pos: float,
    sweep_depth_atr: float,
    signal_index: int | None = None,
) -> None:
    del highs, lows, closes
    if entry_index >= len(opens):
        return
    append_candidate(
        rows,
        symbol=symbol,
        event_index=event_index,
        entry_index=entry_index,
        event_time=times.iloc[event_index],
        entry_time=times.iloc[entry_index],
        direction=direction,
        strategy=strategy,
        entry_price=float(opens[entry_index]),
        stop_anchor=stop_anchor,
        target_rr=target_rr,
        stop_buffer_atr=stop_buffer_atr,
        max_hold_bars=max_hold_bars,
        atr=atrs[event_index],
        liquidity_level=liquidity_level,
        channel_value=channel_value,
        channel_gap_atr=channel_gap_atr,
        channel_tf=channel_tf,
        channel_set=channel_set,
        reclaim_pos=reclaim_pos,
        sweep_depth_atr=sweep_depth_atr,
        entry_delay_bars=entry_index - event_index,
        signal_index=signal_index if signal_index is not None else event_index,
    )


def append_candidate(
    rows: list[dict[str, Any]],
    *,
    symbol: str,
    event_index: int,
    entry_index: int,
    event_time: pd.Timestamp,
    entry_time: pd.Timestamp,
    direction: str,
    strategy: str,
    entry_price: float,
    stop_anchor: float,
    target_rr: float,
    stop_buffer_atr: float,
    max_hold_bars: int,
    atr: float,
    liquidity_level: float,
    channel_value: float,
    channel_gap_atr: float,
    channel_tf: str,
    channel_set: int,
    reclaim_pos: float,
    sweep_depth_atr: float,
    entry_delay_bars: int,
    signal_index: int | None = None,
) -> None:
    if not all(math.isfinite(float(value)) for value in [entry_price, stop_anchor, target_rr, stop_buffer_atr, atr]):
        return
    if atr <= 0.0 or target_rr <= 0.0:
        return
    if direction == "long":
        stop_price = float(stop_anchor) - stop_buffer_atr * atr
        risk = entry_price - stop_price
        target_price = entry_price + target_rr * risk
    else:
        stop_price = float(stop_anchor) + stop_buffer_atr * atr
        risk = stop_price - entry_price
        target_price = entry_price - target_rr * risk
    if not math.isfinite(risk) or risk <= 0.0:
        return
    rows.append(
        {
            "symbol": symbol,
            "entry_strategy": strategy,
            "event_index": int(event_index),
            "signal_index": int(signal_index if signal_index is not None else event_index),
            "entry_index": int(entry_index),
            "event_time": pd.Timestamp(event_time).tz_convert("UTC"),
            "entry_time": pd.Timestamp(entry_time).tz_convert("UTC"),
            "direction": direction,
            "entry_price": float(entry_price),
            "stop_price": float(stop_price),
            "target_price": float(target_price),
            "risk_abs": float(risk),
            "target_rr_planned": float(target_rr),
            "max_hold_bars": int(max_hold_bars),
            "liquidity_level": float(liquidity_level),
            "channel_value": float(channel_value),
            "channel_gap_atr": float(channel_gap_atr),
            "channel_tf": channel_tf,
            "channel_set": int(channel_set),
            "reclaim_pos": float(reclaim_pos),
            "sweep_depth_atr": float(sweep_depth_atr),
            "entry_delay_bars": int(entry_delay_bars),
        }
    )


def find_limit_fill(
    *,
    direction: str,
    limit_price: float,
    start_index: int,
    end_index: int,
    highs: np.ndarray,
    lows: np.ndarray,
    opens: np.ndarray,
) -> tuple[int, float] | None:
    final = min(len(opens) - 1, int(end_index))
    for index in range(start_index, final + 1):
        if direction == "long":
            if opens[index] <= limit_price:
                return index, float(opens[index])
            if lows[index] <= limit_price <= highs[index]:
                return index, float(limit_price)
        else:
            if opens[index] >= limit_price:
                return index, float(opens[index])
            if lows[index] <= limit_price <= highs[index]:
                return index, float(limit_price)
    return None


def find_momentum_confirm(
    *,
    direction: str,
    trigger_high: float,
    trigger_low: float,
    start_index: int,
    end_index: int,
    closes: np.ndarray,
) -> int | None:
    final = min(len(closes) - 2, int(end_index))
    for index in range(start_index, final + 1):
        if direction == "long" and closes[index] > trigger_high:
            return index
        if direction == "short" and closes[index] < trigger_low:
            return index
    return None


def frame_candidates(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(["entry_index", "channel_gap_atr", "sweep_depth_atr"], ascending=[True, True, False])
        .reset_index(drop=True)
    )


def label_and_schedule(
    candidates: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    fee_bps_side: float,
    slippage_bps_side: float,
    risk_fraction: float,
) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    opens = pd.to_numeric(bars["open"], errors="coerce").to_numpy(dtype=float)
    highs = pd.to_numeric(bars["high"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float)
    closes = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float)
    close_times = pd.to_datetime(bars["close_time"], utc=True, errors="coerce").to_list()

    rows: list[dict[str, Any]] = []
    active_until = -1
    ordered = candidates.sort_values(["entry_index", "channel_gap_atr", "sweep_depth_atr"], ascending=[True, True, False])
    for _, candidate in ordered.iterrows():
        entry_index = int(candidate["entry_index"])
        if entry_index <= active_until:
            continue
        max_hold_bars = int(candidate.get("max_hold_bars", candidate.get("hold_bars", 0)) or 0)
        if max_hold_bars <= 0:
            max_hold_bars = 64
        outcome = label_trade(
            direction=str(candidate["direction"]),
            entry_index=entry_index,
            entry_price=float(candidate["entry_price"]),
            stop_price=float(candidate["stop_price"]),
            target_price=float(candidate["target_price"]),
            max_hold_bars=max_hold_bars,
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
            close_times=close_times,
        )
        if outcome is None:
            continue
        risk = float(candidate["risk_abs"])
        cost_r = ((2.0 * fee_bps_side) + (2.0 * slippage_bps_side)) / 10_000.0 * float(candidate["entry_price"]) / risk
        gross_r = float(outcome["r_multiple_gross"])
        net_r = gross_r - cost_r
        row = candidate.to_dict()
        row.update(outcome)
        row["cost_r"] = float(cost_r)
        row["r_multiple_net"] = float(net_r)
        row["return_pct"] = float(risk_fraction * net_r)
        rows.append(row)
        active_until = max(active_until, int(outcome["exit_index"]))
    return pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame()


def label_trade(
    *,
    direction: str,
    entry_index: int,
    entry_price: float,
    stop_price: float,
    target_price: float,
    max_hold_bars: int,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    close_times: list[pd.Timestamp],
) -> dict[str, Any] | None:
    risk = abs(entry_price - stop_price)
    if not math.isfinite(risk) or risk <= 0.0:
        return None
    final_index = min(len(closes) - 1, entry_index + int(max_hold_bars) - 1)
    if entry_index > final_index:
        return None
    mfe_r = 0.0
    mae_r = 0.0
    for index in range(entry_index, final_index + 1):
        if direction == "long":
            mfe_r = max(mfe_r, (highs[index] - entry_price) / risk)
            mae_r = max(mae_r, (entry_price - lows[index]) / risk)
            target_hit = highs[index] >= target_price
            stop_hit = lows[index] <= stop_price
            if target_hit and stop_hit:
                target_first = high_before_low(opens[index], highs[index], lows[index])
                return trade_outcome(index, entry_index, close_times, target_price if target_first else stop_price, target_first, "target_same_bar" if target_first else "stop_same_bar", mfe_r, mae_r, target_rr=(target_price - entry_price) / risk)
            if target_hit:
                return trade_outcome(index, entry_index, close_times, target_price, True, "target", mfe_r, mae_r, target_rr=(target_price - entry_price) / risk)
            if stop_hit:
                return trade_outcome(index, entry_index, close_times, stop_price, False, "stop", mfe_r, mae_r, target_rr=(target_price - entry_price) / risk)
        else:
            mfe_r = max(mfe_r, (entry_price - lows[index]) / risk)
            mae_r = max(mae_r, (highs[index] - entry_price) / risk)
            target_hit = lows[index] <= target_price
            stop_hit = highs[index] >= stop_price
            if target_hit and stop_hit:
                target_first = not high_before_low(opens[index], highs[index], lows[index])
                return trade_outcome(index, entry_index, close_times, target_price if target_first else stop_price, target_first, "target_same_bar" if target_first else "stop_same_bar", mfe_r, mae_r, target_rr=(entry_price - target_price) / risk)
            if target_hit:
                return trade_outcome(index, entry_index, close_times, target_price, True, "target", mfe_r, mae_r, target_rr=(entry_price - target_price) / risk)
            if stop_hit:
                return trade_outcome(index, entry_index, close_times, stop_price, False, "stop", mfe_r, mae_r, target_rr=(entry_price - target_price) / risk)
    if direction == "long":
        gross_r = (closes[final_index] - entry_price) / risk
    else:
        gross_r = (entry_price - closes[final_index]) / risk
    return {
        "exit_index": int(final_index),
        "exit_time": pd.Timestamp(close_times[final_index]).tz_convert("UTC"),
        "exit_price": float(closes[final_index]),
        "exit_reason": "time",
        "hold_bars": int(final_index - entry_index + 1),
        "r_multiple_gross": float(gross_r),
        "mfe_r": float(mfe_r),
        "mae_r": float(mae_r),
    }


def trade_outcome(
    index: int,
    entry_index: int,
    close_times: list[pd.Timestamp],
    exit_price: float,
    target_first: bool,
    reason: str,
    mfe_r: float,
    mae_r: float,
    *,
    target_rr: float,
) -> dict[str, Any]:
    return {
        "exit_index": int(index),
        "exit_time": pd.Timestamp(close_times[index]).tz_convert("UTC"),
        "exit_price": float(exit_price),
        "exit_reason": reason,
        "hold_bars": int(index - entry_index + 1),
        "r_multiple_gross": float(target_rr if target_first else -1.0),
        "mfe_r": float(mfe_r),
        "mae_r": float(mae_r),
    }


def score_metrics(metrics: dict[str, float], min_trades: int) -> float:
    trades = float(metrics.get("trades", 0.0))
    if trades < min_trades:
        return -float("inf")
    total_return = float(metrics.get("total_return", 0.0))
    drawdown = abs(float(metrics.get("max_drawdown", 0.0)))
    profit_factor = float(metrics.get("profit_factor", 0.0))
    if not math.isfinite(profit_factor):
        profit_factor = 5.0
    sharpe = float(metrics.get("sharpe", 0.0))
    if not all(math.isfinite(value) for value in [total_return, drawdown, profit_factor, sharpe]):
        return -float("inf")
    return total_return - 0.35 * drawdown + 0.02 * min(profit_factor, 5.0) + 0.01 * sharpe


def spec_row(spec: StrategySpec) -> dict[str, Any]:
    return {
        "entry_strategy": spec.entry_strategy,
        "lookback_bars": spec.lookback_bars,
        "channel_gap_atr": spec.channel_gap_atr,
        "min_reclaim_pos": spec.min_reclaim_pos,
        "target_rr": spec.target_rr,
        "stop_buffer_atr": spec.stop_buffer_atr,
        "min_sweep_depth_atr": spec.min_sweep_depth_atr,
        "entry_window_bars": spec.entry_window_bars,
        "confirm_window_bars": spec.confirm_window_bars,
        "max_hold_bars": spec.max_hold_bars,
    }


def tf_name(timeframes: list[str], index: int) -> str:
    if index < 0 or index >= len(timeframes):
        return ""
    return timeframes[index]


def format_sets(sets: list[tuple[int, int]]) -> str:
    return ",".join(f"{left}:{right}" for left, right in sets)


if __name__ == "__main__":
    main()
