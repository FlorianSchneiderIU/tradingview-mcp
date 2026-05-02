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
from scripts.channel_state_research.production import load_production_config
from scripts.plot_zone_channel_history import build_bfm_magic_lines, parse_timeframes
from scripts.tune_bfm_support_resistance import LineBundle, Projection, project_lines_to_execution_frame
from scripts.tune_bfm_turtle_soup import (
    OPTIMIZED_BFM_TF_SETS,
    append_candidate,
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
class ScalpSpec:
    entry_strategy: str
    lookback_bars: int
    channel_gap_atr: float
    min_reclaim_pos: float
    target_rr: float
    stop_buffer_atr: float
    min_sweep_depth_atr: float
    min_risk_atr: float
    max_risk_atr: float
    stop_lookback_bars: int
    max_hold_bars: int
    trend_filter: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune lower-timeframe BTC scalp entries that use optimized BFM support/resistance "
            "lines as higher-timeframe features."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("scripts/channel_15m_broad_v2_full5y_config.json"))
    parser.add_argument("--start", default="2021-04-30")
    parser.add_argument("--end", default="2026-04-30")
    parser.add_argument("--cache-dir", type=Path, default=Path("scripts/.cache"))
    parser.add_argument("--exec-timeframe", default="5m")
    parser.add_argument("--line-timeframes", default="1h,4h,1d")
    parser.add_argument("--bfm-tf-sets", default=OPTIMIZED_BFM_TF_SETS)
    parser.add_argument("--bfm-invalidation", choices=["wick", "close", "none"], default="wick")
    parser.add_argument("--bfm-max-extension-bars", type=int, default=300)
    parser.add_argument(
        "--entry-strategies",
        default="hybrid_reclaim,support_tap,support_reclaim,micro_turtle,ema_reclaim,vwap_reclaim",
        help=(
            "Comma-separated trigger families. support_tap is the loosest BFM-zone touch; "
            "support_reclaim requires reclaiming the BFM line; micro_turtle sweeps a local "
            "Donchian level near the line; ema_reclaim/vwap_reclaim add 5m scalp structure; "
            "hybrid_reclaim schedules support_reclaim, ema_reclaim, and micro_turtle together."
        ),
    )
    parser.add_argument("--directions", default="long", help="Comma-separated directions: long,short.")
    parser.add_argument("--lookbacks", default="6,12,24,36", help="Execution bars for micro Turtle Soup sweeps.")
    parser.add_argument("--channel-gap-atrs", default="0.75,1.0,1.5,2.0")
    parser.add_argument("--min-reclaim-positions", default="0.35,0.5")
    parser.add_argument("--target-rrs", default="0.75,1.0,1.5,2.0")
    parser.add_argument("--stop-buffer-atrs", default="0.08,0.12,0.2")
    parser.add_argument("--min-sweep-depth-atrs", default="0.0,0.05")
    parser.add_argument("--min-risk-atrs", default="0.0,1.5,2.5")
    parser.add_argument("--max-risk-atrs", default="2.0,4.0,6.0")
    parser.add_argument("--stop-lookbacks", default="1,6,12", help="Execution bars used for local structure stops.")
    parser.add_argument("--max-hold-bars", default="24,48,96")
    parser.add_argument(
        "--trend-filters",
        default=(
            "none,ema20_up,close_above_ema50,close_above_vwap,ema20_above_ema50,"
            "daily_above_ema50,daily_above_ema100,daily_above_ema200,4h_above_ema200,htf_bull"
        ),
    )
    parser.add_argument("--fee-bps-side", type=float, default=None)
    parser.add_argument("--slippage-bps-side", type=float, default=None)
    parser.add_argument("--risk-fraction", type=float, default=None)
    parser.add_argument("--min-trades-for-score", type=int, default=250)
    parser.add_argument("--max-configs", type=int, default=0)
    parser.add_argument("--output-prefix", type=Path, default=Path("scripts/bfm_scalper_full5y"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_production_config(args.config)
    line_timeframes = parse_timeframes(args.line_timeframes, "1h")
    bfm_sets_by_tf = parse_tf_sets(args.bfm_tf_sets, line_timeframes)
    allowed_directions = set(parse_str_list(args.directions))
    fee_bps_side = float(config.fee_bps_side if args.fee_bps_side is None else args.fee_bps_side)
    slippage_bps_side = float(config.slippage_bps_side if args.slippage_bps_side is None else args.slippage_bps_side)
    risk_fraction = float(config.risk.risk_fraction if args.risk_fraction is None else args.risk_fraction)

    print(
        f"Loading {config.symbol} {config.base_interval} data {args.start} -> {args.end}; "
        f"execution {args.exec_timeframe}; BFM context {','.join(line_timeframes)}"
    )
    base = load_base_candles(
        config.symbol,
        args.start,
        args.end,
        cache_dir=args.cache_dir,
        interval=config.base_interval,
    )
    all_timeframes = unique_preserve_order([args.exec_timeframe, *line_timeframes])
    bars_by_tf = {
        timeframe: prepare_timeframe_bars(base, timeframe, atr_length=config.atr_length)
        for timeframe in all_timeframes
    }
    exec_bars = bars_by_tf[args.exec_timeframe].reset_index(drop=True)

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
    features = build_execution_features(exec_bars, bars_by_tf=bars_by_tf)
    specs = build_specs(args)
    if args.max_configs > 0:
        specs = specs[: int(args.max_configs)]
    print(f"Evaluating {len(specs):,} lower-timeframe BFM scalp variants")

    summaries: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_rank = (False, -float("inf"), -float("inf"), -float("inf"))
    best_trades = pd.DataFrame()
    best_candidates = pd.DataFrame()

    for index, spec in enumerate(specs, start=1):
        candidates = build_candidates(
            exec_bars=exec_bars,
            projection=projection,
            features=features,
            line_timeframes=line_timeframes,
            spec=spec,
            symbol=config.symbol,
            allowed_directions=allowed_directions,
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
                "exec_timeframe": args.exec_timeframe,
                "line_timeframes": ",".join(line_timeframes),
                "fee_bps_side": fee_bps_side,
                "slippage_bps_side": slippage_bps_side,
                "risk_fraction": risk_fraction,
            }
        )
        row.update(metrics)
        summaries.append(row)
        rank = (
            math.isfinite(score),
            float(score),
            float(metrics.get("total_return", 0.0)),
            float(metrics.get("net_r", 0.0)),
        )
        if rank > best_rank:
            best_rank = rank
            best_score = score
            best_trades = trades
            best_candidates = candidates
        if index == 1 or index % 100 == 0 or index == len(specs):
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
            "min_risk_atr",
            "max_risk_atr",
            "stop_lookback_bars",
            "max_hold_bars",
            "trend_filter",
            "raw_candidates",
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


def build_specs(args: argparse.Namespace) -> list[ScalpSpec]:
    strategies = parse_str_list(args.entry_strategies)
    lookbacks = parse_int_list(args.lookbacks)
    channel_gaps = parse_float_list(args.channel_gap_atrs)
    reclaims = parse_float_list(args.min_reclaim_positions)
    target_rrs = parse_float_list(args.target_rrs)
    stop_buffers = parse_float_list(args.stop_buffer_atrs)
    sweep_depths = parse_float_list(args.min_sweep_depth_atrs)
    min_risks = parse_float_list(args.min_risk_atrs)
    max_risks = parse_float_list(args.max_risk_atrs)
    stop_lookbacks = parse_int_list(args.stop_lookbacks)
    hold_windows = parse_int_list(args.max_hold_bars)
    trend_filters = parse_str_list(args.trend_filters)

    specs: list[ScalpSpec] = []
    for strategy, lookback, gap, reclaim, rr, stop_buffer, depth, min_risk, max_risk, stop_lookback, hold, trend_filter in itertools.product(
        strategies,
        lookbacks,
        channel_gaps,
        reclaims,
        target_rrs,
        stop_buffers,
        sweep_depths,
        min_risks,
        max_risks,
        stop_lookbacks,
        hold_windows,
        trend_filters,
    ):
        if min_risk > max_risk:
            continue
        if strategy not in {"micro_turtle", "hybrid_reclaim"} and lookback != lookbacks[0]:
            continue
        if strategy not in {"micro_turtle", "support_reclaim", "hybrid_reclaim"} and depth != sweep_depths[0]:
            continue
        specs.append(
            ScalpSpec(
                entry_strategy=strategy,
                lookback_bars=int(lookback),
                channel_gap_atr=float(gap),
                min_reclaim_pos=float(reclaim),
                target_rr=float(rr),
                stop_buffer_atr=float(stop_buffer),
                min_sweep_depth_atr=float(depth),
                min_risk_atr=float(min_risk),
                max_risk_atr=float(max_risk),
                stop_lookback_bars=int(stop_lookback),
                max_hold_bars=int(hold),
                trend_filter=trend_filter,
            )
        )
    return specs


def build_execution_features(bars: pd.DataFrame, *, bars_by_tf: dict[str, pd.DataFrame] | None = None) -> dict[str, np.ndarray]:
    opens = pd.to_numeric(bars["open"], errors="coerce")
    highs = pd.to_numeric(bars["high"], errors="coerce")
    lows = pd.to_numeric(bars["low"], errors="coerce")
    closes = pd.to_numeric(bars["close"], errors="coerce")
    volumes = pd.to_numeric(bars.get("volume", pd.Series(np.ones(len(bars)))), errors="coerce").fillna(0.0)

    ema20 = closes.ewm(span=20, adjust=False, min_periods=20).mean().to_numpy(dtype=float)
    ema50 = closes.ewm(span=50, adjust=False, min_periods=50).mean().to_numpy(dtype=float)
    ema200 = closes.ewm(span=200, adjust=False, min_periods=200).mean().to_numpy(dtype=float)
    close_arr = closes.to_numpy(dtype=float)
    times = pd.to_datetime(bars["close_time"], utc=True, errors="coerce")
    session = times.dt.floor("D")
    typical = ((highs + lows + closes) / 3.0).to_numpy(dtype=float)
    pv = pd.Series(typical * volumes.to_numpy(dtype=float), index=bars.index)
    cum_pv = pv.groupby(session).cumsum()
    cum_volume = volumes.groupby(session).cumsum()
    vwap = np.divide(
        cum_pv.to_numpy(dtype=float),
        cum_volume.to_numpy(dtype=float),
        out=np.full(len(bars), np.nan, dtype=float),
        where=cum_volume.to_numpy(dtype=float) > 0.0,
    )
    ranges = (highs - lows).to_numpy(dtype=float)
    long_reclaim = np.divide(
        (closes - lows).to_numpy(dtype=float),
        ranges,
        out=np.zeros(len(bars), dtype=float),
        where=ranges > 0.0,
    )
    short_reclaim = np.divide(
        (highs - closes).to_numpy(dtype=float),
        ranges,
        out=np.zeros(len(bars), dtype=float),
        where=ranges > 0.0,
    )
    features = {
        "open": opens.to_numpy(dtype=float),
        "high": highs.to_numpy(dtype=float),
        "low": lows.to_numpy(dtype=float),
        "close": close_arr,
        "ema20": ema20,
        "ema50": ema50,
        "ema200": ema200,
        "vwap": vwap,
        "long_reclaim": long_reclaim,
        "short_reclaim": short_reclaim,
        "ema20_delta_6": ema20 - np.roll(ema20, 6),
        "ema50_delta_12": ema50 - np.roll(ema50, 12),
        "ret_6": close_arr - np.roll(close_arr, 6),
    }
    for key in ["ema20_delta_6", "ema50_delta_12", "ret_6"]:
        features[key][:12] = np.nan
    if bars_by_tf:
        add_higher_timeframe_features(features, bars, bars_by_tf)
    return features


def add_higher_timeframe_features(
    features: dict[str, np.ndarray],
    exec_bars: pd.DataFrame,
    bars_by_tf: dict[str, pd.DataFrame],
) -> None:
    exec_times = pd.to_datetime(exec_bars["close_time"], utc=True, errors="coerce").to_numpy(dtype="datetime64[ns]").astype("int64")
    for timeframe, spans in {"4h": (50, 100, 200), "1d": (50, 100, 200)}.items():
        htf = bars_by_tf.get(timeframe)
        if htf is None or htf.empty:
            continue
        htf_times = pd.to_datetime(htf["close_time"], utc=True, errors="coerce").to_numpy(dtype="datetime64[ns]").astype("int64")
        htf_close = pd.to_numeric(htf["close"], errors="coerce")
        indices = np.searchsorted(htf_times, exec_times, side="right") - 1
        valid = (indices >= 0) & (indices < len(htf_close))
        close_projected = np.full(len(exec_bars), np.nan, dtype=float)
        close_values = htf_close.to_numpy(dtype=float)
        close_projected[valid] = close_values[indices[valid]]
        prefix = "daily" if timeframe == "1d" else "h4"
        features[f"{prefix}_close"] = close_projected
        for span in spans:
            ema = htf_close.ewm(span=span, adjust=False, min_periods=max(5, min(span, 50))).mean().to_numpy(dtype=float)
            projected = np.full(len(exec_bars), np.nan, dtype=float)
            projected[valid] = ema[indices[valid]]
            features[f"{prefix}_ema{span}"] = projected


def build_candidates(
    *,
    exec_bars: pd.DataFrame,
    projection: Projection,
    features: dict[str, np.ndarray],
    line_timeframes: list[str],
    spec: ScalpSpec,
    symbol: str,
    allowed_directions: set[str],
) -> pd.DataFrame:
    if spec.entry_strategy == "hybrid_reclaim":
        frames: list[pd.DataFrame] = []
        for trigger_family in ["support_reclaim", "ema_reclaim", "micro_turtle"]:
            family_spec = ScalpSpec(
                entry_strategy=trigger_family,
                lookback_bars=spec.lookback_bars,
                channel_gap_atr=spec.channel_gap_atr,
                min_reclaim_pos=spec.min_reclaim_pos,
                target_rr=spec.target_rr,
                stop_buffer_atr=spec.stop_buffer_atr,
                min_sweep_depth_atr=spec.min_sweep_depth_atr,
                min_risk_atr=spec.min_risk_atr,
                max_risk_atr=spec.max_risk_atr,
                stop_lookback_bars=spec.stop_lookback_bars,
                max_hold_bars=spec.max_hold_bars,
                trend_filter=spec.trend_filter,
            )
            family_candidates = build_candidates(
                exec_bars=exec_bars,
                projection=projection,
                features=features,
                line_timeframes=line_timeframes,
                spec=family_spec,
                symbol=symbol,
                allowed_directions=allowed_directions,
            )
            if not family_candidates.empty:
                family_candidates = family_candidates.copy()
                family_candidates["trigger_family"] = trigger_family
                family_candidates["entry_strategy"] = spec.entry_strategy
                frames.append(family_candidates)
        if not frames:
            return pd.DataFrame()
        return sort_candidate_frame(pd.concat(frames, ignore_index=True))

    opens = features["open"]
    highs = features["high"]
    lows = features["low"]
    closes = features["close"]
    atrs = pd.to_numeric(exec_bars["atr"], errors="coerce").to_numpy(dtype=float)
    times = pd.to_datetime(exec_bars["close_time"], utc=True, errors="coerce")
    n = len(exec_bars)
    if n < 3:
        return pd.DataFrame()

    support_gap_atr = np.divide(
        projection.support_touch_gap,
        atrs,
        out=np.full(n, np.inf, dtype=float),
        where=atrs > 0.0,
    )
    resistance_gap_atr = np.divide(
        projection.resistance_touch_gap,
        atrs,
        out=np.full(n, np.inf, dtype=float),
        where=atrs > 0.0,
    )
    channel_long_ok = (
        np.isfinite(projection.support_touch_value)
        & np.isfinite(support_gap_atr)
        & (support_gap_atr <= spec.channel_gap_atr)
        & (atrs > 0.0)
    )
    channel_short_ok = (
        np.isfinite(projection.resistance_touch_value)
        & np.isfinite(resistance_gap_atr)
        & (resistance_gap_atr <= spec.channel_gap_atr)
        & (atrs > 0.0)
    )

    long_trend = trend_mask(spec.trend_filter, "long", features)
    short_trend = trend_mask(spec.trend_filter, "short", features)
    rows: list[dict[str, Any]] = []
    stop_lookback = max(1, int(spec.stop_lookback_bars))
    stop_lows = pd.Series(lows).rolling(stop_lookback, min_periods=1).min().to_numpy(dtype=float)
    stop_highs = pd.Series(highs).rolling(stop_lookback, min_periods=1).max().to_numpy(dtype=float)

    if spec.entry_strategy == "micro_turtle":
        lookback = int(spec.lookback_bars)
        prior_low = pd.Series(lows).rolling(lookback, min_periods=lookback).min().shift(1).to_numpy(dtype=float)
        prior_high = pd.Series(highs).rolling(lookback, min_periods=lookback).max().shift(1).to_numpy(dtype=float)
        long_depth = np.divide(prior_low - lows, atrs, out=np.full(n, np.nan), where=atrs > 0.0)
        short_depth = np.divide(highs - prior_high, atrs, out=np.full(n, np.nan), where=atrs > 0.0)
        if "long" in allowed_directions:
            long_events = (
                channel_long_ok
                & long_trend
                & np.isfinite(prior_low)
                & (lows < prior_low)
                & (closes > prior_low)
                & (features["long_reclaim"] >= spec.min_reclaim_pos)
                & (long_depth >= spec.min_sweep_depth_atr)
            )
            for event_index in np.where(long_events)[0]:
                append_scalp_candidate(
                    rows,
                    spec=spec,
                    symbol=symbol,
                    event_index=int(event_index),
                    times=times,
                    opens=opens,
                    lows=lows,
                    highs=highs,
                    atrs=atrs,
                    direction="long",
                    liquidity_level=prior_low[event_index],
                    stop_anchor=stop_lows[event_index],
                    channel_value=projection.support_touch_value[event_index],
                    channel_gap_atr=support_gap_atr[event_index],
                    channel_tf=tf_name(line_timeframes, int(projection.support_touch_tf[event_index])),
                    channel_set=int(projection.support_touch_set[event_index]),
                    reclaim_pos=features["long_reclaim"][event_index],
                    sweep_depth_atr=long_depth[event_index],
                    features=features,
                )
        if "short" in allowed_directions:
            short_events = (
                channel_short_ok
                & short_trend
                & np.isfinite(prior_high)
                & (highs > prior_high)
                & (closes < prior_high)
                & (features["short_reclaim"] >= spec.min_reclaim_pos)
                & (short_depth >= spec.min_sweep_depth_atr)
            )
            for event_index in np.where(short_events)[0]:
                append_scalp_candidate(
                    rows,
                    spec=spec,
                    symbol=symbol,
                    event_index=int(event_index),
                    times=times,
                    opens=opens,
                    lows=lows,
                    highs=highs,
                    atrs=atrs,
                    direction="short",
                    liquidity_level=prior_high[event_index],
                    stop_anchor=stop_highs[event_index],
                    channel_value=projection.resistance_touch_value[event_index],
                    channel_gap_atr=resistance_gap_atr[event_index],
                    channel_tf=tf_name(line_timeframes, int(projection.resistance_touch_tf[event_index])),
                    channel_set=int(projection.resistance_touch_set[event_index]),
                    reclaim_pos=features["short_reclaim"][event_index],
                    sweep_depth_atr=short_depth[event_index],
                    features=features,
                )
        return frame_candidates(rows)

    if spec.entry_strategy == "support_tap":
        long_events = channel_long_ok & long_trend & (features["long_reclaim"] >= spec.min_reclaim_pos) & (closes > opens)
        short_events = channel_short_ok & short_trend & (features["short_reclaim"] >= spec.min_reclaim_pos) & (closes < opens)
    elif spec.entry_strategy == "support_reclaim":
        long_line_depth = np.divide(
            projection.support_touch_value - lows,
            atrs,
            out=np.full(n, np.nan),
            where=atrs > 0.0,
        )
        short_line_depth = np.divide(
            highs - projection.resistance_touch_value,
            atrs,
            out=np.full(n, np.nan),
            where=atrs > 0.0,
        )
        long_events = (
            channel_long_ok
            & long_trend
            & (lows <= projection.support_touch_value)
            & (closes > projection.support_touch_value)
            & (features["long_reclaim"] >= spec.min_reclaim_pos)
            & (long_line_depth >= spec.min_sweep_depth_atr)
        )
        short_events = (
            channel_short_ok
            & short_trend
            & (highs >= projection.resistance_touch_value)
            & (closes < projection.resistance_touch_value)
            & (features["short_reclaim"] >= spec.min_reclaim_pos)
            & (short_line_depth >= spec.min_sweep_depth_atr)
        )
    elif spec.entry_strategy == "ema_reclaim":
        ema20 = features["ema20"]
        long_events = (
            channel_long_ok
            & long_trend
            & np.isfinite(ema20)
            & (lows < ema20)
            & (closes > ema20)
            & (features["long_reclaim"] >= spec.min_reclaim_pos)
        )
        short_events = (
            channel_short_ok
            & short_trend
            & np.isfinite(ema20)
            & (highs > ema20)
            & (closes < ema20)
            & (features["short_reclaim"] >= spec.min_reclaim_pos)
        )
    elif spec.entry_strategy == "vwap_reclaim":
        vwap = features["vwap"]
        long_events = (
            channel_long_ok
            & long_trend
            & np.isfinite(vwap)
            & (lows < vwap)
            & (closes > vwap)
            & (features["long_reclaim"] >= spec.min_reclaim_pos)
        )
        short_events = (
            channel_short_ok
            & short_trend
            & np.isfinite(vwap)
            & (highs > vwap)
            & (closes < vwap)
            & (features["short_reclaim"] >= spec.min_reclaim_pos)
        )
    else:
        raise ValueError(f"Unknown entry strategy {spec.entry_strategy!r}")

    if "long" in allowed_directions:
        line_depth = np.divide(
            projection.support_touch_value - lows,
            atrs,
            out=np.zeros(n, dtype=float),
            where=atrs > 0.0,
        )
        for event_index in np.where(long_events)[0]:
            append_scalp_candidate(
                rows,
                spec=spec,
                symbol=symbol,
                event_index=int(event_index),
                times=times,
                opens=opens,
                lows=lows,
                highs=highs,
                atrs=atrs,
                direction="long",
                liquidity_level=projection.support_touch_value[event_index],
                stop_anchor=stop_lows[event_index],
                channel_value=projection.support_touch_value[event_index],
                channel_gap_atr=support_gap_atr[event_index],
                channel_tf=tf_name(line_timeframes, int(projection.support_touch_tf[event_index])),
                channel_set=int(projection.support_touch_set[event_index]),
                reclaim_pos=features["long_reclaim"][event_index],
                sweep_depth_atr=max(0.0, float(line_depth[event_index])),
                features=features,
            )
    if "short" in allowed_directions:
        line_depth = np.divide(
            highs - projection.resistance_touch_value,
            atrs,
            out=np.zeros(n, dtype=float),
            where=atrs > 0.0,
        )
        for event_index in np.where(short_events)[0]:
            append_scalp_candidate(
                rows,
                spec=spec,
                symbol=symbol,
                event_index=int(event_index),
                times=times,
                opens=opens,
                lows=lows,
                highs=highs,
                atrs=atrs,
                direction="short",
                liquidity_level=projection.resistance_touch_value[event_index],
                stop_anchor=stop_highs[event_index],
                channel_value=projection.resistance_touch_value[event_index],
                channel_gap_atr=resistance_gap_atr[event_index],
                channel_tf=tf_name(line_timeframes, int(projection.resistance_touch_tf[event_index])),
                channel_set=int(projection.resistance_touch_set[event_index]),
                reclaim_pos=features["short_reclaim"][event_index],
                sweep_depth_atr=max(0.0, float(line_depth[event_index])),
                features=features,
            )
    return frame_candidates(rows)


def append_scalp_candidate(
    rows: list[dict[str, Any]],
    *,
    spec: ScalpSpec,
    symbol: str,
    event_index: int,
    times: pd.Series,
    opens: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
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
    features: dict[str, np.ndarray],
) -> None:
    entry_index = event_index + 1
    if entry_index >= len(opens):
        return
    atr = float(atrs[event_index])
    entry_price = float(opens[entry_index])
    if not all(math.isfinite(value) for value in [atr, entry_price, stop_anchor, channel_value, channel_gap_atr]):
        return
    if atr <= 0.0:
        return
    if direction == "long":
        penetration_price = float(lows[event_index])
        stop_price = float(stop_anchor) - spec.stop_buffer_atr * atr
        risk = entry_price - stop_price
    else:
        penetration_price = float(highs[event_index])
        stop_price = float(stop_anchor) + spec.stop_buffer_atr * atr
        risk = stop_price - entry_price
    risk_atr = risk / atr
    if not math.isfinite(risk) or risk <= 0.0 or risk_atr < spec.min_risk_atr or risk_atr > spec.max_risk_atr:
        return

    before = len(rows)
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
        atr=atr,
        liquidity_level=liquidity_level,
        channel_value=channel_value,
        channel_gap_atr=channel_gap_atr,
        channel_tf=channel_tf,
        channel_set=channel_set,
        reclaim_pos=reclaim_pos,
        sweep_depth_atr=sweep_depth_atr,
        entry_delay_bars=1,
        signal_index=event_index,
    )
    if len(rows) > before:
        close = features["close"]
        rows[-1].update(
            {
                "lookback_bars": int(spec.lookback_bars),
                "trend_filter": spec.trend_filter,
                "min_risk_atr": float(spec.min_risk_atr),
                "max_risk_atr": float(spec.max_risk_atr),
                "stop_lookback_bars": int(spec.stop_lookback_bars),
                "penetration_price": float(penetration_price),
                "stop_anchor_price": float(stop_anchor),
                "risk_atr": float(risk_atr),
                "ema20": float(features["ema20"][event_index]),
                "ema50": float(features["ema50"][event_index]),
                "ema200": float(features["ema200"][event_index]),
                "vwap": float(features["vwap"][event_index]),
                "close_vs_ema20_atr": float((close[event_index] - features["ema20"][event_index]) / atr)
                if math.isfinite(float(features["ema20"][event_index]))
                else math.nan,
                "close_vs_ema50_atr": float((close[event_index] - features["ema50"][event_index]) / atr)
                if math.isfinite(float(features["ema50"][event_index]))
                else math.nan,
                "close_vs_ema200_atr": float((close[event_index] - features["ema200"][event_index]) / atr)
                if math.isfinite(float(features["ema200"][event_index]))
                else math.nan,
                "close_vs_vwap_atr": float((close[event_index] - features["vwap"][event_index]) / atr)
                if math.isfinite(float(features["vwap"][event_index]))
                else math.nan,
                "ema20_delta_6_atr": float(features["ema20_delta_6"][event_index] / atr)
                if math.isfinite(float(features["ema20_delta_6"][event_index]))
                else math.nan,
                "ema50_delta_12_atr": float(features["ema50_delta_12"][event_index] / atr)
                if math.isfinite(float(features["ema50_delta_12"][event_index]))
                else math.nan,
                "ret_6_atr": float(features["ret_6"][event_index] / atr)
                if math.isfinite(float(features["ret_6"][event_index]))
                else math.nan,
            }
        )
        append_htf_feature_columns(rows[-1], features, close[event_index], event_index)


def append_htf_feature_columns(row: dict[str, Any], features: dict[str, np.ndarray], close: float, index: int) -> None:
    del close
    for prefix in ["h4", "daily"]:
        htf_close = float(features.get(f"{prefix}_close", np.full(index + 1, np.nan))[index])
        row[f"{prefix}_close"] = htf_close
        for span in [50, 100, 200]:
            key = f"{prefix}_ema{span}"
            values = features.get(key)
            ema = float(values[index]) if values is not None and index < len(values) else math.nan
            row[key] = ema
            row[f"{prefix}_close_vs_ema{span}_pct"] = (
                float((htf_close / ema) - 1.0) if math.isfinite(htf_close) and math.isfinite(ema) and ema != 0.0 else math.nan
            )


def trend_mask(name: str, direction: str, features: dict[str, np.ndarray]) -> np.ndarray:
    close = features["close"]
    ema20 = features["ema20"]
    ema50 = features["ema50"]
    ema200 = features["ema200"]
    vwap = features["vwap"]
    ema20_delta_6 = features["ema20_delta_6"]
    ema50_delta_12 = features["ema50_delta_12"]
    ret_6 = features["ret_6"]
    valid = np.isfinite(close)
    daily_close = features.get("daily_close", np.full_like(close, np.nan))
    daily_ema50 = features.get("daily_ema50", np.full_like(close, np.nan))
    daily_ema100 = features.get("daily_ema100", np.full_like(close, np.nan))
    daily_ema200 = features.get("daily_ema200", np.full_like(close, np.nan))
    h4_close = features.get("h4_close", np.full_like(close, np.nan))
    h4_ema50 = features.get("h4_ema50", np.full_like(close, np.nan))
    h4_ema200 = features.get("h4_ema200", np.full_like(close, np.nan))
    if name == "none":
        return valid
    if direction == "long":
        if name == "ema20_up":
            return valid & np.isfinite(ema20_delta_6) & (ema20_delta_6 > 0.0)
        if name == "ema50_up":
            return valid & np.isfinite(ema50_delta_12) & (ema50_delta_12 > 0.0)
        if name == "close_above_ema20":
            return valid & np.isfinite(ema20) & (close > ema20)
        if name == "close_above_ema50":
            return valid & np.isfinite(ema50) & (close > ema50)
        if name == "close_above_ema200":
            return valid & np.isfinite(ema200) & (close > ema200)
        if name == "ema20_above_ema50":
            return valid & np.isfinite(ema20) & np.isfinite(ema50) & (ema20 > ema50)
        if name == "close_above_vwap":
            return valid & np.isfinite(vwap) & (close > vwap)
        if name == "momentum_6":
            return valid & np.isfinite(ret_6) & (ret_6 > 0.0)
        if name == "daily_above_ema50":
            return valid & np.isfinite(daily_close) & np.isfinite(daily_ema50) & (daily_close > daily_ema50)
        if name == "daily_above_ema100":
            return valid & np.isfinite(daily_close) & np.isfinite(daily_ema100) & (daily_close > daily_ema100)
        if name == "daily_above_ema200":
            return valid & np.isfinite(daily_close) & np.isfinite(daily_ema200) & (daily_close > daily_ema200)
        if name == "4h_above_ema200":
            return valid & np.isfinite(h4_close) & np.isfinite(h4_ema200) & (h4_close > h4_ema200)
        if name == "htf_bull":
            return (
                valid
                & np.isfinite(daily_close)
                & np.isfinite(daily_ema100)
                & np.isfinite(h4_close)
                & np.isfinite(h4_ema50)
                & (daily_close > daily_ema100)
                & (h4_close > h4_ema50)
            )
    else:
        if name == "ema20_up":
            return valid & np.isfinite(ema20_delta_6) & (ema20_delta_6 < 0.0)
        if name == "ema50_up":
            return valid & np.isfinite(ema50_delta_12) & (ema50_delta_12 < 0.0)
        if name == "close_above_ema20":
            return valid & np.isfinite(ema20) & (close < ema20)
        if name == "close_above_ema50":
            return valid & np.isfinite(ema50) & (close < ema50)
        if name == "close_above_ema200":
            return valid & np.isfinite(ema200) & (close < ema200)
        if name == "ema20_above_ema50":
            return valid & np.isfinite(ema20) & np.isfinite(ema50) & (ema20 < ema50)
        if name == "close_above_vwap":
            return valid & np.isfinite(vwap) & (close < vwap)
        if name == "momentum_6":
            return valid & np.isfinite(ret_6) & (ret_6 < 0.0)
        if name == "daily_above_ema50":
            return valid & np.isfinite(daily_close) & np.isfinite(daily_ema50) & (daily_close < daily_ema50)
        if name == "daily_above_ema100":
            return valid & np.isfinite(daily_close) & np.isfinite(daily_ema100) & (daily_close < daily_ema100)
        if name == "daily_above_ema200":
            return valid & np.isfinite(daily_close) & np.isfinite(daily_ema200) & (daily_close < daily_ema200)
        if name == "4h_above_ema200":
            return valid & np.isfinite(h4_close) & np.isfinite(h4_ema200) & (h4_close < h4_ema200)
        if name == "htf_bull":
            return (
                valid
                & np.isfinite(daily_close)
                & np.isfinite(daily_ema100)
                & np.isfinite(h4_close)
                & np.isfinite(h4_ema50)
                & (daily_close < daily_ema100)
                & (h4_close < h4_ema50)
            )
    raise ValueError(f"Unknown trend filter {name!r}")


def frame_candidates(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return sort_candidate_frame(pd.DataFrame(rows))


def sort_candidate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["entry_index", "channel_gap_atr", "risk_atr"], ascending=[True, True, True]).reset_index(drop=True)


def spec_row(spec: ScalpSpec) -> dict[str, Any]:
    return {
        "entry_strategy": spec.entry_strategy,
        "lookback_bars": spec.lookback_bars,
        "channel_gap_atr": spec.channel_gap_atr,
        "min_reclaim_pos": spec.min_reclaim_pos,
        "target_rr": spec.target_rr,
        "stop_buffer_atr": spec.stop_buffer_atr,
        "min_sweep_depth_atr": spec.min_sweep_depth_atr,
        "min_risk_atr": spec.min_risk_atr,
        "max_risk_atr": spec.max_risk_atr,
        "stop_lookback_bars": spec.stop_lookback_bars,
        "max_hold_bars": spec.max_hold_bars,
        "trend_filter": spec.trend_filter,
    }


if __name__ == "__main__":
    main()
