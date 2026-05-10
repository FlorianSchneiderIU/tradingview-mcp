from __future__ import annotations

import argparse
import json
import math
import sys
from bisect import bisect_right
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backtest_turtle_soup import (
    INTERVAL_MS,
    bfm_zone_feature_values,
    build_bfm_feature_projection,
    build_structure_choch_events,
    find_last_opposite_candle,
    long_pre_entry_invalid,
    normalize_binance_spot_symbol,
    normalize_timeframe,
    short_pre_entry_invalid,
)
from scripts.channel_state_research.backtest import strategy_metrics
from scripts.channel_state_research.data import load_base_candles, prepare_timeframe_bars
from scripts.channel_state_research.production import load_production_config
from scripts.crypto_symbol_sets import SYMBOL_SETS, expand_symbol_args
from scripts.plot_zone_channel_history import build_bfm_magic_lines, parse_timeframes
from scripts.train_bfm_scalper_ml import (
    BFM_SCALP_TF_SETS,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    add_ml_columns,
    aggregate_metrics,
    build_model,
    enrich_confluence_features,
    label_all_candidates,
    run_walk_forward,
)
from scripts.tune_bfm_scalper import build_execution_features
from scripts.tune_bfm_support_resistance import LineBundle, project_lines_to_execution_frame
from scripts.tune_bfm_turtle_soup import (
    OPTIMIZED_BFM_TF_SETS,
    format_sets,
    parse_float_list,
    parse_str_list,
    parse_tf_sets,
    unique_preserve_order,
)
from scripts.tune_horizontal_turtle_bfm import (
    HorizontalLevel,
    build_horizontal_levels,
    channel_confluence_for_level,
    parse_horizontal_pivots,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Canonical original-style Turtle Soup on horizontal levels, with BFM/FVG/KER/POC "
            "as confluence features and a walk-forward ML selector."
        )
    )
    parser.add_argument("--symbols", nargs="+", default=[])
    parser.add_argument("--symbol-set", choices=["none", *SYMBOL_SETS.keys()], default="majors5")
    parser.add_argument("--config", type=Path, default=Path("scripts/channel_15m_broad_v2_full5y_config.json"))
    parser.add_argument("--start", default="2021-09-01")
    parser.add_argument("--end", default="2026-04-20")
    parser.add_argument("--cache-dir", type=Path, default=Path("scripts/.cache"))
    parser.add_argument("--exec-timeframe", default="5m")
    parser.add_argument("--horizontal-timeframes", default="15m,1h,4h,1d")
    parser.add_argument("--horizontal-pivots", default="15m=5:5;1h=5:5;4h=3:3;1d=2:2")
    parser.add_argument("--line-timeframes", default="5m,15m,1h,4h,1d")
    parser.add_argument("--bfm-tf-sets", default=BFM_SCALP_TF_SETS)
    parser.add_argument("--bfm-invalidation", choices=["wick", "close", "none"], default="wick")
    parser.add_argument("--bfm-max-extension-bars", type=int, default=300)
    parser.add_argument("--directions", default="long,short")
    parser.add_argument(
        "--entry-mode",
        choices=[
            "reclaim_next_open",
            "level_retest",
            "break_confirm",
            "choch_ob_limit",
            "choch_ob_retest_close",
            "choch_zone_retest",
        ],
        default="reclaim_next_open",
    )
    parser.add_argument("--structure-timeframe", default="15m")
    parser.add_argument("--structure-left", type=int, default=2)
    parser.add_argument("--structure-right", type=int, default=2)
    parser.add_argument("--max-structure-bars-to-choch", type=int, default=32)
    parser.add_argument("--ob-search-bars", type=int, default=60)
    parser.add_argument("--limit-entry-pos", type=float, default=0.50)
    parser.add_argument("--retest-close-pos", type=float, default=0.50)
    parser.add_argument("--retest-valid-bars", type=int, default=60)
    parser.add_argument(
        "--pre-entry-invalidation-mode",
        choices=["OB Or Stop Wick", "OB Or Stop Close", "Stop Sweep", "Zone Boundary"],
        default="OB Or Stop Wick",
    )
    parser.add_argument("--invalidate-on-close", dest="invalidate_on_close", action="store_true", default=True)
    parser.add_argument("--invalidate-on-wick", dest="invalidate_on_close", action="store_false")
    parser.add_argument("--ob-use-body", action="store_true")
    parser.add_argument("--require-structure-fvg", action="store_true")
    parser.add_argument("--skip-bfm-trade-features", action="store_true")
    parser.add_argument("--target-rr", type=float, default=1.5)
    parser.add_argument("--stop-buffer-atr", type=float, default=0.10)
    parser.add_argument("--max-hold-bars", type=int, default=96)
    parser.add_argument("--min-reclaim-pos", type=float, default=0.45)
    parser.add_argument("--min-sweep-depth-atr", type=float, default=0.02)
    parser.add_argument("--max-sweep-depth-atr", type=float, default=2.5)
    parser.add_argument("--min-level-age-bars", type=int, default=12)
    parser.add_argument("--entry-window-bars", type=int, default=12)
    parser.add_argument("--confirm-window-bars", type=int, default=12)
    parser.add_argument("--min-risk-atr", type=float, default=0.10)
    parser.add_argument("--max-risk-atr", type=float, default=8.0)
    parser.add_argument("--level-cluster-atrs", type=float, default=0.25)
    parser.add_argument("--max-level-scan", type=int, default=800)
    parser.add_argument("--fee-bps-side", type=float, default=None)
    parser.add_argument("--slippage-bps-side", type=float, default=None)
    parser.add_argument("--risk-fraction", type=float, default=None)
    parser.add_argument("--model", choices=["rf", "hgb"], default="hgb")
    parser.add_argument("--label-min-r", type=float, default=0.0)
    parser.add_argument("--train-months", type=int, default=18)
    parser.add_argument("--val-months", type=int, default=6)
    parser.add_argument("--test-months", type=int, default=3)
    parser.add_argument("--embargo-hours", type=float, default=48.0)
    parser.add_argument("--thresholds", default="0.45,0.50,0.55,0.60,0.65,0.70,0.75")
    parser.add_argument("--min-val-trades", type=int, default=12)
    parser.add_argument("--schedule-mode", choices=["per_symbol", "global"], default="per_symbol")
    parser.add_argument("--reuse-dataset", type=Path, default=None)
    parser.add_argument("--write-dataset", action="store_true")
    parser.add_argument("--skip-confluence-enrichment", action="store_true")
    parser.add_argument("--confluence-cache", type=Path, default=None)
    parser.add_argument("--max-horizontal-scan", type=int, default=800)
    parser.add_argument("--event-fvg-lookback", type=int, default=500)
    parser.add_argument("--event-poc-lookback-bars", type=int, default=2016)
    parser.add_argument("--event-poc-bins", type=int, default=160)
    parser.add_argument("--output-prefix", type=Path, default=Path("scripts/horizontal_turtle_confluence_ml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_production_config(args.config)
    symbols = expand_symbol_args(args.symbols, args.symbol_set)
    fee_bps_side = float(config.fee_bps_side if args.fee_bps_side is None else args.fee_bps_side)
    slippage_bps_side = float(config.slippage_bps_side if args.slippage_bps_side is None else args.slippage_bps_side)
    risk_fraction = float(config.risk.risk_fraction if args.risk_fraction is None else args.risk_fraction)

    if args.reuse_dataset:
        dataset = pd.read_csv(args.reuse_dataset, parse_dates=["event_time", "entry_time", "exit_time", "label_end_time"])
        print(f"Loaded reusable dataset {args.reuse_dataset}: {len(dataset):,} rows")
    else:
        dataset = build_labeled_dataset(
            args,
            symbols=symbols,
            fee_bps_side=fee_bps_side,
            slippage_bps_side=slippage_bps_side,
            risk_fraction=risk_fraction,
        )
    if dataset.empty:
        raise RuntimeError("No canonical horizontal Turtle Soup candidates were generated.")

    if not args.skip_confluence_enrichment:
        if args.confluence_cache and args.confluence_cache.exists():
            dataset = pd.read_csv(args.confluence_cache, parse_dates=["event_time", "entry_time", "exit_time", "label_end_time"])
            print(f"Loaded enriched confluence dataset {args.confluence_cache}: {len(dataset):,} rows")
        else:
            dataset = enrich_confluence_features(args, dataset)
            if args.confluence_cache:
                args.confluence_cache.parent.mkdir(parents=True, exist_ok=True)
                dataset.to_csv(args.confluence_cache, index=False)
                print(f"Wrote enriched confluence dataset {args.confluence_cache}")

    dataset = add_ml_columns(dataset)
    if float(args.label_min_r) != 0.0:
        dataset["label"] = (pd.to_numeric(dataset["r_multiple_net"], errors="coerce") >= float(args.label_min_r)).astype(int)
    feature_columns = [
        column
        for column in [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
        if column in dataset.columns
    ]
    print(
        f"Dataset: {len(dataset):,} candidates | {dataset['symbol'].nunique()} symbols | "
        f"positive {dataset['label'].mean():.2%} | features {len(feature_columns)}"
    )

    results = run_walk_forward(
        dataset,
        feature_columns=feature_columns,
        args=args,
        risk_fraction=risk_fraction,
    )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    folds_path = args.output_prefix.with_name(f"{args.output_prefix.name}_folds.csv")
    thresholds_path = args.output_prefix.with_name(f"{args.output_prefix.name}_thresholds.csv")
    scored_path = args.output_prefix.with_name(f"{args.output_prefix.name}_scored.csv")
    selected_path = args.output_prefix.with_name(f"{args.output_prefix.name}_selected_trades.csv")
    dataset_path = args.output_prefix.with_name(f"{args.output_prefix.name}_dataset.csv")
    config_path = args.output_prefix.with_name(f"{args.output_prefix.name}_config.json")

    results["folds"].to_csv(folds_path, index=False)
    results["thresholds"].to_csv(thresholds_path, index=False)
    results["scored"].to_csv(scored_path, index=False)
    results["selected_trades"].to_csv(selected_path, index=False)
    if args.write_dataset:
        dataset.to_csv(dataset_path, index=False)
    config_path.write_text(
        json.dumps(
            {
                "symbols": symbols,
                "start": args.start,
                "end": args.end,
                "feature_columns": feature_columns,
                "candidate_count": int(len(dataset)),
                "positive_rate": float(dataset["label"].mean()),
                "entry_definition": "one candidate per confirmed horizontal level sweep/reclaim; stop beyond actual penetration wick",
                "horizontal_timeframes": args.horizontal_timeframes,
                "horizontal_pivots": args.horizontal_pivots,
                "bfm_tf_sets": args.bfm_tf_sets,
            },
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )

    display = results["folds"]
    if not display.empty:
        cols = [
            "fold",
            "test_start",
            "test_end",
            "threshold",
            "test_rows",
            "selected_trades",
            "selected_total_return",
            "selected_net_r",
            "selected_profit_factor",
            "selected_max_drawdown",
            "auc",
        ]
        print("\nWalk-forward folds")
        print(display[[column for column in cols if column in display.columns]].to_string(index=False))
    print("\nAggregate selected")
    print(pd.DataFrame([aggregate_metrics(results["selected_trades"])]).to_string(index=False))
    print(f"\nWrote {folds_path}")
    print(f"Wrote {thresholds_path}")
    print(f"Wrote {scored_path}")
    print(f"Wrote {selected_path}")
    if args.write_dataset:
        print(f"Wrote {dataset_path}")
    print(f"Wrote {config_path}")


def build_labeled_dataset(
    args: argparse.Namespace,
    *,
    symbols: list[str],
    fee_bps_side: float,
    slippage_bps_side: float,
    risk_fraction: float,
) -> pd.DataFrame:
    horizontal_timeframes = parse_timeframes(args.horizontal_timeframes, "1h")
    horizontal_pivots = parse_horizontal_pivots(args.horizontal_pivots, horizontal_timeframes)
    line_timeframes = parse_timeframes(args.line_timeframes, "1h")
    bfm_sets_by_tf = parse_tf_sets(args.bfm_tf_sets, line_timeframes)
    allowed_directions = set(parse_str_list(args.directions))
    all_frames: list[pd.DataFrame] = []

    for raw_symbol in symbols:
        symbol = normalize_binance_spot_symbol(raw_symbol)
        print(f"\nBuilding canonical horizontal Turtle Soup candidates for {symbol}")
        base = load_base_candles(symbol, args.start, args.end, cache_dir=args.cache_dir, interval="5m")
        all_timeframes = unique_preserve_order(
            [args.exec_timeframe, args.structure_timeframe, *horizontal_timeframes, *line_timeframes, "4h", "1d"]
        )
        bars_by_tf = {
            timeframe: prepare_timeframe_bars(base, timeframe, atr_length=14)
            for timeframe in all_timeframes
        }
        exec_bars = bars_by_tf[args.exec_timeframe].reset_index(drop=True)
        levels = build_horizontal_levels(bars_by_tf, horizontal_pivots)
        print(f"  horizontal levels: {len(levels):,}")

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
            print(f"  {timeframe} BFM: {len(pivots):,} pivots, {len(lines):,} lines, sets {format_sets(bfm_sets_by_tf[timeframe])}")

        projection = project_lines_to_execution_frame(exec_bars, bundles)
        bfm_feature_projection = None
        if not args.skip_bfm_trade_features:
            bfm_feature_projection = build_bfm_feature_projection(
                exec_bars,
                timeframes=line_timeframes,
                tf_sets=bfm_sets_by_tf,
                invalidation=args.bfm_invalidation,
                max_extension_bars=args.bfm_max_extension_bars,
            )
        execution_features = build_execution_features(exec_bars, bars_by_tf=bars_by_tf)
        candidates = build_canonical_candidates(
            exec_bars=exec_bars,
            levels=levels,
            projection=projection,
            bfm_feature_projection=bfm_feature_projection,
            line_timeframes=line_timeframes,
            execution_features=execution_features,
            symbol=symbol,
            allowed_directions=allowed_directions,
            args=args,
        )
        labeled = label_all_candidates(
            candidates,
            exec_bars,
            fee_bps_side=fee_bps_side,
            slippage_bps_side=slippage_bps_side,
            risk_fraction=risk_fraction,
        )
        print(
            f"  {len(candidates):,} canonical candidates -> {len(labeled):,} labelled, "
            f"positive {labeled['label'].mean():.2%}" if not labeled.empty else "  no labelled candidates"
        )
        if not labeled.empty:
            all_frames.append(labeled)
    return pd.concat(all_frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True) if all_frames else pd.DataFrame()


def build_canonical_candidates(
    *,
    exec_bars: pd.DataFrame,
    levels: list[HorizontalLevel],
    projection: Any,
    bfm_feature_projection: Any | None,
    line_timeframes: list[str],
    execution_features: dict[str, np.ndarray],
    symbol: str,
    allowed_directions: set[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    if str(args.entry_mode).startswith("choch_"):
        return build_choch_ob_candidates(
            exec_bars=exec_bars,
            levels=levels,
            projection=projection,
            bfm_feature_projection=bfm_feature_projection,
            line_timeframes=line_timeframes,
            execution_features=execution_features,
            symbol=symbol,
            allowed_directions=allowed_directions,
            args=args,
        )

    opens = pd.to_numeric(exec_bars["open"], errors="coerce").to_numpy(dtype=float)
    highs = pd.to_numeric(exec_bars["high"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(exec_bars["low"], errors="coerce").to_numpy(dtype=float)
    closes = pd.to_numeric(exec_bars["close"], errors="coerce").to_numpy(dtype=float)
    atrs = pd.to_numeric(exec_bars["atr"], errors="coerce").bfill().ffill().to_numpy(dtype=float)
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
    ranges = highs - lows
    long_reclaim = np.divide(closes - lows, ranges, out=np.zeros_like(closes), where=ranges > 0.0)
    short_reclaim = np.divide(highs - closes, ranges, out=np.zeros_like(closes), where=ranges > 0.0)
    rows: list[dict[str, Any]] = []

    for index in range(len(exec_bars) - 1):
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
            active_supports, candidates = scan_side(
                direction="long",
                active_levels=active_supports,
                index=index,
                opens=opens,
                highs=highs,
                lows=lows,
                closes=closes,
                atrs=atrs,
                times=times,
                reclaim_pos=float(long_reclaim[index]),
                projection=projection,
                line_timeframes=line_timeframes,
                execution_features=execution_features,
                symbol=symbol,
                level_confirm_indices=level_confirm_indices,
                args=args,
            )
            rows.extend(candidates)
        if "short" in allowed_directions:
            active_resistances, candidates = scan_side(
                direction="short",
                active_levels=active_resistances,
                index=index,
                opens=opens,
                highs=highs,
                lows=lows,
                closes=closes,
                atrs=atrs,
                times=times,
                reclaim_pos=float(short_reclaim[index]),
                projection=projection,
                line_timeframes=line_timeframes,
                execution_features=execution_features,
                symbol=symbol,
                level_confirm_indices=level_confirm_indices,
                args=args,
            )
            rows.extend(candidates)

    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(["entry_index", "horizontal_rank", "sweep_depth_atr", "channel_gap_atr"], ascending=[True, False, False, True])
        .drop_duplicates(["symbol", "entry_index", "direction", "horizontal_cluster_key"])
        .reset_index(drop=True)
    )


def build_choch_ob_candidates(
    *,
    exec_bars: pd.DataFrame,
    levels: list[HorizontalLevel],
    projection: Any,
    bfm_feature_projection: Any | None,
    line_timeframes: list[str],
    execution_features: dict[str, np.ndarray],
    symbol: str,
    allowed_directions: set[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    opens = pd.to_numeric(exec_bars["open"], errors="coerce").to_numpy(dtype=float)
    highs = pd.to_numeric(exec_bars["high"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(exec_bars["low"], errors="coerce").to_numpy(dtype=float)
    closes = pd.to_numeric(exec_bars["close"], errors="coerce").to_numpy(dtype=float)
    atrs = pd.to_numeric(exec_bars["atr"], errors="coerce").bfill().ffill().to_numpy(dtype=float)
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
    ranges = highs - lows
    long_reclaim = np.divide(closes - lows, ranges, out=np.zeros_like(closes), where=ranges > 0.0)
    short_reclaim = np.divide(highs - closes, ranges, out=np.zeros_like(closes), where=ranges > 0.0)

    exec_tf = normalize_timeframe(args.exec_timeframe)
    structure_tf = normalize_timeframe(args.structure_timeframe)
    structure_events = build_structure_choch_events(
        exec_bars,
        structure_tf,
        int(args.structure_left),
        int(args.structure_right),
    )
    for event in structure_events:
        event["time"] = pd.Timestamp(event["time"]).tz_convert("UTC")
    structure_event_times = [event["time"] for event in structure_events]
    max_choch_wait_exec_bars = int(
        math.ceil(
            float(args.max_structure_bars_to_choch)
            * max(1.0, INTERVAL_MS[structure_tf] / INTERVAL_MS[exec_tf])
        )
    )

    long_setup: dict[str, Any] | None = None
    short_setup: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []

    for index in range(len(exec_bars) - 1):
        now = pd.Timestamp(times.iloc[index]).tz_convert("UTC")
        while support_ptr < len(supports) and supports[support_ptr].confirm_time <= now:
            active_supports.append(supports[support_ptr])
            support_ptr += 1
        while resistance_ptr < len(resistances) and resistances[resistance_ptr].confirm_time <= now:
            active_resistances.append(resistances[resistance_ptr])
            resistance_ptr += 1

        atr = float(atrs[index])
        if math.isfinite(atr) and atr > 0.0:
            if long_setup is not None:
                candidate, long_setup = advance_choch_setup(
                    setup=long_setup,
                    index=index,
                    opens=opens,
                    highs=highs,
                    lows=lows,
                    closes=closes,
                    atrs=atrs,
                    times=times,
                    structure_events=structure_events,
                    max_choch_wait_exec_bars=max_choch_wait_exec_bars,
                    bfm_feature_projection=bfm_feature_projection,
                    execution_features=execution_features,
                    symbol=symbol,
                    args=args,
                )
                if candidate is not None:
                    rows.append(candidate)
                    long_setup = None
            if short_setup is not None:
                candidate, short_setup = advance_choch_setup(
                    setup=short_setup,
                    index=index,
                    opens=opens,
                    highs=highs,
                    lows=lows,
                    closes=closes,
                    atrs=atrs,
                    times=times,
                    structure_events=structure_events,
                    max_choch_wait_exec_bars=max_choch_wait_exec_bars,
                    bfm_feature_projection=bfm_feature_projection,
                    execution_features=execution_features,
                    symbol=symbol,
                    args=args,
                )
                if candidate is not None:
                    rows.append(candidate)
                    short_setup = None

        if math.isfinite(atr) and atr > 0.0 and "long" in allowed_directions:
            active_supports, setup = select_horizontal_sweep(
                direction="long",
                active_levels=active_supports,
                index=index,
                highs=highs,
                lows=lows,
                closes=closes,
                atrs=atrs,
                times=times,
                reclaim_pos=float(long_reclaim[index]),
                projection=projection,
                line_timeframes=line_timeframes,
                level_confirm_indices=level_confirm_indices,
                structure_event_times=structure_event_times,
                args=args,
            )
            if setup is not None:
                long_setup = setup
        if math.isfinite(atr) and atr > 0.0 and "short" in allowed_directions:
            active_resistances, setup = select_horizontal_sweep(
                direction="short",
                active_levels=active_resistances,
                index=index,
                highs=highs,
                lows=lows,
                closes=closes,
                atrs=atrs,
                times=times,
                reclaim_pos=float(short_reclaim[index]),
                projection=projection,
                line_timeframes=line_timeframes,
                level_confirm_indices=level_confirm_indices,
                structure_event_times=structure_event_times,
                args=args,
            )
            if setup is not None:
                short_setup = setup

    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(["entry_index", "horizontal_rank", "sweep_depth_atr", "channel_gap_atr"], ascending=[True, False, False, True])
        .drop_duplicates(["symbol", "entry_index", "direction", "horizontal_cluster_key"])
        .reset_index(drop=True)
    )


def select_horizontal_sweep(
    *,
    direction: str,
    active_levels: list[HorizontalLevel],
    index: int,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    atrs: np.ndarray,
    times: pd.Series,
    reclaim_pos: float,
    projection: Any,
    line_timeframes: list[str],
    level_confirm_indices: dict[str, int],
    structure_event_times: list[pd.Timestamp],
    args: argparse.Namespace,
) -> tuple[list[HorizontalLevel], dict[str, Any] | None]:
    atr = float(atrs[index])
    kept: list[HorizontalLevel] = []
    swept: list[dict[str, Any]] = []
    scan_start = max(0, len(active_levels) - max(1, int(args.max_level_scan)))
    scan_ids = {level.level_id for level in active_levels[scan_start:]}
    for level in active_levels:
        value = float(level.value)
        confirm_index = int(level_confirm_indices.get(level.level_id, index))
        age = index - confirm_index
        if direction == "long":
            invalid = closes[index] < value
            event = level.level_id in scan_ids and age >= args.min_level_age_bars and lows[index] < value and closes[index] > value
            penetration_price = float(lows[index])
            depth = (value - penetration_price) / atr
        else:
            invalid = closes[index] > value
            event = level.level_id in scan_ids and age >= args.min_level_age_bars and highs[index] > value and closes[index] < value
            penetration_price = float(highs[index])
            depth = (penetration_price - value) / atr
        if event and reclaim_pos >= args.min_reclaim_pos and args.min_sweep_depth_atr <= depth <= args.max_sweep_depth_atr:
            channel = channel_confluence_for_level(
                direction=direction,
                level=value,
                index=index,
                projection=projection,
                line_timeframes=line_timeframes,
                atr=atr,
            ) or {"value": math.nan, "gap_atr": 999.0, "tf": "none", "set": -1, "source": "none"}
            swept.append(
                {
                    "direction": direction,
                    "level": level,
                    "level_value": value,
                    "confirm_index": confirm_index,
                    "sweep_idx": int(index),
                    "sweep_time": pd.Timestamp(times.iloc[index]).tz_convert("UTC"),
                    "sweep_extreme": penetration_price,
                    "penetration_price": penetration_price,
                    "sweep_depth_atr": float(depth),
                    "reclaim_pos": float(reclaim_pos),
                    "channel": channel,
                    "event_search_idx": bisect_right(structure_event_times, pd.Timestamp(times.iloc[index]).tz_convert("UTC")),
                    "choch_found": False,
                    "retry_armed": False,
                    "horizontal_cluster_key": f"{level.timeframe}|{round(value / max(atr * args.level_cluster_atrs, 1e-12))}",
                }
            )
            continue
        if not invalid:
            kept.append(level)

    if not swept:
        return kept, None
    swept.sort(key=lambda setup: (-float(setup["level"].rank), float(setup["channel"]["gap_atr"]), -float(setup["sweep_depth_atr"])))
    selected = swept[0]
    cluster_width = float(args.level_cluster_atrs) * atr
    selected_level = float(selected["level_value"])
    kept = [level for level in kept if abs(float(level.value) - selected_level) > cluster_width]
    return kept, selected


def advance_choch_setup(
    *,
    setup: dict[str, Any],
    index: int,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    atrs: np.ndarray,
    times: pd.Series,
    structure_events: list[dict[str, Any]],
    max_choch_wait_exec_bars: int,
    bfm_feature_projection: Any | None,
    execution_features: dict[str, np.ndarray],
    symbol: str,
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    direction = str(setup["direction"])
    level = float(setup["level_value"])
    atr = float(atrs[index])
    if not math.isfinite(atr) or atr <= 0.0:
        return None, setup

    if not setup.get("choch_found", False):
        if direction == "long":
            pre_choch_invalid = closes[index] < level if args.invalidate_on_close else lows[index] < level
            setup["sweep_extreme"] = min(float(setup["sweep_extreme"]), float(lows[index]))
        else:
            pre_choch_invalid = closes[index] > level if args.invalidate_on_close else highs[index] > level
            setup["sweep_extreme"] = max(float(setup["sweep_extreme"]), float(highs[index]))
        if index - int(setup["sweep_idx"]) > max_choch_wait_exec_bars or pre_choch_invalid:
            return None, None

        close_time = pd.Timestamp(times.iloc[index]).tz_convert("UTC")
        while int(setup["event_search_idx"]) < len(structure_events) and structure_events[int(setup["event_search_idx"])]["time"] <= close_time:
            event = structure_events[int(setup["event_search_idx"])]
            setup["event_search_idx"] = int(setup["event_search_idx"]) + 1
            expected = "bull" if direction == "long" else "bear"
            if event["direction"] != expected or event["time"] <= setup["sweep_time"] or index <= int(setup["sweep_idx"]):
                continue
            if args.require_structure_fvg and not event.get("has_fvg", False):
                continue
            start_idx = max(int(setup["sweep_idx"]) + 1, index - int(args.ob_search_bars))
            ob = find_last_opposite_candle(
                opens,
                highs,
                lows,
                closes,
                start_idx,
                index - 1,
                direction,
                bool(args.ob_use_body),
            )
            if ob is None:
                return None, None
            ob_top, ob_bottom = float(ob[0]), float(ob[1])
            if not math.isfinite(ob_top) or not math.isfinite(ob_bottom) or ob_top <= ob_bottom:
                return None, None
            setup["choch_found"] = True
            setup["choch_exec_idx"] = int(index)
            setup["choch_time"] = pd.Timestamp(event["time"]).tz_convert("UTC")
            setup["choch_break_level"] = float(event["break_level"])
            setup["ob_top"] = ob_top
            setup["ob_bottom"] = ob_bottom
            if direction == "long":
                setup["planned_stop"] = float(setup["sweep_extreme"]) - atr * float(args.stop_buffer_atr)
                setup["limit_price"] = level if args.entry_mode == "choch_zone_retest" else ob_bottom + (ob_top - ob_bottom) * float(args.limit_entry_pos)
            else:
                setup["planned_stop"] = float(setup["sweep_extreme"]) + atr * float(args.stop_buffer_atr)
                setup["limit_price"] = level if args.entry_mode == "choch_zone_retest" else ob_top - (ob_top - ob_bottom) * float(args.limit_entry_pos)
            return None, setup
        return None, setup

    close_time = pd.Timestamp(times.iloc[index]).tz_convert("UTC")
    while int(setup["event_search_idx"]) < len(structure_events) and structure_events[int(setup["event_search_idx"])]["time"] <= close_time:
        setup["event_search_idx"] = int(setup["event_search_idx"]) + 1

    choch_index = int(setup["choch_exec_idx"])
    if index - choch_index > int(args.retest_valid_bars):
        return None, None
    stop_price = float(setup["planned_stop"])
    if direction == "long":
        invalid_boundary = level if args.entry_mode == "choch_zone_retest" else float(setup["ob_bottom"])
        invalid = long_pre_entry_invalid(
            args.pre_entry_invalidation_mode,
            False,
            bool(args.invalidate_on_close),
            float(closes[index]),
            float(lows[index]),
            invalid_boundary,
            level,
            stop_price,
        )
    else:
        invalid_boundary = level if args.entry_mode == "choch_zone_retest" else float(setup["ob_top"])
        invalid = short_pre_entry_invalid(
            args.pre_entry_invalidation_mode,
            False,
            bool(args.invalidate_on_close),
            float(closes[index]),
            float(highs[index]),
            invalid_boundary,
            level,
            stop_price,
        )
    if invalid:
        return None, None

    if args.entry_mode == "choch_ob_retest_close":
        if direction == "long":
            retest_close = float(setup["ob_bottom"]) + (float(setup["ob_top"]) - float(setup["ob_bottom"])) * float(args.retest_close_pos)
            touched = lows[index] <= float(setup["ob_top"]) and highs[index] >= float(setup["ob_bottom"])
            ready = touched and closes[index] >= retest_close and index + 1 < len(opens)
        else:
            retest_close = float(setup["ob_top"]) - (float(setup["ob_top"]) - float(setup["ob_bottom"])) * float(args.retest_close_pos)
            touched = highs[index] >= float(setup["ob_bottom"]) and lows[index] <= float(setup["ob_top"])
            ready = touched and closes[index] <= retest_close and index + 1 < len(opens)
        if not ready:
            return None, setup
        entry_index = index + 1
        entry_price = float(opens[entry_index])
        signal_index = index
    else:
        if index <= choch_index:
            return None, setup
        limit_price = float(setup["limit_price"])
        entry_price = fill_limit_price(direction, limit_price, float(opens[index]), float(highs[index]), float(lows[index]))
        if entry_price is None:
            return None, setup
        entry_index = index
        signal_index = choch_index

    if direction == "long" and entry_price <= stop_price:
        return None, None
    if direction == "short" and entry_price >= stop_price:
        return None, None
    candidate = make_choch_candidate(
        setup=setup,
        signal_index=signal_index,
        entry_index=entry_index,
        entry_price=entry_price,
        opens=opens,
        highs=highs,
        lows=lows,
        closes=closes,
        atrs=atrs,
        times=times,
        bfm_feature_projection=bfm_feature_projection,
        execution_features=execution_features,
        symbol=symbol,
        args=args,
    )
    return candidate, setup if candidate is None else None


def fill_limit_price(direction: str, limit_price: float, open_price: float, high: float, low: float) -> float | None:
    if direction == "long":
        if open_price <= limit_price:
            return float(open_price)
        if low <= limit_price <= high:
            return float(limit_price)
    else:
        if open_price >= limit_price:
            return float(open_price)
        if low <= limit_price <= high:
            return float(limit_price)
    return None


def make_choch_candidate(
    *,
    setup: dict[str, Any],
    signal_index: int,
    entry_index: int,
    entry_price: float,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    atrs: np.ndarray,
    times: pd.Series,
    bfm_feature_projection: Any | None,
    execution_features: dict[str, np.ndarray],
    symbol: str,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    direction = str(setup["direction"])
    sign = 1.0 if direction == "long" else -1.0
    sweep_index = int(setup["sweep_idx"])
    choch_index = int(setup["choch_exec_idx"])
    atr_sweep = float(atrs[sweep_index])
    atr_signal = float(atrs[signal_index])
    if not math.isfinite(atr_sweep) or atr_sweep <= 0.0 or not math.isfinite(atr_signal) or atr_signal <= 0.0:
        return None
    level = float(setup["level_value"])
    stop_price = float(setup["planned_stop"])
    risk = entry_price - stop_price if direction == "long" else stop_price - entry_price
    risk_atr = risk / atr_signal
    if not math.isfinite(risk) or risk <= 0.0 or risk_atr < args.min_risk_atr or risk_atr > args.max_risk_atr:
        return None
    target_price = entry_price + args.target_rr * risk if direction == "long" else entry_price - args.target_rr * risk
    level_obj: HorizontalLevel = setup["level"]
    channel = setup["channel"]
    ob_top = float(setup["ob_top"])
    ob_bottom = float(setup["ob_bottom"])
    ob_mid = (ob_top + ob_bottom) / 2.0
    stop_beyond_zone = (level - stop_price) / atr_signal if direction == "long" else (stop_price - level) / atr_signal
    sweep_range = float(highs[sweep_index] - lows[sweep_index])
    if direction == "long":
        sweep_reaction = max(0.0, (float(highs[sweep_index]) - level) / atr_sweep)
        sweep_close_reaction = (float(closes[sweep_index]) - level) / atr_sweep
    else:
        sweep_reaction = max(0.0, (level - float(lows[sweep_index])) / atr_sweep)
        sweep_close_reaction = (level - float(closes[sweep_index])) / atr_sweep

    row = {
        "symbol": symbol,
        "entry_strategy": args.entry_mode,
        "trigger_family": "horizontal_turtle_choch_ob",
        "event_index": int(sweep_index),
        "signal_index": int(signal_index),
        "entry_index": int(entry_index),
        "choch_index": int(choch_index),
        "event_time": pd.Timestamp(times.iloc[sweep_index]).tz_convert("UTC"),
        "choch_time": pd.Timestamp(setup["choch_time"]).tz_convert("UTC"),
        "signal_time": pd.Timestamp(times.iloc[signal_index]).tz_convert("UTC"),
        "entry_time": pd.Timestamp(times.iloc[entry_index]).tz_convert("UTC"),
        "direction": direction,
        "entry_price": float(entry_price),
        "stop_price": float(stop_price),
        "target_price": float(target_price),
        "risk_abs": float(risk),
        "target_rr_planned": float(args.target_rr),
        "max_hold_bars": int(args.max_hold_bars),
        "liquidity_level": level,
        "horizontal_level": level,
        "horizontal_tf": level_obj.timeframe,
        "horizontal_rank": float(level_obj.rank),
        "horizontal_cluster_key": str(setup["horizontal_cluster_key"]),
        "level_age_bars": float(sweep_index - int(setup["confirm_index"])),
        "channel_value": float(channel["value"]),
        "channel_gap_atr": float(channel["gap_atr"]),
        "channel_tf": str(channel["tf"]),
        "channel_set": int(channel["set"]),
        "channel_source": str(channel["source"]),
        "reclaim_pos": float(setup["reclaim_pos"]),
        "sweep_depth_atr": float(setup["sweep_depth_atr"]),
        "entry_delay_bars": float(entry_index - sweep_index),
        "lookback_bars": float(args.min_level_age_bars),
        "trend_filter": "none",
        "min_risk_atr": float(args.min_risk_atr),
        "max_risk_atr": float(args.max_risk_atr),
        "stop_lookback_bars": 0.0,
        "penetration_price": float(setup["penetration_price"]),
        "stop_anchor_price": float(setup["sweep_extreme"]),
        "risk_atr": float(risk_atr),
        "level_close_gap_atr": sign * (float(closes[sweep_index]) - level) / atr_sweep,
        "level_body_reclaim_atr": sign * (float(closes[sweep_index]) - float(opens[sweep_index])) / atr_sweep,
        "sweep_same_bar_reaction_atr": float(sweep_reaction),
        "sweep_same_bar_close_reaction_atr": float(sweep_close_reaction),
        "sweep_same_bar_adverse_atr": float(setup["sweep_depth_atr"]),
        "sweep_reclaim_body_atr": sign * (float(closes[sweep_index]) - float(opens[sweep_index])) / atr_sweep,
        "sweep_range_atr": sweep_range / atr_sweep if atr_sweep > 0.0 else math.nan,
        "choch_wait_bars": float(choch_index - sweep_index),
        "signal_after_choch_bars": float(signal_index - choch_index),
        "entry_after_signal_bars": float(entry_index - signal_index),
        "choch_to_entry_bars": float(entry_index - choch_index),
        "choch_break_gap_atr": sign * (float(setup["choch_break_level"]) - level) / atr_signal,
        "ob_top": ob_top,
        "ob_bottom": ob_bottom,
        "ob_width_atr_signal": (ob_top - ob_bottom) / atr_signal,
        "entry_to_ob_mid_atr": sign * (float(entry_price) - ob_mid) / atr_signal,
        "entry_to_zone_mid_atr": sign * (float(entry_price) - level) / atr_signal,
        "stop_beyond_zone_atr": float(stop_beyond_zone),
        "entry_vs_signal_close_atr": sign * (float(entry_price) - float(closes[signal_index])) / atr_signal,
        "signal_close_vs_zone_atr": sign * (float(closes[signal_index]) - level) / atr_signal,
    }
    append_execution_context(row, execution_features, closes, signal_index, atr_signal)
    append_bfm_trade_features(
        row,
        bfm_feature_projection=bfm_feature_projection,
        direction=direction,
        level=level,
        sweep_index=sweep_index,
        signal_index=signal_index,
        highs=highs,
        lows=lows,
        closes=closes,
        atrs=atrs,
    )
    return row


def prefixed_bfm_values(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key[4:]}": value for key, value in values.items() if key.startswith("bfm_")}


def append_bfm_trade_features(
    row: dict[str, Any],
    *,
    bfm_feature_projection: Any | None,
    direction: str,
    level: float,
    sweep_index: int,
    signal_index: int,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    atrs: np.ndarray,
) -> None:
    if bfm_feature_projection is None:
        return
    zone = {"top": float(level), "bottom": float(level)}
    atr_sweep = float(atrs[sweep_index])
    atr_signal = float(atrs[signal_index])
    row.update(
        prefixed_bfm_values(
            "bfm_sweep",
            bfm_zone_feature_values(
                projection=bfm_feature_projection,
                direction=direction,
                zone=zone,
                index=sweep_index,
                atr=atr_sweep,
                close=float(closes[sweep_index]),
                high=float(highs[sweep_index]),
                low=float(lows[sweep_index]),
            ),
        )
    )
    row.update(
        prefixed_bfm_values(
            "bfm_signal",
            bfm_zone_feature_values(
                projection=bfm_feature_projection,
                direction=direction,
                zone=zone,
                index=signal_index,
                atr=atr_signal,
                close=float(closes[signal_index]),
                high=float(highs[signal_index]),
                low=float(lows[signal_index]),
            ),
        )
    )


def scan_side(
    *,
    direction: str,
    active_levels: list[HorizontalLevel],
    index: int,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    atrs: np.ndarray,
    times: pd.Series,
    reclaim_pos: float,
    projection: Any,
    line_timeframes: list[str],
    execution_features: dict[str, np.ndarray],
    symbol: str,
    level_confirm_indices: dict[str, int],
    args: argparse.Namespace,
) -> tuple[list[HorizontalLevel], list[dict[str, Any]]]:
    atr = float(atrs[index])
    kept: list[HorizontalLevel] = []
    swept: list[HorizontalLevel] = []
    scan_start = max(0, len(active_levels) - max(1, int(args.max_level_scan)))
    scan_ids = {level.level_id for level in active_levels[scan_start:]}
    for level in active_levels:
        value = float(level.value)
        age = index - int(level_confirm_indices.get(level.level_id, index))
        if direction == "long":
            invalid = closes[index] < value
            event = level.level_id in scan_ids and age >= args.min_level_age_bars and lows[index] < value and closes[index] > value
            depth = (value - lows[index]) / atr
        else:
            invalid = closes[index] > value
            event = level.level_id in scan_ids and age >= args.min_level_age_bars and highs[index] > value and closes[index] < value
            depth = (highs[index] - value) / atr
        if event and reclaim_pos >= args.min_reclaim_pos and args.min_sweep_depth_atr <= depth <= args.max_sweep_depth_atr:
            swept.append(level)
            continue
        if not invalid:
            kept.append(level)

    candidates: list[dict[str, Any]] = []
    for level in swept:
        candidate = make_candidate(
            direction=direction,
            level=level,
            index=index,
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
            atrs=atrs,
            times=times,
            reclaim_pos=reclaim_pos,
            projection=projection,
            line_timeframes=line_timeframes,
            execution_features=execution_features,
            symbol=symbol,
            level_confirm_indices=level_confirm_indices,
            args=args,
        )
        if candidate is not None:
            candidates.append(candidate)
    if not candidates:
        return kept, []

    candidates.sort(key=lambda row: (-float(row["horizontal_rank"]), float(row["channel_gap_atr"]), -float(row["sweep_depth_atr"])))
    selected = candidates[0]
    cluster_width = float(args.level_cluster_atrs) * atr
    selected_level = float(selected["liquidity_level"])
    kept = [level for level in kept if abs(float(level.value) - selected_level) > cluster_width]
    return kept, [selected]


def make_candidate(
    *,
    direction: str,
    level: HorizontalLevel,
    index: int,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    atrs: np.ndarray,
    times: pd.Series,
    reclaim_pos: float,
    projection: Any,
    line_timeframes: list[str],
    execution_features: dict[str, np.ndarray],
    symbol: str,
    level_confirm_indices: dict[str, int],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    atr = float(atrs[index])
    if not math.isfinite(atr) or atr <= 0.0:
        return None
    value = float(level.value)
    penetration_price = float(lows[index] if direction == "long" else highs[index])
    sweep_depth_atr = (value - penetration_price) / atr if direction == "long" else (penetration_price - value) / atr
    stop_price = penetration_price - args.stop_buffer_atr * atr if direction == "long" else penetration_price + args.stop_buffer_atr * atr
    entry = resolve_entry(
        direction=direction,
        mode=args.entry_mode,
        event_index=index,
        level=value,
        stop_price=stop_price,
        opens=opens,
        highs=highs,
        lows=lows,
        confirm_window_bars=int(args.confirm_window_bars),
        entry_window_bars=int(args.entry_window_bars),
    )
    if entry is None:
        return None
    entry_index, entry_price, signal_index = entry
    risk = entry_price - stop_price if direction == "long" else stop_price - entry_price
    risk_atr = risk / atr
    if not math.isfinite(risk) or risk <= 0.0 or risk_atr < args.min_risk_atr or risk_atr > args.max_risk_atr:
        return None
    target_price = entry_price + args.target_rr * risk if direction == "long" else entry_price - args.target_rr * risk
    channel = channel_confluence_for_level(
        direction=direction,
        level=value,
        index=index,
        projection=projection,
        line_timeframes=line_timeframes,
        atr=atr,
    ) or {"value": math.nan, "gap_atr": 999.0, "tf": "none", "set": -1, "source": "none"}
    confirm_index = int(level_confirm_indices.get(level.level_id, index))
    sign = 1.0 if direction == "long" else -1.0
    row = {
        "symbol": symbol,
        "entry_strategy": args.entry_mode,
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
        "target_rr_planned": float(args.target_rr),
        "max_hold_bars": int(args.max_hold_bars),
        "liquidity_level": value,
        "horizontal_level": value,
        "horizontal_tf": level.timeframe,
        "horizontal_rank": float(level.rank),
        "horizontal_cluster_key": f"{level.timeframe}|{round(value / max(atr * args.level_cluster_atrs, 1e-12))}",
        "level_age_bars": float(index - confirm_index),
        "channel_value": float(channel["value"]),
        "channel_gap_atr": float(channel["gap_atr"]),
        "channel_tf": str(channel["tf"]),
        "channel_set": int(channel["set"]),
        "channel_source": str(channel["source"]),
        "reclaim_pos": float(reclaim_pos),
        "sweep_depth_atr": float(sweep_depth_atr),
        "entry_delay_bars": float(entry_index - index),
        "lookback_bars": float(args.min_level_age_bars),
        "trend_filter": "none",
        "min_risk_atr": float(args.min_risk_atr),
        "max_risk_atr": float(args.max_risk_atr),
        "stop_lookback_bars": 0.0,
        "penetration_price": penetration_price,
        "stop_anchor_price": penetration_price,
        "risk_atr": float(risk_atr),
        "level_close_gap_atr": sign * (float(closes[index]) - value) / atr,
        "level_body_reclaim_atr": sign * (float(closes[index]) - float(opens[index])) / atr,
    }
    append_execution_context(row, execution_features, closes, index, atr)
    return row


def resolve_entry(
    *,
    direction: str,
    mode: str,
    event_index: int,
    level: float,
    stop_price: float,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    confirm_window_bars: int,
    entry_window_bars: int,
) -> tuple[int, float, int] | None:
    if mode == "reclaim_next_open":
        entry_index = event_index + 1
        if entry_index >= len(opens):
            return None
        return entry_index, float(opens[entry_index]), event_index
    if mode == "level_retest":
        final = min(len(opens) - 1, event_index + entry_window_bars)
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
    final = min(len(opens) - 2, event_index + confirm_window_bars)
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


def append_execution_context(row: dict[str, Any], features: dict[str, np.ndarray], closes: np.ndarray, index: int, atr: float) -> None:
    close = float(closes[index])
    for key in ["ema20", "ema50", "ema200", "vwap"]:
        values = features.get(key)
        value = float(values[index]) if values is not None and index < len(values) else math.nan
        row[key] = value
        row[f"close_vs_{key}_atr"] = (close - value) / atr if math.isfinite(value) and atr > 0 else math.nan
    for key in ["ema20_delta_6", "ema50_delta_12", "ret_6"]:
        values = features.get(key)
        value = float(values[index]) if values is not None and index < len(values) else math.nan
        row[f"{key}_atr"] = value / atr if math.isfinite(value) and atr > 0 else math.nan
    for prefix in ["h4", "daily"]:
        htf_close_values = features.get(f"{prefix}_close")
        htf_close = float(htf_close_values[index]) if htf_close_values is not None and index < len(htf_close_values) else math.nan
        row[f"{prefix}_close"] = htf_close
        for span in [50, 100, 200]:
            key = f"{prefix}_ema{span}"
            values = features.get(key)
            ema = float(values[index]) if values is not None and index < len(values) else math.nan
            row[key] = ema
            row[f"{prefix}_close_vs_ema{span}_pct"] = (htf_close / ema - 1.0) if math.isfinite(htf_close) and math.isfinite(ema) and ema != 0 else math.nan


if __name__ == "__main__":
    main()
