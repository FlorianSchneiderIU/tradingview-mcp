from __future__ import annotations

import argparse
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backtest_turtle_soup import Config, Trade, run_backtest, side_metrics, summarize
from scripts.channel_state_research.data import load_base_candles, prepare_timeframe_bars
from scripts.channel_state_research.production import load_production_config
from scripts.plot_zone_channel_history import build_bfm_magic_lines, parse_bfm_sets, parse_timeframes
from scripts.tune_bfm_support_resistance import LineBundle, project_lines_to_execution_frame
from scripts.tune_bfm_turtle_soup import OPTIMIZED_BFM_TF_SETS, parse_tf_sets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the existing Turtle Soup CHoCH/OB strategy and filter trades with optimized BFM channel features."
    )
    parser.add_argument("--config", type=Path, default=Path("scripts/channel_15m_broad_v2_full5y_config.json"))
    parser.add_argument("--start", default="2021-04-30")
    parser.add_argument("--end", default="2026-04-30")
    parser.add_argument("--cache-dir", type=Path, default=Path("scripts/.cache"))
    parser.add_argument("--exec-timeframe", default="5m")
    parser.add_argument("--structure-timeframes", default="15m")
    parser.add_argument("--tf-profiles", default="1h_only,1h_4h,4h_1d")
    parser.add_argument("--entry-modes", default="zone_retest,retest_close,limit_mid")
    parser.add_argument("--max-choch-bars", default="16,32,48")
    parser.add_argument("--target-rrs", default="1.0,1.5,2.0")
    parser.add_argument("--min-reclaim-positions", default="0.5,0.7")
    parser.add_argument("--dead-zone-options", default="false,true")
    parser.add_argument("--channel-gap-atrs", default="0.5,1.0,2.0,999")
    parser.add_argument("--channel-timeframe-filter", default="any,4h,1d")
    parser.add_argument("--line-timeframes", default="1h,4h,1d")
    parser.add_argument("--bfm-tf-sets", default=OPTIMIZED_BFM_TF_SETS)
    parser.add_argument("--bfm-invalidation", choices=["wick", "close", "none"], default="wick")
    parser.add_argument("--bfm-max-extension-bars", type=int, default=300)
    parser.add_argument("--min-trades-for-score", type=int, default=30)
    parser.add_argument("--max-configs", type=int, default=0)
    parser.add_argument("--output-prefix", type=Path, default=Path("scripts/turtle_soup_bfm_filter_full5y"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prod_config = load_production_config(args.config)
    line_timeframes = parse_timeframes(args.line_timeframes, "1h")
    bfm_sets_by_tf = parse_tf_sets(args.bfm_tf_sets, line_timeframes)
    base = load_base_candles(
        prod_config.symbol,
        args.start,
        args.end,
        cache_dir=args.cache_dir,
        interval=prod_config.base_interval,
    )
    exec_bars = prepare_timeframe_bars(base, args.exec_timeframe, atr_length=prod_config.atr_length)
    bundles: dict[str, LineBundle] = {}
    for timeframe in line_timeframes:
        bars = prepare_timeframe_bars(base, timeframe, atr_length=prod_config.atr_length)
        lines, pivots = build_bfm_magic_lines(
            bars,
            bfm_sets_by_tf[timeframe],
            invalidation=args.bfm_invalidation,
            max_extension_bars=args.bfm_max_extension_bars,
        )
        bundles[timeframe] = LineBundle(
            timeframe=timeframe,
            scale=1.0,
            sets=tuple(bfm_sets_by_tf[timeframe]),
            bars=bars,
            lines=tuple(lines),
            pivots_count=len(pivots),
        )
        print(f"{timeframe} BFM: {len(pivots):,} pivots, {len(lines):,} lines")

    projection = project_lines_to_execution_frame(exec_bars, bundles)
    specs = config_specs(args)
    if args.max_configs > 0:
        specs = specs[: args.max_configs]
    print(f"Running {len(specs)} Turtle Soup configs x BFM filters")

    rows: list[dict[str, Any]] = []
    best_trades: list[Trade] = []
    best_rows = pd.DataFrame()
    best_score = -float("inf")
    best_summary: dict[str, Any] | None = None
    for index, (name, cfg) in enumerate(specs, start=1):
        trades = run_backtest(exec_bars.copy(), cfg)
        enriched = enrich_trades(trades, exec_bars, projection, line_timeframes)
        raw_summary = summarize(trades)
        for gap in parse_float_list(args.channel_gap_atrs):
            for tf_filter in parse_str_list(args.channel_timeframe_filter):
                filtered = filter_enriched(enriched, gap, tf_filter)
                filtered_trades = [trades[int(row.trade_order)] for row in filtered.itertuples()]
                metrics = summarize(filtered_trades)
                score = robust_score(metrics, int(args.min_trades_for_score))
                long_metrics = side_metrics(filtered_trades, "long")
                short_metrics = side_metrics(filtered_trades, "short")
                row = {
                    "config_index": index,
                    "config_name": name,
                    "bfm_gap_atr": gap,
                    "bfm_tf_filter": tf_filter,
                    "raw_trades": raw_summary["trades"],
                    "raw_net_r": raw_summary["net_r"],
                    "raw_pf": raw_summary["profit_factor"],
                    "score": score,
                    **config_row(cfg),
                    **{f"filtered_{key}": value for key, value in metrics.items()},
                    "long_net_r": long_metrics["net_r"],
                    "long_trades": long_metrics["trades"],
                    "short_net_r": short_metrics["net_r"],
                    "short_trades": short_metrics["trades"],
                }
                rows.append(row)
                if best_summary is None or score > best_score:
                    best_score = score
                    best_summary = row
                    best_trades = filtered_trades
                    best_rows = filtered
        if index == 1 or index % 10 == 0 or index == len(specs):
            print(
                f"[{index}/{len(specs)}] raw {raw_summary['trades']} trades "
                f"{raw_summary['net_r']}R PF {raw_summary['profit_factor']}; best score {best_score:.3f}"
            )

    summary = pd.DataFrame(rows).sort_values(["score", "filtered_net_r", "filtered_profit_factor"], ascending=[False, False, False])
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_prefix.with_name(f"{args.output_prefix.name}_summary.csv")
    trades_path = args.output_prefix.with_name(f"{args.output_prefix.name}_best_trades.csv")
    config_path = args.output_prefix.with_name(f"{args.output_prefix.name}_best_config.json")
    summary.to_csv(summary_path, index=False)
    best_rows.to_csv(trades_path, index=False)
    config_path.write_text(
        json_dumps({**(best_summary or {}), "bfm_tf_sets": {tf: format_sets(sets) for tf, sets in bfm_sets_by_tf.items()}}),
        encoding="utf-8",
    )

    print("\nBest filtered Turtle Soup config")
    if best_summary:
        for key in [
            "config_name",
            "bfm_gap_atr",
            "bfm_tf_filter",
            "raw_trades",
            "raw_net_r",
            "filtered_trades",
            "filtered_win_rate",
            "filtered_profit_factor",
            "filtered_net_r",
            "filtered_avg_r",
            "long_net_r",
            "short_net_r",
            "score",
        ]:
            print(f"  {key}: {best_summary.get(key)}")
    print(f"\nWrote {summary_path}")
    print(f"Wrote {trades_path}")
    print(f"Wrote {config_path}")


def config_specs(args: argparse.Namespace) -> list[tuple[str, Config]]:
    profiles = {
        "1h_only": dict(tf1="1h", tf2="4h", use_tf1=True, use_tf2=False),
        "1h_4h": dict(tf1="1h", tf2="4h", use_tf1=True, use_tf2=True),
        "4h_1d": dict(tf1="4h", tf2="1d", use_tf1=True, use_tf2=True),
        "1d_only": dict(tf1="1d", tf2="1w", use_tf1=True, use_tf2=False),
    }
    specs: list[tuple[str, Config]] = []
    for profile in parse_str_list(args.tf_profiles):
        for structure_tf in parse_str_list(args.structure_timeframes):
            for entry_mode in parse_str_list(args.entry_modes):
                for max_choch in parse_int_list(args.max_choch_bars):
                    for target_rr in parse_float_list(args.target_rrs):
                        for reclaim in parse_float_list(args.min_reclaim_positions):
                            for dead_zone in parse_bool_list(args.dead_zone_options):
                                kwargs = profiles[profile]
                                name = (
                                    f"{profile}|{structure_tf}|{entry_mode}|choch{max_choch}|"
                                    f"rr{target_rr:g}|reclaim{reclaim:g}|{'dead' if dead_zone else 'nodead'}"
                                )
                                specs.append(
                                    (
                                        name,
                                        Config(
                                            exec_tf=args.exec_timeframe,
                                            structure_tf=structure_tf,
                                            entry_mode=entry_mode,
                                            target_rr=float(target_rr),
                                            max_structure_bars_to_choch=int(max_choch),
                                            min_sweep_reclaim_pos=float(reclaim),
                                            block_dead_zone=bool(dead_zone),
                                            max_zone_scan=250,
                                            **kwargs,
                                        ),
                                    )
                                )
    return specs


def enrich_trades(
    trades: list[Trade],
    exec_bars: pd.DataFrame,
    projection,
    line_timeframes: list[str],
) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    atrs = pd.to_numeric(exec_bars["atr"], errors="coerce").to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for order, trade in enumerate(trades):
        idx = int(trade.sweep_index)
        if idx < 0 or idx >= len(atrs) or not math.isfinite(float(atrs[idx])) or atrs[idx] <= 0:
            continue
        if trade.direction == "long":
            gap = float(projection.support_touch_gap[idx]) / float(atrs[idx])
            channel_value = float(projection.support_touch_value[idx])
            channel_tf = tf_name(line_timeframes, int(projection.support_touch_tf[idx]))
            channel_set = int(projection.support_touch_set[idx])
        else:
            gap = float(projection.resistance_touch_gap[idx]) / float(atrs[idx])
            channel_value = float(projection.resistance_touch_value[idx])
            channel_tf = tf_name(line_timeframes, int(projection.resistance_touch_tf[idx]))
            channel_set = int(projection.resistance_touch_set[idx])
        row = asdict(trade)
        row.update(
            {
                "trade_order": order,
                "bfm_gap_atr": gap,
                "bfm_channel_value": channel_value,
                "bfm_channel_tf": channel_tf,
                "bfm_channel_set": channel_set,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def filter_enriched(frame: pd.DataFrame, gap_atr: float, tf_filter: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    mask = frame["bfm_gap_atr"].astype(float) <= float(gap_atr)
    if tf_filter != "any":
        mask &= frame["bfm_channel_tf"].astype(str) == tf_filter
    return frame.loc[mask].copy()


def robust_score(metrics: dict[str, Any], min_trades: int) -> float:
    trades = int(metrics.get("trades", 0))
    if trades < min_trades:
        return -float("inf")
    net_r = float(metrics.get("net_r", 0.0))
    pf = float(metrics.get("profit_factor", 0.0))
    avg_r = float(metrics.get("avg_r", 0.0))
    if math.isinf(pf):
        pf = 5.0
    return net_r + 5.0 * min(pf - 1.0, 3.0) + 10.0 * avg_r


def config_row(cfg: Config) -> dict[str, Any]:
    return {
        "exec_tf": cfg.exec_tf,
        "structure_tf": cfg.structure_tf,
        "entry_mode": cfg.entry_mode,
        "tf1": cfg.tf1,
        "tf2": cfg.tf2,
        "use_tf2": cfg.use_tf2,
        "target_rr": cfg.target_rr,
        "max_choch": cfg.max_structure_bars_to_choch,
        "min_reclaim_pos": cfg.min_sweep_reclaim_pos,
        "dead_zone": cfg.block_dead_zone,
    }


def parse_str_list(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in str(raw).split(",") if item.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def parse_bool_list(raw: str) -> list[bool]:
    out: list[bool] = []
    for item in parse_str_list(raw):
        out.append(item.lower() in {"1", "true", "yes", "y"})
    return out


def tf_name(timeframes: list[str], index: int) -> str:
    if index < 0 or index >= len(timeframes):
        return ""
    return timeframes[index]


def format_sets(sets: list[tuple[int, int]]) -> str:
    return ",".join(f"{left}:{right}" for left, right in sets)


def json_dumps(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, default=str, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
