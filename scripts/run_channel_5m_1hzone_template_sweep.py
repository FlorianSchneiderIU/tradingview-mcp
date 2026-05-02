from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.channel_state_research.data import build_market_dataset
from scripts.channel_state_research.features import TimeframeFeatureSpec, build_decision_dataset, build_timeframe_state_frame
from scripts.channel_state_research.zone_confluence import ZoneChannelEventSpec, build_zone_channel_event_dataset
from scripts.run_zone_channel_confluence_study import (
    apply_event_filters,
    build_report,
    resolve_thresholds,
    run_walkforward_event_study,
    select_feature_columns,
)


@dataclass(frozen=True)
class TemplateSpec:
    name: str
    entry_mode: str
    stop_mode: str
    label_horizon_bars: int
    target_buffer_atr: float
    stop_buffer_atr: float
    passive_entry_window_bars: int = 6
    passive_entry_buffer_atr: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BTC 5m / 1h-zone template sweep.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-04-20")
    parser.add_argument("--cache-dir", type=Path, default=Path("scripts/.cache"))
    parser.add_argument("--base-interval", default="5m")
    parser.add_argument("--timeframes", default="5m,15m,1h")
    parser.add_argument("--decision-timeframe", default="5m")
    parser.add_argument("--output-prefix", type=Path, default=Path("scripts/channel_5m_1hzone_template_sweep"))
    parser.add_argument("--train-months", type=int, default=4)
    parser.add_argument("--val-months", type=int, default=2)
    parser.add_argument("--test-months", type=int, default=2)
    parser.add_argument("--embargo-bars", type=int, default=24)
    parser.add_argument("--min-validation-trades", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timeframes = [item.strip() for item in args.timeframes.split(",") if item.strip()]
    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    market, decision_frame, decision_groups = build_decision_inputs(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        cache_dir=args.cache_dir,
        base_interval=args.base_interval,
        timeframes=timeframes,
        decision_timeframe=args.decision_timeframe,
    )

    templates = [
        TemplateSpec("mr_zone_h24_tb02_sb02", "market_reclaim", "zone", 24, 0.20, 0.20),
        TemplateSpec("mr_zone_h12_tb05_sb02", "market_reclaim", "zone", 12, 0.50, 0.20),
        TemplateSpec("mr_zone_h08_tb08_sb02", "market_reclaim", "zone", 8, 0.80, 0.20),
        TemplateSpec("mr_zone_h12_tb05_sb00", "market_reclaim", "zone", 12, 0.50, 0.00),
        TemplateSpec("mr_channel_h12_tb05_sb02", "market_reclaim", "channel_anchor", 12, 0.50, 0.20),
        TemplateSpec("mr_reaction_h12_tb05_sb02", "market_reclaim", "reaction_extreme", 12, 0.50, 0.20),
        TemplateSpec("pr_zone_h24_tb02_sb02", "passive_retest", "zone", 24, 0.20, 0.20),
        TemplateSpec("pr_zone_h12_tb05_sb02", "passive_retest", "zone", 12, 0.50, 0.20),
        TemplateSpec("pr_channel_h12_tb05_sb02", "passive_retest", "channel_anchor", 12, 0.50, 0.20),
    ]

    summary_rows: list[dict[str, Any]] = []
    for template in templates:
        print(f"\nRunning {template.name} ...", flush=True)
        event_frame, feature_groups = build_template_dataset(
            market=market,
            decision_frame=decision_frame,
            decision_groups=decision_groups,
            decision_timeframe=args.decision_timeframe,
            timeframes=timeframes,
            template=template,
        )
        result = run_template(
            event_frame=event_frame,
            feature_groups=feature_groups,
            template=template,
            symbol=args.symbol,
            start=args.start,
            end=args.end,
            timeframes=timeframes,
            decision_timeframe=args.decision_timeframe,
            train_months=args.train_months,
            val_months=args.val_months,
            test_months=args.test_months,
            embargo_bars=args.embargo_bars,
            min_validation_trades=args.min_validation_trades,
            output_prefix=output_prefix.with_name(output_prefix.name + "_" + template.name),
        )
        summary_rows.append(result)

    summary = pd.DataFrame(summary_rows).sort_values(
        ["net_r", "profit_factor", "trades", "raw_event_net_r"],
        ascending=[False, False, False, False],
    )
    summary_path = output_prefix.with_name(output_prefix.name + "_summary.csv")
    summary.to_csv(summary_path, index=False)

    report_lines = [
        "# BTC 5m / 1h-Zone Template Sweep",
        "",
        f"- symbol: `{args.symbol}`",
        f"- window: `{args.start}` to `{args.end}`",
        f"- timeframes: `{', '.join(timeframes)}`",
        f"- decision timeframe: `{args.decision_timeframe}`",
        f"- train/val/test months: `{args.train_months}/{args.val_months}/{args.test_months}`",
        "",
        "## Summary",
        "",
    ]
    for row in summary.to_dict(orient="records"):
        report_lines.append(
            f"- `{row['template']}`: trades `{row['trades']}`, net_r `{row['net_r']}`, PF `{row['profit_factor']}`, raw_event_net_r `{row['raw_event_net_r']}`"
        )
    report_path = output_prefix.with_name(output_prefix.name + "_report.md")
    report_path.write_text("\n".join(report_lines).strip() + "\n", encoding="utf-8")

    print("\nSummary")
    print(summary.to_string(index=False))
    print(f"\nSaved summary to {summary_path}")
    print(f"Saved report to {report_path}")


def build_decision_inputs(
    *,
    symbol: str,
    start: str,
    end: str,
    cache_dir: Path,
    base_interval: str,
    timeframes: list[str],
    decision_timeframe: str,
):
    reversal_map = {
        "5m": 1.5,
        "15m": 1.5,
        "1h": 2.0,
        "4h": 2.0,
        "1d": 2.0,
        "1w": 1.5,
    }
    market = build_market_dataset(
        symbol,
        start,
        end,
        timeframes=timeframes,
        cache_dir=cache_dir,
        base_interval=base_interval,
        atr_length=14,
    )
    state_frames: dict[str, pd.DataFrame] = {}
    state_groups: dict[str, dict[str, list[str]]] = {}
    for timeframe in timeframes:
        tf_spec = TimeframeFeatureSpec(
            timeframe=timeframe,
            reversal_mult=reversal_map[timeframe],
            estimator="theil_sen",
            structural_point_count=5,
            min_points=3,
            body_envelope_lookback=12,
            body_envelope_min_separation=2,
            body_envelope_min_move_atr=0.1,
            touch_epsilon_atr=0.2,
            touch_lookback_bars=20,
            persistence_lookback_bars=20,
        )
        state_frame, groups = build_timeframe_state_frame(market.bars_by_timeframe[timeframe], tf_spec)
        state_frames[timeframe] = state_frame
        state_groups[timeframe] = groups

    decision_frame, decision_groups = build_decision_dataset(
        state_frames,
        state_groups,
        decision_timeframe=decision_timeframe,
        context_timeframes=[timeframe for timeframe in timeframes if timeframe != decision_timeframe],
    )
    return market, decision_frame, decision_groups


def build_template_dataset(
    *,
    market,
    decision_frame: pd.DataFrame,
    decision_groups: dict[str, list[str]],
    decision_timeframe: str,
    timeframes: list[str],
    template: TemplateSpec,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    event_frame, feature_groups = build_zone_channel_event_dataset(
        symbol=market.symbol,
        exec_frame=market.bars_by_timeframe[decision_timeframe],
        decision_frame=decision_frame,
        feature_groups=decision_groups,
        spec=ZoneChannelEventSpec(
            zone_timeframes=("1h",),
            zone_left=5,
            zone_right=5,
            zone_ob_search_bars=50,
            zone_penetration_frac=0.5,
            min_reclaim_pos=0.6,
            max_zone_scan=50,
            confluence_epsilon_atr=0.5,
            entry_mode=template.entry_mode,
            passive_entry_window_bars=template.passive_entry_window_bars,
            passive_entry_buffer_atr=template.passive_entry_buffer_atr,
            stop_mode=template.stop_mode,
            stop_buffer_atr=template.stop_buffer_atr,
            target_buffer_atr=template.target_buffer_atr,
            label_horizon_bars=template.label_horizon_bars,
            fee_bps_side=5.0,
            slippage_bps_side=2.0,
            channel_timeframes=tuple(timeframes),
            execution_timeframe=decision_timeframe,
        ),
    )
    return event_frame, feature_groups


def run_template(
    *,
    event_frame: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    template: TemplateSpec,
    symbol: str,
    start: str,
    end: str,
    timeframes: list[str],
    decision_timeframe: str,
    train_months: int,
    val_months: int,
    test_months: int,
    embargo_bars: int,
    min_validation_trades: int,
    output_prefix: Path,
) -> dict[str, Any]:
    filtered = apply_event_filters(
        event_frame,
        long_min_pos_in_body_1d=None,
        long_max_pos_in_body_1d=None,
        short_min_pos_in_body_1d=None,
        short_max_pos_in_body_1d=None,
        long_max_target_rr=None,
        short_max_target_rr=None,
    )
    raw_event_metrics = aggregate_raw_event_metrics(filtered)

    feature_columns = select_feature_columns(
        filtered,
        feature_groups,
        ("zone_context", "zone_reaction", "zone_confluence", "position", "confluence", "regime"),
    )
    results = run_walkforward_event_study(
        dataset=filtered,
        feature_columns=feature_columns,
        model_name="rf",
        train_months=train_months,
        val_months=val_months,
        test_months=test_months,
        embargo_bars=embargo_bars,
        long_thresholds=resolve_thresholds("0.45,0.50,0.55,0.60,0.65", "0.45,0.50,0.55,0.60,0.65", "probability"),
        short_thresholds=resolve_thresholds("0.45,0.50,0.55,0.60,0.65", "0.45,0.50,0.55,0.60,0.65", "probability"),
        long_score_mode="probability",
        short_score_mode="probability",
        selection_gates=(),
        risk_fraction=0.01,
        min_validation_trades=min_validation_trades,
    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_prefix.with_name(output_prefix.name + "_dataset.csv"), index=False)
    results["folds"].to_csv(output_prefix.with_name(output_prefix.name + "_folds.csv"), index=False)
    results["thresholds"].to_csv(output_prefix.with_name(output_prefix.name + "_thresholds.csv"), index=False)
    results["predictions"].to_csv(output_prefix.with_name(output_prefix.name + "_predictions.csv"), index=False)
    results["trades"].to_csv(output_prefix.with_name(output_prefix.name + "_trades.csv"), index=False)
    results["feature_importance"].to_csv(output_prefix.with_name(output_prefix.name + "_feature_importance.csv"), index=False)
    pd.DataFrame({"feature": feature_columns}).to_csv(output_prefix.with_name(output_prefix.name + "_selected_features.csv"), index=False)

    config_payload = {
        "symbol": symbol,
        "start": start,
        "end": end,
        "timeframes": timeframes,
        "zone_timeframes": ["1h"],
        "decision_timeframe": decision_timeframe,
        "channel_estimator": "theil_sen",
        "point_count": 5,
        "confluence_epsilon_atr": 0.5,
        "entry_mode": template.entry_mode,
        "passive_entry_window_bars": template.passive_entry_window_bars,
        "passive_entry_buffer_atr": template.passive_entry_buffer_atr,
        "stop_mode": template.stop_mode,
        "stop_buffer_atr": template.stop_buffer_atr,
        "target_buffer_atr": template.target_buffer_atr,
        "label_horizon_bars": template.label_horizon_bars,
        "long_score_mode": "probability",
        "short_score_mode": "probability",
        "long_min_pos_in_body_1d": None,
        "long_max_pos_in_body_1d": None,
        "short_min_pos_in_body_1d": None,
        "short_max_pos_in_body_1d": None,
        "long_max_target_rr": None,
        "short_max_target_rr": None,
        "selection_gates": [],
        "template_name": template.name,
    }
    output_prefix.with_name(output_prefix.name + "_config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    output_prefix.with_name(output_prefix.name + "_report.md").write_text(build_report(config_payload, results), encoding="utf-8")

    trade_metrics = aggregate_trade_metrics(results["trades"], results["folds"])
    return {
        "template": template.name,
        "entry_mode": template.entry_mode,
        "stop_mode": template.stop_mode,
        "label_horizon_bars": template.label_horizon_bars,
        "target_buffer_atr": template.target_buffer_atr,
        "stop_buffer_atr": template.stop_buffer_atr,
        "feature_count": len(feature_columns),
        **raw_event_metrics,
        **trade_metrics,
    }


def aggregate_raw_event_metrics(event_frame: pd.DataFrame) -> dict[str, Any]:
    if event_frame.empty:
        return {
            "raw_event_rows": 0,
            "raw_event_hold_rate": 0.0,
            "raw_event_net_r": 0.0,
            "raw_event_avg_r": 0.0,
        }
    future_r_net = pd.to_numeric(event_frame["future_r_net"], errors="coerce").fillna(0.0)
    hold_label = pd.to_numeric(event_frame["hold_label"], errors="coerce").fillna(0.0)
    return {
        "raw_event_rows": int(len(event_frame)),
        "raw_event_hold_rate": float(hold_label.mean()),
        "raw_event_net_r": float(future_r_net.sum()),
        "raw_event_avg_r": float(future_r_net.mean()),
    }


def aggregate_trade_metrics(trades: pd.DataFrame, folds: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {
            "folds": 0,
            "trades": 0,
            "trades_per_30d": 0.0,
            "net_r": 0.0,
            "profit_factor": 0.0,
            "hit_rate": 0.0,
            "average_trade": 0.0,
            "total_return": 0.0,
        }

    trade_r = pd.to_numeric(trades["r_multiple_net"], errors="coerce").fillna(0.0)
    winners = trade_r[trade_r > 0.0].sum()
    losers = -trade_r[trade_r < 0.0].sum()

    actual_folds = folds[folds["fold"].astype(str) != "aggregate"].copy() if not folds.empty and "fold" in folds.columns else pd.DataFrame()
    test_days = 0.0
    if not actual_folds.empty:
        val_end = pd.to_datetime(actual_folds["val_end"], utc=True, errors="coerce")
        test_end = pd.to_datetime(actual_folds["test_end"], utc=True, errors="coerce")
        test_days = float(((test_end - val_end).dt.total_seconds() / 86400.0).sum())

    trades_per_30d = float(len(trades) / test_days * 30.0) if test_days > 0.0 else 0.0
    aggregate_row = actual_folds.iloc[0:0]
    aggregate = folds[folds["fold"].astype(str) == "aggregate"] if not folds.empty and "fold" in folds.columns else aggregate_row
    total_return = float(aggregate.iloc[0]["trade_total_return"]) if not aggregate.empty and "trade_total_return" in aggregate.columns else 0.0

    return {
        "folds": int(len(actual_folds)),
        "trades": int(len(trades)),
        "trades_per_30d": trades_per_30d,
        "net_r": float(trade_r.sum()),
        "profit_factor": float(winners / losers) if losers > 0.0 else float("inf"),
        "hit_rate": float((trade_r > 0.0).mean()),
        "average_trade": float(trade_r.mean()),
        "total_return": total_return,
    }


if __name__ == "__main__":
    main()
