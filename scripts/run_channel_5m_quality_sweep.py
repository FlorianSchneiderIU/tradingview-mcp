from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
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
    parse_selection_gates,
    resolve_thresholds,
    run_walkforward_event_study,
    select_feature_columns,
)


@dataclass(frozen=True)
class SweepCandidate:
    name: str
    model: str
    feature_groups: tuple[str, ...]
    long_thresholds: str = "0.45,0.50,0.55,0.60"
    short_thresholds: str = "0.45,0.50,0.55,0.60"
    long_score_mode: str = "probability"
    short_score_mode: str = "probability"
    selection_gates: tuple[str, ...] = ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Focused 5m channel-quality sweep.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-04-20")
    parser.add_argument("--cache-dir", type=Path, default=Path("scripts/.cache"))
    parser.add_argument("--base-interval", default="5m")
    parser.add_argument("--timeframes", default="5m,15m,1h")
    parser.add_argument("--decision-timeframe", default="5m")
    parser.add_argument("--zone-timeframes", default="15m,1h")
    parser.add_argument("--output-prefix", type=Path, default=Path("scripts/channel_5m_quality_sweep"))
    parser.add_argument("--train-months", type=int, default=1)
    parser.add_argument("--val-months", type=int, default=1)
    parser.add_argument("--test-months", type=int, default=1)
    parser.add_argument("--embargo-bars", type=int, default=24)
    parser.add_argument("--min-validation-trades", type=int, default=5)
    parser.add_argument("--skip-dataset-build", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timeframes = [item.strip() for item in args.timeframes.split(",") if item.strip()]
    zone_timeframes = tuple(item.strip() for item in args.zone_timeframes.split(",") if item.strip())
    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    dataset_path = output_prefix.with_name(output_prefix.name + "_dataset.csv")
    feature_groups_path = output_prefix.with_name(output_prefix.name + "_feature_groups.json")
    config_path = output_prefix.with_name(output_prefix.name + "_dataset_config.json")

    if args.skip_dataset_build:
        dataset = pd.read_csv(dataset_path, parse_dates=["event_time", "entry_time", "exit_time", "zone_time"])
        feature_groups = json.loads(feature_groups_path.read_text(encoding="utf-8"))
    else:
        dataset, feature_groups = build_dataset(
            symbol=args.symbol,
            start=args.start,
            end=args.end,
            cache_dir=args.cache_dir,
            base_interval=args.base_interval,
            timeframes=timeframes,
            decision_timeframe=args.decision_timeframe,
            zone_timeframes=zone_timeframes,
        )
        dataset.to_csv(dataset_path, index=False)
        feature_groups_path.write_text(json.dumps(feature_groups, indent=2), encoding="utf-8")
        config_path.write_text(
            json.dumps(
                {
                    "symbol": args.symbol,
                    "start": args.start,
                    "end": args.end,
                    "timeframes": timeframes,
                    "decision_timeframe": args.decision_timeframe,
                    "zone_timeframes": list(zone_timeframes),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    if dataset.empty:
        print("No events were generated; stopping.")
        return

    print(f"Dataset rows: {len(dataset)}")
    print(dataset["zone_tf"].value_counts().to_string())
    print(dataset["direction"].value_counts().to_string())

    candidates = [
        SweepCandidate(
            name="hgb_quality_core",
            model="hgb",
            feature_groups=("zone_context", "zone_reaction", "zone_confluence", "position", "confluence", "regime"),
        ),
        SweepCandidate(
            name="rf_quality_core",
            model="rf",
            feature_groups=("zone_context", "zone_reaction", "zone_confluence", "position", "confluence", "regime"),
        ),
        SweepCandidate(
            name="hgb_quality_plus_structure",
            model="hgb",
            feature_groups=("zone_context", "zone_reaction", "zone_confluence", "structural", "position", "touch_interaction", "confluence", "regime"),
        ),
        SweepCandidate(
            name="logreg_quality_core",
            model="logreg",
            feature_groups=("zone_context", "zone_reaction", "zone_confluence", "position", "confluence", "regime"),
        ),
        SweepCandidate(
            name="rf_quality_core_outer15m",
            model="rf",
            feature_groups=("zone_context", "zone_reaction", "zone_confluence", "position", "confluence", "regime"),
            selection_gates=(
                "long:pos_in_body_15m<=0.20",
                "short:pos_in_body_15m>=0.80",
            ),
        ),
        SweepCandidate(
            name="rf_quality_core_outer15m_age",
            model="rf",
            feature_groups=("zone_context", "zone_reaction", "zone_confluence", "position", "confluence", "regime"),
            selection_gates=(
                "long:pos_in_body_15m<=0.20",
                "short:pos_in_body_15m>=0.80",
                "long:zone_age_hours<=12",
                "short:zone_age_hours>=4",
            ),
        ),
    ]

    summary_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        print(f"\nRunning {candidate.name} ...", flush=True)
        results = run_candidate(
            dataset=dataset,
            feature_groups=feature_groups,
            candidate=candidate,
            study_context={
                "symbol": args.symbol,
                "start": args.start,
                "end": args.end,
                "timeframes": timeframes,
                "zone_timeframes": list(zone_timeframes),
                "decision_timeframe": args.decision_timeframe,
                "channel_estimator": "theil_sen",
                "point_count": 5,
                "confluence_epsilon_atr": 0.5,
                "entry_mode": "market_reclaim",
                "passive_entry_window_bars": 6,
                "passive_entry_buffer_atr": 0.0,
                "stop_mode": "zone",
                "stop_buffer_atr": 0.2,
                "target_buffer_atr": 0.2,
                "label_horizon_bars": 24,
                "long_min_pos_in_body_1d": None,
                "long_max_pos_in_body_1d": None,
                "short_min_pos_in_body_1d": None,
                "short_max_pos_in_body_1d": None,
                "long_max_target_rr": None,
                "short_max_target_rr": None,
            },
            train_months=args.train_months,
            val_months=args.val_months,
            test_months=args.test_months,
            embargo_bars=args.embargo_bars,
            min_validation_trades=args.min_validation_trades,
            output_prefix=output_prefix.with_name(output_prefix.name + "_" + candidate.name),
        )
        summary_rows.append(results)

    summary = pd.DataFrame(summary_rows).sort_values(["net_r", "profit_factor", "trades"], ascending=[False, False, False])
    summary_path = output_prefix.with_name(output_prefix.name + "_summary.csv")
    summary.to_csv(summary_path, index=False)

    report_lines = [
        "# 5m Channel Quality Sweep",
        "",
        f"- symbol: `{args.symbol}`",
        f"- window: `{args.start}` to `{args.end}`",
        f"- timeframes: `{', '.join(timeframes)}`",
        f"- zone timeframes: `{', '.join(zone_timeframes)}`",
        f"- decision timeframe: `{args.decision_timeframe}`",
        f"- dataset rows: `{len(dataset)}`",
        "",
        "## Candidates",
        "",
    ]
    for row in summary.to_dict(orient="records"):
        report_lines.extend(
            [
                f"- `{row['candidate']}`: trades `{row['trades']}`, net_r `{row['net_r']}`, PF `{row['profit_factor']}`, hit_rate `{row['hit_rate']}`",
            ]
        )
    report_path = output_prefix.with_name(output_prefix.name + "_report.md")
    report_path.write_text("\n".join(report_lines).strip() + "\n", encoding="utf-8")

    print("\nSummary")
    print(summary.to_string(index=False))
    print(f"\nSaved summary to {summary_path}")
    print(f"Saved report to {report_path}")


def build_dataset(
    *,
    symbol: str,
    start: str,
    end: str,
    cache_dir: Path,
    base_interval: str,
    timeframes: list[str],
    decision_timeframe: str,
    zone_timeframes: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
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
    event_frame, feature_groups = build_zone_channel_event_dataset(
        symbol=market.symbol,
        exec_frame=market.bars_by_timeframe[decision_timeframe],
        decision_frame=decision_frame,
        feature_groups=decision_groups,
        spec=ZoneChannelEventSpec(
            zone_timeframes=zone_timeframes,
            zone_left=5,
            zone_right=5,
            zone_ob_search_bars=50,
            zone_penetration_frac=0.5,
            min_reclaim_pos=0.6,
            max_zone_scan=50,
            confluence_epsilon_atr=0.5,
            entry_mode="market_reclaim",
            passive_entry_window_bars=6,
            passive_entry_buffer_atr=0.0,
            stop_mode="zone",
            stop_buffer_atr=0.2,
            target_buffer_atr=0.2,
            label_horizon_bars=24,
            fee_bps_side=5.0,
            slippage_bps_side=2.0,
            channel_timeframes=tuple(timeframes),
            execution_timeframe=decision_timeframe,
        ),
    )
    return event_frame, feature_groups


def run_candidate(
    *,
    dataset: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    candidate: SweepCandidate,
    study_context: dict[str, Any],
    train_months: int,
    val_months: int,
    test_months: int,
    embargo_bars: int,
    min_validation_trades: int,
    output_prefix: Path,
) -> dict[str, Any]:
    filtered = apply_event_filters(
        dataset,
        long_min_pos_in_body_1d=None,
        long_max_pos_in_body_1d=None,
        short_min_pos_in_body_1d=None,
        short_max_pos_in_body_1d=None,
        long_max_target_rr=None,
        short_max_target_rr=None,
    )
    feature_columns = select_feature_columns(filtered, feature_groups, candidate.feature_groups)
    results = run_walkforward_event_study(
        dataset=filtered,
        feature_columns=feature_columns,
        model_name=candidate.model,
        train_months=train_months,
        val_months=val_months,
        test_months=test_months,
        embargo_bars=embargo_bars,
        long_thresholds=resolve_thresholds(candidate.long_thresholds, candidate.long_thresholds, candidate.long_score_mode),
        short_thresholds=resolve_thresholds(candidate.short_thresholds, candidate.short_thresholds, candidate.short_score_mode),
        long_score_mode=candidate.long_score_mode,
        short_score_mode=candidate.short_score_mode,
        selection_gates=parse_selection_gates(list(candidate.selection_gates)),
        risk_fraction=0.01,
        min_validation_trades=min_validation_trades,
    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    results["folds"].to_csv(output_prefix.with_name(output_prefix.name + "_folds.csv"), index=False)
    results["thresholds"].to_csv(output_prefix.with_name(output_prefix.name + "_thresholds.csv"), index=False)
    results["predictions"].to_csv(output_prefix.with_name(output_prefix.name + "_predictions.csv"), index=False)
    results["trades"].to_csv(output_prefix.with_name(output_prefix.name + "_trades.csv"), index=False)
    results["feature_importance"].to_csv(output_prefix.with_name(output_prefix.name + "_feature_importance.csv"), index=False)
    pd.DataFrame({"feature": feature_columns}).to_csv(output_prefix.with_name(output_prefix.name + "_selected_features.csv"), index=False)

    config_payload = {
        **study_context,
        "candidate": candidate.name,
        "model": candidate.model,
        "feature_groups": list(candidate.feature_groups),
        "long_thresholds": candidate.long_thresholds,
        "short_thresholds": candidate.short_thresholds,
        "long_score_mode": candidate.long_score_mode,
        "short_score_mode": candidate.short_score_mode,
        "selection_gates": list(candidate.selection_gates),
        "train_months": train_months,
        "val_months": val_months,
        "test_months": test_months,
        "embargo_bars": embargo_bars,
        "min_validation_trades": min_validation_trades,
    }
    output_prefix.with_name(output_prefix.name + "_config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    output_prefix.with_name(output_prefix.name + "_report.md").write_text(build_report(config_payload, results), encoding="utf-8")

    folds = results["folds"]
    trades = results["trades"]
    aggregate = aggregate_trade_metrics(trades)
    actual_fold_count = 0 if folds.empty or "fold" not in folds.columns else int((folds["fold"].astype(str) != "aggregate").sum())
    return {
        "candidate": candidate.name,
        "model": candidate.model,
        "feature_count": len(feature_columns),
        "folds": actual_fold_count,
        **aggregate,
    }


def aggregate_trade_metrics(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {
            "trades": 0,
            "net_r": 0.0,
            "profit_factor": 0.0,
            "hit_rate": 0.0,
            "average_trade": 0.0,
        }
    trade_r = pd.to_numeric(trades["r_multiple_net"], errors="coerce").fillna(0.0)
    winners = trade_r[trade_r > 0.0].sum()
    losers = -trade_r[trade_r < 0.0].sum()
    return {
        "trades": int(len(trades)),
        "net_r": float(trade_r.sum()),
        "profit_factor": float(winners / losers) if losers > 0.0 else float("inf"),
        "hit_rate": float((trade_r > 0.0).mean()),
        "average_trade": float(trade_r.mean()),
    }


if __name__ == "__main__":
    main()
