from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.channel_state_research.data import build_market_dataset
from scripts.channel_state_research.features import (
    TimeframeFeatureSpec,
    build_decision_dataset,
    build_timeframe_state_frame,
)
from scripts.channel_state_research.labels import add_raw_future_return_labels, add_triple_barrier_labels
from scripts.channel_state_research.modeling import WalkForwardSpec, run_walkforward_study


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-timeframe channel-state study for BTCUSDT.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", default="2021-09-01")
    parser.add_argument("--end", default="2026-04-20")
    parser.add_argument("--dataset-path", type=Path, help="Optional prebuilt dataset CSV from a previous channel-state run.")
    parser.add_argument("--feature-groups-path", type=Path, help="Optional feature-groups JSON matching --dataset-path.")
    parser.add_argument("--cache-dir", type=Path, default=Path("scripts/.cache"))
    parser.add_argument("--base-interval", default="5m")
    parser.add_argument("--timeframes", default="1h,4h,1d,1w")
    parser.add_argument("--decision-timeframe", default="1h")
    parser.add_argument("--atr-length", type=int, default=14)
    parser.add_argument("--channel-estimator", choices=["theil_sen", "ols", "ransac", "two_point"], default="theil_sen")
    parser.add_argument("--point-count", type=int, default=5)
    parser.add_argument("--min-points", type=int, default=3)
    parser.add_argument("--reversal-5m", type=float, default=1.5)
    parser.add_argument("--reversal-15m", type=float, default=1.5)
    parser.add_argument("--reversal-1h", type=float, default=2.0)
    parser.add_argument("--reversal-4h", type=float, default=2.0)
    parser.add_argument("--reversal-1d", type=float, default=2.0)
    parser.add_argument("--reversal-1w", type=float, default=1.5)
    parser.add_argument("--body-envelope-lookback", type=int, default=12)
    parser.add_argument("--body-envelope-min-separation", type=int, default=2)
    parser.add_argument("--body-envelope-min-move-atr", type=float, default=0.1)
    parser.add_argument("--touch-epsilon-atr", type=float, default=0.2)
    parser.add_argument("--touch-lookback-bars", type=int, default=20)
    parser.add_argument("--persistence-lookback-bars", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--beta", type=float, default=1.5)
    parser.add_argument("--horizon-bars", type=int, default=24)
    parser.add_argument("--model", choices=["logreg", "rf", "hgb"], default="hgb")
    parser.add_argument("--channel-family", choices=["wick", "body", "both"], default="both")
    parser.add_argument(
        "--feature-groups",
        default="structural,position,excursion_acceptance,touch_interaction,swing_state,channel_evolution,confluence,regime",
    )
    parser.add_argument("--train-months", type=int, default=24)
    parser.add_argument("--val-months", type=int, default=6)
    parser.add_argument("--test-months", type=int, default=6)
    parser.add_argument("--embargo-bars", type=int, default=24)
    parser.add_argument("--threshold-mode", choices=["absolute", "percentile"], default="absolute")
    parser.add_argument("--long-score-mode", choices=["probability", "edge", "logit", "logit_edge"], default="probability")
    parser.add_argument("--short-score-mode", choices=["probability", "edge", "logit", "logit_edge"], default="probability")
    parser.add_argument("--long-thresholds", default="0.55,0.60,0.65")
    parser.add_argument("--short-thresholds", default="0.55,0.60,0.65")
    parser.add_argument("--probability-gap-values", default="0.0")
    parser.add_argument("--gate-presets", default="none")
    parser.add_argument("--fee-bps-side", type=float, default=5.0)
    parser.add_argument("--slippage-bps-side", type=float, default=2.0)
    parser.add_argument("--risk-fraction", type=float, default=0.01)
    parser.add_argument("--min-validation-trades", type=int, default=3)
    parser.add_argument("--output-prefix", type=Path, default=Path("scripts/channel_state_btcusdt_baseline"))
    parser.add_argument("--skip-study", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timeframes = [item.strip() for item in args.timeframes.split(",") if item.strip()]
    if args.decision_timeframe not in timeframes:
        raise ValueError(f"--decision-timeframe {args.decision_timeframe!r} must be included in --timeframes.")
    if bool(args.dataset_path) != bool(args.feature_groups_path):
        raise ValueError("--dataset-path and --feature-groups-path must be provided together.")

    reversal_map = {
        "5m": args.reversal_5m,
        "15m": args.reversal_15m,
        "1h": args.reversal_1h,
        "4h": args.reversal_4h,
        "1d": args.reversal_1d,
        "1w": args.reversal_1w,
    }
    if args.dataset_path and args.feature_groups_path:
        decision_frame = load_dataset(args.dataset_path)
        feature_groups = json.loads(args.feature_groups_path.read_text(encoding="utf-8"))
        market_symbol = args.symbol
        available_timeframes = infer_available_timeframes(decision_frame, feature_groups)
        print(f"Loaded prebuilt dataset from {args.dataset_path}", flush=True)
    else:
        market = build_market_dataset(
            args.symbol,
            args.start,
            args.end,
            timeframes=timeframes,
            cache_dir=args.cache_dir,
            base_interval=args.base_interval,
            atr_length=args.atr_length,
        )

        state_frames: dict[str, pd.DataFrame] = {}
        state_groups: dict[str, dict[str, list[str]]] = {}
        for timeframe in timeframes:
            tf_spec = TimeframeFeatureSpec(
                timeframe=timeframe,
                reversal_mult=reversal_map[timeframe],
                estimator=args.channel_estimator,
                structural_point_count=args.point_count,
                min_points=args.min_points,
                body_envelope_lookback=args.body_envelope_lookback,
                body_envelope_min_separation=args.body_envelope_min_separation,
                body_envelope_min_move_atr=args.body_envelope_min_move_atr,
                touch_epsilon_atr=args.touch_epsilon_atr,
                touch_lookback_bars=args.touch_lookback_bars,
                persistence_lookback_bars=args.persistence_lookback_bars,
            )
            state_frame, groups = build_timeframe_state_frame(market.bars_by_timeframe[timeframe], tf_spec)
            state_frames[timeframe] = state_frame
            state_groups[timeframe] = groups
            print(f"{timeframe}: built {len(state_frame)} causal state rows", flush=True)

        decision_frame, feature_groups = build_decision_dataset(
            state_frames,
            state_groups,
            decision_timeframe=args.decision_timeframe,
            context_timeframes=[timeframe for timeframe in timeframes if timeframe != args.decision_timeframe],
        )
        decision_frame = add_triple_barrier_labels(
            decision_frame,
            close_column=f"close_tf_{args.decision_timeframe}",
            open_column=f"open_tf_{args.decision_timeframe}",
            high_column=f"high_tf_{args.decision_timeframe}",
            low_column=f"low_tf_{args.decision_timeframe}",
            atr_column=f"atr_tf_{args.decision_timeframe}",
            alpha=args.alpha,
            beta=args.beta,
            horizon_bars=args.horizon_bars,
        )
        decision_frame = add_raw_future_return_labels(
            decision_frame,
            close_column=f"close_tf_{args.decision_timeframe}",
            horizon_bars=args.horizon_bars,
        )
        market_symbol = market.symbol
        available_timeframes = timeframes

    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    dataset_path = output_prefix.with_name(output_prefix.name + "_dataset.csv")
    decision_frame.to_csv(dataset_path, index=False)
    feature_group_path = output_prefix.with_name(output_prefix.name + "_feature_groups.json")
    feature_group_path.write_text(json.dumps(feature_groups, indent=2), encoding="utf-8")

    config_payload = {
        "symbol": market_symbol,
        "start": args.start,
        "end": args.end,
        "timeframes": timeframes,
        "decision_timeframe": args.decision_timeframe,
        "atr_length": args.atr_length,
        "channel_estimator": args.channel_estimator,
        "point_count": args.point_count,
        "min_points": args.min_points,
        "reversal_map": reversal_map,
        "alpha": args.alpha,
        "beta": args.beta,
        "horizon_bars": args.horizon_bars,
        "threshold_mode": args.threshold_mode,
        "long_score_mode": args.long_score_mode,
        "short_score_mode": args.short_score_mode,
    }
    config_path = output_prefix.with_name(output_prefix.name + "_config.json")
    config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    print(f"Saved dataset to {dataset_path}", flush=True)
    print(f"Saved feature groups to {feature_group_path}", flush=True)

    if args.skip_study:
        return

    study_spec = WalkForwardSpec(
        model_name=args.model,
        feature_group_names=tuple(item.strip() for item in args.feature_groups.split(",") if item.strip()),
        channel_family=args.channel_family,
        timeframes=tuple(timeframes),
        decision_timeframe=args.decision_timeframe,
        train_months=args.train_months,
        val_months=args.val_months,
        test_months=args.test_months,
        embargo_bars=args.embargo_bars,
        alpha=args.alpha,
        beta=args.beta,
        horizon_bars=args.horizon_bars,
        threshold_mode=args.threshold_mode,
        long_score_mode=args.long_score_mode,
        short_score_mode=args.short_score_mode,
        long_thresholds=tuple(float(item.strip()) for item in args.long_thresholds.split(",") if item.strip()),
        short_thresholds=tuple(float(item.strip()) for item in args.short_thresholds.split(",") if item.strip()),
        probability_gap_values=tuple(float(item.strip()) for item in args.probability_gap_values.split(",") if item.strip()),
        gate_presets=tuple(item.strip() for item in args.gate_presets.split(",") if item.strip()),
        fee_bps_side=args.fee_bps_side,
        slippage_bps_side=args.slippage_bps_side,
        risk_fraction=args.risk_fraction,
        min_validation_trades=args.min_validation_trades,
    )
    results = run_walkforward_study(decision_frame, feature_groups, study_spec, all_timeframes=available_timeframes)

    folds_path = output_prefix.with_name(output_prefix.name + "_folds.csv")
    thresholds_path = output_prefix.with_name(output_prefix.name + "_thresholds.csv")
    predictions_path = output_prefix.with_name(output_prefix.name + "_predictions.csv")
    trades_path = output_prefix.with_name(output_prefix.name + "_trades.csv")
    importance_path = output_prefix.with_name(output_prefix.name + "_feature_importance.csv")
    selected_feature_path = output_prefix.with_name(output_prefix.name + "_selected_features.csv")
    report_path = output_prefix.with_name(output_prefix.name + "_report.md")

    results["folds"].to_csv(folds_path, index=False)
    results["thresholds"].to_csv(thresholds_path, index=False)
    results["predictions"].to_csv(predictions_path, index=False)
    results["trades"].to_csv(trades_path, index=False)
    results["feature_importance"].to_csv(importance_path, index=False)
    results["feature_columns"].to_csv(selected_feature_path, index=False)
    report_path.write_text(build_report(config_payload, results), encoding="utf-8")

    if not results["folds"].empty:
        print()
        print(results["folds"].to_string(index=False))
    print(f"\nSaved folds to {folds_path}")
    print(f"Saved thresholds to {thresholds_path}")
    print(f"Saved predictions to {predictions_path}")
    print(f"Saved trades to {trades_path}")
    print(f"Saved feature importance to {importance_path}")
    print(f"Saved report to {report_path}")


def build_report(config_payload: dict, results: dict[str, pd.DataFrame]) -> str:
    folds = results["folds"]
    aggregate = folds[folds["fold"].astype(str) == "aggregate"] if "fold" in folds.columns else pd.DataFrame()
    aggregate_row = aggregate.iloc[0].to_dict() if not aggregate.empty else {}

    lines = [
        "# Channel-State Study",
        "",
        "## Configuration",
        "",
        f"- symbol: `{config_payload['symbol']}`",
        f"- window: `{config_payload['start']}` to `{config_payload['end']}`",
        f"- timeframes: `{', '.join(config_payload['timeframes'])}`",
        f"- decision timeframe: `{config_payload['decision_timeframe']}`",
        f"- estimator: `{config_payload['channel_estimator']}`",
        f"- point count: `{config_payload['point_count']}`",
        f"- labels: `alpha={config_payload['alpha']}`, `beta={config_payload['beta']}`, `horizon={config_payload['horizon_bars']}`",
        f"- threshold mode: `{aggregate_row.get('threshold_mode', config_payload.get('threshold_mode', 'absolute'))}`",
        f"- score modes: `long={aggregate_row.get('long_score_mode', config_payload.get('long_score_mode', 'probability'))}`, `short={aggregate_row.get('short_score_mode', config_payload.get('short_score_mode', 'probability'))}`",
        "",
        "## Aggregate",
        "",
    ]
    if aggregate_row:
        for key in [
            "trade_trades",
            "trade_total_return",
            "trade_sharpe",
            "trade_sortino",
            "trade_max_drawdown",
            "trade_calmar",
            "trade_hit_rate",
            "trade_profit_factor",
            "trade_long_only_return",
            "trade_short_only_return",
            "trade_long_threshold",
            "trade_short_threshold",
            "trade_probability_gap",
            "trade_gate_preset",
            "long_test_auc",
            "short_test_auc",
        ]:
            if key in aggregate_row:
                lines.append(f"- {key}: `{aggregate_row[key]}`")
    else:
        lines.append("- No aggregate row was produced.")

    lines.extend(
        [
            "",
            "## Top Features",
            "",
        ]
    )
    importance = results["feature_importance"]
    if not importance.empty:
        top = (
            importance.groupby(["direction", "feature"], as_index=False)["importance"]
            .mean()
            .sort_values(["direction", "importance"], ascending=[True, False])
        )
        for direction in sorted(top["direction"].unique()):
            lines.append(f"### {direction.title()}")
            lines.append("")
            for _, row in top[top["direction"] == direction].head(10).iterrows():
                lines.append(f"- `{row['feature']}`: `{row['importance']:.6f}`")
            lines.append("")
    else:
        lines.append("- Feature importance was not available.")

    return "\n".join(lines).strip() + "\n"


def load_dataset(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in ["close_time", "decision_time", "label_timeout_time", "label_hit_time", "label_end_time"]:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    return frame


def infer_available_timeframes(frame: pd.DataFrame, feature_groups: dict[str, list[str]]) -> list[str]:
    pattern = re.compile(r"(?:_tf_|_)(\d+[mhdw])$")
    candidates: set[str] = set()
    for column in frame.columns:
        match = pattern.search(column)
        if match:
            candidates.add(match.group(1))
    for columns in feature_groups.values():
        for column in columns:
            match = pattern.search(column)
            if match:
                candidates.add(match.group(1))
    if not candidates:
        return ["1h", "4h", "1d", "1w"]
    order = {"1m": 0, "5m": 1, "15m": 2, "30m": 3, "1h": 4, "4h": 5, "1d": 6, "1w": 7}
    return sorted(candidates, key=lambda item: (order.get(item, 99), item))


if __name__ == "__main__":
    main()
