from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.channel_state_research.backtest import strategy_metrics
from scripts.channel_state_research.data import build_market_dataset
from scripts.channel_state_research.features import TimeframeFeatureSpec, build_decision_dataset, build_timeframe_state_frame
from scripts.channel_state_research.modeling import binary_metrics, fit_binary_model, predict_binary_model
from scripts.channel_state_research.zone_confluence import ZoneChannelEventSpec, build_zone_channel_event_dataset

SELECTION_GATE_PATTERN = re.compile(
    r"^(long|short):([A-Za-z0-9_]+)\s*(<=|>=|==|!=|<|>)\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zone-channel confluence study for BTCUSDT.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", default="2021-09-01")
    parser.add_argument("--end", default="2026-04-20")
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--feature-groups-path", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path("scripts/.cache"))
    parser.add_argument("--base-interval", default="5m")
    parser.add_argument("--timeframes", default="1h,4h,1d")
    parser.add_argument("--decision-timeframe", default="1h")
    parser.add_argument("--zone-timeframes", default="4h,1d")
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
    parser.add_argument("--zone-left", type=int, default=5)
    parser.add_argument("--zone-right", type=int, default=5)
    parser.add_argument("--zone-ob-search-bars", type=int, default=50)
    parser.add_argument("--zone-penetration-frac", type=float, default=0.50)
    parser.add_argument("--min-reclaim-pos", type=float, default=0.60)
    parser.add_argument("--max-zone-scan", type=int, default=50)
    parser.add_argument("--confluence-epsilon-atr", type=float, default=0.50)
    parser.add_argument("--entry-mode", choices=["market_reclaim", "passive_retest"], default="market_reclaim")
    parser.add_argument("--passive-entry-window-bars", type=int, default=6)
    parser.add_argument("--passive-entry-buffer-atr", type=float, default=0.0)
    parser.add_argument("--stop-mode", choices=["channel_anchor", "zone", "reaction_extreme"], default="channel_anchor")
    parser.add_argument("--stop-buffer-atr", type=float, default=0.20)
    parser.add_argument("--target-buffer-atr", type=float, default=0.20)
    parser.add_argument("--label-horizon-bars", type=int, default=24)
    parser.add_argument("--model", choices=["logreg", "rf", "hgb"], default="hgb")
    parser.add_argument(
        "--feature-groups",
        default="zone_context,zone_reaction,zone_confluence,structural,position,touch_interaction,confluence,regime",
    )
    parser.add_argument("--train-months", type=int, default=12)
    parser.add_argument("--val-months", type=int, default=3)
    parser.add_argument("--test-months", type=int, default=3)
    parser.add_argument("--embargo-bars", type=int, default=24)
    parser.add_argument("--thresholds", default="0.45,0.50,0.55,0.60,0.65")
    parser.add_argument("--long-thresholds", default="")
    parser.add_argument("--short-thresholds", default="")
    parser.add_argument("--long-score-mode", choices=["probability", "ev", "ev_net", "prob_x_rr"], default="probability")
    parser.add_argument("--short-score-mode", choices=["probability", "ev", "ev_net", "prob_x_rr"], default="probability")
    parser.add_argument("--long-min-pos-in-body-1d", type=float)
    parser.add_argument("--long-max-pos-in-body-1d", type=float)
    parser.add_argument("--short-min-pos-in-body-1d", type=float)
    parser.add_argument("--short-max-pos-in-body-1d", type=float)
    parser.add_argument("--long-max-target-rr", type=float)
    parser.add_argument("--short-max-target-rr", type=float)
    parser.add_argument(
        "--selection-gate",
        action="append",
        default=[],
        help="Optional post-score gate applied during validation/test selection, e.g. long:zone_mid_to_decision_close_atr<=2.5",
    )
    parser.add_argument("--fee-bps-side", type=float, default=5.0)
    parser.add_argument("--slippage-bps-side", type=float, default=2.0)
    parser.add_argument("--risk-fraction", type=float, default=0.01)
    parser.add_argument("--min-validation-trades", type=int, default=5)
    parser.add_argument("--output-prefix", type=Path, default=Path("scripts/zone_channel_confluence_btcusdt"))
    parser.add_argument("--skip-study", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timeframes = [item.strip() for item in args.timeframes.split(",") if item.strip()]
    if args.decision_timeframe not in timeframes:
        raise ValueError(f"--decision-timeframe {args.decision_timeframe!r} must be included in --timeframes.")
    zone_timeframes = tuple(item.strip() for item in args.zone_timeframes.split(",") if item.strip())
    if bool(args.dataset_path) != bool(args.feature_groups_path):
        raise ValueError("--dataset-path and --feature-groups-path must be provided together.")

    if args.dataset_path and args.feature_groups_path:
        event_frame = load_dataset(args.dataset_path)
        feature_groups = json.loads(args.feature_groups_path.read_text(encoding="utf-8"))
        market_symbol = args.symbol
        print(f"Loaded prebuilt event dataset from {args.dataset_path}", flush=True)
    else:
        reversal_map = {
            "5m": args.reversal_5m,
            "15m": args.reversal_15m,
            "1h": args.reversal_1h,
            "4h": args.reversal_4h,
            "1d": args.reversal_1d,
            "1w": args.reversal_1w,
        }
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

        decision_frame, decision_groups = build_decision_dataset(
            state_frames,
            state_groups,
            decision_timeframe=args.decision_timeframe,
            context_timeframes=[timeframe for timeframe in timeframes if timeframe != args.decision_timeframe],
        )
        event_frame, feature_groups = build_zone_channel_event_dataset(
            symbol=market.symbol,
            exec_frame=market.bars_by_timeframe[args.decision_timeframe],
            decision_frame=decision_frame,
            feature_groups=decision_groups,
            spec=ZoneChannelEventSpec(
                zone_timeframes=zone_timeframes,
                zone_left=args.zone_left,
                zone_right=args.zone_right,
                zone_ob_search_bars=args.zone_ob_search_bars,
                zone_penetration_frac=args.zone_penetration_frac,
                min_reclaim_pos=args.min_reclaim_pos,
                max_zone_scan=args.max_zone_scan,
                confluence_epsilon_atr=args.confluence_epsilon_atr,
                entry_mode=args.entry_mode,
                passive_entry_window_bars=args.passive_entry_window_bars,
                passive_entry_buffer_atr=args.passive_entry_buffer_atr,
                stop_mode=args.stop_mode,
                stop_buffer_atr=args.stop_buffer_atr,
                target_buffer_atr=args.target_buffer_atr,
                label_horizon_bars=args.label_horizon_bars,
                fee_bps_side=args.fee_bps_side,
                slippage_bps_side=args.slippage_bps_side,
                channel_timeframes=tuple(timeframes),
                execution_timeframe=args.decision_timeframe,
            ),
        )
        market_symbol = market.symbol

    selection_gates = parse_selection_gates(args.selection_gate)
    event_frame = apply_event_filters(
        event_frame,
        long_min_pos_in_body_1d=args.long_min_pos_in_body_1d,
        long_max_pos_in_body_1d=args.long_max_pos_in_body_1d,
        short_min_pos_in_body_1d=args.short_min_pos_in_body_1d,
        short_max_pos_in_body_1d=args.short_max_pos_in_body_1d,
        long_max_target_rr=args.long_max_target_rr,
        short_max_target_rr=args.short_max_target_rr,
    )

    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    dataset_path = output_prefix.with_name(output_prefix.name + "_dataset.csv")
    event_frame.to_csv(dataset_path, index=False)
    feature_group_path = output_prefix.with_name(output_prefix.name + "_feature_groups.json")
    feature_group_path.write_text(json.dumps(feature_groups, indent=2), encoding="utf-8")

    config_payload = {
        "symbol": market_symbol,
        "start": args.start,
        "end": args.end,
        "timeframes": timeframes,
        "zone_timeframes": list(zone_timeframes),
        "decision_timeframe": args.decision_timeframe,
        "channel_estimator": args.channel_estimator,
        "point_count": args.point_count,
        "confluence_epsilon_atr": args.confluence_epsilon_atr,
        "entry_mode": args.entry_mode,
        "passive_entry_window_bars": args.passive_entry_window_bars,
        "passive_entry_buffer_atr": args.passive_entry_buffer_atr,
        "stop_mode": args.stop_mode,
        "stop_buffer_atr": args.stop_buffer_atr,
        "target_buffer_atr": args.target_buffer_atr,
        "label_horizon_bars": args.label_horizon_bars,
        "long_score_mode": args.long_score_mode,
        "short_score_mode": args.short_score_mode,
        "long_min_pos_in_body_1d": args.long_min_pos_in_body_1d,
        "long_max_pos_in_body_1d": args.long_max_pos_in_body_1d,
        "short_min_pos_in_body_1d": args.short_min_pos_in_body_1d,
        "short_max_pos_in_body_1d": args.short_max_pos_in_body_1d,
        "long_max_target_rr": args.long_max_target_rr,
        "short_max_target_rr": args.short_max_target_rr,
        "selection_gates": [format_selection_gate(gate) for gate in selection_gates],
    }
    config_path = output_prefix.with_name(output_prefix.name + "_config.json")
    config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    print(f"Saved event dataset to {dataset_path}", flush=True)
    print(f"Saved feature groups to {feature_group_path}", flush=True)

    if event_frame.empty:
        print("No confluence events were generated for this configuration.", flush=True)
        return

    if args.skip_study:
        return

    selected_groups = tuple(item.strip() for item in args.feature_groups.split(",") if item.strip())
    feature_columns = select_feature_columns(event_frame, feature_groups, selected_groups)
    if not feature_columns:
        print("No usable feature columns were available for the selected feature groups.", flush=True)
        return
    long_thresholds = resolve_thresholds(args.long_thresholds, args.thresholds, args.long_score_mode)
    short_thresholds = resolve_thresholds(args.short_thresholds, args.thresholds, args.short_score_mode)
    results = run_walkforward_event_study(
        dataset=event_frame,
        feature_columns=feature_columns,
        model_name=args.model,
        train_months=args.train_months,
        val_months=args.val_months,
        test_months=args.test_months,
        embargo_bars=args.embargo_bars,
        long_thresholds=long_thresholds,
        short_thresholds=short_thresholds,
        long_score_mode=args.long_score_mode,
        short_score_mode=args.short_score_mode,
        selection_gates=selection_gates,
        risk_fraction=args.risk_fraction,
        min_validation_trades=args.min_validation_trades,
    )

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
    pd.DataFrame({"feature": feature_columns}).to_csv(selected_feature_path, index=False)
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


def select_feature_columns(dataset: pd.DataFrame, feature_groups: dict[str, list[str]], selected_groups: tuple[str, ...]) -> list[str]:
    columns: list[str] = []
    for group_name in selected_groups:
        columns.extend(feature_groups.get(group_name, []))
    unique = list(dict.fromkeys(columns))
    return [column for column in unique if column in dataset.columns and dataset[column].notna().any()]


def apply_event_filters(
    dataset: pd.DataFrame,
    *,
    long_min_pos_in_body_1d: float | None,
    long_max_pos_in_body_1d: float | None,
    short_min_pos_in_body_1d: float | None,
    short_max_pos_in_body_1d: float | None,
    long_max_target_rr: float | None,
    short_max_target_rr: float | None,
) -> pd.DataFrame:
    if dataset.empty:
        return dataset
    filtered = dataset.copy()
    mask = pd.Series(True, index=filtered.index, dtype=bool)
    if "direction" not in filtered.columns:
        return filtered
    if "pos_in_body_1d" in filtered.columns:
        if long_min_pos_in_body_1d is not None:
            mask &= ~((filtered["direction"] == "long") & (filtered["pos_in_body_1d"] < float(long_min_pos_in_body_1d)))
        if long_max_pos_in_body_1d is not None:
            mask &= ~((filtered["direction"] == "long") & (filtered["pos_in_body_1d"] > float(long_max_pos_in_body_1d)))
        if short_min_pos_in_body_1d is not None:
            mask &= ~((filtered["direction"] == "short") & (filtered["pos_in_body_1d"] < float(short_min_pos_in_body_1d)))
        if short_max_pos_in_body_1d is not None:
            mask &= ~((filtered["direction"] == "short") & (filtered["pos_in_body_1d"] > float(short_max_pos_in_body_1d)))
    if "target_rr_planned" in filtered.columns:
        if long_max_target_rr is not None:
            mask &= ~((filtered["direction"] == "long") & (filtered["target_rr_planned"] > float(long_max_target_rr)))
        if short_max_target_rr is not None:
            mask &= ~((filtered["direction"] == "short") & (filtered["target_rr_planned"] > float(short_max_target_rr)))
    return filtered.loc[mask].reset_index(drop=True)


def parse_selection_gates(raw_gates: list[str]) -> tuple[dict[str, Any], ...]:
    gates: list[dict[str, Any]] = []
    for raw_gate in raw_gates:
        text = str(raw_gate).strip()
        if not text:
            continue
        match = SELECTION_GATE_PATTERN.match(text)
        if match is None:
            raise ValueError(
                f"Invalid --selection-gate {text!r}. Expected syntax like long:zone_mid_to_decision_close_atr<=2.5."
            )
        direction, column, operator, value = match.groups()
        gates.append(
            {
                "direction": direction,
                "column": column,
                "operator": operator,
                "value": float(value),
            }
        )
    return tuple(gates)


def format_selection_gate(gate: dict[str, Any]) -> str:
    return f"{gate['direction']}:{gate['column']}{gate['operator']}{gate['value']}"


def selection_gate_mask(frame: pd.DataFrame, selection_gates: tuple[dict[str, Any], ...]) -> pd.Series:
    if frame.empty or not selection_gates:
        return pd.Series(True, index=frame.index, dtype=bool)
    directions = frame["direction"].astype(str) if "direction" in frame.columns else pd.Series("", index=frame.index, dtype=str)
    mask = pd.Series(True, index=frame.index, dtype=bool)
    for gate in selection_gates:
        column = str(gate["column"])
        if column not in frame.columns:
            raise ValueError(f"Selection gate references missing column {column!r}.")
        values = pd.to_numeric(frame[column], errors="coerce")
        direction_mask = directions == str(gate["direction"])
        threshold = float(gate["value"])
        operator = str(gate["operator"])
        if operator == "<=":
            gate_pass = values <= threshold
        elif operator == ">=":
            gate_pass = values >= threshold
        elif operator == "<":
            gate_pass = values < threshold
        elif operator == ">":
            gate_pass = values > threshold
        elif operator == "==":
            gate_pass = values == threshold
        elif operator == "!=":
            gate_pass = values != threshold
        else:
            raise ValueError(f"Unsupported selection gate operator {operator!r}.")
        gate_pass = gate_pass.fillna(False)
        mask &= (~direction_mask) | gate_pass
    return mask


def resolve_thresholds(explicit: str, fallback: str, score_mode: str) -> tuple[float, ...]:
    if explicit.strip():
        return tuple(float(item.strip()) for item in explicit.split(",") if item.strip())
    if score_mode == "probability":
        return tuple(float(item.strip()) for item in fallback.split(",") if item.strip())
    if score_mode == "prob_x_rr":
        return (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
    return (-0.50, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)


def score_series(frame: pd.DataFrame, score_mode: str) -> pd.Series:
    hold_prob = frame["hold_prob"].astype(float)
    target_rr = frame["target_rr_planned"].astype(float)
    cost_r = frame["cost_r"].astype(float) if "cost_r" in frame.columns else pd.Series(0.0, index=frame.index)
    if score_mode == "ev":
        return hold_prob * target_rr - (1.0 - hold_prob)
    if score_mode == "ev_net":
        return hold_prob * target_rr - (1.0 - hold_prob) - cost_r
    if score_mode == "prob_x_rr":
        return hold_prob * target_rr
    return hold_prob


def run_walkforward_event_study(
    *,
    dataset: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
    train_months: int,
    val_months: int,
    test_months: int,
    embargo_bars: int,
    long_thresholds: tuple[float, ...],
    short_thresholds: tuple[float, ...],
    long_score_mode: str,
    short_score_mode: str,
    selection_gates: tuple[dict[str, Any], ...],
    risk_fraction: float,
    min_validation_trades: int,
) -> dict[str, pd.DataFrame]:
    usable = dataset.dropna(subset=["event_time", "exit_time", "hold_label", "future_r_net"]).copy()
    usable = usable.sort_values("event_time").reset_index(drop=True)
    if usable.empty or not feature_columns:
        empty_fold = pd.DataFrame(columns=["fold"])
        empty_importance = pd.DataFrame(columns=["fold", "direction", "feature", "importance"])
        return {
            "folds": empty_fold,
            "thresholds": empty_fold.copy(),
            "predictions": empty_fold.copy(),
            "trades": empty_fold.copy(),
            "feature_importance": empty_importance,
        }
    folds = build_folds(usable, train_months=train_months, val_months=val_months, test_months=test_months)

    fold_rows: list[dict[str, Any]] = []
    threshold_rows: list[pd.DataFrame] = []
    prediction_rows: list[pd.DataFrame] = []
    trade_rows: list[pd.DataFrame] = []
    importance_rows: list[pd.DataFrame] = []
    embargo_delta = pd.Timedelta(hours=embargo_bars)

    for fold_index, fold in enumerate(folds, start=1):
        train_end = fold["train_end"]
        val_end = fold["val_end"]
        test_end = fold["test_end"]

        train = usable[
            (usable["event_time"] >= fold["train_start"])
            & (usable["event_time"] < train_end - embargo_delta)
            & (usable["exit_time"] < train_end)
        ].copy()
        validation = usable[
            (usable["event_time"] >= train_end)
            & (usable["event_time"] < val_end - embargo_delta)
            & (usable["exit_time"] < val_end)
        ].copy()
        test = usable[
            (usable["event_time"] >= val_end)
            & (usable["event_time"] < test_end)
            & (usable["exit_time"] < test_end)
        ].copy()
        if train.empty or validation.empty or test.empty:
            continue

        long_train = train[train["direction"] == "long"].copy()
        short_train = train[train["direction"] == "short"].copy()
        long_validation = validation[validation["direction"] == "long"].copy()
        short_validation = validation[validation["direction"] == "short"].copy()
        long_test = test[test["direction"] == "long"].copy()
        short_test = test[test["direction"] == "short"].copy()
        if long_train.empty or short_train.empty or long_validation.empty or short_validation.empty or long_test.empty or short_test.empty:
            continue
        if long_train["hold_label"].nunique() < 2 or short_train["hold_label"].nunique() < 2:
            continue

        long_model = fit_binary_model(long_train, feature_columns, "hold_label", model_name)
        short_model = fit_binary_model(short_train, feature_columns, "hold_label", model_name)

        long_validation["hold_prob"] = predict_binary_model(long_model, long_validation, feature_columns)
        short_validation["hold_prob"] = predict_binary_model(short_model, short_validation, feature_columns)
        long_validation["model_score"] = score_series(long_validation, long_score_mode)
        short_validation["model_score"] = score_series(short_validation, short_score_mode)
        long_validation["gate_pass"] = selection_gate_mask(long_validation, selection_gates)
        short_validation["gate_pass"] = selection_gate_mask(short_validation, selection_gates)
        threshold_pair, threshold_table = tune_event_thresholds(
            long_validation,
            short_validation,
            long_thresholds=long_thresholds,
            short_thresholds=short_thresholds,
            risk_fraction=risk_fraction,
            min_validation_trades=min_validation_trades,
        )
        threshold_table["fold"] = float(fold_index)
        threshold_rows.append(threshold_table)

        long_test["hold_prob"] = predict_binary_model(long_model, long_test, feature_columns)
        short_test["hold_prob"] = predict_binary_model(short_model, short_test, feature_columns)
        long_test["model_score"] = score_series(long_test, long_score_mode)
        short_test["model_score"] = score_series(short_test, short_score_mode)
        long_test["gate_pass"] = selection_gate_mask(long_test, selection_gates)
        short_test["gate_pass"] = selection_gate_mask(short_test, selection_gates)
        long_test["selected"] = long_test["gate_pass"] & (long_test["model_score"] >= float(threshold_pair["long_threshold"]))
        short_test["selected"] = short_test["gate_pass"] & (short_test["model_score"] >= float(threshold_pair["short_threshold"]))
        fold_predictions = pd.concat([long_test, short_test], ignore_index=True)
        fold_predictions["fold"] = float(fold_index)
        fold_predictions["train_end"] = train_end
        fold_predictions["val_end"] = val_end
        fold_predictions["test_end"] = test_end
        prediction_rows.append(fold_predictions)

        trades = select_merged_trades(
            long_events=long_test[long_test["selected"]].copy(),
            short_events=short_test[short_test["selected"]].copy(),
            risk_fraction=risk_fraction,
        )
        trade_metrics = strategy_metrics(trades)
        if not trades.empty:
            trades["fold"] = float(fold_index)
            trade_rows.append(trades)

        fold_row: dict[str, Any] = {
            "fold": float(fold_index),
            "train_start": fold["train_start"].isoformat(),
            "train_end": train_end.isoformat(),
            "val_end": val_end.isoformat(),
            "test_end": test_end.isoformat(),
            "train_rows": float(len(train)),
            "validation_rows": float(len(validation)),
            "test_rows": float(len(test)),
            "feature_count": float(len(feature_columns)),
            "model_name": model_name,
            "long_score_mode": long_score_mode,
            "short_score_mode": short_score_mode,
            "long_threshold": float(threshold_pair["long_threshold"]),
            "short_threshold": float(threshold_pair["short_threshold"]),
        }
        for prefix, frame in [
            ("long_val", long_validation),
            ("short_val", short_validation),
            ("long_test", long_test),
            ("short_test", short_test),
        ]:
            metrics = binary_metrics(frame["hold_label"], frame["hold_prob"])
            fold_row.update({f"{prefix}_{key}": float(value) if value is not None else np.nan for key, value in metrics.items()})
            if "gate_pass" in frame.columns and frame["gate_pass"].notna().any():
                fold_row[f"{prefix}_gate_pass_rate"] = float(frame["gate_pass"].astype(float).mean())
        for key, value in trade_metrics.items():
            fold_row[f"trade_{key}"] = float(value) if isinstance(value, (int, float, np.floating)) else value
        fold_rows.append(fold_row)

        importance_rows.extend(
            [
                extract_feature_importance(long_model, feature_columns, fold_index, "long"),
                extract_feature_importance(short_model, feature_columns, fold_index, "short"),
            ]
        )

    fold_summary = pd.DataFrame(fold_rows) if fold_rows else pd.DataFrame(columns=["fold"])
    threshold_frame = pd.concat(threshold_rows, ignore_index=True) if threshold_rows else pd.DataFrame(columns=["fold"])
    prediction_frame = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame(columns=["fold"])
    trade_frame = pd.concat(trade_rows, ignore_index=True) if trade_rows else pd.DataFrame(columns=["fold"])
    nonempty_importance = [frame for frame in importance_rows if not frame.empty]
    importance_frame = pd.concat(nonempty_importance, ignore_index=True) if nonempty_importance else pd.DataFrame(columns=["fold", "direction", "feature", "importance"])

    if not trade_frame.empty:
        aggregate = strategy_metrics(trade_frame)
        aggregate_row: dict[str, Any] = {
            "fold": "aggregate",
            "train_start": "",
            "train_end": "",
            "val_end": "",
            "test_end": "",
            "train_rows": float(fold_summary["train_rows"].sum()) if not fold_summary.empty else 0.0,
            "validation_rows": float(fold_summary["validation_rows"].sum()) if not fold_summary.empty else 0.0,
            "test_rows": float(len(prediction_frame)),
            "feature_count": float(len(feature_columns)),
            "model_name": model_name,
        }
        for key, value in aggregate.items():
            aggregate_row[f"trade_{key}"] = float(value) if isinstance(value, (int, float, np.floating)) else value
        if not prediction_frame.empty:
            long_predictions = prediction_frame[prediction_frame["direction"] == "long"]
            short_predictions = prediction_frame[prediction_frame["direction"] == "short"]
            if not long_predictions.empty:
                aggregate_row["long_test_auc"] = binary_metrics(long_predictions["hold_label"], long_predictions["hold_prob"]).get("auc")
            if not short_predictions.empty:
                aggregate_row["short_test_auc"] = binary_metrics(short_predictions["hold_label"], short_predictions["hold_prob"]).get("auc")
            for source, target in [("long_threshold", "trade_long_threshold"), ("short_threshold", "trade_short_threshold")]:
                if source in fold_summary.columns and fold_summary[source].notna().any():
                    aggregate_row[target] = float(fold_summary[source].dropna().mode().iloc[0])
        fold_summary = pd.concat([fold_summary, pd.DataFrame([aggregate_row])], ignore_index=True)

    return {
        "folds": fold_summary,
        "thresholds": threshold_frame,
        "predictions": prediction_frame,
        "trades": trade_frame,
        "feature_importance": importance_frame,
    }


def build_folds(dataset: pd.DataFrame, *, train_months: int, val_months: int, test_months: int) -> list[dict[str, pd.Timestamp]]:
    ordered = dataset.sort_values("event_time")
    first_time = pd.Timestamp(ordered["event_time"].iloc[0]).tz_convert("UTC").floor("D")
    last_time = pd.Timestamp(ordered["event_time"].iloc[-1]).tz_convert("UTC").ceil("D")
    folds: list[dict[str, pd.Timestamp]] = []
    train_start = first_time
    train_end = train_start + pd.DateOffset(months=train_months)
    val_end = train_end + pd.DateOffset(months=val_months)
    test_end = val_end + pd.DateOffset(months=test_months)
    while test_end <= last_time:
        folds.append(
            {
                "train_start": pd.Timestamp(train_start).tz_convert("UTC"),
                "train_end": pd.Timestamp(train_end).tz_convert("UTC"),
                "val_end": pd.Timestamp(val_end).tz_convert("UTC"),
                "test_end": pd.Timestamp(test_end).tz_convert("UTC"),
            }
        )
        train_end = train_end + pd.DateOffset(months=test_months)
        val_end = val_end + pd.DateOffset(months=test_months)
        test_end = test_end + pd.DateOffset(months=test_months)
    return folds


def tune_event_thresholds(
    long_validation: pd.DataFrame,
    short_validation: pd.DataFrame,
    *,
    long_thresholds: tuple[float, ...],
    short_thresholds: tuple[float, ...],
    risk_fraction: float,
    min_validation_trades: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    best_score = -math.inf
    best = {"long_threshold": float(long_thresholds[0]), "short_threshold": float(short_thresholds[0])}
    long_candidates = long_validation[long_validation["gate_pass"]] if "gate_pass" in long_validation.columns else long_validation
    short_candidates = short_validation[short_validation["gate_pass"]] if "gate_pass" in short_validation.columns else short_validation

    for long_threshold in long_thresholds:
        for short_threshold in short_thresholds:
            trades = select_merged_trades(
                long_events=long_candidates[long_candidates["model_score"] >= float(long_threshold)].copy(),
                short_events=short_candidates[short_candidates["model_score"] >= float(short_threshold)].copy(),
                risk_fraction=risk_fraction,
            )
            metrics = strategy_metrics(trades)
            eligible = metrics["trades"] >= float(min_validation_trades)
            score = threshold_score(metrics) if eligible else -10_000.0
            rows.append(
                {
                    "long_threshold": float(long_threshold),
                    "short_threshold": float(short_threshold),
                    "eligible": float(eligible),
                    "score": float(score),
                    **metrics,
                }
            )
            if eligible and score > best_score:
                best_score = score
                best = {"long_threshold": float(long_threshold), "short_threshold": float(short_threshold)}

    table = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    if best_score == -math.inf and not table.empty:
        best = {
            "long_threshold": float(table.iloc[0]["long_threshold"]),
            "short_threshold": float(table.iloc[0]["short_threshold"]),
        }
    return best, table


def select_merged_trades(*, long_events: pd.DataFrame, short_events: pd.DataFrame, risk_fraction: float) -> pd.DataFrame:
    selected = pd.concat([long_events, short_events], ignore_index=True)
    if selected.empty:
        return pd.DataFrame(columns=["direction", "entry_time", "exit_time", "r_multiple_net", "return_pct", "hold_bars"])
    selected = selected.copy()
    cost_r = selected["cost_r"].astype(float) if "cost_r" in selected.columns else pd.Series(0.0, index=selected.index)
    selected["collision_score"] = (
        selected["hold_prob"].astype(float) * selected["target_rr_planned"].astype(float)
        - (1.0 - selected["hold_prob"].astype(float))
        - cost_r
    )
    selected["selection_score"] = selected["model_score"].astype(float) if "model_score" in selected.columns else selected["collision_score"]
    selected = selected.sort_values(
        ["entry_time", "collision_score", "selection_score", "hold_prob", "target_rr_planned"],
        ascending=[True, False, False, False, False],
    ).reset_index(drop=True)

    trades: list[dict[str, Any]] = []
    active_exit: pd.Timestamp | None = None
    for _, row in selected.iterrows():
        entry_time = pd.Timestamp(row["entry_time"]).tz_convert("UTC")
        exit_time = pd.Timestamp(row["exit_time"]).tz_convert("UTC")
        if active_exit is not None and entry_time < active_exit:
            continue
        hold_bars = int(max(1.0, round((exit_time - entry_time).total_seconds() / 3600.0)))
        trades.append(
            {
                "direction": str(row["direction"]),
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_price": float(row["entry_price"]),
                "exit_price": np.nan,
                "stop_price": float(row["stop_price"]),
                "target_price": float(row["target_price"]),
                "hold_prob": float(row["hold_prob"]),
                "selection_score": float(row["selection_score"]),
                "collision_score": float(row["collision_score"]),
                "r_multiple_gross": float(row["future_r"]),
                "r_multiple_net": float(row["future_r_net"]),
                "return_pct": float(risk_fraction * float(row["future_r_net"])),
                "hold_bars": hold_bars,
                "exit_reason": str(row["outcome"]),
                "event_key": str(row["event_key"]),
                "symbol": str(row["symbol"]),
            }
        )
        active_exit = exit_time
    return pd.DataFrame(trades)


def threshold_score(metrics: dict[str, Any]) -> float:
    total_return = float(metrics["total_return"])
    max_drawdown = abs(float(metrics["max_drawdown"]))
    profit_factor = float(metrics["profit_factor"]) if math.isfinite(float(metrics["profit_factor"])) else 5.0
    return total_return + 0.10 * profit_factor - 0.50 * max_drawdown


def extract_feature_importance(model: Any, feature_columns: list[str], fold_index: int, direction: str) -> pd.DataFrame:
    estimator = model.named_steps.get("estimator") if hasattr(model, "named_steps") else None
    if estimator is None:
        return pd.DataFrame()
    if hasattr(estimator, "coef_"):
        importance = np.abs(np.ravel(estimator.coef_))
    elif hasattr(estimator, "feature_importances_"):
        importance = np.asarray(estimator.feature_importances_, dtype=float)
    else:
        return pd.DataFrame()
    if len(importance) != len(feature_columns):
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "fold": float(fold_index),
            "direction": direction,
            "feature": feature_columns,
            "importance": importance,
        }
    ).sort_values("importance", ascending=False)


def build_report(config_payload: dict[str, Any], results: dict[str, pd.DataFrame]) -> str:
    folds = results["folds"]
    aggregate = folds[folds["fold"].astype(str) == "aggregate"] if "fold" in folds.columns else pd.DataFrame()
    aggregate_row = aggregate.iloc[0].to_dict() if not aggregate.empty else {}
    lines = [
        "# Zone-Channel Confluence Study",
        "",
        "## Configuration",
        "",
        f"- symbol: `{config_payload['symbol']}`",
        f"- window: `{config_payload['start']}` to `{config_payload['end']}`",
        f"- timeframes: `{', '.join(config_payload['timeframes'])}`",
        f"- zone timeframes: `{', '.join(config_payload['zone_timeframes'])}`",
        f"- decision timeframe: `{config_payload['decision_timeframe']}`",
        f"- estimator: `{config_payload['channel_estimator']}`",
        f"- point count: `{config_payload['point_count']}`",
        f"- confluence epsilon ATR: `{config_payload['confluence_epsilon_atr']}`",
        f"- entry mode: `{config_payload['entry_mode']}`",
        f"- passive entry window bars: `{config_payload['passive_entry_window_bars']}`",
        f"- passive entry buffer ATR: `{config_payload['passive_entry_buffer_atr']}`",
        f"- stop mode: `{config_payload['stop_mode']}`",
        f"- stop buffer ATR: `{config_payload['stop_buffer_atr']}`",
        f"- target buffer ATR: `{config_payload['target_buffer_atr']}`",
        f"- label horizon bars: `{config_payload['label_horizon_bars']}`",
        f"- long score mode: `{config_payload['long_score_mode']}`",
        f"- short score mode: `{config_payload['short_score_mode']}`",
        f"- long pos_in_body_1d min/max: `{config_payload['long_min_pos_in_body_1d']}` / `{config_payload['long_max_pos_in_body_1d']}`",
        f"- short pos_in_body_1d min/max: `{config_payload['short_min_pos_in_body_1d']}` / `{config_payload['short_max_pos_in_body_1d']}`",
        f"- long max target RR: `{config_payload['long_max_target_rr']}`",
        f"- short max target RR: `{config_payload['short_max_target_rr']}`",
        f"- selection gates: `{', '.join(config_payload['selection_gates']) if config_payload.get('selection_gates') else 'none'}`",
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
            "long_test_auc",
            "short_test_auc",
        ]:
            if key in aggregate_row:
                lines.append(f"- {key}: `{aggregate_row[key]}`")
    else:
        lines.append("- No aggregate row was produced.")

    importance = results["feature_importance"]
    lines.extend(["", "## Top Features", ""])
    if not importance.empty:
        top = importance.groupby(["direction", "feature"], as_index=False)["importance"].mean().sort_values(["direction", "importance"], ascending=[True, False])
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
    for column in ["event_time", "entry_time", "zone_time", "exit_time"]:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    return frame


if __name__ == "__main__":
    main()
