from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.channel_state_research.modeling import binary_metrics, fit_binary_model, predict_binary_model
from scripts.run_zone_channel_confluence_study import (
    build_folds,
    extract_feature_importance,
    load_dataset,
    parse_selection_gates,
    score_series,
    select_merged_trades,
    selection_gate_mask,
    tune_event_thresholds,
)


@dataclass(frozen=True)
class CachedFold:
    fold_index: int
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    test_end: pd.Timestamp
    long_validation: pd.DataFrame
    short_validation: pd.DataFrame
    long_test: pd.DataFrame
    short_test: pd.DataFrame
    long_importance: pd.DataFrame
    short_importance: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast post-model gate sweep for channel event datasets.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--selected-features", type=Path, required=True)
    parser.add_argument("--model", choices=["logreg", "rf", "hgb"], default="rf")
    parser.add_argument("--train-months", type=int, default=4)
    parser.add_argument("--val-months", type=int, default=2)
    parser.add_argument("--test-months", type=int, default=2)
    parser.add_argument("--embargo-bars", type=int, default=24)
    parser.add_argument("--long-thresholds", default="0.45,0.50,0.55,0.60,0.65")
    parser.add_argument("--short-thresholds", default="0.45,0.50,0.55,0.60,0.65")
    parser.add_argument("--long-score-mode", choices=["probability", "ev", "ev_net", "prob_x_rr"], default="probability")
    parser.add_argument("--short-score-mode", choices=["probability", "ev", "ev_net", "prob_x_rr"], default="probability")
    parser.add_argument("--risk-fraction", type=float, default=0.01)
    parser.add_argument("--min-validation-trades", type=int, default=5)
    parser.add_argument("--zone-mode-values", default="ANY")
    parser.add_argument("--long-min-reclaim-values", default="0")
    parser.add_argument("--long-max-reclaim-values", default="0")
    parser.add_argument("--short-min-reclaim-values", default="0")
    parser.add_argument("--short-max-reclaim-values", default="0")
    parser.add_argument("--long-max-target-rr-values", default="0")
    parser.add_argument("--short-max-target-rr-values", default="0")
    parser.add_argument("--long-min-zone-width-pct-values", default="0")
    parser.add_argument("--short-min-zone-width-pct-values", default="0")
    parser.add_argument("--long-max-zone-width-atr-values", default="0")
    parser.add_argument("--short-max-zone-width-atr-values", default="0")
    parser.add_argument("--out-prefix", type=Path, default=Path("scripts/channel_selection_gate_sweep"))
    return parser.parse_args()


def parse_float_values(text: str) -> list[float | None]:
    values: list[float | None] = []
    for item in str(text).split(","):
        raw = item.strip()
        if not raw:
            continue
        if raw.lower() in {"0", "none", "null", "any"}:
            values.append(None)
        else:
            values.append(float(raw))
    return values or [None]


def parse_mode_values(text: str) -> list[str]:
    values = [item.strip() for item in str(text).split(",") if item.strip()]
    return values or ["ANY"]


def resolve_thresholds(text: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in str(text).split(",") if item.strip())


def load_feature_columns(path: Path) -> list[str]:
    frame = pd.read_csv(path)
    if "feature" not in frame.columns:
        raise ValueError(f"{path} must contain a 'feature' column.")
    return frame["feature"].dropna().astype(str).tolist()


def build_cached_folds(
    *,
    dataset: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
    train_months: int,
    val_months: int,
    test_months: int,
    embargo_bars: int,
    long_score_mode: str,
    short_score_mode: str,
) -> list[CachedFold]:
    usable = dataset.copy()
    usable["event_time"] = pd.to_datetime(usable["event_time"], utc=True, errors="coerce")
    usable["entry_time"] = pd.to_datetime(usable["entry_time"], utc=True, errors="coerce")
    usable["exit_time"] = pd.to_datetime(usable["exit_time"], utc=True, errors="coerce")
    usable = usable.dropna(subset=["event_time", "exit_time"]).sort_values("event_time").reset_index(drop=True)

    embargo_delta = pd.Timedelta(hours=max(int(embargo_bars), 0))
    cached: list[CachedFold] = []

    for fold_index, fold in enumerate(
        build_folds(usable, train_months=train_months, val_months=val_months, test_months=test_months),
        start=1,
    ):
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
        long_test["hold_prob"] = predict_binary_model(long_model, long_test, feature_columns)
        short_test["hold_prob"] = predict_binary_model(short_model, short_test, feature_columns)

        long_validation["model_score"] = score_series(long_validation, long_score_mode)
        short_validation["model_score"] = score_series(short_validation, short_score_mode)
        long_test["model_score"] = score_series(long_test, long_score_mode)
        short_test["model_score"] = score_series(short_test, short_score_mode)

        cached.append(
            CachedFold(
                fold_index=fold_index,
                train_end=train_end,
                val_end=val_end,
                test_end=test_end,
                long_validation=long_validation,
                short_validation=short_validation,
                long_test=long_test,
                short_test=short_test,
                long_importance=extract_feature_importance(long_model, feature_columns, fold_index, "long"),
                short_importance=extract_feature_importance(short_model, feature_columns, fold_index, "short"),
            )
        )
    return cached


def gate_strings_from_combo(
    *,
    zone_mode: str,
    long_min_reclaim: float | None,
    long_max_reclaim: float | None,
    short_min_reclaim: float | None,
    short_max_reclaim: float | None,
    long_max_target_rr: float | None,
    short_max_target_rr: float | None,
    long_min_zone_width_pct: float | None,
    short_min_zone_width_pct: float | None,
    long_max_zone_width_atr: float | None,
    short_max_zone_width_atr: float | None,
) -> tuple[str, ...]:
    gates: list[str] = []
    zone_mode = zone_mode.strip()
    if zone_mode not in {"ANY", "1h_only", "4h_only", "1h4h_only"}:
        raise ValueError(f"Unsupported zone mode {zone_mode!r}.")
    if zone_mode == "1h_only":
        gates.extend(["long:zone_tf_1h>=1", "short:zone_tf_1h>=1"])
    elif zone_mode == "4h_only":
        gates.extend(["long:zone_tf_4h>=1", "short:zone_tf_4h>=1"])
    elif zone_mode == "1h4h_only":
        pass
    if long_min_reclaim is not None:
        gates.append(f"long:reclaim_pos>={long_min_reclaim}")
    if long_max_reclaim is not None:
        gates.append(f"long:reclaim_pos<={long_max_reclaim}")
    if short_min_reclaim is not None:
        gates.append(f"short:reclaim_pos>={short_min_reclaim}")
    if short_max_reclaim is not None:
        gates.append(f"short:reclaim_pos<={short_max_reclaim}")
    if long_max_target_rr is not None:
        gates.append(f"long:target_rr_planned<={long_max_target_rr}")
    if short_max_target_rr is not None:
        gates.append(f"short:target_rr_planned<={short_max_target_rr}")
    if long_min_zone_width_pct is not None:
        gates.append(f"long:zone_width_pct>={long_min_zone_width_pct}")
    if short_min_zone_width_pct is not None:
        gates.append(f"short:zone_width_pct>={short_min_zone_width_pct}")
    if long_max_zone_width_atr is not None:
        gates.append(f"long:zone_width_atr<={long_max_zone_width_atr}")
    if short_max_zone_width_atr is not None:
        gates.append(f"short:zone_width_atr<={short_max_zone_width_atr}")
    return tuple(gates)


def candidate_name_from_gates(zone_mode: str, gates: tuple[str, ...]) -> str:
    if not gates:
        return f"{zone_mode.lower()}__base"
    safe = [gate.replace(":", "_").replace(">=", "ge").replace("<=", "le").replace(">", "gt").replace("<", "lt").replace("==", "eq").replace("!=", "ne").replace(".", "p") for gate in gates]
    return f"{zone_mode.lower()}__{'__'.join(safe)}"


def evaluate_candidate(
    *,
    cached_folds: list[CachedFold],
    selection_gates: tuple[dict[str, Any], ...],
    long_thresholds: tuple[float, ...],
    short_thresholds: tuple[float, ...],
    risk_fraction: float,
    min_validation_trades: int,
) -> dict[str, Any]:
    fold_rows: list[dict[str, Any]] = []
    threshold_rows: list[pd.DataFrame] = []
    trade_rows: list[pd.DataFrame] = []
    prediction_rows: list[pd.DataFrame] = []

    for fold in cached_folds:
        long_validation = fold.long_validation.copy()
        short_validation = fold.short_validation.copy()
        long_test = fold.long_test.copy()
        short_test = fold.short_test.copy()

        long_validation["gate_pass"] = selection_gate_mask(long_validation, selection_gates)
        short_validation["gate_pass"] = selection_gate_mask(short_validation, selection_gates)
        best_thresholds, threshold_table = tune_event_thresholds(
            long_validation,
            short_validation,
            long_thresholds=long_thresholds,
            short_thresholds=short_thresholds,
            risk_fraction=risk_fraction,
            min_validation_trades=min_validation_trades,
        )
        threshold_table["fold"] = float(fold.fold_index)
        threshold_rows.append(threshold_table)

        long_test["gate_pass"] = selection_gate_mask(long_test, selection_gates)
        short_test["gate_pass"] = selection_gate_mask(short_test, selection_gates)
        long_test["selected"] = long_test["gate_pass"] & (long_test["model_score"] >= float(best_thresholds["long_threshold"]))
        short_test["selected"] = short_test["gate_pass"] & (short_test["model_score"] >= float(best_thresholds["short_threshold"]))
        fold_predictions = pd.concat([long_test, short_test], ignore_index=True)
        fold_predictions["fold"] = float(fold.fold_index)
        prediction_rows.append(fold_predictions)

        trades = select_merged_trades(
            long_events=long_test[long_test["selected"]].copy(),
            short_events=short_test[short_test["selected"]].copy(),
            risk_fraction=risk_fraction,
        )
        if not trades.empty:
            trades["fold"] = float(fold.fold_index)
            trade_rows.append(trades)

        metrics = strategy_metrics_from_trades(trades)
        val_long_metrics = binary_metrics(long_validation["hold_label"], long_validation["hold_prob"])
        val_short_metrics = binary_metrics(short_validation["hold_label"], short_validation["hold_prob"])
        test_long_metrics = binary_metrics(long_test["hold_label"], long_test["hold_prob"])
        test_short_metrics = binary_metrics(short_test["hold_label"], short_test["hold_prob"])
        row: dict[str, Any] = {
            "fold": float(fold.fold_index),
            "train_end": fold.train_end.isoformat(),
            "val_end": fold.val_end.isoformat(),
            "test_end": fold.test_end.isoformat(),
            "long_threshold": float(best_thresholds["long_threshold"]),
            "short_threshold": float(best_thresholds["short_threshold"]),
            "long_val_gate_pass_rate": float(long_validation["gate_pass"].astype(float).mean()),
            "short_val_gate_pass_rate": float(short_validation["gate_pass"].astype(float).mean()),
            "long_test_gate_pass_rate": float(long_test["gate_pass"].astype(float).mean()),
            "short_test_gate_pass_rate": float(short_test["gate_pass"].astype(float).mean()),
            "long_val_auc": float(val_long_metrics.get("auc", np.nan)),
            "short_val_auc": float(val_short_metrics.get("auc", np.nan)),
            "long_test_auc": float(test_long_metrics.get("auc", np.nan)),
            "short_test_auc": float(test_short_metrics.get("auc", np.nan)),
        }
        row.update({f"trade_{key}": value for key, value in metrics.items()})
        fold_rows.append(row)

    folds_frame = pd.DataFrame(fold_rows)
    thresholds_frame = pd.concat(threshold_rows, ignore_index=True) if threshold_rows else pd.DataFrame()
    predictions_frame = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    trades_frame = pd.concat(trade_rows, ignore_index=True) if trade_rows else pd.DataFrame()

    aggregate = aggregate_fold_results(folds_frame, predictions_frame, trades_frame)
    if aggregate is not None:
        folds_frame = pd.concat([folds_frame, pd.DataFrame([aggregate])], ignore_index=True)
    return {
        "folds": folds_frame,
        "thresholds": thresholds_frame,
        "predictions": predictions_frame,
        "trades": trades_frame,
    }


def strategy_metrics_from_trades(trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty:
        return {
            "trades": 0.0,
            "total_return": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "hit_rate": 0.0,
            "profit_factor": 0.0,
            "average_trade": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "long_only_return": 0.0,
            "short_only_return": 0.0,
            "net_r": 0.0,
        }
    returns = pd.to_numeric(trades["return_pct"], errors="coerce").fillna(0.0)
    net_r = pd.to_numeric(trades["r_multiple_net"], errors="coerce").fillna(0.0)
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    downside = returns[returns < 0.0]
    wins = net_r[net_r > 0.0]
    losses = net_r[net_r < 0.0]
    long_mask = trades["direction"].astype(str) == "long"
    short_mask = trades["direction"].astype(str) == "short"
    total_return = float(equity.iloc[-1] - 1.0)
    sharpe = float(returns.mean() / returns.std(ddof=0)) if len(returns) > 1 and returns.std(ddof=0) > 0 else 0.0
    sortino = float(returns.mean() / downside.std(ddof=0)) if len(downside) > 1 and downside.std(ddof=0) > 0 else 0.0
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0
    calmar = float(total_return / abs(max_drawdown)) if max_drawdown < 0 else 0.0
    gross_profit = float(wins.sum()) if not wins.empty else 0.0
    gross_loss = float(-losses.sum()) if not losses.empty else 0.0
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    return {
        "trades": float(len(trades)),
        "total_return": total_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "hit_rate": float((net_r > 0.0).mean()),
        "profit_factor": profit_factor,
        "average_trade": float(net_r.mean()),
        "average_win": float(wins.mean()) if not wins.empty else 0.0,
        "average_loss": float(losses.mean()) if not losses.empty else 0.0,
        "long_only_return": float(returns[long_mask].sum()) if long_mask.any() else 0.0,
        "short_only_return": float(returns[short_mask].sum()) if short_mask.any() else 0.0,
        "net_r": float(net_r.sum()),
    }


def aggregate_fold_results(
    folds_frame: pd.DataFrame,
    predictions_frame: pd.DataFrame,
    trades_frame: pd.DataFrame,
) -> dict[str, Any] | None:
    if folds_frame.empty:
        return None
    aggregate: dict[str, Any] = {"fold": "aggregate"}
    metrics = strategy_metrics_from_trades(trades_frame)
    for key, value in metrics.items():
        aggregate[f"trade_{key}"] = value
    if not predictions_frame.empty:
        long_predictions = predictions_frame[predictions_frame["direction"].astype(str) == "long"]
        short_predictions = predictions_frame[predictions_frame["direction"].astype(str) == "short"]
        if not long_predictions.empty:
            aggregate["long_test_auc"] = float(binary_metrics(long_predictions["hold_label"], long_predictions["hold_prob"]).get("auc", np.nan))
        if not short_predictions.empty:
            aggregate["short_test_auc"] = float(binary_metrics(short_predictions["hold_label"], short_predictions["hold_prob"]).get("auc", np.nan))
    for source, target in [("long_threshold", "trade_long_threshold"), ("short_threshold", "trade_short_threshold")]:
        if source in folds_frame.columns and folds_frame[source].notna().any():
            aggregate[target] = float(folds_frame[source].dropna().mode().iloc[0])
    if "trade_net_r" in folds_frame.columns:
        aggregate["worst_fold_net_r"] = float(folds_frame["trade_net_r"].min())
    if "trade_trades" in folds_frame.columns:
        aggregate["min_fold_trades"] = float(folds_frame["trade_trades"].min())
    if "trade_profit_factor" in folds_frame.columns:
        pf = pd.to_numeric(folds_frame["trade_profit_factor"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        aggregate["worst_fold_pf"] = float(pf.min()) if not pf.empty else np.nan
    return aggregate


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset)
    feature_columns = load_feature_columns(args.selected_features)
    long_thresholds = resolve_thresholds(args.long_thresholds)
    short_thresholds = resolve_thresholds(args.short_thresholds)

    cached_folds = build_cached_folds(
        dataset=dataset,
        feature_columns=feature_columns,
        model_name=args.model,
        train_months=args.train_months,
        val_months=args.val_months,
        test_months=args.test_months,
        embargo_bars=args.embargo_bars,
        long_score_mode=args.long_score_mode,
        short_score_mode=args.short_score_mode,
    )
    if not cached_folds:
        raise SystemExit("No usable folds were built from the dataset.")

    zone_modes = parse_mode_values(args.zone_mode_values)
    long_min_reclaims = parse_float_values(args.long_min_reclaim_values)
    long_max_reclaims = parse_float_values(args.long_max_reclaim_values)
    short_min_reclaims = parse_float_values(args.short_min_reclaim_values)
    short_max_reclaims = parse_float_values(args.short_max_reclaim_values)
    long_max_target_rrs = parse_float_values(args.long_max_target_rr_values)
    short_max_target_rrs = parse_float_values(args.short_max_target_rr_values)
    long_min_zone_widths = parse_float_values(args.long_min_zone_width_pct_values)
    short_min_zone_widths = parse_float_values(args.short_min_zone_width_pct_values)
    long_max_zone_width_atrs = parse_float_values(args.long_max_zone_width_atr_values)
    short_max_zone_width_atrs = parse_float_values(args.short_max_zone_width_atr_values)

    summary_rows: list[dict[str, Any]] = []
    best_outputs: dict[str, dict[str, pd.DataFrame]] = {}

    total = (
        len(zone_modes)
        * len(long_min_reclaims)
        * len(long_max_reclaims)
        * len(short_min_reclaims)
        * len(short_max_reclaims)
        * len(long_max_target_rrs)
        * len(short_max_target_rrs)
        * len(long_min_zone_widths)
        * len(short_min_zone_widths)
        * len(long_max_zone_width_atrs)
        * len(short_max_zone_width_atrs)
    )
    print(f"Evaluating {total} gate combinations across {len(cached_folds)} cached folds...", flush=True)

    for combo_index, combo in enumerate(
        product(
            zone_modes,
            long_min_reclaims,
            long_max_reclaims,
            short_min_reclaims,
            short_max_reclaims,
            long_max_target_rrs,
            short_max_target_rrs,
            long_min_zone_widths,
            short_min_zone_widths,
            long_max_zone_width_atrs,
            short_max_zone_width_atrs,
        ),
        start=1,
    ):
        (
            zone_mode,
            long_min_reclaim,
            long_max_reclaim,
            short_min_reclaim,
            short_max_reclaim,
            long_max_target_rr,
            short_max_target_rr,
            long_min_zone_width_pct,
            short_min_zone_width_pct,
            long_max_zone_width_atr,
            short_max_zone_width_atr,
        ) = combo
        gate_strings = gate_strings_from_combo(
            zone_mode=zone_mode,
            long_min_reclaim=long_min_reclaim,
            long_max_reclaim=long_max_reclaim,
            short_min_reclaim=short_min_reclaim,
            short_max_reclaim=short_max_reclaim,
            long_max_target_rr=long_max_target_rr,
            short_max_target_rr=short_max_target_rr,
            long_min_zone_width_pct=long_min_zone_width_pct,
            short_min_zone_width_pct=short_min_zone_width_pct,
            long_max_zone_width_atr=long_max_zone_width_atr,
            short_max_zone_width_atr=short_max_zone_width_atr,
        )
        parsed_gates = parse_selection_gates(list(gate_strings))
        result = evaluate_candidate(
            cached_folds=cached_folds,
            selection_gates=parsed_gates,
            long_thresholds=long_thresholds,
            short_thresholds=short_thresholds,
            risk_fraction=args.risk_fraction,
            min_validation_trades=args.min_validation_trades,
        )
        folds = result["folds"]
        aggregate = folds[folds["fold"].astype(str) == "aggregate"]
        if aggregate.empty:
            continue
        aggregate_row = aggregate.iloc[0]
        per_fold = folds[folds["fold"].astype(str) != "aggregate"].copy()
        fold_nets = pd.to_numeric(per_fold.get("trade_net_r", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        fold_pfs = pd.to_numeric(per_fold.get("trade_profit_factor", pd.Series(dtype=float)), errors="coerce")
        finite_pfs = fold_pfs.replace([np.inf, -np.inf], np.nan).fillna(5.0)
        all_folds_pos = bool(len(fold_nets) > 0 and (fold_nets > 0.0).all())
        score = (
            float(aggregate_row.get("trade_net_r", 0.0)) * 0.45
            + float(fold_nets.min() if not fold_nets.empty else 0.0) * 0.35
            + (float(finite_pfs.mean()) - 1.0) * 5.0
            + float(aggregate_row.get("trade_trades", 0.0)) * 0.02
        )
        summary_row = {
            "combo": combo_index,
            "candidate_name": candidate_name_from_gates(zone_mode, gate_strings),
            "zone_mode": zone_mode,
            "selection_gates": " | ".join(gate_strings) if gate_strings else "none",
            "long_min_reclaim": long_min_reclaim,
            "long_max_reclaim": long_max_reclaim,
            "short_min_reclaim": short_min_reclaim,
            "short_max_reclaim": short_max_reclaim,
            "long_max_target_rr": long_max_target_rr,
            "short_max_target_rr": short_max_target_rr,
            "long_min_zone_width_pct": long_min_zone_width_pct,
            "short_min_zone_width_pct": short_min_zone_width_pct,
            "long_max_zone_width_atr": long_max_zone_width_atr,
            "short_max_zone_width_atr": short_max_zone_width_atr,
            "aggregate_trades": float(aggregate_row.get("trade_trades", 0.0)),
            "aggregate_net_r": float(aggregate_row.get("trade_net_r", 0.0)),
            "aggregate_pf": float(aggregate_row.get("trade_profit_factor", 0.0)),
            "aggregate_hit_rate": float(aggregate_row.get("trade_hit_rate", 0.0)),
            "aggregate_total_return": float(aggregate_row.get("trade_total_return", 0.0)),
            "aggregate_max_drawdown": float(aggregate_row.get("trade_max_drawdown", 0.0)),
            "aggregate_avg_trade": float(aggregate_row.get("trade_average_trade", 0.0)),
            "aggregate_long_only_return": float(aggregate_row.get("trade_long_only_return", 0.0)),
            "aggregate_short_only_return": float(aggregate_row.get("trade_short_only_return", 0.0)),
            "aggregate_long_threshold": float(aggregate_row.get("trade_long_threshold", 0.0)),
            "aggregate_short_threshold": float(aggregate_row.get("trade_short_threshold", 0.0)),
            "worst_fold_net_r": float(fold_nets.min() if not fold_nets.empty else 0.0),
            "worst_fold_pf": float(finite_pfs.min() if not finite_pfs.empty else np.nan),
            "min_fold_trades": float(pd.to_numeric(per_fold.get("trade_trades", pd.Series(dtype=float)), errors="coerce").fillna(0.0).min()) if not per_fold.empty else 0.0,
            "all_folds_positive": int(all_folds_pos),
            "positive_aggregate": int(float(aggregate_row.get("trade_net_r", 0.0)) > 0.0),
            "score": float(score),
        }
        summary_rows.append(summary_row)
        best_outputs[summary_row["candidate_name"]] = result
        print(
            f"[{combo_index}/{total}] net_r={summary_row['aggregate_net_r']:.3f} trades={summary_row['aggregate_trades']:.0f} "
            f"pf={summary_row['aggregate_pf']:.3f} gates={summary_row['selection_gates']}",
            flush=True,
        )

    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        raise SystemExit("No candidate results were produced.")
    summary = summary.sort_values(
        [
            "all_folds_positive",
            "positive_aggregate",
            "worst_fold_net_r",
            "aggregate_net_r",
            "aggregate_pf",
            "aggregate_trades",
            "score",
        ],
        ascending=[False, False, False, False, False, False, False],
    ).reset_index(drop=True)

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_prefix.with_name(args.out_prefix.name + "_summary.csv")
    summary.to_csv(summary_path, index=False)

    best_name = str(summary.iloc[0]["candidate_name"])
    best_result = best_outputs[best_name]
    best_folds_path = args.out_prefix.with_name(args.out_prefix.name + "_best_folds.csv")
    best_trades_path = args.out_prefix.with_name(args.out_prefix.name + "_best_trades.csv")
    best_predictions_path = args.out_prefix.with_name(args.out_prefix.name + "_best_predictions.csv")
    best_thresholds_path = args.out_prefix.with_name(args.out_prefix.name + "_best_thresholds.csv")
    best_meta_path = args.out_prefix.with_name(args.out_prefix.name + "_best_meta.json")
    best_result["folds"].to_csv(best_folds_path, index=False)
    best_result["trades"].to_csv(best_trades_path, index=False)
    best_result["predictions"].to_csv(best_predictions_path, index=False)
    best_result["thresholds"].to_csv(best_thresholds_path, index=False)
    best_meta_path.write_text(
        json.dumps(
            {
                "best_candidate_name": best_name,
                "selection_gates": summary.iloc[0]["selection_gates"],
                "aggregate_net_r": float(summary.iloc[0]["aggregate_net_r"]),
                "aggregate_trades": float(summary.iloc[0]["aggregate_trades"]),
                "aggregate_pf": float(summary.iloc[0]["aggregate_pf"]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(summary.head(20).to_string(index=False))
    print(f"\nSaved summary to {summary_path}")
    print(f"Saved best folds to {best_folds_path}")
    print(f"Saved best trades to {best_trades_path}")
    print(f"Saved best predictions to {best_predictions_path}")
    print(f"Saved best thresholds to {best_thresholds_path}")
    print(f"Saved best metadata to {best_meta_path}")


if __name__ == "__main__":
    main()
