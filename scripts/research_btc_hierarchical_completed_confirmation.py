from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research_btc_astro_cycle_timing import load_bybit_cached  # noqa: E402
from scripts.research_btc_hierarchical_cascade_backtest import trade_summary  # noqa: E402
from scripts.research_btc_hierarchical_hot_retest_1m import (  # noqa: E402
    chronological_orders,
    filled_trades,
    find_retest_fill,
    hot_metrics,
    limit_geometry,
    order_summary,
    prepare_hot_frame,
    retest_price,
)
from scripts.research_btc_hierarchical_path_walkforward import (  # noqa: E402
    undo_balanced_prior,
)
from scripts.research_btc_hierarchical_reversal import (  # noqa: E402
    LayerSpec,
    add_indicators,
    balanced_weights,
    build_features,
    clean_features_from_train,
    fit_model,
    json_default,
    make_model,
    parse_float_list,
    parse_str_list,
    parse_utc_datetime,
    predict_probability,
    resample_ohlc,
    safe_average_precision,
    safe_roc_auc,
)
from scripts.research_btc_hierarchical_wyckoff_1m import combine_cascades  # noqa: E402
from scripts.research_btc_ltf_calendar_probability import DEFAULT_CACHE_DIR  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass(frozen=True)
class ConfirmationLayer:
    prefix: str
    name: str
    child_tf: str
    parent_tf: str
    target_column: str
    ancestor_prefixes: tuple[str, ...]
    zone_tf: str
    retest_wait_minutes: int

    @property
    def spec(self) -> LayerSpec:
        return LayerSpec(self.name, self.child_tf, self.parent_tf, (1,))


LAYERS = [
    ConfirmationLayer(
        "l4h",
        "4h_to_1d",
        "4h",
        "1d",
        "target_4h",
        ("l4h",),
        "15m",
        720,
    ),
    ConfirmationLayer(
        "l1h",
        "1h_to_4h",
        "1h",
        "4h",
        "target_1h",
        ("l4h", "l1h"),
        "5m",
        240,
    ),
    ConfirmationLayer(
        "l15",
        "15m_to_1h",
        "15m",
        "1h",
        "target_15m",
        ("l4h", "l1h", "l15"),
        "1m",
        60,
    ),
    ConfirmationLayer(
        "l5",
        "5m_to_15m",
        "5m",
        "15m",
        "target_5m",
        ("l4h", "l1h", "l15", "l5"),
        "1m",
        30,
    ),
]
LAYER_BY_PREFIX = {layer.prefix: layer for layer in LAYERS}


def percentile_against(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ref = np.sort(reference[np.isfinite(reference)].astype(float))
    if len(ref) == 0:
        return np.full(len(values), np.nan, dtype=float)
    return np.searchsorted(ref, values, side="right").astype(float) / float(len(ref))


def safe_float(value: Any, default: float = 0.0) -> float:
    value = float(value)
    return value if math.isfinite(value) else default


def build_frames(raw_1m: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frame_5m = (
        raw_1m.set_index("open_time")
        .resample("5min", label="left", closed="left")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
        .reset_index()
    )
    frame_5m["close_time"] = (
        frame_5m["open_time"] + pd.Timedelta(minutes=5) - pd.Timedelta(milliseconds=1)
    )
    frame_5m = frame_5m[
        ["open_time", "close_time", "open", "high", "low", "close", "volume"]
    ]
    return {
        "1m": raw_1m.reset_index(drop=True),
        "5m": frame_5m.reset_index(drop=True),
        "15m": resample_ohlc(raw_1m, "15m"),
        "1h": resample_ohlc(raw_1m, "1h"),
        "4h": resample_ohlc(raw_1m, "4h"),
    }


def build_layer_table(
    cascade: pd.DataFrame,
    child_frame: pd.DataFrame,
    layer: ConfirmationLayer,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    score_columns: list[str] = []
    for prefix in layer.ancestor_prefixes:
        for direction in ["low", "high"]:
            score_columns.extend([f"{prefix}_{direction}_raw", f"{prefix}_{direction}_pct"])
    base_columns = [
        layer.target_column,
        f"{layer.prefix}_low_truth",
        f"{layer.prefix}_high_truth",
        *score_columns,
    ]
    base = (
        cascade.loc[:, base_columns]
        .sort_values(layer.target_column)
        .drop_duplicates(layer.target_column, keep="first")
        .rename(columns={layer.target_column: "target_time"})
        .reset_index(drop=True)
    )
    base["target_time"] = pd.to_datetime(base["target_time"], utc=True)

    completed = add_indicators(child_frame).reset_index(drop=True)
    completed_features, feature_groups = build_features(completed, layer.spec)
    completed_features = completed_features.add_prefix("completed_")
    completed_rows = pd.concat(
        [
            completed[["open_time", "close_time", "open", "high", "low", "close", "volume"]],
            completed_features,
        ],
        axis=1,
    ).rename(columns={"open_time": "target_time", "close_time": "decision_time"})
    completed_rows["target_time"] = pd.to_datetime(completed_rows["target_time"], utc=True)
    completed_rows["decision_time"] = pd.to_datetime(
        completed_rows["decision_time"],
        utc=True,
    )
    table = base.merge(completed_rows, on="target_time", how="inner")
    prefixed_groups = {
        name: [f"completed_{column}" for column in columns]
        for name, columns in feature_groups.items()
    }
    feature_sets = {
        "prior_scores": list(score_columns),
        "completed_time_plus_prior": [
            *prefixed_groups["time"],
            *score_columns,
        ],
        "completed_price_plus_prior": [
            *prefixed_groups["price"],
            *score_columns,
        ],
        "completed_price_time_no_prior": list(prefixed_groups["price_time"]),
        "completed_price_time_plus_prior": [
            *prefixed_groups["price_time"],
            *score_columns,
        ],
    }
    return table.sort_values("target_time").reset_index(drop=True), feature_sets


def fit_completed_posteriors(
    table: pd.DataFrame,
    layer: ConfirmationLayer,
    feature_sets: dict[str, list[str]],
    *,
    train_end: pd.Timestamp,
    validation_end: pd.Timestamp,
    model_name: str,
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    out = table.copy()
    train_mask = (out["decision_time"] < train_end).to_numpy()
    validation_mask = (
        (out["decision_time"] >= train_end)
        & (out["decision_time"] < validation_end)
    ).to_numpy()
    test_mask = (out["decision_time"] >= validation_end).to_numpy()
    diagnostics: list[dict[str, Any]] = []
    is_last_child = (
        out["completed_parent_is_last_child"].to_numpy(dtype=float) > 0.5
    )
    # The pre-candle hierarchy already carries the calendar/cycle prior. Once
    # the candle closes, price context is the useful Bayesian-style update;
    # completed-candle time features consistently reduced OOS discrimination.
    stored_feature_set = "completed_price_plus_prior"

    for feature_set, feature_columns in feature_sets.items():
        clean = clean_features_from_train(out.loc[:, feature_columns], train_mask)
        x = clean.to_numpy(dtype=np.float32, copy=False)
        for direction in ["low", "high"]:
            y = out[f"{layer.prefix}_{direction}_truth"].astype(np.int8).to_numpy()
            model = make_model(model_name, seed)
            fit_model(
                model,
                x[train_mask],
                y[train_mask],
                balanced_weights(y[train_mask]),
            )
            raw = predict_probability(model, x)
            prior_rate = float(y[train_mask].mean())
            probability = undo_balanced_prior(raw, prior_rate)
            if feature_set == stored_feature_set:
                percentile = percentile_against(probability[train_mask], probability)
                out[f"posterior_{direction}_prob"] = probability
                out[f"posterior_{direction}_pct"] = percentile
            prior_score = out[f"{layer.prefix}_{direction}_pct"].to_numpy(dtype=float)

            for split_name, split_mask in [
                ("validation", validation_mask),
                ("test", test_mask),
            ]:
                for parent_state, state_mask in [
                    ("all", np.ones(len(out), dtype=bool)),
                    ("open", ~is_last_child),
                    ("closed", is_last_child),
                ]:
                    mask = split_mask & state_mask
                    diagnostics.append(
                        {
                            "layer": layer.prefix,
                            "layer_name": layer.name,
                            "direction": direction,
                            "feature_set": feature_set,
                            "parent_state": parent_state,
                            "split": split_name,
                            "rows": int(mask.sum()),
                            "base_rate": float(y[mask].mean()),
                            "prior_ap": safe_average_precision(
                                y[mask],
                                prior_score[mask],
                            ),
                            "posterior_ap": safe_average_precision(
                                y[mask],
                                probability[mask],
                            ),
                            "prior_auc": safe_roc_auc(
                                y[mask],
                                prior_score[mask],
                            ),
                            "posterior_auc": safe_roc_auc(
                                y[mask],
                                probability[mask],
                            ),
                        }
                    )
    return out, diagnostics


def zone_candle(
    zone_frame: pd.DataFrame,
    zone_times: np.ndarray,
    *,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    direction: str,
) -> tuple[int, dict[str, float]] | None:
    start_idx = int(np.searchsorted(zone_times, start_time.value, side="left"))
    end_idx = int(np.searchsorted(zone_times, end_time.value, side="left"))
    if start_idx >= end_idx or start_idx < 20:
        return None
    segment = zone_frame.iloc[start_idx:end_idx]
    if direction == "long":
        extreme_idx = int(segment["low"].idxmin())
    else:
        extreme_idx = int(segment["high"].idxmax())
    best: tuple[int, dict[str, float]] | None = None
    for idx in range(extreme_idx, end_idx):
        metrics = hot_metrics(zone_frame, idx, direction, "reversal")
        if metrics is None:
            continue
        if best is None or metrics["hotness_score"] > best[1]["hotness_score"]:
            best = (idx, metrics)
    return best


def simulate_zone_trade_multi_rr(
    execution_frame: pd.DataFrame,
    *,
    fill_idx: int,
    entry: float,
    stop: float,
    direction: str,
    rr_values: list[float],
    max_hold_bars: int,
    cost_r: float,
) -> dict[float, dict[str, Any]]:
    risk = entry - stop if direction == "long" else stop - entry
    targets = {
        rr: entry + rr * risk if direction == "long" else entry - rr * risk
        for rr in rr_values
    }
    end_idx = min(len(execution_frame) - 1, fill_idx + max_hold_bars - 1)
    unresolved = set(rr_values)
    results: dict[float, dict[str, Any]] = {}
    mfe_r = 0.0
    mae_r = 0.0
    for cursor in range(fill_idx, end_idx + 1):
        high = float(execution_frame["high"].iloc[cursor])
        low = float(execution_frame["low"].iloc[cursor])
        hit_stop = low <= stop if direction == "long" else high >= stop
        if hit_stop:
            adverse = (entry - low) / risk if direction == "long" else (high - entry) / risk
            mae_r = max(mae_r, adverse)
            for rr in unresolved:
                results[rr] = {
                    "result_r": -1.0 - cost_r,
                    "exit_idx": cursor,
                    "exit_time": pd.Timestamp(
                        execution_frame["close_time"].iloc[cursor]
                    ).tz_convert("UTC"),
                    "exit_reason": "stop",
                    "mfe_r": mfe_r,
                    "mae_r": mae_r,
                    "cost_r": cost_r,
                }
            unresolved.clear()
            break

        # No target credit on the fill bar because the OHLC path before the
        # retest is unknown.
        if cursor == fill_idx:
            continue
        favorable = (high - entry) / risk if direction == "long" else (entry - low) / risk
        adverse = (entry - low) / risk if direction == "long" else (high - entry) / risk
        mfe_r = max(mfe_r, favorable)
        mae_r = max(mae_r, adverse)
        for rr in list(unresolved):
            touched = high >= targets[rr] if direction == "long" else low <= targets[rr]
            if touched:
                results[rr] = {
                    "result_r": rr - cost_r,
                    "exit_idx": cursor,
                    "exit_time": pd.Timestamp(
                        execution_frame["close_time"].iloc[cursor]
                    ).tz_convert("UTC"),
                    "exit_reason": "target",
                    "mfe_r": mfe_r,
                    "mae_r": mae_r,
                    "cost_r": cost_r,
                }
                unresolved.remove(rr)
        if not unresolved:
            break

    if unresolved:
        exit_price = float(execution_frame["close"].iloc[end_idx])
        timeout_r = (
            (exit_price - entry) / risk
            if direction == "long"
            else (entry - exit_price) / risk
        )
        for rr in unresolved:
            results[rr] = {
                "result_r": timeout_r - cost_r,
                "exit_idx": end_idx,
                "exit_time": pd.Timestamp(
                    execution_frame["close_time"].iloc[end_idx]
                ).tz_convert("UTC"),
                "exit_reason": "timeout",
                "mfe_r": mfe_r,
                "mae_r": mae_r,
                "cost_r": cost_r,
            }
    return results


def build_confirmation_retest_candidates(
    layer_tables: dict[str, pd.DataFrame],
    zone_frames: dict[str, pd.DataFrame],
    execution_frame: pd.DataFrame,
    *,
    layers: list[ConfirmationLayer],
    minimum_prior_pct: float,
    minimum_posterior_pct: float,
    rr_values: list[float],
    retest_mode: str,
    stop_buffer_atr: float,
    min_risk_pct: float,
    cost_bps_round_trip: float,
    max_friction_r: float,
    max_hold_bars: int,
) -> pd.DataFrame:
    execution_times = pd.DatetimeIndex(
        pd.to_datetime(execution_frame["open_time"], utc=True)
    ).as_unit("ns").asi8
    zone_times = {
        timeframe: pd.DatetimeIndex(
            pd.to_datetime(frame["open_time"], utc=True)
        ).as_unit("ns").asi8
        for timeframe, frame in zone_frames.items()
    }
    rows: list[dict[str, Any]] = []
    for layer in layers:
        table = layer_tables[layer.prefix]
        zone_frame = zone_frames[layer.zone_tf]
        layer_zone_times = zone_times[layer.zone_tf]
        for direction_name, trade_direction in [("low", "long"), ("high", "short")]:
            prior_columns = [
                f"{prefix}_{direction_name}_pct" for prefix in layer.ancestor_prefixes
            ]
            work = table.copy()
            work["prior_chain_min"] = work[prior_columns].min(axis=1)
            work = work[
                (work["prior_chain_min"] >= minimum_prior_pct)
                & (work[f"posterior_{direction_name}_pct"] >= minimum_posterior_pct)
            ]
            print(
                f"  {layer.prefix}/{trade_direction}: "
                f"{len(work):,} completed-candle confirmations to simulate"
            )
            for count, item in enumerate(work.itertuples(index=False), start=1):
                start_time = pd.Timestamp(item.target_time).tz_convert("UTC")
                decision_time = pd.Timestamp(item.decision_time).tz_convert("UTC")
                zone = zone_candle(
                    zone_frame,
                    layer_zone_times,
                    start_time=start_time,
                    end_time=decision_time,
                    direction=trade_direction,
                )
                if zone is None:
                    continue
                zone_idx, zone_metrics = zone
                zone_row = zone_frame.iloc[zone_idx]
                entry = retest_price(zone_row, retest_mode)
                geometry = limit_geometry(
                    zone_row,
                    entry=entry,
                    direction=trade_direction,
                    min_risk_pct=min_risk_pct,
                    stop_buffer_atr=stop_buffer_atr,
                    cost_bps_round_trip=cost_bps_round_trip,
                )
                if geometry is None or geometry["cost_r"] > max_friction_r:
                    continue

                order_active_idx = int(
                    np.searchsorted(execution_times, decision_time.value, side="left")
                )
                if order_active_idx >= len(execution_frame):
                    continue
                fill_idx = find_retest_fill(
                    execution_frame,
                    search_start_idx=order_active_idx,
                    wait_bars=layer.retest_wait_minutes,
                    entry=entry,
                    direction=trade_direction,
                    order_kind="limit_retest",
                )
                order_expiry_idx = min(
                    len(execution_frame) - 1,
                    order_active_idx + layer.retest_wait_minutes - 1,
                )
                outcomes = (
                    simulate_zone_trade_multi_rr(
                        execution_frame,
                        fill_idx=fill_idx,
                        entry=entry,
                        stop=geometry["stop_price"],
                        direction=trade_direction,
                        rr_values=rr_values,
                        max_hold_bars=max_hold_bars,
                        cost_r=geometry["cost_r"],
                    )
                    if fill_idx is not None
                    else {}
                )
                candidate: dict[str, Any] = {
                    "stage": layer.prefix,
                    "stage_name": layer.name,
                    "stage_tf": layer.child_tf,
                    "parent_tf": layer.parent_tf,
                    "zone_tf": layer.zone_tf,
                    "direction": trade_direction,
                    "direction_name": direction_name,
                    "target_time": start_time,
                    "decision_time": decision_time,
                    "signal_idx": order_active_idx,
                    "order_active_idx": order_active_idx,
                    "order_expiry_idx": order_expiry_idx,
                    "entry_idx": float(fill_idx) if fill_idx is not None else np.nan,
                    "filled": float(fill_idx is not None),
                    "entry_price": entry,
                    "stop_price": geometry["stop_price"],
                    "planned_risk_pct": geometry["risk_pct"],
                    "planned_cost_r": geometry["cost_r"],
                    "prior_chain_min": float(item.prior_chain_min),
                    "cascade_min": float(item.prior_chain_min),
                    "posterior_prob": float(
                        getattr(item, f"posterior_{direction_name}_prob")
                    ),
                    "posterior_pct": float(
                        getattr(item, f"posterior_{direction_name}_pct")
                    ),
                    "parent_is_last_child": float(
                        item.completed_parent_is_last_child
                    ),
                    "parent_state": (
                        "closed"
                        if float(item.completed_parent_is_last_child) > 0.5
                        else "open"
                    ),
                    "stage_truth": float(
                        getattr(item, f"{layer.prefix}_{direction_name}_truth")
                    ),
                    "zone_idx": zone_idx,
                    "hot_idx": zone_idx,
                    "zone_time": pd.Timestamp(
                        zone_frame["open_time"].iloc[zone_idx]
                    ).tz_convert("UTC"),
                    **zone_metrics,
                }
                for rr in rr_values:
                    key = f"{rr:g}"
                    trade = outcomes.get(rr)
                    if trade is None:
                        candidate[f"result_r_{key}"] = np.nan
                        candidate[f"exit_idx_{key}"] = np.nan
                        candidate[f"exit_time_{key}"] = pd.NaT
                        candidate[f"exit_reason_{key}"] = ""
                        candidate[f"mfe_r_{key}"] = np.nan
                        candidate[f"mae_r_{key}"] = np.nan
                        candidate[f"cost_r_{key}"] = np.nan
                        candidate[f"label_end_idx_{key}"] = order_expiry_idx
                        candidate[f"label_end_time_{key}"] = pd.Timestamp(
                            execution_frame["close_time"].iloc[order_expiry_idx]
                        ).tz_convert("UTC")
                    else:
                        candidate[f"result_r_{key}"] = trade["result_r"]
                        candidate[f"exit_idx_{key}"] = trade["exit_idx"]
                        candidate[f"exit_time_{key}"] = trade["exit_time"]
                        candidate[f"exit_reason_{key}"] = trade["exit_reason"]
                        candidate[f"mfe_r_{key}"] = trade["mfe_r"]
                        candidate[f"mae_r_{key}"] = trade["mae_r"]
                        candidate[f"cost_r_{key}"] = trade["cost_r"]
                        candidate[f"label_end_idx_{key}"] = trade["exit_idx"]
                        candidate[f"label_end_time_{key}"] = trade["exit_time"]
                rows.append(candidate)
                if count % 2_000 == 0:
                    print(f"    simulated {count:,}/{len(work):,}")

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["decision_time"] = pd.to_datetime(out["decision_time"], utc=True)
    out["target_time"] = pd.to_datetime(out["target_time"], utc=True)
    out["zone_time"] = pd.to_datetime(out["zone_time"], utc=True)
    for rr in rr_values:
        key = f"{rr:g}"
        out[f"exit_time_{key}"] = pd.to_datetime(out[f"exit_time_{key}"], utc=True)
        out[f"label_end_time_{key}"] = pd.to_datetime(
            out[f"label_end_time_{key}"],
            utc=True,
        )
    return out.sort_values(["decision_time", "stage", "direction"]).reset_index(drop=True)


def positive_month_fraction(orders: pd.DataFrame, rr: float) -> float:
    if orders.empty:
        return 0.0
    key = f"{rr:g}"
    values = np.where(
        orders["filled"].to_numpy(dtype=float) > 0.5,
        orders[f"result_r_{key}"].to_numpy(dtype=float),
        0.0,
    )
    monthly = (
        orders.assign(
            month=orders["decision_time"].dt.to_period("M"),
            order_result_r=values,
        )
        .groupby("month")["order_result_r"]
        .sum()
    )
    return float((monthly > 0.0).mean()) if len(monthly) else 0.0


def select_validation_gate(
    validation: pd.DataFrame,
    *,
    rr: float,
    use_posterior: bool,
    prior_thresholds: list[float],
    posterior_thresholds: list[float],
    hotness_coverages: list[float],
    min_trades: int,
    min_positive_month_fraction: float,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for prior_threshold in prior_thresholds:
        prior_part = validation[validation["prior_chain_min"] >= prior_threshold]
        if prior_part.empty:
            continue
        post_grid = posterior_thresholds if use_posterior else [0.0]
        for posterior_threshold in post_grid:
            posterior_part = prior_part[
                prior_part["posterior_pct"] >= posterior_threshold
            ]
            if posterior_part.empty:
                continue
            for hotness_coverage in hotness_coverages:
                hotness_threshold = float(
                    np.quantile(
                        posterior_part["hotness_score"],
                        1.0 - hotness_coverage,
                    )
                )
                selected = posterior_part[
                    posterior_part["hotness_score"] >= hotness_threshold
                ].copy()
                selected["selection_score"] = (
                    selected["prior_chain_min"]
                    * (selected["posterior_pct"] if use_posterior else 1.0)
                    * selected["hotness_score"]
                )
                orders = chronological_orders(selected, rr)
                summary = order_summary(orders, rr)
                positive_fraction = positive_month_fraction(orders, rr)
                if int(summary["trades"]) < min_trades:
                    continue
                if float(summary["net_r"]) <= 0.0 or float(summary["profit_factor"]) <= 1.0:
                    continue
                if positive_fraction < min_positive_month_fraction:
                    continue
                objective = (
                    float(summary["net_r"])
                    / max(abs(float(summary["max_drawdown_r"])), 5.0)
                    + 0.20 * float(summary["avg_r"])
                    + 0.10 * positive_fraction
                )
                record = {
                    "active": True,
                    "prior_threshold": prior_threshold,
                    "posterior_threshold": posterior_threshold,
                    "hotness_threshold": hotness_threshold,
                    "hotness_coverage": hotness_coverage,
                    "positive_month_fraction": positive_fraction,
                    "objective": objective,
                    **summary,
                }
                if best is None or objective > float(best["objective"]):
                    best = record
    if best is None:
        return {
            "active": False,
            "prior_threshold": float("inf"),
            "posterior_threshold": float("inf"),
            "hotness_threshold": float("inf"),
            "hotness_coverage": 0.0,
            "positive_month_fraction": 0.0,
            "objective": 0.0,
            **trade_summary([]),
        }
    return best


def run_locked_test(
    candidates: pd.DataFrame,
    *,
    stage: str,
    direction_scope: str,
    parent_state_scope: str,
    rr: float,
    confirmation: str,
    validation_start: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    prior_thresholds: list[float],
    posterior_thresholds: list[float],
    hotness_coverages: list[float],
    min_validation_trades: int,
    min_positive_month_fraction: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    universe = candidates[candidates["stage"] == stage].copy()
    if direction_scope != "both":
        universe = universe[universe["direction"] == direction_scope]
    if parent_state_scope != "both":
        universe = universe[universe["parent_state"] == parent_state_scope]
    if confirmation == "exact_closed":
        if parent_state_scope != "closed":
            raise ValueError("Exact confirmation is causal only for a closed parent.")
        universe["posterior_pct"] = universe["stage_truth"]
    use_posterior = confirmation != "prior_only"
    key = f"{rr:g}"
    resolved = pd.to_datetime(universe[f"label_end_time_{key}"], utc=True)
    validation = universe[
        (universe["decision_time"] >= validation_start)
        & (universe["decision_time"] < test_start)
        & (resolved < test_start)
    ].copy()
    test = universe[
        (universe["decision_time"] >= test_start)
        & (universe["decision_time"] < test_end)
    ].copy()
    selection = select_validation_gate(
        validation,
        rr=rr,
        use_posterior=use_posterior,
        prior_thresholds=prior_thresholds,
        posterior_thresholds=posterior_thresholds,
        hotness_coverages=hotness_coverages,
        min_trades=min_validation_trades,
        min_positive_month_fraction=min_positive_month_fraction,
    )
    if bool(selection["active"]):
        selected = test[
            (test["prior_chain_min"] >= float(selection["prior_threshold"]))
            & (
                test["posterior_pct"]
                >= (
                    float(selection["posterior_threshold"])
                    if use_posterior
                    else 0.0
                )
            )
            & (test["hotness_score"] >= float(selection["hotness_threshold"]))
        ].copy()
        selected["selection_score"] = (
            selected["prior_chain_min"]
            * (selected["posterior_pct"] if use_posterior else 1.0)
            * selected["hotness_score"]
        )
        orders = chronological_orders(selected, rr)
    else:
        orders = test.iloc[:0].copy()
    trades = filled_trades(orders)
    summary = {
        "stage": stage,
        "direction_scope": direction_scope,
        "parent_state_scope": parent_state_scope,
        "rr": rr,
        "confirmation": confirmation,
        "validation_rows": len(validation),
        "test_rows": len(test),
        **{f"selected_{name}": value for name, value in selection.items()},
        **{f"test_{name}": value for name, value in order_summary(orders, rr).items()},
    }
    return summary, trades


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Completed-candle posterior confirmation at every BTC hierarchy layer, "
            "followed by causal LTF retest execution."
        )
    )
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2026-06-01")
    parser.add_argument("--train-end", default="2024-01-01")
    parser.add_argument("--validation-end", default="2025-01-01")
    parser.add_argument("--model", default="logit")
    parser.add_argument("--stages", default="l4h,l1h,l15,l5")
    parser.add_argument("--directions", default="both,long,short")
    parser.add_argument("--parent-states", default="both,open,closed")
    parser.add_argument("--rr-values", default="2,3,5,10")
    parser.add_argument("--minimum-prior-pct", type=float, default=0.50)
    parser.add_argument("--minimum-posterior-pct", type=float, default=0.50)
    parser.add_argument("--prior-thresholds", default="0.50,0.60,0.70,0.80,0.90")
    parser.add_argument("--posterior-thresholds", default="0.50,0.60,0.70,0.80,0.90,0.95")
    parser.add_argument("--hotness-coverages", default="0.25,0.50,0.75,1.0")
    parser.add_argument("--retest-mode", default="body_mid")
    parser.add_argument("--stop-buffer-atr", type=float, default=0.25)
    parser.add_argument("--min-risk-pct", type=float, default=0.0)
    parser.add_argument("--cost-bps-round-trip", type=float, default=8.0)
    parser.add_argument("--max-friction-r", type=float, default=0.50)
    parser.add_argument("--max-hold-bars", type=int, default=1440)
    parser.add_argument("--min-validation-trades", type=int, default=12)
    parser.add_argument("--min-positive-validation-month-fraction", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--cascade-cache",
        type=Path,
        default=Path("scripts/.cache/astro_cycle/hierarchical_cascade_hgb_price_time.pkl"),
    )
    parser.add_argument(
        "--prior-cascade-cache",
        type=Path,
        default=Path(
            "scripts/.cache/astro_cycle/hierarchical_cascade_hgb_price_time_train2023.pkl"
        ),
    )
    parser.add_argument("--prior-cascade-end", default="2024-01-01")
    parser.add_argument(
        "--posterior-cache",
        type=Path,
        default=Path(
            "scripts/.cache/astro_cycle/hierarchical_completed_confirmation_posteriors.pkl"
        ),
    )
    parser.add_argument("--refresh-posteriors", action="store_true")
    parser.add_argument("--posterior-only", action="store_true")
    parser.add_argument(
        "--candidate-cache",
        type=Path,
        default=Path(
            "scripts/.cache/astro_cycle/hierarchical_completed_confirmation_candidates.pkl"
        ),
    )
    parser.add_argument("--refresh-candidates", action="store_true")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("scripts/hierarchical_completed_confirmation.json"),
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("scripts/hierarchical_completed_confirmation_summary.csv"),
    )
    parser.add_argument(
        "--diagnostics-csv",
        type=Path,
        default=Path("scripts/hierarchical_completed_confirmation_diagnostics.csv"),
    )
    parser.add_argument(
        "--trades-csv",
        type=Path,
        default=Path("scripts/hierarchical_completed_confirmation_trades.csv"),
    )
    args = parser.parse_args()

    start = pd.Timestamp(parse_utc_datetime(args.start))
    end = pd.Timestamp(parse_utc_datetime(args.end))
    train_end = pd.Timestamp(parse_utc_datetime(args.train_end))
    validation_end = pd.Timestamp(parse_utc_datetime(args.validation_end))
    prior_end = pd.Timestamp(parse_utc_datetime(args.prior_cascade_end))
    stages = parse_str_list(args.stages)
    directions = parse_str_list(args.directions)
    parent_states = parse_str_list(args.parent_states)
    rr_values = parse_float_list(args.rr_values)
    prior_thresholds = parse_float_list(args.prior_thresholds)
    posterior_thresholds = parse_float_list(args.posterior_thresholds)
    hotness_coverages = parse_float_list(args.hotness_coverages)
    for stage in stages:
        if stage not in LAYER_BY_PREFIX:
            raise ValueError(f"Unknown stage {stage!r}; available={sorted(LAYER_BY_PREFIX)}")
    for parent_state in parent_states:
        if parent_state not in {"both", "open", "closed"}:
            raise ValueError(
                f"Unknown parent state {parent_state!r}; "
                "available=['both', 'closed', 'open']"
            )

    expected_config = {
        "start": start,
        "end": end,
        "train_end": train_end,
        "validation_end": validation_end,
        "model": args.model,
        "minimum_prior_pct": args.minimum_prior_pct,
        "minimum_posterior_pct": args.minimum_posterior_pct,
        "rr_values": rr_values,
        "retest_mode": args.retest_mode,
        "stop_buffer_atr": args.stop_buffer_atr,
        "min_risk_pct": args.min_risk_pct,
        "cost_bps_round_trip": args.cost_bps_round_trip,
        "max_friction_r": args.max_friction_r,
        "max_hold_bars": args.max_hold_bars,
        "stages": stages,
        "parent_states": parent_states,
    }
    if args.candidate_cache.exists() and not args.refresh_candidates:
        print(f"Loading candidate cache {args.candidate_cache}...")
        cached = pd.read_pickle(args.candidate_cache)
        candidates = cached["candidates"]
        diagnostics = cached["diagnostics"]
        config = cached["config"]
        for name, value in expected_config.items():
            cached_value = config.get(name)
            if isinstance(value, list):
                if not set(value).issubset(set(cached_value or [])):
                    raise ValueError(f"Cache lacks requested {name}; refresh candidates.")
            elif isinstance(value, pd.Timestamp):
                if pd.Timestamp(cached_value) != value:
                    raise ValueError(f"Cache {name} differs; refresh candidates.")
            elif cached_value != value:
                raise ValueError(f"Cache {name} differs; refresh candidates.")
    else:
        print("Loading 1m history and hierarchy cascade...")
        raw_1m = load_bybit_cached(args.symbol, "1m", start, end, args.cache_dir)
        frames = build_frames(raw_1m)
        selected_layers = [LAYER_BY_PREFIX[stage] for stage in stages]
        if args.posterior_cache.exists() and not args.refresh_posteriors:
            print(f"Loading posterior cache {args.posterior_cache}...")
            posterior_cached = pd.read_pickle(args.posterior_cache)
            layer_tables = posterior_cached["layer_tables"]
            diagnostics = posterior_cached["diagnostics"]
            missing = set(stages) - set(layer_tables)
            if missing:
                raise ValueError(
                    f"Posterior cache lacks stages {sorted(missing)}; "
                    "use --refresh-posteriors."
                )
        else:
            cascade = combine_cascades(
                args.cascade_cache,
                args.prior_cascade_cache,
                start,
                prior_end,
            )
            layer_tables = {}
            diagnostics_rows: list[dict[str, Any]] = []
            posterior_layers = LAYERS if args.posterior_only else selected_layers
            for layer in posterior_layers:
                print(f"Training completed-candle posterior for {layer.name}...")
                table, feature_sets = build_layer_table(
                    cascade,
                    frames[layer.child_tf],
                    layer,
                )
                scored, layer_diagnostics = fit_completed_posteriors(
                    table,
                    layer,
                    feature_sets,
                    train_end=train_end,
                    validation_end=validation_end,
                    model_name=args.model,
                    seed=args.seed,
                )
                layer_tables[layer.prefix] = scored
                diagnostics_rows.extend(layer_diagnostics)
            diagnostics = pd.DataFrame(diagnostics_rows)
            args.posterior_cache.parent.mkdir(parents=True, exist_ok=True)
            pd.to_pickle(
                {
                    "layer_tables": layer_tables,
                    "diagnostics": diagnostics,
                    "config": {
                        "start": start,
                        "end": end,
                        "train_end": train_end,
                        "validation_end": validation_end,
                        "model": args.model,
                    },
                },
                args.posterior_cache,
            )
            print(f"Wrote {args.posterior_cache}")

        if args.posterior_only:
            args.diagnostics_csv.write_text(diagnostics.to_csv(index=False), encoding="utf-8")
            args.output_json.write_text(
                json.dumps(
                    {
                        "config": expected_config,
                        "diagnostics": diagnostics.to_dict(orient="records"),
                    },
                    indent=2,
                    default=json_default,
                ),
                encoding="utf-8",
            )
            print(f"Wrote {args.output_json}")
            print(f"Wrote {args.diagnostics_csv}")
            print(diagnostics.to_string(index=False))
            return 0

        print("Preparing LTF zone frames...")
        required_zone_tfs = {layer.zone_tf for layer in selected_layers} | {"1m"}
        zone_frames = {
            timeframe: prepare_hot_frame(frames[timeframe], 20)
            for timeframe in required_zone_tfs
        }
        print("Building confirmation-gated retest candidates...")
        candidates = build_confirmation_retest_candidates(
            layer_tables,
            zone_frames,
            zone_frames["1m"],
            layers=selected_layers,
            minimum_prior_pct=args.minimum_prior_pct,
            minimum_posterior_pct=args.minimum_posterior_pct,
            rr_values=rr_values,
            retest_mode=args.retest_mode,
            stop_buffer_atr=args.stop_buffer_atr,
            min_risk_pct=args.min_risk_pct,
            cost_bps_round_trip=args.cost_bps_round_trip,
            max_friction_r=args.max_friction_r,
            max_hold_bars=args.max_hold_bars,
        )
        args.candidate_cache.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(
            {
                "candidates": candidates,
                "diagnostics": diagnostics,
                "config": expected_config,
            },
            args.candidate_cache,
        )
        print(f"Wrote {args.candidate_cache}")

    if candidates.empty:
        raise RuntimeError("No completed-confirmation retest candidates were built.")
    candidates["decision_time"] = pd.to_datetime(candidates["decision_time"], utc=True)
    if "hot_idx" not in candidates:
        candidates["hot_idx"] = candidates["zone_idx"]
    if "cascade_min" not in candidates:
        candidates["cascade_min"] = candidates["prior_chain_min"]
    for rr in rr_values:
        candidates[f"label_end_time_{rr:g}"] = pd.to_datetime(
            candidates[f"label_end_time_{rr:g}"],
            utc=True,
        )
    print(f"Confirmation retest candidates: {len(candidates):,}")

    summaries: list[dict[str, Any]] = []
    trade_frames: list[pd.DataFrame] = []
    for stage in stages:
        for direction in directions:
            for parent_state in parent_states:
                for rr in rr_values:
                    confirmations = ["prior_only", "prior_posterior"]
                    if parent_state == "closed":
                        confirmations.append("exact_closed")
                    for confirmation in confirmations:
                        summary, trades = run_locked_test(
                            candidates,
                            stage=stage,
                            direction_scope=direction,
                            parent_state_scope=parent_state,
                            rr=rr,
                            confirmation=confirmation,
                            validation_start=train_end,
                            test_start=validation_end,
                            test_end=end,
                            prior_thresholds=prior_thresholds,
                            posterior_thresholds=posterior_thresholds,
                            hotness_coverages=hotness_coverages,
                            min_validation_trades=args.min_validation_trades,
                            min_positive_month_fraction=args.min_positive_validation_month_fraction,
                        )
                        label = (
                            f"stage={stage}/direction={direction}/"
                            f"parent={parent_state}/rr={rr:g}/"
                            f"confirmation={summary['confirmation']}"
                        )
                        summary["experiment"] = label
                        summaries.append(summary)
                        trades.insert(0, "experiment", label)
                        trade_frames.append(trades)

    summary_table = pd.DataFrame(summaries).sort_values(
        ["test_net_r", "test_profit_factor", "test_trades"],
        ascending=[False, False, False],
    )
    trades_table = (
        pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    )
    args.summary_csv.write_text(summary_table.to_csv(index=False), encoding="utf-8")
    args.diagnostics_csv.write_text(diagnostics.to_csv(index=False), encoding="utf-8")
    args.trades_csv.write_text(trades_table.to_csv(index=False), encoding="utf-8")
    result = {
        "config": expected_config,
        "candidate_rows": len(candidates),
        "experiments": summary_table.to_dict(orient="records"),
        "diagnostics": diagnostics.to_dict(orient="records"),
    }
    args.output_json.write_text(
        json.dumps(result, indent=2, default=json_default),
        encoding="utf-8",
    )
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.diagnostics_csv}")
    print(f"Wrote {args.trades_csv}")
    print("\nPosterior diagnostics:")
    print(diagnostics.to_string(index=False))
    print("\nTop experiments:")
    print(summary_table.head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
