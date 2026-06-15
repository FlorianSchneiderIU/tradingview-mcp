from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research_btc_astro_cycle_timing import load_bybit_cached  # noqa: E402
from scripts.research_btc_hierarchical_cascade_backtest import (  # noqa: E402
    trade_summary,
)
from scripts.research_btc_hierarchical_path_walkforward import (  # noqa: E402
    CASCADE_FEATURES,
    PRICE_FEATURES,
    TIME_FEATURES,
    cascade_features,
    clean_matrix,
    directional_price_features,
    make_execution_model,
    period_start,
    time_features,
    undo_balanced_prior,
)
from scripts.research_btc_hierarchical_reversal import (  # noqa: E402
    json_default,
    parse_float_list,
    parse_str_list,
    parse_utc_datetime,
)
from scripts.research_btc_hierarchical_wyckoff_1m import (  # noqa: E402
    combine_cascades,
    prepare_1m_features,
)
from scripts.research_btc_ltf_calendar_probability import DEFAULT_CACHE_DIR  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


RETEST_MODES = ("body_proximal", "body_mid", "body_open", "range_mid")
CANDLE_ROLES = ("reversal", "exhaustion")
HOT_FEATURES = [
    "hotness_score",
    "hot_directional_body_atr",
    "hot_range_atr",
    "hot_body_fraction",
    "hot_directional_close_position",
    "hot_range_expansion",
    "hot_body_expansion",
    "hot_volume_ratio",
    "hot_break_5_atr",
    "hot_break_15_atr",
    "hot_directional_return_3_atr",
    "hot_offset_minutes",
    "planned_risk_pct",
    "planned_cost_r",
]
FEATURE_SETS = {
    "hot_only": list(dict.fromkeys(PRICE_FEATURES + TIME_FEATURES + HOT_FEATURES)),
    "hierarchy_only": list(dict.fromkeys(CASCADE_FEATURES + TIME_FEATURES)),
    "hierarchy_hot": list(
        dict.fromkeys(CASCADE_FEATURES + PRICE_FEATURES + TIME_FEATURES + HOT_FEATURES)
    ),
}


def safe_float(value: Any, default: float = 0.0) -> float:
    value = float(value)
    return value if math.isfinite(value) else default


def prepare_hot_frame(frame: pd.DataFrame, lookback: int) -> pd.DataFrame:
    out = prepare_1m_features(frame, lookback)
    candle_range = (out["high"] - out["low"]).replace(0.0, np.nan)
    body = (out["close"] - out["open"]).abs()
    out["hot_body_fraction_raw"] = body / candle_range
    out["hot_prior_range_median"] = candle_range.shift(1).rolling(
        lookback,
        min_periods=lookback,
    ).median()
    out["hot_prior_body_median"] = body.shift(1).rolling(
        lookback,
        min_periods=lookback,
    ).median()
    out["hot_range_expansion_raw"] = candle_range / out["hot_prior_range_median"].replace(
        0.0,
        np.nan,
    )
    out["hot_body_expansion_raw"] = body / out["hot_prior_body_median"].replace(
        0.0,
        np.nan,
    )
    for bars in [5, 15]:
        out[f"hot_prior_high_{bars}"] = (
            out["high"].shift(1).rolling(bars, min_periods=bars).max()
        )
        out[f"hot_prior_low_{bars}"] = (
            out["low"].shift(1).rolling(bars, min_periods=bars).min()
        )
    return out.replace([np.inf, -np.inf], np.nan)


def hot_metrics(
    frame: pd.DataFrame,
    idx: int,
    trade_direction: str,
    candle_role: str,
) -> dict[str, float] | None:
    row = frame.iloc[idx]
    atr = safe_float(row["atr"], float("nan"))
    if not math.isfinite(atr) or atr <= 0.0:
        return None
    direction = (
        trade_direction
        if candle_role == "reversal"
        else ("short" if trade_direction == "long" else "long")
    )
    is_long = direction == "long"
    sign = 1.0 if is_long else -1.0
    body_atr = sign * safe_float(row["body_atr"])
    close_position = safe_float(row["close_pos"], 0.5)
    directional_close_position = close_position if is_long else 1.0 - close_position
    body_fraction = safe_float(row["hot_body_fraction_raw"])
    range_atr = safe_float(row["range_atr"])
    range_expansion = safe_float(row["hot_range_expansion_raw"], 1.0)
    body_expansion = safe_float(row["hot_body_expansion_raw"], 1.0)
    volume_ratio = safe_float(row["wyckoff_volume_ratio"], 1.0)
    return_3_atr = sign * safe_float(row["return_3_atr"])
    close = safe_float(row["close"])
    break_5 = (
        (close - safe_float(row["hot_prior_high_5"], close)) / atr
        if is_long
        else (safe_float(row["hot_prior_low_5"], close) - close) / atr
    )
    break_15 = (
        (close - safe_float(row["hot_prior_high_15"], close)) / atr
        if is_long
        else (safe_float(row["hot_prior_low_15"], close) - close) / atr
    )
    break_5 = max(0.0, break_5)
    break_15 = max(0.0, break_15)

    if (
        body_atr < 0.20
        or directional_close_position < 0.60
        or body_fraction < 0.40
        or range_expansion < 0.70
    ):
        return None

    hotness = (
        min(body_atr / 1.25, 2.5)
        + 0.45 * min(range_atr / 1.75, 2.5)
        + 0.70 * min(body_fraction, 1.0)
        + 0.70 * min(directional_close_position, 1.0)
        + 0.30 * min(max(math.log2(max(volume_ratio, 0.25)), -1.0), 2.0)
        + 0.40 * min(range_expansion / 2.0, 2.5)
        + 0.25 * min(body_expansion / 2.0, 2.5)
        + 0.65 * min(break_5, 2.0)
        + 0.35 * min(break_15, 2.0)
        + 0.25 * min(max(return_3_atr, 0.0), 3.0)
    )
    return {
        "hotness_score": float(hotness),
        "hot_directional_body_atr": body_atr,
        "hot_range_atr": range_atr,
        "hot_body_fraction": body_fraction,
        "hot_directional_close_position": directional_close_position,
        "hot_range_expansion": range_expansion,
        "hot_body_expansion": body_expansion,
        "hot_volume_ratio": volume_ratio,
        "hot_break_5_atr": break_5,
        "hot_break_15_atr": break_15,
        "hot_directional_return_3_atr": return_3_atr,
        "hot_candle_aligned_with_trade": float(candle_role == "reversal"),
    }


def retest_price(row: pd.Series, mode: str) -> float:
    if mode == "body_proximal":
        return float(row["close"])
    if mode == "body_mid":
        return 0.5 * (float(row["open"]) + float(row["close"]))
    if mode == "body_open":
        return float(row["open"])
    if mode == "range_mid":
        return 0.5 * (float(row["high"]) + float(row["low"]))
    raise ValueError(f"Unknown retest mode: {mode}")


def find_retest_fill(
    frame: pd.DataFrame,
    *,
    search_start_idx: int,
    wait_bars: int,
    entry: float,
    direction: str,
    order_kind: str,
) -> int | None:
    stop_idx = min(len(frame), search_start_idx + wait_bars)
    for idx in range(search_start_idx, stop_idx):
        if order_kind == "limit_retest":
            if direction == "long":
                if float(frame["low"].iloc[idx]) <= entry:
                    return idx
            elif float(frame["high"].iloc[idx]) >= entry:
                return idx
        elif order_kind == "reclaim_stop":
            if direction == "long":
                if float(frame["high"].iloc[idx]) >= entry:
                    return idx
            elif float(frame["low"].iloc[idx]) <= entry:
                return idx
        else:
            raise ValueError(f"Unknown order kind: {order_kind}")
    return None


def limit_geometry(
    hot: pd.Series,
    *,
    entry: float,
    direction: str,
    min_risk_pct: float,
    stop_buffer_atr: float,
    cost_bps_round_trip: float,
) -> dict[str, float] | None:
    atr = safe_float(hot["atr"], float("nan"))
    if not math.isfinite(atr) or atr <= 0.0:
        return None
    if direction == "long":
        stop = min(
            float(hot["low"]) - stop_buffer_atr * atr,
            entry * (1.0 - min_risk_pct),
        )
        risk = entry - stop
    else:
        stop = max(
            float(hot["high"]) + stop_buffer_atr * atr,
            entry * (1.0 + min_risk_pct),
        )
        risk = stop - entry
    if not math.isfinite(risk) or risk <= 0.0:
        return None
    return {
        "stop_price": stop,
        "risk": risk,
        "risk_pct": risk / entry,
        "cost_r": (cost_bps_round_trip / 10_000.0) * entry / risk,
    }


def simulate_limit_multi_rr(
    frame: pd.DataFrame,
    *,
    hot_idx: int,
    fill_idx: int,
    entry: float,
    direction: str,
    rr_values: list[float],
    min_risk_pct: float,
    stop_buffer_atr: float,
    max_hold_bars: int,
    cost_bps_round_trip: float,
    max_friction_r: float,
) -> dict[float, dict[str, Any]] | None:
    hot = frame.iloc[hot_idx]
    geometry = limit_geometry(
        hot,
        entry=entry,
        direction=direction,
        min_risk_pct=min_risk_pct,
        stop_buffer_atr=stop_buffer_atr,
        cost_bps_round_trip=cost_bps_round_trip,
    )
    if geometry is None:
        return None
    stop = geometry["stop_price"]
    risk = geometry["risk"]
    cost_r = geometry["cost_r"]
    if cost_r > max_friction_r:
        return None
    if direction == "long":
        targets = {rr: entry + rr * risk for rr in rr_values}
    else:
        targets = {rr: entry - rr * risk for rr in rr_values}
    end_idx = min(len(frame) - 1, fill_idx + max_hold_bars - 1)
    unresolved = set(rr_values)
    results: dict[float, dict[str, Any]] = {}
    mfe_r = 0.0
    mae_r = 0.0
    for cursor in range(fill_idx, end_idx + 1):
        high = float(frame["high"].iloc[cursor])
        low = float(frame["low"].iloc[cursor])
        hit_stop = low <= stop if direction == "long" else high >= stop
        if hit_stop:
            adverse = (entry - low) / risk if direction == "long" else (high - entry) / risk
            mae_r = max(mae_r, adverse)
            for rr in unresolved:
                results[rr] = {
                    "result_r": -1.0 - cost_r,
                    "exit_idx": cursor,
                    "exit_time": pd.Timestamp(frame["close_time"].iloc[cursor]).tz_convert("UTC"),
                    "exit_reason": "stop",
                    "mfe_r": mfe_r,
                    "mae_r": mae_r,
                    "cost_r": cost_r,
                }
            unresolved.clear()
            break

        # The fill bar's path is unknown. Ignore targets and favorable excursion
        # on that bar so a pre-retest high/low cannot become a false target.
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
                    "exit_time": pd.Timestamp(frame["close_time"].iloc[cursor]).tz_convert("UTC"),
                    "exit_reason": "target",
                    "mfe_r": mfe_r,
                    "mae_r": mae_r,
                    "cost_r": cost_r,
                }
                unresolved.remove(rr)
        if not unresolved:
            break

    if unresolved:
        exit_price = float(frame["close"].iloc[end_idx])
        timeout_r = (
            (exit_price - entry) / risk
            if direction == "long"
            else (entry - exit_price) / risk
        )
        for rr in unresolved:
            results[rr] = {
                "result_r": timeout_r - cost_r,
                "exit_idx": end_idx,
                "exit_time": pd.Timestamp(frame["close_time"].iloc[end_idx]).tz_convert("UTC"),
                "exit_reason": "timeout",
                "mfe_r": mfe_r,
                "mae_r": mae_r,
                "cost_r": cost_r,
            }
    return results


def add_outcomes(
    candidate: dict[str, Any],
    outcomes: dict[float, dict[str, Any]] | None,
    rr_values: list[float],
    *,
    unfilled_expiry_idx: int,
    unfilled_expiry_time: pd.Timestamp,
) -> None:
    for rr in rr_values:
        key = f"{rr:g}"
        trade = outcomes.get(rr) if outcomes is not None else None
        if trade is None:
            candidate[f"result_r_{key}"] = np.nan
            candidate[f"exit_idx_{key}"] = np.nan
            candidate[f"exit_time_{key}"] = pd.NaT
            candidate[f"exit_reason_{key}"] = ""
            candidate[f"mfe_r_{key}"] = np.nan
            candidate[f"mae_r_{key}"] = np.nan
            candidate[f"cost_r_{key}"] = np.nan
            candidate[f"label_end_idx_{key}"] = int(unfilled_expiry_idx)
            candidate[f"label_end_time_{key}"] = unfilled_expiry_time
        else:
            candidate[f"result_r_{key}"] = float(trade["result_r"])
            candidate[f"exit_idx_{key}"] = int(trade["exit_idx"])
            candidate[f"exit_time_{key}"] = trade["exit_time"]
            candidate[f"exit_reason_{key}"] = str(trade["exit_reason"])
            candidate[f"mfe_r_{key}"] = float(trade["mfe_r"])
            candidate[f"mae_r_{key}"] = float(trade["mae_r"])
            candidate[f"cost_r_{key}"] = float(trade["cost_r"])
            candidate[f"label_end_idx_{key}"] = int(trade["exit_idx"])
            candidate[f"label_end_time_{key}"] = trade["exit_time"]


def build_hot_retest_candidates(
    cascade: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    candidate_start: pd.Timestamp,
    minimum_cascade: float,
    candle_roles: list[str],
    retest_modes: list[str],
    retest_wait_bars: int,
    rr_values: list[float],
    min_risk_pcts: list[float],
    max_hold_bars: int,
    stop_buffer_atr: float,
    cost_bps_round_trip: float,
    max_friction_r: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    one_minute_ns = pd.DatetimeIndex(pd.to_datetime(frame["open_time"], utc=True)).as_unit("ns").asi8
    eligible = cascade[pd.to_datetime(cascade["target_5m"], utc=True) >= candidate_start]
    rows: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "eligible_windows": 0,
        "hot_windows": 0,
        "placed_by_role_mode": {
            f"{role}:{mode}": 0 for role in candle_roles for mode in retest_modes
        },
        "filled_by_role_mode": {
            f"{role}:{mode}": 0 for role in candle_roles for mode in retest_modes
        },
    }

    for cascade_direction, direction in [("low", "long"), ("high", "short")]:
        selected = eligible[eligible[f"{cascade_direction}_cascade_min"] >= minimum_cascade]
        stats["eligible_windows"] += len(selected)
        print(f"  {direction}: scanning {len(selected):,} hierarchy windows")
        for count, item in enumerate(selected.itertuples(index=False), start=1):
            window_time = pd.Timestamp(item.target_5m).tz_convert("UTC")
            start_idx = int(np.searchsorted(one_minute_ns, window_time.value, side="left"))
            end_idx = min(len(frame), start_idx + 5)
            if start_idx < 25 or end_idx - start_idx < 5:
                continue

            window_had_hot = False
            for candle_role in candle_roles:
                for idx in range(start_idx, end_idx):
                    metrics = hot_metrics(frame, idx, direction, candle_role)
                    if metrics is None:
                        continue
                    window_had_hot = True
                    hot_idx = idx
                    hot = frame.iloc[hot_idx]
                    order_active_idx = hot_idx + 1
                    if order_active_idx >= len(frame):
                        continue
                    order_expiry_idx = min(
                        len(frame) - 1,
                        order_active_idx + retest_wait_bars - 1,
                    )
                    order_expiry_time = pd.Timestamp(
                        frame["close_time"].iloc[order_expiry_idx]
                    ).tz_convert("UTC")
                    base = {
                        "hot_idx": hot_idx,
                        "order_active_idx": order_active_idx,
                        "order_expiry_idx": order_expiry_idx,
                        "target_time": window_time,
                        "hot_time": pd.Timestamp(frame["open_time"].iloc[hot_idx]).tz_convert("UTC"),
                        "decision_time": pd.Timestamp(
                            frame["open_time"].iloc[order_active_idx]
                        ).tz_convert("UTC"),
                        "direction": direction,
                        "cascade_direction": cascade_direction,
                        "candle_role": candle_role,
                        "order_kind": (
                            "limit_retest"
                            if candle_role == "reversal"
                            else "reclaim_stop"
                        ),
                        "nested_truth": bool(getattr(item, f"nested_{cascade_direction}_truth")),
                        **cascade_features(item, cascade_direction),
                        **directional_price_features(hot, direction),
                        **time_features(
                            pd.Timestamp(frame["close_time"].iloc[hot_idx]).tz_convert("UTC")
                        ),
                        **metrics,
                        "hot_offset_minutes": float(hot_idx - start_idx),
                    }
                    for mode in retest_modes:
                        stats["placed_by_role_mode"][f"{candle_role}:{mode}"] += 1
                        entry = retest_price(hot, mode)
                        fill_idx = find_retest_fill(
                            frame,
                            search_start_idx=order_active_idx,
                            wait_bars=retest_wait_bars,
                            entry=entry,
                            direction=direction,
                            order_kind=base["order_kind"],
                        )
                        if fill_idx is not None:
                            stats["filled_by_role_mode"][f"{candle_role}:{mode}"] += 1
                        for min_risk_pct in min_risk_pcts:
                            geometry = limit_geometry(
                                hot,
                                entry=entry,
                                direction=direction,
                                min_risk_pct=min_risk_pct,
                                stop_buffer_atr=stop_buffer_atr,
                                cost_bps_round_trip=cost_bps_round_trip,
                            )
                            if geometry is None or geometry["cost_r"] > max_friction_r:
                                continue
                            candidate = {
                                **base,
                                "signal_idx": order_active_idx,
                                "entry_idx": (
                                    float(fill_idx) if fill_idx is not None else np.nan
                                ),
                                "filled": float(fill_idx is not None),
                                "retest_mode": mode,
                                "retest_delay_bars": (
                                    float(fill_idx - order_active_idx + 1)
                                    if fill_idx is not None
                                    else np.nan
                                ),
                                "entry_price": entry,
                                "stop_price": geometry["stop_price"],
                                "planned_risk_pct": geometry["risk_pct"],
                                "planned_cost_r": geometry["cost_r"],
                                "min_risk_pct": float(min_risk_pct),
                            }
                            outcomes = (
                                simulate_limit_multi_rr(
                                    frame,
                                    hot_idx=hot_idx,
                                    fill_idx=fill_idx,
                                    entry=entry,
                                    direction=direction,
                                    rr_values=rr_values,
                                    min_risk_pct=min_risk_pct,
                                    stop_buffer_atr=stop_buffer_atr,
                                    max_hold_bars=max_hold_bars,
                                    cost_bps_round_trip=cost_bps_round_trip,
                                    max_friction_r=max_friction_r,
                                )
                                if fill_idx is not None
                                else None
                            )
                            if fill_idx is not None and outcomes is None:
                                continue
                            add_outcomes(
                                candidate,
                                outcomes,
                                rr_values,
                                unfilled_expiry_idx=order_expiry_idx,
                                unfilled_expiry_time=order_expiry_time,
                            )
                            rows.append(candidate)
            if window_had_hot:
                stats["hot_windows"] += 1
            if count % 5_000 == 0:
                print(f"    scanned {count:,}/{len(selected):,}")

    out = pd.DataFrame(rows)
    if out.empty:
        return out, stats
    out["decision_time"] = pd.to_datetime(out["decision_time"], utc=True)
    out["hot_time"] = pd.to_datetime(out["hot_time"], utc=True)
    for rr in rr_values:
        out[f"exit_time_{rr:g}"] = pd.to_datetime(out[f"exit_time_{rr:g}"], utc=True)
        out[f"label_end_time_{rr:g}"] = pd.to_datetime(
            out[f"label_end_time_{rr:g}"],
            utc=True,
        )
    out = out.sort_values(
        [
            "candle_role",
            "retest_mode",
            "min_risk_pct",
            "direction",
            "hot_idx",
            "cascade_min",
            "hotness_score",
        ],
        ascending=[True, True, True, True, True, False, False],
    )
    out = out.drop_duplicates(
        [
            "candle_role",
            "retest_mode",
            "min_risk_pct",
            "direction",
            "hot_idx",
            "target_time",
        ],
        keep="first",
    )
    return (
        out.sort_values(["decision_time", "cascade_min"], ascending=[True, False]).reset_index(drop=True),
        stats,
    )


def rows_to_summary(rows: pd.DataFrame, rr: float) -> dict[str, Any]:
    if rows.empty:
        return trade_summary([])
    key = f"{rr:g}"
    trades = [
        {
            "result_r": float(row[f"result_r_{key}"]),
            "exit_reason": str(row[f"exit_reason_{key}"]),
            "mfe_r": float(row[f"mfe_r_{key}"]),
            "cost_r": float(row[f"cost_r_{key}"]),
        }
        for _, row in rows.iterrows()
    ]
    return trade_summary(trades)


def first_qualifying_orders(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return (
        frame.sort_values(
            ["target_time", "direction", "hot_idx", "cascade_min"],
            ascending=[True, True, True, False],
        )
        .drop_duplicates(["target_time", "direction"], keep="first")
        .copy()
    )


def chronological_orders(frame: pd.DataFrame, rr: float) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    key = f"{rr:g}"
    ordered = first_qualifying_orders(frame).sort_values(
        ["order_active_idx", "selection_score", "cascade_min", "hotness_score"],
        ascending=[True, False, False, False],
    )
    keep: list[int] = []
    blocked_until = -1
    for idx, row in ordered.iterrows():
        order_active_idx = int(row["order_active_idx"])
        label_end_idx = safe_float(row[f"label_end_idx_{key}"], float("nan"))
        if order_active_idx <= blocked_until or not math.isfinite(label_end_idx):
            continue
        keep.append(idx)
        blocked_until = max(blocked_until, int(label_end_idx))
    return ordered.loc[keep].sort_values("decision_time").copy()


def filled_trades(orders: pd.DataFrame) -> pd.DataFrame:
    if orders.empty:
        return orders.copy()
    return orders[orders["filled"] > 0.5].copy()


def order_summary(orders: pd.DataFrame, rr: float) -> dict[str, Any]:
    trades = filled_trades(orders)
    summary = rows_to_summary(trades, rr)
    summary["orders"] = len(orders)
    summary["unfilled_orders"] = len(orders) - len(trades)
    summary["fill_rate"] = len(trades) / len(orders) if len(orders) else float("nan")
    summary["net_r_per_order"] = (
        float(summary["net_r"]) / len(orders) if len(orders) else float("nan")
    )
    if not trades.empty:
        values = trades[f"result_r_{rr:g}"].to_numpy(dtype=float)
        equity = np.concatenate([[0.0], np.cumsum(values)])
        summary["max_drawdown_r"] = float(
            np.min(equity - np.maximum.accumulate(equity))
        )
    return summary


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


def select_gate(
    validation: pd.DataFrame,
    *,
    rr: float,
    score_column: str,
    cascade_thresholds: list[float],
    score_coverages: list[float],
    min_trades: int,
    min_positive_month_fraction: float,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for cascade_threshold in cascade_thresholds:
        hierarchy = validation[validation["cascade_min"] >= cascade_threshold]
        if hierarchy.empty:
            continue
        for coverage in score_coverages:
            score_threshold = float(np.quantile(hierarchy[score_column], 1.0 - coverage))
            eligible = hierarchy[hierarchy[score_column] >= score_threshold].copy()
            eligible["selection_score"] = (
                eligible["cascade_min"] * eligible[score_column]
            )
            orders = chronological_orders(eligible, rr)
            summary = order_summary(orders, rr)
            positive_fraction = positive_month_fraction(orders, rr)
            if int(summary["trades"]) < min_trades:
                continue
            if float(summary["net_r"]) <= 0.0 or float(summary["profit_factor"]) <= 1.0:
                continue
            if positive_fraction < min_positive_month_fraction:
                continue
            objective = (
                float(summary["net_r"]) / max(abs(float(summary["max_drawdown_r"])), 5.0)
                + 0.20 * float(summary["avg_r"])
                + 0.10 * positive_fraction
            )
            record = {
                "active": True,
                "cascade_threshold": cascade_threshold,
                "score_threshold": score_threshold,
                "score_coverage": coverage,
                "positive_month_fraction": positive_fraction,
                "objective": objective,
                **summary,
            }
            if best is None or objective > float(best["objective"]):
                best = record
    if best is None:
        return {
            "active": False,
            "cascade_threshold": float("inf"),
            "score_threshold": float("inf"),
            "score_coverage": 0.0,
            "positive_month_fraction": 0.0,
            "objective": 0.0,
            **trade_summary([]),
        }
    return best


def run_walkforward(
    candidates: pd.DataFrame,
    *,
    candle_role: str,
    retest_mode: str,
    min_risk_pct: float,
    rr: float,
    direction_scope: str,
    validation_months: int,
    start_month: str,
    end_month: str,
    cascade_thresholds: list[float],
    hotness_coverages: list[float],
    min_validation_trades: int,
    min_positive_month_fraction: float,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    key = f"{rr:g}"
    universe = candidates[
        (candidates["candle_role"] == candle_role)
        & (candidates["retest_mode"] == retest_mode)
        & np.isclose(candidates["min_risk_pct"].to_numpy(dtype=float), min_risk_pct)
    ].copy()
    if direction_scope != "both":
        universe = universe[universe["direction"] == direction_scope].copy()

    months = pd.period_range(start_month, end_month, freq="M")
    selected_parts: list[pd.DataFrame] = []
    monthly_rows: list[dict[str, Any]] = []
    for period in months:
        test_start = period_start(period)
        test_end = period_start(period + 1)
        validation_start = test_start - DateOffset(months=validation_months)
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
        if len(validation) < 50 or test.empty:
            monthly_rows.append(
                {
                    "month": str(period),
                    "active": False,
                    "skip_reason": "insufficient_data",
                    "validation_rows": len(validation),
                    "test_rows": len(test),
                }
            )
            continue

        selection = select_gate(
            validation,
            rr=rr,
            score_column="hotness_score",
            cascade_thresholds=cascade_thresholds,
            score_coverages=hotness_coverages,
            min_trades=min_validation_trades,
            min_positive_month_fraction=min_positive_month_fraction,
        )
        if bool(selection["active"]):
            selected_test = test[
                (test["cascade_min"] >= float(selection["cascade_threshold"]))
                & (test["hotness_score"] >= float(selection["score_threshold"]))
            ].copy()
            selected_test["selection_score"] = (
                selected_test["cascade_min"] * selected_test["hotness_score"]
            )
            selected_test["month"] = str(period)
            selected_test["selected_cascade_threshold"] = float(
                selection["cascade_threshold"]
            )
            selected_test["selected_hotness_threshold"] = float(
                selection["score_threshold"]
            )
            selected_parts.append(selected_test)
        else:
            selected_test = test.iloc[:0].copy()
            selected_test["selection_score"] = np.nan
        test_summary = order_summary(chronological_orders(selected_test, rr), rr)
        monthly_rows.append(
            {
                "month": str(period),
                "active": bool(selection["active"]),
                "skip_reason": "" if bool(selection["active"]) else "validation_no_edge",
                "validation_rows": len(validation),
                "test_rows": len(test),
                **{
                    f"validation_{name}": value
                    for name, value in selection.items()
                    if name != "active"
                },
                **{f"test_{name}": value for name, value in test_summary.items()},
            }
        )

    scored = pd.concat(selected_parts, ignore_index=True) if selected_parts else universe.iloc[:0].copy()
    orders = chronological_orders(scored, rr)
    trades = filled_trades(orders)
    overall = order_summary(orders, rr)
    summary = {
        "candle_role": candle_role,
        "retest_mode": retest_mode,
        "min_risk_pct": min_risk_pct,
        "rr": rr,
        "direction_scope": direction_scope,
        "validation_months": validation_months,
        "months": len(months),
        "active_months": int(sum(bool(row.get("active")) for row in monthly_rows)),
        "positive_months": int(
            (
                trades.groupby("month")[f"result_r_{key}"].sum() > 0.0
            ).sum()
        )
        if not trades.empty
        else 0,
        **overall,
    }
    return summary, pd.DataFrame(monthly_rows), trades


def run_model_walkforward(
    candidates: pd.DataFrame,
    *,
    candle_role: str,
    retest_mode: str,
    min_risk_pct: float,
    rr: float,
    direction_scope: str,
    feature_set: str,
    model_name: str,
    validation_months: int,
    start_month: str,
    end_month: str,
    cascade_thresholds: list[float],
    score_coverages: list[float],
    min_validation_trades: int,
    min_positive_month_fraction: float,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    key = f"{rr:g}"
    columns = FEATURE_SETS[feature_set]
    universe = candidates[
        (candidates["candle_role"] == candle_role)
        & (candidates["retest_mode"] == retest_mode)
        & np.isclose(candidates["min_risk_pct"].to_numpy(dtype=float), min_risk_pct)
    ].copy()
    if direction_scope != "both":
        universe = universe[universe["direction"] == direction_scope].copy()
    universe["positive_order"] = (
        (universe["filled"] > 0.5) & (universe[f"result_r_{key}"] > 0.0)
    ).astype(np.int8)

    months = pd.period_range(start_month, end_month, freq="M")
    selected_parts: list[pd.DataFrame] = []
    monthly_rows: list[dict[str, Any]] = []
    for period in months:
        test_start = period_start(period)
        test_end = period_start(period + 1)
        validation_start = test_start - DateOffset(months=validation_months)
        resolved = pd.to_datetime(universe[f"label_end_time_{key}"], utc=True)
        train = universe[
            (universe["decision_time"] < validation_start)
            & (resolved < validation_start)
        ].copy()
        validation = universe[
            (universe["decision_time"] >= validation_start)
            & (universe["decision_time"] < test_start)
            & (resolved < test_start)
        ].copy()
        test = universe[
            (universe["decision_time"] >= test_start)
            & (universe["decision_time"] < test_end)
        ].copy()
        if (
            len(train) < 250
            or len(validation) < 50
            or test.empty
            or train["positive_order"].nunique() < 2
            or int(train["positive_order"].sum()) < 15
        ):
            monthly_rows.append(
                {
                    "month": str(period),
                    "active": False,
                    "skip_reason": "insufficient_data",
                    "train_rows": len(train),
                    "train_positives": int(train["positive_order"].sum()),
                    "validation_rows": len(validation),
                    "test_rows": len(test),
                }
            )
            continue

        train_x = clean_matrix(train, columns)
        validation_x = clean_matrix(validation, columns)
        test_x = clean_matrix(test, columns)
        model = make_execution_model(model_name, seed)
        model.fit(train_x, train["positive_order"])
        prior = float(train["positive_order"].mean())
        validation["model_score"] = undo_balanced_prior(
            model.predict_proba(validation_x)[:, 1],
            prior,
        )
        test["model_score"] = undo_balanced_prior(
            model.predict_proba(test_x)[:, 1],
            prior,
        )
        selection = select_gate(
            validation,
            rr=rr,
            score_column="model_score",
            cascade_thresholds=cascade_thresholds,
            score_coverages=score_coverages,
            min_trades=min_validation_trades,
            min_positive_month_fraction=min_positive_month_fraction,
        )
        if bool(selection["active"]):
            selected_test = test[
                (test["cascade_min"] >= float(selection["cascade_threshold"]))
                & (test["model_score"] >= float(selection["score_threshold"]))
            ].copy()
            selected_test["selection_score"] = (
                selected_test["cascade_min"] * selected_test["model_score"]
            )
            selected_test["month"] = str(period)
            selected_test["selected_cascade_threshold"] = float(
                selection["cascade_threshold"]
            )
            selected_test["selected_model_threshold"] = float(
                selection["score_threshold"]
            )
            selected_parts.append(selected_test)
        else:
            selected_test = test.iloc[:0].copy()
            selected_test["selection_score"] = np.nan
        test_summary = order_summary(chronological_orders(selected_test, rr), rr)
        monthly_rows.append(
            {
                "month": str(period),
                "active": bool(selection["active"]),
                "skip_reason": "" if bool(selection["active"]) else "validation_no_edge",
                "train_rows": len(train),
                "train_positives": int(train["positive_order"].sum()),
                "train_positive_rate": prior,
                "validation_rows": len(validation),
                "test_rows": len(test),
                **{
                    f"validation_{name}": value
                    for name, value in selection.items()
                    if name != "active"
                },
                **{f"test_{name}": value for name, value in test_summary.items()},
            }
        )

    scored = (
        pd.concat(selected_parts, ignore_index=True)
        if selected_parts
        else universe.iloc[:0].copy()
    )
    orders = chronological_orders(scored, rr)
    trades = filled_trades(orders)
    overall = order_summary(orders, rr)
    summary = {
        "candle_role": candle_role,
        "retest_mode": retest_mode,
        "min_risk_pct": min_risk_pct,
        "rr": rr,
        "direction_scope": direction_scope,
        "feature_set": feature_set,
        "model": model_name,
        "validation_months": validation_months,
        "months": len(months),
        "active_months": int(sum(bool(row.get("active")) for row in monthly_rows)),
        "positive_months": int(
            (trades.groupby("month")[f"result_r_{key}"].sum() > 0.0).sum()
        )
        if not trades.empty
        else 0,
        **overall,
    }
    return summary, pd.DataFrame(monthly_rows), trades


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hierarchy-gated hot 1m candle retest research for BTC reversals."
    )
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2026-06-01")
    parser.add_argument("--candidate-start", default="2023-01-01")
    parser.add_argument("--start-month", default="2025-01")
    parser.add_argument("--end-month", default="2026-05")
    parser.add_argument("--validation-months", type=int, default=6)
    parser.add_argument("--minimum-cascade", type=float, default=0.70)
    parser.add_argument("--cascade-thresholds", default="0.70,0.75,0.80,0.85,0.90")
    parser.add_argument("--hotness-coverages", default="0.10,0.20,0.35,0.50,0.70,1.0")
    parser.add_argument("--selection-methods", default="model")
    parser.add_argument("--models", default="logit")
    parser.add_argument("--feature-sets", default="hierarchy_hot")
    parser.add_argument("--candle-roles", default="reversal")
    parser.add_argument("--retest-modes", default="body_mid")
    parser.add_argument("--retest-wait-bars", type=int, default=10)
    parser.add_argument("--rr-values", default="2,5")
    parser.add_argument("--min-risk-pcts", default="0")
    parser.add_argument("--direction-scopes", default="both")
    parser.add_argument("--max-hold-bars", type=int, default=240)
    parser.add_argument("--stop-buffer-atr", type=float, default=0.50)
    parser.add_argument("--cost-bps-round-trip", type=float, default=8.0)
    parser.add_argument("--max-friction-r", type=float, default=0.50)
    parser.add_argument("--hot-lookback", type=int, default=20)
    parser.add_argument("--min-validation-trades", type=int, default=12)
    parser.add_argument("--min-positive-validation-month-fraction", type=float, default=0.50)
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
        "--candidate-cache",
        type=Path,
        default=Path(
            "scripts/.cache/astro_cycle/hierarchical_hot_retest_1m_reversal_q70.pkl"
        ),
    )
    parser.add_argument("--refresh-candidates", action="store_true")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("scripts/hierarchical_hot_retest_1m.json"),
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("scripts/hierarchical_hot_retest_1m_summary.csv"),
    )
    parser.add_argument(
        "--monthly-csv",
        type=Path,
        default=Path("scripts/hierarchical_hot_retest_1m_monthly.csv"),
    )
    parser.add_argument(
        "--trades-csv",
        type=Path,
        default=Path("scripts/hierarchical_hot_retest_1m_trades.csv"),
    )
    args = parser.parse_args()

    start = parse_utc_datetime(args.start)
    end = parse_utc_datetime(args.end)
    candidate_start = pd.Timestamp(parse_utc_datetime(args.candidate_start))
    prior_end = pd.Timestamp(parse_utc_datetime(args.prior_cascade_end))
    cascade_thresholds = parse_float_list(args.cascade_thresholds)
    hotness_coverages = parse_float_list(args.hotness_coverages)
    selection_methods = parse_str_list(args.selection_methods)
    models = parse_str_list(args.models)
    feature_sets = parse_str_list(args.feature_sets)
    candle_roles = parse_str_list(args.candle_roles)
    retest_modes = parse_str_list(args.retest_modes)
    rr_values = parse_float_list(args.rr_values)
    min_risk_pcts = parse_float_list(args.min_risk_pcts)
    direction_scopes = parse_str_list(args.direction_scopes)
    for method in selection_methods:
        if method not in {"rule", "model"}:
            raise ValueError("selection methods must be rule and/or model")
    for feature_set in feature_sets:
        if feature_set not in FEATURE_SETS:
            raise ValueError(
                f"Unknown feature set {feature_set!r}; available={sorted(FEATURE_SETS)}"
            )
    for candle_role in candle_roles:
        if candle_role not in CANDLE_ROLES:
            raise ValueError(
                f"Unknown candle role {candle_role!r}; available={CANDLE_ROLES}"
            )
    for mode in retest_modes:
        if mode not in RETEST_MODES:
            raise ValueError(f"Unknown retest mode {mode!r}; available={RETEST_MODES}")

    expected_config = {
        "start": pd.Timestamp(start),
        "end": pd.Timestamp(end),
        "candidate_start": candidate_start,
        "minimum_cascade": float(args.minimum_cascade),
        "candle_roles": candle_roles,
        "retest_modes": retest_modes,
        "retest_wait_bars": int(args.retest_wait_bars),
        "rr_values": rr_values,
        "min_risk_pcts": min_risk_pcts,
        "max_hold_bars": int(args.max_hold_bars),
        "stop_buffer_atr": float(args.stop_buffer_atr),
        "cost_bps_round_trip": float(args.cost_bps_round_trip),
        "max_friction_r": float(args.max_friction_r),
        "hot_lookback": int(args.hot_lookback),
    }
    if args.candidate_cache.exists() and not args.refresh_candidates:
        print(f"Loading candidate cache {args.candidate_cache}...")
        cached = pd.read_pickle(args.candidate_cache)
        candidates = cached["candidates"]
        build_stats = cached.get("build_stats", {})
        config = cached["config"]
        for name, value in expected_config.items():
            cached_value = config.get(name)
            if isinstance(value, list):
                if not set(value).issubset(set(cached_value or [])):
                    raise ValueError(
                        f"Candidate cache lacks requested {name}; use --refresh-candidates."
                    )
            elif isinstance(value, pd.Timestamp):
                if pd.Timestamp(cached_value) != value:
                    raise ValueError(
                        f"Candidate cache {name} differs; use --refresh-candidates."
                    )
            elif cached_value != value:
                raise ValueError(
                    f"Candidate cache {name} differs; use --refresh-candidates."
                )
    else:
        print("Loading 1m history and hierarchy cascade...")
        raw = load_bybit_cached(args.symbol, "1m", start, end, args.cache_dir)
        frame = prepare_hot_frame(raw, args.hot_lookback)
        cascade = combine_cascades(
            args.cascade_cache,
            args.prior_cascade_cache,
            candidate_start,
            prior_end,
        )
        print("Building hot-candle retest candidates...")
        candidates, build_stats = build_hot_retest_candidates(
            cascade,
            frame,
            candidate_start=candidate_start,
            minimum_cascade=args.minimum_cascade,
            candle_roles=candle_roles,
            retest_modes=retest_modes,
            retest_wait_bars=args.retest_wait_bars,
            rr_values=rr_values,
            min_risk_pcts=min_risk_pcts,
            max_hold_bars=args.max_hold_bars,
            stop_buffer_atr=args.stop_buffer_atr,
            cost_bps_round_trip=args.cost_bps_round_trip,
            max_friction_r=args.max_friction_r,
        )
        args.candidate_cache.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(
            {
                "candidates": candidates,
                "build_stats": build_stats,
                "config": expected_config,
            },
            args.candidate_cache,
        )
        print(f"Wrote {args.candidate_cache}")

    if candidates.empty:
        raise RuntimeError("No filled hot-candle retest candidates were found.")
    candidates = candidates[
        candidates["candle_role"].isin(candle_roles)
        & candidates["retest_mode"].isin(retest_modes)
        & candidates["min_risk_pct"].isin(min_risk_pcts)
    ].copy()
    candidates["decision_time"] = pd.to_datetime(candidates["decision_time"], utc=True)
    for rr in rr_values:
        candidates[f"exit_time_{rr:g}"] = pd.to_datetime(
            candidates[f"exit_time_{rr:g}"],
            utc=True,
        )
        candidates[f"label_end_time_{rr:g}"] = pd.to_datetime(
            candidates[f"label_end_time_{rr:g}"],
            utc=True,
        )
    print(f"Retest order candidates: {len(candidates):,}")

    summaries: list[dict[str, Any]] = []
    monthly_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    for method in selection_methods:
        method_models = models if method == "model" else ["rule"]
        method_feature_sets = feature_sets if method == "model" else ["hotness_score"]
        for model_name in method_models:
            for feature_set in method_feature_sets:
                for candle_role in candle_roles:
                    for mode in retest_modes:
                        for min_risk_pct in min_risk_pcts:
                            for rr in rr_values:
                                for direction_scope in direction_scopes:
                                    label = (
                                        f"selection={method}/model={model_name}/"
                                        f"features={feature_set}/role={candle_role}/"
                                        f"mode={mode}/risk={min_risk_pct:g}/rr={rr:g}/"
                                        f"direction={direction_scope}/"
                                        f"validation={args.validation_months}m"
                                    )
                                    print(f"Running {label}...")
                                    if method == "model":
                                        summary, monthly, trades = run_model_walkforward(
                                            candidates,
                                            candle_role=candle_role,
                                            retest_mode=mode,
                                            min_risk_pct=min_risk_pct,
                                            rr=rr,
                                            direction_scope=direction_scope,
                                            feature_set=feature_set,
                                            model_name=model_name,
                                            validation_months=args.validation_months,
                                            start_month=args.start_month,
                                            end_month=args.end_month,
                                            cascade_thresholds=cascade_thresholds,
                                            score_coverages=hotness_coverages,
                                            min_validation_trades=args.min_validation_trades,
                                            min_positive_month_fraction=args.min_positive_validation_month_fraction,
                                            seed=31,
                                        )
                                    else:
                                        summary, monthly, trades = run_walkforward(
                                            candidates,
                                            candle_role=candle_role,
                                            retest_mode=mode,
                                            min_risk_pct=min_risk_pct,
                                            rr=rr,
                                            direction_scope=direction_scope,
                                            validation_months=args.validation_months,
                                            start_month=args.start_month,
                                            end_month=args.end_month,
                                            cascade_thresholds=cascade_thresholds,
                                            hotness_coverages=hotness_coverages,
                                            min_validation_trades=args.min_validation_trades,
                                            min_positive_month_fraction=args.min_positive_validation_month_fraction,
                                        )
                                    summary["selection_method"] = method
                                    summary["experiment"] = label
                                    summaries.append(summary)
                                    monthly.insert(0, "experiment", label)
                                    trades.insert(0, "experiment", label)
                                    monthly_frames.append(monthly)
                                    trade_frames.append(trades)

    summary_table = pd.DataFrame(summaries).sort_values(
        ["net_r", "profit_factor", "trades"],
        ascending=[False, False, False],
    )
    monthly_table = (
        pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame()
    )
    trades_table = (
        pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    )
    args.summary_csv.write_text(summary_table.to_csv(index=False), encoding="utf-8")
    args.monthly_csv.write_text(monthly_table.to_csv(index=False), encoding="utf-8")
    args.trades_csv.write_text(trades_table.to_csv(index=False), encoding="utf-8")
    result = {
        "config": {
            **expected_config,
            "start_month": args.start_month,
            "end_month": args.end_month,
            "validation_months": args.validation_months,
            "cascade_thresholds": cascade_thresholds,
            "hotness_coverages": hotness_coverages,
            "selection_methods": selection_methods,
            "models": models,
            "feature_sets": feature_sets,
            "direction_scopes": direction_scopes,
        },
        "build_stats": build_stats,
        "candidate_rows": len(candidates),
        "experiments": summary_table.to_dict(orient="records"),
    }
    args.output_json.write_text(
        json.dumps(result, indent=2, default=json_default),
        encoding="utf-8",
    )
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.monthly_csv}")
    print(f"Wrote {args.trades_csv}")
    print("\nTop experiments:")
    print(summary_table.head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
