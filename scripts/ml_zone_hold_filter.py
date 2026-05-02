from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.pipeline import make_pipeline

    SKLEARN_AVAILABLE = True
except ImportError:
    joblib = None
    HistGradientBoostingClassifier = None
    RandomForestClassifier = None
    SimpleImputer = None
    permutation_importance = None
    make_pipeline = None
    SKLEARN_AVAILABLE = False

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backtest_turtle_soup import (
    Config,
    DEFAULT_BFM_ZONE_TF_SETS,
    DEFAULT_BFM_ZONE_TIMEFRAMES,
    add_atr,
    bfm_feature_columns_for_groups,
    bfm_zone_feature_values,
    build_daily_context,
    build_bfm_feature_projection,
    build_htf_bias_events,
    build_htf_zone_events,
    fetch_klines,
    normalize_binance_spot_symbol,
    parse_utc_datetime,
    parse_bfm_feature_tf_sets,
    parse_bfm_feature_timeframes,
    resample_ohlc,
    run_backtest,
    summarize,
)
from scripts.crypto_symbol_sets import SYMBOL_SETS, expand_symbol_args


FEATURE_COLUMNS = [
    "direction_long",
    "zone_age_hours",
    "zone_width_pct",
    "zone_width_atr",
    "penetration_frac",
    "close_distance_pct",
    "reclaim_pos",
    "sweep_range_atr",
    "vol_mult",
    "ret_1h_dir",
    "ret_4h_dir",
    "ret_24h_dir",
    "range_1h_pct",
    "range_4h_pct",
    "bias_4h_aligned",
    "bias_1d_aligned",
    "first4_ret_dir",
    "first4_range_pos",
    "prev_day_ret_dir",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "active_same_dir_zones",
    "active_opp_zones",
    "zone_rank",
    "prior_zone_touches",
    "same_bar_reaction_atr",
    "same_bar_close_reaction_atr",
    "same_bar_adverse_atr",
    "reclaim_body_atr",
    "mbq_zone_health",
    "confluence_count_0_5atr",
    "confluence_same_count_0_5atr",
    "htf_sma50_aligned",
    "last20_known_hold_rate",
]


@dataclass
class LogisticModel:
    columns: list[str]
    median: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float


def symbol_job(params: dict[str, Any]) -> tuple[str, pd.DataFrame, list[Any], str]:
    symbol = params["symbol"]
    cache_path = ensure_cache(symbol, params["interval"], params["start"], params["end"], params["cache_dir"])
    df = pd.read_pickle(cache_path)
    samples = build_zone_hold_samples(
        symbol,
        df,
        zone_tf=params["zone_tf"],
        label_rr=params["label_rr"],
        label_horizon_bars=params["label_horizon_bars"],
        htf_left=params["htf_left"],
        htf_right=params["htf_right"],
        htf_ob_search_bars=params["htf_ob_search_bars"],
        zone_penetration_frac=params["zone_penetration_frac"],
        min_reclaim_pos=params["min_reclaim_pos"],
        mbq_ob_lookback_bars=params["mbq_ob_lookback_bars"],
        mbq_confluence_atr=params["mbq_confluence_atr"],
        max_zone_scan=params["max_zone_scan"],
        use_bfm_features=params["use_bfm_features"],
        bfm_timeframes=params["bfm_timeframes"],
        bfm_tf_sets=params["bfm_tf_sets"],
        bfm_invalidation=params["bfm_invalidation"],
        bfm_max_extension_bars=params["bfm_max_extension_bars"],
    )
    cfg = Config(
        exec_tf=params["interval"],
        structure_tf="15m",
        entry_mode="zone_retest",
        tf1=params["zone_tf"],
        tf2="1d",
        use_tf1=True,
        use_tf2=False,
        block_dead_zone=False,
        htf_left=params["htf_left"],
        htf_right=params["htf_right"],
        htf_ob_search_bars=params["htf_ob_search_bars"],
        max_structure_bars_to_choch=32,
        min_entry_risk_pct=params["strategy_min_entry_risk_pct"],
        max_entry_risk_pct=params["strategy_max_entry_risk_pct"],
        max_zone_scan=params["max_zone_scan"],
        zone_hold_bfm_timeframes=params["bfm_timeframes"],
        zone_hold_bfm_tf_sets=params["bfm_tf_sets"],
        zone_hold_bfm_invalidation=params["bfm_invalidation"],
        zone_hold_bfm_max_extension_bars=params["bfm_max_extension_bars"],
    )
    normalized = normalize_binance_spot_symbol(symbol)
    trades = run_backtest(df, cfg)
    message = f"{normalized}: {len(samples)} zone events, {len(trades)} {params['zone_tf']}-zone strategy trades"
    return normalized, samples, trades, message


def _to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _cache_path(cache_dir: Path, symbol: str, interval: str, start: datetime, end: datetime) -> Path:
    return cache_dir / f"{normalize_binance_spot_symbol(symbol).lower()}_{interval}_{start:%Y%m%d}_{end:%Y%m%d}.pkl"


def ensure_cache(symbol: str, interval: str, start: datetime, end: datetime, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    requested_symbol = normalize_binance_spot_symbol(symbol).lower()

    for candidate in sorted(cache_dir.glob(f"{requested_symbol}_{interval}_*.pkl")):
        try:
            df = pd.read_pickle(candidate)
        except Exception:
            continue
        if df.empty:
            continue
        if df["open_time"].iloc[0].to_pydatetime() <= start and df["close_time"].iloc[-1].to_pydatetime() >= end:
            return candidate

    path = _cache_path(cache_dir, symbol, interval, start, end)
    if path.exists():
        return path

    df = fetch_klines(symbol, interval, _to_ms(start), _to_ms(end))
    df.to_pickle(path)
    return path


def prepare_exec_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("open_time").reset_index(drop=True).copy()
    out = add_atr(out)
    out["vol_sma20"] = out["volume"].rolling(20).mean()
    out["ret_1h"] = out["close"].pct_change(12) * 100.0
    out["ret_4h"] = out["close"].pct_change(48) * 100.0
    out["ret_24h"] = out["close"].pct_change(288) * 100.0
    out["range_1h_pct"] = (out["high"].rolling(12).max() - out["low"].rolling(12).min()) / out["close"] * 100.0
    out["range_4h_pct"] = (out["high"].rolling(48).max() - out["low"].rolling(48).min()) / out["close"] * 100.0
    return out


def zone_key(symbol: str, direction: str, time_value: pd.Timestamp, top: float, bottom: float) -> str:
    return f"{symbol}|{direction}|{pd.Timestamp(time_value).isoformat()}|{top:.8f}|{bottom:.8f}"


def tradingview_high_before_low(open_val: float, high_val: float, low_val: float) -> bool:
    return abs(open_val - high_val) < abs(open_val - low_val)


def label_zone_outcome(
    df: pd.DataFrame,
    start_idx: int,
    direction: str,
    entry_price: float,
    fail_price: float,
    target_rr: float,
    horizon_bars: int,
) -> dict[str, Any] | None:
    risk = abs(entry_price - fail_price)
    if risk <= 0:
        return None

    end_idx = min(len(df), start_idx + horizon_bars)
    if start_idx >= end_idx:
        return None

    sign = 1.0 if direction == "long" else -1.0
    target_price = entry_price + sign * risk * target_rr
    mfe_r = 0.0
    mae_r = 0.0
    last_close_r = 0.0

    highs = df["high"].to_list()
    lows = df["low"].to_list()
    opens = df["open"].to_list()
    closes = df["close"].to_list()

    for j in range(start_idx, end_idx):
        if direction == "long":
            mfe_r = max(mfe_r, (highs[j] - entry_price) / risk)
            mae_r = max(mae_r, (entry_price - lows[j]) / risk)
            target_hit = highs[j] >= target_price
            fail_hit = lows[j] <= fail_price
            if target_hit and fail_hit:
                target_first = tradingview_high_before_low(opens[j], highs[j], lows[j])
                return {
                    "hold_label": 1 if target_first else 0,
                    "outcome": "target_same_bar" if target_first else "fail_same_bar",
                    "future_r": target_rr if target_first else -1.0,
                    "mfe_r": mfe_r,
                    "mae_r": mae_r,
                    "bars_to_outcome": j - start_idx + 1,
                }
            if target_hit:
                return {
                    "hold_label": 1,
                    "outcome": "target",
                    "future_r": target_rr,
                    "mfe_r": mfe_r,
                    "mae_r": mae_r,
                    "bars_to_outcome": j - start_idx + 1,
                }
            if fail_hit:
                return {
                    "hold_label": 0,
                    "outcome": "fail",
                    "future_r": -1.0,
                    "mfe_r": mfe_r,
                    "mae_r": mae_r,
                    "bars_to_outcome": j - start_idx + 1,
                }
            last_close_r = (closes[j] - entry_price) / risk
        else:
            mfe_r = max(mfe_r, (entry_price - lows[j]) / risk)
            mae_r = max(mae_r, (highs[j] - entry_price) / risk)
            target_hit = lows[j] <= target_price
            fail_hit = highs[j] >= fail_price
            if target_hit and fail_hit:
                target_first = not tradingview_high_before_low(opens[j], highs[j], lows[j])
                return {
                    "hold_label": 1 if target_first else 0,
                    "outcome": "target_same_bar" if target_first else "fail_same_bar",
                    "future_r": target_rr if target_first else -1.0,
                    "mfe_r": mfe_r,
                    "mae_r": mae_r,
                    "bars_to_outcome": j - start_idx + 1,
                }
            if target_hit:
                return {
                    "hold_label": 1,
                    "outcome": "target",
                    "future_r": target_rr,
                    "mfe_r": mfe_r,
                    "mae_r": mae_r,
                    "bars_to_outcome": j - start_idx + 1,
                }
            if fail_hit:
                return {
                    "hold_label": 0,
                    "outcome": "fail",
                    "future_r": -1.0,
                    "mfe_r": mfe_r,
                    "mae_r": mae_r,
                    "bars_to_outcome": j - start_idx + 1,
                }
            last_close_r = (entry_price - closes[j]) / risk

    clipped_r = max(-1.0, min(float(target_rr), last_close_r))
    return {
        "hold_label": 1 if clipped_r > 0 else 0,
        "outcome": "timeout",
        "future_r": clipped_r,
        "mfe_r": mfe_r,
        "mae_r": mae_r,
        "bars_to_outcome": end_idx - start_idx,
    }


def current_day_context(day_context: dict[pd.Timestamp, dict], now: pd.Timestamp) -> dict[str, float]:
    day_start = pd.Timestamp(now).floor("D")
    ctx = day_context.get(day_start)
    if ctx is None:
        return {"first4_ret": math.nan, "first4_range_pos": math.nan, "prev_day_ret": math.nan}
    return ctx


def build_htf_sma_bias_events(exec_df: pd.DataFrame, timeframe: str, length: int) -> list[dict[str, Any]]:
    htf = resample_ohlc(exec_df, timeframe)
    htf["sma"] = htf["close"].rolling(length).mean()
    events: list[dict[str, Any]] = []
    for _, row in htf.iterrows():
        bias = 0
        if pd.notna(row["sma"]):
            if row["close"] > row["sma"]:
                bias = 1
            elif row["close"] < row["sma"]:
                bias = -1
        events.append({"time": row["close_time"], "bias": bias})
    return events


def zone_mid(zone: dict[str, Any]) -> float:
    return (float(zone["top"]) + float(zone["bottom"])) / 2.0


def confluence_counts(
    zone: dict[str, Any],
    same_zones: list[dict[str, Any]],
    opp_zones: list[dict[str, Any]],
    threshold: float,
) -> tuple[int, int]:
    mid = zone_mid(zone)
    same_count = sum(
        1
        for other in same_zones
        if other.get("id") != zone.get("id") and abs(mid - zone_mid(other)) <= threshold
    )
    opp_count = sum(1 for other in opp_zones if abs(mid - zone_mid(other)) <= threshold)
    return same_count + opp_count, same_count


def build_zone_hold_samples(
    symbol: str,
    exec_df: pd.DataFrame,
    *,
    zone_tf: str,
    label_rr: float,
    label_horizon_bars: int,
    htf_left: int,
    htf_right: int,
    htf_ob_search_bars: int,
    zone_penetration_frac: float,
    min_reclaim_pos: float,
    mbq_ob_lookback_bars: int,
    mbq_confluence_atr: float,
    max_zone_scan: int,
    use_bfm_features: bool = False,
    bfm_timeframes: str = DEFAULT_BFM_ZONE_TIMEFRAMES,
    bfm_tf_sets: str = DEFAULT_BFM_ZONE_TF_SETS,
    bfm_invalidation: str = "wick",
    bfm_max_extension_bars: int = 300,
) -> pd.DataFrame:
    exec_df = prepare_exec_df(exec_df)
    bfm_projection = None
    if use_bfm_features:
        parsed_bfm_timeframes = parse_bfm_feature_timeframes(bfm_timeframes)
        parsed_bfm_tf_sets = parse_bfm_feature_tf_sets(bfm_tf_sets, parsed_bfm_timeframes)
        bfm_projection = build_bfm_feature_projection(
            exec_df,
            timeframes=parsed_bfm_timeframes,
            tf_sets=parsed_bfm_tf_sets,
            invalidation=bfm_invalidation,
            max_extension_bars=bfm_max_extension_bars,
        )
    supply_events, demand_events = build_htf_zone_events(
        exec_df,
        zone_tf,
        htf_left,
        htf_right,
        0.25,
        htf_ob_search_bars,
        False,
    )
    bias_4h_events = build_htf_bias_events(exec_df, "4h", 20)
    bias_1d_events = build_htf_bias_events(exec_df, "1d", 20)
    htf_sma50_events = build_htf_sma_bias_events(exec_df, zone_tf, 50)
    day_context = build_daily_context(exec_df)

    demand_ptr = 0
    supply_ptr = 0
    bias_4h_ptr = 0
    bias_1d_ptr = 0
    htf_sma50_ptr = 0
    current_bias_4h = 0
    current_bias_1d = 0
    current_htf_sma50_bias = 0
    demand_zones: list[dict[str, Any]] = []
    supply_zones: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    pending_known_outcomes: list[tuple[int, int]] = []
    last20_known_outcomes: list[int] = []
    normalized_symbol = normalize_binance_spot_symbol(symbol)

    opens = exec_df["open"].to_list()
    highs = exec_df["high"].to_list()
    lows = exec_df["low"].to_list()
    closes = exec_df["close"].to_list()
    volumes = exec_df["volume"].to_list()
    atrs = exec_df["atr"].bfill().ffill().to_list()
    vol_sma20 = exec_df["vol_sma20"].bfill().ffill().to_list()
    times = exec_df["open_time"].to_list()
    close_times = exec_df["close_time"].to_list()

    def add_zone(event: dict[str, Any], direction: str, seq: int) -> dict[str, Any]:
        return {
            **event,
            "id": f"{direction}-{seq}-{pd.Timestamp(event['time']).isoformat()}-{event['top']:.8f}-{event['bottom']:.8f}",
            "direction": direction,
            "used": False,
            "touch_count": 0,
        }

    def build_sample(
        zone: dict[str, Any],
        direction: str,
        i: int,
        rank: int,
        active_same: int,
        active_opp: int,
        same_zones: list[dict[str, Any]],
        opp_zones: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        width = float(zone["width"])
        if width <= 0 or atrs[i] <= 0 or closes[i] <= 0:
            return None

        sign = 1.0 if direction == "long" else -1.0
        age_bars = max(0.0, (pd.Timestamp(times[i]) - pd.Timestamp(zone["time"])).total_seconds() / 300.0)
        age_pct = min(100.0, 100.0 * age_bars / max(1.0, float(mbq_ob_lookback_bars)))
        overhit = 20.0 if zone["touch_count"] > 2 else 10.0 if zone["touch_count"] > 1 else 0.0
        health = max(0.0, 100.0 - age_pct - overhit)
        confluence_total, confluence_same = confluence_counts(zone, same_zones, opp_zones, mbq_confluence_atr * atrs[i])

        if direction == "long":
            entry_price = float(zone["top"])
            fail_price = float(zone["bottom"])
            penetration_frac = (float(zone["top"]) - lows[i]) / width
            close_distance_pct = (closes[i] - float(zone["top"])) / closes[i] * 100.0
            reclaim_pos = (closes[i] - lows[i]) / (highs[i] - lows[i]) if highs[i] > lows[i] else 0.0
            same_bar_reaction_atr = max(0.0, (highs[i] - entry_price) / atrs[i])
            same_bar_close_reaction_atr = (closes[i] - entry_price) / atrs[i]
            same_bar_adverse_atr = max(0.0, (entry_price - lows[i]) / atrs[i])
        else:
            entry_price = float(zone["bottom"])
            fail_price = float(zone["top"])
            penetration_frac = (highs[i] - float(zone["bottom"])) / width
            close_distance_pct = (float(zone["bottom"]) - closes[i]) / closes[i] * 100.0
            reclaim_pos = (highs[i] - closes[i]) / (highs[i] - lows[i]) if highs[i] > lows[i] else 0.0
            same_bar_reaction_atr = max(0.0, (entry_price - lows[i]) / atrs[i])
            same_bar_close_reaction_atr = (entry_price - closes[i]) / atrs[i]
            same_bar_adverse_atr = max(0.0, (highs[i] - entry_price) / atrs[i])

        outcome = label_zone_outcome(exec_df, i + 1, direction, entry_price, fail_price, label_rr, label_horizon_bars)
        if outcome is None:
            return None

        now = pd.Timestamp(times[i])
        ctx = current_day_context(day_context, now)
        hour = now.hour + now.minute / 60.0
        dow = now.dayofweek
        vol_mult = volumes[i] / vol_sma20[i] if vol_sma20[i] > 0 else math.nan

        row = {
            "symbol": normalized_symbol,
            "time": now,
            "close_time": pd.Timestamp(close_times[i]),
            "direction": direction,
            "zone_time": pd.Timestamp(zone["time"]),
            "zone_top": float(zone["top"]),
            "zone_bottom": float(zone["bottom"]),
            "entry_price": entry_price,
            "fail_price": fail_price,
            "label_rr": label_rr,
            "label_horizon_bars": label_horizon_bars,
            "event_key": zone_key(normalized_symbol, direction, now, float(zone["top"]), float(zone["bottom"])),
            "direction_long": 1.0 if direction == "long" else 0.0,
            "zone_age_hours": (now - pd.Timestamp(zone["time"])).total_seconds() / 3600.0,
            "zone_width_pct": width / closes[i] * 100.0,
            "zone_width_atr": width / atrs[i],
            "penetration_frac": penetration_frac,
            "close_distance_pct": close_distance_pct,
            "reclaim_pos": reclaim_pos,
            "sweep_range_atr": (highs[i] - lows[i]) / atrs[i],
            "vol_mult": vol_mult,
            "ret_1h_dir": sign * float(exec_df.iloc[i]["ret_1h"]) if pd.notna(exec_df.iloc[i]["ret_1h"]) else math.nan,
            "ret_4h_dir": sign * float(exec_df.iloc[i]["ret_4h"]) if pd.notna(exec_df.iloc[i]["ret_4h"]) else math.nan,
            "ret_24h_dir": sign * float(exec_df.iloc[i]["ret_24h"]) if pd.notna(exec_df.iloc[i]["ret_24h"]) else math.nan,
            "range_1h_pct": float(exec_df.iloc[i]["range_1h_pct"]) if pd.notna(exec_df.iloc[i]["range_1h_pct"]) else math.nan,
            "range_4h_pct": float(exec_df.iloc[i]["range_4h_pct"]) if pd.notna(exec_df.iloc[i]["range_4h_pct"]) else math.nan,
            "bias_4h_aligned": sign * current_bias_4h,
            "bias_1d_aligned": sign * current_bias_1d,
            "first4_ret_dir": sign * float(ctx["first4_ret"]) if pd.notna(ctx["first4_ret"]) else math.nan,
            "first4_range_pos": float(ctx["first4_range_pos"]) if pd.notna(ctx["first4_range_pos"]) else math.nan,
            "prev_day_ret_dir": sign * float(ctx["prev_day_ret"]) if pd.notna(ctx["prev_day_ret"]) else math.nan,
            "hour_sin": math.sin(2.0 * math.pi * hour / 24.0),
            "hour_cos": math.cos(2.0 * math.pi * hour / 24.0),
            "dow_sin": math.sin(2.0 * math.pi * dow / 7.0),
            "dow_cos": math.cos(2.0 * math.pi * dow / 7.0),
            "active_same_dir_zones": float(active_same),
            "active_opp_zones": float(active_opp),
            "zone_rank": float(rank),
            "prior_zone_touches": float(zone["touch_count"]),
            "same_bar_reaction_atr": same_bar_reaction_atr,
            "same_bar_close_reaction_atr": same_bar_close_reaction_atr,
            "same_bar_adverse_atr": same_bar_adverse_atr,
            "reclaim_body_atr": sign * (closes[i] - opens[i]) / atrs[i],
            "mbq_zone_health": health,
            "confluence_count_0_5atr": float(confluence_total),
            "confluence_same_count_0_5atr": float(confluence_same),
            "htf_sma50_aligned": sign * current_htf_sma50_bias,
            "last20_known_hold_rate": float(np.mean(last20_known_outcomes)) if last20_known_outcomes else math.nan,
        }
        if bfm_projection is not None:
            row.update(
                bfm_zone_feature_values(
                    projection=bfm_projection,
                    direction=direction,
                    zone=zone,
                    index=i,
                    atr=atrs[i],
                    close=closes[i],
                    high=highs[i],
                    low=lows[i],
                )
            )
        row.update(outcome)
        return row

    for i in range(len(exec_df)):
        visible_time = pd.Timestamp(close_times[i])

        if pending_known_outcomes:
            still_pending: list[tuple[int, int]] = []
            for outcome_idx, label in pending_known_outcomes:
                if outcome_idx <= i:
                    last20_known_outcomes.append(label)
                    if len(last20_known_outcomes) > 20:
                        last20_known_outcomes.pop(0)
                else:
                    still_pending.append((outcome_idx, label))
            pending_known_outcomes = still_pending

        while demand_ptr < len(demand_events) and demand_events[demand_ptr]["time"] <= visible_time:
            demand_zones.append(add_zone(demand_events[demand_ptr], "long", demand_ptr))
            demand_ptr += 1
        while supply_ptr < len(supply_events) and supply_events[supply_ptr]["time"] <= visible_time:
            supply_zones.append(add_zone(supply_events[supply_ptr], "short", supply_ptr))
            supply_ptr += 1
        while bias_4h_ptr < len(bias_4h_events) and bias_4h_events[bias_4h_ptr]["time"] <= visible_time:
            current_bias_4h = bias_4h_events[bias_4h_ptr]["bias"]
            bias_4h_ptr += 1
        while bias_1d_ptr < len(bias_1d_events) and bias_1d_events[bias_1d_ptr]["time"] <= visible_time:
            current_bias_1d = bias_1d_events[bias_1d_ptr]["bias"]
            bias_1d_ptr += 1
        while htf_sma50_ptr < len(htf_sma50_events) and htf_sma50_events[htf_sma50_ptr]["time"] <= visible_time:
            current_htf_sma50_bias = htf_sma50_events[htf_sma50_ptr]["bias"]
            htf_sma50_ptr += 1

        demand_zones = [zone for zone in demand_zones if not zone["used"] and lows[i] >= zone["bottom"]]
        supply_zones = [zone for zone in supply_zones if not zone["used"] and highs[i] <= zone["top"]]

        demand_candidates = [zone for zone in reversed(demand_zones) if not zone["used"]]
        supply_candidates = [zone for zone in reversed(supply_zones) if not zone["used"]]
        if max_zone_scan > 0:
            demand_candidates = demand_candidates[:max_zone_scan]
            supply_candidates = supply_candidates[:max_zone_scan]
        active_demand_count = len(demand_candidates)
        active_supply_count = len(supply_candidates)

        for rank, zone in enumerate(demand_candidates):
            width = float(zone["width"])
            sweep_range = highs[i] - lows[i]
            reclaim_pos = (closes[i] - lows[i]) / sweep_range if sweep_range > 0 else 0.0
            penetration_limit = float(zone["bottom"]) - width * zone_penetration_frac
            touched = lows[i] <= float(zone["top"]) and lows[i] >= penetration_limit
            if touched and closes[i] > float(zone["top"]) and reclaim_pos >= min_reclaim_pos:
                row = build_sample(zone, "long", i, rank, active_demand_count, active_supply_count, demand_zones, supply_zones)
                if row is not None:
                    samples.append(row)
                    pending_known_outcomes.append((i + int(row["bars_to_outcome"]), int(row["hold_label"])))
                zone["used"] = True
                break
            if touched:
                zone["touch_count"] += 1

        for rank, zone in enumerate(supply_candidates):
            width = float(zone["width"])
            sweep_range = highs[i] - lows[i]
            reclaim_pos = (highs[i] - closes[i]) / sweep_range if sweep_range > 0 else 0.0
            penetration_limit = float(zone["top"]) + width * zone_penetration_frac
            touched = highs[i] >= float(zone["bottom"]) and highs[i] <= penetration_limit
            if touched and closes[i] < float(zone["bottom"]) and reclaim_pos >= min_reclaim_pos:
                row = build_sample(zone, "short", i, rank, active_supply_count, active_demand_count, supply_zones, demand_zones)
                if row is not None:
                    samples.append(row)
                    pending_known_outcomes.append((i + int(row["bars_to_outcome"]), int(row["hold_label"])))
                zone["used"] = True
                break
            if touched:
                zone["touch_count"] += 1

    return pd.DataFrame(samples)


def fit_logistic_regression(train: pd.DataFrame, columns: list[str], *, l2: float, learning_rate: float, epochs: int) -> LogisticModel:
    x = train[columns].astype(float).to_numpy()
    y = train["hold_label"].astype(float).to_numpy()
    median = np.nanmedian(x, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    x = np.where(np.isnan(x), median, x)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    xs = (x - mean) / scale

    coef = np.zeros(xs.shape[1], dtype=float)
    intercept = 0.0
    pos = max(float(y.sum()), 1.0)
    neg = max(float(len(y) - y.sum()), 1.0)
    weights = np.where(y > 0.5, len(y) / (2.0 * pos), len(y) / (2.0 * neg))
    weight_sum = weights.sum()

    for _ in range(epochs):
        logits = np.clip(intercept + xs @ coef, -40.0, 40.0)
        pred = 1.0 / (1.0 + np.exp(-logits))
        error = (pred - y) * weights
        grad_intercept = error.sum() / weight_sum
        grad_coef = xs.T @ error / weight_sum + l2 * coef / len(y)
        intercept -= learning_rate * grad_intercept
        coef -= learning_rate * grad_coef

    return LogisticModel(columns=columns, median=median, mean=mean, scale=scale, coef=coef, intercept=float(intercept))


def predict_proba(model: LogisticModel, frame: pd.DataFrame) -> np.ndarray:
    x = frame[model.columns].astype(float).to_numpy()
    x = np.where(np.isnan(x), model.median, x)
    xs = (x - model.mean) / model.scale
    logits = np.clip(model.intercept + xs @ model.coef, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-logits))


def fit_sklearn_model(train: pd.DataFrame, columns: list[str], model_name: str) -> Any:
    x = train[columns].astype(float)
    y = train["hold_label"].astype(int)
    if model_name == "sklearn_rf":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=500,
                max_depth=5,
                min_samples_leaf=8,
                class_weight="balanced_subsample",
                random_state=7,
                n_jobs=1,
            ),
        ).fit(x, y)

    return HistGradientBoostingClassifier(
        max_iter=350,
        learning_rate=0.035,
        max_leaf_nodes=8,
        min_samples_leaf=10,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=7,
    ).fit(x, y)


def sklearn_feature_rank(estimator: Any, frame: pd.DataFrame, columns: list[str], model_name: str) -> pd.DataFrame:
    if frame.empty or frame["hold_label"].nunique() < 2:
        return pd.DataFrame(columns=["feature", "importance"])

    if model_name == "sklearn_rf":
        forest = estimator.named_steps["randomforestclassifier"]
        return pd.DataFrame({"feature": columns, "importance": forest.feature_importances_}).sort_values("importance", ascending=False)

    result = permutation_importance(
        estimator,
        frame[columns].astype(float),
        frame["hold_label"].astype(int),
        n_repeats=8,
        random_state=7,
        scoring="roc_auc",
    )
    return pd.DataFrame({"feature": columns, "importance": result.importances_mean}).sort_values("importance", ascending=False)


def auc_score(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = y_true.astype(int)
    pos = int(y_true.sum())
    neg = int(len(y_true) - pos)
    if pos == 0 or neg == 0:
        return math.nan
    ranks = pd.Series(score).rank(method="average").to_numpy()
    return float((ranks[y_true == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def classifier_metrics(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {"rows": 0, "hold_rate": 0.0, "auc": math.nan, "brier": math.nan, "log_loss": math.nan}
    y = frame["hold_label"].astype(float).to_numpy()
    p = np.clip(frame["hold_prob"].astype(float).to_numpy(), 1e-6, 1.0 - 1e-6)
    return {
        "rows": int(len(frame)),
        "hold_rate": round(float(y.mean()) * 100.0, 2),
        "auc": round(auc_score(y, p), 3),
        "brier": round(float(np.mean((p - y) ** 2)), 4),
        "log_loss": round(float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))), 4),
    }


def threshold_table(frame: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        kept = frame[frame["hold_prob"] >= threshold]
        if kept.empty:
            rows.append({
                "threshold": threshold,
                "kept": 0,
                "kept_pct": 0.0,
                "hold_rate": 0.0,
                "avg_future_r": 0.0,
                "net_future_r": 0.0,
            })
            continue
        rows.append({
            "threshold": threshold,
            "kept": len(kept),
            "kept_pct": round(100.0 * len(kept) / len(frame), 2) if len(frame) else 0.0,
            "hold_rate": round(100.0 * kept["hold_label"].mean(), 2),
            "avg_future_r": round(float(kept["future_r"].mean()), 3),
            "net_future_r": round(float(kept["future_r"].sum()), 3),
        })
    return pd.DataFrame(rows)


def max_drawdown_r(trades: list[Any]) -> float:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for trade in sorted(trades, key=lambda item: item.exit_time):
        equity += trade.r_multiple
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)
    return round(max_dd, 3)


def trade_window_metrics(trades: list[Any], start: datetime, end: datetime) -> dict[str, Any]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    selected = [trade for trade in trades if start_ts <= trade.entry_time < end_ts]
    out = summarize(selected)
    out["max_dd_r"] = max_drawdown_r(selected)
    return out


def filtered_trade_table(
    trades: list[Any],
    prob_lookup: dict[str, float],
    symbol: str,
    start: datetime,
    end: datetime,
    thresholds: list[float],
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    normalized_symbol = normalize_binance_spot_symbol(symbol)
    window_trades = [trade for trade in trades if start_ts <= trade.entry_time < end_ts]
    rows = []
    for threshold in thresholds:
        kept = []
        missing = 0
        for trade in window_trades:
            key = zone_key(normalized_symbol, trade.direction, trade.sweep_time, trade.zone_top, trade.zone_bottom)
            prob = prob_lookup.get(key)
            if prob is None:
                missing += 1
                continue
            if prob >= threshold:
                kept.append(trade)
        metrics = summarize(kept)
        rows.append({
            "threshold": threshold,
            "trades": metrics["trades"],
            "win_rate": metrics["win_rate"],
            "profit_factor": metrics["profit_factor"],
            "net_r": metrics["net_r"],
            "avg_r": metrics["avg_r"],
            "max_dd_r": max_drawdown_r(kept),
            "missing_probs": missing,
        })
    return pd.DataFrame(rows)


def model_to_json(model: LogisticModel) -> dict[str, Any]:
    return {
        "columns": model.columns,
        "median": model.median.tolist(),
        "mean": model.mean.tolist(),
        "scale": model.scale.tolist(),
        "coef": model.coef.tolist(),
        "intercept": model.intercept,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight SMC zone-hold probability filter.")
    parser.add_argument("--symbols", nargs="+", default=[])
    parser.add_argument("--symbol-set", choices=["none", *SYMBOL_SETS.keys()], default="core3")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--zone-tf", default="4h")
    parser.add_argument("--start", default="2022-04-20")
    parser.add_argument("--split", default="2025-04-20")
    parser.add_argument("--end", default="2026-04-20")
    parser.add_argument("--cache-dir", type=Path, default=Path("scripts/.cache"))
    parser.add_argument("--dataset-out", type=Path, default=Path("scripts/zone_hold_dataset.csv"))
    parser.add_argument("--model-out", type=Path, default=Path("scripts/zone_hold_model.joblib"))
    parser.add_argument("--model", choices=["sklearn_hgb", "sklearn_rf", "logistic"], default="sklearn_rf")
    parser.add_argument("--label-rr", type=float, default=1.0)
    parser.add_argument("--label-horizon-bars", type=int, default=288)
    parser.add_argument("--htf-left", type=int, default=5)
    parser.add_argument("--htf-right", type=int, default=5)
    parser.add_argument("--htf-ob-search-bars", type=int, default=50)
    parser.add_argument("--zone-penetration-frac", type=float, default=0.50)
    parser.add_argument("--min-reclaim-pos", type=float, default=0.70)
    parser.add_argument("--mbq-ob-lookback-bars", type=int, default=200)
    parser.add_argument("--mbq-confluence-atr", type=float, default=0.50)
    parser.add_argument("--max-zone-scan", type=int, default=250, help="Limit newest active zones scanned per side per bar; 0 means unlimited.")
    parser.add_argument("--use-bfm-features", action="store_true", help="Add causal BFM Magic Trendline confluence features to the zone-hold model.")
    parser.add_argument("--bfm-feature-groups", default="line,channel", help="Comma-separated BFM feature groups: line, channel, or all.")
    parser.add_argument("--bfm-timeframes", default=DEFAULT_BFM_ZONE_TIMEFRAMES)
    parser.add_argument("--bfm-tf-sets", default=DEFAULT_BFM_ZONE_TF_SETS)
    parser.add_argument("--bfm-invalidation", choices=["wick", "close", "none"], default="wick")
    parser.add_argument("--bfm-max-extension-bars", type=int, default=300)
    parser.add_argument("--strategy-min-entry-risk-pct", type=float, default=0.0)
    parser.add_argument("--strategy-max-entry-risk-pct", type=float, default=math.inf)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--l2", type=float, default=1.0)
    args = parser.parse_args()

    args.symbols = expand_symbol_args(args.symbols, args.symbol_set)
    feature_columns = FEATURE_COLUMNS + bfm_feature_columns_for_groups(args.bfm_feature_groups) if args.use_bfm_features else FEATURE_COLUMNS

    start = parse_utc_datetime(args.start)
    split = parse_utc_datetime(args.split)
    end = parse_utc_datetime(args.end)
    thresholds = [0.45, 0.50, 0.55, 0.60, 0.65]

    all_samples = []
    all_trades: dict[str, list[Any]] = {}
    job_params = [
        {
            "symbol": symbol,
            "interval": args.interval,
            "start": start,
            "end": end,
            "cache_dir": args.cache_dir,
            "zone_tf": args.zone_tf,
            "label_rr": args.label_rr,
            "label_horizon_bars": args.label_horizon_bars,
            "htf_left": args.htf_left,
            "htf_right": args.htf_right,
            "htf_ob_search_bars": args.htf_ob_search_bars,
            "zone_penetration_frac": args.zone_penetration_frac,
            "min_reclaim_pos": args.min_reclaim_pos,
            "mbq_ob_lookback_bars": args.mbq_ob_lookback_bars,
            "mbq_confluence_atr": args.mbq_confluence_atr,
            "max_zone_scan": args.max_zone_scan,
            "use_bfm_features": args.use_bfm_features,
            "bfm_timeframes": args.bfm_timeframes,
            "bfm_tf_sets": args.bfm_tf_sets,
            "bfm_invalidation": args.bfm_invalidation,
            "bfm_max_extension_bars": args.bfm_max_extension_bars,
            "strategy_min_entry_risk_pct": args.strategy_min_entry_risk_pct,
            "strategy_max_entry_risk_pct": args.strategy_max_entry_risk_pct,
        }
        for symbol in args.symbols
    ]
    if args.workers <= 1:
        for params in job_params:
            normalized, samples, trades, message = symbol_job(params)
            all_samples.append(samples)
            all_trades[normalized] = trades
            print(message, flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(symbol_job, params): params["symbol"] for params in job_params}
            for future in as_completed(futures):
                normalized, samples, trades, message = future.result()
                all_samples.append(samples)
                all_trades[normalized] = trades
                print(message, flush=True)

    dataset = pd.concat(all_samples, ignore_index=True) if all_samples else pd.DataFrame()
    if dataset.empty:
        raise RuntimeError("No zone-hold samples were generated.")

    dataset = dataset.sort_values(["time", "symbol"]).reset_index(drop=True)
    train = dataset[dataset["time"] < pd.Timestamp(split)].copy()
    oos = dataset[(dataset["time"] >= pd.Timestamp(split)) & (dataset["time"] < pd.Timestamp(end))].copy()
    if train["hold_label"].nunique() < 2:
        raise RuntimeError("Training set only has one label class; adjust the label horizon or symbols.")

    use_sklearn = args.model != "logistic" and SKLEARN_AVAILABLE
    if args.model != "logistic" and not SKLEARN_AVAILABLE:
        print("sklearn is not available in this Python, falling back to the built-in logistic model.")

    if use_sklearn:
        model = fit_sklearn_model(train, feature_columns, args.model)
        dataset["hold_prob"] = model.predict_proba(dataset[feature_columns].astype(float))[:, 1]
    else:
        model = fit_logistic_regression(train, feature_columns, l2=args.l2, learning_rate=args.learning_rate, epochs=args.epochs)
        dataset["hold_prob"] = predict_proba(model, dataset)

    train = dataset[dataset["time"] < pd.Timestamp(split)].copy()
    oos = dataset[(dataset["time"] >= pd.Timestamp(split)) & (dataset["time"] < pd.Timestamp(end))].copy()

    args.dataset_out.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.dataset_out, index=False)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    model_path = args.model_out
    if use_sklearn:
        payload = {
            "model": model,
            "feature_columns": feature_columns,
            "model_kind": args.model,
            "zone_tf": args.zone_tf,
        }
        if args.use_bfm_features:
            payload["bfm_feature_config"] = {
                "timeframes": args.bfm_timeframes,
                "tf_sets": args.bfm_tf_sets,
                "feature_groups": args.bfm_feature_groups,
                "invalidation": args.bfm_invalidation,
                "max_extension_bars": int(args.bfm_max_extension_bars),
            }
        joblib.dump(payload, model_path)
    else:
        model_path = args.model_out if args.model_out.suffix.lower() == ".json" else args.model_out.with_suffix(".json")
        model_path.write_text(json.dumps(model_to_json(model), indent=2), encoding="utf-8")

    print()
    print(f"Dataset saved to {args.dataset_out}")
    print(f"Model saved to {model_path}")
    print()
    print("Classifier metrics:")
    metrics = pd.DataFrame([
        {"window": "train", **classifier_metrics(train)},
        {"window": "oos", **classifier_metrics(oos)},
    ])
    print(metrics.to_string(index=False))
    print()
    print("OOS zone-event threshold table:")
    print(threshold_table(oos, thresholds).to_string(index=False))

    print()
    if use_sklearn:
        rank = sklearn_feature_rank(model, oos if len(oos) >= 20 else train, feature_columns, args.model)
        print("Largest sklearn feature importances:")
        print(rank.head(12).to_string(index=False))
    else:
        coef = pd.DataFrame({"feature": feature_columns, "coef": model.coef})
        coef["abs_coef"] = coef["coef"].abs()
        print("Largest standardized logistic coefficients:")
        print(coef.sort_values("abs_coef", ascending=False).head(12)[["feature", "coef"]].to_string(index=False))

    prob_lookup = dict(zip(dataset["event_key"], dataset["hold_prob"]))
    print()
    print(f"{args.zone_tf}-zone strategy baseline and ML-filtered OOS trades:")
    baseline_rows = []
    filtered_rows = []
    for symbol, trades in all_trades.items():
        base_train = trade_window_metrics(trades, start, split)
        base_oos = trade_window_metrics(trades, split, end)
        baseline_rows.append({"symbol": symbol, "window": "train", **base_train})
        baseline_rows.append({"symbol": symbol, "window": "oos", **base_oos})
        table = filtered_trade_table(trades, prob_lookup, symbol, split, end, thresholds)
        table.insert(0, "symbol", symbol)
        filtered_rows.append(table)

    baseline = pd.DataFrame(baseline_rows)
    print(baseline.to_string(index=False))
    print()
    filtered = pd.concat(filtered_rows, ignore_index=True)
    print(filtered.to_string(index=False))


if __name__ == "__main__":
    main()
