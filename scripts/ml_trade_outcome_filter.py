from __future__ import annotations

import argparse
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    joblib = None
    HistGradientBoostingClassifier = None
    RandomForestClassifier = None
    SimpleImputer = None
    permutation_importance = None
    LogisticRegression = None
    brier_score_loss = None
    log_loss = None
    roc_auc_score = None
    make_pipeline = None
    StandardScaler = None
    SKLEARN_AVAILABLE = False

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backtest_turtle_soup import (
    BFM_CHANNEL_FEATURE_COLUMNS,
    BFM_LINE_FEATURE_COLUMNS,
    Config,
    DEFAULT_BFM_ZONE_TF_SETS,
    DEFAULT_BFM_ZONE_TIMEFRAMES,
    INTERVAL_MS,
    add_atr,
    bfm_zone_feature_values,
    build_daily_context,
    build_bfm_feature_projection,
    build_htf_bias_events,
    build_htf_sma_bias_events,
    fetch_klines,
    normalize_binance_spot_symbol,
    normalize_timeframe,
    parse_bfm_feature_tf_sets,
    parse_bfm_feature_timeframes,
    parse_bfm_feature_groups,
    parse_utc_datetime,
    parse_timeframe_list,
    resample_ohlc,
    run_backtest,
    summarize,
)
from scripts.crypto_symbol_sets import SYMBOL_SETS, expand_symbol_args


BASE_FEATURE_COLUMNS = [
    "direction_long",
    "entry_risk_pct",
    "entry_risk_atr",
    "risk_to_zone_width",
    "risk_to_ob_width",
    "target_distance_atr",
    "zone_width_pct_signal",
    "zone_width_atr_signal",
    "zone_age_hours_sweep",
    "zone_age_hours_signal",
    "sweep_penetration_frac",
    "sweep_reclaim_pos",
    "sweep_range_atr",
    "sweep_same_bar_reaction_atr",
    "sweep_same_bar_close_reaction_atr",
    "sweep_same_bar_adverse_atr",
    "sweep_reclaim_body_atr",
    "sweep_vol_mult",
    "choch_wait_bars",
    "signal_after_choch_bars",
    "ob_width_atr_signal",
    "entry_to_ob_mid_atr",
    "entry_to_zone_mid_atr",
    "stop_beyond_zone_atr",
    "entry_vs_signal_close_atr",
    "signal_close_vs_zone_atr",
    "zone_source_sfp",
    "zone_source_ob_break",
    "liquidity_level_width_atr",
    "liquidity_level_age_hours",
    "liquidity_level_age_bars",
    "liquidity_sfp_strict",
    "liquidity_sfp_confirm3",
    "liquidity_sweep_open_reclaimed",
    "liquidity_sweep_close_reclaimed",
    "liquidity_sweep_extreme_is_lookback_extreme",
    "liquidity_sweep_depth_atr",
    "liquidity_level_to_signal_close_atr",
    "liquidity_prev_day_same_gap_atr",
    "liquidity_prev_day_opp_dist_atr",
    "liquidity_prev_day_swept",
    "liquidity_prev_week_same_gap_atr",
    "liquidity_prev_week_opp_dist_atr",
    "liquidity_prev_week_swept",
    "liquidity_prev_month_same_gap_atr",
    "liquidity_prev_month_opp_dist_atr",
    "liquidity_prev_month_swept",
    "ret_1h_dir",
    "ret_4h_dir",
    "ret_24h_dir",
    "range_1h_pct",
    "range_4h_pct",
    "bias_4h_aligned",
    "bias_1d_aligned",
    "htf_sma50_aligned",
    "first4_ret_dir",
    "first4_range_pos",
    "prev_day_ret_dir",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "zone_hold_prob",
    "zone_prob_ge_045",
    "zone_prob_ge_050",
    "zone_prob_ge_055",
    "zone_prob_ge_060",
    "zone_prob_mid_045_050",
    "zone_prob_mid_050_055",
    "first4_range_low_017",
    "first4_ret_neg_108",
    "first4_ret_pos_098",
    "late_hour_cos_low_094",
    "early_week_dow_cos_high_062",
    "sweep_adverse_tiny_0107",
    "sweep_reclaim_near_full_0994",
    "sweep_reclaim_full_100",
    "ret1h_pullback_deep_106",
    "entry_ob_deep_219",
    "stop_beyond_deep_941",
    "range1h_high_120",
    "close_reaction_low_009",
    "rescue_first4_range_low",
    "rescue_late_hour",
    "rescue_sweep_adverse_tiny",
    "rescue_sweep_reclaim_full",
    "rescue_ret1h_pullback",
    "rescue_entry_ob_deep",
    "rescue_stop_beyond_deep",
    "rescue_first4_ret_neg",
    "rescue_first4_ret_pos",
    "rescue_range1h_high",
    "zone_prob_x_first4_range_low",
    "zone_prob_x_sweep_reclaim_full",
    "zone_prob_x_sweep_adverse_tiny",
    "zone_prob_x_ret1h_pullback",
    "tex_rsi22_signal",
    "tex_rsi22_sweep",
    "tex_pressure_signal",
    "tex_pressure_sweep",
    "tex_aligned_strength_recent",
    "tex_aligned_age_bars",
    "tex_counter_strength_recent",
    "tex_counter_age_bars",
    "tex_aligned_signal_bar",
    "tex_counter_signal_bar",
    "tex_aligned_between_sweep_signal",
    "tex_counter_between_sweep_signal",
]

RSI_EXHAUST_FEATURE_COLUMNS = [
    "tex_rsi22_signal",
    "tex_rsi22_sweep",
    "tex_pressure_signal",
    "tex_pressure_sweep",
    "tex_aligned_strength_recent",
    "tex_aligned_age_bars",
    "tex_counter_strength_recent",
    "tex_counter_age_bars",
    "tex_aligned_signal_bar",
    "tex_counter_signal_bar",
    "tex_aligned_between_sweep_signal",
    "tex_counter_between_sweep_signal",
]


def trade_bfm_feature_columns_for_groups(raw: str | None) -> list[str]:
    out: list[str] = []
    for group in parse_bfm_feature_groups(raw):
        columns = BFM_LINE_FEATURE_COLUMNS if group == "line" else BFM_CHANNEL_FEATURE_COLUMNS
        out.extend(f"bfm_sweep_{column[4:]}" for column in columns)
        out.extend(f"bfm_signal_{column[4:]}" for column in columns)
    return out


def symbol_feature_name(symbol: str) -> str:
    normalized = normalize_binance_spot_symbol(symbol).lower()
    return "symbol_" + "".join(ch if ch.isalnum() else "_" for ch in normalized)


def add_symbol_dummy_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = frame.copy()
    symbols = sorted(out["symbol"].dropna().unique())
    columns: list[str] = []
    for symbol in symbols:
        column = symbol_feature_name(str(symbol))
        out[column] = (out["symbol"] == symbol).astype(float)
        columns.append(column)
    return out, columns


def symbol_job(params: dict[str, Any]) -> tuple[str, pd.DataFrame, list[Any], str]:
    symbol = params["symbol"]
    cache_path = ensure_cache(symbol, params["interval"], params["warmup_start"], params["end"], params["cache_dir"])
    df = pd.read_pickle(cache_path)
    cfg = Config(
        exec_tf=params["interval"],
        structure_tf="15m",
        entry_mode="zone_retest",
        tf1=params["tf1"],
        tf2=params["tf2"],
        use_tf1=True,
        use_tf2=params["use_tf2"],
        block_dead_zone=params["dead_zone"],
        max_structure_bars_to_choch=32,
        min_entry_risk_pct=params["min_entry_risk_pct"],
        max_zone_scan=params["max_zone_scan"],
        use_sfp_liquidity_zones=params["use_sfp_liquidity_zones"],
        sfp_timeframes=params["sfp_timeframes"],
        sfp_left=params["sfp_left"],
        sfp_right=params["sfp_right"],
        sfp_level_width_atr=params["sfp_level_width_atr"],
        sfp_strict=params["sfp_strict"],
        sfp_require_open_reclaim=params["sfp_require_open_reclaim"],
    )
    feature_frame, trades = trade_feature_rows(
        symbol,
        df,
        cfg,
        use_bfm_features=params["use_bfm_features"],
        bfm_timeframes=params["bfm_timeframes"],
        bfm_tf_sets=params["bfm_tf_sets"],
        bfm_invalidation=params["bfm_invalidation"],
        bfm_max_extension_bars=params["bfm_max_extension_bars"],
        use_sfp_liquidity_zones=params["use_sfp_liquidity_zones"],
        sfp_timeframes=params["sfp_timeframes"],
        sfp_left=params["sfp_left"],
        sfp_right=params["sfp_right"],
        sfp_level_width_atr=params["sfp_level_width_atr"],
        sfp_strict=params["sfp_strict"],
        sfp_require_open_reclaim=params["sfp_require_open_reclaim"],
    )
    normalized = normalize_binance_spot_symbol(symbol)
    return normalized, feature_frame, trades, f"{normalized}: {len(feature_frame)} trade rows"


def _to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


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

    path = cache_dir / f"{requested_symbol}_{interval}_{start:%Y%m%d}_{end:%Y%m%d}.pkl"
    if path.exists():
        return path

    df = fetch_klines(symbol, interval, _to_ms(start), _to_ms(end))
    df.to_pickle(path)
    return path


def prepare_feature_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("open_time").reset_index(drop=True).copy()
    out = add_atr(out)
    out["vol_sma20"] = out["volume"].rolling(20).mean()
    out["ret_1h"] = out["close"].pct_change(12) * 100.0
    out["ret_4h"] = out["close"].pct_change(48) * 100.0
    out["ret_24h"] = out["close"].pct_change(288) * 100.0
    out["range_1h_pct"] = (out["high"].rolling(12).max() - out["low"].rolling(12).min()) / out["close"] * 100.0
    out["range_4h_pct"] = (out["high"].rolling(48).max() - out["low"].rolling(48).min()) / out["close"] * 100.0
    return out


def rsi_wilder(close: pd.Series, length: int = 22) -> pd.Series:
    delta = close.astype(float).diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_up = up.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    avg_down = down.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    rs = avg_up / avg_down.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.mask((avg_down == 0.0) & (avg_up > 0.0), 100.0)
    rsi = rsi.mask((avg_up == 0.0) & (avg_down > 0.0), 0.0)
    return rsi


def compute_trend_exhaustion_frame(df: pd.DataFrame, *, rsi_len: int = 22, valid_bars: int = 4) -> pd.DataFrame:
    out = df.sort_values("open_time").reset_index(drop=True).copy()
    out["rsi22"] = rsi_wilder(out["close"], rsi_len)
    rsi = out["rsi22"]
    prev = rsi.shift(1)

    long_strength = np.zeros(len(out), dtype=int)
    short_strength = np.zeros(len(out), dtype=int)
    for level, strength in [(30.0, 1), (20.0, 2), (15.0, 3)]:
        mask = ((prev <= level) & (rsi > level)).fillna(False).to_numpy()
        long_strength[mask] = np.maximum(long_strength[mask], strength)
    for level, strength in [(70.0, 1), (80.0, 2), (85.0, 3)]:
        mask = ((prev >= level) & (rsi < level)).fillna(False).to_numpy()
        short_strength[mask] = np.maximum(short_strength[mask], strength)

    out["tex_long_strength"] = long_strength
    out["tex_short_strength"] = short_strength

    for side in ["long", "short"]:
        strengths = out[f"tex_{side}_strength"].astype(int).to_numpy()
        recent_strength = np.zeros(len(out), dtype=float)
        recent_age = np.full(len(out), math.nan, dtype=float)
        last_strength = 0
        last_index: int | None = None
        for idx, strength in enumerate(strengths):
            if strength > 0:
                last_strength = int(strength)
                last_index = idx
            if last_index is not None:
                age = idx - last_index
                if age <= valid_bars:
                    recent_strength[idx] = float(last_strength)
                    recent_age[idx] = float(age)
        out[f"tex_{side}_recent_strength"] = recent_strength
        out[f"tex_{side}_recent_age"] = recent_age
    return out


def trend_exhaustion_trade_features(
    tex: pd.DataFrame,
    *,
    direction: str,
    signal_idx: int,
    sweep_idx: int,
) -> dict[str, float]:
    if signal_idx < 0 or signal_idx >= len(tex):
        return {column: math.nan for column in RSI_EXHAUST_FEATURE_COLUMNS}
    if sweep_idx < 0 or sweep_idx >= len(tex):
        sweep_idx = signal_idx

    aligned_side = "long" if direction == "long" else "short"
    counter_side = "short" if direction == "long" else "long"
    signal_rsi = float(tex.at[signal_idx, "rsi22"])
    sweep_rsi = float(tex.at[sweep_idx, "rsi22"])
    if direction == "long":
        signal_pressure = (50.0 - signal_rsi) / 50.0 if math.isfinite(signal_rsi) else math.nan
        sweep_pressure = (50.0 - sweep_rsi) / 50.0 if math.isfinite(sweep_rsi) else math.nan
    else:
        signal_pressure = (signal_rsi - 50.0) / 50.0 if math.isfinite(signal_rsi) else math.nan
        sweep_pressure = (sweep_rsi - 50.0) / 50.0 if math.isfinite(sweep_rsi) else math.nan

    lo = min(sweep_idx, signal_idx)
    hi = max(sweep_idx, signal_idx)
    return {
        "tex_rsi22_signal": signal_rsi / 100.0 if math.isfinite(signal_rsi) else math.nan,
        "tex_rsi22_sweep": sweep_rsi / 100.0 if math.isfinite(sweep_rsi) else math.nan,
        "tex_pressure_signal": signal_pressure,
        "tex_pressure_sweep": sweep_pressure,
        "tex_aligned_strength_recent": float(tex.at[signal_idx, f"tex_{aligned_side}_recent_strength"]),
        "tex_aligned_age_bars": float(tex.at[signal_idx, f"tex_{aligned_side}_recent_age"]),
        "tex_counter_strength_recent": float(tex.at[signal_idx, f"tex_{counter_side}_recent_strength"]),
        "tex_counter_age_bars": float(tex.at[signal_idx, f"tex_{counter_side}_recent_age"]),
        "tex_aligned_signal_bar": float(tex.at[signal_idx, f"tex_{aligned_side}_strength"]),
        "tex_counter_signal_bar": float(tex.at[signal_idx, f"tex_{counter_side}_strength"]),
        "tex_aligned_between_sweep_signal": float(tex.loc[lo:hi, f"tex_{aligned_side}_strength"].max()),
        "tex_counter_between_sweep_signal": float(tex.loc[lo:hi, f"tex_{counter_side}_strength"].max()),
    }


def event_series(events: list[dict[str, Any]], close_times: list[pd.Timestamp]) -> list[int]:
    values: list[int] = []
    ptr = 0
    current = 0
    for close_time in close_times:
        while ptr < len(events) and events[ptr]["time"] <= close_time:
            current = int(events[ptr]["bias"])
            ptr += 1
        values.append(current)
    return values


def current_day_context(day_context: dict[pd.Timestamp, dict], now: pd.Timestamp) -> dict[str, float]:
    ctx = day_context.get(pd.Timestamp(now).floor("D"))
    if ctx is None:
        return {"first4_ret": math.nan, "first4_range_pos": math.nan, "prev_day_ret": math.nan}
    return ctx


def profit_factor(rs: pd.Series) -> float:
    wins = rs[rs > 0]
    losses = rs[rs <= 0]
    gross_loss = abs(float(losses.sum()))
    if gross_loss == 0:
        return float("inf") if len(wins) else 0.0
    return float(wins.sum()) / gross_loss


def max_drawdown_r(rs: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in rs:
        equity += r
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)
    return round(max_dd, 3)


def frame_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "net_r": 0.0, "avg_r": 0.0, "max_dd_r": 0.0}
    rs = frame.sort_values("exit_time")["r_multiple"].astype(float)
    return {
        "trades": int(len(frame)),
        "win_rate": round(100.0 * float((rs > 0).mean()), 2),
        "profit_factor": round(profit_factor(rs), 3),
        "net_r": round(float(rs.sum()), 3),
        "avg_r": round(float(rs.mean()), 3),
        "max_dd_r": max_drawdown_r(rs.to_list()),
    }


def safe_div(num: float, den: float) -> float:
    return num / den if den and math.isfinite(den) else math.nan


def previous_completed_levels(df: pd.DataFrame, timeframe: str) -> tuple[list[float], list[float]]:
    htf = resample_ohlc(df, timeframe)
    if htf.empty:
        return [math.nan] * len(df), [math.nan] * len(df)
    htf_close = pd.to_datetime(htf["close_time"], utc=True, errors="coerce").to_list()
    exec_close = pd.to_datetime(df["close_time"], utc=True, errors="coerce").to_list()
    highs = htf["high"].astype(float).to_list()
    lows = htf["low"].astype(float).to_list()
    out_high: list[float] = []
    out_low: list[float] = []
    ptr = -1
    for close_time in exec_close:
        while ptr + 1 < len(htf_close) and htf_close[ptr + 1] <= close_time:
            ptr += 1
        if ptr >= 0:
            out_high.append(float(highs[ptr]))
            out_low.append(float(lows[ptr]))
        else:
            out_high.append(math.nan)
            out_low.append(math.nan)
    return out_high, out_low


def previous_calendar_month_levels(df: pd.DataFrame) -> tuple[list[float], list[float]]:
    monthly = (
        df.set_index("open_time")
        .resample("MS", label="left", closed="left")
        .agg({"high": "max", "low": "min"})
        .dropna()
        .reset_index()
    )
    if monthly.empty:
        return [math.nan] * len(df), [math.nan] * len(df)
    monthly["next_open_time"] = monthly["open_time"].shift(-1)
    monthly["close_time"] = monthly["next_open_time"].fillna(monthly["open_time"] + pd.offsets.MonthBegin(1)) - pd.Timedelta(milliseconds=1)
    month_close = pd.to_datetime(monthly["close_time"], utc=True, errors="coerce").to_list()
    exec_close = pd.to_datetime(df["close_time"], utc=True, errors="coerce").to_list()
    highs = monthly["high"].astype(float).to_list()
    lows = monthly["low"].astype(float).to_list()
    out_high: list[float] = []
    out_low: list[float] = []
    ptr = -1
    for close_time in exec_close:
        while ptr + 1 < len(month_close) and month_close[ptr + 1] <= close_time:
            ptr += 1
        if ptr >= 0:
            out_high.append(float(highs[ptr]))
            out_low.append(float(lows[ptr]))
        else:
            out_high.append(math.nan)
            out_low.append(math.nan)
    return out_high, out_low


def swept_prev_level(direction: str, high: float, low: float, close: float, level: float) -> float:
    if not math.isfinite(level):
        return 0.0
    if direction == "long":
        return 1.0 if low < level and close > level else 0.0
    return 1.0 if high > level and close < level else 0.0


def bool_float(mask: pd.Series) -> pd.Series:
    return mask.fillna(False).astype(float)


def add_engineered_rescue_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    z = out["zone_hold_prob"].fillna(0.5)

    out["zone_prob_ge_045"] = bool_float(z >= 0.45)
    out["zone_prob_ge_050"] = bool_float(z >= 0.50)
    out["zone_prob_ge_055"] = bool_float(z >= 0.55)
    out["zone_prob_ge_060"] = bool_float(z >= 0.60)
    out["zone_prob_mid_045_050"] = bool_float((z >= 0.45) & (z < 0.50))
    out["zone_prob_mid_050_055"] = bool_float((z >= 0.50) & (z < 0.55))

    out["first4_range_low_017"] = bool_float(out["first4_range_pos"] <= 0.17118)
    out["first4_ret_neg_108"] = bool_float(out["first4_ret_dir"] <= -1.08814)
    out["first4_ret_pos_098"] = bool_float(out["first4_ret_dir"] >= 0.98018)
    out["late_hour_cos_low_094"] = bool_float(out["hour_cos"] <= -0.94693)
    out["early_week_dow_cos_high_062"] = bool_float(out["dow_cos"] >= 0.62349)
    out["sweep_adverse_tiny_0107"] = bool_float(out["sweep_same_bar_adverse_atr"] <= 0.10749)
    out["sweep_reclaim_near_full_0994"] = bool_float(out["sweep_reclaim_pos"] >= 0.99395)
    out["sweep_reclaim_full_100"] = bool_float(out["sweep_reclaim_pos"] >= 1.0)
    out["ret1h_pullback_deep_106"] = bool_float(out["ret_1h_dir"] <= -1.05996)
    out["entry_ob_deep_219"] = bool_float(out["entry_to_ob_mid_atr"] <= -2.19333)
    out["stop_beyond_deep_941"] = bool_float(out["stop_beyond_zone_atr"] <= -9.41274)
    out["range1h_high_120"] = bool_float(out["range_1h_pct"] >= 1.20047)
    out["close_reaction_low_009"] = bool_float(out["sweep_same_bar_close_reaction_atr"] <= 0.08994)

    low_zone = z < 0.55
    out["rescue_first4_range_low"] = bool_float(low_zone & (out["first4_range_low_017"] > 0))
    out["rescue_late_hour"] = bool_float(low_zone & (out["late_hour_cos_low_094"] > 0))
    out["rescue_sweep_adverse_tiny"] = bool_float(low_zone & (out["sweep_adverse_tiny_0107"] > 0))
    out["rescue_sweep_reclaim_full"] = bool_float(low_zone & (out["sweep_reclaim_near_full_0994"] > 0))
    out["rescue_ret1h_pullback"] = bool_float(low_zone & (out["ret1h_pullback_deep_106"] > 0))
    out["rescue_entry_ob_deep"] = bool_float(low_zone & (out["entry_ob_deep_219"] > 0))
    out["rescue_stop_beyond_deep"] = bool_float(low_zone & (out["stop_beyond_deep_941"] > 0))
    out["rescue_first4_ret_neg"] = bool_float(low_zone & (out["first4_ret_neg_108"] > 0))
    out["rescue_first4_ret_pos"] = bool_float(low_zone & (out["first4_ret_pos_098"] > 0))
    out["rescue_range1h_high"] = bool_float(low_zone & (out["range1h_high_120"] > 0))

    out["zone_prob_x_first4_range_low"] = z * out["first4_range_low_017"]
    out["zone_prob_x_sweep_reclaim_full"] = z * out["sweep_reclaim_near_full_0994"]
    out["zone_prob_x_sweep_adverse_tiny"] = z * out["sweep_adverse_tiny_0107"]
    out["zone_prob_x_ret1h_pullback"] = z * out["ret1h_pullback_deep_106"]
    return out


def zone_key(symbol: str, direction: str, time_value: pd.Timestamp, top: float, bottom: float) -> str:
    return f"{symbol}|{direction}|{pd.Timestamp(time_value).isoformat()}|{top:.8f}|{bottom:.8f}"


def prefixed_bfm_values(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key[4:]}": value for key, value in values.items() if key.startswith("bfm_")}


def trade_feature_rows(
    symbol: str,
    df: pd.DataFrame,
    cfg: Config,
    *,
    use_bfm_features: bool = False,
    bfm_timeframes: str = DEFAULT_BFM_ZONE_TIMEFRAMES,
    bfm_tf_sets: str = DEFAULT_BFM_ZONE_TF_SETS,
    bfm_invalidation: str = "wick",
    bfm_max_extension_bars: int = 300,
    use_sfp_liquidity_zones: bool = False,
    sfp_timeframes: str = "15m,1h,4h",
    sfp_left: int = 15,
    sfp_right: int = 10,
    sfp_level_width_atr: float = 0.15,
    sfp_strict: bool = True,
    sfp_require_open_reclaim: bool = True,
    precomputed_trades: list[Any] | None = None,
    extra_trades: list[Any] | None = None,
) -> tuple[pd.DataFrame, list[Any]]:
    prepared = prepare_feature_df(df)
    if use_sfp_liquidity_zones and not cfg.use_sfp_liquidity_zones:
        cfg = Config(
            **{
                **cfg.__dict__,
                "use_sfp_liquidity_zones": True,
                "sfp_timeframes": ",".join(parse_timeframe_list(sfp_timeframes)),
                "sfp_left": int(sfp_left),
                "sfp_right": int(sfp_right),
                "sfp_level_width_atr": float(sfp_level_width_atr),
                "sfp_strict": bool(sfp_strict),
                "sfp_require_open_reclaim": bool(sfp_require_open_reclaim),
            }
        )
    bfm_projection = None
    if use_bfm_features:
        parsed_bfm_timeframes = parse_bfm_feature_timeframes(bfm_timeframes)
        parsed_bfm_tf_sets = parse_bfm_feature_tf_sets(bfm_tf_sets, parsed_bfm_timeframes)
        bfm_projection = build_bfm_feature_projection(
            prepared,
            timeframes=parsed_bfm_timeframes,
            tf_sets=parsed_bfm_tf_sets,
            invalidation=bfm_invalidation,
            max_extension_bars=bfm_max_extension_bars,
        )
    trades = list(precomputed_trades) if precomputed_trades is not None else run_backtest(df, cfg)
    if extra_trades:
        trades = list(trades) + list(extra_trades)
    normalized = normalize_binance_spot_symbol(symbol)

    opens = prepared["open"].to_list()
    highs = prepared["high"].to_list()
    lows = prepared["low"].to_list()
    closes = prepared["close"].to_list()
    volumes = prepared["volume"].to_list()
    atrs = prepared["atr"].bfill().ffill().to_list()
    vol_sma20 = prepared["vol_sma20"].bfill().ffill().to_list()
    close_times = prepared["close_time"].to_list()
    day_context = build_daily_context(prepared)
    bias_4h = event_series(build_htf_bias_events(prepared, "4h", 20), close_times)
    bias_1d = event_series(build_htf_bias_events(prepared, "1d", 20), close_times)
    sma50_4h = event_series(build_htf_sma_bias_events(prepared, "4h", 50), close_times)
    prev_day_high, prev_day_low = previous_completed_levels(prepared, "1d")
    prev_week_high, prev_week_low = previous_completed_levels(prepared, "1w")
    prev_month_high, prev_month_low = previous_calendar_month_levels(prepared)
    trend_exhaustion = compute_trend_exhaustion_frame(prepared, valid_bars=4)

    rows: list[dict[str, Any]] = []
    for trade in trades:
        signal_idx = int(trade.signal_index)
        sweep_idx = int(trade.sweep_index)
        choch_idx = int(trade.choch_index)
        entry_idx = int(trade.entry_index)
        if signal_idx >= len(prepared) or sweep_idx >= len(prepared) or entry_idx >= len(prepared):
            continue
        atr_signal = atrs[signal_idx]
        atr_sweep = atrs[sweep_idx]
        if atr_signal <= 0 or atr_sweep <= 0:
            continue

        sign = 1.0 if trade.direction == "long" else -1.0
        zone_width = float(trade.zone_top - trade.zone_bottom)
        ob_width = float(trade.ob_top - trade.ob_bottom)
        risk = abs(float(trade.entry_price - trade.stop_price))
        target_distance = abs(float(trade.target_price - trade.entry_price))
        zone_mid = (trade.zone_top + trade.zone_bottom) / 2.0
        ob_mid = (trade.ob_top + trade.ob_bottom) / 2.0
        sweep_range = highs[sweep_idx] - lows[sweep_idx]
        liquidity_level = float(getattr(trade, "liquidity_level", math.nan))
        if not math.isfinite(liquidity_level):
            liquidity_level = float(trade.zone_top if trade.direction == "long" else trade.zone_bottom)
        liquidity_width = float(trade.zone_top - trade.zone_bottom)
        liquidity_confirm_time = getattr(trade, "liquidity_confirm_time", None)
        liquidity_confirm = pd.Timestamp(liquidity_confirm_time) if liquidity_confirm_time is not None else pd.Timestamp(trade.sweep_time)
        liquidity_age_hours = (pd.Timestamp(trade.signal_time) - liquidity_confirm).total_seconds() / 3600.0
        exec_tf_minutes = INTERVAL_MS[normalize_timeframe(trade.exec_tf)] / 60_000.0
        liquidity_age_bars = liquidity_age_hours * 60.0 / exec_tf_minutes if exec_tf_minutes > 0 else math.nan

        if trade.direction == "long":
            sweep_penetration = safe_div(trade.zone_top - lows[sweep_idx], zone_width)
            sweep_reclaim = safe_div(closes[sweep_idx] - lows[sweep_idx], sweep_range)
            sweep_reaction = max(0.0, (highs[sweep_idx] - trade.zone_top) / atr_sweep)
            sweep_close_reaction = (closes[sweep_idx] - trade.zone_top) / atr_sweep
            sweep_adverse = max(0.0, (trade.zone_top - lows[sweep_idx]) / atr_sweep)
            stop_beyond_zone = (trade.zone_bottom - trade.stop_price) / atr_signal
            signal_close_vs_zone = (closes[signal_idx] - trade.zone_top) / atr_signal
            liquidity_sweep_depth_atr = max(0.0, (liquidity_level - lows[sweep_idx]) / atr_sweep)
            liquidity_level_to_signal_close_atr = (closes[signal_idx] - liquidity_level) / atr_signal
            liquidity_open_reclaimed = 1.0 if opens[sweep_idx] > liquidity_level else 0.0
            liquidity_close_reclaimed = 1.0 if closes[sweep_idx] > liquidity_level else 0.0
            look_start = max(0, sweep_idx - int(getattr(cfg, "sfp_left", 15)) + 1)
            liquidity_extreme = 1.0 if lows[sweep_idx] <= min(lows[look_start:sweep_idx + 1]) + 1e-12 else 0.0
            confirm_end = min(len(closes), sweep_idx + 4)
            sfp_confirm3 = 1.0 if confirm_end >= sweep_idx + 4 and min(closes[sweep_idx + 1:confirm_end]) > liquidity_level else 0.0
        else:
            sweep_penetration = safe_div(highs[sweep_idx] - trade.zone_bottom, zone_width)
            sweep_reclaim = safe_div(highs[sweep_idx] - closes[sweep_idx], sweep_range)
            sweep_reaction = max(0.0, (trade.zone_bottom - lows[sweep_idx]) / atr_sweep)
            sweep_close_reaction = (trade.zone_bottom - closes[sweep_idx]) / atr_sweep
            sweep_adverse = max(0.0, (highs[sweep_idx] - trade.zone_bottom) / atr_sweep)
            stop_beyond_zone = (trade.stop_price - trade.zone_top) / atr_signal
            signal_close_vs_zone = (trade.zone_bottom - closes[signal_idx]) / atr_signal
            liquidity_sweep_depth_atr = max(0.0, (highs[sweep_idx] - liquidity_level) / atr_sweep)
            liquidity_level_to_signal_close_atr = (liquidity_level - closes[signal_idx]) / atr_signal
            liquidity_open_reclaimed = 1.0 if opens[sweep_idx] < liquidity_level else 0.0
            liquidity_close_reclaimed = 1.0 if closes[sweep_idx] < liquidity_level else 0.0
            look_start = max(0, sweep_idx - int(getattr(cfg, "sfp_left", 15)) + 1)
            liquidity_extreme = 1.0 if highs[sweep_idx] >= max(highs[look_start:sweep_idx + 1]) - 1e-12 else 0.0
            confirm_end = min(len(closes), sweep_idx + 4)
            sfp_confirm3 = 1.0 if confirm_end >= sweep_idx + 4 and max(closes[sweep_idx + 1:confirm_end]) < liquidity_level else 0.0

        same_day_level = prev_day_low[sweep_idx] if trade.direction == "long" else prev_day_high[sweep_idx]
        opp_day_level = prev_day_high[sweep_idx] if trade.direction == "long" else prev_day_low[sweep_idx]
        same_week_level = prev_week_low[sweep_idx] if trade.direction == "long" else prev_week_high[sweep_idx]
        opp_week_level = prev_week_high[sweep_idx] if trade.direction == "long" else prev_week_low[sweep_idx]
        same_month_level = prev_month_low[sweep_idx] if trade.direction == "long" else prev_month_high[sweep_idx]
        opp_month_level = prev_month_high[sweep_idx] if trade.direction == "long" else prev_month_low[sweep_idx]

        def same_gap(level: float) -> float:
            return abs(liquidity_level - level) / atr_sweep if math.isfinite(level) and atr_sweep > 0 else math.nan

        def opp_dist(level: float) -> float:
            if not math.isfinite(level) or atr_sweep <= 0:
                return math.nan
            distance = level - liquidity_level if trade.direction == "long" else liquidity_level - level
            return distance / atr_sweep if distance > 0 else 0.0

        now = pd.Timestamp(prepared.iloc[signal_idx]["open_time"])
        ctx = current_day_context(day_context, now)
        hour = now.hour + now.minute / 60.0
        dow = now.dayofweek

        def signed_col(column: str) -> float:
            value = prepared.iloc[signal_idx][column]
            return sign * float(value) if pd.notna(value) else math.nan

        row = {
            "symbol": normalized,
            "entry_time": trade.entry_time,
            "exit_time": trade.exit_time,
            "direction": trade.direction,
            "r_multiple": trade.r_multiple,
            "win_label": 1 if trade.r_multiple > 0 else 0,
            "entry_price": trade.entry_price,
            "stop_price": trade.stop_price,
            "target_price": trade.target_price,
            "zone_top": trade.zone_top,
            "zone_bottom": trade.zone_bottom,
            "sweep_time": trade.sweep_time,
            "choch_time": trade.choch_time,
            "signal_time": trade.signal_time,
            "exit_reason": trade.exit_reason,
            "event_key": zone_key(normalized, trade.direction, trade.sweep_time, trade.zone_top, trade.zone_bottom),
            "direction_long": 1.0 if trade.direction == "long" else 0.0,
            "entry_risk_pct": risk / trade.entry_price * 100.0,
            "entry_risk_atr": risk / atr_signal,
            "risk_to_zone_width": safe_div(risk, zone_width),
            "risk_to_ob_width": safe_div(risk, ob_width),
            "target_distance_atr": target_distance / atr_signal,
            "zone_width_pct_signal": zone_width / closes[signal_idx] * 100.0,
            "zone_width_atr_signal": zone_width / atr_signal,
            "zone_age_hours_sweep": (trade.sweep_time - pd.Timestamp(prepared.iloc[sweep_idx]["close_time"]).floor("D")).total_seconds() / 3600.0,
            "zone_age_hours_signal": (trade.signal_time - trade.sweep_time).total_seconds() / 3600.0,
            "sweep_penetration_frac": sweep_penetration,
            "sweep_reclaim_pos": sweep_reclaim,
            "sweep_range_atr": sweep_range / atr_sweep,
            "sweep_same_bar_reaction_atr": sweep_reaction,
            "sweep_same_bar_close_reaction_atr": sweep_close_reaction,
            "sweep_same_bar_adverse_atr": sweep_adverse,
            "sweep_reclaim_body_atr": sign * (closes[sweep_idx] - opens[sweep_idx]) / atr_sweep,
            "sweep_vol_mult": volumes[sweep_idx] / vol_sma20[sweep_idx] if vol_sma20[sweep_idx] > 0 else math.nan,
            "choch_wait_bars": float(choch_idx - sweep_idx),
            "signal_after_choch_bars": float(signal_idx - choch_idx),
            "entry_after_signal_bars": float(entry_idx - signal_idx),
            "choch_to_entry_bars": float(entry_idx - choch_idx),
            "ob_width_atr_signal": ob_width / atr_signal,
            "entry_to_ob_mid_atr": sign * (trade.entry_price - ob_mid) / atr_signal,
            "entry_to_zone_mid_atr": sign * (trade.entry_price - zone_mid) / atr_signal,
            "stop_beyond_zone_atr": stop_beyond_zone,
            "entry_vs_signal_close_atr": sign * (trade.entry_price - closes[signal_idx]) / atr_signal,
            "signal_close_vs_zone_atr": signal_close_vs_zone,
            "zone_source_sfp": 1.0 if getattr(trade, "zone_source", "ob_break") == "sfp_pivot" else 0.0,
            "zone_source_ob_break": 1.0 if getattr(trade, "zone_source", "ob_break") != "sfp_pivot" else 0.0,
            "liquidity_level_width_atr": liquidity_width / atr_signal,
            "liquidity_level_age_hours": liquidity_age_hours,
            "liquidity_level_age_bars": liquidity_age_bars,
            "liquidity_sfp_strict": 1.0 if bool(getattr(trade, "liquidity_sfp_strict", False)) else 0.0,
            "liquidity_sfp_confirm3": sfp_confirm3,
            "liquidity_sweep_open_reclaimed": liquidity_open_reclaimed,
            "liquidity_sweep_close_reclaimed": liquidity_close_reclaimed,
            "liquidity_sweep_extreme_is_lookback_extreme": liquidity_extreme,
            "liquidity_sweep_depth_atr": liquidity_sweep_depth_atr,
            "liquidity_level_to_signal_close_atr": liquidity_level_to_signal_close_atr,
            "liquidity_prev_day_same_gap_atr": same_gap(same_day_level),
            "liquidity_prev_day_opp_dist_atr": opp_dist(opp_day_level),
            "liquidity_prev_day_swept": swept_prev_level(trade.direction, highs[sweep_idx], lows[sweep_idx], closes[sweep_idx], same_day_level),
            "liquidity_prev_week_same_gap_atr": same_gap(same_week_level),
            "liquidity_prev_week_opp_dist_atr": opp_dist(opp_week_level),
            "liquidity_prev_week_swept": swept_prev_level(trade.direction, highs[sweep_idx], lows[sweep_idx], closes[sweep_idx], same_week_level),
            "liquidity_prev_month_same_gap_atr": same_gap(same_month_level),
            "liquidity_prev_month_opp_dist_atr": opp_dist(opp_month_level),
            "liquidity_prev_month_swept": swept_prev_level(trade.direction, highs[sweep_idx], lows[sweep_idx], closes[sweep_idx], same_month_level),
            "ret_1h_dir": signed_col("ret_1h"),
            "ret_4h_dir": signed_col("ret_4h"),
            "ret_24h_dir": signed_col("ret_24h"),
            "range_1h_pct": float(prepared.iloc[signal_idx]["range_1h_pct"]) if pd.notna(prepared.iloc[signal_idx]["range_1h_pct"]) else math.nan,
            "range_4h_pct": float(prepared.iloc[signal_idx]["range_4h_pct"]) if pd.notna(prepared.iloc[signal_idx]["range_4h_pct"]) else math.nan,
            "bias_4h_aligned": sign * bias_4h[signal_idx],
            "bias_1d_aligned": sign * bias_1d[signal_idx],
            "htf_sma50_aligned": sign * sma50_4h[signal_idx],
            "first4_ret_dir": sign * float(ctx["first4_ret"]) if pd.notna(ctx["first4_ret"]) else math.nan,
            "first4_range_pos": float(ctx["first4_range_pos"]) if pd.notna(ctx["first4_range_pos"]) else math.nan,
            "prev_day_ret_dir": sign * float(ctx["prev_day_ret"]) if pd.notna(ctx["prev_day_ret"]) else math.nan,
            "hour_sin": math.sin(2.0 * math.pi * hour / 24.0),
            "hour_cos": math.cos(2.0 * math.pi * hour / 24.0),
            "dow_sin": math.sin(2.0 * math.pi * dow / 7.0),
            "dow_cos": math.cos(2.0 * math.pi * dow / 7.0),
            "zone_hold_prob": 0.5,
        }
        row.update(
            trend_exhaustion_trade_features(
                trend_exhaustion,
                direction=trade.direction,
                signal_idx=signal_idx,
                sweep_idx=sweep_idx,
            )
        )
        if bfm_projection is not None:
            zone = {
                "top": trade.zone_top,
                "bottom": trade.zone_bottom,
            }
            row.update(
                prefixed_bfm_values(
                    "bfm_sweep",
                    bfm_zone_feature_values(
                        projection=bfm_projection,
                        direction=trade.direction,
                        zone=zone,
                        index=sweep_idx,
                        atr=atr_sweep,
                        close=closes[sweep_idx],
                        high=highs[sweep_idx],
                        low=lows[sweep_idx],
                    ),
                )
            )
            row.update(
                prefixed_bfm_values(
                    "bfm_signal",
                    bfm_zone_feature_values(
                        projection=bfm_projection,
                        direction=trade.direction,
                        zone=zone,
                        index=signal_idx,
                        atr=atr_signal,
                        close=closes[signal_idx],
                        high=highs[signal_idx],
                        low=lows[signal_idx],
                    ),
                )
            )
        rows.append(row)

    return pd.DataFrame(rows), trades


def fit_model(train: pd.DataFrame, model_name: str, feature_columns: list[str]) -> Any:
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn is required. Run with .venv\\Scripts\\python.exe.")
    x = train[feature_columns].astype(float)
    y = train["win_label"].astype(int)

    if model_name == "logreg":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2500, class_weight="balanced", C=0.05),
        ).fit(x, y)
    if model_name == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.03,
            max_leaf_nodes=4,
            min_samples_leaf=12,
            l2_regularization=8.0,
            class_weight="balanced",
            random_state=11,
        ).fit(x, y)

    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=500,
            max_depth=4,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            random_state=11,
            n_jobs=1,
        ),
    ).fit(x, y)


def classifier_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty or frame["win_label"].nunique() < 2:
        return {"rows": len(frame), "win_rate": round(100.0 * frame["win_label"].mean(), 2) if len(frame) else 0.0, "auc": math.nan, "brier": math.nan, "log_loss": math.nan}
    y = frame["win_label"].astype(int)
    p = frame["trade_win_prob"].astype(float).clip(1e-6, 1.0 - 1e-6)
    return {
        "rows": int(len(frame)),
        "win_rate": round(100.0 * float(y.mean()), 2),
        "auc": round(float(roc_auc_score(y, p)), 3),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "log_loss": round(float(log_loss(y, p)), 4),
    }


def threshold_table(frame: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        kept = frame[frame["trade_win_prob"] >= threshold].copy()
        metrics = frame_metrics(kept)
        rows.append({"threshold": threshold, "kept_pct": round(100.0 * len(kept) / len(frame), 2) if len(frame) else 0.0, **metrics})
    return pd.DataFrame(rows)


def feature_rank(model: Any, frame: pd.DataFrame, model_name: str, feature_columns: list[str]) -> pd.DataFrame:
    if frame.empty or frame["win_label"].nunique() < 2:
        return pd.DataFrame(columns=["feature", "importance"])
    if model_name == "rf":
        forest = model.named_steps["randomforestclassifier"]
        importances = forest.feature_importances_
        names = list(feature_columns)
        if len(importances) != len(names):
            try:
                names = list(model.named_steps["simpleimputer"].get_feature_names_out(feature_columns))
            except Exception:
                names = names[: len(importances)]
        return pd.DataFrame({"feature": names, "importance": importances}).sort_values("importance", ascending=False)
    result = permutation_importance(
        model,
        frame[feature_columns].astype(float),
        frame["win_label"].astype(int),
        n_repeats=8,
        random_state=11,
        scoring="roc_auc",
    )
    return pd.DataFrame({"feature": feature_columns, "importance": result.importances_mean}).sort_values("importance", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an ML filter on actual turtle-soup trade outcomes.")
    parser.add_argument("--symbols", nargs="+", default=[])
    parser.add_argument("--symbol-set", choices=["none", *SYMBOL_SETS.keys()], default="core3")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--warmup-start", default="2021-09-01")
    parser.add_argument("--train-start", default="2022-04-20")
    parser.add_argument("--split", default="2025-04-20")
    parser.add_argument("--end", default="2026-04-20")
    parser.add_argument("--cache-dir", type=Path, default=Path("scripts/.cache"))
    parser.add_argument("--dataset-out", type=Path, default=Path("scripts/trade_outcome_dataset.csv"))
    parser.add_argument("--model-out", type=Path, default=Path("scripts/trade_outcome_model.joblib"))
    parser.add_argument("--model", choices=["rf", "logreg", "hgb"], default="rf")
    parser.add_argument("--zone-hold-dataset", type=Path, help="Optional zone_hold_dataset_mbq.csv to add zone-hold probability as a stage-one feature.")
    parser.add_argument("--zone-hold-pre-filter", type=float, default=0.0, help="Train/evaluate only trades whose stage-one zone-hold probability is at least this value.")
    parser.add_argument("--min-entry-risk-pct", type=float, default=0.0)
    parser.add_argument("--tf1", default="4h")
    parser.add_argument("--tf2", default="1d")
    parser.add_argument("--use-tf2", action="store_true")
    parser.add_argument("--dead-zone", action="store_true")
    parser.add_argument("--max-zone-scan", type=int, default=0)
    parser.add_argument("--use-bfm-features", action="store_true", help="Add BFM trendline confluence at sweep and signal time.")
    parser.add_argument("--bfm-feature-groups", default="line,channel", help="Comma-separated BFM feature groups: line, channel, or all.")
    parser.add_argument("--bfm-timeframes", default=DEFAULT_BFM_ZONE_TIMEFRAMES)
    parser.add_argument("--bfm-tf-sets", default=DEFAULT_BFM_ZONE_TF_SETS)
    parser.add_argument("--bfm-invalidation", choices=["wick", "close", "none"], default="wick")
    parser.add_argument("--bfm-max-extension-bars", type=int, default=300)
    parser.add_argument("--use-sfp-liquidity-triggers", action="store_true", help="Add strict confirmed-pivot SFP liquidity levels to the Turtle Soup candidate universe.")
    parser.add_argument("--sfp-timeframes", default="15m,1h,4h")
    parser.add_argument("--sfp-left", type=int, default=15)
    parser.add_argument("--sfp-right", type=int, default=10)
    parser.add_argument("--sfp-level-width-atr", type=float, default=0.15)
    parser.add_argument("--no-sfp-strict", action="store_true", help="Use pivot liquidity zones with the normal Turtle sweep test instead of the stricter SFP test.")
    parser.add_argument("--no-sfp-open-reclaim", action="store_true", help="Do not require the sweep candle open to remain beyond the swept pivot level.")
    parser.add_argument("--no-symbol-dummies", action="store_true", help="Disable per-symbol dummy features.")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    args.symbols = expand_symbol_args(args.symbols, args.symbol_set)

    warmup_start = parse_utc_datetime(args.warmup_start)
    train_start = parse_utc_datetime(args.train_start)
    split = parse_utc_datetime(args.split)
    end = parse_utc_datetime(args.end)

    frames = []
    baseline_rows = []
    job_params = [
        {
            "symbol": symbol,
            "interval": args.interval,
            "warmup_start": warmup_start,
            "end": end,
            "cache_dir": args.cache_dir,
            "tf1": args.tf1,
            "tf2": args.tf2,
            "use_tf2": args.use_tf2,
            "dead_zone": args.dead_zone,
            "min_entry_risk_pct": args.min_entry_risk_pct,
            "max_zone_scan": args.max_zone_scan,
            "use_bfm_features": args.use_bfm_features,
            "bfm_feature_groups": args.bfm_feature_groups,
            "bfm_timeframes": args.bfm_timeframes,
            "bfm_tf_sets": args.bfm_tf_sets,
            "bfm_invalidation": args.bfm_invalidation,
            "bfm_max_extension_bars": args.bfm_max_extension_bars,
            "use_sfp_liquidity_zones": args.use_sfp_liquidity_triggers,
            "sfp_timeframes": args.sfp_timeframes,
            "sfp_left": args.sfp_left,
            "sfp_right": args.sfp_right,
            "sfp_level_width_atr": args.sfp_level_width_atr,
            "sfp_strict": not args.no_sfp_strict,
            "sfp_require_open_reclaim": not args.no_sfp_open_reclaim,
        }
        for symbol in args.symbols
    ]
    symbol_results: list[tuple[str, pd.DataFrame, list[Any], str]] = []
    if args.workers <= 1:
        for params in job_params:
            symbol_results.append(symbol_job(params))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(symbol_job, params): params["symbol"] for params in job_params}
            for future in as_completed(futures):
                result = future.result()
                print(result[3], flush=True)
                symbol_results.append(result)

    for normalized, feature_frame, trades, message in symbol_results:
        frames.append(feature_frame)
        train_trades = [trade for trade in trades if pd.Timestamp(train_start) <= trade.entry_time < pd.Timestamp(split)]
        oos_trades = [trade for trade in trades if pd.Timestamp(split) <= trade.entry_time < pd.Timestamp(end)]
        baseline_rows.append({"symbol": normalized, "window": "train", **summarize(train_trades)})
        baseline_rows.append({"symbol": normalized, "window": "oos", **summarize(oos_trades)})
        if args.workers <= 1:
            print(message, flush=True)

    dataset = pd.concat(frames, ignore_index=True).sort_values(["entry_time", "symbol"]).reset_index(drop=True)
    if args.zone_hold_dataset:
        zone_dataset = pd.read_csv(args.zone_hold_dataset)
        zone_prob = dict(zip(zone_dataset["event_key"], zone_dataset["hold_prob"]))
        dataset["zone_hold_prob"] = dataset["event_key"].map(zone_prob)
        missing = int(dataset["zone_hold_prob"].isna().sum())
        if missing:
            print(f"Warning: {missing} trades had no zone-hold probability; filling with 0.5.")
            dataset["zone_hold_prob"] = dataset["zone_hold_prob"].fillna(0.5)
    dataset = add_engineered_rescue_features(dataset)
    if args.zone_hold_pre_filter > 0:
        dataset = dataset[dataset["zone_hold_prob"] >= args.zone_hold_pre_filter].copy()
    feature_columns = list(BASE_FEATURE_COLUMNS)
    if args.use_bfm_features:
        feature_columns.extend(trade_bfm_feature_columns_for_groups(args.bfm_feature_groups))
    if not args.no_symbol_dummies:
        dataset, symbol_columns = add_symbol_dummy_features(dataset)
        feature_columns.extend(symbol_columns)

    dataset = dataset[(dataset["entry_time"] >= pd.Timestamp(train_start)) & (dataset["entry_time"] < pd.Timestamp(end))].copy()
    train = dataset[dataset["entry_time"] < pd.Timestamp(split)].copy()
    oos = dataset[dataset["entry_time"] >= pd.Timestamp(split)].copy()
    if train["win_label"].nunique() < 2:
        raise RuntimeError("Training set has only one class.")

    model = fit_model(train, args.model, feature_columns)
    dataset["trade_win_prob"] = model.predict_proba(dataset[feature_columns].astype(float))[:, 1]
    train = dataset[dataset["entry_time"] < pd.Timestamp(split)].copy()
    oos = dataset[dataset["entry_time"] >= pd.Timestamp(split)].copy()

    args.dataset_out.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.dataset_out, index=False)
    joblib.dump({"model": model, "feature_columns": feature_columns, "model_kind": args.model, "config": vars(args)}, args.model_out)

    thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    print()
    print(f"Dataset saved to {args.dataset_out}")
    print(f"Model saved to {args.model_out}")
    print()
    print("Baseline:")
    baseline = pd.DataFrame(baseline_rows)
    aggregate_rows = []
    for window, frame in dataset.groupby(dataset["entry_time"].lt(pd.Timestamp(split)).map({True: "train", False: "oos"})):
        aggregate_rows.append({"symbol": "AGG", "window": window, **frame_metrics(frame)})
    print(pd.concat([baseline, pd.DataFrame(aggregate_rows)], ignore_index=True).to_string(index=False))
    print()
    print("Classifier metrics:")
    print(pd.DataFrame([
        {"window": "train", **classifier_metrics(train)},
        {"window": "oos", **classifier_metrics(oos)},
    ]).to_string(index=False))
    print()
    print("OOS threshold table:")
    print(threshold_table(oos, thresholds).to_string(index=False))
    print()
    print("OOS by symbol at p>=0.50 and p>=0.55:")
    rows = []
    for symbol, frame in oos.groupby("symbol"):
        for threshold in [0.50, 0.55]:
            rows.append({"symbol": symbol, "threshold": threshold, **frame_metrics(frame[frame["trade_win_prob"] >= threshold])})
    print(pd.DataFrame(rows).to_string(index=False))
    print()
    print("Oracle actual-winner ceiling:")
    oracle_rows = []
    for symbol, frame in oos.groupby("symbol"):
        oracle_rows.append({"symbol": symbol, **frame_metrics(frame[frame["win_label"] == 1])})
    oracle_rows.append({"symbol": "AGG", **frame_metrics(oos[oos["win_label"] == 1])})
    print(pd.DataFrame(oracle_rows).to_string(index=False))
    print()
    print("Largest feature importances:")
    print(feature_rank(model, oos if len(oos) >= 20 else train, args.model, feature_columns).head(14).to_string(index=False))


if __name__ == "__main__":
    main()
