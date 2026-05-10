from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import joblib
    from sklearn.cluster import KMeans
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    joblib = None
    KMeans = None
    SimpleImputer = None
    StandardScaler = None
    make_pipeline = None
    SKLEARN_AVAILABLE = False

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backtest_turtle_soup import (
    add_atr,
    build_confirmed_pivots,
    normalize_binance_spot_symbol,
    parse_utc_datetime,
    resample_ohlc,
)
from scripts.crypto_symbol_sets import SYMBOL_SETS, expand_symbol_args
from scripts.ml_trade_outcome_filter import (
    BASE_FEATURE_COLUMNS as TRADE_BASE_FEATURE_COLUMNS,
    classifier_metrics as trade_classifier_metrics,
    fit_model as fit_trade_model,
    frame_metrics as trade_frame_metrics,
)
from scripts.ml_zone_hold_filter import (
    FEATURE_COLUMNS as ZONE_BASE_FEATURE_COLUMNS,
    classifier_metrics as zone_classifier_metrics,
    fit_sklearn_model as fit_zone_model,
    threshold_table as zone_threshold_table,
)
from scripts.sweep_turtle_soup_oos import ensure_cache


TIMEFRAME_CONFIGS = {
    "15m": {"left": 5, "right": 5, "lookahead": 96},
    "1h": {"left": 5, "right": 5, "lookahead": 96},
    "4h": {"left": 5, "right": 5, "lookahead": 60},
    "1d": {"left": 3, "right": 3, "lookahead": 45},
}

TRADE_ADVANCED_FEATURE_COLUMNS = [
    "sweep_ker20",
    "sweep_ker60",
    "signal_ker20",
    "signal_ker60",
    "sweep_nearest_fvg_dist_atr",
    "sweep_same_side_fvg_dist_atr",
    "sweep_inside_fvg",
    "sweep_open_fvg_count",
    "signal_nearest_fvg_dist_atr",
    "signal_same_side_fvg_dist_atr",
    "signal_inside_fvg",
    "signal_open_fvg_count",
    "signal_weekly_poc_abs_dist_atr",
    "signal_weekly_poc_dir_dist_atr",
    "sweep_volume_share_15m",
    "sweep_volume_share_1h",
    "sweep_volume_rvol20",
    "wick_speed_true_1m_available",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build coin-character/cluster features and test whether they help Turtle Soup modelling."
    )
    parser.add_argument("--symbols", nargs="+", default=[])
    parser.add_argument("--symbol-set", choices=["none", *SYMBOL_SETS.keys()], default="majors20")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--character-start", default="2022-04-20")
    parser.add_argument("--split", default="2025-04-20")
    parser.add_argument("--end", default="2026-04-20")
    parser.add_argument("--cache-dir", type=Path, default=Path("scripts/.cache"))
    parser.add_argument("--zone-dataset", type=Path, default=Path("scripts/zone_hold_dataset_majors20_1h_2022_2026.csv"))
    parser.add_argument("--trade-dataset", type=Path, default=Path("scripts/trade_outcome_dataset_majors20_1h_2stage_pilot.csv"))
    parser.add_argument("--cluster-ks", default="3,4")
    parser.add_argument("--output-prefix", type=Path, default=Path("scripts/coin_cluster_feature_study"))
    parser.add_argument("--skip-candles", action="store_true", help="Use only zone-hold character stats, no candle-derived features.")
    return parser.parse_args()


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den and math.isfinite(den) else math.nan


def median_or_nan(values: pd.Series | np.ndarray) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(arr.median()) if len(arr) else math.nan


def mean_or_nan(values: pd.Series | np.ndarray) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(arr.mean()) if len(arr) else math.nan


def candle_features_for_timeframe(bars: pd.DataFrame, timeframe: str) -> dict[str, float]:
    bars = add_atr(bars).reset_index(drop=True)
    high = pd.to_numeric(bars["high"], errors="coerce")
    low = pd.to_numeric(bars["low"], errors="coerce")
    open_ = pd.to_numeric(bars["open"], errors="coerce")
    close = pd.to_numeric(bars["close"], errors="coerce")
    atr = pd.to_numeric(bars["atr"], errors="coerce")
    body = (close - open_).abs()
    candle_range = high - low
    upper_wick = high - pd.concat([open_, close], axis=1).max(axis=1)
    lower_wick = pd.concat([open_, close], axis=1).min(axis=1) - low
    wick_total = upper_wick + lower_wick
    body_floor = np.maximum(body.to_numpy(dtype=float), 0.05 * atr.bfill().ffill().to_numpy(dtype=float))
    ret = close.pct_change()
    abs_path = close.diff().abs().sum()
    trend_efficiency = safe_div(abs(float(close.iloc[-1] - close.iloc[0])), float(abs_path)) if len(close) > 2 else math.nan

    return {
        f"{timeframe}_wick_body_median": median_or_nan(wick_total.to_numpy(dtype=float) / body_floor),
        f"{timeframe}_wick_range_median": median_or_nan(wick_total / candle_range.replace(0, np.nan)),
        f"{timeframe}_upper_wick_range_median": median_or_nan(upper_wick / candle_range.replace(0, np.nan)),
        f"{timeframe}_lower_wick_range_median": median_or_nan(lower_wick / candle_range.replace(0, np.nan)),
        f"{timeframe}_natr_median": median_or_nan(atr / close * 100.0),
        f"{timeframe}_range_pct_median": median_or_nan(candle_range / close * 100.0),
        f"{timeframe}_return_vol": float(ret.std() * 100.0) if ret.notna().sum() > 2 else math.nan,
        f"{timeframe}_trend_efficiency": trend_efficiency,
    }


def session_activity_features(df: pd.DataFrame) -> dict[str, float]:
    frame = df.copy()
    frame["hour"] = pd.to_datetime(frame["open_time"], utc=True, errors="coerce").dt.hour
    volume = pd.to_numeric(frame["volume"], errors="coerce")
    hourly = volume.groupby(frame["hour"]).mean()
    overall = float(volume.mean())
    sessions = {
        "asia": list(range(0, 8)),
        "london": list(range(7, 16)),
        "newyork": list(range(13, 22)),
    }
    out: dict[str, float] = {}
    session_scores = []
    for name, hours in sessions.items():
        value = mean_or_nan(hourly.reindex(hours))
        rvol = safe_div(value, overall)
        out[f"session_{name}_rvol"] = rvol
        session_scores.append((name, rvol if math.isfinite(rvol) else -1.0))
    total = sum(max(0.0, score) for _, score in session_scores)
    entropy = 0.0
    if total > 0:
        for _, score in session_scores:
            share = max(0.0, score) / total
            if share > 0:
                entropy -= share * math.log(share)
    out["session_rvol_entropy"] = entropy
    best_name, best_score = max(session_scores, key=lambda item: item[1])
    for name, _ in session_scores:
        out[f"session_peak_{name}"] = 1.0 if name == best_name and best_score > 0 else 0.0
    return out


def drawdown_recovery_features(bars: pd.DataFrame, *, drop_threshold: float = -0.05, lookback: int = 12, sma_length: int = 50, horizon: int = 120) -> dict[str, float]:
    close = pd.to_numeric(bars["close"], errors="coerce").reset_index(drop=True)
    future_sma = close.rolling(sma_length).mean()
    drop = close.pct_change(lookback)
    recovery_bars: list[float] = []
    unrecovered = 0
    for index in np.where(drop.to_numpy(dtype=float) <= drop_threshold)[0]:
        if index < sma_length or index >= len(close) - 1:
            continue
        end = min(len(close), index + horizon + 1)
        recovered = False
        for cursor in range(index + 1, end):
            if close.iloc[cursor] >= future_sma.iloc[cursor]:
                recovery_bars.append(float(cursor - index))
                recovered = True
                break
        if not recovered:
            unrecovered += 1
    total = len(recovery_bars) + unrecovered
    return {
        "drop5_recovery_bars_median_1h": median_or_nan(recovery_bars),
        "drop5_recovery_bars_mean_1h": mean_or_nan(recovery_bars),
        "drop5_unrecovered_rate_1h": safe_div(unrecovered, total),
        "drop5_events_per_1000h": safe_div(1000.0 * total, len(close)),
    }


def first_touch_level_stats(bars: pd.DataFrame, timeframe: str, left: int, right: int, lookahead: int) -> dict[str, float]:
    bars = add_atr(bars).reset_index(drop=True)
    highs = pd.to_numeric(bars["high"], errors="coerce").to_list()
    lows = pd.to_numeric(bars["low"], errors="coerce").to_list()
    closes = pd.to_numeric(bars["close"], errors="coerce").to_list()
    atrs = pd.to_numeric(bars["atr"], errors="coerce").bfill().ffill().to_list()
    high_pivots = build_confirmed_pivots(bars["high"], left, right, "high")
    low_pivots = build_confirmed_pivots(bars["low"], left, right, "low")

    support_holds = support_breaks = 0
    resistance_holds = resistance_breaks = 0
    support_penetrations: list[float] = []
    resistance_penetrations: list[float] = []

    for pivot in low_pivots:
        level = float(pivot["value"])
        confirm = int(pivot["pivot_index"]) + right
        for cursor in range(confirm + 1, min(len(bars), confirm + lookahead + 1)):
            if lows[cursor] <= level:
                if closes[cursor] >= level:
                    support_holds += 1
                else:
                    support_breaks += 1
                if atrs[cursor] > 0:
                    support_penetrations.append(max(0.0, (level - lows[cursor]) / atrs[cursor]))
                break

    for pivot in high_pivots:
        level = float(pivot["value"])
        confirm = int(pivot["pivot_index"]) + right
        for cursor in range(confirm + 1, min(len(bars), confirm + lookahead + 1)):
            if highs[cursor] >= level:
                if closes[cursor] <= level:
                    resistance_holds += 1
                else:
                    resistance_breaks += 1
                if atrs[cursor] > 0:
                    resistance_penetrations.append(max(0.0, (highs[cursor] - level) / atrs[cursor]))
                break

    support_total = support_holds + support_breaks
    resistance_total = resistance_holds + resistance_breaks
    total = support_total + resistance_total
    return {
        f"{timeframe}_support_respect_rate": safe_div(support_holds, support_total),
        f"{timeframe}_resistance_respect_rate": safe_div(resistance_holds, resistance_total),
        f"{timeframe}_level_respect_rate": safe_div(support_holds + resistance_holds, total),
        f"{timeframe}_support_touch_rate_1000": safe_div(1000.0 * support_total, len(bars)),
        f"{timeframe}_resistance_touch_rate_1000": safe_div(1000.0 * resistance_total, len(bars)),
        f"{timeframe}_support_penetration_atr_median": median_or_nan(support_penetrations),
        f"{timeframe}_resistance_penetration_atr_median": median_or_nan(resistance_penetrations),
    }


def candle_character_features(
    symbols: list[str],
    *,
    interval: str,
    start: pd.Timestamp,
    split: pd.Timestamp,
    cache_dir: Path,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    rows: list[dict[str, Any]] = []
    hourly_returns: dict[str, pd.Series] = {}

    for raw_symbol in symbols:
        normalized = normalize_binance_spot_symbol(raw_symbol)
        cache = ensure_cache(raw_symbol, interval, start.to_pydatetime(), split.to_pydatetime(), cache_dir)
        df = pd.read_pickle(cache)
        df = df[(df["open_time"] >= start) & (df["open_time"] < split)].sort_values("open_time").reset_index(drop=True)
        if df.empty:
            continue

        row: dict[str, Any] = {"symbol": normalized}
        daily = resample_ohlc(df, "1d")
        row["daily_notional_median"] = median_or_nan(pd.to_numeric(daily["volume"], errors="coerce") * pd.to_numeric(daily["close"], errors="coerce"))
        row["daily_notional_log"] = math.log1p(row["daily_notional_median"]) if math.isfinite(row["daily_notional_median"]) else math.nan
        row["daily_quote_range_pct_median"] = median_or_nan((daily["high"] - daily["low"]) / daily["close"] * 100.0)
        row.update(session_activity_features(df))

        for timeframe, config in TIMEFRAME_CONFIGS.items():
            bars = resample_ohlc(df, timeframe)
            if len(bars) < config["left"] + config["right"] + 10:
                continue
            row.update(candle_features_for_timeframe(bars, timeframe))
            row.update(first_touch_level_stats(bars, timeframe, config["left"], config["right"], config["lookahead"]))
            if timeframe == "1h":
                row.update(drawdown_recovery_features(bars))
                hourly_returns[normalized] = pd.to_numeric(bars["close"], errors="coerce").pct_change().rename(normalized)

        rows.append(row)
        print(f"Character candles {normalized}: {len(df):,} {interval} bars", flush=True)

    return pd.DataFrame(rows), hourly_returns


def add_beta_features(features: pd.DataFrame, hourly_returns: dict[str, pd.Series]) -> pd.DataFrame:
    if not hourly_returns:
        return features
    returns = pd.concat(hourly_returns.values(), axis=1).dropna(how="all")
    out = features.copy()
    for benchmark in ["BTCUSDT", "ETHUSDT"]:
        if benchmark not in returns.columns:
            continue
        bench = returns[benchmark]
        bench_var = float(bench.var())
        for idx, row in out.iterrows():
            symbol = str(row["symbol"])
            if symbol not in returns.columns:
                continue
            pair = pd.concat([returns[symbol], bench], axis=1).dropna()
            if len(pair) < 100 or bench_var <= 0:
                continue
            y = pair.iloc[:, 0]
            x = pair.iloc[:, 1]
            beta = float(y.cov(x) / x.var()) if x.var() > 0 else math.nan
            corr = float(y.corr(x))
            residual = y - beta * x if math.isfinite(beta) else pd.Series(dtype=float)
            out.loc[idx, f"{benchmark.lower()}_beta_1h"] = beta
            out.loc[idx, f"{benchmark.lower()}_corr_1h"] = corr
            out.loc[idx, f"{benchmark.lower()}_idio_vol_1h"] = float(residual.std() * 100.0) if len(residual) else math.nan
            up = pair[x > 0]
            down = pair[x < 0]
            up_var = float(up.iloc[:, 1].var()) if len(up) > 20 else math.nan
            down_var = float(down.iloc[:, 1].var()) if len(down) > 20 else math.nan
            up_beta = float(up.iloc[:, 0].cov(up.iloc[:, 1]) / up_var) if up_var and math.isfinite(up_var) and up_var > 0 else math.nan
            down_beta = float(down.iloc[:, 0].cov(down.iloc[:, 1]) / down_var) if down_var and math.isfinite(down_var) and down_var > 0 else math.nan
            out.loc[idx, f"{benchmark.lower()}_up_beta_1h"] = up_beta
            out.loc[idx, f"{benchmark.lower()}_down_beta_1h"] = down_beta
            out.loc[idx, f"{benchmark.lower()}_beta_asym_1h"] = down_beta - up_beta if math.isfinite(down_beta) and math.isfinite(up_beta) else math.nan
    return out


def zone_character_features(zone_dataset: Path, *, start: pd.Timestamp, split: pd.Timestamp) -> pd.DataFrame:
    if not zone_dataset.exists():
        return pd.DataFrame(columns=["symbol"])
    usecols = [
        "symbol",
        "time",
        "direction",
        "hold_label",
        "future_r",
        "zone_width_atr",
        "penetration_frac",
        "reclaim_pos",
        "sweep_range_atr",
        "same_bar_reaction_atr",
        "same_bar_adverse_atr",
        "vol_mult",
    ]
    frame = pd.read_csv(zone_dataset, usecols=lambda col: col in usecols)
    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
    frame = frame[(frame["time"] >= start) & (frame["time"] < split)].copy()
    if frame.empty:
        return pd.DataFrame(columns=["symbol"])
    rows: list[dict[str, Any]] = []
    for symbol, group in frame.groupby("symbol"):
        long_group = group[group["direction"] == "long"]
        short_group = group[group["direction"] == "short"]
        rows.append({
            "symbol": normalize_binance_spot_symbol(str(symbol)),
            "zone_1h_events": float(len(group)),
            "zone_1h_hold_rate": mean_or_nan(group["hold_label"]),
            "zone_1h_support_hold_rate": mean_or_nan(long_group["hold_label"]),
            "zone_1h_resistance_hold_rate": mean_or_nan(short_group["hold_label"]),
            "zone_1h_future_r_mean": mean_or_nan(group["future_r"]),
            "zone_1h_width_atr_median": median_or_nan(group["zone_width_atr"]),
            "zone_1h_penetration_mean": mean_or_nan(group["penetration_frac"]),
            "zone_1h_reclaim_mean": mean_or_nan(group["reclaim_pos"]),
            "zone_1h_sweep_range_atr_median": median_or_nan(group["sweep_range_atr"]),
            "zone_1h_reaction_atr_mean": mean_or_nan(group["same_bar_reaction_atr"]),
            "zone_1h_adverse_atr_mean": mean_or_nan(group["same_bar_adverse_atr"]),
            "zone_1h_vol_mult_median": median_or_nan(group["vol_mult"]),
        })
    return pd.DataFrame(rows)


def build_character_features(args: argparse.Namespace, symbols: list[str], start: pd.Timestamp, split: pd.Timestamp) -> pd.DataFrame:
    zone_features = zone_character_features(args.zone_dataset, start=start, split=split)
    if args.skip_candles:
        features = zone_features
    else:
        candle_features, hourly_returns = candle_character_features(
            symbols,
            interval=args.interval,
            start=start,
            split=split,
            cache_dir=args.cache_dir,
        )
        candle_features = add_beta_features(candle_features, hourly_returns)
        features = candle_features.merge(zone_features, on="symbol", how="outer")
    features = features.sort_values("symbol").reset_index(drop=True)
    return features


def add_clusters(features: pd.DataFrame, ks: list[int]) -> tuple[pd.DataFrame, list[str]]:
    numeric_columns = [column for column in features.columns if column != "symbol"]
    numeric_columns = [column for column in numeric_columns if pd.to_numeric(features[column], errors="coerce").notna().any()]
    out = features.copy()
    cluster_columns: list[str] = []
    if not SKLEARN_AVAILABLE or not numeric_columns:
        return out, cluster_columns
    x = out[numeric_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    xs = pipeline.fit_transform(x)
    for k in ks:
        if k <= 1 or k > len(out):
            continue
        labels = KMeans(n_clusters=k, n_init=50, random_state=23).fit_predict(xs)
        label_column = f"coin_cluster_k{k}"
        out[label_column] = labels.astype(int)
        for cluster in range(k):
            column = f"{label_column}_{cluster}"
            out[column] = (out[label_column] == cluster).astype(float)
            cluster_columns.append(column)
    return out, cluster_columns


def character_numeric_columns(features: pd.DataFrame) -> list[str]:
    skip = {"symbol"}
    skip.update(column for column in features.columns if column.startswith("coin_cluster"))
    return [
        column
        for column in features.columns
        if column not in skip and pd.to_numeric(features[column], errors="coerce").notna().any()
    ]


def join_character(dataset: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    out = dataset.copy()
    out["symbol"] = out["symbol"].astype(str).map(normalize_binance_spot_symbol)
    return out.merge(features, on="symbol", how="left")


def add_ker_columns(df: pd.DataFrame, lengths: tuple[int, ...] = (20, 60)) -> pd.DataFrame:
    out = df.copy()
    close = pd.to_numeric(out["close"], errors="coerce")
    path = close.diff().abs()
    for length in lengths:
        numerator = (close - close.shift(length)).abs()
        denominator = path.rolling(length).sum()
        out[f"ker{length}"] = numerator / denominator.replace(0, np.nan)
    return out


def open_fvg_snapshot(df: pd.DataFrame, index: int, close: float, atr: float, direction: str, lookback: int = 500) -> dict[str, float]:
    if index < 3 or atr <= 0 or not math.isfinite(atr):
        return {
            "nearest_fvg_dist_atr": math.nan,
            "same_side_fvg_dist_atr": math.nan,
            "inside_fvg": 0.0,
            "open_fvg_count": 0.0,
        }
    highs = pd.to_numeric(df["high"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(df["low"], errors="coerce").to_numpy(dtype=float)
    start = max(2, index - lookback)
    nearest = math.inf
    same_side = math.inf
    inside = 0.0
    count = 0
    for cursor in range(start, index + 1):
        # Bullish imbalance: high[cursor-2] < low[cursor].
        if lows[cursor] > highs[cursor - 2]:
            gap_low = highs[cursor - 2]
            gap_high = lows[cursor]
            mitigated = np.nanmin(lows[cursor + 1:index + 1]) <= gap_low if cursor + 1 <= index else False
            if not mitigated:
                count += 1
                dist = 0.0 if gap_low <= close <= gap_high else min(abs(close - gap_low), abs(close - gap_high))
                nearest = min(nearest, dist)
                if direction == "long" and close >= gap_low:
                    same_side = min(same_side, dist)
                if gap_low <= close <= gap_high:
                    inside = 1.0
        # Bearish imbalance: low[cursor-2] > high[cursor].
        if highs[cursor] < lows[cursor - 2]:
            gap_low = highs[cursor]
            gap_high = lows[cursor - 2]
            mitigated = np.nanmax(highs[cursor + 1:index + 1]) >= gap_high if cursor + 1 <= index else False
            if not mitigated:
                count += 1
                dist = 0.0 if gap_low <= close <= gap_high else min(abs(close - gap_low), abs(close - gap_high))
                nearest = min(nearest, dist)
                if direction == "short" and close <= gap_high:
                    same_side = min(same_side, dist)
                if gap_low <= close <= gap_high:
                    inside = 1.0
    return {
        "nearest_fvg_dist_atr": float(nearest / atr) if math.isfinite(nearest) else 999.0,
        "same_side_fvg_dist_atr": float(same_side / atr) if math.isfinite(same_side) else 999.0,
        "inside_fvg": inside,
        "open_fvg_count": float(count),
    }


def weekly_poc_features(df: pd.DataFrame, index: int, close: float, atr: float, direction: str, lookback_bars: int = 2016, bins: int = 80) -> dict[str, float]:
    if index <= 10 or atr <= 0 or not math.isfinite(atr):
        return {"signal_weekly_poc_abs_dist_atr": math.nan, "signal_weekly_poc_dir_dist_atr": math.nan}
    start = max(0, index - lookback_bars)
    window = df.iloc[start:index]
    prices = pd.to_numeric(window["close"], errors="coerce").to_numpy(dtype=float)
    volumes = pd.to_numeric(window["volume"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(prices) & np.isfinite(volumes)
    prices = prices[mask]
    volumes = volumes[mask]
    if len(prices) < 50 or prices.max() <= prices.min():
        return {"signal_weekly_poc_abs_dist_atr": math.nan, "signal_weekly_poc_dir_dist_atr": math.nan}
    edges = np.linspace(prices.min(), prices.max(), bins + 1)
    bucket = np.clip(np.searchsorted(edges, prices, side="right") - 1, 0, bins - 1)
    volume_by_bucket = np.bincount(bucket, weights=volumes, minlength=bins)
    poc_bucket = int(np.argmax(volume_by_bucket))
    poc = float((edges[poc_bucket] + edges[poc_bucket + 1]) / 2.0)
    sign = 1.0 if direction == "long" else -1.0
    return {
        "signal_weekly_poc_abs_dist_atr": abs(close - poc) / atr,
        "signal_weekly_poc_dir_dist_atr": sign * (close - poc) / atr,
    }


def event_index_for_time(times: pd.Series, value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    values = pd.to_datetime(times, utc=True, errors="coerce").to_numpy(dtype="datetime64[ns]").astype("int64")
    target = int(ts.value)
    pos = int(np.searchsorted(values, target, side="right") - 1)
    if pos < 0 or pos >= len(values):
        return None
    # sweep_time is stored as bar open; signal_time is often stored as bar close
    # (...:xx:59.999). Map both to the containing 5m execution bar.
    if 0 <= target - int(values[pos]) <= pd.Timedelta(minutes=5).value:
        return pos
    return None


def enrich_trade_event_features(
    trades: pd.DataFrame,
    symbols: list[str],
    *,
    interval: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: Path,
) -> pd.DataFrame:
    if trades.empty:
        return trades
    out = trades.copy()
    for column in TRADE_ADVANCED_FEATURE_COLUMNS:
        if column not in out.columns:
            out[column] = math.nan
    out["wick_speed_true_1m_available"] = 0.0

    for raw_symbol in symbols:
        symbol = normalize_binance_spot_symbol(raw_symbol)
        mask = out["symbol"].astype(str).map(normalize_binance_spot_symbol) == symbol
        if not mask.any():
            continue
        cache = ensure_cache(raw_symbol, interval, start.to_pydatetime(), end.to_pydatetime(), cache_dir)
        df = pd.read_pickle(cache)
        df = df[(df["open_time"] >= start) & (df["open_time"] < end)].sort_values("open_time").reset_index(drop=True)
        if df.empty:
            continue
        df = add_ker_columns(add_atr(df))
        df["vol_sma20"] = pd.to_numeric(df["volume"], errors="coerce").rolling(20).mean()
        times = df["open_time"]
        opens = pd.to_datetime(df["open_time"], utc=True, errors="coerce")
        volumes = pd.to_numeric(df["volume"], errors="coerce").to_numpy(dtype=float)
        closes = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)
        atrs = pd.to_numeric(df["atr"], errors="coerce").bfill().ffill().to_numpy(dtype=float)

        for idx, row in out[mask].iterrows():
            direction = str(row.get("direction", "")).lower()
            sweep_idx = event_index_for_time(times, row.get("sweep_time"))
            signal_idx = event_index_for_time(times, row.get("signal_time"))
            if sweep_idx is not None:
                out.loc[idx, "sweep_ker20"] = float(df.iloc[sweep_idx].get("ker20", math.nan))
                out.loc[idx, "sweep_ker60"] = float(df.iloc[sweep_idx].get("ker60", math.nan))
                fvg = open_fvg_snapshot(df, sweep_idx, closes[sweep_idx], atrs[sweep_idx], direction)
                for key, value in fvg.items():
                    out.loc[idx, f"sweep_{key}"] = value
                bucket_15 = opens.dt.floor("15min") == opens.iloc[sweep_idx].floor("15min")
                bucket_1h = opens.dt.floor("1h") == opens.iloc[sweep_idx].floor("1h")
                vol_15 = float(np.nansum(volumes[bucket_15.to_numpy()]))
                vol_1h = float(np.nansum(volumes[bucket_1h.to_numpy()]))
                out.loc[idx, "sweep_volume_share_15m"] = safe_div(volumes[sweep_idx], vol_15)
                out.loc[idx, "sweep_volume_share_1h"] = safe_div(volumes[sweep_idx], vol_1h)
                vol_sma = float(df.iloc[sweep_idx].get("vol_sma20", math.nan))
                out.loc[idx, "sweep_volume_rvol20"] = safe_div(volumes[sweep_idx], vol_sma)
            if signal_idx is not None:
                out.loc[idx, "signal_ker20"] = float(df.iloc[signal_idx].get("ker20", math.nan))
                out.loc[idx, "signal_ker60"] = float(df.iloc[signal_idx].get("ker60", math.nan))
                fvg = open_fvg_snapshot(df, signal_idx, closes[signal_idx], atrs[signal_idx], direction)
                for key, value in fvg.items():
                    out.loc[idx, f"signal_{key}"] = value
                poc = weekly_poc_features(df, signal_idx, closes[signal_idx], atrs[signal_idx], direction)
                for key, value in poc.items():
                    out.loc[idx, key] = value
    return out


def profit_factor(rs: pd.Series) -> float:
    wins = rs[rs > 0]
    losses = rs[rs <= 0]
    loss = abs(float(losses.sum()))
    if loss == 0:
        return float("inf") if len(wins) else 0.0
    return float(wins.sum()) / loss


def zone_variant_metrics(frame: pd.DataFrame, threshold: float) -> dict[str, float]:
    kept = frame[frame["hold_prob"] >= threshold]
    if kept.empty:
        return {"trades": 0, "hold_rate": 0.0, "avg_future_r": 0.0, "net_future_r": 0.0}
    return {
        "trades": int(len(kept)),
        "hold_rate": round(float(kept["hold_label"].mean()) * 100.0, 2),
        "avg_future_r": round(float(kept["future_r"].mean()), 3),
        "net_future_r": round(float(kept["future_r"].sum()), 3),
    }


def evaluate_zone_dataset(dataset: pd.DataFrame, features: pd.DataFrame, cluster_columns: list[str], split: pd.Timestamp) -> pd.DataFrame:
    frame = join_character(dataset, features)
    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
    train = frame[frame["time"] < split].copy()
    oos = frame[frame["time"] >= split].copy()
    char_columns = character_numeric_columns(features)
    symbol_columns = []
    for symbol in sorted(frame["symbol"].dropna().unique()):
        column = "symbol_" + str(symbol).lower()
        frame[column] = (frame["symbol"] == symbol).astype(float)
        symbol_columns.append(column)
    train = frame[frame["time"] < split].copy()
    oos = frame[frame["time"] >= split].copy()

    variants = {
        "base": list(ZONE_BASE_FEATURE_COLUMNS),
        "base_symbol": list(ZONE_BASE_FEATURE_COLUMNS) + symbol_columns,
        "base_character": list(ZONE_BASE_FEATURE_COLUMNS) + char_columns,
        "base_cluster": list(ZONE_BASE_FEATURE_COLUMNS) + cluster_columns,
        "base_character_cluster": list(ZONE_BASE_FEATURE_COLUMNS) + char_columns + cluster_columns,
    }
    rows: list[dict[str, Any]] = []
    for variant, columns in variants.items():
        columns = [column for column in dict.fromkeys(columns) if column in frame.columns]
        model = fit_zone_model(train, columns, "sklearn_rf")
        scored = oos.copy()
        scored["hold_prob"] = model.predict_proba(scored[columns].astype(float))[:, 1]
        metrics = zone_classifier_metrics(scored)
        for threshold in [0.50, 0.55, 0.60, 0.65]:
            rows.append({
                "task": "zone_hold",
                "variant": variant,
                "threshold": threshold,
                "feature_count": len(columns),
                **metrics,
                **zone_variant_metrics(scored, threshold),
            })
    return pd.DataFrame(rows)


def evaluate_trade_dataset(dataset: pd.DataFrame, features: pd.DataFrame, cluster_columns: list[str], split: pd.Timestamp) -> pd.DataFrame:
    frame = join_character(dataset, features)
    frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True, errors="coerce")
    char_columns = character_numeric_columns(features)
    existing_symbol_columns = [column for column in frame.columns if column.startswith("symbol_")]
    base_columns = [column for column in TRADE_BASE_FEATURE_COLUMNS if column in frame.columns]
    event_columns = [column for column in TRADE_ADVANCED_FEATURE_COLUMNS if column in frame.columns]
    variants = {
        "base": base_columns,
        "base_symbol": base_columns + existing_symbol_columns,
        "base_event_advanced": base_columns + event_columns,
        "base_event_cluster": base_columns + event_columns + cluster_columns,
        "base_character": base_columns + char_columns,
        "base_cluster": base_columns + cluster_columns,
        "base_character_cluster": base_columns + char_columns + cluster_columns,
    }
    rows: list[dict[str, Any]] = []
    train = frame[frame["entry_time"] < split].copy()
    oos = frame[frame["entry_time"] >= split].copy()
    for variant, columns in variants.items():
        columns = [column for column in dict.fromkeys(columns) if column in frame.columns]
        if train["win_label"].nunique() < 2:
            continue
        model = fit_trade_model(train, "rf", columns)
        scored = oos.copy()
        scored["trade_win_prob"] = model.predict_proba(scored[columns].astype(float))[:, 1]
        metrics = trade_classifier_metrics(scored)
        for threshold in [0.45, 0.50, 0.55, 0.60]:
            kept = scored[scored["trade_win_prob"] >= threshold]
            rows.append({
                "task": "trade_outcome",
                "variant": variant,
                "threshold": threshold,
                "feature_count": len(columns),
                **metrics,
                **trade_frame_metrics(kept),
            })
    return pd.DataFrame(rows)


def cluster_summary(features: pd.DataFrame, ks: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for k in ks:
        label_column = f"coin_cluster_k{k}"
        if label_column not in features.columns:
            continue
        for cluster, group in features.groupby(label_column):
            rows.append({
                "k": k,
                "cluster": int(cluster),
                "symbols": ",".join(group["symbol"].astype(str).sort_values()),
                "count": int(len(group)),
                "zone_1h_hold_rate": mean_or_nan(group.get("zone_1h_hold_rate", pd.Series(dtype=float))),
                "daily_notional_log": mean_or_nan(group.get("daily_notional_log", pd.Series(dtype=float))),
                "1h_natr_median": mean_or_nan(group.get("1h_natr_median", pd.Series(dtype=float))),
                "1h_level_respect_rate": mean_or_nan(group.get("1h_level_respect_rate", pd.Series(dtype=float))),
                "btcusdt_beta_1h": mean_or_nan(group.get("btcusdt_beta_1h", pd.Series(dtype=float))),
                "btcusdt_beta_asym_1h": mean_or_nan(group.get("btcusdt_beta_asym_1h", pd.Series(dtype=float))),
                "drop5_recovery_bars_median_1h": mean_or_nan(group.get("drop5_recovery_bars_median_1h", pd.Series(dtype=float))),
                "session_asia_rvol": mean_or_nan(group.get("session_asia_rvol", pd.Series(dtype=float))),
                "session_london_rvol": mean_or_nan(group.get("session_london_rvol", pd.Series(dtype=float))),
                "session_newyork_rvol": mean_or_nan(group.get("session_newyork_rvol", pd.Series(dtype=float))),
            })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if not SKLEARN_AVAILABLE:
        raise SystemExit("scikit-learn is required. Run with .venv\\Scripts\\python.exe.")
    symbols = expand_symbol_args(args.symbols, args.symbol_set)
    start = pd.Timestamp(parse_utc_datetime(args.character_start))
    split = pd.Timestamp(parse_utc_datetime(args.split))
    ks = [int(value.strip()) for value in str(args.cluster_ks).split(",") if value.strip()]

    print(f"Building character features for {len(symbols)} symbols from {start.date()} to {split.date()}")
    features = build_character_features(args, symbols, start, split)
    features, cluster_columns = add_clusters(features, ks)
    cluster_info = cluster_summary(features, ks)

    outputs = {}
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    features_path = args.output_prefix.with_name(args.output_prefix.name + "_features.csv")
    clusters_path = args.output_prefix.with_name(args.output_prefix.name + "_clusters.csv")
    features.to_csv(features_path, index=False)
    cluster_info.to_csv(clusters_path, index=False)
    outputs["features"] = features_path
    outputs["clusters"] = clusters_path

    result_frames: list[pd.DataFrame] = []
    if args.zone_dataset.exists():
        zone = pd.read_csv(args.zone_dataset)
        zone["symbol"] = zone["symbol"].astype(str).map(normalize_binance_spot_symbol)
        zone["time"] = pd.to_datetime(zone["time"], utc=True, errors="coerce")
        zone = zone[(zone["time"] >= start) & (zone["time"] < pd.Timestamp(parse_utc_datetime(args.end)))].copy()
        result_frames.append(evaluate_zone_dataset(zone, features, cluster_columns, split))
    if args.trade_dataset.exists():
        trades = pd.read_csv(args.trade_dataset)
        trades["symbol"] = trades["symbol"].astype(str).map(normalize_binance_spot_symbol)
        trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce")
        trades = trades[(trades["entry_time"] >= start) & (trades["entry_time"] < pd.Timestamp(parse_utc_datetime(args.end)))].copy()
        print("Building advanced event features for trade rows", flush=True)
        trades = enrich_trade_event_features(
            trades,
            symbols,
            interval=args.interval,
            start=start,
            end=pd.Timestamp(parse_utc_datetime(args.end)),
            cache_dir=args.cache_dir,
        )
        enriched_trade_path = args.output_prefix.with_name(args.output_prefix.name + "_trade_event_features.csv")
        trades.to_csv(enriched_trade_path, index=False)
        outputs["trade_event_features"] = enriched_trade_path
        result_frames.append(evaluate_trade_dataset(trades, features, cluster_columns, split))

    results = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    results_path = args.output_prefix.with_name(args.output_prefix.name + "_model_results.csv")
    results.to_csv(results_path, index=False)
    outputs["results"] = results_path

    print("\nClusters:")
    print(cluster_info.to_string(index=False))
    print("\nModel results:")
    display_columns = [
        column
        for column in [
            "task",
            "variant",
            "threshold",
            "feature_count",
            "auc",
            "trades",
            "win_rate",
            "hold_rate",
            "profit_factor",
            "net_r",
            "net_future_r",
            "avg_r",
            "avg_future_r",
            "max_dd_r",
        ]
        if column in results.columns
    ]
    print(results[display_columns].to_string(index=False))
    for label, path in outputs.items():
        print(f"Wrote {label}: {path}")


if __name__ == "__main__":
    main()
