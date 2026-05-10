from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline

    SKLEARN_AVAILABLE = True
except ImportError:
    RandomForestClassifier = None
    SimpleImputer = None
    roc_auc_score = None
    make_pipeline = None
    SKLEARN_AVAILABLE = False

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backtest_turtle_soup import add_atr, parse_utc_datetime, resample_ohlc  # noqa: E402


SESSION_SPECS = {
    "asia": {"start_min": 0, "end_min": 8 * 60},
    "london": {"start_min": 7 * 60, "end_min": 12 * 60},
    "ny": {"start_min": 13 * 60 + 30, "end_min": 18 * 60},
}

META_COLUMNS = {
    "symbol",
    "variant",
    "family",
    "session",
    "session_date",
    "direction",
    "signal_index",
    "entry_index",
    "exit_index",
    "signal_time",
    "entry_time",
    "exit_time",
    "entry_price",
    "stop_price",
    "target_price",
    "exit_price",
    "exit_reason",
    "r_multiple_gross",
    "r_multiple",
    "win_label",
    "session_id",
}


@dataclass(frozen=True)
class OrbConfig:
    family: str
    session: str
    or_minutes: int
    rr: float
    max_hold_bars: int
    stop_mode: str
    min_or_width_atr: float = 0.25
    max_or_width_atr: float = 6.0
    min_break_atr: float = 0.0
    retest_wait_bars: int = 12
    retest_tolerance_atr: float = 0.10
    min_sweep_atr: float = 0.05
    entry_mode: str = "immediate"
    stop_buffer_atr: float = 0.05

    @property
    def variant(self) -> str:
        return (
            f"{self.session}_{self.family}_or{self.or_minutes}_rr{self.rr:g}_"
            f"hold{self.max_hold_bars}_{self.stop_mode}_br{self.min_break_atr:g}_"
            f"tol{self.retest_tolerance_atr:g}_sw{self.min_sweep_atr:g}_{self.entry_mode}"
        )


def profit_factor(rs: Iterable[float]) -> float:
    arr = np.asarray(list(rs), dtype=float)
    if len(arr) == 0:
        return 0.0
    wins = arr[arr > 0]
    losses = arr[arr <= 0]
    gross_loss = abs(float(losses.sum()))
    if gross_loss == 0:
        return float("inf") if len(wins) else 0.0
    return float(wins.sum()) / gross_loss


def max_drawdown_r(rs: Iterable[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for value in rs:
        equity += float(value)
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)
    return round(max_dd, 3)


def metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "net_r": 0.0,
            "avg_r": 0.0,
            "max_dd_r": 0.0,
        }
    ordered = frame.sort_values("exit_time")
    rs = ordered["r_multiple"].astype(float).to_list()
    return {
        "trades": int(len(ordered)),
        "win_rate": round(100.0 * float((ordered["r_multiple"].astype(float) > 0).mean()), 2),
        "profit_factor": round(profit_factor(rs), 3),
        "net_r": round(float(np.sum(rs)), 3),
        "avg_r": round(float(np.mean(rs)), 3),
        "max_dd_r": max_drawdown_r(rs),
    }


def safe_div(num: float, den: float) -> float:
    return num / den if den and math.isfinite(den) else math.nan


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.astype(float).ewm(span=length, adjust=False, min_periods=length).mean()


def add_htf_context(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("open_time").reset_index(drop=True).copy()
    out = add_atr(out)
    out["vol_sma20"] = out["volume"].rolling(20).mean()
    out["ema20"] = ema(out["close"], 20)
    out["ema50"] = ema(out["close"], 50)
    out["ema200"] = ema(out["close"], 200)
    out["ret_1h"] = out["close"].pct_change(12) * 100.0
    out["ret_4h"] = out["close"].pct_change(48) * 100.0
    out["ret_24h"] = out["close"].pct_change(288) * 100.0
    out["range_1h_atr"] = (out["high"].rolling(12).max() - out["low"].rolling(12).min()) / out["atr"]
    out["range_4h_atr"] = (out["high"].rolling(48).max() - out["low"].rolling(48).min()) / out["atr"]
    out["minute_of_day"] = out["open_time"].dt.hour * 60 + out["open_time"].dt.minute
    out["session_date"] = out["open_time"].dt.floor("D")

    for tf, prefix in [("1h", "h1"), ("4h", "h4"), ("1d", "d1")]:
        htf = resample_ohlc(out[["open_time", "close_time", "open", "high", "low", "close", "volume"]], tf)
        if htf.empty:
            continue
        htf[f"{prefix}_ema50"] = ema(htf["close"], 50)
        htf[f"{prefix}_ema200"] = ema(htf["close"], 200)
        htf[f"{prefix}_ret3"] = htf["close"].pct_change(3) * 100.0
        merged = pd.merge_asof(
            out[["close_time"]].sort_values("close_time"),
            htf[["close_time", "close", f"{prefix}_ema50", f"{prefix}_ema200", f"{prefix}_ret3"]].sort_values("close_time"),
            on="close_time",
            direction="backward",
        ).sort_index()
        out[f"{prefix}_close"] = merged["close"].to_numpy()
        out[f"{prefix}_close_vs_ema50_pct"] = (merged["close"] - merged[f"{prefix}_ema50"]) / merged[f"{prefix}_ema50"] * 100.0
        out[f"{prefix}_close_vs_ema200_pct"] = (merged["close"] - merged[f"{prefix}_ema200"]) / merged[f"{prefix}_ema200"] * 100.0
        out[f"{prefix}_ret3"] = merged[f"{prefix}_ret3"].to_numpy()

    daily = resample_ohlc(out[["open_time", "close_time", "open", "high", "low", "close", "volume"]], "1d")
    daily["prev_day_high"] = daily["high"].shift(1)
    daily["prev_day_low"] = daily["low"].shift(1)
    daily["prev_day_close"] = daily["close"].shift(1)
    daily_levels = pd.merge_asof(
        out[["close_time"]].sort_values("close_time"),
        daily[["close_time", "prev_day_high", "prev_day_low", "prev_day_close"]].sort_values("close_time"),
        on="close_time",
        direction="backward",
    ).sort_index()
    out["prev_day_high"] = daily_levels["prev_day_high"].to_numpy()
    out["prev_day_low"] = daily_levels["prev_day_low"].to_numpy()
    out["prev_day_close"] = daily_levels["prev_day_close"].to_numpy()

    out["daily_vwap"] = (
        (out["close"] * out["volume"]).groupby(out["session_date"]).cumsum()
        / out["volume"].groupby(out["session_date"]).cumsum().replace(0.0, np.nan)
    )
    return add_fvg_state(out)


def add_fvg_state(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    n = len(out)
    bull_mid = np.full(n, np.nan)
    bear_mid = np.full(n, np.nan)
    bull_width = np.full(n, np.nan)
    bear_width = np.full(n, np.nan)
    bull_age = np.full(n, np.nan)
    bear_age = np.full(n, np.nan)
    last_bull: tuple[float, float, int] | None = None
    last_bear: tuple[float, float, int] | None = None
    highs = out["high"].astype(float).to_numpy()
    lows = out["low"].astype(float).to_numpy()
    atr = out["atr"].bfill().ffill().astype(float).to_numpy()
    for i in range(n):
        if last_bull is not None and lows[i] <= last_bull[0]:
            last_bull = None
        if last_bear is not None and highs[i] >= last_bear[1]:
            last_bear = None
        if i >= 2:
            if lows[i] > highs[i - 2]:
                last_bull = (highs[i - 2], lows[i], i)
            if highs[i] < lows[i - 2]:
                last_bear = (highs[i], lows[i - 2], i)
        if last_bull is not None:
            bottom, top, idx = last_bull
            bull_mid[i] = (top + bottom) / 2.0
            bull_width[i] = (top - bottom) / atr[i] if atr[i] > 0 else math.nan
            bull_age[i] = i - idx
        if last_bear is not None:
            bottom, top, idx = last_bear
            bear_mid[i] = (top + bottom) / 2.0
            bear_width[i] = (top - bottom) / atr[i] if atr[i] > 0 else math.nan
            bear_age[i] = i - idx
    out["bull_fvg_mid"] = bull_mid
    out["bear_fvg_mid"] = bear_mid
    out["bull_fvg_width_atr"] = bull_width
    out["bear_fvg_width_atr"] = bear_width
    out["bull_fvg_age"] = bull_age
    out["bear_fvg_age"] = bear_age
    out["close_to_bull_fvg_atr"] = (out["close"] - out["bull_fvg_mid"]) / out["atr"]
    out["close_to_bear_fvg_atr"] = (out["bear_fvg_mid"] - out["close"]) / out["atr"]
    return out


def load_cached_symbol(symbol: str, cache_dir: Path) -> pd.DataFrame:
    normalized = symbol.lower()
    candidates = sorted(cache_dir.glob(f"{normalized}_5m_*.pkl"), key=lambda p: p.stat().st_size, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No cached 5m file found for {symbol} in {cache_dir}")
    return pd.read_pickle(candidates[0])


def simulate_exit(
    df: pd.DataFrame,
    *,
    entry_idx: int,
    direction: str,
    entry: float,
    stop: float,
    target: float,
    max_hold_bars: int,
    session_end_idx: int,
    fee_bps_per_side: float,
) -> dict[str, Any] | None:
    if entry_idx >= len(df):
        return None
    if direction == "long":
        risk = entry - stop
    else:
        risk = stop - entry
    if risk <= 0 or not math.isfinite(risk):
        return None
    end_idx = min(len(df) - 1, session_end_idx, entry_idx + max_hold_bars)
    exit_idx = end_idx
    exit_price = float(df.at[end_idx, "close"])
    exit_reason = "time"
    for i in range(entry_idx, end_idx + 1):
        high = float(df.at[i, "high"])
        low = float(df.at[i, "low"])
        if direction == "long":
            stop_hit = low <= stop
            target_hit = high >= target
            if stop_hit and target_hit:
                exit_idx, exit_price, exit_reason = i, stop, "stop_and_target_same_bar"
                break
            if stop_hit:
                exit_idx, exit_price, exit_reason = i, stop, "stop"
                break
            if target_hit:
                exit_idx, exit_price, exit_reason = i, target, "target"
                break
        else:
            stop_hit = high >= stop
            target_hit = low <= target
            if stop_hit and target_hit:
                exit_idx, exit_price, exit_reason = i, stop, "stop_and_target_same_bar"
                break
            if stop_hit:
                exit_idx, exit_price, exit_reason = i, stop, "stop"
                break
            if target_hit:
                exit_idx, exit_price, exit_reason = i, target, "target"
                break
    gross = (exit_price - entry) / risk if direction == "long" else (entry - exit_price) / risk
    fee_r = ((entry + exit_price) * (fee_bps_per_side / 10000.0)) / risk
    return {
        "exit_index": exit_idx,
        "exit_time": df.at[exit_idx, "close_time"],
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "r_multiple_gross": gross,
        "r_multiple": gross - fee_r,
    }


def make_trade_row(
    df: pd.DataFrame,
    *,
    symbol: str,
    cfg: OrbConfig,
    session_date: pd.Timestamp,
    or_start_idx: int,
    or_end_idx: int,
    session_end_idx: int,
    signal_idx: int,
    entry_idx: int,
    direction: str,
    entry: float,
    stop: float,
    target: float,
    or_high: float,
    or_low: float,
    first_break_side: int,
    sweep_side: int,
    sweep_depth_atr: float,
    fee_bps_per_side: float,
) -> dict[str, Any] | None:
    outcome = simulate_exit(
        df,
        entry_idx=entry_idx,
        direction=direction,
        entry=entry,
        stop=stop,
        target=target,
        max_hold_bars=cfg.max_hold_bars,
        session_end_idx=session_end_idx,
        fee_bps_per_side=fee_bps_per_side,
    )
    if outcome is None:
        return None
    atr = float(df.at[signal_idx, "atr"])
    or_width = or_high - or_low
    or_mid = (or_high + or_low) / 2.0
    sign = 1.0 if direction == "long" else -1.0
    signal_range = float(df.at[signal_idx, "high"] - df.at[signal_idx, "low"])
    signal_body = float(df.at[signal_idx, "close"] - df.at[signal_idx, "open"])
    prev_high = float(df.at[signal_idx, "prev_day_high"]) if pd.notna(df.at[signal_idx, "prev_day_high"]) else math.nan
    prev_low = float(df.at[signal_idx, "prev_day_low"]) if pd.notna(df.at[signal_idx, "prev_day_low"]) else math.nan
    risk = abs(entry - stop)
    row = {
        "symbol": symbol.upper(),
        "variant": cfg.variant,
        "family": cfg.family,
        "session": cfg.session,
        "session_date": session_date,
        "session_id": f"{session_date.date()}_{cfg.session}",
        "direction": direction,
        "direction_long": 1.0 if direction == "long" else 0.0,
        "signal_index": signal_idx,
        "entry_index": entry_idx,
        "signal_time": df.at[signal_idx, "close_time"],
        "entry_time": df.at[entry_idx, "open_time"],
        "entry_price": entry,
        "stop_price": stop,
        "target_price": target,
        "win_label": 1 if outcome["r_multiple"] > 0 else 0,
        "or_minutes": float(cfg.or_minutes),
        "rr": float(cfg.rr),
        "max_hold_bars": float(cfg.max_hold_bars),
        "min_break_atr": float(cfg.min_break_atr),
        "retest_tolerance_atr": float(cfg.retest_tolerance_atr),
        "min_sweep_atr": float(cfg.min_sweep_atr),
        "or_width_atr": safe_div(or_width, atr),
        "or_width_pct": safe_div(or_width, float(df.at[signal_idx, "close"])) * 100.0,
        "or_close_pos": safe_div(float(df.at[signal_idx, "close"]) - or_low, or_width),
        "entry_vs_or_mid_atr": sign * safe_div(entry - or_mid, atr),
        "entry_risk_atr": safe_div(risk, atr),
        "target_distance_atr": safe_div(abs(target - entry), atr),
        "minutes_after_or": float((signal_idx - or_end_idx + 1) * 5),
        "first_break_side_aligned": 1.0 if (first_break_side > 0 and direction == "long") or (first_break_side < 0 and direction == "short") else 0.0,
        "sweep_side_aligned": 1.0 if (sweep_side < 0 and direction == "long") or (sweep_side > 0 and direction == "short") else 0.0,
        "sweep_depth_atr": sweep_depth_atr,
        "signal_range_atr": safe_div(signal_range, atr),
        "signal_body_atr_dir": sign * safe_div(signal_body, atr),
        "signal_vol_mult": safe_div(float(df.at[signal_idx, "volume"]), float(df.at[signal_idx, "vol_sma20"])),
        "ret_1h_dir": sign * float(df.at[signal_idx, "ret_1h"]) if pd.notna(df.at[signal_idx, "ret_1h"]) else math.nan,
        "ret_4h_dir": sign * float(df.at[signal_idx, "ret_4h"]) if pd.notna(df.at[signal_idx, "ret_4h"]) else math.nan,
        "ret_24h_dir": sign * float(df.at[signal_idx, "ret_24h"]) if pd.notna(df.at[signal_idx, "ret_24h"]) else math.nan,
        "h1_trend_aligned": sign * float(df.at[signal_idx, "h1_close_vs_ema50_pct"]) if "h1_close_vs_ema50_pct" in df and pd.notna(df.at[signal_idx, "h1_close_vs_ema50_pct"]) else math.nan,
        "h4_trend_aligned": sign * float(df.at[signal_idx, "h4_close_vs_ema50_pct"]) if "h4_close_vs_ema50_pct" in df and pd.notna(df.at[signal_idx, "h4_close_vs_ema50_pct"]) else math.nan,
        "d1_trend_aligned": sign * float(df.at[signal_idx, "d1_close_vs_ema50_pct"]) if "d1_close_vs_ema50_pct" in df and pd.notna(df.at[signal_idx, "d1_close_vs_ema50_pct"]) else math.nan,
        "close_vs_ema20_atr_dir": sign * safe_div(float(df.at[signal_idx, "close"] - df.at[signal_idx, "ema20"]), atr),
        "close_vs_ema200_atr_dir": sign * safe_div(float(df.at[signal_idx, "close"] - df.at[signal_idx, "ema200"]), atr),
        "close_vs_daily_vwap_atr_dir": sign * safe_div(float(df.at[signal_idx, "close"] - df.at[signal_idx, "daily_vwap"]), atr),
        "prev_day_same_gap_atr": safe_div(abs((prev_low if direction == "long" else prev_high) - entry), atr) if math.isfinite(prev_low if direction == "long" else prev_high) else math.nan,
        "prev_day_opp_gap_atr": safe_div(abs((prev_high if direction == "long" else prev_low) - entry), atr) if math.isfinite(prev_high if direction == "long" else prev_low) else math.nan,
        "same_side_fvg_dist_atr": float(df.at[signal_idx, "close_to_bull_fvg_atr"]) if direction == "long" else float(df.at[signal_idx, "close_to_bear_fvg_atr"]),
        "opp_side_fvg_dist_atr": float(df.at[signal_idx, "close_to_bear_fvg_atr"]) if direction == "long" else float(df.at[signal_idx, "close_to_bull_fvg_atr"]),
        "same_side_fvg_width_atr": float(df.at[signal_idx, "bull_fvg_width_atr"]) if direction == "long" else float(df.at[signal_idx, "bear_fvg_width_atr"]),
        "same_side_fvg_age": float(df.at[signal_idx, "bull_fvg_age"]) if direction == "long" else float(df.at[signal_idx, "bear_fvg_age"]),
        "session_asia": 1.0 if cfg.session == "asia" else 0.0,
        "session_london": 1.0 if cfg.session == "london" else 0.0,
        "session_ny": 1.0 if cfg.session == "ny" else 0.0,
        "family_breakout": 1.0 if cfg.family == "breakout" else 0.0,
        "family_retest": 1.0 if cfg.family == "retest" else 0.0,
        "family_judas": 1.0 if cfg.family == "judas" else 0.0,
        "mode_fvg_retest": 1.0 if cfg.entry_mode == "fvg_retest" else 0.0,
        "mode_level_retest": 1.0 if cfg.entry_mode == "level_retest" else 0.0,
        **outcome,
    }
    return row


def first_break(trade_slice: pd.DataFrame, or_high: float, or_low: float) -> tuple[int | None, int]:
    for idx, row in trade_slice.iterrows():
        up = float(row["high"]) > or_high
        down = float(row["low"]) < or_low
        if up and down:
            return idx, 1 if float(row["close"]) >= float(row["open"]) else -1
        if up:
            return idx, 1
        if down:
            return idx, -1
    return None, 0


def direction_prices(
    *,
    direction: str,
    entry: float,
    rr: float,
    atr: float,
    signal_high: float,
    signal_low: float,
    or_high: float,
    or_low: float,
    stop_mode: str,
    stop_buffer_atr: float,
    sweep_low: float | None = None,
    sweep_high: float | None = None,
) -> tuple[float, float] | None:
    or_mid = (or_high + or_low) / 2.0
    buffer = stop_buffer_atr * atr
    if direction == "long":
        if stop_mode == "or_mid":
            stop = or_mid - buffer
        elif stop_mode == "or_opposite":
            stop = or_low - buffer
        elif stop_mode == "sweep":
            stop = (sweep_low if sweep_low is not None else signal_low) - buffer
        else:
            stop = signal_low - buffer
        risk = entry - stop
        target = entry + rr * risk
    else:
        if stop_mode == "or_mid":
            stop = or_mid + buffer
        elif stop_mode == "or_opposite":
            stop = or_high + buffer
        elif stop_mode == "sweep":
            stop = (sweep_high if sweep_high is not None else signal_high) + buffer
        else:
            stop = signal_high + buffer
        risk = stop - entry
        target = entry - rr * risk
    if risk <= 0 or risk / entry > 0.08:
        return None
    return stop, target


def generate_trades_for_config(
    df: pd.DataFrame,
    *,
    symbol: str,
    cfg: OrbConfig,
    fee_bps_per_side: float,
    day_groups: list[tuple[pd.Timestamp, pd.DataFrame]] | None = None,
) -> list[dict[str, Any]]:
    spec = SESSION_SPECS[cfg.session]
    start_min = spec["start_min"]
    or_end_min = start_min + cfg.or_minutes
    session_end_min = spec["end_min"]
    rows: list[dict[str, Any]] = []
    grouped = day_groups if day_groups is not None else list(df.groupby("session_date", sort=True))
    for session_date, day in grouped:
        or_slice = day[(day["minute_of_day"] >= start_min) & (day["minute_of_day"] < or_end_min)]
        trade_slice = day[(day["minute_of_day"] >= or_end_min) & (day["minute_of_day"] < session_end_min)]
        if len(or_slice) < max(2, cfg.or_minutes // 5 - 1) or trade_slice.empty:
            continue
        or_high = float(or_slice["high"].max())
        or_low = float(or_slice["low"].min())
        or_width = or_high - or_low
        or_end_idx = int(or_slice.index[-1])
        or_start_idx = int(or_slice.index[0])
        session_end_idx = int(trade_slice.index[-1])
        atr_ref = float(df.at[or_end_idx, "atr"])
        if not math.isfinite(atr_ref) or atr_ref <= 0:
            continue
        or_width_atr = or_width / atr_ref
        if or_width_atr < cfg.min_or_width_atr or or_width_atr > cfg.max_or_width_atr:
            continue
        first_idx, first_side = first_break(trade_slice, or_high, or_low)

        if cfg.family == "breakout":
            for idx, bar in trade_slice.iterrows():
                atr = float(df.at[idx, "atr"])
                if not math.isfinite(atr) or atr <= 0:
                    continue
                close = float(bar["close"])
                direction = ""
                if close > or_high + cfg.min_break_atr * atr:
                    direction = "long"
                elif close < or_low - cfg.min_break_atr * atr:
                    direction = "short"
                if not direction:
                    continue
                entry_idx = idx + 1
                if entry_idx >= len(df):
                    break
                entry = float(df.at[entry_idx, "open"])
                prices = direction_prices(
                    direction=direction,
                    entry=entry,
                    rr=cfg.rr,
                    atr=atr,
                    signal_high=float(bar["high"]),
                    signal_low=float(bar["low"]),
                    or_high=or_high,
                    or_low=or_low,
                    stop_mode=cfg.stop_mode,
                    stop_buffer_atr=cfg.stop_buffer_atr,
                )
                if prices is None:
                    break
                stop, target = prices
                row = make_trade_row(
                    df,
                    symbol=symbol,
                    cfg=cfg,
                    session_date=session_date,
                    or_start_idx=or_start_idx,
                    or_end_idx=or_end_idx,
                    session_end_idx=session_end_idx,
                    signal_idx=idx,
                    entry_idx=entry_idx,
                    direction=direction,
                    entry=entry,
                    stop=stop,
                    target=target,
                    or_high=or_high,
                    or_low=or_low,
                    first_break_side=first_side,
                    sweep_side=0,
                    sweep_depth_atr=0.0,
                    fee_bps_per_side=fee_bps_per_side,
                )
                if row:
                    rows.append(row)
                break

        elif cfg.family == "retest":
            if first_idx is None:
                continue
            breakout_side = first_side
            boundary = or_high if breakout_side > 0 else or_low
            direction = "long" if breakout_side > 0 else "short"
            end_wait = min(session_end_idx, first_idx + cfg.retest_wait_bars)
            for idx in range(first_idx + 1, end_wait + 1):
                atr = float(df.at[idx, "atr"])
                if not math.isfinite(atr) or atr <= 0:
                    continue
                if direction == "long":
                    touched = float(df.at[idx, "low"]) <= boundary + cfg.retest_tolerance_atr * atr
                    accepted = touched and float(df.at[idx, "close"]) > boundary
                else:
                    touched = float(df.at[idx, "high"]) >= boundary - cfg.retest_tolerance_atr * atr
                    accepted = touched and float(df.at[idx, "close"]) < boundary
                if not accepted:
                    continue
                entry_idx = idx + 1
                if entry_idx >= len(df):
                    break
                entry = float(df.at[entry_idx, "open"])
                prices = direction_prices(
                    direction=direction,
                    entry=entry,
                    rr=cfg.rr,
                    atr=atr,
                    signal_high=float(df.at[idx, "high"]),
                    signal_low=float(df.at[idx, "low"]),
                    or_high=or_high,
                    or_low=or_low,
                    stop_mode=cfg.stop_mode,
                    stop_buffer_atr=cfg.stop_buffer_atr,
                )
                if prices is None:
                    break
                stop, target = prices
                row = make_trade_row(
                    df,
                    symbol=symbol,
                    cfg=cfg,
                    session_date=session_date,
                    or_start_idx=or_start_idx,
                    or_end_idx=or_end_idx,
                    session_end_idx=session_end_idx,
                    signal_idx=idx,
                    entry_idx=entry_idx,
                    direction=direction,
                    entry=entry,
                    stop=stop,
                    target=target,
                    or_high=or_high,
                    or_low=or_low,
                    first_break_side=first_side,
                    sweep_side=0,
                    sweep_depth_atr=0.0,
                    fee_bps_per_side=fee_bps_per_side,
                )
                if row:
                    rows.append(row)
                break

        elif cfg.family == "judas":
            for idx, bar in trade_slice.iterrows():
                atr = float(df.at[idx, "atr"])
                if not math.isfinite(atr) or atr <= 0:
                    continue
                high = float(bar["high"])
                low = float(bar["low"])
                close = float(bar["close"])
                direction = ""
                sweep_side = 0
                sweep_depth_atr = 0.0
                if low < or_low - cfg.min_sweep_atr * atr and close > or_low:
                    direction = "long"
                    sweep_side = -1
                    sweep_depth_atr = (or_low - low) / atr
                elif high > or_high + cfg.min_sweep_atr * atr and close < or_high:
                    direction = "short"
                    sweep_side = 1
                    sweep_depth_atr = (high - or_high) / atr
                if not direction:
                    continue

                signal_idx = idx
                entry_idx = idx + 1
                if cfg.entry_mode == "level_retest":
                    boundary = or_low if direction == "long" else or_high
                    for j in range(idx + 1, min(session_end_idx, idx + cfg.retest_wait_bars) + 1):
                        j_atr = float(df.at[j, "atr"])
                        if direction == "long":
                            accepted = float(df.at[j, "low"]) <= boundary + cfg.retest_tolerance_atr * j_atr and float(df.at[j, "close"]) > boundary
                        else:
                            accepted = float(df.at[j, "high"]) >= boundary - cfg.retest_tolerance_atr * j_atr and float(df.at[j, "close"]) < boundary
                        if accepted:
                            signal_idx = j
                            entry_idx = j + 1
                            break
                    else:
                        continue
                elif cfg.entry_mode == "fvg_retest":
                    found = False
                    for j in range(idx + 1, min(session_end_idx, idx + cfg.retest_wait_bars) + 1):
                        if j < 2:
                            continue
                        if direction == "long" and float(df.at[j, "low"]) > float(df.at[j - 2, "high"]):
                            gap_bottom = float(df.at[j - 2, "high"])
                            gap_top = float(df.at[j, "low"])
                            for k in range(j + 1, min(session_end_idx, j + cfg.retest_wait_bars) + 1):
                                if float(df.at[k, "low"]) <= gap_top and float(df.at[k, "close"]) >= (gap_top + gap_bottom) / 2.0:
                                    signal_idx = k
                                    entry_idx = k + 1
                                    found = True
                                    break
                        elif direction == "short" and float(df.at[j, "high"]) < float(df.at[j - 2, "low"]):
                            gap_bottom = float(df.at[j, "high"])
                            gap_top = float(df.at[j - 2, "low"])
                            for k in range(j + 1, min(session_end_idx, j + cfg.retest_wait_bars) + 1):
                                if float(df.at[k, "high"]) >= gap_bottom and float(df.at[k, "close"]) <= (gap_top + gap_bottom) / 2.0:
                                    signal_idx = k
                                    entry_idx = k + 1
                                    found = True
                                    break
                        if found:
                            break
                    if not found:
                        continue

                if entry_idx >= len(df):
                    break
                entry = float(df.at[entry_idx, "open"])
                signal_atr = float(df.at[signal_idx, "atr"])
                prices = direction_prices(
                    direction=direction,
                    entry=entry,
                    rr=cfg.rr,
                    atr=signal_atr,
                    signal_high=float(df.at[signal_idx, "high"]),
                    signal_low=float(df.at[signal_idx, "low"]),
                    or_high=or_high,
                    or_low=or_low,
                    stop_mode=cfg.stop_mode,
                    stop_buffer_atr=cfg.stop_buffer_atr,
                    sweep_low=low,
                    sweep_high=high,
                )
                if prices is None:
                    break
                stop, target = prices
                row = make_trade_row(
                    df,
                    symbol=symbol,
                    cfg=cfg,
                    session_date=session_date,
                    or_start_idx=or_start_idx,
                    or_end_idx=or_end_idx,
                    session_end_idx=session_end_idx,
                    signal_idx=signal_idx,
                    entry_idx=entry_idx,
                    direction=direction,
                    entry=entry,
                    stop=stop,
                    target=target,
                    or_high=or_high,
                    or_low=or_low,
                    first_break_side=first_side,
                    sweep_side=sweep_side,
                    sweep_depth_atr=sweep_depth_atr,
                    fee_bps_per_side=fee_bps_per_side,
                )
                if row:
                    rows.append(row)
                break
    return rows


def build_grid(args: argparse.Namespace) -> list[OrbConfig]:
    sessions = [x.strip() for x in args.sessions.split(",") if x.strip()]
    or_minutes = [int(x) for x in args.or_minutes.split(",") if x.strip()]
    fast = args.grid_mode == "fast"
    configs: list[OrbConfig] = []
    for session in sessions:
        for or_min in or_minutes:
            breakout_rrs = [1.5] if fast else [1.0, 1.5]
            breakout_holds = [48] if fast else [24, 48]
            breakout_stops = ["signal", "or_mid"]
            for rr in breakout_rrs:
                for hold in breakout_holds:
                    for stop in breakout_stops:
                        configs.append(
                            OrbConfig(
                                family="breakout",
                                session=session,
                                or_minutes=or_min,
                                rr=rr,
                                max_hold_bars=hold,
                                stop_mode=stop,
                                min_break_atr=0.05,
                            )
                        )
            retest_rrs = [1.5, 2.0]
            retest_holds = [48] if fast else [48, 72]
            retest_tols = [0.15] if fast else [0.10, 0.25]
            for rr in retest_rrs:
                for hold in retest_holds:
                    for tol in retest_tols:
                        configs.append(
                            OrbConfig(
                                family="retest",
                                session=session,
                                or_minutes=or_min,
                                rr=rr,
                                max_hold_bars=hold,
                                stop_mode="signal",
                                min_break_atr=0.05,
                                retest_tolerance_atr=tol,
                                retest_wait_bars=18,
                            )
                        )
            judas_rrs = [1.0, 1.5, 2.0]
            judas_holds = [48] if fast else [24, 48]
            judas_sweeps = [0.15] if fast else [0.05, 0.15, 0.30]
            judas_modes = ["immediate", "level_retest", "fvg_retest"]
            for rr in judas_rrs:
                for hold in judas_holds:
                    for sweep in judas_sweeps:
                        for mode in judas_modes:
                            configs.append(
                                OrbConfig(
                                    family="judas",
                                    session=session,
                                    or_minutes=or_min,
                                    rr=rr,
                                    max_hold_bars=hold,
                                    stop_mode="sweep",
                                    min_sweep_atr=sweep,
                                    entry_mode=mode,
                                    retest_tolerance_atr=0.15,
                                    retest_wait_bars=18,
                                )
                            )
    return configs


def score_configs(
    df: pd.DataFrame,
    *,
    symbol: str,
    configs: list[OrbConfig],
    split: pd.Timestamp,
    fee_bps_per_side: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = len(configs)
    day_groups = list(df.groupby("session_date", sort=True))
    for pos, cfg in enumerate(configs, 1):
        trades = pd.DataFrame(
            generate_trades_for_config(
                df,
                symbol=symbol,
                cfg=cfg,
                fee_bps_per_side=fee_bps_per_side,
                day_groups=day_groups,
            )
        )
        if trades.empty:
            train_m = metrics(trades)
            oos_m = metrics(trades)
        else:
            trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce")
            train_m = metrics(trades[trades["entry_time"] < split])
            oos_m = metrics(trades[trades["entry_time"] >= split])
        rows.append(
            {
                **asdict(cfg),
                "variant": cfg.variant,
                **{f"train_{k}": v for k, v in train_m.items()},
                **{f"oos_{k}": v for k, v in oos_m.items()},
            }
        )
        if pos % 50 == 0:
            print(f"  scored {pos}/{total} configs", flush=True)
    summary = pd.DataFrame(rows)
    summary["train_score"] = (
        summary["train_net_r"].astype(float)
        + 8.0 * np.log1p(summary["train_profit_factor"].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float))
        - 0.10 * summary["train_trades"].astype(float).clip(lower=0)
        + 0.30 * summary["train_max_dd_r"].astype(float)
    )
    return summary


def feature_columns(frame: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in frame.columns:
        if column in META_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            columns.append(column)
    return columns


def train_ml_ranker(
    trades: pd.DataFrame,
    *,
    split: pd.Timestamp,
    thresholds: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if not SKLEARN_AVAILABLE or trades.empty:
        return pd.DataFrame(), trades, []
    data = trades.copy()
    data["entry_time"] = pd.to_datetime(data["entry_time"], utc=True, errors="coerce")
    train = data[data["entry_time"] < split].copy()
    if len(train) < 200 or train["win_label"].nunique() < 2:
        return pd.DataFrame(), data, []
    cols = [c for c in feature_columns(train) if train[c].notna().any()]
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=400,
            max_depth=5,
            min_samples_leaf=50,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
    )
    model.fit(train[cols].astype(float), train["win_label"].astype(int))
    data["ml_prob"] = model.predict_proba(data[cols].astype(float))[:, 1]
    try:
        auc = float(roc_auc_score(data[data["entry_time"] >= split]["win_label"], data[data["entry_time"] >= split]["ml_prob"]))
    except Exception:
        auc = math.nan

    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        selected_parts = []
        for _, group in data[data["ml_prob"] >= threshold].groupby("session_id", sort=True):
            selected_parts.append(group.sort_values("ml_prob", ascending=False).head(1))
        selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame(columns=data.columns)
        train_sel = selected[selected["entry_time"] < split].copy()
        oos_sel = selected[selected["entry_time"] >= split].copy()
        rows.append(
            {
                "threshold": threshold,
                "oos_auc": round(auc, 3) if math.isfinite(auc) else math.nan,
                **{f"train_{k}": v for k, v in metrics(train_sel).items()},
                **{f"oos_{k}": v for k, v in metrics(oos_sel).items()},
            }
        )
    table = pd.DataFrame(rows)
    return table, data, cols


def parse_thresholds(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Research session ORB/Judas/FVG strategies on Bybit 5m crypto data.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--cache-dir", type=Path, default=Path("scripts/.cache/bybit_linear"))
    parser.add_argument("--split", default="2025-04-20")
    parser.add_argument("--train-start", default="2022-04-20")
    parser.add_argument("--end", default="2026-04-20")
    parser.add_argument("--sessions", default="asia,london,ny")
    parser.add_argument("--or-minutes", default="30,60,90")
    parser.add_argument("--grid-mode", choices=["fast", "full"], default="fast")
    parser.add_argument("--fee-bps-per-side", type=float, default=6.5)
    parser.add_argument("--top-train-variants", type=int, default=24)
    parser.add_argument("--thresholds", default="0.45,0.50,0.55,0.60,0.65,0.70")
    parser.add_argument("--output-prefix", type=Path, default=Path("scripts/session_orb_btc"))
    args = parser.parse_args()

    symbol = args.symbol.upper()
    split = pd.Timestamp(parse_utc_datetime(args.split))
    train_start = pd.Timestamp(parse_utc_datetime(args.train_start))
    end = pd.Timestamp(parse_utc_datetime(args.end))
    thresholds = parse_thresholds(args.thresholds)

    print(f"{symbol}: loading cached 5m data ...", flush=True)
    raw = load_cached_symbol(symbol, args.cache_dir)
    raw["open_time"] = pd.to_datetime(raw["open_time"], utc=True)
    raw["close_time"] = pd.to_datetime(raw["close_time"], utc=True)
    raw = raw[(raw["open_time"] >= train_start - pd.Timedelta(days=90)) & (raw["open_time"] < end)].copy()
    print(f"{symbol}: preparing indicators on {len(raw):,} bars ...", flush=True)
    df = add_htf_context(raw)
    df = df[(df["open_time"] >= train_start) & (df["open_time"] < end)].reset_index(drop=True)
    print(f"{symbol}: experiment bars {len(df):,} from {df['open_time'].iloc[0]} to {df['open_time'].iloc[-1]}", flush=True)

    configs = build_grid(args)
    print(f"{symbol}: scoring {len(configs)} ORB configs ...", flush=True)
    summary = score_configs(df, symbol=symbol, configs=configs, split=split, fee_bps_per_side=args.fee_bps_per_side)
    summary_path = args.output_prefix.with_name(args.output_prefix.name + "_summary.csv")
    summary.sort_values(["train_score", "train_profit_factor", "train_net_r"], ascending=[False, False, False]).to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}", flush=True)

    eligible = summary[(summary["train_trades"] >= 80) & (summary["oos_trades"] >= 15)].copy()
    if eligible.empty:
        eligible = summary[summary["train_trades"] >= 40].copy()
    top = eligible.sort_values(["train_score", "train_profit_factor", "train_net_r"], ascending=[False, False, False]).head(args.top_train_variants)
    print("\nTop train-selected variants with OOS validation:", flush=True)
    cols = ["variant", "train_trades", "train_profit_factor", "train_net_r", "train_max_dd_r", "oos_trades", "oos_profit_factor", "oos_net_r", "oos_max_dd_r"]
    print(top[cols].head(12).to_string(index=False), flush=True)

    all_top_rows: list[dict[str, Any]] = []
    variant_to_cfg = {cfg.variant: cfg for cfg in configs}
    day_groups = list(df.groupby("session_date", sort=True))
    for variant in top["variant"].tolist():
        all_top_rows.extend(
            generate_trades_for_config(
                df,
                symbol=symbol,
                cfg=variant_to_cfg[variant],
                fee_bps_per_side=args.fee_bps_per_side,
                day_groups=day_groups,
            )
        )
    top_trades = pd.DataFrame(all_top_rows)
    trades_path = args.output_prefix.with_name(args.output_prefix.name + "_top_variant_trades.csv")
    top_trades.to_csv(trades_path, index=False)
    print(f"Wrote {trades_path}", flush=True)

    ml_table, scored, ml_cols = train_ml_ranker(top_trades, split=split, thresholds=thresholds)
    ml_path = args.output_prefix.with_name(args.output_prefix.name + "_ml_thresholds.csv")
    scored_path = args.output_prefix.with_name(args.output_prefix.name + "_ml_scored_candidates.csv")
    ml_table.to_csv(ml_path, index=False)
    scored.to_csv(scored_path, index=False)
    print(f"Wrote {ml_path}", flush=True)
    print(f"Wrote {scored_path}", flush=True)
    if not ml_table.empty:
        print("\nML ranker threshold table:", flush=True)
        print(ml_table.to_string(index=False), flush=True)
        train_choices = ml_table[ml_table["train_trades"] >= 80].copy()
        if train_choices.empty:
            train_choices = ml_table.copy()
        chosen = train_choices.sort_values(["train_profit_factor", "train_net_r", "train_trades"], ascending=[False, False, False]).iloc[0]
        print("\nTrain-chosen ML threshold:", flush=True)
        print(chosen.to_string(), flush=True)
        print(f"\nML feature count: {len(ml_cols)}", flush=True)


if __name__ == "__main__":
    main()
