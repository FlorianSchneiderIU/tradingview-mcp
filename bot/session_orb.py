from __future__ import annotations

import logging
import math
import os
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from scripts.experiment_session_orb import OrbConfig, add_htf_context
from scripts.experiment_session_orb_fast import (
    Arrays,
    SessionContext,
    build_contexts,
    first_true,
    stop_target,
    to_arrays,
)


log = logging.getLogger("mm")

SESSION_ORB_INTERVAL = "5m"
SESSION_ORB_BYBIT_INTERVAL = "5"
DEFAULT_SESSION_ORB_SYMBOLS = "ETHUSDT,WIFUSDT,NEARUSDT,ENAUSDT,OPUSDT,ONDOUSDT"


def _feature_snapshot(row: pd.Series, columns: list[str]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for column in columns:
        try:
            value = float(row.get(column, math.nan))
        except (TypeError, ValueError):
            value = math.nan
        out[column] = value if math.isfinite(value) else None
    return out


def bars_to_frame(bars: list[dict], interval: str = SESSION_ORB_INTERVAL) -> pd.DataFrame:
    interval_ms = 5 * 60 * 1000 if interval == "5m" else 60 * 1000
    frame = pd.DataFrame(bars)
    if frame.empty:
        return pd.DataFrame(columns=["open_time", "close_time", "open", "high", "low", "close", "volume"])
    frame = frame.drop_duplicates("ts", keep="last").sort_values("ts").reset_index(drop=True)
    frame["open_time"] = pd.to_datetime(frame["ts"].astype("int64"), unit="ms", utc=True)
    frame["close_time"] = frame["open_time"] + pd.Timedelta(milliseconds=interval_ms - 1)
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
    return frame[["open_time", "close_time", "open", "high", "low", "close", "volume"]]


@dataclass
class SessionOrbModel:
    symbol: str
    model: object
    feature_columns: list[str]
    threshold: float
    configs: list[OrbConfig]


class SessionOrbState:
    def __init__(self, symbol: str, model: SessionOrbModel, max_bars: int):
        self.symbol = symbol
        self.model = model
        self.bars: deque = deque(maxlen=max_bars)
        self.last_entry_time: Optional[pd.Timestamp] = None
        self.last_signal_key: Optional[str] = None
        self._lock = threading.Lock()

    def push_bar(self, bar: dict) -> None:
        with self._lock:
            if self.bars and self.bars[-1]["ts"] == bar["ts"]:
                self.bars[-1] = bar
            else:
                self.bars.append(bar)

    def snapshot(self) -> list[dict]:
        with self._lock:
            return list(self.bars)


def load_session_orb_models(
    *,
    symbols: list[str],
    models_dir: str,
    threshold_override: float | None = None,
) -> dict[str, SessionOrbModel]:
    out: dict[str, SessionOrbModel] = {}
    root = Path(models_dir)
    for symbol in symbols:
        symbol = symbol.upper()
        path = root / f"{symbol.lower()}_session_orb.joblib"
        if not path.exists():
            log.warning(f"[session_orb] {symbol}: model not found at {path}; disabled")
            continue
        try:
            payload = joblib.load(path)
            configs = [OrbConfig(**cfg) for cfg in payload["selected_configs"]]
            threshold = float(threshold_override if threshold_override is not None else payload["threshold"])
            out[symbol] = SessionOrbModel(
                symbol=symbol,
                model=payload["model"],
                feature_columns=list(payload["feature_columns"]),
                threshold=threshold,
                configs=configs,
            )
            log.info(
                f"[session_orb] {symbol}: model loaded threshold={threshold:.2f} "
                f"features={len(payload['feature_columns'])} configs={len(configs)}"
            )
        except Exception as exc:
            log.error(f"[session_orb] {symbol}: failed to load model {path}: {exc}")
    return out


def _finite(value: float) -> bool:
    return math.isfinite(float(value))


def _make_live_row(
    a: Arrays,
    *,
    symbol: str,
    cfg: OrbConfig,
    ctx: SessionContext,
    signal_idx: int,
    direction: str,
    entry: float,
    stop: float,
    target: float,
    sweep_side: int,
    sweep_depth_atr: float,
    fvg_bottom: float,
    fvg_top: float,
) -> dict:
    atr = float(a.atr[signal_idx])
    sign = 1.0 if direction == "long" else -1.0
    or_width = ctx.or_high - ctx.or_low
    or_mid = (ctx.or_high + ctx.or_low) / 2.0
    signal_range = a.high[signal_idx] - a.low[signal_idx]
    signal_body = a.close[signal_idx] - a.open[signal_idx]
    same_fvg_mid = a.bull_fvg_mid[signal_idx] if direction == "long" else a.bear_fvg_mid[signal_idx]
    opp_fvg_mid = a.bear_fvg_mid[signal_idx] if direction == "long" else a.bull_fvg_mid[signal_idx]
    same_fvg_width = a.bull_fvg_width_atr[signal_idx] if direction == "long" else a.bear_fvg_width_atr[signal_idx]
    same_fvg_age = a.bull_fvg_age[signal_idx] if direction == "long" else a.bear_fvg_age[signal_idx]
    prev_same = a.prev_day_low[signal_idx] if direction == "long" else a.prev_day_high[signal_idx]
    prev_opp = a.prev_day_high[signal_idx] if direction == "long" else a.prev_day_low[signal_idx]
    return {
        "symbol": symbol.upper(),
        "variant": cfg.variant,
        "family": cfg.family,
        "session": cfg.session,
        "session_date": ctx.session_date,
        "session_id": ctx.session_id,
        "direction": direction,
        "direction_long": 1.0 if direction == "long" else 0.0,
        "signal_index": signal_idx,
        "entry_index": signal_idx,
        "signal_time": a.close_time[signal_idx],
        "entry_time": a.close_time[signal_idx],
        "entry_price": entry,
        "stop_price": stop,
        "target_price": target,
        "or_minutes": float(cfg.or_minutes),
        "rr": float(cfg.rr),
        "max_hold_bars": float(cfg.max_hold_bars),
        "min_break_atr": float(cfg.min_break_atr),
        "retest_tolerance_atr": float(cfg.retest_tolerance_atr),
        "min_sweep_atr": float(cfg.min_sweep_atr),
        "or_width_atr": ctx.or_width_atr,
        "or_width_pct": (or_width / a.close[signal_idx]) * 100.0,
        "or_close_pos": (a.close[signal_idx] - ctx.or_low) / or_width if or_width > 0 else math.nan,
        "entry_vs_or_mid_atr": sign * (entry - or_mid) / atr,
        "entry_risk_atr": abs(entry - stop) / atr,
        "target_distance_atr": abs(target - entry) / atr,
        "minutes_after_or": float((signal_idx - ctx.or_end_idx + 1) * 5),
        "first_break_side_aligned": 1.0 if (ctx.first_break_side > 0 and direction == "long") or (ctx.first_break_side < 0 and direction == "short") else 0.0,
        "sweep_side_aligned": 1.0 if (sweep_side < 0 and direction == "long") or (sweep_side > 0 and direction == "short") else 0.0,
        "sweep_depth_atr": sweep_depth_atr,
        "signal_range_atr": signal_range / atr if atr > 0 else math.nan,
        "signal_body_atr_dir": sign * signal_body / atr if atr > 0 else math.nan,
        "signal_vol_mult": a.volume[signal_idx] / a.vol_sma20[signal_idx] if a.vol_sma20[signal_idx] > 0 else math.nan,
        "ret_1h_dir": sign * a.ret_1h[signal_idx],
        "ret_4h_dir": sign * a.ret_4h[signal_idx],
        "ret_24h_dir": sign * a.ret_24h[signal_idx],
        "h1_trend_aligned": sign * a.h1_trend[signal_idx],
        "h4_trend_aligned": sign * a.h4_trend[signal_idx],
        "d1_trend_aligned": sign * a.d1_trend[signal_idx],
        "close_vs_ema20_atr_dir": sign * (a.close[signal_idx] - a.ema20[signal_idx]) / atr,
        "close_vs_ema200_atr_dir": sign * (a.close[signal_idx] - a.ema200[signal_idx]) / atr,
        "close_vs_daily_vwap_atr_dir": sign * (a.close[signal_idx] - a.daily_vwap[signal_idx]) / atr,
        "prev_day_same_gap_atr": abs(prev_same - entry) / atr if _finite(prev_same) else math.nan,
        "prev_day_opp_gap_atr": abs(prev_opp - entry) / atr if _finite(prev_opp) else math.nan,
        "same_side_fvg_dist_atr": sign * (a.close[signal_idx] - same_fvg_mid) / atr if _finite(same_fvg_mid) else math.nan,
        "opp_side_fvg_dist_atr": -sign * (a.close[signal_idx] - opp_fvg_mid) / atr if _finite(opp_fvg_mid) else math.nan,
        "same_side_fvg_width_atr": same_fvg_width,
        "same_side_fvg_age": same_fvg_age,
        "session_asia": 1.0 if cfg.session == "asia" else 0.0,
        "session_london": 1.0 if cfg.session == "london" else 0.0,
        "session_ny": 1.0 if cfg.session == "ny" else 0.0,
        "family_breakout": 1.0 if cfg.family == "breakout" else 0.0,
        "family_retest": 1.0 if cfg.family == "retest" else 0.0,
        "family_judas": 1.0 if cfg.family == "judas" else 0.0,
        "mode_fvg_retest": 1.0 if cfg.entry_mode == "fvg_retest" else 0.0,
        "mode_level_retest": 1.0 if cfg.entry_mode == "level_retest" else 0.0,
        "fvg_bottom": fvg_bottom,
        "fvg_top": fvg_top,
    }


def _live_trade_for_context(
    a: Arrays,
    *,
    symbol: str,
    cfg: OrbConfig,
    ctx: SessionContext,
    last_idx: int,
) -> dict | None:
    if cfg.family != "judas" or cfg.entry_mode != "fvg_retest":
        return None
    if ctx.session != cfg.session or ctx.or_minutes != cfg.or_minutes:
        return None
    if ctx.or_width_atr < cfg.min_or_width_atr or ctx.or_width_atr > cfg.max_or_width_atr:
        return None
    if not (ctx.trade_start_idx <= last_idx <= ctx.trade_end_idx):
        return None

    idxs = np.arange(ctx.trade_start_idx, last_idx + 1)
    long_mask = (a.low[idxs] < ctx.or_low - cfg.min_sweep_atr * a.atr[idxs]) & (a.close[idxs] > ctx.or_low)
    short_mask = (a.high[idxs] > ctx.or_high + cfg.min_sweep_atr * a.atr[idxs]) & (a.close[idxs] < ctx.or_high)
    long_first = first_true(long_mask)
    short_first = first_true(short_mask)
    if long_first is None and short_first is None:
        return None
    if long_first is not None and (short_first is None or long_first <= short_first):
        sweep_idx = int(idxs[long_first])
        direction = "long"
        sweep_side = -1
        sweep_depth = (ctx.or_low - a.low[sweep_idx]) / a.atr[sweep_idx]
    else:
        sweep_idx = int(idxs[short_first])
        direction = "short"
        sweep_side = 1
        sweep_depth = (a.high[sweep_idx] - ctx.or_high) / a.atr[sweep_idx]

    found = False
    signal_idx = sweep_idx
    fvg_bottom = math.nan
    fvg_top = math.nan
    for j in range(sweep_idx + 1, min(last_idx, sweep_idx + cfg.retest_wait_bars) + 1):
        if direction == "long" and math.isfinite(a.bull_fvg_top[j]):
            gap_mid = (a.bull_fvg_top[j] + a.bull_fvg_bottom[j]) / 2.0
            for k in range(j + 1, min(last_idx, j + cfg.retest_wait_bars) + 1):
                if a.low[k] <= a.bull_fvg_top[j] and a.close[k] >= gap_mid:
                    signal_idx = k
                    fvg_bottom = float(a.bull_fvg_bottom[j])
                    fvg_top = float(a.bull_fvg_top[j])
                    found = True
                    break
        elif direction == "short" and math.isfinite(a.bear_fvg_bottom[j]):
            gap_mid = (a.bear_fvg_top[j] + a.bear_fvg_bottom[j]) / 2.0
            for k in range(j + 1, min(last_idx, j + cfg.retest_wait_bars) + 1):
                if a.high[k] >= a.bear_fvg_bottom[j] and a.close[k] <= gap_mid:
                    signal_idx = k
                    fvg_bottom = float(a.bear_fvg_bottom[j])
                    fvg_top = float(a.bear_fvg_top[j])
                    found = True
                    break
        if found:
            break
    if not found or signal_idx != last_idx:
        return None

    entry = float(a.close[signal_idx])
    prices = stop_target(
        a,
        ctx,
        cfg,
        direction=direction,
        signal_idx=signal_idx,
        entry=entry,
        sweep_low=float(a.low[sweep_idx]),
        sweep_high=float(a.high[sweep_idx]),
    )
    if prices is None:
        return None
    stop, target = prices
    return _make_live_row(
        a,
        symbol=symbol,
        cfg=cfg,
        ctx=ctx,
        signal_idx=signal_idx,
        direction=direction,
        entry=entry,
        stop=stop,
        target=target,
        sweep_side=sweep_side,
        sweep_depth_atr=sweep_depth,
        fvg_bottom=fvg_bottom,
        fvg_top=fvg_top,
    )


class SessionOrbEngine:
    def __init__(self):
        self.min_bars = int(os.environ.get("SESSION_ORB_MIN_BARS", "1500"))

    def detect_signal(self, state: SessionOrbState) -> Optional[dict]:
        bars = state.snapshot()
        if len(bars) < self.min_bars:
            return None
        frame = bars_to_frame(bars, SESSION_ORB_INTERVAL)
        if frame.empty:
            return None
        try:
            enriched = add_htf_context(frame).reset_index(drop=True)
            arrays = to_arrays(enriched)
            sessions = sorted({cfg.session for cfg in state.model.configs})
            or_minutes = sorted({int(cfg.or_minutes) for cfg in state.model.configs})
            contexts = build_contexts(enriched, arrays, sessions=sessions, or_minutes=or_minutes)
            last_idx = len(enriched) - 1
            rows = []
            for cfg in state.model.configs:
                for ctx in contexts:
                    row = _live_trade_for_context(arrays, symbol=state.symbol, cfg=cfg, ctx=ctx, last_idx=last_idx)
                    if row is not None:
                        rows.append(row)
            if not rows:
                return None
            candidates = pd.DataFrame(rows)
            if state.last_entry_time is not None:
                candidates = candidates[pd.to_datetime(candidates["entry_time"], utc=True) > state.last_entry_time]
            if candidates.empty:
                return None
            for column in state.model.feature_columns:
                if column not in candidates.columns:
                    candidates[column] = math.nan
            candidates["ml_prob"] = state.model.model.predict_proba(
                candidates[state.model.feature_columns].astype(float)
            )[:, 1]
            row = candidates.sort_values("ml_prob", ascending=False).iloc[0]
        except Exception as exc:
            log.warning(f"[session_orb] {state.symbol}: signal evaluation failed: {exc}")
            return None

        entry_time = pd.Timestamp(row["entry_time"])
        direction = str(row["direction"])
        signal_key = f"{state.symbol}|{entry_time.isoformat()}|{direction}|{float(row['entry_price']):.10g}|{row['variant']}"
        if state.last_signal_key == signal_key:
            return None
        state.last_signal_key = signal_key

        entry = float(row["entry_price"])
        stop = float(row["stop_price"])
        target = float(row["target_price"])
        prob = float(row["ml_prob"])
        risk = abs(entry - stop)
        if risk <= 0 or not np.isfinite(risk):
            return None

        sig = {
            "strategy": "session_orb_judas_fvg",
            "signal": direction,
            "entry": entry,
            "sl": stop,
            "tp1": target,
            "take_profit": target,
            "trail_dist": risk,
            "exit_style": "fixed_tp",
            "prob": prob,
            "threshold": state.model.threshold,
            "entry_time": entry_time.isoformat(),
            "session": str(row["session"]),
            "or_minutes": int(row["or_minutes"]),
            "variant": str(row["variant"]),
            "entry_risk_atr": float(row["entry_risk_atr"]),
            "sweep_depth_atr": float(row["sweep_depth_atr"]),
            "fvg_bottom": float(row["fvg_bottom"]),
            "fvg_top": float(row["fvg_top"]),
            "feature_columns": list(state.model.feature_columns),
            "feature_snapshot": _feature_snapshot(row, state.model.feature_columns),
        }
        if prob < state.model.threshold:
            sig["rejected"] = True
            sig["reject_reason"] = f"ML probability {prob:.3f} below threshold {state.model.threshold:.2f}"
            return sig

        if direction == "long" and (entry <= stop or entry >= target):
            return None
        if direction == "short" and (entry >= stop or entry <= target):
            return None
        state.last_entry_time = entry_time
        return sig
