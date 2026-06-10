from __future__ import annotations

import json
import logging
import math
import os
import threading
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Optional

import pandas as pd

from scripts.backtest_wolfe_wave import (
    WolfeConfig,
    bybit_symbol,
    ensure_ohlcv_frame,
    fee_aware_stop_price,
    fee_to_price_risk,
    find_wolfe_signals,
    normalize_timeframe,
)
from turtle_soup import fetch_warmup_bars_interval, parse_symbol_list


log = logging.getLogger("mm")

WOLFE_WAVE_INTERVAL = "5m"
WOLFE_WAVE_BYBIT_INTERVAL = "5"
DEFAULT_WOLFE_WAVE_SYMBOLS = (
    "BTCUSDT,ETHUSDT,LINKUSDT,LTCUSDT,SOLUSDT,UNIUSDT,XRPUSDT,"
    "1000PEPEUSDT,BNBUSDT,DOGEUSDT,STXUSDT"
)
DEFAULT_WOLFE_WAVE_CONFIG = WolfeConfig(
    exec_tf=WOLFE_WAVE_INTERVAL,
    pattern_tf="1h",
    pivot_method="zigzag",
    zigzag_atr_mult=1.4,
    max_time_ratio=3.0,
    max_p5_break_atr=2.2,
    stop_atr_buffer=0.5,
    min_rr=1.5,
    min_score=64.0,
    target_projection_bars=18,
    max_hold_bars=288,
    trend_filter="rsi",
    regime_filter="none",
)


def bars_to_frame(bars: list[dict], interval: str = WOLFE_WAVE_INTERVAL) -> pd.DataFrame:
    interval = normalize_timeframe(interval)
    interval_ms = {
        "1m": 60_000,
        "3m": 180_000,
        "5m": 300_000,
        "15m": 900_000,
        "1h": 3_600_000,
        "4h": 14_400_000,
        "1d": 86_400_000,
    }[interval]
    frame = pd.DataFrame(bars)
    if frame.empty:
        return pd.DataFrame(columns=["open_time", "close_time", "open", "high", "low", "close", "volume"])
    frame = frame.drop_duplicates("ts", keep="last").sort_values("ts").reset_index(drop=True)
    frame["open_time"] = pd.to_datetime(frame["ts"].astype("int64"), unit="ms", utc=True)
    frame["close_time"] = frame["open_time"] + pd.Timedelta(milliseconds=interval_ms - 1)
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
    return ensure_ohlcv_frame(frame[["open_time", "close_time", "open", "high", "low", "close", "volume"]])


def load_wolfe_wave_configs(
    *,
    symbols: list[str],
    config_path: str,
) -> dict[str, WolfeConfig]:
    raw: dict[str, object] = {}
    path = Path(config_path)
    loaded_from_file = False
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                raw = loaded
                loaded_from_file = True
        except Exception as exc:  # noqa: BLE001 - disable bad config entries but keep bot booting.
            log.error(f"[wolfe] Failed to load config {path}: {exc}")
    else:
        log.warning(f"[wolfe] Config not found at {path}; using built-in BTC defaults")

    out: dict[str, WolfeConfig] = {}
    for symbol in symbols:
        normalized = bybit_symbol(symbol)
        if loaded_from_file and normalized not in raw:
            log.warning(f"[wolfe] {normalized}: no config entry in {path}; skipped")
            continue
        payload = raw.get(normalized, {}) if isinstance(raw, dict) else {}
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            log.warning(f"[wolfe] {normalized}: config payload is not an object; skipped")
            continue
        try:
            cfg = WolfeConfig.from_mapping(
                {
                    **DEFAULT_WOLFE_WAVE_CONFIG.__dict__,
                    **payload,
                    "exec_tf": WOLFE_WAVE_INTERVAL,
                }
            )
        except Exception as exc:  # noqa: BLE001
            log.error(f"[wolfe] {normalized}: invalid config: {exc}")
            continue
        out[normalized] = cfg
        log.info(
            f"[wolfe] {normalized}: config loaded pattern_tf={cfg.pattern_tf} "
            f"pivots={cfg.pivot_method}/{cfg.pivot_source} min_score={cfg.min_score:.1f} "
            f"trend={cfg.trend_filter} regime={cfg.regime_filter} "
            f"fee_aware_stop={cfg.fee_aware_stop} max_fee_risk={cfg.max_fee_to_price_risk:.1%} "
            f"directions={'L' if cfg.allow_longs else '-'}{'S' if cfg.allow_shorts else '-'}"
        )
    return out


class WolfeWaveState:
    def __init__(self, symbol: str, config: WolfeConfig, max_bars: int):
        self.symbol = bybit_symbol(symbol)
        self.config = config
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


class WolfeWaveEngine:
    def __init__(self):
        self.min_bars = int(os.environ.get("WOLFE_WAVE_MIN_BARS", "3000"))

    @staticmethod
    def _signal_payload(
        signal,
        *,
        entry: float,
        stop: float,
        target: float,
        risk: float,
        threshold: float,
        target_rr_planned: float | None = None,
        fee_risk: float | None = None,
        entry_risk_pct: float | None = None,
        stop_adjusted_for_fee: bool | None = None,
    ) -> dict:
        planned_rr = float(signal.target_rr_planned if target_rr_planned is None else target_rr_planned)
        return {
            "strategy": "wolfe_wave",
            "signal": signal.direction,
            "entry": entry,
            "model_entry": float(signal.entry_price),
            "sl": stop,
            "model_sl": float(signal.stop_price),
            "structural_sl": float(signal.structural_stop_price),
            "tp1": target,
            "take_profit": target,
            "trail_dist": risk,
            "exit_style": "fixed_tp",
            "prob": signal.score / 100.0,
            "threshold": threshold / 100.0,
            "entry_time": signal.entry_time.isoformat(),
            "pattern_tf": signal.pattern_tf,
            "pivot_method": signal.pivot_method,
            "trend_context": signal.trend_context,
            "target_rr_planned": planned_rr,
            "fee_to_price_risk": None if fee_risk is None else float(fee_risk),
            "entry_risk_pct": None if entry_risk_pct is None else float(entry_risk_pct),
            "stop_adjusted_for_fee": bool(signal.stop_adjusted_for_fee if stop_adjusted_for_fee is None else stop_adjusted_for_fee),
            "score": float(signal.score),
            "p5_break_atr": float(signal.p5_break_atr),
            "symmetry_ratio": float(signal.symmetry_ratio),
            "epa_slope_atr": float(signal.epa_slope_atr),
            "volume_ratio": float(signal.volume_ratio),
            "rsi": float(signal.rsi),
            "p1_horizontal_hit": bool(signal.p1_horizontal_hit),
            "p1_horizontal_distance_bars": float(signal.p1_horizontal_distance_bars),
            "p1_horizontal_error_atr": float(signal.p1_horizontal_error_atr),
            "p1_horizontal_score": float(signal.p1_horizontal_score),
            "p4_contrary_pivots": int(signal.p4_contrary_pivots),
            "p4_contrary_swing_atr": float(signal.p4_contrary_swing_atr),
            "p4_contrary_score": float(signal.p4_contrary_score),
            "impulse_45_bars": int(signal.impulse_45_bars),
            "impulse_45_atr": float(signal.impulse_45_atr),
            "impulse_45_same_dir_ratio": float(signal.impulse_45_same_dir_ratio),
            "sweet_zone_width_atr": float(signal.sweet_zone_width_atr),
            "sweet_zone_expansion_atr_per_bar": float(signal.sweet_zone_expansion_atr_per_bar),
            "p5_volume_ratio": float(signal.p5_volume_ratio),
            "p5_rejection_atr": float(signal.p5_rejection_atr),
            "v2_quality": float(signal.v2_quality),
            "feature_columns": [
                "score",
                "target_rr_planned",
                "fee_to_price_risk",
                "entry_risk_pct",
                "stop_adjusted_for_fee",
                "p5_break_atr",
                "symmetry_ratio",
                "epa_slope_atr",
                "volume_ratio",
                "rsi",
                "p1_horizontal_hit",
                "p1_horizontal_distance_bars",
                "p1_horizontal_error_atr",
                "p1_horizontal_score",
                "p4_contrary_pivots",
                "p4_contrary_swing_atr",
                "p4_contrary_score",
                "impulse_45_bars",
                "impulse_45_atr",
                "impulse_45_same_dir_ratio",
                "sweet_zone_width_atr",
                "sweet_zone_expansion_atr_per_bar",
                "p5_volume_ratio",
                "p5_rejection_atr",
                "v2_quality",
            ],
            "feature_snapshot": {
                "score": float(signal.score),
                "target_rr_planned": planned_rr,
                "fee_to_price_risk": None if fee_risk is None else float(fee_risk),
                "entry_risk_pct": None if entry_risk_pct is None else float(entry_risk_pct),
                "stop_adjusted_for_fee": bool(signal.stop_adjusted_for_fee if stop_adjusted_for_fee is None else stop_adjusted_for_fee),
                "p5_break_atr": float(signal.p5_break_atr),
                "symmetry_ratio": float(signal.symmetry_ratio),
                "epa_slope_atr": float(signal.epa_slope_atr),
                "volume_ratio": float(signal.volume_ratio) if math.isfinite(float(signal.volume_ratio)) else None,
                "rsi": float(signal.rsi) if math.isfinite(float(signal.rsi)) else None,
                "p1_horizontal_hit": bool(signal.p1_horizontal_hit),
                "p1_horizontal_distance_bars": float(signal.p1_horizontal_distance_bars)
                if math.isfinite(float(signal.p1_horizontal_distance_bars))
                else None,
                "p1_horizontal_error_atr": float(signal.p1_horizontal_error_atr)
                if math.isfinite(float(signal.p1_horizontal_error_atr))
                else None,
                "p1_horizontal_score": float(signal.p1_horizontal_score),
                "p4_contrary_pivots": int(signal.p4_contrary_pivots),
                "p4_contrary_swing_atr": float(signal.p4_contrary_swing_atr),
                "p4_contrary_score": float(signal.p4_contrary_score),
                "impulse_45_bars": int(signal.impulse_45_bars),
                "impulse_45_atr": float(signal.impulse_45_atr) if math.isfinite(float(signal.impulse_45_atr)) else None,
                "impulse_45_same_dir_ratio": float(signal.impulse_45_same_dir_ratio)
                if math.isfinite(float(signal.impulse_45_same_dir_ratio))
                else None,
                "sweet_zone_width_atr": float(signal.sweet_zone_width_atr)
                if math.isfinite(float(signal.sweet_zone_width_atr))
                else None,
                "sweet_zone_expansion_atr_per_bar": float(signal.sweet_zone_expansion_atr_per_bar)
                if math.isfinite(float(signal.sweet_zone_expansion_atr_per_bar))
                else None,
                "p5_volume_ratio": float(signal.p5_volume_ratio) if math.isfinite(float(signal.p5_volume_ratio)) else None,
                "p5_rejection_atr": float(signal.p5_rejection_atr) if math.isfinite(float(signal.p5_rejection_atr)) else None,
                "v2_quality": float(signal.v2_quality),
            },
        }

    def detect_signal(self, state: WolfeWaveState) -> Optional[dict]:
        bars = state.snapshot()
        if len(bars) < self.min_bars:
            return None
        try:
            frame = bars_to_frame(bars, WOLFE_WAVE_INTERVAL)
            scan_config = replace(state.config, min_score=0.0)
            signals = find_wolfe_signals(frame, scan_config, symbol=state.symbol)
        except Exception as exc:  # noqa: BLE001
            log.warning(f"[wolfe] {state.symbol}: signal evaluation failed: {exc}")
            return None
        if not signals:
            return None

        recent_idx = max(0, len(frame) - 3)
        recent_cutoff = pd.Timestamp(frame["close_time"].iloc[recent_idx]).tz_convert("UTC")
        candidates = [
            signal
            for signal in signals
            if signal.entry_time >= recent_cutoff
            and (state.last_entry_time is None or signal.entry_time > state.last_entry_time)
        ]
        if not candidates:
            return None
        signal = sorted(candidates, key=lambda item: (item.entry_time, item.score), reverse=True)[0]
        signal_key = signal.event_key
        if signal.score < state.config.min_score:
            signal_key = f"{signal.event_key}|rejected"
        if state.last_signal_key == signal_key:
            return None
        state.last_signal_key = signal_key

        direction = signal.direction
        entry = float(frame["close"].iloc[-1])
        stop, live_stop_adjusted = fee_aware_stop_price(
            direction=direction,
            entry_price=entry,
            structural_stop_price=float(signal.stop_price),
            cfg=state.config,
        )
        target = float(signal.target_price)
        risk = abs(entry - stop)
        if risk <= 0 or not math.isfinite(risk):
            return None
        if direction == "long" and (entry <= stop or entry >= target):
            return None
        if direction == "short" and (entry >= stop or entry <= target):
            return None
        live_rr = (target - entry) / risk if direction == "long" else (entry - target) / risk
        if live_rr < state.config.min_rr or live_rr > state.config.max_rr:
            return None
        live_fee_risk = fee_to_price_risk(entry, stop, state.config)
        if state.config.max_fee_to_price_risk > 0 and live_fee_risk > state.config.max_fee_to_price_risk:
            return None
        entry_risk_pct = risk / entry * 100.0 if entry > 0 else math.inf

        payload = self._signal_payload(
            signal,
            entry=entry,
            stop=stop,
            target=target,
            risk=risk,
            threshold=state.config.min_score,
            target_rr_planned=live_rr,
            fee_risk=live_fee_risk,
            entry_risk_pct=entry_risk_pct,
            stop_adjusted_for_fee=bool(signal.stop_adjusted_for_fee or live_stop_adjusted),
        )
        if signal.score < state.config.min_score:
            payload["rejected"] = True
            payload["reject_reason"] = (
                f"Wolfe score {signal.score:.1f} below threshold "
                f"{state.config.min_score:.1f}"
            )
            return payload

        state.last_entry_time = signal.entry_time
        return payload


__all__ = [
    "DEFAULT_WOLFE_WAVE_SYMBOLS",
    "WOLFE_WAVE_INTERVAL",
    "WolfeWaveEngine",
    "WolfeWaveState",
    "fetch_warmup_bars_interval",
    "load_wolfe_wave_configs",
    "parse_symbol_list",
]
