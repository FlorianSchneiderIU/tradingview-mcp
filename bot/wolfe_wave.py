from __future__ import annotations

import json
import logging
import math
import os
import threading
from collections import deque
from pathlib import Path
from typing import Optional

import pandas as pd

from scripts.backtest_wolfe_wave import (
    WolfeConfig,
    bybit_symbol,
    ensure_ohlcv_frame,
    find_wolfe_signals,
    normalize_timeframe,
)
from turtle_soup import fetch_warmup_bars_interval, parse_symbol_list


log = logging.getLogger("mm")

WOLFE_WAVE_INTERVAL = "5m"
WOLFE_WAVE_BYBIT_INTERVAL = "5"
DEFAULT_WOLFE_WAVE_SYMBOLS = "BTCUSDT,ETHUSDT,XRPUSDT,BNBUSDT,LINKUSDT"
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
            f"pivots={cfg.pivot_method} min_score={cfg.min_score:.1f}"
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

    def detect_signal(self, state: WolfeWaveState) -> Optional[dict]:
        bars = state.snapshot()
        if len(bars) < self.min_bars:
            return None
        try:
            frame = bars_to_frame(bars, WOLFE_WAVE_INTERVAL)
            signals = find_wolfe_signals(frame, state.config, symbol=state.symbol)
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
        if state.last_signal_key == signal.event_key:
            return None
        state.last_signal_key = signal.event_key

        direction = signal.direction
        entry = float(frame["close"].iloc[-1])
        stop = float(signal.stop_price)
        target = float(signal.target_price)
        risk = abs(entry - stop)
        if risk <= 0 or not math.isfinite(risk):
            return None
        if direction == "long" and (entry <= stop or entry >= target):
            return None
        if direction == "short" and (entry >= stop or entry <= target):
            return None

        state.last_entry_time = signal.entry_time
        return {
            "strategy": "wolfe_wave",
            "signal": direction,
            "entry": entry,
            "model_entry": float(signal.entry_price),
            "sl": stop,
            "tp1": target,
            "take_profit": target,
            "trail_dist": risk,
            "exit_style": "fixed_tp",
            "prob": signal.score / 100.0,
            "threshold": state.config.min_score / 100.0,
            "entry_time": signal.entry_time.isoformat(),
            "pattern_tf": signal.pattern_tf,
            "pivot_method": signal.pivot_method,
            "target_rr_planned": float(signal.target_rr_planned),
            "score": float(signal.score),
            "p5_break_atr": float(signal.p5_break_atr),
            "symmetry_ratio": float(signal.symmetry_ratio),
            "epa_slope_atr": float(signal.epa_slope_atr),
            "volume_ratio": float(signal.volume_ratio),
            "rsi": float(signal.rsi),
            "feature_columns": [
                "score",
                "target_rr_planned",
                "p5_break_atr",
                "symmetry_ratio",
                "epa_slope_atr",
                "volume_ratio",
                "rsi",
            ],
            "feature_snapshot": {
                "score": float(signal.score),
                "target_rr_planned": float(signal.target_rr_planned),
                "p5_break_atr": float(signal.p5_break_atr),
                "symmetry_ratio": float(signal.symmetry_ratio),
                "epa_slope_atr": float(signal.epa_slope_atr),
                "volume_ratio": float(signal.volume_ratio) if math.isfinite(float(signal.volume_ratio)) else None,
                "rsi": float(signal.rsi) if math.isfinite(float(signal.rsi)) else None,
            },
        }


__all__ = [
    "DEFAULT_WOLFE_WAVE_SYMBOLS",
    "WOLFE_WAVE_INTERVAL",
    "WolfeWaveEngine",
    "WolfeWaveState",
    "fetch_warmup_bars_interval",
    "load_wolfe_wave_configs",
    "parse_symbol_list",
]
