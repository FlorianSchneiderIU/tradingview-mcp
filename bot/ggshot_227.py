from __future__ import annotations

import json
import logging
import math
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from scripts.backtest_ggshot_227 import (
    filter_mask,
    gg_flips,
    make_frame_cache,
    resample_local,
    rolling_midline,
)


log = logging.getLogger("mm")

GGSHOT_227_INTERVAL = "5m"
DEFAULT_GGSHOT_227_SYMBOLS = "BTCUSDT,BNBUSDT"
DEFAULT_GGSHOT_227_CONFIG = {
    "timeframe": "15m",
    "bb_period": 150,
    "bb_dev": 3.1,
    "sensitivity": 350,
    "filter_mode": "atr_or_rsi",
    "sl_pct": 1.5,
    "tp_pcts": [0.5, 1.1, 2.1, 4.5],
    "qty_pcts": [30.0, 30.0, 15.0, 15.0],
    "exit_style": "multi_tp_be",
    "min_risk_pct": 0.15,
}


def normalize_ggshot_timeframe(value: str) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "5": "5m",
        "5m": "5m",
        "15": "15m",
        "15m": "15m",
        "30": "30m",
        "30m": "30m",
        "45": "45m",
        "45m": "45m",
        "60": "1h",
        "1h": "1h",
        "1hr": "1h",
        "1hour": "1h",
    }
    if text not in aliases:
        raise ValueError(f"Unsupported GGShot timeframe: {value!r}")
    return aliases[text]


@dataclass(frozen=True)
class GgShotLiveConfig:
    timeframe: str
    bb_period: int
    bb_dev: float
    sensitivity: int = 350
    filter_mode: str = "atr_or_rsi"
    sl_pct: float = 1.5
    tp_pcts: tuple[float, ...] = (0.5, 1.1, 2.1, 4.5)
    qty_pcts: tuple[float, ...] = (30.0, 30.0, 15.0, 15.0)
    exit_style: str = "multi_tp_be"
    min_risk_pct: float = 0.15
    min_rr: float = 0.0
    max_rr: float = 20.0

    @classmethod
    def from_mapping(cls, payload: dict) -> "GgShotLiveConfig":
        merged = {**DEFAULT_GGSHOT_227_CONFIG, **payload}
        return cls(
            timeframe=normalize_ggshot_timeframe(str(merged["timeframe"])),
            bb_period=int(merged["bb_period"]),
            bb_dev=float(merged["bb_dev"]),
            sensitivity=int(merged.get("sensitivity", 350)),
            filter_mode=str(merged.get("filter_mode", "atr_or_rsi")),
            sl_pct=float(merged.get("sl_pct", 1.5)),
            tp_pcts=tuple(float(x) for x in merged.get("tp_pcts", DEFAULT_GGSHOT_227_CONFIG["tp_pcts"])),
            qty_pcts=tuple(float(x) for x in merged.get("qty_pcts", DEFAULT_GGSHOT_227_CONFIG["qty_pcts"])),
            exit_style=str(merged.get("exit_style", "multi_tp_be")),
            min_risk_pct=float(merged.get("min_risk_pct", 0.15)),
            min_rr=float(merged.get("min_rr", 0.0)),
            max_rr=float(merged.get("max_rr", 20.0)),
        )


def bars_to_frame(bars: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(bars)
    if frame.empty:
        return pd.DataFrame(columns=["open_time", "close_time", "open", "high", "low", "close", "volume"])
    frame = frame.drop_duplicates("ts", keep="last").sort_values("ts").reset_index(drop=True)
    frame["open_time"] = pd.to_datetime(frame["ts"].astype("int64"), unit="ms", utc=True)
    frame["close_time"] = frame["open_time"] + pd.Timedelta(minutes=5) - pd.Timedelta(milliseconds=1)
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
    return frame[["open_time", "close_time", "open", "high", "low", "close", "volume"]].dropna().reset_index(drop=True)


def load_ggshot_227_configs(*, symbols: list[str], config_path: str) -> dict[str, GgShotLiveConfig]:
    path = Path(config_path)
    loaded: dict[str, object] = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                loaded = {str(k).upper(): v for k, v in payload.items() if not str(k).startswith("_")}
        except Exception as exc:  # noqa: BLE001 - disable bad config entries but keep booting.
            log.error(f"[ggshot] Failed to load config {path}: {exc}")
    else:
        log.warning(f"[ggshot] Config not found at {path}; using built-in BTC/BNB defaults")

    out: dict[str, GgShotLiveConfig] = {}
    for symbol in symbols:
        symbol = str(symbol).upper()
        payload = loaded.get(symbol)
        if payload is None:
            if path.exists():
                log.warning(f"[ggshot] {symbol}: no config entry in {path}; skipped")
                continue
            payload = {}
            if symbol == "BNBUSDT":
                payload = {"timeframe": "30m", "bb_period": 90, "bb_dev": 2.4}
        if not isinstance(payload, dict):
            log.warning(f"[ggshot] {symbol}: config payload is not an object; skipped")
            continue
        if payload.get("enabled") is False:
            log.info(f"[ggshot] {symbol}: disabled by config")
            continue
        try:
            cfg = GgShotLiveConfig.from_mapping(payload)
        except Exception as exc:  # noqa: BLE001
            log.error(f"[ggshot] {symbol}: invalid config: {exc}")
            continue
        out[symbol] = cfg
        log.info(
            f"[ggshot] {symbol}: config loaded tf={cfg.timeframe} "
            f"bb={cfg.bb_period}/{cfg.bb_dev:g} sens={cfg.sensitivity} "
            f"filter={cfg.filter_mode} sl={cfg.sl_pct:g}% exit={cfg.exit_style}"
        )
    return out


class GgShotState:
    def __init__(self, symbol: str, config: GgShotLiveConfig, max_bars: int):
        self.symbol = str(symbol).upper()
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


class GgShotEngine:
    def __init__(self, min_bars: int = 3000):
        self.min_bars = int(min_bars)

    @staticmethod
    def _complete_timeframe(frame5: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        resampled = resample_local(frame5, timeframe)
        if resampled.empty or frame5.empty:
            return resampled
        last_source_close = pd.Timestamp(frame5["close_time"].iloc[-1])
        return resampled[resampled["close_time"] <= last_source_close].reset_index(drop=True)

    @staticmethod
    def _payload(
        *,
        state: GgShotState,
        direction: str,
        entry: float,
        stop: float,
        target_prices: list[float],
        signal_time: pd.Timestamp,
        imba_line: float,
        live_rr: float,
        risk_pct: float,
    ) -> dict:
        cfg = state.config
        return {
            "strategy": "ggshot_227",
            "signal": direction,
            "entry": float(entry),
            "model_entry": float(entry),
            "sl": float(stop),
            "model_sl": float(stop),
            "tp1": float(target_prices[0]),
            "take_profit": float(target_prices[-1]),
            "tp_pcts": list(cfg.tp_pcts),
            "tp_prices": [float(x) for x in target_prices],
            "tp_qty_pcts": list(cfg.qty_pcts),
            "trail_dist": abs(float(entry) - float(stop)),
            "exit_style": cfg.exit_style,
            "move_sl_to_be_after_tp1": cfg.exit_style.endswith("_be"),
            "prob": 1.0,
            "threshold": 0.0,
            "entry_time": signal_time.isoformat(),
            "pattern_tf": cfg.timeframe,
            "imba_line": float(imba_line),
            "target_rr_planned": float(live_rr),
            "entry_risk_pct": float(risk_pct),
            "feature_columns": [
                "target_rr_planned",
                "entry_risk_pct",
                "imba_line",
                "bb_period",
                "bb_dev",
                "sensitivity",
            ],
            "feature_snapshot": {
                "target_rr_planned": float(live_rr),
                "entry_risk_pct": float(risk_pct),
                "imba_line": float(imba_line),
                "bb_period": float(cfg.bb_period),
                "bb_dev": float(cfg.bb_dev),
                "sensitivity": float(cfg.sensitivity),
            },
        }

    def detect_signal(self, state: GgShotState) -> Optional[dict]:
        bars = state.snapshot()
        if len(bars) < self.min_bars:
            return None
        cfg = state.config
        try:
            frame5 = bars_to_frame(bars)
            frame = self._complete_timeframe(frame5, cfg.timeframe)
            min_rows = max(cfg.bb_period, cfg.sensitivity) + 3
            if len(frame) < min_rows:
                return None
            cache = make_frame_cache(frame)
            long_flip, short_flip = gg_flips(cache, cfg.bb_period, cfg.bb_dev)
            imba = rolling_midline(cache, cfg.sensitivity)
            filt = filter_mask(cache, cfg.filter_mode)
        except Exception as exc:  # noqa: BLE001
            log.warning(f"[ggshot] {state.symbol}: signal evaluation failed: {exc}")
            return None

        idx = len(frame) - 1
        if not bool(filt[idx]) or not math.isfinite(float(imba[idx])):
            return None
        direction_int = 1 if bool(long_flip[idx]) else -1 if bool(short_flip[idx]) else 0
        if direction_int == 0:
            return None

        signal_time = pd.Timestamp(frame["close_time"].iloc[idx]).tz_convert("UTC")
        direction = "long" if direction_int > 0 else "short"
        signal_key = f"{state.symbol}|{signal_time.isoformat()}|{direction}|{cfg.timeframe}"
        if state.last_signal_key == signal_key:
            return None

        entry = float(frame5["close"].iloc[-1])
        imba_line = float(imba[idx])
        stop = imba_line - entry * cfg.sl_pct / 100.0 if direction_int > 0 else imba_line + entry * cfg.sl_pct / 100.0
        if direction_int > 0 and stop >= entry:
            return None
        if direction_int < 0 and stop <= entry:
            return None
        risk = abs(entry - stop)
        risk_pct = risk / entry * 100.0 if entry > 0 else math.inf
        if risk <= 0 or risk_pct < cfg.min_risk_pct:
            return None
        target_prices = [entry * (1.0 + direction_int * pct / 100.0) for pct in cfg.tp_pcts]
        live_rr = abs(target_prices[-1] - entry) / risk if risk > 0 else math.inf
        if live_rr < cfg.min_rr or live_rr > cfg.max_rr:
            return None

        state.last_signal_key = signal_key
        state.last_entry_time = signal_time
        return self._payload(
            state=state,
            direction=direction,
            entry=entry,
            stop=stop,
            target_prices=target_prices,
            signal_time=signal_time,
            imba_line=imba_line,
            live_rr=live_rr,
            risk_pct=risk_pct,
        )


__all__ = [
    "DEFAULT_GGSHOT_227_SYMBOLS",
    "GGSHOT_227_INTERVAL",
    "GgShotEngine",
    "GgShotState",
    "load_ggshot_227_configs",
]
