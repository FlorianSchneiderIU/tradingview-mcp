from __future__ import annotations

import logging
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from pybit.unified_trading import HTTP

from scripts.backtest_turtle_soup import (
    Config,
    DEFAULT_BFM_ZONE_TF_SETS,
    DEFAULT_BFM_ZONE_TIMEFRAMES,
    INTERVAL_MS,
    normalize_timeframe,
    run_backtest,
    trade_from_position,
)
from scripts.ml_trade_outcome_filter import (
    BASE_FEATURE_COLUMNS,
    add_engineered_rescue_features,
    trade_bfm_feature_columns_for_groups,
    trade_feature_rows,
)


log = logging.getLogger("mm")

TURTLE_INTERVAL = "5m"
TURTLE_BYBIT_INTERVAL = "5"
DEFAULT_TURTLE_SYMBOLS = (
    "BNBUSDT,TAOUSDT,LINKUSDT,BTCUSDT,ETHUSDT,SOLUSDT"
)
NEW_LIQUIDITY_FEATURE_PREFIXES = ("liquidity_", "zone_source_")


def _feature_snapshot(row: pd.Series, columns: list[str]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for column in columns:
        try:
            value = float(row.get(column, math.nan))
        except (TypeError, ValueError):
            value = math.nan
        out[column] = value if math.isfinite(value) else None
    return out


def parse_symbol_list(raw: str | None) -> list[str]:
    if raw is None or not raw.strip():
        raw = DEFAULT_TURTLE_SYMBOLS
    out: list[str] = []
    for chunk in raw.split(","):
        symbol = chunk.strip().upper()
        if symbol and symbol not in out:
            out.append(symbol)
    return out


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def fetch_warmup_bars_interval(
    http_client: HTTP,
    symbol: str,
    *,
    interval: str,
    n: int,
) -> list[dict]:
    """Fetch newest closed Bybit linear candles, oldest to newest."""
    interval = normalize_timeframe(interval)
    bybit_interval = "5" if interval == "5m" else "15" if interval == "15m" else interval
    all_bars: list[dict] = []
    end_ts: Optional[int] = None
    while len(all_bars) < n:
        kwargs: dict = dict(category="linear", symbol=symbol, interval=bybit_interval, limit=1000)
        if end_ts is not None:
            kwargs["end"] = end_ts
        try:
            resp = http_client.get_kline(**kwargs)
            items = resp.get("result", {}).get("list", [])
        except Exception as exc:
            log.warning(f"  get_kline error for {symbol} {interval}: {exc}")
            break
        if not items:
            break
        for it in reversed(items):
            all_bars.append({
                "ts": int(it[0]),
                "open": float(it[1]),
                "high": float(it[2]),
                "low": float(it[3]),
                "close": float(it[4]),
                "volume": float(it[5]),
            })
        end_ts = int(items[-1][0]) - 1
        if len(items) < 1000:
            break
        time.sleep(0.05)

    seen: set[int] = set()
    unique: list[dict] = []
    for bar in sorted(all_bars, key=lambda x: x["ts"]):
        if bar["ts"] not in seen:
            seen.add(bar["ts"])
            unique.append(bar)
    return unique[-n:]


def bars_to_frame(bars: list[dict], interval: str = TURTLE_INTERVAL) -> pd.DataFrame:
    interval = normalize_timeframe(interval)
    interval_ms = INTERVAL_MS[interval]
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
class TurtleModel:
    symbol: str
    model: object
    feature_columns: list[str]
    threshold: float


def load_turtle_models(
    *,
    symbols: list[str],
    models_dir: str,
    leaderboard_path: str,
) -> dict[str, TurtleModel]:
    thresholds: dict[str, float] = {}
    if leaderboard_path and os.path.exists(leaderboard_path):
        leaderboard = pd.read_csv(leaderboard_path)
        for _, row in leaderboard.iterrows():
            symbol = str(row.get("symbol", "")).upper()
            threshold = row.get("best_threshold")
            if symbol and pd.notna(threshold):
                thresholds[symbol] = float(threshold)
    else:
        log.warning(f"[turtle] Leaderboard not found: {leaderboard_path}")

    out: dict[str, TurtleModel] = {}
    model_root = Path(models_dir)
    for symbol in symbols:
        path = model_root / f"{symbol.lower()}_model.joblib"
        if not path.exists():
            log.warning(f"[turtle] {symbol}: model not found at {path}; disabled")
            continue
        if symbol not in thresholds:
            log.warning(f"[turtle] {symbol}: threshold missing from leaderboard; disabled")
            continue
        try:
            payload = joblib.load(path)
            out[symbol] = TurtleModel(
                symbol=symbol,
                model=payload["model"],
                feature_columns=list(payload["feature_columns"]),
                threshold=thresholds[symbol],
            )
            log.info(
                f"[turtle] {symbol}: model loaded threshold={thresholds[symbol]:.2f} "
                f"features={len(payload['feature_columns'])}"
            )
        except Exception as exc:
            log.error(f"[turtle] {symbol}: failed to load model {path}: {exc}")
    return out


class TurtleSoupState:
    def __init__(self, symbol: str, model: TurtleModel, max_bars: int):
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


class TurtleSoupEngine:
    def __init__(
        self,
        *,
        bfm_timeframes: str = DEFAULT_BFM_ZONE_TIMEFRAMES,
        bfm_tf_sets: str = DEFAULT_BFM_ZONE_TF_SETS,
        bfm_feature_groups: str = "line,channel",
        bfm_invalidation: str = "wick",
        bfm_max_extension_bars: int = 300,
    ):
        self.cfg = Config(
            exec_tf=TURTLE_INTERVAL,
            structure_tf="15m",
            entry_mode="zone_retest",
            tf1="1h",
            tf2="4h",
            use_tf1=True,
            use_tf2=False,
            block_dead_zone=False,
            max_structure_bars_to_choch=32,
            min_entry_risk_pct=0.0,
            max_zone_scan=0,
        )
        self.bfm_timeframes = bfm_timeframes
        self.bfm_tf_sets = bfm_tf_sets
        self.bfm_feature_groups = bfm_feature_groups
        self.bfm_invalidation = bfm_invalidation
        self.bfm_max_extension_bars = bfm_max_extension_bars
        self.sfp_trigger_mode = os.environ.get("TURTLE_USE_SFP_LIQUIDITY_TRIGGERS", "auto").strip().lower()
        self.sfp_timeframes = os.environ.get("TURTLE_SFP_TIMEFRAMES", "15m,1h,4h")
        self.sfp_left = int(os.environ.get("TURTLE_SFP_LEFT", "15"))
        self.sfp_right = int(os.environ.get("TURTLE_SFP_RIGHT", "10"))
        self.sfp_level_width_atr = float(os.environ.get("TURTLE_SFP_LEVEL_WIDTH_ATR", "0.15"))
        self.sfp_strict = env_bool("TURTLE_SFP_STRICT", True)
        self.sfp_require_open_reclaim = env_bool("TURTLE_SFP_REQUIRE_OPEN_RECLAIM", True)
        self.base_feature_columns = list(BASE_FEATURE_COLUMNS)
        self.bfm_feature_columns = trade_bfm_feature_columns_for_groups(bfm_feature_groups)

    def _cfg_for_model(self, model: TurtleModel) -> Config:
        if self.sfp_trigger_mode in {"1", "true", "yes", "on"}:
            enable_sfp = True
        elif self.sfp_trigger_mode in {"0", "false", "no", "off"}:
            enable_sfp = False
        else:
            enable_sfp = any(
                column.startswith(NEW_LIQUIDITY_FEATURE_PREFIXES)
                for column in model.feature_columns
            )
        if not enable_sfp:
            return self.cfg
        return replace(
            self.cfg,
            use_sfp_liquidity_zones=True,
            sfp_timeframes=self.sfp_timeframes,
            sfp_left=self.sfp_left,
            sfp_right=self.sfp_right,
            sfp_level_width_atr=self.sfp_level_width_atr,
            sfp_strict=self.sfp_strict,
            sfp_require_open_reclaim=self.sfp_require_open_reclaim,
        )

    def detect_signal(self, state: TurtleSoupState) -> Optional[dict]:
        bars = state.snapshot()
        if len(bars) < 500:
            return None
        frame = bars_to_frame(bars, TURTLE_INTERVAL)
        if frame.empty:
            return None

        try:
            cfg = self._cfg_for_model(state.model)
            _closed_trades, live_state = run_backtest(frame, cfg, return_state=True)
            position = live_state.get("position")
            if position is None:
                return None

            entry_time = pd.Timestamp(position["entry_time"])
            recent_idx = max(0, len(frame) - 3)
            recent_cutoff = pd.Timestamp(frame["open_time"].iloc[recent_idx])
            if entry_time < recent_cutoff:
                return None
            if state.last_entry_time is not None and entry_time <= state.last_entry_time:
                return None

            last_idx = len(frame) - 1
            live_trade = trade_from_position(
                position,
                last_idx,
                pd.Timestamp(frame["close_time"].iloc[-1]),
                float(frame["close"].iloc[-1]),
                "open_live_candidate",
                cfg,
            )
            feature_frame, _trades = trade_feature_rows(
                state.symbol,
                frame,
                cfg,
                use_bfm_features=True,
                bfm_timeframes=self.bfm_timeframes,
                bfm_tf_sets=self.bfm_tf_sets,
                bfm_invalidation=self.bfm_invalidation,
                bfm_max_extension_bars=self.bfm_max_extension_bars,
                use_sfp_liquidity_zones=cfg.use_sfp_liquidity_zones,
                sfp_timeframes=cfg.sfp_timeframes,
                sfp_left=cfg.sfp_left,
                sfp_right=cfg.sfp_right,
                sfp_level_width_atr=cfg.sfp_level_width_atr,
                sfp_strict=cfg.sfp_strict,
                sfp_require_open_reclaim=cfg.sfp_require_open_reclaim,
                precomputed_trades=[],
                extra_trades=[live_trade],
            )
        except Exception as exc:
            log.warning(f"[turtle] {state.symbol}: signal evaluation failed: {exc}")
            return None

        if feature_frame.empty:
            return None
        data = add_engineered_rescue_features(feature_frame)
        data["entry_time"] = pd.to_datetime(data["entry_time"], utc=True, errors="coerce")
        candidates = data.copy()
        if state.last_entry_time is not None:
            candidates = candidates[candidates["entry_time"] > state.last_entry_time]
        if candidates.empty:
            return None

        feature_columns = state.model.feature_columns
        for column in feature_columns:
            if column not in candidates.columns:
                candidates[column] = math.nan
        candidates["trade_win_prob"] = state.model.model.predict_proba(
            candidates[feature_columns].astype(float)
        )[:, 1]
        selected = candidates[candidates["trade_win_prob"] >= state.model.threshold].copy()
        if selected.empty:
            best = candidates.sort_values("trade_win_prob", ascending=False).iloc[0]
            entry_time = pd.Timestamp(best["entry_time"])
            signal_key = f"{state.symbol}|{entry_time.isoformat()}|{best['direction']}|{float(best['entry_price']):.10g}|rejected"
            if state.last_signal_key == signal_key:
                return None
            state.last_signal_key = signal_key
            direction = str(best["direction"])
            reason = (
                f"ML probability {float(best['trade_win_prob']):.3f} "
                f"below threshold {state.model.threshold:.2f}"
            )
            log.info(
                f"[turtle] {state.symbol}: candidate filtered "
                f"prob={float(best['trade_win_prob']):.3f} < {state.model.threshold:.2f}"
            )
            return {
                "strategy": "turtle_soup",
                "signal": direction,
                "entry": float(frame["close"].iloc[-1]),
                "model_entry": float(best["entry_price"]),
                "sl": float(best["stop_price"]),
                "tp1": float(best["target_price"]),
                "take_profit": float(best["target_price"]),
                "trail_dist": abs(float(frame["close"].iloc[-1]) - float(best["stop_price"])),
                "exit_style": "fixed_tp",
                "prob": float(best["trade_win_prob"]),
                "threshold": state.model.threshold,
                "entry_time": entry_time.isoformat(),
                "zone_top": float(best["zone_top"]),
                "zone_bottom": float(best["zone_bottom"]),
                "zone_source": "sfp_pivot" if float(best.get("zone_source_sfp", 0.0)) > 0.5 else "ob_break",
                "feature_columns": list(feature_columns),
                "feature_snapshot": _feature_snapshot(best, feature_columns),
                "rejected": True,
                "reject_reason": reason,
            }

        row = selected.sort_values(["entry_time", "trade_win_prob"], ascending=[True, False]).iloc[0]
        entry_time = pd.Timestamp(row["entry_time"])
        signal_key = f"{state.symbol}|{entry_time.isoformat()}|{row['direction']}|{float(row['entry_price']):.10g}"
        if state.last_signal_key == signal_key:
            return None
        state.last_entry_time = entry_time
        state.last_signal_key = signal_key

        direction = str(row["direction"])
        model_entry = float(row["entry_price"])
        entry = float(frame["close"].iloc[-1])
        stop = float(row["stop_price"])
        target = float(row["target_price"])
        risk = abs(entry - stop)
        if risk <= 0 or not np.isfinite(risk):
            return None
        if direction == "long" and (entry <= stop or entry >= target):
            log.info(
                f"[turtle] {state.symbol}: live price no longer valid for long "
                f"entry={entry:.8g} stop={stop:.8g} target={target:.8g}"
            )
            return None
        if direction == "short" and (entry >= stop or entry <= target):
            log.info(
                f"[turtle] {state.symbol}: live price no longer valid for short "
                f"entry={entry:.8g} stop={stop:.8g} target={target:.8g}"
            )
            return None

        return {
            "strategy": "turtle_soup",
            "signal": direction,
            "entry": entry,
            "model_entry": model_entry,
            "sl": stop,
            "tp1": target,
            "take_profit": target,
            "trail_dist": risk,
            "exit_style": "fixed_tp",
            "prob": float(row["trade_win_prob"]),
            "threshold": state.model.threshold,
            "entry_time": entry_time.isoformat(),
            "zone_top": float(row["zone_top"]),
            "zone_bottom": float(row["zone_bottom"]),
            "zone_source": "sfp_pivot" if float(row.get("zone_source_sfp", 0.0)) > 0.5 else "ob_break",
            "feature_columns": list(feature_columns),
            "feature_snapshot": _feature_snapshot(row, feature_columns),
        }
