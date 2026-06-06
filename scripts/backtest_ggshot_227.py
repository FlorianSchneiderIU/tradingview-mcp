from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backtest_turtle_soup import parse_utc_datetime  # noqa: E402
from scripts.backtest_wolfe_wave import BYBIT_URL, fetch_bybit_klines  # noqa: E402
from scripts.experiment_pine_strategy_candidates import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    DEFAULT_UNIVERSE,
    clean_symbol,
    find_cache_path,
    load_frame,
    load_universe,
    parse_list,
    rma,
    rsi,
    sma,
    true_range,
)


@dataclass(frozen=True)
class GgConfig:
    timeframe: str
    bb_period: int
    bb_dev: float
    sensitivity: int
    filter_mode: str
    tp_pct: float
    sl_pct: float
    max_hold_bars: int
    symbol: str = ""
    exec_mode: str = "single_tp"
    tp_pcts: tuple[float, ...] = (0.5, 1.1, 2.1, 4.5)
    qty_pcts: tuple[float, ...] = (30.0, 30.0, 15.0, 15.0)
    dca_count: int = 1
    opposite_reverse: bool = True

    @property
    def name(self) -> str:
        prefix = f"{self.symbol.lower()}_" if self.symbol else ""
        suffix = f"_{self.exec_mode}" if self.exec_mode != "single_tp" else ""
        return (
            f"{prefix}ggshot_227_{self.timeframe}_bb{self.bb_period}_dev{self.bb_dev:g}_"
            f"sens{self.sensitivity}_{self.filter_mode}_tp{self.tp_pct:g}_sl{self.sl_pct:g}{suffix}"
        ).replace(".", "p")


@dataclass
class Agg:
    count: int = 0
    net_r: float = 0.0
    wins_r: float = 0.0
    losses_r: float = 0.0
    wins: int = 0
    equity: float = 0.0
    peak: float = 0.0
    max_dd_r: float = 0.0

    def add(self, r_multiple: float) -> None:
        r_multiple = float(r_multiple)
        self.count += 1
        self.net_r += r_multiple
        if r_multiple > 0.0:
            self.wins += 1
            self.wins_r += r_multiple
        elif r_multiple < 0.0:
            self.losses_r += r_multiple
        self.equity += r_multiple
        self.peak = max(self.peak, self.equity)
        self.max_dd_r = max(self.max_dd_r, self.peak - self.equity)

    def metrics(self) -> dict[str, float]:
        pf = self.wins_r / abs(self.losses_r) if self.losses_r < 0.0 else (99.0 if self.wins_r > 0 else 0.0)
        return {
            "trades": int(self.count),
            "net_r": float(self.net_r),
            "avg_r": float(self.net_r / self.count) if self.count else 0.0,
            "win_rate": float(self.wins / self.count) if self.count else 0.0,
            "profit_factor": float(pf),
            "max_dd_r": float(self.max_dd_r),
        }


@dataclass
class ConfigAgg:
    train: Agg
    oos: Agg
    all: Agg
    by_symbol: dict[str, Agg]
    invalid: int = 0
    overlapped: int = 0


@dataclass(frozen=True)
class FrameCache:
    frame: pd.DataFrame
    open_: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    open_time_ns: np.ndarray
    close_time_ns: np.ndarray
    atr_filter: np.ndarray
    atr_ma: np.ndarray
    atr14: np.ndarray
    rsi_filter: np.ndarray


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in str(raw or "").split(",") if item.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in str(raw or "").split(",") if item.strip()]


def parse_str_list(raw: str) -> list[str]:
    return [item.strip().lower() for item in str(raw or "").split(",") if item.strip()]


def dca_count_for_mode(mode: str) -> int:
    if "dca5" in mode:
        return 5
    if "dca3" in mode:
        return 3
    if "dca2" in mode:
        return 2
    return 1


def ensure_5m_cache(
    symbol: str,
    *,
    cache_dir: Path,
    start: datetime,
    end: datetime,
    base_url: str,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    normalized = clean_symbol(symbol).lower()
    for candidate in sorted(cache_dir.glob(f"{normalized}_5m_*.pkl")):
        try:
            frame = pd.read_pickle(candidate)
        except Exception:
            continue
        if frame.empty or "open_time" not in frame or "close_time" not in frame:
            continue
        first = pd.Timestamp(frame["open_time"].iloc[0]).to_pydatetime()
        last = pd.Timestamp(frame["close_time"].iloc[-1]).to_pydatetime()
        if first <= start and last >= end:
            return candidate
    path = cache_dir / f"{normalized}_5m_{start:%Y%m%d}_{end:%Y%m%d}.pkl"
    if path.exists():
        return path
    frame = fetch_bybit_klines(symbol, "5m", start, end, base_url=base_url)
    frame.to_pickle(path)
    return path


PINE_CRYPTO_PRESETS: tuple[tuple[str, str, int, float], ...] = (
    ("BTCUSDT", "30m", 160, 1.5),
    ("BTCUSDT", "1h", 180, 1.0),
    ("BTCUSDT", "15m", 150, 3.1),
    ("ETHUSDT", "15m", 150, 0.5),
    ("QTUMUSDT", "15m", 196, 1.0),
    ("ZILUSDT", "30m", 140, 1.5),
    ("ZILUSDT", "15m", 150, 3.5),
    ("AUDIOUSDT", "15m", 320, 2.0),
    ("AXSUSDT", "15m", 300, 3.0),
    ("BANDUSDT", "30m", 300, 2.0),
    ("BANDUSDT", "15m", 300, 2.0),
    ("TRXUSDT", "30m", 200, 2.0),
    ("OPUSDT", "15m", 170, 2.8),
    ("OPUSDT", "30m", 170, 2.3),
    ("APEUSDT", "15m", 120, 1.8),
    ("APEUSDT", "30m", 170, 2.3),
    ("ENSUSDT", "15m", 170, 3.3),
    ("BAKEUSDT", "30m", 100, 2.6),
    ("CELRUSDT", "30m", 200, 0.8),
    ("1000PEPEUSDT", "15m", 60, 2.0),
    ("1000SHIBUSDT", "45m", 50, 3.0),
    ("CHRUSDT", "15m", 50, 3.0),
    ("CHRUSDT", "45m", 50, 3.0),
    ("HNTUSDT", "15m", 300, 2.0),
    ("BLZUSDT", "15m", 190, 1.0),
    ("EGLDUSDT", "15m", 200, 3.0),
    ("KAVAUSDT", "15m", 200, 2.0),
    ("ZENUSDT", "15m", 200, 2.0),
    ("FLOWUSDT", "30m", 20, 3.0),
    ("YFIUSDT", "15m", 40, 3.0),
    ("LDOUSDT", "15m", 200, 2.0),
    ("MATICUSDT", "15m", 200, 2.5),
    ("ANKRUSDT", "15m", 20, 3.0),
    ("TOMOUSDT", "15m", 20, 3.0),
    ("DOTUSDT", "15m", 20, 2.5),
    ("DASHUSDT", "30m", 200, 1.0),
    ("JASMYUSDT", "15m", 60, 2.0),
    ("WOOUSDT", "15m", 40, 3.0),
    ("SUSHIUSDT", "30m", 40, 2.0),
    ("SUSHIUSDT", "15m", 40, 2.0),
    ("NEARUSDT", "30m", 60, 2.0),
    ("SKLUSDT", "15m", 200, 2.0),
    ("UNFIUSDT", "15m", 180, 2.0),
    ("FTMUSDT", "15m", 80, 2.2),
    ("SFPUSDT", "15m", 150, 2.0),
    ("BNBUSDT", "30m", 90, 2.4),
)


def build_configs(args: argparse.Namespace) -> list[GgConfig]:
    timeframes = parse_list(args.timeframes)
    if args.grid_mode == "pine":
        modes = parse_str_list(args.execution_modes)
        wanted_symbols = {clean_symbol(item) for item in parse_list(args.symbols)} if args.symbols.strip() else set()
        configs = []
        for symbol, timeframe, period, dev in PINE_CRYPTO_PRESETS:
            if wanted_symbols and symbol not in wanted_symbols:
                continue
            if timeframes and timeframe not in timeframes:
                continue
            for mode in modes:
                dca_count = dca_count_for_mode(mode)
                configs.append(
                    GgConfig(
                        timeframe=timeframe,
                        bb_period=period,
                        bb_dev=dev,
                        sensitivity=350,
                        filter_mode="atr_or_rsi",
                        tp_pct=2.1,
                        sl_pct=1.5,
                        max_hold_bars=0,
                        symbol=symbol,
                        exec_mode=mode,
                        tp_pcts=(0.5, 1.1, 2.1, 4.5),
                        qty_pcts=(30.0, 30.0, 15.0, 15.0),
                        dca_count=dca_count,
                    )
                )
        return configs
    if args.grid_mode == "smoke":
        configs: list[GgConfig] = []
        for tf in timeframes:
            hold = hold_for_tf(tf)
            configs.append(GgConfig(tf, 100, 1.5, 200, "none", 1.1, 1.5, hold))
            configs.append(GgConfig(tf, 100, 1.5, 350, "atr_or_rsi", 1.1, 1.5, hold))
        return configs
    if args.grid_mode == "fast":
        periods = [60, 100, 150, 200]
        devs = [1.5, 2.5, 3.5]
        sensitivities = [200, 350]
        filters = ["none", "atr_or_rsi"]
        tps = [0.5, 1.1, 2.1]
        sls = [1.5]
    else:
        periods = [40, 60, 100, 150, 200, 300]
        devs = [0.8, 1.5, 2.0, 2.5, 3.0, 3.5]
        sensitivities = [100, 150, 200, 350]
        filters = ["none", "atr", "rsi", "atr_or_rsi", "atr_and_rsi"]
        tps = [0.5, 1.1, 2.1, 4.5]
        sls = [1.0, 1.5, 2.0]
    configs = []
    for tf in timeframes:
        hold = hold_for_tf(tf)
        for period in periods:
            for dev in devs:
                for sensitivity in sensitivities:
                    for filter_mode in filters:
                        for tp in tps:
                            for sl in sls:
                                configs.append(GgConfig(tf, period, dev, sensitivity, filter_mode, tp, sl, hold))
    return configs


def hold_for_tf(timeframe: str) -> int:
    if timeframe == "5m":
        return 288
    if timeframe == "15m":
        return 96
    if timeframe in {"30m", "1h"}:
        return 48
    return 96


def resample_local(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if timeframe == "5m":
        return frame.copy().reset_index(drop=True)
    rule_map = {"15m": "15min", "30m": "30min", "45m": "45min", "1h": "1h"}
    rule = rule_map.get(timeframe)
    if rule is None:
        raise ValueError(f"unsupported timeframe {timeframe}")
    indexed = frame.set_index("open_time")
    out = indexed.resample(rule, label="left", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    out = out.dropna(subset=["open", "high", "low", "close"]).reset_index()
    out["close_time"] = out["open_time"] + pd.Timedelta(rule) - pd.Timedelta(milliseconds=1)
    return out[["open_time", "close_time", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def make_frame_cache(frame: pd.DataFrame) -> FrameCache:
    open_ = frame["open"].to_numpy(dtype=float)
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    close = frame["close"].to_numpy(dtype=float)
    tr = true_range(high, low, close)
    atr_filter = rma(tr, 5)
    return FrameCache(
        frame=frame,
        open_=open_,
        high=high,
        low=low,
        close=close,
        open_time_ns=frame["open_time"].to_numpy(dtype="datetime64[ns]").astype(np.int64, copy=False),
        close_time_ns=frame["close_time"].to_numpy(dtype="datetime64[ns]").astype(np.int64, copy=False),
        atr_filter=atr_filter,
        atr_ma=sma(atr_filter, 5),
        atr14=rma(tr, 14),
        rsi_filter=rsi(close, 7),
    )


def rolling_midline(cache: FrameCache, sensitivity: int) -> np.ndarray:
    high_line = pd.Series(cache.high).rolling(sensitivity, min_periods=sensitivity).max().to_numpy(dtype=float)
    low_line = pd.Series(cache.low).rolling(sensitivity, min_periods=sensitivity).min().to_numpy(dtype=float)
    return high_line - (high_line - low_line) * 0.5


def gg_flips(cache: FrameCache, bb_period: int, bb_dev: float) -> tuple[np.ndarray, np.ndarray]:
    close_series = pd.Series(cache.close)
    mid = close_series.rolling(bb_period, min_periods=bb_period).mean().to_numpy(dtype=float)
    std = close_series.rolling(bb_period, min_periods=bb_period).std(ddof=0).to_numpy(dtype=float)
    upper = mid + std * bb_dev
    lower = mid - std * bb_dev
    n = len(cache.close)
    trend_line = np.full(n, np.nan, dtype=float)
    itrend = np.zeros(n, dtype=int)
    long_flip = np.zeros(n, dtype=bool)
    short_flip = np.zeros(n, dtype=bool)
    for i in range(n):
        prev_line = trend_line[i - 1] if i > 0 else math.nan
        prev_itrend = itrend[i - 1] if i > 0 else 0
        tl = prev_line
        if math.isfinite(upper[i]) and cache.close[i] > upper[i]:
            tl = float(cache.low[i])
            if math.isfinite(prev_line) and tl < prev_line:
                tl = prev_line
        elif math.isfinite(lower[i]) and cache.close[i] < lower[i]:
            tl = float(cache.high[i])
            if math.isfinite(prev_line) and tl > prev_line:
                tl = prev_line
        trend_line[i] = tl
        itrend[i] = prev_itrend
        if math.isfinite(tl) and math.isfinite(prev_line):
            if tl > prev_line:
                itrend[i] = 1
            elif tl < prev_line:
                itrend[i] = -1
        if i > 0:
            long_flip[i] = prev_itrend == -1 and itrend[i] == 1
            short_flip[i] = prev_itrend == 1 and itrend[i] == -1
    return long_flip, short_flip


def filter_mask(cache: FrameCache, filter_mode: str) -> np.ndarray:
    mode = str(filter_mode or "none").lower()
    atr_ok = np.isfinite(cache.atr_filter) & np.isfinite(cache.atr_ma) & (cache.atr_filter >= cache.atr_ma)
    rsi_ok = np.isfinite(cache.rsi_filter) & ((cache.rsi_filter > 45.0) | (cache.rsi_filter < 10.0))
    if mode in {"none", "off", "no"}:
        return np.isfinite(cache.rsi_filter) & (cache.rsi_filter > 0.0)
    if mode == "atr":
        return atr_ok
    if mode == "rsi":
        return rsi_ok
    if mode == "atr_or_rsi":
        return atr_ok | rsi_ok
    if mode == "atr_and_rsi":
        return atr_ok & rsi_ok
    return np.ones(len(cache.close), dtype=bool)


def new_config_agg() -> ConfigAgg:
    return ConfigAgg(train=Agg(), oos=Agg(), all=Agg(), by_symbol={})


def add_trade(agg: ConfigAgg, symbol: str, r_multiple: float, entry_time_ns: int, split_ns: int) -> None:
    agg.all.add(r_multiple)
    if entry_time_ns < split_ns:
        agg.train.add(r_multiple)
    else:
        agg.oos.add(r_multiple)
    agg.by_symbol.setdefault(symbol, Agg()).add(r_multiple)


def simulate_config_symbol(
    cache: FrameCache,
    cfg: GgConfig,
    long_flip: np.ndarray,
    short_flip: np.ndarray,
    imba: np.ndarray,
    filt: np.ndarray,
    *,
    fee_bps_per_side: float,
    min_risk_pct: float,
) -> tuple[list[tuple[float, int]], int, int]:
    n = len(cache.close)
    trades: list[tuple[float, int]] = []
    invalid = 0
    overlapped = 0

    pos: dict[str, Any] | None = None

    def dca_levels(entry: float, imba_line: float, direction: int) -> list[float]:
        if cfg.dca_count <= 1:
            return []
        if direction > 0 and imba_line >= entry:
            return []
        if direction < 0 and imba_line <= entry:
            return []
        if cfg.dca_count == 2:
            return [float(imba_line)]
        mid = (entry + imba_line) / 2.0
        if cfg.dca_count == 3:
            return [float(mid), float(imba_line)]
        return [float((entry + mid) / 2.0), float(mid), float((mid + imba_line) / 2.0), float(imba_line)]

    def open_position(signal_idx: int, direction: int) -> None:
        nonlocal pos, invalid
        entry_idx = signal_idx + 1
        if entry_idx >= n or not math.isfinite(imba[signal_idx]):
            invalid += 1
            return
        entry = float(cache.open_[entry_idx])
        if entry <= 0.0:
            invalid += 1
            return
        imba_line = float(imba[signal_idx])
        stop = imba_line - entry * cfg.sl_pct / 100.0 if direction > 0 else imba_line + entry * cfg.sl_pct / 100.0
        if direction > 0 and stop >= entry:
            invalid += 1
            return
        if direction < 0 and stop <= entry:
            invalid += 1
            return
        risk = abs(entry - stop)
        risk_pct = risk / entry * 100.0
        if risk_pct < min_risk_pct:
            invalid += 1
            return
        units = 1.0
        pos = {
            "direction": int(direction),
            "entry_time_ns": int(cache.open_time_ns[entry_idx]),
            "entry_idx": int(entry_idx),
            "avg_entry": float(entry),
            "units": units,
            "max_units": units,
            "initial_risk": float(risk),
            "initial_entry": float(entry),
            "imba_line": imba_line,
            "stop": float(stop),
            "tp_hit": [False for _ in cfg.tp_pcts],
            "tp1_hit": False,
            "realized_pnl": 0.0,
            "fees": fee_bps_per_side / 10000.0 * entry * units,
            "dca_levels": dca_levels(entry, imba_line, direction),
            "filled_dca": set(),
            "trailing_short": math.nan,
        }

    def current_stop(idx: int) -> float:
        assert pos is not None
        direction = int(pos["direction"])
        avg_entry = float(pos["avg_entry"])
        if pos["tp1_hit"] and cfg.exec_mode in {"be", "dca2_be"}:
            return avg_entry
        if pos["tp1_hit"] and cfg.exec_mode in {"trailing", "dca2_trailing"}:
            if direction > 0:
                return float(imba[idx]) if math.isfinite(imba[idx]) else float(pos["stop"])
            atr = float(cache.atr14[idx]) if math.isfinite(cache.atr14[idx]) else 0.0
            candidate = float(cache.high[idx] + 5.0 * atr)
            prev = float(pos.get("trailing_short", math.nan))
            stop = candidate if not math.isfinite(prev) else min(prev, candidate)
            pos["trailing_short"] = stop
            return stop
        stop_offset = avg_entry * cfg.sl_pct / 100.0
        return float(pos["imba_line"] - stop_offset) if direction > 0 else float(pos["imba_line"] + stop_offset)

    def close_units(units: float, price: float, reason: str, idx: int) -> None:
        nonlocal pos
        assert pos is not None
        units = min(float(units), float(pos["units"]))
        if units <= 0.0:
            return
        direction = int(pos["direction"])
        avg_entry = float(pos["avg_entry"])
        pos["realized_pnl"] += direction * (float(price) - avg_entry) * units
        pos["fees"] += fee_bps_per_side / 10000.0 * float(price) * units
        pos["units"] = float(pos["units"]) - units
        if pos["units"] <= 1e-9:
            risk_budget = max(float(pos["max_units"]), 1.0) * max(float(pos["initial_risk"]), 1e-12)
            r_multiple = (float(pos["realized_pnl"]) - float(pos["fees"])) / risk_budget
            trades.append((float(r_multiple), int(pos["entry_time_ns"])))
            pos = None

    def add_dca(level: float) -> None:
        assert pos is not None
        units = 1.0
        old_units = float(pos["units"])
        new_units = old_units + units
        pos["avg_entry"] = (float(pos["avg_entry"]) * old_units + float(level) * units) / new_units
        pos["fees"] += fee_bps_per_side / 10000.0 * float(level) * units
        pos["units"] = new_units
        pos["max_units"] = max(float(pos["max_units"]), new_units)

    def process_bar(idx: int) -> None:
        nonlocal pos
        if pos is None:
            return
        direction = int(pos["direction"])
        stop = current_stop(idx)
        stop_hit = cache.low[idx] <= stop if direction > 0 else cache.high[idx] >= stop
        if stop_hit:
            close_units(float(pos["units"]), stop, "stop", idx)
            return

        if cfg.dca_count > 1:
            for level_idx, level in enumerate(pos["dca_levels"]):
                if level_idx in pos["filled_dca"]:
                    continue
                hit = cache.low[idx] <= level if direction > 0 else cache.high[idx] >= level
                if hit:
                    add_dca(float(level))
                    pos["filled_dca"].add(level_idx)
                    stop = current_stop(idx)
                    stop_hit_after = cache.low[idx] <= stop if direction > 0 else cache.high[idx] >= stop
                    if stop_hit_after:
                        close_units(float(pos["units"]), stop, "stop_after_dca", idx)
                        return

        avg_entry = float(pos["avg_entry"])
        for tp_idx, (tp_pct, qty_pct) in enumerate(zip(cfg.tp_pcts, cfg.qty_pcts)):
            if pos["tp_hit"][tp_idx]:
                continue
            target = avg_entry * (1.0 + direction * float(tp_pct) / 100.0)
            hit = cache.high[idx] >= target if direction > 0 else cache.low[idx] <= target
            if not hit:
                continue
            close_units(float(pos["units"]) * float(qty_pct) / 100.0, target, f"tp{tp_idx + 1}", idx)
            if pos is None:
                return
            pos["tp_hit"][tp_idx] = True
            if tp_idx == 0:
                pos["tp1_hit"] = True

        if cfg.max_hold_bars > 0 and pos is not None and idx - int(pos["entry_idx"]) >= cfg.max_hold_bars:
            close_units(float(pos["units"]), float(cache.close[idx]), "timeout", idx)

    for idx in range(n - 1):
        process_bar(idx)
        long_signal = bool(long_flip[idx] and filt[idx] and math.isfinite(imba[idx]))
        short_signal = bool(short_flip[idx] and filt[idx] and math.isfinite(imba[idx]))
        if not long_signal and not short_signal:
            continue
        signal_direction = 1 if long_signal else -1
        if pos is None:
            open_position(idx, signal_direction)
            continue
        if int(pos["direction"]) == signal_direction:
            overlapped += 1
            continue
        if cfg.opposite_reverse:
            reverse_price = float(cache.open_[idx + 1])
            close_units(float(pos["units"]), reverse_price, "opposite", idx + 1)
            open_position(idx, signal_direction)
        else:
            overlapped += 1
    if pos is not None:
        close_units(float(pos["units"]), float(cache.close[-1]), "end", n - 1)
    return trades, invalid, overlapped


def summarize(configs: list[GgConfig], aggs: dict[str, ConfigAgg], min_oos_trades: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cfg in configs:
        agg = aggs[cfg.name]
        row = {
            "spec_name": cfg.name,
            "strategy": "ggshot_227",
            "timeframe": cfg.timeframe,
            "preset_symbol": cfg.symbol,
            "exec_mode": cfg.exec_mode,
            "bb_period": cfg.bb_period,
            "bb_dev": cfg.bb_dev,
            "sensitivity": cfg.sensitivity,
            "filter_mode": cfg.filter_mode,
            "tp_pct": cfg.tp_pct,
            "sl_pct": cfg.sl_pct,
            "symbols": len(agg.by_symbol),
            **{f"train_{k}": v for k, v in agg.train.metrics().items()},
            **{f"oos_{k}": v for k, v in agg.oos.metrics().items()},
            **{f"all_{k}": v for k, v in agg.all.metrics().items()},
            "profitable_symbols": sum(1 for symbol_agg in agg.by_symbol.values() if symbol_agg.net_r > 0.0),
            "invalid": agg.invalid,
            "overlapped": agg.overlapped,
        }
        row["score"] = (
            row["oos_avg_r"] * math.sqrt(max(row["oos_trades"], 1))
            + 0.15 * min(row["oos_profit_factor"], 3.0)
            - 0.015 * row["oos_max_dd_r"]
        )
        row["eligible"] = bool(row["oos_trades"] >= min_oos_trades and row["train_trades"] >= min_oos_trades * 2)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["eligible", "score", "oos_net_r"], ascending=[False, False, False])


def by_symbol_frame(configs: list[GgConfig], aggs: dict[str, ConfigAgg]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cfg in configs:
        for symbol, agg in sorted(aggs[cfg.name].by_symbol.items()):
            rows.append({"spec_name": cfg.name, "symbol": symbol, **agg.metrics()})
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    train_start = pd.Timestamp(parse_utc_datetime(args.train_start))
    split = pd.Timestamp(parse_utc_datetime(args.split))
    end = pd.Timestamp(parse_utc_datetime(args.end))
    split_ns = int(split.value)
    configs = build_configs(args)
    if args.grid_mode == "pine":
        symbols = sorted({cfg.symbol for cfg in configs if cfg.symbol})
    else:
        symbols = [clean_symbol(x) for x in parse_list(args.symbols)] if args.symbols.strip() else load_universe(args.universe, args.max_symbols)
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    aggs = {cfg.name: new_config_agg() for cfg in configs}
    print(f"GGShot 227 configs={len(configs)} symbols={len(symbols)} timeframes={args.timeframes} split={split.date()}", flush=True)

    for sym_idx, symbol in enumerate(symbols, start=1):
        print(f"[{sym_idx}/{len(symbols)}] {symbol}", flush=True)
        if args.fetch_missing and find_cache_path(symbol, args.cache_dir, "5m") is None:
            try:
                cache_path = ensure_5m_cache(
                    symbol,
                    cache_dir=args.cache_dir,
                    start=parse_utc_datetime(args.cache_start),
                    end=end.to_pydatetime(),
                    base_url=args.base_url,
                )
                print(f"  fetched cache: {cache_path}", flush=True)
            except Exception as exc:
                print(f"  failed fetch: {type(exc).__name__}: {exc}", flush=True)
                continue
        try:
            base = load_frame(symbol, args.cache_dir, train_start, end)
        except Exception as exc:
            print(f"  failed load: {type(exc).__name__}: {exc}", flush=True)
            continue
        for timeframe in sorted({cfg.timeframe for cfg in configs}):
            frame = resample_local(base, timeframe)
            frame = frame[frame["open_time"] >= train_start - pd.Timedelta(days=10)].reset_index(drop=True)
            cache = make_frame_cache(frame)
            tf_configs = [cfg for cfg in configs if cfg.timeframe == timeframe and (not cfg.symbol or cfg.symbol == symbol)]
            if not tf_configs:
                continue
            flip_cache: dict[tuple[int, float], tuple[np.ndarray, np.ndarray]] = {}
            imba_cache: dict[int, np.ndarray] = {}
            filter_cache: dict[str, np.ndarray] = {}
            for cfg in tf_configs:
                flips_key = (cfg.bb_period, cfg.bb_dev)
                if flips_key not in flip_cache:
                    flip_cache[flips_key] = gg_flips(cache, cfg.bb_period, cfg.bb_dev)
                if cfg.sensitivity not in imba_cache:
                    imba_cache[cfg.sensitivity] = rolling_midline(cache, cfg.sensitivity)
                if cfg.filter_mode not in filter_cache:
                    filter_cache[cfg.filter_mode] = filter_mask(cache, cfg.filter_mode)
                trades, invalid, overlapped = simulate_config_symbol(
                    cache,
                    cfg,
                    *flip_cache[flips_key],
                    imba_cache[cfg.sensitivity],
                    filter_cache[cfg.filter_mode],
                    fee_bps_per_side=args.fee_bps_per_side,
                    min_risk_pct=args.min_risk_pct,
                )
                agg = aggs[cfg.name]
                agg.invalid += invalid
                agg.overlapped += overlapped
                for r_multiple, entry_time_ns in trades:
                    add_trade(agg, symbol, r_multiple, entry_time_ns, split_ns)

    summary = summarize(configs, aggs, args.min_oos_trades)
    by_symbol = by_symbol_frame(configs, aggs)
    summary_path = args.out_prefix.with_name(f"{args.out_prefix.name}_summary.csv")
    by_symbol_path = args.out_prefix.with_name(f"{args.out_prefix.name}_by_symbol.csv")
    summary.to_csv(summary_path, index=False)
    by_symbol.to_csv(by_symbol_path, index=False)
    print(f"Saved summary: {summary_path}", flush=True)
    print(f"Saved by-symbol: {by_symbol_path}", flush=True)
    if not summary.empty:
        cols = ["timeframe", "oos_trades", "oos_net_r", "oos_avg_r", "oos_profit_factor", "profitable_symbols", "spec_name"]
        print(summary.head(15)[cols].to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lean GGShot 227 migration/backtest.")
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--max-symbols", type=int, default=50)
    parser.add_argument("--symbols", default="")
    parser.add_argument("--timeframes", default="15m,30m,45m,1h")
    parser.add_argument("--train-start", default="2024-01-01")
    parser.add_argument("--split", default="2025-07-01")
    parser.add_argument("--end", default="2026-04-20")
    parser.add_argument("--grid-mode", choices=["smoke", "fast", "full", "pine"], default="fast")
    parser.add_argument("--execution-modes", default="pine,be,trailing,dca2,dca2_be,dca2_trailing")
    parser.add_argument("--fetch-missing", action="store_true", help="Fetch missing Bybit linear 5m cache files before testing.")
    parser.add_argument("--cache-start", default="2021-09-01", help="Start date for fetched 5m cache files.")
    parser.add_argument("--base-url", default=BYBIT_URL)
    parser.add_argument("--fee-bps-per-side", type=float, default=6.5)
    parser.add_argument("--min-risk-pct", type=float, default=0.15)
    parser.add_argument("--min-oos-trades", type=int, default=80)
    parser.add_argument("--out-prefix", type=Path, default=Path("scripts/ggshot_227_research"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
