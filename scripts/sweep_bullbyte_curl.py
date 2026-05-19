from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backtest_turtle_soup import parse_utc_datetime  # noqa: E402
from scripts.experiment_pine_strategy_candidates import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    DEFAULT_UNIVERSE,
    clean_symbol,
    load_frame,
    load_universe,
    resample_frame,
    rma,
    sma,
    ema,
    rsi,
    true_range,
)


DEFAULT_OUT_PREFIX = Path("scripts/bullbyte_curl_top50_15m")

FEATURE_NAMES = [
    "atr_pctile",
    "vol_ratio",
    "ema200_dist",
    "ema200_slope",
    "body_ratio",
    "sma13_dist",
    "hour_utc",
    "direction",
    "mom5",
    "atr_norm",
    "day_of_week",
    "rsi14",
    "rsi4h",
]


@dataclass(frozen=True)
class SignalSpec:
    profile: str
    timeframe: str
    comp_min_bars: int
    comp_max_bars: int
    atr_contraction: float
    extreme_zone: float
    touch_mode: str
    catalyst_mode: bool
    bg_atr_period: int
    session_lookback: int

    @property
    def name(self) -> str:
        return (
            f"{self.profile}_{self.timeframe}"
            f"_min{self.comp_min_bars}_max{self.comp_max_bars}"
            f"_ctr{self.atr_contraction:g}_ext{self.extreme_zone:g}"
            f"_touch{self.touch_mode}_cat{int(self.catalyst_mode)}"
            f"_bg{self.bg_atr_period}_sess{self.session_lookback}"
        ).replace(".", "p")


@dataclass(frozen=True)
class ExitSpec:
    atr_period: int
    sl_buffer: float
    sl_min_dist: float
    sl_max_dist: float
    tp3_r: float
    post_outcome_gap: int

    @property
    def name(self) -> str:
        return (
            f"atr{self.atr_period}_buf{self.sl_buffer:g}"
            f"_min{self.sl_min_dist:g}_max{self.sl_max_dist:g}"
            f"_tp3{self.tp3_r:g}_gap{self.post_outcome_gap}"
        ).replace(".", "p")


@dataclass(frozen=True)
class FullSpec:
    signal: SignalSpec
    exit: ExitSpec

    @property
    def name(self) -> str:
        return f"{self.signal.name}__{self.exit.name}"

    @property
    def params(self) -> dict[str, Any]:
        return {"signal": asdict(self.signal), "exit": asdict(self.exit)}


def signal_spec_from_dict(raw: dict[str, Any]) -> SignalSpec:
    return SignalSpec(
        profile=str(raw["profile"]),
        timeframe=str(raw["timeframe"]),
        comp_min_bars=int(raw["comp_min_bars"]),
        comp_max_bars=int(raw["comp_max_bars"]),
        atr_contraction=float(raw["atr_contraction"]),
        extreme_zone=float(raw["extreme_zone"]),
        touch_mode=str(raw["touch_mode"]),
        catalyst_mode=bool(raw["catalyst_mode"]),
        bg_atr_period=int(raw["bg_atr_period"]),
        session_lookback=int(raw["session_lookback"]),
    )


def exit_spec_from_dict(raw: dict[str, Any]) -> ExitSpec:
    return ExitSpec(
        atr_period=int(raw["atr_period"]),
        sl_buffer=float(raw["sl_buffer"]),
        sl_min_dist=float(raw["sl_min_dist"]),
        sl_max_dist=float(raw["sl_max_dist"]),
        tp3_r=float(raw["tp3_r"]),
        post_outcome_gap=int(raw["post_outcome_gap"]),
    )


@dataclass
class Candidate:
    symbol: str
    signal_spec_name: str
    timeframe: str
    signal_index: int
    signal_time: pd.Timestamp
    direction: int
    entry_price: float
    watch_high: float
    watch_low: float
    watch_len: int
    comp_len: int
    local_atr: float
    bg_atr: float
    session_position: float
    feature_values: list[float]


@dataclass
class Trade:
    symbol: str
    strategy: str
    spec_name: str
    signal_spec_name: str
    exit_spec_name: str
    timeframe: str
    direction: str
    signal_index: int
    entry_index: int
    exit_index: int
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    stop_price: float
    target_price: float
    exit_price: float
    exit_reason: str
    r_multiple: float
    gross_r: float
    fee_r: float
    risk_pct: float
    bars_held: int
    feature_json: str

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for key in ("signal_time", "entry_time", "exit_time"):
            out[key] = pd.Timestamp(out[key]).isoformat()
        return out


def parse_list(raw: str) -> list[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def safe_div(num: float, den: float) -> float:
    if not math.isfinite(num) or not math.isfinite(den) or abs(den) < 1e-12:
        return math.nan
    return float(num / den)


def max_dd(values: pd.Series | np.ndarray | list[float]) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return 0.0
    curve = np.cumsum(arr)
    peaks = np.maximum.accumulate(np.insert(curve, 0, 0.0))[1:]
    return float(np.max(peaks - curve)) if curve.size else 0.0


def profit_factor(values: pd.Series | np.ndarray | list[float]) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return 0.0
    gains = float(arr[arr > 0].sum())
    losses = float(-arr[arr < 0].sum())
    if losses <= 0:
        return 99.0 if gains > 0 else 0.0
    return float(gains / losses)


def metrics_from_values(r_values: list[float], entry_times: list[pd.Timestamp]) -> dict[str, float]:
    if not r_values:
        return {
            "trades": 0,
            "net_r": 0.0,
            "avg_r": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_dd_r": 0.0,
            "trades_per_week": 0.0,
        }
    r = np.asarray(r_values, dtype=float)
    times = pd.to_datetime(pd.Series(entry_times), utc=True, errors="coerce")
    if times.notna().sum() >= 2:
        weeks = max((times.max() - times.min()).total_seconds() / (86400.0 * 7.0), 1e-9)
    else:
        weeks = 1e-9
    return {
        "trades": int(len(r)),
        "net_r": float(r.sum()),
        "avg_r": float(r.mean()),
        "win_rate": float((r > 0).mean()),
        "profit_factor": profit_factor(r),
        "max_dd_r": max_dd(r),
        "trades_per_week": float(len(r) / weeks),
    }


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return metrics_from_values([], [])
    return metrics_from_values(
        pd.to_numeric(frame["r_multiple"], errors="coerce").dropna().to_list(),
        pd.to_datetime(frame["entry_time"], utc=True, errors="coerce").to_list(),
    )


def score_metrics(m: dict[str, float]) -> float:
    trades = float(m.get("trades", 0.0) or 0.0)
    if trades <= 0:
        return -1e9
    pf = min(float(m.get("profit_factor", 0.0) or 0.0), 4.0)
    avg_r = float(m.get("avg_r", 0.0) or 0.0)
    dd = float(m.get("max_dd_r", 0.0) or 0.0)
    return avg_r * math.sqrt(min(trades, 800.0)) + 0.14 * pf - 0.0035 * dd


def prefixed(m: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in m.items()}


def rolling_percentile_exclusive(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(values.shape, 50.0, dtype=float)
    for i in range(window, len(values)):
        cur = values[i]
        if not math.isfinite(float(cur)):
            continue
        hist = values[i - window : i]
        valid = hist[np.isfinite(hist)]
        if valid.size:
            out[i] = float((valid < cur).sum() / valid.size * 100.0)
    return out


def htf_rsi_from_frame(frame: pd.DataFrame, htf: str = "4h", length: int = 14) -> np.ndarray:
    indexed = frame.set_index("open_time")
    closes = indexed["close"].resample(htf, closed="right", label="right").last().dropna()
    if closes.empty:
        return np.full(len(frame), 50.0, dtype=float)
    htf_rsi = pd.Series(rsi(closes.to_numpy(dtype=float), length), index=closes.index)
    return htf_rsi.reindex(pd.DatetimeIndex(frame["open_time"]), method="ffill").fillna(50.0).to_numpy(dtype=float)


class PreparedFrame:
    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        local_lengths: set[int],
        bg_lengths: set[int],
        atr_lengths: set[int],
        session_lookbacks: set[int],
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.open = self.frame["open"].to_numpy(dtype=float)
        self.high = self.frame["high"].to_numpy(dtype=float)
        self.low = self.frame["low"].to_numpy(dtype=float)
        self.close = self.frame["close"].to_numpy(dtype=float)
        self.volume = self.frame["volume"].to_numpy(dtype=float)
        self.open_time = pd.to_datetime(self.frame["open_time"], utc=True, errors="coerce")
        self.close_time = pd.to_datetime(self.frame["close_time"], utc=True, errors="coerce")
        self.n = len(self.frame)
        self.tr = true_range(self.high, self.low, self.close)
        self.atr_by_len = {length: rma(self.tr, length) for length in sorted(atr_lengths | bg_lengths | local_lengths | {14})}
        self.local_atr = {length: self.atr_by_len[length] for length in sorted(local_lengths)}
        self.bg_atr = {length: self.atr_by_len[length] for length in sorted(bg_lengths)}
        self.signal_atr = {length: self.atr_by_len[length] for length in sorted(atr_lengths)}
        self.session_high = {
            length: pd.Series(self.high).rolling(length, min_periods=length).max().to_numpy(dtype=float)
            for length in sorted(session_lookbacks)
        }
        self.session_low = {
            length: pd.Series(self.low).rolling(length, min_periods=length).min().to_numpy(dtype=float)
            for length in sorted(session_lookbacks)
        }
        self.median_range50 = (
            pd.Series(self.high - self.low)
            .rolling(50, min_periods=50)
            .quantile(0.5, interpolation="linear")
            .to_numpy(dtype=float)
        )
        self.feature_arrays = self._feature_arrays()

    def _feature_arrays(self) -> dict[str, np.ndarray]:
        atr14 = self.atr_by_len[14]
        ema200 = ema(self.close, 200)
        sma13 = sma(self.close, 13)
        rsi14 = rsi(self.close, 14)
        rsi4h = htf_rsi_from_frame(self.frame, "4h", 14)
        vol_sma20 = sma(self.volume, 20)
        vol_ratio = np.divide(
            self.volume,
            vol_sma20,
            out=np.ones_like(self.volume, dtype=float),
            where=np.isfinite(vol_sma20) & (vol_sma20 > 0),
        )
        return {
            "atr14": atr14,
            "atr_pctile": rolling_percentile_exclusive(atr14, 100),
            "vol_ratio": np.where(np.isfinite(vol_ratio), vol_ratio, 1.0),
            "ema200": ema200,
            "sma13": sma13,
            "rsi14": rsi14,
            "rsi4h": rsi4h,
        }


def feature_values(prep: PreparedFrame, idx: int, direction: int) -> list[float]:
    close = prep.close
    high = prep.high
    low = prep.low
    open_ = prep.open
    fa = prep.feature_arrays
    ema200 = fa["ema200"]
    sma13 = fa["sma13"]
    atr14 = fa["atr14"]
    ts = pd.Timestamp(prep.close_time.iloc[idx])
    e10 = ema200[max(0, idx - 10)]
    prior5 = close[max(0, idx - 5)]
    rng = high[idx] - low[idx]
    return [
        float(fa["atr_pctile"][idx] / 100.0) if math.isfinite(float(fa["atr_pctile"][idx])) else 0.5,
        float(fa["vol_ratio"][idx]) if math.isfinite(float(fa["vol_ratio"][idx])) else 1.0,
        safe_div(close[idx] - ema200[idx], close[idx]) if math.isfinite(float(ema200[idx])) else 0.0,
        safe_div(ema200[idx] - e10, e10) if math.isfinite(float(e10)) and e10 > 0 else 0.0,
        safe_div(abs(close[idx] - open_[idx]), rng) if rng > 0 else 0.0,
        safe_div(abs(close[idx] - sma13[idx]), close[idx]) if math.isfinite(float(sma13[idx])) else 0.0,
        float(ts.hour / 23.0),
        1.0 if direction > 0 else 0.0,
        safe_div(close[idx] - prior5, prior5) if prior5 > 0 else 0.0,
        safe_div(atr14[idx], close[idx]) if math.isfinite(float(atr14[idx])) else 0.0,
        float(ts.dayofweek / 6.0),
        float(fa["rsi14"][idx] / 100.0) if math.isfinite(float(fa["rsi14"][idx])) else 0.5,
        float(fa["rsi4h"][idx] / 100.0) if math.isfinite(float(fa["rsi4h"][idx])) else 0.5,
    ]


def build_signal_specs(grid_mode: str, timeframes: list[str]) -> list[SignalSpec]:
    specs: dict[str, SignalSpec] = {}

    def add(spec: SignalSpec) -> None:
        specs.setdefault(spec.name, spec)

    preset_rows = [
        ("conservative", 5, 12, 0.70, 0.25, "Full"),
        ("balanced", 4, 14, 0.82, 0.35, "Edge"),
        ("aggressive", 3, 16, 0.92, 0.45, "Mid"),
    ]
    if grid_mode == "smoke":
        bg_periods = [20]
        lookbacks = [50]
        catalyst_modes = [True]
        manual_rows: list[tuple[str, int, int, float, float, str]] = []
    elif grid_mode == "fast":
        bg_periods = [20]
        lookbacks = [36, 50, 72]
        catalyst_modes = [True]
        manual_rows = [
            ("edge_loose", 3, 16, 0.92, 0.45, "Edge"),
            ("edge_tight", 5, 12, 0.70, 0.25, "Edge"),
        ]
    else:
        bg_periods = [20]
        lookbacks = [36, 50, 72]
        catalyst_modes = [True]
        manual_rows = [
            ("edge_fast", 3, 12, 0.82, 0.35, "Edge"),
            ("edge_loose", 3, 16, 0.92, 0.45, "Edge"),
            ("edge_tight", 5, 12, 0.70, 0.25, "Edge"),
        ]

    for tf in timeframes:
        for row in preset_rows + manual_rows:
            profile, comp_min, comp_max, contraction, extreme, touch = row
            for catalyst in catalyst_modes:
                for bg_period in bg_periods:
                    for lookback in lookbacks:
                        add(
                            SignalSpec(
                                profile=profile,
                                timeframe=tf,
                                comp_min_bars=comp_min,
                                comp_max_bars=comp_max,
                                atr_contraction=contraction,
                                extreme_zone=extreme,
                                touch_mode=touch,
                                catalyst_mode=catalyst,
                                bg_atr_period=bg_period,
                                session_lookback=lookback,
                            )
                        )
    return list(specs.values())


def build_exit_specs(grid_mode: str) -> list[ExitSpec]:
    if grid_mode == "smoke":
        atr_periods = [14]
        buffers = [0.2]
        mins = [0.3]
        maxes = [2.0]
        targets = [2.0]
        gaps = [8]
    elif grid_mode == "fast":
        atr_periods = [14]
        buffers = [0.1, 0.2]
        mins = [0.2, 0.3]
        maxes = [2.0]
        targets = [1.5, 2.0, 2.5]
        gaps = [8]
    else:
        atr_periods = [14]
        buffers = [0.1, 0.2, 0.35]
        mins = [0.2, 0.3]
        maxes = [2.0]
        targets = [1.5, 2.0, 2.5, 3.0]
        gaps = [8]
    out: list[ExitSpec] = []
    for atr_period in atr_periods:
        for buffer in buffers:
            for min_dist in mins:
                for max_dist in maxes:
                    if min_dist >= max_dist:
                        continue
                    for target in targets:
                        for gap in gaps:
                            out.append(ExitSpec(atr_period, buffer, min_dist, max_dist, target, gap))
    return out


def max_violations(touch_mode: str) -> int:
    if touch_mode == "Full":
        return 1
    if touch_mode == "Edge":
        return 2
    return 3


def tolerance_multiplier(touch_mode: str) -> float:
    if touch_mode == "Full":
        return 0.30
    if touch_mode == "Edge":
        return 0.50
    return 0.75


def watch_expiry(touch_mode: str) -> int:
    if touch_mode == "Full":
        return 5
    if touch_mode == "Edge":
        return 10
    return 15


def generate_candidates(symbol: str, prep: PreparedFrame, spec: SignalSpec) -> list[Candidate]:
    n = prep.n
    if n < max(spec.session_lookback, 100) + 5:
        return []
    high = prep.high
    low = prep.low
    open_ = prep.open
    close = prep.close
    local_atr = prep.local_atr[max(spec.comp_min_bars, 3)]
    bg_atr = prep.bg_atr[spec.bg_atr_period]
    session_hi = prep.session_high[spec.session_lookback]
    session_lo = prep.session_low[spec.session_lookback]
    median_range = prep.median_range50

    comp_len = 0
    comp_high = math.nan
    comp_low = math.nan
    comp_start = -1
    comp_violations = 0
    watch_active = False
    watch_is_short = False
    watch_high = math.nan
    watch_low = math.nan
    watch_len = 0
    watch_end = -1
    out: list[Candidate] = []
    max_v = max_violations(spec.touch_mode)
    tol_mult = tolerance_multiplier(spec.touch_mode)
    expiry = watch_expiry(spec.touch_mode)

    for i in range(n):
        if not (
            math.isfinite(float(local_atr[i]))
            and math.isfinite(float(bg_atr[i]))
            and bg_atr[i] > 0
            and math.isfinite(float(median_range[i]))
            and math.isfinite(float(session_hi[i]))
            and math.isfinite(float(session_lo[i]))
        ):
            continue

        ratio_contracted = local_atr[i] < bg_atr[i] * spec.atr_contraction
        drift_contracted = (high[i] - low[i]) < median_range[i] * 0.65
        is_contracted = bool(ratio_contracted or drift_contracted)
        if spec.catalyst_mode:
            recent_expansion = False
            for k in range(1, 6):
                j = i - k
                if j >= 0 and math.isfinite(float(bg_atr[j])) and (high[j] - low[j]) > bg_atr[j] * 1.5:
                    recent_expansion = True
                    break
            is_contracted = bool(is_contracted or (recent_expansion and local_atr[i] < bg_atr[i] * 1.10))

        if is_contracted:
            if comp_len == 0:
                comp_len = 1
                comp_high = float(high[i])
                comp_low = float(low[i])
                comp_start = i
                comp_violations = 0
            else:
                tol = bg_atr[i] * tol_mult
                in_range = high[i] <= comp_high + tol and low[i] >= comp_low - tol
                if in_range:
                    comp_len += 1
                    comp_high = max(comp_high, float(high[i]))
                    comp_low = min(comp_low, float(low[i]))
                else:
                    comp_violations += 1
                    if comp_violations <= max_v:
                        comp_len += 1
                        comp_high = max(comp_high, float(high[i]))
                        comp_low = min(comp_low, float(low[i]))
                    else:
                        comp_len = 1
                        comp_high = float(high[i])
                        comp_low = float(low[i])
                        comp_start = i
                        comp_violations = 0
        else:
            if comp_len > 0:
                comp_len -= 1
            if comp_len == 0:
                comp_high = math.nan
                comp_low = math.nan
                comp_start = -1
                comp_violations = 0

        if comp_len > spec.comp_max_bars:
            comp_len = 0
            comp_high = math.nan
            comp_low = math.nan
            comp_start = -1
            comp_violations = 0

        comp_qualified = comp_len >= spec.comp_min_bars
        session_rng = session_hi[i] - session_lo[i]
        comp_at_high = False
        comp_at_low = False
        if comp_qualified and session_rng > 0 and math.isfinite(comp_high) and math.isfinite(comp_low):
            zone_top = session_hi[i] - session_rng * spec.extreme_zone
            zone_bot = session_lo[i] + session_rng * spec.extreme_zone
            comp_mid = (comp_high + comp_low) / 2.0
            if spec.touch_mode == "Full":
                comp_at_high = comp_low >= zone_top
                comp_at_low = comp_high <= zone_bot
            elif spec.touch_mode == "Edge":
                comp_at_high = comp_high >= zone_top
                comp_at_low = comp_low <= zone_bot
            else:
                comp_at_high = comp_high >= zone_top or comp_mid >= zone_top or close[i] >= zone_top
                comp_at_low = comp_low <= zone_bot or comp_mid <= zone_bot or close[i] <= zone_bot

        if comp_at_high and not watch_active:
            watch_active = True
            watch_is_short = True
            watch_high = comp_high
            watch_low = comp_low
            watch_len = comp_len
            watch_end = -1
        elif comp_at_low and not watch_active:
            watch_active = True
            watch_is_short = False
            watch_high = comp_high
            watch_low = comp_low
            watch_len = comp_len
            watch_end = -1

        if watch_active and comp_qualified:
            if watch_is_short and comp_at_high:
                watch_high = max(watch_high, comp_high)
                watch_low = min(watch_low, comp_low)
                watch_len = comp_len
            elif (not watch_is_short) and comp_at_low:
                watch_high = max(watch_high, comp_high)
                watch_low = min(watch_low, comp_low)
                watch_len = comp_len

        if watch_active and (not comp_qualified) and watch_end < 0:
            watch_end = i

        if watch_active and watch_end >= 0 and (i - watch_end) > expiry:
            watch_active = False
            watch_high = math.nan
            watch_low = math.nan
            watch_len = 0
            watch_end = -1

        if not watch_active or not math.isfinite(watch_high) or not math.isfinite(watch_low):
            continue

        candle_range = high[i] - low[i]
        if spec.touch_mode == "Full":
            short_break = close[i] < watch_low
            long_break = close[i] > watch_high
        elif spec.touch_mode == "Edge":
            short_break = close[i] < watch_low or (
                low[i] < watch_low and close[i] < open_[i] and (open_[i] - close[i]) > candle_range * 0.40
            )
            long_break = close[i] > watch_high or (
                high[i] > watch_high and close[i] > open_[i] and (close[i] - open_[i]) > candle_range * 0.40
            )
        else:
            short_break = close[i] < watch_low or low[i] < watch_low - bg_atr[i] * 0.05
            long_break = close[i] > watch_high or high[i] > watch_high + bg_atr[i] * 0.05

        direction = 0
        if watch_is_short and short_break:
            direction = -1
        elif (not watch_is_short) and long_break:
            direction = 1
        if direction == 0:
            continue

        session_pos = safe_div(close[i] - session_lo[i], session_rng) if session_rng > 0 else math.nan
        out.append(
            Candidate(
                symbol=symbol,
                signal_spec_name=spec.name,
                timeframe=spec.timeframe,
                signal_index=i,
                signal_time=pd.Timestamp(prep.close_time.iloc[i]),
                direction=direction,
                entry_price=float(close[i]),
                watch_high=float(watch_high),
                watch_low=float(watch_low),
                watch_len=int(watch_len),
                comp_len=int(comp_len),
                local_atr=float(local_atr[i]),
                bg_atr=float(bg_atr[i]),
                session_position=float(session_pos) if math.isfinite(session_pos) else math.nan,
                feature_values=feature_values(prep, i, direction),
            )
        )
        watch_active = False
        watch_high = math.nan
        watch_low = math.nan
        watch_len = 0
        watch_end = -1

    return out


def calc_levels(candidate: Candidate, atr_value: float, exit_spec: ExitSpec) -> tuple[float, float, float]:
    entry = candidate.entry_price
    buffer = atr_value * exit_spec.sl_buffer
    min_dist = atr_value * exit_spec.sl_min_dist
    max_dist = atr_value * exit_spec.sl_max_dist
    raw_sl = candidate.watch_low - buffer if candidate.direction > 0 else candidate.watch_high + buffer
    raw_dist = abs(entry - raw_sl)
    final_dist = max(min_dist, min(raw_dist, max_dist))
    if candidate.direction > 0:
        return entry - final_dist, entry + final_dist * exit_spec.tp3_r, final_dist
    return entry + final_dist, entry - final_dist * exit_spec.tp3_r, final_dist


def simulate_candidates(
    symbol: str,
    prep: PreparedFrame,
    signal_spec: SignalSpec,
    exit_spec: ExitSpec,
    candidates: list[Candidate],
    *,
    fee_bps_per_side: float,
    min_risk_pct: float,
) -> list[Trade]:
    if not candidates:
        return []
    high = prep.high
    low = prep.low
    close = prep.close
    atr = prep.signal_atr[exit_spec.atr_period]
    n = prep.n
    spec_name = f"{signal_spec.name}__{exit_spec.name}"
    out: list[Trade] = []
    blocked_until = -1
    for candidate in candidates:
        i = candidate.signal_index
        if i >= n - 1 or i <= blocked_until + exit_spec.post_outcome_gap:
            continue
        atr_value = float(atr[i])
        if not math.isfinite(atr_value) or atr_value <= 0:
            continue
        stop, target, risk = calc_levels(candidate, atr_value, exit_spec)
        entry = candidate.entry_price
        if risk <= 0 or entry <= 0:
            continue
        risk_pct = risk / entry * 100.0
        if risk_pct < min_risk_pct:
            continue

        entry_idx = i
        start = i + 1
        exit_idx = n - 1
        exit_price = float(close[-1])
        exit_reason = "open"
        if candidate.direction > 0:
            sl_hits = np.flatnonzero(low[start:n] <= stop)
            tp_hits = np.flatnonzero(high[start:n] >= target)
        else:
            sl_hits = np.flatnonzero(high[start:n] >= stop)
            tp_hits = np.flatnonzero(low[start:n] <= target)
        first_sl = int(sl_hits[0]) if sl_hits.size else None
        first_tp = int(tp_hits[0]) if tp_hits.size else None
        if first_sl is not None or first_tp is not None:
            if first_sl is not None and (first_tp is None or first_sl <= first_tp):
                exit_idx = start + first_sl
                exit_price = stop
                exit_reason = "sl"
            else:
                exit_idx = start + int(first_tp)
                exit_price = target
                exit_reason = "tp3"

        gross_r = candidate.direction * (exit_price - entry) / risk
        fee_r = (2.0 * fee_bps_per_side / 10000.0) * entry / risk
        features = dict(zip(FEATURE_NAMES, candidate.feature_values))
        features.update(
            {
                "bb_watch_len": float(candidate.watch_len),
                "bb_comp_len": float(candidate.comp_len),
                "bb_local_bg_ratio": safe_div(candidate.local_atr, candidate.bg_atr),
                "bb_session_position": candidate.session_position,
                "bb_watch_height_atr": safe_div(candidate.watch_high - candidate.watch_low, atr_value),
            }
        )
        out.append(
            Trade(
                symbol=symbol,
                strategy="bullbyte_curl",
                spec_name=spec_name,
                signal_spec_name=signal_spec.name,
                exit_spec_name=exit_spec.name,
                timeframe=signal_spec.timeframe,
                direction="long" if candidate.direction > 0 else "short",
                signal_index=i,
                entry_index=entry_idx,
                exit_index=exit_idx,
                signal_time=candidate.signal_time,
                entry_time=pd.Timestamp(prep.close_time.iloc[entry_idx]),
                exit_time=pd.Timestamp(prep.close_time.iloc[exit_idx]),
                entry_price=float(entry),
                stop_price=float(stop),
                target_price=float(target),
                exit_price=float(exit_price),
                exit_reason=exit_reason,
                r_multiple=float(gross_r - fee_r),
                gross_r=float(gross_r),
                fee_r=float(fee_r),
                risk_pct=float(risk_pct),
                bars_held=int(exit_idx - entry_idx),
                feature_json=json.dumps(features, sort_keys=True),
            )
        )
        blocked_until = exit_idx
    return out


def feature_table(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    rows: list[dict[str, Any]] = []
    for _, row in trades.iterrows():
        out = row.to_dict()
        try:
            features = json.loads(row.get("feature_json") or "{}")
        except Exception:
            features = {}
        for name in FEATURE_NAMES:
            out[f"f_{name}"] = features.get(name, math.nan)
        rows.append(out)
    return pd.DataFrame(rows)


def summarize_spec_trades(
    trades: list[Trade],
    *,
    split: pd.Timestamp,
    min_train_trades: int,
    min_oos_trades: int,
) -> dict[str, Any] | None:
    if not trades:
        return None
    first = trades[0]
    r_values = [trade.r_multiple for trade in trades]
    times = [trade.entry_time for trade in trades]
    train_r = [r for r, ts in zip(r_values, times) if pd.Timestamp(ts) < split]
    train_t = [ts for ts in times if pd.Timestamp(ts) < split]
    oos_r = [r for r, ts in zip(r_values, times) if pd.Timestamp(ts) >= split]
    oos_t = [ts for ts in times if pd.Timestamp(ts) >= split]
    all_m = metrics_from_values(r_values, times)
    train_m = metrics_from_values(train_r, train_t)
    oos_m = metrics_from_values(oos_r, oos_t)
    row = {
        "symbol": first.symbol,
        "strategy": first.strategy,
        "timeframe": first.timeframe,
        "spec_name": first.spec_name,
        "signal_spec_name": first.signal_spec_name,
        "exit_spec_name": first.exit_spec_name,
        **prefixed(train_m, "train"),
        **prefixed(oos_m, "oos"),
        **prefixed(all_m, "all"),
    }
    row["train_score"] = score_metrics(train_m)
    row["oos_score"] = score_metrics(oos_m)
    row["train_eligible"] = bool(train_m["trades"] >= min_train_trades)
    row["oos_eligible"] = bool(oos_m["trades"] >= min_oos_trades)
    return row


def select_rows(summary: pd.DataFrame, min_train_trades: int, min_oos_trades: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_rows: list[pd.Series] = []
    oracle_rows: list[pd.Series] = []
    if summary.empty:
        return pd.DataFrame(), pd.DataFrame()
    for _, group in summary.groupby("symbol"):
        train_candidates = group[group["train_trades"] >= min_train_trades]
        if train_candidates.empty:
            train_candidates = group
        selected_rows.append(
            train_candidates.sort_values(["train_score", "train_net_r"], ascending=[False, False]).iloc[0]
        )
        oos_candidates = group[group["oos_trades"] >= min_oos_trades]
        if oos_candidates.empty:
            oos_candidates = group
        oracle_rows.append(oos_candidates.sort_values(["oos_score", "oos_net_r"], ascending=[False, False]).iloc[0])
    selected = pd.DataFrame(selected_rows).sort_values(["oos_score", "oos_net_r"], ascending=[False, False])
    oracle = pd.DataFrame(oracle_rows).sort_values(["oos_score", "oos_net_r"], ascending=[False, False])
    return selected.reset_index(drop=True), oracle.reset_index(drop=True)


def run_symbol_job(payload: tuple[str, dict[str, Any], list[SignalSpec], list[ExitSpec]]) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], str | None]:
    symbol, args_dict, signal_specs, exit_specs = payload
    args = argparse.Namespace(**args_dict)
    train_start = pd.Timestamp(parse_utc_datetime(args.train_start))
    split = pd.Timestamp(parse_utc_datetime(args.split))
    end = pd.Timestamp(parse_utc_datetime(args.end))
    try:
        base = load_frame(symbol, Path(args.cache_dir), train_start, end)
        rows: list[dict[str, Any]] = []
        trade_rows: list[dict[str, Any]] = []
        by_tf: dict[str, PreparedFrame] = {}
        local_lengths = {max(spec.comp_min_bars, 3) for spec in signal_specs}
        bg_lengths = {spec.bg_atr_period for spec in signal_specs}
        atr_lengths = {spec.atr_period for spec in exit_specs}
        session_lookbacks = {spec.session_lookback for spec in signal_specs}
        for tf in sorted({spec.timeframe for spec in signal_specs}):
            frame = resample_frame(base, tf)
            frame = frame[frame["open_time"] >= train_start - pd.Timedelta(days=30)].reset_index(drop=True)
            by_tf[tf] = PreparedFrame(
                frame,
                local_lengths=local_lengths,
                bg_lengths=bg_lengths,
                atr_lengths=atr_lengths,
                session_lookbacks=session_lookbacks,
            )

        for signal_spec in signal_specs:
            prep = by_tf[signal_spec.timeframe]
            candidates = generate_candidates(symbol, prep, signal_spec)
            if not candidates:
                continue
            for exit_spec in exit_specs:
                trades = simulate_candidates(
                    symbol,
                    prep,
                    signal_spec,
                    exit_spec,
                    candidates,
                    fee_bps_per_side=float(args.fee_bps_per_side),
                    min_risk_pct=float(args.min_risk_pct),
                )
                row = summarize_spec_trades(
                    trades,
                    split=split,
                    min_train_trades=int(args.min_train_trades),
                    min_oos_trades=int(args.min_oos_trades),
                )
                if row is None:
                    continue
                full = FullSpec(signal_spec, exit_spec)
                row["params_json"] = json.dumps(full.params, sort_keys=True)
                rows.append(row)
                if args.keep_trades and trades:
                    trade_rows.extend(trade.to_dict() for trade in trades)
        return symbol, rows, trade_rows, None
    except Exception as exc:
        return symbol, [], [], f"{type(exc).__name__}: {exc}"


def train_dt_gate_for_symbol(
    symbol: str,
    spec_name: str,
    trades: pd.DataFrame,
    *,
    split: pd.Timestamp,
    val_frac: float,
    thresholds: list[float],
    min_fit: int,
    min_val: int,
    min_oos: int,
    random_state: int,
    model_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    expanded = feature_table(trades)
    feature_cols = [f"f_{name}" for name in FEATURE_NAMES if f"f_{name}" in expanded.columns]
    for col in feature_cols:
        expanded[col] = pd.to_numeric(expanded[col], errors="coerce")
    expanded["entry_time"] = pd.to_datetime(expanded["entry_time"], utc=True, errors="coerce")
    train_all = expanded[expanded["entry_time"] < split].sort_values("entry_time").copy()
    oos = expanded[expanded["entry_time"] >= split].sort_values("entry_time").copy()
    raw_train_m = metrics(train_all)
    raw_oos_m = metrics(oos)
    base = {
        "symbol": symbol,
        "spec_name": spec_name,
        "feature_schema": ",".join(FEATURE_NAMES),
        "feature_count": len(feature_cols),
        **{f"raw_train_{k}": v for k, v in raw_train_m.items()},
        **{f"raw_oos_{k}": v for k, v in raw_oos_m.items()},
        "status": "skipped",
        "reason": "",
        "model": "",
        "threshold": math.nan,
        "model_path": "",
        "fit_trades": 0,
        "val_trades_raw": 0,
        "min_leaf": 0,
        "ml_val_trades": 0,
        "ml_val_net_r": 0.0,
        "ml_val_avg_r": 0.0,
        "ml_val_win_rate": 0.0,
        "ml_val_profit_factor": 0.0,
        "ml_val_max_dd_r": 0.0,
        "ml_oos_trades": 0,
        "ml_oos_net_r": 0.0,
        "ml_oos_avg_r": 0.0,
        "ml_oos_win_rate": 0.0,
        "ml_oos_profit_factor": 0.0,
        "ml_oos_max_dd_r": 0.0,
        "oos_pf_delta": math.nan,
        "oos_net_r_delta": math.nan,
        "oos_trade_retention": math.nan,
    }
    if len(feature_cols) != len(FEATURE_NAMES):
        base["reason"] = "missing standard features"
        return base, pd.DataFrame(), pd.DataFrame()
    if len(train_all) < min_fit + min_val:
        base["reason"] = f"not enough train trades ({len(train_all)})"
        return base, pd.DataFrame(), pd.DataFrame()
    if len(oos) < min_oos:
        base["reason"] = f"not enough OOS trades ({len(oos)})"
        return base, pd.DataFrame(), pd.DataFrame()

    cut = max(min_fit, int(len(train_all) * (1.0 - val_frac)))
    cut = min(cut, len(train_all) - min_val)
    fit = train_all.iloc[:cut].copy()
    val = train_all.iloc[cut:].copy()
    if fit["r_multiple"].gt(0).nunique() < 2:
        base["reason"] = "fit labels single class"
        return base, pd.DataFrame(), pd.DataFrame()

    min_leaf = max(15, min(80, len(fit) // 20))
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        DecisionTreeClassifier(
            max_depth=2,
            min_samples_leaf=min_leaf,
            class_weight="balanced",
            random_state=random_state,
        ),
    )
    model.fit(fit[feature_cols], fit["r_multiple"].gt(0).astype(int))
    val_scored = val.copy()
    oos_scored = oos.copy()
    val_scored["ml_prob"] = model.predict_proba(val[feature_cols])[:, 1]
    oos_scored["ml_prob"] = model.predict_proba(oos[feature_cols])[:, 1]

    detail_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    min_val_selected = max(min_val, int(len(val) * 0.20))
    for threshold in thresholds:
        sel_val = val_scored[val_scored["ml_prob"] >= threshold].copy()
        sel_oos = oos_scored[oos_scored["ml_prob"] >= threshold].copy()
        val_m = metrics(sel_val)
        oos_m = metrics(sel_oos)
        valid = val_m["trades"] >= min_val_selected and val_m["net_r"] > 0
        val_pf = min(float(val_m["profit_factor"]), 5.0) if math.isfinite(float(val_m["profit_factor"])) else 5.0
        score = (
            val_pf
            + 0.35 * val_m["avg_r"] * math.sqrt(max(val_m["trades"], 1))
            + 0.02 * math.log1p(val_m["trades"])
            - 0.01 * val_m["max_dd_r"]
        )
        row = {
            "symbol": symbol,
            "spec_name": spec_name,
            "model": "decision_tree_depth2",
            "threshold": float(threshold),
            "valid_on_val": bool(valid),
            "score": float(score),
            **{f"val_{k}": v for k, v in val_m.items()},
            **{f"oos_{k}": v for k, v in oos_m.items()},
        }
        detail_rows.append(row)
        if valid and (best is None or score > best["score"]):
            best = row

    if best is None:
        base["reason"] = "no validation-positive threshold"
        return base, pd.DataFrame(detail_rows), pd.DataFrame()

    selected_oos = oos_scored[oos_scored["ml_prob"] >= float(best["threshold"])].copy()
    safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in spec_name)
    model_path = model_dir / f"{symbol}_{safe_name}_dt.joblib"
    payload = {
        "model": model,
        "feature_columns": feature_cols,
        "feature_names": FEATURE_NAMES,
        "threshold": float(best["threshold"]),
        "symbol": symbol,
        "strategy": "bullbyte_curl",
        "spec_name": spec_name,
        "split": split.isoformat(),
        "validation_policy": {
            "val_frac": val_frac,
            "thresholds": thresholds,
            "min_fit": min_fit,
            "min_val": min_val,
            "min_oos": min_oos,
            "min_leaf": min_leaf,
            "max_depth": 2,
        },
    }
    joblib.dump(payload, model_path)

    out = {
        **base,
        "status": "ok",
        "reason": "",
        "model": "decision_tree_depth2",
        "threshold": float(best["threshold"]),
        "model_path": str(model_path),
        "fit_trades": int(len(fit)),
        "val_trades_raw": int(len(val)),
        "min_leaf": int(min_leaf),
        **{f"raw_fit_{k}": v for k, v in metrics(fit).items()},
        **{f"raw_val_{k}": v for k, v in metrics(val).items()},
        **{f"ml_val_{k}": best[f"val_{k}"] for k in ("trades", "net_r", "avg_r", "win_rate", "profit_factor", "max_dd_r")},
        **{f"ml_oos_{k}": best[f"oos_{k}"] for k in ("trades", "net_r", "avg_r", "win_rate", "profit_factor", "max_dd_r")},
    }
    out["oos_pf_delta"] = float(out["ml_oos_profit_factor"] - out["raw_oos_profit_factor"])
    out["oos_net_r_delta"] = float(out["ml_oos_net_r"] - out["raw_oos_net_r"])
    out["oos_trade_retention"] = safe_div(float(out["ml_oos_trades"]), float(out["raw_oos_trades"]))
    return out, pd.DataFrame(detail_rows), selected_oos


def markdown_table(frame: pd.DataFrame, columns: list[str], limit: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    shown = frame.head(limit)[columns].copy()
    for column in shown.columns:
        if pd.api.types.is_float_dtype(shown[column]):
            shown[column] = shown[column].map(
                lambda x: f"{float(x):.4f}" if pd.notna(x) and math.isfinite(float(x)) else str(x)
            )
        else:
            shown[column] = shown[column].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(str(row[column]) for column in columns) + " |" for _, row in shown.iterrows()]
    return "\n".join([header, sep, *body])


def write_report(
    args: argparse.Namespace,
    summary: pd.DataFrame,
    selected: pd.DataFrame,
    oracle: pd.DataFrame,
    ml_summary: pd.DataFrame,
    failures: pd.DataFrame,
) -> Path:
    path = args.out_prefix.with_suffix(".md")
    lines: list[str] = []
    lines.append("# BullByte Curl Research Sweep")
    lines.append("")
    lines.append(f"Universe: `{args.universe}` | max symbols: `{args.max_symbols}` | grid: `{args.grid_mode}` | timeframes: `{args.timeframes}`")
    lines.append(f"Window: `{args.train_start}` to `{args.end}` | split: `{args.split}`")
    lines.append("")
    lines.append(
        "Signal logic ports the BullByte Volatility Coil Edge confirmed-bar coil/watch trigger. "
        "Parameter selection uses train metrics only; OOS and oracle rows are diagnostics."
    )
    lines.append("")
    lines.append("## Selected By Train")
    lines.append("")
    cols = [
        "symbol",
        "train_trades",
        "train_net_r",
        "train_avg_r",
        "train_profit_factor",
        "oos_trades",
        "oos_net_r",
        "oos_avg_r",
        "oos_profit_factor",
        "spec_name",
    ]
    lines.append(markdown_table(selected, cols, 30))
    lines.append("")
    lines.append("## Best OOS Oracle Diagnostic")
    lines.append("")
    lines.append(markdown_table(oracle, cols, 20))
    lines.append("")
    lines.append("## ML Gate Compatibility")
    lines.append("")
    if ml_summary.empty:
        lines.append("_No ML rows._")
    else:
        ok = ml_summary[ml_summary["status"] == "ok"].copy()
        improved = ok[(ok["oos_net_r_delta"] > 0) & (ok["ml_oos_trades"] > 0)].copy()
        lines.append(
            f"Model: DecisionTree depth=2 using bot feature schema `{', '.join(FEATURE_NAMES)}`. "
            f"Fit rows: {len(ok)} ok / {len(ml_summary)} attempted. OOS net-R improved on {len(improved)} symbols."
        )
        lines.append("")
        ml_cols = [
            "symbol",
            "status",
            "threshold",
            "raw_oos_trades",
            "raw_oos_net_r",
            "ml_oos_trades",
            "ml_oos_net_r",
            "oos_net_r_delta",
            "oos_trade_retention",
        ]
        ok_display = ok.sort_values(["oos_net_r_delta", "ml_oos_net_r"], ascending=[False, False])
        skipped_display = ml_summary[ml_summary["status"] != "ok"].copy()
        display = pd.concat([ok_display, skipped_display], ignore_index=True)
        lines.append(markdown_table(display, ml_cols, 30))
    if not failures.empty:
        lines.append("")
        lines.append("## Failures")
        lines.append("")
        lines.append(markdown_table(failures, ["symbol", "error"], 80))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-universe BullByte Curl/VCE parameter and ML-gate sweep.")
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    parser.add_argument("--symbols", default="")
    parser.add_argument("--max-symbols", type=int, default=50)
    parser.add_argument("--timeframes", default="15m")
    parser.add_argument("--grid-mode", choices=["smoke", "fast", "full"], default="full")
    parser.add_argument("--train-start", default="2024-01-01")
    parser.add_argument("--split", default="2025-07-01")
    parser.add_argument("--end", default="2026-04-20")
    parser.add_argument("--fee-bps-per-side", type=float, default=6.5)
    parser.add_argument("--min-risk-pct", type=float, default=0.15)
    parser.add_argument("--min-train-trades", type=int, default=35)
    parser.add_argument("--min-oos-trades", type=int, default=15)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--keep-trades", action="store_true", help="Write all simulated trades, not only selected-train trades.")
    parser.add_argument("--run-ml", action="store_true", help="Train DecisionTree gates on train-selected parameter sets.")
    parser.add_argument("--ml-val-frac", type=float, default=0.35)
    parser.add_argument("--ml-thresholds", default="0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75")
    parser.add_argument("--ml-min-fit", type=int, default=60)
    parser.add_argument("--ml-min-val", type=int, default=25)
    parser.add_argument("--ml-min-oos", type=int, default=30)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.symbols:
        symbols = [clean_symbol(x) for x in parse_list(args.symbols)]
    else:
        symbols = load_universe(args.universe, args.max_symbols)
    signal_specs = build_signal_specs(args.grid_mode, parse_list(args.timeframes))
    exit_specs = build_exit_specs(args.grid_mode)
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    model_dir = args.out_prefix.with_name(f"{args.out_prefix.name}_models")
    if args.run_ml:
        model_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Sweeping BullByte Curl: {len(symbols)} symbols x {len(signal_specs)} signal specs "
        f"x {len(exit_specs)} exit specs ({args.grid_mode})",
        flush=True,
    )
    args_dict = {
        "cache_dir": str(args.cache_dir),
        "train_start": args.train_start,
        "split": args.split,
        "end": args.end,
        "fee_bps_per_side": args.fee_bps_per_side,
        "min_risk_pct": args.min_risk_pct,
        "min_train_trades": args.min_train_trades,
        "min_oos_trades": args.min_oos_trades,
        "keep_trades": bool(args.keep_trades),
    }
    jobs = [(symbol, args_dict, signal_specs, exit_specs) for symbol in symbols]
    rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if args.workers <= 1:
        for job in jobs:
            symbol, job_rows, job_trades, error = run_symbol_job(job)
            if error:
                failures.append({"symbol": symbol, "error": error})
                print(f"  {symbol}: failed {error}", flush=True)
            else:
                rows.extend(job_rows)
                trade_rows.extend(job_trades)
                print(f"  {symbol}: {len(job_rows)} rows", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
            future_map = {ex.submit(run_symbol_job, job): job[0] for job in jobs}
            for fut in concurrent.futures.as_completed(future_map):
                symbol = future_map[fut]
                try:
                    symbol, job_rows, job_trades, error = fut.result()
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    job_rows = []
                    job_trades = []
                if error:
                    failures.append({"symbol": symbol, "error": error})
                    print(f"  {symbol}: failed {error}", flush=True)
                else:
                    rows.extend(job_rows)
                    trade_rows.extend(job_trades)
                    print(f"  {symbol}: {len(job_rows)} rows", flush=True)

    summary = pd.DataFrame(rows)
    failures_df = pd.DataFrame(failures)
    if summary.empty:
        summary.to_csv(args.out_prefix.with_name(f"{args.out_prefix.name}_summary.csv"), index=False)
        failures_df.to_csv(args.out_prefix.with_name(f"{args.out_prefix.name}_failures.csv"), index=False)
        raise SystemExit("No summary rows produced.")

    selected, oracle = select_rows(summary, args.min_train_trades, args.min_oos_trades)
    summary_path = args.out_prefix.with_name(f"{args.out_prefix.name}_summary.csv")
    selected_path = args.out_prefix.with_name(f"{args.out_prefix.name}_selected_by_train.csv")
    oracle_path = args.out_prefix.with_name(f"{args.out_prefix.name}_best_oos_oracle.csv")
    failures_path = args.out_prefix.with_name(f"{args.out_prefix.name}_failures.csv")
    summary.to_csv(summary_path, index=False)
    selected.to_csv(selected_path, index=False)
    oracle.to_csv(oracle_path, index=False)
    failures_df.to_csv(failures_path, index=False)

    all_trades = pd.DataFrame(trade_rows)
    selected_trades = pd.DataFrame()
    if not all_trades.empty:
        all_trades["entry_time"] = pd.to_datetime(all_trades["entry_time"], utc=True, errors="coerce")
        keys = set(zip(selected["symbol"].astype(str), selected["spec_name"].astype(str)))
        selected_trades = all_trades[
            [((str(row.symbol), str(row.spec_name)) in keys) for row in all_trades.itertuples(index=False)]
        ].copy()
        if args.keep_trades:
            all_trades.to_csv(args.out_prefix.with_name(f"{args.out_prefix.name}_trades.csv"), index=False)
        selected_trades.to_csv(args.out_prefix.with_name(f"{args.out_prefix.name}_selected_trades.csv"), index=False)

    if args.run_ml and selected_trades.empty:
        print("Re-running train-selected specs to collect ML training trades ...", flush=True)
        selected_trade_rows: list[dict[str, Any]] = []
        keep_args = {**args_dict, "keep_trades": True}
        selected_jobs: list[tuple[str, dict[str, Any], list[SignalSpec], list[ExitSpec]]] = []
        for _, row in selected.iterrows():
            params = json.loads(str(row["params_json"]))
            selected_jobs.append(
                (
                    str(row["symbol"]),
                    keep_args,
                    [signal_spec_from_dict(params["signal"])],
                    [exit_spec_from_dict(params["exit"])],
                )
            )
        if args.workers <= 1:
            for job in selected_jobs:
                symbol, _, job_trades, error = run_symbol_job(job)
                if error:
                    failures.append({"symbol": symbol, "error": f"selected_trade_rerun: {error}"})
                    print(f"  selected trades {symbol}: failed {error}", flush=True)
                else:
                    selected_trade_rows.extend(job_trades)
                    print(f"  selected trades {symbol}: {len(job_trades)} trades", flush=True)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
                future_map = {ex.submit(run_symbol_job, job): job[0] for job in selected_jobs}
                for fut in concurrent.futures.as_completed(future_map):
                    symbol = future_map[fut]
                    try:
                        symbol, _, job_trades, error = fut.result()
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        job_trades = []
                    if error:
                        failures.append({"symbol": symbol, "error": f"selected_trade_rerun: {error}"})
                        print(f"  selected trades {symbol}: failed {error}", flush=True)
                    else:
                        selected_trade_rows.extend(job_trades)
                        print(f"  selected trades {symbol}: {len(job_trades)} trades", flush=True)
        selected_trades = pd.DataFrame(selected_trade_rows)
        if not selected_trades.empty:
            selected_trades["entry_time"] = pd.to_datetime(selected_trades["entry_time"], utc=True, errors="coerce")
        selected_trades.to_csv(args.out_prefix.with_name(f"{args.out_prefix.name}_selected_trades.csv"), index=False)
        failures_df = pd.DataFrame(failures)
        failures_df.to_csv(failures_path, index=False)

    ml_summary = pd.DataFrame()
    ml_details = pd.DataFrame()
    ml_selected = pd.DataFrame()
    if args.run_ml:
        split = pd.Timestamp(parse_utc_datetime(args.split))
        ml_rows: list[dict[str, Any]] = []
        detail_frames: list[pd.DataFrame] = []
        selected_frames: list[pd.DataFrame] = []
        for _, row in selected.iterrows():
            symbol = str(row["symbol"])
            spec_name = str(row["spec_name"])
            trades = selected_trades[
                (selected_trades["symbol"].astype(str) == symbol)
                & (selected_trades["spec_name"].astype(str) == spec_name)
            ].copy()
            result, details, selected_oos = train_dt_gate_for_symbol(
                symbol,
                spec_name,
                trades,
                split=split,
                val_frac=args.ml_val_frac,
                thresholds=parse_float_list(args.ml_thresholds),
                min_fit=args.ml_min_fit,
                min_val=args.ml_min_val,
                min_oos=args.ml_min_oos,
                random_state=args.random_state,
                model_dir=model_dir,
            )
            ml_rows.append(result)
            if not details.empty:
                detail_frames.append(details)
            if not selected_oos.empty:
                selected_frames.append(selected_oos)
            print(
                f"  ML {symbol}: {result['status']} {result.get('reason', '')} "
                f"raw_oos={result.get('raw_oos_net_r', 0):+.2f} "
                f"ml_oos={result.get('ml_oos_net_r', 0):+.2f}",
                flush=True,
            )
        ml_summary = pd.DataFrame(ml_rows)
        ml_details = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
        ml_selected = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
        ml_summary.to_csv(args.out_prefix.with_name(f"{args.out_prefix.name}_ml_summary.csv"), index=False)
        ml_details.to_csv(args.out_prefix.with_name(f"{args.out_prefix.name}_ml_threshold_details.csv"), index=False)
        ml_selected.to_csv(args.out_prefix.with_name(f"{args.out_prefix.name}_ml_selected_trades.csv"), index=False)

    report_path = write_report(args, summary, selected, oracle, ml_summary, failures_df)
    print(f"Saved summary: {summary_path}", flush=True)
    print(f"Saved selected: {selected_path}", flush=True)
    print(f"Saved oracle: {oracle_path}", flush=True)
    if args.run_ml:
        print(f"Saved ML summary: {args.out_prefix.with_name(f'{args.out_prefix.name}_ml_summary.csv')}", flush=True)
    print(f"Saved report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
