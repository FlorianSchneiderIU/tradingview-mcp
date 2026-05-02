from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.backtest_turtle_soup import normalize_binance_spot_symbol, normalize_timeframe, resample_ohlc


@dataclass(frozen=True)
class BreakerCandidatePreset:
    name: str
    symbols: tuple[str, ...]
    directions: tuple[str, ...]
    interval: str
    entry_mode: str
    zone_tf: str
    confirmation_tf: str
    max_retest_bars: int
    max_confirm_bars: int
    max_hold_bars: int
    stop_buffer_atr: float
    target_rr: float
    min_entry_risk_pct: float
    strategy_min_reject_pos: float
    context_min_retest_reject_pos: float
    context_max_confirm_close_pos_dir: float | None = None
    context_min_confirm_gap_r: float | None = None
    min_confirm_fvg_atr: float = 0.0
    breaker_model: Path | None = None
    breaker_prob_threshold: float | None = None
    notes: str = ""


PRESETS: dict[str, BreakerCandidatePreset] = {
    "btc_sol_base_v1": BreakerCandidatePreset(
        name="btc_sol_base_v1",
        symbols=("BTCUSDT", "SOLUSDT"),
        directions=("long",),
        interval="5m",
        entry_mode="fvg_print",
        zone_tf="1h",
        confirmation_tf="15m",
        max_retest_bars=72,
        max_confirm_bars=72,
        max_hold_bars=120,
        stop_buffer_atr=0.10,
        target_rr=2.0,
        min_entry_risk_pct=1.0,
        strategy_min_reject_pos=0.75,
        context_min_retest_reject_pos=0.75,
        context_max_confirm_close_pos_dir=0.90,
        context_min_confirm_gap_r=None,
        notes="BTC+SOL long-only base context gate that matched the 234-trade / +35.276R study.",
    ),
    "btc_sol_candidate_v1": BreakerCandidatePreset(
        name="btc_sol_candidate_v1",
        symbols=("BTCUSDT", "SOLUSDT"),
        directions=("long",),
        interval="5m",
        entry_mode="fvg_print",
        zone_tf="1h",
        confirmation_tf="15m",
        max_retest_bars=72,
        max_confirm_bars=72,
        max_hold_bars=120,
        stop_buffer_atr=0.10,
        target_rr=2.0,
        min_entry_risk_pct=1.0,
        strategy_min_reject_pos=0.75,
        context_min_retest_reject_pos=0.90,
        context_max_confirm_close_pos_dir=0.90,
        context_min_confirm_gap_r=-0.05,
        notes="Higher-quality BTC+SOL long-only continuation gate with stronger rejection and shallow confirmation gap.",
    ),
}


def preset_names() -> list[str]:
    return sorted(PRESETS)


def get_preset(name: str) -> BreakerCandidatePreset:
    try:
        return PRESETS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown breaker preset {name!r}. Available: {', '.join(preset_names())}") from exc


def preset_summary(preset: BreakerCandidatePreset) -> dict[str, Any]:
    data = asdict(preset)
    data["symbols"] = list(preset.symbols)
    data["directions"] = list(preset.directions)
    data["breaker_model"] = str(preset.breaker_model) if preset.breaker_model else None
    return data


def apply_preset_args(args: Any, preset: BreakerCandidatePreset) -> Any:
    args.symbols = list(preset.symbols)
    args.directions = list(preset.directions)
    args.interval = normalize_timeframe(preset.interval)
    args.entry_mode = preset.entry_mode
    args.zone_tf = normalize_timeframe(preset.zone_tf)
    args.confirmation_tf = normalize_timeframe(preset.confirmation_tf)
    args.max_retest_bars = preset.max_retest_bars
    args.max_confirm_bars = preset.max_confirm_bars
    args.max_hold_bars = preset.max_hold_bars
    args.stop_buffer_atr = preset.stop_buffer_atr
    args.target_rr = preset.target_rr
    args.min_reject_pos = preset.strategy_min_reject_pos
    args.min_confirm_fvg_atr = preset.min_confirm_fvg_atr
    args.min_entry_risk_pct = preset.min_entry_risk_pct
    if preset.breaker_model is None:
        args.no_breaker_model = True
    else:
        args.breaker_model = preset.breaker_model
        args.no_breaker_model = False
    if preset.breaker_prob_threshold is not None:
        args.breaker_prob_threshold = preset.breaker_prob_threshold
    return args


def normalize_symbol_list(symbols: list[str] | tuple[str, ...]) -> list[str]:
    return [normalize_binance_spot_symbol(symbol) for symbol in symbols]


def build_confirmation_lookup(frame: pd.DataFrame, confirmation_tf: str) -> pd.DataFrame:
    confirmation = resample_ohlc(frame, confirmation_tf)
    return confirmation.set_index("close_time")[["open", "high", "low", "close"]]


def _to_timestamp(value: Any) -> pd.Timestamp:
    if value is None or pd.isna(value):
        return pd.NaT
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _float(row: Any, key: str, default: float = math.nan) -> float:
    value = row.get(key, default) if hasattr(row, "get") else default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out


def compute_context_metrics(
    row: dict[str, Any] | pd.Series,
    confirmation_lookup: pd.DataFrame | None,
    confirmation_tf: str,
) -> dict[str, float]:
    direction = str(row.get("direction", ""))
    entry_price = _float(row, "entry_price")
    if not math.isfinite(entry_price):
        entry_price = _float(row, "planned_entry_price")
    stop_price = _float(row, "stop_price")
    zone_top = _float(row, "zone_top")
    zone_bottom = _float(row, "zone_bottom")
    confirm_fvg_top = _float(row, "confirm_fvg_top")
    confirm_fvg_bottom = _float(row, "confirm_fvg_bottom")
    confirm_time = _to_timestamp(row.get("confirm_time"))
    retest_reject_pos = _float(row, "retest_reject_pos")

    risk = abs(entry_price - stop_price) if math.isfinite(entry_price) and math.isfinite(stop_price) else math.nan
    risk_pct = risk / entry_price * 100.0 if math.isfinite(risk) and risk > 0 and math.isfinite(entry_price) and entry_price > 0 else math.nan

    confirm_gap_r = math.nan
    if math.isfinite(risk) and risk > 0:
        if direction == "long" and math.isfinite(confirm_fvg_bottom) and math.isfinite(zone_top):
            confirm_gap_r = (confirm_fvg_bottom - zone_top) / risk
        elif direction == "short" and math.isfinite(confirm_fvg_top) and math.isfinite(zone_bottom):
            confirm_gap_r = (zone_bottom - confirm_fvg_top) / risk

    confirm_close_pos_dir = math.nan
    confirm_body_frac = math.nan
    if confirmation_lookup is not None and pd.notna(confirm_time) and confirm_time in confirmation_lookup.index:
        confirm_bar = confirmation_lookup.loc[confirm_time]
        confirm_range = float(confirm_bar["high"]) - float(confirm_bar["low"])
        if confirm_range > 0:
            close_pos = (float(confirm_bar["close"]) - float(confirm_bar["low"])) / confirm_range
            confirm_close_pos_dir = close_pos if direction == "long" else 1.0 - close_pos
            confirm_body_frac = abs(float(confirm_bar["close"]) - float(confirm_bar["open"])) / confirm_range

    return {
        "risk_pct": risk_pct,
        "retest_reject_pos": retest_reject_pos,
        "confirm_gap_r": confirm_gap_r,
        "confirm_close_pos_dir": confirm_close_pos_dir,
        "confirm_body_frac": confirm_body_frac,
        "confirmation_tf": confirmation_tf,
    }


def evaluate_candidate(
    row: dict[str, Any] | pd.Series,
    preset: BreakerCandidatePreset,
    confirmation_lookup: pd.DataFrame | None = None,
) -> tuple[bool, dict[str, float], list[str]]:
    metrics = compute_context_metrics(row, confirmation_lookup, preset.confirmation_tf)
    symbol = normalize_binance_spot_symbol(str(row.get("symbol", "")))
    direction = str(row.get("direction", ""))
    failures: list[str] = []

    if symbol not in normalize_symbol_list(preset.symbols):
        failures.append("symbol")
    if direction not in preset.directions:
        failures.append("direction")

    risk_pct = metrics["risk_pct"]
    if not math.isfinite(risk_pct) or risk_pct < preset.min_entry_risk_pct:
        failures.append("risk_pct")

    reject_pos = metrics["retest_reject_pos"]
    if not math.isfinite(reject_pos) or reject_pos < preset.context_min_retest_reject_pos:
        failures.append("retest_reject_pos")

    if preset.context_max_confirm_close_pos_dir is not None:
        close_pos = metrics["confirm_close_pos_dir"]
        if not math.isfinite(close_pos) or close_pos > preset.context_max_confirm_close_pos_dir:
            failures.append("confirm_close_pos_dir")

    if preset.context_min_confirm_gap_r is not None:
        gap_r = metrics["confirm_gap_r"]
        if not math.isfinite(gap_r) or gap_r < preset.context_min_confirm_gap_r:
            failures.append("confirm_gap_r")

    return len(failures) == 0, metrics, failures


def candidate_mask(frame: pd.DataFrame, preset: BreakerCandidatePreset) -> pd.Series:
    mask = frame["symbol"].astype(str).map(normalize_binance_spot_symbol).isin(normalize_symbol_list(preset.symbols))
    mask &= frame["direction"].astype(str).isin(preset.directions)
    mask &= pd.to_numeric(frame["risk_pct"], errors="coerce") >= preset.min_entry_risk_pct
    mask &= pd.to_numeric(frame["retest_reject_pos"], errors="coerce") >= preset.context_min_retest_reject_pos

    if preset.context_max_confirm_close_pos_dir is not None:
        mask &= pd.to_numeric(frame["confirm_close_pos_dir"], errors="coerce") <= preset.context_max_confirm_close_pos_dir
    if preset.context_min_confirm_gap_r is not None:
        mask &= pd.to_numeric(frame["confirm_gap_r"], errors="coerce") >= preset.context_min_confirm_gap_r
    return mask.fillna(False)
