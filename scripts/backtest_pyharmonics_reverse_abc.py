from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backtest_wolfe_wave import high_before_low  # noqa: E402


DEFAULT_DECISIONS = Path("scripts/pyharmonics_abc_overnight_20260604/abc_selector_pooled_decisions.csv")
DEFAULT_DATA_DIR = Path("scripts/data_pyharmonics_top100_fast_15m_abcd_xabcd_5y")
ROUND_TRIP_COST_RATE = (2.0 * 5.5 + 2.0 * 1.0) / 10_000.0


@dataclass(frozen=True)
class ReverseConfig:
    config_id: str
    stop_basis: str
    stop_value: float
    structure_lookback: int
    target_rr: float
    breakeven_trigger_r: float


@dataclass
class StatAgg:
    count: int = 0
    net_r: float = 0.0
    wins_r: float = 0.0
    losses_r: float = 0.0
    wins: int = 0
    equity: float = 0.0
    peak: float = 0.0
    max_dd: float = 0.0
    values: list[float] | None = None

    def add(self, value: float, *, keep_values: bool = True) -> None:
        value = finite_float(value, 0.0)
        self.count += 1
        self.net_r += value
        if value > 0.0:
            self.wins += 1
            self.wins_r += value
        elif value < 0.0:
            self.losses_r += value
        self.equity += value
        self.peak = max(self.peak, self.equity)
        self.max_dd = min(self.max_dd, self.equity - self.peak)
        if keep_values:
            if self.values is None:
                self.values = []
            self.values.append(value)

    def metrics(self) -> dict[str, float]:
        avg_r = self.net_r / self.count if self.count else 0.0
        median_r = float(np.median(self.values)) if self.values else 0.0
        profit_factor = math.inf if self.losses_r == 0.0 and self.wins_r > 0.0 else (
            self.wins_r / abs(self.losses_r) if self.losses_r < 0.0 else 0.0
        )
        return {
            "trades": int(self.count),
            "net_r": float(self.net_r),
            "avg_r": float(avg_r),
            "median_r": float(median_r),
            "win_rate": float(self.wins / self.count) if self.count else 0.0,
            "profit_factor": float(profit_factor) if math.isfinite(profit_factor) else math.inf,
            "max_dd_r": float(self.max_dd),
        }


@dataclass
class ConfigAgg:
    all: StatAgg
    by_year: dict[int, StatAgg]
    by_symbol: dict[str, StatAgg]
    exit_reasons: dict[str, int]
    invalid: int = 0
    fee_filtered: int = 0
    overlapped: int = 0


@dataclass(frozen=True)
class SymbolArrays:
    open_time: pd.Series
    close_time: pd.Series
    open_time_ns: np.ndarray
    close_time_ns: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    atr: np.ndarray


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in str(raw or "").split(",") if item.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in str(raw or "").split(",") if item.strip()]


def parse_str_list(raw: str) -> list[str]:
    return [item.strip().lower() for item in str(raw or "").split(",") if item.strip()]


def rma(values: pd.Series, length: int) -> pd.Series:
    return values.ewm(alpha=1.0 / max(int(length), 1), adjust=False).mean()


def load_symbol_arrays(data_dir: Path, symbol: str, timeframe: str, atr_length: int) -> SymbolArrays:
    path = data_dir / f"{symbol.lower()}_{timeframe}_bybit.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing candle file for {symbol}: {path}")
    frame = pd.read_csv(path)
    frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True, errors="coerce")
    frame["close_time"] = pd.to_datetime(frame["close_time"], utc=True, errors="coerce")
    for column in ["open", "high", "low", "close"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
    frame = frame.dropna(subset=["open_time", "close_time", "open", "high", "low", "close"]).sort_values("open_time")
    frame = frame.reset_index(drop=True)
    prev_close = frame["close"].shift(1)
    tr = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = rma(tr, atr_length).to_numpy(dtype=float)
    return SymbolArrays(
        open_time=frame["open_time"],
        close_time=frame["close_time"],
        open_time_ns=frame["open_time"].to_numpy(dtype="datetime64[ns]").astype(np.int64, copy=False),
        close_time_ns=frame["close_time"].to_numpy(dtype="datetime64[ns]").astype(np.int64, copy=False),
        open=frame["open"].to_numpy(dtype=float),
        high=frame["high"].to_numpy(dtype=float),
        low=frame["low"].to_numpy(dtype=float),
        close=frame["close"].to_numpy(dtype=float),
        atr=atr,
    )


def top_action_per_event(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    if "event_decision_key" not in out.columns:
        out["event_decision_key"] = out["event_key"].astype(str)
    if "pred_r" not in out.columns:
        out["pred_r"] = pd.to_numeric(out.get("result_r", out.get("r_multiple_net", 0.0)), errors="coerce").fillna(0.0)
    out["pred_r"] = pd.to_numeric(out["pred_r"], errors="coerce").fillna(-999.0)
    return (
        out.sort_values(["event_decision_key", "pred_r", "target_rr_planned", "entry_time"], ascending=[True, False, False, True])
        .drop_duplicates("event_decision_key", keep="first")
        .sort_values(["entry_time", "symbol"])
        .reset_index(drop=True)
    )


def one_trade_per_symbol_forward(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    kept: list[pd.Series] = []
    for _, group in frame.groupby("symbol", dropna=False):
        active_until: pd.Timestamp | None = None
        ordered = group.sort_values(["entry_time", "exit_time"])
        for _, row in ordered.iterrows():
            entry_time = pd.Timestamp(row["entry_time"])
            if active_until is not None and entry_time < active_until:
                continue
            kept.append(row)
            active_until = pd.Timestamp(row["exit_time"])
    return pd.DataFrame(kept).reset_index(drop=True) if kept else pd.DataFrame(columns=frame.columns)


def trade_metrics(frame: pd.DataFrame, result_column: str = "result_r") -> dict[str, float]:
    if frame.empty or result_column not in frame.columns:
        return {"trades": 0, "net_r": 0.0, "avg_r": 0.0, "median_r": 0.0, "win_rate": 0.0, "profit_factor": 0.0, "max_dd_r": 0.0}
    agg = StatAgg()
    for value in pd.to_numeric(frame[result_column], errors="coerce").fillna(0.0):
        agg.add(float(value))
    return agg.metrics()


def build_configs(args: argparse.Namespace) -> list[ReverseConfig]:
    configs: list[ReverseConfig] = []
    stop_bases = set(parse_str_list(args.stop_bases))
    rrs = parse_float_list(args.rrs)
    bes = parse_float_list(args.breakeven_triggers)
    if "mirror" in stop_bases:
        for scale in parse_float_list(args.mirror_scales):
            for rr in rrs:
                for be in bes:
                    configs.append(
                        ReverseConfig(
                            config_id=f"mirror|scale={scale:g}|rr={rr:g}|be={be:g}",
                            stop_basis="mirror",
                            stop_value=float(scale),
                            structure_lookback=0,
                            target_rr=float(rr),
                            breakeven_trigger_r=float(be),
                        )
                    )
    if "structure" in stop_bases:
        for lookback in parse_int_list(args.structure_lookbacks):
            for buffer in parse_float_list(args.structure_buffers):
                for rr in rrs:
                    for be in bes:
                        configs.append(
                            ReverseConfig(
                                config_id=f"structure|lb={lookback:g}|buf={buffer:g}|rr={rr:g}|be={be:g}",
                                stop_basis="structure",
                                stop_value=float(buffer),
                                structure_lookback=int(lookback),
                                target_rr=float(rr),
                                breakeven_trigger_r=float(be),
                            )
                        )
    if args.limit_configs and args.limit_configs > 0:
        configs = configs[: int(args.limit_configs)]
    return configs


def reverse_direction(direction: str) -> str:
    return "short" if str(direction).strip().lower() == "long" else "long"


NUMERIC_EVENT_COLUMNS = [
    "entry_index",
    "trigger_index",
    "entry_price",
    "stop_price",
    "completion_price",
    "completion_min_price",
    "completion_max_price",
    "stop_atr_buffer",
    "max_hold_bars_config",
    "r_multiple_net",
    "pred_r",
]


def prepare_events_by_symbol(top_events: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    events_by_symbol: dict[str, list[dict[str, Any]]] = {}
    ordered = top_events.sort_values(["symbol", "entry_time", "event_decision_key"]).copy()
    for row in ordered.to_dict("records"):
        symbol = str(row.get("symbol", "")).upper()
        for column in NUMERIC_EVENT_COLUMNS:
            row[column] = finite_float(row.get(column))
        row["entry_index"] = int(row["entry_index"]) if math.isfinite(row["entry_index"]) else -1
        row["trigger_index"] = int(row["trigger_index"]) if math.isfinite(row["trigger_index"]) else row["entry_index"]
        entry_time = pd.Timestamp(row.get("entry_time"))
        row["_entry_time_ns"] = int(entry_time.value)
        row["_entry_year"] = int(entry_time.year)
        row["_symbol"] = symbol
        events_by_symbol.setdefault(symbol, []).append(row)
    return events_by_symbol


def structural_stop(row: dict[str, Any], arrays: SymbolArrays, cfg: ReverseConfig, entry_idx: int, direction: str) -> tuple[float, str]:
    entry = float(row.get("entry_price", math.nan))
    if not math.isfinite(entry):
        entry = float(arrays.open[entry_idx])
    if cfg.stop_basis == "mirror":
        original_stop = float(row.get("stop_price", math.nan))
        if not math.isfinite(original_stop):
            return math.nan, "invalid_no_original_stop"
        risk = abs(entry - original_stop) * max(float(cfg.stop_value), 0.0)
        if risk <= 0.0:
            return math.nan, "invalid_mirror_risk"
        return (entry - risk if direction == "long" else entry + risk), "ok"

    if cfg.stop_basis == "structure":
        lookback = max(int(cfg.structure_lookback), 1)
        trigger_idx = int(row.get("trigger_index", entry_idx))
        start_idx = max(0, min(trigger_idx, entry_idx) - max(lookback - 1, 0))
        end_idx = max(start_idx, entry_idx)
        atr = float(arrays.atr[entry_idx]) if 0 <= entry_idx < len(arrays.atr) else math.nan
        if not math.isfinite(atr) or atr <= 0.0:
            atr = infer_original_atr(row)
        if not math.isfinite(atr) or atr <= 0.0:
            return math.nan, "invalid_no_atr"
        if direction == "long":
            stop = float(np.nanmin(arrays.low[start_idx : end_idx + 1])) - float(cfg.stop_value) * atr
        else:
            stop = float(np.nanmax(arrays.high[start_idx : end_idx + 1])) + float(cfg.stop_value) * atr
        return stop, "ok"

    return math.nan, "invalid_stop_basis"


def infer_original_atr(row: dict[str, Any]) -> float:
    entry_direction = str(row.get("direction", "")).lower()
    original_stop = float(row.get("stop_price", math.nan))
    completion_price = float(row.get("completion_price", math.nan))
    completion_min = float(row.get("completion_min_price", math.nan))
    completion_max = float(row.get("completion_max_price", math.nan))
    stop_buffer = float(row.get("stop_atr_buffer", 0.0))
    if not math.isfinite(original_stop) or stop_buffer <= 0.0:
        return math.nan
    if entry_direction == "long":
        values = [x for x in [completion_min, completion_price] if math.isfinite(x)]
        if not values:
            return math.nan
        structural = min(values)
        return (structural - original_stop) / stop_buffer
    if entry_direction == "short":
        values = [x for x in [completion_max, completion_price] if math.isfinite(x)]
        if not values:
            return math.nan
        structural = max(values)
        return (original_stop - structural) / stop_buffer
    return math.nan


def simulate_reverse(
    row: dict[str, Any],
    arrays: SymbolArrays,
    cfg: ReverseConfig,
    *,
    max_fee_to_price_risk: float,
    min_entry_risk_pct: float,
    max_hold_override: int | None,
) -> dict[str, Any]:
    entry_idx = int(row.get("entry_index", -1))
    if entry_idx < 0 or entry_idx >= len(arrays.open) - 1:
        return {"status": "invalid", "reason": "invalid_entry_index"}
    direction = reverse_direction(str(row.get("direction", "")))
    entry = float(row.get("entry_price", math.nan))
    if not math.isfinite(entry) or entry <= 0.0:
        entry = float(arrays.open[entry_idx])
    stop, stop_status = structural_stop(row, arrays, cfg, entry_idx, direction)
    if not math.isfinite(stop):
        return {"status": "invalid", "reason": stop_status}
    if direction == "long":
        if stop >= entry:
            return {"status": "invalid", "reason": "invalid_stop_side"}
        risk = entry - stop
        target = entry + float(cfg.target_rr) * risk
    else:
        if stop <= entry:
            return {"status": "invalid", "reason": "invalid_stop_side"}
        risk = stop - entry
        target = entry - float(cfg.target_rr) * risk
    if risk <= 0.0:
        return {"status": "invalid", "reason": "invalid_risk"}
    entry_risk_pct = risk / entry if entry > 0.0 else math.inf
    if entry_risk_pct < float(min_entry_risk_pct):
        return {"status": "invalid", "reason": "min_entry_risk"}
    cost_r = ROUND_TRIP_COST_RATE * entry / risk
    if max_fee_to_price_risk > 0.0 and cost_r > max_fee_to_price_risk:
        return {"status": "fee_filtered", "reason": "max_fee_to_price_risk", "cost_r": cost_r}

    row_hold = int(row.get("max_hold_bars_config", 96)) if math.isfinite(float(row.get("max_hold_bars_config", math.nan))) else 96
    max_hold_bars = int(max_hold_override) if max_hold_override is not None and max_hold_override > 0 else row_hold
    exit_limit = min(len(arrays.open) - 1, entry_idx + max(1, max_hold_bars))
    exit_idx = exit_limit
    exit_price = float(arrays.close[exit_idx])
    exit_reason = "timeout"
    be = max(float(cfg.breakeven_trigger_r), 0.0)
    breakeven_enabled = be > 0.0 and be < float(cfg.target_rr)
    breakeven_active = False
    breakeven_trigger_price = entry + be * risk if direction == "long" else entry - be * risk

    for idx in range(entry_idx + 1, exit_limit + 1):
        open_value = float(arrays.open[idx])
        high_value = float(arrays.high[idx])
        low_value = float(arrays.low[idx])
        if direction == "long":
            target_hit = high_value >= target
            initial_stop_hit = low_value <= stop
            be_trigger_hit = breakeven_enabled and high_value >= breakeven_trigger_price
            be_stop_hit = breakeven_active and low_value <= entry
            if breakeven_active:
                if target_hit and be_stop_hit:
                    if high_before_low(open_value, high_value, low_value):
                        exit_idx, exit_price, exit_reason = idx, target, "target_same_bar"
                    else:
                        exit_idx, exit_price, exit_reason = idx, entry, "breakeven_same_bar"
                    break
                if be_stop_hit:
                    exit_idx, exit_price, exit_reason = idx, entry, "breakeven"
                    break
                if target_hit:
                    exit_idx, exit_price, exit_reason = idx, target, "target"
                    break
                continue
            if target_hit and initial_stop_hit:
                if high_before_low(open_value, high_value, low_value):
                    breakeven_active = breakeven_active or bool(be_trigger_hit)
                    exit_idx, exit_price, exit_reason = idx, target, "target_same_bar"
                else:
                    exit_idx, exit_price, exit_reason = idx, stop, "stop_same_bar"
                break
            if be_trigger_hit and initial_stop_hit:
                if high_before_low(open_value, high_value, low_value):
                    breakeven_active = True
                    if low_value <= entry:
                        exit_idx, exit_price, exit_reason = idx, entry, "breakeven_same_bar"
                        break
                else:
                    exit_idx, exit_price, exit_reason = idx, stop, "stop_same_bar"
                    break
            if initial_stop_hit:
                exit_idx, exit_price, exit_reason = idx, stop, "stop"
                break
            if target_hit:
                breakeven_active = breakeven_active or bool(be_trigger_hit)
                exit_idx, exit_price, exit_reason = idx, target, "target"
                break
            if be_trigger_hit:
                breakeven_active = True
        else:
            target_hit = low_value <= target
            initial_stop_hit = high_value >= stop
            be_trigger_hit = breakeven_enabled and low_value <= breakeven_trigger_price
            be_stop_hit = breakeven_active and high_value >= entry
            if breakeven_active:
                if target_hit and be_stop_hit:
                    if high_before_low(open_value, high_value, low_value):
                        exit_idx, exit_price, exit_reason = idx, entry, "breakeven_same_bar"
                    else:
                        exit_idx, exit_price, exit_reason = idx, target, "target_same_bar"
                    break
                if be_stop_hit:
                    exit_idx, exit_price, exit_reason = idx, entry, "breakeven"
                    break
                if target_hit:
                    exit_idx, exit_price, exit_reason = idx, target, "target"
                    break
                continue
            if target_hit and initial_stop_hit:
                if high_before_low(open_value, high_value, low_value):
                    exit_idx, exit_price, exit_reason = idx, stop, "stop_same_bar"
                else:
                    breakeven_active = breakeven_active or bool(be_trigger_hit)
                    exit_idx, exit_price, exit_reason = idx, target, "target_same_bar"
                break
            if be_trigger_hit and initial_stop_hit:
                if high_before_low(open_value, high_value, low_value):
                    exit_idx, exit_price, exit_reason = idx, stop, "stop_same_bar"
                    break
                breakeven_active = True
                if high_value >= entry:
                    exit_idx, exit_price, exit_reason = idx, entry, "breakeven_same_bar"
                    break
            if initial_stop_hit:
                exit_idx, exit_price, exit_reason = idx, stop, "stop"
                break
            if target_hit:
                breakeven_active = breakeven_active or bool(be_trigger_hit)
                exit_idx, exit_price, exit_reason = idx, target, "target"
                break
            if be_trigger_hit:
                breakeven_active = True

    gross_r = (exit_price - entry) / risk if direction == "long" else (entry - exit_price) / risk
    net_r = gross_r - cost_r
    exit_time_ns = int(arrays.close_time_ns[exit_idx])
    return {
        "status": "ok",
        "result_r": float(net_r),
        "gross_r": float(gross_r),
        "cost_r": float(cost_r),
        "direction": direction,
        "entry_price": float(entry),
        "exit_price": float(exit_price),
        "stop_price": float(stop),
        "target_price": float(target),
        "entry_index": int(entry_idx),
        "exit_index": int(exit_idx),
        "entry_time_ns": int(arrays.open_time_ns[entry_idx]),
        "exit_time_ns": exit_time_ns,
        "hold_bars": int(exit_idx - entry_idx),
        "exit_reason": exit_reason,
        "entry_risk_pct": float(entry_risk_pct),
        "breakeven_enabled": bool(breakeven_enabled),
        "breakeven_activated": bool(breakeven_active),
    }


def new_config_agg() -> ConfigAgg:
    return ConfigAgg(all=StatAgg(), by_year={}, by_symbol={}, exit_reasons={})


def add_result(agg: ConfigAgg, symbol: str, year: int, exit_reason: str, value: float) -> None:
    agg.all.add(value)
    agg.by_year.setdefault(int(year), StatAgg()).add(value)
    agg.by_symbol.setdefault(symbol, StatAgg()).add(value)
    agg.exit_reasons[exit_reason] = agg.exit_reasons.get(exit_reason, 0) + 1


def summarize_aggs(configs: list[ReverseConfig], aggs: dict[str, ConfigAgg]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cfg in configs:
        agg = aggs[cfg.config_id]
        metrics = agg.all.metrics()
        year_metrics = [year_agg.metrics() for year_agg in agg.by_year.values()]
        symbol_metrics = [symbol_agg.metrics() for symbol_agg in agg.by_symbol.values()]
        min_year_net = min((m["net_r"] for m in year_metrics), default=0.0)
        min_year_avg = min((m["avg_r"] for m in year_metrics), default=0.0)
        years_positive = sum(1 for m in year_metrics if m["net_r"] > 0.0)
        min_symbol_avg = min((m["avg_r"] for m in symbol_metrics if m["trades"] >= 5), default=0.0)
        profitable_symbols = sum(1 for m in symbol_metrics if m["net_r"] > 0.0)
        row = {
            "config_id": cfg.config_id,
            "stop_basis": cfg.stop_basis,
            "stop_value": cfg.stop_value,
            "structure_lookback": cfg.structure_lookback,
            "target_rr": cfg.target_rr,
            "breakeven_trigger_r": cfg.breakeven_trigger_r,
            **metrics,
            "min_year_net_r": float(min_year_net),
            "min_year_avg_r": float(min_year_avg),
            "years_positive": int(years_positive),
            "years_seen": int(len(year_metrics)),
            "active_symbols": int(len(symbol_metrics)),
            "profitable_symbols": int(profitable_symbols),
            "min_symbol_avg_r": float(min_symbol_avg),
            "invalid": int(agg.invalid),
            "fee_filtered": int(agg.fee_filtered),
            "overlapped": int(agg.overlapped),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def add_low_pass_scores(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    out = summary.copy()
    scores: list[float] = []
    min_year_scores: list[float] = []
    trade_counts: list[int] = []
    for _, row in out.iterrows():
        neighbors = out[out["stop_basis"].eq(row["stop_basis"])].copy()
        for column in ["target_rr", "breakeven_trigger_r"]:
            values = sorted(neighbors[column].dropna().unique())
            value_to_idx = {value: idx for idx, value in enumerate(values)}
            center = value_to_idx.get(row[column])
            if center is None:
                continue
            neighbors = neighbors[neighbors[column].map(value_to_idx).sub(center).abs().le(1)]
        if row["stop_basis"] == "mirror":
            values = sorted(neighbors["stop_value"].dropna().unique())
            value_to_idx = {value: idx for idx, value in enumerate(values)}
            center = value_to_idx.get(row["stop_value"])
            if center is not None:
                neighbors = neighbors[neighbors["stop_value"].map(value_to_idx).sub(center).abs().le(1)]
        else:
            for column in ["stop_value", "structure_lookback"]:
                values = sorted(neighbors[column].dropna().unique())
                value_to_idx = {value: idx for idx, value in enumerate(values)}
                center = value_to_idx.get(row[column])
                if center is not None:
                    neighbors = neighbors[neighbors[column].map(value_to_idx).sub(center).abs().le(1)]
        eligible = neighbors[pd.to_numeric(neighbors["trades"], errors="coerce").ge(100)]
        scores.append(float(eligible["avg_r"].median()) if not eligible.empty else -999.0)
        min_year_scores.append(float(eligible["min_year_avg_r"].median()) if not eligible.empty else -999.0)
        trade_counts.append(int(eligible["trades"].median()) if not eligible.empty else 0)
    out["lowpass_avg_r"] = scores
    out["lowpass_min_year_avg_r"] = min_year_scores
    out["lowpass_neighborhood_trades"] = trade_counts
    out["stable_score"] = out["lowpass_avg_r"] + 0.5 * out["lowpass_min_year_avg_r"]
    return out.sort_values(["stable_score", "min_year_net_r", "net_r"], ascending=[False, False, False]).reset_index(drop=True)


def explode_by_year(configs: list[ReverseConfig], aggs: dict[str, ConfigAgg]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cfg in configs:
        for year, agg in sorted(aggs[cfg.config_id].by_year.items()):
            rows.append({"config_id": cfg.config_id, "year": int(year), **agg.metrics()})
    return pd.DataFrame(rows)


def explode_by_symbol(configs: list[ReverseConfig], aggs: dict[str, ConfigAgg]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cfg in configs:
        for symbol, agg in sorted(aggs[cfg.config_id].by_symbol.items()):
            rows.append({"config_id": cfg.config_id, "symbol": symbol, **agg.metrics()})
    return pd.DataFrame(rows)


def explode_exit_reasons(configs: list[ReverseConfig], aggs: dict[str, ConfigAgg]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cfg in configs:
        total = max(aggs[cfg.config_id].all.count, 1)
        for reason, count in sorted(aggs[cfg.config_id].exit_reasons.items()):
            rows.append({"config_id": cfg.config_id, "exit_reason": reason, "count": int(count), "share": float(count / total)})
    return pd.DataFrame(rows)


def simulate_portfolio_for_config(
    events_by_symbol: dict[str, list[dict[str, Any]]],
    cfg: ReverseConfig,
    arrays_by_symbol: dict[str, SymbolArrays],
    args: argparse.Namespace,
    *,
    save_trades: bool = False,
) -> tuple[ConfigAgg, list[dict[str, Any]]]:
    agg = new_config_agg()
    records: list[dict[str, Any]] = []
    for symbol, symbol_events in events_by_symbol.items():
        arrays = arrays_by_symbol[str(symbol)]
        active_until_ns: int | None = None
        for row in symbol_events:
            entry_time_ns = int(row["_entry_time_ns"])
            if active_until_ns is not None and entry_time_ns < active_until_ns:
                agg.overlapped += 1
                continue
            result = simulate_reverse(
                row,
                arrays,
                cfg,
                max_fee_to_price_risk=float(args.max_fee_to_price_risk),
                min_entry_risk_pct=float(args.min_entry_risk_pct),
                max_hold_override=args.max_hold_bars,
            )
            status = result.get("status")
            if status == "invalid":
                agg.invalid += 1
                continue
            if status == "fee_filtered":
                agg.fee_filtered += 1
                continue
            year = int(row["_entry_year"])
            value = float(result["result_r"])
            add_result(agg, str(symbol), year, str(result["exit_reason"]), value)
            active_until_ns = int(result["exit_time_ns"])
            if save_trades:
                records.append(
                    {
                        "config_id": cfg.config_id,
                        "symbol": str(symbol),
                        "event_decision_key": row.get("event_decision_key"),
                        "family": row.get("family"),
                        "pattern_name": row.get("pattern_name"),
                        "pattern_mode": row.get("pattern_mode"),
                        "original_direction": row.get("direction"),
                        "reverse_direction": result["direction"],
                        "entry_time": pd.Timestamp(result["entry_time_ns"], unit="ns", tz="UTC"),
                        "exit_time": pd.Timestamp(result["exit_time_ns"], unit="ns", tz="UTC"),
                        "entry_price": result["entry_price"],
                        "exit_price": result["exit_price"],
                        "stop_price": result["stop_price"],
                        "target_price": result["target_price"],
                        "result_r": result["result_r"],
                        "gross_r": result["gross_r"],
                        "cost_r": result["cost_r"],
                        "exit_reason": result["exit_reason"],
                        "hold_bars": result["hold_bars"],
                        "entry_risk_pct": result["entry_risk_pct"],
                        "forward_result_r": finite_float(row.get("r_multiple_net"), 0.0),
                        "pred_r": finite_float(row.get("pred_r"), math.nan),
                    }
                )
    return agg, records


def load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, SymbolArrays]]:
    decisions = pd.read_csv(args.decisions)
    for column in ["entry_time", "exit_time", "completion_time", "detection_time", "trigger_time"]:
        if column in decisions.columns:
            decisions[column] = pd.to_datetime(decisions[column], utc=True, errors="coerce")
    decisions["result_r"] = pd.to_numeric(decisions.get("result_r", decisions.get("r_multiple_net", 0.0)), errors="coerce").fillna(0.0)
    if args.symbols:
        wanted = {item.strip().upper() for item in str(args.symbols).split(",") if item.strip()}
        decisions = decisions[decisions["symbol"].astype(str).str.upper().isin(wanted)].copy()
    top_events = top_action_per_event(decisions)
    if args.max_events and args.max_events > 0:
        top_events = top_events.head(int(args.max_events)).copy()
    baseline = one_trade_per_symbol_forward(top_events)
    symbols = sorted(top_events["symbol"].astype(str).unique())
    arrays_by_symbol: dict[str, SymbolArrays] = {}
    for idx, symbol in enumerate(symbols, start=1):
        arrays_by_symbol[symbol] = load_symbol_arrays(args.data_dir, symbol, args.timeframe, args.atr_length)
        if idx % 10 == 0 or idx == len(symbols):
            print(f"Loaded candles {idx}/{len(symbols)}", flush=True)
    return top_events, baseline, arrays_by_symbol


def save_top_config_trades(
    events_by_symbol: dict[str, list[dict[str, Any]]],
    summary: pd.DataFrame,
    configs_by_id: dict[str, ReverseConfig],
    arrays_by_symbol: dict[str, SymbolArrays],
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    top_ids = [str(value) for value in summary.head(int(args.save_top_trades)).get("config_id", [])]
    records: list[dict[str, Any]] = []
    for config_id in top_ids:
        cfg = configs_by_id[config_id]
        _, cfg_records = simulate_portfolio_for_config(events_by_symbol, cfg, arrays_by_symbol, args, save_trades=True)
        records.extend(cfg_records)
    if records:
        pd.DataFrame(records).to_csv(output_dir / "reverse_top_trades.csv", index=False)


def run(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    top_events, baseline, arrays_by_symbol = load_inputs(args)
    events_by_symbol = prepare_events_by_symbol(top_events)
    baseline_metrics = trade_metrics(baseline, "result_r")
    pd.DataFrame([{"scope": "forward_baseline_one_symbol", **baseline_metrics}]).to_csv(output_dir / "baseline_guard.csv", index=False)
    configs = build_configs(args)
    print(
        f"Reverse ABC backtest events={len(top_events)} baseline_trades={len(baseline)} "
        f"symbols={len(arrays_by_symbol)} configs={len(configs)}",
        flush=True,
    )
    aggs: dict[str, ConfigAgg] = {}
    for idx, cfg in enumerate(configs, start=1):
        agg, _ = simulate_portfolio_for_config(events_by_symbol, cfg, arrays_by_symbol, args)
        aggs[cfg.config_id] = agg
        if idx % max(int(args.progress_every), 1) == 0 or idx == len(configs):
            metrics = agg.all.metrics()
            print(
                f"config {idx}/{len(configs)} {cfg.config_id} trades={metrics['trades']} "
                f"net={metrics['net_r']:.2f} avg={metrics['avg_r']:.3f}",
                flush=True,
            )

    summary = add_low_pass_scores(summarize_aggs(configs, aggs))
    by_year = explode_by_year(configs, aggs)
    by_symbol = explode_by_symbol(configs, aggs)
    exits = explode_exit_reasons(configs, aggs)
    summary.to_csv(output_dir / "reverse_summary.csv", index=False)
    by_year.to_csv(output_dir / "reverse_by_year.csv", index=False)
    by_symbol.to_csv(output_dir / "reverse_by_symbol.csv", index=False)
    exits.to_csv(output_dir / "reverse_exit_reasons.csv", index=False)
    if args.save_top_trades > 0 and not summary.empty:
        save_top_config_trades(events_by_symbol, summary, {cfg.config_id: cfg for cfg in configs}, arrays_by_symbol, args, output_dir)
    print(f"Saved reverse outputs: {output_dir}", flush=True)
    print(summary.head(12).to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest proper reverse exits on pyharmonics ABC selector events.")
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("scripts/pyharmonics_abc_reverse_20260605"))
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--limit-configs", type=int, default=0)
    parser.add_argument("--stop-bases", default="mirror,structure")
    parser.add_argument("--mirror-scales", default="0.5,0.75,1,1.25")
    parser.add_argument("--structure-lookbacks", default="1,4,12")
    parser.add_argument("--structure-buffers", default="0,0.2,0.5")
    parser.add_argument("--rrs", default="0.5,0.75,1,1.25,1.5")
    parser.add_argument("--breakeven-triggers", default="0,0.5,0.75")
    parser.add_argument("--max-hold-bars", type=int, default=0)
    parser.add_argument("--atr-length", type=int, default=14)
    parser.add_argument("--max-fee-to-price-risk", type=float, default=0.25)
    parser.add_argument("--min-entry-risk-pct", type=float, default=0.001)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--save-top-trades", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
