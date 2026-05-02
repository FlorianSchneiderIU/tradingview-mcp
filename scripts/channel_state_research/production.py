from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.channel_state_research.backtest import strategy_metrics
from scripts.channel_state_research.data import MarketDataset, build_market_dataset, prepare_timeframe_bars
from scripts.channel_state_research.features import TimeframeFeatureSpec, build_decision_dataset, build_timeframe_state_frame
from scripts.channel_state_research.zone_confluence import (
    ZoneChannelEventSpec,
    build_zone_channel_signal_dataset,
    label_channel_trade_outcome,
    label_passive_retest_trade_outcome,
)

SELECTION_GATE_PATTERN = re.compile(
    r"^(long|short):([A-Za-z0-9_]+)\s*(<=|>=|==|!=|<|>)\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$"
)


@dataclass(frozen=True)
class ProductionRiskConfig:
    risk_fraction: float = 0.01
    max_daily_net_r_loss: float = 2.0
    max_consecutive_losses: int = 3
    max_signals_per_day: int = 3
    cooldown_bars_after_exit: int = 0
    one_trade_at_a_time: bool = True


@dataclass(frozen=True)
class ZoneChannelProductionConfig:
    name: str = "zone_channel_passive_width_rr_v1"
    symbol: str = "BTCUSDT"
    base_interval: str = "5m"
    timeframes: tuple[str, ...] = ("1h", "4h", "1d")
    decision_timeframe: str = "1h"
    zone_timeframes: tuple[str, ...] = ("4h", "1d")
    atr_length: int = 14
    channel_estimator: str = "theil_sen"
    point_count: int = 5
    min_points: int = 3
    reversal_5m: float = 1.5
    reversal_15m: float = 1.5
    reversal_1h: float = 2.0
    reversal_4h: float = 2.0
    reversal_1d: float = 2.0
    reversal_1w: float = 1.5
    body_envelope_lookback: int = 12
    body_envelope_min_separation: int = 2
    body_envelope_min_move_atr: float = 0.1
    touch_epsilon_atr: float = 0.2
    touch_lookback_bars: int = 20
    persistence_lookback_bars: int = 20
    zone_left: int = 5
    zone_right: int = 5
    zone_ob_search_bars: int = 50
    zone_penetration_frac: float = 0.5
    min_reclaim_pos: float = 0.6
    max_zone_scan: int = 50
    confluence_epsilon_atr: float = 0.5
    entry_mode: str = "passive_retest"
    passive_entry_window_bars: int = 6
    passive_entry_buffer_atr: float = 0.0
    stop_mode: str = "zone"
    stop_buffer_atr: float = 0.2
    target_buffer_atr: float = 0.2
    label_horizon_bars: int = 24
    fee_bps_side: float = 5.0
    slippage_bps_side: float = 2.0
    selection_gates: tuple[str, ...] = (
        "long:zone_width_atr<=2.5",
        "short:target_rr_planned<=3.5",
    )
    risk: ProductionRiskConfig = field(default_factory=ProductionRiskConfig)


@dataclass(frozen=True)
class ParsedSelectionGate:
    direction: str
    column: str
    operator: str
    value: float


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=str, sort_keys=True) + "\n")


def write_json_file(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(row, default=str, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def load_production_config(path: Path) -> ZoneChannelProductionConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    risk_payload = payload.get("risk", {})
    risk = ProductionRiskConfig(
        risk_fraction=float(risk_payload.get("risk_fraction", 0.01)),
        max_daily_net_r_loss=float(risk_payload.get("max_daily_net_r_loss", 2.0)),
        max_consecutive_losses=int(risk_payload.get("max_consecutive_losses", 3)),
        max_signals_per_day=int(risk_payload.get("max_signals_per_day", 3)),
        cooldown_bars_after_exit=int(risk_payload.get("cooldown_bars_after_exit", 0)),
        one_trade_at_a_time=bool(risk_payload.get("one_trade_at_a_time", True)),
    )
    return ZoneChannelProductionConfig(
        name=str(payload.get("name", "zone_channel_passive_width_rr_v1")),
        symbol=str(payload.get("symbol", "BTCUSDT")),
        base_interval=str(payload.get("base_interval", "5m")),
        timeframes=tuple(payload.get("timeframes", ["1h", "4h", "1d"])),
        decision_timeframe=str(payload.get("decision_timeframe", "1h")),
        zone_timeframes=tuple(payload.get("zone_timeframes", ["4h", "1d"])),
        atr_length=int(payload.get("atr_length", 14)),
        channel_estimator=str(payload.get("channel_estimator", "theil_sen")),
        point_count=int(payload.get("point_count", 5)),
        min_points=int(payload.get("min_points", 3)),
        reversal_5m=float(payload.get("reversal_5m", 1.5)),
        reversal_15m=float(payload.get("reversal_15m", 1.5)),
        reversal_1h=float(payload.get("reversal_1h", 2.0)),
        reversal_4h=float(payload.get("reversal_4h", 2.0)),
        reversal_1d=float(payload.get("reversal_1d", 2.0)),
        reversal_1w=float(payload.get("reversal_1w", 1.5)),
        body_envelope_lookback=int(payload.get("body_envelope_lookback", 12)),
        body_envelope_min_separation=int(payload.get("body_envelope_min_separation", 2)),
        body_envelope_min_move_atr=float(payload.get("body_envelope_min_move_atr", 0.1)),
        touch_epsilon_atr=float(payload.get("touch_epsilon_atr", 0.2)),
        touch_lookback_bars=int(payload.get("touch_lookback_bars", 20)),
        persistence_lookback_bars=int(payload.get("persistence_lookback_bars", 20)),
        zone_left=int(payload.get("zone_left", 5)),
        zone_right=int(payload.get("zone_right", 5)),
        zone_ob_search_bars=int(payload.get("zone_ob_search_bars", 50)),
        zone_penetration_frac=float(payload.get("zone_penetration_frac", 0.5)),
        min_reclaim_pos=float(payload.get("min_reclaim_pos", 0.6)),
        max_zone_scan=int(payload.get("max_zone_scan", 50)),
        confluence_epsilon_atr=float(payload.get("confluence_epsilon_atr", 0.5)),
        entry_mode=str(payload.get("entry_mode", "passive_retest")),
        passive_entry_window_bars=int(payload.get("passive_entry_window_bars", 6)),
        passive_entry_buffer_atr=float(payload.get("passive_entry_buffer_atr", 0.0)),
        stop_mode=str(payload.get("stop_mode", "zone")),
        stop_buffer_atr=float(payload.get("stop_buffer_atr", 0.2)),
        target_buffer_atr=float(payload.get("target_buffer_atr", 0.2)),
        label_horizon_bars=int(payload.get("label_horizon_bars", 24)),
        fee_bps_side=float(payload.get("fee_bps_side", 5.0)),
        slippage_bps_side=float(payload.get("slippage_bps_side", 2.0)),
        selection_gates=tuple(payload.get("selection_gates", [])),
        risk=risk,
    )


def save_production_config(path: Path, config: ZoneChannelProductionConfig) -> None:
    payload = asdict(config)
    payload["timeframes"] = list(config.timeframes)
    payload["zone_timeframes"] = list(config.zone_timeframes)
    payload["selection_gates"] = list(config.selection_gates)
    write_json_file(path, payload)


def parse_selection_gates(raw_gates: tuple[str, ...] | list[str]) -> tuple[ParsedSelectionGate, ...]:
    parsed: list[ParsedSelectionGate] = []
    for raw_gate in raw_gates:
        text = str(raw_gate).strip()
        if not text:
            continue
        match = SELECTION_GATE_PATTERN.match(text)
        if match is None:
            raise ValueError(f"Invalid selection gate {text!r}.")
        direction, column, operator, value = match.groups()
        parsed.append(
            ParsedSelectionGate(
                direction=direction,
                column=column,
                operator=operator,
                value=float(value),
            )
        )
    return tuple(parsed)


def selection_gate_mask(frame: pd.DataFrame, gates: tuple[ParsedSelectionGate, ...]) -> pd.Series:
    if frame.empty or not gates:
        return pd.Series(True, index=frame.index, dtype=bool)
    directions = frame["direction"].astype(str) if "direction" in frame.columns else pd.Series("", index=frame.index, dtype=str)
    mask = pd.Series(True, index=frame.index, dtype=bool)
    for gate in gates:
        if gate.column not in frame.columns:
            raise ValueError(f"Selection gate references missing column {gate.column!r}.")
        values = pd.to_numeric(frame[gate.column], errors="coerce")
        direction_mask = directions == gate.direction
        if gate.operator == "<=":
            gate_pass = values <= gate.value
        elif gate.operator == ">=":
            gate_pass = values >= gate.value
        elif gate.operator == "<":
            gate_pass = values < gate.value
        elif gate.operator == ">":
            gate_pass = values > gate.value
        elif gate.operator == "==":
            gate_pass = values == gate.value
        elif gate.operator == "!=":
            gate_pass = values != gate.value
        else:
            raise ValueError(f"Unsupported gate operator {gate.operator!r}.")
        gate_pass = gate_pass.fillna(False)
        mask &= (~direction_mask) | gate_pass
    return mask


def build_production_inputs(
    config: ZoneChannelProductionConfig,
    *,
    start: str,
    end: str,
    cache_dir: Path,
) -> tuple[MarketDataset, pd.DataFrame, dict[str, list[str]]]:
    market = build_market_dataset(
        config.symbol,
        start,
        end,
        timeframes=list(config.timeframes),
        cache_dir=cache_dir,
        base_interval=config.base_interval,
        atr_length=config.atr_length,
    )
    return build_production_inputs_from_market(config, market)


def build_production_inputs_from_base_frame(
    config: ZoneChannelProductionConfig,
    base_frame: pd.DataFrame,
) -> tuple[MarketDataset, pd.DataFrame, dict[str, list[str]]]:
    base = (
        base_frame.sort_values("open_time")
        .drop_duplicates(subset=["open_time"], keep="last")
        .reset_index(drop=True)
        .copy()
    )
    if base.empty:
        raise RuntimeError("Base frame is empty; cannot build production inputs.")
    bars_by_timeframe = {
        timeframe: prepare_timeframe_bars(base, timeframe, atr_length=config.atr_length)
        for timeframe in config.timeframes
    }
    max_close_time = pd.Timestamp(base["close_time"].iloc[-1]).tz_convert("UTC")
    for timeframe, frame in list(bars_by_timeframe.items()):
        bars_by_timeframe[timeframe] = frame[
            pd.to_datetime(frame["close_time"], utc=True, errors="coerce") <= max_close_time
        ].reset_index(drop=True)
    market = MarketDataset(
        symbol=config.symbol,
        source_interval=config.base_interval,
        base_frame=base,
        bars_by_timeframe=bars_by_timeframe,
    )
    return build_production_inputs_from_market(config, market)


def build_production_inputs_from_market(
    config: ZoneChannelProductionConfig,
    market: MarketDataset,
) -> tuple[MarketDataset, pd.DataFrame, dict[str, list[str]]]:
    reversal_map = {
        "5m": config.reversal_5m,
        "15m": config.reversal_15m,
        "1h": config.reversal_1h,
        "4h": config.reversal_4h,
        "1d": config.reversal_1d,
        "1w": config.reversal_1w,
    }
    state_frames: dict[str, pd.DataFrame] = {}
    state_groups: dict[str, dict[str, list[str]]] = {}
    for timeframe in config.timeframes:
        spec = TimeframeFeatureSpec(
            timeframe=timeframe,
            reversal_mult=reversal_map[timeframe],
            estimator=config.channel_estimator,
            structural_point_count=config.point_count,
            min_points=config.min_points,
            body_envelope_lookback=config.body_envelope_lookback,
            body_envelope_min_separation=config.body_envelope_min_separation,
            body_envelope_min_move_atr=config.body_envelope_min_move_atr,
            touch_epsilon_atr=config.touch_epsilon_atr,
            touch_lookback_bars=config.touch_lookback_bars,
            persistence_lookback_bars=config.persistence_lookback_bars,
        )
        state_frame, groups = build_timeframe_state_frame(market.bars_by_timeframe[timeframe], spec)
        state_frames[timeframe] = state_frame
        state_groups[timeframe] = groups

    decision_frame, decision_groups = build_decision_dataset(
        state_frames,
        state_groups,
        decision_timeframe=config.decision_timeframe,
        context_timeframes=[timeframe for timeframe in config.timeframes if timeframe != config.decision_timeframe],
    )
    signals, feature_groups = build_zone_channel_signal_dataset(
        symbol=market.symbol,
        exec_frame=market.bars_by_timeframe[config.decision_timeframe],
        decision_frame=decision_frame,
        feature_groups=decision_groups,
        spec=_to_event_spec(config),
    )
    gates = parse_selection_gates(config.selection_gates)
    if signals.empty:
        signals["gate_pass"] = pd.Series(dtype=bool)
    else:
        signals["gate_pass"] = selection_gate_mask(signals, gates)
    return market, signals, feature_groups


def replay_production_strategy(
    *,
    config: ZoneChannelProductionConfig,
    exec_frame: pd.DataFrame,
    signals: pd.DataFrame,
) -> dict[str, pd.DataFrame | dict[str, Any]]:
    if signals.empty:
        empty = pd.DataFrame()
        return {
            "decisions": empty,
            "trades": empty,
            "summary": {
                "signal_rows": 0,
                "selected_signal_rows": 0,
                "accepted_orders": 0,
                "filled_trades": 0,
                **strategy_metrics(empty),
            },
        }

    ordered_signals = signals.sort_values(["event_time", "direction", "event_key"]).reset_index(drop=True).copy()
    exec_ordered = exec_frame.sort_values("close_time").reset_index(drop=True).copy()
    close_times = [pd.Timestamp(value).tz_convert("UTC") for value in pd.to_datetime(exec_ordered["close_time"], utc=True, errors="coerce")]
    time_to_index = {timestamp: index for index, timestamp in enumerate(close_times)}
    opens = exec_ordered["open"].astype(float).to_list()
    highs = exec_ordered["high"].astype(float).to_list()
    lows = exec_ordered["low"].astype(float).to_list()
    closes = exec_ordered["close"].astype(float).to_list()

    decisions: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    signal_counts_by_day: dict[pd.Timestamp, int] = defaultdict(int)
    daily_net_r: dict[pd.Timestamp, float] = defaultdict(float)
    active_until: pd.Timestamp | None = None
    cooldown_until: pd.Timestamp | None = None
    consecutive_losses = 0

    for _, row in ordered_signals.iterrows():
        event_time = pd.Timestamp(row["event_time"]).tz_convert("UTC")
        event_day = event_time.floor("D")
        decision: dict[str, Any] = {
            "time": utc_now_iso(),
            "event_time": event_time,
            "event_key": str(row["event_key"]),
            "direction": str(row["direction"]),
            "gate_pass": bool(row.get("gate_pass", True)),
            "signal_price": float(row["signal_price"]),
            "entry_price": float(row["entry_price"]),
            "stop_price": float(row["stop_price"]),
            "target_price": float(row["target_price"]),
            "target_rr_planned": float(row["target_rr_planned"]),
            "daily_signal_count_before": int(signal_counts_by_day[event_day]),
            "daily_net_r_before": float(daily_net_r[event_day]),
            "consecutive_losses_before": int(consecutive_losses),
        }
        if not bool(row.get("gate_pass", True)):
            decision["status"] = "filtered_by_gate"
            decisions.append(decision)
            continue
        if active_until is not None and event_time < active_until and config.risk.one_trade_at_a_time:
            decision["status"] = "blocked_active_trade"
            decision["blocked_until"] = active_until
            decisions.append(decision)
            continue
        if cooldown_until is not None and event_time < cooldown_until:
            decision["status"] = "blocked_cooldown"
            decision["blocked_until"] = cooldown_until
            decisions.append(decision)
            continue
        if config.risk.max_signals_per_day > 0 and signal_counts_by_day[event_day] >= config.risk.max_signals_per_day:
            decision["status"] = "blocked_daily_signal_cap"
            decisions.append(decision)
            continue
        if config.risk.max_daily_net_r_loss > 0.0 and daily_net_r[event_day] <= -abs(config.risk.max_daily_net_r_loss):
            decision["status"] = "blocked_daily_loss_limit"
            decisions.append(decision)
            continue
        if config.risk.max_consecutive_losses > 0 and consecutive_losses >= config.risk.max_consecutive_losses:
            decision["status"] = "blocked_loss_streak"
            decisions.append(decision)
            continue

        signal_counts_by_day[event_day] += 1
        outcome = replay_trade_outcome_for_signal(
            row=row,
            config=config,
            time_to_index=time_to_index,
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
            close_times=close_times,
        )
        decision.update(outcome["decision"])
        decisions.append(decision)

        if config.risk.one_trade_at_a_time and outcome["active_until"] is not None:
            active_until = pd.Timestamp(outcome["active_until"]).tz_convert("UTC")

        if not outcome["filled"]:
            continue

        trade = dict(outcome["trade"])
        trades.append(trade)
        exit_time = pd.Timestamp(trade["exit_time"]).tz_convert("UTC")
        exit_day = exit_time.floor("D")
        daily_net_r[exit_day] += float(trade["r_multiple_net"])
        if float(trade["r_multiple_net"]) < 0.0:
            consecutive_losses += 1
        else:
            consecutive_losses = 0

        if config.risk.cooldown_bars_after_exit > 0:
            exit_index = int(trade["exit_index"])
            cooldown_index = min(len(close_times) - 1, exit_index + int(config.risk.cooldown_bars_after_exit))
            cooldown_until = close_times[cooldown_index]

    decision_frame = pd.DataFrame(decisions)
    trade_frame = pd.DataFrame(trades)
    summary: dict[str, Any] = {
        "signal_rows": int(len(ordered_signals)),
        "selected_signal_rows": int(ordered_signals["gate_pass"].astype(bool).sum()),
        "accepted_orders": int(decision_frame["status"].isin(["order_expired", "trade_filled"]).sum()) if not decision_frame.empty else 0,
        "filled_trades": int(len(trade_frame)),
        "blocked_active_trade": int((decision_frame["status"] == "blocked_active_trade").sum()) if not decision_frame.empty else 0,
        "blocked_cooldown": int((decision_frame["status"] == "blocked_cooldown").sum()) if not decision_frame.empty else 0,
        "blocked_daily_signal_cap": int((decision_frame["status"] == "blocked_daily_signal_cap").sum()) if not decision_frame.empty else 0,
        "blocked_daily_loss_limit": int((decision_frame["status"] == "blocked_daily_loss_limit").sum()) if not decision_frame.empty else 0,
        "blocked_loss_streak": int((decision_frame["status"] == "blocked_loss_streak").sum()) if not decision_frame.empty else 0,
        "expired_orders": int((decision_frame["status"] == "order_expired").sum()) if not decision_frame.empty else 0,
    }
    summary.update(strategy_metrics(trade_frame))
    return {
        "decisions": decision_frame,
        "trades": trade_frame,
        "summary": summary,
    }


def replay_trade_outcome_for_signal(
    *,
    row: pd.Series,
    config: ZoneChannelProductionConfig,
    time_to_index: dict[pd.Timestamp, int],
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    close_times: list[pd.Timestamp],
) -> dict[str, Any]:
    event_time = pd.Timestamp(row["event_time"]).tz_convert("UTC")
    signal_index = time_to_index.get(event_time)
    if signal_index is None:
        return {
            "filled": False,
            "active_until": event_time,
            "decision": {
                "status": "error_missing_signal_index",
            },
            "trade": {},
        }

    if config.entry_mode == "passive_retest":
        outcome = label_passive_retest_trade_outcome(
            direction=str(row["direction"]),
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
            close_times=close_times,
            signal_index=signal_index,
            limit_entry_price=float(row["entry_price"]),
            stop_price=float(row["stop_price"]),
            target_price=float(row["target_price"]),
            entry_window_bars=config.passive_entry_window_bars,
            horizon_bars=config.label_horizon_bars,
        )
        expiry_index = min(len(close_times) - 1, signal_index + max(int(config.passive_entry_window_bars), 1))
    else:
        outcome = label_channel_trade_outcome(
            direction=str(row["direction"]),
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
            close_times=close_times,
            start_index=signal_index + 1,
            entry_price=float(row["entry_price"]),
            stop_price=float(row["stop_price"]),
            target_price=float(row["target_price"]),
            horizon_bars=config.label_horizon_bars,
        )
        expiry_index = signal_index + 1

    if outcome is None:
        expiry_time = close_times[min(expiry_index, len(close_times) - 1)]
        return {
            "filled": False,
            "active_until": expiry_time,
            "decision": {
                "status": "order_expired",
                "expiry_time": expiry_time,
            },
            "trade": {},
        }

    entry_time = pd.Timestamp(outcome["entry_time"]).tz_convert("UTC")
    exit_time = pd.Timestamp(outcome["exit_time"]).tz_convert("UTC")
    entry_index = time_to_index.get(entry_time, signal_index)
    exit_index = time_to_index.get(exit_time, entry_index)
    gross_r = float(outcome["future_r"])
    net_r = gross_r - float(row["cost_r"])
    bar_seconds = _infer_bar_seconds_from_times(close_times)
    hold_bars = int(
        max(
            1,
            round((exit_time - entry_time).total_seconds() / bar_seconds),
        )
    )
    trade = {
        "direction": str(row["direction"]),
        "event_time": event_time,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "signal_price": float(row["signal_price"]),
        "entry_price": float(row["entry_price"]),
        "stop_price": float(row["stop_price"]),
        "target_price": float(row["target_price"]),
        "target_rr_planned": float(row["target_rr_planned"]),
        "cost_r": float(row["cost_r"]),
        "r_multiple_gross": gross_r,
        "r_multiple_net": net_r,
        "return_pct": float(config.risk.risk_fraction * net_r),
        "hold_bars": hold_bars,
        "exit_reason": str(outcome["outcome"]),
        "bars_to_outcome": float(outcome["bars_to_outcome"]),
        "entry_delay_bars": float(outcome.get("entry_delay_bars", 0.0)),
        "mfe_r": float(outcome.get("mfe_r", np.nan)),
        "mae_r": float(outcome.get("mae_r", np.nan)),
        "event_key": str(row["event_key"]),
        "symbol": str(row["symbol"]),
        "entry_index": int(entry_index),
        "exit_index": int(exit_index),
    }
    return {
        "filled": True,
        "active_until": exit_time,
        "decision": {
            "status": "trade_filled",
            "entry_time": entry_time,
            "exit_time": exit_time,
            "outcome": str(outcome["outcome"]),
            "r_multiple_net": net_r,
            "return_pct": float(config.risk.risk_fraction * net_r),
        },
        "trade": trade,
    }


def latest_selected_signal(
    signals: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    max_age_hours: float,
) -> pd.Series | None:
    if signals.empty:
        return None
    cutoff = as_of - pd.Timedelta(hours=max_age_hours)
    eligible = signals[
        signals["gate_pass"].astype(bool)
        & (pd.to_datetime(signals["event_time"], utc=True, errors="coerce") <= as_of)
        & (pd.to_datetime(signals["event_time"], utc=True, errors="coerce") >= cutoff)
    ].copy()
    if eligible.empty:
        return None
    eligible = eligible.sort_values(["event_time", "target_rr_planned"], ascending=[False, False])
    return eligible.iloc[0]


def _infer_bar_seconds_from_times(close_times: list[pd.Timestamp]) -> float:
    if len(close_times) < 2:
        return 3600.0
    series = pd.to_datetime(pd.Series(close_times), utc=True, errors="coerce").dropna().sort_values()
    diffs = series.diff().dropna().dt.total_seconds()
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return 3600.0
    return float(diffs.median())


def build_production_report(
    *,
    config: ZoneChannelProductionConfig,
    start: str,
    end: str,
    summary: dict[str, Any],
) -> str:
    lines = [
        "# Zone-Channel Production Replay",
        "",
        "## Configuration",
        "",
        f"- name: `{config.name}`",
        f"- symbol: `{config.symbol}`",
        f"- window: `{start}` to `{end}`",
        f"- timeframes: `{', '.join(config.timeframes)}`",
        f"- zone timeframes: `{', '.join(config.zone_timeframes)}`",
        f"- decision timeframe: `{config.decision_timeframe}`",
        f"- estimator: `{config.channel_estimator}`",
        f"- entry mode: `{config.entry_mode}`",
        f"- stop mode: `{config.stop_mode}`",
        f"- selection gates: `{', '.join(config.selection_gates) if config.selection_gates else 'none'}`",
        f"- risk fraction: `{config.risk.risk_fraction}`",
        f"- max daily net R loss: `{config.risk.max_daily_net_r_loss}`",
        f"- max consecutive losses: `{config.risk.max_consecutive_losses}`",
        f"- max signals per day: `{config.risk.max_signals_per_day}`",
        "",
        "## Summary",
        "",
    ]
    for key in [
        "signal_rows",
        "selected_signal_rows",
        "accepted_orders",
        "filled_trades",
        "expired_orders",
        "blocked_active_trade",
        "blocked_cooldown",
        "blocked_daily_signal_cap",
        "blocked_daily_loss_limit",
        "blocked_loss_streak",
        "trades",
        "total_return",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "hit_rate",
        "profit_factor",
        "average_trade",
        "average_win",
        "average_loss",
        "long_only_return",
        "short_only_return",
        "net_r",
    ]:
        if key in summary:
            lines.append(f"- {key}: `{summary[key]}`")
    return "\n".join(lines).strip() + "\n"


def build_heartbeat(
    *,
    mode: str,
    config: ZoneChannelProductionConfig,
    start: str,
    end: str,
    log_jsonl: Path,
) -> dict[str, Any]:
    now = utc_now_iso()
    return {
        "mode": mode,
        "pid": os.getpid(),
        "started_at": now,
        "updated_at": now,
        "config_name": config.name,
        "symbol": config.symbol,
        "start": start,
        "end": end,
        "selection_gates": list(config.selection_gates),
        "log_jsonl": str(log_jsonl),
        "last_event": "starting",
        "last_status": None,
    }


def update_heartbeat(path: Path, heartbeat: dict[str, Any], event: str, **updates: Any) -> None:
    heartbeat["updated_at"] = utc_now_iso()
    heartbeat["last_event"] = event
    heartbeat.update(updates)
    write_json_file(path, heartbeat)


def _to_event_spec(config: ZoneChannelProductionConfig) -> ZoneChannelEventSpec:
    return ZoneChannelEventSpec(
        zone_timeframes=tuple(config.zone_timeframes),
        zone_left=config.zone_left,
        zone_right=config.zone_right,
        zone_ob_search_bars=config.zone_ob_search_bars,
        zone_penetration_frac=config.zone_penetration_frac,
        min_reclaim_pos=config.min_reclaim_pos,
        max_zone_scan=config.max_zone_scan,
        confluence_epsilon_atr=config.confluence_epsilon_atr,
        entry_mode=config.entry_mode,
        passive_entry_window_bars=config.passive_entry_window_bars,
        passive_entry_buffer_atr=config.passive_entry_buffer_atr,
        stop_mode=config.stop_mode,
        stop_buffer_atr=config.stop_buffer_atr,
        target_buffer_atr=config.target_buffer_atr,
        label_horizon_bars=config.label_horizon_bars,
        fee_bps_side=config.fee_bps_side,
        slippage_bps_side=config.slippage_bps_side,
        channel_timeframes=tuple(config.timeframes),
        execution_timeframe=config.decision_timeframe,
    )
