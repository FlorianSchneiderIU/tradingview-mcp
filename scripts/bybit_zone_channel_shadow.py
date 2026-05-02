from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from queue import Empty, Queue
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.bybit_demo_turtle_soup import (
    BybitV5Client,
    append_closed_candle,
    bybit_symbol,
    bybit_websocket_interval,
    candle_row_from_ws,
    fetch_bybit_klines,
    load_env_file,
    utc_now_iso,
    write_json_file,
    write_jsonl,
)
from scripts.channel_state_research.backtest import strategy_metrics
from scripts.channel_state_research.data import prepare_timeframe_bars
from scripts.channel_state_research.labels import high_before_low
from scripts.channel_state_research.production import (
    ZoneChannelProductionConfig,
    build_production_inputs_from_base_frame,
    build_production_report,
    load_production_config,
    save_production_config,
)


DEFAULT_BASE_URL = "https://api.bybit.com"


@dataclass
class PendingVirtualOrder:
    symbol: str
    event_key: str
    direction: str
    entry_mode: str
    event_time: pd.Timestamp
    signal_index: int
    signal_price: float
    entry_price: float
    stop_price: float
    target_price: float
    target_rr_planned: float
    cost_r: float
    fill_deadline_index: int
    gate_pass: bool


@dataclass
class OpenVirtualTrade:
    symbol: str
    event_key: str
    direction: str
    event_time: pd.Timestamp
    entry_time: pd.Timestamp
    signal_price: float
    entry_price: float
    stop_price: float
    target_price: float
    target_rr_planned: float
    cost_r: float
    signal_index: int
    entry_index: int
    horizon_end_index: int
    entry_delay_bars: int
    mfe_r: float = 0.0
    mae_r: float = 0.0

    @property
    def risk_abs(self) -> float:
        return abs(self.entry_price - self.stop_price)

    @property
    def target_r(self) -> float:
        risk = self.risk_abs
        return abs(self.target_price - self.entry_price) / risk if risk > 0.0 else 0.0


@dataclass
class ShadowRuntimeState:
    pending_order: PendingVirtualOrder | None = None
    active_trade: OpenVirtualTrade | None = None
    processed_event_keys: set[str] = field(default_factory=set)
    status_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    daily_signal_count: dict[pd.Timestamp, int] = field(default_factory=lambda: defaultdict(int))
    daily_net_r: dict[pd.Timestamp, float] = field(default_factory=lambda: defaultdict(float))
    decisions: list[dict[str, Any]] = field(default_factory=list)
    trades: list[dict[str, Any]] = field(default_factory=list)
    consecutive_losses: int = 0
    cooldown_until: pd.Timestamp | None = None
    last_synced_decision_time: pd.Timestamp | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bybit websocket shadow runner for the frozen zone-channel production config.")
    parser.add_argument("--config", type=Path, default=Path("scripts/zone_channel_production_width_rr_v1.json"))
    parser.add_argument("--symbol", default=None, help="Override the symbol from the frozen config.")
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--env-file", type=Path, default=Path("scripts/bybit_demo.env"))
    parser.add_argument("--base-url", default=os.environ.get("BYBIT_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--scan-on-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--websocket-demo", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--websocket-testnet", action="store_true")
    parser.add_argument("--websocket-ping-interval", type=int, default=20)
    parser.add_argument("--websocket-ping-timeout", type=int, default=10)
    parser.add_argument("--websocket-retries", type=int, default=10)
    parser.add_argument("--websocket-idle-timeout", type=int, default=120)
    parser.add_argument("--websocket-stop-after-events", type=int, default=0)
    parser.add_argument("--log-jsonl", type=Path, default=Path("scripts/live_logs/bybit_zone_channel_shadow.jsonl"))
    parser.add_argument("--heartbeat-json", type=Path, default=Path("scripts/live_logs/bybit_zone_channel_shadow_heartbeat.json"))
    parser.add_argument("--state-json", type=Path, default=Path("scripts/live_logs/bybit_zone_channel_shadow_state.json"))
    parser.add_argument("--decisions-csv", type=Path, default=Path("scripts/live_logs/bybit_zone_channel_shadow_decisions.csv"))
    parser.add_argument("--trades-csv", type=Path, default=Path("scripts/live_logs/bybit_zone_channel_shadow_trades.csv"))
    parser.add_argument("--summary-json", type=Path, default=Path("scripts/live_logs/bybit_zone_channel_shadow_summary.json"))
    parser.add_argument("--report-md", type=Path, default=Path("scripts/live_logs/bybit_zone_channel_shadow_report.md"))
    parser.add_argument("--pid-file", type=Path, default=Path("scripts/bybit_zone_channel_shadow.pid"))
    parser.add_argument("--config-copy", type=Path, default=Path("scripts/live_logs/bybit_zone_channel_shadow_config.json"))
    return parser.parse_args()


def initial_heartbeat(args: argparse.Namespace, config: ZoneChannelProductionConfig, mode: str) -> dict[str, Any]:
    now = utc_now_iso()
    return {
        "mode": mode,
        "pid": os.getpid(),
        "started_at": now,
        "updated_at": now,
        "config_name": config.name,
        "symbol": config.symbol,
        "base_interval": config.base_interval,
        "decision_timeframe": config.decision_timeframe,
        "timeframes": list(config.timeframes),
        "zone_timeframes": list(config.zone_timeframes),
        "entry_mode": config.entry_mode,
        "stop_mode": config.stop_mode,
        "selection_gates": list(config.selection_gates),
        "lookback_days": args.lookback_days,
        "log_jsonl": str(args.log_jsonl),
        "last_websocket_event_time": None,
        "last_base_candle_time": None,
        "last_decision_candle_time": None,
        "last_status": None,
        "pending_order": None,
        "active_trade": None,
        "summary": None,
        "error_count": 0,
        "last_error": None,
    }


def update_heartbeat(args: argparse.Namespace, heartbeat: dict[str, Any], event: str, **updates: Any) -> None:
    heartbeat["updated_at"] = utc_now_iso()
    heartbeat["event"] = event
    heartbeat.update(updates)
    write_json_file(args.heartbeat_json, heartbeat)


def websocket_callback(symbol: str, interval: str, event_queue: Queue) -> Any:
    normalized_symbol = bybit_symbol(symbol)

    def on_message(message: dict[str, Any]) -> None:
        topic = str(message.get("topic", ""))
        topic_symbol = topic.split(".")[-1] if topic else ""
        for candle in message.get("data", []):
            row = candle_row_from_ws(
                bybit_symbol(str(candle.get("symbol") or topic_symbol)),
                interval,
                candle,
            )
            if row is not None and row["symbol"] == normalized_symbol:
                event_queue.put(row)

    return on_message


def bootstrap_base_frame(args: argparse.Namespace, client: BybitV5Client, config: ZoneChannelProductionConfig) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.lookback_days)
    frame = fetch_bybit_klines(client, config.symbol, config.base_interval, start, end)
    last_open = frame["open_time"].iloc[-1] if not frame.empty else "n/a"
    print(f"{config.symbol}: bootstrapped {len(frame)} closed {config.base_interval} candles through {last_open}")
    return frame


def build_runtime_snapshot(
    args: argparse.Namespace,
    config: ZoneChannelProductionConfig,
    state: ShadowRuntimeState,
    *,
    signal_rows: int,
    selected_signal_rows: int,
    last_base_candle_time: pd.Timestamp | None,
    last_decision_time: pd.Timestamp | None,
) -> dict[str, Any]:
    payload = {
        "time": utc_now_iso(),
        "config_name": config.name,
        "symbol": config.symbol,
        "base_interval": config.base_interval,
        "decision_timeframe": config.decision_timeframe,
        "signal_rows": int(signal_rows),
        "selected_signal_rows": int(selected_signal_rows),
        "last_base_candle_time": _ts_iso(last_base_candle_time),
        "last_decision_candle_time": _ts_iso(last_decision_time),
        "pending_order": serialize_pending_order(state.pending_order),
        "active_trade": serialize_active_trade(state.active_trade),
        "summary": shadow_summary(state),
    }
    write_json_file(args.state_json, payload)
    return payload


def shadow_summary(state: ShadowRuntimeState) -> dict[str, Any]:
    decision_frame = pd.DataFrame(state.decisions)
    trade_frame = pd.DataFrame(state.trades)
    summary = {
        "decision_rows": int(len(decision_frame)),
        "accepted_orders": int(state.status_counts.get("accepted_pending_order", 0) + state.status_counts.get("accepted_market_trade", 0)),
        "filled_trades": int(len(trade_frame)),
        "expired_orders": int(state.status_counts.get("order_expired", 0)),
        "filtered_by_gate": int(state.status_counts.get("filtered_by_gate", 0)),
        "blocked_active_trade": int(state.status_counts.get("blocked_active_trade", 0)),
        "blocked_cooldown": int(state.status_counts.get("blocked_cooldown", 0)),
        "blocked_daily_signal_cap": int(state.status_counts.get("blocked_daily_signal_cap", 0)),
        "blocked_daily_loss_limit": int(state.status_counts.get("blocked_daily_loss_limit", 0)),
        "blocked_loss_streak": int(state.status_counts.get("blocked_loss_streak", 0)),
        "open_pending_orders": 1 if state.pending_order is not None else 0,
        "open_trades": 1 if state.active_trade is not None else 0,
        "consecutive_losses": int(state.consecutive_losses),
        "last_synced_decision_time": _ts_iso(state.last_synced_decision_time),
    }
    summary.update(strategy_metrics(trade_frame))
    return summary


def persist_shadow_artifacts(
    args: argparse.Namespace,
    config: ZoneChannelProductionConfig,
    state: ShadowRuntimeState,
    *,
    signal_rows: int,
    selected_signal_rows: int,
    window_start: pd.Timestamp | None,
    window_end: pd.Timestamp | None,
) -> dict[str, Any]:
    args.decisions_csv.parent.mkdir(parents=True, exist_ok=True)
    decision_frame = pd.DataFrame(state.decisions)
    trade_frame = pd.DataFrame(state.trades)
    if decision_frame.empty:
        decision_frame = pd.DataFrame(columns=["time", "kind", "status", "event_time", "bar_time", "direction", "event_key"])
    if trade_frame.empty:
        trade_frame = pd.DataFrame(columns=["symbol", "event_key", "direction", "event_time", "entry_time", "exit_time"])
    decision_frame.to_csv(args.decisions_csv, index=False)
    trade_frame.to_csv(args.trades_csv, index=False)

    summary = shadow_summary(state)
    summary["signal_rows"] = int(signal_rows)
    summary["selected_signal_rows"] = int(selected_signal_rows)
    summary["window_start"] = _ts_iso(window_start)
    summary["window_end"] = _ts_iso(window_end)
    write_json_file(args.summary_json, summary)

    start_text = window_start.isoformat() if window_start is not None else "n/a"
    end_text = window_end.isoformat() if window_end is not None else "n/a"
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(
        build_production_report(
            config=config,
            start=start_text,
            end=end_text,
            summary=summary,
        ),
        encoding="utf-8",
    )
    return summary


def serialize_pending_order(order: PendingVirtualOrder | None) -> dict[str, Any] | None:
    if order is None:
        return None
    return {
        "symbol": order.symbol,
        "event_key": order.event_key,
        "direction": order.direction,
        "entry_mode": order.entry_mode,
        "event_time": _ts_iso(order.event_time),
        "signal_index": int(order.signal_index),
        "signal_price": float(order.signal_price),
        "entry_price": float(order.entry_price),
        "stop_price": float(order.stop_price),
        "target_price": float(order.target_price),
        "target_rr_planned": float(order.target_rr_planned),
        "cost_r": float(order.cost_r),
        "fill_deadline_index": int(order.fill_deadline_index),
        "gate_pass": bool(order.gate_pass),
    }


def serialize_active_trade(trade: OpenVirtualTrade | None) -> dict[str, Any] | None:
    if trade is None:
        return None
    return {
        "symbol": trade.symbol,
        "event_key": trade.event_key,
        "direction": trade.direction,
        "event_time": _ts_iso(trade.event_time),
        "entry_time": _ts_iso(trade.entry_time),
        "signal_price": float(trade.signal_price),
        "entry_price": float(trade.entry_price),
        "stop_price": float(trade.stop_price),
        "target_price": float(trade.target_price),
        "target_rr_planned": float(trade.target_rr_planned),
        "cost_r": float(trade.cost_r),
        "signal_index": int(trade.signal_index),
        "entry_index": int(trade.entry_index),
        "horizon_end_index": int(trade.horizon_end_index),
        "entry_delay_bars": int(trade.entry_delay_bars),
        "mfe_r": float(trade.mfe_r),
        "mae_r": float(trade.mae_r),
    }


def _ts_iso(value: pd.Timestamp | None) -> str | None:
    if value is None:
        return None
    return pd.Timestamp(value).tz_convert("UTC").isoformat()


def _record_status(state: ShadowRuntimeState, status: str) -> None:
    state.status_counts[status] = int(state.status_counts.get(status, 0)) + 1


def record_shadow_rows(
    args: argparse.Namespace,
    state: ShadowRuntimeState,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return
    state.decisions.extend(rows)
    for row in rows:
        write_jsonl(args.log_jsonl, row)


def sync_shadow_state(
    *,
    state: ShadowRuntimeState,
    config: ZoneChannelProductionConfig,
    exec_frame: pd.DataFrame,
    signals: pd.DataFrame,
    after_time: pd.Timestamp | None,
) -> list[dict[str, Any]]:
    exec_ordered = exec_frame.sort_values("close_time").reset_index(drop=True).copy()
    if exec_ordered.empty:
        return []
    exec_ordered["close_time"] = pd.to_datetime(exec_ordered["close_time"], utc=True, errors="coerce")
    time_to_index = {
        pd.Timestamp(close_time).tz_convert("UTC"): int(index)
        for index, close_time in enumerate(exec_ordered["close_time"])
    }
    close_times = exec_ordered["close_time"].to_list()
    signals_ordered = signals.sort_values(["event_time", "direction", "event_key"]).copy()
    if not signals_ordered.empty:
        signals_ordered["event_time"] = pd.to_datetime(signals_ordered["event_time"], utc=True, errors="coerce")

    rows: list[dict[str, Any]] = []
    for bar_index, bar in exec_ordered.iterrows():
        bar_time = pd.Timestamp(bar["close_time"]).tz_convert("UTC")
        if after_time is not None and bar_time <= after_time:
            continue

        rows.extend(
            _advance_open_state_on_bar(
                state=state,
                config=config,
                bar=bar,
                bar_index=int(bar_index),
                bar_time=bar_time,
                close_times=close_times,
            )
        )

        if signals_ordered.empty:
            state.last_synced_decision_time = bar_time
            continue

        bar_signals = signals_ordered.loc[signals_ordered["event_time"] == bar_time]
        for _, signal_row in bar_signals.iterrows():
            event_key = str(signal_row["event_key"])
            if event_key in state.processed_event_keys:
                continue
            rows.append(
                _accept_signal_row(
                    state=state,
                    config=config,
                    signal_row=signal_row,
                    signal_index=time_to_index[bar_time],
                )
            )
            state.processed_event_keys.add(event_key)
        state.last_synced_decision_time = bar_time
    return rows


def _advance_open_state_on_bar(
    *,
    state: ShadowRuntimeState,
    config: ZoneChannelProductionConfig,
    bar: pd.Series,
    bar_index: int,
    bar_time: pd.Timestamp,
    close_times: list[pd.Timestamp],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if state.pending_order is not None:
        order_update = _process_pending_order_on_bar(
            state=state,
            config=config,
            order=state.pending_order,
            bar=bar,
            bar_index=bar_index,
            bar_time=bar_time,
            close_times=close_times,
        )
        if order_update is not None:
            rows.extend(order_update)

    if state.active_trade is not None:
        trade_update = _process_active_trade_on_bar(
            state=state,
            config=config,
            trade=state.active_trade,
            bar=bar,
            bar_index=bar_index,
            bar_time=bar_time,
            close_times=close_times,
        )
        if trade_update is not None:
            rows.append(trade_update)
    return rows


def _process_pending_order_on_bar(
    *,
    state: ShadowRuntimeState,
    config: ZoneChannelProductionConfig,
    order: PendingVirtualOrder,
    bar: pd.Series,
    bar_index: int,
    bar_time: pd.Timestamp,
    close_times: list[pd.Timestamp],
) -> list[dict[str, Any]] | None:
    if bar_index <= order.signal_index:
        return None

    open_value = float(bar["open"])
    high_value = float(bar["high"])
    low_value = float(bar["low"])
    close_value = float(bar["close"])
    risk = abs(order.entry_price - order.stop_price)
    if risk <= 0.0:
        state.pending_order = None
        row = _build_event_row(
            kind="order",
            status="order_invalid_risk",
            event_time=order.event_time,
            bar_time=bar_time,
            direction=order.direction,
            event_key=order.event_key,
            signal_price=order.signal_price,
            entry_price=order.entry_price,
            stop_price=order.stop_price,
            target_price=order.target_price,
            target_rr_planned=order.target_rr_planned,
        )
        _record_status(state, row["status"])
        return [row]

    if order.entry_mode == "market_reclaim":
        if bar_index < order.signal_index + 1:
            return None
        target_r = abs(order.target_price - order.entry_price) / risk
        if order.direction == "long":
            target_hit = high_value >= order.target_price
            stop_hit = low_value <= order.stop_price
            target_first = high_before_low(open_value, high_value, low_value)
            mfe_r = max(0.0, (high_value - order.entry_price) / risk)
            mae_r = max(0.0, (order.entry_price - low_value) / risk)
            if target_hit and stop_hit:
                state.pending_order = None
                return [
                    _finalize_trade(
                        state=state,
                        config=config,
                        event_key=order.event_key,
                        symbol=order.symbol,
                        direction=order.direction,
                        event_time=order.event_time,
                        entry_time=bar_time,
                        exit_time=bar_time,
                        signal_price=order.signal_price,
                        entry_price=order.entry_price,
                        stop_price=order.stop_price,
                        target_price=order.target_price,
                        target_rr_planned=order.target_rr_planned,
                        cost_r=order.cost_r,
                        entry_index=bar_index,
                        exit_index=bar_index,
                        entry_delay_bars=bar_index - order.signal_index,
                        gross_r=float(target_r if target_first else -1.0),
                        exit_reason="target_same_bar" if target_first else "stop_same_bar",
                        bars_to_outcome=1.0,
                        mfe_r=float(mfe_r),
                        mae_r=float(mae_r),
                    )
                ]
            if target_hit:
                state.pending_order = None
                return [
                    _finalize_trade(
                        state=state,
                        config=config,
                        event_key=order.event_key,
                        symbol=order.symbol,
                        direction=order.direction,
                        event_time=order.event_time,
                        entry_time=bar_time,
                        exit_time=bar_time,
                        signal_price=order.signal_price,
                        entry_price=order.entry_price,
                        stop_price=order.stop_price,
                        target_price=order.target_price,
                        target_rr_planned=order.target_rr_planned,
                        cost_r=order.cost_r,
                        entry_index=bar_index,
                        exit_index=bar_index,
                        entry_delay_bars=bar_index - order.signal_index,
                        gross_r=float(target_r),
                        exit_reason="target_same_bar",
                        bars_to_outcome=1.0,
                        mfe_r=float(mfe_r),
                        mae_r=float(mae_r),
                    )
                ]
            if stop_hit:
                state.pending_order = None
                return [
                    _finalize_trade(
                        state=state,
                        config=config,
                        event_key=order.event_key,
                        symbol=order.symbol,
                        direction=order.direction,
                        event_time=order.event_time,
                        entry_time=bar_time,
                        exit_time=bar_time,
                        signal_price=order.signal_price,
                        entry_price=order.entry_price,
                        stop_price=order.stop_price,
                        target_price=order.target_price,
                        target_rr_planned=order.target_rr_planned,
                        cost_r=order.cost_r,
                        entry_index=bar_index,
                        exit_index=bar_index,
                        entry_delay_bars=bar_index - order.signal_index,
                        gross_r=-1.0,
                        exit_reason="stop_same_bar",
                        bars_to_outcome=1.0,
                        mfe_r=float(mfe_r),
                        mae_r=float(mae_r),
                    )
                ]
        else:
            target_hit = low_value <= order.target_price
            stop_hit = high_value >= order.stop_price
            target_first = not high_before_low(open_value, high_value, low_value)
            mfe_r = max(0.0, (order.entry_price - low_value) / risk)
            mae_r = max(0.0, (high_value - order.entry_price) / risk)
            if target_hit and stop_hit:
                state.pending_order = None
                return [
                    _finalize_trade(
                        state=state,
                        config=config,
                        event_key=order.event_key,
                        symbol=order.symbol,
                        direction=order.direction,
                        event_time=order.event_time,
                        entry_time=bar_time,
                        exit_time=bar_time,
                        signal_price=order.signal_price,
                        entry_price=order.entry_price,
                        stop_price=order.stop_price,
                        target_price=order.target_price,
                        target_rr_planned=order.target_rr_planned,
                        cost_r=order.cost_r,
                        entry_index=bar_index,
                        exit_index=bar_index,
                        entry_delay_bars=bar_index - order.signal_index,
                        gross_r=float(target_r if target_first else -1.0),
                        exit_reason="target_same_bar" if target_first else "stop_same_bar",
                        bars_to_outcome=1.0,
                        mfe_r=float(mfe_r),
                        mae_r=float(mae_r),
                    )
                ]
            if target_hit:
                state.pending_order = None
                return [
                    _finalize_trade(
                        state=state,
                        config=config,
                        event_key=order.event_key,
                        symbol=order.symbol,
                        direction=order.direction,
                        event_time=order.event_time,
                        entry_time=bar_time,
                        exit_time=bar_time,
                        signal_price=order.signal_price,
                        entry_price=order.entry_price,
                        stop_price=order.stop_price,
                        target_price=order.target_price,
                        target_rr_planned=order.target_rr_planned,
                        cost_r=order.cost_r,
                        entry_index=bar_index,
                        exit_index=bar_index,
                        entry_delay_bars=bar_index - order.signal_index,
                        gross_r=float(target_r),
                        exit_reason="target_same_bar",
                        bars_to_outcome=1.0,
                        mfe_r=float(mfe_r),
                        mae_r=float(mae_r),
                    )
                ]
            if stop_hit:
                state.pending_order = None
                return [
                    _finalize_trade(
                        state=state,
                        config=config,
                        event_key=order.event_key,
                        symbol=order.symbol,
                        direction=order.direction,
                        event_time=order.event_time,
                        entry_time=bar_time,
                        exit_time=bar_time,
                        signal_price=order.signal_price,
                        entry_price=order.entry_price,
                        stop_price=order.stop_price,
                        target_price=order.target_price,
                        target_rr_planned=order.target_rr_planned,
                        cost_r=order.cost_r,
                        entry_index=bar_index,
                        exit_index=bar_index,
                        entry_delay_bars=bar_index - order.signal_index,
                        gross_r=-1.0,
                        exit_reason="stop_same_bar",
                        bars_to_outcome=1.0,
                        mfe_r=float(mfe_r),
                        mae_r=float(mae_r),
                    )
                ]
        state.pending_order = None
        state.active_trade = OpenVirtualTrade(
            symbol=order.symbol,
            event_key=order.event_key,
            direction=order.direction,
            event_time=order.event_time,
            entry_time=bar_time,
            signal_price=order.signal_price,
            entry_price=order.entry_price,
            stop_price=order.stop_price,
            target_price=order.target_price,
            target_rr_planned=order.target_rr_planned,
            cost_r=order.cost_r,
            signal_index=order.signal_index,
            entry_index=bar_index,
            horizon_end_index=bar_index + int(config.label_horizon_bars),
            entry_delay_bars=bar_index - order.signal_index,
            mfe_r=float(mfe_r),
            mae_r=float(mae_r),
        )
        row = _build_event_row(
            kind="trade",
            status="trade_filled",
            event_time=order.event_time,
            bar_time=bar_time,
            direction=order.direction,
            event_key=order.event_key,
            signal_price=order.signal_price,
            entry_price=order.entry_price,
            stop_price=order.stop_price,
            target_price=order.target_price,
            target_rr_planned=order.target_rr_planned,
            entry_time=bar_time,
            entry_delay_bars=float(bar_index - order.signal_index),
            mfe_r=float(mfe_r),
            mae_r=float(mae_r),
        )
        _record_status(state, row["status"])
        return [row]

    filled = False
    if order.direction == "long" and low_value <= order.entry_price:
        filled = True
    if order.direction == "short" and high_value >= order.entry_price:
        filled = True

    if not filled:
        if bar_index >= order.fill_deadline_index:
            state.pending_order = None
            row = _build_event_row(
                kind="order",
                status="order_expired",
                event_time=order.event_time,
                bar_time=bar_time,
                direction=order.direction,
                event_key=order.event_key,
                signal_price=order.signal_price,
                entry_price=order.entry_price,
                stop_price=order.stop_price,
                target_price=order.target_price,
                target_rr_planned=order.target_rr_planned,
            )
            _record_status(state, row["status"])
            return [row]
        return None

    target_r = abs(order.target_price - order.entry_price) / risk
    rows: list[dict[str, Any]] = []
    if order.direction == "long":
        low_before_high = not high_before_low(open_value, high_value, low_value)
        mfe_r = (high_value - order.entry_price) / risk if low_before_high else 0.0
        mae_r = (order.entry_price - low_value) / risk
        stop_hit = low_value <= order.stop_price
        target_hit = high_value >= order.target_price
        if stop_hit:
            state.pending_order = None
            rows.append(
                _finalize_trade(
                    state=state,
                    config=config,
                    event_key=order.event_key,
                    symbol=order.symbol,
                    direction=order.direction,
                    event_time=order.event_time,
                    entry_time=bar_time,
                    exit_time=bar_time,
                    signal_price=order.signal_price,
                    entry_price=order.entry_price,
                    stop_price=order.stop_price,
                    target_price=order.target_price,
                    target_rr_planned=order.target_rr_planned,
                    cost_r=order.cost_r,
                    entry_index=bar_index,
                    exit_index=bar_index,
                    entry_delay_bars=bar_index - order.signal_index,
                    gross_r=-1.0,
                    exit_reason="stop_on_fill_bar",
                    bars_to_outcome=1.0,
                    mfe_r=max(0.0, float(mfe_r)),
                    mae_r=max(0.0, float(mae_r)),
                )
            )
            return rows
        if target_hit and low_before_high:
            state.pending_order = None
            rows.append(
                _finalize_trade(
                    state=state,
                    config=config,
                    event_key=order.event_key,
                    symbol=order.symbol,
                    direction=order.direction,
                    event_time=order.event_time,
                    entry_time=bar_time,
                    exit_time=bar_time,
                    signal_price=order.signal_price,
                    entry_price=order.entry_price,
                    stop_price=order.stop_price,
                    target_price=order.target_price,
                    target_rr_planned=order.target_rr_planned,
                    cost_r=order.cost_r,
                    entry_index=bar_index,
                    exit_index=bar_index,
                    entry_delay_bars=bar_index - order.signal_index,
                    gross_r=float(target_r),
                    exit_reason="target_on_fill_bar",
                    bars_to_outcome=1.0,
                    mfe_r=max(0.0, float(mfe_r)),
                    mae_r=max(0.0, float(mae_r)),
                )
            )
            return rows
    else:
        high_before_low_bar = high_before_low(open_value, high_value, low_value)
        mfe_r = (order.entry_price - low_value) / risk if high_before_low_bar else 0.0
        mae_r = (high_value - order.entry_price) / risk
        stop_hit = high_value >= order.stop_price
        target_hit = low_value <= order.target_price
        if stop_hit:
            state.pending_order = None
            rows.append(
                _finalize_trade(
                    state=state,
                    config=config,
                    event_key=order.event_key,
                    symbol=order.symbol,
                    direction=order.direction,
                    event_time=order.event_time,
                    entry_time=bar_time,
                    exit_time=bar_time,
                    signal_price=order.signal_price,
                    entry_price=order.entry_price,
                    stop_price=order.stop_price,
                    target_price=order.target_price,
                    target_rr_planned=order.target_rr_planned,
                    cost_r=order.cost_r,
                    entry_index=bar_index,
                    exit_index=bar_index,
                    entry_delay_bars=bar_index - order.signal_index,
                    gross_r=-1.0,
                    exit_reason="stop_on_fill_bar",
                    bars_to_outcome=1.0,
                    mfe_r=max(0.0, float(mfe_r)),
                    mae_r=max(0.0, float(mae_r)),
                )
            )
            return rows
        if target_hit and high_before_low_bar:
            state.pending_order = None
            rows.append(
                _finalize_trade(
                    state=state,
                    config=config,
                    event_key=order.event_key,
                    symbol=order.symbol,
                    direction=order.direction,
                    event_time=order.event_time,
                    entry_time=bar_time,
                    exit_time=bar_time,
                    signal_price=order.signal_price,
                    entry_price=order.entry_price,
                    stop_price=order.stop_price,
                    target_price=order.target_price,
                    target_rr_planned=order.target_rr_planned,
                    cost_r=order.cost_r,
                    entry_index=bar_index,
                    exit_index=bar_index,
                    entry_delay_bars=bar_index - order.signal_index,
                    gross_r=float(target_r),
                    exit_reason="target_on_fill_bar",
                    bars_to_outcome=1.0,
                    mfe_r=max(0.0, float(mfe_r)),
                    mae_r=max(0.0, float(mae_r)),
                )
            )
            return rows

    state.pending_order = None
    state.active_trade = OpenVirtualTrade(
        symbol=order.symbol,
        event_key=order.event_key,
        direction=order.direction,
        event_time=order.event_time,
        entry_time=bar_time,
        signal_price=order.signal_price,
        entry_price=order.entry_price,
        stop_price=order.stop_price,
        target_price=order.target_price,
        target_rr_planned=order.target_rr_planned,
        cost_r=order.cost_r,
        signal_index=order.signal_index,
        entry_index=bar_index,
        horizon_end_index=bar_index + int(config.label_horizon_bars),
        entry_delay_bars=bar_index - order.signal_index,
        mfe_r=max(0.0, float(mfe_r)),
        mae_r=max(0.0, float(mae_r)),
    )
    row = _build_event_row(
        kind="trade",
        status="trade_filled",
        event_time=order.event_time,
        bar_time=bar_time,
        direction=order.direction,
        event_key=order.event_key,
        signal_price=order.signal_price,
        entry_price=order.entry_price,
        stop_price=order.stop_price,
        target_price=order.target_price,
        target_rr_planned=order.target_rr_planned,
        entry_time=bar_time,
        entry_delay_bars=float(bar_index - order.signal_index),
        mfe_r=max(0.0, float(mfe_r)),
        mae_r=max(0.0, float(mae_r)),
    )
    _record_status(state, row["status"])
    return [row]


def _process_active_trade_on_bar(
    *,
    state: ShadowRuntimeState,
    config: ZoneChannelProductionConfig,
    trade: OpenVirtualTrade,
    bar: pd.Series,
    bar_index: int,
    bar_time: pd.Timestamp,
    close_times: list[pd.Timestamp],
) -> dict[str, Any] | None:
    if bar_index <= trade.entry_index:
        return None

    high_value = float(bar["high"])
    low_value = float(bar["low"])
    close_value = float(bar["close"])
    risk = trade.risk_abs
    if risk <= 0.0:
        state.active_trade = None
        row = _build_event_row(
            kind="trade",
            status="trade_invalid_risk",
            event_time=trade.event_time,
            bar_time=bar_time,
            direction=trade.direction,
            event_key=trade.event_key,
            signal_price=trade.signal_price,
            entry_price=trade.entry_price,
            stop_price=trade.stop_price,
            target_price=trade.target_price,
            target_rr_planned=trade.target_rr_planned,
        )
        _record_status(state, row["status"])
        return row

    if trade.direction == "long":
        trade.mfe_r = max(float(trade.mfe_r), (high_value - trade.entry_price) / risk)
        trade.mae_r = max(float(trade.mae_r), (trade.entry_price - low_value) / risk)
        stop_hit = low_value <= trade.stop_price
        target_hit = high_value >= trade.target_price
        last_close_r = (close_value - trade.entry_price) / risk
    else:
        trade.mfe_r = max(float(trade.mfe_r), (trade.entry_price - low_value) / risk)
        trade.mae_r = max(float(trade.mae_r), (high_value - trade.entry_price) / risk)
        stop_hit = high_value >= trade.stop_price
        target_hit = low_value <= trade.target_price
        last_close_r = (trade.entry_price - close_value) / risk

    if stop_hit:
        state.active_trade = None
        return _finalize_trade(
            state=state,
            config=config,
            event_key=trade.event_key,
            symbol=trade.symbol,
            direction=trade.direction,
            event_time=trade.event_time,
            entry_time=trade.entry_time,
            exit_time=bar_time,
            signal_price=trade.signal_price,
            entry_price=trade.entry_price,
            stop_price=trade.stop_price,
            target_price=trade.target_price,
            target_rr_planned=trade.target_rr_planned,
            cost_r=trade.cost_r,
            entry_index=trade.entry_index,
            exit_index=bar_index,
            entry_delay_bars=trade.entry_delay_bars,
            gross_r=-1.0,
            exit_reason="stop",
            bars_to_outcome=float(bar_index - trade.entry_index + 1),
            mfe_r=float(trade.mfe_r),
            mae_r=float(trade.mae_r),
        )
    if target_hit:
        state.active_trade = None
        return _finalize_trade(
            state=state,
            config=config,
            event_key=trade.event_key,
            symbol=trade.symbol,
            direction=trade.direction,
            event_time=trade.event_time,
            entry_time=trade.entry_time,
            exit_time=bar_time,
            signal_price=trade.signal_price,
            entry_price=trade.entry_price,
            stop_price=trade.stop_price,
            target_price=trade.target_price,
            target_rr_planned=trade.target_rr_planned,
            cost_r=trade.cost_r,
            entry_index=trade.entry_index,
            exit_index=bar_index,
            entry_delay_bars=trade.entry_delay_bars,
            gross_r=float(trade.target_r),
            exit_reason="target",
            bars_to_outcome=float(bar_index - trade.entry_index + 1),
            mfe_r=float(trade.mfe_r),
            mae_r=float(trade.mae_r),
        )
    if bar_index >= trade.horizon_end_index:
        clipped_r = max(-1.0, min(float(trade.target_r), float(last_close_r)))
        state.active_trade = None
        return _finalize_trade(
            state=state,
            config=config,
            event_key=trade.event_key,
            symbol=trade.symbol,
            direction=trade.direction,
            event_time=trade.event_time,
            entry_time=trade.entry_time,
            exit_time=bar_time,
            signal_price=trade.signal_price,
            entry_price=trade.entry_price,
            stop_price=trade.stop_price,
            target_price=trade.target_price,
            target_rr_planned=trade.target_rr_planned,
            cost_r=trade.cost_r,
            entry_index=trade.entry_index,
            exit_index=bar_index,
            entry_delay_bars=trade.entry_delay_bars,
            gross_r=float(clipped_r),
            exit_reason="timeout",
            bars_to_outcome=float(bar_index - trade.entry_index + 1),
            mfe_r=float(trade.mfe_r),
            mae_r=float(trade.mae_r),
        )
    return None


def _finalize_trade(
    *,
    state: ShadowRuntimeState,
    config: ZoneChannelProductionConfig,
    event_key: str,
    symbol: str,
    direction: str,
    event_time: pd.Timestamp,
    entry_time: pd.Timestamp,
    exit_time: pd.Timestamp,
    signal_price: float,
    entry_price: float,
    stop_price: float,
    target_price: float,
    target_rr_planned: float,
    cost_r: float,
    entry_index: int,
    exit_index: int,
    entry_delay_bars: int,
    gross_r: float,
    exit_reason: str,
    bars_to_outcome: float,
    mfe_r: float,
    mae_r: float,
) -> dict[str, Any]:
    net_r = float(gross_r) - float(cost_r)
    trade_row = {
        "symbol": symbol,
        "event_key": event_key,
        "direction": direction,
        "event_time": event_time,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "signal_price": float(signal_price),
        "entry_price": float(entry_price),
        "stop_price": float(stop_price),
        "target_price": float(target_price),
        "target_rr_planned": float(target_rr_planned),
        "cost_r": float(cost_r),
        "r_multiple_gross": float(gross_r),
        "r_multiple_net": float(net_r),
        "return_pct": float(config.risk.risk_fraction * net_r),
        "hold_bars": int(max(1, round((exit_time - entry_time).total_seconds() / 3600.0))),
        "exit_reason": exit_reason,
        "bars_to_outcome": float(bars_to_outcome),
        "entry_delay_bars": float(entry_delay_bars),
        "mfe_r": float(mfe_r),
        "mae_r": float(mae_r),
        "entry_index": int(entry_index),
        "exit_index": int(exit_index),
    }
    state.trades.append(trade_row)
    exit_day = pd.Timestamp(exit_time).tz_convert("UTC").floor("D")
    state.daily_net_r[exit_day] += float(net_r)
    if net_r < 0.0:
        state.consecutive_losses += 1
    else:
        state.consecutive_losses = 0
    if config.risk.cooldown_bars_after_exit > 0:
        state.cooldown_until = exit_time + pd.Timedelta(hours=int(config.risk.cooldown_bars_after_exit))
    row = _build_event_row(
        kind="trade",
        status=f"trade_closed_{exit_reason}",
        event_time=event_time,
        bar_time=exit_time,
        direction=direction,
        event_key=event_key,
        signal_price=signal_price,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        target_rr_planned=target_rr_planned,
        entry_time=entry_time,
        exit_time=exit_time,
        r_multiple_net=float(net_r),
        return_pct=float(config.risk.risk_fraction * net_r),
        bars_to_outcome=float(bars_to_outcome),
        entry_delay_bars=float(entry_delay_bars),
        mfe_r=float(mfe_r),
        mae_r=float(mae_r),
    )
    _record_status(state, row["status"])
    return row


def _accept_signal_row(
    *,
    state: ShadowRuntimeState,
    config: ZoneChannelProductionConfig,
    signal_row: pd.Series,
    signal_index: int,
) -> dict[str, Any]:
    event_time = pd.Timestamp(signal_row["event_time"]).tz_convert("UTC")
    direction = str(signal_row["direction"])
    decision = _build_event_row(
        kind="decision",
        status="pending",
        event_time=event_time,
        bar_time=event_time,
        direction=direction,
        event_key=str(signal_row["event_key"]),
        signal_price=float(signal_row["signal_price"]),
        entry_price=float(signal_row["entry_price"]),
        stop_price=float(signal_row["stop_price"]),
        target_price=float(signal_row["target_price"]),
        target_rr_planned=float(signal_row["target_rr_planned"]),
        gate_pass=bool(signal_row.get("gate_pass", True)),
        daily_signal_count_before=int(state.daily_signal_count[event_time.floor("D")]),
        daily_net_r_before=float(state.daily_net_r[event_time.floor("D")]),
        consecutive_losses_before=int(state.consecutive_losses),
    )
    event_day = event_time.floor("D")

    if not bool(signal_row.get("gate_pass", True)):
        decision["status"] = "filtered_by_gate"
        _record_status(state, decision["status"])
        return decision
    if config.risk.one_trade_at_a_time and (state.pending_order is not None or state.active_trade is not None):
        decision["status"] = "blocked_active_trade"
        _record_status(state, decision["status"])
        return decision
    if state.cooldown_until is not None and event_time < state.cooldown_until:
        decision["status"] = "blocked_cooldown"
        decision["blocked_until"] = _ts_iso(state.cooldown_until)
        _record_status(state, decision["status"])
        return decision
    if config.risk.max_signals_per_day > 0 and state.daily_signal_count[event_day] >= config.risk.max_signals_per_day:
        decision["status"] = "blocked_daily_signal_cap"
        _record_status(state, decision["status"])
        return decision
    if config.risk.max_daily_net_r_loss > 0.0 and state.daily_net_r[event_day] <= -abs(config.risk.max_daily_net_r_loss):
        decision["status"] = "blocked_daily_loss_limit"
        _record_status(state, decision["status"])
        return decision
    if config.risk.max_consecutive_losses > 0 and state.consecutive_losses >= config.risk.max_consecutive_losses:
        decision["status"] = "blocked_loss_streak"
        _record_status(state, decision["status"])
        return decision

    state.daily_signal_count[event_day] += 1
    if config.entry_mode not in {"passive_retest", "market_reclaim"}:
        decision["status"] = "blocked_unsupported_entry_mode"
        _record_status(state, decision["status"])
        return decision

    state.pending_order = PendingVirtualOrder(
        symbol=str(signal_row["symbol"]),
        event_key=str(signal_row["event_key"]),
        direction=direction,
        entry_mode=str(config.entry_mode),
        event_time=event_time,
        signal_index=int(signal_index),
        signal_price=float(signal_row["signal_price"]),
        entry_price=float(signal_row["entry_price"]),
        stop_price=float(signal_row["stop_price"]),
        target_price=float(signal_row["target_price"]),
        target_rr_planned=float(signal_row["target_rr_planned"]),
        cost_r=float(signal_row["cost_r"]),
        fill_deadline_index=int(signal_index + max(int(config.passive_entry_window_bars), 1))
        if config.entry_mode == "passive_retest"
        else int(signal_index + 1),
        gate_pass=bool(signal_row.get("gate_pass", True)),
    )
    decision["status"] = "accepted_pending_order" if config.entry_mode == "passive_retest" else "accepted_market_trade"
    decision["fill_deadline_index"] = int(state.pending_order.fill_deadline_index)
    _record_status(state, decision["status"])
    return decision


def _build_event_row(
    *,
    kind: str,
    status: str,
    event_time: pd.Timestamp,
    bar_time: pd.Timestamp,
    direction: str,
    event_key: str,
    signal_price: float,
    entry_price: float,
    stop_price: float,
    target_price: float,
    target_rr_planned: float,
    **extra: Any,
) -> dict[str, Any]:
    row = {
        "time": utc_now_iso(),
        "kind": kind,
        "status": status,
        "event_time": _ts_iso(event_time),
        "bar_time": _ts_iso(bar_time),
        "direction": direction,
        "event_key": event_key,
        "signal_price": float(signal_price),
        "entry_price": float(entry_price),
        "stop_price": float(stop_price),
        "target_price": float(target_price),
        "target_rr_planned": float(target_rr_planned),
    }
    row.update(extra)
    return row


def latest_complete_decision_time(base_frame: pd.DataFrame, config: ZoneChannelProductionConfig) -> pd.Timestamp | None:
    decision_bars = prepare_timeframe_bars(base_frame, config.decision_timeframe, atr_length=config.atr_length)
    if decision_bars.empty:
        return None
    max_base_close = pd.Timestamp(base_frame["close_time"].iloc[-1]).tz_convert("UTC")
    decision_bars = decision_bars[pd.to_datetime(decision_bars["close_time"], utc=True, errors="coerce") <= max_base_close].reset_index(drop=True)
    if decision_bars.empty:
        return None
    return pd.Timestamp(decision_bars["close_time"].iloc[-1]).tz_convert("UTC")


def run_single_scan(
    args: argparse.Namespace,
    client: BybitV5Client,
    config: ZoneChannelProductionConfig,
    heartbeat: dict[str, Any],
) -> None:
    update_heartbeat(args, heartbeat, "bootstrapping")
    base_frame = bootstrap_base_frame(args, client, config)
    market, signals, _ = build_production_inputs_from_base_frame(config, base_frame)
    state = ShadowRuntimeState()
    rows = sync_shadow_state(
        state=state,
        config=config,
        exec_frame=market.bars_by_timeframe[config.decision_timeframe],
        signals=signals,
        after_time=None,
    )
    record_shadow_rows(args, state, rows)
    snapshot = build_runtime_snapshot(
        args,
        config,
        state,
        signal_rows=len(signals),
        selected_signal_rows=int(signals["gate_pass"].astype(bool).sum()) if not signals.empty and "gate_pass" in signals.columns else 0,
        last_base_candle_time=pd.Timestamp(base_frame["close_time"].iloc[-1]).tz_convert("UTC") if not base_frame.empty else None,
        last_decision_time=state.last_synced_decision_time,
    )
    startup_row = {
        "time": utc_now_iso(),
        "kind": "startup_sync",
        "status": "synced",
        "symbol": config.symbol,
        "bootstrapped_base_bars": int(len(base_frame)),
        "decision_bars": int(len(market.bars_by_timeframe[config.decision_timeframe])),
        "signal_rows": int(len(signals)),
        "selected_signal_rows": int(signals["gate_pass"].astype(bool).sum()) if not signals.empty and "gate_pass" in signals.columns else 0,
        "summary": snapshot["summary"],
        "pending_order": snapshot["pending_order"],
        "active_trade": snapshot["active_trade"],
    }
    record_shadow_rows(args, state, [startup_row])
    summary = persist_shadow_artifacts(
        args,
        config,
        state,
        signal_rows=len(signals),
        selected_signal_rows=int(signals["gate_pass"].astype(bool).sum()) if not signals.empty and "gate_pass" in signals.columns else 0,
        window_start=pd.Timestamp(base_frame["open_time"].iloc[0]).tz_convert("UTC") if not base_frame.empty else None,
        window_end=pd.Timestamp(base_frame["close_time"].iloc[-1]).tz_convert("UTC") if not base_frame.empty else None,
    )
    update_heartbeat(
        args,
        heartbeat,
        "scan_complete",
        last_status="synced",
        last_base_candle_time=_ts_iso(pd.Timestamp(base_frame["close_time"].iloc[-1]).tz_convert("UTC")) if not base_frame.empty else None,
        last_decision_candle_time=_ts_iso(state.last_synced_decision_time),
        pending_order=snapshot["pending_order"],
        active_trade=snapshot["active_trade"],
        summary=summary,
    )
    print(f"{config.symbol}: synced {len(base_frame)} base candles, {len(signals)} signal rows")
    print(summary)


def run_websocket_loop(
    args: argparse.Namespace,
    client: BybitV5Client,
    config: ZoneChannelProductionConfig,
    heartbeat: dict[str, Any],
) -> None:
    try:
        from pybit.unified_trading import WebSocket
    except ImportError as exc:
        raise SystemExit("pybit is required for websocket mode. Install pybit in the active venv.") from exc

    update_heartbeat(args, heartbeat, "bootstrapping")
    base_frame = bootstrap_base_frame(args, client, config)
    market, signals, _ = build_production_inputs_from_base_frame(config, base_frame)
    state = ShadowRuntimeState()
    initial_rows = sync_shadow_state(
        state=state,
        config=config,
        exec_frame=market.bars_by_timeframe[config.decision_timeframe],
        signals=signals,
        after_time=None,
    )
    record_shadow_rows(args, state, initial_rows)
    snapshot = build_runtime_snapshot(
        args,
        config,
        state,
        signal_rows=len(signals),
        selected_signal_rows=int(signals["gate_pass"].astype(bool).sum()) if not signals.empty and "gate_pass" in signals.columns else 0,
        last_base_candle_time=pd.Timestamp(base_frame["close_time"].iloc[-1]).tz_convert("UTC") if not base_frame.empty else None,
        last_decision_time=state.last_synced_decision_time,
    )
    update_heartbeat(
        args,
        heartbeat,
        "bootstrapped",
        last_status="synced",
        last_base_candle_time=snapshot["last_base_candle_time"],
        last_decision_candle_time=snapshot["last_decision_candle_time"],
        pending_order=snapshot["pending_order"],
        active_trade=snapshot["active_trade"],
        summary=snapshot["summary"],
    )
    if args.scan_on_start:
        startup_row = {
            "time": utc_now_iso(),
            "kind": "startup_sync",
            "status": "synced",
            "symbol": config.symbol,
            "signal_rows": int(len(signals)),
            "selected_signal_rows": int(signals["gate_pass"].astype(bool).sum()) if not signals.empty and "gate_pass" in signals.columns else 0,
            "pending_order": snapshot["pending_order"],
            "active_trade": snapshot["active_trade"],
            "summary": snapshot["summary"],
        }
        record_shadow_rows(args, state, [startup_row])
    persist_shadow_artifacts(
        args,
        config,
        state,
        signal_rows=len(signals),
        selected_signal_rows=int(signals["gate_pass"].astype(bool).sum()) if not signals.empty and "gate_pass" in signals.columns else 0,
        window_start=pd.Timestamp(base_frame["open_time"].iloc[0]).tz_convert("UTC") if not base_frame.empty else None,
        window_end=pd.Timestamp(base_frame["close_time"].iloc[-1]).tz_convert("UTC") if not base_frame.empty else None,
    )

    event_queue: Queue = Queue()
    seen_candles: set[pd.Timestamp] = set()
    ws = WebSocket(
        channel_type="linear",
        testnet=args.websocket_testnet,
        demo=args.websocket_demo,
        ping_interval=args.websocket_ping_interval,
        ping_timeout=args.websocket_ping_timeout,
        retries=args.websocket_retries,
    )
    ws.kline_stream(
        interval=bybit_websocket_interval(config.base_interval),
        symbol=[config.symbol],
        callback=websocket_callback(config.symbol, config.base_interval, event_queue),
    )
    print(f"Zone-channel shadow websocket active symbol={config.symbol} interval={config.base_interval}")
    update_heartbeat(args, heartbeat, "websocket_active")

    processed = 0
    try:
        while True:
            try:
                candle = event_queue.get(timeout=args.websocket_idle_timeout)
            except Empty:
                print(f"WebSocket idle: no confirmed {config.base_interval} candles in {args.websocket_idle_timeout}s")
                update_heartbeat(args, heartbeat, "websocket_idle", last_idle_time=utc_now_iso())
                continue

            candle_time = pd.Timestamp(candle["open_time"]).tz_convert("UTC")
            if candle_time in seen_candles:
                continue
            seen_candles.add(candle_time)
            base_frame = append_closed_candle(base_frame, candle, args.lookback_days)
            last_base_close = pd.Timestamp(candle["close_time"]).tz_convert("UTC")
            heartbeat["last_websocket_event_time"] = utc_now_iso()
            heartbeat["last_base_candle_time"] = _ts_iso(last_base_close)
            print(f"{config.symbol}: confirmed {config.base_interval} candle {candle_time.isoformat()}")

            next_decision_time = latest_complete_decision_time(base_frame, config)
            if next_decision_time is None or (state.last_synced_decision_time is not None and next_decision_time <= state.last_synced_decision_time):
                update_heartbeat(args, heartbeat, "base_candle_appended")
                processed += 1
                if args.websocket_stop_after_events > 0 and processed >= args.websocket_stop_after_events:
                    break
                continue

            market, signals, _ = build_production_inputs_from_base_frame(config, base_frame)
            rows = sync_shadow_state(
                state=state,
                config=config,
                exec_frame=market.bars_by_timeframe[config.decision_timeframe],
                signals=signals,
                after_time=state.last_synced_decision_time,
            )
            record_shadow_rows(args, state, rows)
            for row in rows:
                print(f"{config.symbol}: {row['kind']} {row['status']}")

            snapshot = build_runtime_snapshot(
                args,
                config,
                state,
                signal_rows=len(signals),
                selected_signal_rows=int(signals["gate_pass"].astype(bool).sum()) if not signals.empty and "gate_pass" in signals.columns else 0,
                last_base_candle_time=last_base_close,
                last_decision_time=state.last_synced_decision_time,
            )
            summary = persist_shadow_artifacts(
                args,
                config,
                state,
                signal_rows=len(signals),
                selected_signal_rows=int(signals["gate_pass"].astype(bool).sum()) if not signals.empty and "gate_pass" in signals.columns else 0,
                window_start=pd.Timestamp(base_frame["open_time"].iloc[0]).tz_convert("UTC") if not base_frame.empty else None,
                window_end=last_base_close,
            )
            update_heartbeat(
                args,
                heartbeat,
                "decision_sync_complete",
                last_status=rows[-1]["status"] if rows else "synced_no_events",
                last_decision_candle_time=snapshot["last_decision_candle_time"],
                pending_order=snapshot["pending_order"],
                active_trade=snapshot["active_trade"],
                summary=summary,
            )
            processed += 1
            if args.websocket_stop_after_events > 0 and processed >= args.websocket_stop_after_events:
                break
    finally:
        ws.exit()


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    config = load_production_config(args.config)
    if args.symbol:
        config = replace(config, symbol=bybit_symbol(args.symbol))
    args.pid_file.parent.mkdir(parents=True, exist_ok=True)
    args.pid_file.write_text(str(os.getpid()), encoding="utf-8")
    save_production_config(args.config_copy, config)

    client = BybitV5Client(base_url=args.base_url)
    heartbeat = initial_heartbeat(args, config, "websocket" if args.loop else "single_scan")
    update_heartbeat(args, heartbeat, "starting")

    if args.loop:
        run_websocket_loop(args, client, config, heartbeat)
        return

    run_single_scan(args, client, config, heartbeat)


if __name__ == "__main__":
    main()
