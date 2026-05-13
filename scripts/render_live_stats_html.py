#!/usr/bin/env python3
"""Render an offline live-statistics HTML report from the bot ledger.

This is intentionally not imported or scheduled by the trading bot. Run it
manually from the repo root when you want a local dashboard snapshot:

    python scripts/render_live_stats_html.py

The report reads the local JSONL ledger and state files, optionally fetches
public Bybit OHLCV for recent closed trades, then writes a Plotly HTML page.
Plotly is loaded in the browser from the official CDN, so this script does not
add Plotly to the bot container or block the listener.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import urllib.parse
import urllib.request
import webbrowser
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / "bot" / "logs" / "trade_ledger.jsonl"
DEFAULT_ACTIVE = ROOT / "bot" / "logs" / "active_trades.json"
DEFAULT_RISK = ROOT / "bot" / "logs" / "risk_state.json"
DEFAULT_OUT = ROOT / "bot" / "logs" / "live_stats.html"
DEFAULT_CHART_CACHE = ROOT / "bot" / "logs" / "ohlcv_cache"

EXIT_TOKENS = ("EXIT", "TAKE PROFIT", "STOP LOSS", "TRAILING", "FLATTEN")
BYBIT_KLINE_URL = "https://api.bybit.com/v5/market/kline"
VALID_INTERVALS: tuple[tuple[int, str], ...] = (
    (1, "1"),
    (3, "3"),
    (5, "5"),
    (15, "15"),
    (30, "30"),
    (60, "60"),
    (120, "120"),
    (240, "240"),
    (360, "360"),
    (720, "720"),
    (1440, "D"),
)


def parse_float(value: Any) -> float | None:
    if value in (None, "", "-", "None"):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def parse_dt(value: Any) -> datetime | None:
    if value in (None, "", 0, "0"):
        return None
    if isinstance(value, (int, float)) or (isinstance(value, str) and value.isdigit()):
        try:
            number = float(value)
            if number > 1e12:
                number /= 1000.0
            return datetime.fromtimestamp(number, tz=timezone.utc)
        except (OSError, ValueError):
            return None
    try:
        text = str(value).replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def dt_to_iso(dt: datetime | None) -> str:
    return dt.isoformat().replace("+00:00", "Z") if dt else ""


def fmt_number(value: float | None, digits: int = 2) -> str:
    if value is None or not math.isfinite(value):
        return "-"
    return f"{value:,.{digits}f}"


def fmt_pct(value: float | None, digits: int = 1) -> str:
    if value is None or not math.isfinite(value):
        return "-"
    return f"{value * 100:.{digits}f}%"


def is_exit_fill(event: dict[str, Any]) -> bool:
    if event.get("type") != "fill":
        return False
    name = str(event.get("event") or "").upper()
    return any(token in name for token in EXIT_TOKENS)


def strategy_key(value: Any) -> str:
    text = str(value or "").strip()
    return text if text else "unknown"


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return default


def load_events(path: Path, since: datetime | None) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not path.exists():
        return events
    with path.open("r", encoding="utf-8-sig") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            ts = parse_dt(event.get("ts"))
            if ts is None:
                continue
            if since is not None and ts < since:
                continue
            event["_dt"] = ts
            events.append(event)
    events.sort(key=lambda item: item["_dt"])
    return events


@dataclass
class ClosedTrade:
    exit_time: datetime
    entry_time: datetime | None
    symbol: str
    strategy: str
    direction: str
    pnl: float
    entry: float | None = None
    sl: float | None = None
    tp: float | None = None
    exit_price: float | None = None
    qty: float | None = None
    risk_at_sl: float | None = None
    r_multiple: float | None = None
    event: str = ""
    exit_style: str = ""
    order_id: str = ""
    order_link_id: str = ""
    matched: bool = False
    time_in_trade_ms: float | None = None


def accepted_key(event: dict[str, Any]) -> str:
    extra = event.get("extra") if isinstance(event.get("extra"), dict) else {}
    for key in (
        event.get("order_link_id"),
        extra.get("order_link_id"),
        extra.get("Order link id"),
        event.get("order_id"),
    ):
        if key:
            return str(key)
    return ""


def exit_parent_key(event: dict[str, Any]) -> str:
    order_raw = event.get("order_raw") if isinstance(event.get("order_raw"), dict) else {}
    for key in (
        order_raw.get("parentOrderLinkId"),
        event.get("order_link_id"),
        order_raw.get("orderLinkId"),
        event.get("order_id"),
    ):
        if key:
            return str(key)
    return ""


def risk_from_signal(event: dict[str, Any]) -> float | None:
    extra = event.get("extra") if isinstance(event.get("extra"), dict) else {}
    for key in ("risk_at_sl", "Risk at SL"):
        value = extra.get(key)
        if value is None:
            continue
        number = parse_float(str(value).split()[0])
        if number is not None:
            return abs(number)
    return None


def signal_extra(event: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(event, dict):
        return {}
    extra = event.get("extra")
    return extra if isinstance(extra, dict) else {}


def build_closed_trades(events: list[dict[str, Any]]) -> list[ClosedTrade]:
    accepted_by_key: dict[str, dict[str, Any]] = {}
    pending: defaultdict[tuple[str, str, str], deque[dict[str, Any]]] = defaultdict(deque)
    closed: list[ClosedTrade] = []

    for event in events:
        if event.get("type") == "signal" and event.get("status") == "accepted":
            key = accepted_key(event)
            symbol = str(event.get("symbol") or "")
            strategy = strategy_key(event.get("strategy"))
            direction = str(event.get("direction") or "").lower()
            if key:
                accepted_by_key[key] = event
            pending[(symbol, strategy, direction)].append(event)
            continue

        if not is_exit_fill(event):
            continue
        pnl = parse_float(event.get("closed_pnl"))
        if pnl is None:
            continue

        exit_time = parse_dt(event.get("fill_time_ms")) or event["_dt"]
        symbol = str(event.get("symbol") or "")
        strategy = strategy_key(event.get("strategy"))
        direction = str(event.get("direction") or "").lower()
        parent_key = exit_parent_key(event)
        signal = accepted_by_key.get(parent_key) if parent_key else None
        if signal is None:
            queue_key = (symbol, strategy, direction)
            queue = pending.get(queue_key)
            if queue:
                signal = queue.pop()
        elif parent_key:
            queue_key = (
                str(signal.get("symbol") or ""),
                strategy_key(signal.get("strategy")),
                str(signal.get("direction") or "").lower(),
            )
            queue = pending.get(queue_key)
            if queue:
                try:
                    queue.remove(signal)
                except ValueError:
                    pass

        extra = signal_extra(signal)
        risk = risk_from_signal(signal) if signal else None
        r_multiple = pnl / risk if risk not in (None, 0.0) else None
        closed.append(
            ClosedTrade(
                exit_time=exit_time,
                entry_time=parse_dt(signal.get("ts")) if signal else None,
                symbol=symbol,
                strategy=strategy,
                direction=direction,
                pnl=pnl,
                entry=parse_float(signal.get("entry")) if signal else parse_float(event.get("expected_entry")),
                sl=parse_float(signal.get("sl")) if signal else None,
                tp=parse_float(signal.get("tp")) if signal else None,
                exit_price=parse_float(event.get("price")),
                qty=parse_float(event.get("qty")),
                risk_at_sl=risk,
                r_multiple=r_multiple,
                event=str(event.get("event") or ""),
                exit_style=str(extra.get("exit_style") or extra.get("Exit style") or ""),
                order_id=str(event.get("order_id") or ""),
                order_link_id=parent_key,
                matched=signal is not None,
                time_in_trade_ms=parse_float(event.get("time_in_trade_ms")),
            )
        )
    closed.sort(key=lambda item: item.exit_time)
    return closed


def perf_stats(trades: list[ClosedTrade]) -> dict[str, Any]:
    pnls = [t.pnl for t in trades]
    wins = [x for x in pnls if x > 0]
    losses = [x for x in pnls if x < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for pnl in pnls:
        cum += pnl
        peak = max(peak, cum)
        max_dd = min(max_dd, cum - peak)

    streak_side = "-"
    streak_count = 0
    for pnl in reversed(pnls):
        side = "W" if pnl > 0 else "L" if pnl < 0 else "BE"
        if streak_side == "-":
            streak_side = side
        if side == streak_side:
            streak_count += 1
        else:
            break

    r_values = [t.r_multiple for t in trades if t.r_multiple is not None and math.isfinite(t.r_multiple)]
    return {
        "closed": len(trades),
        "pnl": sum(pnls),
        "wins": len(wins),
        "losses": len(losses),
        "breakeven": len(pnls) - len(wins) - len(losses),
        "winrate": len(wins) / len(trades) if trades else None,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else None,
        "avg_pnl": statistics.mean(pnls) if pnls else None,
        "median_pnl": statistics.median(pnls) if pnls else None,
        "max_win": max(pnls) if pnls else None,
        "max_loss": min(pnls) if pnls else None,
        "max_drawdown": max_dd,
        "current_streak": f"{streak_side}{streak_count}" if pnls else "-",
        "avg_r": statistics.mean(r_values) if r_values else None,
    }


def group_stats(trades: list[ClosedTrade], attr: str) -> list[dict[str, Any]]:
    groups: defaultdict[str, list[ClosedTrade]] = defaultdict(list)
    for trade in trades:
        groups[str(getattr(trade, attr) or "unknown")].append(trade)
    rows = []
    for name, items in groups.items():
        stats = perf_stats(items)
        rows.append({"name": name, **stats})
    rows.sort(key=lambda row: (row["pnl"], row["closed"]), reverse=True)
    return rows


def choose_trade_interval(entry_time: datetime, exit_time: datetime, target_bars: int = 50) -> tuple[int, str]:
    duration_minutes = max((exit_time - entry_time).total_seconds() / 60.0, 1.0)
    target_minutes = max(duration_minutes / max(target_bars, 1), 1.0)
    return min(VALID_INTERVALS, key=lambda item: abs(math.log(item[0] / target_minutes)))


def cache_key(symbol: str, interval: str, start_ms: int, end_ms: int) -> str:
    safe_interval = interval.replace("/", "_")
    return f"{symbol}_{safe_interval}_{start_ms}_{end_ms}.json"


def fetch_bybit_ohlcv(
    *,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    cache_dir: Path | None,
) -> list[dict[str, Any]]:
    cache_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / cache_key(symbol, interval, start_ms, end_ms)
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(cached, list):
                    return cached
            except Exception:
                pass

    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "start": str(start_ms),
        "end": str(end_ms),
        "limit": "1000",
    }
    url = BYBIT_KLINE_URL + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("retCode") != 0:
        raise RuntimeError(f"{symbol} kline fetch failed: {payload.get('retMsg', '?')}")

    rows = payload.get("result", {}).get("list", []) or []
    candles: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) < 5:
            continue
        try:
            candles.append(
                {
                    "ts": int(float(row[0])),
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]) if len(row) > 5 else None,
                }
            )
        except (TypeError, ValueError):
            continue
    candles.sort(key=lambda item: item["ts"])
    if cache_path is not None:
        cache_path.write_text(json.dumps(candles, separators=(",", ":")), encoding="utf-8")
    return candles


def build_trade_chart_data(
    trades: list[ClosedTrade],
    *,
    max_charts: int,
    cache_dir: Path | None,
    target_trade_bars: int = 50,
    front_bars: int = 50,
) -> list[dict[str, Any]]:
    if max_charts <= 0:
        return []
    out: list[dict[str, Any]] = []
    recent = [trade for trade in trades if trade.entry_time is not None][-max_charts:]
    for idx, trade in enumerate(reversed(recent), 1):
        if trade.entry_time is None:
            continue
        interval_minutes, interval = choose_trade_interval(
            trade.entry_time,
            trade.exit_time,
            target_bars=target_trade_bars,
        )
        interval_ms = interval_minutes * 60 * 1000
        start_ms = int(trade.entry_time.timestamp() * 1000) - front_bars * interval_ms
        end_padding_ms = max(10 * interval_ms, int((trade.exit_time - trade.entry_time).total_seconds() * 1000 * 0.2))
        end_ms = int(trade.exit_time.timestamp() * 1000) + end_padding_ms
        try:
            candles = fetch_bybit_ohlcv(
                symbol=trade.symbol,
                interval=interval,
                start_ms=start_ms,
                end_ms=end_ms,
                cache_dir=cache_dir,
            )
        except Exception as exc:
            out.append(
                {
                    "id": f"trade_{idx}",
                    "label": f"{trade.exit_time:%Y-%m-%d %H:%M} {trade.symbol} {trade.strategy} ({trade.pnl:+.2f})",
                    "error": str(exc),
                }
            )
            continue
        if not candles:
            out.append(
                {
                    "id": f"trade_{idx}",
                    "label": f"{trade.exit_time:%Y-%m-%d %H:%M} {trade.symbol} {trade.strategy} ({trade.pnl:+.2f})",
                    "error": "no candles returned",
                }
            )
            continue
        x = [
            datetime.fromtimestamp(candle["ts"] / 1000.0, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
            for candle in candles
        ]
        fixed_tp = (trade.exit_style or "").lower() == "fixed_tp"
        out.append(
            {
                "id": f"trade_{idx}",
                "label": f"{trade.exit_time:%Y-%m-%d %H:%M} {trade.symbol} {trade.strategy} {trade.direction} ({trade.pnl:+.2f})",
                "symbol": trade.symbol,
                "strategy": trade.strategy,
                "direction": trade.direction,
                "event": trade.event,
                "pnl": trade.pnl,
                "r_multiple": trade.r_multiple,
                "interval": interval,
                "interval_minutes": interval_minutes,
                "bars": len(candles),
                "entry_time": dt_to_iso(trade.entry_time),
                "exit_time": dt_to_iso(trade.exit_time),
                "entry": trade.entry,
                "sl": trade.sl,
                "tp": trade.tp if fixed_tp else None,
                "planned_tp": trade.tp,
                "exit_price": trade.exit_price,
                "exit_style": trade.exit_style or "-",
                "candles": {
                    "x": x,
                    "open": [candle["open"] for candle in candles],
                    "high": [candle["high"] for candle in candles],
                    "low": [candle["low"] for candle in candles],
                    "close": [candle["close"] for candle in candles],
                },
            }
        )
    return out


def daily_summary(trades: list[ClosedTrade], events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    days: defaultdict[str, dict[str, Any]] = defaultdict(
        lambda: {"day": "", "pnl": 0.0, "closed": 0, "wins": 0, "losses": 0, "accepted": 0, "rejected": 0}
    )
    for trade in trades:
        key = trade.exit_time.date().isoformat()
        row = days[key]
        row["day"] = key
        row["pnl"] += trade.pnl
        row["closed"] += 1
        row["wins"] += int(trade.pnl > 0)
        row["losses"] += int(trade.pnl < 0)
    for event in events:
        if event.get("type") != "signal":
            continue
        key = event["_dt"].date().isoformat()
        row = days[key]
        row["day"] = key
        if event.get("status") == "accepted":
            row["accepted"] += 1
        elif event.get("status") == "rejected":
            row["rejected"] += 1
    return [days[key] for key in sorted(days)]


def estimate_equity_base(
    trades: list[ClosedTrade],
    events: list[dict[str, Any]],
    risk_state: dict[str, Any],
    manual_base: float | None,
) -> tuple[float | None, str]:
    if manual_base is not None:
        return manual_base, "manual --base-equity"

    heartbeats = [
        event for event in events
        if event.get("type") == "heartbeat" and parse_float(event.get("Current equity")) is not None
    ]
    if heartbeats:
        latest = max(heartbeats, key=lambda item: item["_dt"])
        equity = parse_float(latest.get("Current equity"))
        cum_until = sum(t.pnl for t in trades if t.exit_time <= latest["_dt"])
        if equity is not None:
            return equity - cum_until, f"latest heartbeat {dt_to_iso(latest['_dt'])}"

    day_start_equity = parse_float(risk_state.get("day_start_equity"))
    day_date = str(risk_state.get("day_date") or "")
    if day_start_equity is not None and day_date:
        try:
            day_start = datetime.fromisoformat(day_date).replace(tzinfo=timezone.utc)
            cum_before = sum(t.pnl for t in trades if t.exit_time < day_start)
            return day_start_equity - cum_before, f"risk_state day start {day_date}"
        except ValueError:
            pass

    return None, "unavailable; showing realized PnL only"


def build_chart_payload(
    trades: list[ClosedTrade],
    events: list[dict[str, Any]],
    risk_state: dict[str, Any],
    base_equity: float | None,
    trade_charts: list[dict[str, Any]],
) -> dict[str, Any]:
    x: list[str] = []
    cum_pnl: list[float] = []
    equity: list[float | None] = []
    drawdown: list[float] = []
    scatter_size: list[float] = []
    scatter_text: list[str] = []
    cumulative = 0.0
    peak = 0.0
    for trade in trades:
        cumulative += trade.pnl
        peak = max(peak, cumulative)
        x.append(dt_to_iso(trade.exit_time))
        cum_pnl.append(round(cumulative, 8))
        equity.append(round(base_equity + cumulative, 8) if base_equity is not None else None)
        drawdown.append(round(cumulative - peak, 8))
        scatter_size.append(max(7.0, min(22.0, 6.0 + abs(trade.pnl) / 4.0)))
        scatter_text.append(
            f"{trade.symbol}<br>{trade.strategy}<br>{trade.direction}<br>"
            f"PnL {trade.pnl:.2f}<br>{trade.event}"
        )

    strategy_curves = []
    for row in group_stats(trades, "strategy"):
        name = row["name"]
        c = 0.0
        xs: list[str] = []
        ys: list[float] = []
        for trade in trades:
            if trade.strategy != name:
                continue
            c += trade.pnl
            xs.append(dt_to_iso(trade.exit_time))
            ys.append(round(c, 8))
        strategy_curves.append({"name": name, "x": xs, "y": ys})

    daily = daily_summary(trades, events)
    hourly = [[0.0 for _ in range(24)] for _ in range(7)]
    hourly_counts = [[0 for _ in range(24)] for _ in range(7)]
    for trade in trades:
        dow = trade.exit_time.weekday()
        hour = trade.exit_time.hour
        hourly[dow][hour] += trade.pnl
        hourly_counts[dow][hour] += 1

    return {
        "equity": {
            "x": x,
            "cum_pnl": cum_pnl,
            "equity": equity,
            "drawdown": drawdown,
            "trade_pnl": [round(t.pnl, 8) for t in trades],
            "trade_color": ["#1f9d55" if t.pnl > 0 else "#dc3545" if t.pnl < 0 else "#6c757d" for t in trades],
            "trade_size": scatter_size,
            "trade_text": scatter_text,
        },
        "strategy_curves": strategy_curves,
        "daily": daily,
        "hourly": {
            "z": hourly,
            "counts": hourly_counts,
            "x": [str(h) for h in range(24)],
            "y": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        },
        "trade_charts": trade_charts,
        "risk_state": risk_state,
    }


def rejection_rows(events: list[dict[str, Any]], limit: int = 12) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    for event in events:
        if event.get("type") == "signal" and event.get("status") == "rejected":
            reason = str(event.get("reason") or "unknown")
            if len(reason) > 120:
                reason = reason[:117] + "..."
            counts[reason] += 1
    return counts.most_common(limit)


def active_trade_rows(active: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    if not isinstance(active, dict):
        return rows
    for symbol, trade in active.items():
        if not isinstance(trade, dict):
            continue
        rows.append({
            "symbol": symbol,
            "strategy": strategy_key(trade.get("strategy")),
            "direction": trade.get("direction", "-"),
            "entry": parse_float(trade.get("entry")),
            "sl": parse_float(trade.get("sl")),
            "tp": parse_float(trade.get("tp1")),
            "opened": parse_dt(trade.get("opened_at")),
        })
    rows.sort(key=lambda row: row["symbol"])
    return rows


def html_table(headers: list[str], rows: list[list[Any]], *, empty: str = "No rows") -> str:
    if not rows:
        return f"<p class='muted'>{escape(empty)}</p>"
    head = "".join(f"<th>{escape(str(col))}</th>" for col in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in row) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def render_html(
    *,
    events: list[dict[str, Any]],
    trades: list[ClosedTrade],
    trade_charts: list[dict[str, Any]],
    active: dict[str, Any],
    risk_state: dict[str, Any],
    output: Path,
    ledger: Path,
    base_equity: float | None,
    base_source: str,
) -> str:
    stats = perf_stats(trades)
    strategy_stats = group_stats(trades, "strategy")
    symbol_stats = group_stats(trades, "symbol")
    accepted_count = sum(1 for e in events if e.get("type") == "signal" and e.get("status") == "accepted")
    rejected_count = sum(1 for e in events if e.get("type") == "signal" and e.get("status") == "rejected")
    fill_count = sum(1 for e in events if e.get("type") == "fill")
    latest_event_dt = max((e["_dt"] for e in events), default=None)
    active_rows = active_trade_rows(active)
    chart_payload = build_chart_payload(trades, events, risk_state, base_equity, trade_charts)

    kpis = [
        ("Closed PnL", fmt_number(stats["pnl"])),
        ("Closed Trades", stats["closed"]),
        ("Winrate", fmt_pct(stats["winrate"])),
        ("Profit Factor", fmt_number(stats["profit_factor"])),
        ("Max Drawdown", fmt_number(stats["max_drawdown"])),
        ("Current Streak", stats["current_streak"]),
        ("Accepted / Rejected", f"{accepted_count} / {rejected_count}"),
        ("Open Trades", len(active_rows)),
    ]
    kpi_html = "".join(
        f"<div class='kpi'><div class='kpi-label'>{escape(str(label))}</div>"
        f"<div class='kpi-value'>{escape(str(value))}</div></div>"
        for label, value in kpis
    )

    strategy_table = html_table(
        ["Strategy", "Closed", "PnL", "Winrate", "PF", "Avg", "Max DD", "Avg R"],
        [
            [
                row["name"],
                row["closed"],
                fmt_number(row["pnl"]),
                fmt_pct(row["winrate"]),
                fmt_number(row["profit_factor"]),
                fmt_number(row["avg_pnl"]),
                fmt_number(row["max_drawdown"]),
                fmt_number(row["avg_r"]),
            ]
            for row in strategy_stats
        ],
    )
    symbol_table = html_table(
        ["Symbol", "Closed", "PnL", "Winrate", "PF", "Avg", "Max DD"],
        [
            [
                row["name"],
                row["closed"],
                fmt_number(row["pnl"]),
                fmt_pct(row["winrate"]),
                fmt_number(row["profit_factor"]),
                fmt_number(row["avg_pnl"]),
                fmt_number(row["max_drawdown"]),
            ]
            for row in symbol_stats
        ],
    )
    recent_table = html_table(
        ["Exit UTC", "Symbol", "Strategy", "Dir", "Event", "Exit Style", "PnL", "R", "Entry", "SL", "TP", "Exit", "Matched"],
        [
            [
                dt_to_iso(t.exit_time),
                t.symbol,
                t.strategy,
                t.direction,
                t.event,
                t.exit_style or "-",
                fmt_number(t.pnl),
                fmt_number(t.r_multiple),
                fmt_number(t.entry, 5),
                fmt_number(t.sl, 5),
                fmt_number(t.tp, 5),
                fmt_number(t.exit_price, 5),
                "yes" if t.matched else "no",
            ]
            for t in trades[-40:][::-1]
        ],
        empty="No closed trades in the selected ledger window.",
    )
    active_table = html_table(
        ["Symbol", "Strategy", "Dir", "Entry", "SL", "TP", "Opened UTC"],
        [
            [
                row["symbol"],
                row["strategy"],
                row["direction"],
                fmt_number(row["entry"], 5),
                fmt_number(row["sl"], 5),
                fmt_number(row["tp"], 5),
                dt_to_iso(row["opened"]),
            ]
            for row in active_rows
        ],
        empty="No active trade metadata found.",
    )
    rejection_table = html_table(
        ["Rejected Reason", "Count"],
        [[reason, count] for reason, count in rejection_rows(events)],
        empty="No rejected signals in the selected ledger window.",
    )
    if trade_charts:
        chart_options = "".join(
            f"<option value='{idx}'>{escape(str(chart.get('label', f'Trade {idx + 1}')))}</option>"
            for idx, chart in enumerate(trade_charts)
        )
        trade_chart_section = f"""
    <section class="panel">
      <div class="trade-chart-head">
        <h2>Recent Trade Chart</h2>
        <div class="trade-chart-controls">
          <button id="prevTrade" type="button">Prev</button>
          <select id="tradeSelect" aria-label="Select recent trade">{chart_options}</select>
          <button id="nextTrade" type="button">Next</button>
          <button id="downloadTradeChart" type="button">Download Image</button>
        </div>
      </div>
      <div id="tradeChartMeta" class="muted"></div>
      <div id="tradeChart" class="chart trade"></div>
    </section>
"""
    else:
        trade_chart_section = """
    <section class="panel">
      <h2>Recent Trade Chart</h2>
      <p class="muted">No trade charts were generated. Use --trade-charts N to fetch recent OHLCV snapshots.</p>
    </section>
"""

    report_meta = {
        "generated_at": dt_to_iso(datetime.now(timezone.utc)),
        "ledger": str(ledger),
        "output": str(output),
        "events": len(events),
        "fills": fill_count,
        "latest_event": dt_to_iso(latest_event_dt),
        "base_equity_source": base_source,
    }
    payload = json.dumps(chart_payload, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Bot Live Stats</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #202733;
      --muted: #657083;
      --grid: #e5e8ef;
      --green: #1f9d55;
      --red: #dc3545;
      --blue: #2f6fed;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .wrap {{ width: min(1500px, calc(100vw - 48px)); margin: 28px auto 56px; }}
    h1 {{ margin: 0 0 4px; font-size: 28px; font-weight: 700; letter-spacing: 0; }}
    h2 {{ margin: 0 0 14px; font-size: 18px; font-weight: 650; letter-spacing: 0; }}
    .muted {{ color: var(--muted); font-size: 13px; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(180px, 1fr)); gap: 12px; margin: 20px 0; }}
    .kpi, .panel {{
      background: var(--panel);
      border: 1px solid var(--grid);
      border-radius: 8px;
      box-shadow: 0 1px 2px rgba(18, 25, 38, 0.05);
    }}
    .kpi {{ padding: 14px 16px; min-height: 82px; }}
    .kpi-label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
    .kpi-value {{ margin-top: 8px; font-size: 23px; font-weight: 700; white-space: nowrap; }}
    .panel {{ padding: 16px; margin: 14px 0; }}
    .two {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
    .chart {{ width: 100%; height: 430px; }}
    .chart.small {{ height: 360px; }}
    .chart.trade {{ height: 620px; }}
    .trade-chart-head {{ display: flex; align-items: center; justify-content: space-between; gap: 14px; flex-wrap: wrap; }}
    .trade-chart-controls {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }}
    .trade-chart-controls select {{
      min-width: min(680px, 72vw);
      max-width: 100%;
      padding: 8px 10px;
      border: 1px solid var(--grid);
      border-radius: 6px;
      background: white;
      color: var(--text);
    }}
    button {{
      padding: 8px 11px;
      border: 1px solid var(--grid);
      border-radius: 6px;
      background: #f8fafc;
      color: var(--text);
      cursor: pointer;
    }}
    button:hover {{ background: #eef2f7; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 8px 9px; border-bottom: 1px solid var(--grid); text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-weight: 650; background: #fafbfc; position: sticky; top: 0; }}
    .table-scroll {{ max-height: 520px; overflow: auto; }}
    .meta {{ display: flex; gap: 16px; flex-wrap: wrap; margin-top: 8px; }}
    .note {{ margin-top: 10px; padding: 10px 12px; border-left: 3px solid var(--blue); background: #eef4ff; color: #31415f; }}
    @media (max-width: 980px) {{
      .wrap {{ width: min(100vw - 24px, 1500px); margin-top: 18px; }}
      .grid {{ grid-template-columns: repeat(2, minmax(150px, 1fr)); }}
      .two {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class="wrap">
    <h1>Bot Live Stats</h1>
    <div class="muted">Generated {escape(report_meta["generated_at"])} from {escape(report_meta["ledger"])}</div>
    <div class="meta muted">
      <span>Events: {report_meta["events"]}</span>
      <span>Fills: {report_meta["fills"]}</span>
      <span>Latest event: {escape(report_meta["latest_event"] or "-")}</span>
      <span>Equity basis: {escape(report_meta["base_equity_source"])}</span>
    </div>
    <div class="note">Equity is reconstructed from local closed PnL plus the newest heartbeat or risk-state baseline when available. It does not include unrealized PnL unless it was already reflected in the baseline.</div>

    <section class="grid">{kpi_html}</section>

    <section class="panel">
      <h2>Equity Curve</h2>
      <div id="equityChart" class="chart"></div>
    </section>

    <section class="two">
      <div class="panel">
        <h2>Daily PnL And Signals</h2>
        <div id="dailyChart" class="chart small"></div>
      </div>
      <div class="panel">
        <h2>Strategy Curves</h2>
        <div id="strategyChart" class="chart small"></div>
      </div>
    </section>

    <section class="two">
      <div class="panel">
        <h2>Trade PnL Scatter</h2>
        <div id="scatterChart" class="chart small"></div>
      </div>
      <div class="panel">
        <h2>Exit Hour PnL Heatmap</h2>
        <div id="hourlyChart" class="chart small"></div>
      </div>
    </section>

    {trade_chart_section}

    <section class="two">
      <div class="panel"><h2>By Strategy</h2><div class="table-scroll">{strategy_table}</div></div>
      <div class="panel"><h2>By Symbol</h2><div class="table-scroll">{symbol_table}</div></div>
    </section>

    <section class="panel"><h2>Open Trade Metadata</h2><div class="table-scroll">{active_table}</div></section>
    <section class="panel"><h2>Recent Closed Trades</h2><div class="table-scroll">{recent_table}</div></section>
    <section class="panel"><h2>Top Rejection Reasons</h2><div class="table-scroll">{rejection_table}</div></section>
  </main>

  <script>
    const payload = {payload};
    const plotConfig = {{responsive: true, displaylogo: false}};
    const axisStyle = {{gridcolor: "#e5e8ef", zerolinecolor: "#d9dee8"}};
    const numeric = arr => arr.map(v => (v === null || v === undefined ? null : Number(v)));
    const hasEquity = payload.equity.equity.some(v => v !== null);
    const equityY = hasEquity ? numeric(payload.equity.equity) : numeric(payload.equity.cum_pnl);

    Plotly.newPlot("equityChart", [
      {{x: payload.equity.x, y: equityY,
        type: "scatter", mode: "lines+markers", name: hasEquity ? "Estimated equity" : "Realized PnL",
        line: {{color: "#2f6fed", width: 2}}}},
      {{x: payload.equity.x, y: numeric(payload.equity.drawdown), type: "scatter", mode: "lines", name: "Drawdown",
        yaxis: "y2", fill: "tozeroy", line: {{color: "#dc3545", width: 1.5}}}}
    ], {{
      margin: {{l: 58, r: 58, t: 8, b: 42}},
      paper_bgcolor: "white", plot_bgcolor: "white",
      legend: {{orientation: "h", y: 1.08}},
      xaxis: axisStyle,
      yaxis: {{...axisStyle, type: "linear", tickformat: ",.2f", title: "Equity / PnL"}},
      yaxis2: {{...axisStyle, type: "linear", tickformat: ",.2f", title: "Drawdown", overlaying: "y", side: "right"}}
    }}, plotConfig);

    Plotly.newPlot("dailyChart", [
      {{x: payload.daily.map(d => d.day), y: numeric(payload.daily.map(d => d.pnl)), type: "bar", name: "Daily PnL",
        marker: {{color: payload.daily.map(d => d.pnl >= 0 ? "#1f9d55" : "#dc3545")}}}},
      {{x: payload.daily.map(d => d.day), y: numeric(payload.daily.map(d => d.accepted)), type: "scatter", mode: "lines+markers",
        name: "Accepted", yaxis: "y2", line: {{color: "#2f6fed"}}}},
      {{x: payload.daily.map(d => d.day), y: numeric(payload.daily.map(d => d.rejected)), type: "scatter", mode: "lines+markers",
        name: "Rejected", yaxis: "y2", line: {{color: "#8a93a5"}}}}
    ], {{
      margin: {{l: 52, r: 48, t: 8, b: 42}},
      paper_bgcolor: "white", plot_bgcolor: "white",
      legend: {{orientation: "h", y: 1.1}},
      xaxis: axisStyle,
      yaxis: {{...axisStyle, type: "linear", tickformat: ",.2f", title: "PnL"}},
      yaxis2: {{...axisStyle, type: "linear", tickformat: "d", title: "Signals", overlaying: "y", side: "right"}}
    }}, plotConfig);

    Plotly.newPlot("strategyChart", payload.strategy_curves.map(curve => ({{
      x: curve.x, y: numeric(curve.y), type: "scatter", mode: "lines+markers", name: curve.name
    }})), {{
      margin: {{l: 52, r: 16, t: 8, b: 42}},
      paper_bgcolor: "white", plot_bgcolor: "white",
      legend: {{orientation: "h", y: 1.1}},
      xaxis: axisStyle,
      yaxis: {{...axisStyle, type: "linear", tickformat: ",.2f", title: "Cumulative PnL"}}
    }}, plotConfig);

    Plotly.newPlot("scatterChart", [{{
      x: payload.equity.x,
      y: numeric(payload.equity.trade_pnl),
      text: payload.equity.trade_text,
      type: "scatter",
      mode: "markers",
      marker: {{size: payload.equity.trade_size, color: payload.equity.trade_color, opacity: 0.82}},
      name: "Closed trade"
    }}], {{
      margin: {{l: 52, r: 16, t: 8, b: 42}},
      paper_bgcolor: "white", plot_bgcolor: "white",
      xaxis: axisStyle,
      yaxis: {{...axisStyle, type: "linear", tickformat: ",.2f", title: "Trade PnL"}},
      showlegend: false
    }}, plotConfig);

    Plotly.newPlot("hourlyChart", [{{
      x: payload.hourly.x, y: payload.hourly.y, z: payload.hourly.z,
      type: "heatmap", colorscale: [[0, "#dc3545"], [0.5, "#f8f9fa"], [1, "#1f9d55"]],
      hovertemplate: "%{{y}} %{{x}}:00<br>PnL %{{z:.2f}}<extra></extra>"
    }}], {{
      margin: {{l: 52, r: 16, t: 8, b: 42}},
      paper_bgcolor: "white", plot_bgcolor: "white",
      xaxis: {{...axisStyle, type: "category", title: "UTC hour"}},
      yaxis: {{...axisStyle, type: "category"}}
    }}, plotConfig);

    function finiteNumber(value) {{
      return value !== null && value !== undefined && Number.isFinite(Number(value));
    }}

    function horizontalTrace(name, x0, x1, y, color, dash) {{
      if (!finiteNumber(y)) return null;
      return {{
        x: [x0, x1],
        y: [Number(y), Number(y)],
        type: "scatter",
        mode: "lines",
        name: name,
        line: {{color: color, width: 1.4, dash: dash || "solid"}},
        hoverinfo: "skip"
      }};
    }}

    function renderTradeChart(index) {{
      const charts = payload.trade_charts || [];
      const chart = charts[index];
      const meta = document.getElementById("tradeChartMeta");
      if (!chart) {{
        if (meta) meta.textContent = "No chart selected.";
        return;
      }}
      if (chart.error) {{
        if (meta) meta.textContent = chart.label + " | " + chart.error;
        Plotly.purge("tradeChart");
        return;
      }}
      if (meta) {{
        meta.textContent = chart.symbol + " | " + chart.strategy + " | " + chart.direction +
          " | " + chart.event + " | interval " + chart.interval + " | " + chart.bars +
          " bars | PnL " + Number(chart.pnl).toFixed(2) +
          (finiteNumber(chart.r_multiple) ? " | R " + Number(chart.r_multiple).toFixed(2) : "") +
          " | exit style " + chart.exit_style;
      }}
      const x0 = chart.entry_time;
      const x1 = chart.exit_time;
      const candleOpen = chart.candles.open.map(Number);
      const candleHigh = chart.candles.high.map(Number);
      const candleLow = chart.candles.low.map(Number);
      const candleClose = chart.candles.close.map(Number);
      const traces = [
        {{
          x: chart.candles.x,
          open: candleOpen,
          high: candleHigh,
          low: candleLow,
          close: candleClose,
          type: "candlestick",
          name: chart.symbol,
          increasing: {{line: {{color: "#26a69a", width: 1}}, fillcolor: "rgba(38,166,154,0.72)"}},
          decreasing: {{line: {{color: "#ef5350", width: 1}}, fillcolor: "rgba(239,83,80,0.72)"}}
        }}
      ];
      [
        horizontalTrace("Entry", x0, x1, chart.entry, "#111827", "solid"),
        horizontalTrace("SL", x0, x1, chart.sl, "#dc3545", "dot"),
        horizontalTrace("TP", x0, x1, chart.tp, "#1f9d55", "dot")
      ].forEach(trace => {{ if (trace) traces.push(trace); }});
      if (finiteNumber(chart.entry)) {{
        traces.push({{
          x: [x0], y: [Number(chart.entry)], type: "scatter", mode: "markers", name: "Entry marker",
          marker: {{symbol: chart.direction === "long" ? "triangle-up" : "triangle-down", color: "#111827", size: 11}}
        }});
      }}
      if (finiteNumber(chart.exit_price)) {{
        traces.push({{
          x: [x1], y: [Number(chart.exit_price)], type: "scatter", mode: "markers", name: "Exit marker",
          marker: {{symbol: "x", color: chart.pnl >= 0 ? "#1f9d55" : "#dc3545", size: 12, line: {{width: 2}}}}
        }});
      }}

      const shapes = [];
      if (finiteNumber(chart.entry) && finiteNumber(chart.sl)) {{
        shapes.push({{
          type: "rect", xref: "x", yref: "y",
          x0: x0, x1: x1,
          y0: Math.min(Number(chart.entry), Number(chart.sl)),
          y1: Math.max(Number(chart.entry), Number(chart.sl)),
          fillcolor: "rgba(220,53,69,0.18)",
          line: {{color: "rgba(220,53,69,0.75)", width: 1}}
        }});
      }}
      if (finiteNumber(chart.entry) && finiteNumber(chart.tp)) {{
        shapes.push({{
          type: "rect", xref: "x", yref: "y",
          x0: x0, x1: x1,
          y0: Math.min(Number(chart.entry), Number(chart.tp)),
          y1: Math.max(Number(chart.entry), Number(chart.tp)),
          fillcolor: "rgba(31,157,85,0.18)",
          line: {{color: "rgba(31,157,85,0.75)", width: 1}}
        }});
      }}
      shapes.push(
        {{type: "line", xref: "x", yref: "paper", x0: x0, x1: x0, y0: 0, y1: 1, line: {{color: "#111827", width: 1, dash: "dot"}}}},
        {{type: "line", xref: "x", yref: "paper", x0: x1, x1: x1, y0: 0, y1: 1, line: {{color: chart.pnl >= 0 ? "#1f9d55" : "#dc3545", width: 1, dash: "dot"}}}}
      );

      Plotly.newPlot("tradeChart", traces, {{
        margin: {{l: 58, r: 24, t: 16, b: 42}},
        paper_bgcolor: "white",
        plot_bgcolor: "white",
        legend: {{orientation: "h", y: 1.06}},
        xaxis: {{...axisStyle, rangeslider: {{visible: false}}, title: "UTC"}},
        yaxis: {{...axisStyle, type: "linear", tickformat: ",.6f", title: "Price", fixedrange: false}},
        shapes: shapes,
        hovermode: "x unified"
      }}, plotConfig);
    }}

    const tradeSelect = document.getElementById("tradeSelect");
    if (tradeSelect) {{
      tradeSelect.addEventListener("change", () => renderTradeChart(Number(tradeSelect.value)));
      document.getElementById("prevTrade").addEventListener("click", () => {{
        tradeSelect.value = String(Math.max(0, Number(tradeSelect.value) - 1));
        renderTradeChart(Number(tradeSelect.value));
      }});
      document.getElementById("nextTrade").addEventListener("click", () => {{
        const maxIdx = (payload.trade_charts || []).length - 1;
        tradeSelect.value = String(Math.min(maxIdx, Number(tradeSelect.value) + 1));
        renderTradeChart(Number(tradeSelect.value));
      }});
      document.getElementById("downloadTradeChart").addEventListener("click", () => {{
        Plotly.downloadImage("tradeChart", {{format: "png", filename: "trade_chart", width: 1440, height: 900}});
      }});
      renderTradeChart(Number(tradeSelect.value || 0));
    }}
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER, help="Path to trade_ledger.jsonl")
    parser.add_argument("--active", type=Path, default=DEFAULT_ACTIVE, help="Path to active_trades.json")
    parser.add_argument("--risk", type=Path, default=DEFAULT_RISK, help="Path to risk_state.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT, help="Output HTML path")
    parser.add_argument("--since", default="", help="Optional UTC ISO date/datetime filter, e.g. 2026-05-10")
    parser.add_argument("--base-equity", type=float, default=None, help="Optional manual equity baseline")
    parser.add_argument("--trade-charts", type=int, default=20, help="Fetch and embed OHLCV charts for the N most recent closed trades")
    parser.add_argument("--chart-cache-dir", type=Path, default=DEFAULT_CHART_CACHE, help="Directory for cached Bybit OHLCV snapshots")
    parser.add_argument("--no-chart-cache", action="store_true", help="Disable OHLCV cache reads/writes")
    parser.add_argument("--open", action="store_true", help="Open the generated report in your default browser")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    since = parse_dt(args.since) if args.since else None
    events = load_events(args.ledger, since)
    active = read_json(args.active, {})
    risk_state = read_json(args.risk, {})
    trades = build_closed_trades(events)
    base_equity, base_source = estimate_equity_base(trades, events, risk_state, args.base_equity)
    chart_cache_dir = None if args.no_chart_cache else args.chart_cache_dir
    trade_charts = build_trade_chart_data(
        trades,
        max_charts=max(args.trade_charts, 0),
        cache_dir=chart_cache_dir,
    )
    html = render_html(
        events=events,
        trades=trades,
        trade_charts=trade_charts,
        active=active,
        risk_state=risk_state,
        output=args.output,
        ledger=args.ledger,
        base_equity=base_equity,
        base_source=base_source,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"Wrote {args.output}")
    chart_errors = sum(1 for chart in trade_charts if chart.get("error"))
    print(
        f"Events: {len(events)} | Closed trades: {len(trades)} | "
        f"Trade charts: {len(trade_charts)} ({chart_errors} errors) | "
        f"Equity basis: {base_source}"
    )
    if args.open:
        webbrowser.open(args.output.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
