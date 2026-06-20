"""
fill_notifier.py — standalone TP/SL fill notification service

Connects to the Bybit private WebSocket and sends Telegram messages for
TP/SL/exit fills.  Reads active_trades.json (shared volume, written by
the main bot) to enrich messages with strategy and direction.

Env vars (all shared with the main bot via the same .env):
  BYBIT_API_KEY, BYBIT_API_SECRET
  BYBIT_PRIVATE_WS_DEMO             (default false)
  TELEGRAM_BOT_TOKEN
  TELEGRAM_ACCEPTED_SIGNALS_CHAT_ID
  TELEGRAM_NOTIFY_ENTRY_FILLS       (default false)
  ACTIVE_TRADES_STATE_PATH          (default /app/logs/active_trades.json)
  TAKER_FEE_RATE                    (default 0.00055)
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import time
from html import escape
from typing import Optional

import requests
from pybit.unified_trading import WebSocket

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)sZ [%(levelname)-5s] [fill_notifier] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("fill_notifier")

# ── Config ────────────────────────────────────────────────────────────────────
BYBIT_API_KEY    = os.environ["BYBIT_API_KEY"]
BYBIT_API_SECRET = os.environ["BYBIT_API_SECRET"]
PRIVATE_WS_DEMO  = os.environ.get("BYBIT_PRIVATE_WS_DEMO", "false").lower() in ("1", "true", "yes")
TG_TOKEN         = os.environ.get("TELEGRAM_BOT_TOKEN", "")
NOTIFY_ENTRIES   = os.environ.get("TELEGRAM_NOTIFY_ENTRY_FILLS", "false").lower() in ("1", "true", "yes")

# Parse "chatid_threadid" format (e.g. "-1003755718647_118")
def _parse_chat_target(raw: str) -> tuple[str, Optional[int]]:
    text = str(raw or "").strip()
    if not text:
        return "", None
    for sep in ("_", ":"):
        head, s, tail = text.rpartition(sep)
        if s and head and tail.isdigit():
            return head, int(tail)
    return text, None

_raw_fill_chat = os.environ.get("TELEGRAM_ACCEPTED_SIGNALS_CHAT_ID", "")
FILL_CHAT_ID, FILL_THREAD_ID = _parse_chat_target(_raw_fill_chat)
ACTIVE_TRADES_PATH = os.environ.get(
    "ACTIVE_TRADES_STATE_PATH",
    os.path.join(os.environ.get("LOG_DIR", "/app/logs"), "active_trades.json"),
)
TRADE_LEDGER_PATH = os.environ.get(
    "TRADE_LEDGER_PATH",
    os.path.join(os.environ.get("LOG_DIR", "/app/logs"), "trade_ledger.jsonl"),
)

# ── In-memory dedup set ────────────────────────────────────────────────────────
_notified: set[str] = set()
_ledger_context_mtime: float = -1.0
_ledger_context_by_key: dict[str, dict] = {}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fmt(value: object) -> str:
    if value is None or str(value).strip() in ("", "-"):
        return "-"
    try:
        f = float(str(value))
        if f == int(f) and abs(f) < 1e9:
            return str(int(f))
        return f"{f:g}"
    except (ValueError, TypeError):
        return str(value)


def _fmt_time_ms(value: object) -> str:
    try:
        ts = float(str(value)) / 1000.0
        return datetime.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return str(value or "-")


def _to_float(value: object) -> Optional[float]:
    try:
        return float(str(value))
    except Exception:
        return None


def _send_tg(lines: list[str], *, reply_to_message_id: Optional[int] = None) -> None:
    if not TG_TOKEN or not FILL_CHAT_ID:
        return
    payload: dict = {
        "chat_id": FILL_CHAT_ID,
        "text": "\n".join(lines),
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    if FILL_THREAD_ID is not None:
        payload["message_thread_id"] = FILL_THREAD_ID
    if reply_to_message_id is not None:
        payload["reply_to_message_id"] = reply_to_message_id
        payload["allow_sending_without_reply"] = True  # send normally if original was deleted
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json=payload,
            timeout=10,
        )
        if not resp.ok:
            log.warning(f"Telegram send failed: {resp.status_code} {resp.text[:200]}")
    except Exception as exc:
        log.warning(f"Telegram send error: {exc}")


def _read_active_trades() -> dict:
    try:
        with open(ACTIVE_TRADES_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


# ── Fill classification (mirrors bot.py logic) ─────────────────────────────────

def _strategy_from_order_link_id(value: object) -> dict:
    text = str(value or "").strip()
    if not text:
        return {}
    parts = text.split("-")
    if len(parts) < 5 or parts[0] not in {"E", "X"}:
        return {}
    strategy_by_code = {
        "MM": "million_moves",
        "ORB": "session_orb_judas_fvg",
        "WW": "wolfe_wave",
        "GG": "ggshot_227",
    }
    strategy = strategy_by_code.get(parts[1].upper())
    direction = {"L": "long", "S": "short"}.get(parts[-3].upper())
    out: dict = {"trade_id": text, "order_link_id": text}
    if strategy:
        out["strategy"] = strategy
    if direction:
        out["direction"] = direction
    return out


def _order_context_keys(order: dict) -> list[str]:
    keys: list[str] = []
    for key in ("orderId", "orderLinkId", "parentOrderId", "parentOrderLinkId"):
        value = str(order.get(key) or "").strip()
        if value and value not in keys:
            keys.append(value)
    return keys


def _remember_ledger_context(keys: list[object], context: dict) -> None:
    clean_context = {k: v for k, v in context.items() if v not in (None, "")}
    if not clean_context:
        return
    for key in keys:
        text = str(key or "").strip()
        if text:
            _ledger_context_by_key[text] = clean_context


def _refresh_ledger_context_cache() -> None:
    global _ledger_context_mtime, _ledger_context_by_key
    try:
        mtime = os.path.getmtime(TRADE_LEDGER_PATH)
    except OSError:
        return
    if mtime == _ledger_context_mtime:
        return
    previous = _ledger_context_by_key
    _ledger_context_by_key = {}
    try:
        with open(TRADE_LEDGER_PATH, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                event_type = event.get("type")
                if event_type == "signal":
                    context = {
                        "strategy": event.get("strategy"),
                        "direction": event.get("direction"),
                        "symbol": event.get("symbol"),
                        "trade_id": event.get("order_link_id") or event.get("order_id"),
                        "order_link_id": event.get("order_link_id"),
                        "tg_message_id": event.get("tg_message_id"),
                    }
                    _remember_ledger_context(
                        [event.get("order_id"), event.get("order_link_id")],
                        context,
                    )
                    continue
                if event_type != "fill" or not str(event.get("strategy") or "").strip():
                    continue
                raw = event.get("order_raw") if isinstance(event.get("order_raw"), dict) else {}
                context = {
                    "strategy": event.get("strategy"),
                    "direction": event.get("direction"),
                    "symbol": event.get("symbol"),
                    "trade_id": event.get("trade_id") or event.get("entry_order_link_id"),
                    "order_link_id": event.get("entry_order_link_id") or event.get("order_link_id"),
                }
                _remember_ledger_context(
                    [
                        event.get("order_id"),
                        event.get("order_link_id"),
                        event.get("entry_order_id"),
                        event.get("entry_order_link_id"),
                        event.get("trade_id"),
                        raw.get("orderId"),
                        raw.get("orderLinkId"),
                        raw.get("parentOrderId"),
                        raw.get("parentOrderLinkId"),
                    ],
                    context,
                )
        _ledger_context_mtime = mtime
    except Exception as exc:
        _ledger_context_by_key = previous
        log.warning(f"Could not read trade ledger context {TRADE_LEDGER_PATH}: {exc}")


def _lookup_ledger_context(order: dict) -> dict:
    _refresh_ledger_context_cache()
    for key in _order_context_keys(order):
        context = _ledger_context_by_key.get(key)
        if context:
            return dict(context)
    for key in _order_context_keys(order):
        context = _strategy_from_order_link_id(key)
        if context:
            return context
    return {}


def _clean_stop_order_type(value: object) -> str:
    text = str(value or "").strip()
    if text.upper() in {"", "UNKNOWN", "NONE", "NA", "N/A"}:
        return ""
    return text


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _classify(order: dict) -> tuple[Optional[str], bool]:
    """Return (event_label, is_exit). event_label is None if message should be suppressed."""
    stop_type = _clean_stop_order_type(order.get("stopOrderType"))
    descriptor = " ".join(
        str(order.get(key) or "") for key in ("stopOrderType", "createType", "orderLinkId")
    ).lower()
    is_exit = (
        bool(stop_type)
        or _truthy(order.get("reduceOnly"))
        or _truthy(order.get("closeOnTrigger"))
        or any(
            token in str(order.get("createType", "")).lower()
            for token in ("takeprofit", "stoploss", "trailing")
        )
    )
    if "takeprofit" in descriptor or "take_profit" in descriptor:
        return "TAKE PROFIT FILLED", True
    if "stoploss" in descriptor or "stop_loss" in descriptor:
        return "STOP LOSS FILLED", True
    if "trailing" in descriptor:
        return "TRAILING STOP FILLED", True
    if is_exit:
        return "POSITION EXIT FILLED", True
    if NOTIFY_ENTRIES:
        return "ENTRY FILLED", False
    return None, False


def _fill_price(order: dict) -> object:
    for key in ("avgPrice", "execPrice", "price", "triggerPrice"):
        v = order.get(key)
        if v not in (None, "", "0", 0):
            return v
    return "-"


def _fill_qty(order: dict) -> object:
    for key in ("cumExecQty", "execQty", "qty", "orderQty"):
        v = order.get(key)
        if v not in (None, "", "0", 0):
            return v
    return "-"


# ── WebSocket callback ─────────────────────────────────────────────────────────

def _on_order(msg: dict) -> None:
    global _notified
    try:
        data = msg.get("data", [])
        items: list[dict] = [data] if isinstance(data, dict) else (
            [i for i in data if isinstance(i, dict)] if isinstance(data, list) else []
        )
        for order in items:
            if order.get("category") not in (None, "", "linear"):
                continue
            if str(order.get("orderStatus", "")).lower() != "filled":
                continue

            symbol   = str(order.get("symbol", ""))
            order_id = str(order.get("orderId") or "")
            dedupe   = order_id or (
                f"{symbol}:{order.get('updatedTime')}:{order.get('cumExecQty')}:"
                f"{order.get('avgPrice')}:{order.get('stopOrderType')}"
            )
            if dedupe in _notified:
                continue

            event, is_exit = _classify(order)
            if event is None:
                continue

            _notified.add(dedupe)
            if len(_notified) > 1000:
                _notified = set(list(_notified)[-500:])

            # active_trades.json can be cleared before this separate process sees
            # the TP/SL fill, so prefer order-specific context from the ledger.
            trade          = _read_active_trades().get(symbol, {})
            ledger_trade   = _lookup_ledger_context(order)
            trade          = {**trade, **ledger_trade}
            strategy       = trade.get("strategy")
            direction      = trade.get("direction")
            tg_message_id  = trade.get("tg_message_id")  # reply-to the original signal message
            side           = str(order.get("side", ""))
            if not direction and is_exit:
                direction = "long" if side == "Sell" else "short" if side == "Buy" else None

            stop_type  = _clean_stop_order_type(order.get("stopOrderType"))
            price      = _fill_price(order)
            qty        = _fill_qty(order)

            # ── Header line: emoji + event label ──────────────────────────────
            EVENT_EMOJI = {
                "TAKE PROFIT FILLED":    "✅",
                "STOP LOSS FILLED":      "❌",
                "TRAILING STOP FILLED":  "🔶",
                "POSITION EXIT FILLED":  "🔷",
                "ENTRY FILLED":          "🟢" if (direction or "").lower() == "long" else "🔴",
            }
            emoji = EVENT_EMOJI.get(event, "🔔")

            DIR_EMOJI = {"long": "📈", "short": "📉"}
            dir_str = str(direction or "").lower()
            dir_label = f"{DIR_EMOJI.get(dir_str, '')} {dir_str.upper()}".strip() if dir_str else None

            strategy_tag = f"[{escape(strategy.upper())}] " if strategy else ""
            lines: list[str] = [
                f"{emoji} <b>{strategy_tag}{escape(event)}</b>",
                f"<b>{escape(symbol)}</b>" + (f"  ·  {escape(dir_label)}" if dir_label else ""),
            ]

            # ── Price / qty on one line ────────────────────────────────────────
            lines.append(f"💰 <code>{_fmt(price)}</code>  ×  <code>{_fmt(qty)}</code>")

            # ── PnL prominently if present ─────────────────────────────────────
            closed_pnl = order.get("closedPnl")
            if closed_pnl not in (None, "", "0", 0):
                pnl_f = _to_float(closed_pnl)
                pnl_emoji = "📗" if (pnl_f or 0) >= 0 else "📕"
                lines.append(f"{pnl_emoji} PnL: <code>{_fmt(closed_pnl)}</code>")

            # ── Secondary detail ──────────────────────────────────────────────
            updated_time = order.get("updatedTime")
            if updated_time:
                lines.append(f"🕐 <code>{_fmt_time_ms(updated_time)}</code>")
            if order_id:
                lines.append(f"🆔 <code>{escape(order_id)}</code>")

            log.info(
                f"[{symbol}] {event}: side={side or '-'} price={_fmt(price)} "
                f"qty={_fmt(qty)} stop_type={stop_type or '-'}"
                + (f" reply_to={tg_message_id}" if tg_message_id else "")
            )
            _send_tg(lines, reply_to_message_id=tg_message_id)

    except Exception:
        log.exception("Error in order callback")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    log.info(
        f"starting  demo={PRIVATE_WS_DEMO}  "
        f"notify_entries={NOTIFY_ENTRIES}  active_trades={ACTIVE_TRADES_PATH}"
    )
    if not TG_TOKEN or not FILL_CHAT_ID:
        log.warning("TELEGRAM_BOT_TOKEN or TELEGRAM_ACCEPTED_SIGNALS_CHAT_ID not set — messages will be dropped")

    while True:
        try:
            ws = WebSocket(
                testnet=False,
                demo=PRIVATE_WS_DEMO,
                channel_type="private",
                api_key=BYBIT_API_KEY,
                api_secret=BYBIT_API_SECRET,
            )
            ws.order_stream(_on_order)
            log.info("Private WS connected — listening for fills")
            while True:
                time.sleep(30)
        except Exception as exc:
            log.error(f"WebSocket error: {exc}; reconnecting in 10s")
            time.sleep(10)


if __name__ == "__main__":
    main()
