#!/usr/bin/env python3
"""Capitulation Spring strategy sidecar.

Standalone strategy bot (modelled on weekday_edge_bot.py). Listens to Bybit 5m
klines for a basket of liquid perps, detects the validated "Capitulation Spring"
long setup (deep-sweep Wyckoff spring + early-week + negative-funding flush), and:
  * forwards every ACCEPTED signal to the RL execution sidecar (rl_signal_v1), and
  * optionally executes it on a separate Bybit Demo account with a scaled exit
    (25% @ 4R / 50% @ 12R / 25% @ 30R, stop -> breakeven after the first partial).

Execution is OFF by default (signal + RL-forward only) until
CAPITULATION_TRADING_ENABLED=true. Demo by default.
"""

from __future__ import annotations

import json
import logging
import math
import os
import queue
import signal as signal_module
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import requests
from pybit.unified_trading import HTTP, WebSocket

from capitulation_spring import SpringConfig, detect_spring


def _csv(name: str, default: str) -> list[str]:
    return [s.strip().upper() for s in os.environ.get(name, default).split(",") if s.strip()]


SYMBOLS = _csv(
    "CAPITULATION_SYMBOLS",
    "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,AVAXUSDT,LINKUSDT,"
    "DOTUSDT,LTCUSDT,ATOMUSDT,NEARUSDT,FILUSDT,ARBUSDT,OPUSDT,INJUSDT,SUIUSDT",
)
CATEGORY = "linear"
STRATEGY_NAME = os.environ.get("CAPITULATION_STRATEGY_NAME", "capitulation_spring").strip() or "capitulation_spring"
INTERVAL_MIN = 5

PUBLIC_WS_DEMO = os.environ.get("CAPITULATION_PUBLIC_WS_DEMO", os.environ.get("BYBIT_PUBLIC_WS_DEMO", "false")).lower() in {"1", "true", "yes"}
TRADING_DEMO = os.environ.get("CAPITULATION_BYBIT_DEMO", "true").lower() in {"1", "true", "yes"}
# Execute on demo by default (nothing to lose on demo; exercises the real fill/exit path).
TRADING_ENABLED = os.environ.get("CAPITULATION_TRADING_ENABLED", "true").lower() in {"1", "true", "yes"}
# Guard against accidentally going LIVE: real-money trading needs an explicit confirm.
LIVE_CONFIRM = os.environ.get("CAPITULATION_LIVE_CONFIRM", "false").lower() in {"1", "true", "yes"}
# Hedge mode (BothSide): longs use positionIdx=1. The bot auto-detects at startup and
# falls back to this hint. Set false only for one-way (MergedSingle, positionIdx=0) accounts.
HEDGE_MODE = os.environ.get("CAPITULATION_HEDGE_MODE", "true").lower() in {"1", "true", "yes"}
API_KEY = os.environ.get("CAPITULATION_BYBIT_API_KEY", "").strip()
API_SECRET = os.environ.get("CAPITULATION_BYBIT_API_SECRET", "").strip()

TELEGRAM_BOT_TOKEN = os.environ.get("CAPITULATION_TELEGRAM_BOT_TOKEN", os.environ.get("TELEGRAM_BOT_TOKEN", "")).strip()
TELEGRAM_CHAT_ID = os.environ.get("CAPITULATION_TELEGRAM_CHAT_ID", os.environ.get("TELEGRAM_ACCEPTED_SIGNALS_CHAT_ID", "")).strip()

LOG_DIR = Path(os.environ.get("CAPITULATION_LOG_DIR", os.environ.get("LOG_DIR", "/app/logs")))
LEDGER_PATH = Path(os.environ.get("CAPITULATION_LEDGER_PATH", str(LOG_DIR / "capitulation_spring_signals.jsonl")))

# Strategy params (defaults = the validated 5m spec).
SWEEP_LOOKBACK = int(os.environ.get("CAPITULATION_SWEEP_LOOKBACK", "4320"))   # 15d of 5m
WICK_FRAC = float(os.environ.get("CAPITULATION_WICK_FRAC", "0.5"))
CLOSE_POS = float(os.environ.get("CAPITULATION_CLOSE_POS", "0.5"))
WEEK_FRAC_MAX = float(os.environ.get("CAPITULATION_WEEK_FRAC_MAX", "0.40"))
FUNDING_Z_THR = float(os.environ.get("CAPITULATION_FUNDING_Z_THR", "-1.0"))
ATR_LEN = int(os.environ.get("CAPITULATION_ATR_LEN", "14"))
STOP_BUFFER_ATR = float(os.environ.get("CAPITULATION_STOP_BUFFER_ATR", "0.05"))
TP_R = tuple(float(x) for x in _csv("CAPITULATION_TP_R", "4,12,30"))
TP_QTY_PCT = tuple(float(x) for x in _csv("CAPITULATION_TP_QTY_PCT", "25,50,25"))

# Risk / portfolio.
RISK_PCT = float(os.environ.get("CAPITULATION_RISK_PCT", "0.005"))            # 0.5% equity; 0 -> use RISK_USDT
RISK_USDT = float(os.environ.get("CAPITULATION_RISK_USDT", "50"))
MAX_CONCURRENT = int(os.environ.get("CAPITULATION_MAX_CONCURRENT", "4"))
TAKER_FEE_RATE = float(os.environ.get("CAPITULATION_TAKER_FEE_RATE", os.environ.get("TAKER_FEE_RATE", "0.00055")))
MIN_STOP_DISTANCE_PCT = float(os.environ.get("CAPITULATION_MIN_STOP_DISTANCE_PCT", "0.0005"))

# Funding cache + housekeeping.
FUNDING_TTL_SECONDS = int(os.environ.get("CAPITULATION_FUNDING_TTL_SECONDS", "1800"))
FUNDING_Z_WINDOW = int(os.environ.get("CAPITULATION_FUNDING_Z_WINDOW", "30"))
WARMUP_BARS = int(os.environ.get("CAPITULATION_WARMUP_BARS", str(SWEEP_LOOKBACK + ATR_LEN + 120)))
BE_POLL_SECONDS = int(os.environ.get("CAPITULATION_BE_POLL_SECONDS", "10"))
WS_STALE_SECONDS = int(os.environ.get("CAPITULATION_WS_STALE_SECONDS", os.environ.get("WS_STALE_SECONDS", "900")))
HEARTBEAT_SECONDS = int(os.environ.get("CAPITULATION_HEARTBEAT_SECONDS", "3600"))

RL_EXECUTION_URL = os.environ.get("CAPITULATION_RL_EXECUTION_URL", os.environ.get("RL_EXECUTION_URL", "")).strip()
RL_EXECUTION_TIMEOUT_SECONDS = float(os.environ.get("CAPITULATION_RL_EXECUTION_TIMEOUT_SECONDS", os.environ.get("RL_EXECUTION_TIMEOUT_SECONDS", "1.0")))
RL_EXECUTION_QUEUE_SIZE = int(os.environ.get("CAPITULATION_RL_EXECUTION_QUEUE_SIZE", "1000"))
RL_FORWARD_REJECTED = os.environ.get("CAPITULATION_RL_FORWARD_REJECTED", "false").lower() in {"1", "true", "yes"}

LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)sZ [%(levelname)-5s] [capitulation] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_DIR / "capitulation_spring_bot.log")],
)
log = logging.getLogger("capitulation")

SPRING_CFG = SpringConfig(
    sweep_lookback=SWEEP_LOOKBACK, wick_frac=WICK_FRAC, close_pos=CLOSE_POS,
    week_frac_max=WEEK_FRAC_MAX, funding_z_thr=FUNDING_Z_THR, atr_len=ATR_LEN,
    stop_buffer_atr=STOP_BUFFER_ATR, tp_r=TP_R, tp_qty_pct=TP_QTY_PCT,
)


# --------------------------------------------------------------------------- helpers
def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def round_to_step(value: float, step: float) -> float:
    if step <= 0:
        return float(value)
    precision = max(0, int(round(-math.log10(step))))
    return round(round(float(value) / step) * step, precision)


def floor_to_step(value: float, step: float) -> float:
    if step <= 0:
        return float(value)
    precision = max(0, int(round(-math.log10(step))))
    return round(math.floor(float(value) / step) * step, precision)


def qty_to_str(value: float, step: float = 0.0) -> str:
    if step > 0:
        precision = max(0, int(round(-math.log10(step))))
        return f"{value:.{precision}f}"
    return f"{round(value, 8):.8f}".rstrip("0").rstrip(".") or "0"


def fmt(value: float, digits: int = 2) -> str:
    return "-" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def fetch_5m_klines(http: HTTP, symbol: str, limit: int) -> list[dict]:
    """Fetch the last `limit` closed 5m bars (chronological list of dict)."""
    rows: dict[int, list[Any]] = {}
    end_ms: int | None = None
    while len(rows) < limit:
        kwargs: dict[str, Any] = {"category": CATEGORY, "symbol": symbol, "interval": str(INTERVAL_MIN),
                                  "limit": min(1000, max(1, limit - len(rows)))}
        if end_ms is not None:
            kwargs["end"] = end_ms
        resp = http.get_kline(**kwargs)
        batch = resp.get("result", {}).get("list", [])
        if not batch:
            break
        for item in batch:
            rows[int(item[0])] = item
        end_ms = min(int(item[0]) for item in batch) - 1
        if len(batch) < kwargs["limit"]:
            break
        time.sleep(0.05)
    bars = []
    for ts in sorted(rows):
        r = rows[ts]
        bars.append({"ts": int(ts), "open": float(r[1]), "high": float(r[2]),
                     "low": float(r[3]), "close": float(r[4]), "volume": float(r[5])})
    return bars


# --------------------------------------------------------------------------- telegram / RL
class TelegramNotifier:
    def __init__(self, token: str, chat_id: str) -> None:
        self.token = token
        head, _, tail = chat_id.rpartition("_")
        if tail.isdigit() and head:
            self.chat_id, self.thread_id = head, int(tail)
        else:
            self.chat_id, self.thread_id = chat_id, None
        self.enabled = bool(self.token and self.chat_id)

    def send(self, lines: list[str]) -> None:
        if not self.enabled:
            return
        payload: dict[str, Any] = {"chat_id": self.chat_id, "text": "\n".join(lines),
                                   "parse_mode": "HTML", "disable_web_page_preview": True}
        if self.thread_id is not None:
            payload["message_thread_id"] = self.thread_id
        try:
            requests.post(f"https://api.telegram.org/bot{self.token}/sendMessage", json=payload, timeout=10)
        except Exception as exc:
            log.warning("Telegram send failed: %s", str(exc).replace(self.token, "<redacted>"))


class RlSidecarClient:
    def __init__(self, url: str, timeout_seconds: float, queue_size: int) -> None:
        self.url = url
        self.timeout = max(0.1, timeout_seconds)
        self.queue: queue.Queue[dict[str, Any]] | None = None
        if self.url:
            self.queue = queue.Queue(maxsize=max(1, queue_size))
            threading.Thread(target=self._worker, name="capitulation-rl-dispatch", daemon=True).start()
            log.info("RL execution sidecar enabled url=%s", self.url)
        else:
            log.info("RL execution sidecar disabled; RL_EXECUTION_URL is empty")

    def enqueue(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.url or self.queue is None:
            return {"enabled": False, "queued": False}
        try:
            self.queue.put_nowait(jsonable(payload))
            return {"enabled": True, "queued": True}
        except queue.Full:
            log.warning("[rl] dispatch queue full; dropping %s %s", payload.get("status"), payload.get("symbol"))
            return {"enabled": True, "queued": False}

    def _worker(self) -> None:
        assert self.queue is not None
        while True:
            payload = self.queue.get()
            try:
                resp = requests.post(self.url, json=payload, timeout=self.timeout)
                if resp.status_code >= 300:
                    log.warning("[rl] sidecar HTTP %s: %s", resp.status_code, resp.text[:200])
                else:
                    log.info("[rl] dispatched %s %s", payload.get("status"), payload.get("symbol"))
            except Exception as exc:
                log.warning("[rl] dispatch failed: %s", exc)
            finally:
                self.queue.task_done()


# --------------------------------------------------------------------------- funding
class FundingCache:
    def __init__(self, http: HTTP) -> None:
        self.http = http
        self._cache: dict[str, tuple[float, float | None]] = {}   # symbol -> (fetched_at, z)
        self._lock = threading.Lock()

    def funding_z(self, symbol: str) -> float | None:
        now = time.time()
        with self._lock:
            cached = self._cache.get(symbol)
            if cached and now - cached[0] < FUNDING_TTL_SECONDS:
                return cached[1]
        z = self._compute(symbol)
        with self._lock:
            self._cache[symbol] = (now, z)
        return z

    def _compute(self, symbol: str) -> float | None:
        try:
            resp = self.http.get_funding_rate_history(category=CATEGORY, symbol=symbol, limit=50)
            rows = resp.get("result", {}).get("list", [])
            vals = [finite_float(r.get("fundingRate")) for r in rows]
            vals = [v for v in vals if v is not None]
            if len(vals) < 8:
                return None
            # Bybit returns newest first; reverse to chronological, current = newest.
            series = list(reversed(vals))
            current = series[-1]
            window = series[-(FUNDING_Z_WINDOW + 1):-1] if len(series) > FUNDING_Z_WINDOW else series[:-1]
            arr = np.asarray(window, dtype=float)
            std = float(np.std(arr))
            if arr.size < 5 or std <= 0:
                return None
            z = (current - float(np.mean(arr))) / std
            return float(z) if math.isfinite(z) else None
        except Exception as exc:
            log.debug("[%s] funding z fetch failed: %s", symbol, exc)
            return None


# --------------------------------------------------------------------------- executor
class ScaledExecutor:
    """Demo scaled-exit executor: one BRACKETED market leg per TP share (each carries its
    own [stopLoss, takeProfit_i] as a native Partial bracket - no reduce-only limits), and
    a poller that moves the stop to breakeven once the first partial fills. Enforces a
    global concurrency cap and one position per symbol (no pyramiding). No-op unless
    TRADING_ENABLED."""

    def __init__(self, telegram: "TelegramNotifier | None" = None) -> None:
        self.http: HTTP | None = None
        self.telegram = telegram
        self.enabled = False
        self.long_idx = 1 if HEDGE_MODE else 0   # positionIdx for the long side
        self.info_cache: dict[str, dict[str, Any]] = {}
        self.open_trades: dict[str, dict[str, Any]] = {}     # symbol -> trade dict
        self.lock = threading.Lock()
        if not TRADING_ENABLED:
            log.info("Execution disabled (CAPITULATION_TRADING_ENABLED=false); signal + RL-forward only")
            return
        if not TRADING_DEMO and not LIVE_CONFIRM:
            log.critical("CAPITULATION_BYBIT_DEMO=false requires CAPITULATION_LIVE_CONFIRM=true; "
                         "refusing to trade LIVE -> running signal-only")
            return
        if not API_KEY or not API_SECRET:
            log.warning("Execution requested but CAPITULATION_BYBIT_API_KEY/SECRET missing; "
                        "running signal-only (signals still forwarded to RL)")
            return
        self.http = HTTP(testnet=False, demo=TRADING_DEMO, api_key=API_KEY, api_secret=API_SECRET)
        self.long_idx = self._detect_long_idx()
        self.enabled = True
        threading.Thread(target=self._be_poller, name="capitulation-be-poller", daemon=True).start()
        log.info("Execution ENABLED on Bybit %s account (%s mode, long positionIdx=%d)",
                 "DEMO" if TRADING_DEMO else "LIVE",
                 "hedge" if self.long_idx == 1 else "one-way", self.long_idx)

    def _detect_long_idx(self) -> int:
        """Detect account position mode from get_positions; long side is idx 1 (hedge)
        or 0 (one-way). Falls back to the HEDGE_MODE hint."""
        idx = 1 if HEDGE_MODE else 0
        try:
            rows = self.http.get_positions(category=CATEGORY, symbol=SYMBOLS[0]).get("result", {}).get("list", [])
            pidxs = {int(r.get("positionIdx", 0)) for r in rows}
            if pidxs == {0}:
                idx = 0
            elif 1 in pidxs or 2 in pidxs:
                idx = 1
        except Exception as exc:
            log.warning("position-mode detect failed (%s); using HEDGE_MODE=%s", exc, HEDGE_MODE)
        return idx

    def _info(self, symbol: str) -> dict[str, Any]:
        if symbol not in self.info_cache:
            item = self.http.get_instruments_info(category=CATEGORY, symbol=symbol)["result"]["list"][0]
            lot, price, lev = item.get("lotSizeFilter", {}), item.get("priceFilter", {}), item.get("leverageFilter", {})
            self.info_cache[symbol] = {
                "status": str(item.get("status", "")),
                "qty_step": float(lot.get("qtyStep", "0.001")), "min_qty": float(lot.get("minOrderQty", "0.001")),
                "tick_size": float(price.get("tickSize", "0.01")), "max_leverage": float(lev.get("maxLeverage", "1") or "1"),
            }
        return self.info_cache[symbol]

    def _equity(self) -> float:
        row = self.http.get_wallet_balance(accountType="UNIFIED")["result"]["list"][0]
        for coin in row.get("coin", []):
            if coin.get("coin") == "USDT":
                return float(coin.get("equity", 0) or 0)
        return float(row.get("totalEquity", 0) or 0)

    def open_count(self) -> int:
        with self.lock:
            return len(self.open_trades)

    def has_symbol(self, symbol: str) -> bool:
        with self.lock:
            return symbol in self.open_trades

    def execute(self, symbol: str, setup: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled or self.http is None:
            return {"ok": False, "skipped": True, "message": "trading disabled (signal-only)"}
        with self.lock:
            if symbol in self.open_trades:
                return {"ok": False, "skipped": True, "message": "position already open for symbol"}
            if len(self.open_trades) >= MAX_CONCURRENT:
                return {"ok": False, "skipped": True, "message": f"concurrency cap {MAX_CONCURRENT} reached"}
        info = self._info(symbol)
        if info.get("status") and info["status"] != "Trading":
            return {"ok": False, "message": f"{symbol} status={info['status']}"}
        entry, stop = float(setup["entry"]), float(setup["stop"])
        unit_risk = entry - stop
        if unit_risk <= 0 or unit_risk / entry < MIN_STOP_DISTANCE_PCT:
            return {"ok": False, "message": "stop distance too small"}
        q_step, min_qty, tick = info["qty_step"], info["min_qty"], info["tick_size"]
        max_lev = max(info["max_leverage"], 1.0)
        equity = self._equity()
        if equity <= 0:
            return {"ok": False, "message": "invalid equity"}
        risk_usdt = equity * RISK_PCT if RISK_PCT > 0 else RISK_USDT
        fee_per_unit = TAKER_FEE_RATE * (entry + stop)
        qty = floor_to_step(risk_usdt / (unit_risk + fee_per_unit), q_step)
        if qty < min_qty:
            return {"ok": False, "message": f"risk too small for min qty ({qty} < {min_qty})"}

        self._ensure_leverage(symbol, max_lev)
        sl_price = round_to_step(stop, tick)
        link = self._link_id(symbol)
        targets, pcts, rs = setup["targets"], setup["tp_qty_pct"], setup["tp_r"]

        # One BRACKETED market leg per TP share: each leg carries [stopLoss=SL,
        # takeProfit=TP_i] as a native Partial bracket (market triggers, NOT reduce-only
        # limits). Hedge mode aggregates them into one positionIdx=1 long; partial SLs at
        # the same price sum to a full-position stop. We never pyramid (one position/symbol).
        legs: list[dict[str, Any]] = []
        placed_qty = 0.0
        remaining = qty
        n = len(targets)
        for i, (price, pct) in enumerate(zip(targets, pcts)):
            leg_qty = remaining if i == n - 1 else floor_to_step(qty * pct / 100.0, q_step)
            leg_qty = min(leg_qty, remaining)
            if leg_qty < min_qty:
                if i < n - 1:
                    continue                       # roll a sub-min share into the final leg
                if leg_qty <= 0:
                    break
            tp_price = round_to_step(price, tick)
            try:
                resp = self.http.place_order(
                    category=CATEGORY, symbol=symbol, side="Buy", orderType="Market",
                    qty=qty_to_str(leg_qty, q_step), takeProfit=str(tp_price), stopLoss=str(sl_price),
                    tpslMode="Partial", tpOrderType="Market", slOrderType="Market",
                    tpTriggerBy="LastPrice", slTriggerBy="LastPrice",
                    positionIdx=self.long_idx, orderLinkId=f"{link}-L{i+1}")
            except Exception as exc:
                log.warning("[%s] bracket leg %d exception: %s", symbol, i + 1, exc)
                continue
            if int(resp.get("retCode", -1)) != 0:
                log.warning("[%s] bracket leg %d rejected: %s %s", symbol, i + 1,
                            resp.get("retCode"), resp.get("retMsg"))
                continue
            placed_qty += leg_qty
            remaining = round(remaining - leg_qty, 10)
            legs.append({"r": rs[i], "tp": tp_price, "qty": leg_qty,
                         "order_id": str(resp.get("result", {}).get("orderId") or "")})

        if placed_qty <= 0 or not legs:
            return {"ok": False, "message": "no bracket legs placed (all rejected)"}
        order_id = legs[0]["order_id"]
        trade = {"symbol": symbol, "entry": entry, "stop": stop, "qty": placed_qty, "order_id": order_id,
                 "order_link_id": link, "be_done": False, "legs": legs, "opened_at": time.time()}
        with self.lock:
            self.open_trades[symbol] = trade
        return {"ok": True, "order_id": order_id, "order_link_id": link, "qty": qty_to_str(placed_qty, q_step),
                "qty_float": placed_qty, "notional": placed_qty * entry, "legs": legs,
                "expected_total_risk": placed_qty * (unit_risk + fee_per_unit)}

    def _ensure_leverage(self, symbol: str, max_lev: float) -> None:
        try:
            self.http.set_leverage(category=CATEGORY, symbol=symbol,
                                   buyLeverage=qty_to_str(max_lev), sellLeverage=qty_to_str(max_lev))
        except Exception as exc:
            if "not modified" not in str(exc).lower():
                log.debug("[%s] set_leverage: %s", symbol, exc)

    @staticmethod
    def _link_id(symbol: str) -> str:
        ts = datetime.now(timezone.utc).strftime("%m%d%H%M")
        return f"CS-{symbol.replace('USDT', '')[:7]}-{ts}-{uuid.uuid4().hex[:5].upper()}"[:36]

    def _tg(self, lines: list[str]) -> None:
        if self.telegram is not None:
            self.telegram.send(lines)

    def _realized_since(self, symbol: str, since_ts: float) -> float | None:
        """Sum closed-PnL rows for this symbol's long side since the trade opened."""
        try:
            rows = self.http.get_closed_pnl(category=CATEGORY, symbol=symbol, limit=50).get("result", {}).get("list", [])
        except Exception:
            return None
        total, n = 0.0, 0
        for r in rows:
            try:
                ut = int(r.get("updatedTime", 0) or 0) / 1000.0
            except (TypeError, ValueError):
                ut = 0.0
            if ut >= since_ts - 5 and str(r.get("side", "")).lower() in ("sell", ""):  # long closed by a Sell
                total += float(r.get("closedPnl", 0) or 0)
                n += 1
        return total if n else None

    def _be_poller(self) -> None:
        """Move stop to breakeven once the first partial fills; clean up closed trades."""
        while True:
            time.sleep(max(2, BE_POLL_SECONDS))
            try:
                with self.lock:
                    items = list(self.open_trades.items())
                for symbol, trade in items:
                    try:
                        size = self._position_size(symbol)
                    except Exception:
                        continue
                    if size <= 0:
                        with self.lock:
                            self.open_trades.pop(symbol, None)
                        pnl = self._realized_since(symbol, trade.get("opened_at", 0.0))
                        if pnl is None:
                            result = "in profit (TP)" if trade.get("be_done") else "stopped out (SL)"
                            pnl_line = ""
                        else:
                            result = "WIN" if pnl > 0.01 else ("LOSS" if pnl < -0.01 else "breakeven")
                            pnl_line = f"  realized <code>{fmt(pnl, 2)} USDT</code>"
                        log.info("[%s] position closed (%s)%s", symbol, result,
                                 f" pnl={fmt(pnl, 2)}" if pnl is not None else "")
                        self._tg([f"<b>[CAPITULATION] exit</b> {escape(symbol)} — {result}{pnl_line}",
                                  f"entry <code>{fmt(trade.get('entry'), 6)}</code>  "
                                  f"held <code>{int((time.time() - trade.get('opened_at', time.time())) / 60)}m</code>"])
                        continue
                    if not trade["be_done"] and size < trade["qty"] - 1e-12:
                        try:
                            info = self._info(symbol)
                            self.http.set_trading_stop(
                                category=CATEGORY, symbol=symbol,
                                stopLoss=str(round_to_step(trade["entry"], info["tick_size"])),
                                slTriggerBy="LastPrice", tpslMode="Partial",
                                slSize=qty_to_str(size, info["qty_step"]), positionIdx=self.long_idx)
                            trade["be_done"] = True
                            log.info("[%s] first partial filled -> stop moved to breakeven (rem %s)",
                                     symbol, qty_to_str(size, info["qty_step"]))
                            self._tg([f"<b>[CAPITULATION] TP fill</b> {escape(symbol)} — first partial filled",
                                      f"stop → breakeven, remaining <code>{qty_to_str(size, info['qty_step'])}</code>"])
                        except Exception as exc:
                            log.warning("[%s] BE move failed: %s", symbol, exc)
            except Exception:
                log.exception("BE poller error")

    def _position_size(self, symbol: str) -> float:
        """Size of the LONG side only (hedge mode returns both idx 1 and idx 2 rows)."""
        resp = self.http.get_positions(category=CATEGORY, symbol=symbol)
        rows = resp.get("result", {}).get("list", [])
        for row in rows:
            if int(row.get("positionIdx", 0)) == self.long_idx and str(row.get("side", "")).lower() in ("buy", ""):
                return abs(float(row.get("size", 0) or 0))
        # Fallback: first Buy-side row.
        for row in rows:
            if str(row.get("side", "")).lower() == "buy":
                return abs(float(row.get("size", 0) or 0))
        return 0.0


# --------------------------------------------------------------------------- bot
class CapitulationSpringBot:
    def __init__(self) -> None:
        self.public_http = HTTP(testnet=False, demo=False)
        self.bars: dict[str, deque] = {s: deque(maxlen=WARMUP_BARS) for s in SYMBOLS}
        self.bar_locks: dict[str, threading.Lock] = {s: threading.Lock() for s in SYMBOLS}
        self.funding = FundingCache(self.public_http)
        self.telegram = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        self.executor = ScaledExecutor(telegram=self.telegram)
        self.rl = RlSidecarClient(RL_EXECUTION_URL, RL_EXECUTION_TIMEOUT_SECONDS, RL_EXECUTION_QUEUE_SIZE)
        self.stop_event = threading.Event()
        self.ws: WebSocket | None = None
        self.last_ws_ts = time.time()
        self.last_heartbeat = 0.0
        self.processed: set[str] = set()
        self.ledger_lock = threading.Lock()

    # ---- lifecycle
    def start(self) -> None:
        self._warmup()
        self._send_startup()
        self._open_ws()
        while not self.stop_event.is_set():
            now = time.time()
            if HEARTBEAT_SECONDS > 0 and now - self.last_heartbeat >= HEARTBEAT_SECONDS:
                log.info("Heartbeat: %d symbols, open=%d/%d, exec=%s",
                         len(SYMBOLS), self.executor.open_count(), MAX_CONCURRENT, TRADING_ENABLED)
                self.last_heartbeat = now
            if WS_STALE_SECONDS > 0 and now - self.last_ws_ts > WS_STALE_SECONDS:
                raise SystemExit(f"No Bybit public WS message for {WS_STALE_SECONDS}s")
            time.sleep(1)

    def _warmup(self) -> None:
        log.info("Warming up %d symbols x %d 5m bars ...", len(SYMBOLS), WARMUP_BARS)
        for sym in SYMBOLS:
            try:
                bars = fetch_5m_klines(self.public_http, sym, WARMUP_BARS)
                with self.bar_locks[sym]:
                    self.bars[sym].extend(bars)
                log.info("  %s: %d bars", sym, len(bars))
            except Exception as exc:
                log.warning("  %s warmup failed: %s", sym, exc)

    def _open_ws(self) -> None:
        log.info("Opening 5m kline WS for %d symbols (demo=%s)", len(SYMBOLS), PUBLIC_WS_DEMO)
        # ping_timeout=None: public Bybit sockets miss control pongs while the data
        # stream stays healthy; the WS_STALE_SECONDS watchdog handles real stalls.
        # Without this the connection flaps with "ping/pong timed out". (Matches mm-bot.)
        self.ws = WebSocket(testnet=False, channel_type="linear", demo=PUBLIC_WS_DEMO, ping_timeout=None)
        for sym in SYMBOLS:
            self.ws.kline_stream(interval=INTERVAL_MIN, symbol=sym, callback=self._on_kline)

    def _send_startup(self) -> None:
        if self.executor.enabled:
            mode = f"EXECUTION ON ({'demo' if TRADING_DEMO else 'LIVE'})"
        else:
            mode = "signal + RL forward only"
        self.telegram.send([
            "<b>[CAPITULATION SPRING] started</b>",
            f"Symbols: <code>{len(SYMBOLS)}</code>  Mode: <code>{escape(mode)}</code>",
            f"RL sidecar: <code>{'on' if RL_EXECUTION_URL else 'off'}</code>",
            f"Filter: sweep {SWEEP_LOOKBACK} bars, week&lt;{WEEK_FRAC_MAX}, funding_z&le;{FUNDING_Z_THR}",
            f"Exit: {'/'.join(str(int(r)) for r in TP_R)}R @ {'/'.join(str(int(p)) for p in TP_QTY_PCT)}% , stop-&gt;BE",
            f"Risk: <code>{RISK_PCT*100:.2f}% equity</code>  Max concurrent: <code>{MAX_CONCURRENT}</code>",
        ])

    # ---- data
    def _on_kline(self, msg: dict[str, Any]) -> None:
        self.last_ws_ts = time.time()
        try:
            data = msg.get("data") or []
            topic = str(msg.get("topic", ""))
            sym = topic.split(".")[-1] if "." in topic else None
            if not data or sym not in self.bars:
                return
            candle = data[0]
            if not candle.get("confirm", False):
                return
            bar = {"ts": int(candle["start"]), "open": float(candle["open"]), "high": float(candle["high"]),
                   "low": float(candle["low"]), "close": float(candle["close"]), "volume": float(candle.get("volume", 0.0))}
            with self.bar_locks[sym]:
                buf = self.bars[sym]
                if buf and buf[-1]["ts"] == bar["ts"]:
                    buf[-1] = bar
                else:
                    buf.append(bar)
            key = f"{sym}:{bar['ts']}"
            if key in self.processed:
                return
            self.processed.add(key)
            if len(self.processed) > 4000:
                self.processed = set(sorted(self.processed)[-2000:])
            threading.Thread(target=self._evaluate, args=(sym,), daemon=True).start()
        except Exception:
            log.exception("kline callback error")

    # ---- signal
    def _evaluate(self, symbol: str) -> None:
        try:
            with self.bar_locks[symbol]:
                bars = list(self.bars[symbol])
            if len(bars) < SWEEP_LOOKBACK + ATR_LEN + 2:
                return
            fz = self.funding.funding_z(symbol)
            setup = detect_spring(bars, fz, SPRING_CFG)
            if setup is None:
                return
            event_id = uuid.uuid4().hex
            entry_time = datetime.fromtimestamp(bars[-1]["ts"] / 1000, tz=timezone.utc).isoformat()
            # Skip (still report) if portfolio gates would block execution.
            blocked = None
            if self.executor.has_symbol(symbol):
                blocked = "symbol already in position"
            elif self.executor.enabled and self.executor.open_count() >= MAX_CONCURRENT:
                blocked = f"concurrency cap {MAX_CONCURRENT}"

            execution = {"ok": False, "skipped": True, "message": blocked or "signal-only"}
            if blocked is None:
                try:
                    execution = self.executor.execute(symbol, setup)
                except Exception as exc:
                    log.exception("execution error")
                    execution = {"ok": False, "message": f"execution error: {exc}"}

            signal_data = {"event_id": event_id, "symbol": symbol, "strategy": STRATEGY_NAME,
                           "entry_time": entry_time, **setup, "execution": execution}
            payload = self._build_rl_payload(signal_data)
            signal_data["rl_dispatch"] = self.rl.enqueue(payload)
            self._write_ledger(signal_data)
            self._notify(signal_data)
            log.info("SIGNAL %s long entry=%.6g sl=%.6g fz=%.2f exec=%s", symbol, setup["entry"], setup["stop"],
                     setup["funding_z"], execution.get("order_id") or execution.get("message"))
        except Exception:
            log.exception("evaluate error for %s", symbol)

    def _build_rl_payload(self, sd: dict[str, Any]) -> dict[str, Any]:
        entry, stop = float(sd["entry"]), float(sd["stop"])
        risk = entry - stop
        execution = sd.get("execution") if isinstance(sd.get("execution"), dict) else {}
        features = {
            "funding_z": sd.get("funding_z"), "week_fraction": sd.get("week_fraction"),
            "lower_wick_frac": sd.get("lower_wick_frac"), "close_pos": sd.get("close_pos"),
            "atr": sd.get("atr"), "sweep_depth_pct": (sd.get("prev_deep_low", 0) - sd.get("spring_low", 0)) / entry if entry else None,
            "normal_bot_status_accepted": 1.0, "normal_bot_status_rejected": 0.0,
        }
        return jsonable({
            "schema_version": "rl_signal_v1", "event_id": sd.get("event_id"), "source": "capitulation-spring-bot",
            "sent_at": datetime.now(timezone.utc).isoformat(), "status": "accepted", "reason": None,
            "symbol": sd.get("symbol"), "strategy": STRATEGY_NAME, "direction": "long",
            "setup": {
                "entry": entry,
                "stop_loss": stop,
                # RL sidecar reads take_profit / take_profit_levels and picks ONE tier
                # (its own full-position exit). Give it the real R-tiers to choose from;
                # single fallback = the middle (12R) tier.
                "take_profit": (sd.get("targets") or [sd.get("target")])[len(sd.get("targets") or [1]) // 2],
                "take_profit_levels": sd.get("targets"),
                # Strategy's scaled plan (the NORMAL sidecar executes this; RL ignores it).
                "tp_prices": sd.get("targets"), "tp_qty_pcts": sd.get("tp_qty_pct"), "tp_r": sd.get("tp_r"),
                "trail_dist": None, "exit_style": "multi_tp_be", "move_sl_to_be_after_tp1": True,
                "entry_time": sd.get("entry_time"), "atr": sd.get("atr"),
                "stop_distance_pct": risk / entry if entry > 0 else None,
                "fee_to_price_risk": (TAKER_FEE_RATE * (entry + stop) / risk) if risk > 0 else None,
                "upstream_order_id": execution.get("order_id"), "upstream_order_link_id": execution.get("order_link_id"),
            },
            "features": features, "feature_columns": list(features.keys()),
            "default_risk_pct": RISK_PCT,
            "risk_config": {"risk_pct": RISK_PCT, "risk_usdt": RISK_USDT, "max_concurrent": MAX_CONCURRENT,
                            "taker_fee_rate": TAKER_FEE_RATE, "min_stop_distance_pct": MIN_STOP_DISTANCE_PCT,
                            "funding_z_thr": FUNDING_Z_THR, "sweep_lookback": SWEEP_LOOKBACK},
            "raw_signal": sd,
            "extra": {"direct_execution": execution, "spring_low": sd.get("spring_low"),
                      "prev_deep_low": sd.get("prev_deep_low")},
        })

    def _notify(self, sd: dict[str, Any]) -> None:
        execution = sd.get("execution") or {}
        lines = [
            "<b>[CAPITULATION SPRING] LONG SIGNAL</b>",
            f"Symbol: <b>{escape(sd['symbol'])}</b>  Time: <code>{escape(sd['entry_time'])}</code>",
            f"Entry: <code>{fmt(sd['entry'], 6)}</code>  SL: <code>{fmt(sd['stop'], 6)}</code>",
            f"TPs: <code>{', '.join(f'{int(r)}R={fmt(p,6)}' for r, p in zip(sd['tp_r'], sd['targets']))}</code>",
            f"funding_z: <code>{fmt(sd['funding_z'], 2)}</code>  week: <code>{fmt(sd['week_fraction'], 2)}</code>  "
            f"wick: <code>{fmt(sd['lower_wick_frac'], 2)}</code>",
        ]
        if execution.get("ok"):
            lines.append(f"Demo order: <b>placed</b> qty <code>{escape(str(execution.get('qty')))}</code> "
                         f"risk~<code>{fmt(float(execution.get('expected_total_risk', 0.0)), 2)} USDT</code>")
        else:
            tag = "skipped" if execution.get("skipped") else "not placed"
            lines.append(f"Demo order: <b>{escape(tag)}</b> - <code>{escape(str(execution.get('message', '-')))}</code>")
        rl = sd.get("rl_dispatch") or {}
        if rl.get("enabled"):
            lines.append(f"RL sidecar: <b>{'queued' if rl.get('queued') else 'not queued'}</b>")
        self.telegram.send(lines)

    def _write_ledger(self, payload: dict[str, Any]) -> None:
        LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_lock:
            with LEDGER_PATH.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(jsonable(payload), separators=(",", ":")) + "\n")

    def stop(self) -> None:
        self.stop_event.set()
        try:
            if self.ws is not None:
                self.ws.exit()
        except Exception:
            pass


def main() -> None:
    bot = CapitulationSpringBot()

    def handle(signum: int, _frame: Any) -> None:
        log.info("signal %s -> stopping", signum)
        bot.stop()

    signal_module.signal(signal_module.SIGTERM, handle)
    signal_module.signal(signal_module.SIGINT, handle)
    bot.start()


if __name__ == "__main__":
    main()
