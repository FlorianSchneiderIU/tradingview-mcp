#!/usr/bin/env python3
"""
matrix_signal_bot.py — Matrix room signal follower
===================================================
Joins a Matrix room, parses trading signals from messages, and places
market orders on Bybit (Unified Trading linear perpetuals).

Supported signal formats
------------------------
1. Bot's own Telegram-style format (HTML or plain text):
       [STRATEGY] ACCEPTED SIGNAL
       Symbol: BTCUSDT
       Direction: LONG
       Entry: 105000
       Stop Loss: 104000
       Target: 107000

2. Compact key=value format (single line or multi-line):
       BTCUSDT LONG entry=105000 sl=104000 tp=107000

3. JSON (sent as a Matrix code block or bare JSON):
       {"symbol":"BTCUSDT","direction":"long","entry":105000,"sl":104000,"tp":107000}

Signal fields recognised
------------------------
  symbol    : trading pair, e.g. BTCUSDT
  direction : long / buy   or   short / sell
  entry     : entry price (used for sizing; actual fill is market)
  sl        : stop-loss price
  tp / tp1  : take-profit / target price (optional)
  strategy  : label for logging (optional, default "matrix")

Required env vars
-----------------
  BYBIT_API_KEY, BYBIT_API_SECRET
  BYBIT_DEMO                  — "true" (default) = demo, "false" = live
  LIVE_TRADING_CONFIRM        — must be "true" when BYBIT_DEMO=false
  MATRIX_HOMESERVER           — e.g. https://matrix.org
  MATRIX_ACCESS_TOKEN         — bot account access token
  MATRIX_ROOM_ID              — !roomid:homeserver  (must already be joined)

Optional env vars
-----------------
  NOTIONAL_PCT                — fraction of equity risked (default 0.01)
  TAKER_FEE_RATE              — estimated one-way taker fee (default 0.00055)
  MAX_FEE_TO_PRICE_RISK       — reject when fees > this fraction of SL risk (default 0.25)
  ORDER_LEVERAGE_BUFFER       — dynamic leverage buffer (default 2.0)
  MIN_STOP_DISTANCE_PCT       — minimum SL distance as fraction of entry (default 0.001)
  MAX_OPEN_POSITIONS          — max simultaneous positions (default 5)
  MATRIX_SIGNAL_SENDER        — restrict signals to this Matrix user ID (optional)
  MATRIX_POST_REPLY           — "true" to post order confirmations back to room (default true)
  LOG_DIR                     — log directory (default /app/logs)
  ACTIVE_TRADES_STATE_PATH    — shared active-trades JSON (default /app/logs/active_trades.json)
"""
from __future__ import annotations

import asyncio
import html
import json
import logging
import math
import os
import re
import sys
import threading
import time
import uuid
from typing import Optional

import numpy as np
import requests
from pybit.unified_trading import HTTP

# --- Config -------------------------------------------------------------------
DEMO = os.environ.get("BYBIT_DEMO", "true").lower() in ("1", "true", "yes")
LIVE_TRADING_CONFIRM = os.environ.get("LIVE_TRADING_CONFIRM", "false").lower() in ("1", "true", "yes")
NOTIONAL_PCT = float(os.environ.get("NOTIONAL_PCT", os.environ.get("RISK_PCT", "0.01")))
TAKER_FEE_RATE = float(os.environ.get("TAKER_FEE_RATE", "0.00055"))
MAX_FEE_TO_PRICE_RISK = float(os.environ.get("MAX_FEE_TO_PRICE_RISK", "0.25"))
ORDER_LEVERAGE_BUFFER = float(os.environ.get("ORDER_LEVERAGE_BUFFER", "2.0"))
MIN_STOP_DISTANCE_PCT = float(os.environ.get("MIN_STOP_DISTANCE_PCT", "0.001"))
MAX_OPEN = int(os.environ.get("MAX_OPEN_POSITIONS", "5"))
LOG_DIR = os.environ.get("LOG_DIR", "/app/logs")
ACTIVE_TRADES_STATE_PATH = os.environ.get(
    "ACTIVE_TRADES_STATE_PATH",
    os.path.join(LOG_DIR, "active_trades.json"),
)

MATRIX_HOMESERVER = os.environ.get("MATRIX_HOMESERVER", "").strip()
MATRIX_ACCESS_TOKEN = os.environ.get("MATRIX_ACCESS_TOKEN", "").strip()
MATRIX_ROOM_ID = os.environ.get("MATRIX_ROOM_ID", "").strip()
MATRIX_SIGNAL_SENDER = os.environ.get("MATRIX_SIGNAL_SENDER", "").strip()
MATRIX_POST_REPLY = os.environ.get("MATRIX_POST_REPLY", "true").lower() in ("1", "true", "yes")

# --- Logging ------------------------------------------------------------------
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-5s] [matrix-bot] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, "matrix_signal_bot.log")),
    ],
)
log = logging.getLogger("matrix_bot")


# --- Signal parser ------------------------------------------------------------

# ── Bandit / LDZ format helpers ───────────────────────────────────────────────
# Two real-world formats seen in the rooms:
#
# Format A — 🔔 OPEN LONG/SHORT (TnB Party / TnB Pro / Bandit Tips):
#   🔔 OPEN LONG ▲
#   #XRPUSDT.P - 2m
#   ...
#   ➾ Long Entry Zone: 1.3395
#   TP1:  1.3516
#   ❌ Stop: 1.2993
#
# Format B — inline header (ETH / ADA / SOL / XRP per-coin rooms):
#   #ETHUSDT.P 30m | ⅂ⅆℤ ᵛ³⁻⁴
#   ➾ Short Entry Zone: 2324.96
#   TP1:  2300.78
#   ❌ Stop: 2394.71
#
# Format C — wolfe_entry / custom keyword (Wolfe room, TradingView alerts):
#   wolfe_entry long BTCUSDT entry=50000 sl=49000 tp=52000
#   (or any line containing wolfe_entry/wolfe_long/wolfe_short)
#
# Format D — 🌀 OPEN LONG/SHORT (Curling / ⅂C channel):
#   🌀 OPEN LONG #XAUUSDT.P 5m | ⅂C
#   ➾ Long Entry Zone: 4514.88
#   TP1: 4560.03  ...  TP10: 5417.86
#   ❌ Stop: 4424.58
#   Leverage: 300x Cross

_ENTRY_ZONE_RE = re.compile(
    r"➾\s+(Long|Short)\s+Entry\s+Zone:\s*([\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)
_STOP_RE  = re.compile(r"❌\s+Stop:\s*([\d,]+(?:\.\d+)?)")
_LDZ_LEVEL_RE = re.compile(r"🧲\s*⅂ⅆℤ:\s*([\d,]+(?:\.\d+)?)")
_TP1_RE   = re.compile(r"\bTP1:\s*([\d,]+(?:\.\d+)?)")
_TP_ALL_RE = re.compile(r"\bTP(\d{1,2}):\s*([\d,]+(?:\.\d+)?)")  # TP1..TP10
# Symbol: #XRPUSDT.P, #ETHUSDT.P 30m, #XRP (short alias)
_SYM_RE   = re.compile(r"#([A-Z]{2,10}(?:USDT|USDC|BTC|ETH|USD|XAU|XAG|XRP|XLM|SOL|ADA|BTC|BNB)?)(?:\.P)?", re.IGNORECASE)
# Format A header (🔔) and Format D header (🌀 curling)
_OPEN_DIR_RE   = re.compile(r"🔔\s+OPEN\s+(LONG|SHORT)", re.IGNORECASE)
_CURLING_DIR_RE = re.compile(r"🌀\s+OPEN\s+(LONG|SHORT)", re.IGNORECASE)
# Preliminary signals (need confirmation — skip for auto-trading)
_PRELIMINARY_RE = re.compile(r"⚡\s*//\s*PRELIMINARY", re.IGNORECASE)
# wolfe_entry / wolfe_long / wolfe_short keywords
_WOLFE_RE = re.compile(
    r"\bwolfe[_\s]?(entry|long|short|buy|sell)\b",
    re.IGNORECASE,
)

# Generic key=value fallback
_KV_RE = re.compile(
    r"(?:symbol|sym)[:\s=]+(?P<sym>[A-Z]{3,12}USDT)\b"
    r"|(?:direction|side|signal)[:\s=]+(?P<dir>long|short|buy|sell)"
    r"|(?:entry|entry[_\s]?price)[:\s=]+(?P<entry>[0-9]+(?:\.[0-9]+)?)"
    r"|(?:stop[_\s]?loss|sl|stop)[:\s=]+(?P<sl>[0-9]+(?:\.[0-9]+)?)"
    r"|(?:take[_\s]?profit|tp1?|target(?:[_\s]?price)?)[:\s=]+(?P<tp>[0-9]+(?:\.[0-9]+)?)"
    r"|(?:strategy)[:\s=]+(?P<strat>[A-Za-z0-9_\-]+)",
    re.IGNORECASE,
)
_INLINE_RE = re.compile(
    r"(?P<sym>[A-Z]{3,12}USDT)\s+(?P<dir>long|short|buy|sell)"
    r"(?:.*?entry[=:\s]+(?P<entry>[0-9]+(?:\.[0-9]+)?))?"
    r"(?:.*?(?:sl|stop)[=:\s]+(?P<sl>[0-9]+(?:\.[0-9]+)?))?"
    r"(?:.*?(?:tp1?|target)[=:\s]+(?P<tp>[0-9]+(?:\.[0-9]+)?))?",
    re.IGNORECASE | re.DOTALL,
)

# Short-alias → canonical USDT symbol map (add more as needed)
_SHORT_ALIAS: dict[str, str] = {
    "XAU": "XAUUSDT", "XAG": "XAGUSDT",
    "BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT",
    "ADA": "ADAUSDT", "XRP": "XRPUSDT", "XLM": "XLMUSDT",
    "BNB": "BNBUSDT", "DOT": "DOTUSDT", "LINK": "LINKUSDT",
    "AVAX": "AVAXUSDT", "OP": "OPUSDT", "FIL": "FILUSDT",
    "WIF": "WIFUSDT",
}


def _parse_price(s: str) -> float:
    """Parse price string that may contain commas (e.g. '4,515.86')."""
    return float(s.replace(",", ""))


def _normalise_symbol(raw: str) -> str:
    """Ensure symbol ends with USDT; expand short aliases."""
    sym = raw.upper()
    if sym in _SHORT_ALIAS:
        return _SHORT_ALIAS[sym]
    if not any(sym.endswith(suf) for suf in ("USDT", "USDC", "BTC", "ETH", "PERP")):
        return sym + "USDT"
    return sym


def _strip_html(text: str) -> str:
    """Remove HTML tags and decode entities."""
    return html.unescape(re.sub(r"<[^>]+>", " ", text))


def parse_signal(text: str) -> Optional[dict]:
    """
    Try to extract a trading signal from a message string.
    Returns dict with keys: symbol, signal, entry, sl, tp1, strategy.
    Returns None if no valid signal detected.

    Parser priority:
      1. Preliminary signals (⚡ PRELIMINARY) → skip, return None
      2. Bandit/LDZ format A (🔔 OPEN LONG/SHORT)
      3. Bandit/LDZ format B (#SYMBOL TF | ⅂ⅆℤ)
      4. wolfe_entry / wolfe_long / wolfe_short keyword lines
      5. JSON fragments
      6. Generic key=value scan
      7. Inline compact (SYMBOL LONG entry=X sl=Y)
    """
    clean = _strip_html(text).strip()

    # 1. Skip preliminary / unconfirmed signals
    if _PRELIMINARY_RE.search(clean):
        return None

    # 2 & 3 — Bandit / LDZ format (both share the same ➾ Entry Zone / ❌ Stop structure)
    entry_m = _ENTRY_ZONE_RE.search(clean)
    stop_m  = _STOP_RE.search(clean)
    if entry_m and stop_m:
        direction = "long" if entry_m.group(1).lower() == "long" else "short"
        entry = _parse_price(entry_m.group(2))
        stop  = _parse_price(stop_m.group(1))
        tp1_m = _TP1_RE.search(clean)
        tp1   = _parse_price(tp1_m.group(1)) if tp1_m else None

        # Prefer the explicit OPEN LONG/SHORT direction if present
        open_m    = _OPEN_DIR_RE.search(clean)    # 🔔 Format A
        curling_m = _CURLING_DIR_RE.search(clean)  # 🌀 Format D
        if open_m:
            direction = "long" if open_m.group(1).upper() == "LONG" else "short"
            strategy  = "bandit_open"
        elif curling_m:
            direction = "long" if curling_m.group(1).upper() == "LONG" else "short"
            strategy  = "curling"
        else:
            strategy  = "bandit_ldz"

        sym_m = _SYM_RE.search(clean)
        if sym_m:
            symbol = _normalise_symbol(sym_m.group(1))
            sig: dict = {
                "symbol":   symbol,
                "signal":   direction,
                "entry":    entry,
                "sl":       stop,
                "strategy": strategy,
            }
            if tp1 is not None:
                sig["tp1"] = tp1
            # Capture all TPs (TP1..TP10) if present
            all_tps = {int(m.group(1)): _parse_price(m.group(2)) for m in _TP_ALL_RE.finditer(clean)}
            if len(all_tps) > 1:
                sig["tps"] = [all_tps[k] for k in sorted(all_tps)]
            # Capture LDZ level if present (🧲 ⅂ⅆℤ: X.XX)
            ldz_m = _LDZ_LEVEL_RE.search(clean)
            if ldz_m:
                sig["ldz_level"] = _parse_price(ldz_m.group(1))
            if _is_valid_signal(sig):
                return sig

    # 4. wolfe_entry / wolfe_long / wolfe_short keyword
    wolfe_m = _WOLFE_RE.search(clean)
    if wolfe_m:
        kw = wolfe_m.group(1).lower()
        wdir: Optional[str] = None
        if kw in ("long", "buy"):
            wdir = "long"
        elif kw in ("short", "sell"):
            wdir = "short"

        wsig: dict = {"strategy": "wolfe_entry"}
        if wdir:
            wsig["signal"] = wdir

        # Try key=value pairs first (symbol=X entry=Y sl=Z)
        for m in _KV_RE.finditer(clean):
            if m.group("sym") and "symbol" not in wsig:
                wsig["symbol"] = m.group("sym").upper()
            if m.group("dir") and "signal" not in wsig:
                wsig["signal"] = _normalise_direction(m.group("dir"))
            if m.group("entry") and "entry" not in wsig:
                wsig["entry"] = float(m.group("entry"))
            if m.group("sl") and "sl" not in wsig:
                wsig["sl"] = float(m.group("sl"))
            if m.group("tp") and "tp1" not in wsig:
                wsig["tp1"] = float(m.group("tp"))

        # Fall back to SYMBOL LONG/SHORT inline pattern or reversed
        if not _is_valid_signal(wsig):
            # Try "SYMBOL dir entry=X sl=Y" order
            m2 = _INLINE_RE.search(clean)
            if m2:
                if "symbol" not in wsig and m2.group("sym"):
                    wsig["symbol"] = m2.group("sym").upper()
                if "signal" not in wsig and m2.group("dir"):
                    wsig["signal"] = _normalise_direction(m2.group("dir"))
                if "entry" not in wsig and m2.group("entry"):
                    wsig["entry"] = float(m2.group("entry"))
                if "sl" not in wsig and m2.group("sl"):
                    wsig["sl"] = float(m2.group("sl"))
                if "tp1" not in wsig and m2.group("tp"):
                    wsig["tp1"] = float(m2.group("tp"))
            # Also try reversed "dir SYMBOL entry=X sl=Y" order
            if not _is_valid_signal(wsig):
                rev_m = re.search(
                    r"(?P<dir>long|short|buy|sell)\s+(?P<sym>[A-Z]{3,12}USDT)"
                    r"(?:.*?entry[=:\s]+(?P<entry>[\d.]+))?"
                    r"(?:.*?(?:sl|stop)[=:\s]+(?P<sl>[\d.]+))?"
                    r"(?:.*?(?:tp1?|target)[=:\s]+(?P<tp>[\d.]+))?",
                    clean, re.IGNORECASE | re.DOTALL,
                )
                if rev_m:
                    if "symbol" not in wsig and rev_m.group("sym"):
                        wsig["symbol"] = rev_m.group("sym").upper()
                    if "signal" not in wsig and rev_m.group("dir"):
                        wsig["signal"] = _normalise_direction(rev_m.group("dir"))
                    if "entry" not in wsig and rev_m.group("entry"):
                        wsig["entry"] = float(rev_m.group("entry"))
                    if "sl" not in wsig and rev_m.group("sl"):
                        wsig["sl"] = float(rev_m.group("sl"))
                    if "tp1" not in wsig and rev_m.group("tp"):
                        wsig["tp1"] = float(rev_m.group("tp"))

        if _is_valid_signal(wsig):
            return wsig

    # 5. JSON fragments
    for fragment in re.findall(r"\{[^{}]+\}", clean, re.DOTALL):
        try:
            obj = json.loads(fragment)
            sig2 = _build_from_dict(obj)
            if sig2:
                return sig2
        except json.JSONDecodeError:
            pass

    # 6. Generic key=value scan
    kvsig: dict = {}
    for m in _KV_RE.finditer(clean):
        if m.group("sym") and "symbol" not in kvsig:
            kvsig["symbol"] = m.group("sym").upper()
        if m.group("dir") and "signal" not in kvsig:
            kvsig["signal"] = _normalise_direction(m.group("dir"))
        if m.group("entry") and "entry" not in kvsig:
            kvsig["entry"] = float(m.group("entry"))
        if m.group("sl") and "sl" not in kvsig:
            kvsig["sl"] = float(m.group("sl"))
        if m.group("tp") and "tp1" not in kvsig:
            kvsig["tp1"] = float(m.group("tp"))
        if m.group("strat") and "strategy" not in kvsig:
            kvsig["strategy"] = m.group("strat")
    if _is_valid_signal(kvsig):
        kvsig.setdefault("strategy", "matrix")
        return kvsig

    # 7. Inline compact: "BTCUSDT LONG entry=X sl=Y tp=Z"
    m2 = _INLINE_RE.search(clean)
    if m2:
        inlsig: dict = {
            "symbol":   m2.group("sym").upper(),
            "signal":   _normalise_direction(m2.group("dir")),
            "strategy": "matrix",
        }
        if m2.group("entry"):
            inlsig["entry"] = float(m2.group("entry"))
        if m2.group("sl"):
            inlsig["sl"] = float(m2.group("sl"))
        if m2.group("tp"):
            inlsig["tp1"] = float(m2.group("tp"))
        if _is_valid_signal(inlsig):
            return inlsig

    return None


def _build_from_dict(obj: dict) -> Optional[dict]:
    sig: dict = {}
    for key in ("symbol", "sym", "ticker"):
        if key in obj and isinstance(obj[key], str):
            sig["symbol"] = obj[key].upper()
            break
    for key in ("direction", "side", "signal", "action"):
        if key in obj:
            norm = _normalise_direction(str(obj[key]))
            if norm:
                sig["signal"] = norm
                break
    for field, aliases in (
        ("entry",  ("entry", "entry_price", "price")),
        ("sl",     ("sl", "stop_loss", "stop", "stoploss")),
        ("tp1",    ("tp1", "tp", "take_profit", "target", "target_price")),
    ):
        for alias in aliases:
            if alias in obj:
                try:
                    sig[field] = float(obj[alias])
                    break
                except (TypeError, ValueError):
                    pass
    sig.setdefault("strategy", str(obj.get("strategy", "matrix")))
    return sig if _is_valid_signal(sig) else None


def _normalise_direction(raw: str) -> Optional[str]:
    v = raw.strip().lower()
    if v in ("long", "buy"):
        return "long"
    if v in ("short", "sell"):
        return "short"
    return None


def _is_valid_signal(sig: dict) -> bool:
    return (
        bool(sig.get("symbol"))
        and sig.get("signal") in ("long", "short")
        and sig.get("entry", 0) > 0
        and sig.get("sl", 0) > 0
    )


# --- Bybit helpers ------------------------------------------------------------

def round_to_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    precision = max(0, round(-math.log10(step)))
    return round(round(value / step) * step, precision)


def floor_to_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    precision = max(0, round(-math.log10(step)))
    return round(math.floor(value / step) * step, precision)


def qty_to_str(value: float, step: float = 0.0) -> str:
    if step > 0:
        precision = max(0, round(-math.log10(step)))
        return f"{value:.{precision}f}"
    return f"{round(value, 8):.8f}".rstrip("0").rstrip(".") or "0"


def get_instrument_info(http: HTTP, symbol: str) -> dict:
    resp = http.get_instruments_info(category="linear", symbol=symbol)
    items = resp.get("result", {}).get("list", [])
    if not items:
        return {}
    item = items[0]
    lot   = item.get("lotSizeFilter", {})
    price = item.get("priceFilter", {})
    lev   = item.get("leverageFilter", {})
    return {
        "status":       str(item.get("status", "")),
        "qty_step":     float(lot.get("qtyStep",     "0.001")),
        "min_qty":      float(lot.get("minOrderQty", "0.001")),
        "tick_size":    float(price.get("tickSize",  "0.01")),
        "max_leverage": float(lev.get("maxLeverage", "1") or "1"),
    }


def get_balance_metrics(http: HTTP) -> dict[str, float]:
    resp = http.get_wallet_balance(accountType="UNIFIED")
    row  = resp.get("result", {}).get("list", [{}])[0]
    total_equity    = float(row.get("totalEquity", 0) or 0)
    total_available = float(row.get("totalAvailableBalance", 0) or 0)
    usdt_equity = usdt_available = 0.0
    for coin in row.get("coin", []):
        if coin.get("coin") != "USDT":
            continue
        usdt_equity    = float(coin.get("equity", 0) or 0)
        usdt_available = float(
            coin.get("availableToWithdraw")
            or coin.get("availableToBorrow")
            or 0
        )
        break
    equity    = usdt_equity    if usdt_equity    > 0 else total_equity
    available = usdt_available if usdt_available > 0 else total_available
    if usdt_available > 0 and total_available > 0:
        available = min(usdt_available, total_available)
    return {"equity": equity, "available": available}


# --- Order executor -----------------------------------------------------------

class OrderExecutor:
    """Thin wrapper around Bybit HTTP that executes a single signal."""

    def __init__(self, http: HTTP, *, open_count_fn=None):
        self._http = http
        self._lock = threading.Lock()
        # Callable returning current open position count for gate-check.
        self._open_count_fn = open_count_fn or (lambda: 0)
        self._info_cache: dict[str, dict] = {}

    def _get_info(self, symbol: str) -> dict:
        if symbol not in self._info_cache:
            self._info_cache[symbol] = get_instrument_info(self._http, symbol)
        return self._info_cache[symbol]

    def _set_leverage(self, symbol: str, leverage: int) -> None:
        try:
            self._http.set_leverage(
                category="linear",
                symbol=symbol,
                buyLeverage=str(leverage),
                sellLeverage=str(leverage),
            )
        except Exception as exc:
            log.debug("[%s] set_leverage skipped: %s", symbol, exc)

    def execute(self, sig: dict) -> dict:
        """
        Place a market order from a parsed signal dict.
        Returns a result dict with keys: ok, order_id, message.
        """
        symbol = sig["symbol"]
        direction = sig["signal"]
        entry  = float(sig["entry"])
        stop   = float(sig["sl"])
        tp1    = float(sig.get("tp1") or 0.0)
        strategy = str(sig.get("strategy", "matrix"))

        with self._lock:
            open_count = self._open_count_fn()
            if open_count >= MAX_OPEN:
                return {
                    "ok": False,
                    "message": f"Position limit reached ({open_count}/{MAX_OPEN})",
                }

        info = self._get_info(symbol)
        if not info:
            return {"ok": False, "message": f"Unknown symbol: {symbol}"}
        if info.get("status") and info["status"] != "Trading":
            return {"ok": False, "message": f"{symbol} status={info['status']}"}

        unit_risk = abs(entry - stop)
        if unit_risk <= 0:
            return {"ok": False, "message": "entry == sl"}
        if unit_risk / entry < MIN_STOP_DISTANCE_PCT:
            return {
                "ok": False,
                "message": (
                    f"SL too close: {unit_risk / entry:.4%} < {MIN_STOP_DISTANCE_PCT:.4%}"
                ),
            }
        fee_risk_per_unit = TAKER_FEE_RATE * (entry + stop)
        if fee_risk_per_unit / unit_risk > MAX_FEE_TO_PRICE_RISK:
            return {
                "ok": False,
                "message": (
                    f"Fee/risk ratio too high: "
                    f"{fee_risk_per_unit / unit_risk:.2%} > {MAX_FEE_TO_PRICE_RISK:.2%}"
                ),
            }

        balances  = get_balance_metrics(self._http)
        equity    = float(balances.get("equity", 0.0))
        available = float(balances.get("available", 0.0))
        if equity <= 0:
            return {"ok": False, "message": f"Bad equity: {equity}"}

        max_leverage = max(float(info.get("max_leverage", 1.0)), 1.0)
        risk_budget  = equity * NOTIONAL_PCT
        raw_qty      = risk_budget / (unit_risk + fee_risk_per_unit)
        margin_basis = available if available > 0 else equity
        max_qty_by_margin = (margin_basis * max_leverage * 0.95) / entry
        if raw_qty > max_qty_by_margin:
            log.warning(
                "[%s] qty capped by margin: raw=%.6g max=%.6g", symbol, raw_qty, max_qty_by_margin
            )
            raw_qty = max_qty_by_margin

        q_step = info["qty_step"]
        min_q  = info["min_qty"]
        tick   = info["tick_size"]
        qty = floor_to_step(raw_qty, q_step)
        if qty < min_q:
            qty = min_q

        # Dynamic leverage: set to the minimum needed to cover the notional
        notional   = qty * entry
        lev_needed = math.ceil(notional / max(equity * NOTIONAL_PCT, 1.0) * ORDER_LEVERAGE_BUFFER)
        order_lev  = max(1, min(int(lev_needed), int(max_leverage)))
        try:
            self._set_leverage(symbol, order_lev)
        except Exception:
            pass

        sl_price  = round_to_step(stop, tick)
        tp_price  = round_to_step(tp1, tick) if tp1 > 0 else None
        side      = "Buy" if direction == "long" else "Sell"
        # positionIdx: 0=one-way, 1=hedge-long, 2=hedge-short
        # Try one-way first; if the account is in hedge mode we retry with the
        # correct hedge-mode idx (1 for long, 2 for short).
        pos_idx   = 0
        order_link_id = f"mx_{strategy[:4]}_{symbol[:6]}_{direction[0]}_{uuid.uuid4().hex[:8]}"

        order_kwargs: dict = dict(
            category    = "linear",
            symbol      = symbol,
            side        = side,
            orderType   = "Market",
            qty         = qty_to_str(qty, q_step),
            stopLoss    = str(sl_price),
            slTriggerBy = "LastPrice",
            positionIdx = pos_idx,
            orderLinkId = order_link_id,
        )
        if tp_price is not None and tp_price > 0:
            order_kwargs["takeProfit"]  = str(tp_price)
            order_kwargs["tpTriggerBy"] = "LastPrice"
            order_kwargs["tpslMode"]    = "Full"   # close 100% of position at TP1

        log.info(
            "[%s] SIGNAL %s | strategy=%s entry~%.5g sl=%s tp=%s "
            "qty=%s notional=%.2f lev=%dx equity=%.2f",
            symbol, direction.upper(), strategy,
            entry, sl_price, tp_price or "-",
            qty_to_str(qty, q_step), notional, order_lev, equity,
        )

        try:
            resp = self._http.place_order(**order_kwargs)
        except Exception as exc:
            exc_str = str(exc).lower()
            # pybit raises the Bybit error as an exception — check for hedge-mode mismatch
            if "position idx" in exc_str or ("10001" in exc_str and "position" in exc_str):
                hedge_idx = 1 if direction == "long" else 2
                log.warning(
                    "[%s] Account is in hedge mode — retrying with positionIdx=%d", symbol, hedge_idx
                )
                order_kwargs["positionIdx"] = hedge_idx
                order_kwargs["orderLinkId"] = (
                    f"mx_{strategy[:4]}_{symbol[:6]}_{direction[0]}_{uuid.uuid4().hex[:8]}"
                )
                try:
                    resp = self._http.place_order(**order_kwargs)
                except Exception as exc2:
                    log.error("[%s] place_order (hedge retry) exception: %s", symbol, exc2)
                    return {"ok": False, "message": f"API exception: {exc2}"}
            else:
                log.error("[%s] place_order exception: %s", symbol, exc)
                return {"ok": False, "message": f"API exception: {exc}"}

        ret_code = resp.get("retCode", -1)
        # Also handle the case where it comes back as a retCode (non-raising clients)
        if ret_code == 10001 and "position idx" in resp.get("retMsg", "").lower():
            hedge_idx = 1 if direction == "long" else 2
            log.warning(
                "[%s] Account is in hedge mode (retCode) — retrying with positionIdx=%d", symbol, hedge_idx
            )
            order_kwargs["positionIdx"] = hedge_idx
            order_kwargs["orderLinkId"] = (
                f"mx_{strategy[:4]}_{symbol[:6]}_{direction[0]}_{uuid.uuid4().hex[:8]}"
            )
            try:
                resp = self._http.place_order(**order_kwargs)
            except Exception as exc:
                log.error("[%s] place_order (hedge retry) exception: %s", symbol, exc)
                return {"ok": False, "message": f"API exception: {exc}"}
            ret_code = resp.get("retCode", -1)

        if ret_code != 0:
            msg = resp.get("retMsg", "?")
            log.error("[%s] Order rejected retCode=%s: %s", symbol, ret_code, msg)
            return {"ok": False, "message": f"Order rejected ({ret_code}): {msg}"}

        order_id = resp.get("result", {}).get("orderId", "?")
        log.info("[%s] Order accepted orderId=%s linkId=%s", symbol, order_id, order_link_id)
        return {
            "ok":      True,
            "order_id": order_id,
            "link_id": order_link_id,
            "qty":     qty_to_str(qty, q_step),
            "notional": f"{notional:.2f}",
            "message": (
                f"Order placed: {side} {qty_to_str(qty, q_step)} {symbol} "
                f"| sl={sl_price} tp={tp_price or '-'} | id={order_id}"
            ),
        }


# --- Matrix client (nio) ------------------------------------------------------

try:
    from nio import AsyncClient, InviteMemberEvent, MatrixRoom, RoomMessageText
except ImportError:
    sys.exit(
        "matrix-nio is not installed. "
        "Run: pip install matrix-nio  (or add it to requirements.txt)"
    )


class MatrixSignalBot:
    def __init__(self, executor: OrderExecutor):
        self._executor = executor
        self._client = AsyncClient(MATRIX_HOMESERVER)
        self._client.access_token = MATRIX_ACCESS_TOKEN
        self._client.user_id = None  # will be populated by whoami()
        self._processed_event_ids: set[str] = set()

    async def start(self) -> None:
        log.info(
            "Matrix bot starting | homeserver=%s room=%s sender_filter=%s",
            MATRIX_HOMESERVER,
            MATRIX_ROOM_ID or "(all joined rooms)",
            MATRIX_SIGNAL_SENDER or "(any)",
        )
        # Verify credentials with a whoami call
        try:
            whoami = await self._client.whoami()
            log.info("Authenticated as %s", whoami.user_id)
        except Exception as exc:
            log.error("Matrix authentication failed: %s", exc)
            raise

        # Advance the sync token to "now" so we only process future messages.
        # We do a single sync with timeout=0 to fast-forward without processing
        # any existing messages that predate this run.
        await self._client.sync(timeout=0, full_state=True)
        log.info("Matrix initial sync complete — listening for new messages")

        self._client.add_event_callback(self._on_invite, InviteMemberEvent)
        self._client.add_event_callback(self._on_message, RoomMessageText)

        # Long-poll sync loop
        while True:
            try:
                await self._client.sync(timeout=30_000)
            except Exception as exc:
                log.warning("Sync error: %s — retrying in 5s", exc)
                await asyncio.sleep(5)

    async def _on_invite(self, room: MatrixRoom, event: InviteMemberEvent) -> None:
        """Auto-join any room the bot is invited to."""
        if event.membership != "invite":
            return
        log.info("Invite received for room %s — joining", room.room_id)
        result = await self._client.join(room.room_id)
        if hasattr(result, "room_id"):
            log.info("Joined room %s", result.room_id)
        else:
            log.warning("Failed to join room %s: %s", room.room_id, result)

    async def _on_message(self, room: MatrixRoom, event: RoomMessageText) -> None:
        # Filter by configured room if set; otherwise accept any joined room
        if MATRIX_ROOM_ID and room.room_id != MATRIX_ROOM_ID:
            return
        # Dedup (nio may deliver the same event twice after reconnect)
        if event.event_id in self._processed_event_ids:
            return
        self._processed_event_ids.add(event.event_id)
        # Trim dedup set to avoid unbounded growth
        if len(self._processed_event_ids) > 10_000:
            self._processed_event_ids = set(list(self._processed_event_ids)[-5_000:])

        # Filter by sender
        if MATRIX_SIGNAL_SENDER and event.sender != MATRIX_SIGNAL_SENDER:
            return

        body = str(event.body or "").strip()
        if not body:
            return

        log.debug("Message from %s: %s", event.sender, body[:200])

        sig = parse_signal(body)
        if sig is None:
            return  # Not a signal message

        log.info(
            "Signal detected | symbol=%s dir=%s entry=%s sl=%s tp=%s strategy=%s | from=%s",
            sig["symbol"], sig["signal"], sig.get("entry"),
            sig.get("sl"), sig.get("tp1", "-"), sig.get("strategy"), event.sender,
        )

        result = await asyncio.get_event_loop().run_in_executor(
            None, self._executor.execute, sig
        )

        reply = self._format_reply(sig, result)
        log.info("Execution result: %s", result.get("message"))

        if MATRIX_POST_REPLY:
            await self._send_message(reply, reply_to=event.event_id)

    def _format_reply(self, sig: dict, result: dict) -> str:
        symbol    = sig["symbol"]
        direction = sig["signal"].upper()
        status    = "✅ ACCEPTED" if result["ok"] else "❌ REJECTED"
        lines = [
            f"{status} | {symbol} {direction}",
            f"Entry: {sig.get('entry', '?')}  SL: {sig.get('sl', '?')}  TP: {sig.get('tp1', '-')}",
        ]
        if result["ok"]:
            lines.append(f"Qty: {result.get('qty')}  Notional: {result.get('notional')}")
            lines.append(f"Order ID: {result.get('order_id')}")
        else:
            lines.append(f"Reason: {result.get('message')}")
        return "\n".join(lines)

    async def _send_message(self, body: str, *, reply_to: Optional[str] = None) -> None:
        content: dict = {
            "msgtype": "m.text",
            "body": body,
        }
        if reply_to:
            content["m.relates_to"] = {
                "m.in_reply_to": {"event_id": reply_to}
            }
        try:
            await self._client.room_send(
                room_id=MATRIX_ROOM_ID,
                message_type="m.room.message",
                content=content,
            )
        except Exception as exc:
            log.warning("Failed to send Matrix reply: %s", exc)


# --- Entry point --------------------------------------------------------------

def _validate_config() -> None:
    missing = [
        name
        for name, val in [
            ("BYBIT_API_KEY",       os.environ.get("BYBIT_API_KEY")),
            ("BYBIT_API_SECRET",    os.environ.get("BYBIT_API_SECRET")),
            ("MATRIX_HOMESERVER",   MATRIX_HOMESERVER),
            ("MATRIX_ACCESS_TOKEN", MATRIX_ACCESS_TOKEN),
            # MATRIX_ROOM_ID is optional: if blank the bot handles all joined rooms
        ]
        if not val
    ]
    if missing:
        sys.exit(f"Missing required env vars: {', '.join(missing)}")


def _enforce_live_trading_confirmation() -> None:
    if not DEMO:
        if not LIVE_TRADING_CONFIRM:
            sys.exit(
                "BYBIT_DEMO=false but LIVE_TRADING_CONFIRM is not 'true'. "
                "Set LIVE_TRADING_CONFIRM=true to enable live trading."
            )
        log.warning("*** LIVE TRADING MODE — real funds at risk ***")
    else:
        log.info("Demo trading mode (BYBIT_DEMO=true)")


async def main() -> None:
    _validate_config()
    _enforce_live_trading_confirmation()

    api_key    = os.environ["BYBIT_API_KEY"]
    api_secret = os.environ["BYBIT_API_SECRET"]

    http = HTTP(testnet=False, demo=DEMO, api_key=api_key, api_secret=api_secret)

    executor = OrderExecutor(http)
    bot      = MatrixSignalBot(executor)
    await bot.start()


if __name__ == "__main__":
    asyncio.run(main())
