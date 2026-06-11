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
  MATRIX_HOMESERVER           — e.g. https://matrix.org
  MATRIX_ACCESS_TOKEN         — bot account access token
  MATRIX_ROOM_IDS             — comma-separated room IDs (!roomid:homeserver) (optional; if blank, listens to all joined rooms)
  MATRIX_RL_EXECUTION_URL     — RL sidecar /v1/signals endpoint

Optional env vars
-----------------
  NOTIONAL_PCT                — fraction of equity risked (default 0.01)
  TAKER_FEE_RATE              — estimated one-way taker fee (default 0.00055)
  MAX_FEE_TO_PRICE_RISK       — reject when fees > this fraction of SL risk (default 0.25)
  ORDER_LEVERAGE_BUFFER       — legacy setting; leverage is now based on SL distance
  MIN_STOP_DISTANCE_PCT       — minimum SL distance as fraction of entry (default 0.001)
  MAX_OPEN_POSITIONS          — max simultaneous positions (default 5)
  MATRIX_MAX_SYMBOL_POSITIONS — max simultaneous positions for one symbol (-1 disables; default -1)
  MATRIX_DEDUP_SECONDS        — ignore identical repeated signals within this window (0 disables; default 0)
  MATRIX_MAX_RISK_MULTIPLIER  — reject orders whose rounded size exceeds target risk by this multiple (default 1.05)
  MATRIX_SIGNAL_SENDER        — restrict signals to this Matrix user ID (optional)
  MATRIX_POST_REPLY           — "true" to post order confirmations back to room (default true)
    MATRIX_RL_EXECUTION_TIMEOUT_SECONDS — RL sidecar HTTP timeout (default 1.0)
    MATRIX_RL_EXECUTION_QUEUE_SIZE — queued RL dispatch items (default 1000)
  LOG_DIR                     — log directory (default /app/logs)
"""
from __future__ import annotations

import asyncio
import html
import json
import logging
import math
import os
import queue
import re
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import requests
from pybit.unified_trading import HTTP
from market_context import MarketContextEnricher


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_csv_set(name: str, *, lower: bool = True, default: str = "") -> set[str]:
    raw = os.environ.get(name, default)
    values = {item.strip() for item in raw.split(",") if item.strip()}
    return {value.lower() for value in values} if lower else values


# --- Config -------------------------------------------------------------------
DEMO = os.environ.get("BYBIT_DEMO", "true").lower() in ("1", "true", "yes")
LIVE_TRADING_CONFIRM = os.environ.get("LIVE_TRADING_CONFIRM", "false").lower() in ("1", "true", "yes")
NOTIONAL_PCT = float(os.environ.get("NOTIONAL_PCT", os.environ.get("RISK_PCT", "0.01")))
TAKER_FEE_RATE = float(os.environ.get("TAKER_FEE_RATE", "0.00055"))
MAX_FEE_TO_PRICE_RISK = float(os.environ.get("MAX_FEE_TO_PRICE_RISK", "0.25"))
# Kept for env-file compatibility; order leverage is derived from stop distance.
ORDER_LEVERAGE_BUFFER = float(os.environ.get("ORDER_LEVERAGE_BUFFER", "2.0"))
MIN_STOP_DISTANCE_PCT = float(os.environ.get("MIN_STOP_DISTANCE_PCT", "0.001"))
MAX_OPEN = int(os.environ.get("MAX_OPEN_POSITIONS", "5"))
MAX_SYMBOL_POSITIONS = int(os.environ.get("MATRIX_MAX_SYMBOL_POSITIONS", "-1"))
MATRIX_DEDUP_SECONDS = float(os.environ.get("MATRIX_DEDUP_SECONDS", "0"))
MATRIX_MAX_RISK_MULTIPLIER = float(os.environ.get("MATRIX_MAX_RISK_MULTIPLIER", "1.05"))
LOG_DIR = os.environ.get("LOG_DIR", "/app/logs")
ACTIVE_TRADES_STATE_PATH = os.environ.get(
    "ACTIVE_TRADES_STATE_PATH",
    os.path.join(LOG_DIR, "active_trades.json"),
)
RL_ENTRY_REFS_STATE_PATH = os.environ.get(
    "RL_ENTRY_REFS_STATE_PATH",
    os.path.join(LOG_DIR, "rl_entry_refs.json"),
)
RL_ENTRY_REFS_MAX_AGE_DAYS = float(os.environ.get("RL_ENTRY_REFS_MAX_AGE_DAYS", "14"))
RL_ENTRY_REFS_MAX_ENTRIES = int(os.environ.get("RL_ENTRY_REFS_MAX_ENTRIES", "5000"))

MATRIX_HOMESERVER = os.environ.get("MATRIX_HOMESERVER", "").strip()
MATRIX_ACCESS_TOKEN = os.environ.get("MATRIX_ACCESS_TOKEN", "").strip()
# Support both legacy MATRIX_ROOM_ID (single) and new MATRIX_ROOM_IDS (comma-separated list)
_matrix_room_ids_raw = (
    os.environ.get("MATRIX_ROOM_IDS", "")
    or os.environ.get("MATRIX_ROOM_ID", "")
).strip()
MATRIX_ROOM_IDS = set(r.strip() for r in _matrix_room_ids_raw.split(",") if r.strip()) if _matrix_room_ids_raw else set()
MATRIX_SIGNAL_SENDER = os.environ.get("MATRIX_SIGNAL_SENDER", "").strip()
MATRIX_POST_REPLY = os.environ.get("MATRIX_POST_REPLY", "true").lower() in ("1", "true", "yes")
MATRIX_RL_EXECUTION_URL = os.environ.get("MATRIX_RL_EXECUTION_URL", os.environ.get("RL_EXECUTION_URL", "")).strip()
MATRIX_RL_EXECUTION_TIMEOUT_SECONDS = float(
    os.environ.get(
        "MATRIX_RL_EXECUTION_TIMEOUT_SECONDS",
        os.environ.get("RL_EXECUTION_TIMEOUT_SECONDS", "1.0"),
    )
)
MATRIX_RL_EXECUTION_QUEUE_SIZE = int(
    os.environ.get(
        "MATRIX_RL_EXECUTION_QUEUE_SIZE",
        os.environ.get("RL_EXECUTION_QUEUE_SIZE", "1000"),
    )
)
MATRIX_FUNDED_EXECUTION_ENABLED = _env_bool("MATRIX_FUNDED_EXECUTION_ENABLED", False)
MATRIX_FUNDED_BYBIT_API_KEY = os.environ.get("MATRIX_FUNDED_BYBIT_API_KEY", "").strip()
MATRIX_FUNDED_BYBIT_API_SECRET = os.environ.get("MATRIX_FUNDED_BYBIT_API_SECRET", "").strip()
MATRIX_FUNDED_BYBIT_DEMO = _env_bool("MATRIX_FUNDED_BYBIT_DEMO", False)
MATRIX_FUNDED_SYMBOLS = _env_csv_set("MATRIX_FUNDED_SYMBOLS", default="XLMUSDT,ADAUSDT")
MATRIX_FUNDED_STRATEGIES = _env_csv_set("MATRIX_FUNDED_STRATEGIES", default="wolfe_channel")
MATRIX_FUNDED_STATUSES = _env_csv_set("MATRIX_FUNDED_STATUSES", default="accepted")
MATRIX_FUNDED_ROOM_IDS = _env_csv_set("MATRIX_FUNDED_ROOM_IDS", lower=False)
MATRIX_FUNDED_RISK_USDT = _env_float("MATRIX_FUNDED_RISK_USDT", 7.5)
MATRIX_FUNDED_MAX_TOTAL_OPEN_RISK_USDT = _env_float("MATRIX_FUNDED_MAX_TOTAL_OPEN_RISK_USDT", 15.0)
MATRIX_FUNDED_MAX_OPEN_POSITIONS = _env_int("MATRIX_FUNDED_MAX_OPEN_POSITIONS", 2)
MATRIX_FUNDED_MAX_SYMBOL_POSITIONS = _env_int("MATRIX_FUNDED_MAX_SYMBOL_POSITIONS", 1)
MATRIX_FUNDED_ACCOUNT_EQUITY_FLOOR = _env_float("MATRIX_FUNDED_ACCOUNT_EQUITY_FLOOR", 4500.0)
MATRIX_FUNDED_ACCOUNT_EQUITY_BUFFER_USDT = _env_float("MATRIX_FUNDED_ACCOUNT_EQUITY_BUFFER_USDT", 10.0)
MATRIX_FUNDED_ACCOUNT_TARGET_EQUITY = _env_float("MATRIX_FUNDED_ACCOUNT_TARGET_EQUITY", 5300.0)
MATRIX_FUNDED_MAX_RISK_MULTIPLIER = _env_float("MATRIX_FUNDED_MAX_RISK_MULTIPLIER", 1.05)
MATRIX_WOLFE_SETUP_TTL_SECONDS = _env_float("MATRIX_WOLFE_SETUP_TTL_SECONDS", 6 * 60 * 60)
MATRIX_OPI_COOLDOWN_SECONDS = _env_float("MATRIX_OPI_COOLDOWN_SECONDS", 30 * 60)
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
#
# Format E — wolf channel compact multiline:
#   🐺 #XRP 5m BEAR @ $1.3455
#   🌊 [entry] · T $1.3335
#   SL $1.3471 · R/R 4.01 · 300xn

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
# Wolf channel compact multiline format
_WOLFE_CHANNEL_HEADER_RE = re.compile(
    r"(?:🐺\s*)?#(?P<sym>[A-Z]{2,10})\s+\d+[smhdw]\s+(?P<dir>BULL|BEAR)\s*@\s*\$?(?P<entry>[\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)
_WOLFE_CHANNEL_TARGET_RE = re.compile(
    r"(?:🌊\s*)?(?:\[[^\]]+\]|[A-Za-z_][A-Za-z0-9_]*)?\s*[|·•\-]?\s*T\s*\$?(?P<tp>[\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)
_WOLFE_CHANNEL_SL_RE = re.compile(r"\bSL\s*\$?(?P<sl>[\d,]+(?:\.\d+)?)", re.IGNORECASE)
_WOLFE_LIFECYCLE_STAGE_RE = re.compile(r"\[(?P<stage>[A-Za-z0-9_]+)\]", re.IGNORECASE)
_WOLFE_LIFECYCLE_MARKER_RE = re.compile(
    r"#(?P<sym>[A-Z0-9]{2,10})\s+(?P<tf>\d+[smhdw])(?:\s+(?P<dir>BULL|BEAR))?",
    re.IGNORECASE,
)
_WOLFE_LIFECYCLE_HEADER_DIR_RE = re.compile(r"\]\s+(?P<dir>BULL|BEAR)\b", re.IGNORECASE)
_WOLFE_LIFECYCLE_CURRENT_RE = re.compile(
    r"\$\s*(?P<price>[\d,]+(?:\.\d+)?)\s*-\s*\d{1,2}/\d{1,2}",
    re.IGNORECASE,
)
_WOLFE_LIFECYCLE_ENTRY_ZONE_RE = re.compile(
    r"(?P<side>Long|Short)\s+Entry\s+Zone:\s*(?P<low>[\d,]+(?:\.\d+)?)\s*-\s*(?P<high>[\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)
_WOLFE_LIFECYCLE_STOP_RE = re.compile(r"\bStop:\s*(?P<sl>[\d,]+(?:\.\d+)?)", re.IGNORECASE)
_WOLFE_LIFECYCLE_SCORE_RE = re.compile(r"\((?P<score>[\d.]+)\s*/\s*8\)")
_WOLFE_LIFECYCLE_RR_RE = re.compile(r"\bR/R\s+(?P<rr>[\d.]+)", re.IGNORECASE)
_WOLFE_LIFECYCLE_ID_RE = re.compile(
    r"\b(?P<wave_id>[A-Za-z0-9]{4})\s*//\s*Regime:\s*(?P<regime>[^\n]+)",
    re.IGNORECASE,
)
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
_OPI_FULL_RE = re.compile(
    r"\bOp\S*\s*//\s*(?P<symbol>[A-Z0-9]{2,16})\s+"
    r"(?P<timeframe>\d+[smhdw])\s+(?P<direction>LONG|SHORT)\s*@\s*\$?(?P<entry>[\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)
_OPI_TARGET_RE = re.compile(r"\b(?:T|TP)\s*:?\s*\$?(?P<target>[\d,]+(?:\.\d+)?)", re.IGNORECASE)
_OPI_SL_RE = re.compile(r"\bSL\s*:?\s*\$?(?P<sl>[\d,]+(?:\.\d+)?)", re.IGNORECASE)
_OPI_SCORE_RE = re.compile(
    r"(?:\b1C\s*//.*?)?(?:\*|\bscore\b)?\s*(?P<score>[\d.]+)\s*/\s*8",
    re.IGNORECASE | re.DOTALL,
)
_OPI_NEXT_EVENT_RE = re.compile(r"\bNext\s+event:\s*(?P<event>[^\n\r]+)", re.IGNORECASE)
_OPI_TF_FLOOR_RE = re.compile(r"(?P<floor>\d+[smhdw]\s+TF-floor[^\n\r]*)", re.IGNORECASE)
_OPI_MULTI_TF_RE = re.compile(r"(?P<multi_tf>\d+[smhdw](?:/\d+[smhdw]){1,})")
_OPI_BIAS_FLIP_RE = re.compile(
    r"(?P<symbol>[A-Z0-9]{2,16})\s+(?P<timeframe>\d+[smhdw])\s*//\s*Bias\s+Flip\s*"
    r"\[(?P<bias>[^\]]+)\]\s*@\s*\$?(?P<entry>[\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)
_OPI_MARKER_PATTERN = r"(?:[\u25b2\u25b3\u25bc\u25bd]|\U0001f53a|\U0001f53b)"
_OPI_DOMINO_RE = re.compile(
    rf"(?P<marker>{_OPI_MARKER_PATTERN})?\s*"
    r"(?P<symbol>[A-Z0-9]{2,16})\s*//\s*DOMINO\s*@\s*\$?(?P<entry>[\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)
_OPI_TURN_RE = re.compile(
    r"(?P<timeframes>\d+[smhdw](?:\+\d+[smhdw])*)\s*<(?P<minutes>\d+)min\s*\[Turn\s+Detected\]",
    re.IGNORECASE,
)
_OPI_MOVE_RE = re.compile(
    rf"(?P<marker>{_OPI_MARKER_PATTERN})?\s*"
    r"(?P<symbol>[A-Z0-9]{2,16})\s*\[(?P<timeframe>\d+[smhdw])\]\s*"
    r"(?P<pct>[+-]?\d+(?:\.\d+)?)%\s+Move\s*@\s*\$?(?P<entry>[\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)
_OPI_TEXT_TRANSLATION = str.maketrans(
    {
        "\u1d3c": "O",
        "\u1d52": "o",
        "\u1d56": "p",
        "\u1da6": "i",
        "\u1d38": "L",
        "\u1d5b": "v",
        "\u02e1": "l",
        "\u1dbb": "z",
        "\u2142": "L",
        "\u2146": "d",
        "\u2124": "Z",
        "\u2605": "*",
        "\u00d7": "x",
        "\u00b7": " ",
        "\u2192": " -> ",
        "\u279c": " -> ",
        "\u27f6": " -> ",
    }
)
_OPI_FULL_CANDIDATES = {
    ("BTCUSDT", "5m"),
    ("XLMUSDT", "5m"),
    ("XAUUSDT", "5m"),
}
_OPI_STRUCTURAL_CANDIDATES = {
    ("move_alert", "XLMUSDT", "30m"),
    ("domino_turn", "XRPUSDT", "3m"),
}
_OPI_STRUCTURAL_LOOKBACK = 80
_OPI_STRUCTURAL_TARGET_R = 3.0
_OPI_KLINE_CACHE: dict[tuple[str, str, int], tuple[float, list[dict[str, float]]]] = {}
_OPI_HTTP: HTTP | None = None

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


_NON_ACTIONABLE_SIGNAL = object()


def _normalise_opi_alert_text(text: str) -> str:
    return str(text).translate(_OPI_TEXT_TRANSLATION)


def _opi_direction_from_marker(raw: str | None) -> Optional[str]:
    if raw in {"\u25b2", "\u25b3", "\U0001f53a"}:
        return "long"
    if raw in {"\u25bc", "\u25bd", "\U0001f53b"}:
        return "short"
    return None


def _opi_direction_from_nearby_marker(text: str, symbol_start: int) -> Optional[str]:
    prefix = text[max(0, int(symbol_start) - 16):int(symbol_start)]
    for char in reversed(prefix):
        direction = _opi_direction_from_marker(char)
        if direction:
            return direction
    return None


def _opi_direction_from_bias(raw: str) -> Optional[str]:
    value = str(raw or "").upper()
    if "SHORT" in value or "BEAR" in value:
        return "short"
    if "LONG" in value or "BULL" in value:
        return "long"
    return None


def _opi_split_bias_flip(raw: str) -> tuple[str, str, Optional[str]]:
    text = str(raw or "").strip()
    states = re.findall(r"(?:LEAN\s+)?(?:SHORT|LONG|BEAR|BULL)", text, flags=re.IGNORECASE)
    to_bias = states[-1] if states else text
    from_bias = states[0] if len(states) > 1 else ""
    return from_bias, to_bias, _opi_direction_from_bias(to_bias)


def _opi_levels_are_valid(direction: str, entry: float, sl: float, tp: float) -> bool:
    if entry <= 0 or sl <= 0 or tp <= 0:
        return False
    if direction == "long":
        return sl < entry < tp
    return tp < entry < sl


def _opi_fee_to_stop_risk(entry: float, sl: float) -> float:
    risk = abs(float(entry) - float(sl))
    if risk <= 0:
        return math.inf
    return TAKER_FEE_RATE * (float(entry) + float(sl)) / risk


def _opi_get_http() -> HTTP:
    global _OPI_HTTP
    if _OPI_HTTP is None:
        _OPI_HTTP = HTTP(testnet=False, demo=DEMO)
    return _OPI_HTTP


def _opi_fetch_recent_bars(symbol: str, *, interval: str = "1", limit: int = 140) -> list[dict[str, float]]:
    key = (symbol.upper(), interval, int(limit))
    now = time.time()
    cached = _OPI_KLINE_CACHE.get(key)
    if cached and now - cached[0] <= 15.0:
        return cached[1]

    resp = _opi_get_http().get_kline(category="linear", symbol=symbol.upper(), interval=interval, limit=limit)
    if not isinstance(resp, dict) or resp.get("retCode", 0) not in (0, "0"):
        raise RuntimeError(f"Bybit kline retCode={resp.get('retCode') if isinstance(resp, dict) else '?'}")
    items = resp.get("result", {}).get("list", [])
    if not isinstance(items, list):
        return []

    interval_ms = int(interval) * 60_000 if str(interval).isdigit() else 60_000
    now_ms = int(now * 1000)
    bars: list[dict[str, float]] = []
    for item in items:
        if not isinstance(item, (list, tuple)) or len(item) < 6:
            continue
        try:
            ts = int(float(item[0]))
            if ts + interval_ms > now_ms:
                continue
            bars.append(
                {
                    "ts": float(ts),
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5]),
                }
            )
        except (TypeError, ValueError):
            continue
    bars.sort(key=lambda row: row["ts"])
    _OPI_KLINE_CACHE[key] = (now, bars)
    return bars


def _with_opi_signal_controls(sig: dict, *, key: str) -> dict:
    sig["dedup_key"] = key
    sig["dedup_seconds"] = MATRIX_OPI_COOLDOWN_SECONDS
    sig["drop_duplicate"] = True
    return sig


def _build_opi_structural_signal(
    *,
    kind: str,
    symbol: str,
    timeframe: str,
    direction: str,
    entry: float,
    extra: dict[str, Any] | None = None,
) -> Optional[dict]:
    try:
        bars = _opi_fetch_recent_bars(symbol, interval="1", limit=max(_OPI_STRUCTURAL_LOOKBACK + 20, 120))
    except Exception as exc:  # noqa: BLE001 - keep live parser resilient.
        log.warning("[opi] %s %s structural level fetch failed: %s", symbol, timeframe, exc)
        return None
    if len(bars) < _OPI_STRUCTURAL_LOOKBACK:
        log.debug("[opi] %s %s skipped: only %s 1m bars", symbol, timeframe, len(bars))
        return None

    window = bars[-_OPI_STRUCTURAL_LOOKBACK:]
    if direction == "long":
        sl = min(float(row["low"]) for row in window)
        risk = entry - sl
        tp = entry + _OPI_STRUCTURAL_TARGET_R * risk
    else:
        sl = max(float(row["high"]) for row in window)
        risk = sl - entry
        tp = entry - _OPI_STRUCTURAL_TARGET_R * risk

    risk_pct = risk / entry if entry > 0 else 0.0
    if risk <= 0 or risk_pct < max(0.0002, MIN_STOP_DISTANCE_PCT):
        log.debug("[opi] %s %s skipped: risk_pct=%.5f", symbol, timeframe, risk_pct)
        return None
    fee_ratio = _opi_fee_to_stop_risk(entry, sl)
    if fee_ratio > MAX_FEE_TO_PRICE_RISK:
        log.debug("[opi] %s %s skipped: fee_to_risk=%.3f", symbol, timeframe, fee_ratio)
        return None
    if not _opi_levels_are_valid(direction, entry, sl, tp):
        return None

    sig = {
        "symbol": symbol,
        "signal": direction,
        "entry": entry,
        "sl": sl,
        "tp1": tp,
        "strategy": "opi_curl_reversal",
        "source_format": "opi_matrix_structural",
        "opi_kind": kind,
        "timeframe": timeframe,
        "structure_lookback_bars": _OPI_STRUCTURAL_LOOKBACK,
        "target_r": _OPI_STRUCTURAL_TARGET_R,
        "risk_pct": risk_pct,
        "fee_to_stop_risk": fee_ratio,
    }
    if extra:
        sig.update(extra)
    return _with_opi_signal_controls(
        sig,
        key=f"opi|opi_curl_reversal|{kind}|{symbol}|{timeframe}|{direction}",
    )


def _parse_opi_signal(clean: str) -> Optional[dict]:
    match_text = _normalise_opi_alert_text(clean)
    if not any(token in match_text for token in ("//", "DOMINO", "Bias Flip", "Move @")):
        return None

    full_m = _OPI_FULL_RE.search(match_text)
    if full_m:
        symbol = _normalise_symbol(full_m.group("symbol"))
        timeframe = full_m.group("timeframe").lower()
        if (symbol, timeframe) not in _OPI_FULL_CANDIDATES:
            return None
        direction = full_m.group("direction").lower()
        entry = _parse_price(full_m.group("entry"))
        target_m = _OPI_TARGET_RE.search(match_text)
        sl_m = _OPI_SL_RE.search(match_text)
        if not target_m or not sl_m:
            return None
        sl = _parse_price(sl_m.group("sl"))
        tp = _parse_price(target_m.group("target"))
        if not _opi_levels_are_valid(direction, entry, sl, tp):
            return None
        risk_pct = abs(entry - sl) / entry if entry > 0 else 0.0
        fee_ratio = _opi_fee_to_stop_risk(entry, sl)
        if risk_pct < max(0.0002, MIN_STOP_DISTANCE_PCT) or fee_ratio > MAX_FEE_TO_PRICE_RISK:
            return None

        sig: dict = {
            "symbol": symbol,
            "signal": direction,
            "entry": entry,
            "sl": sl,
            "tp1": tp,
            "strategy": "opi_full",
            "source_format": "opi_matrix_full",
            "opi_kind": "opi_full",
            "timeframe": timeframe,
            "risk_pct": risk_pct,
            "fee_to_stop_risk": fee_ratio,
        }
        score_m = _OPI_SCORE_RE.search(match_text)
        if score_m:
            try:
                sig["score"] = float(score_m.group("score"))
            except (TypeError, ValueError):
                pass
        floor_m = _OPI_TF_FLOOR_RE.search(match_text)
        if floor_m:
            sig["tf_floor"] = floor_m.group("floor").strip()
        multi_tf_m = _OPI_MULTI_TF_RE.search(match_text)
        if multi_tf_m:
            sig["multi_tf"] = multi_tf_m.group("multi_tf")
        event_m = _OPI_NEXT_EVENT_RE.search(match_text)
        if event_m:
            sig["next_event"] = event_m.group("event").strip()
        return _with_opi_signal_controls(
            sig,
            key=f"opi|opi_full|{symbol}|{timeframe}|{direction}",
        )

    if domino_m := _OPI_DOMINO_RE.search(match_text):
        direction = _opi_direction_from_marker(domino_m.group("marker")) or _opi_direction_from_nearby_marker(
            match_text,
            domino_m.start("symbol"),
        )
        turn_m = _OPI_TURN_RE.search(match_text)
        timeframe = turn_m.group("timeframes").split("+")[-1].lower() if turn_m else "1m"
        symbol = _normalise_symbol(domino_m.group("symbol"))
        kind = "domino_turn"
        if direction and (kind, symbol, timeframe) in _OPI_STRUCTURAL_CANDIDATES:
            return _build_opi_structural_signal(
                kind=kind,
                symbol=symbol,
                timeframe=timeframe,
                direction=direction,
                entry=_parse_price(domino_m.group("entry")),
                extra={"multi_tf": turn_m.group("timeframes") if turn_m else ""},
            )

    if move_m := _OPI_MOVE_RE.search(match_text):
        direction = _opi_direction_from_marker(move_m.group("marker")) or _opi_direction_from_nearby_marker(
            match_text,
            move_m.start("symbol"),
        )
        if not direction:
            direction = "long" if float(move_m.group("pct")) > 0 else "short"
        symbol = _normalise_symbol(move_m.group("symbol"))
        timeframe = move_m.group("timeframe").lower()
        kind = "move_alert"
        if direction and (kind, symbol, timeframe) in _OPI_STRUCTURAL_CANDIDATES:
            return _build_opi_structural_signal(
                kind=kind,
                symbol=symbol,
                timeframe=timeframe,
                direction=direction,
                entry=_parse_price(move_m.group("entry")),
                extra={"move_pct": float(move_m.group("pct"))},
            )

    if bias_m := _OPI_BIAS_FLIP_RE.search(match_text):
        from_bias, to_bias, direction = _opi_split_bias_flip(bias_m.group("bias"))
        symbol = _normalise_symbol(bias_m.group("symbol"))
        timeframe = bias_m.group("timeframe").lower()
        kind = "bias_flip"
        if direction and (kind, symbol, timeframe) in _OPI_STRUCTURAL_CANDIDATES:
            return _build_opi_structural_signal(
                kind=kind,
                symbol=symbol,
                timeframe=timeframe,
                direction=direction,
                entry=_parse_price(bias_m.group("entry")),
                extra={"from_bias": from_bias, "to_bias": to_bias},
            )

    return None


def _wolfe_direction(raw: str | None) -> Optional[str]:
    if str(raw or "").upper() == "BULL":
        return "long"
    if str(raw or "").upper() == "BEAR":
        return "short"
    return None


def _extract_wolfe_lifecycle_event(clean: str) -> Optional[dict]:
    stage_m = _WOLFE_LIFECYCLE_STAGE_RE.search(clean)
    marker_m = _WOLFE_LIFECYCLE_MARKER_RE.search(clean)
    if not stage_m or not marker_m:
        return None

    stage = stage_m.group("stage").lower()
    direction_raw = marker_m.group("dir")
    if not direction_raw:
        header_dir_m = _WOLFE_LIFECYCLE_HEADER_DIR_RE.search(clean)
        direction_raw = header_dir_m.group("dir") if header_dir_m else ""
    direction = _wolfe_direction(direction_raw)

    wave_id = None
    regime = None
    id_m = _WOLFE_LIFECYCLE_ID_RE.search(clean)
    if id_m:
        wave_id = id_m.group("wave_id")
        regime = id_m.group("regime").strip()
    else:
        trailing_id_m = re.search(r"\]\s*.*?\b([A-Za-z0-9]{4,})\s*$", clean.strip(), re.IGNORECASE)
        if trailing_id_m:
            wave_id = trailing_id_m.group(1)

    event = {
        "stage": stage,
        "symbol": _normalise_symbol(marker_m.group("sym")),
        "timeframe": marker_m.group("tf"),
        "direction": direction,
        "wave_id": wave_id,
        "strategy": "wolfe_channel",
    }

    compact_head_m = _WOLFE_CHANNEL_HEADER_RE.search(clean)
    if stage == "entry" and compact_head_m and direction:
        sig = {
            "symbol": _normalise_symbol(compact_head_m.group("sym")),
            "signal": direction,
            "entry": _parse_price(compact_head_m.group("entry")),
            "strategy": "wolfe_channel",
            "wave_stage": stage,
            "timeframe": marker_m.group("tf"),
            "source_format": "wolfe_compact",
        }
        if wave_id:
            sig["wave_id"] = wave_id
        wolfe_sl_m = _WOLFE_CHANNEL_SL_RE.search(clean)
        if wolfe_sl_m:
            sig["sl"] = _parse_price(wolfe_sl_m.group("sl"))
        wolfe_tp_m = _WOLFE_CHANNEL_TARGET_RE.search(clean)
        if wolfe_tp_m:
            sig["tp1"] = _parse_price(wolfe_tp_m.group("tp"))
        rr_m = _WOLFE_LIFECYCLE_RR_RE.search(clean)
        if rr_m:
            sig["rr"] = float(rr_m.group("rr"))
        if _is_valid_signal(sig):
            event["signal_payload"] = sig
        return event

    if stage != "bona_fide" or not direction:
        return event

    entry_zone_m = _WOLFE_LIFECYCLE_ENTRY_ZONE_RE.search(clean)
    stop_m = _WOLFE_LIFECYCLE_STOP_RE.search(clean)
    current_m = _WOLFE_LIFECYCLE_CURRENT_RE.search(clean)
    tp1_m = _TP1_RE.search(clean)
    if not (entry_zone_m and stop_m and current_m and tp1_m):
        return event

    zone_low = _parse_price(entry_zone_m.group("low"))
    zone_high = _parse_price(entry_zone_m.group("high"))
    signal_price = _parse_price(current_m.group("price"))
    stop_price = _parse_price(stop_m.group("sl"))
    raw_tps = [_parse_price(m.group(2)) for m in _TP_ALL_RE.finditer(clean)]
    if not raw_tps:
        raw_tps = [_parse_price(tp1_m.group(1))]

    if direction == "long":
        entry_candidates = [zone_low, zone_high, signal_price]
        stop_ok = lambda entry: stop_price < entry
        valid_tp = lambda entry, tp: tp > entry
    else:
        entry_candidates = [zone_high, zone_low, signal_price]
        stop_ok = lambda entry: stop_price > entry
        valid_tp = lambda entry, tp: tp < entry

    entry_ref = None
    selected_tps: list[float] = []
    for candidate in entry_candidates:
        profit_tps = [tp for tp in raw_tps if valid_tp(candidate, tp)]
        if stop_ok(candidate) and profit_tps:
            entry_ref = candidate
            selected_tps = profit_tps
            break
    if entry_ref is None:
        return event

    sig = {
        "symbol": event["symbol"],
        "signal": direction,
        "entry": entry_ref,
        "sl": stop_price,
        "tp1": selected_tps[0],
        "strategy": "wolfe_channel",
        "wave_stage": stage,
        "timeframe": marker_m.group("tf"),
        "source_format": "wolfe_lifecycle",
        "signal_price": signal_price,
        "entry_zone_low": zone_low,
        "entry_zone_high": zone_high,
    }
    if wave_id:
        sig["wave_id"] = wave_id
    if regime:
        sig["regime"] = regime
    score_m = _WOLFE_LIFECYCLE_SCORE_RE.search(clean)
    if score_m:
        sig["score"] = float(score_m.group("score"))
    rr_m = _WOLFE_LIFECYCLE_RR_RE.search(clean)
    if rr_m:
        sig["rr"] = float(rr_m.group("rr"))
    ldz_m = _LDZ_LEVEL_RE.search(clean)
    if ldz_m:
        sig["ldz_level"] = _parse_price(ldz_m.group(1))
    if len(selected_tps) > 1:
        sig["tps"] = selected_tps
    if len(raw_tps) > 1:
        sig["raw_tps"] = raw_tps
    if _is_valid_signal(sig):
        event["signal_payload"] = sig
    return event


def _parse_wolfe_lifecycle_signal(clean: str):
    event = _extract_wolfe_lifecycle_event(clean)
    if event is None:
        return None
    if event.get("stage") == "entry" and isinstance(event.get("signal_payload"), dict):
        return event["signal_payload"]
    return _NON_ACTIONABLE_SIGNAL


def parse_signal(text: str) -> Optional[dict]:
    """
    Try to extract a trading signal from a message string.
    Returns dict with keys: symbol, signal, entry, sl, tp1, strategy.
    Returns None if no valid signal detected.

    Parser priority:
      1. Preliminary signals (⚡ PRELIMINARY) → skip, return None
      2. Bandit/LDZ format A (🔔 OPEN LONG/SHORT)
      3. Bandit/LDZ format B (#SYMBOL TF | ⅂ⅆℤ)
            4. Wolf channel compact multiline (🐺 ... BULL/BEAR @ ...)
            5. wolfe_entry / wolfe_long / wolfe_short keyword lines
            6. JSON fragments
            7. Generic key=value scan
            8. Inline compact (SYMBOL LONG entry=X sl=Y)
    """
    clean = _strip_html(text).strip()

    # 1. Skip preliminary / unconfirmed signals
    if _PRELIMINARY_RE.search(clean):
        return None

    # 2 & 3 — Bandit / LDZ format (both share the same ➾ Entry Zone / ❌ Stop structure)
    opi_sig = _parse_opi_signal(clean)
    if opi_sig:
        return opi_sig

    wolfe_lifecycle_sig = _parse_wolfe_lifecycle_signal(clean)
    if wolfe_lifecycle_sig is _NON_ACTIONABLE_SIGNAL:
        return None
    if wolfe_lifecycle_sig:
        return wolfe_lifecycle_sig

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

    # 4. Wolf channel compact multiline format
    wolfe_head_m = _WOLFE_CHANNEL_HEADER_RE.search(clean)
    if wolfe_head_m:
        direction = "long" if wolfe_head_m.group("dir").upper() == "BULL" else "short"
        wcsig: dict = {
            "symbol": _normalise_symbol(wolfe_head_m.group("sym")),
            "signal": direction,
            "entry": _parse_price(wolfe_head_m.group("entry")),
            "strategy": "wolfe_channel",
        }
        wolfe_sl_m = _WOLFE_CHANNEL_SL_RE.search(clean)
        if wolfe_sl_m:
            wcsig["sl"] = _parse_price(wolfe_sl_m.group("sl"))
        wolfe_tp_m = _WOLFE_CHANNEL_TARGET_RE.search(clean)
        if wolfe_tp_m:
            wcsig["tp1"] = _parse_price(wolfe_tp_m.group("tp"))
        if _is_valid_signal(wcsig):
            return wcsig

    # 5. wolfe_entry / wolfe_long / wolfe_short keyword
    parsed_all_tps = {int(m.group(1)): _parse_price(m.group(2)) for m in _TP_ALL_RE.finditer(clean)}

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
            if len(parsed_all_tps) > 1 and "tps" not in wsig:
                wsig["tps"] = [parsed_all_tps[k] for k in sorted(parsed_all_tps)]
            return wsig

    # 6. JSON fragments
    for fragment in re.findall(r"\{[^{}]+\}", clean, re.DOTALL):
        try:
            obj = json.loads(fragment)
            sig2 = _build_from_dict(obj)
            if sig2:
                return sig2
        except json.JSONDecodeError:
            pass

    # 7. Generic key=value scan
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
        if len(parsed_all_tps) > 1 and "tps" not in kvsig:
            kvsig["tps"] = [parsed_all_tps[k] for k in sorted(parsed_all_tps)]
        return kvsig

    # 8. Inline compact: "BTCUSDT LONG entry=X sl=Y tp=Z"
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
            if len(parsed_all_tps) > 1 and "tps" not in inlsig:
                inlsig["tps"] = [parsed_all_tps[k] for k in sorted(parsed_all_tps)]
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
    for ladder_key in ("tps", "tp_levels", "take_profit_levels", "take_profits"):
        raw = obj.get(ladder_key)
        if isinstance(raw, list):
            tps = []
            for value in raw:
                try:
                    parsed = float(value)
                except (TypeError, ValueError):
                    continue
                if parsed > 0:
                    tps.append(parsed)
            if len(tps) > 1:
                sig["tps"] = tps
                break
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


# --- Exit message parser ─────────────────────────────────────────────────────

# Exit message patterns
_EXIT_TP_RE = re.compile(
    r"#([A-Z]{2,10}(?:USDT)?)\s+(LONG|SHORT)\s+TP\s+\d+",
    re.IGNORECASE,
)
_EXIT_RATCHET_RE = re.compile(
    r"Ratchet\s+armed\s+([A-Z]{2,10})\s+(LONG|SHORT|BUY|SELL)\s*[—-]\s*\[SL→BE\]",
    re.IGNORECASE,
)
# Entry price pattern: "Entry: 1234.56" or "entry 1234.56"
_ENTRY_PRICE_RE = re.compile(
    r"[Ee]ntry\s*:?\s*([\d.]+)",
)


def parse_exit_message(text: str) -> Optional[dict]:
    """
    Try to extract an exit signal from a message string.
    Returns dict with keys: action, symbol, direction, entry_price (optional), reason.
    Actions:
      - "set_sl_to_be": Set stop loss to breakeven for the specific subposition
    Returns None if no exit signal detected.
    Note: Trailing stops and stop hits are handled automatically by Bybit,
    so we don't need to take action on those messages.
    """
    clean = _strip_html(text).strip()

    # 1. TP hit message → set SL to BE
    tp_m = _EXIT_TP_RE.search(clean)
    if tp_m:
        symbol = _normalise_symbol(tp_m.group(1))
        direction = _normalise_direction(tp_m.group(2))
        entry_price = None
        entry_m = _ENTRY_PRICE_RE.search(clean)
        if entry_m:
            try:
                entry_price = float(entry_m.group(1))
            except (TypeError, ValueError):
                pass
        return {
            "action": "set_sl_to_be",
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry_price,
            "reason": "TP hit - setting SL to breakeven",
        }

    # 2. Ratchet armed message → set SL to BE
    ratchet_m = _EXIT_RATCHET_RE.search(clean)
    if ratchet_m:
        symbol = _normalise_symbol(ratchet_m.group(1))
        direction = _normalise_direction(ratchet_m.group(2))
        entry_price = None
        entry_m = _ENTRY_PRICE_RE.search(clean)
        if entry_m:
            try:
                entry_price = float(entry_m.group(1))
            except (TypeError, ValueError):
                pass
        return {
            "action": "set_sl_to_be",
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry_price,
            "reason": "Ratchet armed - setting SL to breakeven",
        }

    return None


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


def ceil_to_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    precision = max(0, round(-math.log10(step)))
    return round(math.ceil(value / step) * step, precision)


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
        "min_leverage": float(lev.get("minLeverage", "1") or "1"),
        "max_leverage": float(lev.get("maxLeverage", "1") or "1"),
        "leverage_step": float(lev.get("leverageStep", "0.01") or "0.01"),
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


def get_open_positions(http: HTTP) -> list[dict]:
    """Return non-flat USDT linear positions across one-way and hedge modes."""
    positions: list[dict] = []
    cursor = None
    while True:
        kwargs: dict = {"category": "linear", "settleCoin": "USDT", "limit": 200}
        if cursor:
            kwargs["cursor"] = cursor
        resp = http.get_positions(**kwargs)
        ret_code = int(resp.get("retCode", 0) or 0)
        if ret_code != 0:
            raise RuntimeError(
                f"get_positions failed ({ret_code}): {resp.get('retMsg', '?')}"
            )
        result = resp.get("result", {})
        for pos in result.get("list", []) or []:
            try:
                size = abs(float(pos.get("size", 0) or 0))
            except (TypeError, ValueError):
                size = 0.0
            if size <= 0:
                continue
            positions.append(pos)
        cursor = result.get("nextPageCursor")
        if not cursor:
            return positions


# --- Order executor -----------------------------------------------------------

class OrderExecutor:
    """Thin wrapper around Bybit HTTP that executes a single signal."""

    def __init__(self, http: HTTP, *, open_count_fn=None, position_snapshot_fn=None):
        self._http = http
        self._lock = threading.Lock()
        # Test hooks. Production uses get_positions directly and fails closed if
        # the account state cannot be read.
        self._open_count_fn = open_count_fn
        self._position_snapshot_fn = position_snapshot_fn
        self._info_cache: dict[str, dict] = {}

    def _get_info(self, symbol: str) -> dict:
        if symbol not in self._info_cache:
            self._info_cache[symbol] = get_instrument_info(self._http, symbol)
        return self._info_cache[symbol]

    def _fetch_open_positions(self) -> list[dict]:
        if self._position_snapshot_fn is not None:
            return list(self._position_snapshot_fn())
        return get_open_positions(self._http)

    def _position_counts(self, symbol: str) -> tuple[int, int]:
        if self._open_count_fn is not None and self._position_snapshot_fn is None:
            return int(self._open_count_fn()), 0
        positions = self._fetch_open_positions()
        symbol_positions = [
            pos for pos in positions
            if str(pos.get("symbol", "")).upper() == symbol.upper()
        ]
        return len(positions), len(symbol_positions)

    def _set_leverage(self, symbol: str, leverage: float, step: float) -> None:
        leverage_text = qty_to_str(leverage, step)
        try:
            self._http.set_leverage(
                category="linear",
                symbol=symbol,
                buyLeverage=leverage_text,
                sellLeverage=leverage_text,
            )
        except Exception as exc:
            log.debug("[%s] set_leverage skipped: %s", symbol, exc)

    def _determine_order_leverage(self, info: dict, entry: float, stop: float) -> float:
        max_leverage = max(float(info.get("max_leverage", 1.0)), 1.0)
        min_leverage = max(float(info.get("min_leverage", 1.0)), 1.0)
        leverage_step = float(info.get("leverage_step", 0.01) or 0.01)
        stop_distance_pct = abs(entry - stop) / entry if entry > 0 else 0.0
        risk_leverage = (1.0 / stop_distance_pct) if stop_distance_pct > 0 else max_leverage
        return min(max_leverage, max(min_leverage, ceil_to_step(risk_leverage, leverage_step)))

    def execute(self, sig: dict) -> dict:
        with self._lock:
            return self._execute_locked(sig)

    def _execute_locked(self, sig: dict) -> dict:
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

        try:
            open_count, symbol_count = self._position_counts(symbol)
        except Exception as exc:
            log.warning("[%s] Could not fetch open positions; rejecting signal: %s", symbol, exc)
            return {
                "ok": False,
                "message": f"Could not verify open positions: {exc}",
            }
        if open_count >= MAX_OPEN:
            return {
                "ok": False,
                "message": f"Position limit reached ({open_count}/{MAX_OPEN})",
            }
        if MAX_SYMBOL_POSITIONS >= 0 and symbol_count >= MAX_SYMBOL_POSITIONS:
            return {
                "ok": False,
                "message": (
                    f"Symbol position limit reached for {symbol} "
                    f"({symbol_count}/{MAX_SYMBOL_POSITIONS})"
                ),
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

        leverage_step = float(info.get("leverage_step", 0.01) or 0.01)
        order_lev = self._determine_order_leverage(info, entry, stop)
        risk_budget  = equity * NOTIONAL_PCT
        raw_qty      = risk_budget / (unit_risk + fee_risk_per_unit)
        margin_basis = available if available > 0 else equity
        max_qty_by_margin = (margin_basis * order_lev * 0.95) / entry
        if raw_qty > max_qty_by_margin:
            log.warning(
                "[%s] qty capped by margin: raw=%.6g max=%.6g lev=%.6gx",
                symbol, raw_qty, max_qty_by_margin, order_lev,
            )
            raw_qty = max_qty_by_margin

        q_step = info["qty_step"]
        min_q  = info["min_qty"]
        tick   = info["tick_size"]
        qty = floor_to_step(raw_qty, q_step)
        if qty < min_q:
            qty = min_q

        notional = qty * entry
        expected_price_sl_loss = qty * unit_risk
        expected_fee_loss = qty * fee_risk_per_unit
        expected_sl_loss = expected_price_sl_loss + expected_fee_loss
        if expected_sl_loss > risk_budget * MATRIX_MAX_RISK_MULTIPLIER:
            return {
                "ok": False,
                "message": (
                    f"Rounded order risk too high: {expected_sl_loss:.2f} "
                    f"> target {risk_budget:.2f}"
                ),
            }
        margin_est = notional / max(order_lev, 1.0)
        try:
            self._set_leverage(symbol, order_lev, leverage_step)
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

        # Partial TP/SL makes Bybit create protection for the actual filled
        # quantity of this entry, instead of replacing the whole position TP/SL.
        order_kwargs: dict = dict(
            category    = "linear",
            symbol      = symbol,
            side        = side,
            orderType   = "Market",
            qty         = qty_to_str(qty, q_step),
            stopLoss    = str(sl_price),
            slTriggerBy = "LastPrice",
            tpslMode    = "Partial",
            slOrderType = "Market",
            positionIdx = pos_idx,
            orderLinkId = order_link_id,
        )
        if tp_price is not None and tp_price > 0:
            order_kwargs["takeProfit"]  = str(tp_price)
            order_kwargs["tpTriggerBy"] = "LastPrice"
            order_kwargs["tpOrderType"] = "Market"

        log.info(
            "[%s] SIGNAL %s | strategy=%s entry~%.5g sl=%s tp=%s "
            "qty=%s notional=%.2f margin~%.2f lev=%sx "
            "price_risk~%.2f fees~%.2f risk_at_sl~%.2f (target=%.2f, %.2f%% equity) "
            "equity=%.2f open=%d/%d symbol_open=%d/%s",
            symbol, direction.upper(), strategy,
            entry, sl_price, tp_price or "-",
            qty_to_str(qty, q_step), notional, margin_est, qty_to_str(order_lev, leverage_step),
            expected_price_sl_loss, expected_fee_loss, expected_sl_loss, risk_budget,
            (expected_sl_loss / equity * 100.0) if equity > 0 else 0.0,
            equity,
            open_count,
            MAX_OPEN,
            symbol_count,
            "off" if MAX_SYMBOL_POSITIONS < 0 else str(MAX_SYMBOL_POSITIONS),
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
                order_link_id = f"mx_{strategy[:4]}_{symbol[:6]}_{direction[0]}_{uuid.uuid4().hex[:8]}"
                order_kwargs["orderLinkId"] = order_link_id
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
            order_link_id = f"mx_{strategy[:4]}_{symbol[:6]}_{direction[0]}_{uuid.uuid4().hex[:8]}"
            order_kwargs["orderLinkId"] = order_link_id
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
            "margin": f"{margin_est:.2f}",
            "leverage": f"{qty_to_str(order_lev, leverage_step)}x",
            "price_risk_at_sl": f"{expected_price_sl_loss:.2f}",
            "estimated_fees": f"{expected_fee_loss:.2f}",
            "risk_at_sl": f"{expected_sl_loss:.2f}",
            "risk_pct": f"{expected_sl_loss / equity:.4%}",
            "tpsl_mode": "Partial",
            "message": (
                f"Order placed: {side} {qty_to_str(qty, q_step)} {symbol} "
                f"| risk_at_sl~{expected_sl_loss:.2f} ({expected_sl_loss / equity:.2%}) "
                f"| sl={sl_price} tp={tp_price or '-'} | id={order_id}"
            ),
        }

    def set_stop_to_breakeven(self, symbol: str, entry_price: Optional[float] = None) -> dict:
        """
        Set stop loss to breakeven for one specific open subposition.
        entry_price is required to avoid modifying other entries accidentally.
        """
        with self._lock:
            return self._set_stop_to_breakeven_locked(symbol, entry_price)

    def _set_stop_to_breakeven_locked(self, symbol: str, entry_price: Optional[float] = None) -> dict:
        """Set SL to BE for open position(s)."""
        if entry_price is None or entry_price <= 0:
            return {
                "ok": False,
                "message": f"Entry price required for targeted SL->BE on {symbol}",
            }

        try:
            positions = self._fetch_open_positions()
        except Exception as exc:
            log.warning("[%s] Could not fetch positions for SL→BE: %s", symbol, exc)
            return {"ok": False, "message": f"Could not fetch positions: {exc}"}

        symbol_upper = symbol.upper()
        matching_positions = [
            pos for pos in positions
            if str(pos.get("symbol", "")).upper() == symbol_upper
        ]

        if not matching_positions:
            log.info("[%s] No open positions for SL→BE", symbol)
            return {"ok": True, "message": f"No open positions for {symbol}"}

        info = self._get_info(symbol)
        if not info:
            return {"ok": False, "message": f"Unknown symbol: {symbol}"}

        tick = info["tick_size"]
        updated_summary = []

        for pos in matching_positions:
            pos_size = float(pos.get("size", 0) or 0)
            if pos_size <= 0:
                continue

            pos_entry_price = float(pos.get("avgPrice", 0) or 0)
            if pos_entry_price <= 0:
                continue

            # Allow small tolerance for floating point comparison (0.1% tolerance)
            if abs(pos_entry_price - entry_price) / entry_price > 0.001:
                log.debug(
                    "[%s] Skipping subposition entry=%.8g (target=%.8g)",
                    symbol, pos_entry_price, entry_price,
                )
                continue

            be_sl = round_to_step(pos_entry_price, tick)
            pos_idx = int(pos.get("positionIdx", 0) or 0)

            modify_kwargs = dict(
                category="linear",
                symbol=symbol_upper,
                stopLoss=str(be_sl),
                slTriggerBy="LastPrice",
                positionIdx=pos_idx,
            )

            try:
                resp = self._http.set_trading_stop(**modify_kwargs)
                ret_code = int(resp.get("retCode", 0) or 0)
                if ret_code != 0:
                    msg = resp.get("retMsg", "?")
                    log.warning("[%s] SL→BE failed retCode=%s: %s", symbol, ret_code, msg)
                    updated_summary.append(f"{symbol} [entry={pos_entry_price}] FAILED: {msg}")
                else:
                    log.info("[%s] SL set to BE (%.5g) | entry=%.8g positionIdx=%d", symbol, be_sl, pos_entry_price, pos_idx)
                    updated_summary.append(f"{symbol} entry={pos_entry_price} SL→{be_sl}")
            except Exception as exc:
                log.error("[%s] set_trading_stop exception: %s", symbol, exc)
                updated_summary.append(f"{symbol} [entry={pos_entry_price}] ERROR: {exc}")

        if not updated_summary:
            msg = f"No matching subposition found for {symbol}"
            if entry_price:
                msg += f" at entry={entry_price}"
            log.info("[%s] %s", symbol, msg)
            return {"ok": True, "message": msg}

        return {
            "ok": True,
            "message": f"SL→BE updates: {', '.join(updated_summary)}",
            "updated_count": len(updated_summary),
        }


class FundedFixedRiskExecutor(OrderExecutor):
    """Direct fixed-risk executor for selected Matrix Wolfe signals."""

    def __init__(self) -> None:
        self.enabled = bool(MATRIX_FUNDED_EXECUTION_ENABLED)
        self.ready = bool(MATRIX_FUNDED_BYBIT_API_KEY and MATRIX_FUNDED_BYBIT_API_SECRET)
        if self.ready:
            http = HTTP(
                testnet=False,
                demo=MATRIX_FUNDED_BYBIT_DEMO,
                api_key=MATRIX_FUNDED_BYBIT_API_KEY,
                api_secret=MATRIX_FUNDED_BYBIT_API_SECRET,
            )
        else:
            http = HTTP(testnet=False, demo=MATRIX_FUNDED_BYBIT_DEMO)
        super().__init__(http)
        if self.enabled and not self.ready:
            log.error("Funded Wolfe executor enabled but MATRIX_FUNDED_BYBIT_API_KEY/SECRET are missing")
        log.info(
            "Funded Wolfe executor %s demo=%s symbols=%s strategies=%s risk=%.2f floor=%.2f buffer=%.2f target=%.2f",
            "enabled" if self.enabled else "disabled",
            MATRIX_FUNDED_BYBIT_DEMO,
            ",".join(sorted(MATRIX_FUNDED_SYMBOLS)) if MATRIX_FUNDED_SYMBOLS else "-",
            ",".join(sorted(MATRIX_FUNDED_STRATEGIES)) if MATRIX_FUNDED_STRATEGIES else "-",
            MATRIX_FUNDED_RISK_USDT,
            MATRIX_FUNDED_ACCOUNT_EQUITY_FLOOR,
            MATRIX_FUNDED_ACCOUNT_EQUITY_BUFFER_USDT,
            MATRIX_FUNDED_ACCOUNT_TARGET_EQUITY,
        )

    @staticmethod
    def matches(sig: dict, *, status: str) -> bool:
        symbol = str(sig.get("symbol") or "").strip().lower()
        strategy = str(sig.get("strategy") or "").strip().lower()
        room_id = str(sig.get("room_id") or "").strip()
        status_key = str(status or "").strip().lower()
        if MATRIX_FUNDED_SYMBOLS and symbol not in MATRIX_FUNDED_SYMBOLS:
            return False
        if MATRIX_FUNDED_STRATEGIES and strategy not in MATRIX_FUNDED_STRATEGIES:
            return False
        if MATRIX_FUNDED_STATUSES and status_key not in MATRIX_FUNDED_STATUSES:
            return False
        if MATRIX_FUNDED_ROOM_IDS and room_id not in MATRIX_FUNDED_ROOM_IDS:
            return False
        return True

    def execute_if_relevant(self, sig: dict, *, status: str) -> dict:
        if not self.matches(sig, status=status):
            return {"enabled": self.enabled, "matched": False, "executed": False, "message": "not a funded Wolfe signal"}
        if not self.enabled:
            return {"enabled": False, "matched": True, "executed": False, "message": "MATRIX_FUNDED_EXECUTION_ENABLED is false"}
        if not self.ready:
            return {"enabled": True, "matched": True, "executed": False, "message": "funded Bybit credentials missing"}
        with self._lock:
            return self._execute_funded_locked(sig)

    def _execute_funded_locked(self, sig: dict) -> dict:
        symbol = str(sig["symbol"]).upper()
        direction = str(sig["signal"]).lower()
        entry = float(sig["entry"])
        stop = float(sig["sl"])
        tp1 = float(sig.get("tp1") or 0.0)
        strategy = str(sig.get("strategy", "matrix"))
        if direction not in {"long", "short"}:
            return {"enabled": True, "matched": True, "executed": False, "message": f"invalid direction {direction}"}
        if tp1 <= 0:
            return {"enabled": True, "matched": True, "executed": False, "message": "missing TP; funded executor requires fixed TP/SL"}

        try:
            open_positions = self._fetch_open_positions()
        except Exception as exc:
            return {"enabled": True, "matched": True, "executed": False, "message": f"Could not verify funded positions: {exc}"}
        symbol_positions = [pos for pos in open_positions if str(pos.get("symbol", "")).upper() == symbol]
        open_count = len(open_positions)
        symbol_count = len(symbol_positions)
        if MATRIX_FUNDED_MAX_OPEN_POSITIONS >= 0 and open_count >= MATRIX_FUNDED_MAX_OPEN_POSITIONS:
            return {"enabled": True, "matched": True, "executed": False, "message": f"funded position limit reached ({open_count}/{MATRIX_FUNDED_MAX_OPEN_POSITIONS})"}
        if MATRIX_FUNDED_MAX_SYMBOL_POSITIONS >= 0 and symbol_count >= MATRIX_FUNDED_MAX_SYMBOL_POSITIONS:
            return {"enabled": True, "matched": True, "executed": False, "message": f"funded symbol position limit reached for {symbol} ({symbol_count}/{MATRIX_FUNDED_MAX_SYMBOL_POSITIONS})"}

        balances = get_balance_metrics(self._http)
        equity = float(balances.get("equity", 0.0))
        available = float(balances.get("available", 0.0))
        if equity <= 0:
            return {"enabled": True, "matched": True, "executed": False, "message": f"Bad funded equity: {equity}"}
        if MATRIX_FUNDED_ACCOUNT_TARGET_EQUITY > 0 and equity >= MATRIX_FUNDED_ACCOUNT_TARGET_EQUITY:
            return {"enabled": True, "matched": True, "executed": False, "message": f"funded target reached: equity {equity:.2f} >= {MATRIX_FUNDED_ACCOUNT_TARGET_EQUITY:.2f}"}

        unit_risk = abs(entry - stop)
        if unit_risk <= 0 or entry <= 0:
            return {"enabled": True, "matched": True, "executed": False, "message": "invalid entry/SL"}
        stop_distance_pct = unit_risk / entry
        if MIN_STOP_DISTANCE_PCT > 0 and stop_distance_pct < MIN_STOP_DISTANCE_PCT:
            return {"enabled": True, "matched": True, "executed": False, "message": f"funded SL too close: {stop_distance_pct:.4%} < {MIN_STOP_DISTANCE_PCT:.4%}"}
        fee_risk_per_unit = max(TAKER_FEE_RATE, 0.0) * (entry + stop)
        fee_to_price_risk = fee_risk_per_unit / unit_risk
        if MAX_FEE_TO_PRICE_RISK > 0 and fee_to_price_risk > MAX_FEE_TO_PRICE_RISK:
            return {"enabled": True, "matched": True, "executed": False, "message": f"funded fee/risk ratio too high: {fee_to_price_risk:.2%} > {MAX_FEE_TO_PRICE_RISK:.2%}"}

        open_risk_estimate = open_count * max(MATRIX_FUNDED_RISK_USDT, 0.0)
        risk_budget = max(MATRIX_FUNDED_RISK_USDT, 0.0)
        if MATRIX_FUNDED_MAX_TOTAL_OPEN_RISK_USDT > 0:
            remaining = MATRIX_FUNDED_MAX_TOTAL_OPEN_RISK_USDT - open_risk_estimate
            if remaining <= 0:
                return {"enabled": True, "matched": True, "executed": False, "message": f"funded open-risk limit reached: {open_risk_estimate:.2f}/{MATRIX_FUNDED_MAX_TOTAL_OPEN_RISK_USDT:.2f}"}
            risk_budget = min(risk_budget, remaining)
        if MATRIX_FUNDED_ACCOUNT_EQUITY_FLOOR > 0:
            floor_budget = equity - MATRIX_FUNDED_ACCOUNT_EQUITY_FLOOR - max(MATRIX_FUNDED_ACCOUNT_EQUITY_BUFFER_USDT, 0.0)
            remaining_floor_budget = floor_budget - open_risk_estimate
            if remaining_floor_budget <= 0:
                return {
                    "enabled": True,
                    "matched": True,
                    "executed": False,
                    "message": (
                        f"funded loss budget exhausted: equity={equity:.2f} "
                        f"floor={MATRIX_FUNDED_ACCOUNT_EQUITY_FLOOR:.2f} buffer={MATRIX_FUNDED_ACCOUNT_EQUITY_BUFFER_USDT:.2f}"
                    ),
                }
            risk_budget = min(risk_budget, remaining_floor_budget)
        if risk_budget <= 0:
            return {"enabled": True, "matched": True, "executed": False, "message": "funded risk budget is zero"}

        info = self._get_info(symbol)
        if not info:
            return {"enabled": True, "matched": True, "executed": False, "message": f"Unknown funded symbol: {symbol}"}
        if info.get("status") and info["status"] != "Trading":
            return {"enabled": True, "matched": True, "executed": False, "message": f"{symbol} status={info['status']}"}

        q_step = float(info["qty_step"])
        min_q = float(info["min_qty"])
        tick = float(info["tick_size"])
        leverage_step = float(info.get("leverage_step", 0.01) or 0.01)
        order_lev = self._determine_order_leverage(info, entry, stop)
        raw_qty = risk_budget / (unit_risk + fee_risk_per_unit)
        margin_basis = available if available > 0 else equity
        raw_qty = min(raw_qty, (margin_basis * order_lev * 0.95) / entry)
        qty = floor_to_step(raw_qty, q_step)
        if qty < min_q:
            return {"enabled": True, "matched": True, "executed": False, "message": f"funded risk too small for min qty: {qty_to_str(qty, q_step)} < {qty_to_str(min_q, q_step)}"}

        expected_price_sl_loss = qty * unit_risk
        expected_fee_loss = qty * fee_risk_per_unit
        expected_sl_loss = expected_price_sl_loss + expected_fee_loss
        if expected_sl_loss > risk_budget * MATRIX_FUNDED_MAX_RISK_MULTIPLIER:
            return {"enabled": True, "matched": True, "executed": False, "message": f"funded rounded risk too high: {expected_sl_loss:.2f} > target {risk_budget:.2f}"}

        self._set_leverage(symbol, order_lev, leverage_step)
        sl_price = round_to_step(stop, tick)
        tp_price = round_to_step(tp1, tick)
        side = "Buy" if direction == "long" else "Sell"
        pos_idx = 0
        order_link_id = f"fd_{strategy[:4]}_{symbol[:6]}_{direction[0]}_{uuid.uuid4().hex[:8]}"
        order_kwargs: dict = dict(
            category="linear",
            symbol=symbol,
            side=side,
            orderType="Market",
            qty=qty_to_str(qty, q_step),
            stopLoss=str(sl_price),
            takeProfit=str(tp_price),
            slTriggerBy="LastPrice",
            tpTriggerBy="LastPrice",
            tpslMode="Partial",
            slOrderType="Market",
            tpOrderType="Market",
            positionIdx=pos_idx,
            orderLinkId=order_link_id,
        )
        try:
            resp = self._http.place_order(**order_kwargs)
        except Exception as exc:
            exc_str = str(exc).lower()
            if "position idx" in exc_str or ("10001" in exc_str and "position" in exc_str):
                pos_idx = 1 if direction == "long" else 2
                order_kwargs["positionIdx"] = pos_idx
                order_link_id = f"fd_{strategy[:4]}_{symbol[:6]}_{direction[0]}_{uuid.uuid4().hex[:8]}"
                order_kwargs["orderLinkId"] = order_link_id
                resp = self._http.place_order(**order_kwargs)
            else:
                raise
        ret_code = int(resp.get("retCode", -1))
        if ret_code == 10001 and "position idx" in str(resp.get("retMsg", "")).lower():
            pos_idx = 1 if direction == "long" else 2
            order_kwargs["positionIdx"] = pos_idx
            order_link_id = f"fd_{strategy[:4]}_{symbol[:6]}_{direction[0]}_{uuid.uuid4().hex[:8]}"
            order_kwargs["orderLinkId"] = order_link_id
            resp = self._http.place_order(**order_kwargs)
            ret_code = int(resp.get("retCode", -1))
        if ret_code != 0:
            return {"enabled": True, "matched": True, "executed": False, "message": f"funded order rejected ({ret_code}): {resp.get('retMsg', '?')}"}

        order_id = resp.get("result", {}).get("orderId", "?")
        log.info(
            "[funded-wolfe] %s %s qty=%s risk~%.2f equity=%.2f sl=%s tp=%s order=%s",
            symbol,
            direction.upper(),
            qty_to_str(qty, q_step),
            expected_sl_loss,
            equity,
            sl_price,
            tp_price,
            order_id,
        )
        return {
            "enabled": True,
            "matched": True,
            "executed": True,
            "order_id": order_id,
            "link_id": order_link_id,
            "qty": qty_to_str(qty, q_step),
            "risk_at_sl": f"{expected_sl_loss:.2f}",
            "risk_budget": f"{risk_budget:.2f}",
            "equity": f"{equity:.2f}",
            "message": f"funded order placed: {side} {qty_to_str(qty, q_step)} {symbol} risk~{expected_sl_loss:.2f}",
        }


# --- RL sidecar forwarding ----------------------------------------------------

class MatrixRlSidecarClient:
    """Asynchronous dispatch client for matrix->RL sidecar forwarding."""

    def __init__(self) -> None:
        self._url = MATRIX_RL_EXECUTION_URL
        base_url = self._url.rsplit("/v1/signals", 1)[0] if "/v1/signals" in self._url else self._url.rstrip("/")
        self._decision_url = f"{base_url}/v1/decisions" if base_url else ""
        self._timeout = max(0.1, MATRIX_RL_EXECUTION_TIMEOUT_SECONDS)
        self._queue: queue.Queue[dict] | None = None
        self._dispatch_thread: threading.Thread | None = None
        self._market_http = HTTP(testnet=False, demo=DEMO)
        self._market_enricher = MarketContextEnricher(self._market_http, logger=log)

        if self._url:
            self._queue = queue.Queue(maxsize=max(1, MATRIX_RL_EXECUTION_QUEUE_SIZE))
            self._dispatch_thread = threading.Thread(
                target=self._dispatch_worker,
                daemon=True,
                name="matrix-rl-dispatch",
            )
            self._dispatch_thread.start()
            log.info("Matrix RL sidecar enabled  signal_url=%s", self._url)
        else:
            log.info("Matrix RL sidecar disabled; MATRIX_RL_EXECUTION_URL is empty")

    def _market_features_for_signal(self, sig: dict) -> dict:
        symbol = str(sig.get("symbol") or "").upper()
        direction = str(sig.get("signal") or "").lower()
        if not symbol or direction not in {"long", "short"}:
            return {"features": {}, "market_context": {}}

        shared_context = self._market_enricher.build_context(
            symbol=symbol,
            sig={
                "symbol": symbol,
                "signal": direction,
                "strategy": sig.get("strategy", "matrix"),
                "entry": sig.get("entry"),
                "sl": sig.get("sl"),
                "tp1": sig.get("tp1"),
                "entry_time": datetime.now(timezone.utc).isoformat(),
                "session": sig.get("session"),
            },
            instrument_info={},
            provenance={
                "bot_version": "matrix",
                "strategy": sig.get("strategy", "matrix"),
                "symbol": symbol,
                "timeframe": str(sig.get("timeframe") or "5"),
                "runtime": {"bybit_demo": DEMO},
                "build": {
                    "git_commit": os.environ.get("GIT_COMMIT") or os.environ.get("COMMIT_SHA"),
                    "image_tag": os.environ.get("IMAGE_TAG") or os.environ.get("DOCKER_IMAGE_TAG"),
                },
            },
        )
        derived = shared_context.get("derived") if isinstance(shared_context.get("derived"), dict) else {}

        sign = 1.0 if direction == "long" else -1.0
        features: dict[str, float] = {}
        if isinstance(derived.get("ret_1h"), (int, float)):
            features["ret_1h_dir"] = sign * float(derived["ret_1h"])
        if isinstance(derived.get("ret_4h"), (int, float)):
            features["ret_4h_dir"] = sign * float(derived["ret_4h"])
        if isinstance(derived.get("symbol_vol_mult"), (int, float)):
            features["symbol_vol_mult"] = float(derived["symbol_vol_mult"])
        return {"features": features, "market_context": shared_context}

    def _build_signal_payload(
        self,
        status: str,
        *,
        sig: dict,
        reason: str | None,
        source_event_id: str | None,
    ) -> dict:
        probability = None
        threshold = None
        market_enrichment = self._market_features_for_signal(sig)
        feature_payload = {
            "ml_probability": probability,
            "ml_threshold": threshold,
            "matrix_status_accepted": 1.0 if status == "accepted" else 0.0,
            "matrix_status_rejected": 1.0 if status == "rejected" else 0.0,
        }
        for name in ("score", "risk_pct", "fee_to_stop_risk", "target_r", "move_pct"):
            try:
                value = float(sig.get(name))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                feature_payload[f"signal.{name}"] = value
        kind = str(sig.get("opi_kind") or "").strip().lower()
        if kind:
            for candidate in ("opi_full", "move_alert", "domino_turn", "bias_flip"):
                feature_payload[f"signal.opi_kind_{candidate}"] = 1.0 if kind == candidate else 0.0
        feature_payload.update(market_enrichment.get("features") or {})
        feature_columns = list(feature_payload.keys())
        return {
            "schema_version": "rl_signal_v1",
            "event_id": str(source_event_id or uuid.uuid4().hex),
            "source": "matrix-bot",
            "sent_at": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "reason": reason,
            "symbol": sig.get("symbol"),
            "strategy": sig.get("strategy", "matrix"),
            "direction": sig.get("signal"),
            "room_id": sig.get("room_id"),
            "setup": {
                "entry": sig.get("entry"),
                "stop_loss": sig.get("sl"),
                "take_profit": sig.get("tp1"),
                "take_profit_levels": sig.get("tps"),
                "timeframe": sig.get("timeframe"),
                "source_format": sig.get("source_format"),
                "entry_time": datetime.now(timezone.utc).isoformat(),
            },
            "features": feature_payload,
            "feature_columns": feature_columns,
            "ml_probability": probability,
            "ml_threshold": threshold,
            "market_context": market_enrichment.get("market_context") or {},
            "raw_signal": sig,
            "extra": {},
        }

    def enqueue_signal(self, sig: dict, status: str, reason: str | None, source_event_id: str | None = None) -> dict:
        if not self._url or self._queue is None:
            return {
                "enabled": False,
                "queued": False,
                "message": "MATRIX_RL_EXECUTION_URL is empty",
            }
        payload = self._build_signal_payload(
            status,
            sig=sig,
            reason=reason,
            source_event_id=source_event_id,
        )
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            log.warning(
                "[matrix-rl] Dispatch queue full; dropping %s signal %s %s",
                status,
                sig.get("symbol"),
                sig.get("strategy", "matrix"),
            )
            return {
                "enabled": True,
                "queued": False,
                "message": "RL dispatch queue full",
            }
        return {
            "enabled": True,
            "queued": True,
            "message": "RL sidecar signal queued",
            "event_id": payload.get("event_id"),
        }

    def fetch_decision_by_event_id(self, event_id: str) -> Optional[dict]:
        if not self._decision_url or not event_id:
            return None
        try:
            response = requests.get(
                f"{self._decision_url}/{event_id}",
                timeout=max(1.0, self._timeout),
            )
        except Exception as exc:
            log.debug("[matrix-rl] Decision lookup failed for event=%s: %s", event_id, exc)
            return None
        if response.status_code == 404:
            return None
        if response.status_code >= 300:
            log.warning(
                "[matrix-rl] Decision lookup HTTP %s for event=%s: %s",
                response.status_code,
                event_id,
                response.text[:200],
            )
            return None
        try:
            payload = response.json()
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def _dispatch_worker(self) -> None:
        assert self._queue is not None
        while True:
            payload = self._queue.get()
            try:
                self._dispatch_signal(payload)
            except Exception as exc:
                log.warning("[matrix-rl] Dispatch failed: %s", exc)
            finally:
                self._queue.task_done()

    def _dispatch_signal(self, payload: dict) -> None:
        response = requests.post(
            self._url,
            json=payload,
            timeout=self._timeout,
        )
        if response.status_code >= 300:
            log.warning(
                "[matrix-rl] Sidecar returned HTTP %s: %s",
                response.status_code,
                response.text[:300],
            )
            return
        try:
            reply = response.json()
        except Exception:
            reply = {}

        log.info(
            "[matrix-rl] Dispatched %s signal %s %s decision=%s status=%s action=%s",
            payload.get("status"),
            payload.get("symbol"),
            payload.get("strategy"),
            reply.get("decision_id", "-") if isinstance(reply, dict) else "-",
            reply.get("execution_status", "-") if isinstance(reply, dict) else "-",
            (
                f"{float(reply.get('action')):.3f}"
                if isinstance(reply, dict) and reply.get("action") is not None
                else "-"
            ),
        )


# --- Matrix client (nio) ------------------------------------------------------

try:
    from nio import AsyncClient, InviteMemberEvent, MatrixRoom, RoomMessageText
except ImportError:
    sys.exit(
        "matrix-nio is not installed. "
        "Run: pip install matrix-nio  (or add it to requirements.txt)"
    )


class MatrixSignalBot:
    def __init__(self, rl_client: MatrixRlSidecarClient):
        self._rl_client = rl_client
        self._client = AsyncClient(MATRIX_HOMESERVER)
        self._client.access_token = MATRIX_ACCESS_TOKEN
        self._client.user_id = None  # will be populated by whoami()
        self._processed_event_ids: set[str] = set()
        self._recent_signal_keys: dict[str, float] = {}
        self._wolfe_lifecycle_setups: dict[str, dict] = {}
        self._rl_entry_refs: dict[str, dict] = {}
        self._rl_entry_refs_dirty = False
        self._rl_entry_refs_last_save_at = 0.0
        # Order executor for managing positions (entry, close, SL modifications)
        self._order_executor = OrderExecutor(HTTP(testnet=False, demo=DEMO))
        self._funded_executor = FundedFixedRiskExecutor()
        self._load_rl_entry_refs_state()

    @staticmethod
    def _normalise_rl_entry_ref(raw: object) -> Optional[dict]:
        if not isinstance(raw, dict):
            return None
        event_id = str(raw.get("event_id") or "").strip()
        if not event_id:
            return None
        entry_price_raw = raw.get("entry_price")
        entry_price = None
        if entry_price_raw is not None:
            try:
                entry_price = float(entry_price_raw)
            except (TypeError, ValueError):
                entry_price = None
        return {
            "event_id": event_id,
            "decision_id": raw.get("decision_id"),
            "order_id": raw.get("order_id"),
            "order_link_id": raw.get("order_link_id"),
            "execution_status": raw.get("execution_status"),
            "symbol": str(raw.get("symbol") or "").upper() or None,
            "direction": str(raw.get("direction") or "").lower() or None,
            "entry_price": entry_price,
            "updated_at": str(raw.get("updated_at") or ""),
        }

    @staticmethod
    def _parse_ref_updated_at(value: object) -> Optional[datetime]:
        raw = str(value or "").strip()
        if not raw:
            return None
        if raw.endswith("Z"):
            raw = f"{raw[:-1]}+00:00"
        try:
            dt = datetime.fromisoformat(raw)
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def _prune_rl_entry_refs(self) -> bool:
        if not self._rl_entry_refs:
            return False

        changed = False
        now_utc = datetime.now(timezone.utc)
        max_age_days = max(0.0, RL_ENTRY_REFS_MAX_AGE_DAYS)
        max_entries = max(1, RL_ENTRY_REFS_MAX_ENTRIES)

        if max_age_days > 0:
            cutoff = now_utc.timestamp() - (max_age_days * 86400.0)
            for event_id, ref in list(self._rl_entry_refs.items()):
                updated_dt = self._parse_ref_updated_at(ref.get("updated_at"))
                updated_ts = updated_dt.timestamp() if updated_dt else 0.0
                if updated_ts <= 0 or updated_ts < cutoff:
                    self._rl_entry_refs.pop(event_id, None)
                    changed = True

        if len(self._rl_entry_refs) > max_entries:
            sorted_ids = sorted(
                self._rl_entry_refs,
                key=lambda key: (
                    self._parse_ref_updated_at(self._rl_entry_refs.get(key, {}).get("updated_at"))
                    or datetime.fromtimestamp(0, tz=timezone.utc)
                ),
                reverse=True,
            )
            keep = set(sorted_ids[:max_entries])
            for event_id in list(self._rl_entry_refs.keys()):
                if event_id not in keep:
                    self._rl_entry_refs.pop(event_id, None)
                    changed = True

        return changed

    def _save_rl_entry_refs_state(self, *, force: bool = False) -> None:
        if self._prune_rl_entry_refs():
            self._rl_entry_refs_dirty = True

        if not self._rl_entry_refs_dirty and not force:
            return
        now = time.time()
        if not force and (now - self._rl_entry_refs_last_save_at) < 1.0:
            return

        rows = sorted(
            self._rl_entry_refs.values(),
            key=lambda row: str(row.get("updated_at") or ""),
            reverse=True,
        )
        payload = {
            "version": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "entries": rows,
        }

        try:
            tmp_path = f"{RL_ENTRY_REFS_STATE_PATH}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=True, indent=2)
            os.replace(tmp_path, RL_ENTRY_REFS_STATE_PATH)
            self._rl_entry_refs_dirty = False
            self._rl_entry_refs_last_save_at = now
        except Exception as exc:
            log.warning("Failed saving RL entry ref state (%s): %s", RL_ENTRY_REFS_STATE_PATH, exc)

    def _load_rl_entry_refs_state(self) -> None:
        if not RL_ENTRY_REFS_STATE_PATH:
            return
        if not os.path.exists(RL_ENTRY_REFS_STATE_PATH):
            return
        try:
            with open(RL_ENTRY_REFS_STATE_PATH, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception as exc:
            log.warning("Failed loading RL entry ref state (%s): %s", RL_ENTRY_REFS_STATE_PATH, exc)
            return

        raw_entries = payload.get("entries") if isinstance(payload, dict) else None
        if not isinstance(raw_entries, list):
            return

        restored: dict[str, dict] = {}
        for raw in raw_entries:
            item = self._normalise_rl_entry_ref(raw)
            if item is None:
                continue
            restored[str(item["event_id"])] = item

        if restored:
            self._rl_entry_refs = restored
            if self._prune_rl_entry_refs():
                self._rl_entry_refs_dirty = True
                self._save_rl_entry_refs_state(force=True)
            log.info("Restored %d RL entry refs from %s", len(restored), RL_ENTRY_REFS_STATE_PATH)

    def _remember_rl_entry_ref(self, ref: dict) -> None:
        event_id = str(ref.get("event_id") or "")
        if not event_id:
            return
        normalised = self._normalise_rl_entry_ref(ref)
        if normalised is None:
            return
        self._rl_entry_refs[event_id] = normalised
        self._rl_entry_refs_dirty = True
        if self._prune_rl_entry_refs():
            self._rl_entry_refs_dirty = True
        self._save_rl_entry_refs_state()

    def _find_recent_entry_ref(self, symbol: str, direction: Optional[str]) -> Optional[dict]:
        symbol_u = str(symbol or "").upper()
        dir_l = str(direction or "").lower() if direction else ""
        matches = [
            ref
            for ref in self._rl_entry_refs.values()
            if str(ref.get("symbol") or "").upper() == symbol_u
            and (not dir_l or str(ref.get("direction") or "").lower() == dir_l)
        ]
        if not matches:
            return None
        return sorted(matches, key=lambda row: str(row.get("updated_at") or ""), reverse=True)[0]

    async def _sync_rl_decision_ref(self, *, event_id: str, sig: dict) -> None:
        for _ in range(15):
            decision = await asyncio.get_event_loop().run_in_executor(
                None,
                self._rl_client.fetch_decision_by_event_id,
                event_id,
            )
            if isinstance(decision, dict):
                resolved_entry = decision.get("setup_entry")
                if resolved_entry is None:
                    resolved_entry = sig.get("entry")
                ref = {
                    "event_id": event_id,
                    "decision_id": decision.get("decision_id"),
                    "order_id": decision.get("order_id"),
                    "order_link_id": decision.get("order_link_id"),
                    "execution_status": decision.get("execution_status"),
                    "symbol": sig.get("symbol"),
                    "direction": sig.get("signal"),
                    "entry_price": resolved_entry,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                self._remember_rl_entry_ref(ref)
                if str(decision.get("execution_status") or "").lower() in {
                    "executed", "failed", "skipped", "queue_full"
                }:
                    return
            await asyncio.sleep(2)

    @staticmethod
    def _signal_key(sig: dict) -> str:
        if sig.get("dedup_key"):
            return str(sig["dedup_key"])

        def fmt(value: object) -> str:
            try:
                return f"{float(value):.8g}"
            except (TypeError, ValueError):
                return str(value)
        return "|".join(
            [
                str(sig.get("symbol", "")).upper(),
                str(sig.get("signal", "")).lower(),
                fmt(sig.get("entry")),
                fmt(sig.get("sl")),
                fmt(sig.get("tp1", "")),
            ]
        )

    def _claim_signal(self, sig: dict) -> tuple[bool, float]:
        try:
            dedup_seconds = float(sig.get("dedup_seconds", MATRIX_DEDUP_SECONDS))
        except (TypeError, ValueError):
            dedup_seconds = MATRIX_DEDUP_SECONDS
        if dedup_seconds <= 0:
            return True, 0.0
        now = time.time()
        expired = [key for key, expires_at in self._recent_signal_keys.items() if expires_at <= now]
        for key in expired:
            self._recent_signal_keys.pop(key, None)
        key = self._signal_key(sig)
        expires_at = self._recent_signal_keys.get(key, 0.0)
        if expires_at > now:
            return False, expires_at - now
        self._recent_signal_keys[key] = now + dedup_seconds
        return True, 0.0

    @staticmethod
    def _wolfe_lifecycle_key(event: dict) -> str:
        return "|".join(
            [
                str(event.get("wave_id") or "").lower(),
                str(event.get("symbol") or "").upper(),
                str(event.get("timeframe") or "").lower(),
                str(event.get("direction") or "").lower(),
            ]
        )

    def _prune_wolfe_lifecycle_setups(self) -> None:
        if MATRIX_WOLFE_SETUP_TTL_SECONDS <= 0:
            return
        cutoff = time.time() - MATRIX_WOLFE_SETUP_TTL_SECONDS
        expired = [
            key
            for key, row in self._wolfe_lifecycle_setups.items()
            if float(row.get("updated_at") or 0.0) < cutoff
        ]
        for key in expired:
            self._wolfe_lifecycle_setups.pop(key, None)

    def _pop_wolfe_lifecycle_setup(self, event: dict) -> Optional[dict]:
        key = self._wolfe_lifecycle_key(event)
        if key in self._wolfe_lifecycle_setups:
            return self._wolfe_lifecycle_setups.pop(key, None)
        wave_id = str(event.get("wave_id") or "").lower()
        if not wave_id:
            return None
        matches = [
            (key, row)
            for key, row in self._wolfe_lifecycle_setups.items()
            if str(row.get("wave_id") or "").lower() == wave_id
        ]
        if not matches:
            return None
        key, row = sorted(matches, key=lambda item: float(item[1].get("updated_at") or 0.0), reverse=True)[0]
        self._wolfe_lifecycle_setups.pop(key, None)
        return row

    def _handle_wolfe_lifecycle_message(self, body: str, *, room_id: str) -> Optional[dict]:
        event = _extract_wolfe_lifecycle_event(_strip_html(body).strip())
        if not event:
            return None

        self._prune_wolfe_lifecycle_setups()
        stage = str(event.get("stage") or "").lower()
        wave_id = str(event.get("wave_id") or "")
        payload = event.get("signal_payload")

        if stage == "bona_fide":
            if isinstance(payload, dict):
                key = self._wolfe_lifecycle_key(event)
                self._wolfe_lifecycle_setups[key] = {
                    "wave_id": wave_id,
                    "symbol": event.get("symbol"),
                    "timeframe": event.get("timeframe"),
                    "direction": event.get("direction"),
                    "signal_payload": payload,
                    "updated_at": time.time(),
                }
                log.info(
                    "Wolfe setup cached | id=%s symbol=%s tf=%s dir=%s entry_ref=%s sl=%s tp=%s",
                    wave_id or "-",
                    payload.get("symbol"),
                    payload.get("timeframe"),
                    payload.get("signal"),
                    payload.get("entry"),
                    payload.get("sl"),
                    payload.get("tp1"),
                )
            return {"handled": True, "signal": None, "event": event}

        if stage == "entry":
            if isinstance(payload, dict):
                payload = dict(payload)
                payload["room_id"] = room_id
                return {"handled": True, "signal": payload, "event": event}
            cached = self._pop_wolfe_lifecycle_setup(event)
            if cached and isinstance(cached.get("signal_payload"), dict):
                sig = dict(cached["signal_payload"])
                sig["wave_stage"] = "entry"
                sig["source_format"] = "wolfe_lifecycle_entry"
                sig["room_id"] = room_id
                log.info(
                    "Wolfe entry trigger matched | id=%s symbol=%s tf=%s dir=%s",
                    wave_id or "-",
                    sig.get("symbol"),
                    sig.get("timeframe"),
                    sig.get("signal"),
                )
                return {"handled": True, "signal": sig, "event": event}
            log.info(
                "Wolfe entry trigger ignored; no cached setup | id=%s symbol=%s tf=%s dir=%s",
                wave_id or "-",
                event.get("symbol"),
                event.get("timeframe"),
                event.get("direction"),
            )
            return {"handled": True, "signal": None, "event": event}

        if stage in {"canceled", "stop_out", "target_hit"}:
            removed = self._pop_wolfe_lifecycle_setup(event)
            log.info(
                "Wolfe lifecycle %s | id=%s symbol=%s removed_cached=%s",
                stage,
                wave_id or "-",
                event.get("symbol"),
                bool(removed),
            )
            return {"handled": True, "signal": None, "event": event}

        return {"handled": True, "signal": None, "event": event}

    async def start(self) -> None:
        room_list = ", ".join(sorted(MATRIX_ROOM_IDS)) if MATRIX_ROOM_IDS else "(all joined rooms)"
        log.info(
            "Matrix bot starting | homeserver=%s rooms=%s sender_filter=%s",
            MATRIX_HOMESERVER,
            room_list,
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
        try:
            while True:
                try:
                    await self._client.sync(timeout=30_000)
                except Exception as exc:
                    log.warning("Sync error: %s — retrying in 5s", exc)
                    await asyncio.sleep(5)
        finally:
            self._save_rl_entry_refs_state(force=True)

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
        # Filter by configured rooms if set; otherwise accept any joined room
        if MATRIX_ROOM_IDS and room.room_id not in MATRIX_ROOM_IDS:
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

        # 1. Wolfe channel lifecycle messages are stateful: [bona_fide] defines
        # levels, [entry] triggers execution, and outcome states must not trade.
        wolfe_lifecycle = self._handle_wolfe_lifecycle_message(body, room_id=room.room_id)
        if wolfe_lifecycle is not None:
            sig = wolfe_lifecycle.get("signal")
            if sig is None:
                return
        else:
            sig = parse_signal(body)
        if sig is not None:
            # Add room metadata to signal for RL feature vector
            sig["room_id"] = room.room_id

            log.info(
                "Signal detected | symbol=%s dir=%s entry=%s sl=%s tp=%s strategy=%s | from=%s room=%s",
                sig["symbol"], sig["signal"], sig.get("entry"),
                sig.get("sl"), sig.get("tp1", "-"), sig.get("strategy"), event.sender, room.room_id,
            )

            claimed, wait_seconds = self._claim_signal(sig)
            dispatch_result: dict
            if not claimed:
                reason = f"Duplicate signal ignored for {wait_seconds:.0f}s"
                if bool(sig.get("drop_duplicate")):
                    log.info("%s | symbol=%s strategy=%s", reason, sig.get("symbol"), sig.get("strategy"))
                    return
                dispatch_result = self._rl_client.enqueue_signal(
                    sig,
                    status="rejected",
                    reason=reason,
                    source_event_id=event.event_id,
                )
                result = {
                    "forwarded": bool(dispatch_result.get("queued")),
                    "status": "rejected",
                    "message": reason,
                    "dispatch": dispatch_result,
                }
            else:
                try:
                    dispatch_result = await asyncio.get_event_loop().run_in_executor(
                        None,
                        self._rl_client.enqueue_signal,
                        sig,
                        "accepted",
                        None,
                        event.event_id,
                    )
                    result = {
                        "forwarded": bool(dispatch_result.get("queued")),
                        "status": "accepted",
                        "message": str(dispatch_result.get("message") or ""),
                        "dispatch": dispatch_result,
                    }
                except Exception as exc:
                    log.exception("RL forward failed")
                    result = {
                        "forwarded": False,
                        "status": "accepted",
                        "message": f"RL forward failed: {exc}",
                        "dispatch": {"enabled": True, "queued": False, "message": str(exc)},
                    }
                funded_result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._funded_executor.execute_if_relevant(sig, status="accepted"),
                )
                result["funded"] = funded_result
                if funded_result.get("matched"):
                    log.info(
                        "Funded Wolfe result | symbol=%s strategy=%s executed=%s message=%s",
                        sig.get("symbol"),
                        sig.get("strategy"),
                        funded_result.get("executed"),
                        funded_result.get("message"),
                    )

            reply = self._format_reply(sig, result)
            log.info("Forward result: %s", result.get("message"))

            if claimed and bool(dispatch_result.get("queued")):
                asyncio.create_task(self._sync_rl_decision_ref(event_id=event.event_id, sig=sig))

            if MATRIX_POST_REPLY:
                await self._send_message(reply, room_id=room.room_id, reply_to=event.event_id)
            return

        # 2. Try to parse as exit signal (if not an entry signal)
        exit_sig = parse_exit_message(body)
        if exit_sig is not None:
            action = exit_sig["action"]
            symbol = exit_sig["symbol"]
            direction = exit_sig.get("direction")
            reason = exit_sig.get("reason", "Exit signal")
            entry_price = exit_sig.get("entry_price")

            if entry_price is None:
                ref = self._find_recent_entry_ref(symbol, direction)
                if ref and ref.get("entry_price"):
                    entry_price = float(ref["entry_price"])
                    log.info(
                        "[%s] Exit matched via RL ref event=%s decision=%s order=%s link=%s entry=%s",
                        symbol,
                        ref.get("event_id"),
                        ref.get("decision_id"),
                        ref.get("order_id"),
                        ref.get("order_link_id"),
                        entry_price,
                    )

            log.info("Exit signal detected | action=%s symbol=%s entry_price=%s reason=%s | from=%s", action, symbol, entry_price, reason, event.sender)

            if action == "set_sl_to_be":
                if entry_price is None:
                    reply_text = (
                        f"⚠️ {reason}\n{symbol}\n"
                        "Skipped: no entry reference found for a specific subposition"
                    )
                    if MATRIX_POST_REPLY:
                        await self._send_message(reply_text, room_id=room.room_id, reply_to=event.event_id)
                    return

                # Set SL to breakeven for the specific subposition only.
                exit_result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._order_executor.set_stop_to_breakeven,
                    symbol,
                    entry_price,
                )
                message = exit_result.get("message", "")
                log.info("[%s] SL→BE result: %s", symbol, message)
                reply_text = f"✅ {reason}\n{symbol}\n{message}"

            if MATRIX_POST_REPLY:
                await self._send_message(reply_text, room_id=room.room_id, reply_to=event.event_id)
            return

    def _format_reply(self, sig: dict, result: dict) -> str:
        symbol    = sig["symbol"]
        direction = sig["signal"].upper()
        forwarded = bool(result.get("forwarded"))
        upstream_status = str(result.get("status") or "accepted").upper()
        status = "✅ FORWARDED" if forwarded else "⚠️ NOT FORWARDED"
        lines = [
            f"{status} | {symbol} {direction} ({upstream_status})",
            f"Entry: {sig.get('entry', '?')}  SL: {sig.get('sl', '?')}  TP: {sig.get('tp1', '-')}",
        ]
        dispatch = result.get("dispatch") if isinstance(result.get("dispatch"), dict) else {}
        rl_state = "queued" if dispatch.get("queued") else "not queued"
        if dispatch.get("enabled"):
            lines.append(f"RL sidecar: {rl_state}")
        funded = result.get("funded") if isinstance(result.get("funded"), dict) else {}
        if funded.get("matched"):
            funded_state = "executed" if funded.get("executed") else "not executed"
            lines.append(f"Funded Wolfe: {funded_state}")
            lines.append(f"Funded message: {funded.get('message')}")
        lines.append(f"Message: {result.get('message')}")
        return "\n".join(lines)

    async def _send_message(self, body: str, *, room_id: str, reply_to: Optional[str] = None) -> None:
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
                room_id=room_id,
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
            ("MATRIX_HOMESERVER",   MATRIX_HOMESERVER),
            ("MATRIX_ACCESS_TOKEN", MATRIX_ACCESS_TOKEN),
            ("MATRIX_RL_EXECUTION_URL", MATRIX_RL_EXECUTION_URL),
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

    rl_client = MatrixRlSidecarClient()
    bot      = MatrixSignalBot(rl_client)
    await bot.start()


if __name__ == "__main__":
    asyncio.run(main())
