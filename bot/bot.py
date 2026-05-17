#!/usr/bin/env python3
"""
Million Moves V4.3 — Bybit Live Trading Bot
============================================
Listens to closed 15-minute candles via Bybit WebSocket (linear perpetuals),
detects Supertrend + EMA200 + SMA13 signals, and places a market order.

Exit logic
----------
  1. Hard stop-loss  on the order itself
  2. Trailing stop   distance = trail_mult x ATR(14), activating at TP1 price

Decision Tree filter
--------------------
  For coins where `use_dt=true` in configs, a pre-trained sklearn
  DecisionTreeClassifier is loaded from MODELS_DIR at startup.
  Signals with predicted probability < dt_threshold are skipped.
  Run `train_dt.py` to (re)generate model files.

Env variables
-------------
  BYBIT_API_KEY, BYBIT_API_SECRET  -- credentials
  BYBIT_DEMO           -- "true" (default) = demo endpoint, "false" = live
  LIVE_TRADING_CONFIRM -- must be "true" when BYBIT_DEMO=false, otherwise the
                          bot exits before opening REST/order clients
  BYBIT_PUBLIC_WS_DEMO -- "false" (default) uses normal public market-data WS;
                          demo public WS currently returns 404 for linear klines
  NOTIONAL_PCT         -- target fraction of equity lost if initial SL is hit
                          (default 0.01; legacy RISK_PCT env is accepted)
  MIN_STOP_DISTANCE_PCT -- reject signals whose initial SL distance is smaller
                          than this fraction of entry (default 0.001 = 0.10%)
  TAKER_FEE_RATE       -- estimated one-way taker fee for fee-aware sizing
                          (default 0.00055 = 0.055%)
  MAX_FEE_TO_PRICE_RISK -- reject when estimated round-trip fee exceeds this
                          fraction of raw stop risk (default 0.25)
  ORDER_LEVERAGE_BUFFER -- dynamic order leverage buffer over notional/equity
                          (default 2.0). Lower leverage can increase Bybit risk-tier
                          position limits, so orders no longer force max leverage.
  WS_STALE_SECONDS      -- exit for Docker restart if no kline WS message arrives
                          for this many seconds (default 900; 0 disables)
  MAX_OPEN_POSITIONS   -- max simultaneous positions     (default 5)
  MAX_DAILY_DD         -- max allowed daily drawdown fraction (default 0.05 = 5%)
                          measured as (equity_now - equity_day_start) / equity_day_start
                          closes tracked positions and blocks new entries until next UTC day
  MAX_WEEKLY_DD        -- optional weekly drawdown fraction (default 0 = disabled);
                          blocks until the next UTC ISO week when breached
  DD_CLOSE_POSITIONS_ON_BREACH -- "true" (default) closes tracked positions on DD breach
  MAX_CLUSTER_A        -- max simultaneous positions in the correlated cluster (default 3)
                          Research (16 months, 19 coins) shows 18 of 19 coins form one
                          densely correlated cluster (return corr >= 0.60 across 149/171
                          pairs). TRXUSDT is the only independent coin and is excluded from
                          this limit. Set to MAX_OPEN to disable the cluster guard.
  Leverage caps are fetched per symbol from Bybit. Before each order the bot sets
  leverage dynamically from the intended notional instead of forcing max leverage.
  CONFIGS_PATH         -- path to top20_configs.json
  MODELS_DIR           -- directory containing <SYMBOL>_dt.pkl files
  ALLOW_MM_WITHOUT_DT  -- "true" allows use_dt symbols to fall back to trail-only
                          when their DT file is missing (default false)
  TELEGRAM_BOT_TOKEN   -- optional Telegram bot token for signal notifications
  TELEGRAM_ACCEPTED_SIGNALS_CHAT_ID -- chat/channel ID for accepted signals;
                          forum topic syntax is supported as <chat_id>_<thread_id>
  TELEGRAM_REJECTED_SIGNALS_CHAT_ID -- chat/channel ID for rejected signals;
                          forum topic syntax is supported as <chat_id>_<thread_id>
  TELEGRAM_ADMIN_CHAT_ID -- optional admin chat/topic for warnings and reconciliation
  ENABLE_PRIVATE_ORDER_WS -- "true" (default) subscribes to private order fills
  BYBIT_PRIVATE_WS_DEMO -- demo flag for private order WS (default follows BYBIT_DEMO)
  TELEGRAM_NOTIFY_ENTRY_FILLS -- "true" also sends entry fill messages
                          (default false; accepted signals are already sent)
  ENABLE_SESSION_ORB   -- enable Session ORB/Judas/FVG strategy (default false)
  SESSION_ORB_BLOCK_WEEKEND_SESSIONS -- comma-separated ORB sessions to skip
                          on Saturday/Sunday UTC (default asia,london)
  SESSION_ORB_SYMBOLS  -- comma-separated symbols for Session ORB models
  SESSION_ORB_MODELS_DIR -- directory containing <symbol>_session_orb.joblib files
  SESSION_ORB_THRESHOLD -- fixed ML threshold override (default 0.50)
  LOG_DIR              -- directory for bot.log
  ACTIVE_TRADES_STATE_PATH -- optional JSON path for persisted open-trade metadata
  TRADE_LEDGER_PATH    -- optional JSONL path for signal/fill/risk events
  HEARTBEAT_STATE_PATH -- optional JSON path for daily heartbeat state
  OPEN_ORDER_AUDIT_SECONDS -- interval for admin-only open-order sanity checks
  MAX_DAILY_LOSSES_PER_STRATEGY -- strategy kill switch; 0 disables
  MAX_DAILY_LOSSES_PER_SYMBOL -- symbol kill switch; 0 disables
  MAX_CONSECUTIVE_LOSSES_PER_STRATEGY -- strategy streak kill switch; 0 disables
  MAX_CONSECUTIVE_LOSSES_PER_SYMBOL -- symbol streak kill switch; 0 disables
"""
from __future__ import annotations

import json
import logging
import math
import os
import pickle
import signal
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timedelta, timezone
from html import escape
from typing import Optional

import numpy as np
import requests
from pybit.unified_trading import HTTP, WebSocket

from indicators import (
    ST_MULT, ST_ATR_LEN, EMA_LEN, SMA_LEN, ATR_LEN,
    VOL_WIN, ATR_PCTILE_WIN, ATR_LO, ATR_HI, VOL_THR,
    FEATURE_NAMES,
    ind_atr, ind_ema, ind_sma, ind_supertrend,
    atr_pctile_last, vol_ratio_last,
    extract_live_feature_vector,
)
from turtle_soup import (
    TURTLE_INTERVAL,
    TurtleSoupEngine,
    TurtleSoupState,
    fetch_warmup_bars_interval,
    load_turtle_models,
    parse_symbol_list,
)
from session_orb import (
    DEFAULT_SESSION_ORB_SYMBOLS,
    SESSION_ORB_INTERVAL,
    SessionOrbEngine,
    SessionOrbState,
    load_session_orb_models,
)

# --- Configuration ------------------------------------------------------------
# Credentials are read inside Bot.__init__ so this module can be imported
# by train_dt.py without BYBIT_API_KEY being set.
DEMO         = os.environ.get("BYBIT_DEMO", "true").lower() in ("1", "true", "yes")
LIVE_TRADING_CONFIRM = os.environ.get("LIVE_TRADING_CONFIRM", "false").lower() in ("1", "true", "yes")
PUBLIC_WS_DEMO = os.environ.get("BYBIT_PUBLIC_WS_DEMO", "false").lower() in ("1", "true", "yes")
NOTIONAL_PCT = float(os.environ.get("NOTIONAL_PCT", os.environ.get("RISK_PCT", "0.01")))
MIN_STOP_DISTANCE_PCT = float(os.environ.get("MIN_STOP_DISTANCE_PCT", "0.001"))
TAKER_FEE_RATE = float(os.environ.get("TAKER_FEE_RATE", "0.00055"))
MAX_FEE_TO_PRICE_RISK = float(os.environ.get("MAX_FEE_TO_PRICE_RISK", "0.25"))
ORDER_LEVERAGE_BUFFER = float(os.environ.get("ORDER_LEVERAGE_BUFFER", "2.0"))
WS_STALE_SECONDS = int(os.environ.get("WS_STALE_SECONDS", "900"))
MAX_OPEN     = int(os.environ.get("MAX_OPEN_POSITIONS", "5"))
MAX_DAILY_DD = float(os.environ.get("MAX_DAILY_DD", "0.05"))
MAX_WEEKLY_DD = float(os.environ.get("MAX_WEEKLY_DD", "0.0"))

# Correlation cluster (from research/correlation_research.py, 16 months of 15m data):
# 18 of 19 coins form a single densely correlated cluster (149/171 pairs at corr >= 0.60).
# TRXUSDT is the only independent coin and is NOT included in this set.
CLUSTER_A: frozenset[str] = frozenset({
    "AAVEUSDT", "ADAUSDT",   "ARBUSDT",    "BNBUSDT",   "CAKEUSDT",
    "ETCUSDT",  "ETHUSDT",   "GRTUSDT",    "JUPUSDT",   "LINKUSDT",
    "NEARUSDT", "RENDERUSDT","STXUSDT",    "TIAUSDT",   "WIFUSDT",
    "WLDUSDT",  "XLMUSDT",   "XRPUSDT",
})
# Max simultaneous open positions inside CLUSTER_A. Configurable so it can be
# loosened or tightened without code change. Default 3 reserves one slot for TRX
# while keeping total at MAX_OPEN=5.
CLUSTER_A_MAX = int(os.environ.get("MAX_CLUSTER_A", "3"))

CONFIGS_PATH = os.environ.get("CONFIGS_PATH", "/app/configs/top20_configs.json")
MODELS_DIR   = os.environ.get("MODELS_DIR",   "/app/models")
LOG_DIR      = os.environ.get("LOG_DIR",       "/app/logs")
RISK_STATE_PATH = os.environ.get("RISK_STATE_PATH", os.path.join(LOG_DIR, "risk_state.json"))
ACTIVE_TRADES_STATE_PATH = os.environ.get(
    "ACTIVE_TRADES_STATE_PATH",
    os.path.join(LOG_DIR, "active_trades.json"),
)
TRADE_LEDGER_PATH = os.environ.get("TRADE_LEDGER_PATH", os.path.join(LOG_DIR, "trade_ledger.jsonl"))
HEARTBEAT_STATE_PATH = os.environ.get("HEARTBEAT_STATE_PATH", os.path.join(LOG_DIR, "heartbeat_state.json"))
PROTECTION_AUDIT_SECONDS = int(os.environ.get("PROTECTION_AUDIT_SECONDS", "300"))
OPEN_ORDER_AUDIT_SECONDS = int(os.environ.get("OPEN_ORDER_AUDIT_SECONDS", "300"))
DAILY_HEARTBEAT_UTC_HOUR = int(os.environ.get("DAILY_HEARTBEAT_UTC_HOUR", "0"))
DAILY_HEARTBEAT_UTC_MINUTE = int(os.environ.get("DAILY_HEARTBEAT_UTC_MINUTE", "5"))
MAX_DAILY_LOSSES_PER_STRATEGY = int(os.environ.get("MAX_DAILY_LOSSES_PER_STRATEGY", "0"))
MAX_DAILY_LOSSES_PER_SYMBOL = int(os.environ.get("MAX_DAILY_LOSSES_PER_SYMBOL", "0"))
MAX_CONSECUTIVE_LOSSES_PER_STRATEGY = int(os.environ.get("MAX_CONSECUTIVE_LOSSES_PER_STRATEGY", "0"))
MAX_CONSECUTIVE_LOSSES_PER_SYMBOL = int(os.environ.get("MAX_CONSECUTIVE_LOSSES_PER_SYMBOL", "0"))
DD_CLOSE_POSITIONS_ON_BREACH = os.environ.get(
    "DD_CLOSE_POSITIONS_ON_BREACH", "true"
).lower() in ("1", "true", "yes")
ALLOW_MM_WITHOUT_DT = os.environ.get("ALLOW_MM_WITHOUT_DT", "false").lower() in ("1", "true", "yes")

ENABLE_TURTLE_SOUP = os.environ.get("ENABLE_TURTLE_SOUP", "true").lower() in ("1", "true", "yes")
TURTLE_SYMBOLS = parse_symbol_list(os.environ.get("TURTLE_SYMBOLS"))
TURTLE_MODELS_DIR = os.environ.get("TURTLE_MODELS_DIR", "/app/turtle_models")
TURTLE_LEADERBOARD_PATH = os.environ.get("TURTLE_LEADERBOARD_PATH", "/app/turtle_leaderboard.csv")
TURTLE_WARMUP_BARS = int(os.environ.get("TURTLE_WARMUP_BARS", "20000"))

ENABLE_SESSION_ORB = os.environ.get("ENABLE_SESSION_ORB", "false").lower() in ("1", "true", "yes")
SESSION_ORB_SYMBOLS = parse_symbol_list(os.environ.get("SESSION_ORB_SYMBOLS") or DEFAULT_SESSION_ORB_SYMBOLS)
SESSION_ORB_MODELS_DIR = os.environ.get("SESSION_ORB_MODELS_DIR", "/app/session_orb_models")
SESSION_ORB_THRESHOLD_RAW = os.environ.get("SESSION_ORB_THRESHOLD", "0.50").strip()
SESSION_ORB_THRESHOLD = float(SESSION_ORB_THRESHOLD_RAW) if SESSION_ORB_THRESHOLD_RAW else 0.50
SESSION_ORB_WARMUP_BARS = int(os.environ.get("SESSION_ORB_WARMUP_BARS", "20000"))
SESSION_ORB_BLOCK_WEEKEND_SESSIONS = {
    chunk.strip().lower()
    for chunk in os.environ.get("SESSION_ORB_BLOCK_WEEKEND_SESSIONS", "asia,london").split(",")
    if chunk.strip()
}

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_ACCEPTED_SIGNALS_CHAT_ID = os.environ.get("TELEGRAM_ACCEPTED_SIGNALS_CHAT_ID", "").strip()
TELEGRAM_REJECTED_SIGNALS_CHAT_ID = os.environ.get("TELEGRAM_REJECTED_SIGNALS_CHAT_ID", "").strip()
TELEGRAM_ADMIN_CHAT_ID = os.environ.get("TELEGRAM_ADMIN_CHAT_ID", "").strip()
ENABLE_PRIVATE_ORDER_WS = os.environ.get("ENABLE_PRIVATE_ORDER_WS", "true").lower() in ("1", "true", "yes")
PRIVATE_WS_DEMO = os.environ.get("BYBIT_PRIVATE_WS_DEMO", str(DEMO).lower()).lower() in ("1", "true", "yes")
TELEGRAM_NOTIFY_ENTRY_FILLS = os.environ.get("TELEGRAM_NOTIFY_ENTRY_FILLS", "false").lower() in ("1", "true", "yes")
PRIVATE_POSITION_ENTRY_DEBOUNCE_SECONDS = float(os.environ.get("PRIVATE_POSITION_ENTRY_DEBOUNCE_SECONDS", "10"))

TIMEFRAME   = "15"
WARMUP_BARS = 500

# --- Logging ------------------------------------------------------------------
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, "bot.log")),
    ],
)
log = logging.getLogger("mm")


# --- Signal detection ---------------------------------------------------------

def detect_signal(bars: list[dict], cfg: dict) -> Optional[dict]:
    """
    Evaluate the last closed bar for an entry signal.
    Returns dict {signal, entry, sl, tp1, trail_dist, atr} or None.
    """
    n = len(bars)
    if n < max(EMA_LEN + 5, ATR_PCTILE_WIN + 5, SMA_LEN + 5):
        return None

    o  = np.array([b["open"]   for b in bars], dtype=np.float64)
    h  = np.array([b["high"]   for b in bars], dtype=np.float64)
    l  = np.array([b["low"]    for b in bars], dtype=np.float64)
    c  = np.array([b["close"]  for b in bars], dtype=np.float64)
    v  = np.array([b["volume"] for b in bars], dtype=np.float64)

    atr_st  = ind_atr(h, l, c, ST_ATR_LEN)
    atr14   = ind_atr(h, l, c, ATR_LEN)
    ema200  = ind_ema(c, EMA_LEN)
    sma13   = ind_sma(c, SMA_LEN)

    atr_p = atr_pctile_last(atr14, ATR_PCTILE_WIN)
    if not (ATR_LO < atr_p < ATR_HI):
        return None
    if vol_ratio_last(v, VOL_WIN) < VOL_THR:
        return None

    st = ind_supertrend(o, c, atr_st, ST_MULT)

    i = n - 1
    if any(np.isnan(x) for x in (st[i], st[i-1], c[i], c[i-1],
                                   ema200[i], ema200[i-1], sma13[i])):
        return None

    pc, ps = c[i-1], st[i-1]
    cc, cs = c[i],   st[i]
    above  = (pc > ema200[i-1]) and (cc > ema200[i])
    co = (pc < ps) and (cc > cs)
    cu = (pc > ps) and (cc < cs)

    sl_m = cfg["sl"]; tp1_r = cfg["tp1"]; trail_m = cfg["trail"]

    if co and (cc >= sma13[i]) and above:
        atr  = float(atr14[i])
        sl_p = l[i] - atr * sl_m
        risk = max(cc - sl_p, 1e-10)
        return {"signal": "long",  "entry": cc, "sl": sl_p,
                "tp1": cc + tp1_r * risk, "trail_dist": trail_m * atr, "atr": atr}

    if cu and (cc <= sma13[i]) and (not above):
        atr  = float(atr14[i])
        sl_p = h[i] + atr * sl_m
        risk = max(sl_p - cc, 1e-10)
        return {"signal": "short", "entry": cc, "sl": sl_p,
                "tp1": cc - tp1_r * risk, "trail_dist": trail_m * atr, "atr": atr}

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


def fetch_warmup_bars(http_client: HTTP, symbol: str, n: int = WARMUP_BARS) -> list[dict]:
    all_bars: list[dict] = []
    end_ts: Optional[int] = None
    while len(all_bars) < n:
        kwargs: dict = dict(category="linear", symbol=symbol, interval=TIMEFRAME, limit=200)
        if end_ts is not None:
            kwargs["end"] = end_ts
        try:
            resp  = http_client.get_kline(**kwargs)
            items = resp.get("result", {}).get("list", [])
        except Exception as exc:
            log.warning(f"  get_kline error for {symbol}: {exc}"); break
        if not items:
            break
        for it in reversed(items):
            all_bars.append({"ts": int(it[0]), "open": float(it[1]), "high": float(it[2]),
                              "low": float(it[3]), "close": float(it[4]), "volume": float(it[5])})
        end_ts = int(items[-1][0]) - 1
        if len(items) < 200:
            break
        time.sleep(0.12)

    seen: set[int] = set()
    unique: list[dict] = []
    for bar in sorted(all_bars, key=lambda x: x["ts"]):
        if bar["ts"] not in seen:
            seen.add(bar["ts"]); unique.append(bar)
    return unique[-n:]


def get_balance_metrics(http_client: HTTP) -> dict[str, float]:
    """Return account equity and available USDT balance for sizing decisions."""
    resp = http_client.get_wallet_balance(accountType="UNIFIED")
    row = resp.get("result", {}).get("list", [{}])[0]
    total_equity = float(row.get("totalEquity", 0) or 0)
    total_available = float(row.get("totalAvailableBalance", 0) or 0)

    usdt_equity = 0.0
    usdt_available = 0.0
    for coin in row.get("coin", []):
        if coin.get("coin") != "USDT":
            continue
        usdt_equity = float(coin.get("equity", 0) or 0)
        # Prefer explicit withdrawable USDT if present; otherwise account-level available balance.
        usdt_available = float(
            coin.get("availableToWithdraw")
            or coin.get("availableToBorrow")
            or 0
        )
        break

    return {
        "equity": usdt_equity if usdt_equity > 0 else total_equity,
        "available": min(usdt_available, total_available)
        if (usdt_available > 0 and total_available > 0)
        else (usdt_available if usdt_available > 0 else total_available),
    }


def get_equity(http_client: HTTP) -> float:
    return get_balance_metrics(http_client).get("equity", 0.0)


def get_instrument_info(http_client: HTTP, symbol: str) -> dict:
    resp  = http_client.get_instruments_info(category="linear", symbol=symbol)
    items = resp.get("result", {}).get("list", [])
    if not items:
        return {}
    lot   = items[0].get("lotSizeFilter", {})
    price = items[0].get("priceFilter", {})
    leverage = items[0].get("leverageFilter", {})
    min_leverage_raw = str(leverage.get("minLeverage", "1") or "1")
    max_leverage_raw = str(leverage.get("maxLeverage", "1") or "1")
    leverage_step_raw = str(leverage.get("leverageStep", "0.01") or "0.01")
    return {
        "status": str(items[0].get("status", "")),
        "qty_step":  float(lot.get("qtyStep",     "0.001")),
        "min_qty":   float(lot.get("minOrderQty", "0.001")),
        "tick_size": float(price.get("tickSize",  "0.01")),
        "min_leverage": float(min_leverage_raw),
        "min_leverage_raw": min_leverage_raw,
        "max_leverage": float(max_leverage_raw),
        "max_leverage_raw": max_leverage_raw,
        "leverage_step": float(leverage_step_raw),
        "leverage_step_raw": leverage_step_raw,
    }


def qty_to_str(value: float, step: float = 0.0) -> str:
    if step > 0:
        precision = max(0, round(-math.log10(step)))
        text = f"{value:.{precision}f}"
    else:
        # Fall back: format with enough digits then strip trailing zeros,
        # but round first to 8 significant decimals to avoid fp noise.
        text = f"{round(value, 8):.8f}".rstrip("0").rstrip(".")
    return text if text else "0"


# --- Per-symbol state ---------------------------------------------------------

class SymbolState:
    def __init__(self, symbol: str, cfg: dict, info: dict):
        self.symbol          = symbol
        self.cfg             = cfg
        self.info            = info
        self.mm_enabled      = bool(cfg.get("enable_mm", True))
        self.bars: deque     = deque(maxlen=WARMUP_BARS)
        self.in_position     = False
        self.position_side: Optional[str] = None
        self.active_trade: Optional[dict] = None
        self.pending_entry_until_ms: int = 0
        self.use_dt          = bool(cfg.get("use_dt", False))
        self.dt_model        = None
        self.dt_meta: dict[str, object] = {}
        self.dt_threshold: float = 0.55
        self._lock           = threading.Lock()

    def push_bar(self, bar: dict) -> None:
        with self._lock:
            if self.bars and self.bars[-1]["ts"] == bar["ts"]:
                self.bars[-1] = bar
            else:
                self.bars.append(bar)

    def snapshot(self) -> list[dict]:
        with self._lock:
            return list(self.bars)


class TelegramNotifier:
    @staticmethod
    def _redact(value: object, token: str = "") -> str:
        text = str(value)
        if token:
            text = text.replace(token, "<redacted>")
        return text

    @staticmethod
    def _parse_chat_target(raw: str) -> tuple[str, Optional[int]]:
        text = str(raw or "").strip()
        if not text:
            return "", None
        for separator in ("_", ":"):
            head, sep, tail = text.rpartition(separator)
            if sep and head and tail.isdigit():
                return head, int(tail)
        return text, None

    def __init__(
        self,
        *,
        token: str,
        accepted_chat_id: str,
        rejected_chat_id: str,
        admin_chat_id: str,
    ):
        self.token = token
        self.accepted_chat_id, self.accepted_thread_id = self._parse_chat_target(accepted_chat_id)
        self.rejected_chat_id, self.rejected_thread_id = self._parse_chat_target(rejected_chat_id)
        self.admin_chat_id, self.admin_thread_id = self._parse_chat_target(admin_chat_id)
        self.enabled = bool(
            token and (self.accepted_chat_id or self.rejected_chat_id or self.admin_chat_id)
        )

    @staticmethod
    def _fmt(value: object) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "-"
        if not math.isfinite(number):
            return "-"
        return f"{number:.8g}"

    @staticmethod
    def _fmt_time_ms(value: object) -> str:
        try:
            millis = int(float(value))
        except (TypeError, ValueError):
            return "-"
        if millis <= 0:
            return "-"
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(millis / 1000.0))

    def _send_to_target(self, *, chat_id: str, thread_id: Optional[int], lines: list[str]) -> None:
        if not self.enabled:
            return
        if not chat_id:
            return
        try:
            payload = {
                "chat_id": chat_id,
                "text": "\n".join(lines),
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            }
            if thread_id is not None:
                payload["message_thread_id"] = thread_id
            resp = requests.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                json=payload,
                timeout=10,
            )
            if resp.status_code >= 400:
                log.warning(
                    "[telegram] sendMessage failed status=%s: %s",
                    resp.status_code,
                    self._redact(resp.text[:200], self.token),
                )
        except Exception as exc:
            log.warning("[telegram] sendMessage failed: %s", self._redact(exc, self.token))

    def _send_lines(self, *, accepted: bool, lines: list[str]) -> None:
        chat_id = self.accepted_chat_id if accepted else self.rejected_chat_id
        thread_id = self.accepted_thread_id if accepted else self.rejected_thread_id
        self._send_to_target(chat_id=chat_id, thread_id=thread_id, lines=lines)

    def _send_admin_lines(self, lines: list[str]) -> None:
        self._send_to_target(
            chat_id=self.admin_chat_id,
            thread_id=self.admin_thread_id,
            lines=lines,
        )

    def send_signal(
        self,
        status: str,
        *,
        symbol: str,
        sig: dict,
        reason: str | None = None,
        order_id: str | None = None,
        extra: dict[str, object] | None = None,
    ) -> None:
        if not self.enabled:
            return
        if status == "accepted":
            chat_id = self.accepted_chat_id
            thread_id = self.accepted_thread_id
        else:
            chat_id = self.rejected_chat_id
            thread_id = self.rejected_thread_id
        if not chat_id:
            return

        strategy = str(sig.get("strategy", "strategy"))
        direction = str(sig.get("signal", sig.get("direction", "-"))).upper()
        target = sig.get("tp1", sig.get("take_profit", sig.get("target", sig.get("target_price"))))
        lines = [
            f"<b>[{escape(strategy.upper())}] {escape(status.upper())} SIGNAL</b>",
            f"Symbol: <code>{escape(symbol)}</code>",
            f"Direction: <b>{escape(direction)}</b>",
            f"Entry: <code>{self._fmt(sig.get('entry', sig.get('entry_price')))}</code>",
            f"Stop Loss: <code>{self._fmt(sig.get('sl', sig.get('stop_price')))}</code>",
            f"Target: <code>{self._fmt(target)}</code>",
        ]
        if "prob" in sig:
            threshold = sig.get("threshold")
            threshold_text = f" / {self._fmt(threshold)}" if threshold is not None else ""
            lines.append(f"Model probability: <code>{self._fmt(sig.get('prob'))}{threshold_text}</code>")
        if "dt_prob" in sig:
            lines.append(
                f"DT probability: <code>{self._fmt(sig.get('dt_prob'))} / "
                f"{self._fmt(sig.get('dt_threshold'))}</code>"
            )
        if "stop_distance_pct" in sig:
            try:
                lines.append(f"SL distance: <code>{100.0 * float(sig['stop_distance_pct']):.4g}%</code>")
            except (TypeError, ValueError):
                pass
        if "fee_to_price_risk" in sig:
            try:
                lines.append(f"Fee / price risk: <code>{100.0 * float(sig['fee_to_price_risk']):.4g}%</code>")
            except (TypeError, ValueError):
                pass
        if sig.get("entry_time"):
            lines.append(f"Signal time: <code>{escape(str(sig['entry_time']))}</code>")
        for key, label in (
            ("session", "Session"),
            ("or_minutes", "OR minutes"),
            ("entry_risk_atr", "Entry risk ATR"),
            ("sweep_depth_atr", "Sweep depth ATR"),
            ("variant", "Variant"),
        ):
            if key in sig:
                lines.append(f"{label}: <code>{escape(str(sig[key]))}</code>")
        if reason:
            lines.append(f"Reason: {escape(reason)}")
        if order_id:
            lines.append(f"Order ID: <code>{escape(str(order_id))}</code>")
        if extra:
            for key, value in extra.items():
                lines.append(f"{escape(str(key))}: <code>{escape(str(value))}</code>")

        self._send_lines(accepted=(status == "accepted"), lines=lines)

    def send_fill(
        self,
        *,
        symbol: str,
        event: str,
        side: str,
        price: object,
        qty: object,
        strategy: str | None = None,
        direction: str | None = None,
        order_id: str | None = None,
        stop_order_type: str | None = None,
        order_type: str | None = None,
        closed_pnl: object | None = None,
        updated_time: object | None = None,
        extra: dict[str, object] | None = None,
    ) -> None:
        if not self.enabled:
            return
        header = strategy.upper() if strategy else "ORDER_FILL"
        lines = [
            f"<b>[{escape(header)}] {escape(event.upper())}</b>",
            f"Symbol: <code>{escape(symbol)}</code>",
        ]
        if direction:
            lines.append(f"Position: <b>{escape(str(direction).upper())}</b>")
        lines.extend([
            f"Order side: <code>{escape(str(side or '-'))}</code>",
            f"Fill price: <code>{self._fmt(price)}</code>",
            f"Fill qty: <code>{self._fmt(qty)}</code>",
        ])
        if closed_pnl not in (None, "", "0", 0):
            lines.append(f"Closed PnL: <code>{self._fmt(closed_pnl)}</code>")
        if stop_order_type:
            lines.append(f"Stop order type: <code>{escape(str(stop_order_type))}</code>")
        if order_type:
            lines.append(f"Order type: <code>{escape(str(order_type))}</code>")
        if updated_time:
            lines.append(f"Fill time: <code>{escape(self._fmt_time_ms(updated_time))}</code>")
        if order_id:
            lines.append(f"Order ID: <code>{escape(str(order_id))}</code>")
        if extra:
            for key, value in extra.items():
                lines.append(f"{escape(str(key))}: <code>{escape(str(value))}</code>")
        self._send_lines(accepted=True, lines=lines)

    def send_risk_event(self, title: str, *, fields: dict[str, object] | None = None) -> None:
        if not self.enabled or not self.admin_chat_id:
            return
        lines = [f"<b>[RISK] {escape(str(title).upper())}</b>"]
        if fields:
            for key, value in fields.items():
                lines.append(f"{escape(str(key))}: <code>{escape(str(value))}</code>")
        self._send_admin_lines(lines)

    def send_admin_event(self, title: str, *, fields: dict[str, object] | None = None) -> None:
        if not self.enabled or not self.admin_chat_id:
            return
        lines = [f"<b>[ADMIN] {escape(str(title).upper())}</b>"]
        if fields:
            for key, value in fields.items():
                lines.append(f"{escape(str(key))}: <code>{escape(str(value))}</code>")
        self._send_admin_lines(lines)

    def send_daily_heartbeat(self, *, fields: dict[str, object]) -> None:
        if not self.enabled or not self.accepted_chat_id:
            return
        lines = ["<b>[DAILY HEARTBEAT]</b>"]
        for key, value in fields.items():
            lines.append(f"{escape(str(key))}: <code>{escape(str(value))}</code>")
        self._send_lines(accepted=True, lines=lines)


# --- Trading bot --------------------------------------------------------------

class Bot:

    def __init__(self):
        api_key    = os.environ["BYBIT_API_KEY"]
        api_secret = os.environ["BYBIT_API_SECRET"]
        self._telegram = TelegramNotifier(
            token=TELEGRAM_BOT_TOKEN,
            accepted_chat_id=TELEGRAM_ACCEPTED_SIGNALS_CHAT_ID,
            rejected_chat_id=TELEGRAM_REJECTED_SIGNALS_CHAT_ID,
            admin_chat_id=TELEGRAM_ADMIN_CHAT_ID,
        )

        log.info(
            f"MM V4.3 Bot starting  |  demo={DEMO}  "
            f"risk_at_sl={NOTIONAL_PCT:.1%}  min_sl_distance={MIN_STOP_DISTANCE_PCT:.3%}  "
            f"taker_fee={TAKER_FEE_RATE:.3%}  max_fee_to_risk={MAX_FEE_TO_PRICE_RISK:.1%}  "
            f"order_leverage=dynamic(buffer={ORDER_LEVERAGE_BUFFER:g})  ws_stale={WS_STALE_SECONDS}s  "
            f"max_open={MAX_OPEN}  cluster_a_max={CLUSTER_A_MAX}  "
            f"max_daily_dd={MAX_DAILY_DD:.1%}  max_weekly_dd={MAX_WEEKLY_DD:.1%}  "
            f"dd_close_positions={DD_CLOSE_POSITIONS_ON_BREACH}  private_order_ws={ENABLE_PRIVATE_ORDER_WS}"
        )
        if SESSION_ORB_BLOCK_WEEKEND_SESSIONS:
            log.info(
                "Session ORB weekend filter: blocking UTC weekend sessions "
                f"{','.join(sorted(SESSION_ORB_BLOCK_WEEKEND_SESSIONS))}"
            )
        if self._telegram.enabled:
            log.info(
                "Telegram notifications enabled  "
                f"accepted_chat={'set' if TELEGRAM_ACCEPTED_SIGNALS_CHAT_ID else 'missing'}  "
                f"rejected_chat={'set' if TELEGRAM_REJECTED_SIGNALS_CHAT_ID else 'missing'}  "
                f"admin_chat={'set' if TELEGRAM_ADMIN_CHAT_ID else 'missing'}"
            )
        else:
            log.info("Telegram notifications disabled")

        self._send_trading_mode_banner()
        self._enforce_live_trading_confirmation()

        with open(CONFIGS_PATH) as fh:
            raw_configs: dict[str, dict] = json.load(fh)
        raw_configs = {k: v for k, v in raw_configs.items() if not k.startswith("_")}
        log.info(f"Loaded {len(raw_configs)} coin configs from {CONFIGS_PATH}")

        self._http = HTTP(testnet=False, demo=DEMO,
                          api_key=api_key, api_secret=api_secret)

        self._pos_lock        = threading.Lock()
        self._open_count      = 0
        self._cluster_a_count = 0   # positions open in the 18-coin correlated cluster
        self._dd_blocked      = False   # True when daily drawdown >= MAX_DAILY_DD
        self._day_start_equity: float = 0.0
        self._day_date: str           = ""  # UTC date string "YYYY-MM-DD"
        self._week_start_equity: float = 0.0
        self._week_key: str = ""  # UTC ISO week key "YYYY-Www"
        self._dd_block_scope: str = ""
        self._dd_block_key: str = ""
        self._dd_block_reason: str = ""
        self._risk_flattened_block_id: str = ""
        self._states: dict[str, SymbolState] = {}
        self._turtle_engine: Optional[TurtleSoupEngine] = None
        self._turtle_states: dict[str, TurtleSoupState] = {}
        self._turtle_models = {}
        self._session_orb_engine: Optional[SessionOrbEngine] = None
        self._session_orb_states: dict[str, SessionOrbState] = {}
        self._session_orb_models = {}
        self._private_ws = None
        self._notified_order_fill_ids: set[str] = set()
        self._manual_exit_orders: dict[str, dict] = {}
        self._admin_warning_keys: set[str] = set()
        self._last_daily_heartbeat_key = ""
        self._last_protection_audit_ts = 0.0
        self._last_open_order_audit_ts = 0.0
        self._oi_history: dict[str, deque[dict]] = {}
        self._funding_history: dict[str, deque[dict]] = {}
        self._load_risk_state()
        self._load_heartbeat_state()

        if ENABLE_TURTLE_SOUP:
            self._turtle_models = load_turtle_models(
                symbols=TURTLE_SYMBOLS,
                models_dir=TURTLE_MODELS_DIR,
                leaderboard_path=TURTLE_LEADERBOARD_PATH,
            )
            self._turtle_engine = TurtleSoupEngine()
            log.info(
                f"Turtle Soup enabled  symbols={len(self._turtle_models)}  "
                f"warmup_5m={TURTLE_WARMUP_BARS}"
            )
        else:
            log.info("Turtle Soup disabled")

        if ENABLE_SESSION_ORB:
            self._session_orb_models = load_session_orb_models(
                symbols=SESSION_ORB_SYMBOLS,
                models_dir=SESSION_ORB_MODELS_DIR,
                threshold_override=SESSION_ORB_THRESHOLD,
            )
            self._session_orb_engine = SessionOrbEngine()
            log.info(
                f"Session ORB enabled  symbols={len(self._session_orb_models)}  "
                f"warmup_5m={SESSION_ORB_WARMUP_BARS} threshold={SESSION_ORB_THRESHOLD:.2f}"
            )
        else:
            log.info("Session ORB disabled")

        for sym in self._turtle_models:
            raw_configs.setdefault(sym, {"enable_mm": False, "sl": 2.0, "tp1": 1.0, "trail": 0.5, "use_dt": False})
        for sym in self._session_orb_models:
            raw_configs.setdefault(sym, {"enable_mm": False, "sl": 2.0, "tp1": 1.0, "trail": 0.5, "use_dt": False})

        for sym, cfg in raw_configs.items():
            info = get_instrument_info(self._http, sym)
            if not info:
                log.warning(f"  {sym}: instrument info missing -- skipping"); continue
            if info.get("status") and info.get("status") != "Trading":
                log.warning(f"  {sym}: instrument status={info.get('status')} -- skipping")
                continue

            log.info(
                f"  {sym}: leverage cap {info.get('max_leverage_raw', '?')}x "
                f"(orders set leverage dynamically)"
            )
            try:
                self._http.switch_position_mode(category="linear", symbol=sym, mode=0)
            except Exception:
                pass

            state = SymbolState(sym, cfg, info)
            self._states[sym] = state

            if state.mm_enabled:
                log.info(f"  {sym}: fetching {WARMUP_BARS} warmup 15m bars ...")
                for bar in fetch_warmup_bars(self._http, sym, WARMUP_BARS):
                    state.bars.append(bar)
                log.info(
                    f"  {sym}: {len(state.bars)} 15m bars loaded  "
                    f"sl={cfg['sl']} tp1={cfg['tp1']} trail={cfg['trail']}  "
                    f"use_dt={cfg.get('use_dt', False)}"
                )
            else:
                log.info(f"  {sym}: MM disabled; state used for shared risk/execution")

            if sym in self._turtle_models:
                turtle_state = TurtleSoupState(sym, self._turtle_models[sym], TURTLE_WARMUP_BARS)
                log.info(f"  {sym}: fetching {TURTLE_WARMUP_BARS} warmup 5m bars for Turtle Soup ...")
                for bar in fetch_warmup_bars_interval(self._http, sym, interval=TURTLE_INTERVAL, n=TURTLE_WARMUP_BARS):
                    turtle_state.bars.append(bar)
                self._turtle_states[sym] = turtle_state
                log.info(f"  {sym}: {len(turtle_state.bars)} 5m bars loaded for Turtle Soup")

            if sym in self._session_orb_models:
                orb_state = SessionOrbState(sym, self._session_orb_models[sym], SESSION_ORB_WARMUP_BARS)
                log.info(f"  {sym}: fetching {SESSION_ORB_WARMUP_BARS} warmup 5m bars for Session ORB ...")
                for bar in fetch_warmup_bars_interval(self._http, sym, interval=SESSION_ORB_INTERVAL, n=SESSION_ORB_WARMUP_BARS):
                    orb_state.bars.append(bar)
                self._session_orb_states[sym] = orb_state
                log.info(f"  {sym}: {len(orb_state.bars)} 5m bars loaded for Session ORB")

        self._load_dt_models()
        self._run_startup_config_linter(raw_configs)
        self._load_active_trade_state()
        self._sync_positions()
        self._refresh_risk_state()
        self._send_startup_reconciliation_report()
        self._audit_position_protection(force=True)
        self._audit_open_orders(force=True)
        self._maybe_send_daily_heartbeat()
        self._last_ws_message_ts = time.time()

        log.info(
            f"Opening public kline WebSocket  |  demo_ws={PUBLIC_WS_DEMO}  "
            f"(REST/order demo={DEMO})"
        )
        self._ws = WebSocket(testnet=False, demo=PUBLIC_WS_DEMO, channel_type="linear")
        symbols = [sym for sym, state in self._states.items() if state.mm_enabled]
        log.info(f"Subscribing to {len(symbols)} kline.{TIMEFRAME} MM streams ...")
        for sym in symbols:
            self._ws.kline_stream(interval=int(TIMEFRAME), symbol=sym,
                                  callback=self._on_kline)
        turtle_symbols = set(self._turtle_states.keys())
        orb_symbols = set(self._session_orb_states.keys())
        strategy5_symbols = sorted(turtle_symbols | orb_symbols)
        log.info(
            f"Subscribing to {len(strategy5_symbols)} kline.5 strategy streams "
            f"(turtle={len(turtle_symbols)}, session_orb={len(orb_symbols)}) ..."
        )
        for sym in strategy5_symbols:
            self._ws.kline_stream(interval=5, symbol=sym, callback=self._on_strategy5_kline)
        self._open_private_order_ws(api_key=api_key, api_secret=api_secret)
        log.info("Bot live. Waiting for closed candles ...")

    # -- Startup safety/admin checks -----------------------------------------

    def _send_trading_mode_banner(self) -> None:
        mode = "DEMO / DRY RUN" if DEMO else "LIVE TRADING"
        fields = {
            "Mode": mode,
            "BYBIT_DEMO": DEMO,
            "LIVE_TRADING_CONFIRM": LIVE_TRADING_CONFIRM,
            "Public WS demo": PUBLIC_WS_DEMO,
            "Private WS demo": PRIVATE_WS_DEMO,
            "Risk at SL": f"{NOTIONAL_PCT:.2%}",
            "Max open positions": MAX_OPEN,
            "Daily DD limit": f"{MAX_DAILY_DD:.2%}" if MAX_DAILY_DD > 0 else "disabled",
            "Weekly DD limit": f"{MAX_WEEKLY_DD:.2%}" if MAX_WEEKLY_DD > 0 else "disabled",
        }
        if DEMO:
            log.info(
                "[mode] DEMO / DRY RUN mode active. Orders use Bybit demo trading; "
                "live funds are not touched."
            )
            self._telegram.send_admin_event("dry-run safety mode", fields=fields)
        else:
            log.critical("[mode] LIVE TRADING mode requested. Real funds can be touched.")
            self._telegram.send_admin_event("live trading mode", fields=fields)

    def _enforce_live_trading_confirmation(self) -> None:
        if DEMO or LIVE_TRADING_CONFIRM:
            return
        reason = "BYBIT_DEMO=false requires LIVE_TRADING_CONFIRM=true"
        log.critical(f"[mode] {reason}; exiting before REST/order startup.")
        self._telegram.send_admin_event(
            "live trading blocked",
            fields={
                "Reason": reason,
                "Action": "bot exited before opening trading clients",
            },
        )
        raise SystemExit(3)

    @staticmethod
    def _summarize_items(items: list[str], *, limit: int = 12) -> str:
        if not items:
            return "-"
        head = items[:limit]
        suffix = "" if len(items) <= limit else f"; +{len(items) - limit} more"
        return "; ".join(head) + suffix

    def _run_startup_config_linter(self, raw_configs: dict[str, dict]) -> None:
        warnings: list[str] = []
        errors: list[str] = []

        def warn(message: str) -> None:
            warnings.append(message)
            log.warning(f"[config] {message}")

        def error(message: str) -> None:
            errors.append(message)
            log.error(f"[config] {message}")

        if MAX_OPEN <= 0:
            error("MAX_OPEN_POSITIONS must be greater than 0")
        if NOTIONAL_PCT <= 0:
            error("NOTIONAL_PCT must be greater than 0")
        elif NOTIONAL_PCT > 0.02:
            warn(f"NOTIONAL_PCT={NOTIONAL_PCT:.2%} is high for unattended trading")

        if not 0 <= DAILY_HEARTBEAT_UTC_HOUR <= 23:
            error("DAILY_HEARTBEAT_UTC_HOUR must be between 0 and 23")
        if not 0 <= DAILY_HEARTBEAT_UTC_MINUTE <= 59:
            error("DAILY_HEARTBEAT_UTC_MINUTE must be between 0 and 59")

        if CLUSTER_A_MAX > MAX_OPEN:
            warn("MAX_CLUSTER_A is greater than MAX_OPEN_POSITIONS; cluster guard is ineffective")
        if CLUSTER_A_MAX <= 0:
            warn("MAX_CLUSTER_A <= 0 blocks all entries in the correlated cluster")
        if MIN_STOP_DISTANCE_PCT <= 0:
            warn("MIN_STOP_DISTANCE_PCT is disabled; tiny-stop signals can reach Bybit risk limits")
        elif MIN_STOP_DISTANCE_PCT < 0.0005:
            warn(f"MIN_STOP_DISTANCE_PCT={MIN_STOP_DISTANCE_PCT:.3%} is very small")
        if MAX_DAILY_DD <= 0:
            warn("MAX_DAILY_DD is disabled")
        if not DD_CLOSE_POSITIONS_ON_BREACH:
            warn("DD_CLOSE_POSITIONS_ON_BREACH=false; drawdown breach blocks entries but leaves positions open")
        if (
            MAX_DAILY_LOSSES_PER_STRATEGY <= 0
            and MAX_DAILY_LOSSES_PER_SYMBOL <= 0
            and MAX_CONSECUTIVE_LOSSES_PER_STRATEGY <= 0
            and MAX_CONSECUTIVE_LOSSES_PER_SYMBOL <= 0
        ):
            warn("loss-count circuit breakers are disabled")

        if DEMO and PUBLIC_WS_DEMO:
            warn("BYBIT_PUBLIC_WS_DEMO=true in demo mode can 404 for linear public kline streams")
        if ENABLE_PRIVATE_ORDER_WS and PRIVATE_WS_DEMO != DEMO:
            warn("BYBIT_PRIVATE_WS_DEMO does not match BYBIT_DEMO; private fills may subscribe to the wrong account")
        if not ENABLE_PRIVATE_ORDER_WS:
            warn("ENABLE_PRIVATE_ORDER_WS=false; Telegram exit/fill notifications are disabled")
        if PROTECTION_AUDIT_SECONDS <= 0:
            warn("PROTECTION_AUDIT_SECONDS <= 0; position TP/SL protection audit is disabled")
        if OPEN_ORDER_AUDIT_SECONDS <= 0:
            warn("OPEN_ORDER_AUDIT_SECONDS <= 0; open-order auditor is disabled")

        if self._telegram.enabled:
            if not TELEGRAM_ACCEPTED_SIGNALS_CHAT_ID:
                warn("Telegram accepted signal chat is missing")
            if not TELEGRAM_REJECTED_SIGNALS_CHAT_ID:
                warn("Telegram rejected signal chat is missing")
            if not TELEGRAM_ADMIN_CHAT_ID:
                warn("Telegram admin chat is missing; admin warnings only reach logs")
        else:
            warn("Telegram is disabled")

        path_checks = [
            ("CONFIGS_PATH", CONFIGS_PATH, True),
            ("MODELS_DIR", MODELS_DIR, False),
            ("LOG_DIR", LOG_DIR, False),
        ]
        if ENABLE_TURTLE_SOUP:
            path_checks.extend([
                ("TURTLE_MODELS_DIR", TURTLE_MODELS_DIR, False),
                ("TURTLE_LEADERBOARD_PATH", TURTLE_LEADERBOARD_PATH, True),
            ])
        if ENABLE_SESSION_ORB:
            path_checks.append(("SESSION_ORB_MODELS_DIR", SESSION_ORB_MODELS_DIR, False))
        for label, path, must_be_file in path_checks:
            exists = os.path.isfile(path) if must_be_file else os.path.isdir(path)
            if not exists:
                warn(f"{label} does not exist or is not mounted: {path}")

        missing_dt = sorted(
            sym
            for sym, cfg in raw_configs.items()
            if bool(cfg.get("enable_mm", True))
            and bool(cfg.get("use_dt", False))
            and not os.path.exists(os.path.join(MODELS_DIR, f"{sym}_dt.pkl"))
        )
        if missing_dt:
            behavior = "fallback without DT" if ALLOW_MM_WITHOUT_DT else "MM disabled for those symbols"
            warn(f"missing DT models ({behavior}): {','.join(missing_dt[:20])}")

        if ENABLE_TURTLE_SOUP and not self._turtle_models:
            warn("Turtle Soup is enabled but no Turtle Soup models were loaded")
        if ENABLE_SESSION_ORB and not self._session_orb_models:
            warn("Session ORB is enabled but no Session ORB models were loaded")
        if (
            not any(state.mm_enabled for state in self._states.values())
            and not self._turtle_states
            and not self._session_orb_states
        ):
            error("no active strategy streams are configured")

        status = "ERROR" if errors else "WARN" if warnings else "PASS"
        fields = {
            "Status": status,
            "Errors": len(errors),
            "Warnings": len(warnings),
            "Tracked symbols": len(self._states),
            "MM streams": sum(1 for state in self._states.values() if state.mm_enabled),
            "Turtle models": len(self._turtle_models),
            "Session ORB models": len(self._session_orb_models),
            "Fatal": self._summarize_items(errors),
            "Warnings detail": self._summarize_items(warnings),
        }
        self._telegram.send_admin_event("config linter", fields=fields)
        self._append_ledger_event(
            "admin",
            event="config_linter",
            status=status,
            errors=errors,
            warnings=warnings,
        )
        if errors:
            raise SystemExit(4)

    # -- Private order/fill WebSocket ----------------------------------------

    def _open_private_order_ws(self, *, api_key: str, api_secret: str) -> None:
        if not ENABLE_PRIVATE_ORDER_WS:
            log.info("Private order WebSocket disabled")
            return
        try:
            log.info(
                f"Opening private order WebSocket  |  demo_ws={PRIVATE_WS_DEMO}  "
                f"entry_fill_notifications={TELEGRAM_NOTIFY_ENTRY_FILLS}"
            )
            self._private_ws = WebSocket(
                testnet=False,
                demo=PRIVATE_WS_DEMO,
                channel_type="private",
                api_key=api_key,
                api_secret=api_secret,
            )
            self._private_ws.order_stream(self._on_private_order)
            self._private_ws.position_stream(self._on_private_position)
            log.info("Private order WebSocket live (order + position topics)")
        except Exception as exc:
            self._private_ws = None
            log.warning(f"Private order WebSocket unavailable; fill Telegram disabled: {exc}")

    @staticmethod
    def _private_items(msg: dict) -> list[dict]:
        data = msg.get("data", [])
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        return []

    @staticmethod
    def _truthy_field(value: object) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes"}

    @staticmethod
    def _clean_stop_order_type(value: object) -> str:
        text = str(value or "").strip()
        if text.upper() in {"", "UNKNOWN", "NONE", "NA", "N/A"}:
            return ""
        return text

    @staticmethod
    def _classify_fill_event(order: dict, *, is_exit: bool) -> str | None:
        stop_type = Bot._clean_stop_order_type(order.get("stopOrderType"))
        descriptor = " ".join(
            str(order.get(key) or "") for key in ("stopOrderType", "createType", "orderLinkId")
        ).lower()
        if "takeprofit" in descriptor or "take_profit" in descriptor:
            return "TAKE PROFIT FILLED"
        if "stoploss" in descriptor or "stop_loss" in descriptor:
            return "STOP LOSS FILLED"
        if "trailing" in descriptor:
            return "TRAILING STOP FILLED"
        if is_exit:
            return "POSITION EXIT FILLED"
        if TELEGRAM_NOTIFY_ENTRY_FILLS:
            return "ENTRY FILLED"
        return None

    @staticmethod
    def _fill_price(order: dict) -> object:
        for key in ("avgPrice", "execPrice", "price", "triggerPrice"):
            value = order.get(key)
            if value not in (None, "", "0", 0):
                return value
        return "-"

    @staticmethod
    def _fill_qty(order: dict) -> object:
        for key in ("cumExecQty", "execQty", "qty", "orderQty"):
            value = order.get(key)
            if value not in (None, "", "0", 0):
                return value
        return "-"

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    def _release_position_slot_from_private(self, symbol: str, *, clear_active_trade: bool) -> None:
        state = self._states.get(symbol)
        if state is None:
            return
        with self._pos_lock:
            if state.in_position:
                self._open_count = max(0, self._open_count - 1)
                if symbol in CLUSTER_A:
                    self._cluster_a_count = max(0, self._cluster_a_count - 1)
            state.in_position = False
            state.position_side = None
            state.pending_entry_until_ms = 0
            if clear_active_trade:
                state.active_trade = None
                self._save_active_trade_state_locked()

    def _on_private_order(self, msg: dict) -> None:
        try:
            for order in self._private_items(msg):
                if order.get("category") not in (None, "", "linear"):
                    continue
                symbol = str(order.get("symbol", ""))
                if symbol not in self._states:
                    continue
                if str(order.get("orderStatus", "")).lower() != "filled":
                    continue

                order_id = str(order.get("orderId") or "")
                dedupe_key = order_id or (
                    f"{symbol}:{order.get('updatedTime')}:{order.get('cumExecQty')}:"
                    f"{order.get('avgPrice')}:{order.get('stopOrderType')}"
                )
                if dedupe_key in self._notified_order_fill_ids:
                    continue
                self._notified_order_fill_ids.add(dedupe_key)
                if len(self._notified_order_fill_ids) > 1000:
                    self._notified_order_fill_ids = set(list(self._notified_order_fill_ids)[-500:])

                manual_exit = self._manual_exit_orders.get(order_id) if order_id else None
                stop_type = self._clean_stop_order_type(order.get("stopOrderType"))
                is_exit = (
                    bool(manual_exit)
                    or bool(stop_type)
                    or self._truthy_field(order.get("reduceOnly"))
                    or self._truthy_field(order.get("closeOnTrigger"))
                    or any(
                        token in str(order.get("createType", "")).lower()
                        for token in ("takeprofit", "stoploss", "trailing")
                    )
                )
                event = self._classify_fill_event(order, is_exit=is_exit)
                if manual_exit:
                    event = "RISK FLATTEN EXIT FILLED"
                if event is None:
                    continue

                state = self._states.get(symbol)
                active_trade = dict(state.active_trade or {}) if state is not None else {}
                side = str(order.get("side", ""))
                direction = active_trade.get("direction")
                if not direction and is_exit:
                    direction = "long" if side == "Sell" else "short" if side == "Buy" else None
                strategy = active_trade.get("strategy") or (manual_exit or {}).get("strategy")
                exit_reason = (manual_exit or {}).get("reason") if is_exit else None

                price = self._fill_price(order)
                qty = self._fill_qty(order)
                fill_price_f = self._to_float(price)
                fill_qty_f = self._to_float(qty)
                expected_entry = self._to_float(active_trade.get("entry"))
                slippage_abs = None
                slippage_bps = None
                if fill_price_f is not None and expected_entry not in (None, 0.0):
                    slippage_abs = fill_price_f - expected_entry
                    slippage_bps = (slippage_abs / expected_entry) * 1e4
                fill_notional = (
                    fill_price_f * fill_qty_f
                    if (fill_price_f is not None and fill_qty_f is not None)
                    else None
                )
                est_fill_fee = (fill_notional * TAKER_FEE_RATE) if fill_notional is not None else None
                fill_time_ms = int(self._to_float(order.get("updatedTime")) or 0)
                opened_at_ms = int(self._to_float(active_trade.get("opened_at")) or 0)
                time_to_fill_ms = (fill_time_ms - opened_at_ms) if (fill_time_ms > 0 and opened_at_ms > 0) else None
                time_in_trade_ms = time_to_fill_ms if is_exit else None
                log.info(
                    f"[{symbol}] {event}: side={side or '-'} price={TelegramNotifier._fmt(price)} "
                    f"qty={TelegramNotifier._fmt(qty)} stop_type={stop_type or '-'} "
                    f"reason={exit_reason or '-'}"
                )
                extra = {
                    "Order link id": order.get("orderLinkId", "-"),
                    "Trigger": order.get("triggerBy", "-"),
                    "Create type": order.get("createType", "-"),
                }
                if exit_reason:
                    extra["Exit reason"] = exit_reason
                self._telegram.send_fill(
                    symbol=symbol,
                    event=event,
                    side=side,
                    price=price,
                    qty=qty,
                    strategy=strategy,
                    direction=direction,
                    order_id=order_id or None,
                    stop_order_type=stop_type or None,
                    order_type=str(order.get("orderType") or "") or None,
                    closed_pnl=order.get("closedPnl"),
                    updated_time=order.get("updatedTime"),
                    extra=extra,
                )
                self._append_ledger_event(
                    "fill",
                    symbol=symbol,
                    event=event,
                    side=side,
                    direction=direction,
                    strategy=strategy,
                    price=price,
                    qty=qty,
                    order_id=order_id or None,
                    order_link_id=order.get("orderLinkId"),
                    stop_order_type=stop_type or None,
                    order_type=order.get("orderType"),
                    closed_pnl=order.get("closedPnl"),
                    fill_time_ms=order.get("updatedTime"),
                    expected_entry=active_trade.get("entry"),
                    slippage_abs=slippage_abs,
                    slippage_bps=slippage_bps,
                    fill_notional=fill_notional,
                    estimated_fill_fee=est_fill_fee,
                    time_to_fill_ms=time_to_fill_ms,
                    time_in_trade_ms=time_in_trade_ms,
                    order_raw=order,
                    exit_reason=exit_reason,
                    create_type=order.get("createType"),
                    trigger_by=order.get("triggerBy"),
                )
                if is_exit:
                    self._release_position_slot_from_private(symbol, clear_active_trade=True)
                    if order_id:
                        self._manual_exit_orders.pop(order_id, None)
        except Exception:
            log.exception("Error in private order callback")

    def _on_private_position(self, msg: dict) -> None:
        try:
            for pos in self._private_items(msg):
                if pos.get("category") not in (None, "", "linear"):
                    continue
                symbol = str(pos.get("symbol", ""))
                state = self._states.get(symbol)
                if state is None:
                    continue
                try:
                    size = float(pos.get("size", 0) or 0)
                except (TypeError, ValueError):
                    continue
                side = str(pos.get("side", ""))
                with self._pos_lock:
                    was_open = state.in_position
                    if size > 0:
                        if not state.in_position:
                            self._open_count += 1
                            if symbol in CLUSTER_A:
                                self._cluster_a_count += 1
                        state.in_position = True
                        state.position_side = side
                        state.pending_entry_until_ms = 0
                    else:
                        now_ms = self._now_ms()
                        active_trade = state.active_trade or {}
                        opened_at = int(active_trade.get("opened_at") or 0)
                        debounce_ms = int(PRIVATE_POSITION_ENTRY_DEBOUNCE_SECONDS * 1000)
                        if state.pending_entry_until_ms and now_ms < state.pending_entry_until_ms:
                            log.debug(f"[{symbol}] Ignoring transient private flat during pending entry")
                            continue
                        if opened_at and debounce_ms > 0 and now_ms - opened_at < debounce_ms:
                            log.debug(f"[{symbol}] Ignoring transient private flat right after entry")
                            continue
                        if state.in_position:
                            self._open_count = max(0, self._open_count - 1)
                            if symbol in CLUSTER_A:
                                self._cluster_a_count = max(0, self._cluster_a_count - 1)
                        state.in_position = False
                        state.position_side = None
                        state.pending_entry_until_ms = 0
                        if state.active_trade is not None:
                            state.active_trade = None
                            self._save_active_trade_state_locked()
                if was_open and size <= 0:
                    log.info(f"[{symbol}] Private position stream: position is flat")
        except Exception:
            log.exception("Error in private position callback")

    # -- Drawdown guard --------------------------------------------------------

    @staticmethod
    def _utc_week_key(now) -> str:
        iso = now.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"

    def _dd_block_id_locked(self) -> str:
        if not self._dd_blocked:
            return ""
        return f"{self._dd_block_scope}:{self._dd_block_key}:{self._dd_block_reason}"

    def _load_risk_state(self) -> None:
        if not os.path.exists(RISK_STATE_PATH):
            return
        try:
            with open(RISK_STATE_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self._day_date = str(data.get("day_date", ""))
            self._day_start_equity = float(data.get("day_start_equity", 0.0) or 0.0)
            self._week_key = str(data.get("week_key", ""))
            self._week_start_equity = float(data.get("week_start_equity", 0.0) or 0.0)
            self._dd_blocked = bool(data.get("dd_blocked", False))
            self._dd_block_scope = str(data.get("dd_block_scope", ""))
            self._dd_block_key = str(data.get("dd_block_key", ""))
            self._dd_block_reason = str(data.get("dd_block_reason", ""))
            self._risk_flattened_block_id = str(data.get("risk_flattened_block_id", ""))
            log.info(
                f"[risk] Loaded persisted risk state from {RISK_STATE_PATH}  "
                f"day={self._day_date or '-'} week={self._week_key or '-'} "
                f"blocked={self._dd_blocked}"
            )
        except Exception as exc:
            log.warning(f"[risk] Could not load risk state {RISK_STATE_PATH}: {exc}")

    def _save_risk_state_locked(self) -> None:
        data = {
            "day_date": self._day_date,
            "day_start_equity": self._day_start_equity,
            "week_key": self._week_key,
            "week_start_equity": self._week_start_equity,
            "dd_blocked": self._dd_blocked,
            "dd_block_scope": self._dd_block_scope,
            "dd_block_key": self._dd_block_key,
            "dd_block_reason": self._dd_block_reason,
            "risk_flattened_block_id": self._risk_flattened_block_id,
        }
        try:
            os.makedirs(os.path.dirname(RISK_STATE_PATH) or ".", exist_ok=True)
            tmp_path = f"{RISK_STATE_PATH}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, sort_keys=True)
            os.replace(tmp_path, RISK_STATE_PATH)
        except Exception as exc:
            log.warning(f"[risk] Could not save risk state {RISK_STATE_PATH}: {exc}")

    def _load_active_trade_state(self) -> None:
        if not os.path.exists(ACTIVE_TRADES_STATE_PATH):
            return
        try:
            with open(ACTIVE_TRADES_STATE_PATH, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            if not isinstance(raw, dict):
                raise ValueError("active trade state must be a JSON object")
            loaded = 0
            with self._pos_lock:
                for sym, trade in raw.items():
                    state = self._states.get(str(sym))
                    if state is None or not isinstance(trade, dict):
                        continue
                    state.active_trade = dict(trade)
                    loaded += 1
            if loaded:
                log.info(
                    f"[state] Loaded {loaded} persisted active trades from "
                    f"{ACTIVE_TRADES_STATE_PATH}"
                )
        except Exception as exc:
            log.warning(f"[state] Could not load active trades {ACTIVE_TRADES_STATE_PATH}: {exc}")

    def _save_active_trade_state_locked(self) -> None:
        data = {
            sym: dict(state.active_trade)
            for sym, state in self._states.items()
            if state.active_trade
        }
        try:
            os.makedirs(os.path.dirname(ACTIVE_TRADES_STATE_PATH) or ".", exist_ok=True)
            tmp_path = f"{ACTIVE_TRADES_STATE_PATH}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, sort_keys=True)
            os.replace(tmp_path, ACTIVE_TRADES_STATE_PATH)
        except Exception as exc:
            log.warning(f"[state] Could not save active trades {ACTIVE_TRADES_STATE_PATH}: {exc}")

    @staticmethod
    def _jsonable(value: object) -> object:
        if isinstance(value, dict):
            return {str(k): Bot._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [Bot._jsonable(v) for v in value]
        if isinstance(value, np.generic):
            return Bot._jsonable(value.item())
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    @staticmethod
    def _compact_token(value: object, *, max_len: int) -> str:
        text = "".join(ch for ch in str(value or "").upper() if ch.isalnum())
        return (text or "X")[:max_len]

    @staticmethod
    def _strategy_code(strategy: object) -> str:
        text = str(strategy or "").lower()
        if text == "million_moves":
            return "MM"
        if text == "turtle_soup":
            return "TS"
        if text.startswith("session_orb"):
            return "ORB"
        if text.startswith("risk"):
            return "RISK"
        return Bot._compact_token(text, max_len=4)

    @staticmethod
    def _direction_code(direction: object) -> str:
        text = str(direction or "").lower()
        if text in {"long", "buy"}:
            return "L"
        if text in {"short", "sell"}:
            return "S"
        return Bot._compact_token(text, max_len=1)

    def _make_order_link_id(
        self,
        *,
        kind: str,
        strategy: object,
        symbol: object,
        direction: object,
    ) -> str:
        ts = datetime.now(timezone.utc).strftime("%y%m%d%H%M%S")
        suffix = uuid.uuid4().hex[:4].upper()
        code = self._strategy_code(strategy)
        base = self._compact_token(str(symbol).replace("USDT", ""), max_len=8)
        side = self._direction_code(direction)
        link_id = f"{kind[:1].upper()}-{code}-{base}-{side}-{ts}-{suffix}"
        return link_id[:36]

    @staticmethod
    def _feature_snapshot(columns: list[str] | tuple[str, ...], values: object) -> dict[str, object]:
        try:
            arr = np.asarray(values, dtype=float).reshape(-1)
        except Exception:
            return {}
        out: dict[str, object] = {}
        for idx, column in enumerate(columns):
            if idx >= len(arr):
                break
            value = float(arr[idx])
            out[str(column)] = value if math.isfinite(value) else None
        return out

    def _append_ledger_event(self, event_type: str, **payload: object) -> None:
        event = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            **payload,
        }
        try:
            os.makedirs(os.path.dirname(TRADE_LEDGER_PATH) or ".", exist_ok=True)
            with open(TRADE_LEDGER_PATH, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(self._jsonable(event), sort_keys=True) + "\n")
        except Exception as exc:
            log.warning(f"[ledger] Could not append event to {TRADE_LEDGER_PATH}: {exc}")

    def _record_signal_event(
        self,
        status: str,
        *,
        symbol: str,
        sig: dict,
        reason: str | None = None,
        order_id: str | None = None,
        extra: dict[str, object] | None = None,
    ) -> None:
        self._append_ledger_event(
            "signal",
            status=status,
            symbol=symbol,
            strategy=sig.get("strategy"),
            direction=sig.get("signal", sig.get("direction")),
            entry=sig.get("entry", sig.get("entry_price")),
            sl=sig.get("sl", sig.get("stop_price")),
            tp=sig.get("tp1", sig.get("target", sig.get("target_price"))),
            prob=sig.get("prob", sig.get("dt_prob")),
            threshold=sig.get("threshold", sig.get("dt_threshold")),
            reason=reason,
            order_id=order_id,
            order_link_id=sig.get("order_link_id"),
            signal_time=sig.get("entry_time"),
            feature_columns=sig.get("feature_columns"),
            feature_snapshot=sig.get("feature_snapshot"),
            market_snapshot=sig.get("market_snapshot"),
            market_context=sig.get("market_context"),
            order_request=sig.get("order_request"),
            order_error=sig.get("order_error"),
            extra=extra or {},
        )

    @staticmethod
    def _to_float(value: object) -> float | None:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(out):
            return None
        return out

    @staticmethod
    def _zscore(value: float | None, history_values: list[float]) -> float | None:
        if value is None or len(history_values) < 5:
            return None
        arr = np.asarray(history_values, dtype=np.float64)
        if arr.size < 5:
            return None
        std = float(np.std(arr))
        if std <= 0 or not math.isfinite(std):
            return None
        mean = float(np.mean(arr))
        z = (value - mean) / std
        return float(z) if math.isfinite(z) else None

    def _extract_response_list(self, resp: dict) -> list[dict]:
        items = resp.get("result", {}).get("list", []) if isinstance(resp, dict) else []
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
        return []

    def _build_regime_context(self, state: SymbolState, sig: dict) -> dict:
        bars = state.snapshot()
        if len(bars) < max(ATR_LEN + 5, VOL_WIN + 5, ATR_PCTILE_WIN + 5):
            return {}
        try:
            c = np.array([b["close"] for b in bars], dtype=np.float64)
            h = np.array([b["high"] for b in bars], dtype=np.float64)
            l = np.array([b["low"] for b in bars], dtype=np.float64)
            v = np.array([b["volume"] for b in bars], dtype=np.float64)
            atr = ind_atr(h, l, c, ATR_LEN)
            atr_last = float(atr[-1]) if len(atr) else float("nan")
            atr_pctile = atr_pctile_last(atr, ATR_PCTILE_WIN)
            vol_ratio = vol_ratio_last(v, VOL_WIN)

            rets = np.diff(np.log(np.maximum(c, 1e-12)))
            rv_20 = float(np.std(rets[-20:])) if len(rets) >= 20 else float("nan")
            rv_96 = float(np.std(rets[-96:])) if len(rets) >= 96 else float("nan")
            vol_window = v[-200:] if len(v) >= 200 else v
            vol_pct = float(np.mean(vol_window <= v[-1])) if len(vol_window) else float("nan")

            session = str(sig.get("session", "")).strip().lower()
            entry_time = datetime.now(timezone.utc)
            raw_time = sig.get("entry_time")
            if raw_time:
                try:
                    text = str(raw_time).replace("Z", "+00:00")
                    parsed = datetime.fromisoformat(text)
                    entry_time = parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
                    entry_time = entry_time.astimezone(timezone.utc)
                except (TypeError, ValueError):
                    pass

            session_ranges = {
                "asia": (0, 8),
                "london": (7, 15),
                "newyork": (13, 21),
                "ny": (13, 21),
            }
            minutes_to_session_open = None
            minutes_to_session_close = None
            if session in session_ranges:
                start_h, end_h = session_ranges[session]
                now_min = entry_time.hour * 60 + entry_time.minute
                start_min = start_h * 60
                end_min = end_h * 60
                if now_min < start_min:
                    minutes_to_session_open = start_min - now_min
                    minutes_to_session_close = end_min - now_min
                elif now_min <= end_min:
                    minutes_to_session_open = 0
                    minutes_to_session_close = end_min - now_min
                else:
                    minutes_to_session_open = (24 * 60 - now_min) + start_min
                    minutes_to_session_close = (24 * 60 - now_min) + end_min

            out = {
                "session_tag": session or "unspecified",
                "minutes_to_session_open": minutes_to_session_open,
                "minutes_to_session_close": minutes_to_session_close,
                "atr": atr_last if math.isfinite(atr_last) else None,
                "atr_pctile": atr_pctile if math.isfinite(atr_pctile) else None,
                "volume_ratio": vol_ratio if math.isfinite(vol_ratio) else None,
                "realized_vol_20": rv_20 if math.isfinite(rv_20) else None,
                "realized_vol_96": rv_96 if math.isfinite(rv_96) else None,
                "volume_percentile_200": vol_pct if math.isfinite(vol_pct) else None,
            }
            return out
        except Exception as exc:
            log.debug(f"[{state.symbol}] regime context build failed: {exc}")
            return {}

    def _build_provenance_context(self, state: SymbolState, sig: dict) -> dict:
        return {
            "bot_version": "4.3",
            "strategy": sig.get("strategy"),
            "strategy_variant": sig.get("variant"),
            "symbol": state.symbol,
            "timeframe": TIMEFRAME,
            "dt_enabled": state.use_dt,
            "dt_threshold": state.dt_threshold,
            "dt_meta": dict(state.dt_meta),
            "feature_schema": list(FEATURE_NAMES),
            "model_dirs": {
                "dt": MODELS_DIR,
                "turtle": TURTLE_MODELS_DIR,
                "session_orb": SESSION_ORB_MODELS_DIR,
            },
            "runtime": {
                "bybit_demo": DEMO,
                "public_ws_demo": PUBLIC_WS_DEMO,
                "private_ws_demo": PRIVATE_WS_DEMO,
            },
            "build": {
                "git_commit": os.environ.get("GIT_COMMIT") or os.environ.get("COMMIT_SHA"),
                "image_tag": os.environ.get("IMAGE_TAG") or os.environ.get("DOCKER_IMAGE_TAG"),
            },
        }

    def _fetch_market_context(self, state: SymbolState, sig: dict) -> dict:
        symbol = state.symbol
        local_dt = datetime.now(timezone.utc)
        context: dict[str, object] = {
            "captured_at_utc": local_dt.isoformat(),
            "captured_at_ms": int(local_dt.timestamp() * 1000),
            "symbol": symbol,
            "strategy": sig.get("strategy"),
            "direction": sig.get("signal"),
        }

        try:
            server_resp = self._http.get_server_time()
            context["server_time"] = server_resp.get("result") if isinstance(server_resp, dict) else None
        except Exception as exc:
            log.debug(f"[{symbol}] server time fetch failed: {exc}")

        ticker = self._fetch_market_snapshot(symbol)
        context["ticker"] = ticker

        mark = self._to_float(ticker.get("markPrice")) if ticker else None
        index = self._to_float(ticker.get("indexPrice")) if ticker else None
        basis_pct = ((mark - index) / index) if (mark is not None and index and index != 0) else None
        context["basis"] = {
            "mark_price": mark,
            "index_price": index,
            "basis_pct": basis_pct if basis_pct is None or math.isfinite(basis_pct) else None,
        }

        try:
            ob_resp = self._http.get_orderbook(category="linear", symbol=symbol, limit=50)
            ob = ob_resp.get("result", {}) if isinstance(ob_resp, dict) else {}
            bids_raw = ob.get("b", []) if isinstance(ob, dict) else []
            asks_raw = ob.get("a", []) if isinstance(ob, dict) else []
            bids = [entry for entry in bids_raw if isinstance(entry, (list, tuple)) and len(entry) >= 2]
            asks = [entry for entry in asks_raw if isinstance(entry, (list, tuple)) and len(entry) >= 2]
            best_bid = self._to_float(bids[0][0]) if bids else None
            best_ask = self._to_float(asks[0][0]) if asks else None
            spread = (best_ask - best_bid) if (best_ask is not None and best_bid is not None) else None
            mid = ((best_ask + best_bid) / 2.0) if (best_ask is not None and best_bid is not None) else None
            spread_bps = (spread / mid * 1e4) if (spread is not None and mid not in (None, 0)) else None
            bid_notional_top10 = 0.0
            ask_notional_top10 = 0.0
            for entry in bids[:10]:
                p = self._to_float(entry[0])
                q = self._to_float(entry[1])
                if p is not None and q is not None:
                    bid_notional_top10 += p * q
            for entry in asks[:10]:
                p = self._to_float(entry[0])
                q = self._to_float(entry[1])
                if p is not None and q is not None:
                    ask_notional_top10 += p * q
            denom = bid_notional_top10 + ask_notional_top10
            imbalance = ((bid_notional_top10 - ask_notional_top10) / denom) if denom > 0 else None
            context["orderbook"] = {
                "best_bid": best_bid,
                "best_ask": best_ask,
                "spread": spread,
                "spread_bps": spread_bps,
                "bid_notional_top10": bid_notional_top10,
                "ask_notional_top10": ask_notional_top10,
                "imbalance_top10": imbalance,
                "raw": ob,
            }
        except Exception as exc:
            log.debug(f"[{symbol}] orderbook fetch failed: {exc}")

        try:
            trades_resp = self._http.get_public_trade_history(category="linear", symbol=symbol, limit=200)
            trades = self._extract_response_list(trades_resp)
            now_ms = int(local_dt.timestamp() * 1000)
            buy_notional = 0.0
            sell_notional = 0.0
            buy_qty = 0.0
            sell_qty = 0.0
            count = 0
            for trade in trades:
                ts = int(self._to_float(trade.get("time")) or self._to_float(trade.get("T")) or 0)
                if ts <= 0 or now_ms - ts > 60_000:
                    continue
                side = str(trade.get("side", "")).lower()
                price = self._to_float(trade.get("price") or trade.get("p"))
                qty = self._to_float(trade.get("size") or trade.get("v"))
                if price is None or qty is None:
                    continue
                count += 1
                if side == "buy":
                    buy_notional += price * qty
                    buy_qty += qty
                elif side == "sell":
                    sell_notional += price * qty
                    sell_qty += qty
            denom = buy_notional + sell_notional
            imbalance = ((buy_notional - sell_notional) / denom) if denom > 0 else None
            context["trade_flow_60s"] = {
                "trade_count": count,
                "buy_notional": buy_notional,
                "sell_notional": sell_notional,
                "buy_qty": buy_qty,
                "sell_qty": sell_qty,
                "notional_imbalance": imbalance,
                "raw": trades[:100],
            }
        except Exception as exc:
            log.debug(f"[{symbol}] public trade history fetch failed: {exc}")

        oi_current = self._to_float(ticker.get("openInterest")) if ticker else None
        oi_history: list[dict] = []
        try:
            oi_resp = self._http.get_open_interest(
                category="linear",
                symbol=symbol,
                intervalTime="5min",
                limit=24,
            )
            oi_rows = self._extract_response_list(oi_resp)
            for row in oi_rows:
                ts = int(self._to_float(row.get("timestamp")) or 0)
                oi = self._to_float(row.get("openInterest"))
                if ts > 0 and oi is not None:
                    oi_history.append({"ts": ts, "open_interest": oi})
            oi_history.sort(key=lambda item: item["ts"])
        except Exception as exc:
            log.debug(f"[{symbol}] open interest history fetch failed: {exc}")

        if oi_current is None and oi_history:
            oi_current = oi_history[-1]["open_interest"]

        if symbol not in self._oi_history:
            self._oi_history[symbol] = deque(maxlen=512)
        for row in oi_history:
            if not self._oi_history[symbol] or self._oi_history[symbol][-1].get("ts") != row.get("ts"):
                self._oi_history[symbol].append(row)

        oi_series = [self._to_float(item.get("open_interest")) for item in self._oi_history[symbol]]
        oi_values = [x for x in oi_series if x is not None]
        oi_delta_5m = None
        oi_delta_15m = None
        oi_delta_1h = None
        if oi_current is not None and len(oi_values) >= 2:
            oi_delta_5m = oi_current - oi_values[-2]
        if oi_current is not None and len(oi_values) >= 4:
            oi_delta_15m = oi_current - oi_values[-4]
        if oi_current is not None and len(oi_values) >= 13:
            oi_delta_1h = oi_current - oi_values[-13]

        context["open_interest"] = {
            "current": oi_current,
            "delta_5m": oi_delta_5m,
            "delta_15m": oi_delta_15m,
            "delta_1h": oi_delta_1h,
            "zscore": self._zscore(oi_current, oi_values[-100:]),
            "history": list(self._oi_history[symbol])[-120:],
        }

        funding_current = self._to_float(ticker.get("fundingRate")) if ticker else None
        funding_history: list[dict] = []
        try:
            funding_resp = self._http.get_funding_rate_history(category="linear", symbol=symbol, limit=50)
            funding_rows = self._extract_response_list(funding_resp)
            for row in funding_rows:
                ts = int(self._to_float(row.get("fundingRateTimestamp")) or self._to_float(row.get("fundingRateTs")) or 0)
                rate = self._to_float(row.get("fundingRate"))
                if ts > 0 and rate is not None:
                    funding_history.append({"ts": ts, "funding_rate": rate})
            funding_history.sort(key=lambda item: item["ts"])
        except Exception as exc:
            log.debug(f"[{symbol}] funding history fetch failed: {exc}")

        if symbol not in self._funding_history:
            self._funding_history[symbol] = deque(maxlen=512)
        for row in funding_history:
            if not self._funding_history[symbol] or self._funding_history[symbol][-1].get("ts") != row.get("ts"):
                self._funding_history[symbol].append(row)

        funding_series = [self._to_float(item.get("funding_rate")) for item in self._funding_history[symbol]]
        funding_values = [x for x in funding_series if x is not None]
        if funding_current is None and funding_values:
            funding_current = funding_values[-1]

        next_funding_ms = int(self._to_float(ticker.get("nextFundingTime")) or 0) if ticker else 0
        now_ms = int(local_dt.timestamp() * 1000)
        mins_to_next_funding = ((next_funding_ms - now_ms) / 60000.0) if next_funding_ms > 0 else None
        context["funding"] = {
            "current": funding_current,
            "next_funding_time_ms": next_funding_ms if next_funding_ms > 0 else None,
            "minutes_to_next_funding": mins_to_next_funding,
            "zscore": self._zscore(funding_current, funding_values[-100:]),
            "history": list(self._funding_history[symbol])[-120:],
        }

        context["regime"] = self._build_regime_context(state, sig)
        entry_f = self._to_float(sig.get("entry"))
        sl_f = self._to_float(sig.get("sl"))
        tp_f = self._to_float(sig.get("tp1", sig.get("target")))
        tick = self._to_float(state.info.get("tick_size"))
        stop_ticks = None
        target_ticks = None
        if entry_f is not None and sl_f is not None and tick and tick > 0:
            stop_ticks = abs(entry_f - sl_f) / tick
        if entry_f is not None and tp_f is not None and tick and tick > 0:
            target_ticks = abs(tp_f - entry_f) / tick
        context["execution_plan"] = {
            "expected_entry": sig.get("entry"),
            "expected_stop": sig.get("sl"),
            "expected_target": sig.get("tp1", sig.get("target")),
            "trail_dist": sig.get("trail_dist"),
            "tick_size": state.info.get("tick_size"),
            "stop_distance_ticks": stop_ticks,
            "target_distance_ticks": target_ticks,
            "qty_step": state.info.get("qty_step"),
            "min_qty": state.info.get("min_qty"),
        }
        context["instrument_constraints"] = {
            "qty_step": state.info.get("qty_step"),
            "min_qty": state.info.get("min_qty"),
            "tick_size": state.info.get("tick_size"),
            "min_leverage": state.info.get("min_leverage_raw", state.info.get("min_leverage")),
            "max_leverage": state.info.get("max_leverage_raw", state.info.get("max_leverage")),
            "leverage_step": state.info.get("leverage_step_raw", state.info.get("leverage_step")),
        }
        context["provenance"] = self._build_provenance_context(state, sig)
        return context

    def _load_heartbeat_state(self) -> None:
        if not os.path.exists(HEARTBEAT_STATE_PATH):
            return
        try:
            with open(HEARTBEAT_STATE_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self._last_daily_heartbeat_key = str(data.get("last_daily_heartbeat_key", ""))
            log.info(
                f"[heartbeat] Loaded state from {HEARTBEAT_STATE_PATH}  "
                f"last_daily={self._last_daily_heartbeat_key or '-'}"
            )
        except Exception as exc:
            log.warning(f"[heartbeat] Could not load state {HEARTBEAT_STATE_PATH}: {exc}")

    def _save_heartbeat_state(self) -> None:
        try:
            os.makedirs(os.path.dirname(HEARTBEAT_STATE_PATH) or ".", exist_ok=True)
            tmp_path = f"{HEARTBEAT_STATE_PATH}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(
                    {"last_daily_heartbeat_key": self._last_daily_heartbeat_key},
                    fh,
                    indent=2,
                    sort_keys=True,
                )
            os.replace(tmp_path, HEARTBEAT_STATE_PATH)
        except Exception as exc:
            log.warning(f"[heartbeat] Could not save state {HEARTBEAT_STATE_PATH}: {exc}")

    def _set_drawdown_block_locked(
        self,
        *,
        scope: str,
        key: str,
        reason: str,
        dd: float,
        limit: float,
        equity: float,
        start_equity: float,
    ) -> None:
        self._dd_blocked = True
        self._dd_block_scope = scope
        self._dd_block_key = key
        self._dd_block_reason = reason
        log.warning(
            f"[risk] {reason} drawdown limit hit: {dd:.2%}  "
            f"(limit={limit:.1%})  equity={equity:.2f}  start={start_equity:.2f}  "
            f"-- new entries BLOCKED"
        )

    def _refresh_risk_state(self) -> None:
        """Update drawdown baselines, enforce cooldowns, and flatten on breach."""
        import datetime as _dt

        now = _dt.datetime.now(_dt.timezone.utc)
        day_key = now.strftime("%Y-%m-%d")
        week_key = self._utc_week_key(now)
        try:
            equity = get_equity(self._http)
        except Exception as exc:
            log.warning(f"[risk] Could not fetch equity: {exc}")
            return

        flatten_reason: Optional[str] = None
        flatten_block_id = ""
        with self._pos_lock:
            changed = False

            if self._day_date != day_key:
                self._day_date = day_key
                self._day_start_equity = equity
                changed = True
                log.info(f"[risk] Day start ({day_key} UTC): baseline equity={equity:.2f}")
            elif self._day_start_equity <= 0:
                self._day_start_equity = equity
                changed = True

            if self._week_key != week_key:
                self._week_key = week_key
                self._week_start_equity = equity
                changed = True
                log.info(f"[risk] Week start ({week_key} UTC): baseline equity={equity:.2f}")
            elif self._week_start_equity <= 0:
                self._week_start_equity = equity
                changed = True

            if self._dd_blocked:
                expired = (
                    (self._dd_block_scope == "day" and self._dd_block_key != day_key)
                    or (self._dd_block_scope == "week" and self._dd_block_key != week_key)
                    or self._dd_block_scope not in {"day", "week"}
                )
                if expired:
                    log.info(
                        f"[risk] Drawdown cooldown expired "
                        f"({self._dd_block_scope or 'unknown'} {self._dd_block_key or '-'}); "
                        f"trading resumed  equity={equity:.2f}"
                    )
                    self._dd_blocked = False
                    self._dd_block_scope = ""
                    self._dd_block_key = ""
                    self._dd_block_reason = ""
                    self._risk_flattened_block_id = ""
                    changed = True

            daily_dd = (
                (equity - self._day_start_equity) / self._day_start_equity
                if self._day_start_equity > 0 else 0.0
            )
            weekly_dd = (
                (equity - self._week_start_equity) / self._week_start_equity
                if self._week_start_equity > 0 else 0.0
            )

            if (
                self._dd_blocked
                and self._dd_block_scope != "week"
                and MAX_WEEKLY_DD > 0
                and weekly_dd <= -MAX_WEEKLY_DD
            ):
                self._set_drawdown_block_locked(
                    scope="week",
                    key=week_key,
                    reason="Weekly",
                    dd=weekly_dd,
                    limit=MAX_WEEKLY_DD,
                    equity=equity,
                    start_equity=self._week_start_equity,
                )
                changed = True

            if not self._dd_blocked:
                if MAX_WEEKLY_DD > 0 and weekly_dd <= -MAX_WEEKLY_DD:
                    self._set_drawdown_block_locked(
                        scope="week",
                        key=week_key,
                        reason="Weekly",
                        dd=weekly_dd,
                        limit=MAX_WEEKLY_DD,
                        equity=equity,
                        start_equity=self._week_start_equity,
                    )
                    changed = True
                elif MAX_DAILY_DD > 0 and daily_dd <= -MAX_DAILY_DD:
                    self._set_drawdown_block_locked(
                        scope="day",
                        key=day_key,
                        reason="Daily",
                        dd=daily_dd,
                        limit=MAX_DAILY_DD,
                        equity=equity,
                        start_equity=self._day_start_equity,
                    )
                    changed = True

            if self._dd_blocked and DD_CLOSE_POSITIONS_ON_BREACH:
                block_id = self._dd_block_id_locked()
                if block_id and self._risk_flattened_block_id != block_id:
                    flatten_block_id = block_id
                    flatten_reason = self._dd_block_reason or "Drawdown"

            if changed:
                self._save_risk_state_locked()

        if flatten_reason:
            if self._flatten_all_positions(f"{flatten_reason} drawdown breach"):
                with self._pos_lock:
                    self._risk_flattened_block_id = flatten_block_id
                    self._save_risk_state_locked()

    # -- DT model loading ------------------------------------------------------

    def _load_dt_models(self) -> None:
        os.makedirs(MODELS_DIR, exist_ok=True)
        loaded = 0
        for sym, state in self._states.items():
            if not state.use_dt:
                continue
            model_path = os.path.join(MODELS_DIR, f"{sym}_dt.pkl")
            if not os.path.exists(model_path):
                if ALLOW_MM_WITHOUT_DT:
                    log.warning(
                        f"  {sym}: use_dt=true but {model_path} not found -- "
                        f"running WITHOUT DT filter because ALLOW_MM_WITHOUT_DT=true."
                    )
                    state.use_dt = False
                else:
                    log.warning(
                        f"  {sym}: use_dt=true but {model_path} not found -- "
                        f"Million Moves disabled for this symbol. Run train_dt.py, "
                        f"set use_dt=false, or set ALLOW_MM_WITHOUT_DT=true to allow trail-only fallback."
                    )
                    state.mm_enabled = False
                continue
            try:
                with open(model_path, "rb") as fh:
                    saved = pickle.load(fh)
                state.dt_model     = saved["model"]
                state.dt_threshold = float(saved["threshold"])
                state.dt_meta = {
                    "path": model_path,
                    "trained_on": saved.get("trained_on"),
                    "oos_dt_sh": saved.get("oos_dt_sh"),
                    "threshold": saved.get("threshold"),
                }
                log.info(
                    f"  {sym}: DT model loaded  "
                    f"threshold={state.dt_threshold:.2f}  "
                    f"trained_on={saved.get('trained_on', '?')} bars  "
                    f"oos_dt_sh={saved.get('oos_dt_sh', '?')}"
                )
                loaded += 1
            except Exception as exc:
                log.error(f"  {sym}: failed to load DT model: {exc}")
                state.use_dt = False
        log.info(f"DT models: {loaded} loaded")

    # -- Position sync ---------------------------------------------------------

    def _sync_positions(self) -> None:
        try:
            resp      = self._http.get_positions(category="linear", settleCoin="USDT")
            positions = resp.get("result", {}).get("list", [])
        except Exception as exc:
            log.error(f"Failed to sync positions: {exc}"); return

        with self._pos_lock:
            self._open_count = 0
            self._cluster_a_count = 0
            open_symbols: set[str] = set()
            active_state_changed = False
            for st in self._states.values():
                st.in_position = False; st.position_side = None
            for pos in positions:
                sym  = pos.get("symbol", ""); size = float(pos.get("size", 0))
                side = pos.get("side", "")
                if sym in self._states and size > 0:
                    open_symbols.add(sym)
                    self._states[sym].in_position   = True
                    self._states[sym].position_side = side
                    self._open_count += 1
                    if sym in CLUSTER_A:
                        self._cluster_a_count += 1
                    log.info(f"  Synced open position: {sym} {side} size={size}")
            for sym, st in self._states.items():
                if sym not in open_symbols and st.active_trade is not None:
                    st.active_trade = None
                    active_state_changed = True
            if active_state_changed:
                self._save_active_trade_state_locked()

    def _flatten_all_positions(self, reason: str) -> bool:
        """Cancel open orders and market-close all tracked Bybit linear positions."""
        log.warning(f"[risk] Flattening tracked positions: {reason}")
        self._telegram.send_risk_event(
            "drawdown flatten",
            fields={
                "Reason": reason,
                "Action": "cancel open orders and submit reduce-only market closes",
            },
        )
        self._append_ledger_event(
            "risk",
            event="drawdown_flatten",
            reason=reason,
            action="cancel open orders and submit reduce-only market closes",
        )
        try:
            resp = self._http.get_positions(category="linear", settleCoin="USDT")
            positions = resp.get("result", {}).get("list", [])
        except Exception as exc:
            log.error(f"[risk] Could not fetch positions for drawdown flatten: {exc}")
            return False

        failures = 0
        closed = 0
        for pos in positions:
            sym = pos.get("symbol", "")
            if sym not in self._states:
                continue
            try:
                size = float(pos.get("size", 0))
            except (TypeError, ValueError):
                continue
            if size <= 0:
                continue

            side = pos.get("side", "")
            if side == "Buy":
                close_side = "Sell"
            elif side == "Sell":
                close_side = "Buy"
            else:
                log.warning(f"[risk] {sym}: unknown position side {side!r}; skipping flatten")
                failures += 1
                continue

            try:
                cancel_resp = self._http.cancel_all_orders(category="linear", symbol=sym)
                if cancel_resp.get("retCode", 0) != 0:
                    log.warning(f"[risk] {sym}: cancel_all_orders: {cancel_resp.get('retMsg', '?')}")
            except Exception as exc:
                log.warning(f"[risk] {sym}: cancel_all_orders failed before flatten: {exc}")

            try:
                state = self._states.get(sym)
                active_trade = dict(state.active_trade or {}) if state is not None else {}
                order_link_id = self._make_order_link_id(
                    kind="X",
                    strategy=active_trade.get("strategy") or "risk",
                    symbol=sym,
                    direction=active_trade.get("direction") or close_side,
                )
                order_resp = self._http.place_order(
                    category="linear",
                    symbol=sym,
                    side=close_side,
                    orderType="Market",
                    qty=qty_to_str(size),
                    reduceOnly=True,
                    positionIdx=0,
                    orderLinkId=order_link_id,
                )
                if order_resp.get("retCode", -1) != 0:
                    failures += 1
                    log.error(
                        f"[risk] {sym}: flatten order rejected "
                        f"(retCode={order_resp.get('retCode')}): {order_resp.get('retMsg', '?')}"
                    )
                    continue
                closed += 1
                order_id = str(order_resp.get("result", {}).get("orderId") or "")
                if order_id:
                    self._manual_exit_orders[order_id] = {
                        "symbol": sym,
                        "reason": reason,
                        "strategy": active_trade.get("strategy"),
                        "direction": active_trade.get("direction"),
                        "order_link_id": order_link_id,
                    }
                log.warning(
                    f"[risk] {sym}: reduce-only market close submitted "
                    f"size={qty_to_str(size)} orderId={order_id or '-'} "
                    f"orderLinkId={order_link_id or '-'}"
                )
            except Exception as exc:
                failures += 1
                log.error(f"[risk] {sym}: flatten order failed: {exc}")

        if closed:
            time.sleep(0.8)
            self._sync_positions()
        else:
            log.info("[risk] No tracked open positions needed flattening")

        return failures == 0

    @staticmethod
    def _optional_float(value: object) -> Optional[float]:
        if value in (None, "", "0", 0):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number) or number == 0:
            return None
        return number

    @staticmethod
    def _numeric_float(value: object) -> Optional[float]:
        if value in (None, ""):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        return number

    def _send_startup_reconciliation_report(self) -> None:
        with self._pos_lock:
            open_positions = {
                sym: {
                    "side": state.position_side or "-",
                    "strategy": (state.active_trade or {}).get("strategy", "-"),
                    "has_metadata": bool(state.active_trade),
                }
                for sym, state in self._states.items()
                if state.in_position
            }
            missing_meta = sorted(
                sym for sym, info in open_positions.items() if not info["has_metadata"]
            )
            active_meta = sorted(
                sym for sym, state in self._states.items() if state.active_trade
            )
            fields = {
                "Open positions": len(open_positions),
                "Active metadata": len(active_meta),
                "Open without metadata": ",".join(missing_meta) if missing_meta else "-",
                "DD blocked": self._dd_blocked,
                "DD reason": self._dd_block_reason or "-",
                "Open slots": f"{self._open_count}/{MAX_OPEN}",
                "Cluster A slots": f"{self._cluster_a_count}/{CLUSTER_A_MAX}",
            }
        try:
            fields["Equity"] = f"{get_equity(self._http):.2f}"
        except Exception as exc:
            fields["Equity"] = f"unavailable: {exc}"
        self._telegram.send_admin_event("startup reconciliation", fields=fields)
        self._append_ledger_event("admin", event="startup_reconciliation", **fields)

    @staticmethod
    def _order_identity(order: dict) -> str:
        return str(
            order.get("orderId")
            or order.get("orderLinkId")
            or ":".join(
                str(order.get(key) or "")
                for key in ("symbol", "side", "orderType", "price", "triggerPrice", "qty")
            )
        )

    @staticmethod
    def _order_status_is_open(order: dict) -> bool:
        status = str(order.get("orderStatus") or "").strip().lower()
        return not status or status in {"new", "partiallyfilled", "untriggered", "created"}

    @staticmethod
    def _order_is_protection(order: dict) -> bool:
        if Bot._truthy_field(order.get("reduceOnly")) or Bot._truthy_field(order.get("closeOnTrigger")):
            return True
        stop_type = Bot._clean_stop_order_type(order.get("stopOrderType")).lower()
        if stop_type in {
            "takeprofit",
            "stoploss",
            "trailingstop",
            "partialtakeprofit",
            "partialstoploss",
            "tpslorder",
            "ocostoporder",
            "bidirectionaltpslorder",
        }:
            return True
        descriptor = " ".join(
            str(order.get(key) or "")
            for key in ("stopOrderType", "createType", "orderLinkId", "orderFilter")
        ).lower()
        return any(token in descriptor for token in ("takeprofit", "stoploss", "trailing", "tpsl"))

    def _fetch_open_orders(self) -> list[dict]:
        def pages(base_kwargs: dict) -> list[dict]:
            out: list[dict] = []
            cursor = ""
            seen_ids: set[str] = set()
            for _ in range(5):
                kwargs = dict(base_kwargs)
                if cursor:
                    kwargs["cursor"] = cursor
                resp = self._http.get_open_orders(**kwargs)
                if resp.get("retCode", 0) != 0:
                    raise RuntimeError(resp.get("retMsg", "?"))
                result = resp.get("result", {}) or {}
                for order in result.get("list", []) or []:
                    if not isinstance(order, dict) or not self._order_status_is_open(order):
                        continue
                    identity = self._order_identity(order)
                    if identity in seen_ids:
                        continue
                    seen_ids.add(identity)
                    out.append(order)
                cursor = str(result.get("nextPageCursor") or "")
                if not cursor:
                    break
            return out

        try:
            return pages(
                {
                    "category": "linear",
                    "settleCoin": "USDT",
                    "openOnly": 0,
                    "limit": 50,
                }
            )
        except Exception as global_exc:
            log.info(f"[orders] Global open-order query unavailable; falling back per symbol: {global_exc}")

        by_id: dict[str, dict] = {}
        failures: list[str] = []
        filters: tuple[str | None, ...] = (None, "Order", "StopOrder", "tpslOrder")
        for sym in sorted(self._states):
            for order_filter in filters:
                try:
                    kwargs = {
                        "category": "linear",
                        "symbol": sym,
                        "openOnly": 0,
                        "limit": 50,
                    }
                    if order_filter:
                        kwargs["orderFilter"] = order_filter
                    for order in pages(kwargs):
                        by_id[self._order_identity(order)] = order
                except Exception as exc:
                    if order_filter is None:
                        failures.append(f"{sym}: {exc}")
                        continue
                    break
        if failures and not by_id:
            raise RuntimeError("; ".join(failures[:5]))
        return list(by_id.values())

    def _audit_open_orders(self, *, force: bool = False) -> None:
        if OPEN_ORDER_AUDIT_SECONDS <= 0:
            return
        now = time.time()
        if not force and now - self._last_open_order_audit_ts < OPEN_ORDER_AUDIT_SECONDS:
            return
        self._last_open_order_audit_ts = now

        try:
            orders = self._fetch_open_orders()
        except Exception as exc:
            log.warning(f"[orders] Could not fetch open orders for audit: {exc}")
            self._admin_warn_once(
                "open-order-audit-fetch-failed",
                "open-order audit failed",
                {"Reason": f"Could not fetch open orders: {exc}"},
            )
            return

        try:
            resp = self._http.get_positions(category="linear", settleCoin="USDT")
            positions = resp.get("result", {}).get("list", [])
        except Exception as exc:
            log.warning(f"[orders] Could not fetch positions for open-order audit: {exc}")
            self._admin_warn_once(
                "open-order-position-fetch-failed",
                "open-order audit failed",
                {"Reason": f"Could not fetch positions: {exc}"},
            )
            return

        open_positions: dict[str, dict] = {}
        for pos in positions:
            try:
                size = float(pos.get("size", 0) or 0)
            except (TypeError, ValueError):
                continue
            if size > 0:
                open_positions[str(pos.get("symbol", ""))] = pos

        protection_counts: dict[str, list[dict]] = {}
        for order in orders:
            sym = str(order.get("symbol") or "")
            if not sym:
                continue
            state = self._states.get(sym)
            identity = self._order_identity(order)
            order_link_id = str(order.get("orderLinkId") or "")
            is_protection = self._order_is_protection(order)
            common_fields = {
                "Symbol": sym,
                "Side": order.get("side", "-"),
                "Qty": order.get("qty", "-"),
                "Order type": order.get("orderType", "-"),
                "Status": order.get("orderStatus", "-"),
                "Price": order.get("price", "-"),
                "Trigger price": order.get("triggerPrice", "-"),
                "Stop type": self._clean_stop_order_type(order.get("stopOrderType")) or "-",
                "Order link id": order_link_id or "-",
                "Order ID": identity,
            }

            if state is None:
                self._admin_warn_once(
                    f"open-order-unknown-symbol:{identity}",
                    "open order on untracked symbol",
                    {**common_fields, "Action": "manual review recommended"},
                )
                continue

            if not is_protection:
                title = "entry-like open order detected"
                fields = {
                    **common_fields,
                    "Action": "manual review recommended; bot normally uses market entries",
                }
                if order_link_id and not (order_link_id.startswith("E-") or order_link_id.startswith("X-")):
                    title = "non-bot open order detected"
                    fields["Action"] = "manual review recommended; orderLinkId is not bot-prefixed"
                self._admin_warn_once(f"entry-like-open-order:{identity}", title, fields)
                continue

            protection_counts.setdefault(sym, []).append(order)
            if sym not in open_positions:
                self._admin_warn_once(
                    f"orphan-protection-order:{identity}",
                    "protection order without position",
                    {**common_fields, "Action": "manual review/cancel if stale"},
                )

        for sym, sym_orders in protection_counts.items():
            if len(sym_orders) <= 4:
                continue
            identities = [self._order_identity(order) for order in sym_orders[:4]]
            self._admin_warn_once(
                f"many-protection-orders:{sym}:{len(sym_orders)}",
                "many protection orders open",
                {
                    "Symbol": sym,
                    "Count": len(sym_orders),
                    "Sample order IDs": ",".join(identities),
                    "Action": "manual review recommended",
                },
            )

    def _audit_position_protection(self, *, force: bool = False) -> None:
        if PROTECTION_AUDIT_SECONDS <= 0:
            return
        now = time.time()
        if not force and now - self._last_protection_audit_ts < PROTECTION_AUDIT_SECONDS:
            return
        self._last_protection_audit_ts = now

        try:
            resp = self._http.get_positions(category="linear", settleCoin="USDT")
            positions = resp.get("result", {}).get("list", [])
        except Exception as exc:
            log.warning(f"[protection] Could not fetch positions for audit: {exc}")
            self._telegram.send_admin_event(
                "protection audit failed",
                fields={"Reason": f"Could not fetch positions: {exc}"},
            )
            return

        for pos in positions:
            sym = str(pos.get("symbol", ""))
            state = self._states.get(sym)
            if state is None:
                continue
            try:
                size = float(pos.get("size", 0) or 0)
            except (TypeError, ValueError):
                continue
            if size <= 0:
                continue

            active_trade = dict(state.active_trade or {})
            if not active_trade:
                key = f"missing-active-trade:{sym}:{pos.get('side')}:{size:g}"
                if key not in self._admin_warning_keys:
                    self._admin_warning_keys.add(key)
                    self._telegram.send_admin_event(
                        "position without metadata",
                        fields={
                            "Symbol": sym,
                            "Side": pos.get("side", "-"),
                            "Size": size,
                            "Action": "manual inspection recommended",
                        },
                    )
                continue

            pos_side = str(pos.get("side", ""))
            expected_side = "Buy" if active_trade.get("direction") == "long" else "Sell"
            if pos_side and expected_side and pos_side != expected_side:
                key = f"side-mismatch:{sym}:{pos_side}:{expected_side}"
                if key not in self._admin_warning_keys:
                    self._admin_warning_keys.add(key)
                    self._telegram.send_admin_event(
                        "position side mismatch",
                        fields={
                            "Symbol": sym,
                            "Bybit side": pos_side,
                            "Metadata side": expected_side,
                            "Strategy": active_trade.get("strategy", "-"),
                        },
                    )

            tick = float(state.info.get("tick_size", 0.0) or 0.0)
            tolerance = max(tick * 2.0, 1e-12)
            pos_sl = self._optional_float(pos.get("stopLoss"))
            pos_tp = self._optional_float(pos.get("takeProfit"))
            pos_trailing = self._optional_float(pos.get("trailingStop"))
            expected_sl = self._optional_float(active_trade.get("sl"))
            expected_tp = self._optional_float(active_trade.get("tp1"))
            exit_style = str(active_trade.get("exit_style", "trailing"))

            def differs(actual: Optional[float], expected: Optional[float]) -> bool:
                return expected is not None and (
                    actual is None or abs(actual - expected) > tolerance
                )

            if exit_style == "fixed_tp" and (
                differs(pos_sl, expected_sl) or differs(pos_tp, expected_tp)
            ):
                key = (
                    f"fixed-protection:{sym}:{expected_sl}:{expected_tp}:"
                    f"{pos_sl}:{pos_tp}"
                )
                if key not in self._admin_warning_keys:
                    self._admin_warning_keys.add(key)
                    self._telegram.send_admin_event(
                        "fixed TP/SL mismatch",
                        fields={
                            "Symbol": sym,
                            "Strategy": active_trade.get("strategy", "-"),
                            "Bybit SL": pos_sl if pos_sl is not None else "-",
                            "Expected SL": expected_sl if expected_sl is not None else "-",
                            "Bybit TP": pos_tp if pos_tp is not None else "-",
                            "Expected TP": expected_tp if expected_tp is not None else "-",
                            "Action": "re-sync fixed TP/SL",
                        },
                    )
                    if expected_sl is not None and expected_tp is not None:
                        self._sync_fixed_exit_orders(sym, sl_price=expected_sl, tp_price=expected_tp)
                continue

            if exit_style != "fixed_tp" and differs(pos_sl, expected_sl):
                key = f"trailing-sl:{sym}:{expected_sl}:{pos_sl}:{pos_trailing}"
                if key not in self._admin_warning_keys:
                    self._admin_warning_keys.add(key)
                    self._telegram.send_admin_event(
                        "trailing SL mismatch",
                        fields={
                            "Symbol": sym,
                            "Strategy": active_trade.get("strategy", "-"),
                            "Bybit SL": pos_sl if pos_sl is not None else "-",
                            "Expected SL": expected_sl if expected_sl is not None else "-",
                            "Bybit trailing": pos_trailing if pos_trailing is not None else "-",
                            "Action": "re-sync hard SL/trailing config",
                        },
                    )
                    trail_dist = self._optional_float(active_trade.get("trail_dist"))
                    active_price = self._optional_float(active_trade.get("tp1"))
                    if expected_sl is not None:
                        try:
                            kwargs = {
                                "category": "linear",
                                "symbol": sym,
                                "stopLoss": str(expected_sl),
                                "slTriggerBy": "LastPrice",
                                "tpslMode": "Full",
                                "positionIdx": 0,
                            }
                            if trail_dist is not None and active_price is not None:
                                kwargs["trailingStop"] = str(trail_dist)
                                kwargs["activePrice"] = str(active_price)
                            resp = self._http.set_trading_stop(**kwargs)
                            if resp.get("retCode", -1) != 0:
                                self._telegram.send_admin_event(
                                    "trailing protection resync failed",
                                    fields={
                                        "Symbol": sym,
                                        "Reason": resp.get("retMsg", "?"),
                                        "retCode": resp.get("retCode"),
                                    },
                                )
                        except Exception as exc:
                            self._telegram.send_admin_event(
                                "trailing protection resync failed",
                                fields={"Symbol": sym, "Reason": exc},
                            )

    @staticmethod
    def _ledger_event_day(event: dict) -> str:
        fill_time_ms = event.get("fill_time_ms")
        try:
            if fill_time_ms not in (None, "", 0, "0"):
                return datetime.fromtimestamp(
                    int(float(fill_time_ms)) / 1000.0,
                    tz=timezone.utc,
                ).date().isoformat()
        except (TypeError, ValueError, OSError):
            pass
        try:
            text = str(event.get("ts", "")).replace("Z", "+00:00")
            dt = datetime.fromisoformat(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).date().isoformat()
        except (TypeError, ValueError):
            return ""

    @staticmethod
    def _blank_perf_stats() -> dict[str, object]:
        return {
            "pnl": 0.0,
            "closed": 0,
            "wins": 0,
            "losses": 0,
            "breakeven": 0,
            "accepted": 0,
            "rejected": 0,
        }

    @staticmethod
    def _is_exit_event_name(event_name: object) -> bool:
        text = str(event_name or "").upper()
        return (
            "EXIT" in text
            or "TAKE PROFIT" in text
            or "STOP LOSS" in text
            or "TRAILING" in text
        )

    @staticmethod
    def _strategy_key(value: object) -> str:
        text = str(value or "").strip()
        return text if text else "unknown"

    def _ledger_stats_for_day(self, day_key: str) -> dict[str, object]:
        stats = self._blank_perf_stats()
        strategies: dict[str, dict[str, object]] = {}

        def strategy_stats(strategy: object) -> dict[str, object]:
            key = self._strategy_key(strategy)
            if key not in strategies:
                strategies[key] = self._blank_perf_stats()
            return strategies[key]

        def update_signal(target: dict[str, object], status: object) -> None:
            if status == "accepted":
                target["accepted"] = int(target["accepted"]) + 1
            elif status == "rejected":
                target["rejected"] = int(target["rejected"]) + 1

        def update_exit(target: dict[str, object], pnl: float) -> None:
            target["pnl"] = float(target["pnl"]) + pnl
            target["closed"] = int(target["closed"]) + 1
            if pnl > 0:
                target["wins"] = int(target["wins"]) + 1
            elif pnl < 0:
                target["losses"] = int(target["losses"]) + 1
            else:
                target["breakeven"] = int(target["breakeven"]) + 1

        if not os.path.exists(TRADE_LEDGER_PATH):
            stats["strategies"] = strategies
            return stats
        try:
            with open(TRADE_LEDGER_PATH, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if self._ledger_event_day(event) != day_key:
                        continue
                    event_type = event.get("type")
                    if event_type == "signal":
                        update_signal(stats, event.get("status"))
                        update_signal(strategy_stats(event.get("strategy")), event.get("status"))
                    elif event_type == "fill":
                        if not self._is_exit_event_name(event.get("event")):
                            continue
                        pnl = self._numeric_float(event.get("closed_pnl"))
                        if pnl is None:
                            continue
                        update_exit(stats, pnl)
                        update_exit(strategy_stats(event.get("strategy")), pnl)
        except Exception as exc:
            log.warning(f"[heartbeat] Could not read ledger {TRADE_LEDGER_PATH}: {exc}")
        stats["strategies"] = strategies
        return stats

    def _ledger_loss_counters(self, day_key: str) -> dict[str, dict[str, int]]:
        counters: dict[str, dict[str, int]] = {
            "daily_strategy": {},
            "daily_symbol": {},
            "streak_strategy": {},
            "streak_symbol": {},
        }
        if not os.path.exists(TRADE_LEDGER_PATH):
            return counters
        try:
            with open(TRADE_LEDGER_PATH, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("type") != "fill" or not self._is_exit_event_name(event.get("event")):
                        continue
                    pnl = self._numeric_float(event.get("closed_pnl"))
                    if pnl is None:
                        continue
                    strategy = self._strategy_key(event.get("strategy"))
                    symbol = str(event.get("symbol") or "").upper() or "UNKNOWN"
                    if pnl < 0 and self._ledger_event_day(event) == day_key:
                        counters["daily_strategy"][strategy] = counters["daily_strategy"].get(strategy, 0) + 1
                        counters["daily_symbol"][symbol] = counters["daily_symbol"].get(symbol, 0) + 1
                    if pnl < 0:
                        counters["streak_strategy"][strategy] = counters["streak_strategy"].get(strategy, 0) + 1
                        counters["streak_symbol"][symbol] = counters["streak_symbol"].get(symbol, 0) + 1
                    else:
                        counters["streak_strategy"][strategy] = 0
                        counters["streak_symbol"][symbol] = 0
        except Exception as exc:
            log.warning(f"[risk] Could not read ledger for circuit breakers: {exc}")
        return counters

    def _admin_warn_once(self, key: str, title: str, fields: dict[str, object]) -> None:
        if key in self._admin_warning_keys:
            return
        self._admin_warning_keys.add(key)
        log.warning(
            "[admin] %s: %s",
            title,
            "; ".join(f"{name}={value}" for name, value in fields.items()),
        )
        self._telegram.send_admin_event(title, fields=fields)
        self._append_ledger_event(
            "admin",
            event=title,
            warning_key=key,
            **fields,
        )

    def _circuit_breaker_reject_reason(self, symbol: str, sig: dict) -> str | None:
        if (
            MAX_DAILY_LOSSES_PER_STRATEGY <= 0
            and MAX_DAILY_LOSSES_PER_SYMBOL <= 0
            and MAX_CONSECUTIVE_LOSSES_PER_STRATEGY <= 0
            and MAX_CONSECUTIVE_LOSSES_PER_SYMBOL <= 0
        ):
            return None

        day_key = datetime.now(timezone.utc).date().isoformat()
        counters = self._ledger_loss_counters(day_key)
        strategy = self._strategy_key(sig.get("strategy"))
        symbol = symbol.upper()

        checks = [
            (
                "daily_strategy",
                strategy,
                MAX_DAILY_LOSSES_PER_STRATEGY,
                f"{strategy} has reached {MAX_DAILY_LOSSES_PER_STRATEGY} daily losses",
            ),
            (
                "daily_symbol",
                symbol,
                MAX_DAILY_LOSSES_PER_SYMBOL,
                f"{symbol} has reached {MAX_DAILY_LOSSES_PER_SYMBOL} daily losses",
            ),
            (
                "streak_strategy",
                strategy,
                MAX_CONSECUTIVE_LOSSES_PER_STRATEGY,
                f"{strategy} has reached {MAX_CONSECUTIVE_LOSSES_PER_STRATEGY} consecutive losses",
            ),
            (
                "streak_symbol",
                symbol,
                MAX_CONSECUTIVE_LOSSES_PER_SYMBOL,
                f"{symbol} has reached {MAX_CONSECUTIVE_LOSSES_PER_SYMBOL} consecutive losses",
            ),
        ]
        for bucket, key, limit, reason in checks:
            if limit <= 0:
                continue
            count = counters[bucket].get(key, 0)
            if count >= limit:
                warn_key = f"circuit:{bucket}:{key}:{day_key}:{count}"
                self._admin_warn_once(
                    warn_key,
                    "circuit breaker active",
                    {
                        "Scope": bucket,
                        "Key": key,
                        "Count": count,
                        "Limit": limit,
                        "Action": "new signal rejected",
                    },
                )
                return reason
        return None

    def _maybe_send_daily_heartbeat(self) -> None:
        now = datetime.now(timezone.utc)
        if (
            now.hour < DAILY_HEARTBEAT_UTC_HOUR
            or (
                now.hour == DAILY_HEARTBEAT_UTC_HOUR
                and now.minute < DAILY_HEARTBEAT_UTC_MINUTE
            )
        ):
            return
        day_key = (now - timedelta(days=1)).date().isoformat()
        if self._last_daily_heartbeat_key == day_key:
            return
        stats = self._ledger_stats_for_day(day_key)
        closed = int(stats["closed"])
        wins = int(stats["wins"])
        losses = int(stats["losses"])
        breakeven = int(stats["breakeven"])
        winrate = (wins / closed * 100.0) if closed else 0.0
        fields = {
            "Date UTC": day_key,
            "Combined": (
                f"PnL {float(stats['pnl']):.2f} | trades {closed} | WR {winrate:.1f}% | "
                f"W/L/BE {wins}/{losses}/{breakeven} | A/R {stats['accepted']}/{stats['rejected']}"
            ),
            "Closed PnL": f"{float(stats['pnl']):.2f}",
            "Closed trades": closed,
            "Winrate": f"{winrate:.1f}%",
            "Wins / Losses / BE": f"{wins}/{losses}/{breakeven}",
            "Accepted / Rejected signals": f"{stats['accepted']}/{stats['rejected']}",
            "Open positions now": self._open_count,
            "DD blocked": self._dd_blocked,
        }
        try:
            fields["Current equity"] = f"{get_equity(self._http):.2f}"
        except Exception as exc:
            fields["Current equity"] = f"unavailable: {exc}"
        strategies = stats.get("strategies", {})
        if isinstance(strategies, dict):
            for strategy, strategy_stats in sorted(strategies.items()):
                if not isinstance(strategy_stats, dict):
                    continue
                s_closed = int(strategy_stats.get("closed", 0) or 0)
                s_wins = int(strategy_stats.get("wins", 0) or 0)
                s_losses = int(strategy_stats.get("losses", 0) or 0)
                s_be = int(strategy_stats.get("breakeven", 0) or 0)
                s_wr = (s_wins / s_closed * 100.0) if s_closed else 0.0
                fields[f"Strategy {strategy}"] = (
                    f"PnL {float(strategy_stats.get('pnl', 0.0) or 0.0):.2f} | "
                    f"trades {s_closed} | WR {s_wr:.1f}% | "
                    f"W/L/BE {s_wins}/{s_losses}/{s_be} | "
                    f"A/R {strategy_stats.get('accepted', 0)}/{strategy_stats.get('rejected', 0)}"
                )
        self._telegram.send_daily_heartbeat(fields=fields)
        self._append_ledger_event("heartbeat", event="daily", reported_day=day_key, **fields)
        self._last_daily_heartbeat_key = day_key
        self._save_heartbeat_state()

    # -- WebSocket callback ----------------------------------------------------

    def _mark_ws_message(self) -> None:
        self._last_ws_message_ts = time.time()

    def _on_kline(self, msg: dict) -> None:
        try:
            data = msg.get("data")
            if not data:
                return
            self._mark_ws_message()
            candle = data[0]
            if not candle.get("confirm", False):
                return

            parts = msg.get("topic", "").split(".")
            if len(parts) != 3:
                return
            sym   = parts[2]
            state = self._states.get(sym)
            if state is None:
                return

            bar = {
                "ts":     int(candle["start"]),
                "open":   float(candle["open"]),
                "high":   float(candle["high"]),
                "low":    float(candle["low"]),
                "close":  float(candle["close"]),
                "volume": float(candle["volume"]),
            }
            state.push_bar(bar)
            log.debug(f"[{sym}] bar closed @ {bar['close']} vol={bar['volume']:.0f}")

            if not state.in_position:
                threading.Thread(target=self._check_and_trade,
                                 args=(state,), daemon=True).start()
        except Exception:
            log.exception("Error in kline callback")

    def _on_strategy5_kline(self, msg: dict) -> None:
        try:
            data = msg.get("data")
            if not data:
                return
            self._mark_ws_message()
            candle = data[0]
            if not candle.get("confirm", False):
                return
            parts = msg.get("topic", "").split(".")
            if len(parts) != 3:
                return
            sym = parts[2]
            turtle_state = self._turtle_states.get(sym)
            orb_state = self._session_orb_states.get(sym)
            trade_state = self._states.get(sym)
            if (turtle_state is None and orb_state is None) or trade_state is None:
                return

            bar = {
                "ts": int(candle["start"]),
                "open": float(candle["open"]),
                "high": float(candle["high"]),
                "low": float(candle["low"]),
                "close": float(candle["close"]),
                "volume": float(candle["volume"]),
            }
            if turtle_state is not None:
                turtle_state.push_bar(bar)
            if orb_state is not None:
                orb_state.push_bar(bar)

            if turtle_state is not None and self._turtle_engine is not None and not trade_state.in_position:
                threading.Thread(
                    target=self._check_turtle_and_trade,
                    args=(trade_state, turtle_state),
                    daemon=True,
                ).start()
            if orb_state is not None and self._session_orb_engine is not None and not trade_state.in_position:
                threading.Thread(
                    target=self._check_session_orb_and_trade,
                    args=(trade_state, orb_state),
                    daemon=True,
                ).start()
        except Exception:
            log.exception("Error in 5m strategy kline callback")

    # -- Signal -> DT filter -> order ------------------------------------------

    def _check_and_trade(self, state: SymbolState) -> None:
        if not state.mm_enabled:
            return
        bars = state.snapshot()
        sig  = detect_signal(bars, state.cfg)
        if sig is None:
            return
        sig["strategy"] = "million_moves"

        # DT filter
        if state.use_dt and state.dt_model is not None:
            is_long = sig["signal"] == "long"
            fvec = extract_live_feature_vector(bars, is_long)
            if fvec is None:
                log.debug(f"[{state.symbol}] DT: could not build feature vector -- passing through")
            else:
                sig["feature_columns"] = list(FEATURE_NAMES)
                sig["feature_snapshot"] = self._feature_snapshot(FEATURE_NAMES, fvec)
                prob = float(state.dt_model.predict_proba(fvec.reshape(1, -1))[0, 1])
                log.debug(f"[{state.symbol}] DT prob={prob:.3f} threshold={state.dt_threshold:.2f}")
                if prob < state.dt_threshold:
                    sig["dt_prob"] = prob
                    sig["dt_threshold"] = state.dt_threshold
                    reason = f"DT probability {prob:.3f} below threshold {state.dt_threshold:.2f}"
                    log.info(
                        f"[{state.symbol}] {sig['signal'].upper()} filtered by DT "
                        f"(prob={prob:.3f} < {state.dt_threshold:.2f})"
                    )
                    self._record_signal_event("rejected", symbol=state.symbol, sig=sig, reason=reason)
                    self._telegram.send_signal("rejected", symbol=state.symbol, sig=sig, reason=reason)
                    return

        self._submit_signal(state, sig)

    def _check_turtle_and_trade(self, trade_state: SymbolState, turtle_state: TurtleSoupState) -> None:
        if self._turtle_engine is None:
            return
        sig = self._turtle_engine.detect_signal(turtle_state)
        if sig is None:
            return
        if sig.get("rejected"):
            reason = str(sig.get("reject_reason", "Rejected by Turtle Soup filter"))
            log.info(
                f"[{trade_state.symbol}] TURTLE candidate {sig['signal'].upper()} rejected "
                f"-- {reason}"
            )
            self._record_signal_event("rejected", symbol=trade_state.symbol, sig=sig, reason=reason)
            self._telegram.send_signal("rejected", symbol=trade_state.symbol, sig=sig, reason=reason)
            return
        log.info(
            f"[{trade_state.symbol}] TURTLE candidate {sig['signal'].upper()} "
            f"prob={sig.get('prob', float('nan')):.3f} threshold={sig.get('threshold', float('nan')):.2f}"
        )
        self._submit_signal(trade_state, sig)

    def _check_session_orb_and_trade(self, trade_state: SymbolState, orb_state: SessionOrbState) -> None:
        if self._session_orb_engine is None:
            return
        sig = self._session_orb_engine.detect_signal(orb_state)
        if sig is None:
            return
        if sig.get("rejected"):
            reason = str(sig.get("reject_reason", "Rejected by Session ORB filter"))
            log.info(
                f"[{trade_state.symbol}] SESSION_ORB candidate {sig['signal'].upper()} rejected "
                f"-- {reason}"
            )
            self._record_signal_event("rejected", symbol=trade_state.symbol, sig=sig, reason=reason)
            self._telegram.send_signal("rejected", symbol=trade_state.symbol, sig=sig, reason=reason)
            return
        log.info(
            f"[{trade_state.symbol}] SESSION_ORB candidate {sig['signal'].upper()} "
            f"prob={sig.get('prob', float('nan')):.3f} threshold={sig.get('threshold', float('nan')):.2f} "
            f"session={sig.get('session', '-')} or={sig.get('or_minutes', '-')}"
        )
        self._submit_signal(trade_state, sig)

    @staticmethod
    def _stop_distance_reject_reason(sig: dict) -> str | None:
        try:
            entry = float(sig.get("entry"))
            stop = float(sig.get("sl"))
        except (TypeError, ValueError):
            return None
        if entry <= 0 or not np.isfinite(entry) or not np.isfinite(stop):
            return None
        distance_pct = abs(entry - stop) / entry
        sig["stop_distance_pct"] = distance_pct
        if MIN_STOP_DISTANCE_PCT > 0 and distance_pct < MIN_STOP_DISTANCE_PCT:
            return (
                f"SL distance {distance_pct:.4%} below minimum "
                f"{MIN_STOP_DISTANCE_PCT:.4%}"
            )
        return None

    @staticmethod
    def _fee_drag_reject_reason(sig: dict) -> str | None:
        if TAKER_FEE_RATE <= 0 or MAX_FEE_TO_PRICE_RISK <= 0:
            return None
        try:
            entry = float(sig.get("entry"))
            stop = float(sig.get("sl"))
        except (TypeError, ValueError):
            return None
        if entry <= 0 or stop <= 0 or not np.isfinite(entry) or not np.isfinite(stop):
            return None
        price_risk = abs(entry - stop)
        if price_risk <= 0:
            return None
        fee_risk_per_unit = TAKER_FEE_RATE * (entry + stop)
        fee_to_price_risk = fee_risk_per_unit / price_risk
        sig["fee_to_price_risk"] = fee_to_price_risk
        sig["fee_risk_per_unit"] = fee_risk_per_unit
        if fee_to_price_risk > MAX_FEE_TO_PRICE_RISK:
            return (
                f"estimated round-trip fee is {fee_to_price_risk:.1%} of raw stop risk "
                f"(limit {MAX_FEE_TO_PRICE_RISK:.1%})"
            )
        return None

    @staticmethod
    def _session_orb_weekend_reject_reason(sig: dict) -> str | None:
        if not SESSION_ORB_BLOCK_WEEKEND_SESSIONS:
            return None
        if str(sig.get("strategy", "")).lower() != "session_orb_judas_fvg":
            return None
        session = str(sig.get("session", "")).strip().lower()
        if session not in SESSION_ORB_BLOCK_WEEKEND_SESSIONS:
            return None
        raw_time = sig.get("entry_time")
        try:
            if raw_time:
                text = str(raw_time).replace("Z", "+00:00")
                entry_time = datetime.fromisoformat(text)
                if entry_time.tzinfo is None:
                    entry_time = entry_time.replace(tzinfo=timezone.utc)
                entry_time = entry_time.astimezone(timezone.utc)
            else:
                entry_time = datetime.now(timezone.utc)
        except (TypeError, ValueError):
            entry_time = datetime.now(timezone.utc)
        if entry_time.weekday() >= 5:
            sig["weekend_session_blocked"] = f"{entry_time.date()} {session}"
            return f"Session ORB {session} session blocked on UTC weekend ({entry_time.date()})"
        return None

    def _fetch_market_snapshot(self, symbol: str) -> dict:
        """Fetch full ticker info for a symbol from Bybit. Returns empty dict on failure."""
        try:
            resp = self._http.get_tickers(category="linear", symbol=symbol)
            items = self._extract_response_list(resp)
            if items:
                return dict(items[0])
        except Exception as exc:
            log.debug(f"[{symbol}] market snapshot fetch failed: {exc}")
        return {}

    def _submit_signal(self, state: SymbolState, sig: dict) -> None:
        market_context = self._fetch_market_context(state, sig)
        sig["market_context"] = market_context
        market_snapshot = market_context.get("ticker") if isinstance(market_context, dict) else None
        if isinstance(market_snapshot, dict) and market_snapshot:
            sig["market_snapshot"] = market_snapshot
        reject_reason: str | None = None
        stop_distance_reason = self._stop_distance_reject_reason(sig)
        if stop_distance_reason:
            reject_reason = stop_distance_reason
            log.info(
                f"[{state.symbol}] {sig['signal'].upper()} {sig.get('strategy', 'strategy')} signal skipped "
                f"-- {reject_reason}"
            )
        if reject_reason is None:
            fee_drag_reason = self._fee_drag_reject_reason(sig)
            if fee_drag_reason:
                reject_reason = fee_drag_reason
                log.info(
                    f"[{state.symbol}] {sig['signal'].upper()} {sig.get('strategy', 'strategy')} signal skipped "
                    f"-- {reject_reason}"
                )
        if reject_reason is None:
            weekend_reason = self._session_orb_weekend_reject_reason(sig)
            if weekend_reason:
                reject_reason = weekend_reason
                log.info(
                    f"[{state.symbol}] {sig['signal'].upper()} {sig.get('strategy', 'strategy')} signal skipped "
                    f"-- {reject_reason}"
                )
        if reject_reason is None:
            circuit_reason = self._circuit_breaker_reject_reason(state.symbol, sig)
            if circuit_reason:
                reject_reason = circuit_reason
                log.info(
                    f"[{state.symbol}] {sig['signal'].upper()} {sig.get('strategy', 'strategy')} signal skipped "
                    f"-- {reject_reason}"
                )

        if reject_reason:
            self._record_signal_event("rejected", symbol=state.symbol, sig=sig, reason=reject_reason)
            self._telegram.send_signal("rejected", symbol=state.symbol, sig=sig, reason=reject_reason)
            return

        with self._pos_lock:
            if state.in_position:
                reject_reason = "symbol already has a tracked open position"
            elif self._dd_blocked:
                reason = self._dd_block_reason or "drawdown"
                reject_reason = f"{reason.lower()} drawdown cooldown active"
                log.info(
                    f"[{state.symbol}] {sig['signal'].upper()} {sig.get('strategy', 'strategy')} signal skipped "
                    f"-- {reject_reason}"
                )
            elif self._open_count >= MAX_OPEN:
                reject_reason = f"MAX_OPEN={MAX_OPEN} reached"
                log.info(
                    f"[{state.symbol}] {sig['signal'].upper()} {sig.get('strategy', 'strategy')} signal skipped "
                    f"-- {reject_reason}"
                )
            elif state.symbol in CLUSTER_A and self._cluster_a_count >= CLUSTER_A_MAX:
                reject_reason = (
                    f"CLUSTER_A_MAX={CLUSTER_A_MAX} reached "
                    f"({self._cluster_a_count} correlated positions open)"
                )
                log.info(
                    f"[{state.symbol}] {sig['signal'].upper()} {sig.get('strategy', 'strategy')} signal skipped "
                    f"-- {reject_reason}"
                )
            else:
                state.in_position   = True
                state.position_side = "Buy" if sig["signal"] == "long" else "Sell"
                state.pending_entry_until_ms = self._now_ms() + int(PRIVATE_POSITION_ENTRY_DEBOUNCE_SECONDS * 1000)
                self._open_count   += 1
                if state.symbol in CLUSTER_A:
                    self._cluster_a_count += 1

        if reject_reason:
            self._record_signal_event("rejected", symbol=state.symbol, sig=sig, reason=reject_reason)
            self._telegram.send_signal("rejected", symbol=state.symbol, sig=sig, reason=reject_reason)
            return

        try:
            self._execute_trade(state, sig)
        except Exception as exc:
            log.error(f"[{state.symbol}] Trade failed: {exc}")
            with self._pos_lock:
                state.in_position   = False
                state.position_side = None
                state.active_trade = None
                state.pending_entry_until_ms = 0
                self._open_count = max(0, self._open_count - 1)
                if state.symbol in CLUSTER_A:
                    self._cluster_a_count = max(0, self._cluster_a_count - 1)
                self._save_active_trade_state_locked()
            self._record_signal_event(
                "rejected",
                symbol=state.symbol,
                sig=sig,
                reason=f"Trade failed: {exc}",
            )
            self._telegram.send_signal("rejected", symbol=state.symbol, sig=sig, reason=f"Trade failed: {exc}")

    def _set_order_leverage(
        self,
        state: SymbolState,
        entry: float,
        stop: float,
    ) -> float:
        """Set leverage to the minimum required so that a stop-out loses exactly the
        margin allocated to this position (i.e. leverage = 1 / stop_distance_pct).
        Clamped to [min_leverage, max_leverage] and rounded up to the exchange step.
        """
        max_leverage = max(float(state.info.get("max_leverage", 1.0)), 1.0)
        min_leverage = max(float(state.info.get("min_leverage", 1.0)), 1.0)
        leverage_step = float(state.info.get("leverage_step", 0.01))

        stop_distance_pct = abs(entry - stop) / entry if entry > 0 else 0.0
        risk_leverage = (1.0 / stop_distance_pct) if stop_distance_pct > 0 else max_leverage
        target_leverage = min(max_leverage, max(min_leverage, ceil_to_step(risk_leverage, leverage_step)))
        target_text = qty_to_str(target_leverage)

        try:
            leverage_resp = self._http.set_leverage(
                category="linear",
                symbol=state.symbol,
                buyLeverage=target_text,
                sellLeverage=target_text,
            )
        except Exception as exc:
            if "leverage not modified" in str(exc):
                log.debug(f"[{state.symbol}] Order leverage already {target_text}x")
                return target_leverage
            raise RuntimeError(f"Failed to set order leverage {target_text}x: {exc}") from exc

        if leverage_resp.get("retCode", 0) != 0:
            ret_msg = str(leverage_resp.get("retMsg", "?"))
            if "leverage not modified" in ret_msg.lower():
                log.debug(f"[{state.symbol}] Order leverage already {target_text}x")
                return target_leverage
            raise RuntimeError(
                f"Failed to set order leverage {target_text}x "
                f"(retCode={leverage_resp.get('retCode')}): {ret_msg}"
            )

        log.info(
            f"[{state.symbol}] Order leverage set to {target_text}x "
            f"(sl_dist={stop_distance_pct:.4%}, risk_based={risk_leverage:.2f}x, cap={max_leverage:g}x)"
        )
        return target_leverage

    def _sync_fixed_exit_orders(self, sym: str, *, sl_price: float, tp_price: float) -> bool:
        """Explicitly set full-position fixed TP/SL after the market entry exists."""
        for attempt in range(1, 4):
            try:
                resp = self._http.set_trading_stop(
                    category="linear",
                    symbol=sym,
                    tpslMode="Full",
                    stopLoss=str(sl_price),
                    takeProfit=str(tp_price),
                    slTriggerBy="LastPrice",
                    tpTriggerBy="LastPrice",
                    positionIdx=0,
                )
                ret_code = int(resp.get("retCode", -1) or -1)
                ret_msg = str(resp.get("retMsg", "?"))
                if ret_code == 0:
                    log.info(f"[{sym}] Fixed TP/SL synchronized  sl={sl_price} tp={tp_price}")
                    return True
                if ret_code == 34040 or "not modified" in ret_msg.lower():
                    log.debug(f"[{sym}] Fixed TP/SL already set  sl={sl_price} tp={tp_price}")
                    return True
                log.warning(
                    f"[{sym}] set fixed TP/SL attempt {attempt}/3 failed: "
                    f"{ret_msg} (retCode={ret_code})"
                )
            except Exception as exc:
                exc_msg = str(exc)
                exc_lower = exc_msg.lower()
                if "not modified" in exc_lower or "34040" in exc_lower:
                    log.debug(f"[{sym}] Fixed TP/SL already set  sl={sl_price} tp={tp_price}")
                    return True
                log.warning(f"[{sym}] set fixed TP/SL attempt {attempt}/3 failed: {exc}")
            time.sleep(0.8)

        self._telegram.send_risk_event(
            "protection setup failed",
            fields={
                "Symbol": sym,
                "Stop Loss": sl_price,
                "Target": tp_price,
                "Action": "manual inspection recommended",
            },
        )
        return False

    def _execute_trade(self, state: SymbolState, sig: dict) -> None:
        sym    = state.symbol
        side   = "Buy" if sig["signal"] == "long" else "Sell"
        tick   = state.info["tick_size"]
        q_step = state.info["qty_step"]
        min_q  = state.info["min_qty"]
        max_leverage = max(float(state.info.get("max_leverage", 1.0)), 1.0)

        balances = get_balance_metrics(self._http)
        equity = float(balances.get("equity", 0.0))
        available_balance = float(balances.get("available", 0.0))
        if equity <= 0:
            raise ValueError(f"Invalid equity returned: {equity}")
        entry = float(sig["entry"])
        stop = float(sig["sl"])
        unit_risk = abs(entry - stop)
        if entry <= 0 or unit_risk <= 0 or not np.isfinite(unit_risk):
            raise ValueError(f"Invalid entry/SL for sizing: entry={entry} sl={stop}")

        risk_budget = equity * NOTIONAL_PCT
        fee_risk_per_unit = max(TAKER_FEE_RATE, 0.0) * (entry + stop)
        unit_risk_with_fees = unit_risk + fee_risk_per_unit
        raw_qty = risk_budget / unit_risk_with_fees
        margin_safety_buffer = 0.95
        margin_basis = available_balance if available_balance > 0 else equity
        max_qty_by_margin = (margin_basis * max_leverage * margin_safety_buffer) / entry
        if raw_qty > max_qty_by_margin:
            log.warning(
                f"[{sym}] Risk-sized qty capped by available margin: "
                f"raw_qty={raw_qty:.8g} max_qty={max_qty_by_margin:.8g} "
                f"max_leverage={max_leverage:g}x available={available_balance:.2f}"
            )
            raw_qty = max_qty_by_margin

        qty = floor_to_step(raw_qty, q_step)
        if qty < min_q:
            qty = min_q
        notional = qty * entry
        expected_price_sl_loss = qty * unit_risk
        expected_fee_loss = qty * fee_risk_per_unit
        expected_sl_loss = expected_price_sl_loss + expected_fee_loss
        order_leverage = self._set_order_leverage(
            state,
            entry=entry,
            stop=stop,
        )
        margin_est = notional / order_leverage
        if expected_sl_loss > risk_budget * 1.02:
            log.warning(
                f"[{sym}] Minimum/order-step sizing exceeds target SL risk: "
                f"target={risk_budget:.2f} actual~{expected_sl_loss:.2f}"
            )

        sl_price   = round_to_step(sig["sl"], tick)
        tp1_price  = round_to_step(sig["tp1"], tick)
        trail_dist = round_to_step(float(sig.get("trail_dist", 0.0)), tick)
        strategy = sig.get("strategy", "million_moves")
        exit_style = sig.get("exit_style", "trailing")
        order_link_id = self._make_order_link_id(
            kind="E",
            strategy=strategy,
            symbol=sym,
            direction=sig["signal"],
        )
        sig["order_link_id"] = order_link_id

        dt_tag = " [DT]" if (state.use_dt and state.dt_model is not None) else ""
        log.info(
            f"[{sym}] SIGNAL {sig['signal'].upper()} {strategy}{dt_tag}  "
            f"entry~{sig['entry']:.5g}  sl={sl_price}  "
            f"tp1={tp1_price}  trail_dist={trail_dist}  "
            f"qty={qty}  notional={notional:.2f}  margin~{margin_est:.2f}  "
            f"lev={order_leverage:g}x  "
            f"price_risk~{expected_price_sl_loss:.2f}  fees~{expected_fee_loss:.2f}  "
            f"risk_at_sl~{expected_sl_loss:.2f} ({expected_sl_loss / equity:.2%})  "
            f"equity={equity:.2f}  available={available_balance:.2f}"
        )

        order_kwargs = dict(
            category    = "linear",
            symbol      = sym,
            side        = side,
            orderType   = "Market",
            qty         = qty_to_str(qty, q_step),
            stopLoss    = str(sl_price),
            slTriggerBy = "LastPrice",
            positionIdx = 0,
            orderLinkId = order_link_id,
        )
        if exit_style == "fixed_tp":
            order_kwargs["takeProfit"] = str(tp1_price)
            order_kwargs["tpTriggerBy"] = "LastPrice"
        sig["order_request"] = dict(order_kwargs)

        order_resp = self._http.place_order(**order_kwargs)
        ret_code = order_resp.get("retCode", -1)
        if ret_code != 0:
            sig["order_error"] = {
                "retCode": ret_code,
                "retMsg": order_resp.get("retMsg", "?"),
                "retExtInfo": order_resp.get("retExtInfo"),
                "response": order_resp,
            }
            raise RuntimeError(
                f"Order rejected (retCode={ret_code}): {order_resp.get('retMsg', '?')}"
            )
        order_id = order_resp.get("result", {}).get("orderId", "?")
        log.info(f"[{sym}] Market order accepted  orderId={order_id} orderLinkId={order_link_id}")
        with self._pos_lock:
            state.active_trade = {
                "strategy": strategy,
                "direction": sig["signal"],
                "order_link_id": order_link_id,
                "entry": entry,
                "sl": sl_price,
                "tp1": tp1_price,
                "trail_dist": trail_dist,
                "qty": qty_to_str(qty),
                "notional": f"{notional:.2f}",
                "risk_at_sl": f"{expected_sl_loss:.2f}",
                "price_risk_at_sl": f"{expected_price_sl_loss:.2f}",
                "estimated_fees": f"{expected_fee_loss:.2f}",
                "exit_style": exit_style,
                "entry_order_id": str(order_id),
                "opened_at": int(time.time() * 1000),
                "available_balance": f"{available_balance:.2f}",
            }
            state.pending_entry_until_ms = self._now_ms() + int(PRIVATE_POSITION_ENTRY_DEBOUNCE_SECONDS * 1000)
            self._save_active_trade_state_locked()
        notify_sig = {
            **sig,
            "entry": entry,
            "sl": sl_price,
            "tp1": tp1_price,
        }
        self._telegram.send_signal(
            "accepted",
            symbol=sym,
            sig=notify_sig,
            order_id=str(order_id),
            extra={
                "Qty": qty_to_str(qty),
                "Notional": f"{notional:.2f}",
                "Order leverage": f"{order_leverage:g}x",
                "Order link id": order_link_id,
                "Price risk": f"{expected_price_sl_loss:.2f}",
                "Estimated fees": f"{expected_fee_loss:.2f}",
                "Risk at SL": f"{expected_sl_loss:.2f} ({expected_sl_loss / equity:.2%})",
                "Exit style": exit_style,
            },
        )
        self._record_signal_event(
            "accepted",
            symbol=sym,
            sig=notify_sig,
            order_id=str(order_id),
            extra={
                "qty": qty_to_str(qty),
                "notional": f"{notional:.2f}",
                "order_leverage": f"{order_leverage:g}",
                "order_link_id": order_link_id,
                "risk_at_sl": f"{expected_sl_loss:.2f}",
                "exit_style": exit_style,
                "available_balance": f"{available_balance:.2f}",
                "order_request": dict(order_kwargs),
            },
        )

        if exit_style == "fixed_tp":
            time.sleep(0.8)
            self._sync_fixed_exit_orders(sym, sl_price=sl_price, tp_price=tp1_price)
            return

        time.sleep(0.8)

        try:
            ts_resp = self._http.set_trading_stop(
                category     = "linear",
                symbol       = sym,
                stopLoss     = str(sl_price),
                slTriggerBy  = "LastPrice",
                trailingStop = str(trail_dist),
                activePrice  = str(tp1_price),
                tpslMode     = "Full",
                positionIdx  = 0,
            )
            if ts_resp.get("retCode", -1) != 0:
                log.warning(f"[{sym}] set_trading_stop: {ts_resp.get('retMsg', '?')}")
            else:
                log.info(
                    f"[{sym}] Hard SL/trailing stop set  sl={sl_price}  "
                    f"trail_dist={trail_dist}  activates_at={tp1_price}"
                )
        except Exception as exc:
            log.warning(f"[{sym}] Failed to set trailing stop: {exc}")

    # -- Main loop -------------------------------------------------------------

    def run(self) -> None:
        def _shutdown(sig_num, _frame):
            log.info("Shutdown signal received -- exiting.")
            raise SystemExit(0)

        signal.signal(signal.SIGTERM, _shutdown)
        signal.signal(signal.SIGINT,  _shutdown)

        while True:
            time.sleep(60)
            if WS_STALE_SECONDS > 0:
                stale_for = time.time() - self._last_ws_message_ts
                if stale_for > WS_STALE_SECONDS:
                    log.error(
                        f"WebSocket stale for {stale_for:.0f}s "
                        f"(limit={WS_STALE_SECONDS}s); exiting for Docker restart"
                    )
                    raise SystemExit(2)
            self._sync_positions()
            self._refresh_risk_state()
            self._audit_position_protection()
            self._audit_open_orders()
            self._maybe_send_daily_heartbeat()


# --- Entry point --------------------------------------------------------------

if __name__ == "__main__":
    Bot().run()
