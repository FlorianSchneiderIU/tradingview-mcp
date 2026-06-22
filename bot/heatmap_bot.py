#!/usr/bin/env python3
"""Bybit heatmap bot — three layers of liquidity / liquidation data in one service.

A fixed universe of Bybit linear perps (HEATMAP_SYMBOLS) is tracked across three layers,
each persisted to a SQLite DB on a shared named volume (survives restarts & rebuilds) and
served over REST so other fleet bots (and the Telegram bot) can fetch any of them:

  Layer 1 — ORDER-BOOK LIQUIDITY  (observable)
      Live L2 book from the public orderbook WebSocket. Each interval the resting size is
      binned around mid; a persistence/lifetime filter drops fleeting (spoof) levels so the
      heatmap reflects liquidity that actually rested. -> GET /v1/liquidity/{symbol}

  Layer 2 — ACTUAL LIQUIDATIONS   (ground truth)
      Real force-liquidation prints from the Bybit `allLiquidation.{symbol}` WebSocket
      (Sell print = long liquidation, Buy = short). Stored as events.
      -> GET /v1/liquidations/actual/{symbol}

  Layer 3 — PREDICTIVE LIQUIDATIONS  (estimate, Coinglass-style)
      A position-cohort model from public derivatives data: each candle, ΔOI (open-interest
      change) sizes newly opened notional, split long/short by taker imbalance (candle-range
      position proxy), distributed across leverage buckets; liquidation prices computed off
      MARK price; cohorts decay when OI falls and are consumed when mark price crosses their
      level. Computed for several timeframes (5m/15m/1h). -> GET /v1/liquidations/estimated/{symbol}?tf=1h

Public market data only — no API keys required. A predictive estimate is NOT ground truth;
Layer 2 is what actually happened and is the calibration target for Layer 3.

Threads: orderbook WS (L1) + allLiquidation WS (L2) feed in-memory state; an OB snapshotter
(L1) and a predictive worker (L3, REST-polls klines + open interest) compute + persist;
a liquidation-event flusher (L2) batches inserts; the main thread serves REST.
"""
from __future__ import annotations

import json
import logging
import math
import os
import signal as signal_module
import sqlite3
import threading
import time
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

import requests
from pybit.unified_trading import HTTP, WebSocket

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)sZ [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("heatmap")

CATEGORY = "linear"


# ── config ────────────────────────────────────────────────────────────────────
def _load_file_config(path: str) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        return {k: v for k, v in raw.items() if not k.startswith("_")}
    except FileNotFoundError:
        return {}
    except Exception as exc:  # noqa: BLE001
        log.warning("could not read config %s: %s", path, exc)
        return {}


_FILE_CFG = _load_file_config(os.environ.get("HEATMAP_CONFIG_PATH", "/app/configs/heatmap_configs.json"))


def _cfg(key: str, env: str, default: Any, cast) -> Any:
    if env in os.environ and os.environ[env].strip() != "":
        return cast(os.environ[env])
    if key in _FILE_CFG and _FILE_CFG[key] is not None:
        return cast(_FILE_CFG[key])
    return cast(default)


def _as_bool(v: Any) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _floats(value: Any, default: list[float]) -> list[float]:
    if value is None or value == "":
        return list(default)
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    return [float(x.strip()) for x in str(value).split(",") if x.strip()]


# Universe — fixed HEATMAP_SYMBOLS preferred; dynamic top-N fallback.
def _resolve_fixed_symbols() -> list[str]:
    env = os.environ.get("HEATMAP_SYMBOLS", "").strip()
    if env:
        return [s.strip().upper() for s in env.split(",") if s.strip()]
    cfg = _FILE_CFG.get("symbols")
    if isinstance(cfg, list) and cfg:
        return [str(s).strip().upper() for s in cfg if str(s).strip()]
    return []


FIXED_SYMBOLS = _resolve_fixed_symbols()
UNIVERSE_SIZE = _cfg("universe_size", "HEATMAP_UNIVERSE_SIZE", 50, int)  # fallback only
UNIVERSE_REFRESH_SECONDS = _cfg("universe_refresh_seconds", "HEATMAP_UNIVERSE_REFRESH_SECONDS", 3600, int)
SETTLE_COIN = _cfg("settle_coin", "HEATMAP_SETTLE_COIN", "USDT", str).upper()
TESTNET = _cfg("testnet", "HEATMAP_TESTNET", False, _as_bool)

# Layer 1 — order book
OB_ENABLED = _cfg("orderbook_enabled", "HEATMAP_ORDERBOOK_ENABLED", True, _as_bool)
OB_DEPTH = _cfg("depth", "HEATMAP_DEPTH", 200, int)
OB_WS_CHUNK = _cfg("ws_chunk", "HEATMAP_WS_CHUNK", 10, int)
OB_SNAPSHOT_INTERVAL = _cfg("ob_snapshot_interval_seconds", "HEATMAP_OB_SNAPSHOT_INTERVAL_SECONDS", 10, int)
OB_BIN_BPS = _cfg("ob_bin_bps", "HEATMAP_OB_BIN_BPS", 5.0, float)
OB_RANGE_PCT = _cfg("ob_range_pct", "HEATMAP_OB_RANGE_PCT", 0.02, float)
OB_MIN_LIFETIME_S = _cfg("ob_min_lifetime_seconds", "HEATMAP_OB_MIN_LIFETIME_SECONDS", 3.0, float)
OB_RETENTION_HOURS = _cfg("ob_retention_hours", "HEATMAP_OB_RETENTION_HOURS", 72, int)

# Layer 2 — actual liquidations
LIQ_ENABLED = _cfg("liquidations_enabled", "HEATMAP_LIQUIDATIONS_ENABLED", True, _as_bool)
LIQ_RETENTION_HOURS = _cfg("liq_retention_hours", "HEATMAP_LIQ_RETENTION_HOURS", 720, int)

# Layer 3 — predictive
PRED_ENABLED = _cfg("predictive_enabled", "HEATMAP_PREDICTIVE_ENABLED", True, _as_bool)
LEVERAGES = _floats(_FILE_CFG.get("leverages") or os.environ.get("HEATMAP_LEVERAGES"),
                    [2, 3, 5, 10, 20, 25, 50, 75, 100])
_LEV_WEIGHTS = _floats(_FILE_CFG.get("leverage_weights") or os.environ.get("HEATMAP_LEVERAGE_WEIGHTS"),
                       [0.05, 0.07, 0.12, 0.22, 0.24, 0.12, 0.10, 0.05, 0.03])
if len(_LEV_WEIGHTS) != len(LEVERAGES):
    _LEV_WEIGHTS = [1.0] * len(LEVERAGES)
_WSUM = sum(_LEV_WEIGHTS) or 1.0
LEVERAGE_WEIGHTS = [w / _WSUM for w in _LEV_WEIGHTS]
_DEFAULT_LEVERAGE_WEIGHTS = list(LEVERAGE_WEIGHTS)  # Dirichlet prior mean (the initial guess)
MMR = _cfg("maintenance_margin_rate", "HEATMAP_MMR", 0.005, float)
USE_OI = _cfg("use_open_interest", "HEATMAP_USE_OI", True, _as_bool)
PRED_BIN_PCT = _cfg("pred_bin_pct", "HEATMAP_PRED_BIN_PCT", 0.001, float)  # geometric bin width
_LOG_BIN = math.log1p(PRED_BIN_PCT)
PRED_UPDATE_INTERVAL = _cfg("pred_update_interval_seconds", "HEATMAP_PRED_UPDATE_INTERVAL_SECONDS", 60, int)
TOP_MAGNETS = _cfg("top_magnets", "HEATMAP_TOP_MAGNETS", 40, int)

# Auto-calibration: once/day, fit leverage weights to actual liquidation prints (mixture EM).
CALIB_ENABLED = _cfg("calibration_enabled", "HEATMAP_CALIBRATION_ENABLED", True, _as_bool)
CALIB_INTERVAL_HOURS = _cfg("calibration_interval_hours", "HEATMAP_CALIBRATION_INTERVAL_HOURS", 24.0, float)
CALIB_WINDOW_HOURS = _cfg("calib_window_hours", "HEATMAP_CALIB_WINDOW_HOURS", 72, int)
CALIB_TF = _cfg("calib_tf", "HEATMAP_CALIB_TF", "5m", str)
CALIB_MIN_EVENTS = _cfg("calib_min_events", "HEATMAP_CALIB_MIN_EVENTS", 200, int)
# Bayesian (Dirichlet) update: prior_strength = pseudo-event mass of the initial guess
# (larger = weights move slower); forget = decay applied to accumulated counts each run
# (1.0 = pure accumulation; <1.0 = exponential forgetting for regime adaptation).
CALIB_PRIOR_STRENGTH = _cfg("calib_prior_strength", "HEATMAP_CALIB_PRIOR_STRENGTH", 300.0, float)
CALIB_FORGET = _cfg("calib_forget", "HEATMAP_CALIB_FORGET", 1.0, float)
WEIGHTS_FILE = _cfg("weights_file", "HEATMAP_WEIGHTS_FILE", "/app/data/leverage_weights.json", str)


def _prior_alpha() -> list[float]:
    return [max(1e-6, CALIB_PRIOR_STRENGTH * w) for w in _DEFAULT_LEVERAGE_WEIGHTS]


# Layer 4 — buy/sell (delta) volume profile
VP_ENABLED = _cfg("volume_profile_enabled", "HEATMAP_VOLUME_PROFILE_ENABLED", True, _as_bool)
VP_BIN_PCT = _cfg("vp_bin_pct", "HEATMAP_VP_BIN_PCT", 0.0015, float)  # geometric price-bin width
_VP_LOG = math.log1p(VP_BIN_PCT)
VP_RETENTION_HOURS = _cfg("vp_retention_hours", "HEATMAP_VP_RETENTION_HOURS", 192, int)  # 8 days
VP_SEED_HOURS = _cfg("vp_seed_hours", "HEATMAP_VP_SEED_HOURS", 168, int)  # 7-day kline seed
VP_FLUSH_SECONDS = _cfg("vp_flush_seconds", "HEATMAP_VP_FLUSH_SECONDS", 10, int)
VP_VALUE_AREA = _cfg("vp_value_area", "HEATMAP_VP_VALUE_AREA", 0.70, float)
VP_TOP_LEVELS = _cfg("vp_top_levels", "HEATMAP_VP_TOP_LEVELS", 8, int)
VP_WINDOWS = {"4h", "24h", "7d", "daily", "weekly"}


def vp_bin_index(price: float) -> int:
    return int(math.floor(math.log(price) / _VP_LOG)) if price > 0 else 0


def vp_bin_low(idx: int) -> float:
    return math.exp(idx * _VP_LOG)


def vp_window_start(window: str, now: int) -> int:
    """Epoch-ms start for a profile window — rolling (4h/24h/7d) or UTC session-anchored
    (daily = since 00:00 UTC, weekly = since Monday 00:00 UTC)."""
    w = (window or "24h").lower()
    if w == "daily":
        return now - (now % 86_400_000)
    if w == "weekly":
        days = now // 86_400_000
        dow = (days + 3) % 7  # epoch day 0 (1970-01-01) was Thursday -> Monday-based index
        return (days - dow) * 86_400_000
    hours = {"4h": 4, "24h": 24, "7d": 168}.get(w, 24)
    return now - hours * 3_600_000

_DEFAULT_TFS = {
    "5m": {"interval": "5", "oi_interval": "5min", "lookback_hours": 72},
    "15m": {"interval": "15", "oi_interval": "15min", "lookback_hours": 168},
    "1h": {"interval": "60", "oi_interval": "1h", "lookback_hours": 720},
}
TIMEFRAMES: dict[str, dict[str, Any]] = _FILE_CFG.get("timeframes") or _DEFAULT_TFS

# Persistence / REST
DB_PATH = _cfg("db_path", "HEATMAP_DB_PATH", "/app/data/heatmap.db", str)
API_HOST = _cfg("api_host", "HEATMAP_API_HOST", "0.0.0.0", str)
API_PORT = _cfg("api_port", "HEATMAP_API_PORT", 8110, int)
DEFAULT_SERIES_LIMIT = _cfg("default_series_limit", "HEATMAP_DEFAULT_SERIES_LIMIT", 200, int)
MAX_SERIES_LIMIT = _cfg("max_series_limit", "HEATMAP_MAX_SERIES_LIMIT", 5000, int)

API_KEY = os.environ.get("HEATMAP_BYBIT_API_KEY", "").strip()
API_SECRET = os.environ.get("HEATMAP_BYBIT_API_SECRET", "").strip()

# Alerts + watchdog (heatmap-bot shares bot/.env.heatmap, so it has the TG token/uids)
TG_TOKEN = os.environ.get("HEATMAP_TG_BOT_TOKEN", os.environ.get("TELEGRAM_BOT_TOKEN", "")).strip()
TG_UIDS = [int(x) for x in os.environ.get("HEATMAP_TG_ALLOWED_UIDS", "").split(",") if x.strip().lstrip("-").isdigit()]
_admin = [int(x) for x in os.environ.get("HEATMAP_TG_ADMIN_UIDS", "").split(",") if x.strip().lstrip("-").isdigit()]
ADMIN_UIDS = _admin or TG_UIDS
ALERTS_ENABLED = _cfg("alerts_enabled", "HEATMAP_ALERTS_ENABLED", True, _as_bool)
ALERT_INTERVAL_S = _cfg("alert_interval_seconds", "HEATMAP_ALERT_INTERVAL_SECONDS", 15, int)
CASCADE_WINDOW_S = _cfg("cascade_window_seconds", "HEATMAP_CASCADE_WINDOW_SECONDS", 60, int)
CASCADE_MIN_USD = _cfg("cascade_min_usd", "HEATMAP_CASCADE_MIN_USD", 1_000_000.0, float)
CASCADE_COOLDOWN_S = _cfg("cascade_cooldown_seconds", "HEATMAP_CASCADE_COOLDOWN_SECONDS", 300, int)
PROX_PCT = _cfg("proximity_pct", "HEATMAP_PROXIMITY_PCT", 0.005, float)
PROX_COOLDOWN_S = _cfg("proximity_cooldown_seconds", "HEATMAP_PROXIMITY_COOLDOWN_SECONDS", 1800, int)
WATCHDOG_ENABLED = _cfg("watchdog_enabled", "HEATMAP_WATCHDOG_ENABLED", True, _as_bool)
WATCHDOG_STALE_S = _cfg("watchdog_stale_seconds", "HEATMAP_WATCHDOG_STALE_SECONDS", 120, int)
WATCHDOG_COOLDOWN_S = _cfg("watchdog_cooldown_seconds", "HEATMAP_WATCHDOG_COOLDOWN_SECONDS", 600, int)

_STARTED_AT_MS = 0


def _fmt_price(p: Optional[float]) -> str:
    if not p:
        return "?"
    if p >= 100:
        return f"{p:,.2f}"
    if p >= 1:
        return f"{p:,.4f}"
    return f"{p:.6f}"


def tg_send(text: str, chat_ids: list[int]) -> None:
    if not TG_TOKEN or not chat_ids:
        return
    for cid in chat_ids:
        try:
            requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                          json={"chat_id": cid, "text": text, "disable_web_page_preview": True}, timeout=10)
        except Exception:  # noqa: BLE001 - alerts are best-effort
            pass


def now_ms() -> int:
    return int(time.time() * 1000)


def bin_index(price: float) -> int:
    return int(math.floor(math.log(price) / _LOG_BIN)) if price > 0 else 0


def bin_low(idx: int) -> float:
    return math.exp(idx * _LOG_BIN)


def tf_minutes(tf: str) -> int:
    return max(1, int(TIMEFRAMES[tf]["interval"]))


# ════════════════════════════════════════════════════════════════════════════════
# Layer 1 — order-book liquidity
# ════════════════════════════════════════════════════════════════════════════════
class Book:
    """Per-symbol live L2 book; tracks first-seen time per price for the lifetime filter."""

    __slots__ = ("bids", "asks", "lock", "last_update_ms")

    def __init__(self) -> None:
        self.bids: dict[float, tuple[float, int]] = {}   # price -> (size, first_seen_ms)
        self.asks: dict[float, tuple[float, int]] = {}
        self.lock = threading.Lock()
        self.last_update_ms = 0


# ════════════════════════════════════════════════════════════════════════════════
# Layer 3 — predictive cohorts
# ════════════════════════════════════════════════════════════════════════════════
class Level:
    __slots__ = ("side", "bin_idx", "magnitude", "created_ts", "consumed_ts")

    def __init__(self, side: str, bin_idx: int, magnitude: float, created_ts: int) -> None:
        self.side = side
        self.bin_idx = bin_idx
        self.magnitude = magnitude
        self.created_ts = created_ts
        self.consumed_ts: Optional[int] = None


class MapState:
    """Predictive leverage map for one (symbol, timeframe)."""

    def __init__(self, symbol: str, tf: str) -> None:
        self.symbol = symbol
        self.tf = tf
        self.alive: dict[tuple[str, int], Level] = {}
        self.closed: list[Level] = []
        self.klines: list[tuple[int, float, float, float, float]] = []  # (ts, close(mark), high, low, new_notional)
        self.prev_close: Optional[float] = None
        self.prev_oi: Optional[float] = None
        self.last_ts: int = 0

    def apply_candle(self, ts: int, high: float, low: float, close: float,
                     oi: Optional[float], turnover: float) -> None:
        # 1) consume levels mark price traded through (relative to prior mark close)
        if self.prev_close is not None:
            dn_lo, dn_hi = low, self.prev_close      # longs liquidate as price falls
            up_lo, up_hi = self.prev_close, high      # shorts liquidate as price rises
            for key, lvl in list(self.alive.items()):
                p = bin_low(lvl.bin_idx)
                if (lvl.side == "long" and dn_lo <= p <= dn_hi) or \
                   (lvl.side == "short" and up_lo <= p <= up_hi):
                    lvl.consumed_ts = ts
                    self.closed.append(lvl)
                    del self.alive[key]

        # 2) size newly opened / closed positions from ΔOI (preferred) or turnover
        new_notional = 0.0
        if USE_OI and oi is not None and self.prev_oi is not None and close > 0:
            d_oi = oi - self.prev_oi
            if d_oi > 0:
                new_notional = d_oi * close
            elif d_oi < 0 and self.prev_oi > 0:
                # OI fell: positions closed/liquidated -> decay alive cohorts pro-rata
                frac = min(1.0, -d_oi / self.prev_oi)
                if frac > 0:
                    for key, lvl in list(self.alive.items()):
                        lvl.magnitude *= (1.0 - frac)
                        if lvl.magnitude <= 1e-9:
                            del self.alive[key]
        elif not USE_OI:
            new_notional = turnover

        if new_notional > 0 and close > 0 and high > low:
            buy_pressure = min(1.0, max(0.0, (close - low) / (high - low)))  # taker imbalance proxy
            long_notional = new_notional * buy_pressure
            short_notional = new_notional * (1.0 - buy_pressure)
            for lev, w in zip(LEVERAGES, LEVERAGE_WEIGHTS):
                self._add("long", close * (1.0 - 1.0 / lev + MMR), w * long_notional, ts)
                self._add("short", close * (1.0 + 1.0 / lev - MMR), w * short_notional, ts)

        self.prev_close = close
        self.prev_oi = oi if oi is not None else self.prev_oi
        self.last_ts = ts
        self.klines.append((ts, close, high, low, new_notional))

    def _add(self, side: str, price: float, mag: float, ts: int) -> None:
        if price <= 0 or mag <= 0:
            return
        key = (side, bin_index(price))
        lvl = self.alive.get(key)
        if lvl is None:
            self.alive[key] = Level(side, key[1], mag, ts)
        else:
            lvl.magnitude += mag

    def prune(self, window_start: int) -> None:
        self.closed = [l for l in self.closed if (l.consumed_ts or 0) >= window_start]
        self.klines = [k for k in self.klines if k[0] >= window_start]

    def levels(self) -> list[Level]:
        return list(self.alive.values()) + self.closed


# ════════════════════════════════════════════════════════════════════════════════
# Service
# ════════════════════════════════════════════════════════════════════════════════
class HeatmapService:
    def __init__(self) -> None:
        self.http = HTTP(testnet=TESTNET)
        self.universe: list[str] = []
        self.universe_meta: dict[str, dict[str, Any]] = {}
        self.state_lock = threading.Lock()
        self.stop = threading.Event()

        # Layer 1
        self.books: dict[str, Book] = {}
        self.ob_subscribed: set[str] = set()
        # Layer 2
        self.liq_buffer: deque[tuple] = deque(maxlen=100_000)
        self.liq_subscribed: set[str] = set()
        # Layer 4 (volume profile)
        self.vp_accum: dict[tuple[str, int, int], list[float]] = {}  # (sym,hour,bin)->[total,delta]
        self.vp_lock = threading.Lock()
        self.vp_subscribed: set[str] = set()
        self.vp_seeded: set[str] = set()
        self._vp_db: Optional[sqlite3.Connection] = None
        self.last_trade_ms = 0  # for the watchdog
        # Layer 3
        self.states: dict[tuple[str, str], MapState] = {}
        # Calibration (Dirichlet posterior over leverage buckets)
        self.needs_rebuild = threading.Event()
        self.last_calibration: dict[str, Any] = {}
        self.alpha: list[float] = _prior_alpha()   # Dirichlet pseudo-counts
        self.last_event_ts: int = 0

        self.ws: Optional[WebSocket] = None
        self._ob_db: Optional[sqlite3.Connection] = None
        self._pred_db: Optional[sqlite3.Connection] = None
        self._liq_db: Optional[sqlite3.Connection] = None

    # ── database ────────────────────────────────────────────────────────────────
    @staticmethod
    def _connect(read_only: bool = False) -> sqlite3.Connection:
        if read_only:
            conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=10.0)
        else:
            conn = sqlite3.connect(DB_PATH, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def init_db(self) -> None:
        Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS universe (
                    symbol TEXT PRIMARY KEY, rank INTEGER, turnover24h REAL, updated_ts INTEGER
                );

                -- Layer 1: order-book liquidity
                CREATE TABLE IF NOT EXISTS ob_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL, ts INTEGER NOT NULL,
                    mid REAL, best_bid REAL, best_ask REAL, spread_bps REAL,
                    bid_notional_total REAL, ask_notional_total REAL
                );
                CREATE INDEX IF NOT EXISTS idx_ob_snap ON ob_snapshots(symbol, ts);
                CREATE TABLE IF NOT EXISTS ob_bins (
                    snapshot_id INTEGER NOT NULL, symbol TEXT NOT NULL, ts INTEGER NOT NULL,
                    price_low REAL, price_high REAL, bid_qty REAL, ask_qty REAL,
                    bid_notional REAL, ask_notional REAL
                );
                CREATE INDEX IF NOT EXISTS idx_ob_bins ON ob_bins(symbol, ts);

                -- Layer 2: actual liquidations
                CREATE TABLE IF NOT EXISTS liq_events (
                    ts INTEGER NOT NULL, symbol TEXT NOT NULL, side TEXT, pos_side TEXT,
                    price REAL, qty REAL, notional REAL
                );
                CREATE INDEX IF NOT EXISTS idx_liq_events ON liq_events(symbol, ts);

                -- Layer 3: predictive levels
                CREATE TABLE IF NOT EXISTS liq_levels (
                    symbol TEXT NOT NULL, tf TEXT NOT NULL, side TEXT NOT NULL, bin_idx INTEGER NOT NULL,
                    price_low REAL, price_high REAL, magnitude REAL, created_ts INTEGER, consumed_ts INTEGER
                );
                CREATE INDEX IF NOT EXISTS idx_levels ON liq_levels(symbol, tf);
                CREATE TABLE IF NOT EXISTS liq_klines (
                    symbol TEXT NOT NULL, tf TEXT NOT NULL, ts INTEGER NOT NULL,
                    close REAL, high REAL, low REAL, new_notional REAL
                );
                CREATE INDEX IF NOT EXISTS idx_liq_klines ON liq_klines(symbol, tf, ts);

                -- Layer 4: buy/sell volume profile (hourly buckets, geometric price bins)
                CREATE TABLE IF NOT EXISTS vp_buckets (
                    symbol TEXT NOT NULL, hour_bucket INTEGER NOT NULL, bin_idx INTEGER NOT NULL,
                    total REAL, delta REAL,
                    PRIMARY KEY (symbol, hour_bucket, bin_idx)
                );
                CREATE INDEX IF NOT EXISTS idx_vp ON vp_buckets(symbol, hour_bucket);

                -- Proximity-alert subscriptions (uid x symbol)
                CREATE TABLE IF NOT EXISTS watches (
                    uid INTEGER NOT NULL, symbol TEXT NOT NULL, PRIMARY KEY (uid, symbol)
                );

                -- Auto-calibration audit trail
                CREATE TABLE IF NOT EXISTS calibration_runs (
                    ts INTEGER NOT NULL, tf TEXT, n_events INTEGER, n_explained INTEGER,
                    explained_frac REAL, applied INTEGER, leverages_json TEXT, weights_json TEXT
                );
                """
            )
            # migrate older DBs that predate new_notional
            try:
                conn.execute("ALTER TABLE liq_klines ADD COLUMN new_notional REAL")
            except sqlite3.OperationalError:
                pass
            conn.commit()
        finally:
            conn.close()

    # ── universe ────────────────────────────────────────────────────────────────
    def _turnover_map(self) -> dict[str, float]:
        out: dict[str, float] = {}
        try:
            tickers = self.http.get_tickers(category=CATEGORY)
        except Exception as exc:  # noqa: BLE001
            log.warning("get_tickers failed: %s", exc)
            return out
        for item in (tickers.get("result", {}) or {}).get("list", []) or []:
            sym = item.get("symbol")
            if sym:
                try:
                    out[sym] = float(item.get("turnover24h") or 0.0)
                except (TypeError, ValueError):
                    out[sym] = 0.0
        return out

    def resolve_universe(self) -> list[str]:
        updated = now_ms()
        if FIXED_SYMBOLS:
            turnover = self._turnover_map()
            symbols = FIXED_SYMBOLS
            meta = {s: {"rank": i + 1, "turnover24h": turnover.get(s, 0.0)} for i, s in enumerate(symbols)}
        else:
            instruments = self.http.get_instruments_info(category=CATEGORY)
            tradable = {
                it.get("symbol") for it in (instruments.get("result", {}) or {}).get("list", []) or []
                if it.get("status") == "Trading" and (it.get("quoteCoin") or "").upper() == SETTLE_COIN
            }
            ranked = [(s, t) for s, t in self._turnover_map().items() if s in tradable]
            ranked.sort(key=lambda x: x[1], reverse=True)
            top = ranked[: max(1, UNIVERSE_SIZE)]
            meta = {s: {"rank": i + 1, "turnover24h": t} for i, (s, t) in enumerate(top)}
            symbols = [s for s, _ in top]

        with self.state_lock:
            self.universe = symbols
            self.universe_meta = meta
            for s in symbols:
                self.books.setdefault(s, Book())

        conn = self._connect()
        try:
            conn.execute("DELETE FROM universe")
            conn.executemany(
                "INSERT INTO universe(symbol, rank, turnover24h, updated_ts) VALUES (?,?,?,?)",
                [(s, meta[s]["rank"], meta[s]["turnover24h"], updated) for s in symbols],
            )
            conn.commit()
        finally:
            conn.close()
        log.info("universe resolved: %d symbols (top=%s)", len(symbols), ", ".join(symbols[:5]))
        return symbols

    # ── websockets (L1 + L2) ──────────────────────────────────────────────────--
    def start_ws(self) -> None:
        if self.ws is None:
            self.ws = WebSocket(testnet=TESTNET, channel_type=CATEGORY)

    def subscribe_ws(self) -> None:
        self.start_ws()
        with self.state_lock:
            symbols = list(self.universe)
        if OB_ENABLED:
            pending = [s for s in symbols if s not in self.ob_subscribed]
            for i in range(0, len(pending), OB_WS_CHUNK):
                chunk = pending[i:i + OB_WS_CHUNK]
                try:
                    self.ws.orderbook_stream(OB_DEPTH, chunk, self._on_orderbook)
                    self.ob_subscribed.update(chunk)
                except Exception as exc:  # noqa: BLE001
                    log.warning("orderbook subscribe failed %s: %s", chunk, exc)
            if pending:
                log.info("L1 subscribed orderbook.%d for %d symbols", OB_DEPTH, len(pending))
        if LIQ_ENABLED:
            stream = getattr(self.ws, "all_liquidation_stream", None) or getattr(self.ws, "liquidation_stream", None)
            if stream is None:
                log.warning("pybit has no (all_)liquidation_stream — Layer 2 disabled")
                return
            pending = [s for s in symbols if s not in self.liq_subscribed]
            for sym in pending:
                try:
                    stream(symbol=sym, callback=self._on_liquidation)
                    self.liq_subscribed.add(sym)
                except Exception as exc:  # noqa: BLE001
                    log.warning("liquidation subscribe failed %s: %s", sym, exc)
            if pending:
                log.info("L2 subscribed %s for %d symbols", getattr(stream, "__name__", "liquidation"), len(pending))
        if VP_ENABLED:
            tstream = getattr(self.ws, "trade_stream", None)
            if tstream is None:
                log.warning("pybit has no trade_stream — Layer 4 (volume profile) disabled")
            else:
                pending = [s for s in symbols if s not in self.vp_subscribed]
                for sym in pending:
                    try:
                        tstream(symbol=sym, callback=self._on_trade)
                        self.vp_subscribed.add(sym)
                    except Exception as exc:  # noqa: BLE001
                        log.warning("trade subscribe failed %s: %s", sym, exc)
                if pending:
                    log.info("L4 subscribed publicTrade for %d symbols", len(pending))

    def _on_orderbook(self, msg: dict[str, Any]) -> None:
        data = msg.get("data") or {}
        symbol = data.get("s")
        book = self.books.get(symbol) if symbol else None
        if book is None:
            return
        ts = int(msg.get("ts") or now_ms())
        with book.lock:
            if msg.get("type") == "snapshot":
                book.bids = {float(p): (float(s), ts) for p, s in data.get("b", []) if float(s) > 0}
                book.asks = {float(p): (float(s), ts) for p, s in data.get("a", []) if float(s) > 0}
            else:
                for p, s in data.get("b", []):
                    price, size = float(p), float(s)
                    if size <= 0:
                        book.bids.pop(price, None)
                    else:
                        prev = book.bids.get(price)
                        book.bids[price] = (size, prev[1] if prev else ts)
                for p, s in data.get("a", []):
                    price, size = float(p), float(s)
                    if size <= 0:
                        book.asks.pop(price, None)
                    else:
                        prev = book.asks.get(price)
                        book.asks[price] = (size, prev[1] if prev else ts)
            book.last_update_ms = ts

    def _on_liquidation(self, msg: dict[str, Any]) -> None:
        rows = msg.get("data") or []
        if isinstance(rows, dict):
            rows = [rows]
        for d in rows:
            try:
                symbol = d.get("s")
                side = d.get("S")  # order side of the liquidation
                price = float(d.get("p"))
                qty = float(d.get("v"))
            except (TypeError, ValueError):
                continue
            ts = int(d.get("T") or msg.get("ts") or now_ms())
            pos_side = "long" if side == "Sell" else "short"  # Sell print = long liquidated
            self.liq_buffer.append((ts, symbol, side, pos_side, price, qty, price * qty))

    # ── Layer 4: trade ingestion + volume profile ───────────────────────────────
    def _on_trade(self, msg: dict[str, Any]) -> None:
        rows = msg.get("data") or []
        if isinstance(rows, dict):
            rows = [rows]
        with self.vp_lock:
            for d in rows:
                try:
                    sym = d.get("s")
                    side = d.get("S")  # taker side: Buy = aggressive buy, Sell = aggressive sell
                    price = float(d.get("p"))
                    size = float(d.get("v"))
                except (TypeError, ValueError):
                    continue
                if not sym or price <= 0 or size <= 0:
                    continue
                ts = int(d.get("T") or msg.get("ts") or now_ms())
                hb = (ts // 3_600_000) * 3_600_000
                notional = price * size
                key = (sym, hb, vp_bin_index(price))
                acc = self.vp_accum.get(key)
                if acc is None:
                    acc = [0.0, 0.0]
                    self.vp_accum[key] = acc
                acc[0] += notional
                acc[1] += notional if side == "Buy" else -notional
                self.last_trade_ms = ts

    def _vp_seed(self, sym: str) -> None:
        """Seed a total-volume profile from 1h klines (delta=0; the live trade stream adds
        the real buy/sell split on top). ON CONFLICT DO NOTHING so it never overwrites
        live-accumulated buckets or double-seeds on restart."""
        if sym in self.vp_seeded:
            return
        bars = self._fetch_total_klines(sym, "60", VP_SEED_HOURS + 2)
        rows = []
        for b in bars:
            turnover, low, high = b["turnover"], b["low"], b["high"]
            if turnover <= 0 or low <= 0 or high < low:
                continue
            i0, i1 = vp_bin_index(low), vp_bin_index(high)
            n = max(1, i1 - i0 + 1)
            per = turnover / n
            for i in range(i0, i1 + 1):
                rows.append((sym, b["ts"], i, per, 0.0))
        if rows:
            self._vp_db.executemany(
                """INSERT INTO vp_buckets(symbol, hour_bucket, bin_idx, total, delta)
                   VALUES (?,?,?,?,?) ON CONFLICT(symbol, hour_bucket, bin_idx) DO NOTHING""", rows)
            self._vp_db.commit()
        self.vp_seeded.add(sym)

    def _fetch_total_klines(self, symbol: str, interval: str, limit: int):
        """Closed candles with turnover (USD volume): {ts, high, low, turnover}."""
        rows: dict[int, list[Any]] = {}
        end_ms: Optional[int] = None
        while len(rows) < limit:
            kwargs: dict[str, Any] = {"category": CATEGORY, "symbol": symbol,
                                      "interval": str(interval), "limit": min(1000, max(1, limit - len(rows)))}
            if end_ms is not None:
                kwargs["end"] = end_ms
            resp = self.http.get_kline(**kwargs)
            batch = (resp.get("result", {}) or {}).get("list", []) or []
            if not batch:
                break
            for item in batch:
                rows[int(item[0])] = item
            end_ms = min(int(item[0]) for item in batch) - 1
            if len(batch) < kwargs["limit"]:
                break
            time.sleep(0.05)
        return [{"ts": int(ts), "high": float(r[2]), "low": float(r[3]), "turnover": float(r[6])}
                for ts, r in sorted(rows.items())]

    def fetch_ohlc(self, symbol: str, interval: str, start_ms: int):
        """Full OHLC candles since start_ms (chronological): [ts, open, high, low, close]."""
        rows: dict[int, list[Any]] = {}
        end_ms: Optional[int] = None
        for _ in range(20):  # paginate back from now to start_ms (bounded window)
            kwargs: dict[str, Any] = {"category": CATEGORY, "symbol": symbol,
                                      "interval": str(interval), "start": start_ms, "limit": 1000}
            if end_ms is not None:
                kwargs["end"] = end_ms
            resp = self.http.get_kline(**kwargs)
            batch = (resp.get("result", {}) or {}).get("list", []) or []
            if not batch:
                break
            for item in batch:
                rows[int(item[0])] = item
            oldest = min(int(item[0]) for item in batch)
            if oldest <= start_ms or len(batch) < 1000:
                break
            end_ms = oldest - 1
            time.sleep(0.05)
        return [[int(ts), float(r[1]), float(r[2]), float(r[3]), float(r[4])]
                for ts, r in sorted(rows.items()) if int(ts) >= start_ms]

    def vp_loop(self) -> None:
        self._vp_db = self._connect()
        with self.state_lock:
            symbols = list(self.universe)
        for sym in symbols:
            if self.stop.is_set():
                return
            try:
                self._vp_seed(sym)
            except Exception:  # noqa: BLE001
                log.exception("L4 seed failed %s", sym)
        log.info("L4 volume-profile seed complete (%d symbols)", len(self.vp_seeded))
        cycle = 0
        while not self.stop.is_set():
            with self.vp_lock:
                batch = list(self.vp_accum.items())
                self.vp_accum.clear()
            if batch:
                try:
                    self._vp_db.executemany(
                        """INSERT INTO vp_buckets(symbol, hour_bucket, bin_idx, total, delta)
                           VALUES (?,?,?,?,?)
                           ON CONFLICT(symbol, hour_bucket, bin_idx)
                           DO UPDATE SET total = total + excluded.total, delta = delta + excluded.delta""",
                        [(s, hb, b, acc[0], acc[1]) for (s, hb, b), acc in batch])
                    self._vp_db.commit()
                except Exception:  # noqa: BLE001
                    log.exception("L4 flush failed")
            if cycle % 30 == 0:
                try:
                    cutoff = ((now_ms() // 3_600_000) - VP_RETENTION_HOURS) * 3_600_000
                    self._vp_db.execute("DELETE FROM vp_buckets WHERE hour_bucket < ?", (cutoff,))
                    self._vp_db.commit()
                    self._vp_db.execute("PRAGMA wal_checkpoint(PASSIVE)")
                except Exception:  # noqa: BLE001
                    pass
            cycle += 1
            self.stop.wait(VP_FLUSH_SECONDS)

    # ── Layer 1 worker: snapshot books ───────────────────────────────────────────
    def _ob_aggregate(self, symbol: str, bids: list[tuple[float, float]], asks: list[tuple[float, float]]):
        if not bids or not asks:
            return None
        best_bid = max(bids, key=lambda x: x[0])[0]
        best_ask = min(asks, key=lambda x: x[0])[0]
        if best_bid <= 0 or best_ask < best_bid:
            return None
        mid = (best_bid + best_ask) / 2.0
        bin_w = mid * (OB_BIN_BPS / 1e4)
        if bin_w <= 0:
            return None
        lower, upper = mid * (1 - OB_RANGE_PCT), mid * (1 + OB_RANGE_PCT)
        n = max(1, int(round((upper - lower) / bin_w)))
        bins = [[0.0, 0.0, 0.0, 0.0] for _ in range(n)]
        bid_tot = ask_tot = 0.0

        def idx(p):
            if p < lower or p >= upper:
                return None
            i = int((p - lower) / bin_w)
            return i if 0 <= i < n else None

        for price, qty in bids:
            notional = price * qty
            bid_tot += notional
            i = idx(price)
            if i is not None:
                bins[i][0] += qty
                bins[i][2] += notional
        for price, qty in asks:
            notional = price * qty
            ask_tot += notional
            i = idx(price)
            if i is not None:
                bins[i][1] += qty
                bins[i][3] += notional
        rows = [(lower + i * bin_w, lower + (i + 1) * bin_w, b[0], b[1], b[2], b[3])
                for i, b in enumerate(bins) if b[0] > 0 or b[1] > 0]
        return {"mid": mid, "best_bid": best_bid, "best_ask": best_ask,
                "spread_bps": (best_ask - best_bid) / mid * 1e4,
                "bid_total": bid_tot, "ask_total": ask_tot, "bins": rows}

    def ob_loop(self) -> None:
        self._ob_db = self._connect()
        cycle = 0
        min_life_ms = OB_MIN_LIFETIME_S * 1000.0
        while not self.stop.is_set():
            start = time.time()
            with self.state_lock:
                symbols = list(self.universe)
            ts = now_ms()
            written = 0
            for symbol in symbols:
                book = self.books.get(symbol)
                if book is None:
                    continue
                with book.lock:
                    if not book.bids or not book.asks:
                        continue
                    # lifetime/persistence filter: drop levels younger than the threshold (spoofs)
                    bids = [(p, sz) for p, (sz, seen) in book.bids.items() if ts - seen >= min_life_ms]
                    asks = [(p, sz) for p, (sz, seen) in book.asks.items() if ts - seen >= min_life_ms]
                try:
                    agg = self._ob_aggregate(symbol, bids, asks)
                    if agg is None:
                        continue
                    cur = self._ob_db.execute(
                        """INSERT INTO ob_snapshots(symbol, ts, mid, best_bid, best_ask, spread_bps,
                           bid_notional_total, ask_notional_total) VALUES (?,?,?,?,?,?,?,?)""",
                        (symbol, ts, agg["mid"], agg["best_bid"], agg["best_ask"], agg["spread_bps"],
                         agg["bid_total"], agg["ask_total"]),
                    )
                    sid = cur.lastrowid
                    if agg["bins"]:
                        self._ob_db.executemany(
                            """INSERT INTO ob_bins(snapshot_id, symbol, ts, price_low, price_high,
                               bid_qty, ask_qty, bid_notional, ask_notional) VALUES (?,?,?,?,?,?,?,?,?)""",
                            [(sid, symbol, ts, lo, hi, bq, aq, bn, an) for (lo, hi, bq, aq, bn, an) in agg["bins"]],
                        )
                    written += 1
                except Exception:  # noqa: BLE001
                    log.exception("ob snapshot failed for %s", symbol)
            try:
                if cycle % 30 == 0:
                    cutoff = now_ms() - OB_RETENTION_HOURS * 3600_000
                    self._ob_db.execute("DELETE FROM ob_bins WHERE ts < ?", (cutoff,))
                    self._ob_db.execute("DELETE FROM ob_snapshots WHERE ts < ?", (cutoff,))
                self._ob_db.commit()
                if cycle % 20 == 0:
                    self._ob_db.execute("PRAGMA wal_checkpoint(PASSIVE)")
            except Exception:  # noqa: BLE001
                log.exception("ob commit failed")
            cycle += 1
            self.stop.wait(max(0.5, OB_SNAPSHOT_INTERVAL - (time.time() - start)))

    # ── Layer 2 worker: flush liquidation events ─────────────────────────────────
    def liq_loop(self) -> None:
        self._liq_db = self._connect()
        cycle = 0
        while not self.stop.is_set():
            batch = []
            while self.liq_buffer:
                try:
                    batch.append(self.liq_buffer.popleft())
                except IndexError:
                    break
            if batch:
                try:
                    self._liq_db.executemany(
                        """INSERT INTO liq_events(ts, symbol, side, pos_side, price, qty, notional)
                           VALUES (?,?,?,?,?,?,?)""", batch)
                    self._liq_db.commit()
                except Exception:  # noqa: BLE001
                    log.exception("liq flush failed")
            if cycle % 60 == 0:
                try:
                    cutoff = now_ms() - LIQ_RETENTION_HOURS * 3600_000
                    self._liq_db.execute("DELETE FROM liq_events WHERE ts < ?", (cutoff,))
                    self._liq_db.commit()
                except Exception:  # noqa: BLE001
                    pass
            cycle += 1
            self.stop.wait(2.0)

    # ── Layer 3 worker: predictive ───────────────────────────────────────────────
    def fetch_mark_klines(self, symbol: str, interval: str, limit: int, start_ms: Optional[int] = None):
        """Closed MARK-price candles, chronological: {ts, high, low, close}."""
        rows: dict[int, list[Any]] = {}
        end_ms: Optional[int] = None
        while len(rows) < limit:
            kwargs: dict[str, Any] = {"category": CATEGORY, "symbol": symbol,
                                      "interval": str(interval), "limit": min(1000, max(1, limit - len(rows)))}
            if end_ms is not None:
                kwargs["end"] = end_ms
            if start_ms is not None:
                kwargs["start"] = start_ms
            resp = self.http.get_mark_price_kline(**kwargs)
            batch = (resp.get("result", {}) or {}).get("list", []) or []
            if not batch:
                break
            for item in batch:
                rows[int(item[0])] = item
            end_ms = min(int(item[0]) for item in batch) - 1
            if len(batch) < kwargs["limit"] or (start_ms is not None and end_ms < start_ms):
                break
            time.sleep(0.05)
        out = []
        for ts in sorted(rows):
            r = rows[ts]
            out.append({"ts": int(ts), "high": float(r[2]), "low": float(r[3]), "close": float(r[4])})
        return out

    def fetch_open_interest(self, symbol: str, oi_interval: str, limit: int) -> dict[int, float]:
        """ts(ms) -> open interest (contracts). Paginated by cursor."""
        out: dict[int, float] = {}
        cursor: Optional[str] = None
        guard = 0
        while len(out) < limit and guard < 60:
            guard += 1
            kwargs: dict[str, Any] = {"category": CATEGORY, "symbol": symbol,
                                      "intervalTime": oi_interval, "limit": 200}
            if cursor:
                kwargs["cursor"] = cursor
            try:
                resp = self.http.get_open_interest(**kwargs)
            except Exception as exc:  # noqa: BLE001
                log.warning("get_open_interest failed %s: %s", symbol, exc)
                break
            result = resp.get("result", {}) or {}
            for item in result.get("list", []) or []:
                try:
                    out[int(item["timestamp"])] = float(item["openInterest"])
                except (TypeError, ValueError, KeyError):
                    continue
            cursor = result.get("nextPageCursor")
            if not cursor or not (result.get("list")):
                break
            time.sleep(0.05)
        return out

    def _build_state(self, symbol: str, tf: str) -> MapState:
        spec = TIMEFRAMES[tf]
        limit = int(spec["lookback_hours"] * 60 / tf_minutes(tf)) + 2
        candles = self.fetch_mark_klines(symbol, spec["interval"], limit)
        oi = self.fetch_open_interest(symbol, spec["oi_interval"], limit) if USE_OI else {}
        state = MapState(symbol, tf)
        forming_cutoff = now_ms() - tf_minutes(tf) * 60_000
        for c in candles:
            if c["ts"] > forming_cutoff:
                continue
            state.apply_candle(c["ts"], c["high"], c["low"], c["close"], oi.get(c["ts"]), 0.0)
        state.prune(now_ms() - int(spec["lookback_hours"]) * 3600_000)
        return state

    def flush_pred(self, conn: sqlite3.Connection, state: MapState) -> None:
        sym, tf = state.symbol, state.tf
        conn.execute("DELETE FROM liq_levels WHERE symbol=? AND tf=?", (sym, tf))
        conn.executemany(
            """INSERT INTO liq_levels(symbol, tf, side, bin_idx, price_low, price_high,
               magnitude, created_ts, consumed_ts) VALUES (?,?,?,?,?,?,?,?,?)""",
            [(sym, tf, l.side, l.bin_idx, bin_low(l.bin_idx), bin_low(l.bin_idx + 1),
              l.magnitude, l.created_ts, l.consumed_ts) for l in state.levels()],
        )
        conn.execute("DELETE FROM liq_klines WHERE symbol=? AND tf=?", (sym, tf))
        conn.executemany(
            "INSERT INTO liq_klines(symbol, tf, ts, close, high, low, new_notional) VALUES (?,?,?,?,?,?,?)",
            [(sym, tf, ts, c, h, lo, nn) for (ts, c, h, lo, nn) in state.klines],
        )
        conn.commit()

    def _do_backfill(self) -> None:
        with self.state_lock:
            symbols = list(self.universe)
        for sym in symbols:
            for tf in TIMEFRAMES:
                if self.stop.is_set():
                    return
                try:
                    state = self._build_state(sym, tf)
                    with self.state_lock:
                        self.states[(sym, tf)] = state
                    self.flush_pred(self._pred_db, state)
                    log.info("L3 backfill %s %s: %d live / %d consumed levels",
                             sym, tf, len(state.alive), len(state.closed))
                except Exception:  # noqa: BLE001
                    log.exception("L3 backfill failed %s %s", sym, tf)
        log.info("L3 backfill complete")

    def pred_loop(self) -> None:
        self._pred_db = self._connect()
        self._do_backfill()
        log.info("L3 entering update loop")
        while not self.stop.is_set():
            if self.needs_rebuild.is_set():
                self.needs_rebuild.clear()
                log.info("L3 rebuilding all states with recalibrated weights")
                self._do_backfill()
            start = time.time()
            with self.state_lock:
                symbols = list(self.universe)
            advanced = 0
            for sym in symbols:
                for tf in TIMEFRAMES:
                    if self.stop.is_set():
                        return
                    try:
                        advanced += self._pred_update(sym, tf)
                    except Exception:  # noqa: BLE001
                        log.exception("L3 update failed %s %s", sym, tf)
            try:
                self._pred_db.execute("PRAGMA wal_checkpoint(PASSIVE)")
            except Exception:  # noqa: BLE001
                pass
            if advanced:
                log.info("L3 update: advanced %d candles in %.1fs", advanced, time.time() - start)
            self.stop.wait(max(2.0, PRED_UPDATE_INTERVAL - (time.time() - start)))

    def _pred_update(self, sym: str, tf: str) -> int:
        with self.state_lock:
            state = self.states.get((sym, tf))
        if state is None:
            return 0
        spec = TIMEFRAMES[tf]
        forming_cutoff = now_ms() - tf_minutes(tf) * 60_000
        new = [c for c in self.fetch_mark_klines(sym, spec["interval"], 200, start_ms=state.last_ts + 1)
               if c["ts"] > state.last_ts and c["ts"] <= forming_cutoff]
        if not new:
            return 0
        oi = self.fetch_open_interest(sym, spec["oi_interval"], 50) if USE_OI else {}
        for c in new:
            state.apply_candle(c["ts"], c["high"], c["low"], c["close"], oi.get(c["ts"]), 0.0)
        state.prune(now_ms() - int(spec["lookback_hours"]) * 3600_000)
        self.flush_pred(self._pred_db, state)
        return len(new)

    # ── weight persistence (Dirichlet posterior survives restarts) ────────────────
    def load_weights(self) -> None:
        global LEVERAGE_WEIGHTS
        try:
            with open(WEIGHTS_FILE, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            alpha = data.get("alpha")
            if isinstance(alpha, list) and len(alpha) == len(LEVERAGES) and sum(alpha) > 0:
                self.alpha = [float(a) for a in alpha]
                LEVERAGE_WEIGHTS = [a / sum(self.alpha) for a in self.alpha]
                self.last_event_ts = int(data.get("last_event_ts", 0))
                log.info("loaded calibrated weights (concentration=%.0f): %s",
                         sum(self.alpha), [round(x, 4) for x in LEVERAGE_WEIGHTS])
        except FileNotFoundError:
            pass
        except Exception as exc:  # noqa: BLE001
            log.warning("could not load weights file %s: %s", WEIGHTS_FILE, exc)

    def save_weights(self, meta: dict[str, Any]) -> None:
        try:
            Path(WEIGHTS_FILE).parent.mkdir(parents=True, exist_ok=True)
            payload = {"leverages": LEVERAGES, "alpha": self.alpha, "weights": LEVERAGE_WEIGHTS,
                       "last_event_ts": self.last_event_ts, "updated_ms": now_ms(), **meta}
            with open(WEIGHTS_FILE, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not save weights file: %s", exc)

    # ── auto-calibration: Bayesian Dirichlet update from actual liquidations ───────
    def calibrate(self) -> dict[str, Any]:
        """Fit the leverage-bucket weights to actual liquidation prints as a Dirichlet
        posterior. Each real liquidation is attributed across buckets by the posterior
        responsibility r_L ∝ weight_L · profile_L(price) (profile_L = predicted intensity
        that bucket L places at that price, from recent candle entries). Notional-weighted
        responsibilities of NEW events become pseudo-counts folded into the Dirichlet:
            alpha ← forget·alpha + counts ;  weights = alpha / Σalpha (posterior mean).
        Self-annealing (step shrinks as Σalpha grows) and yields per-weight uncertainty."""
        global LEVERAGE_WEIGHTS
        tf = CALIB_TF if CALIB_TF in TIMEFRAMES else next(iter(TIMEFRAMES))
        nlev = len(LEVERAGES)
        # only fold NEW events since the last applied run (bounded by the window for safety)
        lower = max(self.last_event_ts, now_ms() - CALIB_WINDOW_HOURS * 3600_000)
        acc = [0.0] * nlev
        n_events = n_explained = 0
        explained_notional = total_notional = 0.0
        max_seen_ts = self.last_event_ts

        conn = self._connect(read_only=True)
        try:
            with self.state_lock:
                symbols = list(self.universe)
            for sym in symbols:
                events = conn.execute(
                    "SELECT ts, pos_side, price, notional FROM liq_events WHERE symbol=? AND ts>?",
                    (sym, lower)).fetchall()
                if not events:
                    continue
                candles = conn.execute(
                    "SELECT close, high, low, new_notional FROM liq_klines WHERE symbol=? AND tf=? AND ts>=?",
                    (sym, tf, now_ms() - CALIB_WINDOW_HOURS * 3600_000)).fetchall()
                if not candles:
                    continue
                prof: dict[tuple[str, int], list[float]] = {}
                for c in candles:
                    close, high, low = c["close"], c["high"], c["low"]
                    nn = c["new_notional"] or 0.0
                    if nn <= 0 or close <= 0:
                        continue
                    bp = (close - low) / (high - low) if high > low else 0.5
                    for i, lev in enumerate(LEVERAGES):
                        for side, liq, frac in (
                            ("long", close * (1 - 1 / lev + MMR), bp),
                            ("short", close * (1 + 1 / lev - MMR), 1 - bp),
                        ):
                            if liq <= 0 or frac <= 0:
                                continue
                            key = (side, bin_index(liq))
                            vec = prof.get(key)
                            if vec is None:
                                vec = [0.0] * nlev
                                prof[key] = vec
                            vec[i] += nn * frac
                for ev in events:
                    n_events += 1
                    max_seen_ts = max(max_seen_ts, int(ev["ts"]))
                    notional = ev["notional"] or 0.0
                    total_notional += notional
                    vec = prof.get((ev["pos_side"], bin_index(ev["price"])))
                    if not vec:
                        continue
                    mix = [LEVERAGE_WEIGHTS[i] * vec[i] for i in range(nlev)]
                    s = sum(mix)
                    if s <= 0:
                        continue
                    n_explained += 1
                    explained_notional += notional
                    for i in range(nlev):
                        acc[i] += notional * mix[i] / s  # Bayesian responsibility (notional-weighted)
        finally:
            conn.close()

        explained_frac = (explained_notional / total_notional) if total_notional > 0 else 0.0
        old = list(LEVERAGE_WEIGHTS)
        applied = False
        if n_explained >= CALIB_MIN_EVENTS and sum(acc) > 0:
            # scale responsibilities to "effective event counts" so prior_strength is comparable
            scale = n_explained / sum(acc)
            counts = [a * scale for a in acc]
            self.alpha = [CALIB_FORGET * self.alpha[i] + counts[i] for i in range(nlev)]
            LEVERAGE_WEIGHTS = [a / sum(self.alpha) for a in self.alpha]
            self.last_event_ts = max_seen_ts
            applied = True
            self.save_weights({"calib_tf": tf, "n_explained": n_explained,
                               "explained_frac": explained_frac})
            self.needs_rebuild.set()

        concentration = sum(self.alpha)
        # Dirichlet marginal std per weight: Beta(a, A-a) -> sqrt(m(1-m)/(A+1))
        stds = [round((w * (1 - w) / (concentration + 1)) ** 0.5, 4) for w in LEVERAGE_WEIGHTS]
        # predictive score: share of actual liquidations the model placed intensity at
        # (count-based hit rate + notional-weighted capture) — grades the predictive layer
        hit_rate = round(n_explained / n_events, 4) if n_events else 0.0
        report = {"ts": now_ms(), "tf": tf, "n_events": n_events, "n_explained": n_explained,
                  "hit_rate": hit_rate, "explained_frac": round(explained_frac, 4), "applied": applied,
                  "leverages": LEVERAGES, "old_weights": [round(x, 4) for x in old],
                  "weights": [round(x, 4) for x in LEVERAGE_WEIGHTS],
                  "weight_std": stds, "concentration": round(concentration, 1)}
        self.last_calibration = report
        try:
            wconn = self._connect()
            wconn.execute(
                """INSERT INTO calibration_runs(ts, tf, n_events, n_explained, explained_frac,
                   applied, leverages_json, weights_json) VALUES (?,?,?,?,?,?,?,?)""",
                (report["ts"], tf, n_events, n_explained, explained_frac, int(applied),
                 json.dumps(LEVERAGES), json.dumps(LEVERAGE_WEIGHTS)))
            wconn.commit()
            wconn.close()
        except Exception:  # noqa: BLE001
            log.exception("could not persist calibration run")
        log.info("calibration: %d new events, %d explained (%.1f%% notional), applied=%s, "
                 "concentration=%.0f; weights %s -> %s", n_events, n_explained, explained_frac * 100,
                 applied, concentration, [round(x, 3) for x in old], [round(x, 3) for x in LEVERAGE_WEIGHTS])
        return report

    def calibration_loop(self) -> None:
        while not self.stop.is_set():
            self.stop.wait(CALIB_INTERVAL_HOURS * 3600)
            if self.stop.is_set():
                return
            try:
                self.calibrate()
            except Exception:  # noqa: BLE001
                log.exception("calibration failed")

    def universe_loop(self) -> None:
        while not self.stop.is_set():
            self.stop.wait(UNIVERSE_REFRESH_SECONDS)
            if self.stop.is_set():
                return
            try:
                self.resolve_universe()
                self.subscribe_ws()
            except Exception:  # noqa: BLE001
                log.exception("universe refresh failed")

    # ── REST queries ─────────────────────────────────────────────────────────────
    def status(self) -> dict[str, Any]:
        conn = self._connect(read_only=True)
        try:
            with self.state_lock:
                symbols = list(self.universe)
            now = now_ms()
            per = []
            for sym in symbols:
                ob = conn.execute("SELECT MAX(ts) AS t FROM ob_snapshots WHERE symbol=?", (sym,)).fetchone()
                liq = conn.execute("SELECT COUNT(*) AS n, MAX(ts) AS t FROM liq_events WHERE symbol=?", (sym,)).fetchone()
                tfs = {}
                for tf in TIMEFRAMES:
                    a = conn.execute("SELECT COUNT(*) AS n FROM liq_levels WHERE symbol=? AND tf=? AND consumed_ts IS NULL",
                                     (sym, tf)).fetchone()["n"]
                    tfs[tf] = a
                per.append({
                    "symbol": sym,
                    "ob_age_s": round((now - ob["t"]) / 1000.0, 1) if ob and ob["t"] else None,
                    "liq_events": liq["n"] if liq else 0,
                    "pred_live_levels": tfs,
                })
            totals = {
                "ob_snapshots": conn.execute("SELECT COUNT(*) AS n FROM ob_snapshots").fetchone()["n"],
                "liq_events": conn.execute("SELECT COUNT(*) AS n FROM liq_events").fetchone()["n"],
                "pred_levels": conn.execute("SELECT COUNT(*) AS n FROM liq_levels").fetchone()["n"],
                "vp_buckets": conn.execute("SELECT COUNT(*) AS n FROM vp_buckets").fetchone()["n"],
            }
        finally:
            conn.close()
        try:
            db_bytes = Path(DB_PATH).stat().st_size
        except OSError:
            db_bytes = 0
        return {
            "success": True, "service": "heatmap-bot (3-layer)",
            "uptime_s": round((now_ms() - _STARTED_AT_MS) / 1000.0, 1),
            "tracked_symbols": len(symbols),
            "layers": {"orderbook": OB_ENABLED, "actual_liquidations": LIQ_ENABLED,
                       "predictive": PRED_ENABLED, "volume_profile": VP_ENABLED},
            "timeframes": list(TIMEFRAMES.keys()), "leverages": LEVERAGES,
            "leverage_weights": [round(x, 4) for x in LEVERAGE_WEIGHTS],
            "calibration": {"enabled": CALIB_ENABLED, "interval_hours": CALIB_INTERVAL_HOURS,
                            "last_run": self.last_calibration or None},
            "totals": totals, "db_path": DB_PATH, "db_size_bytes": db_bytes, "symbols": per,
        }

    def get_calibration(self) -> dict[str, Any]:
        conn = self._connect(read_only=True)
        try:
            rows = conn.execute(
                """SELECT ts, tf, n_events, n_explained, explained_frac, applied, weights_json
                   FROM calibration_runs ORDER BY ts DESC LIMIT 30""").fetchall()
        finally:
            conn.close()
        history = [{"ts": r["ts"], "tf": r["tf"], "n_events": r["n_events"],
                    "n_explained": r["n_explained"], "explained_frac": r["explained_frac"],
                    "hit_rate": round(r["n_explained"] / r["n_events"], 4) if r["n_events"] else 0.0,
                    "applied": bool(r["applied"]), "weights": json.loads(r["weights_json"])}
                   for r in rows]
        return {"success": True, "enabled": CALIB_ENABLED, "interval_hours": CALIB_INTERVAL_HOURS,
                "method": "bayesian_dirichlet", "leverages": LEVERAGES,
                "current_weights": [round(x, 4) for x in LEVERAGE_WEIGHTS],
                "alpha": [round(a, 2) for a in self.alpha], "concentration": round(sum(self.alpha), 1),
                "last_run": self.last_calibration or None, "history": history}

    def get_universe(self) -> dict[str, Any]:
        conn = self._connect(read_only=True)
        try:
            rows = conn.execute("SELECT symbol, rank, turnover24h, updated_ts FROM universe ORDER BY rank").fetchall()
        finally:
            conn.close()
        return {"success": True, "count": len(rows), "universe": [dict(r) for r in rows]}

    def liquidity(self, symbol: str, since: Optional[int], limit: int):
        conn = self._connect(read_only=True)
        try:
            params: list[Any] = [symbol]
            where = "symbol = ?"
            if since is not None:
                where += " AND ts >= ?"
                params.append(since)
            rows = list(reversed(conn.execute(
                f"SELECT * FROM ob_snapshots WHERE {where} ORDER BY ts DESC LIMIT ?", params + [limit]).fetchall()))
            if not rows:
                return None
            min_ts, max_ts = rows[0]["ts"], rows[-1]["ts"]
            bins_by_ts: dict[int, list[dict[str, Any]]] = {}
            for br in conn.execute(
                """SELECT ts, price_low, price_high, bid_qty, ask_qty, bid_notional, ask_notional
                   FROM ob_bins WHERE symbol=? AND ts BETWEEN ? AND ? ORDER BY price_low""",
                (symbol, min_ts, max_ts)):
                d = dict(br)
                bins_by_ts.setdefault(d.pop("ts"), []).append(d)
            snaps = []
            for r in rows:
                d = dict(r)
                d["bins"] = bins_by_ts.get(r["ts"], [])
                snaps.append(d)
        finally:
            conn.close()
        return {"success": True, "symbol": symbol, "layer": "liquidity", "count": len(snaps), "snapshots": snaps}

    def actual_liquidations(self, symbol: str, since: Optional[int], limit: int):
        conn = self._connect(read_only=True)
        try:
            params: list[Any] = [symbol]
            where = "symbol = ?"
            if since is not None:
                where += " AND ts >= ?"
                params.append(since)
            rows = list(reversed(conn.execute(
                f"SELECT ts, side, pos_side, price, qty, notional FROM liq_events WHERE {where} ORDER BY ts DESC LIMIT ?",
                params + [limit]).fetchall()))
        finally:
            conn.close()
        return {"success": True, "symbol": symbol, "layer": "actual_liquidations",
                "count": len(rows), "events": [dict(r) for r in rows]}

    def estimated(self, symbol: str, tf: str):
        if tf not in TIMEFRAMES:
            return None
        conn = self._connect(read_only=True)
        try:
            klines = conn.execute(
                "SELECT ts, close, high, low FROM liq_klines WHERE symbol=? AND tf=? ORDER BY ts", (symbol, tf)).fetchall()
            if not klines:
                return None
            levels = conn.execute(
                """SELECT side, bin_idx, price_low, price_high, magnitude, created_ts, consumed_ts
                   FROM liq_levels WHERE symbol=? AND tf=? ORDER BY price_low""", (symbol, tf)).fetchall()
        finally:
            conn.close()
        return {"success": True, "symbol": symbol, "tf": tf, "layer": "estimated",
                "window_start": klines[0]["ts"], "window_end": klines[-1]["ts"],
                "last_price": klines[-1]["close"], "leverages": LEVERAGES,
                "price_series": [[r["ts"], r["close"]] for r in klines],
                "levels": [dict(r) for r in levels]}

    def estimated_magnets(self, symbol: str, tf: str):
        if tf not in TIMEFRAMES:
            return None
        conn = self._connect(read_only=True)
        try:
            last = conn.execute("SELECT close FROM liq_klines WHERE symbol=? AND tf=? ORDER BY ts DESC LIMIT 1",
                                (symbol, tf)).fetchone()
            if last is None:
                return None
            rows = conn.execute(
                """SELECT side, price_low, price_high, magnitude, created_ts FROM liq_levels
                   WHERE symbol=? AND tf=? AND consumed_ts IS NULL ORDER BY magnitude DESC LIMIT ?""",
                (symbol, tf, TOP_MAGNETS)).fetchall()
        finally:
            conn.close()
        price = last["close"]
        magnets = [{"side": r["side"], "price": round(0.5 * (r["price_low"] + r["price_high"]), 10),
                    "distance_pct": round((0.5 * (r["price_low"] + r["price_high"]) / price - 1) * 100, 3),
                    "magnitude": r["magnitude"]} for r in rows]
        return {"success": True, "symbol": symbol, "tf": tf, "last_price": price,
                "count": len(magnets), "magnets": magnets}

    # ── Layer 4 queries ───────────────────────────────────────────────────────--
    @staticmethod
    def _last_price(conn: sqlite3.Connection, symbol: str) -> Optional[float]:
        row = conn.execute("SELECT mid FROM ob_snapshots WHERE symbol=? ORDER BY ts DESC LIMIT 1",
                           (symbol,)).fetchone()
        if row and row["mid"]:
            return row["mid"]
        row = conn.execute("SELECT close FROM liq_klines WHERE symbol=? ORDER BY ts DESC LIMIT 1",
                           (symbol,)).fetchone()
        return row["close"] if row else None

    def volume_profile(self, symbol: str, window: str):
        if window not in VP_WINDOWS:
            window = "24h"
        conn = self._connect(read_only=True)
        try:
            ws = vp_window_start(window, now_ms())
            rows = conn.execute(
                """SELECT bin_idx, SUM(total) AS total, SUM(delta) AS delta FROM vp_buckets
                   WHERE symbol=? AND hour_bucket>=? GROUP BY bin_idx HAVING total>0 ORDER BY bin_idx""",
                (symbol, ws)).fetchall()
            if not rows:
                return None
            last_price = self._last_price(conn, symbol)
        finally:
            conn.close()

        idxs = [r["bin_idx"] for r in rows]
        totals = [r["total"] for r in rows]
        deltas = [r["delta"] for r in rows]
        grand = sum(totals)
        poc_i = max(range(len(rows)), key=lambda i: totals[i])

        # value area: expand from POC to the larger neighbour until VP_VALUE_AREA of volume
        covered = totals[poc_i]
        lo = hi = poc_i
        target = VP_VALUE_AREA * grand
        while covered < target and (lo > 0 or hi < len(rows) - 1):
            left = totals[lo - 1] if lo > 0 else -1.0
            right = totals[hi + 1] if hi < len(rows) - 1 else -1.0
            if right >= left:
                hi += 1
                covered += totals[hi]
            else:
                lo -= 1
                covered += totals[lo]

        bins = [{"price_low": vp_bin_low(idxs[i]), "price_high": vp_bin_low(idxs[i] + 1),
                 "total": totals[i], "delta": deltas[i],
                 "buy": (totals[i] + deltas[i]) / 2.0, "sell": (totals[i] - deltas[i]) / 2.0,
                 "imbalance": (deltas[i] / totals[i]) if totals[i] else 0.0}
                for i in range(len(rows))]
        order_total = sorted(range(len(rows)), key=lambda i: totals[i], reverse=True)
        order_delta = sorted(range(len(rows)), key=lambda i: deltas[i], reverse=True)

        def lvl(i):
            return {"price": round(0.5 * (vp_bin_low(idxs[i]) + vp_bin_low(idxs[i] + 1)), 10),
                    "total": round(totals[i], 2), "delta": round(deltas[i], 2),
                    "imbalance": round((deltas[i] / totals[i]) if totals[i] else 0.0, 4)}

        return {
            "success": True, "symbol": symbol, "window": window, "window_start": ws,
            "last_price": last_price, "total_notional": grand,
            "poc": round(0.5 * (vp_bin_low(idxs[poc_i]) + vp_bin_low(idxs[poc_i] + 1)), 10),
            "vah": vp_bin_low(idxs[hi] + 1), "val": vp_bin_low(idxs[lo]),
            "hvns": [lvl(i) for i in order_total[:VP_TOP_LEVELS]],
            "long_levels": [lvl(i) for i in order_delta[:VP_TOP_LEVELS] if deltas[i] > 0],
            "short_levels": [lvl(i) for i in order_delta[::-1][:VP_TOP_LEVELS] if deltas[i] < 0],
            "bins": bins,
        }

    # OHLC interval per window (for the price overlay on the volume heatmap)
    _VP_OHLC_INTERVAL = {"4h": "5", "24h": "15", "7d": "60", "daily": "15", "weekly": "60"}

    def volume_profile_heatmap(self, symbol: str, window: str):
        if window not in VP_WINDOWS:
            window = "24h"
        conn = self._connect(read_only=True)
        try:
            ws = vp_window_start(window, now_ms())
            rows = conn.execute(
                """SELECT hour_bucket, bin_idx, total, delta FROM vp_buckets
                   WHERE symbol=? AND hour_bucket>=? ORDER BY hour_bucket""",
                (symbol, ws)).fetchall()
            last_price = self._last_price(conn, symbol)
        finally:
            conn.close()
        if not rows:
            return None
        cells = [{"ts": r["hour_bucket"], "price_low": vp_bin_low(r["bin_idx"]),
                  "price_high": vp_bin_low(r["bin_idx"] + 1), "total": r["total"], "delta": r["delta"]}
                 for r in rows]
        interval = self._VP_OHLC_INTERVAL.get(window, "15")
        try:
            ohlc = self.fetch_ohlc(symbol, interval, ws)
        except Exception as exc:  # noqa: BLE001 - price overlay is best-effort
            log.warning("ohlc fetch failed %s: %s", symbol, exc)
            ohlc = []
        return {"success": True, "symbol": symbol, "window": window, "window_start": ws,
                "last_price": last_price, "ohlc_interval": interval, "ohlc": ohlc, "cells": cells}

    def ohlc(self, symbol: str, interval: str, start_ms: int):
        """Reusable OHLC candles for price overlays on any chart: [ts, open, high, low, close]."""
        try:
            data = self.fetch_ohlc(symbol, str(interval), int(start_ms))
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "error": str(exc)}
        return {"success": True, "symbol": symbol, "interval": str(interval), "ohlc": data}

    # ── consolidated key levels across all four layers (with confluence) ──────────
    def levels(self, symbol: str, tf: str = "1h", window: str = "24h", n: int = 12):
        """Merge significant price levels from every layer and cluster nearby ones into
        confluence zones (a level confirmed by several layers scores higher)."""
        cand: list[tuple[str, float, float]] = []  # (type, price, strength 0..1)
        last_price = None

        vp = self.volume_profile(symbol, window)
        if vp:
            last_price = vp.get("last_price")
            if vp.get("poc"):
                cand.append(("POC", vp["poc"], 1.0))
            if vp.get("vah"):
                cand.append(("VAH", vp["vah"], 0.55))
            if vp.get("val"):
                cand.append(("VAL", vp["val"], 0.55))
            hv = vp.get("hvns", [])
            mx = max((h["total"] for h in hv), default=0.0) or 1.0
            for h in hv[:6]:
                cand.append(("HVN", h["price"], 0.5 * h["total"] / mx))

        em = self.estimated_magnets(symbol, tf)
        if em:
            last_price = last_price or em.get("last_price")
            mags = em.get("magnets", [])
            mx = max((m["magnitude"] for m in mags), default=0.0) or 1.0
            for m in mags[:8]:
                cand.append(("LIQ" + ("L" if m["side"] == "long" else "S"), m["price"], m["magnitude"] / mx))

        liq = self.liquidity(symbol, None, 1)
        if liq and liq.get("snapshots"):
            snap = liq["snapshots"][-1]
            last_price = last_price or snap.get("mid")
            walls = sorted(snap.get("bins", []),
                           key=lambda b: max(b.get("bid_notional") or 0, b.get("ask_notional") or 0), reverse=True)[:6]
            mx = max((max(b.get("bid_notional") or 0, b.get("ask_notional") or 0) for b in walls), default=0.0) or 1.0
            for b in walls:
                wn = max(b.get("bid_notional") or 0, b.get("ask_notional") or 0)
                cand.append(("WALL", 0.5 * (b["price_low"] + b["price_high"]), wn / mx))

        conn = self._connect(read_only=True)
        try:
            rows = conn.execute("SELECT price, notional FROM liq_events WHERE symbol=? AND ts>=?",
                                (symbol, now_ms() - 24 * 3_600_000)).fetchall()
        finally:
            conn.close()
        if rows:
            agg: dict[int, float] = {}
            for r in rows:
                agg[vp_bin_index(r["price"])] = agg.get(vp_bin_index(r["price"]), 0.0) + (r["notional"] or 0.0)
            mx = max(agg.values()) or 1.0
            for bi, nt in sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:5]:
                cand.append(("LIQDONE", vp_bin_low(bi) * (1 + VP_BIN_PCT / 2), nt / mx))

        cand = [(t, p, max(s, 1e-6)) for t, p, s in cand if p and p > 0]
        cand.sort(key=lambda x: x[1])
        groups: list[dict[str, Any]] = []
        for typ, price, strength in cand:
            if groups:
                g = groups[-1]
                if abs(price / (g["ps"] / g["w"]) - 1) <= 0.0025:  # within 0.25% -> same zone
                    g["types"].append(typ); g["w"] += strength; g["ps"] += price * strength
                    continue
            groups.append({"types": [typ], "w": strength, "ps": price * strength})

        levels = []
        for g in groups:
            price = g["ps"] / g["w"]
            distinct = len({t.rstrip("LS") if t.startswith("LIQ") else t for t in g["types"]})
            score = g["w"] * (1 + 0.5 * (distinct - 1))  # confluence bonus
            levels.append({"price": round(price, 8), "types": sorted(set(g["types"])), "layers": distinct,
                           "score": round(score, 3),
                           "distance_pct": round((price / last_price - 1) * 100, 3) if last_price else None})
        levels.sort(key=lambda x: x["score"], reverse=True)
        return {"success": True, "symbol": symbol, "tf": tf, "window": window,
                "last_price": last_price, "levels": levels[:n]}

    # ── CVD, market-structure snapshot, cross-symbol screener ─────────────────────
    def _ticker(self, symbol: str) -> dict[str, Any]:
        try:
            t = self.http.get_tickers(category=CATEGORY, symbol=symbol)
            lst = (t.get("result", {}) or {}).get("list", []) or []
            return lst[0] if lst else {}
        except Exception:  # noqa: BLE001
            return {}

    def cvd(self, symbol: str, window: str):
        """Cumulative volume delta series (taker buy-sell), hourly, with close price."""
        if window not in VP_WINDOWS:
            window = "24h"
        conn = self._connect(read_only=True)
        try:
            ws = vp_window_start(window, now_ms())
            rows = conn.execute(
                "SELECT hour_bucket AS h, SUM(delta) AS d, SUM(total) AS t FROM vp_buckets "
                "WHERE symbol=? AND hour_bucket>=? GROUP BY hour_bucket ORDER BY hour_bucket",
                (symbol, ws)).fetchall()
        finally:
            conn.close()
        if not rows:
            return None
        try:
            ohlc = self.fetch_ohlc(symbol, "60", ws)
        except Exception:  # noqa: BLE001
            ohlc = []
        close_by = {o[0]: o[4] for o in ohlc}
        cum = 0.0
        series = []
        for r in rows:
            cum += r["d"] or 0.0
            series.append({"ts": r["h"], "net_delta": r["d"] or 0.0, "cvd": cum, "close": close_by.get(r["h"])})
        return {"success": True, "symbol": symbol, "window": window, "series": series, "ohlc": ohlc}

    def structure(self, symbol: str):
        """One decision-ready market-structure snapshot combining all layers."""
        lv = self.levels(symbol, "1h", "24h", n=20)
        price = lv.get("last_price")
        levels = lv.get("levels", [])
        below = [L for L in levels if price and L["price"] < price]
        above = [L for L in levels if price and L["price"] > price]
        support = max(below, key=lambda L: L["price"]) if below else None
        resistance = min(above, key=lambda L: L["price"]) if above else None
        strongest = max(levels, key=lambda L: L["score"]) if levels else None

        em = self.estimated_magnets(symbol, "1h") or {}
        mags = em.get("magnets", [])
        lmag = sum(m["magnitude"] for m in mags if m["side"] == "long")
        smag = sum(m["magnitude"] for m in mags if m["side"] == "short")
        skew = (lmag - smag) / (lmag + smag) if (lmag + smag) > 0 else 0.0

        cv = self.cvd(symbol, "24h")
        series = cv["series"] if cv else []
        cvd_now = series[-1]["cvd"] if series else 0.0
        cvd_half = series[len(series) // 2]["cvd"] if len(series) > 3 else 0.0

        conn = self._connect(read_only=True)
        try:
            vr = conn.execute("SELECT SUM(delta) d, SUM(total) t FROM vp_buckets WHERE symbol=? AND hour_bucket>=?",
                              (symbol, vp_window_start("24h", now_ms()))).fetchone()
        finally:
            conn.close()
        vt = (vr["t"] or 0.0) if vr else 0.0
        vd = (vr["d"] or 0.0) if vr else 0.0
        vol_imb = vd / vt if vt else 0.0

        tk = self._ticker(symbol)
        funding = float(tk.get("fundingRate") or 0.0) if tk else 0.0
        oi_usd = float(tk.get("openInterestValue") or 0.0) if tk else 0.0
        bias = ("bullish" if vol_imb > 0.05 and (cvd_now - cvd_half) > 0
                else "bearish" if vol_imb < -0.05 and (cvd_now - cvd_half) < 0 else "neutral")

        def sl(L):
            return None if not L else {"price": L["price"], "distance_pct": L["distance_pct"],
                                       "types": L["types"], "score": L["score"]}

        return {
            "success": True, "symbol": symbol, "last_price": price, "bias": bias,
            "cvd_24h": round(cvd_now, 2), "cvd_recent_12h": round(cvd_now - cvd_half, 2),
            "volume_imbalance": round(vol_imb, 4),
            "nearest_support": sl(support), "nearest_resistance": sl(resistance),
            "strongest_level": sl(strongest),
            "liquidation_skew": {
                "long_notional": round(lmag, 2), "short_notional": round(smag, 2), "skew": round(skew, 3),
                "lean": ("longs vulnerable (downside liq)" if skew > 0.1
                         else "shorts vulnerable (upside liq)" if skew < -0.1 else "balanced")},
            "funding_rate": funding, "open_interest_usd": oi_usd,
            "levels": levels[:6],
        }

    def screener(self, metric: str, n: int):
        """Rank the universe by a market-structure metric."""
        metric = (metric or "liq").lower()
        conn = self._connect(read_only=True)
        try:
            with self.state_lock:
                symbols = list(self.universe)
            ws24 = vp_window_start("24h", now_ms())
            h1 = now_ms() - 3_600_000
            out = []
            for sym in symbols:
                vr = conn.execute("SELECT SUM(delta) d, SUM(total) t FROM vp_buckets WHERE symbol=? AND hour_bucket>=?",
                                  (sym, ws24)).fetchone()
                d = (vr["d"] or 0.0) if vr else 0.0
                t = (vr["t"] or 0.0) if vr else 0.0
                lr = conn.execute(
                    "SELECT SUM(notional) n, SUM(CASE WHEN pos_side='long' THEN notional ELSE -notional END) s "
                    "FROM liq_events WHERE symbol=? AND ts>=?", (sym, h1)).fetchone()
                liq = (lr["n"] or 0.0) if lr else 0.0
                liqskew = (lr["s"] or 0.0) if lr else 0.0
                out.append({"symbol": sym, "cvd_24h": round(d, 2), "vol_24h": round(t, 2),
                            "vol_imbalance": round(d / t, 4) if t else 0.0,
                            "liq_1h": round(liq, 2), "liq_skew_1h": round(liqskew, 2)})
        finally:
            conn.close()
        keyf = {"liq": lambda x: x["liq_1h"], "cvd": lambda x: x["cvd_24h"],
                "imbalance": lambda x: abs(x["vol_imbalance"]), "volume": lambda x: x["vol_24h"]}.get(
            metric, lambda x: x["liq_1h"])
        out.sort(key=keyf, reverse=True)
        return {"success": True, "metric": metric, "count": len(out), "results": out[:n]}

    # ── watch subscriptions (for proximity alerts) ────────────────────────────────
    def set_watch(self, uid: int, symbol: str, add: bool) -> dict[str, Any]:
        conn = self._connect()
        try:
            if add:
                conn.execute("INSERT OR IGNORE INTO watches(uid, symbol) VALUES (?,?)", (uid, symbol))
            else:
                conn.execute("DELETE FROM watches WHERE uid=? AND symbol=?", (uid, symbol))
            conn.commit()
            rows = conn.execute("SELECT symbol FROM watches WHERE uid=? ORDER BY symbol", (uid,)).fetchall()
        finally:
            conn.close()
        return {"success": True, "uid": uid, "watches": [r["symbol"] for r in rows]}

    def get_watches(self, uid: int) -> dict[str, Any]:
        conn = self._connect(read_only=True)
        try:
            rows = conn.execute("SELECT symbol FROM watches WHERE uid=? ORDER BY symbol", (uid,)).fetchall()
        finally:
            conn.close()
        return {"success": True, "uid": uid, "watches": [r["symbol"] for r in rows]}

    # ── proactive alerts: liquidation cascades + level proximity ──────────────────
    def alert_loop(self) -> None:
        conn = self._connect(read_only=True)
        last_cascade: dict[str, int] = {}
        last_prox: dict[tuple[int, str], int] = {}
        while not self.stop.is_set():
            self.stop.wait(ALERT_INTERVAL_S)
            if self.stop.is_set():
                return
            now = now_ms()
            try:
                since = now - CASCADE_WINDOW_S * 1000
                for r in conn.execute(
                    "SELECT symbol, SUM(notional) AS n, COUNT(*) AS c, "
                    "SUM(price*notional) AS pn FROM liq_events WHERE ts>=? GROUP BY symbol",
                    (since,)).fetchall():
                    if (r["n"] or 0) >= CASCADE_MIN_USD and now - last_cascade.get(r["symbol"], 0) > CASCADE_COOLDOWN_S * 1000:
                        sd = conn.execute(
                            "SELECT pos_side, SUM(notional) s FROM liq_events WHERE symbol=? AND ts>=? "
                            "GROUP BY pos_side ORDER BY s DESC LIMIT 1", (r["symbol"], since)).fetchone()
                        side = sd["pos_side"] if sd else "?"
                        level = (r["pn"] / r["n"]) if r["n"] else None  # notional-weighted liq price
                        cmp = self._last_price(conn, r["symbol"])
                        loc = f"@ {_fmt_price(level)}"
                        if cmp and (not level or abs(cmp / level - 1) > 0.0005):
                            loc += f" · cmp {_fmt_price(cmp)}"
                        tg_send(f"⚡ {r['symbol']}: ${r['n']/1e6:.1f}M {side} liquidations {loc} in "
                                f"{CASCADE_WINDOW_S}s ({r['c']} prints)", TG_UIDS)
                        last_cascade[r["symbol"]] = now
            except Exception:  # noqa: BLE001
                log.exception("cascade alert check failed")
            try:
                watches = conn.execute("SELECT uid, symbol FROM watches").fetchall()
            except Exception:  # noqa: BLE001
                watches = []
            for w in watches:
                key = (w["uid"], w["symbol"])
                if now - last_prox.get(key, 0) < PROX_COOLDOWN_S * 1000:
                    continue
                try:
                    lv = self.levels(w["symbol"], "1h", "24h", n=8)
                    price = lv.get("last_price")
                    if not price:
                        continue
                    for L in lv["levels"]:
                        if abs(L["price"] / price - 1) <= PROX_PCT:
                            tg_send(f"🎯 {w['symbol']} {price:g} → approaching {L['price']:g} "
                                    f"({(L['price']/price-1)*100:+.2f}%) [{','.join(L['types'])}]", [w["uid"]])
                            last_prox[key] = now
                            break
                except Exception:  # noqa: BLE001
                    log.exception("proximity alert failed %s", w["symbol"])

    def watchdog_loop(self) -> None:
        last_alert = 0
        while not self.stop.is_set():
            self.stop.wait(max(15, WATCHDOG_STALE_S // 2))
            if self.stop.is_set():
                return
            now = now_ms()
            stale = []
            if OB_ENABLED:
                fresh = max((b.last_update_ms for b in self.books.values() if b.last_update_ms), default=0)
                if fresh and now - fresh > WATCHDOG_STALE_S * 1000:
                    stale.append(f"orderbook {int((now-fresh)/1000)}s")
            if VP_ENABLED and self.last_trade_ms and now - self.last_trade_ms > WATCHDOG_STALE_S * 1000:
                stale.append(f"trades {int((now-self.last_trade_ms)/1000)}s")
            if stale and now - last_alert > WATCHDOG_COOLDOWN_S * 1000:
                msg = "⚠️ heatmap-bot stream stale: " + ", ".join(stale) + " — reconnecting WS"
                log.warning(msg)
                tg_send(msg, ADMIN_UIDS)
                last_alert = now
                try:  # force a fresh WS connection + resubscribe
                    self.ws = None
                    self.ob_subscribed.clear(); self.liq_subscribed.clear(); self.vp_subscribed.clear()
                    self.subscribe_ws()
                except Exception:  # noqa: BLE001
                    log.exception("watchdog resubscribe failed")


# ── REST server ─────────────────────────────────────────────────────────────--
class RequestHandler(BaseHTTPRequestHandler):
    service: HeatmapService

    def log_message(self, fmt: str, *args: Any) -> None:
        log.debug("http: " + fmt, *args)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, default=str).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            log.debug("client disconnected before response: %s", exc)

    def _tf(self, qs):
        tf = (qs.get("tf", ["1h"])[0] or "1h").strip()
        return tf if tf in TIMEFRAMES else "1h"

    def _since(self, qs):
        try:
            v = qs.get("since", [None])[0]
            if v is None:
                return None
            n = float(v)
            return int(n if n > 1e12 else n * 1000)
        except (ValueError, TypeError):
            return None

    def _limit(self, qs):
        try:
            return max(1, min(int(qs.get("limit", [DEFAULT_SERIES_LIMIT])[0]), MAX_SERIES_LIMIT))
        except (ValueError, IndexError):
            return DEFAULT_SERIES_LIMIT

    def do_GET(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            parts = [p for p in parsed.path.split("/") if p]
            qs = parse_qs(parsed.query)
            svc = self.service

            if not parts or parts == ["health"]:
                return self._send_json(HTTPStatus.OK, svc.status())
            if parts == ["v1", "universe"]:
                return self._send_json(HTTPStatus.OK, svc.get_universe())
            if parts == ["v1", "calibration"]:
                return self._send_json(HTTPStatus.OK, svc.get_calibration())
            # Consolidated key levels across all layers: /v1/levels/{symbol}
            if len(parts) == 3 and parts[0] == "v1" and parts[1] == "levels":
                window = (qs.get("window", ["24h"])[0] or "24h").strip()
                try:
                    n = int(qs.get("n", ["12"])[0])
                except (ValueError, IndexError):
                    n = 12
                return self._send_json(HTTPStatus.OK, svc.levels(parts[2].upper(), self._tf(qs), window, n))
            if parts == ["v1", "watches"]:
                try:
                    uid = int(qs.get("uid", ["0"])[0])
                except (ValueError, IndexError):
                    uid = 0
                return self._send_json(HTTPStatus.OK, svc.get_watches(uid))
            # Market-structure snapshot, CVD series, cross-symbol screener
            if len(parts) == 3 and parts[0] == "v1" and parts[1] == "structure":
                return self._send_json(HTTPStatus.OK, svc.structure(parts[2].upper()))
            if len(parts) == 3 and parts[0] == "v1" and parts[1] == "cvd":
                window = (qs.get("window", ["24h"])[0] or "24h").strip()
                payload = svc.cvd(parts[2].upper(), window)
                return self._send_json(HTTPStatus.OK if payload else HTTPStatus.NOT_FOUND,
                                       payload or {"success": False, "error": "no cvd data"})
            if parts == ["v1", "screener"]:
                try:
                    n = int(qs.get("n", ["20"])[0])
                except (ValueError, IndexError):
                    n = 20
                return self._send_json(HTTPStatus.OK, svc.screener(qs.get("metric", ["liq"])[0], n))
            # Reusable OHLC for price overlays: /v1/ohlc/{symbol}?interval=&start=&hours=
            if len(parts) == 3 and parts[0] == "v1" and parts[1] == "ohlc":
                interval = (qs.get("interval", ["60"])[0] or "60").strip()
                if "start" in qs:
                    start_ms = int(float(qs["start"][0]))
                else:
                    hours = float(qs.get("hours", ["24"])[0] or 24)
                    start_ms = now_ms() - int(hours * 3_600_000)
                return self._send_json(HTTPStatus.OK, svc.ohlc(parts[2].upper(), interval, start_ms))

            # Layer 1: /v1/liquidity/{symbol}   (alias: /v1/heatmap/{symbol})
            if len(parts) == 3 and parts[0] == "v1" and parts[1] in ("liquidity", "heatmap"):
                payload = svc.liquidity(parts[2].upper(), self._since(qs), self._limit(qs))
                return self._send_json(HTTPStatus.OK if payload else HTTPStatus.NOT_FOUND,
                                       payload or {"success": False, "error": "no data"})

            # Layer 4: /v1/volume_profile/{symbol}[/heatmap]
            if len(parts) >= 3 and parts[0] == "v1" and parts[1] == "volume_profile":
                symbol = parts[2].upper()
                window = (qs.get("window", ["24h"])[0] or "24h").strip()
                if len(parts) == 4 and parts[3] == "heatmap":
                    payload = svc.volume_profile_heatmap(symbol, window)
                elif len(parts) == 3:
                    payload = svc.volume_profile(symbol, window)
                else:
                    payload = None
                return self._send_json(HTTPStatus.OK if payload else HTTPStatus.NOT_FOUND,
                                       payload or {"success": False, "error": f"no volume data for {symbol}"})

            # Layer 2: /v1/liquidations/actual/{symbol}
            if len(parts) == 4 and parts[:3] == ["v1", "liquidations", "actual"]:
                payload = svc.actual_liquidations(parts[3].upper(), self._since(qs), self._limit(qs))
                return self._send_json(HTTPStatus.OK, payload)

            # Layer 3: /v1/liquidations/estimated/{symbol}[/latest]
            if len(parts) >= 4 and parts[:3] == ["v1", "liquidations", "estimated"]:
                symbol = parts[3].upper()
                tf = self._tf(qs)
                if len(parts) == 5 and parts[4] == "latest":
                    payload = svc.estimated_magnets(symbol, tf)
                else:
                    payload = svc.estimated(symbol, tf)
                return self._send_json(HTTPStatus.OK if payload else HTTPStatus.NOT_FOUND,
                                       payload or {"success": False, "error": f"no data for {symbol} tf={tf}"})

            self._send_json(HTTPStatus.NOT_FOUND, {"success": False, "error": "not found"})
        except Exception as exc:  # noqa: BLE001
            log.exception("HTTP request failed")
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"success": False, "error": str(exc)})

    def do_POST(self) -> None:  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}
            path = urlparse(self.path).path.rstrip("/")
            if path == "/v1/watch":
                symbol = str(body.get("symbol", "")).upper().strip()
                if not symbol or body.get("uid") is None:
                    return self._send_json(HTTPStatus.BAD_REQUEST, {"success": False, "error": "uid and symbol required"})
                return self._send_json(HTTPStatus.OK,
                                       self.service.set_watch(int(body["uid"]), symbol, bool(body.get("add", True))))
            self._send_json(HTTPStatus.NOT_FOUND, {"success": False, "error": "not found"})
        except json.JSONDecodeError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"success": False, "error": f"invalid json: {exc}"})
        except Exception as exc:  # noqa: BLE001
            log.exception("HTTP POST failed")
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"success": False, "error": str(exc)})


def main() -> None:
    global _STARTED_AT_MS
    _STARTED_AT_MS = now_ms()

    service = HeatmapService()
    service.init_db()
    service.load_weights()  # apply persisted Dirichlet weights from a previous calibration, if any
    try:
        service.resolve_universe()
        if OB_ENABLED or LIQ_ENABLED:
            service.subscribe_ws()
    except Exception:  # noqa: BLE001
        log.exception("initial setup failed; background loops will retry")

    if OB_ENABLED:
        threading.Thread(target=service.ob_loop, name="ob", daemon=True).start()
    if LIQ_ENABLED:
        threading.Thread(target=service.liq_loop, name="liq", daemon=True).start()
    if VP_ENABLED:
        threading.Thread(target=service.vp_loop, name="vp", daemon=True).start()
    if PRED_ENABLED:
        threading.Thread(target=service.pred_loop, name="pred", daemon=True).start()
        if CALIB_ENABLED:
            threading.Thread(target=service.calibration_loop, name="calibration", daemon=True).start()
    if ALERTS_ENABLED:
        threading.Thread(target=service.alert_loop, name="alerts", daemon=True).start()
    if WATCHDOG_ENABLED:
        threading.Thread(target=service.watchdog_loop, name="watchdog", daemon=True).start()
    threading.Thread(target=service.universe_loop, name="universe", daemon=True).start()

    RequestHandler.service = service
    server = ThreadingHTTPServer((API_HOST, int(API_PORT)), RequestHandler)

    def _shutdown(_sig: int, _frame: Any) -> None:
        log.info("shutdown signal received")
        service.stop.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal_module.signal(signal_module.SIGTERM, _shutdown)
    signal_module.signal(signal_module.SIGINT, _shutdown)
    log.info("heatmap API on %s:%s  layers: ob=%s liq=%s pred=%s  tfs=%s leverages=%s db=%s",
             API_HOST, API_PORT, OB_ENABLED, LIQ_ENABLED, PRED_ENABLED,
             ",".join(TIMEFRAMES), LEVERAGES, DB_PATH)
    server.serve_forever()
    log.info("heatmap bot stopped")


if __name__ == "__main__":
    main()
