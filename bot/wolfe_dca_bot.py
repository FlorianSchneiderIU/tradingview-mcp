#!/usr/bin/env python3
"""Wolfe DCA bot — standalone runner for the gated Wolfe strategy with a DCA second leg.

Self-contained (own process, own Bybit Demo account, own state/ledger). It reuses the
shared, validated Wolfe detection (bot.wolfe_wave.WolfeWaveEngine) for signals, then
executes + monitors the DCA structure entirely on its own — it never touches the main
mm-bot's positions or state. Position management is POLL-based (no private WS), mirroring
capitulation_spring_bot.py.

DCA logic (long; short mirrored), per gated signal E=entry, SL=stop, T=tp1, R=|E-SL|,
k=dca_stop_frac_k (from the config, default 0.25):
  * leg1: market entry with a Full bracket -> hard stop SL2 = SL - k*R, take-profit T.
  * leg2: a resting LIMIT add at SL (not reduce-only).
  * Path A (price hits T first): leg1 TPs (full RR); poller cancels the resting leg2.
  * Path B (price hits SL first): leg2 fills -> poller amends the position TP from T to E
    (the bounce target), keeps stop SL2.  bounce->E = +1R combined ; continue->SL2 = -(1+2k)R.

Validation: see docs/wolfe_dca_implementation_plan.md. Config:
bot/configs/wolfe_wave_shared_v1_dca_configs.json (dca_enabled=true, k=0.25).
"""
from __future__ import annotations

import json
import logging
import os
import signal as signal_module
import sys
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pybit.unified_trading import HTTP, WebSocket

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))            # bot/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from wolfe_wave import WolfeWaveEngine, WolfeWaveState, load_wolfe_wave_configs  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)sZ [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%dT%H:%M:%S")
log = logging.getLogger("wolfe-dca")

CATEGORY = "linear"
INTERVAL_MIN = "5"

# ── config ──────────────────────────────────────────────────────────────────
def _csv(name: str, default: str) -> list[str]:
    return [s.strip().upper() for s in os.environ.get(name, default).split(",") if s.strip()]

CONFIG_PATH   = os.environ.get("WOLFE_DCA_CONFIG_PATH", "/app/configs/wolfe_wave_shared_v1_dca_configs.json")
TRADING_DEMO  = os.environ.get("WOLFE_DCA_BYBIT_DEMO", "true").lower() in {"1", "true", "yes"}
LIVE_CONFIRM  = os.environ.get("WOLFE_DCA_LIVE_CONFIRM", "false").lower() in {"1", "true", "yes"}
TRADING_ENABLED = os.environ.get("WOLFE_DCA_TRADING_ENABLED", "true").lower() in {"1", "true", "yes"}
HEDGE_MODE    = os.environ.get("WOLFE_DCA_HEDGE_MODE", "true").lower() in {"1", "true", "yes"}
PUBLIC_WS_DEMO = os.environ.get("WOLFE_DCA_PUBLIC_WS_DEMO", "false").lower() in {"1", "true", "yes"}
API_KEY       = os.environ.get("WOLFE_DCA_BYBIT_API_KEY", "").strip()
API_SECRET    = os.environ.get("WOLFE_DCA_BYBIT_API_SECRET", "").strip()
STRATEGY_NAME = "wolfe_dca"

RISK_PCT      = float(os.environ.get("WOLFE_DCA_RISK_PCT", "0.005"))   # 0.5% equity; 0 -> RISK_USDT
RISK_USDT     = float(os.environ.get("WOLFE_DCA_RISK_USDT", "25"))
MAX_CONCURRENT = int(os.environ.get("WOLFE_DCA_MAX_CONCURRENT", "5"))
TAKER_FEE_RATE = float(os.environ.get("WOLFE_DCA_TAKER_FEE_RATE", os.environ.get("TAKER_FEE_RATE", "0.00055")))
MIN_STOP_DISTANCE_PCT = float(os.environ.get("WOLFE_DCA_MIN_STOP_DISTANCE_PCT", "0.0005"))
DEFAULT_K     = float(os.environ.get("WOLFE_DCA_K", "0.25"))
DEFAULT_MAX_HOLD_BARS = int(os.environ.get("WOLFE_DCA_MAX_HOLD_BARS", "144"))

WARMUP_BARS   = int(os.environ.get("WOLFE_DCA_WARMUP_BARS", "20000"))
POLL_SECONDS  = int(os.environ.get("WOLFE_DCA_POLL_SECONDS", "10"))
WS_STALE_SECONDS = int(os.environ.get("WOLFE_DCA_WS_STALE_SECONDS", "900"))
WS_CHUNK      = int(os.environ.get("WOLFE_DCA_WS_CHUNK", "20"))   # symbols per WS shard
HEARTBEAT_SECONDS = int(os.environ.get("WOLFE_DCA_HEARTBEAT_SECONDS", "3600"))

LOG_DIR    = Path(os.environ.get("WOLFE_DCA_LOG_DIR", os.environ.get("LOG_DIR", "/app/logs")))
LEDGER_PATH = Path(os.environ.get("WOLFE_DCA_LEDGER_PATH", str(LOG_DIR / "wolfe_dca_ledger.jsonl")))
TELEGRAM_BOT_TOKEN = os.environ.get("WOLFE_DCA_TELEGRAM_BOT_TOKEN", os.environ.get("TELEGRAM_BOT_TOKEN", "")).strip()
TELEGRAM_CHAT_ID = os.environ.get("WOLFE_DCA_TELEGRAM_CHAT_ID", os.environ.get("TELEGRAM_ACCEPTED_SIGNALS_CHAT_ID", "")).strip()

_SYMBOLS_ENV = os.environ.get("WOLFE_DCA_SYMBOLS", "").strip()


# ── helpers ─────────────────────────────────────────────────────────────────
def round_to_step(value: float, step: float) -> float:
    if step <= 0:
        return float(value)
    return round(round(float(value) / step) * step, 10)

def floor_to_step(value: float, step: float) -> float:
    if step <= 0:
        return float(value)
    import math
    return round(math.floor(float(value) / step) * step, 10)

def qty_to_str(value: float, step: float = 0.0) -> str:
    if step and step > 0:
        s = f"{step:.10f}".rstrip("0")
        decimals = len(s.split(".")[-1]) if "." in s else 0
        text = f"{value:.{decimals}f}"
    else:
        text = f"{value:.10f}".rstrip("0").rstrip(".")
    return text if text and text not in ("-0", "") else "0"

def fmt(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"

def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)

def fetch_5m_klines(http: HTTP, symbol: str, limit: int) -> list[dict]:
    out: list[dict] = []
    end = None
    while len(out) < limit:
        kwargs: dict[str, Any] = dict(category=CATEGORY, symbol=symbol, interval=INTERVAL_MIN, limit=1000)
        if end is not None:
            kwargs["end"] = end
        rows = http.get_kline(**kwargs).get("result", {}).get("list", [])
        if not rows:
            break
        for r in rows:
            out.append({"ts": int(r[0]), "open": float(r[1]), "high": float(r[2]),
                        "low": float(r[3]), "close": float(r[4]), "volume": float(r[5])})
        end = int(rows[-1][0]) - 1
        if len(rows) < 1000:
            break
        time.sleep(0.05)
    seen: set[int] = set(); uniq: list[dict] = []
    for b in sorted(out, key=lambda x: x["ts"]):
        if b["ts"] not in seen:
            seen.add(b["ts"]); uniq.append(b)
    return uniq[-limit:]


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str) -> None:
        self.token = token
        self.chat_id, self.thread_id = self._parse_target(chat_id)
        self.enabled = bool(token and self.chat_id)

    @staticmethod
    def _parse_target(raw: str) -> tuple[str, int | None]:
        text = str(raw or "").strip()
        for sep in ("_", ":"):
            head, s, tail = text.rpartition(sep)
            if s and head and tail.isdigit():
                return head, int(tail)
        return text, None

    def send(self, lines: list[str]) -> None:
        if not self.enabled:
            return
        try:
            import urllib.request
            payload: dict[str, Any] = {"chat_id": self.chat_id, "text": "\n".join(lines),
                                       "parse_mode": "HTML", "disable_web_page_preview": True}
            if self.thread_id is not None:
                payload["message_thread_id"] = self.thread_id
            data = json.dumps(payload).encode()
            req = urllib.request.Request(f"https://api.telegram.org/bot{self.token}/sendMessage",
                                         data=data, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10).read()
        except Exception as exc:  # noqa: BLE001
            log.warning("telegram send failed: %s", exc)


# ── executor (poll-based DCA state machine) ─────────────────────────────────
class DcaExecutor:
    def __init__(self, telegram: "TelegramNotifier | None" = None) -> None:
        self.http: HTTP | None = None
        self.telegram = telegram
        self.enabled = False
        self.long_idx = 1 if HEDGE_MODE else 0
        self.short_idx = 2 if HEDGE_MODE else 0
        self.info_cache: dict[str, dict[str, Any]] = {}
        self.open_trades: dict[str, dict[str, Any]] = {}   # symbol -> trade
        self.lock = threading.Lock()
        if not TRADING_ENABLED:
            log.info("Execution disabled (WOLFE_DCA_TRADING_ENABLED=false); signal-only")
            return
        if not TRADING_DEMO and not LIVE_CONFIRM:
            log.critical("WOLFE_DCA_BYBIT_DEMO=false requires WOLFE_DCA_LIVE_CONFIRM=true; refusing LIVE -> signal-only")
            return
        if not API_KEY or not API_SECRET:
            log.warning("WOLFE_DCA_BYBIT_API_KEY/SECRET missing; running signal-only")
            return
        self.http = HTTP(testnet=False, demo=TRADING_DEMO, api_key=API_KEY, api_secret=API_SECRET)
        self._detect_idx()
        self.enabled = True
        threading.Thread(target=self._poller, name="wolfe-dca-poller", daemon=True).start()
        log.info("Execution ENABLED on Bybit %s (%s mode, long_idx=%d short_idx=%d)",
                 "DEMO" if TRADING_DEMO else "LIVE", "hedge" if HEDGE_MODE else "one-way",
                 self.long_idx, self.short_idx)

    def _detect_idx(self) -> None:
        try:
            rows = self.http.get_positions(category=CATEGORY, settleCoin="USDT").get("result", {}).get("list", [])
            pidxs = {int(r.get("positionIdx", 0)) for r in rows}
            if pidxs == {0}:
                self.long_idx = self.short_idx = 0
        except Exception as exc:  # noqa: BLE001
            log.warning("position-mode detect failed (%s); using HEDGE_MODE=%s", exc, HEDGE_MODE)

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

    def execute(self, symbol: str, sig: dict[str, Any], *, k: float, max_hold_bars: int) -> dict[str, Any]:
        if not self.enabled or self.http is None:
            return {"ok": False, "skipped": True, "message": "trading disabled"}
        with self.lock:
            if symbol in self.open_trades:
                return {"ok": False, "skipped": True, "message": "already in position"}
            if len(self.open_trades) >= MAX_CONCURRENT:
                return {"ok": False, "skipped": True, "message": f"concurrency cap {MAX_CONCURRENT}"}
        info = self._info(symbol)
        if info.get("status") and info["status"] != "Trading":
            return {"ok": False, "message": f"{symbol} status={info['status']}"}
        direction = str(sig["signal"]).lower()
        long = direction == "long"
        entry = float(sig["entry"]); sl = float(sig["sl"]); target_t = float(sig["tp1"])
        risk = abs(entry - sl)
        if risk <= 0 or risk / entry < MIN_STOP_DISTANCE_PCT:
            return {"ok": False, "message": "stop distance too small"}
        q_step, min_qty, tick = info["qty_step"], info["min_qty"], info["tick_size"]
        idx = self.long_idx if long else self.short_idx
        side = "Buy" if long else "Sell"
        close_side = "Sell" if long else "Buy"
        sl2 = round_to_step(sl - k * risk if long else sl + k * risk, tick)
        sl_lvl = round_to_step(sl, tick)
        t_lvl = round_to_step(target_t, tick)
        e_lvl = round_to_step(entry, tick)
        equity = self._equity()
        if equity <= 0:
            return {"ok": False, "message": "invalid equity"}
        risk_usdt = equity * RISK_PCT if RISK_PCT > 0 else RISK_USDT
        fee_per_unit = TAKER_FEE_RATE * (entry + sl)
        qty = floor_to_step(risk_usdt / (risk + fee_per_unit), q_step)   # 1 leg = ~RISK_PCT per R
        if qty < min_qty:
            return {"ok": False, "message": f"risk too small for min qty ({qty}<{min_qty})"}
        self._ensure_leverage(symbol, max(info["max_leverage"], 1.0))
        link = self._link_id(symbol)
        # leg1 market with a Full bracket: hard stop SL2, take-profit T.
        try:
            r1 = self.http.place_order(
                category=CATEGORY, symbol=symbol, side=side, orderType="Market",
                qty=qty_to_str(qty, q_step), stopLoss=str(sl2), takeProfit=str(t_lvl),
                tpslMode="Full", tpTriggerBy="LastPrice", slTriggerBy="LastPrice",
                positionIdx=idx, orderLinkId=f"{link}-L1")
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "message": f"leg1 error: {exc}"}
        if int(r1.get("retCode", -1)) != 0:
            return {"ok": False, "message": f"leg1 rejected: {r1.get('retMsg')}"}
        leg1_id = str(r1.get("result", {}).get("orderId") or "")
        # leg2 resting limit add at SL (not reduce-only).
        leg2_id = ""
        try:
            r2 = self.http.place_order(
                category=CATEGORY, symbol=symbol, side=side, orderType="Limit",
                qty=qty_to_str(qty, q_step), price=str(sl_lvl), timeInForce="GTC",
                positionIdx=idx, orderLinkId=f"{link}-L2")
            if int(r2.get("retCode", -1)) == 0:
                leg2_id = str(r2.get("result", {}).get("orderId") or "")
            else:
                log.warning("[%s] DCA leg2 rejected: %s", symbol, r2.get("retMsg"))
        except Exception as exc:  # noqa: BLE001
            log.warning("[%s] DCA leg2 error: %s", symbol, exc)

        trade = {"symbol": symbol, "direction": direction, "idx": idx, "side": side, "close_side": close_side,
                 "entry": entry, "sl": sl_lvl, "sl2": sl2, "target_t": t_lvl, "target_e": e_lvl,
                 "leg1_qty": qty, "leg2_qty": qty, "leg1_id": leg1_id, "leg2_id": leg2_id,
                 "phase": "leg1_only", "order_link_id": link, "opened_at": time.time(),
                 "max_hold_s": max(1, int(max_hold_bars)) * 5 * 60, "q_step": q_step}
        with self.lock:
            self.open_trades[symbol] = trade
        log.info("[%s] DCA %s entry=%.6g qty=%s  stop(SL2)=%.6g tp(T)=%.6g  leg2@SL=%.6g (id=%s)",
                 symbol, direction, entry, qty_to_str(qty, q_step), sl2, t_lvl, sl_lvl, leg2_id or "-")
        return {"ok": True, "order_id": leg1_id, "order_link_id": link, "qty": qty_to_str(qty, q_step),
                "sl2": sl2, "target_t": t_lvl, "leg2_id": leg2_id}

    def _ensure_leverage(self, symbol: str, max_lev: float) -> None:
        try:
            self.http.set_leverage(category=CATEGORY, symbol=symbol,
                                   buyLeverage=qty_to_str(max_lev), sellLeverage=qty_to_str(max_lev))
        except Exception as exc:  # noqa: BLE001
            if "not modified" not in str(exc).lower():
                log.debug("[%s] set_leverage: %s", symbol, exc)

    def _tg(self, lines: list[str]) -> None:
        if self.telegram is not None:
            self.telegram.send(lines)

    @staticmethod
    def _link_id(symbol: str) -> str:
        ts = datetime.now(timezone.utc).strftime("%m%d%H%M")
        return f"WDCA-{symbol.replace('USDT', '')[:6]}-{ts}-{uuid.uuid4().hex[:5].upper()}"[:36]

    def _position_size(self, symbol: str, idx: int, side: str) -> float:
        rows = self.http.get_positions(category=CATEGORY, symbol=symbol).get("result", {}).get("list", [])
        for row in rows:
            if int(row.get("positionIdx", 0)) == idx and str(row.get("side", "")).lower() in (side.lower(), ""):
                return abs(float(row.get("size", 0) or 0))
        for row in rows:
            if str(row.get("side", "")).lower() == side.lower():
                return abs(float(row.get("size", 0) or 0))
        return 0.0

    def _cancel(self, symbol: str, order_id: str) -> None:
        if not order_id:
            return
        try:
            resp = self.http.cancel_order(category=CATEGORY, symbol=symbol, orderId=order_id)
            if int(resp.get("retCode", -1)) not in (0, 110001):
                log.debug("[%s] cancel leg2: %s", symbol, resp.get("retMsg"))
        except Exception as exc:  # noqa: BLE001
            log.debug("[%s] cancel leg2 err: %s", symbol, exc)

    def _amend_to_combined(self, trade: dict[str, Any]) -> None:
        sym = trade["symbol"]
        self.http.set_trading_stop(
            category=CATEGORY, symbol=sym, takeProfit=str(trade["target_e"]), tpTriggerBy="LastPrice",
            stopLoss=str(trade["sl2"]), slTriggerBy="LastPrice", tpslMode="Full", positionIdx=trade["idx"])

    def _market_close(self, trade: dict[str, Any], size: float) -> None:
        sym = trade["symbol"]
        self.http.place_order(category=CATEGORY, symbol=sym, side=trade["close_side"], orderType="Market",
                              qty=qty_to_str(size, trade["q_step"]), reduceOnly=True, positionIdx=trade["idx"])

    def _poller(self) -> None:
        eps = 1e-9
        while True:
            time.sleep(max(2, POLL_SECONDS))
            try:
                with self.lock:
                    items = list(self.open_trades.items())
                for symbol, trade in items:
                    try:
                        size = self._position_size(symbol, trade["idx"], trade["side"])
                    except Exception:
                        continue
                    leg1 = trade["leg1_qty"]; combined = trade["leg1_qty"] + trade["leg2_qty"]
                    timed_out = (time.time() - trade["opened_at"]) > trade["max_hold_s"]

                    if size <= eps:
                        # Position fully closed by exchange (TP at T/E or hard stop SL2).
                        if trade["phase"] == "leg1_only":
                            self._cancel(symbol, trade["leg2_id"])
                        self._finalize(symbol, trade, "closed")
                        continue

                    if timed_out:
                        try:
                            self._market_close(trade, size)
                        except Exception as exc:  # noqa: BLE001
                            log.warning("[%s] timeout close failed: %s", symbol, exc)
                        if trade["phase"] == "leg1_only":
                            self._cancel(symbol, trade["leg2_id"])
                        self._finalize(symbol, trade, "timeout")
                        continue

                    if trade["phase"] == "leg1_only" and size >= leg1 + trade["leg2_qty"] * 0.5 - eps:
                        # leg2 filled -> retarget the combined position from T to E.
                        try:
                            self._amend_to_combined(trade)
                            with self.lock:
                                if symbol in self.open_trades:
                                    self.open_trades[symbol]["phase"] = "combined"
                            self._ledger({"event": "DCA_LEG2_FILLED", "symbol": symbol,
                                          "direction": trade["direction"], "combined_target": trade["target_e"],
                                          "stop": trade["sl2"], "size": size})
                            log.info("[%s] DCA leg2 filled -> combined; target T->%.6g (size %s)",
                                     symbol, trade["target_e"], qty_to_str(size, trade["q_step"]))
                            self._tg([f"<b>[WOLFE DCA] leg2 filled</b> {symbol} {trade['direction']}",
                                      f"averaged in @ SL <code>{fmt(trade['sl'], 6)}</code>; "
                                      f"target → entry <code>{fmt(trade['target_e'], 6)}</code>, stop <code>{fmt(trade['sl2'], 6)}</code>"])
                        except Exception as exc:  # noqa: BLE001
                            log.warning("[%s] DCA retarget failed: %s", symbol, exc)
            except Exception:
                log.exception("DCA poller error")

    def _finalize(self, symbol: str, trade: dict[str, Any], reason: str) -> None:
        with self.lock:
            self.open_trades.pop(symbol, None)
        self._ledger({"event": "DCA_EXIT", "symbol": symbol, "direction": trade["direction"],
                      "reason": reason, "phase": trade["phase"], "entry": trade["entry"],
                      "held_s": round(time.time() - trade["opened_at"], 0)})
        log.info("[%s] DCA position finalized (%s, phase=%s)", symbol, reason, trade["phase"])
        self._tg([f"<b>[WOLFE DCA] exit</b> {symbol} {trade['direction']} — <code>{reason}</code> "
                  f"(phase {trade['phase']})"])

    @staticmethod
    def _ledger(payload: dict[str, Any]) -> None:
        try:
            LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
            payload = {"ts": datetime.now(timezone.utc).isoformat(), "type": "fill", "strategy": STRATEGY_NAME, **payload}
            with LEDGER_PATH.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(jsonable(payload), separators=(",", ":")) + "\n")
        except Exception as exc:  # noqa: BLE001
            log.debug("ledger write failed: %s", exc)


# ── bot ─────────────────────────────────────────────────────────────────────
class WolfeDcaBot:
    def __init__(self) -> None:
        self.public_http = HTTP(testnet=False, demo=False)
        self.configs = load_wolfe_wave_configs(
            symbols=_csv("WOLFE_DCA_SYMBOLS", "") or _config_symbols(CONFIG_PATH),
            config_path=CONFIG_PATH)
        self.symbols = list(self.configs.keys())
        self.engine = WolfeWaveEngine(strategy_name=STRATEGY_NAME)
        self.states = {s: WolfeWaveState(s, self.configs[s], WARMUP_BARS) for s in self.symbols}
        self.telegram = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        self.executor = DcaExecutor(telegram=self.telegram)
        self.stop_event = threading.Event()
        self.ws_connections: list[WebSocket] = []
        self.last_ws_ts = time.time()
        self.last_heartbeat = 0.0
        self.processed: set[str] = set()

    def start(self) -> None:
        log.info("Wolfe DCA bot: %d symbols, config=%s, exec=%s, max_concurrent=%d, risk=%.2f%%",
                 len(self.symbols), CONFIG_PATH, TRADING_ENABLED, MAX_CONCURRENT, RISK_PCT * 100)
        self._warmup()
        self.telegram.send(["<b>[WOLFE DCA] started</b>",
                            f"symbols=<code>{len(self.symbols)}</code>  exec=<code>{TRADING_ENABLED}</code> "
                            f"demo=<code>{TRADING_DEMO}</code>  max_open=<code>{MAX_CONCURRENT}</code>"])
        self._open_ws()
        while not self.stop_event.is_set():
            now = time.time()
            if HEARTBEAT_SECONDS > 0 and now - self.last_heartbeat >= HEARTBEAT_SECONDS:
                log.info("Heartbeat: open=%d/%d exec=%s", self.executor.open_count(), MAX_CONCURRENT, TRADING_ENABLED)
                self.last_heartbeat = now
            if WS_STALE_SECONDS > 0 and now - self.last_ws_ts > WS_STALE_SECONDS:
                raise SystemExit(f"No Bybit public WS message for {WS_STALE_SECONDS}s")
            time.sleep(1)

    def _warmup(self) -> None:
        log.info("Warming up %d symbols x %d 5m bars ...", len(self.symbols), WARMUP_BARS)
        for sym in self.symbols:
            try:
                bars = fetch_5m_klines(self.public_http, sym, WARMUP_BARS)
                for b in bars:
                    self.states[sym].push_bar(b)
                log.info("  %s: %d bars", sym, len(bars))
            except Exception as exc:  # noqa: BLE001
                log.warning("  %s warmup failed: %s", sym, exc)

    def _open_ws(self) -> None:
        # Shard subscriptions across multiple WS connections and disable pybit's
        # control-ping timeout (public Bybit sockets miss control pongs while the
        # data stream is healthy; our WS_STALE watchdog handles real stalls). This
        # mirrors the production mm-bot — a single connection for 53 symbols flaps
        # with "ping/pong timed out".
        chunks = [self.symbols[i:i + WS_CHUNK] for i in range(0, len(self.symbols), max(1, WS_CHUNK))]
        log.info("Opening %d kline WS shard(s) for %d symbols (demo=%s, chunk=%d)",
                 len(chunks), len(self.symbols), PUBLIC_WS_DEMO, WS_CHUNK)
        for idx, chunk in enumerate(chunks, start=1):
            ws = WebSocket(testnet=False, channel_type="linear", demo=PUBLIC_WS_DEMO, ping_timeout=None)
            self.ws_connections.append(ws)
            log.info("  shard %d/%d: subscribing %d symbols", idx, len(chunks), len(chunk))
            for sym in chunk:
                ws.kline_stream(interval=INTERVAL_MIN, symbol=sym, callback=self._on_kline)

    def _on_kline(self, msg: dict[str, Any]) -> None:
        self.last_ws_ts = time.time()
        try:
            data = msg.get("data") or []
            topic = str(msg.get("topic", ""))
            sym = topic.split(".")[-1] if "." in topic else None
            if not data or sym not in self.states:
                return
            candle = data[0]
            if not candle.get("confirm", False):
                return
            bar = {"ts": int(candle["start"]), "open": float(candle["open"]), "high": float(candle["high"]),
                   "low": float(candle["low"]), "close": float(candle["close"]), "volume": float(candle.get("volume", 0.0))}
            self.states[sym].push_bar(bar)
            key = f"{sym}:{bar['ts']}"
            if key in self.processed:
                return
            self.processed.add(key)
            if len(self.processed) > 4000:
                self.processed = set(sorted(self.processed)[-2000:])
            threading.Thread(target=self._evaluate, args=(sym,), daemon=True).start()
        except Exception:
            log.exception("kline callback error")

    def _evaluate(self, symbol: str) -> None:
        try:
            if self.executor.has_symbol(symbol):
                return
            state = self.states[symbol]
            sig = self.engine.detect_signal(state)
            if sig is None or sig.get("rejected"):
                return
            cfg = self.configs[symbol]
            k = float(getattr(cfg, "dca_stop_frac_k", DEFAULT_K) or DEFAULT_K)
            max_hold = int(getattr(cfg, "max_hold_bars", DEFAULT_MAX_HOLD_BARS) or DEFAULT_MAX_HOLD_BARS)
            execution = self.executor.execute(symbol, sig, k=k, max_hold_bars=max_hold)
            self._ledger_signal(symbol, sig, k, execution)
            log.info("SIGNAL %s %s score=%.1f entry=%.6g sl=%.6g tp=%.6g exec=%s",
                     symbol, sig.get("signal"), float(sig.get("score", 0.0)), float(sig["entry"]),
                     float(sig["sl"]), float(sig["tp1"]), execution.get("order_id") or execution.get("message"))
            if execution.get("ok"):
                self.telegram.send([
                    f"<b>[WOLFE DCA] {str(sig.get('signal')).upper()}</b> {symbol}  score <code>{fmt(sig.get('score'), 1)}</code>",
                    f"entry <code>{fmt(sig['entry'], 6)}</code>  qty <code>{execution.get('qty')}</code>",
                    f"DCA leg2@SL <code>{fmt(sig['sl'], 6)}</code>  hard stop(SL2) <code>{fmt(execution.get('sl2'), 6)}</code>  TP(T) <code>{fmt(execution.get('target_t'), 6)}</code>",
                ])
            elif not execution.get("skipped"):
                self.telegram.send([f"<b>[WOLFE DCA] {symbol} not placed</b> — <code>{execution.get('message', '-')}</code>"])
        except Exception:
            log.exception("evaluate error for %s", symbol)

    def _ledger_signal(self, symbol: str, sig: dict[str, Any], k: float, execution: dict[str, Any]) -> None:
        try:
            LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
            payload = {"ts": datetime.now(timezone.utc).isoformat(), "type": "signal", "strategy": STRATEGY_NAME,
                       "symbol": symbol, "direction": sig.get("signal"), "score": sig.get("score"),
                       "entry": sig.get("entry"), "sl": sig.get("sl"), "tp1": sig.get("tp1"), "dca_k": k,
                       "execution": execution}
            with LEDGER_PATH.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(jsonable(payload), separators=(",", ":")) + "\n")
        except Exception as exc:  # noqa: BLE001
            log.debug("signal ledger failed: %s", exc)

    def stop(self) -> None:
        self.stop_event.set()
        for ws in self.ws_connections:
            try:
                ws.exit()
            except Exception:
                pass


def _config_symbols(path: str) -> list[str]:
    try:
        d = json.load(open(path))
        return [k for k in d if not k.startswith("_")]
    except Exception:
        return []


def main() -> None:
    bot = WolfeDcaBot()

    def handle(signum: int, _frame: Any) -> None:
        log.info("signal %s -> stopping", signum)
        bot.stop()

    signal_module.signal(signal_module.SIGTERM, handle)
    signal_module.signal(signal_module.SIGINT, handle)
    bot.start()


if __name__ == "__main__":
    main()
