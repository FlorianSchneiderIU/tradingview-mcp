#!/usr/bin/env python3
"""Heatmap forward-test bot — paper-trades the heatmap setup signal on a SEPARATE Bybit
demo account so we can validate it out-of-sample.

It polls heatmap-bot's GET /v1/setup/{symbol} for the tracked universe; when a valid setup
appears (confluence + bias gated, already done server-side) and we're flat on that symbol,
it places a risk-sized market entry with a NATIVE bracket (full-position stopLoss=setup.sl,
takeProfit=setup.tp). Bybit manages the exits; we poll positions to detect closes and record
realized PnL / R-multiple to a JSONL ledger on the shared logs volume.

Isolation (mirrors wolfe_dca_bot / capitulation_spring_bot): its OWN demo account
(bot/.env.heatmap_demo), its OWN ledger (logs/heatmap_forward_ledger.jsonl) and state file.
It never touches any other bot's positions. Sizing is risk-based: qty = RISK_USDT / |entry-SL|,
so hitting the stop ≈ -1R. The R-multiple per trade = realized PnL / RISK_USDT.

Going live (real funds) requires HEATMAP_FWD_BYBIT_DEMO=false AND HEATMAP_FWD_LIVE_CONFIRM=true.
"""
from __future__ import annotations

import json
import logging
import math
import os
import signal as signal_module
import time
from pathlib import Path
from typing import Any, Optional

import requests
from pybit.unified_trading import HTTP

logging.basicConfig(level=logging.INFO, format="%(asctime)sZ [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%dT%H:%M:%S")
log = logging.getLogger("heatmap-fwd")

CATEGORY = "linear"

API_URL = os.environ.get("HEATMAP_API_URL", "http://heatmap-bot:8110").rstrip("/")
API_KEY = os.environ.get("HEATMAP_FWD_BYBIT_API_KEY", "").strip()
API_SECRET = os.environ.get("HEATMAP_FWD_BYBIT_API_SECRET", "").strip()
DEMO = os.environ.get("HEATMAP_FWD_BYBIT_DEMO", "true").lower() in {"1", "true", "yes"}
LIVE_CONFIRM = os.environ.get("HEATMAP_FWD_LIVE_CONFIRM", "false").lower() in {"1", "true", "yes"}
TRADING_ENABLED = os.environ.get("HEATMAP_FWD_TRADING_ENABLED", "true").lower() in {"1", "true", "yes"}

SYMBOLS_ENV = os.environ.get("HEATMAP_FWD_SYMBOLS", "").strip()
POLL_S = int(os.environ.get("HEATMAP_FWD_POLL_SECONDS", "30"))
RISK_USDT = float(os.environ.get("HEATMAP_FWD_RISK_USDT", "10"))
MIN_CONF = int(os.environ.get("HEATMAP_FWD_MIN_CONFIDENCE", "1"))
MAX_OPEN = int(os.environ.get("HEATMAP_FWD_MAX_OPEN", "5"))
LEVERAGE = float(os.environ.get("HEATMAP_FWD_LEVERAGE", "10"))
HEDGE_MODE = os.environ.get("HEATMAP_FWD_HEDGE_MODE", "true").lower() in {"1", "true", "yes"}
# Scaled exits: % of size at [TP1, TP2]. One value -> single full TP at TP1.
TP_SPLIT = [float(x) for x in os.environ.get("HEATMAP_FWD_TP_SPLIT", "50,50").split(",") if x.strip()]
COOLDOWN_S = int(os.environ.get("HEATMAP_FWD_COOLDOWN_SECONDS", "1800"))
MIN_STOP_PCT = float(os.environ.get("HEATMAP_FWD_MIN_STOP_PCT", "0.0015"))
# Move the runner's stop to breakeven once TP1 fills (makes the TP2 leg risk-free).
BREAKEVEN = os.environ.get("HEATMAP_FWD_BREAKEVEN", "true").lower() in {"1", "true", "yes"}
BE_OFFSET_PCT = float(os.environ.get("HEATMAP_FWD_BE_OFFSET_PCT", "0.0008"))  # nudge past entry to cover fees
# Pyramiding: add a tranche on a later same-direction signal (each its own partial bracket).
# Same-direction adds aggregate into one Bybit position, so we track an "episode" (symbol+side).
MAX_TRANCHES = int(os.environ.get("HEATMAP_FWD_MAX_TRANCHES", "3"))
ADD_COOLDOWN_S = int(os.environ.get("HEATMAP_FWD_ADD_COOLDOWN_SECONDS", "900"))  # min spacing between adds


def pidx(side: str) -> int:
    if not HEDGE_MODE:
        return 0
    return 1 if side == "LONG" else 2


def ekey(symbol: str, side: str) -> str:
    return f"{symbol}|{side}"

LOG_DIR = Path(os.environ.get("LOG_DIR", "/app/logs"))
LEDGER = Path(os.environ.get("HEATMAP_FWD_LEDGER_PATH", str(LOG_DIR / "heatmap_forward_ledger.jsonl")))
STATE = Path(os.environ.get("HEATMAP_FWD_STATE_PATH", str(LOG_DIR / "heatmap_forward_state.json")))

TG_TOKEN = os.environ.get("HEATMAP_TG_BOT_TOKEN", os.environ.get("TELEGRAM_BOT_TOKEN", "")).strip()
TG_UIDS = [int(x) for x in os.environ.get("HEATMAP_TG_ALLOWED_UIDS", "").split(",") if x.strip().lstrip("-").isdigit()]


def now_ms() -> int:
    return int(time.time() * 1000)


def round_to_step(v: float, step: float) -> float:
    if step <= 0:
        return float(v)
    p = max(0, int(round(-math.log10(step))))
    return round(round(float(v) / step) * step, p)


def floor_to_step(v: float, step: float) -> float:
    if step <= 0:
        return float(v)
    p = max(0, int(round(-math.log10(step))))
    return round(math.floor(float(v) / step) * step, p)


def qty_str(v: float, step: float) -> str:
    p = max(0, int(round(-math.log10(step)))) if step > 0 else 8
    return f"{v:.{p}f}"


def tg(text: str) -> None:
    if not TG_TOKEN or not TG_UIDS:
        return
    for uid in TG_UIDS:
        try:
            requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                          json={"chat_id": uid, "text": text, "disable_web_page_preview": True}, timeout=10)
        except Exception:  # noqa: BLE001
            pass


def append_ledger(row: dict[str, Any]) -> None:
    try:
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with open(LEDGER, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    except Exception:  # noqa: BLE001
        log.exception("ledger write failed")


class ForwardBot:
    def __init__(self) -> None:
        self.http: Optional[HTTP] = None
        self.info_cache: dict[str, dict[str, Any]] = {}
        self.lev_set: set[str] = set()
        self.open: dict[str, dict[str, Any]] = {}   # symbol -> trade meta
        self.cooldown: dict[str, int] = {}
        self.long_idx = 0
        self.short_idx = 0
        self.stop = False
        self._load_state()

    # ── persistence ──────────────────────────────────────────────────────────────
    def _load_state(self) -> None:
        try:
            with open(STATE, "r", encoding="utf-8") as fh:
                d = json.load(fh)
            raw = d.get("open", {})
            out: dict[str, Any] = {}
            for k, v in raw.items():
                if isinstance(v, dict) and "tranches" in v:      # already episode format
                    out[k] = v
                elif isinstance(v, dict):                         # migrate old single-trade meta
                    side = v.get("side", "LONG")
                    sym = v.get("symbol", k.split("|")[0])
                    tr = {"entry": v.get("entry"), "sl": v.get("sl"), "tps": v.get("tps") or [],
                          "qty": v.get("qty", 0), "risk": v.get("risk", RISK_USDT), "rr1": v.get("rr1"),
                          "confidence": v.get("confidence"), "types": v.get("types"),
                          "opened_ms": v.get("opened_ms", now_ms()), "link": v.get("link")}
                    out[ekey(sym, side)] = {
                        "symbol": sym, "side": side, "idx": pidx(side), "tranches": [tr],
                        "total_qty": v.get("qty", 0), "total_risk": v.get("risk", RISK_USDT),
                        "be_done": v.get("be_done", False), "opened_ms": v.get("opened_ms", now_ms()),
                        "last_add_ms": v.get("opened_ms", now_ms())}
            self.open = out
            self.cooldown = {k: int(v) for k, v in d.get("cooldown", {}).items()}
            log.info("loaded state: %d open episodes", len(self.open))
        except FileNotFoundError:
            pass
        except Exception:  # noqa: BLE001
            log.exception("state load failed")

    def _save_state(self) -> None:
        try:
            STATE.parent.mkdir(parents=True, exist_ok=True)
            with open(STATE, "w", encoding="utf-8") as fh:
                json.dump({"open": self.open, "cooldown": self.cooldown}, fh)
        except Exception:  # noqa: BLE001
            log.exception("state save failed")

    # ── exchange helpers ─────────────────────────────────────────────────────────
    def _apply_mode(self) -> None:
        # Config is authoritative (detection is unreliable on a fresh account with no
        # positions). Hedge: long positionIdx=1, short=2. One-way: both 0.
        self.long_idx = 1 if HEDGE_MODE else 0
        self.short_idx = 2 if HEDGE_MODE else 0
        try:
            rows = self.http.get_positions(category=CATEGORY, settleCoin="USDT").get("result", {}).get("list", [])
            idxs = {int(r.get("positionIdx", 0)) for r in rows}
            if idxs == {0} and HEDGE_MODE:
                log.warning("HEDGE_MODE=true but account looks one-way (positionIdx=0) — check account setting")
            if (idxs & {1, 2}) and not HEDGE_MODE:
                log.warning("HEDGE_MODE=false but account looks hedge — set HEATMAP_FWD_HEDGE_MODE=true")
        except Exception as exc:  # noqa: BLE001
            log.debug("position-mode verify skipped: %s", exc)

    def _info(self, symbol: str) -> dict[str, Any]:
        if symbol not in self.info_cache:
            it = self.http.get_instruments_info(category=CATEGORY, symbol=symbol)["result"]["list"][0]
            lot, price, lev = it.get("lotSizeFilter", {}), it.get("priceFilter", {}), it.get("leverageFilter", {})
            self.info_cache[symbol] = {
                "status": str(it.get("status", "")),
                "qty_step": float(lot.get("qtyStep", "0.001")), "min_qty": float(lot.get("minOrderQty", "0.001")),
                "tick": float(price.get("tickSize", "0.01")), "max_lev": float(lev.get("maxLeverage", "10") or "10")}
        return self.info_cache[symbol]

    def _ensure_leverage(self, symbol: str, max_lev: float) -> None:
        if symbol in self.lev_set:
            return
        lev = str(min(LEVERAGE, max_lev))
        try:
            self.http.set_leverage(category=CATEGORY, symbol=symbol, buyLeverage=lev, sellLeverage=lev)
        except Exception as exc:  # noqa: BLE001
            if "110043" not in str(exc):  # 110043 = leverage not modified
                log.debug("[%s] set_leverage: %s", symbol, exc)
        self.lev_set.add(symbol)

    # ── REST signal source ───────────────────────────────────────────────────────
    def universe(self) -> list[str]:
        if SYMBOLS_ENV:
            return [s.strip().upper() for s in SYMBOLS_ENV.split(",") if s.strip()]
        try:
            r = requests.get(f"{API_URL}/v1/universe", timeout=10)
            return [x["symbol"] for x in r.json().get("universe", [])]
        except Exception:  # noqa: BLE001
            return []

    def get_setup(self, symbol: str) -> Optional[dict[str, Any]]:
        try:
            d = requests.get(f"{API_URL}/v1/setup/{symbol}", timeout=15).json()
            return d.get("setup")
        except Exception:  # noqa: BLE001
            return None

    # ── trade lifecycle ──────────────────────────────────────────────────────────
    def _place_bracket(self, symbol: str, s: dict[str, Any], info: dict[str, Any]):
        """Place ONE tranche: a risk-sized market entry split into native partial-bracket
        legs (TP1/TP2 + SL). Returns (placed_qty, legs, entry, sl) or (0, [], ...)."""
        entry, sl = float(s["entry"]), float(s["sl"])
        unit_risk = abs(entry - sl)
        if unit_risk <= 0 or unit_risk / entry < MIN_STOP_PCT:
            return 0.0, [], entry, sl
        q_step, min_qty, tick = info["qty_step"], info["min_qty"], info["tick"]
        qty = floor_to_step(RISK_USDT / unit_risk, q_step)
        if qty < min_qty:
            log.info("[%s] risk too small for min qty (%s < %s) — skip", symbol, qty, min_qty)
            return 0.0, [], entry, sl
        side = "Buy" if s["side"] == "LONG" else "Sell"
        idx = pidx(s["side"])
        sl_price = round_to_step(sl, tick)
        tp1, tp2 = float(s["tp1"]), s.get("tp2")
        targets = [(tp1, TP_SPLIT[0]), (float(tp2), TP_SPLIT[1])] if (tp2 and len(TP_SPLIT) >= 2) else [(tp1, 100.0)]
        self._ensure_leverage(symbol, info["max_lev"])
        link = f"HF-{symbol}-{now_ms()}"
        legs, placed, remaining, n = [], 0.0, qty, len(targets)
        for i, (tpp, pct) in enumerate(targets):
            leg_qty = remaining if i == n - 1 else floor_to_step(qty * pct / 100.0, q_step)
            leg_qty = min(leg_qty, remaining)
            if leg_qty < min_qty:
                if i < n - 1:
                    continue
                if leg_qty <= 0:
                    break
            tp_price = round_to_step(tpp, tick)
            try:
                resp = self.http.place_order(
                    category=CATEGORY, symbol=symbol, side=side, orderType="Market",
                    qty=qty_str(leg_qty, q_step), takeProfit=str(tp_price), stopLoss=str(sl_price),
                    tpslMode="Partial", tpOrderType="Market", slOrderType="Market",
                    tpTriggerBy="MarkPrice", slTriggerBy="MarkPrice",
                    positionIdx=idx, orderLinkId=f"{link}-L{i+1}")
            except Exception as exc:  # noqa: BLE001
                log.warning("[%s] bracket leg %d failed: %s", symbol, i + 1, exc)
                continue
            if int(resp.get("retCode", -1)) != 0:
                log.warning("[%s] bracket leg %d rejected: %s %s", symbol, i + 1,
                            resp.get("retCode"), resp.get("retMsg"))
                continue
            placed += leg_qty
            remaining = round(remaining - leg_qty, 10)
            legs.append({"tp": tp_price, "qty": leg_qty})
        return placed, legs, entry, sl

    def execute(self, symbol: str, s: dict[str, Any]) -> None:
        info = self._info(symbol)
        if info["status"] and info["status"] != "Trading":
            return
        side = s["side"]
        key = ekey(symbol, side)
        opp = ekey(symbol, "SHORT" if side == "LONG" else "LONG")
        now = now_ms()
        ep = self.open.get(key)
        if opp in self.open:
            return  # one direction per symbol at a time (keeps exits unambiguous)
        if ep is None:
            if len(self.open) >= MAX_OPEN:
                return
            if now - self.cooldown.get(key, 0) < COOLDOWN_S * 1000:
                return
        else:  # pyramiding gate
            if ep.get("be_done") or len(ep["tranches"]) >= MAX_TRANCHES:
                return
            if now - ep.get("last_add_ms", 0) < ADD_COOLDOWN_S * 1000:
                return

        placed, legs, entry, sl = self._place_bracket(symbol, s, info)
        if placed <= 0 or not legs:
            return
        q_step = info["qty_step"]
        tranche = {"entry": entry, "sl": sl, "tps": [l["tp"] for l in legs], "qty": placed,
                   "risk": RISK_USDT, "rr1": s.get("rr1"), "confidence": s.get("confidence"),
                   "types": s.get("types"), "opened_ms": now}
        tps_str = "/".join(f"{l['tp']:g}" for l in legs)
        if ep is None:
            self.open[key] = {"symbol": symbol, "side": side, "idx": pidx(side), "tranches": [tranche],
                              "total_qty": placed, "total_risk": RISK_USDT, "be_done": False,
                              "opened_ms": now, "last_add_ms": now}
            append_ledger({"event": "open", "symbol": symbol, "side": side, "tranche": 1, "entry": entry,
                           "sl": sl, "tps": tranche["tps"], "qty": placed, "risk": RISK_USDT,
                           "rr1": s.get("rr1"), "confidence": s.get("confidence"), "types": s.get("types"),
                           "opened_ms": now})
            log.info("OPEN %s %s @%.6g SL %.6g TP %s qty %s (%sR, conf %s)", symbol, side, entry, sl,
                     tps_str, qty_str(placed, q_step), s.get("rr1"), s.get("confidence"))
            tg(f"📈 FWD OPEN {symbol} {side} @ {entry:g}\n   SL {sl:g} · TP {tps_str} · "
               f"qty {qty_str(placed, q_step)} · risk ${RISK_USDT:g} · conf {s.get('confidence')}/3 [{s.get('types')}]")
        else:
            ep["tranches"].append(tranche)
            ep["total_qty"] += placed
            ep["total_risk"] += RISK_USDT
            ep["last_add_ms"] = now
            tn = len(ep["tranches"])
            append_ledger({"event": "add", "symbol": symbol, "side": side, "tranche": tn, "entry": entry,
                           "sl": sl, "tps": tranche["tps"], "qty": placed, "risk": RISK_USDT,
                           "total_risk": ep["total_risk"], "opened_ms": now})
            log.info("ADD  %s %s tranche %d @%.6g SL %.6g TP %s qty %s (total risk $%.0f)", symbol, side, tn,
                     entry, sl, tps_str, qty_str(placed, q_step), ep["total_risk"])
            tg(f"➕ FWD ADD {symbol} {side} tranche {tn} @ {entry:g} · SL {sl:g} · TP {tps_str} · "
               f"total risk ${ep['total_risk']:g}")
        self._save_state()

    def _record_close(self, key: str) -> None:
        ep = self.open.pop(key, None)
        self._save_state()
        if ep is None:
            return
        symbol, opened = ep["symbol"], ep.get("opened_ms", 0)
        pnl, exit_px, fills = 0.0, None, 0
        try:
            rows = self.http.get_closed_pnl(category=CATEGORY, symbol=symbol, limit=100).get("result", {}).get("list", [])
            for r in rows:  # sum all closes for this symbol since the episode opened (one side at a time)
                ct = int(r.get("updatedTime") or r.get("createdTime") or 0)
                if ct >= opened - 5000:
                    pnl += float(r.get("closedPnl", 0) or 0)
                    exit_px = exit_px or (float(r.get("avgExitPrice", 0) or 0) or None)
                    fills += 1
        except Exception as exc:  # noqa: BLE001
            log.warning("[%s] closed_pnl fetch failed: %s", symbol, exc)
        risk = ep.get("total_risk", RISK_USDT) or RISK_USDT
        r_mult = pnl / risk if risk else 0.0
        outcome = "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "FLAT"
        self.cooldown[key] = now_ms()
        append_ledger({"event": "close", "symbol": symbol, "side": ep["side"],
                       "tranches": len(ep.get("tranches", [])), "total_qty": ep.get("total_qty"),
                       "total_risk": risk, "pnl": round(pnl, 4), "r_multiple": round(r_mult, 2),
                       "exit_price": exit_px, "fills": fills, "outcome": outcome,
                       "opened_ms": opened, "closed_ms": now_ms()})
        log.info("CLOSE %s %s %s pnl %.4f (%.2fR, %d tranches, %d fills)", symbol, ep["side"], outcome,
                 pnl, r_mult, len(ep.get("tranches", [])), fills)
        tg(f"{'✅' if pnl > 0 else '❌'} FWD CLOSE {symbol} {ep['side']} {outcome} — "
           f"pnl ${pnl:.2f} ({r_mult:+.2f}R, {len(ep.get('tranches', []))} tranches)")

    def _move_to_breakeven(self, key: str, ep: dict[str, Any], cur_size: float, avg_price: float) -> None:
        """Once any TP fills (aggregate size drops), move the whole remaining position's stop to
        breakeven on the position's AVERAGE entry price (+fee offset), staying in PARTIAL mode.
        Cancel only the SL legs; the partial TP conditionals keep running."""
        symbol = ep["symbol"]
        info = self._info(symbol)
        tick, q_step = info["tick"], info["qty_step"]
        side, idx = ep["side"], ep["idx"]
        base = avg_price if avg_price > 0 else ep["tranches"][0]["entry"]
        be = base * (1 + BE_OFFSET_PCT) if side == "LONG" else base * (1 - BE_OFFSET_PCT)
        try:
            oo = self.http.get_open_orders(category=CATEGORY, symbol=symbol).get("result", {}).get("list", [])
            for o in oo:
                if "stoploss" in str(o.get("stopOrderType", "")).lower() and int(o.get("positionIdx", 0)) == idx:
                    try:
                        self.http.cancel_order(category=CATEGORY, symbol=symbol, orderId=o.get("orderId"))
                    except Exception:  # noqa: BLE001
                        pass
        except Exception as exc:  # noqa: BLE001
            log.warning("[%s] BE: list/cancel SL orders failed: %s", symbol, exc)
        try:
            self.http.set_trading_stop(category=CATEGORY, symbol=symbol, tpslMode="Partial",
                                       stopLoss=str(round_to_step(be, tick)), slSize=qty_str(cur_size, q_step),
                                       slTriggerBy="MarkPrice", positionIdx=idx)
            ep["be_done"] = True
            self._save_state()
            append_ledger({"event": "breakeven", "symbol": symbol, "side": side, "be": round(be, 8),
                           "avg_price": round(base, 8), "remaining": cur_size, "ts": now_ms()})
            log.info("BE %s %s -> partial stop %.6g @ avg %.6g (remaining %s)", symbol, side, be, base, cur_size)
            tg(f"🟦 FWD BE {symbol} {side} — stop -> breakeven {be:g} (avg entry) after first TP")
        except Exception as exc:  # noqa: BLE001
            log.warning("[%s] BE set_trading_stop failed (will retry): %s", symbol, exc)

    def sync(self) -> None:
        try:
            rows = self.http.get_positions(category=CATEGORY, settleCoin="USDT").get("result", {}).get("list", [])
        except Exception as exc:  # noqa: BLE001
            log.warning("get_positions failed: %s", exc)
            return
        live = {(r.get("symbol"), int(r.get("positionIdx", 0))): r
                for r in rows if float(r.get("size", 0) or 0) > 0}
        for key in list(self.open):
            ep = self.open[key]
            row = live.get((ep["symbol"], ep["idx"]))
            size = float(row.get("size", 0) or 0) if row else 0.0
            if size <= 0:
                self._record_close(key)
            elif BREAKEVEN and not ep.get("be_done") and ep.get("total_qty") and size < ep["total_qty"] * 0.99:
                self._move_to_breakeven(key, ep, size, float(row.get("avgPrice", 0) or 0))

    # ── main loop ────────────────────────────────────────────────────────────────
    def run(self) -> None:
        self._apply_mode()
        log.info("forward bot live: demo=%s hedge=%s risk=$%.0f/trade max_open=%d tp_split=%s min_conf=%d "
                 "(long_idx=%d short_idx=%d)", DEMO, HEDGE_MODE, RISK_USDT, MAX_OPEN, TP_SPLIT, MIN_CONF,
                 self.long_idx, self.short_idx)
        while not self.stop:
            try:
                self.sync()
                for sym in self.universe():
                    if self.stop:
                        break
                    s = self.get_setup(sym)
                    if s and (s.get("confidence") or 0) >= MIN_CONF:
                        self.execute(sym, s)  # decides: open / pyramid-add / skip (gates inside)
                    time.sleep(0.1)
            except Exception:  # noqa: BLE001
                log.exception("forward loop error")
            for _ in range(POLL_S):
                if self.stop:
                    break
                time.sleep(1)


def main() -> None:
    if not API_KEY or not API_SECRET or not TRADING_ENABLED:
        log.error("no demo API keys / trading disabled — idling. Set HEATMAP_FWD_BYBIT_API_KEY/SECRET "
                  "in bot/.env.heatmap_demo to forward-test.")
        while True:
            time.sleep(3600)
    if not DEMO and not LIVE_CONFIRM:
        raise SystemExit("refusing LIVE trading without HEATMAP_FWD_LIVE_CONFIRM=true")

    bot = ForwardBot()
    bot.http = HTTP(testnet=False, demo=DEMO, api_key=API_KEY, api_secret=API_SECRET)

    def _shutdown(_sig: int, _frame: Any) -> None:
        log.info("shutdown")
        bot.stop = True

    signal_module.signal(signal_module.SIGTERM, _shutdown)
    signal_module.signal(signal_module.SIGINT, _shutdown)
    bot.run()


if __name__ == "__main__":
    main()
