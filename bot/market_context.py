from __future__ import annotations

import math
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np
from pybit.unified_trading import HTTP

from indicators import ATR_LEN, ATR_PCTILE_WIN, VOL_WIN, atr_pctile_last, ind_atr, vol_ratio_last


class MarketContextEnricher:
    """Build a consistent market_context payload for RL consumers."""

    def __init__(
        self,
        http_client: HTTP,
        *,
        fetch_market_snapshot: Callable[[str], dict[str, Any]] | None = None,
        logger: Any | None = None,
    ) -> None:
        self.http = http_client
        self.fetch_market_snapshot = fetch_market_snapshot
        self.log = logger
        self.oi_history: dict[str, deque[dict[str, Any]]] = {}
        self.funding_history: dict[str, deque[dict[str, Any]]] = {}
        self.basis_history: dict[str, deque[dict[str, Any]]] = {}
        self.spread_bps_history: dict[str, deque[dict[str, Any]]] = {}
        self.price_history: dict[str, deque[dict[str, Any]]] = {}

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

    @staticmethod
    def _percentile_rank(value: float | None, history_values: list[float]) -> float | None:
        if value is None or not history_values:
            return None
        arr = np.asarray(history_values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None
        rank = float(np.mean(arr <= float(value)))
        return rank if math.isfinite(rank) else None

    @staticmethod
    def _extract_response_list(resp: dict[str, Any]) -> list[dict[str, Any]]:
        items = resp.get("result", {}).get("list", []) if isinstance(resp, dict) else []
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
        return []

    def _debug(self, msg: str, *args: Any) -> None:
        if self.log is not None:
            self.log.debug(msg, *args)

    def _fetch_snapshot(self, symbol: str) -> dict[str, Any]:
        if self.fetch_market_snapshot is not None:
            try:
                snapshot = self.fetch_market_snapshot(symbol)
                return dict(snapshot) if isinstance(snapshot, dict) else {}
            except Exception as exc:
                self._debug("[%s] shared market snapshot callback failed: %s", symbol, exc)
        try:
            resp = self.http.get_tickers(category="linear", symbol=symbol)
            items = self._extract_response_list(resp)
            if items:
                return dict(items[0])
        except Exception as exc:
            self._debug("[%s] shared market snapshot fetch failed: %s", symbol, exc)
        return {}

    def _fetch_recent_bars(self, symbol: str, limit: int = 300) -> list[dict[str, Any]]:
        try:
            resp = self.http.get_kline(category="linear", symbol=symbol, interval="5", limit=limit)
            items = resp.get("result", {}).get("list", []) if isinstance(resp, dict) else []
            if not isinstance(items, list):
                return []
            bars: list[dict[str, Any]] = []
            for it in reversed(items):
                if not isinstance(it, (list, tuple)) or len(it) < 6:
                    continue
                ts = int(self._to_float(it[0]) or 0)
                o = self._to_float(it[1])
                h = self._to_float(it[2])
                l = self._to_float(it[3])
                c = self._to_float(it[4])
                v = self._to_float(it[5])
                if ts <= 0 or None in (o, h, l, c, v):
                    continue
                bars.append({"ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": v})
            return bars
        except Exception as exc:
            self._debug("[%s] shared kline fetch failed: %s", symbol, exc)
            return []

    def _build_regime_context(self, bars: list[dict[str, Any]], sig: dict[str, Any]) -> dict[str, Any]:
        if len(bars) < max(ATR_LEN + 5, VOL_WIN + 5, ATR_PCTILE_WIN + 5):
            return {}
        try:
            c = np.array([float(b["close"]) for b in bars], dtype=np.float64)
            h = np.array([float(b["high"]) for b in bars], dtype=np.float64)
            l = np.array([float(b["low"]) for b in bars], dtype=np.float64)
            v = np.array([float(b["volume"]) for b in bars], dtype=np.float64)

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

            return {
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
        except Exception as exc:
            self._debug("[%s] shared regime context build failed: %s", sig.get("symbol") or "-", exc)
            return {}

    def build_context(
        self,
        *,
        symbol: str,
        sig: dict[str, Any],
        instrument_info: dict[str, Any] | None = None,
        provenance: dict[str, Any] | None = None,
        recent_bars: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        local_dt = datetime.now(timezone.utc)
        context_now_ms = int(local_dt.timestamp() * 1000)
        context: dict[str, Any] = {
            "captured_at_utc": local_dt.isoformat(),
            "captured_at_ms": context_now_ms,
            "symbol": symbol,
            "strategy": sig.get("strategy"),
            "direction": sig.get("signal") or sig.get("direction"),
        }

        try:
            server_resp = self.http.get_server_time()
            context["server_time"] = server_resp.get("result") if isinstance(server_resp, dict) else None
        except Exception as exc:
            self._debug("[%s] shared server time fetch failed: %s", symbol, exc)

        ticker = self._fetch_snapshot(symbol)
        context["ticker"] = ticker

        mark = self._to_float(ticker.get("markPrice")) if ticker else None
        index = self._to_float(ticker.get("indexPrice")) if ticker else None
        basis_pct = ((mark - index) / index) if (mark is not None and index and index != 0) else None
        self.basis_history.setdefault(symbol, deque(maxlen=512))
        if basis_pct is not None and math.isfinite(basis_pct):
            self.basis_history[symbol].append({"ts": context_now_ms, "basis_pct": basis_pct})
        basis_values = [
            v for v in (self._to_float(item.get("basis_pct")) for item in self.basis_history[symbol]) if v is not None
        ]
        context["basis"] = {
            "mark_price": mark,
            "index_price": index,
            "basis_pct": basis_pct if basis_pct is None or math.isfinite(basis_pct) else None,
            "divergence_pct": basis_pct if basis_pct is None or math.isfinite(basis_pct) else None,
            "divergence_zscore": self._zscore(basis_pct, basis_values[-120:]),
        }

        try:
            ob_resp = self.http.get_orderbook(category="linear", symbol=symbol, limit=50)
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

            def notional_sum(entries: list[Any]) -> float:
                total = 0.0
                for entry in entries:
                    p = self._to_float(entry[0])
                    q = self._to_float(entry[1])
                    if p is not None and q is not None:
                        total += p * q
                return total

            bid_notional_top10 = notional_sum(bids[:10])
            ask_notional_top10 = notional_sum(asks[:10])
            bid_notional_top3 = notional_sum(bids[:3])
            ask_notional_top3 = notional_sum(asks[:3])
            bid_notional_far = notional_sum(bids[3:10])
            ask_notional_far = notional_sum(asks[3:10])

            denom = bid_notional_top10 + ask_notional_top10
            imbalance = ((bid_notional_top10 - ask_notional_top10) / denom) if denom > 0 else None
            near_denom = bid_notional_top3 + ask_notional_top3
            near_imbalance = ((bid_notional_top3 - ask_notional_top3) / near_denom) if near_denom > 0 else None
            far_denom = bid_notional_far + ask_notional_far
            far_imbalance = ((bid_notional_far - ask_notional_far) / far_denom) if far_denom > 0 else None
            depth_skew_near_far = near_imbalance - far_imbalance if near_imbalance is not None and far_imbalance is not None else None

            self.spread_bps_history.setdefault(symbol, deque(maxlen=512))
            if spread_bps is not None and math.isfinite(spread_bps):
                self.spread_bps_history[symbol].append({"ts": context_now_ms, "spread_bps": spread_bps})
            spread_values = [
                v for v in (self._to_float(item.get("spread_bps")) for item in self.spread_bps_history[symbol]) if v is not None
            ]
            context["orderbook"] = {
                "best_bid": best_bid,
                "best_ask": best_ask,
                "spread": spread,
                "spread_bps": spread_bps,
                "spread_bps_zscore": self._zscore(spread_bps, spread_values[-200:]),
                "spread_bps_percentile_200": self._percentile_rank(spread_bps, spread_values[-200:]),
                "bid_notional_top10": bid_notional_top10,
                "ask_notional_top10": ask_notional_top10,
                "imbalance_top10": imbalance,
                "bid_notional_top3": bid_notional_top3,
                "ask_notional_top3": ask_notional_top3,
                "imbalance_top3": near_imbalance,
                "bid_notional_far_4_10": bid_notional_far,
                "ask_notional_far_4_10": ask_notional_far,
                "imbalance_far_4_10": far_imbalance,
                "depth_skew_near_far": depth_skew_near_far,
                "raw": ob,
            }
        except Exception as exc:
            self._debug("[%s] shared orderbook fetch failed: %s", symbol, exc)

        try:
            trades_resp = self.http.get_public_trade_history(category="linear", symbol=symbol, limit=200)
            trades = self._extract_response_list(trades_resp)
            buy_notional = 0.0
            sell_notional = 0.0
            buy_qty = 0.0
            sell_qty = 0.0
            count = 0
            for trade in trades:
                ts = int(self._to_float(trade.get("time")) or self._to_float(trade.get("T")) or 0)
                if ts <= 0 or context_now_ms - ts > 60_000:
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
            self._debug("[%s] shared public trade history fetch failed: %s", symbol, exc)

        oi_current = self._to_float(ticker.get("openInterest")) if ticker else None
        oi_history: list[dict[str, Any]] = []
        try:
            oi_resp = self.http.get_open_interest(category="linear", symbol=symbol, intervalTime="5min", limit=24)
            for row in self._extract_response_list(oi_resp):
                ts = int(self._to_float(row.get("timestamp")) or 0)
                oi = self._to_float(row.get("openInterest"))
                if ts > 0 and oi is not None:
                    oi_history.append({"ts": ts, "open_interest": oi})
            oi_history.sort(key=lambda item: int(item["ts"]))
        except Exception as exc:
            self._debug("[%s] shared open interest history fetch failed: %s", symbol, exc)

        if oi_current is None and oi_history:
            oi_current = self._to_float(oi_history[-1].get("open_interest"))

        self.oi_history.setdefault(symbol, deque(maxlen=512))
        for row in oi_history:
            if not self.oi_history[symbol] or self.oi_history[symbol][-1].get("ts") != row.get("ts"):
                self.oi_history[symbol].append(row)
        oi_values = [v for v in (self._to_float(item.get("open_interest")) for item in self.oi_history[symbol]) if v is not None]

        oi_delta_5m = oi_current - oi_values[-2] if oi_current is not None and len(oi_values) >= 2 else None
        oi_delta_15m = oi_current - oi_values[-4] if oi_current is not None and len(oi_values) >= 4 else None
        oi_delta_1h = oi_current - oi_values[-13] if oi_current is not None and len(oi_values) >= 13 else None

        self.price_history.setdefault(symbol, deque(maxlen=1024))
        current_price = mark or self._to_float(ticker.get("lastPrice") if ticker else None) or index
        if current_price is not None and current_price > 0 and math.isfinite(current_price):
            if not self.price_history[symbol] or int(self.price_history[symbol][-1].get("ts") or 0) != context_now_ms:
                self.price_history[symbol].append({"ts": context_now_ms, "price": current_price})
        price_1h_ago = None
        for item in reversed(self.price_history[symbol]):
            ts = int(self._to_float(item.get("ts")) or 0)
            if ts <= context_now_ms - 3_600_000:
                price_1h_ago = self._to_float(item.get("price"))
                break
        price_ret_1h = (current_price - price_1h_ago) / price_1h_ago if current_price is not None and price_1h_ago not in (None, 0.0) else None
        oi_delta_1h_pct = (oi_delta_1h / oi_current) if oi_current not in (None, 0.0) and oi_delta_1h is not None else None

        oi_price_state_code = 0.0
        if oi_delta_1h_pct is not None and price_ret_1h is not None:
            eps = 1e-6
            if oi_delta_1h_pct > eps and price_ret_1h > eps:
                oi_price_state_code = 2.0
            elif oi_delta_1h_pct < -eps and price_ret_1h < -eps:
                oi_price_state_code = -2.0
            elif oi_delta_1h_pct < -eps and price_ret_1h > eps:
                oi_price_state_code = 1.0
            elif oi_delta_1h_pct > eps and price_ret_1h < -eps:
                oi_price_state_code = -1.0

        context["open_interest"] = {
            "current": oi_current,
            "delta_5m": oi_delta_5m,
            "delta_15m": oi_delta_15m,
            "delta_1h": oi_delta_1h,
            "delta_1h_pct": oi_delta_1h_pct,
            "price_ret_1h": price_ret_1h,
            "oi_price_state_code": oi_price_state_code,
            "zscore": self._zscore(oi_current, oi_values[-100:]),
            "history": list(self.oi_history[symbol])[-120:],
        }

        funding_current = self._to_float(ticker.get("fundingRate")) if ticker else None
        funding_history: list[dict[str, Any]] = []
        try:
            funding_resp = self.http.get_funding_rate_history(category="linear", symbol=symbol, limit=50)
            for row in self._extract_response_list(funding_resp):
                ts = int(self._to_float(row.get("fundingRateTimestamp")) or self._to_float(row.get("fundingRateTs")) or 0)
                rate = self._to_float(row.get("fundingRate"))
                if ts > 0 and rate is not None:
                    funding_history.append({"ts": ts, "funding_rate": rate})
            funding_history.sort(key=lambda item: int(item["ts"]))
        except Exception as exc:
            self._debug("[%s] shared funding history fetch failed: %s", symbol, exc)

        self.funding_history.setdefault(symbol, deque(maxlen=512))
        for row in funding_history:
            if not self.funding_history[symbol] or self.funding_history[symbol][-1].get("ts") != row.get("ts"):
                self.funding_history[symbol].append(row)
        funding_values = [
            v for v in (self._to_float(item.get("funding_rate")) for item in self.funding_history[symbol]) if v is not None
        ]
        if funding_current is None and funding_values:
            funding_current = funding_values[-1]

        funding_1h_ago = None
        funding_4h_ago = None
        for item in reversed(self.funding_history[symbol]):
            ts = int(self._to_float(item.get("ts")) or 0)
            if funding_1h_ago is None and ts <= context_now_ms - 3_600_000:
                funding_1h_ago = self._to_float(item.get("funding_rate"))
            if funding_4h_ago is None and ts <= context_now_ms - 14_400_000:
                funding_4h_ago = self._to_float(item.get("funding_rate"))
            if funding_1h_ago is not None and funding_4h_ago is not None:
                break
        funding_delta_1h = (funding_current - funding_1h_ago) if funding_current is not None and funding_1h_ago is not None else None
        funding_delta_4h = (funding_current - funding_4h_ago) if funding_current is not None and funding_4h_ago is not None else None
        funding_accel = (funding_delta_1h - funding_delta_4h) if funding_delta_1h is not None and funding_delta_4h is not None else None

        next_funding_ms = int(self._to_float(ticker.get("nextFundingTime")) or 0) if ticker else 0
        mins_to_next_funding = ((next_funding_ms - context_now_ms) / 60000.0) if next_funding_ms > 0 else None
        context["funding"] = {
            "current": funding_current,
            "delta_1h": funding_delta_1h,
            "delta_4h": funding_delta_4h,
            "accel_1h_minus_4h": funding_accel,
            "next_funding_time_ms": next_funding_ms if next_funding_ms > 0 else None,
            "minutes_to_next_funding": mins_to_next_funding,
            "zscore": self._zscore(funding_current, funding_values[-100:]),
            "history": list(self.funding_history[symbol])[-120:],
        }

        bars = recent_bars or self._fetch_recent_bars(symbol)
        if bars:
            closes = [self._to_float(bar.get("close")) for bar in bars]
            closes = [value for value in closes if value is not None]
            vols = [self._to_float(bar.get("volume")) for bar in bars]
            vols = [value for value in vols if value is not None]
            ret_1h = ((closes[-1] / closes[-13]) - 1.0) * 100.0 if len(closes) >= 13 and closes[-13] not in (None, 0.0) else None
            ret_4h = ((closes[-1] / closes[-49]) - 1.0) * 100.0 if len(closes) >= 49 and closes[-49] not in (None, 0.0) else None
            vol_mult = None
            if len(vols) >= 20:
                vol_sma20 = float(np.mean(vols[-20:]))
                if vol_sma20 > 0 and vols[-1] is not None:
                    vol_mult = float(vols[-1]) / vol_sma20
            context["derived"] = {
                "ret_1h": ret_1h,
                "ret_4h": ret_4h,
                "symbol_vol_mult": vol_mult,
            }
            context["regime"] = self._build_regime_context(bars, sig)
        else:
            context["derived"] = {}
            context["regime"] = {}

        tick_size = self._to_float((instrument_info or {}).get("tick_size"))
        entry_f = self._to_float(sig.get("entry"))
        sl_f = self._to_float(sig.get("sl") or sig.get("stop_loss"))
        tp_f = self._to_float(sig.get("tp1", sig.get("target", sig.get("take_profit"))))
        stop_ticks = abs(entry_f - sl_f) / tick_size if entry_f is not None and sl_f is not None and tick_size and tick_size > 0 else None
        target_ticks = abs(tp_f - entry_f) / tick_size if entry_f is not None and tp_f is not None and tick_size and tick_size > 0 else None
        context["execution_plan"] = {
            "expected_entry": sig.get("entry"),
            "expected_stop": sig.get("sl") or sig.get("stop_loss"),
            "expected_target": sig.get("tp1", sig.get("target", sig.get("take_profit"))),
            "trail_dist": sig.get("trail_dist"),
            "tick_size": (instrument_info or {}).get("tick_size"),
            "stop_distance_ticks": stop_ticks,
            "target_distance_ticks": target_ticks,
            "qty_step": (instrument_info or {}).get("qty_step"),
            "min_qty": (instrument_info or {}).get("min_qty"),
        }
        if instrument_info:
            context["instrument_constraints"] = {
                "qty_step": instrument_info.get("qty_step"),
                "min_qty": instrument_info.get("min_qty"),
                "tick_size": instrument_info.get("tick_size"),
                "min_leverage": instrument_info.get("min_leverage_raw", instrument_info.get("min_leverage")),
                "max_leverage": instrument_info.get("max_leverage_raw", instrument_info.get("max_leverage")),
                "leverage_step": instrument_info.get("leverage_step_raw", instrument_info.get("leverage_step")),
            }

        if provenance:
            context["provenance"] = provenance
        return context
