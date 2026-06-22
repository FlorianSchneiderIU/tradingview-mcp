"""Tiny client for the heatmap-bot REST API — import this from other fleet bots
instead of hand-rolling HTTP.

    from heatmap_client import HeatmapClient
    hm = HeatmapClient()                       # defaults to http://heatmap-bot:8110
    s  = hm.structure("BTCUSDT")               # decision-ready market-structure snapshot
    if s and s["nearest_resistance"] and s["nearest_resistance"]["distance_pct"] < 0.3:
        ...                                    # e.g. don't long into a wall just above

All calls are best-effort: on any error they return None (or [] for lists) and never raise,
so a heatmap-bot outage can't take down a strategy bot. Set HEATMAP_API_URL to override the
base URL. Public-data service — no auth.
"""
from __future__ import annotations

import os
from typing import Any, Optional

import requests

DEFAULT_URL = os.environ.get("HEATMAP_API_URL", "http://heatmap-bot:8110").rstrip("/")


class HeatmapClient:
    def __init__(self, base_url: str = DEFAULT_URL, timeout: float = 5.0) -> None:
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    def _get(self, path: str, params: Optional[dict] = None) -> Optional[dict]:
        try:
            r = requests.get(f"{self.base}{path}", params=params or {}, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            return data if isinstance(data, dict) and data.get("success", True) else None
        except Exception:  # noqa: BLE001 - never break the caller
            return None

    # ── reads ────────────────────────────────────────────────────────────────────
    def health(self) -> Optional[dict]:
        return self._get("/health")

    def universe(self) -> list[str]:
        d = self._get("/v1/universe")
        return [r["symbol"] for r in d.get("universe", [])] if d else []

    def structure(self, symbol: str) -> Optional[dict]:
        """CMP, bias, nearest support/resistance, liquidation skew, vol imbalance, funding, OI."""
        return self._get(f"/v1/structure/{symbol.upper()}")

    def levels(self, symbol: str, tf: str = "1h", window: str = "24h", n: int = 12) -> list[dict]:
        d = self._get(f"/v1/levels/{symbol.upper()}", {"tf": tf, "window": window, "n": n})
        return d.get("levels", []) if d else []

    def nearest_magnet(self, symbol: str, tf: str = "1h") -> Optional[dict]:
        d = self._get(f"/v1/liquidations/estimated/{symbol.upper()}/latest", {"tf": tf})
        mags = d.get("magnets", []) if d else []
        price = d.get("last_price") if d else None
        if not mags or not price:
            return None
        return min(mags, key=lambda m: abs(m["price"] / price - 1))

    def estimated(self, symbol: str, tf: str = "1h") -> Optional[dict]:
        return self._get(f"/v1/liquidations/estimated/{symbol.upper()}", {"tf": tf})

    def actual_liquidations(self, symbol: str, limit: int = 200) -> list[dict]:
        d = self._get(f"/v1/liquidations/actual/{symbol.upper()}", {"limit": limit})
        return d.get("events", []) if d else []

    def liquidity(self, symbol: str, limit: int = 1) -> list[dict]:
        d = self._get(f"/v1/liquidity/{symbol.upper()}", {"limit": limit})
        return d.get("snapshots", []) if d else []

    def volume_profile(self, symbol: str, window: str = "24h") -> Optional[dict]:
        return self._get(f"/v1/volume_profile/{symbol.upper()}", {"window": window})

    def cvd(self, symbol: str, window: str = "24h") -> Optional[dict]:
        return self._get(f"/v1/cvd/{symbol.upper()}", {"window": window})

    def screener(self, metric: str = "liq", n: int = 20) -> list[dict]:
        d = self._get("/v1/screener", {"metric": metric, "n": n})
        return d.get("results", []) if d else []

    # ── convenience ───────────────────────────────────────────────────────────────
    def is_near_level(self, symbol: str, price: float, pct: float = 0.003) -> Optional[dict]:
        """Return the nearest key level within `pct` of `price`, else None."""
        for L in self.levels(symbol):
            if price and abs(L["price"] / price - 1) <= pct:
                return L
        return None
