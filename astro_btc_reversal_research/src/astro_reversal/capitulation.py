"""Free Bybit funding-rate + open-interest history, and capitulation features.

A deep liquidity sweep that coincides with a funding flush (deeply negative funding)
and/or a sharp open-interest drop (forced long liquidations) is a higher-conviction
capitulation bottom. These helpers fetch + cache the data and align it to the LTF
bars with no lookahead (only values at-or-before each bar are used).

OI history is available from ~2023-09 on Bybit; funding goes further back.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import reuse

BASE = "https://api.bybit.com"


def _get(path: str, params: dict) -> dict:
    for attempt in range(5):
        try:
            r = requests.get(BASE + path, params=params, timeout=20)
            j = r.json()
            if j.get("retCode") == 0:
                return j["result"]
        except Exception:
            pass
        time.sleep(0.5 * (attempt + 1))
    return {"list": [], "nextPageCursor": ""}


def fetch_funding(symbol: str, start, end, cache_dir: Path | None = None) -> pd.DataFrame:
    cache_dir = cache_dir or reuse.DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"bybit_funding_{reuse.safe_symbol(symbol)}.pkl"
    if path.exists():
        df = pd.read_pickle(path)
    else:
        s_ms = int(pd.Timestamp(reuse.parse_utc_datetime(str(start))).timestamp() * 1000)
        rows, end_ms = [], int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
        while True:
            res = _get("/v5/market/funding/history",
                       {"category": "linear", "symbol": symbol, "endTime": end_ms, "limit": 200})
            lst = res.get("list", [])
            if not lst:
                break
            rows.extend(lst)
            oldest = int(lst[-1]["fundingRateTimestamp"])
            if oldest <= s_ms or len(lst) < 200:
                break
            end_ms = oldest - 1
            time.sleep(0.15)
        df = pd.DataFrame({
            "time": pd.to_datetime([int(x["fundingRateTimestamp"]) for x in rows], unit="ms", utc=True),
            "funding": [float(x["fundingRate"]) for x in rows],
        }).drop_duplicates("time").sort_values("time").reset_index(drop=True)
        df.to_pickle(path)
    return df


def fetch_oi(symbol: str, interval_time: str = "4h", start=None, end=None,
             cache_dir: Path | None = None) -> pd.DataFrame:
    cache_dir = cache_dir or reuse.DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"bybit_oi_{reuse.safe_symbol(symbol)}_{interval_time}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    s_ms = int(pd.Timestamp(reuse.parse_utc_datetime(str(start))).timestamp() * 1000) if start else 0
    rows, cursor, pages = [], "", 0
    while pages < 400:
        params = {"category": "linear", "symbol": symbol, "intervalTime": interval_time, "limit": 200}
        if cursor:
            params["cursor"] = cursor
        res = _get("/v5/market/open-interest", params)
        lst = res.get("list", [])
        if not lst:
            break
        rows.extend(lst)
        cursor = res.get("nextPageCursor", "")
        oldest = int(lst[-1]["timestamp"])
        pages += 1
        if not cursor or oldest <= s_ms:
            break
        time.sleep(0.12)
    df = pd.DataFrame({
        "time": pd.to_datetime([int(x["timestamp"]) for x in rows], unit="ms", utc=True),
        "oi": [float(x["openInterest"]) for x in rows],
    }).drop_duplicates("time").sort_values("time").reset_index(drop=True)
    df.to_pickle(path)
    return df


def capitulation_features(frame: pd.DataFrame, funding: pd.DataFrame, oi: pd.DataFrame) -> pd.DataFrame:
    """Per-bar funding/OI capitulation features aligned to ``frame`` (no lookahead)."""
    out = pd.DataFrame(index=frame.index)
    bars = pd.DataFrame({"open_time": pd.to_datetime(frame["open_time"], utc=True)}).sort_values("open_time")

    if funding is not None and len(funding):
        f = funding.sort_values("time").copy()
        f["funding_z"] = (f["funding"] - f["funding"].rolling(30, min_periods=10).mean()) / \
                         f["funding"].rolling(30, min_periods=10).std()
        m = pd.merge_asof(bars, f.rename(columns={"time": "open_time"}), on="open_time", direction="backward")
        out["funding"] = m["funding"].to_numpy()
        out["funding_z"] = m["funding_z"].to_numpy()

    if oi is not None and len(oi):
        o = oi.sort_values("time").copy()
        o["oi_chg_24h"] = o["oi"] / o["oi"].shift(6) - 1.0      # 6 x 4h = 24h
        o["oi_z"] = (o["oi_chg_24h"] - o["oi_chg_24h"].rolling(60, min_periods=20).mean()) / \
                    o["oi_chg_24h"].rolling(60, min_periods=20).std()
        m = pd.merge_asof(bars, o.rename(columns={"time": "open_time"}), on="open_time", direction="backward")
        out["oi_chg_24h"] = m["oi_chg_24h"].to_numpy()
        out["oi_z"] = m["oi_z"].to_numpy()

    return out.replace([np.inf, -np.inf], np.nan)
