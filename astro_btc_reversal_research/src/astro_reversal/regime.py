"""Market-regime context for the alt-basket long: BTC trend + BTC dominance (BTC.D).

BTC.D is fetched as Binance's BTCDOMUSDT perp (free, from ~2021-06). For an *alt*-long
book the regime that matters:
  * BTC freefall  -> whole market cascading (knife-catch risk),
  * BTC.D ripping  -> capital fleeing alts into BTC (alt rallies fail).

Regime features are computed on daily bars and shifted one day, so a trade entered
intraday only ever sees the previous completed day's regime (no lookahead).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import reuse

BINANCE = "https://fapi.binance.com/fapi/v1/klines"


def fetch_btcdom(interval: str = "1d", start: str = "2021-01-01", end: str | None = None,
                 cache_dir: Path | None = None) -> pd.DataFrame:
    """Paginated Binance BTCDOMUSDT klines (cached)."""
    cache_dir = cache_dir or reuse.DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"binance_btcdom_{interval}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    start_ms = int(pd.Timestamp(reuse.parse_utc_datetime(start)).timestamp() * 1000)
    end_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000) if end is None \
        else int(pd.Timestamp(reuse.parse_utc_datetime(end)).timestamp() * 1000)
    rows, cur = [], start_ms
    while cur < end_ms:
        try:
            r = requests.get(BINANCE, params={"symbol": "BTCDOMUSDT", "interval": interval,
                                              "startTime": cur, "limit": 1500}, timeout=20).json()
        except Exception:
            time.sleep(0.5)
            continue
        if not isinstance(r, list) or not r:
            break
        rows.extend(r)
        last_open = r[-1][0]
        if last_open <= cur or len(r) < 1500:
            break
        cur = last_open + 1
        time.sleep(0.1)
    df = pd.DataFrame({
        "time": pd.to_datetime([x[0] for x in rows], unit="ms", utc=True),
        "open": [float(x[1]) for x in rows], "high": [float(x[2]) for x in rows],
        "low": [float(x[3]) for x in rows], "close": [float(x[4]) for x in rows],
    }).drop_duplicates("time").sort_values("time").reset_index(drop=True)
    df.to_pickle(path)
    return df


def build_daily_regime(btc_daily: pd.DataFrame, btcdom_daily: pd.DataFrame) -> pd.DataFrame:
    """Daily regime features, shifted one day (only prior-day info is visible intraday)."""
    b = pd.DataFrame({"time": pd.to_datetime(btc_daily["open_time"], utc=True).dt.normalize(),
                      "btc_close": btc_daily["close"].astype(float)}).drop_duplicates("time")
    b["btc_7d_ret"] = b["btc_close"] / b["btc_close"].shift(7) - 1.0

    d = btcdom_daily.copy()
    d["time"] = pd.to_datetime(d["time"], utc=True).dt.normalize()
    d["btcd_14d_chg"] = d["close"] / d["close"].shift(14) - 1.0
    d["btcd_ema_dist"] = d["close"] / d["close"].ewm(span=20, adjust=False).mean() - 1.0

    reg = pd.merge(b[["time", "btc_7d_ret"]], d[["time", "btcd_14d_chg", "btcd_ema_dist"]],
                   on="time", how="outer").sort_values("time").reset_index(drop=True)
    # Shift so each calendar day exposes only the PREVIOUS day's completed regime.
    for c in ["btc_7d_ret", "btcd_14d_chg", "btcd_ema_dist"]:
        reg[c] = reg[c].shift(1)
    return reg


def regime_for_trades(trades: list[dict], regime: pd.DataFrame) -> pd.DataFrame:
    """Look up each trade's entry-day regime (most recent completed day)."""
    if not trades:
        return pd.DataFrame()
    et = pd.DataFrame({"time": [pd.Timestamp(t["entry_time"]) for t in trades]}).reset_index()
    et = et.sort_values("time")
    m = pd.merge_asof(et, regime.sort_values("time"), on="time", direction="backward")
    return m.sort_values("index").reset_index(drop=True)


def skip_mask(reg_rows: pd.DataFrame, freefall_thr: float | None, btcd_up_thr: float | None) -> np.ndarray:
    """True where a long should be SKIPPED given the regime rules."""
    n = len(reg_rows)
    skip = np.zeros(n, dtype=bool)
    if freefall_thr is not None:
        ff = reg_rows["btc_7d_ret"].to_numpy()
        skip |= np.where(np.isfinite(ff), ff < freefall_thr, False)
    if btcd_up_thr is not None:
        bd = reg_rows["btcd_14d_chg"].to_numpy()
        skip |= np.where(np.isfinite(bd), bd > btcd_up_thr, False)
    return skip
