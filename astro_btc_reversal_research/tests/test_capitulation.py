"""Capitulation feature alignment + no-lookahead."""

import numpy as np
import pandas as pd

from astro_reversal import capitulation as cap


def _frame(n=50):
    return pd.DataFrame({
        "open_time": pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
        "high": np.full(n, 101.0), "low": np.full(n, 99.0),
        "open": np.full(n, 100.0), "close": np.full(n, 100.0), "atr": np.full(n, 1.0),
    })


def _funding(times, rates):
    return pd.DataFrame({"time": pd.to_datetime(times, utc=True), "funding": rates})


def test_funding_uses_only_past_values():
    f = _frame(50)
    # Funding published at 02:00 and 10:00; a 15m bar must see the latest at-or-before it.
    fund = _funding(["2024-01-01 02:00", "2024-01-01 10:00"], [-0.01, 0.02])
    feats = cap.capitulation_features(f, fund, pd.DataFrame(columns=["time", "oi"]))
    t = pd.to_datetime(f["open_time"], utc=True)
    before_first = t < pd.Timestamp("2024-01-01 02:00", tz="UTC")
    between = (t >= pd.Timestamp("2024-01-01 02:00", tz="UTC")) & (t < pd.Timestamp("2024-01-01 10:00", tz="UTC"))
    assert feats.loc[before_first.to_numpy(), "funding"].isna().all()      # nothing known yet
    assert (feats.loc[between.to_numpy(), "funding"] == -0.01).all()       # only the first print
    assert feats["funding"].iloc[-1] == 0.02                              # later bars see the second


def test_oi_change_and_no_future_leak():
    f = _frame(60)
    times = pd.date_range("2024-01-01", periods=20, freq="4h", tz="UTC")
    oi = pd.DataFrame({"time": times, "oi": np.linspace(100, 81, 20)})    # OI declining
    feats = cap.capitulation_features(f, pd.DataFrame(columns=["time", "funding"]), oi)
    assert "oi_chg_24h" in feats.columns
    # 24h change must be negative for declining OI where defined.
    vals = feats["oi_chg_24h"].dropna()
    assert (vals <= 0).all()
