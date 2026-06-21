"""Equity simulator (fixed-fractional + concurrency cap) and regime gate."""

import numpy as np
import pandas as pd

from astro_reversal import portfolio, regime


def _trade(entry, exit_, r):
    return {"entry_time": pd.Timestamp(entry, tz="UTC"), "exit_time": pd.Timestamp(exit_, tz="UTC"),
            "result_r": r}


def test_equity_compounds_per_trade():
    trades = [_trade("2024-01-01", "2024-01-02", 2.0), _trade("2024-01-03", "2024-01-04", -1.0)]
    s = portfolio.simulate_equity(trades, risk_pct=0.01, max_concurrent=4)
    # 1.0 -> 1.0 + 0.01*2 = 1.02 -> 1.02 - 0.01*1.02 = 1.0098 (reported rounded to 3dp)
    assert abs(s["final_equity"] - 1.01) < 1e-6
    assert s["trades_taken"] == 2


def test_concurrency_cap_skips_overlap():
    # Two overlapping trades; K=1 -> the second can't open.
    trades = [_trade("2024-01-01", "2024-01-05", 1.0), _trade("2024-01-02", "2024-01-06", 1.0)]
    s = portfolio.simulate_equity(trades, risk_pct=0.01, max_concurrent=1)
    assert s["trades_taken"] == 1


def test_max_drawdown_pct():
    # +1R then -1R at 100% risk: equity 1 -> 2 -> 0  => 100% drawdown from the peak.
    trades = [_trade("2024-01-01", "2024-01-02", 1.0), _trade("2024-01-03", "2024-01-04", -1.0)]
    s = portfolio.simulate_equity(trades, risk_pct=1.0, max_concurrent=4)
    assert abs(s["max_dd_pct"] - 100.0) < 1e-6


def test_regime_skip_mask():
    rows = pd.DataFrame({"btc_7d_ret": [-0.30, -0.10, 0.05, np.nan],
                         "btcd_14d_chg": [0.02, 0.10, np.nan, 0.20]})
    skip = regime.skip_mask(rows, freefall_thr=-0.20, btcd_up_thr=0.08)
    assert list(skip) == [True, True, False, True]   # row0 freefall, row1 btcd, row3 btcd


def test_build_daily_regime_no_lookahead():
    n = 30
    btc = pd.DataFrame({"open_time": pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
                        "close": np.linspace(100, 130, n)})
    dom = pd.DataFrame({"time": pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
                        "close": np.linspace(50, 65, n)})
    reg = regime.build_daily_regime(btc, dom)
    # 7d return + a one-day shift -> first 8 rows undefined; finite afterwards.
    assert reg["btc_7d_ret"].iloc[:8].isna().all()
    assert reg["btc_7d_ret"].iloc[10:].notna().all()
