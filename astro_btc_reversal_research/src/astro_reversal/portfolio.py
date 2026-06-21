"""Fixed-fractional equity simulator with a concurrency cap.

Turns a list of R-denominated trades (each with entry_time, exit_time, result_r) into
a real compounding equity curve: risk a fixed fraction of *current* equity per trade,
allow at most K concurrent positions (skip signals when full), realize PnL at exit.
Reports real account drawdown %, CAGR and MAR (CAGR / MaxDD) - not R-space numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_equity(trades: list[dict], risk_pct: float = 0.005, max_concurrent: int = 4,
                    start_equity: float = 1.0) -> dict:
    if not trades:
        return {"trades_taken": 0}
    events = []
    for i, t in enumerate(trades):
        events.append((pd.Timestamp(t["entry_time"]), 1, i))   # entry
        events.append((pd.Timestamp(t["exit_time"]), 0, i))    # exit
    # Exits (type 0) before entries (type 1) at the same timestamp, to free slots.
    events.sort(key=lambda e: (e[0], e[1]))

    equity = start_equity
    open_count = 0
    risk_at: dict[int, float] = {}
    taken: set[int] = set()
    peak = equity
    max_dd = 0.0
    curve = []
    for tstamp, typ, i in events:
        if typ == 0:
            if i in taken:
                equity += risk_at[i] * trades[i]["result_r"]
                open_count -= 1
                peak = max(peak, equity)
                max_dd = min(max_dd, (equity - peak) / peak)
                curve.append((tstamp, equity))
        else:
            if open_count < max_concurrent:
                risk_at[i] = risk_pct * equity
                taken.add(i)
                open_count += 1

    times = [pd.Timestamp(t["entry_time"]) for t in trades]
    years = max((max(times) - min(times)).days / 365.25, 1e-6)
    total_return = equity / start_equity - 1.0
    cagr = (equity / start_equity) ** (1.0 / years) - 1.0
    max_dd_pct = -max_dd
    wins = [trades[i]["result_r"] for i in taken if trades[i]["result_r"] > 0]
    return {
        "trades_taken": len(taken),
        "trades_per_week": round(len(taken) / (years * 52.0), 2),
        "win_rate": round(len(wins) / len(taken) * 100, 1) if taken else float("nan"),
        "total_return_pct": round(total_return * 100, 1),
        "cagr_pct": round(cagr * 100, 1),
        "max_dd_pct": round(max_dd_pct * 100, 1),
        "mar": round(cagr / max_dd_pct, 2) if max_dd_pct > 1e-9 else float("nan"),
        "final_equity": round(equity, 3),
        "risk_pct": risk_pct,
        "max_concurrent": max_concurrent,
    }
