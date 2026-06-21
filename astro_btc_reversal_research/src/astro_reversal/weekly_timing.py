"""When (within the week) do the weekly low and weekly high print, and do they get
retested later in the same week?

Weeks are Monday 00:00 -> Sunday 23:59 UTC (matches the Bybit weekly candle). For
each week we locate, at 15m resolution, the bar that made the weekly low and the
weekly high, record its time-of-week (day-of-week, UTC hour, 15m slot, fraction of
week elapsed), and check whether price returns to that level later the same week.

A 'retest' of the low: after the low, price first rises away by ``move_away_frac`` of
the weekly range, then trades back down to within ``retest_tol_frac`` of the low
before the week ends (mirror for the high). This isolates a genuine return-to-level,
not mere consolidation at the extreme.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SLOTS_PER_WEEK = 7 * 96  # 15m slots Mon..Sun


def time_of_week(frame: pd.DataFrame):
    """Return (dow, frac_into_week) arrays for every bar (Mon 00:00 UTC = week start)."""
    ot = pd.to_datetime(frame["open_time"], utc=True)
    week_start = ot.dt.to_period("W").dt.start_time.dt.tz_localize("UTC")
    frac = ((ot - week_start).dt.total_seconds() / (7 * 24 * 3600)).to_numpy()
    return ot.dt.dayofweek.to_numpy(), frac


def _retest_after_low(low, high, idx, p_low, weekly_range, tol_frac, move_away_frac) -> int:
    tol = tol_frac * weekly_range
    move = move_away_frac * weekly_range
    moved = False
    for j in range(idx + 1, len(low)):
        if high[j] >= p_low + move:
            moved = True
        if moved and low[j] <= p_low + tol:
            return j
    return -1


def _retest_after_high(low, high, idx, p_high, weekly_range, tol_frac, move_away_frac) -> int:
    tol = tol_frac * weekly_range
    move = move_away_frac * weekly_range
    moved = False
    for j in range(idx + 1, len(high)):
        if low[j] <= p_high - move:
            moved = True
        if moved and high[j] >= p_high - tol:
            return j
    return -1


def weekly_records(
    frame: pd.DataFrame,
    retest_tol_frac: float = 0.1,
    move_away_frac: float = 0.25,
    min_week_bars: int = 400,
) -> pd.DataFrame:
    """One row per complete week with extreme timings and retest info."""
    ot = pd.to_datetime(frame["open_time"], utc=True).reset_index(drop=True)
    low = frame["low"].to_numpy(float)
    high = frame["high"].to_numpy(float)
    open_ = frame["open"].to_numpy(float)
    close = frame["close"].to_numpy(float)
    week = ot.dt.to_period("W")  # Mon-Sun

    rows = []
    for wk, pos in pd.Series(np.arange(len(ot)), index=week.values).groupby(level=0):
        idx = pos.to_numpy()
        if idx.size < min_week_bars:
            continue
        wlow = low[idx]
        whigh = high[idx]
        i_low = int(idx[np.argmin(wlow)])
        i_high = int(idx[np.argmax(whigh)])
        p_low = float(low[i_low])
        p_high = float(high[i_high])
        wrange = p_high - p_low
        if wrange <= 0:
            continue
        week_start = ot.iloc[idx[0]].normalize()  # Monday 00:00 of this week
        # Local arrays for retest scan (within-week).
        lo_w, hi_w = low[idx], high[idx]
        rel_low = int(np.argmin(wlow))
        rel_high = int(np.argmax(whigh))
        j_low = _retest_after_low(lo_w, hi_w, rel_low, p_low, wrange, retest_tol_frac, move_away_frac)
        j_high = _retest_after_high(lo_w, hi_w, rel_high, p_high, wrange, retest_tol_frac, move_away_frac)

        def tow(ts):
            mins = (ts - week_start).total_seconds() / 60.0
            return {"dow": int(ts.dayofweek), "hour": int(ts.hour),
                    "slot15": int(mins // 15), "frac": float(mins / (7 * 24 * 60))}

        t_low, t_high = ot.iloc[i_low], ot.iloc[i_high]
        rec = {"week": str(wk), "week_start": week_start, "bars": int(idx.size),
               "weekly_range_pct": float(wrange / p_low),
               "open": float(open_[idx[0]]), "close": float(close[idx[-1]]),
               "low_price": p_low, "high_price": p_high,
               "low_time": t_low, "high_time": t_high,
               "low_first": bool(i_low < i_high)}
        for name, ts in (("low", t_low), ("high", t_high)):
            for k, v in tow(ts).items():
                rec[f"{name}_{k}"] = v
        rec["retest_low"] = j_low >= 0
        rec["retest_high"] = j_high >= 0
        if j_low >= 0:
            ts = ot.iloc[idx[j_low]]
            rec["retest_low_dow"] = int(ts.dayofweek)
            rec["retest_low_bars_after"] = int(j_low - rel_low)
        if j_high >= 0:
            ts = ot.iloc[idx[j_high]]
            rec["retest_high_dow"] = int(ts.dayofweek)
            rec["retest_high_bars_after"] = int(j_high - rel_high)
        rows.append(rec)
    return pd.DataFrame(rows)


DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SESSIONS = [("Asia", 0, 8), ("EU", 8, 13), ("US", 13, 21), ("Late", 21, 24)]


def session_of(hour: int) -> str:
    for name, lo, hi in SESSIONS:
        if lo <= hour < hi:
            return name
    return "Late"


def distribution(records: pd.DataFrame, which: str) -> dict:
    """Day-of-week and hour distributions (as fraction of weeks) for 'low' or 'high'."""
    n = len(records)
    dow = records[f"{which}_dow"].value_counts(normalize=True).reindex(range(7), fill_value=0.0)
    hour = records[f"{which}_hour"].value_counts(normalize=True).reindex(range(24), fill_value=0.0)
    sess = records[f"{which}_hour"].map(session_of).value_counts(normalize=True)
    return {
        "n_weeks": int(n),
        "dow": {DOW_NAMES[i]: float(dow[i]) for i in range(7)},
        "hour": {int(h): float(hour[h]) for h in range(24)},
        "session": {k: float(v) for k, v in sess.items()},
        "mean_frac_into_week": float(records[f"{which}_frac"].mean()),
        "median_frac_into_week": float(records[f"{which}_frac"].median()),
    }
