"""Exact planetary aspect event calendar via root-finding.

The reused Skyfield helper gives per-sample geocentric ecliptic longitudes; here
we sample on a fine regular grid (default hourly), form the *signed phase*
``(lon_1 - lon_2) mod 360`` and locate the exact times the phase crosses each
target aspect angle (0/90/180/270 for the Dark Pivot, or any list). Crossings are
refined by linear interpolation between adjacent grid samples.

These events are deterministic and known in advance, so mapping them onto candles
is NOT market-data leakage (proposal section 11/32).
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import reuse

ASPECT_NAMES = {
    0: "conjunction",
    30: "semisextile",
    45: "semisquare",
    60: "sextile",
    72: "quintile",
    90: "square",
    120: "trine",
    135: "sesquiquadrate",
    144: "biquintile",
    150: "quincunx",
    180: "opposition",
    270: "square_lower",
}


def aspect_name(angle: float) -> str:
    return ASPECT_NAMES.get(int(round(angle)), f"aspect_{angle:g}")


def _wrap_180(values: np.ndarray) -> np.ndarray:
    """Wrap degrees into [-180, 180)."""
    return (values + 180.0) % 360.0 - 180.0


def _hourly_grid(start: pd.Timestamp, end: pd.Timestamp, grid_minutes: int) -> pd.Series:
    start = pd.Timestamp(start).tz_convert("UTC").floor("h")
    end = pd.Timestamp(end).tz_convert("UTC").ceil("h")
    index = pd.date_range(start, end, freq=f"{grid_minutes}min", tz="UTC")
    return pd.Series(index, name="time")


def _grid_timeframe_label(grid_minutes: int) -> str:
    """Map a grid step to a timeframe label accepted by ``normalize_timeframe``."""
    if grid_minutes % 60 == 0:
        return f"{grid_minutes // 60}h"
    return f"{grid_minutes}m"


def _body_longitudes(
    times: pd.Series,
    cache_dir: Path,
    grid_minutes: int,
) -> pd.DataFrame:
    label = _grid_timeframe_label(grid_minutes)
    raw = reuse.compute_skyfield_positions(times, cache_dir, label)
    if raw.empty:
        raise RuntimeError("Skyfield ephemeris unavailable; cannot compute aspect events.")
    return raw.reset_index(drop=True)


def compute_aspect_events(
    body_1: str,
    body_2: str,
    aspects: list[float],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    cache_dir: Path | None = None,
    grid_minutes: int = 60,
) -> pd.DataFrame:
    """Return a DataFrame of exact aspect events between two bodies.

    Columns: timestamp_utc, body_1, body_2, aspect_angle, aspect_name,
    orb_resid_deg, relative_speed, body_1_speed, body_2_speed, crossing_rising.
    """
    cache_dir = cache_dir or reuse.DEFAULT_CACHE_DIR
    start_ts = pd.Timestamp(reuse.parse_utc_datetime(str(start)))
    end_ts = pd.Timestamp(reuse.parse_utc_datetime(str(end)))
    grid = _hourly_grid(start_ts, end_ts, grid_minutes)
    raw = _body_longitudes(grid, cache_dir, grid_minutes)

    lon1 = raw[f"{body_1}_lon"].to_numpy(dtype=float)
    lon2 = raw[f"{body_2}_lon"].to_numpy(dtype=float)
    phase = (lon1 - lon2) % 360.0

    # Longitude speeds (deg/day) via gradient of unwrapped longitudes.
    grid_days = (grid - grid.iloc[0]).dt.total_seconds().to_numpy(dtype=float) / reuse.SECONDS_PER_DAY
    speed1 = np.gradient(np.unwrap(np.deg2rad(lon1)), grid_days) * 180.0 / np.pi
    speed2 = np.gradient(np.unwrap(np.deg2rad(lon2)), grid_days) * 180.0 / np.pi

    grid_ts = grid.reset_index(drop=True)
    records: list[dict] = []
    # Guard against the +/-180 wrap discontinuity (real motion per step is tiny).
    max_step_deg = 90.0

    for aspect in aspects:
        resid = _wrap_180(phase - float(aspect))
        sign = np.sign(resid)
        for i in range(len(resid) - 1):
            r0, r1 = resid[i], resid[i + 1]
            if r0 == 0.0:
                frac = 0.0
            elif sign[i] != 0 and sign[i + 1] != 0 and sign[i] != sign[i + 1]:
                if abs(r1 - r0) > max_step_deg:
                    continue  # wrap artifact, not a real crossing
                frac = -r0 / (r1 - r0)
            else:
                continue
            ts = grid_ts.iloc[i] + (grid_ts.iloc[i + 1] - grid_ts.iloc[i]) * float(frac)
            s1 = float(speed1[i] + frac * (speed1[i + 1] - speed1[i]))
            s2 = float(speed2[i] + frac * (speed2[i + 1] - speed2[i]))
            records.append(
                {
                    "timestamp_utc": ts,
                    "body_1": body_1,
                    "body_2": body_2,
                    "aspect_angle": float(aspect),
                    "aspect_name": aspect_name(aspect),
                    "orb_resid_deg": float(abs(r0 + frac * (r1 - r0))),
                    "relative_speed": s1 - s2,
                    "body_1_speed": s1,
                    "body_2_speed": s2,
                    "crossing_rising": bool(r1 > r0),
                }
            )

    if not records:
        return pd.DataFrame(
            columns=[
                "timestamp_utc", "body_1", "body_2", "aspect_angle", "aspect_name",
                "orb_resid_deg", "relative_speed", "body_1_speed", "body_2_speed", "crossing_rising",
            ]
        )
    events = pd.DataFrame.from_records(records).sort_values("timestamp_utc").reset_index(drop=True)
    return events


def map_events_to_candles(events: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the containing candle's bar_index to each event.

    An event maps to candle ``c`` if ``open_time[c] <= t < open_time[c+1]``.
    Events outside the frame span get bar_index = -1.
    """
    if events.empty:
        out = events.copy()
        out["bar_index"] = pd.Series(dtype=int)
        return out
    opens = pd.to_datetime(frame["open_time"], utc=True).to_numpy()
    ts = pd.to_datetime(events["timestamp_utc"], utc=True).to_numpy()
    idx = np.searchsorted(opens, ts, side="right") - 1
    idx[idx < 0] = -1
    out = events.copy()
    out["bar_index"] = idx.astype(int)
    out["candle_open_time"] = [
        pd.Timestamp(frame["open_time"].iloc[i]) if i >= 0 else pd.NaT for i in idx
    ]
    return out


def candle_event_mask(events: pd.DataFrame, n_bars: int) -> np.ndarray:
    """Boolean array (len n_bars) marking candles that contain at least one event."""
    mask = np.zeros(n_bars, dtype=bool)
    if events.empty or "bar_index" not in events.columns:
        return mask
    valid = events["bar_index"].to_numpy(dtype=int)
    valid = valid[(valid >= 0) & (valid < n_bars)]
    mask[valid] = True
    return mask
