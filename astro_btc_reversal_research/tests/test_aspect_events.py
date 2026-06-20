"""Aspect-event engine correctness.

Anchor: the proposal's known Dark Pivot date 2026-06-24 (16:36 UTC) must appear
as a Moon-Pluto HARD aspect in the computed calendar.
"""

import pandas as pd
import pytest

from astro_reversal import ephemeris_events as ee


@pytest.fixture(scope="module")
def moon_pluto():
    return ee.compute_aspect_events("moon", "pluto", [0, 90, 180, 270], "2026-06-01", "2026-07-15")


def test_known_dark_pivot_date(moon_pluto):
    target = pd.Timestamp("2026-06-24 16:36", tz="UTC")
    deltas = (pd.to_datetime(moon_pluto["timestamp_utc"], utc=True) - target).abs()
    nearest = deltas.min()
    # A hard aspect must land within an hour of the proposal's stated timestamp.
    assert nearest <= pd.Timedelta(hours=1), f"nearest event {nearest} from {target}"


def test_events_are_exact_and_sorted(moon_pluto):
    assert not moon_pluto.empty
    # Refined crossings sit on the target angle (orb residual ~ 0).
    assert moon_pluto["orb_resid_deg"].max() < 0.5
    ts = pd.to_datetime(moon_pluto["timestamp_utc"], utc=True)
    assert ts.is_monotonic_increasing


def test_hard_aspect_cadence(moon_pluto):
    # Moon-Pluto hard aspects recur ~ every 6.8 days; spacing between consecutive
    # hard aspects should be a few days, never months.
    ts = pd.to_datetime(moon_pluto["timestamp_utc"], utc=True).sort_values()
    gaps_days = ts.diff().dropna().dt.total_seconds() / 86_400.0
    assert gaps_days.median() < 9.0


def test_map_events_to_candles_alignment():
    events = ee.compute_aspect_events("moon", "pluto", [0], "2026-06-01", "2026-06-20")
    frame = pd.DataFrame({
        "open_time": pd.date_range("2026-06-01", "2026-06-20", freq="D", tz="UTC"),
        "close_time": pd.date_range("2026-06-02", "2026-06-21", freq="D", tz="UTC"),
    })
    mapped = ee.map_events_to_candles(events, frame)
    for _, row in mapped[mapped["bar_index"] >= 0].iterrows():
        c = int(row["bar_index"])
        assert frame["open_time"].iloc[c] <= row["timestamp_utc"]
