"""Port of the "Power Order Blocks" TradingView indicator into our SMC framework.

Original (Pine v6, MPL-2.0): single-candle *displacement* order block detector.

Definition (faithful to the indicator):
    Bullish OB (demand) at bar i when
        close[i-1] < open[i-1]                          # prior candle bearish
        close[i]   > open[i]                            # current candle bullish
        close[i]   > high[i-1]                          # engulfs prior high (displacement)
        (close[i]-open[i]) > (high[i-1]-low[i-1])*disp  # body large vs prior range
      -> zone = [low[i-1], high[i-1]]   (the prior bearish candle's full range)

    Bearish OB (supply) at bar i when
        close[i-1] > open[i-1]
        close[i]   < open[i]
        close[i]   < low[i-1]
        (open[i]-close[i]) > (high[i-1]-low[i-1])*disp
      -> zone = [low[i-1], high[i-1]]   (the prior bullish candle's full range)

"power %" = (zone_top - zone_bottom) / highest(high-low, str_lookback) * 100, evaluated
at the displacement bar. It is just the OB candle's size relative to the largest range in
the lookback window -- the indicator uses it only for transparency shading, but it is a
natural "strength" score we can filter on.

This module emits supply/demand zone events in the exact dict shape used by
``build_htf_zone_events`` in backtest_turtle_soup.py (keys: time, top, bottom, width) plus
a ``power`` field, so the same downstream zone-retest engine can consume them unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backtest_turtle_soup import resample_ohlc


def build_power_ob_events(
    exec_df: pd.DataFrame,
    timeframe: str,
    disp_thresh: float = 0.5,
    str_lookback: int = 100,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Detect Power Order Blocks on ``timeframe`` candles.

    Returns ``(supply_events, demand_events)`` matching ``build_htf_zone_events``'s shape.
    ``supply_events`` are bearish OBs (resistance / short zones); ``demand_events`` are
    bullish OBs (support / long zones). Each event also carries ``power`` (0-100) and
    ``direction`` ("long"/"short").
    """
    htf = resample_ohlc(exec_df, timeframe).reset_index(drop=True)
    opens = htf["open"].astype(float).to_list()
    highs = htf["high"].astype(float).to_list()
    lows = htf["low"].astype(float).to_list()
    closes = htf["close"].astype(float).to_list()
    close_times = htf["close_time"].to_list()

    rng = htf["high"].astype(float) - htf["low"].astype(float)
    # ta.highest(high-low, lookback) includes the current bar.
    max_range = rng.rolling(str_lookback, min_periods=1).max().to_list()

    supply_events: list[dict[str, Any]] = []
    demand_events: list[dict[str, Any]] = []

    for i in range(1, len(htf)):
        prior_range = highs[i - 1] - lows[i - 1]
        zone_top = highs[i - 1]
        zone_bottom = lows[i - 1]
        denom = max_range[i]
        power = ((zone_top - zone_bottom) / denom) * 100.0 if denom > 0 else 0.0

        is_bearish_prev = closes[i - 1] < opens[i - 1]
        is_bullish_disp = (
            closes[i] > opens[i]
            and closes[i] > highs[i - 1]
            and (closes[i] - opens[i]) > prior_range * disp_thresh
        )
        if is_bearish_prev and is_bullish_disp:
            demand_events.append({
                "time": close_times[i],
                "top": zone_top,
                "bottom": zone_bottom,
                "width": zone_top - zone_bottom,
                "power": power,
                "direction": "long",
            })

        is_bullish_prev = closes[i - 1] > opens[i - 1]
        is_bearish_disp = (
            closes[i] < opens[i]
            and closes[i] < lows[i - 1]
            and (opens[i] - closes[i]) > prior_range * disp_thresh
        )
        if is_bullish_prev and is_bearish_disp:
            supply_events.append({
                "time": close_times[i],
                "top": zone_top,
                "bottom": zone_bottom,
                "width": zone_top - zone_bottom,
                "power": power,
                "direction": "short",
            })

    return supply_events, demand_events
