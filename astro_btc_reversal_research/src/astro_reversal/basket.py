"""Multi-symbol basket of liquid Bybit linear perps (for breadth of setups)."""

from __future__ import annotations

from datetime import timezone
from pathlib import Path

import pandas as pd

from . import reuse

# Liquid Bybit USDT perps with reasonable history. Newer listings simply contribute
# fewer bars (the loader skips ranges before listing).
BASKET = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "LTCUSDT", "ATOMUSDT",
    "NEARUSDT", "FILUSDT", "ARBUSDT", "OPUSDT", "INJUSDT", "SUIUSDT",
]


def load_symbol(symbol: str, interval: str, start: str, end: str,
                cache_dir: Path | None = None) -> pd.DataFrame:
    """Cached OHLCV for one symbol at ``interval`` (fetches from Bybit if missing)."""
    cache_dir = cache_dir or reuse.DEFAULT_CACHE_DIR
    s = reuse.parse_utc_datetime(start)
    e = reuse.parse_utc_datetime(end)
    base = reuse.load_bybit_cached(symbol, interval, s, e, cache_dir)
    frame = reuse.prepare_frame(base, interval)
    return frame
