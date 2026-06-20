"""Config + OHLCV loading helpers.

Loads a fine base interval from the offline Bybit cache and resamples up to the
target timeframe via the reused ``prepare_frame`` (which also attaches ATR and a
``bar_index`` column).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from . import reuse

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "aspects.yaml"
REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"


def load_config(path: Path | None = None) -> dict[str, Any]:
    path = path or CONFIG_PATH
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_ohlcv(
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    base_interval: str = "15m",
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Return an OHLCV frame at ``timeframe`` with ATR + bar_index columns.

    The base interval (default 15m) is loaded from the shared cache and resampled
    up to the (coarser) target timeframe. This keeps the run offline + deterministic.
    """
    cache_dir = cache_dir or reuse.DEFAULT_CACHE_DIR
    start_dt = reuse.parse_utc_datetime(start)
    end_dt = reuse.parse_utc_datetime(end)
    base = reuse.load_bybit_cached(symbol, base_interval, start_dt, end_dt, cache_dir)
    frame = reuse.prepare_frame(base, timeframe)
    return frame
