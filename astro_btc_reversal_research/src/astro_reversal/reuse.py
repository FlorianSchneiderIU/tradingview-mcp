"""Single stable import surface for utilities that already live in ``scripts/``.

We deliberately do NOT reimplement Bybit loading, ATR indicators, ATR-based
zigzag pivots, the Skyfield ephemeris, or the trade-stat helpers. They are
battle-tested in ``scripts/research_btc_astro_cycle_timing.py``; we re-export
them here so the rest of the package has one place to import from.
"""

from __future__ import annotations

import sys
from pathlib import Path

# .../astro_btc_reversal_research/src/astro_reversal/reuse.py -> repo root is parents[3]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research_btc_astro_cycle_timing import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    SECONDS_PER_DAY,
    PivotEvent,
    add_ltf_indicators,
    angular_distance_to_aspect,
    astro_features,
    build_feature_matrix,
    calendar_features,
    clean_features,
    compute_skyfield_positions,
    ensure_ohlcv_frame,
    fixed_cycle_features,
    json_default,
    load_bybit_cached,
    make_forward_labels,
    make_model,
    max_drawdown,
    parse_utc_datetime,
    prepare_frame,
    profit_factor,
    safe_average_precision,
    safe_roc_auc,
    safe_symbol,
    shifted_placebo,
    simulate_sweep_trade,
    trade_summary,
    zigzag_pivots,
)
from scripts.backtest_wolfe_wave import (  # noqa: E402
    add_indicators,
    fetch_bybit_klines,
    normalize_timeframe,
    resample_ohlc,
)

__all__ = [
    "REPO_ROOT",
    "DEFAULT_CACHE_DIR",
    "SECONDS_PER_DAY",
    "PivotEvent",
    "angular_distance_to_aspect",
    "astro_features",
    "build_feature_matrix",
    "calendar_features",
    "clean_features",
    "compute_skyfield_positions",
    "ensure_ohlcv_frame",
    "fixed_cycle_features",
    "json_default",
    "load_bybit_cached",
    "make_forward_labels",
    "make_model",
    "max_drawdown",
    "parse_utc_datetime",
    "prepare_frame",
    "profit_factor",
    "safe_average_precision",
    "safe_roc_auc",
    "safe_symbol",
    "shifted_placebo",
    "trade_summary",
    "zigzag_pivots",
    "add_indicators",
    "fetch_bybit_klines",
    "normalize_timeframe",
    "resample_ohlc",
]
