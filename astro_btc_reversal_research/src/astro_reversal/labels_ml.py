"""Forward reversal-window targets for M3 (proposal section 12.3).

Reuses the existing ``make_forward_labels``: candle ``t`` is labelled 1 if a pivot
of the relevant kind occurs within the next ``horizon`` candles (inclusive). These
are forward-looking -> LABELS ONLY.
"""

from __future__ import annotations

import pandas as pd

from . import reuse

# proposal target name -> label column produced by make_forward_labels
TARGET_TO_COLUMN = {
    "bottom": "y_low",   # bottom_within_N
    "top": "y_high",     # top_within_N
    "pivot": "y_any",    # pivot_within_N
}


def forward_labels(frame: pd.DataFrame, pivots, horizon: int) -> pd.DataFrame:
    return reuse.make_forward_labels(frame, pivots, horizon)


def target_column(target: str) -> str:
    if target not in TARGET_TO_COLUMN:
        raise ValueError(f"unknown target {target!r}; choose from {list(TARGET_TO_COLUMN)}")
    return TARGET_TO_COLUMN[target]
