"""Walk-forward splits with embargo/purging (proposal section 19).

Random splits are forbidden. We use expanding-window walk-forward folds over the
development period, then a final untouched holdout. Because targets look forward
``horizon`` candles, the last ``embargo`` (= horizon) candles of every training
block are purged so their label windows cannot overlap the test block.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def split_dev_holdout(times: pd.Series, holdout_start: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
    ts = pd.to_datetime(times, utc=True)
    dev = np.flatnonzero((ts < holdout_start).to_numpy())
    holdout = np.flatnonzero((ts >= holdout_start).to_numpy())
    return dev, holdout


def walk_forward_folds(
    dev_idx: np.ndarray,
    n_folds: int,
    embargo: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-train / next-block-test folds over ``dev_idx``.

    The first block seeds the initial training set; each subsequent block is an
    out-of-sample test set, trained on everything before it minus the embargo.
    """
    dev_idx = np.asarray(dev_idx, dtype=int)
    if dev_idx.size == 0 or n_folds < 1:
        return []
    blocks = [b for b in np.array_split(dev_idx, n_folds + 1) if b.size > 0]
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    train_end = blocks[0][-1]
    for i in range(1, len(blocks)):
        test = blocks[i]
        train = dev_idx[dev_idx <= train_end]
        if embargo > 0 and train.size > embargo:
            train = train[:-embargo]
        elif embargo > 0:
            train = train[:0]
        if train.size > 0 and test.size > 0:
            folds.append((train, test))
        train_end = test[-1]
    return folds
