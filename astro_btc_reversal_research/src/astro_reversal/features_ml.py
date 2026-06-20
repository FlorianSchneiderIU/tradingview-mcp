"""Feature assembly for the M3 pivot-window models.

Combines:
  * price features (new here)              - past/current only, prefix ``px_``
  * calendar + cycle + astro features      - reused ``build_feature_matrix``

and exposes the ablation feature-sets from proposal section 17 (a useful subset).

Leakage note: every price feature is a trailing/current-bar transform (rolling
windows look backward only). Astro/calendar/cycle features are deterministic
functions of time. None reference future bars. Forward-looking targets live in
:mod:`astro_reversal.labels_ml`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import reuse


def _clean(frame: pd.DataFrame) -> pd.DataFrame:
    # inf -> nan -> 0.0 (no global-median fill, to avoid full-sample leakage).
    return frame.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)


def price_features(frame: pd.DataFrame) -> pd.DataFrame:
    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    open_ = frame["open"].astype(float)
    volume = frame["volume"].astype(float)
    atr = frame["atr"].astype(float)

    out = pd.DataFrame(index=frame.index)
    logret1 = np.log(close).diff()
    for k in (1, 3, 6, 12):
        out[f"px_ret_{k}"] = np.log(close).diff(k)
    for w in (6, 24):
        out[f"px_vol_{w}"] = logret1.rolling(w, min_periods=2).std()
    out["px_atr_norm"] = atr / close
    out["px_atr_pctile"] = atr.rolling(100, min_periods=20).rank(pct=True)
    for span in (20, 100):
        ema = close.ewm(span=span, adjust=False).mean()
        out[f"px_ema{span}_dist"] = (close - ema) / close
        out[f"px_ema{span}_slope"] = ema.diff(10) / ema
    rng = (high - low).replace(0.0, np.nan)
    out["px_range_atr"] = (high - low) / atr
    out["px_range_compress"] = (high - low) / (high - low).rolling(20, min_periods=5).mean()
    out["px_body_ratio"] = (close - open_).abs() / rng
    out["px_upper_wick"] = (high - np.maximum(close, open_)) / rng
    out["px_lower_wick"] = (np.minimum(close, open_) - low) / rng
    out["px_vol_ratio"] = volume / volume.rolling(20, min_periods=5).mean()
    # RSI(14), Wilder.
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0.0, np.nan)
    out["px_rsi14"] = (100 - 100 / (1 + rs)) / 100.0
    return _clean(out)


def build_features(
    frame: pd.DataFrame,
    timeframe: str,
    cache_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Return (feature_matrix, feature_sets) aligned to ``frame`` rows."""
    cache_dir = cache_dir or reuse.DEFAULT_CACHE_DIR
    times = pd.to_datetime(frame["open_time"], utc=True).reset_index(drop=True)
    astro_cal, groups = reuse.build_feature_matrix(times, cache_dir, timeframe)
    astro_cal = astro_cal.reset_index(drop=True)
    px = price_features(frame).reset_index(drop=True)

    features = pd.concat([astro_cal, px], axis=1)
    features = _clean(features)

    calendar = list(groups.get("calendar", []))
    cycles = list(groups.get("cycles", []))
    astro = list(groups.get("astro", []))
    astro_cycle = list(groups.get("astro_cycle", []))
    price = list(px.columns)
    lunar = [c for c in astro if "moon_phase" in c] + [c for c in cycles if "moon" in c]

    feature_sets = {
        "price_only": price,
        "calendar_only": calendar,
        "lunar_only": lunar,
        "astro_only": astro if astro else cycles,
        "astro_cycle": astro_cycle if astro_cycle else cycles,
        "calendar_plus_astro": calendar + astro_cycle,
        "astro_plus_price": astro_cycle + price,
        "full": astro_cycle + calendar + price,
    }
    # Drop empty sets (e.g. astro unavailable) and de-duplicate columns.
    feature_sets = {
        name: list(dict.fromkeys([c for c in cols if c in features.columns]))
        for name, cols in feature_sets.items()
    }
    feature_sets = {name: cols for name, cols in feature_sets.items() if cols}
    return features, feature_sets
