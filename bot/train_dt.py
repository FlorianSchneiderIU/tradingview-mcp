#!/usr/bin/env python3
"""
train_dt.py — Train per-coin DecisionTree filters with rigorous validation.
==========================================================================
Walk-forward + held-out-final-year training procedure
-----------------------------------------------------

  1. Fetch all 15m bars from Bybit back to SINCE_DATE (>= 4 years where listed)
  2. Compute indicators + detect signals + extract 13 features (once, globally)
  3. Carve off the most recent HOLDOUT_DAYS as a FINAL HOLDOUT that is never
     touched during any tuning / selection / threshold search.
  4. On the development span (everything before the holdout) run a rolling
     walk-forward: expanding train (>= TRAIN_MIN_DAYS) + VAL_DAYS validation +
     OOS_DAYS out-of-sample, stepping STEP_DAYS forward (~4-5 folds).
  5. Jointly select (sl, tp1, trail, threshold) over a small grid, scoring on
     net-of-cost R-multiples (fees + slippage). A config is accepted only if it
     clears the trade-count, profit-factor, net-expectancy and fold-consistency
     gates below.
  6. Multiple-testing guard: the DT filter must beat a random-selection null on
     the pooled OOS at the 95th percentile (so PF/avg-R that are just the best of
     a large grid of noise are rejected).
  7. Stability guard: the selected config must not be a fragile peak — its pooled
     OOS net avg-R must be >= the median of its grid neighbours.
  8. Final model = a single DT fit on ALL development data (NOT the holdout) with
     the selected params. The held-out final year is then scored exactly once.
  9. Save model + threshold + per-fold metrics + holdout report.

All simulations are NET OF COST. Costs come from indicators.DEFAULT_FEE_BPS_SIDE
and DEFAULT_SLIPPAGE_BPS_SIDE (match live taker fee + slippage).

Only runs for coins with "use_dt": true in the configs file.

Usage (local, from repo root, with .venv active):
    python bot/train_dt.py
    python bot/train_dt.py --symbols BTCUSDT,ETHUSDT --fee-bps 5.5 --slip-bps 1.0

Reads:  bot/configs/top20_configs.json
Writes: bot/models/<SYMBOL>_dt.pkl
        bot/models/<SYMBOL>_holdout_report.json
        bot/models/train_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# pybit is only needed for live Bybit fetching; offline runs (--data-dir) don't
# require it. Imported lazily in main() so the module loads without the dep.

# indicators.py must be on the Python path (same directory as this script)
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from indicators import (
    ST_MULT, ST_ATR_LEN, EMA_LEN, SMA_LEN, ATR_LEN,
    VOL_WIN, ATR_PCTILE_WIN, ATR_LO, ATR_HI, VOL_THR,
    N_FEATURES,
    DEFAULT_FEE_BPS_SIDE, DEFAULT_SLIPPAGE_BPS_SIDE, DEFAULT_LOCKIN_R,
    ind_atr, ind_ema, ind_sma,
    build_signals,
    rolling_atr_pctile, rolling_vol_ratio,
    compute_rsi, compute_rsi_htf,
    extract_features_batch,
    sim_trail, metrics,
)

# ── Config ────────────────────────────────────────────────────────────────────
SINCE_DATE   = "2022-01-01"   # fetch >= 4y where the instrument allows
TIMEFRAME    = "15"

DT_DEPTH     = 2
DT_MIN_LEAF  = 15

# Walk-forward window sizes (calendar days)
HOLDOUT_DAYS    = 365     # final-year holdout, never touched until the end
TRAIN_MIN_DAYS  = 540     # >= 18 months initial (expanding) train window
VAL_DAYS        = 180     # 6-month validation window (threshold selection)
OOS_DAYS        = 180     # 6-month rolling out-of-sample window
STEP_DAYS       = 180     # step the fold origin forward by 6 months

# Parameter grid (jointly re-tuned). tp1/trail floors enforce a positive-
# expectancy exit profile (see Phase 3 of the remediation plan).
GRID_SL    = [1.5, 2.0, 2.5, 3.0, 3.5]
GRID_TP1   = [1.5, 2.0, 2.5]
GRID_TRAIL = [1.5, 2.0, 3.0]
GRID_THR   = [round(x, 2) for x in np.arange(0.40, 0.91, 0.05)]

# Lock-in fraction of R applied to the runner after TP1 (mirrors live behaviour).
LOCKIN_R   = DEFAULT_LOCKIN_R

# Acceptance gates
MIN_TRAIN_TRADES   = 30
MIN_VAL_TRADES     = 15
MIN_OOS_TRADES     = 15
MIN_HOLDOUT_TRADES = 30
PF_FLOOR           = 1.15    # train + validation + OOS
PF_FLOOR_HOLDOUT   = 1.10
AVG_R_FLOOR        = 0.05    # net avg R on validation + OOS
MIN_FOLD_POS_FRAC  = 0.60    # >= 60% of OOS folds net-positive

# Multiple-testing / overfitting guard
N_NULL          = 1000      # random-selection null replicates
NULL_PCTILE     = 95.0      # observed must beat this percentile of the null

CONFIGS_PATH = os.environ.get("CONFIGS_PATH", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "configs", "top20_configs.json"))
MODELS_DIR   = os.environ.get("MODELS_DIR", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models"))

_DAY_MS = 86_400_000


# ── Data fetch ────────────────────────────────────────────────────────────────

def fetch_all_bars(symbol: str, since_date: str, http: HTTP) -> list[dict]:
    """Fetch all 15m bars from since_date to now using Bybit REST pagination."""
    import datetime as _dt
    since_dt = _dt.datetime.strptime(since_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    since_ms = int(since_dt.timestamp() * 1000)
    end_ts = int(datetime.now(timezone.utc).timestamp() * 1000)

    all_bars: list[dict] = []

    while True:
        kwargs: dict = dict(
            category="linear",
            symbol=symbol,
            interval=TIMEFRAME,
            limit=1000,
            start=since_ms,
            end=end_ts,
        )

        resp  = http.get_kline(**kwargs)
        items = resp.get("result", {}).get("list", [])
        if not items:
            break

        for it in reversed(items):
            ts = int(it[0])
            if ts < since_ms:
                continue
            all_bars.append({
                "ts":     ts,
                "open":   float(it[1]),
                "high":   float(it[2]),
                "low":    float(it[3]),
                "close":  float(it[4]),
                "volume": float(it[5]),
            })

        oldest_ts = int(items[-1][0])
        if oldest_ts <= since_ms or len(items) < 1000:
            break
        end_ts = oldest_ts - 1
        time.sleep(0.12)

    # Deduplicate and sort
    seen: set[int] = set()
    unique: list[dict] = []
    for bar in sorted(all_bars, key=lambda x: x["ts"]):
        if bar["ts"] not in seen:
            seen.add(bar["ts"]); unique.append(bar)
    return unique


def load_bars_from_csv(symbol: str, data_dir: str) -> list[dict]:
    """Load cached 5m Bybit klines and resample to the 15m training timeframe.

    Files are named e.g. `btcusdt_5m_bybit.csv` with columns
    open_time, close_time, open, high, low, close, volume. Lets the full
    optimization run offline (no pybit / network).
    """
    path = os.path.join(data_dir, f"{symbol.lower()}_5m_bybit.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, usecols=["open_time", "open", "high", "low", "close", "volume"])
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.set_index("open_time").sort_index()
    agg = pd.concat(
        [
            df["open"].resample("15min").first(),
            df["high"].resample("15min").max(),
            df["low"].resample("15min").min(),
            df["close"].resample("15min").last(),
            df["volume"].resample("15min").sum(),
        ],
        axis=1,
    ).dropna()
    bars: list[dict] = []
    for ts, row in agg.iterrows():
        bars.append({
            "ts":     int(ts.timestamp() * 1000),
            "open":   float(row["open"]),
            "high":   float(row["high"]),
            "low":    float(row["low"]),
            "close":  float(row["close"]),
            "volume": float(row["volume"]),
        })
    return bars


def instrument_status(symbol: str, http: "HTTP") -> str:
    resp = http.get_instruments_info(category="linear", symbol=symbol)
    items = resp.get("result", {}).get("list", [])
    if not items:
        return ""
    return str(items[0].get("status", ""))


# ── Series precompute ─────────────────────────────────────────────────────────

class Series:
    """Precomputed indicators / signals / features for a full bar history.

    Computed once per symbol; all walk-forward windows are index slices into it.
    Feature rows are global (do not depend on sl/tp1/trail), so they are cached
    in `feat_map` keyed by global bar index.
    """

    def __init__(self, bars: list[dict]):
        self.n = len(bars)
        self.ts_ms  = np.array([b["ts"]     for b in bars], dtype=np.int64)
        self.o  = np.array([b["open"]   for b in bars], dtype=np.float64)
        self.h  = np.array([b["high"]   for b in bars], dtype=np.float64)
        self.l  = np.array([b["low"]    for b in bars], dtype=np.float64)
        self.c  = np.array([b["close"]  for b in bars], dtype=np.float64)
        self.v  = np.array([b["volume"] for b in bars], dtype=np.float64)

        ts_idx = pd.DatetimeIndex(pd.to_datetime(self.ts_ms, unit="ms", utc=True))

        self.atr14   = ind_atr(self.h, self.l, self.c, ATR_LEN)
        atr_st       = ind_atr(self.h, self.l, self.c, ST_ATR_LEN)
        self.ema200  = ind_ema(self.c, EMA_LEN)
        self.sma13   = ind_sma(self.c, SMA_LEN)
        self.atr_pct = rolling_atr_pctile(self.atr14, ATR_PCTILE_WIN)
        self.vol_rat = rolling_vol_ratio(self.v, VOL_WIN)
        rsi14        = compute_rsi(self.c, 14)
        rsi4h        = compute_rsi_htf(ts_idx, self.c, "4h", 14)

        self.sbull, self.sbear = build_signals(
            self.c, self.o, self.sma13, self.ema200, atr_st)

        # All valid signal indices (pass ATR percentile + volume gates).
        self.sig_idx = [
            i for i in range(self.n)
            if (self.sbull[i] or self.sbear[i])
            and ATR_LO < self.atr_pct[i] < ATR_HI
            and self.vol_rat[i] >= VOL_THR
        ]

        dts = [datetime.fromtimestamp(t / 1000, tz=timezone.utc) for t in self.ts_ms]
        feat_rows = extract_features_batch(
            self.sig_idx, self.c, self.h, self.l, self.o, self.v,
            self.atr14, self.atr_pct, self.vol_rat, self.ema200, self.sma13,
            self.sbull, self.sbear, dts, rsi14, rsi4h,
        )
        self.feat_map: dict[int, np.ndarray] = {}
        for k, gi in enumerate(self.sig_idx):
            row = feat_rows[k]
            if row is not None:
                self.feat_map[gi] = row

    def idx_at_or_after(self, ts: int) -> int:
        """First bar index with ts_ms >= ts (n if none)."""
        return int(np.searchsorted(self.ts_ms, ts, side="left"))


# ── Window simulation helpers (all net of cost) ───────────────────────────────

def sim_window(s: Series, a: int, b: int, sl: float, tp1: float, trail: float,
               mask: Optional[np.ndarray], fee: float, slip: float):
    """Run sim_trail on slice [a, b). Returns (r_arr, trades_global) where each
    trade's entry/exit indices are converted back to global bar indices."""
    r, td = sim_trail(
        s.c[a:b], s.h[a:b], s.l[a:b],
        s.sbull[a:b], s.sbear[a:b],
        s.atr14[a:b], s.atr_pct[a:b], s.vol_rat[a:b],
        sl_mult=sl, tp1_r=tp1, trail_mult=trail,
        signal_mask=mask,
        fee_bps_side=fee, slippage_bps_side=slip, lockin_r=LOCKIN_R,
    )
    for t in td:
        t["entry_gi"] = a + t["entry_i"]
    return r, td


def fit_dt(s: Series, a: int, b: int, sl: float, tp1: float, trail: float,
           fee: float, slip: float) -> Optional[DecisionTreeClassifier]:
    """Fit a DT on the trades produced inside window [a, b)."""
    _, td = sim_window(s, a, b, sl, tp1, trail, None, fee, slip)
    X, y = [], []
    for t in td:
        gi = t["entry_gi"]
        row = s.feat_map.get(gi)
        if row is None:
            continue
        X.append(row); y.append(1 if t["r"] > 0 else 0)
    if len(X) < MIN_TRAIN_TRADES or len(set(y)) < 2:
        return None
    clf = DecisionTreeClassifier(
        max_depth=DT_DEPTH, min_samples_leaf=DT_MIN_LEAF,
        class_weight="balanced", random_state=42,
    )
    clf.fit(np.array(X, dtype=np.float64), np.array(y, dtype=int))
    return clf


def trade_probs(s: Series, clf: DecisionTreeClassifier, a: int, b: int,
                sl: float, tp1: float, trail: float, fee: float, slip: float):
    """Unfiltered trades in [a, b) with the DT probability of each entry.

    Returns list of (prob, r). Trades whose entry has no feature row are dropped.
    """
    _, td = sim_window(s, a, b, sl, tp1, trail, None, fee, slip)
    rows, keep = [], []
    for t in td:
        row = s.feat_map.get(t["entry_gi"])
        if row is None:
            continue
        rows.append(row); keep.append(t["r"])
    if not rows:
        return []
    probs = clf.predict_proba(np.array(rows, dtype=np.float64))[:, 1]
    return list(zip([float(p) for p in probs], [float(r) for r in keep]))


def _net_avg(rs: list[float]) -> float:
    return float(np.mean(rs)) if rs else 0.0


def _pf(rs: list[float]) -> float:
    gw = sum(r for r in rs if r > 0)
    gl = -sum(r for r in rs if r < 0)
    return (gw / gl) if gl > 1e-12 else (0.0 if gw == 0 else 999.0)


# ── Fold construction ─────────────────────────────────────────────────────────

def build_folds(s: Series, dev_end_i: int) -> list[dict]:
    """Rolling expanding-train / val / oos folds within [0, dev_end_i)."""
    if s.n == 0:
        return []
    t0 = int(s.ts_ms[0])
    folds = []
    origin_day = TRAIN_MIN_DAYS
    while True:
        train_end_ts = t0 + origin_day * _DAY_MS
        val_end_ts   = train_end_ts + VAL_DAYS * _DAY_MS
        oos_end_ts   = val_end_ts + OOS_DAYS * _DAY_MS

        train_a, train_b = 0, s.idx_at_or_after(train_end_ts)
        val_a,   val_b   = train_b, s.idx_at_or_after(val_end_ts)
        oos_a,   oos_b   = val_b, min(s.idx_at_or_after(oos_end_ts), dev_end_i)

        if oos_a >= dev_end_i:
            break
        if (val_b - val_a) > 0 and (oos_b - oos_a) > 0 and train_b > train_a:
            folds.append({
                "train": (train_a, train_b),
                "val":   (val_a, val_b),
                "oos":   (oos_a, oos_b),
            })
        if oos_b >= dev_end_i:
            break
        origin_day += STEP_DAYS
    return folds


# ── Config evaluation across folds ────────────────────────────────────────────

def eval_config(s: Series, folds: list[dict], sl: float, tp1: float,
                trail: float, fee: float, slip: float) -> Optional[dict]:
    """Evaluate one (sl, tp1, trail) across all folds. Selects the best common
    threshold on pooled validation, then measures pooled / per-fold OOS.

    Returns a result dict (always — gate decision is made by the caller) or None
    if the config could not be evaluated (too few trades to fit a DT anywhere).
    """
    pooled_train: list[tuple[float, float]] = []
    pooled_val:   list[tuple[float, float]] = []
    pooled_oos:   list[tuple[float, float]] = []
    per_fold_oos: list[list[tuple[float, float]]] = []
    n_fit = 0

    for f in folds:
        clf = fit_dt(s, *f["train"], sl, tp1, trail, fee, slip)
        if clf is None:
            per_fold_oos.append([])
            continue
        n_fit += 1
        # All buckets measured on DT-filtered trades (prob >= thr), consistently:
        # the strategy that actually trades live is the filtered one.
        pooled_train.extend(trade_probs(s, clf, *f["train"], sl, tp1, trail, fee, slip))
        pooled_val.extend(trade_probs(s, clf, *f["val"], sl, tp1, trail, fee, slip))
        oos = trade_probs(s, clf, *f["oos"], sl, tp1, trail, fee, slip)
        pooled_oos.extend(oos)
        per_fold_oos.append(oos)

    if n_fit == 0 or not pooled_val:
        return None

    # Select threshold maximising net avg R on pooled validation.
    best_thr, best_val_avg = None, -1e9
    for thr in GRID_THR:
        sel = [r for p, r in pooled_val if p >= thr]
        if len(sel) < MIN_VAL_TRADES:
            continue
        a = _net_avg(sel)
        if a > best_val_avg:
            best_val_avg, best_thr = a, thr
    if best_thr is None:
        return None

    train_sel = [r for p, r in pooled_train if p >= best_thr]
    val_sel   = [r for p, r in pooled_val if p >= best_thr]
    oos_sel   = [r for p, r in pooled_oos if p >= best_thr]

    fold_pos = 0
    fold_total = 0
    for oos in per_fold_oos:
        sel = [r for p, r in oos if p >= best_thr]
        if len(sel) < 3:
            continue
        fold_total += 1
        if _net_avg(sel) > 0:
            fold_pos += 1
    fold_pos_frac = (fold_pos / fold_total) if fold_total else 0.0

    return {
        "sl": sl, "tp1": tp1, "trail": trail, "threshold": best_thr,
        "train_n": len(train_sel), "val_n": len(val_sel), "oos_n": len(oos_sel),
        "train_pf": round(_pf(train_sel), 3),
        "val_pf":   round(_pf(val_sel), 3),
        "oos_pf":   round(_pf(oos_sel), 3),
        "val_avg_r": round(_net_avg(val_sel), 4),
        "oos_avg_r": round(_net_avg(oos_sel), 4),
        "fold_pos_frac": round(fold_pos_frac, 3),
        "fold_total": fold_total,
        # Kept for the null test (DT-selected vs random subset on pooled OOS).
        "_pooled_oos": pooled_oos,
        "_oos_sel": oos_sel,
    }


def passes_gates(r: dict) -> bool:
    return (
        r["train_n"] >= MIN_TRAIN_TRADES
        and r["val_n"] >= MIN_VAL_TRADES
        and r["oos_n"] >= MIN_OOS_TRADES
        and r["train_pf"] >= PF_FLOOR
        and r["val_pf"]   >= PF_FLOOR
        and r["oos_pf"]   >= PF_FLOOR
        and r["val_avg_r"] >= AVG_R_FLOOR
        and r["oos_avg_r"] >= AVG_R_FLOOR
        and r["fold_pos_frac"] >= MIN_FOLD_POS_FRAC
    )


def null_pass(r: dict, rng: np.random.Generator) -> tuple[bool, float]:
    """Random-selection null: can a random subset of the same size match the
    DT-selected pooled-OOS net avg R? Returns (passed, null_pctile_value)."""
    pool = [rr for _, rr in r["_pooled_oos"]]
    k = len(r["_oos_sel"])
    if k == 0 or k >= len(pool):
        return False, 0.0
    observed = _net_avg(r["_oos_sel"])
    pool_arr = np.asarray(pool, dtype=np.float64)
    draws = np.empty(N_NULL, dtype=np.float64)
    for j in range(N_NULL):
        draws[j] = float(np.mean(rng.choice(pool_arr, size=k, replace=False)))
    thresh = float(np.percentile(draws, NULL_PCTILE))
    return observed > thresh, round(thresh, 4)


def stability_ok(selected: dict, all_results: dict) -> bool:
    """Selected config must not be a fragile peak: its pooled OOS net avg R must
    be >= the median of its immediate grid neighbours (one step in sl/tp1/trail)."""
    def neigh(vals, v):
        i = vals.index(v)
        out = []
        if i > 0: out.append(vals[i - 1])
        if i < len(vals) - 1: out.append(vals[i + 1])
        return out

    keys = []
    for nsl in neigh(GRID_SL, selected["sl"]):
        keys.append((nsl, selected["tp1"], selected["trail"]))
    for ntp in neigh(GRID_TP1, selected["tp1"]):
        keys.append((selected["sl"], ntp, selected["trail"]))
    for ntr in neigh(GRID_TRAIL, selected["trail"]):
        keys.append((selected["sl"], selected["tp1"], ntr))

    vals = [all_results[k]["oos_avg_r"] for k in keys if k in all_results]
    if not vals:
        return True  # edge of grid; nothing to compare against
    return selected["oos_avg_r"] >= float(np.median(vals))


# ── Holdout scoring (faithful, masked re-simulation) ──────────────────────────

def score_holdout(s: Series, a: int, b: int, clf: DecisionTreeClassifier,
                  thr: float, sl: float, tp1: float, trail: float,
                  fee: float, slip: float) -> dict:
    """Score the final-year holdout once with the frozen model, using a faithful
    masked re-simulation (the DT mask gates which signals may open positions)."""
    width = b - a
    mask = np.zeros(width, dtype=bool)
    for gi in s.sig_idx:
        if a <= gi < b:
            row = s.feat_map.get(gi)
            if row is None:
                continue
            p = float(clf.predict_proba(row.reshape(1, -1))[:, 1][0])
            if p >= thr:
                mask[gi - a] = True
    r, _ = sim_window(s, a, b, sl, tp1, trail, mask, fee, slip)
    m = metrics(r, min_n=1)
    return {
        "n": int(m["n"]),
        "pf": float(m["pf"]),
        "avg_r": round(float(np.mean(r)) if len(r) else 0.0, 4),
        "total_r": float(m["total_r"]),
        "win_rate": float(m["win_rate"]),
    }


# ── Training ──────────────────────────────────────────────────────────────────

def train_symbol(symbol: str, bars: list[dict], fee: float, slip: float) -> Optional[dict]:
    n = len(bars)
    if n < 2000:
        print(f"  {symbol}: only {n} bars — skipping")
        return None

    s = Series(bars)
    if len(s.sig_idx) < (MIN_TRAIN_TRADES + MIN_VAL_TRADES + MIN_OOS_TRADES) * 2:
        print(f"  {symbol}: only {len(s.sig_idx)} signals — skipping")
        return None

    # Carve off the final-year holdout.
    holdout_start_ts = int(s.ts_ms[-1]) - HOLDOUT_DAYS * _DAY_MS
    dev_end_i = s.idx_at_or_after(holdout_start_ts)
    holdout_a, holdout_b = dev_end_i, s.n
    if dev_end_i < s.idx_at_or_after(int(s.ts_ms[0]) + (TRAIN_MIN_DAYS + VAL_DAYS + OOS_DAYS) * _DAY_MS):
        print(f"  {symbol}: insufficient development span before holdout — skipping")
        return None

    folds = build_folds(s, dev_end_i)
    if len(folds) < 2:
        print(f"  {symbol}: only {len(folds)} folds — skipping")
        return None

    # Sweep the grid.
    all_results: dict[tuple, dict] = {}
    n_tried = 0
    for sl in GRID_SL:
        for tp1 in GRID_TP1:
            for trail in GRID_TRAIL:
                n_tried += 1
                res = eval_config(s, folds, sl, tp1, trail, fee, slip)
                if res is not None:
                    all_results[(sl, tp1, trail)] = res

    def _clean(r):
        return {k: v for k, v in r.items() if not k.startswith("_")}

    passing = [r for r in all_results.values() if passes_gates(r)]
    if not passing:
        best = max(all_results.values(), key=lambda r: r["oos_avg_r"], default=None)
        if best is not None:
            print(
                f"  {symbol}: NO PASS (tried {n_tried})  best-by-oos_avg_r: "
                f"sl={best['sl']} tp1={best['tp1']} trail={best['trail']} thr={best['threshold']}  "
                f"train_pf={best['train_pf']} val_pf={best['val_pf']} oos_pf={best['oos_pf']}  "
                f"val_avg_r={best['val_avg_r']} oos_avg_r={best['oos_avg_r']}  "
                f"folds={best['fold_total']}({best['fold_pos_frac']:.0%}+)  "
                f"n(tr/val/oos)={best['train_n']}/{best['val_n']}/{best['oos_n']}"
            )
            return {"_report": {"symbol": symbol, "status": "no_config_passed_gates",
                                "best": _clean(best), "n_configs_tried": n_tried,
                                "n_folds": len(folds)}, "_rejected": True}
        print(f"  {symbol}: no config evaluable (tried {n_tried}) — skipping")
        return None

    # Best passing config by pooled OOS net avg R.
    passing.sort(key=lambda r: r["oos_avg_r"], reverse=True)
    rng = np.random.default_rng(42)

    selected = None
    for cand in passing:
        ok_null, null_thresh = null_pass(cand, rng)
        if not ok_null:
            continue
        if not stability_ok(cand, all_results):
            continue
        cand["null_pctile_value"] = null_thresh
        selected = cand
        break

    if selected is None:
        print(f"  {symbol}: best configs failed the null / stability guard — skipping")
        return None

    sl, tp1, trail, thr = (selected["sl"], selected["tp1"],
                           selected["trail"], selected["threshold"])

    # Final model: fit on ALL development data (NOT the holdout).
    clf_final = fit_dt(s, 0, dev_end_i, sl, tp1, trail, fee, slip)
    if clf_final is None:
        print(f"  {symbol}: final DT fit failed on dev span — skipping")
        return None

    # Score the held-out final year exactly once.
    hold = score_holdout(s, holdout_a, holdout_b, clf_final, thr,
                         sl, tp1, trail, fee, slip)
    holdout_passed = (hold["n"] >= MIN_HOLDOUT_TRADES
                      and hold["pf"] >= PF_FLOOR_HOLDOUT
                      and hold["avg_r"] > 0.0)

    status = "ACCEPTED" if holdout_passed else "REJECTED_HOLDOUT"
    print(
        f"  {symbol}: {status}  sl={sl} tp1={tp1} trail={trail} thr={thr:.2f}  "
        f"OOS pf={selected['oos_pf']} avg_r={selected['oos_avg_r']} "
        f"folds={selected['fold_total']}({selected['fold_pos_frac']:.0%}+)  "
        f"HOLDOUT n={hold['n']} pf={hold['pf']} avg_r={hold['avg_r']}  "
        f"tried={n_tried}"
    )

    report = {
        "symbol": symbol,
        "selected": {k: v for k, v in selected.items() if not k.startswith("_")},
        "holdout": hold,
        "holdout_passed": holdout_passed,
        "n_configs_tried": n_tried,
        "n_configs_passing_gates": len(passing),
        "n_folds": len(folds),
        "fee_bps_side": fee,
        "slippage_bps_side": slip,
        "bars": n,
        "dev_bars": dev_end_i,
        "trained_at": datetime.now(tz=timezone.utc).isoformat(),
    }

    if not holdout_passed:
        return {"_report": report, "_rejected": True}

    return {
        "model":      clf_final,
        "threshold":  thr,
        "sl":         sl,
        "tp1":        tp1,
        "trail":      trail,
        "oos_pf":     selected["oos_pf"],
        "oos_avg_r":  selected["oos_avg_r"],
        "holdout_pf": hold["pf"],
        "holdout_avg_r": hold["avg_r"],
        "n_configs_tried": n_tried,
        "fee_bps_side": fee,
        "slippage_bps_side": slip,
        "trained_on": dev_end_i,
        "trained_at": report["trained_at"],
        "symbol":     symbol,
        "_report":    report,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MM DT walk-forward training")
    parser.add_argument("--since",   default=SINCE_DATE,
                        help="Fetch bars from this date (YYYY-MM-DD)")
    parser.add_argument("--symbols", default=None,
                        help="Comma-separated symbol list (default: all use_dt=true in config)")
    parser.add_argument("--fee-bps", type=float, default=DEFAULT_FEE_BPS_SIDE,
                        help="Taker fee per side in basis points")
    parser.add_argument("--slip-bps", type=float, default=DEFAULT_SLIPPAGE_BPS_SIDE,
                        help="Slippage per side in basis points")
    parser.add_argument("--demo",    action="store_true",
                        help="Use Bybit demo endpoint for fetching data")
    parser.add_argument("--data-dir", default=None,
                        help="Run OFFLINE from cached <symbol>_5m_bybit.csv files in this "
                             "directory (resampled to 15m); no pybit/network needed.")
    args = parser.parse_args()

    offline = bool(args.data_dir)

    with open(CONFIGS_PATH) as fh:
        all_configs: dict[str, dict] = json.load(fh)
    all_configs = {k: v for k, v in all_configs.items() if not k.startswith("_")}

    if args.symbols:
        target = {x.strip().upper() for x in args.symbols.split(",")}
        symbols = sorted(target)
    elif offline:
        # All symbols with a cached CSV in the data dir.
        import glob
        symbols = sorted(
            os.path.basename(p)[: -len("_5m_bybit.csv")].upper()
            for p in glob.glob(os.path.join(args.data_dir, "*_5m_bybit.csv"))
        )
    else:
        symbols = [k for k, v in all_configs.items() if v.get("use_dt", False)]

    if not symbols:
        print("No symbols selected. Nothing to train.")
        return

    http = None
    if not offline:
        from pybit.unified_trading import HTTP
        demo = args.demo or os.environ.get("BYBIT_DEMO", "true").lower() in ("1", "true", "yes")
        http = HTTP(
            testnet=False,
            demo=demo,
            api_key=os.environ.get("BYBIT_API_KEY", ""),
            api_secret=os.environ.get("BYBIT_API_SECRET", ""),
        )

    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"Walk-forward DT training for {len(symbols)} coins  (since={args.since})")
    print(f"holdout={HOLDOUT_DAYS}d  train>={TRAIN_MIN_DAYS}d  val={VAL_DAYS}d  "
          f"oos={OOS_DAYS}d  step={STEP_DAYS}d  cost={args.fee_bps}+{args.slip_bps}bps/side\n")

    saved = 0; failed = 0; t0_total = time.time()
    summary_rows: list[dict] = []

    for sym in symbols:
        t0 = time.time()
        if not offline:
            try:
                status = instrument_status(sym, http)
                if status and status != "Trading":
                    print(f"{sym}: instrument status={status} -- skipping")
                    summary_rows.append({"symbol": sym, "status": "instrument_not_trading", "instrument_status": status})
                    failed += 1
                    continue
            except Exception as exc:
                print(f"{sym}: instrument status check failed: {exc}")
                summary_rows.append({"symbol": sym, "status": "instrument_status_error", "detail": str(exc)})
                failed += 1
                continue

        print(f"{sym}: loading bars ...", end=" ", flush=True)
        try:
            bars = (load_bars_from_csv(sym, args.data_dir) if offline
                    else fetch_all_bars(sym, args.since, http))
            if bars:
                first = datetime.fromtimestamp(bars[0]["ts"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
                last = datetime.fromtimestamp(bars[-1]["ts"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
                print(f"{len(bars)} bars  {first}..{last}  [{time.time()-t0:.0f}s]")
            else:
                print(f"0 bars  [{time.time()-t0:.0f}s]")
        except Exception as exc:
            print(f"FETCH ERROR: {exc}")
            summary_rows.append({"symbol": sym, "status": "fetch_error", "detail": str(exc)})
            failed += 1
            continue

        try:
            result = train_symbol(sym, bars, args.fee_bps, args.slip_bps)
        except Exception as exc:  # noqa: BLE001 - keep the batch going
            print(f"  {sym}: training error: {exc}")
            summary_rows.append({"symbol": sym, "status": "training_error", "detail": str(exc)})
            failed += 1
            continue

        # Always write a holdout report when we got far enough to score one.
        if result is not None and "_report" in result:
            rep_path = os.path.join(MODELS_DIR, f"{sym}_holdout_report.json")
            with open(rep_path, "w", encoding="utf-8") as fh:
                json.dump(result["_report"], fh, indent=2, sort_keys=True)

        if result is None or result.get("_rejected"):
            summary_rows.append({"symbol": sym, "status": "skipped_or_rejected", "bars": len(bars)})
            failed += 1
            continue

        out_path = os.path.join(MODELS_DIR, f"{sym}_dt.pkl")
        save_obj = {k: v for k, v in result.items() if not k.startswith("_")}
        with open(out_path, "wb") as fh:
            pickle.dump(save_obj, fh)
        print(f"  Saved -> {out_path}")
        summary_rows.append({"symbol": sym, "status": "saved", "bars": len(bars), "path": out_path})
        saved += 1

    total = time.time() - t0_total
    summary_path = os.path.join(MODELS_DIR, "train_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "trained_at": datetime.now(tz=timezone.utc).isoformat(),
                "models_dir": MODELS_DIR,
                "saved": saved,
                "failed_or_skipped": failed,
                "fee_bps_side": args.fee_bps,
                "slippage_bps_side": args.slip_bps,
                "rows": summary_rows,
            },
            fh,
            indent=2,
            sort_keys=True,
        )
    print(f"\n{'='*70}")
    print(f"Done  |  {saved} models saved  |  {failed} skipped/failed  |  {total:.0f}s total")
    print(f"Models directory: {MODELS_DIR}")
    print(f"Training summary: {summary_path}")


if __name__ == "__main__":
    main()
