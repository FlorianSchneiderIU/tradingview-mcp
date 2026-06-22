#!/usr/bin/env python3
"""Rigorous re-validation of the session_orb / Judas-swing strategy.

The deployed pipeline (train_session_orb_models.py + sweep_session_orb_top50.py) is
overfit: a single train/OOS split with NO held-out year, the decision threshold is a
grid point inspected on the OOS period, and symbols are KEPT by their OOS profit factor
(selection-on-the-test-set) with no multiple-testing null. Live is PF 0.89.

This script reuses the SAME strategy logic (add_htf_context / build_grid /
generate_trades_for_config / select_scored_trades — all causally clean, audited for
lookahead) but replaces the broken selection with the methodology now used in
bot/train_dt.py and the Wolfe revalidation:

  * walk-forward folds: expanding train >= TRAIN_MIN_DAYS, then VAL_DAYS (threshold is
    chosen HERE, never on OOS), then OOS_DAYS, step STEP_DAYS;
  * a final HOLDOUT_DAYS year that nothing in selection ever touches, scored ONCE;
  * a random-label null (shuffle TRAIN labels only, keep val/oos outcomes real) — the
    real pooled-OOS net-R must beat the NULL_PCTILE of the null to count as edge;
  * cost = fees + slippage (fee_bps_per_side default 8.0 ~= 6.5 taker + 1.5 slippage).

Offline: reads scripts/data/<sym>_5m_bybit.csv (no pybit). Per-symbol verdict + a
summary are written next to --out.

Usage:
  python scripts/revalidate_session_orb.py --symbols ethusdt,adausdt --n-null 100
  python scripts/revalidate_session_orb.py --all
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.experiment_session_orb import (  # noqa: E402
    OrbConfig,
    add_htf_context,
    feature_columns,
    select_scored_trades,
)
from scripts.experiment_session_orb_fast import (  # noqa: E402  (vectorized generation)
    apply_candidate_filter,
    build_contexts,
    generate_trades,
    to_arrays,
)
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402

DATA_DIR = Path("scripts/data")

# Walk-forward windows (mirror bot/train_dt.py).
TRAIN_MIN_DAYS = 540
VAL_DAYS = 180
OOS_DAYS = 180
STEP_DAYS = 180
HOLDOUT_DAYS = 365

FEE_BPS_SIDE = 8.0                       # ~6.5 taker + 1.5 slippage per side
THRESHOLDS = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
SELECTION_MODE = "nonoverlap"

# Acceptance gates.
PF_MIN = 1.15
AVG_R_FLOOR = 0.05
MIN_FOLD_POS_FRAC = 0.60
MIN_OOS_TRADES = 20
MIN_VAL_TRADES = 8
HOLDOUT_PF_MIN = 1.10
NULL_PCTILE = 95.0

DEPLOYED = ["adausdt", "avaxusdt", "dotusdt", "enausdt", "ethusdt", "filusdt",
            "linkusdt", "nearusdt", "ondousdt", "opusdt", "wifusdt"]


def _model(n_estimators: int = 400) -> Any:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(n_estimators=n_estimators, max_depth=5, min_samples_leaf=50,
                               random_state=42, n_jobs=-1, class_weight="balanced_subsample"),
    )


def _judas_fvg_configs() -> list[OrbConfig]:
    """The live-relevant subset: family=judas, entry_mode=fvg_retest (fast grid = 27)."""
    cfgs: list[OrbConfig] = []
    for session in ("asia", "london", "ny"):
        for or_min in (30, 60, 90):
            for rr in (1.0, 1.5, 2.0):
                cfgs.append(OrbConfig(
                    family="judas", session=session, or_minutes=or_min, rr=rr,
                    max_hold_bars=48, stop_mode="sweep", min_sweep_atr=0.15,
                    entry_mode="fvg_retest", retest_tolerance_atr=0.15, retest_wait_bars=18,
                ))
    return cfgs


def load_candidates(symbol: str) -> pd.DataFrame | None:
    path = DATA_DIR / f"{symbol.lower()}_5m_bybit.csv"
    if not path.exists():
        print(f"  {symbol}: no CSV at {path}", flush=True)
        return None
    raw = pd.read_csv(path)
    raw["open_time"] = pd.to_datetime(raw["open_time"], utc=True)
    raw["close_time"] = pd.to_datetime(raw["close_time"], utc=True)
    for c in ("open", "high", "low", "close", "volume"):
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = raw.dropna(subset=["open", "high", "low", "close"]).sort_values("open_time").reset_index(drop=True)
    df = add_htf_context(raw).reset_index(drop=True)
    arrays = to_arrays(df)
    contexts = build_contexts(df, arrays, sessions=["asia", "london", "ny"], or_minutes=[30, 60, 90])
    frames = []
    for cfg in _judas_fvg_configs():
        t = generate_trades(arrays, symbol=symbol, cfg=cfg, contexts=contexts, fee_bps_per_side=FEE_BPS_SIDE)
        if not t.empty:
            frames.append(t)
    if not frames:
        return None
    cand = pd.concat(frames, ignore_index=True)
    cand = apply_candidate_filter(cand, "judas_fvg_risk2")   # deployed strategy: entry_risk_atr>=2.0
    if cand.empty:
        return None
    cand["entry_time"] = pd.to_datetime(cand["entry_time"], utc=True, errors="coerce")
    cand = cand.dropna(subset=["entry_time", "win_label", "r_multiple"]).sort_values("entry_time").reset_index(drop=True)
    return cand


def _pooled(rs: np.ndarray) -> dict[str, float]:
    rs = np.asarray(rs, dtype=float)
    n = len(rs)
    if n == 0:
        return {"trades": 0, "net_r": 0.0, "avg_r": 0.0, "pf": 0.0, "win_rate": 0.0}
    gp = float(rs[rs > 0].sum())
    gl = float(-rs[rs < 0].sum())
    return {"trades": int(n), "net_r": float(rs.sum()), "avg_r": float(rs.mean()),
            "pf": (gp / gl) if gl > 0 else (math.inf if gp > 0 else 0.0),
            "win_rate": float((rs > 0).mean())}


def _fit_score(train: pd.DataFrame, scoreset: pd.DataFrame, cols: list[str], n_estimators: int = 400) -> pd.DataFrame:
    m = _model(n_estimators)
    m.fit(train[cols].astype(float), train["win_label"].astype(int))
    out = scoreset.copy()
    out["ml_prob"] = m.predict_proba(out[cols].astype(float))[:, 1]
    return out


def _pick_threshold(val_scored: pd.DataFrame, split: pd.Timestamp) -> float | None:
    """Choose the threshold maximizing VAL net-R subject to a min trade count."""
    best, best_net = None, -math.inf
    fallback, fallback_trades = None, -1
    for th in THRESHOLDS:
        sel = select_scored_trades(val_scored, threshold=th, split=split, selection_mode=SELECTION_MODE)
        rs = sel["r_multiple"].to_numpy(dtype=float)
        if len(rs) > fallback_trades:
            fallback, fallback_trades = th, len(rs)
        if len(rs) >= MIN_VAL_TRADES and rs.sum() > best_net:
            best, best_net = th, float(rs.sum())
    return best if best is not None else fallback


def _eval_fold(cand: pd.DataFrame, tr0, tr1, va1, oo1, cols, *, shuffle_seed: int | None = None,
               n_estimators: int = 400) -> dict | None:
    train = cand[(cand["entry_time"] >= tr0) & (cand["entry_time"] < tr1)].copy()
    val = cand[(cand["entry_time"] >= tr1) & (cand["entry_time"] < va1)].copy()
    oos = cand[(cand["entry_time"] >= va1) & (cand["entry_time"] < oo1)].copy()
    if len(train) < 200 or train["win_label"].nunique() < 2 or val.empty or oos.empty:
        return None
    if shuffle_seed is not None:                      # null: break only the train mapping
        train = train.copy()
        train["win_label"] = np.random.default_rng(shuffle_seed).permutation(train["win_label"].to_numpy())
    m = _model(n_estimators)                           # fit ONCE; score val + oos
    m.fit(train[cols].astype(float), train["win_label"].astype(int))
    val_scored = val.copy(); val_scored["ml_prob"] = m.predict_proba(val[cols].astype(float))[:, 1]
    th = _pick_threshold(val_scored, tr1)
    if th is None:
        return None
    oos_scored = oos.copy(); oos_scored["ml_prob"] = m.predict_proba(oos[cols].astype(float))[:, 1]
    sel = select_scored_trades(oos_scored, threshold=th, split=va1, selection_mode=SELECTION_MODE)
    rs = sel["r_multiple"].to_numpy(dtype=float)
    return {"threshold": th, "rs": rs, **_pooled(rs)}


def _folds(t0: pd.Timestamp, holdout_start: pd.Timestamp) -> list[tuple]:
    out = []
    train_end = t0 + pd.Timedelta(days=TRAIN_MIN_DAYS)
    while True:
        val_end = train_end + pd.Timedelta(days=VAL_DAYS)
        oos_end = val_end + pd.Timedelta(days=OOS_DAYS)
        if oos_end > holdout_start:
            break
        out.append((t0, train_end, val_end, oos_end))
        train_end = train_end + pd.Timedelta(days=STEP_DAYS)
    return out


def revalidate(symbol: str, n_null: int, direction: str = "") -> dict:
    cand = load_candidates(symbol)
    if cand is None or cand.empty:
        return {"symbol": symbol, "status": "no_data"}
    if direction in ("long", "short"):
        cand = cand[cand["direction"].astype(str).str.lower() == direction].reset_index(drop=True)
        if cand.empty:
            return {"symbol": symbol, "status": "no_data_after_direction_filter"}
    cols = [c for c in feature_columns(cand) if cand[c].notna().any()]
    t0 = cand["entry_time"].iloc[0]
    t_end = cand["entry_time"].iloc[-1]
    holdout_start = t_end - pd.Timedelta(days=HOLDOUT_DAYS)
    span_days = (t_end - t0).days
    folds = _folds(t0, holdout_start)
    if not folds:
        return {"symbol": symbol, "status": "too_short", "span_days": span_days,
                "candidates": int(len(cand))}

    # --- walk-forward (real) ---
    fold_results = [_eval_fold(cand, *f, cols) for f in folds]
    fold_results = [r for r in fold_results if r is not None]
    if not fold_results:
        return {"symbol": symbol, "status": "no_valid_folds", "span_days": span_days,
                "candidates": int(len(cand))}
    all_rs = np.concatenate([r["rs"] for r in fold_results]) if fold_results else np.array([])
    pooled = _pooled(all_rs)
    fold_pos = float(np.mean([1.0 if r["net_r"] > 0 else 0.0 for r in fold_results]))
    real_net = pooled["net_r"]

    # --- null: shuffle train labels only ---
    null_nets = []
    for i in range(n_null):
        nr = []
        for f in folds:
            r = _eval_fold(cand, *f, cols, shuffle_seed=1000 + i, n_estimators=200)
            if r is not None:
                nr.append(r["net_r"])
        null_nets.append(float(np.sum(nr)) if nr else 0.0)
    null_p = float(np.percentile(null_nets, NULL_PCTILE)) if null_nets else math.inf
    beats_null = real_net > null_p

    # --- holdout (scored once) ---
    dev = cand[cand["entry_time"] < holdout_start]
    hold = cand[cand["entry_time"] >= holdout_start]
    holdout = {"trades": 0, "net_r": 0.0, "avg_r": 0.0, "pf": 0.0, "win_rate": 0.0, "threshold": None}
    if len(dev) >= 200 and not hold.empty and dev["win_label"].nunique() >= 2:
        val_cut = holdout_start - pd.Timedelta(days=VAL_DAYS)
        tr = dev[dev["entry_time"] < val_cut]
        va = dev[dev["entry_time"] >= val_cut]
        if len(tr) >= 200 and not va.empty and tr["win_label"].nunique() >= 2:
            th = _pick_threshold(_fit_score(tr, va, cols), val_cut)
            if th is not None:
                hs = select_scored_trades(_fit_score(dev, hold, cols), threshold=th,
                                          split=holdout_start, selection_mode=SELECTION_MODE)
                holdout = {**_pooled(hs["r_multiple"].to_numpy(dtype=float)), "threshold": th}

    accept = bool(
        pooled["trades"] >= MIN_OOS_TRADES and pooled["pf"] >= PF_MIN and pooled["avg_r"] >= AVG_R_FLOOR
        and fold_pos >= MIN_FOLD_POS_FRAC and beats_null
        and holdout["trades"] >= 10 and holdout["pf"] >= HOLDOUT_PF_MIN
    )
    return {
        "symbol": symbol, "status": "ok", "accept": accept, "span_days": span_days,
        "candidates": int(len(cand)), "n_folds": len(fold_results),
        "oos": {k: round(v, 3) if isinstance(v, float) else v for k, v in pooled.items()},
        "fold_pos_frac": round(fold_pos, 2),
        "null": {"real_net_r": round(real_net, 2), "p95_net_r": round(null_p, 2),
                 "beats_null": beats_null, "n_null": len(null_nets)},
        "holdout": {k: round(v, 3) if isinstance(v, float) else v for k, v in holdout.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--n-null", type=int, default=100)
    ap.add_argument("--direction", default="", choices=["", "long", "short"])
    ap.add_argument("--out", type=Path, default=Path("scripts/session_orb_revalidation.json"))
    args = ap.parse_args()
    syms = DEPLOYED if args.all else [s.strip().lower() for s in args.symbols.split(",") if s.strip()]
    if not syms:
        ap.error("pass --symbols a,b or --all")

    results = []
    for s in syms:
        print(f"=== {s} (n_null={args.n_null}) ===", flush=True)
        try:
            r = revalidate(s, args.n_null, direction=args.direction)
        except Exception as exc:  # noqa: BLE001
            r = {"symbol": s, "status": "error", "error": str(exc)}
        results.append(r)
        print("  " + json.dumps({k: r.get(k) for k in ("status", "accept", "oos", "fold_pos_frac", "null", "holdout")},
                                default=str), flush=True)

    args.out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {args.out}", flush=True)
    acc = [r["symbol"] for r in results if r.get("accept")]
    print(f"ACCEPTED ({len(acc)}/{len(results)}): {', '.join(acc) if acc else 'NONE'}", flush=True)


if __name__ == "__main__":
    main()
