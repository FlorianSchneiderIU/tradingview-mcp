#!/usr/bin/env python3
"""
train_dt.py — Train per-coin DecisionTree filters and save to MODELS_DIR.
==========================================================================
Training procedure (single IS/OOS fold on all available data):

  1. Fetch all 15m bars from Bybit back to SINCE_DATE
  2. Compute indicators + detect signals + extract 13 features
  3. IS = first 75% of bars | OOS = last 25% of bars
  4. Train DecisionTree(depth=2, min_leaf=15) on IS signals
  5. Tune prediction threshold on IS to maximize IS Sharpe under DT filter
  6. Validate: measure OOS Sharpe with and without DT
     - Accept if OOS DT Sharpe >= 0 and does not degrade raw OOS by more than 20%
  7. If accepted: retrain on ALL data with the same threshold
  8. Save model + threshold to MODELS_DIR/<SYMBOL>_dt.pkl

Only runs for coins with "use_dt": true in the configs file.

Usage (local, from repo root, with .venv active):
    python bot/train_dt.py

Usage (Docker one-off, after `docker compose build mm-bot`):
    docker compose run --rm mm-train

Reads:  bot/configs/top20_configs.json
Writes: bot/models/<SYMBOL>_dt.pkl
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
from pybit.unified_trading import HTTP
from sklearn.tree import DecisionTreeClassifier

# indicators.py must be on the Python path (same directory as this script)
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from indicators import (
    ST_MULT, ST_ATR_LEN, EMA_LEN, SMA_LEN, ATR_LEN,
    VOL_WIN, ATR_PCTILE_WIN, ATR_LO, ATR_HI, VOL_THR,
    N_FEATURES,
    ind_atr, ind_ema, ind_sma,
    build_signals,
    rolling_atr_pctile, rolling_vol_ratio,
    compute_rsi, compute_rsi_htf,
    extract_features_batch,
    sim_trail, metrics,
)

# ── Config ────────────────────────────────────────────────────────────────────
SINCE_DATE   = "2024-01-01"
TIMEFRAME    = "15"
IS_FRACTION  = 0.75     # first 75% of data used for IS training
MIN_IS_SIGS  = 20       # minimum IS signals needed to fit a DT
DT_DEPTH     = 2
DT_MIN_LEAF  = 15
DT_MIN_IS_SH = 0.5      # IS Sharpe gate: skip if DT-filtered IS Sharpe < this
# Accept OOS if (DT OOS Sharpe >= 0) AND (DT degrades raw OOS by < 20%)
OOS_MIN_SH        = 0.0
OOS_MAX_DEGRADATION = 0.20   # fraction

CONFIGS_PATH = os.environ.get("CONFIGS_PATH", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "configs", "top20_configs.json"))
MODELS_DIR   = os.environ.get("MODELS_DIR", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models"))


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


def instrument_status(symbol: str, http: HTTP) -> str:
    resp = http.get_instruments_info(category="linear", symbol=symbol)
    items = resp.get("result", {}).get("list", [])
    if not items:
        return ""
    return str(items[0].get("status", ""))


# ── Training ──────────────────────────────────────────────────────────────────

def train_symbol(symbol: str, cfg: dict, bars: list[dict]) -> Optional[dict]:
    """
    Train DT for one symbol.  Returns save dict or None if validation fails.
    """
    n = len(bars)
    if n < 500:
        print(f"  {symbol}: only {n} bars — skipping")
        return None

    o  = np.array([b["open"]   for b in bars], dtype=np.float64)
    h  = np.array([b["high"]   for b in bars], dtype=np.float64)
    l  = np.array([b["low"]    for b in bars], dtype=np.float64)
    c  = np.array([b["close"]  for b in bars], dtype=np.float64)
    v  = np.array([b["volume"] for b in bars], dtype=np.float64)

    ts_ms  = np.array([b["ts"] for b in bars], dtype=np.int64)
    ts_idx = pd.DatetimeIndex(
        pd.to_datetime(ts_ms, unit="ms", utc=True)
    )

    # Indicators
    atr_st  = ind_atr(h, l, c, ST_ATR_LEN)
    atr14   = ind_atr(h, l, c, ATR_LEN)
    ema200  = ind_ema(c, EMA_LEN)
    sma13   = ind_sma(c, SMA_LEN)
    atr_pct = rolling_atr_pctile(atr14, ATR_PCTILE_WIN)
    vol_rat = rolling_vol_ratio(v, VOL_WIN)
    rsi14   = compute_rsi(c, 14)
    rsi4h   = compute_rsi_htf(ts_idx, c, "4h", 14)

    sbull, sbear = build_signals(c, o, sma13, ema200, atr_st)

    # All valid signal indices (pass ATR + volume gates)
    sig_idx = [i for i in range(n)
               if (sbull[i] or sbear[i])
               and ATR_LO < atr_pct[i] < ATR_HI
               and vol_rat[i] >= VOL_THR]

    if len(sig_idx) < MIN_IS_SIGS * 2:
        print(f"  {symbol}: only {len(sig_idx)} total signals — skipping")
        return None

    # IS / OOS split
    split_i = int(n * IS_FRACTION)
    is_sig  = [i for i in sig_idx if i < split_i]
    oos_sig = [i for i in sig_idx if i >= split_i]

    if len(is_sig) < MIN_IS_SIGS:
        print(f"  {symbol}: only {len(is_sig)} IS signals — skipping")
        return None

    sl_m = cfg["sl"]; tp1_r = cfg["tp1"]; trail_m = cfg["trail"]

    # ── IS simulation (needed for labels) ─────────────────────────────────────
    r_is, td_is = sim_trail(
        c[:split_i], h[:split_i], l[:split_i],
        sbull[:split_i], sbear[:split_i],
        atr14[:split_i], atr_pct[:split_i], vol_rat[:split_i],
        sl_mult=sl_m, tp1_r=tp1_r, trail_mult=trail_m,
    )

    if len(td_is) < MIN_IS_SIGS:
        print(f"  {symbol}: only {len(td_is)} IS trades — skipping")
        return None

    is_label_map = {t["entry_i"]: (1 if t["r"] > 0 else 0) for t in td_is}

    # Build IS feature matrix
    feat_rows_is = extract_features_batch(
        is_sig, c, h, l, o, v,
        atr14, atr_pct, vol_rat, ema200, sma13,
        sbull, sbear,
        [datetime.fromtimestamp(ts_ms[i] / 1000, tz=timezone.utc) for i in range(n)],
        rsi14, rsi4h,
    )

    X_list, y_list, valid_idx = [], [], []
    for k, gi in enumerate(is_sig):
        row = feat_rows_is[k]
        lbl = is_label_map.get(gi)
        if row is None or lbl is None:
            continue
        X_list.append(row); y_list.append(lbl); valid_idx.append(gi)

    if len(X_list) < MIN_IS_SIGS or len(set(y_list)) < 2:
        print(f"  {symbol}: insufficient IS feature rows — skipping")
        return None

    X_is = np.array(X_list, dtype=np.float64)
    y_is = np.array(y_list, dtype=int)

    # ── Train DT on IS ────────────────────────────────────────────────────────
    clf_is = DecisionTreeClassifier(
        max_depth=DT_DEPTH, min_samples_leaf=DT_MIN_LEAF,
        class_weight="balanced", random_state=42,
    )
    clf_is.fit(X_is, y_is)

    # Tune threshold on IS
    is_r_map   = {t["entry_i"]: t["r"] for t in td_is}
    is_probas  = clf_is.predict_proba(X_is)[:, 1]
    is_gp      = [(gi, float(is_probas[k])) for k, gi in enumerate(valid_idx)]
    best_thr   = 0.55; best_is_sh = -99.0
    for thr in np.arange(0.40, 0.91, 0.05):
        sel_r = [is_r_map[gi] for gi, p in is_gp if p >= thr and gi in is_r_map]
        if len(sel_r) < 8:
            continue
        sh = metrics(np.array(sel_r), min_n=5)["sharpe"]
        if sh > best_is_sh:
            best_is_sh = sh; best_thr = thr

    if best_is_sh < DT_MIN_IS_SH:
        print(
            f"  {symbol}: IS Sharpe with DT={best_is_sh:.3f} < "
            f"gate={DT_MIN_IS_SH} — skipping"
        )
        return None

    # ── OOS evaluation ────────────────────────────────────────────────────────
    r_oos_raw, _ = sim_trail(
        c[split_i:], h[split_i:], l[split_i:],
        sbull[split_i:], sbear[split_i:],
        atr14[split_i:], atr_pct[split_i:], vol_rat[split_i:],
        sl_mult=sl_m, tp1_r=tp1_r, trail_mult=trail_m,
    )
    m_raw = metrics(r_oos_raw, min_n=3)

    # Build OOS mask using clf_is + best_thr
    if len(oos_sig) > 0:
        feat_rows_oos = extract_features_batch(
            oos_sig, c, h, l, o, v,
            atr14, atr_pct, vol_rat, ema200, sma13,
            sbull, sbear,
            [datetime.fromtimestamp(ts_ms[i] / 1000, tz=timezone.utc) for i in range(n)],
            rsi14, rsi4h,
        )
        X_oos_rows, valid_oos = [], []
        for k, gi in enumerate(oos_sig):
            row = feat_rows_oos[k]
            if row is not None:
                X_oos_rows.append(row); valid_oos.append(gi)

        oos_mask = np.zeros(n - split_i, dtype=bool)
        if X_oos_rows:
            probs_oos = clf_is.predict_proba(
                np.array(X_oos_rows, dtype=np.float64))[:, 1]
            for k, gi in enumerate(valid_oos):
                if probs_oos[k] >= best_thr:
                    oos_mask[gi - split_i] = True

        r_oos_dt, _ = sim_trail(
            c[split_i:], h[split_i:], l[split_i:],
            sbull[split_i:], sbear[split_i:],
            atr14[split_i:], atr_pct[split_i:], vol_rat[split_i:],
            sl_mult=sl_m, tp1_r=tp1_r, trail_mult=trail_m,
            signal_mask=oos_mask,
        )
        m_dt = metrics(r_oos_dt, min_n=3)
    else:
        m_dt = {"sharpe": -99.0, "n": 0}

    oos_raw_sh = m_raw["sharpe"]
    oos_dt_sh  = m_dt["sharpe"]

    # Acceptance decision
    raw_floor = oos_raw_sh - abs(oos_raw_sh) * OOS_MAX_DEGRADATION
    accepted  = (oos_dt_sh >= OOS_MIN_SH) and (oos_dt_sh >= raw_floor)

    if not accepted:
        print(
            f"  {symbol}: OOS validation FAILED  "
            f"raw={oos_raw_sh:+.3f}  dt={oos_dt_sh:+.3f}  "
            f"(floor={raw_floor:+.3f}) — model not saved"
        )
        return None

    # ── Retrain on ALL data ───────────────────────────────────────────────────
    r_all, td_all = sim_trail(
        c, h, l, sbull, sbear, atr14, atr_pct, vol_rat,
        sl_mult=sl_m, tp1_r=tp1_r, trail_mult=trail_m,
    )
    all_label_map = {t["entry_i"]: (1 if t["r"] > 0 else 0) for t in td_all}

    feat_all = extract_features_batch(
        sig_idx, c, h, l, o, v,
        atr14, atr_pct, vol_rat, ema200, sma13,
        sbull, sbear,
        [datetime.fromtimestamp(ts_ms[i] / 1000, tz=timezone.utc) for i in range(n)],
        rsi14, rsi4h,
    )
    X_all_rows, y_all = [], []
    for k, gi in enumerate(sig_idx):
        row = feat_all[k]
        lbl = all_label_map.get(gi)
        if row is not None and lbl is not None:
            X_all_rows.append(row); y_all.append(lbl)

    clf_all = DecisionTreeClassifier(
        max_depth=DT_DEPTH, min_samples_leaf=DT_MIN_LEAF,
        class_weight="balanced", random_state=42,
    )
    clf_all.fit(np.array(X_all_rows, dtype=np.float64), np.array(y_all, dtype=int))

    print(
        f"  {symbol}: ACCEPTED  "
        f"IS_sh={best_is_sh:+.3f}  OOS_raw={oos_raw_sh:+.3f}  OOS_dt={oos_dt_sh:+.3f}  "
        f"thr={best_thr:.2f}  n_all={n}  sigs_all={len(X_all_rows)}"
    )

    return {
        "model":      clf_all,
        "threshold":  best_thr,
        "is_sh":      round(best_is_sh, 4),
        "oos_raw_sh": round(oos_raw_sh, 4),
        "oos_dt_sh":  round(oos_dt_sh, 4),
        "trained_on": n,
        "trained_at": datetime.now(tz=timezone.utc).isoformat(),
        "symbol":     symbol,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MM V4.3 DT Training")
    parser.add_argument("--since",   default=SINCE_DATE,
                        help="Fetch bars from this date (YYYY-MM-DD)")
    parser.add_argument("--symbols", default=None,
                        help="Comma-separated symbol list (default: all use_dt=true in config)")
    parser.add_argument("--demo",    action="store_true",
                        help="Use Bybit demo endpoint for fetching data")
    args = parser.parse_args()

    with open(CONFIGS_PATH) as fh:
        all_configs: dict[str, dict] = json.load(fh)
    all_configs = {k: v for k, v in all_configs.items() if not k.startswith("_")}

    if args.symbols:
        target = {s.strip() for s in args.symbols.split(",")}
        configs = {k: v for k, v in all_configs.items() if k in target}
    else:
        configs = {k: v for k, v in all_configs.items() if v.get("use_dt", False)}

    if not configs:
        print("No coins with use_dt=true found in config. Nothing to train.")
        return

    demo = args.demo or os.environ.get("BYBIT_DEMO", "true").lower() in ("1", "true", "yes")
    http = HTTP(
        testnet=False,
        demo=demo,
        api_key=os.environ.get("BYBIT_API_KEY", ""),
        api_secret=os.environ.get("BYBIT_API_SECRET", ""),
    )

    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"Training DT filters for {len(configs)} coins  (since={args.since})")
    print(f"IS={int(IS_FRACTION*100)}%  OOS={int((1-IS_FRACTION)*100)}%  "
          f"depth={DT_DEPTH}  min_leaf={DT_MIN_LEAF}\n")

    saved = 0; failed = 0; t0_total = time.time()
    summary_rows: list[dict] = []

    for sym, cfg in configs.items():
        t0 = time.time()
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

        print(f"{sym}: fetching bars ...", end=" ", flush=True)
        try:
            bars = fetch_all_bars(sym, args.since, http)
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

        result = train_symbol(sym, cfg, bars)
        if result is None:
            summary_rows.append({"symbol": sym, "status": "skipped_or_rejected", "bars": len(bars)})
            failed += 1
            continue

        out_path = os.path.join(MODELS_DIR, f"{sym}_dt.pkl")
        with open(out_path, "wb") as fh:
            pickle.dump(result, fh)
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
