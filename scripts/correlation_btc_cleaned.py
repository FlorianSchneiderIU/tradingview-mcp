"""
correlation_btc_cleaned.py — BTC-cleaned idiosyncratic correlation analysis
============================================================================
Same analysis as correlation_research.py but with BTC's influence removed first.

Method (for each coin):
  1. Fit OLS: coin_15m_log_return = alpha + beta * btc_15m_log_return + residual
  2. The RESIDUAL is the "BTC-cleaned" idiosyncratic return — the part of each
     coin's movement that BTC does NOT explain.
  3. Compute the full correlation matrix on those residuals.
  4. Compare raw vs BTC-cleaned correlations, cluster on residuals.

This answers: are coins genuinely correlated with each other, or does BTC
just drag them all around together?

Outputs printed to console + saved as CSV in scripts/:
  - btc_sensitivity.csv              : beta, R², idio_vol for each coin
  - btc_cleaned_corr_matrix.csv      : full residual correlation matrix
  - btc_cleaned_corr_comparison.csv  : raw vs cleaned corr for every pair

Usage:
  python scripts/correlation_btc_cleaned.py
  python scripts/correlation_btc_cleaned.py --since 2024-01-01 --threshold 0.3
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import ccxt

warnings.filterwarnings("ignore")

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_DIR     = os.path.dirname(SCRIPT_DIR)
CONFIGS_PATH = os.path.join(REPO_DIR, "bot", "configs", "top20_configs.json")

SINCE_DATE = "2024-01-01"
TIMEFRAME  = "15m"


def to_ccxt(sym: str) -> str:
    base = sym.replace("USDT", "").rstrip("/")
    return f"{base}/USDT:USDT"


# ── Data fetch ────────────────────────────────────────────────────────────────

def fetch(sym_ccxt: str, exchange, since_date: str) -> pd.DataFrame:
    since_ms = exchange.parse8601(f"{since_date}T00:00:00Z")
    now_ms   = int(time.time() * 1000)
    bars: list = []
    while True:
        chunk = exchange.fetch_ohlcv(sym_ccxt, TIMEFRAME, since=since_ms, limit=1000)
        if not chunk:
            break
        bars.extend(chunk)
        last_ts = chunk[-1][0]
        if last_ts >= now_ms - 15 * 60 * 1000:
            break
        if len(chunk) < 10:
            break
        since_ms = last_ts + 1
        time.sleep(0.12)
    df = pd.DataFrame(bars, columns=["ts", "open", "high", "low", "close", "volume"])
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("dt").sort_index()
    df = df.loc[~df.index.duplicated()]
    return df


# ── BTC regression helpers ────────────────────────────────────────────────────

def ols_residuals(y: np.ndarray, x: np.ndarray):
    """
    Regress y on x (with intercept). Returns (alpha, beta, R², residuals).
    Uses numpy least squares — no external dependency needed.
    """
    X = np.column_stack([np.ones(len(x)), x])
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = float(coeffs[0]), float(coeffs[1])
    y_hat = alpha + beta * x
    resid = y - y_hat
    ss_tot = float(np.var(y)) * len(y)
    ss_res = float(np.var(resid)) * len(resid)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return alpha, beta, r2, resid


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since",     default=SINCE_DATE)
    ap.add_argument("--top",       type=int,   default=20)
    ap.add_argument("--threshold", type=float, default=0.30,
                    help="Correlation threshold for HIGH list in BTC-cleaned matrix (default 0.30)")
    args = ap.parse_args()

    with open(CONFIGS_PATH) as fh:
        raw = json.load(fh)
    configs  = {k: v for k, v in raw.items() if not k.startswith("_")}
    symbols  = list(configs.keys())[: args.top]
    btc_sym  = "BTC/USDT:USDT"

    exchange = ccxt.bybit({
        "enableRateLimit": True,
        "options": {"defaultType": "linear"},
    })

    # ── 1. Fetch all data (BTC first) ─────────────────────────────────────────
    print(f"Analysing {len(symbols)} coins + BTC | since={args.since}\n")

    dfs: dict[str, pd.DataFrame] = {}

    # Fetch BTC
    print(f"  Fetching {btc_sym} (BTC) ...", end=" ", flush=True)
    try:
        dfs["BTCUSDT"] = fetch(btc_sym, exchange, args.since)
        print(f"{len(dfs['BTCUSDT'])} bars")
    except Exception as e:
        print(f"ERROR: {e}")
        return

    # Fetch portfolio coins
    for sym in symbols:
        ccxt_sym = to_ccxt(sym)
        print(f"  Fetching {ccxt_sym} ...", end=" ", flush=True)
        try:
            df = fetch(ccxt_sym, exchange, args.since)
            print(f"{len(df)} bars")
            dfs[sym] = df
        except Exception as e:
            print(f"ERROR: {e}")

    symbols = [s for s in symbols if s in dfs]
    print(f"\n{len(symbols)} portfolio coins loaded + BTC\n")

    # ── 2. Compute 15m log-returns, align on common timestamps ───────────────
    ret_series: dict[str, pd.Series] = {}
    for sym, df in dfs.items():
        r = np.log(df["close"] / df["close"].shift(1)).rename(sym)
        ret_series[sym] = r

    # Stack all together (including BTC) and drop any row with a NaN in any coin
    ret_df = pd.DataFrame(ret_series).dropna()
    btc_ret = ret_df["BTCUSDT"].values

    # ── 3. Raw correlation matrix (same as correlation_research.py section 1) ─
    print("=" * 70)
    print("1. RAW PRICE RETURN CORRELATIONS (15m log-returns, reference)")
    print("=" * 70)

    raw_corr = ret_df[symbols].corr()
    raw_pairs = []
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            c = raw_corr.iloc[i, j]
            raw_pairs.append((symbols[i], symbols[j], round(c, 3)))
    raw_pairs.sort(key=lambda x: -x[2])

    high_raw = [(a, b, c) for a, b, c in raw_pairs if c >= 0.60]
    print(f"\nPairs with raw return corr >= 0.60 ({len(high_raw)} pairs):")
    print(f"  {'Coin A':<14} {'Coin B':<14} {'Raw corr':>9}")
    print(f"  {'-'*14} {'-'*14} {'-'*9}")
    for a, b, c in high_raw[:20]:
        print(f"  {a:<14} {b:<14} {c:>9.3f}")
    if len(high_raw) > 20:
        print(f"  ... ({len(high_raw) - 20} more pairs not shown)")

    # ── 4. BTC sensitivity per coin ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("2. BTC SENSITIVITY — how much of each coin's variance does BTC explain?")
    print("=" * 70)

    sensitivity_rows = []
    residuals: dict[str, np.ndarray] = {}

    print(f"\n  {'Coin':<14} {'Beta':>7} {'Alpha(ann%)':>12} {'R²':>7} {'Idio_vol%':>10}")
    print(f"  {'-'*14} {'-'*7} {'-'*12} {'-'*7} {'-'*10}")

    for sym in symbols:
        y = ret_df[sym].values
        alpha, beta, r2, resid = ols_residuals(y, btc_ret)
        residuals[sym] = resid

        # Annualise: 15m bars → 26,280 bars / year
        BARS_PER_YEAR = 26_280
        alpha_ann_pct = alpha * BARS_PER_YEAR * 100
        total_vol   = float(np.std(y))       * np.sqrt(BARS_PER_YEAR) * 100
        idio_vol    = float(np.std(resid))   * np.sqrt(BARS_PER_YEAR) * 100

        print(f"  {sym:<14} {beta:>7.3f} {alpha_ann_pct:>11.1f}% {r2:>7.3f} {idio_vol:>9.1f}%")
        sensitivity_rows.append({
            "coin":          sym,
            "btc_beta":      round(beta, 4),
            "alpha_ann_pct": round(alpha_ann_pct, 2),
            "btc_r2":        round(r2, 4),
            "total_vol_ann": round(total_vol, 2),
            "idio_vol_ann":  round(idio_vol, 2),
        })

    sens_path = os.path.join(SCRIPT_DIR, "btc_sensitivity.csv")
    pd.DataFrame(sensitivity_rows).to_csv(sens_path, index=False)
    print(f"\nSaved: {sens_path}")

    # ── 5. BTC-cleaned (residual) correlation matrix ──────────────────────────
    print("\n" + "=" * 70)
    print("3. BTC-CLEANED RESIDUAL CORRELATIONS")
    print("=" * 70)
    print("   (these are correlations AFTER removing what BTC explains)")

    resid_df  = pd.DataFrame(residuals, index=ret_df.index)
    clean_corr = resid_df.corr()

    clean_pairs = []
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            rc = clean_corr.iloc[i, j]
            clean_pairs.append((symbols[i], symbols[j], round(rc, 3)))
    clean_pairs.sort(key=lambda x: -x[2])

    high_clean = [(a, b, c) for a, b, c in clean_pairs if c >= args.threshold]
    print(f"\nPairs with BTC-cleaned corr >= {args.threshold:.2f} ({len(high_clean)} pairs):")
    print(f"  {'Coin A':<14} {'Coin B':<14} {'Clean corr':>10}")
    print(f"  {'-'*14} {'-'*14} {'-'*10}")
    for a, b, c in high_clean:
        print(f"  {a:<14} {b:<14} {c:>10.3f}")

    # ── 6. Comparison table: raw vs BTC-cleaned ───────────────────────────────
    print("\n" + "=" * 70)
    print("4. COMPARISON: raw return corr vs BTC-cleaned corr")
    print("=" * 70)
    print("   Delta = clean - raw  (positive means BTC was MASKING correlation;")
    print("                         negative means BTC was the main driver)")

    raw_dict   = {(a, b): c for a, b, c in raw_pairs}
    clean_dict = {(a, b): c for a, b, c in clean_pairs}

    comp_rows = []
    for (a, b) in raw_dict:
        raw_c   = raw_dict[(a, b)]
        clean_c = clean_dict.get((a, b), 0.0)
        delta   = round(clean_c - raw_c, 3)
        comp_rows.append((a, b, raw_c, clean_c, delta))

    # Sort by absolute BTC-cleaned corr descending
    comp_rows.sort(key=lambda x: -abs(x[3]))

    print(f"\nTop 30 pairs by |BTC-cleaned corr| (was high-raw vs low-clean tells you BTC was the glue):")
    print(f"  {'Coin A':<14} {'Coin B':<14} {'Raw':>7} {'Cleaned':>9} {'Delta':>7}")
    print(f"  {'-'*14} {'-'*14} {'-'*7} {'-'*9} {'-'*7}")
    for a, b, raw_c, clean_c, delta in comp_rows[:30]:
        flag = " ← BTC driven" if raw_c >= 0.60 and clean_c < 0.30 else (
               " ← GENUINE"   if clean_c >= 0.30 else "")
        print(f"  {a:<14} {b:<14} {raw_c:>7.3f} {clean_c:>9.3f} {delta:>+7.3f}{flag}")

    # Show biggest drop-offs: pairs that were highly correlated but aren't after BTC removal
    big_drops = [(a, b, r, cl, d) for a, b, r, cl, d in comp_rows if r >= 0.60 and cl < 0.30]
    big_drops.sort(key=lambda x: x[4])  # most negative delta first
    if big_drops:
        print(f"\nPairs that COLLAPSE after BTC removal (raw >= 0.60, clean < 0.30):")
        print(f"  — these correlations were almost entirely BTC-driven ({len(big_drops)} pairs)")
        for a, b, r, cl, d in big_drops[:15]:
            print(f"  {a:<14} {b:<14}  raw={r:.3f}  clean={cl:.3f}  drop={d:+.3f}")

    # Show pairs that remain correlated despite BTC removal (genuine idiosyncratic correlation)
    genuines = [(a, b, r, cl, d) for a, b, r, cl, d in comp_rows if cl >= 0.30]
    genuines.sort(key=lambda x: -x[3])
    if genuines:
        print(f"\nPairs that SURVIVE BTC removal (clean >= 0.30, {len(genuines)} pairs):")
        print(f"  — these have genuine sector/idiosyncratic correlation beyond BTC")
        for a, b, r, cl, d in genuines[:15]:
            print(f"  {a:<14} {b:<14}  raw={r:.3f}  clean={cl:.3f}  delta={d:+.3f}")

    # ── 7. Cluster analysis on BTC-cleaned residuals ──────────────────────────
    print("\n" + "=" * 70)
    print("5. COIN CLUSTERS on BTC-CLEANED residuals (threshold >= 0.30)")
    print("=" * 70)

    parent = {s: s for s in symbols}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        parent[find(x)] = find(y)

    for a, b, c in high_clean:
        if a in parent and b in parent:
            union(a, b)

    clusters: dict[str, list[str]] = defaultdict(list)
    for s in symbols:
        clusters[find(s)].append(s)

    print("\nClusters (coins whose IDIOSYNCRATIC moves are correlated, beyond BTC):")
    for root, members in sorted(clusters.items(), key=lambda x: -len(x[1])):
        if len(members) > 2:
            tag = "CORRELATED CLUSTER"
        elif len(members) == 2:
            tag = "PAIR"
        else:
            tag = "solo / BTC-only"
        print(f"  [{tag}] {', '.join(sorted(members))}")

    # ── 8. BTC correlation with each coin (how BTC-like each coin is) ─────────
    print("\n" + "=" * 70)
    print("6. BTC ↔ COIN CORRELATION (raw direct correlation with BTC)")
    print("=" * 70)

    btc_coin_corrs = []
    for sym in symbols:
        c = float(np.corrcoef(btc_ret, ret_df[sym].values)[0, 1])
        btc_coin_corrs.append((sym, round(c, 3)))
    btc_coin_corrs.sort(key=lambda x: -x[1])

    print(f"\n  {'Coin':<14} {'Corr w/ BTC':>12}")
    print(f"  {'-'*14} {'-'*12}")
    for sym, c in btc_coin_corrs:
        bar = "█" * int(c * 20)
        print(f"  {sym:<14} {c:>12.3f}  {bar}")

    # ── Save all CSVs ─────────────────────────────────────────────────────────
    clean_corr_path = os.path.join(SCRIPT_DIR, "btc_cleaned_corr_matrix.csv")
    clean_corr.to_csv(clean_corr_path)

    comp_df = pd.DataFrame([
        {"coin_a": a, "coin_b": b,
         "raw_corr": raw_c, "btc_cleaned_corr": clean_c, "delta": delta}
        for a, b, raw_c, clean_c, delta in comp_rows
    ])
    comp_path = os.path.join(SCRIPT_DIR, "btc_cleaned_corr_comparison.csv")
    comp_df.to_csv(comp_path, index=False)

    print(f"\nSaved: {clean_corr_path}")
    print(f"Saved: {comp_path}")
    print(f"Saved: {sens_path}")

    print("\n" + "=" * 70)
    print("RESEARCH COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
