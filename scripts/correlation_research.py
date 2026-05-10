"""
correlation_research.py — Correlation analysis for MM V4.3 bot coins
======================================================================
Answers three questions:
  1. How correlated are 15m price returns between our 20 coins?
  2. How often do signals fire on multiple coins at the same 15m bar (signal clustering)?
  3. When we hold concurrent positions, how often do they all lose together (correlated loss)?

For question 3 we simulate ALL trades (trail exit, best configs from top20_configs.json)
and build a "concurrent loss events" table: which coin groups were in losing trades
simultaneously. This shows the actual worst-case portfolio drawdown clusters.

Outputs (printed to console + saved as CSV in scripts/):
  - Price return correlation matrix (15m)
  - Signal timing overlap matrix: P(coin_j signals within ±2 bars | coin_i signals)
  - Concurrent position loss rate: for each pair, fraction of overlapping holding periods
    where BOTH end as losses
  - "Big loss days": UTC days where ≥3 coins had a losing trade — full list

Usage:
  python scripts/correlation_research.py
  python scripts/correlation_research.py --since 2024-01-01 --top 20
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import ccxt

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR   = os.path.dirname(SCRIPT_DIR)
CONFIGS_PATH = os.path.join(REPO_DIR, "bot", "configs", "top20_configs.json")

# Map Bybit perp symbol (e.g. MKRUSDT) -> ccxt linear perp format (e.g. MKR/USDT:USDT)
def to_ccxt(sym: str) -> str:
    base = sym.replace("USDT", "").rstrip("/")
    return f"{base}/USDT:USDT"

# ── Config (must match multi.py / indicators.py) ──────────────────────────────
SINCE_DATE     = "2024-01-01"
TIMEFRAME      = "15m"
ST_MULT        = 3.5; ST_ATR_LEN = 11
EMA_LEN        = 200; SMA_LEN = 13; ATR_LEN = 14
VOL_WIN        = 20;  ATR_PCTILE_WIN = 100
ATR_LO = 10; ATR_HI = 90; VOL_THR = 1.05

SIGNAL_WINDOW  = 2   # bars; ±2 × 15m = ±30 min for "near-simultaneous signal" check

# ── Indicators ────────────────────────────────────────────────────────────────

def _rma(vals, n):
    alpha = 1.0 / n
    out = np.full(len(vals), np.nan)
    s = 0
    while s < len(vals) and np.isnan(vals[s]): s += 1
    se = s + n
    if se > len(vals): return out
    out[se-1] = float(np.nanmean(vals[s:se]))
    for i in range(se, len(vals)):
        v = vals[i]
        out[i] = alpha * v + (1-alpha)*out[i-1] if not np.isnan(v) else out[i-1]
    return out

def atr(h, l, c, n):
    pc = np.empty_like(c); pc[0] = np.nan; pc[1:] = c[:-1]
    tr = np.maximum(h-l, np.maximum(np.abs(h-pc), np.abs(l-pc)))
    return _rma(tr, n)

def ema(c, n):
    a = 2/(n+1); out = np.full(len(c), np.nan)
    for i,v in enumerate(c):
        if not np.isnan(v):
            out[i] = v
            for j in range(i+1, len(c)): out[j] = a*c[j] + (1-a)*out[j-1]
            break
    return out

def sma(arr, n): return pd.Series(arr).rolling(n).mean().values

def supertrend(open_, close, atr_arr, mult):
    n = len(open_)
    ur = open_ + mult*atr_arr; lr = open_ - mult*atr_arr
    upper, lower = ur.copy(), lr.copy()
    direction = np.full(n, 2.0); st = np.full(n, np.nan)
    for i in range(1, n):
        if np.isnan(atr_arr[i-1]): direction[i]=2.0; upper[i]=ur[i]; lower[i]=lr[i]
        else:
            lower[i] = lr[i] if (lr[i]>lower[i-1] or close[i-1]<lower[i-1]) else lower[i-1]
            upper[i] = ur[i] if (ur[i]<upper[i-1] or close[i-1]>upper[i-1]) else upper[i-1]
            ps = st[i-1]
            if np.isnan(ps): ps = upper[i-1] if not np.isnan(upper[i-1]) else lower[i-1]
            if ps == upper[i-1]: direction[i] = -1.0 if close[i]>upper[i] else 1.0
            else:                direction[i] =  1.0 if close[i]<lower[i] else -1.0
        st[i] = lower[i] if direction[i]==-1.0 else upper[i]
    return st

def signals(close, open_, sma13, ema200, atr_st):
    st = supertrend(open_, close, atr_st, ST_MULT)
    n = len(close)
    pc=np.empty(n); pc[0]=np.nan; pc[1:]=close[:-1]
    ps=np.empty(n); ps[0]=np.nan; ps[1:]=st[:-1]
    pe=np.empty(n); pe[0]=np.nan; pe[1:]=ema200[:-1]
    co  = (~np.isnan(pc))&(~np.isnan(ps))&(~np.isnan(st))&(pc<ps)&(close>st)
    cu  = (~np.isnan(pc))&(~np.isnan(ps))&(~np.isnan(st))&(pc>ps)&(close<st)
    abv = (~np.isnan(pe))&(~np.isnan(ema200))&(pc>pe)&(close>ema200)
    sb  = co&(~np.isnan(sma13))&(close>=sma13)& abv
    se  = cu&(~np.isnan(sma13))&(close<=sma13)&(~abv)
    return sb.astype(bool), se.astype(bool)

def rolling_atr_pct(atr14, w=ATR_PCTILE_WIN):
    n=len(atr14); out=np.full(n,50.0)
    for i in range(w,n):
        ww=atr14[i-w:i]
        if not np.isnan(atr14[i]) and not np.all(np.isnan(ww)):
            v=ww[~np.isnan(ww)]; out[i]=float(np.sum(v<atr14[i]))/len(v)*100
    return out

def vol_ratio(volume, w=VOL_WIN):
    vs=sma(volume,w)
    with np.errstate(invalid="ignore",divide="ignore"):
        r=volume/vs
    return np.where(np.isnan(r)|np.isinf(r),1.0,r)

def sim_trail(close, high, low, sbull, sbear, atr14, atr_pct, vol_rat,
              sl_m, tp1_r, trail_m):
    n=len(close); trades=[]
    active=False; is_long=False; entry=sl_=tp1=0.0; risk=1.0
    tp1_hit=False; trail_sl=acc_r=0.0; entry_i=0
    for i in range(1,n):
        h=high[i]; l=low[i]; c=close[i]; a=atr14[i]
        if active:
            if is_long:
                if not tp1_hit and h>=tp1: acc_r+=0.5*tp1_r; trail_sl=entry; tp1_hit=True
                if tp1_hit:
                    if not math.isnan(a):
                        cand=h-trail_m*a
                        if cand>trail_sl: trail_sl=cand
                    if l<=trail_sl:
                        tot=acc_r+0.5*max(0.0,(trail_sl-entry)/risk)
                        trades.append(dict(entry_i=entry_i,exit_i=i,r=tot,long=True)); active=False; continue
                else:
                    if l<=sl_:
                        trades.append(dict(entry_i=entry_i,exit_i=i,r=-1.0,long=True)); active=False
            else:
                if not tp1_hit and l<=tp1: acc_r+=0.5*tp1_r; trail_sl=entry; tp1_hit=True
                if tp1_hit:
                    if not math.isnan(a):
                        cand=l+trail_m*a
                        if cand<trail_sl: trail_sl=cand
                    if h>=trail_sl:
                        tot=acc_r+0.5*max(0.0,(entry-trail_sl)/risk)
                        trades.append(dict(entry_i=entry_i,exit_i=i,r=tot,long=False)); active=False; continue
                else:
                    if h>=sl_:
                        trades.append(dict(entry_i=entry_i,exit_i=i,r=-1.0,long=False)); active=False
        if active and is_long  and sbear[i]:
            rem=0.5 if tp1_hit else 1.0
            trades.append(dict(entry_i=entry_i,exit_i=i,r=acc_r+rem*(c-entry)/risk,long=True)); active=False
        if active and not is_long and sbull[i]:
            rem=0.5 if tp1_hit else 1.0
            trades.append(dict(entry_i=entry_i,exit_i=i,r=acc_r+rem*(entry-c)/risk,long=False)); active=False
        if not active and not math.isnan(a):
            ap=atr_pct[i]
            if not(ATR_LO<ap<ATR_HI): continue
            if vol_rat[i]<VOL_THR: continue
            if sbull[i]:
                sl_=l-a*sl_m; risk=max(c-sl_,1e-10); entry=c; is_long=True
                tp1=c+tp1_r*risk; tp1_hit=False; trail_sl=sl_; acc_r=0.0; active=True; entry_i=i
            elif sbear[i]:
                sl_=h+a*sl_m; risk=max(sl_-c,1e-10); entry=c; is_long=False
                tp1=c-tp1_r*risk; tp1_hit=False; trail_sl=sl_; acc_r=0.0; active=True; entry_i=i
    if active:
        cl=close[-1]; rem=0.5 if tp1_hit else 1.0
        tot=acc_r+rem*((cl-entry) if is_long else (entry-cl))/risk
        trades.append(dict(entry_i=entry_i,exit_i=n-1,r=tot,long=is_long))
    return trades

# ── Data fetch ────────────────────────────────────────────────────────────────

def fetch(sym_ccxt, exchange, since_date):
    since_ms = exchange.parse8601(f"{since_date}T00:00:00Z")
    now_ms   = int(time.time() * 1000)
    bars = []
    while True:
        chunk = exchange.fetch_ohlcv(sym_ccxt, TIMEFRAME, since=since_ms, limit=1000)
        if not chunk:
            break
        bars.extend(chunk)
        last_ts = chunk[-1][0]
        # Stop only when we've caught up to within 1 bar of present
        if last_ts >= now_ms - 15 * 60 * 1000:
            break
        if len(chunk) < 10:   # genuine sparse end-of-data
            break
        since_ms = last_ts + 1
        time.sleep(0.12)
    df = pd.DataFrame(bars, columns=["ts","open","high","low","close","volume"])
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.set_index("dt").sort_index().loc[~df.set_index("dt").index.duplicated()]

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=SINCE_DATE)
    ap.add_argument("--top",   type=int, default=20, help="Use top N coins from config")
    args = ap.parse_args()

    with open(CONFIGS_PATH) as fh:
        raw = json.load(fh)
    configs = {k: v for k, v in raw.items() if not k.startswith("_")}
    symbols = list(configs.keys())[:args.top]
    print(f"Analysing {len(symbols)} coins | since={args.since}\n")

    # Use Binance spot for historical OHLCV data — most complete symbol coverage
    # Price correlations are effectively identical between spot and perps
    exchange = ccxt.bybit({
        "enableRateLimit": True,
        "options": {"defaultType": "linear"},
    })

    # ── Step 1: fetch all data ─────────────────────────────────────────────────
    dfs: dict[str, pd.DataFrame] = {}
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
    print(f"\n{len(symbols)} coins loaded\n")

    # ── Step 2: price return correlation ──────────────────────────────────────
    # Align on common timestamps, compute 15m log returns
    print("=" * 70)
    print("1. PRICE RETURN CORRELATIONS (15m log-returns)")
    print("=" * 70)

    ret_series = {}
    for sym, df in dfs.items():
        r = np.log(df["close"] / df["close"].shift(1)).rename(sym)
        ret_series[sym] = r
    ret_df = pd.DataFrame(ret_series).dropna()
    corr = ret_df.corr()

    print("\nReturn correlation matrix (upper triangle):")
    syms = corr.columns.tolist()
    # Print condensed view — only pairs with corr > 0.6
    high_corr_pairs = []
    for i in range(len(syms)):
        for j in range(i+1, len(syms)):
            c = corr.iloc[i, j]
            if c >= 0.6:
                high_corr_pairs.append((syms[i], syms[j], round(c, 3)))

    high_corr_pairs.sort(key=lambda x: -x[2])
    print(f"\nPairs with return correlation >= 0.6 ({len(high_corr_pairs)} pairs):")
    print(f"  {'Coin A':<14} {'Coin B':<14} {'Corr':>6}")
    print(f"  {'-'*14} {'-'*14} {'-'*6}")
    for a, b, c in high_corr_pairs:
        print(f"  {a:<14} {b:<14} {c:>6.3f}")

    # Save full matrix
    corr_path = os.path.join(SCRIPT_DIR, "correlation_return_matrix.csv")
    corr.to_csv(corr_path)
    print(f"\nFull correlation matrix saved: {corr_path}")

    # ── Step 3: indicator computation + trade simulation ──────────────────────
    print("\n" + "=" * 70)
    print("2. SIMULATED TRADES — concurrent position & loss analysis")
    print("=" * 70)

    # Build a common bar-level time index
    all_ts = sorted(set.union(*[set(df.index) for df in dfs.values()]))
    ts_to_idx = {ts: i for i, ts in enumerate(all_ts)}
    N = len(all_ts)

    coin_trades: dict[str, list[dict]] = {}       # per-coin trade list with abs timestamps
    coin_signal_bar: dict[str, np.ndarray] = {}   # bool array over all_ts: had signal here?

    print()
    for sym in symbols:
        df    = dfs[sym]
        cfg   = configs[sym]
        close = df["close"].values
        high  = df["high"].values
        low   = df["low"].values
        open_ = df["open"].values
        vol   = df["volume"].values

        atr_st  = atr(high, low, close, ST_ATR_LEN)
        atr14   = atr(high, low, close, ATR_LEN)
        ema200  = ema(close, EMA_LEN)
        sma13   = sma(close, SMA_LEN)
        atr_pct = rolling_atr_pct(atr14)
        vol_rat = vol_ratio(vol)

        sbull, sbear = signals(close, open_, sma13, ema200, atr_st)

        trades = sim_trail(
            close, high, low, sbull, sbear, atr14, atr_pct, vol_rat,
            sl_m=cfg["sl"], tp1_r=cfg["tp1"], trail_m=cfg["trail"]
        )

        local_idx = df.index  # DatetimeIndex

        # Convert local bar indices to global all_ts positions
        abs_trades = []
        for t in trades:
            ei = t["entry_i"]; xi = t["exit_i"]
            if ei >= len(local_idx) or xi >= len(local_idx):
                continue
            abs_trades.append({
                "sym":      sym,
                "entry_ts": local_idx[ei],
                "exit_ts":  local_idx[xi],
                "r":        t["r"],
                "long":     t["long"],
                "entry_gi": ts_to_idx.get(local_idx[ei], -1),
                "exit_gi":  ts_to_idx.get(local_idx[xi], -1),
            })

        coin_trades[sym] = abs_trades

        # Signal bar mask over global time
        sig_arr = np.zeros(N, dtype=bool)
        for t in abs_trades:
            gi = t["entry_gi"]
            if gi >= 0:
                sig_arr[gi] = True
        coin_signal_bar[sym] = sig_arr

        n_trades = len(abs_trades)
        n_loss   = sum(1 for t in abs_trades if t["r"] < 0)
        print(f"  {sym:<14}: {n_trades:3d} trades  {n_loss:3d} losses  "
              f"({100*n_loss/max(n_trades,1):.0f}% loss rate)")

    # ── Step 4: signal clustering ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("3. SIGNAL CLUSTERING — how often do coins signal within ±2 bars?")
    print("=" * 70)

    # For each coin count how many other coins signalled within ±SIGNAL_WINDOW bars
    signal_overlap = np.zeros((len(symbols), len(symbols)))
    for i, a_sym in enumerate(symbols):
        a_arr = coin_signal_bar[a_sym]
        # Build padded version: any signal in ±W window
        a_win = np.zeros(N, dtype=bool)
        for w in range(-SIGNAL_WINDOW, SIGNAL_WINDOW+1):
            shifted = np.roll(a_arr, w)
            if w < 0: shifted[N+w:] = False
            elif w > 0: shifted[:w] = False
            a_win |= shifted

        a_sig_bars = np.where(a_arr)[0]  # where THIS coin signals
        for j, b_sym in enumerate(symbols):
            if i == j: signal_overlap[i, j] = 1.0; continue
            b_arr = coin_signal_bar[b_sym]
            # Of all bars where a signals, how many have b signal in ±W window?
            if len(a_sig_bars) == 0: continue
            hit = sum(1 for gi in a_sig_bars if b_arr[max(0,gi-SIGNAL_WINDOW):gi+SIGNAL_WINDOW+1].any())
            signal_overlap[i, j] = hit / len(a_sig_bars)

    # Print pairs with > 15% signal overlap
    print(f"\nPairs where P(coinB signals ±{SIGNAL_WINDOW} bars | coinA signals) > 15%:")
    print(f"  {'Coin A':<14} {'Coin B':<14} {'P(overlap)':>12}")
    print(f"  {'-'*14} {'-'*14} {'-'*12}")
    overlap_pairs = []
    for i in range(len(symbols)):
        for j in range(i+1, len(symbols)):
            p = (signal_overlap[i,j] + signal_overlap[j,i]) / 2
            if p > 0.15:
                overlap_pairs.append((symbols[i], symbols[j], round(p, 3)))
    overlap_pairs.sort(key=lambda x: -x[2])
    for a_s, b_s, p in overlap_pairs:
        print(f"  {a_s:<14} {b_s:<14} {p:>12.1%}")

    # ── Step 5: concurrent position holding + loss correlation ────────────────
    print("\n" + "=" * 70)
    print("4. CONCURRENT POSITION LOSS CORRELATION")
    print("=" * 70)

    # Build per-coin position occupancy arrays (1 = in position at this global bar)
    # and P&L contribution per bar
    pos_active   = {}   # sym -> bool array len N
    pos_loss_bar = {}   # sym -> bool array: in losing trade at this bar

    for sym in symbols:
        active_arr   = np.zeros(N, dtype=bool)
        losing_arr   = np.zeros(N, dtype=bool)
        for t in coin_trades[sym]:
            gi0 = t["entry_gi"]; gi1 = t["exit_gi"]
            if gi0 < 0 or gi1 < 0 or gi0 >= N or gi1 >= N: continue
            active_arr[gi0:gi1+1] = True
            if t["r"] < 0:
                losing_arr[gi0:gi1+1] = True
        pos_active[sym]   = active_arr
        pos_loss_bar[sym] = losing_arr

    # Pairwise: of bars where BOTH coins are in position, what fraction are BOTH losing?
    print("\nPairwise concurrent-loss rate (both in position AND both in a losing trade):")
    print(f"  {'Coin A':<14} {'Coin B':<14} {'P(both_in)':>10} {'P(both_lose|both_in)':>22} {'N_concurrent_bars':>18}")
    print(f"  {'-'*14} {'-'*14} {'-'*10} {'-'*22} {'-'*18}")
    pair_loss_rows = []
    for i in range(len(symbols)):
        for j in range(i+1, len(symbols)):
            a_sym = symbols[i]; b_sym = symbols[j]
            both_in   = pos_active[a_sym] & pos_active[b_sym]
            both_lose = pos_loss_bar[a_sym] & pos_loss_bar[b_sym]
            n_both    = int(both_in.sum())
            n_lose    = int((both_in & both_lose).sum())
            p_in      = n_both / N
            p_lose    = n_lose / n_both if n_both > 0 else 0.0
            pair_loss_rows.append((a_sym, b_sym, p_in, p_lose, n_both))

    # Sort by P(both_lose|both_in)
    pair_loss_rows.sort(key=lambda x: -x[3])
    for row in pair_loss_rows[:30]:  # top 30 most correlated losers
        a_s, b_s, p_in, p_lose, n_both = row
        marker = " *** HIGH ***" if p_lose > 0.5 else ""
        print(f"  {a_s:<14} {b_s:<14} {p_in:>10.2%} {p_lose:>22.1%} {n_both:>18,}{marker}")

    # ── Step 6: daily multi-coin loss events ──────────────────────────────────
    print("\n" + "=" * 70)
    print("5. DAILY MULTI-COIN LOSS EVENTS (≥3 coins had a losing trade exit)")
    print("=" * 70)

    # Collect all trade exits per UTC day
    from collections import defaultdict
    daily_losses: dict[str, list[str]] = defaultdict(list)
    daily_pnl:    dict[str, dict[str, float]] = defaultdict(dict)

    for sym in symbols:
        for t in coin_trades[sym]:
            day = t["exit_ts"].strftime("%Y-%m-%d")
            if t["r"] < 0:
                daily_losses[day].append(sym)
            daily_pnl[day][sym] = t["r"]

    multi_loss_days = {d: coins for d, coins in daily_losses.items() if len(coins) >= 3}
    sorted_days = sorted(multi_loss_days.items(), key=lambda x: -len(x[1]))

    print(f"\n  Total days with ≥3 coin losses: {len(multi_loss_days)}")
    print(f"\n  {'Date':<12} {'N_losing':>8} {'Coins'}")
    print(f"  {'-'*12} {'-'*8} {'-'*50}")
    for day, coins in sorted_days[:30]:
        # Also show total portfolio R for that day
        day_r = sum(daily_pnl[day].values())
        print(f"  {day:<12} {len(coins):>8}   {', '.join(coins)}  [day_R={day_r:+.2f}]")

    # ── Step 7: return correlation of CONCURRENT trades ───────────────────────
    print("\n" + "=" * 70)
    print("6. TRADE-LEVEL: return correlation between concurrent trades")
    print("=" * 70)
    print("\n(Each trade pair: entry/exit windows that overlap in time)")

    # For each pair, collect trade R values where the two trades overlapped
    pair_r_corr = []
    for i in range(len(symbols)):
        for j in range(i+1, len(symbols)):
            a_sym = symbols[i]; b_sym = symbols[j]
            a_trades = coin_trades[a_sym]; b_trades = coin_trades[b_sym]
            r_pairs = []
            for ta in a_trades:
                for tb in b_trades:
                    # Do they overlap in time?
                    overlap_start = max(ta["entry_ts"], tb["entry_ts"])
                    overlap_end   = min(ta["exit_ts"],  tb["exit_ts"])
                    if overlap_start < overlap_end:
                        r_pairs.append((ta["r"], tb["r"]))
            if len(r_pairs) < 10:
                continue
            ra = np.array([p[0] for p in r_pairs])
            rb = np.array([p[1] for p in r_pairs])
            if np.std(ra) < 1e-9 or np.std(rb) < 1e-9: continue
            c = float(np.corrcoef(ra, rb)[0, 1])
            both_neg = sum(1 for a, b in r_pairs if a < 0 and b < 0)
            pair_r_corr.append((a_sym, b_sym, round(c,3), len(r_pairs), both_neg))

    pair_r_corr.sort(key=lambda x: -x[2])
    print(f"\n  {'Coin A':<14} {'Coin B':<14} {'R_corr':>8} {'N_pairs':>8} {'Both_loss':>10}")
    print(f"  {'-'*14} {'-'*14} {'-'*8} {'-'*8} {'-'*10}")
    for row in pair_r_corr[:25]:
        a_s, b_s, rc, np_, bl = row
        marker = " ***" if rc > 0.3 else ""
        print(f"  {a_s:<14} {b_s:<14} {rc:>8.3f} {np_:>8} {bl:>10}{marker}")

    # Save concurrent loss summary
    loss_rows = []
    for a_s, b_s, p_in, p_lose, n_both in pair_loss_rows:
        loss_rows.append({"coin_a": a_s, "coin_b": b_s,
                          "p_concurrent": round(p_in, 4),
                          "p_both_lose_given_concurrent": round(p_lose, 4),
                          "n_concurrent_bars": n_both})
    pd.DataFrame(loss_rows).to_csv(
        os.path.join(SCRIPT_DIR, "correlation_concurrent_loss.csv"), index=False)

    # Save multi-loss days
    multi_rows = []
    for day, coins in sorted(multi_loss_days.items()):
        multi_rows.append({"date": day, "n_losing_coins": len(coins),
                           "coins": ", ".join(coins),
                           "total_day_r": round(sum(daily_pnl[day].values()), 3)})
    pd.DataFrame(multi_rows).to_csv(
        os.path.join(SCRIPT_DIR, "correlation_multi_loss_days.csv"), index=False)

    print(f"\n\nSaved: correlation_concurrent_loss.csv")
    print(f"Saved: correlation_multi_loss_days.csv")
    print(f"Saved: correlation_return_matrix.csv")

    # ── Step 8: cluster memberships ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("7. COIN CLUSTERS by 15m return correlation (threshold ≥ 0.60)")
    print("=" * 70)

    # Simple union-find clustering
    parent = {s: s for s in symbols}
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(x, y): parent[find(x)] = find(y)

    for a_s, b_s, c in high_corr_pairs:
        if a_s in parent and b_s in parent:
            union(a_s, b_s)

    clusters: dict[str, list[str]] = defaultdict(list)
    for s in symbols:
        clusters[find(s)].append(s)

    print("\nClusters (coins that move together, corr ≥ 0.60):")
    for root, members in sorted(clusters.items(), key=lambda x: -len(x[1])):
        tag = "HIGHLY CORRELATED" if len(members) > 2 else ("PAIR" if len(members)==2 else "solo")
        print(f"  [{tag}] {', '.join(sorted(members))}")

    print("\n" + "=" * 70)
    print("RESEARCH COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
