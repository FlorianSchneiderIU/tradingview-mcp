"""
Million Moves V4.3 — Regime Config Selection via Logistic Regression
===================================================================

Idea:
  1) In each fold IS window, define 3 coarse regimes from market state:
       - trend      (high ADX)
       - chop       (low ADX, high ATR percentile)
       - sideways   (everything else)
  2) For each regime, pick the best supertrend config by IS total_R.
  3) Train a small multinomial logistic model to classify regime from
     causal bar features.
  4) In OOS, when one of the selected configs fires a signal, use the
     logistic regime prediction to choose which config to trade.

This intentionally keeps model complexity low compared to tree-based per-config
win-probability models.
"""

from __future__ import annotations

import argparse
import math
import os
import warnings
from typing import NamedTuple

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

import million_moves_v43_adaptive_lag as al

warnings.filterwarnings("ignore")


OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_OUT = os.path.join(OUT_DIR, "regime_logit_results.csv")

REGIME_NAMES = ["trend", "chop", "sideways"]
R_TREND = 0
R_CHOP = 1
R_SIDE = 2


class RegimeCuts(NamedTuple):
    adx_lo: float
    adx_hi: float
    atr_hi: float


def extract_regime_features(
    i: int,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    open_: np.ndarray,
    ema200: np.ndarray,
    atr_pctile: np.ndarray,
    vol_ratio: np.ndarray,
    rsi14: np.ndarray,
    adx14: np.ndarray,
) -> np.ndarray | None:
    """8D causal regime feature vector (no direction feature)."""
    if i < max(al.ATR_WIN + 14, al.EMA_LEN + 10):
        return None
    c = close[i]
    o = open_[i]
    h = high[i]
    l = low[i]
    em = ema200[i]
    em10 = ema200[max(0, i - 10)]
    ap = atr_pctile[i]
    vr = vol_ratio[i]
    rsi = rsi14[i]
    adx = adx14[i]
    if any(np.isnan(x) for x in [c, o, h, l, em, em10, ap, vr, rsi, adx]):
        return None

    ema_dist = (c - em) / c if c != 0 else 0.0
    ema_slope = (em - em10) / em10 if em10 != 0 else 0.0
    mom5_base = close[max(0, i - 5)]
    mom5 = (c - mom5_base) / mom5_base if mom5_base != 0 else 0.0
    rng = h - l
    body_ratio = abs(c - o) / rng if rng > 0 else 0.0

    return np.array([
        adx / 100.0,
        ap / 100.0,
        rsi / 100.0,
        ema_dist,
        ema_slope,
        vr,
        mom5,
        body_ratio,
    ], dtype=np.float64)


def compute_regime_labels(adx14: np.ndarray, atr_pctile: np.ndarray, i0: int, i1: int) -> tuple[np.ndarray, RegimeCuts]:
    """Build simple 3-class regime labels from IS quantile cuts."""
    adx_is = adx14[i0:i1]
    atr_is = atr_pctile[i0:i1]
    adx_lo = float(np.nanpercentile(adx_is, 35))
    adx_hi = float(np.nanpercentile(adx_is, 65))
    atr_hi = float(np.nanpercentile(atr_is, 60))

    labels = np.full(len(adx14), R_SIDE, dtype=np.int32)
    labels[adx14 >= adx_hi] = R_TREND
    labels[(adx14 <= adx_lo) & (atr_pctile >= atr_hi)] = R_CHOP
    return labels, RegimeCuts(adx_lo=adx_lo, adx_hi=adx_hi, atr_hi=atr_hi)


def combo_is_performance(
    i0: int,
    i1: int,
    regime_mask: np.ndarray,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr14: np.ndarray,
    atr_pctile: np.ndarray,
    vol_ratio: np.ndarray,
    combo_sigs: list[tuple[np.ndarray, np.ndarray]],
) -> list[tuple[int, float, int]]:
    """Returns [(combo_idx, total_R, n_trades)] sorted by total_R desc."""
    rows: list[tuple[int, float, int]] = []
    for c_idx, (sb, ss) in enumerate(combo_sigs):
        sb_m = sb.copy()
        ss_m = ss.copy()
        sb_m[:i0] = False
        ss_m[:i0] = False
        sb_m[i1:] = False
        ss_m[i1:] = False
        sb_m[~regime_mask] = False
        ss_m[~regime_mask] = False
        rs, _ = al.sim_trail(
            close, high, low, sb_m, ss_m, atr14, atr_pctile, vol_ratio, max_i=i1
        )
        rows.append((c_idx, float(rs.sum()) if len(rs) else 0.0, int(len(rs))))
    rows.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return rows


def select_regime_combos(
    i0: int,
    i1: int,
    regime_labels: np.ndarray,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr14: np.ndarray,
    atr_pctile: np.ndarray,
    vol_ratio: np.ndarray,
    combo_sigs: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[int, int], dict[int, list[tuple[int, float, int]]]]:
    """
    Picks one combo per regime; tries to keep picks distinct when possible.
    Returns:
      selected: regime -> combo_idx
      rankings: regime -> sorted ranking list[(combo, total_R, n)]
    """
    rankings: dict[int, list[tuple[int, float, int]]] = {}
    for r in [R_TREND, R_CHOP, R_SIDE]:
        mask = (regime_labels == r)
        rankings[r] = combo_is_performance(
            i0, i1, mask, close, high, low, atr14, atr_pctile, vol_ratio, combo_sigs
        )

    selected: dict[int, int] = {}
    used: set[int] = set()
    for r in [R_TREND, R_CHOP, R_SIDE]:
        choice = al.BASELINE_IDX
        for c_idx, _tot, n_trades in rankings[r]:
            if n_trades < 8:
                continue
            if c_idx not in used:
                choice = c_idx
                break
            if choice == al.BASELINE_IDX:
                choice = c_idx
        selected[r] = choice
        used.add(choice)

    return selected, rankings


def train_regime_logit(
    i0: int,
    i1: int,
    selected: dict[int, int],
    regime_labels: np.ndarray,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    open_: np.ndarray,
    ema200: np.ndarray,
    atr_pctile: np.ndarray,
    vol_ratio: np.ndarray,
    rsi14: np.ndarray,
    adx14: np.ndarray,
    combo_sigs: list[tuple[np.ndarray, np.ndarray]],
) -> LogisticRegression | None:
    """Train small multinomial logistic model to classify bar regime."""
    chosen = set(selected.values())
    X: list[np.ndarray] = []
    y: list[int] = []
    for i in range(i0, i1):
        # Train on bars where at least one selected combo can act.
        if not any(combo_sigs[c][0][i] or combo_sigs[c][1][i] for c in chosen):
            continue
        feat = extract_regime_features(i, close, high, low, open_, ema200, atr_pctile, vol_ratio, rsi14, adx14)
        if feat is None:
            continue
        X.append(feat)
        y.append(int(regime_labels[i]))

    if len(X) < 80:
        return None
    y_np = np.array(y, dtype=int)
    if len(np.unique(y_np)) < 2:
        return None

    model = LogisticRegression(
        max_iter=600,
        C=0.3,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(np.vstack(X), y_np)
    return model


def sim_regime_logit_oos(
    i0: int,
    i1: int,
    selected: dict[int, int],
    model: LogisticRegression | None,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    open_: np.ndarray,
    ema200: np.ndarray,
    atr14: np.ndarray,
    atr_pctile: np.ndarray,
    vol_ratio: np.ndarray,
    rsi14: np.ndarray,
    adx14: np.ndarray,
    combo_sigs: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, list[dict], np.ndarray]:
    """
    OOS sim that picks one of the selected regime combos per entry.
    Returns (r_multiples, trades, combo_usage).
    """
    r_list: list[float] = []
    trades: list[dict] = []
    usage = np.zeros(al.N_COMBOS, dtype=int)

    regime_to_combo = selected
    combo_to_regime: dict[int, int] = {}
    for r, c_idx in regime_to_combo.items():
        combo_to_regime.setdefault(c_idx, r)

    active = False
    is_long = False
    entry = sl_ = tp1 = risk = trail_sl_ = 0.0
    tp1_hit = False
    acc_r = 0.0
    entry_i = 0
    sel_combo = al.BASELINE_IDX

    for i in range(max(1, i0), i1):
        h = high[i]
        l = low[i]
        c = close[i]
        atr = atr14[i]
        ap = atr_pctile[i]

        # Manage open trade
        if active:
            sb_rev, ss_rev = combo_sigs[sel_combo]
            if is_long:
                if not tp1_hit and h >= tp1:
                    acc_r += 0.5 * al.TP1_R
                    trail_sl_ = entry
                    tp1_hit = True
                if tp1_hit:
                    if not math.isnan(atr):
                        cand = h - al.TRAIL_MULT * atr
                        if cand > trail_sl_:
                            trail_sl_ = cand
                    if l <= trail_sl_:
                        total_r = acc_r + 0.5 * max(0.0, (trail_sl_ - entry) / risk)
                        r_list.append(total_r)
                        trades.append({"entry_i": entry_i, "exit_i": i, "direction": "long", "r": total_r, "reason": "Trail", "combo": sel_combo})
                        active = False
                        continue
                else:
                    if l <= sl_:
                        r_list.append(-1.0)
                        trades.append({"entry_i": entry_i, "exit_i": i, "direction": "long", "r": -1.0, "reason": "SL", "combo": sel_combo})
                        active = False
            else:
                if not tp1_hit and l <= tp1:
                    acc_r += 0.5 * al.TP1_R
                    trail_sl_ = entry
                    tp1_hit = True
                if tp1_hit:
                    if not math.isnan(atr):
                        cand = l + al.TRAIL_MULT * atr
                        if cand < trail_sl_:
                            trail_sl_ = cand
                    if h >= trail_sl_:
                        total_r = acc_r + 0.5 * max(0.0, (entry - trail_sl_) / risk)
                        r_list.append(total_r)
                        trades.append({"entry_i": entry_i, "exit_i": i, "direction": "short", "r": total_r, "reason": "Trail", "combo": sel_combo})
                        active = False
                        continue
                else:
                    if h >= sl_:
                        r_list.append(-1.0)
                        trades.append({"entry_i": entry_i, "exit_i": i, "direction": "short", "r": -1.0, "reason": "SL", "combo": sel_combo})
                        active = False

            if active and is_long and ss_rev[i]:
                rem = 0.5 if tp1_hit else 1.0
                total_r = acc_r + rem * (c - entry) / risk
                r_list.append(total_r)
                trades.append({"entry_i": entry_i, "exit_i": i, "direction": "long", "r": total_r, "reason": "Rev", "combo": sel_combo})
                active = False
            if active and (not is_long) and sb_rev[i]:
                rem = 0.5 if tp1_hit else 1.0
                total_r = acc_r + rem * (entry - c) / risk
                r_list.append(total_r)
                trades.append({"entry_i": entry_i, "exit_i": i, "direction": "short", "r": total_r, "reason": "Rev", "combo": sel_combo})
                active = False

        # Entry selection
        if active or math.isnan(atr):
            continue
        if not (al.ATR_LO < ap < al.ATR_HI):
            continue
        if vol_ratio[i] < al.VOL_THR:
            continue

        candidates: list[tuple[int, bool]] = []
        for c_idx in set(regime_to_combo.values()):
            sb, ss = combo_sigs[c_idx]
            if sb[i]:
                candidates.append((c_idx, True))
            if ss[i]:
                candidates.append((c_idx, False))
        if not candidates:
            continue

        feat = extract_regime_features(i, close, high, low, open_, ema200, atr_pctile, vol_ratio, rsi14, adx14)
        if feat is None:
            continue

        if model is not None:
            probs = model.predict_proba(feat.reshape(1, -1))[0]
        else:
            probs = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)

        # Rank regimes by model confidence
        regime_order = list(np.argsort(probs)[::-1])

        best: tuple[int, bool] | None = None
        for r in regime_order:
            want_c = regime_to_combo[int(r)]
            hit = next((x for x in candidates if x[0] == want_c), None)
            if hit is not None:
                best = hit
                break

        if best is None:
            # Fallback: candidate whose owning regime has highest predicted prob
            cand_scores = []
            for cand in candidates:
                c_idx = cand[0]
                r = combo_to_regime.get(c_idx, R_SIDE)
                cand_scores.append((float(probs[r]), cand))
            cand_scores.sort(key=lambda x: x[0], reverse=True)
            best = cand_scores[0][1]

        sel_combo, is_bull = best
        usage[sel_combo] += 1

        if is_bull:
            sl_ = l - al.SL_MULT * atr
            risk = max(c - sl_, 1e-10)
            entry = c
            is_long = True
            tp1 = c + al.TP1_R * risk
        else:
            sl_ = h + al.SL_MULT * atr
            risk = max(sl_ - c, 1e-10)
            entry = c
            is_long = False
            tp1 = c - al.TP1_R * risk

        tp1_hit = False
        trail_sl_ = sl_
        acc_r = 0.0
        active = True
        entry_i = i

    if active:
        cl = close[i1 - 1]
        rem = 0.5 if tp1_hit else 1.0
        total_r = acc_r + rem * ((cl - entry) if is_long else (entry - cl)) / risk
        r_list.append(total_r)
        trades.append({"entry_i": entry_i, "exit_i": i1 - 1, "direction": "long" if is_long else "short", "r": total_r, "reason": "Open", "combo": sel_combo})

    return np.array(r_list, dtype=np.float64), trades, usage


def run_coin(name: str, symbol: str, since: str) -> list[dict]:
    print(f"\n{'='*60}")
    print(f"  {name}  ({symbol})")
    print(f"{'='*60}")

    try:
        df = al.fetch_ohlcv(symbol, since)
    except Exception as e:
        print(f"  [ERROR] fetch failed: {e}")
        return []

    print(f"  {len(df):,} bars  ({df.index[0].date()} -> {df.index[-1].date()})")

    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    open_ = df["open"].values.astype(float)
    volume = df["volume"].values.astype(float)

    ema200 = al.compute_ema(close, al.EMA_LEN)
    atr14 = al.compute_atr(high, low, close, 14)
    atr_pctile = al.compute_atr_pctile(atr14, al.ATR_WIN)
    vol_ratio = al.compute_vol_ratio(volume, al.VOL_WIN)
    rsi14 = al.compute_rsi(close, 14)
    adx14 = al.compute_adx(high, low, close, 14)

    print("  Pre-computing signals for all combos ...", flush=True)
    combo_sigs = al.precompute_combo_signals(close, open_, high, low, ema200)

    folds = al.generate_folds(df.index)
    print(f"  {len(folds)} walk-forward folds")

    rows: list[dict] = []
    for fold in folds:
        fid = fold["fold_id"]
        t0, t1 = fold["train_i0"], fold["train_i1"]
        o0, o1 = fold["oos_i0"], fold["oos_i1"]

        regime_labels, cuts = compute_regime_labels(adx14, atr_pctile, t0, t1)
        selected, rankings = select_regime_combos(
            t0, t1, regime_labels, close, high, low,
            atr14, atr_pctile, vol_ratio, combo_sigs,
        )

        model = train_regime_logit(
            t0, t1, selected, regime_labels,
            close, high, low, open_, ema200,
            atr_pctile, vol_ratio, rsi14, adx14,
            combo_sigs,
        )

        print(
            f"\n  Fold {fid}  IS {fold['train_start'].date()}->{fold['train_end'].date()}"
            f"  OOS {fold['oos_start'].date()}->{fold['oos_end'].date()}"
        )
        print(
            f"    Regime cuts: adx_lo={cuts.adx_lo:.1f}  adx_hi={cuts.adx_hi:.1f}  atr_hi={cuts.atr_hi:.1f}"
        )

        for r in [R_TREND, R_CHOP, R_SIDE]:
            c = selected[r]
            top = rankings[r][0]
            print(
                f"    {REGIME_NAMES[r]:<8} -> {al.COMBO_NAMES[c]}"
                f"  (best raw: {al.COMBO_NAMES[top[0]]} {top[1]:+.2f}R n={top[2]})"
            )

        # OOS baseline
        r_base, _ = al.sim_baseline_oos(
            o0, o1, close, high, low, atr14, atr_pctile, vol_ratio, combo_sigs
        )
        s_base = al.stats(r_base, "baseline")

        # OOS regime-logit
        r_logit, _, usage = sim_regime_logit_oos(
            o0, o1, selected, model,
            close, high, low, open_, ema200,
            atr14, atr_pctile, vol_ratio, rsi14, adx14, combo_sigs,
        )
        s_logit = al.stats(r_logit, "regime_logit")

        print(
            f"    BASELINE   n={s_base['n']:>3}  total_R={s_base['total_r']:>+7.2f}"
            f"  win={s_base['win_rate']:.0%}  sharpe={s_base['sharpe']:>+6.3f}"
        )
        print(
            f"    REG_LOGIT  n={s_logit['n']:>3}  total_R={s_logit['total_r']:>+7.2f}"
            f"  win={s_logit['win_rate']:.0%}  sharpe={s_logit['sharpe']:>+6.3f}"
            f"   dR={s_logit['total_r']-s_base['total_r']:>+.2f}R"
        )
        print(
            "    Usage: " + "  ".join(f"{al.COMBO_NAMES[c]}:{usage[c]}" for c in np.where(usage > 0)[0])
            if usage.sum() > 0 else "    Usage: [none]"
        )

        rows.append({
            "coin": name,
            "fold": fid,
            "oos_start": str(fold["oos_start"].date()),
            "oos_end": str(fold["oos_end"].date()),
            "adx_lo": round(cuts.adx_lo, 3),
            "adx_hi": round(cuts.adx_hi, 3),
            "atr_hi": round(cuts.atr_hi, 3),
            "sel_trend_combo": selected[R_TREND],
            "sel_chop_combo": selected[R_CHOP],
            "sel_side_combo": selected[R_SIDE],
            **{f"base_{k}": v for k, v in s_base.items() if k != "label"},
            **{f"logit_{k}": v for k, v in s_logit.items() if k != "label"},
            "delta_r": round(s_logit["total_r"] - s_base["total_r"], 2),
            **{f"usage_{al.COMBO_NAMES[c]}": int(usage[c]) for c in range(al.N_COMBOS)},
        })

    if rows:
        dfr = pd.DataFrame(rows)
        br = dfr["base_total_r"].sum()
        lr = dfr["logit_total_r"].sum()
        nb = (dfr["base_total_r"] > 0).sum()
        nl = (dfr["logit_total_r"] > 0).sum()
        print(f"\n  -- {name} SUMMARY ({len(rows)} folds) --")
        print(f"     BASELINE:  total={br:+.2f}R  pos_folds={nb}/{len(rows)}")
        print(f"     REG_LOGIT: total={lr:+.2f}R  pos_folds={nl}/{len(rows)}  dR={lr-br:+.2f}R")

    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Regime-based supertrend selector with logistic regression")
    ap.add_argument("--coins", nargs="*", default=None, help="Coin tickers (e.g. ETH BTC). Default: all")
    ap.add_argument("--since", default=al.SINCE_DATE, help="Start date YYYY-MM-DD")
    args = ap.parse_args()

    if args.coins:
        ticker_to_sym = {name: sym for name, sym in al.DEFAULT_COINS}
        coin_list = []
        for t in args.coins:
            u = t.upper()
            coin_list.append((u, ticker_to_sym.get(u, f"{u}/USDT:USDT")))
    else:
        coin_list = al.DEFAULT_COINS

    print("\nRegime Logistic Selector")
    print(f"Combos: {al.N_COMBOS} ({', '.join(al.COMBO_NAMES)})")
    print(f"Coins: {[n for n, _ in coin_list]}")
    print(f"Since: {args.since}")

    all_rows: list[dict] = []
    for name, symbol in coin_list:
        all_rows.extend(run_coin(name, symbol, args.since))

    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(CSV_OUT, index=False)

        print(f"\n{'='*60}")
        print("  OVERALL SUMMARY")
        print(f"{'='*60}")
        for coin in df["coin"].unique():
            sub = df[df["coin"] == coin]
            br = sub["base_total_r"].sum()
            lr = sub["logit_total_r"].sum()
            nb = (sub["base_total_r"] > 0).sum()
            nl = (sub["logit_total_r"] > 0).sum()
            print(
                f"  {coin:<6}  baseline={br:+7.2f}R ({nb}/{len(sub)})"
                f"  reg_logit={lr:+7.2f}R ({nl}/{len(sub)})"
                f"  dR={lr-br:+6.2f}R"
            )

        br_t = df["base_total_r"].sum()
        lr_t = df["logit_total_r"].sum()
        nb_t = (df["base_total_r"] > 0).sum()
        nl_t = (df["logit_total_r"] > 0).sum()
        print(
            f"\n  TOTAL   baseline={br_t:+8.2f}R ({nb_t}/{len(df)} folds)"
            f"  reg_logit={lr_t:+8.2f}R ({nl_t}/{len(df)} folds)"
            f"  dR={lr_t-br_t:+6.2f}R"
        )
        print(f"\n  Results CSV: {CSV_OUT}")
    else:
        print("\nNo results generated.")


if __name__ == "__main__":
    main()
