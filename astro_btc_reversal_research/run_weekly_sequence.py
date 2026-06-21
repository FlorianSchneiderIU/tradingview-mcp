"""Is this week's reversal window predictable from prior weeks? (cycle / ML test)

Descriptive: persistence + transition chi-square, autocorrelation of within-week
extreme timing, inter-low spacing.
Predictive: walk-forward classifier for 'low is early (<40% of week)' using lagged
weekly features +/- price position, compared to the unconditional baseline.

Outputs (reports/): weekly_sequence.json and weekly_sequence.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

PKG_SRC = Path(__file__).resolve().parent / "src"
if str(PKG_SRC) not in sys.path:
    sys.path.insert(0, str(PKG_SRC))

from astro_reversal import (  # noqa: E402
    data, models_ml, report, walk_forward, weekly_sequence as wseq, weekly_timing as wt,
)

FEATURE_SETS = {
    "lag1": lambda cols: [c for c in cols if c.endswith("_l1")],
    "lags123": lambda cols: [c for c in cols if c[-2] == "l" and c[-1] in "123"],
    "lags+price": lambda cols: [c for c in cols if (c[-2] == "l" and c[-1] in "123")
                                or c in ("open_vs_prevlow", "open_vs_prevhigh", "ret_prev_week")],
}


def main() -> int:
    p = argparse.ArgumentParser(description="Weekly reversal-window predictability from prior weeks.")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--symbol", default=None)
    p.add_argument("--n-lags", type=int, default=3)
    p.add_argument("--n-folds", type=int, default=4)
    p.add_argument("--outdir", type=Path, default=data.REPORTS_DIR)
    args = p.parse_args()

    cfg = data.load_config(args.config)
    w = cfg["weekly_timing"]
    symbol = args.symbol or cfg["data"]["symbol"]
    start, end = cfg["data"]["start"], cfg["data"]["end"]
    holdout_start = pd.Timestamp(cfg["holdout_start"], tz="UTC")

    frame = data.load_ohlcv(symbol, w["interval"], start, end, base_interval=w["interval"])
    recs = wt.weekly_records(frame, w["retest_tol_frac"], w["move_away_frac"], w["min_week_bars"])
    seq = wseq.build_sequence(recs, args.n_lags)

    # --- Descriptive cycle structure ---
    low_dow = recs["low_dow"].to_numpy()
    low_frac = recs["low_frac"].to_numpy()
    high_frac = recs["high_frac"].to_numpy()
    low_early = (low_frac < wseq.EARLY_FRAC).astype(int)

    _, chi_e, p_e = wseq.transition_matrix(low_early[:-1], low_early[1:], 2)
    base_early = float(low_early.mean())
    prev_early = low_early[:-1].astype(bool)
    cur = low_early[1:]
    persistence = {
        "base_rate_early": base_early,
        "p_early_given_prev_early": float(cur[prev_early].mean()) if prev_early.any() else float("nan"),
        "p_early_given_prev_late": float(cur[~prev_early].mean()) if (~prev_early).any() else float("nan"),
        "chi2_p": p_e,
    }
    bucket = recs["low_frac"].map(wseq.bucket3).to_numpy()
    _, chi_b, p_b = wseq.transition_matrix(bucket[:-1], bucket[1:], 3)
    _, chi_d, p_d = wseq.transition_matrix(low_dow[:-1], low_dow[1:], 7)
    ac = {"low_frac": wseq.autocorr(low_frac, [1, 2, 3, 4]),
          "high_frac": wseq.autocorr(high_frac, [1, 2, 3, 4])}
    spacing = wseq.low_spacing_days(recs)
    spacing_stats = {
        "median_days": float(np.median(spacing)) if spacing.size else float("nan"),
        "frac_7pm1": float(np.mean(np.abs(spacing - 7) <= 1)) if spacing.size else float("nan"),
        "frac_le3": float(np.mean(spacing <= 3)) if spacing.size else float("nan"),
    }
    ac_ci = 1.96 / np.sqrt(len(low_frac))

    # --- Predictive: walk-forward 'low_early' classifier vs baseline ---
    y = seq["low_early"].to_numpy(int)
    times = pd.to_datetime(seq["week_start"], utc=True)
    dev_idx, hold_idx = walk_forward.split_dev_holdout(times, holdout_start)
    folds = walk_forward.walk_forward_folds(dev_idx, args.n_folds, embargo=args.n_lags)
    all_cols = [c for c in seq.columns if c not in
                ("week_start", "low_frac", "low_dow", "low_early", "low_bucket",
                 "high_frac", "high_early", "low_time")]

    base_rate = float(y.mean())
    baseline_logloss = float(log_loss(y, np.full(len(y), base_rate), labels=[0, 1]))
    pred_results = {}
    for fsname, picker in FEATURE_SETS.items():
        cols = picker(all_cols)
        if not cols:
            continue
        X = seq[cols]
        for model_name in ("logistic_l2", "hgb"):
            oos = np.full(len(y), np.nan)
            for tr, te in folds:
                if len(np.unique(y[tr])) < 2:
                    continue
                m = models_ml.build_model(model_name)
                m.fit(X.iloc[tr], y[tr])
                oos[te] = m.predict_proba(X.iloc[te])[:, 1]
            sc = np.isfinite(oos)
            if sc.sum() < 10 or len(np.unique(y[sc])) < 2:
                continue
            top = sc & (oos >= np.nanquantile(oos[sc], 0.8))
            pred_results[f"{fsname}|{model_name}"] = {
                "n_oos": int(sc.sum()),
                "roc_auc": float(roc_auc_score(y[sc], oos[sc])),
                "log_loss": float(log_loss(y[sc], oos[sc], labels=[0, 1])),
                "accuracy": float(accuracy_score(y[sc], (oos[sc] >= 0.5).astype(int))),
                "top20pct_early_rate": float(y[top].mean()) if top.any() else float("nan"),
            }

    results = {
        "config": {"symbol": symbol, "interval": w["interval"], "start": start, "end": end,
                   "n_lags": args.n_lags, "early_frac": wseq.EARLY_FRAC, "holdout_start": str(holdout_start)},
        "n_weeks": int(len(recs)), "n_seq": int(len(seq)),
        "persistence_low_early": persistence,
        "transition_chi2_p": {"low_early_2x2": p_e, "low_bucket_3x3": p_b, "low_dow_7x7": p_d},
        "autocorr": ac, "autocorr_95ci": float(ac_ci),
        "low_spacing": spacing_stats,
        "baseline": {"base_rate_early": base_rate, "baseline_log_loss": baseline_logloss,
                     "majority_accuracy": float(max(base_rate, 1 - base_rate))},
        "predictive": pred_results,
    }
    md = _markdown(results)
    args.outdir.mkdir(parents=True, exist_ok=True)
    report.write_json(args.outdir / "weekly_sequence.json", results)
    (args.outdir / "weekly_sequence.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def _f(v, d=3):
    if isinstance(v, float):
        return "n/a" if v != v else f"{v:.{d}f}"
    return str(v)


def _markdown(r: dict) -> str:
    pe = r["persistence_low_early"]
    b = r["baseline"]
    L = [
        "# Weekly reversal-window predictability from prior weeks",
        "",
        f"**{r['config']['symbol']}** {r['config']['interval']} | {r['n_weeks']} weeks "
        f"({r['n_seq']} usable after {r['config']['n_lags']} lags). 'early low' = first "
        f"{int(r['config']['early_frac']*100)}% of the week.",
        "",
        "## Does the low-timing persist week to week?",
        "",
        f"- base rate early: {_f(pe['base_rate_early'])}",
        f"- P(early | prev week early): {_f(pe['p_early_given_prev_early'])}",
        f"- P(early | prev week late): {_f(pe['p_early_given_prev_late'])}",
        f"- transition chi-square p: low_early {_f(r['transition_chi2_p']['low_early_2x2'], 4)} | "
        f"3-bucket {_f(r['transition_chi2_p']['low_bucket_3x3'], 4)} | "
        f"7-dow {_f(r['transition_chi2_p']['low_dow_7x7'], 4)}",
        "",
        "## Autocorrelation of within-week extreme timing "
        f"(95% noise band +/-{_f(r['autocorr_95ci'])})",
        "",
        "| lag (weeks) | low_frac ac | high_frac ac |",
        "|---|---|---|",
    ]
    for lag in (1, 2, 3, 4):
        L.append(f"| {lag} | {_f(r['autocorr']['low_frac'].get(lag))} | "
                 f"{_f(r['autocorr']['high_frac'].get(lag))} |")
    sp = r["low_spacing"]
    L += [
        "",
        f"Spacing between consecutive weekly lows: median {_f(sp['median_days'],1)}d | "
        f"share 7+/-1d {_f(sp['frac_7pm1'])} | share <=3d {_f(sp['frac_le3'])}.",
        "",
        "## Predictive test (walk-forward, OOS) vs baseline",
        "",
        f"Baseline: predict base rate -> log-loss {_f(b['baseline_log_loss'])}, "
        f"majority accuracy {_f(b['majority_accuracy'])}. A model is useful only if OOS "
        "ROC-AUC > 0.5, log-loss < baseline, and the top-20% slice's early-rate beats the base rate.",
        "",
        "| Features | Model | OOS n | ROC-AUC | log-loss | accuracy | top-20% early rate |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, m in r["predictive"].items():
        fs, mdl = key.split("|")
        L.append(f"| {fs} | {mdl} | {m['n_oos']} | {_f(m['roc_auc'])} | {_f(m['log_loss'])} | "
                 f"{_f(m['accuracy'])} | {_f(m['top20pct_early_rate'])} |")
    L += [
        "",
        "## Reading guide",
        "",
        "If chi-square p > 0.05, autocorrelations sit inside the noise band, spacing is just the trivial "
        "~7d (Monday-to-Monday) peak, and OOS ROC-AUC ~ 0.5 with log-loss ~ baseline, then prior weeks "
        "carry no extra information beyond the standing Monday/early-week prior. A genuine cycle shows up "
        "as significant persistence AND an OOS model that beats baseline.",
        "",
    ]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
