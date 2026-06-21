"""Learn the weekly-low zone in real time: rank 5m springs by reach-20R probability.

Walk-forward classifier over springs (real-time HTF/MTF features) predicting whether
a spring reaches 20R. If the top-ranked slice is net-positive at a 20R target out-of-
sample AND on the holdout, the weekly-low timing is learnable and tradeable.

Outputs (reports/): spring_model.json and spring_model.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

PKG_SRC = Path(__file__).resolve().parent / "src"
if str(PKG_SRC) not in sys.path:
    sys.path.insert(0, str(PKG_SRC))

from astro_reversal import (  # noqa: E402
    data, evaluate_ml, ltf_structure as lts, models_ml, report, spring_model, walk_forward,
)


def _slice_stats(fixed_r, label, sel):
    if sel.sum() == 0:
        return {"n": 0, "reach20_rate": float("nan"), "avg_r": float("nan"), "net_r": float("nan")}
    return {"n": int(sel.sum()),
            "reach20_rate": float(label[sel].mean()),
            "avg_r": float(fixed_r[sel].mean()),
            "net_r": float(fixed_r[sel].sum())}


def main() -> int:
    p = argparse.ArgumentParser(description="Walk-forward spring reach-20R model.")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--symbol", default=None)
    p.add_argument("--n-folds", type=int, default=4)
    p.add_argument("--top-frac", type=float, default=0.1)
    p.add_argument("--outdir", type=Path, default=data.REPORTS_DIR)
    args = p.parse_args()

    cfg = data.load_config(args.config)
    s = cfg["ltf_structure"]
    symbol = args.symbol or cfg["data"]["symbol"]
    start, end = cfg["data"]["start"], cfg["data"]["end"]
    holdout_start = pd.Timestamp(cfg["holdout_start"], tz="UTC")
    fixed_rr = 20

    daily = data.load_ohlcv(symbol, s["htf_interval"], start, end, base_interval=cfg["data"]["base_interval"])
    ltf = data.load_ohlcv(symbol, s["ltf_interval"], start, end, base_interval=s["ltf_interval"])

    spring = lts.spring_long_signals(ltf, s["spring_lookback"], s["wick_frac"], s["close_pos"])
    spring_idx = np.flatnonzero(spring)
    out = spring_model.spring_outcomes(ltf, spring_idx, s["excursion_rr"], fixed_rr,
                                       s["max_hold_bars"], s["stop_buffer_atr"], s["cost_bps_round_trip"])
    feats_all = spring_model.spring_features(ltf, daily, s["spring_lookback"])
    X = feats_all.iloc[out["idx"].to_numpy()].reset_index(drop=True)
    times = pd.to_datetime(out["time"], utc=True).reset_index(drop=True)
    fixed_r = out["fixed_r"].to_numpy(float)
    label = (out["mfe_r"].to_numpy(float) >= 20).astype(int)

    order = np.argsort(times.to_numpy())
    X, times, fixed_r, label = X.iloc[order].reset_index(drop=True), times.iloc[order].reset_index(drop=True), fixed_r[order], label[order]

    dev_idx, hold_idx = walk_forward.split_dev_holdout(times, holdout_start)
    folds = walk_forward.walk_forward_folds(dev_idx, args.n_folds, embargo=16)

    base_rate = float(label.mean())
    base_avg_r = float(np.nanmean(fixed_r))

    models = {}
    for model_name in ("hgb", "logistic_l2"):
        oos_pred = np.full(len(label), np.nan)
        for tr, te in folds:
            if label[tr].sum() < 3:
                continue
            m = models_ml.build_model(model_name)
            m.fit(X.iloc[tr], label[tr])
            oos_pred[te] = m.predict_proba(X.iloc[te])[:, 1]
        scored = np.isfinite(oos_pred)
        # Top slice of pooled OOS predictions.
        thr = np.nanquantile(oos_pred[scored], 1 - args.top_frac) if scored.any() else np.nan
        top = scored & (oos_pred >= thr)
        oos = {
            "n_scored": int(scored.sum()),
            "pr_auc": evaluate_ml.classification_metrics(label[scored], oos_pred[scored])["pr_auc"],
            "all": _slice_stats(fixed_r, label, scored),
            "top": _slice_stats(fixed_r, label, top),
        }
        # Holdout: train on all dev, threshold from dev predictions.
        hold = {"n": int(hold_idx.size)}
        if label[dev_idx].sum() >= 3 and hold_idx.size:
            m = models_ml.build_model(model_name)
            m.fit(X.iloc[dev_idx], label[dev_idx])
            dev_pred = m.predict_proba(X.iloc[dev_idx])[:, 1]
            h_pred = m.predict_proba(X.iloc[hold_idx])[:, 1]
            hthr = np.quantile(dev_pred, 1 - args.top_frac)
            htop = h_pred >= hthr
            hold["top"] = _slice_stats(fixed_r[hold_idx], label[hold_idx], htop)
            hold["all"] = _slice_stats(fixed_r[hold_idx], label[hold_idx], np.ones(hold_idx.size, bool))
        models[model_name] = {"oos": oos, "holdout": hold}

    # Interpretable importance: standardized logistic coefficients on all dev.
    importance = {}
    if label[dev_idx].sum() >= 3:
        sc = StandardScaler().fit(X.iloc[dev_idx])
        lr = LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced").fit(
            sc.transform(X.iloc[dev_idx]), label[dev_idx])
        coefs = sorted(zip(X.columns, lr.coef_[0]), key=lambda kv: abs(kv[1]), reverse=True)
        importance = {k: float(v) for k, v in coefs[:10]}

    results = {
        "config": {"symbol": symbol, "start": start, "end": end, "ltf": s["ltf_interval"],
                   "fixed_rr": fixed_rr, "top_frac": args.top_frac, "n_folds": args.n_folds,
                   "holdout_start": str(holdout_start)},
        "data": {"n_springs": int(spring_idx.size), "n_labelled": int(len(label)),
                 "reach20_base_rate": base_rate, "base_avg_r_at_20": base_avg_r,
                 "n_dev": int(dev_idx.size), "n_holdout": int(hold_idx.size)},
        "models": models,
        "logistic_importance": importance,
    }
    md = _markdown(results)
    args.outdir.mkdir(parents=True, exist_ok=True)
    report.write_json(args.outdir / "spring_model.json", results)
    (args.outdir / "spring_model.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def _f(v, d=3):
    if isinstance(v, float):
        return "n/a" if v != v else f"{v:.{d}f}"
    return str(v)


def _markdown(r: dict) -> str:
    d = r["data"]
    L = [
        "# Learnable weekly-low timing: ranking 5m springs by reach-20R",
        "",
        f"{d['n_labelled']} springs | reach-20R base rate {_f(d['reach20_base_rate'])} | "
        f"base avg R @20 {_f(d['base_avg_r_at_20'])} | dev {d['n_dev']} / holdout {d['n_holdout']}.",
        f"Top-frac selected: {r['config']['top_frac']}. A tradeable edge = top-slice avg R > 0 "
        "out-of-sample AND on holdout.",
        "",
        "| Model | OOS PR-AUC | top-slice n | top reach20 | top avg R | all avg R | holdout top avg R |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, m in r["models"].items():
        o = m["oos"]
        h = m.get("holdout", {})
        htop = h.get("top", {})
        L.append(f"| {name} | {_f(o['pr_auc'])} | {o['top']['n']} | {_f(o['top']['reach20_rate'])} | "
                 f"{_f(o['top']['avg_r'])} | {_f(o['all']['avg_r'])} | {_f(htop.get('avg_r'))} |")
    L += ["", "## Top logistic structural drivers (standardized coef, +=more likely 20R)", ""]
    for k, v in r["logistic_importance"].items():
        L.append(f"- {k}: {_f(v)}")
    L += [
        "",
        "## Reading guide",
        "",
        "base avg R @20 is what a blind spring earns (negative). If a model's **top avg R** is clearly "
        "positive OOS and on the holdout, it has learned to find the weekly-low springs in real time - "
        "a tradeable high-RR edge. If top avg R ~ base, the structure at the spring does not reveal the "
        "weekly low ahead of time, and the edge stays hindsight-only.",
        "",
    ]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
