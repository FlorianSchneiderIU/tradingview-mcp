"""Milestone 3 - ML pivot-window model (CLI).

Predicts bottom/top/pivot within N candles, comparing feature-set ablations
(price-only, calendar-only, lunar-only, astro-only, astro+cycle, astro+price,
full) and models (logistic, HistGradientBoosting) under walk-forward validation
with embargo, a final holdout, and a shifted-placebo astro control.

Outputs (reports/): ml_<target>_<tf>.json and ml_<target>_<tf>.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PKG_SRC = Path(__file__).resolve().parent / "src"
if str(PKG_SRC) not in sys.path:
    sys.path.insert(0, str(PKG_SRC))

from astro_reversal import (  # noqa: E402
    data,
    evaluate_ml,
    features_ml,
    labels_ml,
    models_ml,
    pivots,
    report,
    reuse,
    walk_forward,
)


def main() -> int:
    p = argparse.ArgumentParser(description="ML pivot-window model (Milestone 3).")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--timeframe", default="1d", choices=["1h", "4h", "1d"])
    p.add_argument("--target", default="pivot", choices=["bottom", "top", "pivot"])
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--pivot-threshold-atr", type=float, default=None)
    p.add_argument("--symbol", default=None)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--outdir", type=Path, default=data.REPORTS_DIR)
    args = p.parse_args()

    cfg = data.load_config(args.config)
    mlc = cfg["ml"]
    symbol = args.symbol or cfg["data"]["symbol"]
    start = args.start or cfg["data"]["start"]
    end = args.end or cfg["data"]["end"]
    tf = args.timeframe
    horizon = args.horizon or int(mlc["horizons"][tf])
    embargo = horizon  # purge label-overlap region around each split
    pivot_thr = args.pivot_threshold_atr if args.pivot_threshold_atr is not None else float(mlc["pivot_threshold_atr"][tf])
    holdout_start = pd.Timestamp(cfg["holdout_start"], tz="UTC")
    k_fracs = tuple(mlc["k_fracs"])

    frame = data.load_ohlcv(symbol, tf, start, end, base_interval=cfg["data"]["base_interval"])
    n = len(frame)
    times = pd.to_datetime(frame["open_time"], utc=True)

    # Labels (forward) + features (past/current + deterministic astro).
    piv = pivots.atr_directional_pivots(frame, pivot_thr)
    labels = labels_ml.forward_labels(frame, piv, horizon)
    y = labels[labels_ml.target_column(args.target)].to_numpy(dtype=int)
    features, feature_sets = features_ml.build_features(frame, tf)
    wanted = [fs for fs in mlc["feature_sets"] if fs in feature_sets]

    # Walk-forward folds over dev, plus untouched holdout.
    dev_idx, holdout_idx = walk_forward.split_dev_holdout(times, holdout_start)
    folds = walk_forward.walk_forward_folds(dev_idx, int(mlc["n_folds"]), embargo)

    results = []
    for fs in wanted:
        cols = feature_sets[fs]
        X = features.loc[:, cols]
        for model_name in mlc["models"]:
            yo, so, fold_pr = evaluate_ml.walk_forward_predictions(
                model_name, models_ml.build_model, X, y, folds)
            oos = evaluate_ml.classification_metrics(yo, so, k_fracs)
            oos["fold_pr_auc"] = [float(x) for x in fold_pr]
            yh, sh = evaluate_ml.holdout_predictions(
                model_name, models_ml.build_model, X, y, dev_idx, holdout_idx, embargo)
            hold = evaluate_ml.classification_metrics(yh, sh, k_fracs)
            results.append({"feature_set": fs, "model": model_name, "oos": oos, "holdout": hold})

    # Shifted-placebo control: shift astro/cycle columns ~37 days and re-score the
    # astro-bearing sets; real astro should beat the placebo if it carries signal.
    placebo = []
    if mlc.get("run_shifted_placebo"):
        astro_cycle_cols = feature_sets.get("astro_cycle", [])
        bpd = (n - 1) / max(1e-9, (times.iloc[-1] - times.iloc[0]).total_seconds() / 86_400.0)
        shift_bars = max(1, int(round(37.0 * bpd)))
        placebo_features = reuse.shifted_placebo(features, astro_cycle_cols, shift_bars)
        for fs in [s for s in ("astro_cycle", "astro_plus_price", "full") if s in feature_sets]:
            cols = feature_sets[fs]
            for model_name in mlc["models"]:
                yo_real, so_real, _ = evaluate_ml.walk_forward_predictions(
                    model_name, models_ml.build_model, features.loc[:, cols], y, folds)
                yo_plac, so_plac, _ = evaluate_ml.walk_forward_predictions(
                    model_name, models_ml.build_model, placebo_features.loc[:, cols], y, folds)
                placebo.append({
                    "feature_set": fs, "model": model_name,
                    "real_pr_auc": reuse.safe_average_precision(yo_real, so_real),
                    "placebo_pr_auc": reuse.safe_average_precision(yo_plac, so_plac),
                })

    out = {
        "config": {
            "symbol": symbol, "timeframe": tf, "start": start, "end": end,
            "target": args.target, "horizon": horizon, "embargo": embargo,
            "pivot_threshold_atr": pivot_thr, "n_folds": int(mlc["n_folds"]),
            "holdout_start": str(holdout_start), "feature_sets": wanted, "models": list(mlc["models"]),
        },
        "data": {"bars": n, "n_pivots": len(piv), "base_rate": float(np.mean(y)),
                 "pivot_stats": pivots.pivot_stats(frame, piv, horizon),
                 "n_dev": int(dev_idx.size), "n_holdout": int(holdout_idx.size), "n_folds_used": len(folds)},
        "results": results,
        "placebo": placebo,
    }

    outdir = args.outdir
    stem = f"ml_{args.target}_{tf}"
    report.write_json(outdir / f"{stem}.json", out)
    (outdir / f"{stem}.md").write_text(report.ml_markdown(out), encoding="utf-8")
    print(report.ml_markdown(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
