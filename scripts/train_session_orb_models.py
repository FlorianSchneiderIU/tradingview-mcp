from __future__ import annotations

import argparse
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backtest_turtle_soup import parse_utc_datetime  # noqa: E402
from scripts.experiment_session_orb import feature_columns, parse_thresholds  # noqa: E402
from scripts.experiment_session_orb_fast import apply_candidate_filter, build_contexts, to_arrays  # noqa: E402
from scripts.sweep_session_orb_top50 import (  # noqa: E402
    build_strategy_configs,
    clean_symbol,
    find_cache_path,
    prefixed_metrics,
    select_top_config_trades,
    summary_from_selected,
)
from scripts.experiment_session_orb import add_htf_context  # noqa: E402
from scripts.experiment_session_orb_fast import select_ranked_trades  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train deployable Session ORB/Judas/FVG ML rankers.")
    parser.add_argument("--symbols", nargs="+", default=["ETHUSDT", "WIFUSDT", "NEARUSDT", "ENAUSDT", "OPUSDT", "ONDOUSDT"])
    parser.add_argument("--cache-dir", type=Path, default=Path("scripts/.cache/bybit_linear"))
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--train-start", default="2022-04-20")
    parser.add_argument("--split", default="2025-04-20")
    parser.add_argument("--end", default="2026-04-20")
    parser.add_argument("--sessions", default="asia,london,ny")
    parser.add_argument("--or-minutes", default="30,60,90")
    parser.add_argument("--grid-mode", choices=["fast", "full"], default="fast")
    parser.add_argument("--family", choices=["judas"], default="judas")
    parser.add_argument("--entry-mode", choices=["fvg_retest", "level_retest", "immediate"], default="fvg_retest")
    parser.add_argument("--rank-config-scope", choices=["all", "strategy"], default="all")
    parser.add_argument("--top-train-variants", type=int, default=24)
    parser.add_argument("--min-config-train-trades", type=int, default=80)
    parser.add_argument("--min-config-oos-trades", type=int, default=15)
    parser.add_argument(
        "--candidate-filter",
        choices=["none", "judas_fvg_risk2", "judas_fvg_risk25", "asia_ny_judas_fvg_risk25"],
        default="judas_fvg_risk2",
    )
    parser.add_argument("--fee-bps-per-side", type=float, default=6.5)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--thresholds", default="0.45,0.50,0.55,0.60,0.65,0.70")
    parser.add_argument("--output-dir", type=Path, default=Path("scripts/session_orb_top50_judas_fvg_risk2_v1_models"))
    parser.add_argument("--summary-path", type=Path, default=Path("scripts/session_orb_top50_judas_fvg_risk2_v1_model_summary.csv"))
    return parser.parse_args()


def train_model(data: pd.DataFrame, split: pd.Timestamp) -> tuple[Any, list[str]]:
    train = data[data["entry_time"] < split].copy()
    if len(train) < 200 or train["win_label"].nunique() < 2:
        raise ValueError(f"not enough train rows/classes: rows={len(train)} classes={train['win_label'].nunique()}")
    cols = [column for column in feature_columns(train) if train[column].notna().any()]
    if not cols:
        raise ValueError("no non-empty feature columns")
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=400,
            max_depth=5,
            min_samples_leaf=50,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
    )
    model.fit(train[cols].astype(float), train["win_label"].astype(int))
    return model, cols


def threshold_table(scored: pd.DataFrame, split: pd.Timestamp, thresholds: list[float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        selected = select_ranked_trades(scored, threshold=threshold, split=split)
        rows.append(
            {
                "threshold": threshold,
                **summary_from_selected(selected, split, prefix="selected"),
            }
        )
    return pd.DataFrame(rows)


def train_symbol(symbol: str, args: argparse.Namespace, train_start: pd.Timestamp, split: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    symbol = clean_symbol(symbol)
    params = {
        "symbol": symbol,
        "interval": args.interval,
        "sessions": args.sessions,
        "or_minutes": args.or_minutes,
        "grid_mode": args.grid_mode,
        "family": args.family,
        "entry_mode": args.entry_mode,
        "rank_config_scope": args.rank_config_scope,
        "top_train_variants": args.top_train_variants,
        "min_config_train_trades": args.min_config_train_trades,
        "min_config_oos_trades": args.min_config_oos_trades,
        "candidate_filter": args.candidate_filter,
        "fee_bps_per_side": args.fee_bps_per_side,
        "split": split,
    }
    cache_path = find_cache_path(symbol, args.cache_dir, args.interval)
    if cache_path is None:
        raise FileNotFoundError(f"missing cache for {symbol} in {args.cache_dir}")
    raw = pd.read_pickle(cache_path)
    raw["open_time"] = pd.to_datetime(raw["open_time"], utc=True, errors="coerce")
    raw["close_time"] = pd.to_datetime(raw["close_time"], utc=True, errors="coerce")
    raw = raw[(raw["open_time"] >= train_start - pd.Timedelta(days=90)) & (raw["open_time"] < end)].copy()
    frame = add_htf_context(raw)
    frame = frame[(frame["open_time"] >= train_start) & (frame["open_time"] < end)].reset_index(drop=True)
    arrays = to_arrays(frame)
    sessions = [x.strip() for x in args.sessions.split(",") if x.strip()]
    or_minutes = [int(x.strip()) for x in args.or_minutes.split(",") if x.strip()]
    contexts = build_contexts(frame, arrays, sessions=sessions, or_minutes=or_minutes)
    selected_configs, selected_trades, diagnostics = select_top_config_trades(
        arrays,
        symbol=symbol,
        contexts=contexts,
        params=params,
    )
    candidates = apply_candidate_filter(selected_trades, args.candidate_filter)
    if candidates.empty:
        raise ValueError("no candidates after filter")
    candidates = candidates.copy()
    candidates["entry_time"] = pd.to_datetime(candidates["entry_time"], utc=True, errors="coerce")
    model, cols = train_model(candidates, split)
    scored = candidates.copy()
    scored["ml_prob"] = model.predict_proba(scored[cols].astype(float))[:, 1]
    selected = select_ranked_trades(scored, threshold=args.threshold, split=split)
    table = threshold_table(scored, split, parse_thresholds(args.thresholds))
    strategy_configs = [
        cfg
        for cfg in selected_configs
        if cfg.family == args.family and cfg.entry_mode == args.entry_mode
    ]
    if not strategy_configs:
        raise ValueError("top-ranked set contains no deployable strategy configs")
    payload = {
        "model": model,
        "feature_columns": cols,
        "threshold": float(args.threshold),
        "symbol": symbol,
        "strategy": "session_orb_judas_fvg",
        "selected_configs": [asdict(cfg) for cfg in strategy_configs],
        "config": {
            "sessions": args.sessions,
            "or_minutes": args.or_minutes,
            "grid_mode": args.grid_mode,
            "rank_config_scope": args.rank_config_scope,
            "top_train_variants": args.top_train_variants,
            "candidate_filter": args.candidate_filter,
            "fee_bps_per_side": args.fee_bps_per_side,
            "train_start": str(train_start),
            "split": str(split),
            "end": str(end),
        },
        "threshold_table": table.to_dict(orient="records"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / f"{symbol.lower()}_session_orb.joblib"
    joblib.dump(payload, model_path)
    fixed = selected[selected.get("sample", "") == "oos"].copy() if not selected.empty else selected
    row = {
        "symbol": symbol,
        "status": "trained",
        "model_path": str(model_path),
        "bars": int(len(frame)),
        "contexts": int(len(contexts)),
        "candidates": int(len(candidates)),
        "train_candidates": int((candidates["entry_time"] < split).sum()),
        "oos_candidates": int((candidates["entry_time"] >= split).sum()),
        "feature_count": int(len(cols)),
        "selected_strategy_configs": int(len(strategy_configs)),
        **diagnostics,
        **prefixed_metrics(candidates[candidates["entry_time"] < split], "candidate_train"),
        **prefixed_metrics(candidates[candidates["entry_time"] >= split], "candidate_oos"),
        **summary_from_selected(selected, split, prefix="fixed"),
    }
    row["fixed_oos_net_r_exact"] = round(float(fixed["r_multiple"].sum()), 6) if not fixed.empty else 0.0
    return row


def main() -> None:
    args = parse_args()
    train_start = pd.Timestamp(parse_utc_datetime(args.train_start))
    split = pd.Timestamp(parse_utc_datetime(args.split))
    end = pd.Timestamp(parse_utc_datetime(args.end))
    rows: list[dict[str, Any]] = []
    for symbol in args.symbols:
        symbol = clean_symbol(symbol)
        print(f"{symbol}: training Session ORB ranker ...", flush=True)
        try:
            row = train_symbol(symbol, args, train_start, split, end)
            print(
                f"  saved {row['model_path']}  "
                f"OOS={row.get('fixed_oos_trades', 0)}tr PF={row.get('fixed_oos_profit_factor', 0)} "
                f"R={row.get('fixed_oos_net_r', 0)}",
                flush=True,
            )
        except Exception as exc:
            row = {"symbol": symbol, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            print(f"  failed: {row['error']}", flush=True)
        rows.append(row)
    pd.DataFrame(rows).to_csv(args.summary_path, index=False)
    print(f"Summary -> {args.summary_path}", flush=True)


if __name__ == "__main__":
    main()
