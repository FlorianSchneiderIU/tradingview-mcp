from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backtest_turtle_soup import normalize_binance_spot_symbol
from scripts.channel_state_research.backtest import strategy_metrics
from scripts.channel_state_research.data import load_base_candles, prepare_timeframe_bars
from scripts.channel_state_research.production import load_production_config
from scripts.plot_zone_channel_history import build_bfm_magic_lines, parse_timeframes
from scripts.tune_bfm_scalper import (
    ScalpSpec,
    build_candidates,
    build_execution_features,
)
from scripts.tune_bfm_support_resistance import LineBundle, project_lines_to_execution_frame
from scripts.tune_bfm_turtle_soup import (
    OPTIMIZED_BFM_TF_SETS,
    format_sets,
    label_trade,
    parse_float_list,
    parse_int_list,
    parse_str_list,
    parse_tf_sets,
    unique_preserve_order,
)

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when local env lacks sklearn
    SKLEARN_AVAILABLE = False


BFM_SCALP_TF_SETS = (
    "5m=144:96,115:77,92:61,74:49;"
    "15m=120:80,96:64,77:51,62:41;"
    f"{OPTIMIZED_BFM_TF_SETS}"
)


NUMERIC_FEATURES = [
    "channel_gap_atr",
    "channel_set",
    "reclaim_pos",
    "sweep_depth_atr",
    "entry_delay_bars",
    "lookback_bars",
    "min_risk_atr",
    "max_risk_atr",
    "stop_lookback_bars",
    "risk_atr",
    "target_rr_planned",
    "max_hold_bars",
    "close_vs_ema20_atr",
    "close_vs_ema50_atr",
    "close_vs_ema200_atr",
    "close_vs_vwap_atr",
    "ema20_delta_6_atr",
    "ema50_delta_12_atr",
    "ret_6_atr",
    "h4_close_vs_ema50_pct",
    "h4_close_vs_ema100_pct",
    "h4_close_vs_ema200_pct",
    "daily_close_vs_ema50_pct",
    "daily_close_vs_ema100_pct",
    "daily_close_vs_ema200_pct",
    "event_hour",
    "event_dayofweek",
]


CATEGORICAL_FEATURES = [
    "symbol",
    "direction",
    "entry_strategy",
    "trigger_family",
    "channel_tf",
    "trend_filter",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk-forward ML selector for lower-timeframe BFM scalp/Turtle Soup candidates."
    )
    parser.add_argument("--config", type=Path, default=Path("scripts/channel_15m_broad_v2_full5y_config.json"))
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,BNBUSDT")
    parser.add_argument("--start", default="2021-09-01")
    parser.add_argument("--end", default="2026-04-20")
    parser.add_argument("--cache-dir", type=Path, default=Path("scripts/.cache"))
    parser.add_argument("--exec-timeframe", default="5m")
    parser.add_argument("--line-timeframes", default="5m,15m,1h,4h,1d")
    parser.add_argument("--bfm-tf-sets", default=BFM_SCALP_TF_SETS)
    parser.add_argument("--bfm-invalidation", choices=["wick", "close", "none"], default="wick")
    parser.add_argument("--bfm-max-extension-bars", type=int, default=300)
    parser.add_argument("--directions", default="long,short")
    parser.add_argument("--entry-strategy", default="hybrid_reclaim")
    parser.add_argument("--lookbacks", default="12")
    parser.add_argument("--channel-gap-atrs", default="1.0")
    parser.add_argument("--min-reclaim-positions", default="0.5")
    parser.add_argument("--target-rrs", default="1.5,2.0")
    parser.add_argument("--stop-buffer-atrs", default="0.12")
    parser.add_argument("--min-sweep-depth-atrs", default="0.0")
    parser.add_argument("--min-risk-atrs", default="2.5,3.5")
    parser.add_argument("--max-risk-atrs", default="6.0")
    parser.add_argument("--stop-lookbacks", default="12,24", help="5m bars for known structure stop; 3=15m, 12=1h.")
    parser.add_argument("--max-hold-bars", default="48,96")
    parser.add_argument("--trend-filter", default="none")
    parser.add_argument("--fee-bps-side", type=float, default=None)
    parser.add_argument("--slippage-bps-side", type=float, default=None)
    parser.add_argument("--risk-fraction", type=float, default=None)
    parser.add_argument("--model", choices=["rf", "hgb"], default="rf")
    parser.add_argument("--label-min-r", type=float, default=0.0, help="Train label is r_multiple_net >= this value.")
    parser.add_argument("--train-months", type=int, default=18)
    parser.add_argument("--val-months", type=int, default=6)
    parser.add_argument("--test-months", type=int, default=3)
    parser.add_argument("--embargo-hours", type=float, default=48.0)
    parser.add_argument("--thresholds", default="0.50,0.55,0.60,0.65,0.70,0.75")
    parser.add_argument("--min-val-trades", type=int, default=15)
    parser.add_argument("--schedule-mode", choices=["per_symbol", "global"], default="per_symbol")
    parser.add_argument("--write-dataset", action="store_true")
    parser.add_argument("--reuse-dataset", type=Path, default=None)
    parser.add_argument("--output-prefix", type=Path, default=Path("scripts/bfm_scalper_ml_majors5"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn is required. Run with the repo virtualenv: .\\.venv\\Scripts\\python.exe")

    config = load_production_config(args.config)
    fee_bps_side = float(config.fee_bps_side if args.fee_bps_side is None else args.fee_bps_side)
    slippage_bps_side = float(config.slippage_bps_side if args.slippage_bps_side is None else args.slippage_bps_side)
    risk_fraction = float(config.risk.risk_fraction if args.risk_fraction is None else args.risk_fraction)

    if args.reuse_dataset:
        dataset = pd.read_csv(args.reuse_dataset, parse_dates=["event_time", "entry_time", "exit_time", "label_end_time"])
        print(f"Loaded reusable dataset {args.reuse_dataset}: {len(dataset):,} labeled candidates")
    else:
        dataset = build_labeled_dataset(
            args,
            fee_bps_side=fee_bps_side,
            slippage_bps_side=slippage_bps_side,
            risk_fraction=risk_fraction,
        )

    if dataset.empty:
        raise RuntimeError("No labeled candidates were generated.")

    dataset = add_ml_columns(dataset)
    if float(args.label_min_r) != 0.0:
        dataset["label"] = (pd.to_numeric(dataset["r_multiple_net"], errors="coerce") >= float(args.label_min_r)).astype(int)
    feature_columns = [column for column in [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES] if column in dataset.columns]
    print(
        f"Dataset: {len(dataset):,} candidates | {dataset['symbol'].nunique()} symbols | "
        f"positive rate {dataset['label'].mean():.2%} | features {len(feature_columns)}"
    )

    results = run_walk_forward(
        dataset,
        feature_columns=feature_columns,
        args=args,
        risk_fraction=risk_fraction,
    )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_prefix.with_name(f"{args.output_prefix.name}_folds.csv")
    thresholds_path = args.output_prefix.with_name(f"{args.output_prefix.name}_thresholds.csv")
    scored_path = args.output_prefix.with_name(f"{args.output_prefix.name}_scored.csv")
    trades_path = args.output_prefix.with_name(f"{args.output_prefix.name}_selected_trades.csv")
    dataset_path = args.output_prefix.with_name(f"{args.output_prefix.name}_dataset.csv")
    config_path = args.output_prefix.with_name(f"{args.output_prefix.name}_config.json")

    results["folds"].to_csv(summary_path, index=False)
    results["thresholds"].to_csv(thresholds_path, index=False)
    results["scored"].to_csv(scored_path, index=False)
    results["selected_trades"].to_csv(trades_path, index=False)
    if args.write_dataset:
        dataset.to_csv(dataset_path, index=False)

    payload = {
        "symbols": parse_str_list(args.symbols),
        "start": args.start,
        "end": args.end,
        "model": args.model,
        "feature_columns": feature_columns,
        "numeric_features": [column for column in NUMERIC_FEATURES if column in feature_columns],
        "categorical_features": [column for column in CATEGORICAL_FEATURES if column in feature_columns],
        "candidate_count": int(len(dataset)),
        "positive_rate": float(dataset["label"].mean()),
        "bfm_tf_sets": args.bfm_tf_sets,
        "stop_note": "stop_anchor_price is the known rolling wick extreme over stop_lookback_bars 5m bars including the penetration bar.",
    }
    config_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")

    display = results["folds"]
    if not display.empty:
        cols = [
            "fold",
            "test_start",
            "test_end",
            "threshold",
            "test_rows",
            "selected_trades",
            "selected_total_return",
            "selected_net_r",
            "selected_profit_factor",
            "selected_max_drawdown",
            "auc",
        ]
        print("\nWalk-forward folds")
        print(display[[column for column in cols if column in display.columns]].to_string(index=False))
    aggregate = aggregate_metrics(results["selected_trades"])
    print("\nAggregate selected")
    print(pd.DataFrame([aggregate]).to_string(index=False))
    print(f"\nWrote {summary_path}")
    print(f"Wrote {thresholds_path}")
    print(f"Wrote {scored_path}")
    print(f"Wrote {trades_path}")
    if args.write_dataset:
        print(f"Wrote {dataset_path}")
    print(f"Wrote {config_path}")


def build_labeled_dataset(
    args: argparse.Namespace,
    *,
    fee_bps_side: float,
    slippage_bps_side: float,
    risk_fraction: float,
) -> pd.DataFrame:
    line_timeframes = parse_timeframes(args.line_timeframes, "1h")
    bfm_sets_by_tf = parse_tf_sets(args.bfm_tf_sets, line_timeframes)
    allowed_directions = set(parse_str_list(args.directions))
    specs = build_specs(args)
    frames: list[pd.DataFrame] = []
    for symbol in parse_str_list(args.symbols):
        normalized = normalize_binance_spot_symbol(symbol)
        print(f"\nBuilding {normalized} candidates")
        base = load_base_candles(
            normalized,
            args.start,
            args.end,
            cache_dir=args.cache_dir,
            interval="5m",
        )
        all_timeframes = unique_preserve_order([args.exec_timeframe, *line_timeframes])
        bars_by_tf = {
            timeframe: prepare_timeframe_bars(base, timeframe, atr_length=14)
            for timeframe in all_timeframes
        }
        exec_bars = bars_by_tf[args.exec_timeframe].reset_index(drop=True)
        bundles: dict[str, LineBundle] = {}
        for timeframe in line_timeframes:
            lines, pivots = build_bfm_magic_lines(
                bars_by_tf[timeframe],
                bfm_sets_by_tf[timeframe],
                invalidation=args.bfm_invalidation,
                max_extension_bars=args.bfm_max_extension_bars,
            )
            bundles[timeframe] = LineBundle(
                timeframe=timeframe,
                scale=1.0,
                sets=tuple(bfm_sets_by_tf[timeframe]),
                bars=bars_by_tf[timeframe],
                lines=tuple(lines),
                pivots_count=len(pivots),
            )
            print(f"  {timeframe}: {len(pivots):,} pivots, {len(lines):,} lines, sets {format_sets(bfm_sets_by_tf[timeframe])}")

        projection = project_lines_to_execution_frame(exec_bars, bundles)
        features = build_execution_features(exec_bars, bars_by_tf=bars_by_tf)
        candidate_frames: list[pd.DataFrame] = []
        for spec_index, spec in enumerate(specs, start=1):
            candidates = build_candidates(
                exec_bars=exec_bars,
                projection=projection,
                features=features,
                line_timeframes=line_timeframes,
                spec=spec,
                symbol=normalized,
                allowed_directions=allowed_directions,
            )
            if candidates.empty:
                continue
            candidates = candidates.copy()
            candidates["candidate_spec_id"] = spec_index
            candidates["spec_channel_gap_atr"] = spec.channel_gap_atr
            candidates["spec_target_rr"] = spec.target_rr
            candidates["spec_min_risk_atr"] = spec.min_risk_atr
            candidate_frames.append(candidates)
        symbol_candidates = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
        labeled = label_all_candidates(
            symbol_candidates,
            exec_bars,
            fee_bps_side=fee_bps_side,
            slippage_bps_side=slippage_bps_side,
            risk_fraction=risk_fraction,
        )
        print(f"  {len(symbol_candidates):,} raw candidates -> {len(labeled):,} labeled, positive {labeled['label'].mean():.2%}" if not labeled.empty else "  no labeled candidates")
        if not labeled.empty:
            frames.append(labeled)
    return pd.concat(frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True) if frames else pd.DataFrame()


def build_specs(args: argparse.Namespace) -> list[ScalpSpec]:
    specs: list[ScalpSpec] = []
    for lookback in parse_int_list(args.lookbacks):
        for gap in parse_float_list(args.channel_gap_atrs):
            for reclaim in parse_float_list(args.min_reclaim_positions):
                for rr in parse_float_list(args.target_rrs):
                    for stop_buffer in parse_float_list(args.stop_buffer_atrs):
                        for depth in parse_float_list(args.min_sweep_depth_atrs):
                            for min_risk in parse_float_list(args.min_risk_atrs):
                                for max_risk in parse_float_list(args.max_risk_atrs):
                                    for stop_lookback in parse_int_list(args.stop_lookbacks):
                                        for hold in parse_int_list(args.max_hold_bars):
                                            if min_risk > max_risk:
                                                continue
                                            specs.append(
                                                ScalpSpec(
                                                    entry_strategy=args.entry_strategy,
                                                    lookback_bars=int(lookback),
                                                    channel_gap_atr=float(gap),
                                                    min_reclaim_pos=float(reclaim),
                                                    target_rr=float(rr),
                                                    stop_buffer_atr=float(stop_buffer),
                                                    min_sweep_depth_atr=float(depth),
                                                    min_risk_atr=float(min_risk),
                                                    max_risk_atr=float(max_risk),
                                                    stop_lookback_bars=int(stop_lookback),
                                                    max_hold_bars=int(hold),
                                                    trend_filter=args.trend_filter,
                                                )
                                            )
    return specs


def label_all_candidates(
    candidates: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    fee_bps_side: float,
    slippage_bps_side: float,
    risk_fraction: float,
) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    opens = pd.to_numeric(bars["open"], errors="coerce").to_numpy(dtype=float)
    highs = pd.to_numeric(bars["high"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float)
    closes = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float)
    close_times = pd.to_datetime(bars["close_time"], utc=True, errors="coerce").to_list()
    rows: list[dict[str, Any]] = []
    for _, candidate in candidates.iterrows():
        outcome = label_trade(
            direction=str(candidate["direction"]),
            entry_index=int(candidate["entry_index"]),
            entry_price=float(candidate["entry_price"]),
            stop_price=float(candidate["stop_price"]),
            target_price=float(candidate["target_price"]),
            max_hold_bars=int(candidate["max_hold_bars"]),
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
            close_times=close_times,
        )
        if outcome is None:
            continue
        risk = float(candidate["risk_abs"])
        if not math.isfinite(risk) or risk <= 0.0:
            continue
        cost_r = ((2.0 * fee_bps_side) + (2.0 * slippage_bps_side)) / 10_000.0 * float(candidate["entry_price"]) / risk
        net_r = float(outcome["r_multiple_gross"]) - cost_r
        row = candidate.to_dict()
        row.update(outcome)
        row["cost_r"] = float(cost_r)
        row["r_multiple_net"] = float(net_r)
        row["return_pct"] = float(risk_fraction * net_r)
        row["label"] = int(net_r > 0.0)
        row["label_end_time"] = pd.Timestamp(outcome["exit_time"]).tz_convert("UTC")
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame()


def add_ml_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["entry_time"] = pd.to_datetime(out["entry_time"], utc=True, errors="coerce")
    out["event_time"] = pd.to_datetime(out["event_time"], utc=True, errors="coerce")
    out["exit_time"] = pd.to_datetime(out["exit_time"], utc=True, errors="coerce")
    out["label_end_time"] = pd.to_datetime(out["label_end_time"], utc=True, errors="coerce")
    out["event_hour"] = out["event_time"].dt.hour.astype(float)
    out["event_dayofweek"] = out["event_time"].dt.dayofweek.astype(float)
    if "trigger_family" not in out.columns:
        out["trigger_family"] = out["entry_strategy"]
    for column in CATEGORICAL_FEATURES:
        if column in out.columns:
            out[column] = out[column].fillna("missing").astype(str)
    for column in NUMERIC_FEATURES:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out["label"] = out["label"].astype(int)
    return out.sort_values("entry_time").reset_index(drop=True)


def run_walk_forward(
    dataset: pd.DataFrame,
    *,
    feature_columns: list[str],
    args: argparse.Namespace,
    risk_fraction: float,
) -> dict[str, pd.DataFrame]:
    thresholds = parse_float_list(args.thresholds)
    rows: list[dict[str, Any]] = []
    threshold_rows: list[pd.DataFrame] = []
    scored_frames: list[pd.DataFrame] = []
    selected_frames: list[pd.DataFrame] = []

    first = pd.Timestamp(args.start, tz="UTC")
    last = pd.Timestamp(args.end, tz="UTC")
    train_start = first
    train_end = train_start + pd.DateOffset(months=args.train_months)
    val_end = train_end + pd.DateOffset(months=args.val_months)
    test_end = val_end + pd.DateOffset(months=args.test_months)
    fold = 1
    embargo = pd.Timedelta(hours=float(args.embargo_hours))

    while val_end < last:
        capped_test_end = min(test_end, last)
        train = dataset[
            (dataset["entry_time"] >= train_start)
            & (dataset["entry_time"] < train_end)
            & (dataset["label_end_time"] < train_end - embargo)
        ].copy()
        validation = dataset[(dataset["entry_time"] >= train_end) & (dataset["entry_time"] < val_end)].copy()
        test = dataset[(dataset["entry_time"] >= val_end) & (dataset["entry_time"] < capped_test_end)].copy()
        if train.empty or validation.empty or test.empty or train["label"].nunique() < 2:
            train_end += pd.DateOffset(months=args.test_months)
            val_end += pd.DateOffset(months=args.test_months)
            test_end += pd.DateOffset(months=args.test_months)
            fold += 1
            continue

        model = build_model(args.model, feature_columns)
        model.fit(train[feature_columns], train["label"].astype(int))
        validation = score_frame(model, validation, feature_columns)
        test = score_frame(model, test, feature_columns)
        best_threshold, table = tune_threshold(
            validation,
            thresholds,
            min_trades=int(args.min_val_trades),
            schedule_mode=args.schedule_mode,
        )
        table["fold"] = fold
        table["train_end"] = train_end
        table["val_end"] = val_end
        threshold_rows.append(table)

        selected = schedule_selected(test[test["ml_prob"] >= best_threshold], schedule_mode=args.schedule_mode).copy()
        selected["fold"] = fold
        selected["selected_threshold"] = best_threshold
        test["fold"] = fold
        test["selected_threshold"] = best_threshold
        scored_frames.append(test)
        selected_frames.append(selected)

        metrics = strategy_metrics(selected)
        all_metrics = strategy_metrics(schedule_selected(test, schedule_mode=args.schedule_mode))
        auc = safe_auc(test["label"], test["ml_prob"])
        rows.append(
            {
                "fold": fold,
                "train_start": train_start,
                "train_end": train_end,
                "val_end": val_end,
                "test_start": val_end,
                "test_end": capped_test_end,
                "threshold": best_threshold,
                "train_rows": len(train),
                "val_rows": len(validation),
                "test_rows": len(test),
                "selected_trades": metrics["trades"],
                "selected_total_return": metrics["total_return"],
                "selected_net_r": metrics["net_r"],
                "selected_profit_factor": metrics["profit_factor"],
                "selected_max_drawdown": metrics["max_drawdown"],
                "selected_hit_rate": metrics["hit_rate"],
                "all_scheduled_trades": all_metrics["trades"],
                "all_scheduled_net_r": all_metrics["net_r"],
                "auc": auc,
                "positive_rate_test": float(test["label"].mean()) if len(test) else np.nan,
            }
        )

        train_end += pd.DateOffset(months=args.test_months)
        val_end += pd.DateOffset(months=args.test_months)
        test_end += pd.DateOffset(months=args.test_months)
        fold += 1

    return {
        "folds": pd.DataFrame(rows),
        "thresholds": pd.concat(threshold_rows, ignore_index=True) if threshold_rows else pd.DataFrame(),
        "scored": pd.concat(scored_frames, ignore_index=True) if scored_frames else pd.DataFrame(),
        "selected_trades": pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame(),
    }


def build_model(model_name: str, feature_columns: list[str]) -> Pipeline:
    numeric = [column for column in feature_columns if column in NUMERIC_FEATURES]
    categorical = [column for column in feature_columns if column in CATEGORICAL_FEATURES]
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - older sklearn
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    preprocessor = ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", encoder)]), categorical),
        ],
        remainder="drop",
    )
    if model_name == "hgb":
        estimator = HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.035,
            max_leaf_nodes=12,
            min_samples_leaf=25,
            l2_regularization=0.5,
            class_weight="balanced",
            random_state=17,
        )
    else:
        estimator = RandomForestClassifier(
            n_estimators=500,
            max_depth=7,
            min_samples_leaf=12,
            class_weight="balanced_subsample",
            random_state=17,
            n_jobs=1,
        )
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def score_frame(model: Pipeline, frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    out["ml_prob"] = model.predict_proba(out[feature_columns])[:, 1]
    return out


def tune_threshold(
    validation: pd.DataFrame,
    thresholds: list[float],
    *,
    min_trades: int,
    schedule_mode: str,
) -> tuple[float, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    best_threshold = thresholds[0]
    best_score = -math.inf
    for threshold in thresholds:
        selected = schedule_selected(validation[validation["ml_prob"] >= threshold], schedule_mode=schedule_mode)
        metrics = strategy_metrics(selected)
        eligible = metrics["trades"] >= min_trades
        pf = float(metrics["profit_factor"])
        if not math.isfinite(pf):
            pf = 5.0
        score = (
            float(metrics["net_r"])
            - 12.0 * abs(float(metrics["max_drawdown"]))
            + 0.8 * min(pf, 5.0)
            + 0.02 * float(metrics["trades"])
        )
        rows.append({"threshold": threshold, "eligible": eligible, "score": score if eligible else -10_000.0, **metrics})
        if eligible and score > best_score:
            best_score = score
            best_threshold = threshold
    table = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    if best_score == -math.inf and not table.empty:
        best_threshold = float(table.iloc[0]["threshold"])
    return float(best_threshold), table


def schedule_selected(frame: pd.DataFrame, *, schedule_mode: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    sort_columns = ["entry_time", "ml_prob", "risk_atr"]
    ascending = [True, False, True]
    ordered = frame.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    active_until_by_key: dict[str, pd.Timestamp] = {}
    for _, row in ordered.iterrows():
        key = "portfolio" if schedule_mode == "global" else str(row["symbol"])
        entry_time = pd.Timestamp(row["entry_time"]).tz_convert("UTC")
        active_until = active_until_by_key.get(key)
        if active_until is not None and entry_time <= active_until:
            continue
        rows.append(row.to_dict())
        active_until_by_key[key] = pd.Timestamp(row["exit_time"]).tz_convert("UTC")
    return pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame(columns=frame.columns)


def aggregate_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    metrics = strategy_metrics(frame) if not frame.empty else strategy_metrics(pd.DataFrame())
    out: dict[str, Any] = {f"selected_{key}": value for key, value in metrics.items()}
    out["symbols"] = ",".join(sorted(frame["symbol"].astype(str).unique())) if not frame.empty else ""
    out["trades_per_week"] = trades_per_week(frame)
    return out


def trades_per_week(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    start = pd.Timestamp(frame["entry_time"].min())
    end = pd.Timestamp(frame["entry_time"].max())
    weeks = max((end - start).total_seconds() / (7.0 * 24.0 * 3600.0), 1e-9)
    return float(len(frame) / weeks)


def safe_auc(label: pd.Series, prob: pd.Series) -> float:
    try:
        if label.nunique() < 2:
            return np.nan
        return float(roc_auc_score(label.astype(int), prob.astype(float)))
    except Exception:
        return np.nan


if __name__ == "__main__":
    main()
