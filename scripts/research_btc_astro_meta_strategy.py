from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research_btc_astro_cycle_timing import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    SECONDS_PER_DAY,
    SplitSpec,
    add_ltf_indicators,
    build_feature_matrix,
    clean_features,
    evaluate_feature_set,
    json_default,
    load_bybit_cached,
    make_forward_labels,
    parse_utc_datetime,
    prepare_frame,
    select_threshold_from_validation,
    shifted_placebo,
    simulate_sweep_trade,
    split_masks,
    trade_summary,
    zigzag_pivots,
)

warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass(frozen=True)
class MetaSpec:
    name: str
    model: str
    feature_set: str
    rr: float
    threshold: float
    coverage: float
    validation: dict[str, Any]
    test: dict[str, Any]


def htf_market_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    atr = frame["atr"].replace(0.0, np.nan)
    close = frame["close"].astype(float)
    open_ = frame["open"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    ema = frame["ema"].astype(float)
    out["mkt_range_atr"] = (high - low) / atr
    out["mkt_body_atr"] = (close - open_).abs() / atr
    out["mkt_body_signed_atr"] = (close - open_) / atr
    out["mkt_ret1_atr"] = (close - close.shift(1)) / atr
    out["mkt_ret4_atr"] = (close - close.shift(4)) / atr
    out["mkt_ret12_atr"] = (close - close.shift(12)) / atr
    out["mkt_dist_ema_atr"] = (close - ema) / atr
    out["mkt_ema_slope_atr"] = frame["ema_slope_atr"]
    out["mkt_atr_ratio"] = frame["atr_ratio"]
    out["mkt_rsi"] = frame["rsi"] / 100.0
    out["mkt_volume_ratio"] = frame["volume_ratio"]
    # Timing scores are evaluated at the HTF open. Price-derived state must come
    # from the previous closed HTF candle to avoid intrabar lookahead.
    return clean_features(out.shift(1))


def fit_scores(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    groups: dict[str, list[str]],
    masks: dict[str, np.ndarray],
    pivots: list[Any],
    horizon_bars: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    interval_seconds = (
        pd.Timestamp(labels["open_time"].iloc[1]) - pd.Timestamp(labels["open_time"].iloc[0])
    ).total_seconds()
    shift_bars = max(1, int(round(37.0 * SECONDS_PER_DAY / interval_seconds)))
    placebo = shifted_placebo(features, groups["astro_cycle"], shift_bars)

    score_frame = labels[["open_time", "close_time", "y_any", "y_high", "y_low"]].copy()
    metrics: dict[str, Any] = {}
    specs = {
        "calendar": (features, groups["calendar"]),
        "real": (features, groups["all"]),
        "placebo": (placebo, groups["all"]),
    }
    for prefix, (feature_frame, columns) in specs.items():
        metrics[prefix] = {}
        for target in ["y_any", "y_high", "y_low"]:
            target_name = target.removeprefix("y_")
            result, scores = evaluate_feature_set(
                feature_frame,
                labels,
                target,
                columns,
                masks,
                pivots,
                horizon_bars,
                [0.005, 0.01, 0.02, 0.05],
            )
            score_frame[f"{prefix}_{target_name}_score"] = scores
            metrics[prefix][target_name] = result
    return score_frame, metrics


def add_candidate_features(row: pd.Series, direction: str, lookback: int, prev_level: float) -> dict[str, float]:
    atr = float(row["atr"])
    high = float(row["high"])
    low = float(row["low"])
    open_ = float(row["open"])
    close = float(row["close"])
    rng = max(high - low, 1e-12)
    if direction == "long":
        sweep_depth = (prev_level - low) / atr
        rejection = (close - low) / rng
        reclaim = (close - prev_level) / atr
        body_dir = (close - open_) / atr
        close_from_extreme = (close - low) / atr
        real_dir = float(row["real_low_score"])
        real_opp = float(row["real_high_score"])
        calendar_dir = float(row["calendar_low_score"])
        placebo_dir = float(row["placebo_low_score"])
        placebo_opp = float(row["placebo_high_score"])
    else:
        sweep_depth = (high - prev_level) / atr
        rejection = (high - close) / rng
        reclaim = (prev_level - close) / atr
        body_dir = (open_ - close) / atr
        close_from_extreme = (high - close) / atr
        real_dir = float(row["real_high_score"])
        real_opp = float(row["real_low_score"])
        calendar_dir = float(row["calendar_high_score"])
        placebo_dir = float(row["placebo_high_score"])
        placebo_opp = float(row["placebo_low_score"])

    ts = pd.Timestamp(row["close_time"]).tz_convert("UTC")
    hour = ts.hour + ts.minute / 60.0
    dow = ts.dayofweek
    return {
        "lookback": float(lookback),
        "is_long": 1.0 if direction == "long" else 0.0,
        "sweep_depth_atr": float(sweep_depth),
        "rejection_frac": float(rejection),
        "reclaim_atr": float(reclaim),
        "body_dir_atr": float(body_dir),
        "body_abs_atr": float(abs(close - open_) / atr),
        "range_atr": float((high - low) / atr),
        "close_from_extreme_atr": float(close_from_extreme),
        "ltf_rsi": float(row["rsi"]) / 100.0,
        "ltf_ema_slope_atr": float(row["ema_slope_atr"]),
        "ltf_atr_ratio": float(row["atr_ratio"]),
        "ltf_volume_ratio": float(row["volume_ratio"]),
        "ltf_dist_ema_atr": float((close - float(row["ema"])) / atr),
        "tod_sin": float(np.sin(2.0 * np.pi * hour / 24.0)),
        "tod_cos": float(np.cos(2.0 * np.pi * hour / 24.0)),
        "dow_sin": float(np.sin(2.0 * np.pi * dow / 7.0)),
        "dow_cos": float(np.cos(2.0 * np.pi * dow / 7.0)),
        "real_any_score": float(row["real_any_score"]),
        "real_dir_score": real_dir,
        "real_opp_score": real_opp,
        "real_edge_score": real_dir - real_opp,
        "calendar_any_score": float(row["calendar_any_score"]),
        "calendar_dir_score": calendar_dir,
        "placebo_any_score": float(row["placebo_any_score"]),
        "placebo_dir_score": placebo_dir,
        "placebo_opp_score": placebo_opp,
        "placebo_edge_score": placebo_dir - placebo_opp,
    }


def build_candidates(
    ltf: pd.DataFrame,
    score_frame: pd.DataFrame,
    split_spec: SplitSpec,
    lookbacks: list[int],
    rr_values: list[float],
    max_hold_bars: int,
    stop_buffer_atr: float,
    cost_bps_round_trip: float,
) -> pd.DataFrame:
    ltf = add_ltf_indicators(ltf)
    for lookback in lookbacks:
        ltf[f"prev_high_{lookback}"] = ltf["high"].shift(1).rolling(lookback, min_periods=lookback).max()
        ltf[f"prev_low_{lookback}"] = ltf["low"].shift(1).rolling(lookback, min_periods=lookback).min()

    aligned = pd.merge_asof(
        ltf.sort_values("open_time"),
        score_frame.sort_values("open_time"),
        on="open_time",
        direction="backward",
        suffixes=("", "_htf"),
    )
    rows: list[dict[str, Any]] = []
    max_lookback = max(lookbacks)
    for idx in range(max_lookback + 1, len(aligned) - max_hold_bars - 2):
        row = aligned.iloc[idx]
        if not math.isfinite(float(row["atr"])) or float(row["atr"]) <= 0:
            continue
        signal_time = pd.Timestamp(row["close_time"]).tz_convert("UTC")
        if signal_time <= split_spec.train_end:
            split = "train"
        elif signal_time <= split_spec.validation_end:
            split = "validation"
        else:
            split = "test"
        for lookback in lookbacks:
            prev_high = float(row[f"prev_high_{lookback}"])
            prev_low = float(row[f"prev_low_{lookback}"])
            if not math.isfinite(prev_high) or not math.isfinite(prev_low):
                continue
            high_sweep = float(row["high"]) > prev_high and float(row["close"]) < prev_high
            low_sweep = float(row["low"]) < prev_low and float(row["close"]) > prev_low
            if high_sweep == low_sweep:
                continue
            direction = "short" if high_sweep else "long"
            prev_level = prev_high if high_sweep else prev_low
            features = add_candidate_features(row, direction, lookback, prev_level)
            candidate: dict[str, Any] = {
                "signal_idx": int(idx),
                "signal_time": signal_time,
                "split": split,
                "direction": direction,
                **features,
            }
            for rr in rr_values:
                trade = simulate_sweep_trade(
                    aligned,
                    idx,
                    direction,
                    rr,
                    max_hold_bars,
                    stop_buffer_atr,
                    cost_bps_round_trip,
                )
                key = f"{rr:g}"
                if trade is None:
                    candidate[f"result_r_{key}"] = np.nan
                    candidate[f"exit_idx_{key}"] = np.nan
                    candidate[f"exit_reason_{key}"] = ""
                    candidate[f"risk_pct_{key}"] = np.nan
                    candidate[f"mfe_r_{key}"] = np.nan
                    continue
                candidate[f"result_r_{key}"] = float(trade["result_r"])
                candidate[f"exit_idx_{key}"] = int(trade["exit_idx"])
                candidate[f"exit_reason_{key}"] = str(trade["exit_reason"])
                candidate[f"risk_pct_{key}"] = float(trade["risk_pct"])
                candidate[f"mfe_r_{key}"] = float(trade["mfe_r"])
                candidate[f"cost_r_{key}"] = (cost_bps_round_trip / 10_000.0) / max(float(trade["risk_pct"]), 1e-12)
            rows.append(candidate)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    numeric = [column for column in out.columns if column not in {"signal_time", "split", "direction"} and not column.startswith("exit_reason")]
    out[numeric] = out[numeric].replace([np.inf, -np.inf], np.nan)
    medians = out.loc[out["split"] == "train", numeric].median(numeric_only=True)
    out[numeric] = out[numeric].fillna(medians).fillna(0.0)
    return out.reset_index(drop=True)


def candidate_summary(frame: pd.DataFrame, rr: float, score_col: str | None = None, threshold: float | None = None) -> dict[str, Any]:
    key = f"{rr:g}"
    selected = frame
    if score_col is not None and threshold is not None:
        selected = selected[selected[score_col] >= threshold]
    selected = selected.sort_values(["signal_idx", score_col] if score_col else ["signal_idx"], ascending=[True, False] if score_col else True)
    trades: list[dict[str, Any]] = []
    blocked_until = -1
    for row in selected.itertuples(index=False):
        signal_idx = int(getattr(row, "signal_idx"))
        if signal_idx <= blocked_until:
            continue
        result_r = float(getattr(row, f"result_r_{key}"))
        exit_idx = int(getattr(row, f"exit_idx_{key}"))
        if not math.isfinite(result_r) or not math.isfinite(exit_idx):
            continue
        trades.append(
            {
                "result_r": result_r,
                "exit_reason": str(getattr(row, f"exit_reason_{key}")),
                "mfe_r": float(getattr(row, f"mfe_r_{key}")),
            }
        )
        blocked_until = max(blocked_until, exit_idx)
    return trade_summary(trades)


def make_meta_model(kind: str) -> Any:
    if kind == "logit":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.2,
                        class_weight="balanced",
                        max_iter=2_000,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=160,
            max_leaf_nodes=9,
            min_samples_leaf=80,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=42,
        )
    if kind == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=80,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown meta model: {kind}")


def score_model(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    raw = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-raw))


def select_validation_threshold(
    candidates: pd.DataFrame,
    rr: float,
    score_col: str,
    coverages: list[float],
    min_trades: int,
) -> tuple[float, float, dict[str, Any]]:
    validation = candidates[candidates["split"] == "validation"].copy()
    scores = validation[score_col].to_numpy(dtype=float)
    best: tuple[float, float, dict[str, Any], float] | None = None
    for coverage in coverages:
        threshold = float(np.quantile(scores, 1.0 - coverage))
        summary = candidate_summary(validation, rr, score_col, threshold)
        trades = int(summary["trades"])
        if trades < min_trades:
            continue
        dd = abs(float(summary["max_drawdown_r"]))
        score = float(summary["net_r"]) / max(dd, 10.0) + 0.25 * float(summary["avg_r"])
        if best is None or score > best[3]:
            best = (threshold, coverage, summary, score)
    if best is None:
        threshold = float(np.quantile(scores, 0.98))
        summary = candidate_summary(validation, rr, score_col, threshold)
        return threshold, 0.02, summary
    return best[0], best[1], best[2]


def run_meta_experiments(
    candidates: pd.DataFrame,
    rr_values: list[float],
    output_prefix: str,
) -> dict[str, Any]:
    base_columns = [
        "lookback",
        "is_long",
        "sweep_depth_atr",
        "rejection_frac",
        "reclaim_atr",
        "body_dir_atr",
        "body_abs_atr",
        "range_atr",
        "close_from_extreme_atr",
        "ltf_rsi",
        "ltf_ema_slope_atr",
        "ltf_atr_ratio",
        "ltf_volume_ratio",
        "ltf_dist_ema_atr",
    ]
    calendar_columns = ["tod_sin", "tod_cos", "dow_sin", "dow_cos", "calendar_any_score", "calendar_dir_score"]
    real_columns = ["real_any_score", "real_dir_score", "real_opp_score", "real_edge_score"]
    placebo_columns = ["placebo_any_score", "placebo_dir_score", "placebo_opp_score", "placebo_edge_score"]
    feature_sets = {
        "price_only": base_columns,
        "price_calendar": base_columns + calendar_columns,
        "price_real": base_columns + calendar_columns + real_columns,
        "price_placebo": base_columns + calendar_columns + placebo_columns,
        "price_real_placebo": base_columns + calendar_columns + real_columns + placebo_columns,
    }
    coverages = [0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.10, 0.15, 0.20]
    model_kinds = ["logit", "hgb", "extra_trees"]
    results: dict[str, Any] = {"baselines": {}, "experiments": []}

    for rr in rr_values:
        results["baselines"][f"rr_{rr:g}"] = {
            split: candidate_summary(candidates[candidates["split"] == split], rr)
            for split in ["train", "validation", "test"]
        }

    for rr in rr_values:
        key = f"{rr:g}"
        train = candidates[(candidates["split"] == "train") & np.isfinite(candidates[f"result_r_{key}"])].copy()
        validation = candidates[(candidates["split"] == "validation") & np.isfinite(candidates[f"result_r_{key}"])].copy()
        test = candidates[(candidates["split"] == "test") & np.isfinite(candidates[f"result_r_{key}"])].copy()
        for feature_name, columns in feature_sets.items():
            for model_kind in model_kinds:
                model = make_meta_model(model_kind)
                y_train = (train[f"result_r_{key}"].to_numpy(dtype=float) > 0.0).astype(int)
                if len(np.unique(y_train)) < 2:
                    continue
                model.fit(train[columns], y_train)
                score_col = f"{output_prefix}_{model_kind}_{feature_name}_rr{key}"
                candidates.loc[train.index, score_col] = score_model(model, train[columns])
                candidates.loc[validation.index, score_col] = score_model(model, validation[columns])
                candidates.loc[test.index, score_col] = score_model(model, test[columns])
                threshold, coverage, val_summary = select_validation_threshold(
                    candidates,
                    rr,
                    score_col,
                    coverages,
                    min_trades=25,
                )
                train_summary = candidate_summary(candidates[candidates["split"] == "train"], rr, score_col, threshold)
                test_summary = candidate_summary(candidates[candidates["split"] == "test"], rr, score_col, threshold)
                results["experiments"].append(
                    {
                        "rr": rr,
                        "feature_set": feature_name,
                        "model": model_kind,
                        "score_col": score_col,
                        "threshold": threshold,
                        "coverage": coverage,
                        "train": train_summary,
                        "validation": val_summary,
                        "test": test_summary,
                    }
                )

    results["experiments"] = sorted(
        results["experiments"],
        key=lambda item: (
            float(item["validation"]["net_r"]),
            float(item["validation"]["avg_r"]),
            float(item["test"]["net_r"]),
        ),
        reverse=True,
    )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Optimize BTC astro/cycle timing into LTF trade selectors.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2026-06-01")
    parser.add_argument("--train-end", default="2023-12-31T23:59:59Z")
    parser.add_argument("--validation-end", default="2024-12-31T23:59:59Z")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path, default=Path("scripts/astro_meta_strategy_results.json"))
    parser.add_argument("--candidate-cache", type=Path, default=Path("scripts/.cache/astro_cycle/meta_candidates.pkl"))
    parser.add_argument("--pivot-threshold-atr", type=float, default=8.0)
    parser.add_argument("--horizon-bars", type=int, default=1)
    parser.add_argument("--lookbacks", default="6,12,24")
    parser.add_argument("--rr-values", default="10,20,30")
    parser.add_argument("--max-hold-bars", type=int, default=288)
    parser.add_argument("--stop-buffer-atr", type=float, default=0.05)
    parser.add_argument("--cost-bps-round-trip", type=float, default=8.0)
    parser.add_argument("--rebuild-candidates", action="store_true")
    args = parser.parse_args()

    start = parse_utc_datetime(args.start)
    end = parse_utc_datetime(args.end)
    split_spec = SplitSpec(
        train_end=pd.Timestamp(parse_utc_datetime(args.train_end)),
        validation_end=pd.Timestamp(parse_utc_datetime(args.validation_end)),
    )
    rr_values = [float(item.strip()) for item in args.rr_values.split(",") if item.strip()]
    lookbacks = [int(item.strip()) for item in args.lookbacks.split(",") if item.strip()]

    base_15m = load_bybit_cached(args.symbol, "15m", start, end, args.cache_dir)
    htf = prepare_frame(base_15m, "15m")
    pivots = zigzag_pivots(htf, args.pivot_threshold_atr)
    labels = make_forward_labels(htf, pivots, args.horizon_bars)
    features, groups = build_feature_matrix(labels["open_time"], args.cache_dir, "15m")
    market = htf_market_features(htf).reset_index(drop=True)
    features = pd.concat([features.reset_index(drop=True), market], axis=1)
    groups["market"] = list(market.columns)
    groups["all_market"] = groups["all"] + groups["market"]
    masks = split_masks(labels["open_time"], split_spec)

    # Keep the timing heads astrology/calendar based. The trade meta model gets
    # price features separately, which makes real-vs-placebo comparisons cleaner.
    score_frame, timing_metrics = fit_scores(
        features,
        labels,
        {**groups, "all": groups["all"]},
        masks,
        pivots,
        args.horizon_bars,
    )

    if args.candidate_cache.exists() and not args.rebuild_candidates:
        candidates = pd.read_pickle(args.candidate_cache)
    else:
        ltf = load_bybit_cached(args.symbol, "5m", start, end, args.cache_dir)
        candidates = build_candidates(
            ltf,
            score_frame,
            split_spec,
            lookbacks,
            rr_values,
            args.max_hold_bars,
            args.stop_buffer_atr,
            args.cost_bps_round_trip,
        )
        args.candidate_cache.parent.mkdir(parents=True, exist_ok=True)
        candidates.to_pickle(args.candidate_cache)

    results = {
        "config": {
            "symbol": args.symbol,
            "start": start,
            "end": end,
            "train_end": split_spec.train_end,
            "validation_end": split_spec.validation_end,
            "pivot_threshold_atr": args.pivot_threshold_atr,
            "horizon_bars": args.horizon_bars,
            "lookbacks": lookbacks,
            "rr_values": rr_values,
            "max_hold_bars": args.max_hold_bars,
            "stop_buffer_atr": args.stop_buffer_atr,
            "cost_bps_round_trip": args.cost_bps_round_trip,
        },
        "data": {
            "htf_bars": len(htf),
            "pivots": len(pivots),
            "candidates": len(candidates),
            "candidates_by_split": candidates["split"].value_counts().to_dict() if not candidates.empty else {},
        },
        "timing_metrics": timing_metrics,
        "meta": run_meta_experiments(candidates, rr_values, "meta"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, default=json_default), encoding="utf-8")
    print(json.dumps(results["data"], indent=2, default=json_default))
    print(json.dumps(results["meta"]["experiments"][:10], indent=2, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
