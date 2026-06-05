from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import joblib
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - only for environments without sklearn
    SKLEARN_AVAILABLE = False

from scripts.train_btc_liquidity_sweep_ml_head import (
    NUMERIC_FEATURES,
    classifier_metrics,
    one_trade_at_a_time,
    parse_float_list,
    trade_metrics,
    with_prefix,
)


DEFAULT_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "BNBUSDT",
    "1000PEPEUSDT",
    "DOGEUSDT",
    "AVAXUSDT",
    "LINKUSDT",
]

CATEGORICAL_FEATURES = [
    "symbol",
    "direction",
    "session",
    "trend_side",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def parse_symbols(raw: str) -> list[str]:
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - older sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_model(name: str) -> Pipeline:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", one_hot_encoder())])
    preprocessor = ColumnTransformer(
        [("num", numeric, NUMERIC_FEATURES), ("cat", categorical, CATEGORICAL_FEATURES)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    if name == "logreg":
        estimator = LogisticRegression(C=0.35, class_weight="balanced", max_iter=2500, random_state=31)
    elif name == "rf":
        estimator = RandomForestClassifier(
            n_estimators=400,
            max_depth=4,
            min_samples_leaf=12,
            class_weight="balanced_subsample",
            random_state=31,
        )
    elif name == "hgb":
        estimator = HistGradientBoostingClassifier(
            learning_rate=0.03,
            max_iter=110,
            max_leaf_nodes=7,
            min_samples_leaf=16,
            l2_regularization=1.0,
            random_state=31,
        )
    else:
        raise ValueError(f"Unknown model: {name}")
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def event_file(symbol_dir: Path) -> Path:
    preferred = symbol_dir / "liquidity_sweep_events.csv"
    if preferred.exists():
        return preferred
    return symbol_dir / "btc_liquidity_sweep_events.csv"


def load_symbol_dataset(symbol: str, args: argparse.Namespace) -> pd.DataFrame:
    symbol_dir = args.input_dir / symbol.lower()
    events_path = event_file(symbol_dir)
    trades_path = symbol_dir / "preferred_backtest_trades.csv"
    if not events_path.exists() or not trades_path.exists():
        raise FileNotFoundError(f"Missing research outputs for {symbol} in {symbol_dir}")

    events = pd.read_csv(events_path)
    trades = pd.read_csv(trades_path)
    for frame in [events, trades]:
        if "symbol" not in frame.columns:
            frame.insert(0, "symbol", symbol)
        frame["symbol"] = frame["symbol"].astype(str).str.upper()
        for column in ["event_time", "entry_time", "exit_time"]:
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")

    trades = trades[
        (trades["candidate_filter"].astype(str) == args.candidate_filter)
        & (trades["config"].astype(str) == args.config)
        & (pd.to_numeric(trades["stop_buffer_atr"], errors="coerce").round(6) == round(float(args.stop_buffer_atr), 6))
    ].copy()
    if trades.empty:
        return pd.DataFrame()

    merge_cols = ["symbol", "level_id", "entry_time", "direction"]
    event_cols = [
        "symbol",
        "level_id",
        "entry_time",
        "direction",
        "hour",
        "weekday",
        "level_age_bars",
        "sweep_wick_to_body_atr",
        "sweep_depth_atr",
        "reclaim_pos",
        "sweep_range_atr",
        "sweep_body_atr",
        "volume_ratio",
        "atr_ratio",
        "rsi",
        "dist_ema200_atr",
        "ema200_slope_atr",
        "pre_return_4_atr",
        "pre_return_12_atr",
        "pre_range_12_atr",
        "session",
        "trend_side",
        "level",
        "sweep_atr",
        "entry_price",
        "stop_price",
        "candidate_4h_session_trend",
        "candidate_4h_session_trend_bounded",
    ]
    event_features = events[[column for column in event_cols if column in events.columns]].copy()
    merged = trades.merge(event_features, on=merge_cols, how="left", suffixes=("", "_event"))
    missing = int(merged["sweep_depth_atr"].isna().sum()) if "sweep_depth_atr" in merged else len(merged)
    if missing:
        raise RuntimeError(f"Failed to merge event features for {missing} {symbol} trades.")

    direction_sign = np.where(merged["direction"].astype(str) == "long", 1.0, -1.0)
    merged["direction_sign"] = direction_sign
    merged["trend_aligned_sign"] = np.where(
        ((merged["direction"].astype(str) == "long") & (merged["trend_side"].astype(str) == "up"))
        | ((merged["direction"].astype(str) == "short") & (merged["trend_side"].astype(str) == "down")),
        1.0,
        0.0,
    )
    merged["entry_vs_level_atr"] = (
        (pd.to_numeric(merged["entry_price"], errors="coerce") - pd.to_numeric(merged["level"], errors="coerce"))
        / pd.to_numeric(merged["sweep_atr"], errors="coerce").replace(0.0, np.nan)
        * direction_sign
    )
    merged["stop_distance_atr"] = (
        (pd.to_numeric(merged["entry_price"], errors="coerce") - pd.to_numeric(merged["stop_price"], errors="coerce")).abs()
        / pd.to_numeric(merged["sweep_atr"], errors="coerce").replace(0.0, np.nan)
    )
    merged["entry_year"] = pd.to_datetime(merged["entry_time"], utc=True).dt.year
    merged["label"] = (pd.to_numeric(merged["result_r"], errors="coerce") > float(args.label_min_r)).astype(int)
    return merged.sort_values("entry_time").reset_index(drop=True)


def load_dataset(args: argparse.Namespace) -> pd.DataFrame:
    parts = [load_symbol_dataset(symbol, args) for symbol in parse_symbols(args.symbols)]
    dataset = pd.concat([part for part in parts if not part.empty], ignore_index=True)
    if dataset.empty:
        raise RuntimeError("No preferred setup trades found in the requested universe.")
    return dataset.sort_values(["entry_time", "symbol"]).reset_index(drop=True)


def score_frame(model: Pipeline, frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["ml_prob"] = model.predict_proba(out[FEATURE_COLUMNS])[:, 1]
    return out


def candidate_thresholds(validation: pd.DataFrame, thresholds: list[float], keep_fracs: list[float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [{"gate": "all", "threshold": -math.inf, "keep_frac": 1.0}]
    probs = pd.to_numeric(validation["ml_prob"], errors="coerce").dropna()
    for threshold in thresholds:
        rows.append({"gate": f"prob_ge_{threshold:.2f}", "threshold": float(threshold), "keep_frac": math.nan})
    for keep_frac in keep_fracs:
        keep_frac = min(max(float(keep_frac), 0.0), 1.0)
        if probs.empty:
            continue
        threshold = float(probs.quantile(1.0 - keep_frac))
        rows.append({"gate": f"val_top_{keep_frac:.2f}", "threshold": threshold, "keep_frac": keep_frac})
    return pd.DataFrame(rows).drop_duplicates(["gate", "threshold"]).reset_index(drop=True)


def choose_stable_gate(
    validation: pd.DataFrame,
    thresholds: list[float],
    keep_fracs: list[float],
    *,
    min_val_trades: int,
    smooth_radius: int,
    min_edge_r: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for _, candidate in candidate_thresholds(validation, thresholds, keep_fracs).iterrows():
        selected = validation[pd.to_numeric(validation["ml_prob"], errors="coerce") >= float(candidate["threshold"])].copy()
        metrics = trade_metrics(selected)
        row = candidate.to_dict()
        row.update(metrics)
        row["eligible"] = bool(metrics["trades"] >= min_val_trades)
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    if table.empty:
        return {"gate": "all", "threshold": -math.inf, "keep_frac": 1.0}

    smooth_scores: list[float] = []
    smooth_trade_counts: list[int] = []
    for idx in range(len(table)):
        start = max(0, idx - smooth_radius)
        end = min(len(table), idx + smooth_radius + 1)
        neighborhood = table.iloc[start:end]
        eligible = neighborhood[neighborhood["eligible"]]
        smooth_scores.append(float(eligible["avg_r"].median()) if not eligible.empty else -999.0)
        smooth_trade_counts.append(int(eligible["trades"].median()) if not eligible.empty else 0)
    table["selection_score"] = smooth_scores
    table["smooth_neighborhood_trades"] = smooth_trade_counts

    all_row = table[table["gate"].eq("all")].iloc[0]
    best = table.sort_values(["selection_score", "trades", "net_r"], ascending=[False, False, False]).iloc[0]
    if best["gate"] != "all" and float(best["selection_score"]) < float(all_row["avg_r"]) + min_edge_r:
        best = all_row
    out = best.to_dict()
    out["threshold_table"] = table
    return out


def fold_schedule(dataset: pd.DataFrame, first_test_year: int | None) -> list[tuple[int, int, int]]:
    years = sorted(int(year) for year in dataset["entry_year"].dropna().unique())
    if len(years) < 4:
        return []
    start = first_test_year if first_test_year is not None else max(years[0] + 3, years[2])
    folds: list[tuple[int, int, int]] = []
    for test_year in years:
        if test_year < start:
            continue
        val_year = test_year - 1
        if val_year not in years:
            continue
        train_years = [year for year in years if year < val_year]
        if train_years:
            folds.append((max(train_years), val_year, test_year))
    return folds


def aggregate_summary(scored: pd.DataFrame, selected: pd.DataFrame, model_name: str, scope: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for portfolio, all_frame, selected_frame in [
        ("all_signals", scored, selected),
        ("one_trade_at_a_time", one_trade_at_a_time(scored), one_trade_at_a_time(selected)),
        ("one_trade_per_symbol", one_trade_per_symbol_at_a_time(scored), one_trade_per_symbol_at_a_time(selected)),
    ]:
        row: dict[str, Any] = {"scope": scope, "model": model_name, "portfolio": portfolio}
        row.update(with_prefix("all", trade_metrics(all_frame)))
        row.update(with_prefix("selected", trade_metrics(selected_frame)))
        row["delta_avg_r"] = row["selected_avg_r"] - row["all_avg_r"]
        row["delta_net_r"] = row["selected_net_r"] - row["all_net_r"]
        row["kept_share"] = row["selected_trades"] / row["all_trades"] if row["all_trades"] else 0.0
        rows.append(row)
    return rows


def one_trade_per_symbol_at_a_time(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    parts = [one_trade_at_a_time(group) for _, group in frame.groupby("symbol", dropna=False)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def aggregate_by_symbol(scored: pd.DataFrame, selected: pd.DataFrame, model_name: str, scope: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected_keys = set(selected.index.tolist())
    for symbol, group in scored.groupby("symbol", dropna=False):
        selected_group = group[group.index.isin(selected_keys)].copy()
        for portfolio, all_frame, selected_frame in [
            ("all_signals", group, selected_group),
            ("one_trade_at_a_time", one_trade_at_a_time(group), one_trade_at_a_time(selected_group)),
        ]:
            row: dict[str, Any] = {"scope": scope, "model": model_name, "symbol": symbol, "portfolio": portfolio}
            row.update(with_prefix("all", trade_metrics(all_frame)))
            row.update(with_prefix("selected", trade_metrics(selected_frame)))
            row["delta_avg_r"] = row["selected_avg_r"] - row["all_avg_r"]
            row["delta_net_r"] = row["selected_net_r"] - row["all_net_r"]
            row["kept_share"] = row["selected_trades"] / row["all_trades"] if row["all_trades"] else 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def run_pooled_walk_forward(dataset: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    thresholds = parse_float_list(args.thresholds)
    keep_fracs = parse_float_list(args.keep_fracs)
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    folds = fold_schedule(dataset, args.first_test_year)
    if not folds:
        raise RuntimeError("Not enough annual data to build train/validation/test folds.")

    fold_rows: list[dict[str, Any]] = []
    threshold_rows: list[pd.DataFrame] = []
    scored_parts: list[pd.DataFrame] = []
    selected_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    by_symbol_parts: list[pd.DataFrame] = []

    for model_name in models:
        model_scored_parts: list[pd.DataFrame] = []
        model_selected_parts: list[pd.DataFrame] = []
        for fold_index, (_, val_year, test_year) in enumerate(folds, start=1):
            train = dataset[dataset["entry_year"] < val_year].copy()
            validation = dataset[dataset["entry_year"] == val_year].copy()
            test = dataset[dataset["entry_year"] == test_year].copy()
            if len(train) < args.min_train_trades or len(validation) < args.min_val_trades or test.empty:
                continue
            if train["label"].nunique() < 2:
                continue
            model = build_model(model_name)
            model.fit(train[FEATURE_COLUMNS], train["label"].astype(int))
            validation = score_frame(model, validation)
            test = score_frame(model, test)
            gate = choose_stable_gate(
                validation,
                thresholds,
                keep_fracs,
                min_val_trades=args.min_val_trades,
                smooth_radius=args.smooth_radius,
                min_edge_r=args.min_edge_r,
            )
            threshold_table = gate.pop("threshold_table", pd.DataFrame())
            if not threshold_table.empty:
                threshold_table.insert(0, "fold", fold_index)
                threshold_table.insert(1, "model", model_name)
                threshold_table.insert(2, "val_year", val_year)
                threshold_table.insert(3, "test_year", test_year)
                threshold_rows.append(threshold_table)

            threshold = float(gate["threshold"])
            selected_test = test[pd.to_numeric(test["ml_prob"], errors="coerce") >= threshold].copy()
            for frame in [test, selected_test]:
                frame["fold"] = fold_index
                frame["model"] = model_name
                frame["val_year"] = val_year
                frame["test_year"] = test_year
                frame["selected_gate"] = gate["gate"]
                frame["selected_threshold"] = threshold

            row: dict[str, Any] = {
                "fold": fold_index,
                "model": model_name,
                "train_years": f"{int(train['entry_year'].min())}-{int(train['entry_year'].max())}",
                "val_year": int(val_year),
                "test_year": int(test_year),
                "train_trades": int(len(train)),
                "val_trades": int(len(validation)),
                "test_trades": int(len(test)),
                "gate": gate["gate"],
                "threshold": threshold,
                "smooth_score": float(gate.get("selection_score", math.nan)),
                "smooth_neighborhood_trades": int(gate.get("smooth_neighborhood_trades", 0)),
            }
            row.update(with_prefix("val_clf", classifier_metrics(validation)))
            row.update(with_prefix("test_clf", classifier_metrics(test)))
            row.update(with_prefix("val_all", trade_metrics(validation)))
            row.update(with_prefix("val_selected", trade_metrics(validation[validation["ml_prob"] >= threshold])))
            row.update(with_prefix("test_all", trade_metrics(test)))
            row.update(with_prefix("test_selected", trade_metrics(selected_test)))
            fold_rows.append(row)

            model_scored_parts.append(test)
            model_selected_parts.append(selected_test)
            scored_parts.append(test)
            selected_parts.append(selected_test)

        scored_model = pd.concat(model_scored_parts, ignore_index=False) if model_scored_parts else pd.DataFrame()
        selected_model = pd.concat(model_selected_parts, ignore_index=False) if model_selected_parts else pd.DataFrame()
        summary_rows.extend(aggregate_summary(scored_model, selected_model, model_name, "pooled_annual"))
        if not scored_model.empty:
            by_symbol_parts.append(aggregate_by_symbol(scored_model, selected_model, model_name, "pooled_annual"))

    folds_out = pd.DataFrame(fold_rows)
    thresholds_out = pd.concat(threshold_rows, ignore_index=True) if threshold_rows else pd.DataFrame()
    scored_out = pd.concat(scored_parts, ignore_index=False) if scored_parts else pd.DataFrame()
    selected_out = pd.concat(selected_parts, ignore_index=False) if selected_parts else pd.DataFrame()
    summary_out = pd.DataFrame(summary_rows).sort_values(["portfolio", "delta_avg_r", "selected_avg_r"], ascending=[True, False, False])
    by_symbol_out = pd.concat(by_symbol_parts, ignore_index=True) if by_symbol_parts else pd.DataFrame()
    scored_selected = pd.concat(
        [scored_out.assign(_selected_oos=False), selected_out.assign(_selected_oos=True)],
        ignore_index=True,
    )
    return summary_out, by_symbol_out, folds_out, thresholds_out, scored_selected


def run_symbol_holdout_walk_forward(dataset: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    thresholds = parse_float_list(args.thresholds)
    keep_fracs = parse_float_list(args.keep_fracs)
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    folds = fold_schedule(dataset, args.first_test_year)
    symbols = sorted(dataset["symbol"].dropna().astype(str).unique())

    fold_rows: list[dict[str, Any]] = []
    threshold_rows: list[pd.DataFrame] = []
    scored_parts: list[pd.DataFrame] = []
    selected_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    by_symbol_parts: list[pd.DataFrame] = []

    for model_name in models:
        model_scored_parts: list[pd.DataFrame] = []
        model_selected_parts: list[pd.DataFrame] = []
        for holdout_symbol in symbols:
            for fold_index, (_, val_year, test_year) in enumerate(folds, start=1):
                train = dataset[(dataset["entry_year"] < val_year) & (dataset["symbol"] != holdout_symbol)].copy()
                validation = dataset[(dataset["entry_year"] == val_year) & (dataset["symbol"] != holdout_symbol)].copy()
                test = dataset[(dataset["entry_year"] == test_year) & (dataset["symbol"] == holdout_symbol)].copy()
                if len(train) < args.min_train_trades or len(validation) < args.min_val_trades or len(test) < args.min_holdout_test_trades:
                    continue
                if train["label"].nunique() < 2:
                    continue
                model = build_model(model_name)
                model.fit(train[FEATURE_COLUMNS], train["label"].astype(int))
                validation = score_frame(model, validation)
                test = score_frame(model, test)
                gate = choose_stable_gate(
                    validation,
                    thresholds,
                    keep_fracs,
                    min_val_trades=args.min_val_trades,
                    smooth_radius=args.smooth_radius,
                    min_edge_r=args.min_edge_r,
                )
                threshold_table = gate.pop("threshold_table", pd.DataFrame())
                if not threshold_table.empty:
                    threshold_table.insert(0, "fold", fold_index)
                    threshold_table.insert(1, "model", model_name)
                    threshold_table.insert(2, "holdout_symbol", holdout_symbol)
                    threshold_table.insert(3, "val_year", val_year)
                    threshold_table.insert(4, "test_year", test_year)
                    threshold_rows.append(threshold_table)

                threshold = float(gate["threshold"])
                selected_test = test[pd.to_numeric(test["ml_prob"], errors="coerce") >= threshold].copy()
                for frame in [test, selected_test]:
                    frame["fold"] = fold_index
                    frame["model"] = model_name
                    frame["holdout_symbol"] = holdout_symbol
                    frame["val_year"] = val_year
                    frame["test_year"] = test_year
                    frame["selected_gate"] = gate["gate"]
                    frame["selected_threshold"] = threshold

                row: dict[str, Any] = {
                    "fold": fold_index,
                    "model": model_name,
                    "holdout_symbol": holdout_symbol,
                    "train_years": f"{int(train['entry_year'].min())}-{int(train['entry_year'].max())}",
                    "val_year": int(val_year),
                    "test_year": int(test_year),
                    "train_trades": int(len(train)),
                    "val_trades": int(len(validation)),
                    "test_trades": int(len(test)),
                    "gate": gate["gate"],
                    "threshold": threshold,
                    "smooth_score": float(gate.get("selection_score", math.nan)),
                    "smooth_neighborhood_trades": int(gate.get("smooth_neighborhood_trades", 0)),
                }
                row.update(with_prefix("val_clf", classifier_metrics(validation)))
                row.update(with_prefix("test_clf", classifier_metrics(test)))
                row.update(with_prefix("val_all", trade_metrics(validation)))
                row.update(with_prefix("val_selected", trade_metrics(validation[validation["ml_prob"] >= threshold])))
                row.update(with_prefix("test_all", trade_metrics(test)))
                row.update(with_prefix("test_selected", trade_metrics(selected_test)))
                fold_rows.append(row)

                model_scored_parts.append(test)
                model_selected_parts.append(selected_test)
                scored_parts.append(test)
                selected_parts.append(selected_test)

        scored_model = pd.concat(model_scored_parts, ignore_index=False) if model_scored_parts else pd.DataFrame()
        selected_model = pd.concat(model_selected_parts, ignore_index=False) if model_selected_parts else pd.DataFrame()
        summary_rows.extend(aggregate_summary(scored_model, selected_model, model_name, "leave_one_symbol_annual"))
        if not scored_model.empty:
            by_symbol_parts.append(aggregate_by_symbol(scored_model, selected_model, model_name, "leave_one_symbol_annual"))

    folds_out = pd.DataFrame(fold_rows)
    thresholds_out = pd.concat(threshold_rows, ignore_index=True) if threshold_rows else pd.DataFrame()
    scored_out = pd.concat(scored_parts, ignore_index=False) if scored_parts else pd.DataFrame()
    selected_out = pd.concat(selected_parts, ignore_index=False) if selected_parts else pd.DataFrame()
    summary_out = pd.DataFrame(summary_rows).sort_values(["portfolio", "delta_avg_r", "selected_avg_r"], ascending=[True, False, False])
    by_symbol_out = pd.concat(by_symbol_parts, ignore_index=True) if by_symbol_parts else pd.DataFrame()
    scored_selected = pd.concat(
        [scored_out.assign(_selected_oos=False), selected_out.assign(_selected_oos=True)],
        ignore_index=True,
    )
    return summary_out, by_symbol_out, folds_out, thresholds_out, scored_selected


def train_final_model(dataset: pd.DataFrame, model_name: str, output_path: Path, args: argparse.Namespace) -> None:
    model = build_model(model_name)
    model.fit(dataset[FEATURE_COLUMNS], dataset["label"].astype(int))
    payload = {
        "model": model,
        "model_name": model_name,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "candidate_filter": args.candidate_filter,
        "config": args.config,
        "stop_buffer_atr": float(args.stop_buffer_atr),
        "label_min_r": float(args.label_min_r),
        "symbols": parse_symbols(args.symbols),
        "trained_rows": int(len(dataset)),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward ML head for a multi-symbol liquidity sweep universe.")
    parser.add_argument("--input-dir", type=Path, default=Path("scripts/liquidity_sweep_universe"))
    parser.add_argument("--output-prefix", type=Path, default=Path("scripts/liquidity_sweep_universe/universe_liquidity_sweep_ml_head"))
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--candidate-filter", default="4h_session_trend")
    parser.add_argument("--config", default="partial_2p5_7p5")
    parser.add_argument("--stop-buffer-atr", type=float, default=0.15)
    parser.add_argument("--label-min-r", type=float, default=0.0)
    parser.add_argument("--models", default="logreg,rf,hgb")
    parser.add_argument("--thresholds", default="0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70")
    parser.add_argument("--keep-fracs", default="0.35,0.50,0.65,0.80")
    parser.add_argument("--min-train-trades", type=int, default=120)
    parser.add_argument("--min-val-trades", type=int, default=25)
    parser.add_argument("--min-holdout-test-trades", type=int, default=3)
    parser.add_argument("--first-test-year", type=int, default=2024)
    parser.add_argument("--smooth-radius", type=int, default=1)
    parser.add_argument("--min-edge-r", type=float, default=0.03)
    parser.add_argument("--skip-symbol-holdout", action="store_true")
    parser.add_argument("--write-final-model", action="store_true")
    parser.add_argument("--final-model-name", choices=["logreg", "rf", "hgb"], default="hgb")
    return parser.parse_args()


def write_table(path: Path, table: pd.DataFrame) -> None:
    table.to_csv(path, index=False)
    print(f"Saved {path}", flush=True)


def main() -> None:
    args = parse_args()
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn and joblib are required for this script.")
    dataset = load_dataset(args)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    dataset_path = args.output_prefix.with_name(args.output_prefix.name + "_dataset.csv")
    write_table(dataset_path, dataset)

    pooled_summary, pooled_by_symbol, pooled_folds, pooled_thresholds, pooled_scored = run_pooled_walk_forward(dataset, args)
    write_table(args.output_prefix.with_name(args.output_prefix.name + "_pooled_summary.csv"), pooled_summary)
    write_table(args.output_prefix.with_name(args.output_prefix.name + "_pooled_summary_by_symbol.csv"), pooled_by_symbol)
    write_table(args.output_prefix.with_name(args.output_prefix.name + "_pooled_folds.csv"), pooled_folds)
    write_table(args.output_prefix.with_name(args.output_prefix.name + "_pooled_thresholds.csv"), pooled_thresholds)
    write_table(args.output_prefix.with_name(args.output_prefix.name + "_pooled_scored_selected.csv"), pooled_scored)

    if not args.skip_symbol_holdout:
        holdout_summary, holdout_by_symbol, holdout_folds, holdout_thresholds, holdout_scored = run_symbol_holdout_walk_forward(dataset, args)
        write_table(args.output_prefix.with_name(args.output_prefix.name + "_symbol_holdout_summary.csv"), holdout_summary)
        write_table(args.output_prefix.with_name(args.output_prefix.name + "_symbol_holdout_summary_by_symbol.csv"), holdout_by_symbol)
        write_table(args.output_prefix.with_name(args.output_prefix.name + "_symbol_holdout_folds.csv"), holdout_folds)
        write_table(args.output_prefix.with_name(args.output_prefix.name + "_symbol_holdout_thresholds.csv"), holdout_thresholds)
        write_table(args.output_prefix.with_name(args.output_prefix.name + "_symbol_holdout_scored_selected.csv"), holdout_scored)

    config_path = args.output_prefix.with_name(args.output_prefix.name + "_config.json")
    config_payload = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
    config_path.write_text(json.dumps(config_payload, indent=2, sort_keys=True), encoding="utf-8")

    if args.write_final_model:
        model_path = args.output_prefix.with_name(args.output_prefix.name + f"_{args.final_model_name}.joblib")
        train_final_model(dataset, args.final_model_name, model_path, args)
        print(f"Saved final model to {model_path}", flush=True)

    print("\nPooled annual summary:")
    print(pooled_summary.to_string(index=False))
    if not args.skip_symbol_holdout:
        print("\nLeave-one-symbol annual summary:")
        print(holdout_summary.to_string(index=False))


if __name__ == "__main__":
    main()
