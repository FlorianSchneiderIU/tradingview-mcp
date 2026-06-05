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
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - only used when local env lacks sklearn
    SKLEARN_AVAILABLE = False

try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except Exception:  # pragma: no cover - xgboost is optional
    XGBOOST_AVAILABLE = False

from scripts.train_pyharmonics_survivor_ml import (  # noqa: E402
    BASE_CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    one_trade_at_a_time,
    one_trade_per_symbol_at_a_time,
    parse_csv_values,
    parse_symbols,
    trade_metrics,
    with_prefix,
)


DEFAULT_DATASET = Path("scripts/pyharmonics_survivor_ml_entry_models_expanded_20260604_dataset.csv")


def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - older sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def feature_columns(feature_set: str) -> tuple[list[str], list[str]]:
    mode = str(feature_set or "no_symbol").strip().lower()
    categorical = list(BASE_CATEGORICAL_FEATURES)
    if mode == "with_symbol":
        categorical = ["symbol", *categorical]
    elif mode != "no_symbol":
        raise ValueError(f"Unsupported feature set: {feature_set!r}")
    return list(NUMERIC_FEATURES), categorical


def available_models(raw: str) -> list[str]:
    out: list[str] = []
    for item in parse_csv_values(raw, str):
        model = item.strip().lower()
        if model == "xgb" and not XGBOOST_AVAILABLE:
            print("Skipping xgb: xgboost is not installed.", flush=True)
            continue
        out.append(model)
    return out


def build_regressor(name: str, numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", one_hot_encoder())])
    preprocessor = ColumnTransformer(
        [("num", numeric, numeric_features), ("cat", categorical, categorical_features)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    model_name = str(name or "").strip().lower()
    if model_name == "ridge":
        estimator = Ridge(alpha=3.0, random_state=31)
    elif model_name == "rf":
        estimator = RandomForestRegressor(
            n_estimators=500,
            max_depth=5,
            min_samples_leaf=10,
            random_state=31,
            n_jobs=-1,
        )
    elif model_name == "extratrees":
        estimator = ExtraTreesRegressor(
            n_estimators=500,
            max_depth=5,
            min_samples_leaf=10,
            random_state=31,
            n_jobs=-1,
        )
    elif model_name == "hgb":
        estimator = HistGradientBoostingRegressor(
            learning_rate=0.035,
            max_iter=140,
            max_leaf_nodes=9,
            min_samples_leaf=18,
            l2_regularization=1.0,
            random_state=31,
        )
    elif model_name == "xgb":
        if not XGBOOST_AVAILABLE:
            raise RuntimeError("xgboost is not installed in this Python environment.")
        estimator = XGBRegressor(
            n_estimators=260,
            max_depth=3,
            learning_rate=0.035,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=8,
            reg_lambda=2.0,
            objective="reg:squarederror",
            random_state=31,
            n_jobs=2,
        )
    else:
        raise ValueError(f"Unknown model: {name!r}")
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def load_dataset(args: argparse.Namespace) -> pd.DataFrame:
    dataset = pd.read_csv(args.dataset)
    for column in ["completion_time", "detection_time", "trigger_time", "entry_time", "exit_time", "trigger_break_time"]:
        if column in dataset.columns:
            dataset[column] = pd.to_datetime(dataset[column], utc=True, errors="coerce")
    if "result_r" not in dataset.columns:
        dataset["result_r"] = pd.to_numeric(dataset["r_multiple_net"], errors="coerce").fillna(0.0)
    dataset["result_r"] = pd.to_numeric(dataset["result_r"], errors="coerce").fillna(0.0)
    dataset["entry_year"] = pd.to_datetime(dataset["entry_time"], utc=True, errors="coerce").dt.year

    symbols = set(parse_symbols(args.symbols))
    if symbols:
        dataset = dataset[dataset["symbol"].astype(str).str.upper().isin(symbols)].copy()
    filters = {
        "family": parse_csv_values(args.families, str),
        "pattern_name": parse_csv_values(args.pattern_name_filters, str),
        "entry_mode": parse_csv_values(args.entry_modes, str),
        "time_filter": parse_csv_values(args.time_filters, str),
        "peak_spacing": parse_csv_values(args.peak_spacings, int),
        "target_rr_planned": parse_csv_values(args.rrs, float),
        "stop_atr_buffer": parse_csv_values(args.stop_buffers, float),
        "breakeven_trigger_r": parse_csv_values(args.breakeven_triggers, float),
        "min_harmonic_quality_score": parse_csv_values(args.min_quality_scores, float),
    }
    numeric_filter_columns = {
        "peak_spacing",
        "target_rr_planned",
        "stop_atr_buffer",
        "breakeven_trigger_r",
        "min_harmonic_quality_score",
    }
    for column, values in filters.items():
        if not values or column not in dataset.columns:
            continue
        if column in numeric_filter_columns:
            numeric = pd.to_numeric(dataset[column], errors="coerce")
            allowed = {round(float(value), 8) for value in values}
            dataset = dataset[numeric.round(8).isin(allowed)].copy()
        else:
            if column == "pattern_name":
                allowed = {str(value).strip().lower().replace("_", " ") for value in values}
                normalized = dataset[column].astype(str).str.lower().str.replace("_", " ", regex=False)
                dataset = dataset[normalized.isin(allowed)].copy()
            else:
                allowed = {str(value).strip().lower() for value in values}
                dataset = dataset[dataset[column].astype(str).str.lower().isin(allowed)].copy()
    if dataset.empty:
        raise RuntimeError("No action candidates remain after filters.")
    dataset["event_decision_key"] = dataset["event_key"].astype(str)

    def action_part(column: str, label: str, *, numeric: bool = False) -> pd.Series:
        if column not in dataset.columns:
            return pd.Series([f"{label}=na"] * len(dataset), index=dataset.index)
        if numeric:
            return pd.to_numeric(dataset[column], errors="coerce").map(lambda value: f"{label}={value:.6g}")
        return label + "=" + dataset[column].astype(str)

    dataset["action_key"] = (
        action_part("entry_mode", "entry")
        + "|"
        + action_part("peak_spacing", "peak", numeric=True)
        + "|"
        + action_part("time_filter", "time")
        + "|"
        + action_part("target_rr_planned", "rr", numeric=True)
        + "|"
        + action_part("stop_atr_buffer", "stop", numeric=True)
        + "|"
        + action_part("breakeven_trigger_r", "be", numeric=True)
        + "|"
        + action_part("min_harmonic_quality_score", "q", numeric=True)
    )
    return dataset.sort_values(["entry_time", "symbol", "event_decision_key", "action_key"]).reset_index(drop=True)


def fold_schedule(dataset: pd.DataFrame, first_test_year: int | None) -> list[tuple[int, int, int]]:
    years = sorted(int(year) for year in dataset["entry_year"].dropna().unique())
    if len(years) < 3:
        return []
    start = int(first_test_year) if first_test_year is not None else max(years[0] + 2, years[1])
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


def sample_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("event_decision_key")["event_decision_key"].transform("count")
    weights = 1.0 / pd.to_numeric(counts, errors="coerce").replace(0.0, np.nan)
    return weights.fillna(1.0).to_numpy(dtype=float)


def fit_regressor(model: Pipeline, train: pd.DataFrame, features: list[str]) -> Pipeline:
    weights = sample_weights(train)
    try:
        model.fit(train[features], train["result_r"], model__sample_weight=weights)
    except TypeError:
        model.fit(train[features], train["result_r"])
    return model


def score_frame(model: Pipeline, frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = frame.copy()
    out["pred_r"] = model.predict(out[features])
    return out


def top_action_per_event(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    out["pred_r"] = pd.to_numeric(out["pred_r"], errors="coerce").fillna(-999.0)
    sort_cols = ["event_decision_key", "pred_r", "target_rr_planned", "entry_time"]
    top = (
        out.sort_values(sort_cols, ascending=[True, False, False, True])
        .drop_duplicates("event_decision_key", keep="first")
        .sort_values(["entry_time", "symbol"])
        .reset_index(drop=True)
    )
    return top


def portfolio(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    mode = str(mode or "one_symbol").strip().lower()
    if frame.empty:
        return frame.copy()
    if mode in {"raw", "events"}:
        return frame.copy()
    if mode == "one_trade":
        return one_trade_at_a_time(frame)
    if mode == "one_symbol":
        return one_trade_per_symbol_at_a_time(frame)
    raise ValueError(f"Unsupported portfolio mode: {mode!r}")


def decision_frame(scored: pd.DataFrame, threshold: float, portfolio_mode: str) -> pd.DataFrame:
    top = top_action_per_event(scored)
    if top.empty:
        return top.copy()
    top["decision_threshold"] = float(threshold)
    top["selected"] = pd.to_numeric(top["pred_r"], errors="coerce") >= float(threshold)
    top["decision"] = np.where(top["selected"], top["action_key"], "reject")
    selected = portfolio(top[top["selected"]].copy(), portfolio_mode)
    selected_ids = set(selected["candidate_id"].astype(str)) if "candidate_id" in selected.columns else set()
    top["portfolio_selected"] = top["candidate_id"].astype(str).isin(selected_ids)
    top["decision"] = np.where(top["portfolio_selected"], top["decision"], "reject")
    top["selected"] = top["portfolio_selected"]
    return top


def selected_trades(decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty:
        return decisions.copy()
    return decisions[decisions["selected"]].copy()


def threshold_candidates(validation_top: pd.DataFrame, thresholds: list[float], keep_fracs: list[float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [{"gate": "all", "threshold": -math.inf, "keep_frac": 1.0}]
    for threshold in thresholds:
        rows.append({"gate": f"pred_ge_{threshold:.2f}", "threshold": float(threshold), "keep_frac": math.nan})
    preds = pd.to_numeric(validation_top["pred_r"], errors="coerce").dropna()
    for keep_frac in keep_fracs:
        keep_frac = min(max(float(keep_frac), 0.0), 1.0)
        if preds.empty:
            continue
        threshold = float(preds.quantile(1.0 - keep_frac))
        rows.append({"gate": f"val_top_{keep_frac:.2f}", "threshold": threshold, "keep_frac": keep_frac})
    return pd.DataFrame(rows).drop_duplicates(["gate", "threshold"]).sort_values("threshold").reset_index(drop=True)


def choose_threshold(
    validation: pd.DataFrame,
    *,
    thresholds: list[float],
    keep_fracs: list[float],
    portfolio_mode: str,
    min_val_trades: int,
    smooth_radius: int,
    min_edge_r: float,
) -> dict[str, Any]:
    top = top_action_per_event(validation)
    baseline_metrics = trade_metrics(portfolio(top, portfolio_mode))
    rows: list[dict[str, Any]] = []
    for _, candidate in threshold_candidates(top, thresholds, keep_fracs).iterrows():
        decisions = decision_frame(validation, float(candidate["threshold"]), portfolio_mode)
        metrics = trade_metrics(selected_trades(decisions))
        row = candidate.to_dict()
        row.update(metrics)
        row["eligible"] = bool(metrics["trades"] >= int(min_val_trades))
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    if table.empty:
        return {"gate": "all", "threshold": -math.inf, "threshold_table": table}

    smooth_scores: list[float] = []
    smooth_trade_counts: list[int] = []
    for idx in range(len(table)):
        start = max(0, idx - int(smooth_radius))
        end = min(len(table), idx + int(smooth_radius) + 1)
        neighborhood = table.iloc[start:end]
        eligible = neighborhood[neighborhood["eligible"]]
        smooth_scores.append(float(eligible["avg_r"].median()) if not eligible.empty else -999.0)
        smooth_trade_counts.append(int(eligible["trades"].median()) if not eligible.empty else 0)
    table["selection_score"] = smooth_scores
    table["smooth_neighborhood_trades"] = smooth_trade_counts

    best = table.sort_values(["selection_score", "trades", "net_r"], ascending=[False, False, False]).iloc[0]
    all_row = table[table["gate"].eq("all")].iloc[0]
    if best["gate"] != "all" and float(best["selection_score"]) < float(baseline_metrics["avg_r"]) + float(min_edge_r):
        best = all_row
    out = best.to_dict()
    out["threshold_table"] = table
    out.update({f"baseline_{key}": value for key, value in baseline_metrics.items()})
    return out


def summarize(scope: str, model_name: str, feature_set: str, scored: pd.DataFrame, decisions: pd.DataFrame, portfolio_mode: str) -> dict[str, Any]:
    baseline = portfolio(top_action_per_event(scored), portfolio_mode)
    selected = selected_trades(decisions)
    rejected_events = int((~decisions["selected"]).sum()) if not decisions.empty and "selected" in decisions.columns else 0
    row: dict[str, Any] = {
        "scope": scope,
        "model": model_name,
        "feature_set": feature_set,
        "portfolio": portfolio_mode,
        "events": int(len(decisions)),
        "rejected_events": rejected_events,
    }
    row.update(with_prefix("baseline", trade_metrics(baseline)))
    row.update(with_prefix("selected", trade_metrics(selected)))
    row["delta_avg_r"] = row["selected_avg_r"] - row["baseline_avg_r"]
    row["delta_net_r"] = row["selected_net_r"] - row["baseline_net_r"]
    row["kept_share"] = row["selected_trades"] / row["baseline_trades"] if row["baseline_trades"] else 0.0
    return row


def aggregate_groups(decisions: pd.DataFrame, scored: pd.DataFrame, *, scope: str, model_name: str, feature_set: str, group_column: str, portfolio_mode: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if decisions.empty or group_column not in decisions.columns:
        return pd.DataFrame()
    selected_ids = set(selected_trades(decisions)["candidate_id"].astype(str))
    baseline = portfolio(top_action_per_event(scored), portfolio_mode)
    for value, group in baseline.groupby(group_column, dropna=False):
        selected_group = group[group["candidate_id"].astype(str).isin(selected_ids)].copy()
        row: dict[str, Any] = {
            "scope": scope,
            "model": model_name,
            "feature_set": feature_set,
            group_column: value,
            "portfolio": portfolio_mode,
        }
        row.update(with_prefix("baseline", trade_metrics(group)))
        row.update(with_prefix("selected", trade_metrics(selected_group)))
        row["delta_avg_r"] = row["selected_avg_r"] - row["baseline_avg_r"]
        row["delta_net_r"] = row["selected_net_r"] - row["baseline_net_r"]
        row["kept_share"] = row["selected_trades"] / row["baseline_trades"] if row["baseline_trades"] else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def run_pooled(dataset: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    folds = fold_schedule(dataset, args.first_test_year)
    thresholds = parse_csv_values(args.thresholds, float)
    keep_fracs = parse_csv_values(args.keep_fracs, float)
    summary_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    threshold_parts: list[pd.DataFrame] = []
    decision_parts: list[pd.DataFrame] = []
    group_parts: list[pd.DataFrame] = []

    for feature_set in parse_csv_values(args.feature_sets, str):
        numeric, categorical = feature_columns(feature_set)
        features = numeric + categorical
        for model_name in available_models(args.models):
            model_scored: list[pd.DataFrame] = []
            model_decisions: list[pd.DataFrame] = []
            for fold_index, (_, val_year, test_year) in enumerate(folds, start=1):
                train = dataset[dataset["entry_year"] < val_year].copy()
                validation = dataset[dataset["entry_year"] == val_year].copy()
                test = dataset[dataset["entry_year"] == test_year].copy()
                if len(train) < args.min_train_rows or len(validation) < args.min_val_rows or test.empty:
                    continue
                model = build_regressor(model_name, numeric, categorical)
                fit_regressor(model, train, features)
                validation = score_frame(model, validation, features)
                test = score_frame(model, test, features)
                gate = choose_threshold(
                    validation,
                    thresholds=thresholds,
                    keep_fracs=keep_fracs,
                    portfolio_mode=args.portfolio,
                    min_val_trades=args.min_val_trades,
                    smooth_radius=args.smooth_radius,
                    min_edge_r=args.min_edge_r,
                )
                threshold_table = gate.pop("threshold_table", pd.DataFrame())
                if not threshold_table.empty:
                    threshold_table.insert(0, "scope", "pooled_annual")
                    threshold_table.insert(1, "fold", fold_index)
                    threshold_table.insert(2, "model", model_name)
                    threshold_table.insert(3, "feature_set", feature_set)
                    threshold_table.insert(4, "val_year", val_year)
                    threshold_table.insert(5, "test_year", test_year)
                    threshold_parts.append(threshold_table)
                threshold = float(gate["threshold"])
                decisions = decision_frame(test, threshold, args.portfolio)
                for frame in [test, decisions]:
                    frame["scope"] = "pooled_annual"
                    frame["fold"] = fold_index
                    frame["model"] = model_name
                    frame["feature_set"] = feature_set
                    frame["val_year"] = val_year
                    frame["test_year"] = test_year
                    frame["selected_gate"] = gate["gate"]
                    frame["selected_threshold"] = threshold

                row = {
                    "scope": "pooled_annual",
                    "fold": fold_index,
                    "model": model_name,
                    "feature_set": feature_set,
                    "train_years": f"{int(train['entry_year'].min())}-{int(train['entry_year'].max())}",
                    "val_year": int(val_year),
                    "test_year": int(test_year),
                    "train_rows": int(len(train)),
                    "val_rows": int(len(validation)),
                    "test_rows": int(len(test)),
                    "gate": gate["gate"],
                    "threshold": threshold,
                    "smooth_score": float(gate.get("selection_score", math.nan)),
                    "smooth_neighborhood_trades": int(gate.get("smooth_neighborhood_trades", 0)),
                }
                row.update(with_prefix("val_baseline", trade_metrics(portfolio(top_action_per_event(validation), args.portfolio))))
                row.update(with_prefix("val_selected", trade_metrics(selected_trades(decision_frame(validation, threshold, args.portfolio)))))
                row.update(with_prefix("test_baseline", trade_metrics(portfolio(top_action_per_event(test), args.portfolio))))
                row.update(with_prefix("test_selected", trade_metrics(selected_trades(decisions))))
                fold_rows.append(row)
                model_scored.append(test)
                model_decisions.append(decisions)
                decision_parts.append(decisions)

            scored_model = pd.concat(model_scored, ignore_index=True) if model_scored else pd.DataFrame()
            decisions_model = pd.concat(model_decisions, ignore_index=True) if model_decisions else pd.DataFrame()
            summary_rows.append(summarize("pooled_annual", model_name, feature_set, scored_model, decisions_model, args.portfolio))
            for group_column in [
                "symbol",
                "family",
                "pattern_name",
                "entry_mode",
                "time_filter",
                "target_rr_planned",
                "peak_spacing",
                "stop_atr_buffer",
                "breakeven_trigger_r",
                "min_harmonic_quality_score",
                "action_key",
            ]:
                group_parts.append(
                    aggregate_groups(
                        decisions_model,
                        scored_model,
                        scope="pooled_annual",
                        model_name=model_name,
                        feature_set=feature_set,
                        group_column=group_column,
                        portfolio_mode=args.portfolio,
                    )
                )

    summary = pd.DataFrame(summary_rows).sort_values(["delta_avg_r", "selected_avg_r"], ascending=[False, False])
    folds_out = pd.DataFrame(fold_rows)
    thresholds_out = pd.concat(threshold_parts, ignore_index=True) if threshold_parts else pd.DataFrame()
    decisions_out = pd.concat(decision_parts, ignore_index=True) if decision_parts else pd.DataFrame()
    groups_out = pd.concat([part for part in group_parts if not part.empty], ignore_index=True) if group_parts else pd.DataFrame()
    return summary, groups_out, folds_out, thresholds_out, decisions_out


def run_symbol_holdout(dataset: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    folds = fold_schedule(dataset, args.first_test_year)
    thresholds = parse_csv_values(args.thresholds, float)
    keep_fracs = parse_csv_values(args.keep_fracs, float)
    symbols = sorted(dataset["symbol"].dropna().astype(str).unique())
    summary_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    threshold_parts: list[pd.DataFrame] = []
    decision_parts: list[pd.DataFrame] = []
    group_parts: list[pd.DataFrame] = []

    for feature_set in parse_csv_values(args.feature_sets, str):
        numeric, categorical = feature_columns(feature_set)
        features = numeric + categorical
        for model_name in available_models(args.models):
            model_scored: list[pd.DataFrame] = []
            model_decisions: list[pd.DataFrame] = []
            for holdout_symbol in symbols:
                for fold_index, (_, val_year, test_year) in enumerate(folds, start=1):
                    train = dataset[(dataset["entry_year"] < val_year) & (dataset["symbol"] != holdout_symbol)].copy()
                    validation = dataset[(dataset["entry_year"] == val_year) & (dataset["symbol"] != holdout_symbol)].copy()
                    test = dataset[(dataset["entry_year"] == test_year) & (dataset["symbol"] == holdout_symbol)].copy()
                    if (
                        len(train) < args.min_train_rows
                        or len(validation) < args.min_val_rows
                        or len(test) < args.min_holdout_test_rows
                    ):
                        continue
                    model = build_regressor(model_name, numeric, categorical)
                    fit_regressor(model, train, features)
                    validation = score_frame(model, validation, features)
                    test = score_frame(model, test, features)
                    gate = choose_threshold(
                        validation,
                        thresholds=thresholds,
                        keep_fracs=keep_fracs,
                        portfolio_mode=args.portfolio,
                        min_val_trades=args.min_val_trades,
                        smooth_radius=args.smooth_radius,
                        min_edge_r=args.min_edge_r,
                    )
                    threshold_table = gate.pop("threshold_table", pd.DataFrame())
                    if not threshold_table.empty:
                        threshold_table.insert(0, "scope", "leave_one_symbol_annual")
                        threshold_table.insert(1, "fold", fold_index)
                        threshold_table.insert(2, "model", model_name)
                        threshold_table.insert(3, "feature_set", feature_set)
                        threshold_table.insert(4, "holdout_symbol", holdout_symbol)
                        threshold_table.insert(5, "val_year", val_year)
                        threshold_table.insert(6, "test_year", test_year)
                        threshold_parts.append(threshold_table)
                    threshold = float(gate["threshold"])
                    decisions = decision_frame(test, threshold, args.portfolio)
                    for frame in [test, decisions]:
                        frame["scope"] = "leave_one_symbol_annual"
                        frame["fold"] = fold_index
                        frame["model"] = model_name
                        frame["feature_set"] = feature_set
                        frame["holdout_symbol"] = holdout_symbol
                        frame["val_year"] = val_year
                        frame["test_year"] = test_year
                        frame["selected_gate"] = gate["gate"]
                        frame["selected_threshold"] = threshold

                    row = {
                        "scope": "leave_one_symbol_annual",
                        "fold": fold_index,
                        "model": model_name,
                        "feature_set": feature_set,
                        "holdout_symbol": holdout_symbol,
                        "train_years": f"{int(train['entry_year'].min())}-{int(train['entry_year'].max())}",
                        "val_year": int(val_year),
                        "test_year": int(test_year),
                        "train_rows": int(len(train)),
                        "val_rows": int(len(validation)),
                        "test_rows": int(len(test)),
                        "gate": gate["gate"],
                        "threshold": threshold,
                        "smooth_score": float(gate.get("selection_score", math.nan)),
                        "smooth_neighborhood_trades": int(gate.get("smooth_neighborhood_trades", 0)),
                    }
                    row.update(with_prefix("val_baseline", trade_metrics(portfolio(top_action_per_event(validation), args.portfolio))))
                    row.update(with_prefix("val_selected", trade_metrics(selected_trades(decision_frame(validation, threshold, args.portfolio)))))
                    row.update(with_prefix("test_baseline", trade_metrics(portfolio(top_action_per_event(test), args.portfolio))))
                    row.update(with_prefix("test_selected", trade_metrics(selected_trades(decisions))))
                    fold_rows.append(row)
                    model_scored.append(test)
                    model_decisions.append(decisions)
                    decision_parts.append(decisions)

            scored_model = pd.concat(model_scored, ignore_index=True) if model_scored else pd.DataFrame()
            decisions_model = pd.concat(model_decisions, ignore_index=True) if model_decisions else pd.DataFrame()
            summary_rows.append(
                summarize("leave_one_symbol_annual", model_name, feature_set, scored_model, decisions_model, args.portfolio)
            )
            for group_column in [
                "symbol",
                "family",
                "pattern_name",
                "entry_mode",
                "time_filter",
                "target_rr_planned",
                "peak_spacing",
                "stop_atr_buffer",
                "breakeven_trigger_r",
                "min_harmonic_quality_score",
                "action_key",
            ]:
                group_parts.append(
                    aggregate_groups(
                        decisions_model,
                        scored_model,
                        scope="leave_one_symbol_annual",
                        model_name=model_name,
                        feature_set=feature_set,
                        group_column=group_column,
                        portfolio_mode=args.portfolio,
                    )
                )

    summary = pd.DataFrame(summary_rows).sort_values(["delta_avg_r", "selected_avg_r"], ascending=[False, False])
    folds_out = pd.DataFrame(fold_rows)
    thresholds_out = pd.concat(threshold_parts, ignore_index=True) if threshold_parts else pd.DataFrame()
    decisions_out = pd.concat(decision_parts, ignore_index=True) if decision_parts else pd.DataFrame()
    groups_out = pd.concat([part for part in group_parts if not part.empty], ignore_index=True) if group_parts else pd.DataFrame()
    return summary, groups_out, folds_out, thresholds_out, decisions_out


def train_final_model(dataset: pd.DataFrame, args: argparse.Namespace) -> None:
    model_name = args.final_model.strip().lower()
    feature_set = args.final_feature_set.strip().lower()
    numeric, categorical = feature_columns(feature_set)
    features = numeric + categorical
    model = build_regressor(model_name, numeric, categorical)
    fit_regressor(model, dataset, features)
    payload = {
        "model": model,
        "model_name": model_name,
        "feature_set": feature_set,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "portfolio": args.portfolio,
        "symbols": parse_symbols(args.symbols),
        "trained_rows": int(len(dataset)),
        "config": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
    }
    output_path = args.output_prefix.with_name(args.output_prefix.name + f"_{feature_set}_{model_name}.joblib")
    joblib.dump(payload, output_path)
    print(f"Saved final action selector {output_path}", flush=True)


def write_table(path: Path, table: pd.DataFrame) -> None:
    table.to_csv(path, index=False)
    print(f"Saved {path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expected-R multi-action selector for pyharmonics events.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-prefix", type=Path, default=Path("scripts/pyharmonics_action_selector_20260604"))
    parser.add_argument("--symbols", default="", help="Optional comma-separated symbol filter; empty uses all dataset symbols.")
    parser.add_argument("--families", default="")
    parser.add_argument("--pattern-name-filters", default="")
    parser.add_argument("--entry-modes", default="next_open,trigger_break,trigger_close_break")
    parser.add_argument("--time-filters", default="all,eu_us")
    parser.add_argument("--peak-spacings", default="10,16")
    parser.add_argument("--rrs", default="2.0,2.5")
    parser.add_argument("--stop-buffers", default="")
    parser.add_argument("--breakeven-triggers", default="")
    parser.add_argument("--min-quality-scores", default="")
    parser.add_argument("--models", default="ridge,rf,extratrees,hgb,xgb")
    parser.add_argument("--feature-sets", default="no_symbol,with_symbol")
    parser.add_argument("--thresholds", default="-0.20,-0.10,0.00,0.05,0.10,0.15,0.20,0.30,0.40,0.50")
    parser.add_argument("--keep-fracs", default="0.20,0.30,0.40,0.50,0.65,0.80")
    parser.add_argument("--portfolio", choices=["raw", "events", "one_trade", "one_symbol"], default="one_symbol")
    parser.add_argument("--first-test-year", type=int, default=2024)
    parser.add_argument("--min-train-rows", type=int, default=100)
    parser.add_argument("--min-val-rows", type=int, default=40)
    parser.add_argument("--min-val-trades", type=int, default=5)
    parser.add_argument("--min-holdout-test-rows", type=int, default=3)
    parser.add_argument("--smooth-radius", type=int, default=1)
    parser.add_argument("--min-edge-r", type=float, default=0.02)
    parser.add_argument("--skip-symbol-holdout", action="store_true")
    parser.add_argument("--write-final-model", action="store_true")
    parser.add_argument("--final-model", default="hgb")
    parser.add_argument("--final-feature-set", default="with_symbol")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn is required for the action selector.")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args)
    write_table(args.output_prefix.with_name(args.output_prefix.name + "_dataset.csv"), dataset)

    pooled_summary, pooled_groups, pooled_folds, pooled_thresholds, pooled_decisions = run_pooled(dataset, args)
    write_table(args.output_prefix.with_name(args.output_prefix.name + "_pooled_summary.csv"), pooled_summary)
    write_table(args.output_prefix.with_name(args.output_prefix.name + "_pooled_groups.csv"), pooled_groups)
    write_table(args.output_prefix.with_name(args.output_prefix.name + "_pooled_folds.csv"), pooled_folds)
    write_table(args.output_prefix.with_name(args.output_prefix.name + "_pooled_thresholds.csv"), pooled_thresholds)
    write_table(args.output_prefix.with_name(args.output_prefix.name + "_pooled_decisions.csv"), pooled_decisions)

    if not args.skip_symbol_holdout:
        holdout_summary, holdout_groups, holdout_folds, holdout_thresholds, holdout_decisions = run_symbol_holdout(dataset, args)
        write_table(args.output_prefix.with_name(args.output_prefix.name + "_symbol_holdout_summary.csv"), holdout_summary)
        write_table(args.output_prefix.with_name(args.output_prefix.name + "_symbol_holdout_groups.csv"), holdout_groups)
        write_table(args.output_prefix.with_name(args.output_prefix.name + "_symbol_holdout_folds.csv"), holdout_folds)
        write_table(args.output_prefix.with_name(args.output_prefix.name + "_symbol_holdout_thresholds.csv"), holdout_thresholds)
        write_table(args.output_prefix.with_name(args.output_prefix.name + "_symbol_holdout_decisions.csv"), holdout_decisions)

    config_path = args.output_prefix.with_name(args.output_prefix.name + "_config.json")
    config_payload = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
    config_path.write_text(json.dumps(config_payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved {config_path}", flush=True)

    if args.write_final_model:
        train_final_model(dataset, args)

    print("\nPooled annual action-selector summary:")
    print(pooled_summary.to_string(index=False))
    if not args.skip_symbol_holdout:
        print("\nLeave-one-symbol annual action-selector summary:")
        print(holdout_summary.to_string(index=False))


if __name__ == "__main__":
    main()
