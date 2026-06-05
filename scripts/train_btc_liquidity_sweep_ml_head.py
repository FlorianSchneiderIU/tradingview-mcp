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
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - only for environments without sklearn
    SKLEARN_AVAILABLE = False


NUMERIC_FEATURES = [
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
    "risk_pct",
    "entry_vs_level_atr",
    "stop_distance_atr",
    "direction_sign",
    "trend_aligned_sign",
]

CATEGORICAL_FEATURES = [
    "direction",
    "session",
    "trend_side",
]


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


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
        estimator = LogisticRegression(C=0.45, class_weight="balanced", max_iter=2000, random_state=31)
    elif name == "rf":
        estimator = RandomForestClassifier(
            n_estimators=300,
            max_depth=3,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            random_state=31,
        )
    elif name == "hgb":
        estimator = HistGradientBoostingClassifier(
            learning_rate=0.035,
            max_iter=90,
            max_leaf_nodes=7,
            l2_regularization=0.7,
            random_state=31,
        )
    else:
        raise ValueError(f"Unknown model: {name}")
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def trade_metrics(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {
            "trades": 0,
            "net_r": 0.0,
            "avg_r": 0.0,
            "median_r": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_dd_r": 0.0,
        }
    result = pd.to_numeric(frame["result_r"], errors="coerce").fillna(0.0)
    wins = result[result > 0.0].sum()
    losses = result[result < 0.0].sum()
    equity = result.cumsum()
    drawdown = equity - equity.cummax()
    pf = math.inf if losses == 0.0 and wins > 0.0 else (float(wins / abs(losses)) if losses < 0.0 else 0.0)
    return {
        "trades": int(len(frame)),
        "net_r": float(result.sum()),
        "avg_r": float(result.mean()),
        "median_r": float(result.median()),
        "win_rate": float((result > 0.0).mean()),
        "profit_factor": pf,
        "max_dd_r": float(drawdown.min()) if len(drawdown) else 0.0,
    }


def classifier_metrics(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty or frame["label"].nunique() < 2:
        return {"auc": math.nan, "brier": math.nan}
    labels = frame["label"].astype(int)
    probs = pd.to_numeric(frame["ml_prob"], errors="coerce").clip(0.0, 1.0)
    return {
        "auc": float(roc_auc_score(labels, probs)),
        "brier": float(brier_score_loss(labels, probs)),
    }


def with_prefix(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def truthy(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def load_dataset(args: argparse.Namespace) -> pd.DataFrame:
    events = pd.read_csv(args.events)
    trades = pd.read_csv(args.trades)
    for frame in [events, trades]:
        for column in ["event_time", "entry_time", "exit_time"]:
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")

    trades = trades[
        (trades["candidate_filter"].astype(str) == args.candidate_filter)
        & (trades["config"].astype(str) == args.config)
        & (pd.to_numeric(trades["stop_buffer_atr"], errors="coerce").round(6) == round(float(args.stop_buffer_atr), 6))
    ].copy()
    if trades.empty:
        raise RuntimeError("No preferred backtest trades matched the requested config.")

    merge_cols = ["level_id", "entry_time", "direction"]
    event_cols = [
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
        raise RuntimeError(f"Failed to merge event features for {missing} trades.")

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
    merged = merged.sort_values("entry_time").reset_index(drop=True)
    return merged


def score_frame(model: Pipeline, frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["ml_prob"] = model.predict_proba(out[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1]
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


def choose_gate(validation: pd.DataFrame, thresholds: list[float], keep_fracs: list[float], min_val_trades: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for _, candidate in candidate_thresholds(validation, thresholds, keep_fracs).iterrows():
        selected = validation[pd.to_numeric(validation["ml_prob"], errors="coerce") >= float(candidate["threshold"])].copy()
        metrics = trade_metrics(selected)
        penalty = min(1.0, metrics["trades"] / max(float(min_val_trades), 1.0))
        score = metrics["avg_r"] * penalty if metrics["trades"] >= min_val_trades else -999.0
        row = candidate.to_dict()
        row.update(metrics)
        row["selection_score"] = float(score)
        rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        return {"gate": "all", "threshold": -math.inf, "keep_frac": 1.0}
    best = table.sort_values(["selection_score", "trades", "net_r"], ascending=[False, False, False]).iloc[0].to_dict()
    best["threshold_table"] = table
    return best


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
        if not train_years:
            continue
        folds.append((max(train_years), val_year, test_year))
    return folds


def one_trade_at_a_time(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    kept: list[int] = []
    active_until: pd.Timestamp | None = None
    ordered = frame.sort_values(["entry_time", "exit_time"]).reset_index(drop=True)
    for idx, row in ordered.iterrows():
        entry_time = pd.Timestamp(row["entry_time"])
        if active_until is not None and entry_time < active_until:
            continue
        kept.append(idx)
        active_until = pd.Timestamp(row["exit_time"])
    return ordered.loc[kept].reset_index(drop=True)


def aggregate_summary(scored: pd.DataFrame, selected: pd.DataFrame, model_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for portfolio, all_frame, selected_frame in [
        ("all_signals", scored, selected),
        ("one_trade_at_a_time", one_trade_at_a_time(scored), one_trade_at_a_time(selected)),
    ]:
        row: dict[str, Any] = {"model": model_name, "portfolio": portfolio}
        row.update(with_prefix("all", trade_metrics(all_frame)))
        row.update(with_prefix("selected", trade_metrics(selected_frame)))
        row["delta_avg_r"] = row["selected_avg_r"] - row["all_avg_r"]
        row["delta_net_r"] = row["selected_net_r"] - row["all_net_r"]
        row["kept_share"] = row["selected_trades"] / row["all_trades"] if row["all_trades"] else 0.0
        rows.append(row)
    return rows


def run_walk_forward(dataset: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
            model.fit(train[NUMERIC_FEATURES + CATEGORICAL_FEATURES], train["label"].astype(int))
            validation = score_frame(model, validation)
            test = score_frame(model, test)
            gate = choose_gate(validation, thresholds, keep_fracs, args.min_val_trades)
            threshold_table = gate.pop("threshold_table", pd.DataFrame())
            if not threshold_table.empty:
                threshold_table.insert(0, "fold", fold_index)
                threshold_table.insert(1, "model", model_name)
                threshold_table.insert(2, "val_year", val_year)
                threshold_table.insert(3, "test_year", test_year)
                threshold_rows.append(threshold_table)

            threshold = float(gate["threshold"])
            selected_test = test[pd.to_numeric(test["ml_prob"], errors="coerce") >= threshold].copy()
            test["fold"] = fold_index
            test["model"] = model_name
            test["val_year"] = val_year
            test["test_year"] = test_year
            test["selected_gate"] = gate["gate"]
            test["selected_threshold"] = threshold
            selected_test["fold"] = fold_index
            selected_test["model"] = model_name
            selected_test["val_year"] = val_year
            selected_test["test_year"] = test_year
            selected_test["selected_gate"] = gate["gate"]
            selected_test["selected_threshold"] = threshold

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

        scored_model = pd.concat(model_scored_parts, ignore_index=True) if model_scored_parts else pd.DataFrame()
        selected_model = pd.concat(model_selected_parts, ignore_index=True) if model_selected_parts else pd.DataFrame()
        summary_rows.extend(aggregate_summary(scored_model, selected_model, model_name))

    folds_out = pd.DataFrame(fold_rows)
    thresholds_out = pd.concat(threshold_rows, ignore_index=True) if threshold_rows else pd.DataFrame()
    scored_out = pd.concat(scored_parts, ignore_index=True) if scored_parts else pd.DataFrame()
    selected_out = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    summary_out = pd.DataFrame(summary_rows).sort_values(["portfolio", "delta_avg_r", "selected_avg_r"], ascending=[True, False, False])
    return summary_out, folds_out, thresholds_out, pd.concat([scored_out.assign(_selected_oos=False), selected_out.assign(_selected_oos=True)], ignore_index=True)


def train_final_model(dataset: pd.DataFrame, model_name: str, output_path: Path, args: argparse.Namespace) -> None:
    model = build_model(model_name)
    model.fit(dataset[NUMERIC_FEATURES + CATEGORICAL_FEATURES], dataset["label"].astype(int))
    payload = {
        "model": model,
        "model_name": model_name,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "candidate_filter": args.candidate_filter,
        "config": args.config,
        "stop_buffer_atr": float(args.stop_buffer_atr),
        "label_min_r": float(args.label_min_r),
        "trained_rows": int(len(dataset)),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward ML head for BTC 4h liquidity sweep preferred setup.")
    parser.add_argument("--events", type=Path, default=Path("scripts/btc_liquidity_sweep_study/btc_liquidity_sweep_events.csv"))
    parser.add_argument("--trades", type=Path, default=Path("scripts/btc_liquidity_sweep_study/preferred_backtest_trades.csv"))
    parser.add_argument("--output-prefix", type=Path, default=Path("scripts/btc_liquidity_sweep_study/btc_liquidity_sweep_ml_head"))
    parser.add_argument("--candidate-filter", default="4h_session_trend")
    parser.add_argument("--config", default="partial_2p5_7p5")
    parser.add_argument("--stop-buffer-atr", type=float, default=0.15)
    parser.add_argument("--label-min-r", type=float, default=0.0)
    parser.add_argument("--models", default="logreg,rf,hgb")
    parser.add_argument("--thresholds", default="0.40,0.45,0.50,0.55,0.60,0.65,0.70")
    parser.add_argument("--keep-fracs", default="0.35,0.50,0.65,0.80")
    parser.add_argument("--min-train-trades", type=int, default=40)
    parser.add_argument("--min-val-trades", type=int, default=8)
    parser.add_argument("--first-test-year", type=int, default=2024)
    parser.add_argument("--write-final-model", action="store_true")
    parser.add_argument("--final-model-name", choices=["logreg", "rf", "hgb"], default="logreg")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn and joblib are required for this script.")
    dataset = load_dataset(args)
    summary, folds, thresholds, scored_and_selected = run_walk_forward(dataset, args)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    dataset_path = args.output_prefix.with_name(args.output_prefix.name + "_dataset.csv")
    summary_path = args.output_prefix.with_name(args.output_prefix.name + "_summary.csv")
    folds_path = args.output_prefix.with_name(args.output_prefix.name + "_folds.csv")
    thresholds_path = args.output_prefix.with_name(args.output_prefix.name + "_thresholds.csv")
    scored_path = args.output_prefix.with_name(args.output_prefix.name + "_scored_selected.csv")
    config_path = args.output_prefix.with_name(args.output_prefix.name + "_config.json")
    dataset.to_csv(dataset_path, index=False)
    summary.to_csv(summary_path, index=False)
    folds.to_csv(folds_path, index=False)
    thresholds.to_csv(thresholds_path, index=False)
    scored_and_selected.to_csv(scored_path, index=False)
    config_payload = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
    config_path.write_text(json.dumps(config_payload, indent=2, sort_keys=True), encoding="utf-8")

    if args.write_final_model:
        model_path = args.output_prefix.with_name(args.output_prefix.name + f"_{args.final_model_name}.joblib")
        train_final_model(dataset, args.final_model_name, model_path, args)
        print(f"Saved final model to {model_path}")

    print(summary.to_string(index=False))
    print(f"\nSaved summary to {summary_path}")
    print(f"Saved folds to {folds_path}")
    print(f"Saved scored rows to {scored_path}")


if __name__ == "__main__":
    main()
