from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import joblib
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - only for environments without sklearn
    SKLEARN_AVAILABLE = False

from scripts.train_liquidity_sweep_universe_ml_head import (
    CATEGORICAL_FEATURES,
    DEFAULT_SYMBOLS,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    aggregate_by_symbol,
    fold_schedule,
    load_dataset,
    one_trade_at_a_time,
    one_trade_per_symbol_at_a_time,
    parse_float_list,
    parse_symbols,
    trade_metrics,
    with_prefix,
)


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
    if name == "ridge":
        estimator = Ridge(alpha=20.0)
    elif name == "rfreg":
        estimator = RandomForestRegressor(
            n_estimators=500,
            max_depth=4,
            min_samples_leaf=14,
            random_state=31,
        )
    elif name == "hgbreg":
        estimator = HistGradientBoostingRegressor(
            learning_rate=0.025,
            max_iter=120,
            max_leaf_nodes=7,
            min_samples_leaf=16,
            l2_regularization=1.0,
            random_state=31,
        )
    else:
        raise ValueError(f"Unknown regression model: {name}")
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def score_frame(model: Pipeline, frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["pred_r"] = model.predict(out[FEATURE_COLUMNS])
    return out


def candidate_thresholds(validation: pd.DataFrame, thresholds: list[float], keep_fracs: list[float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [{"gate": "all", "threshold": -math.inf, "keep_frac": 1.0}]
    preds = pd.to_numeric(validation["pred_r"], errors="coerce").dropna()
    for threshold in thresholds:
        rows.append({"gate": f"pred_ge_{threshold:.2f}", "threshold": float(threshold), "keep_frac": math.nan})
    for keep_frac in keep_fracs:
        keep_frac = min(max(float(keep_frac), 0.0), 1.0)
        if preds.empty:
            continue
        rows.append({"gate": f"val_top_{keep_frac:.2f}", "threshold": float(preds.quantile(1.0 - keep_frac)), "keep_frac": keep_frac})
    return pd.DataFrame(rows).drop_duplicates(["gate", "threshold"]).sort_values("threshold").reset_index(drop=True)


def choose_stable_gate(
    validation: pd.DataFrame,
    thresholds: list[float],
    keep_fracs: list[float],
    *,
    min_val_trades: int,
    smooth_radius: int,
    objective: str,
    min_edge: float,
) -> dict[str, Any]:
    score_col = "net_r" if objective == "net" else "avg_r"
    rows: list[dict[str, Any]] = []
    for _, candidate in candidate_thresholds(validation, thresholds, keep_fracs).iterrows():
        selected = validation[pd.to_numeric(validation["pred_r"], errors="coerce") >= float(candidate["threshold"])]
        metrics = trade_metrics(selected)
        row = candidate.to_dict()
        row.update(metrics)
        row["eligible"] = bool(metrics["trades"] >= min_val_trades)
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)

    scores: list[float] = []
    smooth_trades: list[int] = []
    for idx in range(len(table)):
        neighborhood = table.iloc[max(0, idx - smooth_radius) : min(len(table), idx + smooth_radius + 1)]
        eligible = neighborhood[neighborhood["eligible"]]
        scores.append(float(eligible[score_col].median()) if not eligible.empty else -999.0)
        smooth_trades.append(int(eligible["trades"].median()) if not eligible.empty else 0)
    table["selection_score"] = scores
    table["smooth_neighborhood_trades"] = smooth_trades

    all_row = table[table["gate"].eq("all")].iloc[0]
    best = table.sort_values(["selection_score", "trades", "net_r"], ascending=[False, False, False]).iloc[0]
    if best["gate"] != "all" and float(best["selection_score"]) < float(all_row[score_col]) + min_edge:
        best = all_row
    out = best.to_dict()
    out["threshold_table"] = table
    return out


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


def run_walk_forward(dataset: pd.DataFrame, args: argparse.Namespace, *, holdout: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    thresholds = parse_float_list(args.thresholds)
    keep_fracs = parse_float_list(args.keep_fracs)
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    folds = fold_schedule(dataset, args.first_test_year)
    symbols = sorted(dataset["symbol"].dropna().astype(str).unique()) if holdout else [None]
    scope = "regression_leave_one_symbol_annual" if holdout else "regression_pooled_annual"

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
                if holdout_symbol is None:
                    train = dataset[dataset["entry_year"] < val_year].copy()
                    validation = dataset[dataset["entry_year"] == val_year].copy()
                    test = dataset[dataset["entry_year"] == test_year].copy()
                else:
                    train = dataset[(dataset["entry_year"] < val_year) & (dataset["symbol"] != holdout_symbol)].copy()
                    validation = dataset[(dataset["entry_year"] == val_year) & (dataset["symbol"] != holdout_symbol)].copy()
                    test = dataset[(dataset["entry_year"] == test_year) & (dataset["symbol"] == holdout_symbol)].copy()

                min_test = args.min_holdout_test_trades if holdout_symbol is not None else 1
                if len(train) < args.min_train_trades or len(validation) < args.min_val_trades or len(test) < min_test:
                    continue

                model = build_model(model_name)
                model.fit(train[FEATURE_COLUMNS], pd.to_numeric(train["result_r"], errors="coerce").fillna(0.0))
                validation = score_frame(model, validation)
                test = score_frame(model, test)
                gate = choose_stable_gate(
                    validation,
                    thresholds,
                    keep_fracs,
                    min_val_trades=args.min_val_trades,
                    smooth_radius=args.smooth_radius,
                    objective=args.gate_objective,
                    min_edge=args.min_edge_r,
                )
                threshold_table = gate.pop("threshold_table", pd.DataFrame())
                if not threshold_table.empty:
                    threshold_table.insert(0, "fold", fold_index)
                    threshold_table.insert(1, "model", model_name)
                    if holdout_symbol is not None:
                        threshold_table.insert(2, "holdout_symbol", holdout_symbol)
                    threshold_table.insert(3 if holdout_symbol is not None else 2, "val_year", val_year)
                    threshold_table.insert(4 if holdout_symbol is not None else 3, "test_year", test_year)
                    threshold_rows.append(threshold_table)

                threshold = float(gate["threshold"])
                selected_test = test[pd.to_numeric(test["pred_r"], errors="coerce") >= threshold].copy()
                for frame in [test, selected_test]:
                    frame["fold"] = fold_index
                    frame["model"] = model_name
                    frame["val_year"] = val_year
                    frame["test_year"] = test_year
                    frame["selected_gate"] = gate["gate"]
                    frame["selected_threshold"] = threshold
                    if holdout_symbol is not None:
                        frame["holdout_symbol"] = holdout_symbol

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
                if holdout_symbol is not None:
                    row["holdout_symbol"] = holdout_symbol
                row.update(with_prefix("val_all", trade_metrics(validation)))
                row.update(with_prefix("val_selected", trade_metrics(validation[validation["pred_r"] >= threshold])))
                row.update(with_prefix("test_all", trade_metrics(test)))
                row.update(with_prefix("test_selected", trade_metrics(selected_test)))
                fold_rows.append(row)

                model_scored_parts.append(test)
                model_selected_parts.append(selected_test)
                scored_parts.append(test)
                selected_parts.append(selected_test)

        scored_model = pd.concat(model_scored_parts, ignore_index=False) if model_scored_parts else pd.DataFrame()
        selected_model = pd.concat(model_selected_parts, ignore_index=False) if model_selected_parts else pd.DataFrame()
        summary_rows.extend(aggregate_summary(scored_model, selected_model, model_name, scope))
        if not scored_model.empty:
            by_symbol_parts.append(aggregate_by_symbol(scored_model, selected_model, model_name, scope))

    summary = pd.DataFrame(summary_rows).sort_values(["portfolio", "delta_net_r", "delta_avg_r"], ascending=[True, False, False])
    by_symbol = pd.concat(by_symbol_parts, ignore_index=True) if by_symbol_parts else pd.DataFrame()
    folds_out = pd.DataFrame(fold_rows)
    thresholds_out = pd.concat(threshold_rows, ignore_index=True) if threshold_rows else pd.DataFrame()
    scored_out = pd.concat(scored_parts, ignore_index=False) if scored_parts else pd.DataFrame()
    selected_out = pd.concat(selected_parts, ignore_index=False) if selected_parts else pd.DataFrame()
    scored_selected = pd.concat(
        [scored_out.assign(_selected_oos=False), selected_out.assign(_selected_oos=True)],
        ignore_index=True,
    )
    return summary, by_symbol, folds_out, thresholds_out, scored_selected


def train_final_model(dataset: pd.DataFrame, model_name: str, output_path: Path, args: argparse.Namespace) -> None:
    model = build_model(model_name)
    model.fit(dataset[FEATURE_COLUMNS], pd.to_numeric(dataset["result_r"], errors="coerce").fillna(0.0))
    payload = {
        "model": model,
        "model_name": model_name,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "candidate_filter": args.candidate_filter,
        "config": args.config,
        "stop_buffer_atr": float(args.stop_buffer_atr),
        "symbols": parse_symbols(args.symbols),
        "trained_rows": int(len(dataset)),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward expected-R ML head for the liquidity sweep universe.")
    parser.add_argument("--input-dir", type=Path, default=Path("scripts/liquidity_sweep_universe"))
    parser.add_argument("--output-prefix", type=Path, default=Path("scripts/liquidity_sweep_universe/universe_liquidity_sweep_ml_head_regression"))
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--candidate-filter", default="4h_session_trend")
    parser.add_argument("--config", default="partial_2p5_7p5")
    parser.add_argument("--stop-buffer-atr", type=float, default=0.15)
    parser.add_argument("--label-min-r", type=float, default=0.0)
    parser.add_argument("--models", default="ridge,rfreg,hgbreg")
    parser.add_argument("--thresholds", default="-0.40,-0.25,-0.10,0.00,0.10,0.20,0.35,0.50,0.75,1.00")
    parser.add_argument("--keep-fracs", default="0.35,0.50,0.65,0.80,0.90")
    parser.add_argument("--gate-objective", choices=["net", "avg"], default="net")
    parser.add_argument("--min-train-trades", type=int, default=120)
    parser.add_argument("--min-val-trades", type=int, default=25)
    parser.add_argument("--min-holdout-test-trades", type=int, default=3)
    parser.add_argument("--first-test-year", type=int, default=2024)
    parser.add_argument("--smooth-radius", type=int, default=1)
    parser.add_argument("--min-edge-r", type=float, default=0.0)
    parser.add_argument("--skip-symbol-holdout", action="store_true")
    parser.add_argument("--write-final-model", action="store_true")
    parser.add_argument("--final-model-name", choices=["ridge", "rfreg", "hgbreg"], default="hgbreg")
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

    config_path = args.output_prefix.with_name(args.output_prefix.name + "_config.json")
    config_payload = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
    config_path.write_text(json.dumps(config_payload, indent=2, sort_keys=True), encoding="utf-8")

    pooled_summary, pooled_by_symbol, pooled_folds, pooled_thresholds, pooled_scored = run_walk_forward(dataset, args, holdout=False)
    write_table(args.output_prefix.with_name(args.output_prefix.name + "_pooled_summary.csv"), pooled_summary)
    write_table(args.output_prefix.with_name(args.output_prefix.name + "_pooled_summary_by_symbol.csv"), pooled_by_symbol)
    write_table(args.output_prefix.with_name(args.output_prefix.name + "_pooled_folds.csv"), pooled_folds)
    write_table(args.output_prefix.with_name(args.output_prefix.name + "_pooled_thresholds.csv"), pooled_thresholds)
    write_table(args.output_prefix.with_name(args.output_prefix.name + "_pooled_scored_selected.csv"), pooled_scored)

    if not args.skip_symbol_holdout:
        holdout_summary, holdout_by_symbol, holdout_folds, holdout_thresholds, holdout_scored = run_walk_forward(dataset, args, holdout=True)
        write_table(args.output_prefix.with_name(args.output_prefix.name + "_symbol_holdout_summary.csv"), holdout_summary)
        write_table(args.output_prefix.with_name(args.output_prefix.name + "_symbol_holdout_summary_by_symbol.csv"), holdout_by_symbol)
        write_table(args.output_prefix.with_name(args.output_prefix.name + "_symbol_holdout_folds.csv"), holdout_folds)
        write_table(args.output_prefix.with_name(args.output_prefix.name + "_symbol_holdout_thresholds.csv"), holdout_thresholds)
        write_table(args.output_prefix.with_name(args.output_prefix.name + "_symbol_holdout_scored_selected.csv"), holdout_scored)

    if args.write_final_model:
        model_path = args.output_prefix.with_name(args.output_prefix.name + f"_{args.final_model_name}.joblib")
        train_final_model(dataset, args.final_model_name, model_path, args)
        print(f"Saved final model to {model_path}", flush=True)

    print("\nRegression pooled annual summary:")
    print(pooled_summary.to_string(index=False))
    if not args.skip_symbol_holdout:
        print("\nRegression leave-one-symbol annual summary:")
        print(holdout_summary.to_string(index=False))


if __name__ == "__main__":
    main()
