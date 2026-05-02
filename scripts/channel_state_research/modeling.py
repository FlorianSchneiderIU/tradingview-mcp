from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    ColumnTransformer = None
    HistGradientBoostingClassifier = None
    RandomForestClassifier = None
    SimpleImputer = None
    permutation_importance = None
    LogisticRegression = None
    Pipeline = None
    StandardScaler = None
    SKLEARN_AVAILABLE = False

from scripts.channel_state_research.backtest import SignalGateSpec, choose_signal_direction, simulate_threshold_strategy, strategy_metrics


@dataclass(frozen=True)
class WalkForwardSpec:
    model_name: str = "hgb"
    feature_group_names: tuple[str, ...] = (
        "structural",
        "position",
        "excursion_acceptance",
        "touch_interaction",
        "swing_state",
        "channel_evolution",
        "confluence",
        "regime",
    )
    channel_family: str = "both"
    timeframes: tuple[str, ...] = ("1h", "4h", "1d", "1w")
    decision_timeframe: str = "1h"
    train_months: int = 24
    val_months: int = 6
    test_months: int = 6
    embargo_bars: int = 24
    alpha: float = 1.5
    beta: float = 1.5
    horizon_bars: int = 24
    threshold_mode: str = "absolute"
    long_score_mode: str = "probability"
    short_score_mode: str = "probability"
    long_thresholds: tuple[float, ...] = (0.55, 0.60, 0.65)
    short_thresholds: tuple[float, ...] = (0.55, 0.60, 0.65)
    probability_gap_values: tuple[float, ...] = (0.0,)
    gate_presets: tuple[str, ...] = ("none",)
    fee_bps_side: float = 5.0
    slippage_bps_side: float = 2.0
    risk_fraction: float = 0.01
    min_validation_trades: int = 3
    compute_feature_importance: bool = True


@dataclass(frozen=True)
class FallbackLogisticModel:
    columns: list[str]
    median: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float


def build_walkforward_folds(
    dataset: pd.DataFrame,
    spec: WalkForwardSpec,
    *,
    time_column: str = "decision_time",
) -> list[dict[str, pd.Timestamp]]:
    ordered = dataset.sort_values(time_column)
    first_time = pd.Timestamp(ordered[time_column].iloc[0]).tz_convert("UTC").floor("D")
    last_time = pd.Timestamp(ordered[time_column].iloc[-1]).tz_convert("UTC").ceil("D")

    folds: list[dict[str, pd.Timestamp]] = []
    train_start = first_time
    train_end = train_start + pd.DateOffset(months=spec.train_months)
    val_end = train_end + pd.DateOffset(months=spec.val_months)
    test_end = val_end + pd.DateOffset(months=spec.test_months)

    while test_end <= last_time:
        folds.append(
            {
                "train_start": pd.Timestamp(train_start).tz_convert("UTC"),
                "train_end": pd.Timestamp(train_end).tz_convert("UTC"),
                "val_end": pd.Timestamp(val_end).tz_convert("UTC"),
                "test_end": pd.Timestamp(test_end).tz_convert("UTC"),
            }
        )
        train_end = train_end + pd.DateOffset(months=spec.test_months)
        val_end = val_end + pd.DateOffset(months=spec.test_months)
        test_end = test_end + pd.DateOffset(months=spec.test_months)

    return folds


def select_feature_columns(
    feature_groups: dict[str, list[str]],
    spec: WalkForwardSpec,
    *,
    all_timeframes: list[str],
) -> list[str]:
    columns: list[str] = []
    for group_name in spec.feature_group_names:
        columns.extend(feature_groups.get(group_name, []))

    selected = list(dict.fromkeys(columns))
    excluded_timeframes = [timeframe for timeframe in all_timeframes if timeframe not in spec.timeframes]

    def column_allowed(column: str) -> bool:
        if excluded_timeframes and any(f"_{timeframe}" in column for timeframe in excluded_timeframes):
            return False
        lower = column.lower()
        if spec.channel_family == "wick":
            if "body" in lower and "body_to_wick" not in lower and "midline_gap" not in lower:
                return False
            if "body_to_wick" in lower or "midline_gap" in lower or "slope_difference_body_vs_wick" in lower:
                return False
        if spec.channel_family == "body":
            if "wick" in lower and "body_to_wick" not in lower and "midline_gap" not in lower:
                return False
            if "body_to_wick" in lower or "midline_gap" in lower or "slope_difference_body_vs_wick" in lower:
                return False
        return True

    filtered = [column for column in selected if column_allowed(column)]
    return filtered


def fit_binary_model(
    train: pd.DataFrame,
    columns: list[str],
    label_column: str,
    model_name: str,
) -> Any:
    if not SKLEARN_AVAILABLE:
        if model_name != "logreg":
            raise RuntimeError("scikit-learn is required for hgb/rf. Use .venv\\Scripts\\python.exe or --model logreg.")
        return fit_fallback_logistic(train, columns, label_column)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True)),
        ]
    )
    preprocessor = ColumnTransformer([("num", numeric_transformer, columns)], remainder="drop")

    if model_name == "logreg":
        estimator = LogisticRegression(max_iter=2500, class_weight="balanced", C=0.10)
    elif model_name == "rf":
        estimator = RandomForestClassifier(
            n_estimators=500,
            max_depth=6,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=17,
            n_jobs=1,
        )
    else:
        estimator = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.035,
            max_leaf_nodes=12,
            min_samples_leaf=20,
            l2_regularization=0.5,
            class_weight="balanced",
            random_state=17,
        )

    model = Pipeline(steps=[("preprocessor", preprocessor), ("estimator", estimator)])
    model.fit(train[columns], train[label_column].astype(int))
    return model


def fit_fallback_logistic(train: pd.DataFrame, columns: list[str], label_column: str) -> FallbackLogisticModel:
    x = train[columns].astype(float).to_numpy()
    y = train[label_column].astype(float).to_numpy()
    median = np.nanmedian(x, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    x = np.where(np.isnan(x), median, x)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    xs = (x - mean) / scale

    coef = np.zeros(xs.shape[1], dtype=float)
    intercept = 0.0
    pos = max(float(y.sum()), 1.0)
    neg = max(float(len(y) - y.sum()), 1.0)
    weights = np.where(y > 0.5, len(y) / (2.0 * pos), len(y) / (2.0 * neg))
    weight_sum = weights.sum()

    for _ in range(1_500):
        logits = np.clip(intercept + xs @ coef, -40.0, 40.0)
        pred = 1.0 / (1.0 + np.exp(-logits))
        error = (pred - y) * weights
        grad_intercept = error.sum() / weight_sum
        grad_coef = xs.T @ error / weight_sum + 0.05 * coef / len(y)
        intercept -= 0.05 * grad_intercept
        coef -= 0.05 * grad_coef

    return FallbackLogisticModel(columns=columns, median=median, mean=mean, scale=scale, coef=coef, intercept=float(intercept))


def predict_binary_model(model: Any, frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    if isinstance(model, FallbackLogisticModel):
        x = frame[model.columns].astype(float).to_numpy()
        x = np.where(np.isnan(x), model.median, x)
        xs = (x - model.mean) / model.scale
        logits = np.clip(model.intercept + xs @ model.coef, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-logits))
    return model.predict_proba(frame[columns])[:, 1]


def binary_metrics(y_true: pd.Series, prob: pd.Series) -> dict[str, float]:
    values = y_true.astype(int).to_numpy()
    score = prob.astype(float).clip(1e-6, 1.0 - 1e-6).to_numpy()
    return {
        "rows": float(len(values)),
        "rate": float(values.mean()) if len(values) else 0.0,
        "auc": auc_score(values, score),
        "brier": float(np.mean((score - values) ** 2)) if len(values) else np.nan,
        "log_loss": float(-np.mean(values * np.log(score) + (1.0 - values) * np.log(1.0 - score))) if len(values) else np.nan,
    }


def auc_score(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(y_true) == 0:
        return np.nan
    pos = int(y_true.sum())
    neg = int(len(y_true) - pos)
    if pos == 0 or neg == 0:
        return np.nan
    ranks = pd.Series(score).rank(method="average").to_numpy()
    return float((ranks[y_true == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def _signal_score_series(frame: pd.DataFrame, direction: str, score_mode: str) -> pd.Series:
    p_long = pd.to_numeric(frame["p_long"], errors="coerce").astype(float)
    p_short = pd.to_numeric(frame["p_short"], errors="coerce").astype(float)
    mode = score_mode.strip().lower()
    if mode == "probability":
        return p_long if direction == "long" else p_short
    if mode == "edge":
        return p_long - p_short if direction == "long" else p_short - p_long
    if mode == "logit":
        target = p_long if direction == "long" else p_short
        clipped = target.clip(1e-6, 1.0 - 1e-6)
        return np.log(clipped / (1.0 - clipped))
    if mode == "logit_edge":
        long_logit = np.log(p_long.clip(1e-6, 1.0 - 1e-6) / (1.0 - p_long.clip(1e-6, 1.0 - 1e-6)))
        short_logit = np.log(p_short.clip(1e-6, 1.0 - 1e-6) / (1.0 - p_short.clip(1e-6, 1.0 - 1e-6)))
        return long_logit - short_logit if direction == "long" else short_logit - long_logit
    raise ValueError(f"Unsupported score mode: {score_mode!r}")


def _empirical_percentile(reference: pd.Series, values: pd.Series) -> pd.Series:
    ref = np.sort(pd.to_numeric(reference, errors="coerce").dropna().astype(float).to_numpy())
    if len(ref) == 0:
        return pd.Series(np.nan, index=values.index, dtype=float)
    target = pd.to_numeric(values, errors="coerce").astype(float).to_numpy()
    percentile = np.searchsorted(ref, target, side="right") / float(len(ref))
    percentile = np.where(np.isfinite(target), percentile, np.nan)
    return pd.Series(percentile, index=values.index, dtype=float)


def _add_signal_score_columns(
    frame: pd.DataFrame,
    spec: WalkForwardSpec,
    *,
    reference: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = frame.copy()
    out["long_signal_score"] = _signal_score_series(out, "long", spec.long_score_mode)
    out["short_signal_score"] = _signal_score_series(out, "short", spec.short_score_mode)
    if spec.threshold_mode == "percentile":
        reference_frame = reference if reference is not None else out
        out["long_signal_percentile"] = _empirical_percentile(reference_frame["long_signal_score"], out["long_signal_score"])
        out["short_signal_percentile"] = _empirical_percentile(reference_frame["short_signal_score"], out["short_signal_score"])
    return out


def _score_columns(spec: WalkForwardSpec) -> tuple[str, str]:
    if spec.threshold_mode == "percentile":
        return "long_signal_percentile", "short_signal_percentile"
    return "long_signal_score", "short_signal_score"


def tune_thresholds(validation: pd.DataFrame, spec: WalkForwardSpec) -> tuple[dict[str, float | str], pd.DataFrame]:
    rows: list[dict[str, float | str]] = []
    best_score = -math.inf
    best_pair: dict[str, float | str] = {
        "long_threshold": spec.long_thresholds[0],
        "short_threshold": spec.short_thresholds[0],
        "probability_gap": spec.probability_gap_values[0],
        "gate_preset": spec.gate_presets[0],
    }
    long_score_column, short_score_column = _score_columns(spec)
    long_raw = pd.to_numeric(validation["long_signal_score"], errors="coerce")
    short_raw = pd.to_numeric(validation["short_signal_score"], errors="coerce")

    for long_threshold in spec.long_thresholds:
        for short_threshold in spec.short_thresholds:
            for probability_gap in spec.probability_gap_values:
                for gate_preset in spec.gate_presets:
                    trades, metrics = simulate_threshold_strategy(
                        validation,
                        long_threshold=long_threshold,
                        short_threshold=short_threshold,
                        alpha=spec.alpha,
                        beta=spec.beta,
                        fee_bps_side=spec.fee_bps_side,
                        slippage_bps_side=spec.slippage_bps_side,
                        risk_fraction=spec.risk_fraction,
                        gate_spec=SignalGateSpec(min_probability_gap=float(probability_gap), preset=str(gate_preset)),
                        long_score_column=long_score_column,
                        short_score_column=short_score_column,
                    )
                    score = _threshold_score(metrics)
                    eligible = metrics["trades"] >= float(spec.min_validation_trades)
                    rows.append(
                        {
                            "long_threshold": float(long_threshold),
                            "short_threshold": float(short_threshold),
                            "probability_gap": float(probability_gap),
                            "gate_preset": str(gate_preset),
                            "threshold_mode": spec.threshold_mode,
                            "long_score_mode": spec.long_score_mode,
                            "short_score_mode": spec.short_score_mode,
                            "long_score_cut_raw": (
                                float(long_raw.quantile(long_threshold))
                                if spec.threshold_mode == "percentile" and len(long_raw.dropna())
                                else float(long_threshold)
                            ),
                            "short_score_cut_raw": (
                                float(short_raw.quantile(short_threshold))
                                if spec.threshold_mode == "percentile" and len(short_raw.dropna())
                                else float(short_threshold)
                            ),
                            "eligible": float(eligible),
                            "score": float(score if eligible else -10_000.0),
                            **metrics,
                        }
                    )
                    if eligible and score > best_score:
                        best_score = score
                        best_pair = {
                            "long_threshold": float(long_threshold),
                            "short_threshold": float(short_threshold),
                            "probability_gap": float(probability_gap),
                            "gate_preset": str(gate_preset),
                        }

    threshold_table = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    if best_score == -math.inf and not threshold_table.empty:
        best_row = threshold_table.iloc[0]
        best_pair = {
            "long_threshold": float(best_row["long_threshold"]),
            "short_threshold": float(best_row["short_threshold"]),
            "probability_gap": float(best_row.get("probability_gap", 0.0)),
            "gate_preset": str(best_row.get("gate_preset", "none")),
        }
    return best_pair, threshold_table


def run_walkforward_study(
    dataset: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    spec: WalkForwardSpec,
    *,
    all_timeframes: list[str],
) -> dict[str, pd.DataFrame]:
    feature_columns = select_feature_columns(feature_groups, spec, all_timeframes=all_timeframes)
    atr_column = f"atr_tf_{spec.decision_timeframe}"
    required_columns = ["decision_time", "label_end_time", "decision_close", atr_column, "tb_label", "long_label", "short_label", *feature_columns]
    drop_columns = ["decision_time", "label_end_time", "decision_close", atr_column, "tb_label", "long_label", "short_label"]
    usable = dataset.dropna(subset=drop_columns).copy()
    usable = usable.sort_values("decision_time").reset_index(drop=True)
    feature_columns = [column for column in feature_columns if column in usable.columns and usable[column].notna().any()]
    folds = build_walkforward_folds(usable, spec)

    fold_rows: list[dict[str, float | str]] = []
    threshold_rows: list[pd.DataFrame] = []
    prediction_rows: list[pd.DataFrame] = []
    trade_rows: list[pd.DataFrame] = []
    importance_rows: list[pd.DataFrame] = []

    embargo_delta = pd.Timedelta(hours=spec.embargo_bars)
    long_score_column, short_score_column = _score_columns(spec)

    for fold_index, fold in enumerate(folds, start=1):
        train_end = fold["train_end"]
        val_end = fold["val_end"]
        test_end = fold["test_end"]

        train = usable[
            (usable["decision_time"] >= fold["train_start"])
            & (usable["decision_time"] < train_end)
            & (usable["label_end_time"] < train_end - embargo_delta)
        ].copy()
        validation = usable[
            (usable["decision_time"] >= train_end)
            & (usable["decision_time"] < val_end)
        ].copy()
        test = usable[
            (usable["decision_time"] >= val_end)
            & (usable["decision_time"] < test_end)
        ].copy()
        if train.empty or validation.empty or test.empty:
            continue
        if train["long_label"].nunique() < 2 or train["short_label"].nunique() < 2:
            continue

        long_model = fit_binary_model(train, feature_columns, "long_label", spec.model_name)
        short_model = fit_binary_model(train, feature_columns, "short_label", spec.model_name)

        validation = validation.copy()
        validation["p_long"] = predict_binary_model(long_model, validation, feature_columns)
        validation["p_short"] = predict_binary_model(short_model, validation, feature_columns)
        validation = _add_signal_score_columns(validation, spec)
        threshold_pair, threshold_table = tune_thresholds(validation, spec)
        threshold_table["fold"] = float(fold_index)
        threshold_rows.append(threshold_table)

        test = test.copy()
        test["p_long"] = predict_binary_model(long_model, test, feature_columns)
        test["p_short"] = predict_binary_model(short_model, test, feature_columns)
        test = _add_signal_score_columns(test, spec, reference=validation)
        test["selected_direction"] = [
            choose_signal_direction(
                row,
                float(threshold_pair["long_threshold"]),
                float(threshold_pair["short_threshold"]),
                gate_spec=SignalGateSpec(
                    min_probability_gap=float(threshold_pair.get("probability_gap", 0.0)),
                    preset=str(threshold_pair.get("gate_preset", "none")),
                ),
                long_score_column=long_score_column,
                short_score_column=short_score_column,
            )
            for _, row in test.iterrows()
        ]
        test["fold"] = float(fold_index)
        test["train_end"] = train_end
        test["val_end"] = val_end
        test["test_end"] = test_end
        prediction_columns = required_columns + [
            "p_long",
            "p_short",
            "long_signal_score",
            "short_signal_score",
            "selected_direction",
            "fold",
            "train_end",
            "val_end",
            "test_end",
        ]
        if spec.threshold_mode == "percentile":
            prediction_columns.extend(["long_signal_percentile", "short_signal_percentile"])
        prediction_rows.append(test[prediction_columns])

        trades, trade_metrics = simulate_threshold_strategy(
            test,
            long_threshold=float(threshold_pair["long_threshold"]),
            short_threshold=float(threshold_pair["short_threshold"]),
            alpha=spec.alpha,
            beta=spec.beta,
            fee_bps_side=spec.fee_bps_side,
            slippage_bps_side=spec.slippage_bps_side,
            risk_fraction=spec.risk_fraction,
            gate_spec=SignalGateSpec(
                min_probability_gap=float(threshold_pair.get("probability_gap", 0.0)),
                preset=str(threshold_pair.get("gate_preset", "none")),
            ),
            long_score_column=long_score_column,
            short_score_column=short_score_column,
        )
        if not trades.empty:
            trades["fold"] = float(fold_index)
            trade_rows.append(trades)

        fold_row: dict[str, float | str] = {
            "fold": float(fold_index),
            "train_start": fold["train_start"].isoformat(),
            "train_end": train_end.isoformat(),
            "val_end": val_end.isoformat(),
            "test_end": test_end.isoformat(),
            "train_rows": float(len(train)),
            "validation_rows": float(len(validation)),
            "test_rows": float(len(test)),
            "long_threshold": threshold_pair["long_threshold"],
            "short_threshold": threshold_pair["short_threshold"],
            "probability_gap": threshold_pair.get("probability_gap", 0.0),
            "gate_preset": threshold_pair.get("gate_preset", "none"),
            "threshold_mode": spec.threshold_mode,
            "long_score_mode": spec.long_score_mode,
            "short_score_mode": spec.short_score_mode,
            "feature_count": float(len(feature_columns)),
            "model_name": spec.model_name,
            "channel_family": spec.channel_family,
            "timeframes": ",".join(spec.timeframes),
        }
        long_val_metrics = binary_metrics(validation["long_label"], validation["p_long"])
        short_val_metrics = binary_metrics(validation["short_label"], validation["p_short"])
        long_test_metrics = binary_metrics(test["long_label"], test["p_long"])
        short_test_metrics = binary_metrics(test["short_label"], test["p_short"])
        for prefix, metrics in [
            ("long_val", long_val_metrics),
            ("short_val", short_val_metrics),
            ("long_test", long_test_metrics),
            ("short_test", short_test_metrics),
        ]:
            fold_row.update({f"{prefix}_{key}": float(value) if value is not None else np.nan for key, value in metrics.items()})
        for key, value in trade_metrics.items():
            fold_row[f"trade_{key}"] = float(value) if isinstance(value, (int, float, np.floating)) else value
        fold_rows.append(fold_row)

        if spec.compute_feature_importance:
            importance_rows.extend(
                [
                    _feature_importance_frame(long_model, test, feature_columns, "long_label", fold_index, "long"),
                    _feature_importance_frame(short_model, test, feature_columns, "short_label", fold_index, "short"),
                ]
            )

    fold_summary = pd.DataFrame(fold_rows) if fold_rows else pd.DataFrame(columns=["fold"])
    thresholds = pd.concat(threshold_rows, ignore_index=True) if threshold_rows else pd.DataFrame(columns=["fold"])
    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame(columns=["fold"])
    trades = pd.concat(trade_rows, ignore_index=True) if trade_rows else pd.DataFrame(columns=["fold"])
    importances = pd.concat(importance_rows, ignore_index=True) if importance_rows else pd.DataFrame(columns=["fold", "direction", "feature", "importance"])

    if not trades.empty:
        aggregate = strategy_metrics(trades)
        aggregate_row = {
            "fold": "aggregate",
            "train_start": "",
            "train_end": "",
            "val_end": "",
            "test_end": "",
            "train_rows": float(fold_summary["train_rows"].sum()) if not fold_summary.empty else 0.0,
            "validation_rows": float(fold_summary["validation_rows"].sum()) if not fold_summary.empty else 0.0,
            "test_rows": float(len(predictions)),
            "long_threshold": np.nan,
            "short_threshold": np.nan,
            "probability_gap": np.nan,
            "gate_preset": "",
            "threshold_mode": spec.threshold_mode,
            "long_score_mode": spec.long_score_mode,
            "short_score_mode": spec.short_score_mode,
            "feature_count": float(len(feature_columns)),
            "model_name": spec.model_name,
            "channel_family": spec.channel_family,
            "timeframes": ",".join(spec.timeframes),
        }
        for key, value in aggregate.items():
            aggregate_row[f"trade_{key}"] = float(value) if isinstance(value, (int, float, np.floating)) else value
        if not predictions.empty:
            aggregate_row.update(
                {
                    "long_test_auc": auc_score(predictions["long_label"].astype(int).to_numpy(), predictions["p_long"].astype(float).to_numpy()),
            "short_test_auc": auc_score(predictions["short_label"].astype(int).to_numpy(), predictions["p_short"].astype(float).to_numpy()),
                }
            )
        if not fold_summary.empty:
            for source, target in [
                ("long_threshold", "trade_long_threshold"),
                ("short_threshold", "trade_short_threshold"),
                ("probability_gap", "trade_probability_gap"),
                ("gate_preset", "trade_gate_preset"),
            ]:
                values = fold_summary[source].dropna() if source in fold_summary.columns else pd.Series(dtype=float)
                if values.empty:
                    continue
                if pd.api.types.is_numeric_dtype(values):
                    aggregate_row[target] = float(values.mode().iloc[0])
                else:
                    aggregate_row[target] = str(values.astype(str).mode().iloc[0])
        fold_summary = pd.concat([fold_summary, pd.DataFrame([aggregate_row])], ignore_index=True)

    return {
        "folds": fold_summary,
        "thresholds": thresholds,
        "predictions": predictions,
        "trades": trades,
        "feature_importance": importances,
        "feature_columns": pd.DataFrame({"feature": feature_columns}),
    }


def _threshold_score(metrics: dict[str, float]) -> float:
    total_return = float(metrics["total_return"])
    max_drawdown = abs(float(metrics["max_drawdown"]))
    profit_factor = float(metrics["profit_factor"]) if math.isfinite(float(metrics["profit_factor"])) else 5.0
    return total_return + 0.10 * profit_factor - 0.50 * max_drawdown


def _feature_importance_frame(
    model: Any,
    frame: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    fold_index: int,
    direction: str,
) -> pd.DataFrame:
    if frame.empty or frame[label_column].nunique() < 2:
        return pd.DataFrame()
    if isinstance(model, FallbackLogisticModel):
        importance = np.abs(model.coef)
        return pd.DataFrame(
            {
                "fold": float(fold_index),
                "direction": direction,
                "feature": feature_columns,
                "importance": importance,
            }
        ).sort_values("importance", ascending=False)
    estimator = None
    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get("estimator")
    if estimator is not None and hasattr(estimator, "coef_"):
        importance = np.abs(np.ravel(estimator.coef_))
        if len(importance) != len(feature_columns):
            return pd.DataFrame()
        return pd.DataFrame(
            {
                "fold": float(fold_index),
                "direction": direction,
                "feature": feature_columns,
                "importance": importance,
            }
        ).sort_values("importance", ascending=False)
    if estimator is not None and hasattr(estimator, "feature_importances_"):
        importance = np.asarray(estimator.feature_importances_, dtype=float)
        if len(importance) != len(feature_columns):
            return pd.DataFrame()
        return pd.DataFrame(
            {
                "fold": float(fold_index),
                "direction": direction,
                "feature": feature_columns,
                "importance": importance,
            }
        ).sort_values("importance", ascending=False)
    if not SKLEARN_AVAILABLE or permutation_importance is None:
        return pd.DataFrame()
    sample = frame.sample(n=min(len(frame), 300), random_state=17) if len(frame) > 300 else frame
    result = permutation_importance(
        model,
        sample[feature_columns],
        sample[label_column].astype(int),
        n_repeats=3,
        random_state=17,
        scoring="roc_auc",
    )
    return pd.DataFrame(
        {
            "fold": float(fold_index),
            "direction": direction,
            "feature": feature_columns,
            "importance": result.importances_mean,
        }
    ).sort_values("importance", ascending=False)
