from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from scripts.backtest_turtle_soup import add_atr, normalize_binance_spot_symbol, resample_ohlc


NUMERIC_FEATURES = [
    "direction_long",
    "risk_pct",
    "zone_width_pct",
    "zone_age_hours_log",
    "retest_delay_hours_log",
    "confirm_delay_hours_log",
    "confirm_fvg_atr",
    "confirm_fvg_height_pct",
    "confirm_fvg_r",
    "retest_reject_pos",
    "entry_extension_r",
    "confirm_gap_r",
    "confirm_break_r",
    "confirm_close_pos_dir",
    "confirm_body_frac",
    "retest_depth_frac",
    "retest_close_margin_r",
    "retest_range_atr",
    "rejection_speed",
    "chase_score",
    "confirm_strength_score",
    "entry_hour_sin",
    "entry_hour_cos",
    "entry_dow_sin",
    "entry_dow_cos",
    "break_hour_sin",
    "break_hour_cos",
]
CATEGORICAL_FEATURES = ["symbol", "direction"]


def _latest_cache_file(symbol: str, cache_dir: Path) -> Path | None:
    normalized = normalize_binance_spot_symbol(symbol).lower()
    matches = sorted(cache_dir.glob(f"{normalized}_5m_*.pkl"))
    if not matches:
        return None
    return matches[-1]


def _safe_div(numerator: pd.Series | float, denominator: pd.Series | float) -> pd.Series | float:
    if isinstance(denominator, pd.Series):
        return numerator / denominator.replace(0.0, np.nan)
    return numerator / denominator if denominator != 0 else math.nan


def _augment_context_features(out: pd.DataFrame, cache_dir: Path) -> pd.DataFrame:
    if out.empty:
        return out

    out = out.copy()
    out["entry_extension_r"] = np.nan
    out["confirm_gap_r"] = np.nan
    out["confirm_break_r"] = np.nan
    out["confirm_close_pos_dir"] = np.nan
    out["confirm_body_frac"] = np.nan
    out["retest_depth_frac"] = np.nan
    out["retest_close_margin_r"] = np.nan
    out["retest_range_atr"] = np.nan

    for (symbol, confirmation_tf), group in out.groupby(["symbol", "confirmation_tf"], dropna=False):
        cache_file = _latest_cache_file(str(symbol), cache_dir)
        if cache_file is None:
            continue

        candles = pd.read_pickle(cache_file).sort_values("open_time").reset_index(drop=True)
        candles = add_atr(candles)
        retest_lookup = candles.set_index("open_time")[["open", "high", "low", "close", "atr"]]

        tf = str(confirmation_tf) if pd.notna(confirmation_tf) else "15m"
        confirm = resample_ohlc(candles, tf)
        confirm_lookup = confirm.set_index("close_time")[["open", "high", "low", "close"]]

        for idx, row in group.iterrows():
            risk = abs(float(row["entry_price"]) - float(row["stop_price"]))
            if not math.isfinite(risk) or risk <= 0:
                continue
            zone_top = float(row["zone_top"])
            zone_bottom = float(row["zone_bottom"])
            zone_height = abs(zone_top - zone_bottom)
            direction = str(row["direction"])

            if direction == "long":
                out.at[idx, "entry_extension_r"] = (float(row["entry_price"]) - zone_top) / risk
                out.at[idx, "confirm_gap_r"] = (float(row["confirm_fvg_bottom"]) - zone_top) / risk if pd.notna(row["confirm_fvg_bottom"]) else math.nan
                out.at[idx, "confirm_break_r"] = (float(row["confirm_break_level"]) - zone_top) / risk if pd.notna(row["confirm_break_level"]) else math.nan
            else:
                out.at[idx, "entry_extension_r"] = (zone_bottom - float(row["entry_price"])) / risk
                out.at[idx, "confirm_gap_r"] = (zone_bottom - float(row["confirm_fvg_top"])) / risk if pd.notna(row["confirm_fvg_top"]) else math.nan
                out.at[idx, "confirm_break_r"] = (zone_bottom - float(row["confirm_break_level"])) / risk if pd.notna(row["confirm_break_level"]) else math.nan

            retest_time = pd.Timestamp(row["retest_time"]) if pd.notna(row["retest_time"]) else pd.NaT
            if pd.notna(retest_time) and retest_time in retest_lookup.index:
                retest_bar = retest_lookup.loc[retest_time]
                if zone_height > 0:
                    if direction == "long":
                        out.at[idx, "retest_depth_frac"] = (zone_top - float(retest_bar["low"])) / zone_height
                        out.at[idx, "retest_close_margin_r"] = (float(retest_bar["close"]) - zone_top) / risk
                    else:
                        out.at[idx, "retest_depth_frac"] = (float(retest_bar["high"]) - zone_bottom) / zone_height
                        out.at[idx, "retest_close_margin_r"] = (zone_bottom - float(retest_bar["close"])) / risk
                atr = float(retest_bar["atr"])
                if math.isfinite(atr) and atr > 0:
                    out.at[idx, "retest_range_atr"] = (float(retest_bar["high"]) - float(retest_bar["low"])) / atr

            confirm_time = pd.Timestamp(row["confirm_time"]) if pd.notna(row["confirm_time"]) else pd.NaT
            if pd.notna(confirm_time) and confirm_time in confirm_lookup.index:
                confirm_bar = confirm_lookup.loc[confirm_time]
                confirm_range = float(confirm_bar["high"]) - float(confirm_bar["low"])
                if confirm_range > 0:
                    close_pos = (float(confirm_bar["close"]) - float(confirm_bar["low"])) / confirm_range
                    out.at[idx, "confirm_close_pos_dir"] = close_pos if direction == "long" else 1.0 - close_pos
                    out.at[idx, "confirm_body_frac"] = abs(float(confirm_bar["close"]) - float(confirm_bar["open"])) / confirm_range

    return out


def profit_factor(rs: pd.Series) -> float:
    wins = rs[rs > 0]
    losses = rs[rs <= 0]
    gross_loss = abs(float(losses.sum()))
    if gross_loss == 0:
        return float("inf") if len(wins) else 0.0
    return float(wins.sum()) / gross_loss


def max_drawdown(values: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for value in values:
        equity += float(value)
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)
    return max_dd


def trade_metrics(frame: pd.DataFrame, r_column: str = "r_net_cost") -> dict[str, Any]:
    if frame.empty:
        return {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "net_r": 0.0, "avg_r": 0.0, "max_dd_r": 0.0}
    ordered = frame.sort_values("exit_time")
    rs = ordered[r_column].astype(float)
    return {
        "trades": int(len(ordered)),
        "win_rate": round(100.0 * float((rs > 0).mean()), 2),
        "profit_factor": round(profit_factor(rs), 3),
        "net_r": round(float(rs.sum()), 3),
        "avg_r": round(float(rs.mean()), 4),
        "max_dd_r": round(max_drawdown(rs.to_list()), 3),
    }


def classifier_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty or frame["label"].nunique() < 2:
        return {"rows": int(len(frame)), "hit_rate": 0.0, "auc": math.nan, "brier": math.nan, "log_loss": math.nan}
    labels = frame["label"].astype(int)
    probs = frame["breaker_prob"].astype(float)
    return {
        "rows": int(len(frame)),
        "hit_rate": round(100.0 * float(labels.mean()), 2),
        "auc": round(float(roc_auc_score(labels, probs)), 3),
        "brier": round(float(brier_score_loss(labels, probs)), 4),
        "log_loss": round(float(log_loss(labels, probs, labels=[0, 1])), 4),
    }


def enrich(frame: pd.DataFrame, fee_bps_side: float) -> pd.DataFrame:
    out = frame.copy()
    for column in ["entry_time", "exit_time", "break_time", "retest_time", "zone_time"]:
        out[column] = pd.to_datetime(out[column], utc=True)
    risk = (out["entry_price"].astype(float) - out["stop_price"].astype(float)).abs()
    out["risk_pct"] = risk / out["entry_price"].astype(float) * 100.0
    out["zone_width_pct"] = (out["zone_top"].astype(float) - out["zone_bottom"].astype(float)).abs() / out["entry_price"].astype(float) * 100.0
    out["zone_age_hours"] = (out["break_time"] - out["zone_time"]).dt.total_seconds() / 3600.0
    out["retest_delay_hours"] = (out["retest_time"] - out["break_time"]).dt.total_seconds() / 3600.0
    if "confirm_time" in out.columns:
        out["confirm_time"] = pd.to_datetime(out["confirm_time"], utc=True, errors="coerce")
        out["confirm_delay_hours"] = (out["confirm_time"] - out["retest_time"]).dt.total_seconds() / 3600.0
    else:
        out["confirm_delay_hours"] = 0.0
    if "confirm_fvg_atr" not in out.columns:
        out["confirm_fvg_atr"] = 0.0
    if "confirm_fvg_height" not in out.columns:
        out["confirm_fvg_height"] = 0.0
    out["confirm_fvg_atr"] = pd.to_numeric(out["confirm_fvg_atr"], errors="coerce").fillna(0.0)
    out["confirm_fvg_height"] = pd.to_numeric(out["confirm_fvg_height"], errors="coerce").fillna(0.0)
    out["confirm_fvg_height_pct"] = out["confirm_fvg_height"].abs() / out["entry_price"].astype(float) * 100.0
    out["confirm_fvg_r"] = _safe_div(out["confirm_fvg_height"].abs(), risk)
    out["zone_age_hours_log"] = np.log1p(out["zone_age_hours"].clip(lower=0.0))
    out["retest_delay_hours_log"] = np.log1p(out["retest_delay_hours"].clip(lower=0.0))
    out["confirm_delay_hours_log"] = np.log1p(out["confirm_delay_hours"].clip(lower=0.0))
    out["direction_long"] = (out["direction"].astype(str) == "long").astype(float)

    if "retest_reject_pos" not in out.columns:
        out["retest_reject_pos"] = 0.0
    out["retest_reject_pos"] = pd.to_numeric(out["retest_reject_pos"], errors="coerce")

    entry_hour = out["entry_time"].dt.hour + out["entry_time"].dt.minute / 60.0
    entry_dow = out["entry_time"].dt.dayofweek.astype(float)
    break_hour = out["break_time"].dt.hour + out["break_time"].dt.minute / 60.0
    out["entry_hour_sin"] = np.sin(2.0 * math.pi * entry_hour / 24.0)
    out["entry_hour_cos"] = np.cos(2.0 * math.pi * entry_hour / 24.0)
    out["entry_dow_sin"] = np.sin(2.0 * math.pi * entry_dow / 7.0)
    out["entry_dow_cos"] = np.cos(2.0 * math.pi * entry_dow / 7.0)
    out["break_hour_sin"] = np.sin(2.0 * math.pi * break_hour / 24.0)
    out["break_hour_cos"] = np.cos(2.0 * math.pi * break_hour / 24.0)

    cache_dir = Path("scripts/.cache")
    if cache_dir.exists():
        out = _augment_context_features(out, cache_dir)
    else:
        for column in [
            "entry_extension_r",
            "confirm_gap_r",
            "confirm_break_r",
            "confirm_close_pos_dir",
            "confirm_body_frac",
            "retest_depth_frac",
            "retest_close_margin_r",
            "retest_range_atr",
        ]:
            out[column] = np.nan

    out["rejection_speed"] = out["retest_reject_pos"] / (1.0 + out["confirm_delay_hours"].clip(lower=0.0))
    out["chase_score"] = out["entry_extension_r"] * out["confirm_close_pos_dir"]
    out["confirm_strength_score"] = out["retest_reject_pos"] * out["confirm_body_frac"] * (1.0 + out["confirm_fvg_atr"].clip(lower=0.0))

    for column in [
        "confirm_fvg_r",
        "entry_extension_r",
        "confirm_gap_r",
        "confirm_break_r",
        "confirm_close_pos_dir",
        "confirm_body_frac",
        "retest_depth_frac",
        "retest_close_margin_r",
        "retest_range_atr",
        "rejection_speed",
        "chase_score",
        "confirm_strength_score",
    ]:
        out[column] = pd.to_numeric(out[column], errors="coerce").replace([np.inf, -np.inf], np.nan)

    cost_r = (out["entry_price"].abs() + out["exit_price"].abs()) * fee_bps_side / 10000.0 / risk
    out["r_net_cost"] = out["r_multiple"].astype(float) - cost_r
    out["label"] = (out["r_net_cost"] > 0).astype(int)
    return out


def build_model(model_name: str) -> Any:
    numeric_pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    categorical_pipe = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"),
    )
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    if model_name == "hgb":
        estimator = HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.04,
            max_leaf_nodes=15,
            l2_regularization=0.05,
            random_state=42,
        )
    else:
        estimator = RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=25,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=1,
        )
    return make_pipeline(preprocessor, estimator)


def threshold_table(frame: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        selected = frame[frame["breaker_prob"] >= threshold].copy()
        rows.append({"threshold": threshold, **trade_metrics(selected)})
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a breaker-continuation ML filter.")
    parser.add_argument("--trades", type=Path, default=Path("scripts/breaker_continuation_core3_1h_retest72_2022_2026.csv"))
    parser.add_argument("--split", default="2025-04-20")
    parser.add_argument("--end", default="2026-04-20")
    parser.add_argument("--fee-bps-side", type=float, default=5.0)
    parser.add_argument("--model", choices=["rf", "hgb"], default="rf")
    parser.add_argument("--model-out", type=Path, default=Path("scripts/breaker_continuation_model.joblib"))
    parser.add_argument("--dataset-out", type=Path, default=Path("scripts/breaker_continuation_ml_dataset.csv"))
    parser.add_argument("--thresholds", default="0.45,0.50,0.55,0.60,0.65,0.70")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = pd.Timestamp(args.split, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    thresholds = [float(value.strip()) for value in args.thresholds.split(",") if value.strip()]

    dataset = enrich(pd.read_csv(args.trades), args.fee_bps_side)
    train = dataset[dataset["entry_time"] < split].copy()
    oos = dataset[(dataset["entry_time"] >= split) & (dataset["entry_time"] < end)].copy()
    if train["label"].nunique() < 2:
        raise RuntimeError("Training labels contain only one class.")

    model = build_model(args.model)
    model.fit(train[NUMERIC_FEATURES + CATEGORICAL_FEATURES], train["label"].astype(int))
    dataset["breaker_prob"] = model.predict_proba(dataset[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1]
    train = dataset[dataset["entry_time"] < split].copy()
    oos = dataset[(dataset["entry_time"] >= split) & (dataset["entry_time"] < end)].copy()

    payload = {
        "model": model,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "model_kind": args.model,
        "fee_bps_side": args.fee_bps_side,
        "source_trades": str(args.trades),
    }
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, args.model_out)
    args.dataset_out.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.dataset_out, index=False)

    print(f"Model saved to {args.model_out}")
    print(f"Dataset saved to {args.dataset_out}")
    print()
    print("Classifier metrics:")
    print(pd.DataFrame([
        {"window": "train", **classifier_metrics(train)},
        {"window": "oos", **classifier_metrics(oos)},
    ]).to_string(index=False))
    print()
    print("Trade metrics after 5bps/side-style cost:")
    print(pd.DataFrame([
        {"window": "train_all", **trade_metrics(train)},
        {"window": "oos_all", **trade_metrics(oos)},
    ]).to_string(index=False))
    print()
    print("OOS threshold table:")
    print(threshold_table(oos, thresholds).to_string(index=False))


if __name__ == "__main__":
    main()
