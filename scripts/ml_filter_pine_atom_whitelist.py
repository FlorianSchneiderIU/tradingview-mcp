from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backtest_turtle_soup import parse_utc_datetime  # noqa: E402
from scripts.experiment_pine_strategy_candidates import (  # noqa: E402
    CandidateSpec,
    SIGNAL_BUILDERS,
    Trade,
    base_context,
    feature_table,
    load_frame,
    prefixed_metrics,
    resample_frame,
    simulate_signals,
)


DEFAULT_WHITELIST = Path("scripts/pine_atom_per_coin_full_sweep_top50_15m_recommended_coin_atoms.csv")
DEFAULT_CACHE_DIR = Path("scripts/.cache/bybit_linear")
DEFAULT_OUT_PREFIX = Path("scripts/pine_atom_whitelist_ml_filter_top15_15m")


@dataclass
class ComboPayload:
    symbol: str
    atom: str
    spec: CandidateSpec
    raw_row: dict[str, Any]


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def profit_factor(values: pd.Series | np.ndarray | list[float]) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return 0.0
    gains = arr[arr > 0].sum()
    losses = -arr[arr < 0].sum()
    if losses <= 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def max_dd(values: pd.Series | np.ndarray | list[float]) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return 0.0
    curve = np.cumsum(arr)
    peaks = np.maximum.accumulate(np.insert(curve, 0, 0.0))[1:]
    return float(np.max(peaks - curve)) if curve.size else 0.0


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {
            "trades": 0,
            "net_r": 0.0,
            "avg_r": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_dd_r": 0.0,
        }
    r = pd.to_numeric(frame["r_multiple"], errors="coerce").dropna()
    return {
        "trades": int(len(r)),
        "net_r": float(r.sum()),
        "avg_r": float(r.mean()) if len(r) else 0.0,
        "win_rate": float((r > 0).mean()) if len(r) else 0.0,
        "profit_factor": profit_factor(r),
        "max_dd_r": max_dd(r),
    }


def safe_div(num: float, den: float) -> float:
    if not math.isfinite(num) or not math.isfinite(den) or abs(den) < 1e-12:
        return math.nan
    return num / den


def kaufman_er(close: np.ndarray, length: int = 20) -> np.ndarray:
    out = np.full(close.shape, np.nan, dtype=float)
    changes = np.abs(np.diff(close, prepend=np.nan))
    for i in range(length, len(close)):
        direction = abs(close[i] - close[i - length])
        noise = np.nansum(changes[i - length + 1 : i + 1])
        out[i] = safe_div(direction, noise)
    return out


def rolling_percentile(values: np.ndarray, length: int = 200) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    series = pd.Series(values)
    for i in range(length - 1, len(values)):
        window = series.iloc[i - length + 1 : i + 1].dropna().to_numpy(dtype=float)
        if window.size:
            out[i] = float((window <= values[i]).mean())
    return out


def daily_vwap(frame: pd.DataFrame) -> np.ndarray:
    typical = (frame["high"].to_numpy(dtype=float) + frame["low"].to_numpy(dtype=float) + frame["close"].to_numpy(dtype=float)) / 3.0
    volume = frame["volume"].to_numpy(dtype=float)
    dates = pd.to_datetime(frame["open_time"], utc=True).dt.date
    out = np.full(len(frame), np.nan, dtype=float)
    cum_pv = 0.0
    cum_v = 0.0
    last_date = None
    for i, date in enumerate(dates):
        if date != last_date:
            cum_pv = 0.0
            cum_v = 0.0
            last_date = date
        cum_pv += typical[i] * volume[i]
        cum_v += volume[i]
        out[i] = safe_div(cum_pv, cum_v)
    return out


def enrich_trade_features(frame: pd.DataFrame, ctx: dict[str, np.ndarray], trade: Trade) -> dict[str, Any]:
    row = trade.to_dict()
    try:
        features = json.loads(row.get("feature_json") or "{}")
    except Exception:
        features = {}

    i = int(trade.signal_index)
    direction = 1.0 if trade.direction == "long" else -1.0
    open_ = frame["open"].to_numpy(dtype=float)
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    close = frame["close"].to_numpy(dtype=float)
    volume = frame["volume"].to_numpy(dtype=float)
    atr = ctx["atr14"]
    atr20 = ctx["atr20"]
    vwap = ctx["daily_vwap"]
    er20 = ctx["er20"]
    atr_pctile = ctx["atr_pctile200"]
    candle_range = max(high[i] - low[i], 1e-12)
    upper_wick = high[i] - max(open_[i], close[i])
    lower_wick = min(open_[i], close[i]) - low[i]
    hour = pd.Timestamp(frame["close_time"].iloc[i]).hour

    features.update(
        {
            "hour_sin": math.sin(2.0 * math.pi * hour / 24.0),
            "hour_cos": math.cos(2.0 * math.pi * hour / 24.0),
            "session_asia": 1.0 if 0 <= hour < 8 else 0.0,
            "session_london": 1.0 if 7 <= hour < 16 else 0.0,
            "session_ny": 1.0 if 13 <= hour < 22 else 0.0,
            "signal_range_atr": safe_div(candle_range, atr[i]),
            "signal_body_atr_dir": direction * safe_div(close[i] - open_[i], atr[i]),
            "signal_wick_reject_pct": safe_div(lower_wick if direction > 0 else upper_wick, candle_range),
            "opp_wick_pct": safe_div(upper_wick if direction > 0 else lower_wick, candle_range),
            "volume_ratio50": safe_div(volume[i], ctx["vol_sma50"][i]),
            "atr20_vs_atr14": safe_div(atr20[i], atr[i]),
            "atr_pctile200": atr_pctile[i],
            "er20": er20[i],
            "close_vs_daily_vwap_atr_dir": direction * safe_div(close[i] - vwap[i], atr[i]),
            "ema20_slope_12_atr_dir": direction * safe_div(ctx["ema20"][i] - ctx["ema20"][max(0, i - 12)], atr[i]),
            "ema50_slope_24_atr_dir": direction * safe_div(ctx["ema50"][i] - ctx["ema50"][max(0, i - 24)], atr[i]),
        }
    )
    for lookback in (1, 3, 6, 12, 24, 48):
        j = i - lookback
        features[f"ret_{lookback}_dir_pct"] = (
            direction * (close[i] / close[j] - 1.0) * 100.0
            if j >= 0 and close[j] > 0
            else math.nan
        )
    row["feature_json"] = json.dumps(features, sort_keys=True)
    return row


def load_whitelist(path: Path, max_combos: int, best_per_coin: bool) -> list[ComboPayload]:
    frame = pd.read_csv(path)
    frame["oos_net_r"] = pd.to_numeric(frame["oos_net_r"], errors="coerce")
    frame["oos_profit_factor"] = pd.to_numeric(frame["oos_profit_factor"], errors="coerce")
    frame = frame.sort_values(["oos_net_r", "oos_profit_factor"], ascending=[False, False]).copy()
    if best_per_coin:
        frame = frame.groupby("symbol", as_index=False).head(1)
    if max_combos > 0:
        frame = frame.head(max_combos)
    combos: list[ComboPayload] = []
    for _, row in frame.iterrows():
        params = json.loads(row["params_json"])
        spec = CandidateSpec(strategy=str(row["strategy"]), timeframe=str(row["timeframe"]), params=params)
        combos.append(
            ComboPayload(
                symbol=str(row["symbol"]),
                atom=str(row["atom"]),
                spec=spec,
                raw_row=row.to_dict(),
            )
        )
    return combos


def trades_for_combo(
    combo: ComboPayload,
    *,
    cache_dir: Path,
    train_start: pd.Timestamp,
    end: pd.Timestamp,
    fee_bps_per_side: float,
    min_risk_pct: float,
) -> pd.DataFrame:
    base = load_frame(combo.symbol, cache_dir, train_start, end)
    frame = resample_frame(base, combo.spec.timeframe)
    frame = frame[frame["open_time"] >= train_start - pd.Timedelta(days=10)].reset_index(drop=True)
    ctx = base_context(frame)
    close = frame["close"].to_numpy(dtype=float)
    ctx["daily_vwap"] = daily_vwap(frame)
    ctx["er20"] = kaufman_er(close, 20)
    ctx["atr_pctile200"] = rolling_percentile(ctx["atr14"] / close * 100.0, 200)
    builder = SIGNAL_BUILDERS[combo.spec.strategy]
    signals = builder(frame, ctx, combo.spec)
    trades = simulate_signals(
        frame,
        ctx,
        symbol=combo.symbol,
        spec=combo.spec,
        signals=signals,
        rr=float(combo.spec.params["rr"]),
        max_hold_bars=int(combo.spec.params["max_hold_bars"]),
        fee_bps_per_side=fee_bps_per_side,
        min_risk_pct=min_risk_pct,
    )
    rows = [enrich_trade_features(frame, ctx, trade) for trade in trades]
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["atom"] = combo.atom
    out["combo_key"] = f"{combo.symbol}|{combo.atom}|{combo.spec.name}"
    out["entry_time"] = pd.to_datetime(out["entry_time"], utc=True, errors="coerce")
    out = out[out["entry_time"] >= train_start].copy()
    return out


def make_models(random_state: int, train_len: int) -> dict[str, Any]:
    leaf = max(12, min(75, train_len // 20))
    return {
        "logit_l2": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                C=0.5,
                class_weight="balanced",
                max_iter=1000,
                random_state=random_state,
            ),
        ),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=450,
                max_depth=4,
                min_samples_leaf=leaf,
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        "extra_trees": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=500,
                max_depth=4,
                min_samples_leaf=leaf,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
    }


def expanded_features(trades: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    expanded = feature_table(trades)
    feature_cols = [c for c in expanded.columns if c.startswith("f_")]
    keep = []
    for col in feature_cols:
        numeric = pd.to_numeric(expanded[col], errors="coerce")
        if numeric.notna().sum() >= 10 and numeric.nunique(dropna=True) > 1:
            expanded[col] = numeric
            keep.append(col)
    return expanded, keep


def choose_filter(
    combo: ComboPayload,
    trades: pd.DataFrame,
    *,
    split: pd.Timestamp,
    val_frac: float,
    thresholds: list[float],
    min_fit: int,
    min_val: int,
    min_oos: int,
    model_dir: Path,
    random_state: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    expanded, feature_cols = expanded_features(trades)
    expanded["entry_time"] = pd.to_datetime(expanded["entry_time"], utc=True, errors="coerce")
    train_all = expanded[expanded["entry_time"] < split].sort_values("entry_time").copy()
    oos = expanded[expanded["entry_time"] >= split].sort_values("entry_time").copy()
    raw_train_m = metrics(train_all)
    raw_oos_m = metrics(oos)

    base_row: dict[str, Any] = {
        "symbol": combo.symbol,
        "atom": combo.atom,
        "strategy": combo.spec.strategy,
        "timeframe": combo.spec.timeframe,
        "spec_name": combo.spec.name,
        "params_json": json.dumps(combo.spec.params, sort_keys=True),
        **{f"raw_train_{k}": v for k, v in raw_train_m.items()},
        **{f"raw_oos_{k}": v for k, v in raw_oos_m.items()},
        "feature_count": len(feature_cols),
        "status": "skipped",
        "reason": "",
    }
    if len(train_all) < min_fit + min_val:
        base_row["reason"] = f"not enough train trades ({len(train_all)})"
        return base_row, pd.DataFrame()
    if len(oos) < min_oos:
        base_row["reason"] = f"not enough OOS trades ({len(oos)})"
        return base_row, pd.DataFrame()
    if len(feature_cols) < 4:
        base_row["reason"] = f"not enough features ({len(feature_cols)})"
        return base_row, pd.DataFrame()

    cut = max(min_fit, int(len(train_all) * (1.0 - val_frac)))
    cut = min(cut, len(train_all) - min_val)
    fit = train_all.iloc[:cut].copy()
    val = train_all.iloc[cut:].copy()
    if fit["r_multiple"].gt(0).nunique() < 2:
        base_row["reason"] = "fit labels single class"
        return base_row, pd.DataFrame()

    raw_fit_m = metrics(fit)
    raw_val_m = metrics(val)
    detail_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_model = None
    for model_name, model in make_models(random_state, len(fit)).items():
        try:
            model.fit(fit[feature_cols], fit["r_multiple"].gt(0).astype(int))
            val_prob = model.predict_proba(val[feature_cols])[:, 1]
            oos_prob = model.predict_proba(oos[feature_cols])[:, 1]
        except Exception as exc:
            detail_rows.append(
                {
                    "symbol": combo.symbol,
                    "atom": combo.atom,
                    "spec_name": combo.spec.name,
                    "model": model_name,
                    "threshold": math.nan,
                    "status": f"fit_failed: {type(exc).__name__}: {exc}",
                }
            )
            continue
        val_scored = val.copy()
        oos_scored = oos.copy()
        val_scored["ml_prob"] = val_prob
        oos_scored["ml_prob"] = oos_prob
        for threshold in thresholds:
            sel_val = val_scored[val_scored["ml_prob"] >= threshold].copy()
            sel_oos = oos_scored[oos_scored["ml_prob"] >= threshold].copy()
            val_m = metrics(sel_val)
            oos_m = metrics(sel_oos)
            min_val_selected = max(min_val, int(len(val) * 0.20))
            valid = val_m["trades"] >= min_val_selected and val_m["net_r"] > 0
            val_pf = min(float(val_m["profit_factor"]), 5.0) if math.isfinite(float(val_m["profit_factor"])) else 5.0
            score = (
                val_pf
                + 0.35 * val_m["avg_r"] * math.sqrt(max(val_m["trades"], 1))
                + 0.02 * math.log1p(val_m["trades"])
                - 0.01 * val_m["max_dd_r"]
            )
            row = {
                "symbol": combo.symbol,
                "atom": combo.atom,
                "strategy": combo.spec.strategy,
                "timeframe": combo.spec.timeframe,
                "spec_name": combo.spec.name,
                "model": model_name,
                "threshold": threshold,
                "valid_on_val": bool(valid),
                "score": float(score),
                **{f"val_{k}": v for k, v in val_m.items()},
                **{f"oos_{k}": v for k, v in oos_m.items()},
            }
            detail_rows.append(row)
            if valid and (best is None or score > best["score"]):
                best = row
                best_model = model

    if best is None or best_model is None:
        base_row["reason"] = "no validation-positive threshold"
        return base_row, pd.DataFrame(detail_rows)

    model_path = model_dir / f"{combo.symbol}_{combo.atom}_{combo.spec.name}_{best['model']}.joblib"
    model_path = Path(str(model_path).replace("/", "-"))
    payload = {
        "model": best_model,
        "feature_columns": feature_cols,
        "threshold": float(best["threshold"]),
        "symbol": combo.symbol,
        "atom": combo.atom,
        "strategy": combo.spec.strategy,
        "timeframe": combo.spec.timeframe,
        "spec_name": combo.spec.name,
        "params": combo.spec.params,
        "split": split.isoformat(),
        "validation_policy": {
            "val_frac": val_frac,
            "thresholds": thresholds,
            "min_fit": min_fit,
            "min_val": min_val,
            "min_oos": min_oos,
        },
    }
    joblib.dump(payload, model_path)

    out = {
        **base_row,
        "status": "ok",
        "reason": "",
        "model": best["model"],
        "threshold": best["threshold"],
        "model_path": str(model_path),
        "fit_trades": len(fit),
        "val_trades_raw": len(val),
        **{f"raw_fit_{k}": v for k, v in raw_fit_m.items()},
        **{f"raw_val_{k}": v for k, v in raw_val_m.items()},
        **{f"ml_val_{k}": best[f"val_{k}"] for k in ("trades", "net_r", "avg_r", "win_rate", "profit_factor", "max_dd_r")},
        **{f"ml_oos_{k}": best[f"oos_{k}"] for k in ("trades", "net_r", "avg_r", "win_rate", "profit_factor", "max_dd_r")},
    }
    out["oos_pf_delta"] = float(out["ml_oos_profit_factor"] - out["raw_oos_profit_factor"])
    out["oos_net_r_delta"] = float(out["ml_oos_net_r"] - out["raw_oos_net_r"])
    out["oos_trade_retention"] = safe_div(float(out["ml_oos_trades"]), float(out["raw_oos_trades"]))
    return out, pd.DataFrame(detail_rows)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows._"
    shown = frame[columns].copy()
    for column in shown.columns:
        if pd.api.types.is_float_dtype(shown[column]):
            shown[column] = shown[column].map(lambda x: f"{float(x):.4f}" if pd.notna(x) and math.isfinite(float(x)) else str(x))
        else:
            shown[column] = shown[column].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(str(row[column]) for column in columns) + " |" for _, row in shown.iterrows()]
    return "\n".join([header, sep, *body])


def write_report(args: argparse.Namespace, result: pd.DataFrame, all_trades: pd.DataFrame) -> Path:
    path = args.out_prefix.with_suffix(".md")
    ok = result[result["status"].eq("ok")].copy() if not result.empty else pd.DataFrame()
    improved = ok[ok["oos_pf_delta"] > 0].copy() if not ok.empty else pd.DataFrame()
    split = pd.Timestamp(parse_utc_datetime(args.split))
    all_trades_for_agg = all_trades.copy()
    if not all_trades_for_agg.empty:
        all_trades_for_agg["entry_time"] = pd.to_datetime(all_trades_for_agg["entry_time"], utc=True, errors="coerce")
    ml_selected = score_ml_selected_trades(result, all_trades)
    if not ml_selected.empty:
        ml_selected["entry_time"] = pd.to_datetime(ml_selected["entry_time"], utc=True, errors="coerce")

    def fmt_metrics(label: str, frame: pd.DataFrame) -> dict[str, Any]:
        m = metrics(frame)
        return {
            "set": label,
            "trades": m["trades"],
            "net_r": m["net_r"],
            "avg_r": m["avg_r"],
            "profit_factor": m["profit_factor"],
            "max_dd_r": m["max_dd_r"],
            "win_rate": m["win_rate"],
        }

    aggregate_rows: list[dict[str, Any]] = []
    if not all_trades_for_agg.empty:
        raw_oos = all_trades_for_agg[all_trades_for_agg["entry_time"] >= split]
        ml_oos = ml_selected[ml_selected["entry_time"] >= split] if not ml_selected.empty else pd.DataFrame()
        aggregate_rows.append(fmt_metrics("all raw OOS", raw_oos))
        aggregate_rows.append(fmt_metrics("all ML-selected OOS", ml_oos))
        if not improved.empty:
            keys = set((improved["symbol"] + "|" + improved["atom"] + "|" + improved["spec_name"]).tolist())
            raw_improved = raw_oos[raw_oos["combo_key"].isin(keys)]
            ml_improved = ml_oos[ml_oos["combo_key"].isin(keys)] if not ml_oos.empty else pd.DataFrame()
            aggregate_rows.append(fmt_metrics("improved-combos raw OOS", raw_improved))
            aggregate_rows.append(fmt_metrics("improved-combos ML-selected OOS", ml_improved))
    aggregate = pd.DataFrame(aggregate_rows)

    lines: list[str] = []
    lines.append("# Pine Atom Whitelist ML Filter")
    lines.append("")
    lines.append(f"Whitelist: `{args.whitelist}`")
    lines.append(f"Split: `{args.split}` | validation fraction inside train: `{args.val_frac}`")
    lines.append("")
    lines.append("Parameter selection is inherited from the prior train-only atom sweep. The ML model/threshold is chosen on the pre-OOS validation slice only; OOS is used only for evaluation.")
    lines.append("")
    lines.append(f"Combos evaluated: {len(result)} | model-ready: {len(ok)} | OOS PF improved: {len(improved)}")
    lines.append(f"Trades generated before filtering: {len(all_trades):,}")
    lines.append("")
    if not aggregate.empty:
        lines.append("## Aggregate OOS")
        lines.append("")
        lines.append(markdown_table(aggregate, ["set", "trades", "net_r", "avg_r", "profit_factor", "max_dd_r", "win_rate"]))
        lines.append("")
    if not ok.empty:
        lines.append("## Best OOS PF Improvements")
        lines.append("")
        cols = [
            "symbol",
            "atom",
            "model",
            "threshold",
            "raw_oos_trades",
            "raw_oos_profit_factor",
            "ml_oos_trades",
            "ml_oos_profit_factor",
            "oos_pf_delta",
            "ml_oos_net_r",
            "oos_trade_retention",
        ]
        lines.append(markdown_table(ok.sort_values(["oos_pf_delta", "ml_oos_profit_factor"], ascending=False).head(20), cols))
        lines.append("")
    if not improved.empty:
        lines.append("## Suggested Research Whitelist")
        lines.append("")
        suggested = improved[
            (improved["ml_oos_trades"] >= args.min_oos)
            & (improved["ml_oos_profit_factor"] >= improved["raw_oos_profit_factor"])
            & (improved["ml_oos_net_r"] > 0)
        ].sort_values(["ml_oos_profit_factor", "ml_oos_net_r"], ascending=False)
        cols = ["symbol", "atom", "model", "threshold", "ml_oos_trades", "ml_oos_net_r", "ml_oos_avg_r", "ml_oos_profit_factor", "model_path"]
        lines.append(markdown_table(suggested, cols))
        lines.append("")
    skipped = result[~result["status"].eq("ok")].copy() if not result.empty else pd.DataFrame()
    if not skipped.empty:
        lines.append("## Skipped")
        lines.append("")
        lines.append(markdown_table(skipped, ["symbol", "atom", "raw_train_trades", "raw_oos_trades", "feature_count", "reason"]))
        lines.append("")
    lines.append("## Caveat")
    lines.append("")
    lines.append("Filtered metrics are conservative because the base simulator first removes overlapping raw trades. If ML rejects an early trade, a live system could sometimes take later signals that were blocked in this research path.")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def score_ml_selected_trades(result: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    if result.empty or trades.empty:
        return pd.DataFrame()
    selected_frames: list[pd.DataFrame] = []
    ok = result[result["status"].eq("ok")].copy()
    for _, row in ok.iterrows():
        model_path = row.get("model_path")
        if not isinstance(model_path, str) or not model_path:
            continue
        path = Path(model_path)
        if not path.exists():
            continue
        payload = joblib.load(path)
        feature_cols = payload["feature_columns"]
        threshold = float(payload["threshold"])
        combo_key = f"{row['symbol']}|{row['atom']}|{row['spec_name']}"
        combo_trades = trades[trades["combo_key"].eq(combo_key)].copy()
        if combo_trades.empty:
            continue
        expanded = feature_table(combo_trades)
        for col in feature_cols:
            if col not in expanded:
                expanded[col] = math.nan
            expanded[col] = pd.to_numeric(expanded[col], errors="coerce")
        expanded["ml_prob"] = payload["model"].predict_proba(expanded[feature_cols])[:, 1]
        expanded["ml_threshold"] = threshold
        expanded["ml_model"] = row.get("model", "")
        selected_frames.append(expanded[expanded["ml_prob"] >= threshold].copy())
    return pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ML filters on selected Pine atom coin/parameter whitelist.")
    parser.add_argument("--whitelist", type=Path, default=DEFAULT_WHITELIST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    parser.add_argument("--train-start", default="2024-01-01")
    parser.add_argument("--split", default="2025-07-01")
    parser.add_argument("--end", default="2026-04-20")
    parser.add_argument("--max-combos", type=int, default=0)
    parser.add_argument("--best-per-coin", action="store_true")
    parser.add_argument("--fee-bps-per-side", type=float, default=6.5)
    parser.add_argument("--min-risk-pct", type=float, default=0.15)
    parser.add_argument("--val-frac", type=float, default=0.35)
    parser.add_argument("--thresholds", default="0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75")
    parser.add_argument("--min-fit", type=int, default=80)
    parser.add_argument("--min-val", type=int, default=35)
    parser.add_argument("--min-oos", type=int, default=80)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    model_dir = args.out_prefix.parent / f"{args.out_prefix.name}_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    train_start = pd.Timestamp(parse_utc_datetime(args.train_start))
    split = pd.Timestamp(parse_utc_datetime(args.split))
    end = pd.Timestamp(parse_utc_datetime(args.end))
    combos = load_whitelist(args.whitelist, args.max_combos, args.best_per_coin)
    print(f"ML filtering {len(combos)} coin/atom combos from {args.whitelist}", flush=True)
    result_rows: list[dict[str, Any]] = []
    detail_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    for pos, combo in enumerate(combos, start=1):
        print(f"[{pos}/{len(combos)}] {combo.symbol} {combo.atom} {combo.spec.name}", flush=True)
        try:
            trades = trades_for_combo(
                combo,
                cache_dir=args.cache_dir,
                train_start=train_start,
                end=end,
                fee_bps_per_side=args.fee_bps_per_side,
                min_risk_pct=args.min_risk_pct,
            )
            trade_frames.append(trades)
            row, details = choose_filter(
                combo,
                trades,
                split=split,
                val_frac=args.val_frac,
                thresholds=parse_float_list(args.thresholds),
                min_fit=args.min_fit,
                min_val=args.min_val,
                min_oos=args.min_oos,
                model_dir=model_dir,
                random_state=args.random_state,
            )
            result_rows.append(row)
            if not details.empty:
                detail_frames.append(details)
            if row.get("status") == "ok":
                print(
                    f"  raw_oos_pf={row['raw_oos_profit_factor']:.3f} -> "
                    f"ml_oos_pf={row['ml_oos_profit_factor']:.3f} "
                    f"trades={row['raw_oos_trades']}->{row['ml_oos_trades']} "
                    f"model={row['model']} thr={row['threshold']}",
                    flush=True,
                )
            else:
                print(f"  skipped: {row.get('reason')}", flush=True)
        except Exception as exc:
            result_rows.append(
                {
                    "symbol": combo.symbol,
                    "atom": combo.atom,
                    "strategy": combo.spec.strategy,
                    "timeframe": combo.spec.timeframe,
                    "spec_name": combo.spec.name,
                    "params_json": json.dumps(combo.spec.params, sort_keys=True),
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"  failed: {type(exc).__name__}: {exc}", flush=True)

    result = pd.DataFrame(result_rows)
    details = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    all_trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    ml_selected_trades = score_ml_selected_trades(result, all_trades)
    result_path = args.out_prefix.with_name(f"{args.out_prefix.name}_summary.csv")
    details_path = args.out_prefix.with_name(f"{args.out_prefix.name}_threshold_details.csv")
    trades_path = args.out_prefix.with_name(f"{args.out_prefix.name}_trades.csv")
    ml_trades_path = args.out_prefix.with_name(f"{args.out_prefix.name}_ml_selected_trades.csv")
    suggested_path = args.out_prefix.with_name(f"{args.out_prefix.name}_suggested.csv")
    result.to_csv(result_path, index=False)
    details.to_csv(details_path, index=False)
    all_trades.to_csv(trades_path, index=False)
    ml_selected_trades.to_csv(ml_trades_path, index=False)
    if not result.empty and "status" in result:
        suggested = result[
            result["status"].eq("ok")
            & (pd.to_numeric(result["ml_oos_trades"], errors="coerce") >= args.min_oos)
            & (pd.to_numeric(result["ml_oos_profit_factor"], errors="coerce") > pd.to_numeric(result["raw_oos_profit_factor"], errors="coerce"))
            & (pd.to_numeric(result["ml_oos_net_r"], errors="coerce") > 0)
        ].sort_values(["ml_oos_profit_factor", "ml_oos_net_r"], ascending=False)
    else:
        suggested = pd.DataFrame()
    suggested.to_csv(suggested_path, index=False)
    report = write_report(args, result, all_trades)
    print(f"Saved summary: {result_path}", flush=True)
    print(f"Saved threshold details: {details_path}", flush=True)
    print(f"Saved trades: {trades_path}", flush=True)
    print(f"Saved ML-selected trades: {ml_trades_path}", flush=True)
    print(f"Saved suggested: {suggested_path}", flush=True)
    print(f"Saved report: {report}", flush=True)
    if not result.empty:
        cols = [
            "symbol",
            "atom",
            "status",
            "model",
            "threshold",
            "raw_oos_trades",
            "raw_oos_profit_factor",
            "ml_oos_trades",
            "ml_oos_profit_factor",
            "oos_pf_delta",
        ]
        existing = [c for c in cols if c in result.columns]
        print(result.sort_values("oos_pf_delta", ascending=False, na_position="last")[existing].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
