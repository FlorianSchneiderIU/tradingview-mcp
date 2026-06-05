from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
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
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - only used when local env lacks sklearn
    SKLEARN_AVAILABLE = False

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except Exception:  # pragma: no cover - xgboost is optional in this repo
    XGBOOST_AVAILABLE = False

from scripts.backtest_pyharmonics_strategy import (  # noqa: E402
    HarmonicConfig,
    event_filter_matches,
    find_harmonic_events,
    run_backtest,
)
from scripts.backtest_wolfe_wave import (  # noqa: E402
    ensure_ohlcv_frame,
    load_ohlcv_csv,
    normalize_timeframe,
    resample_ohlc,
)
from scripts.run_pyharmonics_top100_lowpass import (  # noqa: E402
    event_cache_file,
    family_filter,
    load_event_cache,
    save_event_cache,
)


DEFAULT_SYMBOLS = "TAOUSDT,INJUSDT,DASHUSDT,ETHUSDT,FILUSDT"

NUMERIC_FEATURES = [
    "peak_spacing",
    "fib_tolerance",
    "confirm_bars",
    "entry_window_bars",
    "prz_atr_buffer",
    "stop_atr_buffer",
    "target_rr_planned",
    "breakeven_trigger_r",
    "max_hold_bars_config",
    "min_harmonic_quality_score",
    "entry_hour_utc",
    "entry_weekday",
    "trigger_touched_prz_num",
    "trigger_delay_bars",
    "completion_to_entry_bars",
    "detection_to_entry_bars",
    "entry_risk_pct",
    "fee_to_price_risk",
    "stop_distance_pct",
    "structural_gap_pct",
    "target_distance_pct",
    "completion_to_entry_pct",
    "completion_zone_width_pct",
    "direction_sign",
    "htf_1h_rsi",
    "htf_1h_ema_dist_atr",
    "htf_1h_ema_slope_atr",
    "htf_1h_atr_ratio",
    "htf_4h_rsi",
    "htf_4h_ema_dist_atr",
    "htf_4h_ema_slope_atr",
    "htf_4h_atr_ratio",
    "harmonic_point_count",
    "harmonic_quality_score",
    "harmonic_time_score",
    "harmonic_slope_score",
    "harmonic_compactness_score",
    "harmonic_bc_time_score",
    "harmonic_fib_score",
    "harmonic_ab_bars",
    "harmonic_bc_bars",
    "harmonic_cd_bars",
    "harmonic_ab_move_pct",
    "harmonic_bc_move_pct",
    "harmonic_cd_move_pct",
    "harmonic_cd_ab_time_ratio",
    "harmonic_ab_cd_time_balance",
    "harmonic_bc_ab_time_ratio",
    "harmonic_cd_ab_price_ratio",
    "harmonic_ab_cd_slope_balance",
    "harmonic_cd_prior_time_ratio",
    "harmonic_abc_retrace",
    "harmonic_bcd_extension",
]

BASE_CATEGORICAL_FEATURES = [
    "direction",
    "family",
    "pattern_name",
    "pattern_mode",
    "entry_mode",
    "entry_trigger",
    "trigger_candle_primary",
    "candle_filter",
    "trigger_candle_filter",
    "time_filter",
    "htf_filter",
    "trend_filter",
]


def parse_csv_values(raw: str, cast: Any = str) -> list[Any]:
    values: list[Any] = []
    for chunk in str(raw or "").split(","):
        text = chunk.strip()
        if text:
            values.append(cast(text))
    return values


def parse_symbols(raw: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in parse_csv_values(raw, str):
        symbol = item.strip().upper()
        if symbol and symbol not in seen:
            seen.add(symbol)
            out.append(symbol)
    return out


def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - older sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def pct_distance(a: pd.Series, b: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = pd.to_numeric(denominator, errors="coerce").replace(0.0, np.nan).abs()
    return (pd.to_numeric(a, errors="coerce") - pd.to_numeric(b, errors="coerce")).abs() / denom


def signed_pct_distance(a: pd.Series, b: pd.Series, denominator: pd.Series, sign: pd.Series) -> pd.Series:
    denom = pd.to_numeric(denominator, errors="coerce").replace(0.0, np.nan).abs()
    return (pd.to_numeric(a, errors="coerce") - pd.to_numeric(b, errors="coerce")) / denom * pd.to_numeric(sign, errors="coerce")


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
    wins = float(result[result > 0.0].sum())
    losses = float(result[result < 0.0].sum())
    equity = result.cumsum()
    drawdown = equity - equity.cummax()
    profit_factor = math.inf if losses == 0.0 and wins > 0.0 else (wins / abs(losses) if losses < 0.0 else 0.0)
    return {
        "trades": int(len(frame)),
        "net_r": float(result.sum()),
        "avg_r": float(result.mean()),
        "median_r": float(result.median()),
        "win_rate": float((result > 0.0).mean()),
        "profit_factor": float(profit_factor) if math.isfinite(profit_factor) else math.inf,
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


def one_trade_per_symbol_at_a_time(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    parts = [one_trade_at_a_time(group) for _, group in frame.groupby("symbol", dropna=False)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def top_candidate_per_event(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    scored = frame.copy()
    if "ml_prob" not in scored.columns:
        scored["ml_prob"] = 0.0
    sort_cols = ["event_key", "ml_prob", "target_rr_planned", "entry_time"]
    return (
        scored.sort_values(sort_cols, ascending=[True, False, False, True])
        .drop_duplicates("event_key", keep="first")
        .sort_values(["entry_time", "symbol"])
        .reset_index(drop=True)
    )


def portfolio_frame(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    mode = str(mode or "per_event_one_symbol").strip().lower()
    if mode == "candidate_rows":
        return frame.copy()
    if mode == "per_event":
        return top_candidate_per_event(frame)
    if mode == "per_event_one_trade":
        return one_trade_at_a_time(top_candidate_per_event(frame))
    if mode == "per_event_one_symbol":
        return one_trade_per_symbol_at_a_time(top_candidate_per_event(frame))
    raise ValueError(f"Unsupported portfolio mode: {mode!r}")


def feature_columns(feature_set: str) -> tuple[list[str], list[str]]:
    mode = str(feature_set or "no_symbol").strip().lower()
    categorical = list(BASE_CATEGORICAL_FEATURES)
    if mode == "with_symbol":
        categorical = ["symbol", *categorical]
    elif mode != "no_symbol":
        raise ValueError(f"Unsupported feature set: {feature_set!r}")
    return list(NUMERIC_FEATURES), categorical


def build_model(name: str, numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", one_hot_encoder())])
    preprocessor = ColumnTransformer(
        [("num", numeric, numeric_features), ("cat", categorical, categorical_features)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    model_name = str(name).strip().lower()
    if model_name == "logreg":
        estimator = LogisticRegression(C=0.35, class_weight="balanced", max_iter=3000, random_state=31)
    elif model_name == "rf":
        estimator = RandomForestClassifier(
            n_estimators=500,
            max_depth=5,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=31,
            n_jobs=-1,
        )
    elif model_name == "extratrees":
        estimator = ExtraTreesClassifier(
            n_estimators=500,
            max_depth=5,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=31,
            n_jobs=-1,
        )
    elif model_name == "hgb":
        estimator = HistGradientBoostingClassifier(
            learning_rate=0.035,
            max_iter=130,
            max_leaf_nodes=9,
            min_samples_leaf=18,
            l2_regularization=1.0,
            random_state=31,
        )
    elif model_name == "xgb":
        if not XGBOOST_AVAILABLE:
            raise RuntimeError("xgboost is not installed in this Python environment.")
        estimator = XGBClassifier(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.035,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=8,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=31,
            n_jobs=2,
        )
    else:
        raise ValueError(f"Unknown model: {name!r}")
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def available_models(raw: str) -> list[str]:
    models: list[str] = []
    for name in parse_csv_values(raw, str):
        model = name.strip().lower()
        if model == "xgb" and not XGBOOST_AVAILABLE:
            print("Skipping xgb: xgboost is not installed.", flush=True)
            continue
        models.append(model)
    return models


def build_configs(args: argparse.Namespace) -> list[HarmonicConfig]:
    configs: list[HarmonicConfig] = []
    for pattern_tf in parse_csv_values(args.pattern_tfs, str):
        for family in parse_csv_values(args.families, str):
            for peak_spacing in parse_csv_values(args.peak_spacings, int):
                for fib_tolerance in parse_csv_values(args.fib_tolerances, float):
                    for confirm_bars in parse_csv_values(args.confirm_bars, int):
                        for entry_window in parse_csv_values(args.entry_window_bars, int):
                            for entry_mode in parse_csv_values(args.entry_modes, str):
                                for time_filter in parse_csv_values(args.time_filters, str):
                                    for candle_filter in parse_csv_values(args.candle_filters, str):
                                        for stop_buffer in parse_csv_values(args.stop_buffers, float):
                                            for min_quality in parse_csv_values(args.min_quality_scores, float):
                                                for rr in parse_csv_values(args.rrs, float):
                                                    for breakeven in parse_csv_values(args.breakeven_triggers, float):
                                                        configs.append(
                                                            HarmonicConfig(
                                                                pattern_tf=normalize_timeframe(pattern_tf),
                                                                family=family.strip().upper(),
                                                                pattern_mode=args.pattern_mode.strip().lower(),
                                                                forming_percent_c_to_d=float(args.forming_percent_c_to_d),
                                                                peak_spacing=int(peak_spacing),
                                                                fib_tolerance=float(fib_tolerance),
                                                                pattern_lookback_bars=int(args.pattern_lookback_bars),
                                                                pattern_step_bars=int(args.pattern_step_bars),
                                                                search_limit_to=int(args.search_limit_to),
                                                                confirm_bars=int(confirm_bars),
                                                                entry_window_bars=int(entry_window),
                                                                entry_mode=entry_mode.strip().lower(),
                                                                prz_atr_buffer=float(args.prz_atr_buffer),
                                                                candle_filter=candle_filter.strip().lower(),
                                                                trigger_candle_filter=args.trigger_candle_filter.strip().lower(),
                                                                pattern_name_filter=args.pattern_name_filter.strip().upper(),
                                                                direction_filter=args.direction_filter.strip().lower(),
                                                                stop_atr_buffer=float(stop_buffer),
                                                                rr=float(rr),
                                                                breakeven_trigger_r=float(breakeven),
                                                                min_harmonic_quality_score=float(min_quality),
                                                                max_hold_bars=int(args.max_hold_bars),
                                                                trend_filter=args.trend_filter.strip().lower(),
                                                                time_filter=time_filter.strip().lower(),
                                                                htf_filter=args.htf_filter.strip().lower(),
                                                                htf_stretch_atr=float(args.htf_stretch_atr),
                                                                htf_rsi_extreme=float(args.htf_rsi_extreme),
                                                                fee_bps_side=float(args.fee_bps_side),
                                                                slippage_bps_side=float(args.slippage_bps_side),
                                                                max_fee_to_price_risk=float(args.max_fee_to_price_risk),
                                                                min_entry_risk_pct=float(args.min_entry_risk_pct),
                                                                risk_fraction=float(args.risk_fraction),
                                                                one_trade_at_a_time=False,
                                                            )
                                                        )
    if args.limit_configs > 0:
        configs = configs[: int(args.limit_configs)]
    return configs


def load_symbol_frame(symbol: str, data_dir: Path, exec_tf: str) -> pd.DataFrame:
    path = data_dir / f"{symbol.lower()}_{normalize_timeframe(exec_tf)}_bybit.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing cached candle CSV for {symbol}: {path}")
    return ensure_ohlcv_frame(load_ohlcv_csv(path))


def build_symbol_candidates(symbol: str, frame: pd.DataFrame, configs: list[HarmonicConfig], args: argparse.Namespace) -> pd.DataFrame:
    event_cache: dict[tuple[Any, ...], list[Any]] = {}
    parts: list[pd.DataFrame] = []
    cache_family = args.event_cache_family.strip()
    if not cache_family:
        cache_family = "+".join(sorted({token for cfg in configs for token in family_filter(cfg.family)}))
    event_cache_symbol_dir = Path(args.event_cache_symbol_dir) if args.event_cache_symbol_dir else None
    write_event_cache_symbol_dir = Path(args.write_event_cache_symbol_dir) if args.write_event_cache_symbol_dir else None
    for idx, cfg in enumerate(configs, start=1):
        key = (
            cfg.pattern_tf,
            cfg.pattern_mode,
            cfg.peak_spacing,
            cfg.fib_tolerance,
            cfg.forming_percent_c_to_d,
            cfg.pattern_lookback_bars,
            cfg.pattern_step_bars,
            cfg.search_limit_to,
        )
        if key not in event_cache:
            pattern_df = resample_ohlc(frame, cfg.pattern_tf)
            cached_events = None
            if event_cache_symbol_dir is not None:
                disk_path = event_cache_file(
                    symbol_dir=event_cache_symbol_dir,
                    symbol=symbol,
                    cache_key=key,
                    cache_family=cache_family or cfg.family,
                    frame=frame,
                )
                cached_events = load_event_cache(disk_path)
                if cached_events is not None and args.progress_every:
                    print(f"[{symbol}] event cache hit key={len(event_cache) + 1} events={len(cached_events)}", flush=True)
            if cached_events is None:
                event_cfg = HarmonicConfig(**{**asdict(cfg), "family": cache_family or cfg.family})
                cached_events = find_harmonic_events(
                    pattern_df,
                    event_cfg,
                    symbol=symbol,
                    progress_label=f"[{symbol}]" if args.event_progress_every > 0 else None,
                    progress_every_chunks=int(args.event_progress_every),
                )
                if write_event_cache_symbol_dir is not None:
                    disk_path = event_cache_file(
                        symbol_dir=write_event_cache_symbol_dir,
                        symbol=symbol,
                        cache_key=key,
                        cache_family=cache_family or cfg.family,
                        frame=frame,
                    )
                    save_event_cache(disk_path, cached_events)
            event_cache[key] = cached_events
        events = [event for event in event_cache[key] if event_filter_matches(event, cfg)]
        trades = run_backtest(frame, cfg, symbol=symbol, precomputed_events=events)
        if trades.empty:
            continue
        trades.insert(0, "candidate_config_id", idx)
        trades["candidate_config_json"] = json.dumps(asdict(cfg), sort_keys=True)
        parts.append(trades)
        if args.progress_every > 0 and idx % args.progress_every == 0:
            print(
                f"[{symbol}] config {idx}/{len(configs)} cache_keys={len(event_cache)} rows={sum(len(part) for part in parts)}",
                flush=True,
            )
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def engineer_features(dataset: pd.DataFrame, *, label_min_r: float, exec_tf: str) -> pd.DataFrame:
    out = dataset.copy()
    for column in ["completion_time", "detection_time", "trigger_time", "entry_time", "exit_time", "trigger_break_time"]:
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], utc=True, errors="coerce")
    out["result_r"] = pd.to_numeric(out["r_multiple_net"], errors="coerce").fillna(0.0)
    out["label"] = (out["result_r"] >= float(label_min_r)).astype(int)
    out["entry_year"] = pd.to_datetime(out["entry_time"], utc=True, errors="coerce").dt.year
    out["direction_sign"] = np.where(out["direction"].astype(str).str.lower().eq("long"), 1.0, -1.0)
    out["trigger_touched_prz_num"] = out.get("trigger_touched_prz", False).astype(str).str.lower().isin(["true", "1", "yes"]).astype(float)
    out["max_hold_bars_config"] = pd.to_numeric(out.get("max_hold_bars", np.nan), errors="coerce")
    if out["max_hold_bars_config"].notna().sum() == 0:
        out["max_hold_bars_config"] = 192.0

    entry = pd.to_numeric(out["entry_price"], errors="coerce")
    stop = pd.to_numeric(out["stop_price"], errors="coerce")
    structural_stop = pd.to_numeric(out["structural_stop_price"], errors="coerce")
    target = pd.to_numeric(out["target_price"], errors="coerce")
    completion = pd.to_numeric(out["completion_price"], errors="coerce")
    completion_min = pd.to_numeric(out["completion_min_price"], errors="coerce")
    completion_max = pd.to_numeric(out["completion_max_price"], errors="coerce")
    sign = pd.to_numeric(out["direction_sign"], errors="coerce")

    out["stop_distance_pct"] = pct_distance(entry, stop, entry)
    out["structural_gap_pct"] = pct_distance(stop, structural_stop, entry)
    out["target_distance_pct"] = pct_distance(entry, target, entry)
    out["completion_to_entry_pct"] = signed_pct_distance(entry, completion, entry, sign)
    out["completion_zone_width_pct"] = (completion_max - completion_min).abs() / completion.replace(0.0, np.nan).abs()
    for source, target_col in [
        ("harmonic_ab_move", "harmonic_ab_move_pct"),
        ("harmonic_bc_move", "harmonic_bc_move_pct"),
        ("harmonic_cd_move", "harmonic_cd_move_pct"),
    ]:
        out[target_col] = pd.to_numeric(out.get(source, np.nan), errors="coerce").abs() / completion.replace(0.0, np.nan).abs()

    bar_minutes = max(1.0, pd.Timedelta(exec_tf).total_seconds() / 60.0) if exec_tf.endswith("min") else 15.0
    if exec_tf.endswith("m") and not exec_tf.endswith("min"):
        bar_minutes = float(exec_tf[:-1])
    out["completion_to_entry_bars"] = (
        (out["entry_time"] - out["completion_time"]).dt.total_seconds() / (bar_minutes * 60.0)
    )
    out["detection_to_entry_bars"] = (
        (out["entry_time"] - out["detection_time"]).dt.total_seconds() / (bar_minutes * 60.0)
    )
    trigger_time = out["trigger_time"].where(out["trigger_time"].notna(), out["detection_time"])
    out["trigger_delay_bars"] = (out["entry_time"] - trigger_time).dt.total_seconds() / (bar_minutes * 60.0)

    for column in NUMERIC_FEATURES:
        if column not in out.columns:
            out[column] = np.nan
        out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in ["symbol", *BASE_CATEGORICAL_FEATURES]:
        if column not in out.columns:
            out[column] = "unknown"
        out[column] = out[column].fillna("unknown").astype(str)
    out["candidate_id"] = (
        out["event_key"].astype(str)
        + "|cfg="
        + out["candidate_config_id"].astype(str)
        + "|entry="
        + out["entry_time"].astype(str)
    )
    return out.sort_values(["entry_time", "symbol", "candidate_config_id"]).reset_index(drop=True)


def build_dataset(args: argparse.Namespace) -> pd.DataFrame:
    configs = build_configs(args)
    if not configs:
        raise RuntimeError("No pyharmonics configs were generated.")
    print(f"Building pyharmonics ML dataset symbols={args.symbols} configs={len(configs)}", flush=True)
    parts: list[pd.DataFrame] = []
    for symbol in parse_symbols(args.symbols):
        frame = load_symbol_frame(symbol, args.data_dir, args.exec_tf)
        candidates = build_symbol_candidates(symbol, frame, configs, args)
        print(f"[{symbol}] candidate rows={len(candidates)}", flush=True)
        if not candidates.empty:
            parts.append(candidates)
    if not parts:
        raise RuntimeError("No candidate trades were produced for the requested survivor cluster.")
    dataset = pd.concat(parts, ignore_index=True)
    return engineer_features(dataset, label_min_r=float(args.label_min_r), exec_tf=normalize_timeframe(args.exec_tf))


def infer_entry_mode(frame: pd.DataFrame) -> pd.Series:
    if "entry_mode" in frame.columns:
        values = frame["entry_mode"].fillna("").astype(str)
    else:
        values = pd.Series([""] * len(frame), index=frame.index)
    trigger = frame.get("entry_trigger", pd.Series([""] * len(frame), index=frame.index)).fillna("").astype(str).str.lower()
    values = values.where(values.str.len() > 0, np.where(trigger.str.contains("trigger_close_break"), "trigger_close_break", ""))
    values = values.where(values.astype(str).str.len() > 0, np.where(trigger.str.contains("trigger_break"), "trigger_break", ""))
    values = values.where(values.astype(str).str.len() > 0, "next_open")
    return values.astype(str).str.lower()


def cached_trade_files(args: argparse.Namespace) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for item in parse_csv_values(args.input_trade_files, str):
        path = Path(item)
        if path.exists() and path.resolve() not in seen:
            seen.add(path.resolve())
            files.append(path)
    patterns = ["*_pyharmonics_selected_trades.csv", "*_quality_be_best_trades.csv"]
    for item in parse_csv_values(args.input_trade_dirs, str):
        root = Path(item)
        if not root.exists():
            continue
        for pattern in patterns:
            for path in root.glob(pattern):
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    files.append(path)
    return sorted(files)


def load_cached_trade_dataset(args: argparse.Namespace) -> pd.DataFrame:
    symbols = set(parse_symbols(args.symbols))
    parts: list[pd.DataFrame] = []
    files = cached_trade_files(args)
    if not files:
        raise RuntimeError("No cached pyharmonics trade files found.")
    for file_index, path in enumerate(files, start=1):
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            print(f"Skipping unreadable trade file {path}: {exc}", flush=True)
            continue
        if frame.empty or "symbol" not in frame.columns:
            continue
        frame["symbol"] = frame["symbol"].astype(str).str.upper()
        frame = frame[frame["symbol"].isin(symbols)].copy()
        if frame.empty:
            continue
        frame["source_file"] = str(path)
        frame["source_run"] = path.parent.name
        if "candidate_config_id" not in frame.columns:
            frame.insert(0, "candidate_config_id", file_index)
        else:
            frame["candidate_config_id"] = pd.to_numeric(frame["candidate_config_id"], errors="coerce").fillna(file_index).astype(int)
        if "candidate_config_json" not in frame.columns:
            config_cols = [
                "pattern_tf",
                "family",
                "pattern_mode",
                "peak_spacing",
                "fib_tolerance",
                "confirm_bars",
                "entry_window_bars",
                "entry_mode",
                "prz_atr_buffer",
                "candle_filter",
                "trigger_candle_filter",
                "pattern_name_filter",
                "direction_filter",
                "stop_atr_buffer",
                "target_rr_planned",
                "breakeven_trigger_r",
                "max_hold_bars",
                "trend_filter",
                "time_filter",
                "htf_filter",
            ]
            payload = {column: str(frame[column].iloc[0]) for column in config_cols if column in frame.columns}
            frame["candidate_config_json"] = json.dumps(payload, sort_keys=True)
        frame["entry_mode"] = infer_entry_mode(frame)
        if "trigger_candle_filter" not in frame.columns:
            frame["trigger_candle_filter"] = "none"
        if "time_filter" not in frame.columns:
            frame["time_filter"] = "unknown"
        if "htf_filter" not in frame.columns:
            frame["htf_filter"] = "unknown"
        if "trend_filter" not in frame.columns:
            frame["trend_filter"] = "unknown"
        if "candle_filter" not in frame.columns:
            frame["candle_filter"] = "unknown"
        if "min_harmonic_quality_score" not in frame.columns:
            frame["min_harmonic_quality_score"] = 0.0
        if "max_hold_bars" not in frame.columns:
            frame["max_hold_bars"] = np.nan
        parts.append(frame)
    if not parts:
        raise RuntimeError("Cached pyharmonics files did not contain any requested survivor-cluster trades.")
    dataset = pd.concat(parts, ignore_index=True)
    before = len(dataset)
    dedupe_cols = [col for col in ["symbol", "event_key", "candidate_config_id", "entry_time", "stop_price", "target_price"] if col in dataset.columns]
    dataset = dataset.drop_duplicates(dedupe_cols).reset_index(drop=True)
    print(f"Loaded cached pyharmonics trades rows={len(dataset)} deduped_from={before} files={len(files)}", flush=True)
    return engineer_features(dataset, label_min_r=float(args.label_min_r), exec_tf=normalize_timeframe(args.exec_tf))


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
    portfolio_mode: str,
    min_val_trades: int,
    smooth_radius: int,
    min_edge_r: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for _, candidate in candidate_thresholds(validation, thresholds, keep_fracs).iterrows():
        threshold = float(candidate["threshold"])
        selected = validation[pd.to_numeric(validation["ml_prob"], errors="coerce") >= threshold].copy()
        metrics = trade_metrics(portfolio_frame(selected, portfolio_mode))
        row = candidate.to_dict()
        row.update(metrics)
        row["eligible"] = bool(metrics["trades"] >= int(min_val_trades))
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    if table.empty:
        return {"gate": "all", "threshold": -math.inf, "keep_frac": 1.0, "threshold_table": table}

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

    all_row = table[table["gate"].eq("all")].iloc[0]
    best = table.sort_values(["selection_score", "trades", "net_r"], ascending=[False, False, False]).iloc[0]
    if best["gate"] != "all" and float(best["selection_score"]) < float(all_row["avg_r"]) + float(min_edge_r):
        best = all_row
    out = best.to_dict()
    out["threshold_table"] = table
    return out


def score_frame(model: Pipeline, frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = frame.copy()
    out["ml_prob"] = model.predict_proba(out[features])[:, 1]
    return out


def summarize_scored(scored: pd.DataFrame, selected: pd.DataFrame, *, scope: str, model: str, feature_set: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for portfolio in ["candidate_rows", "per_event", "per_event_one_symbol"]:
        all_frame = portfolio_frame(scored, portfolio)
        selected_frame = portfolio_frame(selected, portfolio)
        row: dict[str, Any] = {"scope": scope, "model": model, "feature_set": feature_set, "portfolio": portfolio}
        row.update(with_prefix("all", trade_metrics(all_frame)))
        row.update(with_prefix("selected", trade_metrics(selected_frame)))
        row["delta_avg_r"] = row["selected_avg_r"] - row["all_avg_r"]
        row["delta_net_r"] = row["selected_net_r"] - row["all_net_r"]
        row["kept_share"] = row["selected_trades"] / row["all_trades"] if row["all_trades"] else 0.0
        rows.append(row)
    return rows


def aggregate_group(
    scored: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    scope: str,
    model: str,
    feature_set: str,
    group_column: str,
    portfolio_mode: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if scored.empty or group_column not in scored.columns:
        return pd.DataFrame()
    selected_ids = set(selected["candidate_id"].astype(str)) if "candidate_id" in selected.columns else set()
    for value, group in scored.groupby(group_column, dropna=False):
        selected_group = group[group["candidate_id"].astype(str).isin(selected_ids)].copy()
        all_frame = portfolio_frame(group, portfolio_mode)
        selected_frame = portfolio_frame(selected_group, portfolio_mode)
        row: dict[str, Any] = {
            "scope": scope,
            "model": model,
            "feature_set": feature_set,
            group_column: value,
            "portfolio": portfolio_mode,
        }
        row.update(with_prefix("all", trade_metrics(all_frame)))
        row.update(with_prefix("selected", trade_metrics(selected_frame)))
        row["delta_avg_r"] = row["selected_avg_r"] - row["all_avg_r"]
        row["delta_net_r"] = row["selected_net_r"] - row["all_net_r"]
        row["kept_share"] = row["selected_trades"] / row["all_trades"] if row["all_trades"] else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def run_pooled_walk_forward(dataset: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    thresholds = parse_csv_values(args.thresholds, float)
    keep_fracs = parse_csv_values(args.keep_fracs, float)
    folds = fold_schedule(dataset, args.first_test_year)
    if not folds:
        raise RuntimeError("Not enough annual data to build pyharmonics ML folds.")

    fold_rows: list[dict[str, Any]] = []
    threshold_parts: list[pd.DataFrame] = []
    scored_parts: list[pd.DataFrame] = []
    selected_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    group_parts: list[pd.DataFrame] = []

    for feature_set in parse_csv_values(args.feature_sets, str):
        numeric, categorical = feature_columns(feature_set)
        features = numeric + categorical
        for model_name in available_models(args.models):
            model_scored: list[pd.DataFrame] = []
            model_selected: list[pd.DataFrame] = []
            for fold_index, (_, val_year, test_year) in enumerate(folds, start=1):
                train = dataset[dataset["entry_year"] < val_year].copy()
                validation = dataset[dataset["entry_year"] == val_year].copy()
                test = dataset[dataset["entry_year"] == test_year].copy()
                if len(train) < args.min_train_rows or len(validation) < args.min_val_rows or test.empty:
                    continue
                if train["label"].nunique() < 2:
                    continue
                estimator = build_model(model_name, numeric, categorical)
                estimator.fit(train[features], train["label"].astype(int))
                validation = score_frame(estimator, validation, features)
                test = score_frame(estimator, test, features)
                gate = choose_stable_gate(
                    validation,
                    thresholds,
                    keep_fracs,
                    portfolio_mode=args.selection_portfolio,
                    min_val_trades=args.min_val_trades,
                    smooth_radius=args.smooth_radius,
                    min_edge_r=args.min_edge_r,
                )
                threshold_table = gate.pop("threshold_table", pd.DataFrame())
                if not threshold_table.empty:
                    threshold_table.insert(0, "fold", fold_index)
                    threshold_table.insert(1, "scope", "pooled_annual")
                    threshold_table.insert(2, "model", model_name)
                    threshold_table.insert(3, "feature_set", feature_set)
                    threshold_table.insert(4, "val_year", val_year)
                    threshold_table.insert(5, "test_year", test_year)
                    threshold_parts.append(threshold_table)
                threshold = float(gate["threshold"])
                selected_test = test[pd.to_numeric(test["ml_prob"], errors="coerce") >= threshold].copy()
                for frame in [test, selected_test]:
                    frame["scope"] = "pooled_annual"
                    frame["fold"] = fold_index
                    frame["model"] = model_name
                    frame["feature_set"] = feature_set
                    frame["val_year"] = val_year
                    frame["test_year"] = test_year
                    frame["selected_gate"] = gate["gate"]
                    frame["selected_threshold"] = threshold
                row: dict[str, Any] = {
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
                row.update(with_prefix("val_clf", classifier_metrics(validation)))
                row.update(with_prefix("test_clf", classifier_metrics(test)))
                row.update(with_prefix("val_all", trade_metrics(portfolio_frame(validation, args.selection_portfolio))))
                row.update(with_prefix("val_selected", trade_metrics(portfolio_frame(validation[validation["ml_prob"] >= threshold], args.selection_portfolio))))
                row.update(with_prefix("test_all", trade_metrics(portfolio_frame(test, args.selection_portfolio))))
                row.update(with_prefix("test_selected", trade_metrics(portfolio_frame(selected_test, args.selection_portfolio))))
                fold_rows.append(row)
                model_scored.append(test)
                model_selected.append(selected_test)
                scored_parts.append(test)
                selected_parts.append(selected_test)
            scored_model = pd.concat(model_scored, ignore_index=True) if model_scored else pd.DataFrame()
            selected_model = pd.concat(model_selected, ignore_index=True) if model_selected else pd.DataFrame()
            summary_rows.extend(
                summarize_scored(scored_model, selected_model, scope="pooled_annual", model=model_name, feature_set=feature_set)
            )
            for group_column in ["symbol", "entry_mode", "pattern_name"]:
                group_parts.append(
                    aggregate_group(
                        scored_model,
                        selected_model,
                        scope="pooled_annual",
                        model=model_name,
                        feature_set=feature_set,
                        group_column=group_column,
                        portfolio_mode=args.selection_portfolio,
                    )
                )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["portfolio", "delta_avg_r", "selected_avg_r"], ascending=[True, False, False]
    )
    folds_out = pd.DataFrame(fold_rows)
    thresholds_out = pd.concat(threshold_parts, ignore_index=True) if threshold_parts else pd.DataFrame()
    scored_out = pd.concat(scored_parts, ignore_index=True) if scored_parts else pd.DataFrame()
    selected_out = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    groups_out = pd.concat([part for part in group_parts if not part.empty], ignore_index=True) if group_parts else pd.DataFrame()
    scored_selected = pd.concat(
        [scored_out.assign(_selected_oos=False), selected_out.assign(_selected_oos=True)],
        ignore_index=True,
    )
    return summary, groups_out, folds_out, thresholds_out, scored_selected


def run_symbol_holdout_walk_forward(dataset: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    thresholds = parse_csv_values(args.thresholds, float)
    keep_fracs = parse_csv_values(args.keep_fracs, float)
    folds = fold_schedule(dataset, args.first_test_year)
    symbols = sorted(dataset["symbol"].dropna().astype(str).unique())

    fold_rows: list[dict[str, Any]] = []
    threshold_parts: list[pd.DataFrame] = []
    scored_parts: list[pd.DataFrame] = []
    selected_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    group_parts: list[pd.DataFrame] = []

    for feature_set in parse_csv_values(args.feature_sets, str):
        numeric, categorical = feature_columns(feature_set)
        features = numeric + categorical
        for model_name in available_models(args.models):
            model_scored: list[pd.DataFrame] = []
            model_selected: list[pd.DataFrame] = []
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
                    if train["label"].nunique() < 2:
                        continue
                    estimator = build_model(model_name, numeric, categorical)
                    estimator.fit(train[features], train["label"].astype(int))
                    validation = score_frame(estimator, validation, features)
                    test = score_frame(estimator, test, features)
                    gate = choose_stable_gate(
                        validation,
                        thresholds,
                        keep_fracs,
                        portfolio_mode=args.selection_portfolio,
                        min_val_trades=args.min_val_trades,
                        smooth_radius=args.smooth_radius,
                        min_edge_r=args.min_edge_r,
                    )
                    threshold_table = gate.pop("threshold_table", pd.DataFrame())
                    if not threshold_table.empty:
                        threshold_table.insert(0, "fold", fold_index)
                        threshold_table.insert(1, "scope", "leave_one_symbol_annual")
                        threshold_table.insert(2, "model", model_name)
                        threshold_table.insert(3, "feature_set", feature_set)
                        threshold_table.insert(4, "holdout_symbol", holdout_symbol)
                        threshold_table.insert(5, "val_year", val_year)
                        threshold_table.insert(6, "test_year", test_year)
                        threshold_parts.append(threshold_table)
                    threshold = float(gate["threshold"])
                    selected_test = test[pd.to_numeric(test["ml_prob"], errors="coerce") >= threshold].copy()
                    for frame in [test, selected_test]:
                        frame["scope"] = "leave_one_symbol_annual"
                        frame["fold"] = fold_index
                        frame["model"] = model_name
                        frame["feature_set"] = feature_set
                        frame["holdout_symbol"] = holdout_symbol
                        frame["val_year"] = val_year
                        frame["test_year"] = test_year
                        frame["selected_gate"] = gate["gate"]
                        frame["selected_threshold"] = threshold
                    row: dict[str, Any] = {
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
                    row.update(with_prefix("val_clf", classifier_metrics(validation)))
                    row.update(with_prefix("test_clf", classifier_metrics(test)))
                    row.update(with_prefix("val_all", trade_metrics(portfolio_frame(validation, args.selection_portfolio))))
                    row.update(with_prefix("val_selected", trade_metrics(portfolio_frame(validation[validation["ml_prob"] >= threshold], args.selection_portfolio))))
                    row.update(with_prefix("test_all", trade_metrics(portfolio_frame(test, args.selection_portfolio))))
                    row.update(with_prefix("test_selected", trade_metrics(portfolio_frame(selected_test, args.selection_portfolio))))
                    fold_rows.append(row)
                    model_scored.append(test)
                    model_selected.append(selected_test)
                    scored_parts.append(test)
                    selected_parts.append(selected_test)
            scored_model = pd.concat(model_scored, ignore_index=True) if model_scored else pd.DataFrame()
            selected_model = pd.concat(model_selected, ignore_index=True) if model_selected else pd.DataFrame()
            summary_rows.extend(
                summarize_scored(
                    scored_model,
                    selected_model,
                    scope="leave_one_symbol_annual",
                    model=model_name,
                    feature_set=feature_set,
                )
            )
            for group_column in ["symbol", "entry_mode", "pattern_name"]:
                group_parts.append(
                    aggregate_group(
                        scored_model,
                        selected_model,
                        scope="leave_one_symbol_annual",
                        model=model_name,
                        feature_set=feature_set,
                        group_column=group_column,
                        portfolio_mode=args.selection_portfolio,
                    )
                )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["portfolio", "delta_avg_r", "selected_avg_r"], ascending=[True, False, False]
    )
    folds_out = pd.DataFrame(fold_rows)
    thresholds_out = pd.concat(threshold_parts, ignore_index=True) if threshold_parts else pd.DataFrame()
    scored_out = pd.concat(scored_parts, ignore_index=True) if scored_parts else pd.DataFrame()
    selected_out = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    groups_out = pd.concat([part for part in group_parts if not part.empty], ignore_index=True) if group_parts else pd.DataFrame()
    scored_selected = pd.concat(
        [scored_out.assign(_selected_oos=False), selected_out.assign(_selected_oos=True)],
        ignore_index=True,
    )
    return summary, groups_out, folds_out, thresholds_out, scored_selected


def final_feature_importance(dataset: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature_set in parse_csv_values(args.feature_sets, str):
        numeric, categorical = feature_columns(feature_set)
        features = numeric + categorical
        for model_name in available_models(args.models):
            if model_name == "hgb":
                continue
            if dataset["label"].nunique() < 2:
                continue
            estimator = build_model(model_name, numeric, categorical)
            estimator.fit(dataset[features], dataset["label"].astype(int))
            preprocessor = estimator.named_steps["preprocessor"]
            feature_names = list(preprocessor.get_feature_names_out())
            model = estimator.named_steps["model"]
            if hasattr(model, "feature_importances_"):
                values = np.asarray(model.feature_importances_, dtype=float)
                for name, value in zip(feature_names, values):
                    rows.append(
                        {
                            "model": model_name,
                            "feature_set": feature_set,
                            "method": "model_importance",
                            "feature": name,
                            "importance": float(value),
                        }
                    )
            elif hasattr(model, "coef_"):
                values = np.asarray(model.coef_[0], dtype=float)
                for name, value in zip(feature_names, values):
                    rows.append(
                        {
                            "model": model_name,
                            "feature_set": feature_set,
                            "method": "coefficient",
                            "feature": name,
                            "importance": float(value),
                            "abs_importance": float(abs(value)),
                        }
                    )
            if args.permutation_importance_rows > 0 and len(dataset) >= 50:
                sample = dataset.sample(
                    n=min(int(args.permutation_importance_rows), len(dataset)),
                    random_state=31,
                )
                try:
                    permutation = permutation_importance(
                        estimator,
                        sample[features],
                        sample["label"].astype(int),
                        n_repeats=3,
                        random_state=31,
                        n_jobs=1,
                    )
                except Exception:
                    continue
                for name, value, std in zip(features, permutation.importances_mean, permutation.importances_std):
                    rows.append(
                        {
                            "model": model_name,
                            "feature_set": feature_set,
                            "method": "permutation_auc",
                            "feature": name,
                            "importance": float(value),
                            "importance_std": float(std),
                        }
                    )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    sort_col = "abs_importance" if "abs_importance" in out.columns else "importance"
    return out.sort_values(["method", "model", sort_col], ascending=[True, True, False]).reset_index(drop=True)


def train_final_model(dataset: pd.DataFrame, args: argparse.Namespace) -> None:
    model_name = str(args.final_model).strip().lower()
    feature_set = str(args.final_feature_set).strip().lower()
    numeric, categorical = feature_columns(feature_set)
    features = numeric + categorical
    estimator = build_model(model_name, numeric, categorical)
    estimator.fit(dataset[features], dataset["label"].astype(int))
    payload = {
        "model": estimator,
        "model_name": model_name,
        "feature_set": feature_set,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "label_min_r": float(args.label_min_r),
        "symbols": parse_symbols(args.symbols),
        "trained_rows": int(len(dataset)),
        "config": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
    }
    output_path = args.output_prefix.with_name(args.output_prefix.name + f"_{feature_set}_{model_name}.joblib")
    joblib.dump(payload, output_path)
    print(f"Saved final model {output_path}", flush=True)


def write_table(path: Path, table: pd.DataFrame) -> None:
    table.to_csv(path, index=False)
    print(f"Saved {path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ML entry selectors for the pyharmonics survivor cluster.")
    parser.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    parser.add_argument("--data-dir", type=Path, default=Path("scripts/data_pyharmonics_top100_fast_15m_abcd_xabcd_5y"))
    parser.add_argument("--dataset-source", choices=["generated", "cached_trades"], default="generated")
    parser.add_argument(
        "--input-trade-dirs",
        default=(
            "scripts/pyharmonics_inj_config_portability_65_20260603/per_symbol,"
            "scripts/pyharmonics_top100_fast_15m_abcd_xabcd_5y_20260603_100738/per_symbol,"
            "scripts/pyharmonics_context_deep_inj_link_ltc_20260603,"
            "scripts/pyharmonics_quality_be_targeted_inj_link_ltc_20260603"
        ),
    )
    parser.add_argument("--input-trade-files", default="")
    parser.add_argument("--reuse-dataset", type=Path, default=None)
    parser.add_argument("--output-prefix", type=Path, default=Path("scripts/pyharmonics_survivor_ml_15m"))
    parser.add_argument("--exec-tf", default="15m")
    parser.add_argument("--pattern-tfs", default="15m")
    parser.add_argument("--families", default="ABCD")
    parser.add_argument("--pattern-mode", default="formed")
    parser.add_argument("--forming-percent-c-to-d", type=float, default=0.85)
    parser.add_argument("--peak-spacings", default="10,16")
    parser.add_argument("--fib-tolerances", default="0.03")
    parser.add_argument("--pattern-lookback-bars", type=int, default=800)
    parser.add_argument("--pattern-step-bars", type=int, default=200)
    parser.add_argument("--search-limit-to", type=int, default=12)
    parser.add_argument("--confirm-bars", default="0,4,8")
    parser.add_argument("--entry-window-bars", default="4,8")
    parser.add_argument("--entry-modes", default="next_open,trigger_break,trigger_close_break")
    parser.add_argument("--time-filters", default="all,eu_us")
    parser.add_argument("--candle-filters", default="none")
    parser.add_argument("--trigger-candle-filter", default="none")
    parser.add_argument("--pattern-name-filter", default="ALL")
    parser.add_argument("--direction-filter", default="both")
    parser.add_argument("--prz-atr-buffer", type=float, default=0.10)
    parser.add_argument("--stop-buffers", default="0.35,0.8")
    parser.add_argument("--min-quality-scores", default="0")
    parser.add_argument("--rrs", default="1.5,2.5")
    parser.add_argument("--breakeven-triggers", default="0,1")
    parser.add_argument("--max-hold-bars", type=int, default=192)
    parser.add_argument("--trend-filter", default="none")
    parser.add_argument("--htf-filter", default="none")
    parser.add_argument("--htf-stretch-atr", type=float, default=0.75)
    parser.add_argument("--htf-rsi-extreme", type=float, default=55.0)
    parser.add_argument("--fee-bps-side", type=float, default=5.5)
    parser.add_argument("--slippage-bps-side", type=float, default=1.0)
    parser.add_argument("--max-fee-to-price-risk", type=float, default=0.25)
    parser.add_argument("--min-entry-risk-pct", type=float, default=0.0015)
    parser.add_argument("--risk-fraction", type=float, default=0.01)
    parser.add_argument("--limit-configs", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=48)
    parser.add_argument("--event-progress-every", type=int, default=0)
    parser.add_argument(
        "--event-cache-symbol-dir",
        default="scripts/pyharmonics_top100_fast_15m_abcd_xabcd_5y_20260603_100738/per_symbol",
        help="Directory containing the shared pyharmonics event_cache folder from a prior universe run.",
    )
    parser.add_argument("--event-cache-family", default="ABCD+XABCD")
    parser.add_argument(
        "--write-event-cache-symbol-dir",
        default="",
        help="Optional per_symbol-style directory where newly scanned events should be cached.",
    )
    parser.add_argument("--label-min-r", type=float, default=0.0)
    parser.add_argument("--models", default="logreg,rf,extratrees,hgb,xgb")
    parser.add_argument("--feature-sets", default="no_symbol,with_symbol")
    parser.add_argument("--thresholds", default="0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70")
    parser.add_argument("--keep-fracs", default="0.25,0.35,0.50,0.65,0.80")
    parser.add_argument("--selection-portfolio", default="per_event_one_symbol")
    parser.add_argument("--first-test-year", type=int, default=2024)
    parser.add_argument("--min-train-rows", type=int, default=200)
    parser.add_argument("--min-val-rows", type=int, default=60)
    parser.add_argument("--min-val-trades", type=int, default=12)
    parser.add_argument("--min-holdout-test-rows", type=int, default=10)
    parser.add_argument("--smooth-radius", type=int, default=1)
    parser.add_argument("--min-edge-r", type=float, default=0.02)
    parser.add_argument("--skip-symbol-holdout", action="store_true")
    parser.add_argument("--permutation-importance-rows", type=int, default=0)
    parser.add_argument("--write-final-model", action="store_true")
    parser.add_argument("--final-model", default="rf")
    parser.add_argument("--final-feature-set", default="no_symbol")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn is required. Use the system Python that has sklearn installed.")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    if args.reuse_dataset:
        dataset = pd.read_csv(args.reuse_dataset)
        for column in ["completion_time", "detection_time", "trigger_time", "entry_time", "exit_time", "trigger_break_time"]:
            if column in dataset.columns:
                dataset[column] = pd.to_datetime(dataset[column], utc=True, errors="coerce")
        print(f"Loaded dataset {args.reuse_dataset}: rows={len(dataset)}", flush=True)
    elif args.dataset_source == "cached_trades":
        dataset = load_cached_trade_dataset(args)
    else:
        dataset = build_dataset(args)
    dataset_path = args.output_prefix.with_name(args.output_prefix.name + "_dataset.csv")
    write_table(dataset_path, dataset)

    pooled_summary, pooled_groups, pooled_folds, pooled_thresholds, pooled_scored = run_pooled_walk_forward(dataset, args)
    write_table(args.output_prefix.with_name(args.output_prefix.name + "_pooled_summary.csv"), pooled_summary)
    write_table(args.output_prefix.with_name(args.output_prefix.name + "_pooled_groups.csv"), pooled_groups)
    write_table(args.output_prefix.with_name(args.output_prefix.name + "_pooled_folds.csv"), pooled_folds)
    write_table(args.output_prefix.with_name(args.output_prefix.name + "_pooled_thresholds.csv"), pooled_thresholds)
    write_table(args.output_prefix.with_name(args.output_prefix.name + "_pooled_scored_selected.csv"), pooled_scored)

    if not args.skip_symbol_holdout:
        holdout_summary, holdout_groups, holdout_folds, holdout_thresholds, holdout_scored = run_symbol_holdout_walk_forward(
            dataset,
            args,
        )
        write_table(args.output_prefix.with_name(args.output_prefix.name + "_symbol_holdout_summary.csv"), holdout_summary)
        write_table(args.output_prefix.with_name(args.output_prefix.name + "_symbol_holdout_groups.csv"), holdout_groups)
        write_table(args.output_prefix.with_name(args.output_prefix.name + "_symbol_holdout_folds.csv"), holdout_folds)
        write_table(args.output_prefix.with_name(args.output_prefix.name + "_symbol_holdout_thresholds.csv"), holdout_thresholds)
        write_table(args.output_prefix.with_name(args.output_prefix.name + "_symbol_holdout_scored_selected.csv"), holdout_scored)

    importance = final_feature_importance(dataset, args)
    write_table(args.output_prefix.with_name(args.output_prefix.name + "_feature_importance.csv"), importance)

    config_path = args.output_prefix.with_name(args.output_prefix.name + "_config.json")
    config_payload = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
    config_path.write_text(json.dumps(config_payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved {config_path}", flush=True)

    if args.write_final_model:
        train_final_model(dataset, args)

    print("\nPooled annual summary:")
    print(pooled_summary.to_string(index=False))
    if not args.skip_symbol_holdout:
        print("\nLeave-one-symbol annual summary:")
        print(holdout_summary.to_string(index=False))


if __name__ == "__main__":
    main()
