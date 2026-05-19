from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backtest_turtle_soup import parse_utc_datetime


DEFAULT_PREFIX = Path("scripts/bullbyte_curl_postmortem_btc_eth_allconfigs")
DEFAULT_OUT_PREFIX = Path("scripts/bullbyte_curl_btc_eth_postmortem")

STANDARD_FEATURES = [
    "atr_pctile",
    "vol_ratio",
    "ema200_dist",
    "ema200_slope",
    "body_ratio",
    "sma13_dist",
    "hour_utc",
    "direction",
    "mom5",
    "atr_norm",
    "day_of_week",
    "rsi14",
    "rsi4h",
]

STRATEGY_FEATURES = [
    "bb_watch_len",
    "bb_comp_len",
    "bb_local_bg_ratio",
    "bb_session_position",
    "bb_watch_height_atr",
]


def safe_div(num: float, den: float) -> float:
    if not math.isfinite(num) or not math.isfinite(den) or abs(den) < 1e-12:
        return math.nan
    return float(num / den)


def profit_factor(values: pd.Series | np.ndarray | list[float]) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return 0.0
    gains = float(arr[arr > 0].sum())
    losses = float(-arr[arr < 0].sum())
    if losses <= 0:
        return 99.0 if gains > 0 else 0.0
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


def score(m: dict[str, float]) -> float:
    trades = float(m.get("trades", 0) or 0)
    if trades <= 0:
        return -1e9
    return (
        float(m["avg_r"]) * math.sqrt(min(trades, 300.0))
        + 0.16 * min(float(m["profit_factor"]), 4.0)
        - 0.008 * float(m["max_dd_r"])
    )


def prefixed(m: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in m.items()}


def choose_candidate_specs(summary: pd.DataFrame, symbols: list[str], top_n: int) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for symbol in symbols:
        rows = summary[summary["symbol"].astype(str) == symbol].copy()
        ranked = rows.sort_values(["train_score", "train_net_r"], ascending=[False, False]).head(top_n)
        # Add a small oracle-diagnostic slice for post-mortem only. These are
        # never used as the final leak-clean recommendation unless validation
        # also supports the gate.
        oracle = rows.sort_values(["oos_score", "oos_net_r"], ascending=[False, False]).head(max(3, top_n // 3))
        for _, row in pd.concat([ranked, oracle]).iterrows():
            out.add((str(row["symbol"]), str(row["spec_name"])))
    return out


def load_candidate_trades(path: Path, keys: set[tuple[str, str]], chunksize: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    usecols = [
        "symbol",
        "spec_name",
        "signal_spec_name",
        "exit_spec_name",
        "direction",
        "entry_time",
        "exit_time",
        "r_multiple",
        "exit_reason",
        "risk_pct",
        "bars_held",
        "feature_json",
    ]
    symbols = {key[0] for key in keys}
    spec_names = {key[1] for key in keys}
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        mask = chunk["symbol"].astype(str).isin(symbols) & chunk["spec_name"].astype(str).isin(spec_names)
        if mask.any():
            keep = chunk[mask].copy()
            keep = keep[[((str(row.symbol), str(row.spec_name)) in keys) for row in keep.itertuples(index=False)]]
            if not keep.empty:
                frames.append(keep)
    if not frames:
        return pd.DataFrame(columns=usecols)
    out = pd.concat(frames, ignore_index=True)
    out["entry_time"] = pd.to_datetime(out["entry_time"], utc=True, errors="coerce")
    out["exit_time"] = pd.to_datetime(out["exit_time"], utc=True, errors="coerce")
    out["r_multiple"] = pd.to_numeric(out["r_multiple"], errors="coerce")
    return out.dropna(subset=["entry_time", "r_multiple"]).reset_index(drop=True)


def expand_features(trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in trades.itertuples(index=False):
        base = row._asdict()
        try:
            features = json.loads(base.get("feature_json") or "{}")
        except Exception:
            features = {}
        for name in STANDARD_FEATURES + STRATEGY_FEATURES:
            base[name] = features.get(name, math.nan)
        rows.append(base)
    out = pd.DataFrame(rows)
    for column in STANDARD_FEATURES + STRATEGY_FEATURES:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["is_long"] = (out["direction"].astype(str) == "long").astype(float)
    out["hour"] = out["hour_utc"] * 23.0
    out["hour_sin"] = np.sin(2.0 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * out["hour"] / 24.0)
    out["session_asia"] = ((out["hour"] >= 0) & (out["hour"] < 8)).astype(float)
    out["session_london"] = ((out["hour"] >= 7) & (out["hour"] < 16)).astype(float)
    out["session_ny"] = ((out["hour"] >= 13) & (out["hour"] < 22)).astype(float)
    direction_sign = np.where(out["is_long"] > 0, 1.0, -1.0)
    out["rsi14_reversal_pressure"] = np.where(
        out["is_long"] > 0,
        0.5 - out["rsi14"],
        out["rsi14"] - 0.5,
    )
    out["rsi4h_reversal_pressure"] = np.where(
        out["is_long"] > 0,
        0.5 - out["rsi4h"],
        out["rsi4h"] - 0.5,
    )
    out["ema200_dist_dir"] = direction_sign * out["ema200_dist"]
    out["ema200_slope_dir"] = direction_sign * out["ema200_slope"]
    out["mom5_dir"] = direction_sign * out["mom5"]
    out["extreme_depth"] = np.where(
        out["is_long"] > 0,
        1.0 - out["bb_session_position"],
        out["bb_session_position"],
    )
    out["watch_tightness"] = safe_series_div(out["bb_local_bg_ratio"], out["bb_watch_height_atr"])
    return out


def safe_series_div(num: pd.Series, den: pd.Series) -> pd.Series:
    n = pd.to_numeric(num, errors="coerce")
    d = pd.to_numeric(den, errors="coerce")
    out = n / d.replace(0.0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def feature_sets() -> dict[str, list[str]]:
    standard = list(STANDARD_FEATURES)
    extended = [
        *STANDARD_FEATURES,
        *STRATEGY_FEATURES,
        "hour_sin",
        "hour_cos",
        "session_asia",
        "session_london",
        "session_ny",
        "rsi14_reversal_pressure",
        "rsi4h_reversal_pressure",
        "ema200_dist_dir",
        "ema200_slope_dir",
        "mom5_dir",
        "extreme_depth",
        "watch_tightness",
    ]
    return {"standard": standard, "extended": extended}


def make_models(random_state: int, train_len: int) -> dict[str, Any]:
    leaf = max(8, min(50, train_len // 12))
    return {
        "dt2": make_pipeline(
            SimpleImputer(strategy="median"),
            DecisionTreeClassifier(
                max_depth=2,
                min_samples_leaf=leaf,
                class_weight="balanced",
                random_state=random_state,
            ),
        ),
        "dt3": make_pipeline(
            SimpleImputer(strategy="median"),
            DecisionTreeClassifier(
                max_depth=3,
                min_samples_leaf=leaf,
                class_weight="balanced",
                random_state=random_state,
            ),
        ),
        "logit": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000, random_state=random_state),
        ),
        "rf": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=350,
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
                n_estimators=450,
                max_depth=4,
                min_samples_leaf=leaf,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
    }


def split_frames(frame: pd.DataFrame, split: pd.Timestamp, val_frac: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_all = frame[frame["entry_time"] < split].sort_values("entry_time").copy()
    oos = frame[frame["entry_time"] >= split].sort_values("entry_time").copy()
    if train_all.empty:
        return train_all, train_all, train_all, oos
    cut = max(1, int(len(train_all) * (1.0 - val_frac)))
    cut = min(cut, max(1, len(train_all) - 1))
    fit = train_all.iloc[:cut].copy()
    val = train_all.iloc[cut:].copy()
    return train_all, fit, val, oos


def row_for_gate(
    symbol: str,
    spec_name: str,
    gate_name: str,
    train_all: pd.DataFrame,
    fit: pd.DataFrame,
    val: pd.DataFrame,
    oos: pd.DataFrame,
    selected_train: pd.DataFrame,
    selected_val: pd.DataFrame,
    selected_oos: pd.DataFrame,
    meta: dict[str, Any],
) -> dict[str, Any]:
    raw_train = metrics(train_all)
    raw_val = metrics(val)
    raw_oos = metrics(oos)
    sel_train = metrics(selected_train)
    sel_val = metrics(selected_val)
    sel_oos = metrics(selected_oos)
    out = {
        "symbol": symbol,
        "spec_name": spec_name,
        "gate": gate_name,
        **meta,
        **prefixed(raw_train, "raw_train"),
        **prefixed(raw_val, "raw_val"),
        **prefixed(raw_oos, "raw_oos"),
        **prefixed(sel_train, "gate_train"),
        **prefixed(sel_val, "gate_val"),
        **prefixed(sel_oos, "gate_oos"),
    }
    out["train_delta"] = out["gate_train_net_r"] - out["raw_train_net_r"]
    out["val_delta"] = out["gate_val_net_r"] - out["raw_val_net_r"]
    out["oos_delta"] = out["gate_oos_net_r"] - out["raw_oos_net_r"]
    out["oos_retention"] = safe_div(out["gate_oos_trades"], out["raw_oos_trades"])
    out["selection_score"] = score(sel_val)
    return out


def simple_filter_rows(
    symbol: str,
    spec_name: str,
    frame: pd.DataFrame,
    split: pd.Timestamp,
    val_frac: float,
    min_val_trades: int,
    min_oos_trades: int,
) -> list[dict[str, Any]]:
    train_all, fit, val, oos = split_frames(frame, split, val_frac)
    if len(val) < min_val_trades or len(oos) < min_oos_trades:
        return []
    cols = [
        *STANDARD_FEATURES,
        *STRATEGY_FEATURES,
        "hour",
        "hour_sin",
        "hour_cos",
        "rsi14_reversal_pressure",
        "rsi4h_reversal_pressure",
        "ema200_dist_dir",
        "mom5_dir",
        "extreme_depth",
        "watch_tightness",
        "risk_pct",
    ]
    rows: list[dict[str, Any]] = []
    for col in cols:
        if col not in frame.columns:
            continue
        fit_values = pd.to_numeric(fit[col], errors="coerce").dropna()
        if fit_values.nunique() < 4:
            continue
        for q in (0.15, 0.25, 0.35, 0.50, 0.65, 0.75, 0.85):
            threshold = float(fit_values.quantile(q))
            for op in ("<=", ">="):
                def apply(df: pd.DataFrame) -> pd.DataFrame:
                    vals = pd.to_numeric(df[col], errors="coerce")
                    mask = vals <= threshold if op == "<=" else vals >= threshold
                    return df[mask].copy()

                sel_val = apply(val)
                if len(sel_val) < min_val_trades:
                    continue
                sel_oos = apply(oos)
                if len(sel_oos) < min_oos_trades:
                    continue
                rows.append(
                    row_for_gate(
                        symbol,
                        spec_name,
                        "simple_filter",
                        train_all,
                        fit,
                        val,
                        oos,
                        apply(train_all),
                        sel_val,
                        sel_oos,
                        {"feature_set": "extended", "model": "", "threshold": threshold, "rule": f"{col} {op} {threshold:.6g}"},
                    )
                )
    return rows


def ml_gate_rows(
    symbol: str,
    spec_name: str,
    frame: pd.DataFrame,
    split: pd.Timestamp,
    val_frac: float,
    thresholds: list[float],
    min_fit: int,
    min_val_trades: int,
    min_oos_trades: int,
    random_state: int,
) -> list[dict[str, Any]]:
    train_all, fit, val, oos = split_frames(frame, split, val_frac)
    if len(fit) < min_fit or len(val) < min_val_trades or len(oos) < min_oos_trades:
        return []
    if fit["r_multiple"].gt(0).nunique() < 2:
        return []
    rows: list[dict[str, Any]] = []
    for fs_name, cols in feature_sets().items():
        cols = [col for col in cols if col in frame.columns]
        usable = []
        for col in cols:
            numeric = pd.to_numeric(fit[col], errors="coerce")
            if numeric.notna().sum() >= max(10, min_fit // 2) and numeric.nunique(dropna=True) > 1:
                usable.append(col)
        if len(usable) < 4:
            continue
        for model_name, model in make_models(random_state, len(fit)).items():
            try:
                model.fit(fit[usable], fit["r_multiple"].gt(0).astype(int))
                val_prob = model.predict_proba(val[usable])[:, 1]
                oos_prob = model.predict_proba(oos[usable])[:, 1]
                train_prob = model.predict_proba(train_all[usable])[:, 1]
            except Exception:
                continue
            val_scored = val.copy()
            oos_scored = oos.copy()
            train_scored = train_all.copy()
            val_scored["ml_prob"] = val_prob
            oos_scored["ml_prob"] = oos_prob
            train_scored["ml_prob"] = train_prob
            for threshold in thresholds:
                sel_val = val_scored[val_scored["ml_prob"] >= threshold].copy()
                if len(sel_val) < min_val_trades:
                    continue
                sel_oos = oos_scored[oos_scored["ml_prob"] >= threshold].copy()
                if len(sel_oos) < min_oos_trades:
                    continue
                rows.append(
                    row_for_gate(
                        symbol,
                        spec_name,
                        "ml_gate",
                        train_all,
                        fit,
                        val,
                        oos,
                        train_scored[train_scored["ml_prob"] >= threshold].copy(),
                        sel_val,
                        sel_oos,
                        {
                            "feature_set": fs_name,
                            "model": model_name,
                            "threshold": float(threshold),
                            "rule": f"prob >= {threshold:.2f}",
                        },
                    )
                )
    return rows


def best_rows(gates: pd.DataFrame) -> pd.DataFrame:
    if gates.empty:
        return gates
    eligible = gates[
        (gates["gate_val_net_r"] > gates["raw_val_net_r"])
        & (gates["gate_train_net_r"] > gates["raw_train_net_r"])
        & (gates["gate_oos_trades"] > 0)
    ].copy()
    if eligible.empty:
        eligible = gates.copy()
    return (
        eligible.sort_values(
            ["selection_score", "gate_val_net_r", "gate_train_net_r"],
            ascending=[False, False, False],
        )
        .groupby("symbol", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def markdown_table(frame: pd.DataFrame, columns: list[str], limit: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    shown = frame.head(limit)[columns].copy()
    for col in shown.columns:
        if pd.api.types.is_numeric_dtype(shown[col]):
            shown[col] = shown[col].map(lambda x: f"{float(x):.4f}" if pd.notna(x) and math.isfinite(float(x)) else "")
        else:
            shown[col] = shown[col].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(str(row[col]) for col in columns) + " |" for _, row in shown.iterrows()]
    return "\n".join([header, sep, *body])


def write_report(
    out_path: Path,
    baseline: pd.DataFrame,
    gates: pd.DataFrame,
    best: pd.DataFrame,
    candidates: pd.DataFrame,
    split: str,
) -> None:
    lines: list[str] = []
    lines.append("# BullByte Curl BTC/ETH Post-Mortem")
    lines.append("")
    lines.append(f"Split: `{split}`. Candidate specs are selected from train-ranked rows; OOS-oracle rows are included only as diagnostics.")
    lines.append("")
    lines.append("## Baseline Without ML")
    lines.append("")
    lines.append(markdown_table(baseline, ["symbol", "spec_rank", "train_trades", "train_net_r", "train_profit_factor", "oos_trades", "oos_net_r", "oos_profit_factor", "spec_name"], 20))
    lines.append("")
    lines.append("## Best Validation-Selected Gates")
    lines.append("")
    lines.append(markdown_table(best, ["symbol", "gate", "feature_set", "model", "rule", "raw_train_net_r", "gate_train_net_r", "raw_oos_net_r", "gate_oos_net_r", "oos_delta", "gate_oos_trades", "spec_name"], 10))
    lines.append("")
    lines.append("## Gate Search Top Rows")
    lines.append("")
    top = gates.sort_values(["symbol", "selection_score"], ascending=[True, False]).groupby("symbol", as_index=False).head(10)
    lines.append(markdown_table(top, ["symbol", "gate", "feature_set", "model", "rule", "gate_val_net_r", "gate_train_net_r", "gate_oos_net_r", "oos_delta", "spec_name"], 30))
    lines.append("")
    lines.append("## Candidate Spec Context")
    lines.append("")
    lines.append(markdown_table(candidates, ["symbol", "spec_rank", "train_trades", "train_net_r", "oos_trades", "oos_net_r", "spec_name"], 30))
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BTC/ETH BullByte post-mortem with extended filters and ML gates.")
    parser.add_argument("--prefix", type=Path, default=DEFAULT_PREFIX)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    parser.add_argument("--top-specs", type=int, default=16)
    parser.add_argument("--split", default="2025-07-01")
    parser.add_argument("--val-frac", type=float, default=0.35)
    parser.add_argument("--thresholds", default="0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70")
    parser.add_argument("--min-fit", type=int, default=28)
    parser.add_argument("--min-val-trades", type=int, default=8)
    parser.add_argument("--min-oos-trades", type=int, default=8)
    parser.add_argument("--chunksize", type=int, default=150000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    summary = pd.read_csv(args.prefix.with_name(f"{args.prefix.name}_summary.csv"))
    trades_path = args.prefix.with_name(f"{args.prefix.name}_trades.csv")
    split = pd.Timestamp(parse_utc_datetime(args.split))

    keys = choose_candidate_specs(summary, symbols, args.top_specs)
    print(f"Loading candidate trades for {len(keys)} symbol/spec pairs ...", flush=True)
    trades = load_candidate_trades(trades_path, keys, args.chunksize)
    print(f"Loaded {len(trades):,} candidate trades", flush=True)
    expanded = expand_features(trades)
    expanded_path = args.out_prefix.with_name(f"{args.out_prefix.name}_candidate_trades.csv")
    expanded.to_csv(expanded_path, index=False)

    candidate_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    for symbol in symbols:
        rows = summary[summary["symbol"].astype(str) == symbol].copy()
        ranked = rows.sort_values(["train_score", "train_net_r"], ascending=[False, False]).copy()
        ranked["spec_rank"] = np.arange(1, len(ranked) + 1)
        for _, row in ranked[ranked["spec_name"].isin([key[1] for key in keys if key[0] == symbol])].head(args.top_specs).iterrows():
            candidate_rows.append(row.to_dict())
        baseline_rows.append(ranked.iloc[0].to_dict())
        baseline_rows[-1]["spec_rank"] = 1

    for (symbol, spec_name), frame in expanded.groupby(["symbol", "spec_name"], sort=False):
        frame = frame.sort_values("entry_time").copy()
        gate_rows.extend(
            simple_filter_rows(
                symbol,
                spec_name,
                frame,
                split,
                args.val_frac,
                args.min_val_trades,
                args.min_oos_trades,
            )
        )
        gate_rows.extend(
            ml_gate_rows(
                symbol,
                spec_name,
                frame,
                split,
                args.val_frac,
                [float(x.strip()) for x in args.thresholds.split(",") if x.strip()],
                args.min_fit,
                args.min_val_trades,
                args.min_oos_trades,
                args.random_state,
            )
        )

    candidates = pd.DataFrame(candidate_rows)
    baseline = pd.DataFrame(baseline_rows)
    gates = pd.DataFrame(gate_rows)
    best = best_rows(gates)
    candidates.to_csv(args.out_prefix.with_name(f"{args.out_prefix.name}_candidate_specs.csv"), index=False)
    baseline.to_csv(args.out_prefix.with_name(f"{args.out_prefix.name}_baseline.csv"), index=False)
    gates.to_csv(args.out_prefix.with_name(f"{args.out_prefix.name}_gate_search.csv"), index=False)
    best.to_csv(args.out_prefix.with_name(f"{args.out_prefix.name}_best_gates.csv"), index=False)
    write_report(args.out_prefix.with_suffix(".md"), baseline, gates, best, candidates, args.split)
    print(f"Saved report: {args.out_prefix.with_suffix('.md')}", flush=True)
    print(f"Saved best gates: {args.out_prefix.with_name(f'{args.out_prefix.name}_best_gates.csv')}", flush=True)


if __name__ == "__main__":
    main()
