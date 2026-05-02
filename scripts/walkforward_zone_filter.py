from __future__ import annotations

import argparse
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backtest_turtle_soup import Config, normalize_binance_spot_symbol, parse_utc_datetime, run_backtest
from scripts.crypto_symbol_sets import SYMBOL_SETS, expand_symbol_args
from scripts.ml_zone_hold_filter import FEATURE_COLUMNS, ensure_cache, fit_sklearn_model, symbol_job


def month_add(value: pd.Timestamp, months: int) -> pd.Timestamp:
    return value + pd.DateOffset(months=months)


def profit_factor(rs: pd.Series) -> float:
    wins = rs[rs > 0].sum()
    losses = -rs[rs <= 0].sum()
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return float(wins / losses)


def max_drawdown(values: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    dd = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        dd = min(dd, equity - peak)
    return round(dd, 3)


def metrics(frame: pd.DataFrame, r_col: str = "r_net") -> dict[str, Any]:
    if frame.empty:
        return {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "net_r": 0.0, "avg_r": 0.0, "max_dd_r": 0.0}
    ordered = frame.sort_values("exit_time")
    rs = ordered[r_col].astype(float)
    return {
        "trades": int(len(frame)),
        "win_rate": round(float((rs > 0).mean()) * 100.0, 2),
        "profit_factor": round(profit_factor(rs), 3),
        "net_r": round(float(rs.sum()), 3),
        "avg_r": round(float(rs.mean()), 3),
        "max_dd_r": max_drawdown(rs.to_list()),
    }


def classifier_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"rows": 0, "hold_rate": 0.0, "auc": math.nan, "brier": math.nan}
    y = frame["hold_label"].astype(int).to_numpy()
    p = frame["hold_prob"].astype(float).to_numpy()
    if len(np.unique(y)) < 2:
        auc = math.nan
    else:
        ranks = pd.Series(p).rank(method="average").to_numpy()
        pos = int(y.sum())
        neg = int(len(y) - pos)
        auc = float((ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))
    return {
        "rows": int(len(frame)),
        "hold_rate": round(float(y.mean()) * 100.0, 2) if len(frame) else 0.0,
        "auc": round(auc, 3) if math.isfinite(auc) else math.nan,
        "brier": round(float(np.mean((p - y) ** 2)), 4) if len(frame) else math.nan,
    }


def trade_key(symbol: str, direction: str, sweep_time: Any, zone_top: float, zone_bottom: float) -> str:
    return f"{normalize_binance_spot_symbol(symbol)}|{direction}|{pd.Timestamp(sweep_time).isoformat()}|{zone_top:.8f}|{zone_bottom:.8f}"


def trade_rows_from_backtest(symbol: str, df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    normalized = normalize_binance_spot_symbol(symbol)
    rows = []
    for trade in run_backtest(df, cfg):
        risk = abs(float(trade.entry_price - trade.stop_price))
        rows.append({
            "symbol": normalized,
            "direction": trade.direction,
            "entry_time": trade.entry_time,
            "exit_time": trade.exit_time,
            "sweep_time": trade.sweep_time,
            "zone_top": trade.zone_top,
            "zone_bottom": trade.zone_bottom,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "stop_price": trade.stop_price,
            "risk": risk,
            "r_multiple": trade.r_multiple,
            "event_key": trade_key(normalized, trade.direction, trade.sweep_time, trade.zone_top, trade.zone_bottom),
        })
    return pd.DataFrame(rows)


def trade_symbol_job(params: dict[str, Any]) -> tuple[str, pd.DataFrame]:
    symbol = params["symbol"]
    cache_path = ensure_cache(symbol, params["interval"], params["warmup_start"], params["end"], params["cache_dir"])
    df = pd.read_pickle(cache_path)
    cfg = Config(
        exec_tf=params["interval"],
        structure_tf="15m",
        entry_mode="zone_retest",
        tf1=params["zone_tf"],
        tf2="1d",
        use_tf1=True,
        use_tf2=False,
        block_dead_zone=False,
        max_structure_bars_to_choch=32,
        htf_left=params["htf_left"],
        htf_right=params["htf_right"],
        htf_ob_search_bars=params["htf_ob_search_bars"],
        max_zone_scan=params["max_zone_scan"],
    )
    rows = trade_rows_from_backtest(symbol, df, cfg)
    return normalize_binance_spot_symbol(symbol), rows


def build_zone_dataset(args: argparse.Namespace, symbols: list[str], start: datetime, end: datetime) -> pd.DataFrame:
    job_params = [
        {
            "symbol": symbol,
            "interval": args.interval,
            "start": start,
            "end": end,
            "cache_dir": args.cache_dir,
            "zone_tf": args.zone_tf,
            "label_rr": args.label_rr,
            "label_horizon_bars": args.label_horizon_bars,
            "htf_left": args.htf_left,
            "htf_right": args.htf_right,
            "htf_ob_search_bars": args.htf_ob_search_bars,
            "zone_penetration_frac": args.zone_penetration_frac,
            "min_reclaim_pos": args.min_reclaim_pos,
            "mbq_ob_lookback_bars": args.mbq_ob_lookback_bars,
            "mbq_confluence_atr": args.mbq_confluence_atr,
            "max_zone_scan": args.max_zone_scan,
            "strategy_min_entry_risk_pct": 0.0,
            "strategy_max_entry_risk_pct": math.inf,
        }
        for symbol in symbols
    ]
    frames = []
    if args.workers <= 1:
        for params in job_params:
            normalized, samples, _trades, message = symbol_job(params)
            frames.append(samples)
            print(message, flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(symbol_job, params): params["symbol"] for params in job_params}
            for future in as_completed(futures):
                normalized, samples, _trades, message = future.result()
                frames.append(samples)
                print(message, flush=True)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_trade_dataset(args: argparse.Namespace, symbols: list[str], warmup_start: datetime, end: datetime) -> pd.DataFrame:
    job_params = [
        {
            "symbol": symbol,
            "interval": args.interval,
            "warmup_start": warmup_start,
            "end": end,
            "cache_dir": args.cache_dir,
            "zone_tf": args.zone_tf,
            "htf_left": args.htf_left,
            "htf_right": args.htf_right,
            "htf_ob_search_bars": args.htf_ob_search_bars,
            "max_zone_scan": args.max_zone_scan,
        }
        for symbol in symbols
    ]
    frames = []
    if args.workers <= 1:
        for params in job_params:
            normalized, rows = trade_symbol_job(params)
            frames.append(rows)
            print(f"{normalized}: {len(rows)} trades", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(trade_symbol_job, params): params["symbol"] for params in job_params}
            for future in as_completed(futures):
                normalized, rows = future.result()
                frames.append(rows)
                print(f"{normalized}: {len(rows)} trades", flush=True)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def add_fee_adjusted_r(frame: pd.DataFrame, fee_bps: float) -> pd.DataFrame:
    out = frame.copy()
    if out.empty:
        out["r_net"] = []
        return out
    fee_cost = (out["entry_price"].abs() + out["exit_price"].abs()) * fee_bps / 10000.0
    r_fee = fee_cost / out["risk"].replace(0, np.nan)
    out["r_net"] = out["r_multiple"].astype(float) - r_fee.fillna(0.0)
    return out


def score_threshold(row: dict[str, Any]) -> float:
    pf = float(row["profit_factor"])
    pf_part = min(pf, 5.0) if math.isfinite(pf) else 5.0
    return float(row["net_r"]) + (pf_part - 1.0) * 2.0 + float(row["max_dd_r"]) * 0.10


def evaluate_thresholds(trades: pd.DataFrame, thresholds: list[float], min_trades: int) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        kept = trades[trades["hold_prob"] >= threshold].copy()
        row = {"threshold": threshold, **metrics(kept)}
        row["eligible"] = row["trades"] >= min_trades
        row["score"] = score_threshold(row) if row["eligible"] else -9999.0
        rows.append(row)
    return pd.DataFrame(rows)


def calibrate_bins(frame: pd.DataFrame, bins: list[float]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows = []
    tmp = frame.copy()
    tmp["prob_bin"] = pd.cut(tmp["hold_prob"], bins=bins, include_lowest=True)
    for bin_value, group in tmp.groupby("prob_bin", observed=False):
        rows.append({
            "prob_bin": str(bin_value),
            "rows": int(len(group)),
            "avg_prob": round(float(group["hold_prob"].mean()), 3) if len(group) else math.nan,
            "hold_rate": round(float(group["hold_label"].mean()) * 100.0, 2) if len(group) else math.nan,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rolling walk-forward study for the SMC zone-hold filter.")
    parser.add_argument("--train-symbol-set", choices=["none", *SYMBOL_SETS.keys()], default="majors20")
    parser.add_argument("--train-symbols", nargs="+", default=[])
    parser.add_argument("--trade-symbol-set", choices=["none", *SYMBOL_SETS.keys()], default="core3")
    parser.add_argument("--trade-symbols", nargs="+", default=[])
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--zone-tf", default="1h")
    parser.add_argument("--warmup-start", default="2024-04-20")
    parser.add_argument("--start", default="2024-04-20")
    parser.add_argument("--end", default="2026-04-20")
    parser.add_argument("--train-months", type=int, default=12)
    parser.add_argument("--val-months", type=int, default=3)
    parser.add_argument("--test-months", type=int, default=3)
    parser.add_argument("--thresholds", default="0.55,0.60,0.65")
    parser.add_argument("--default-threshold", type=float, default=0.60)
    parser.add_argument("--min-val-trades", type=int, default=5)
    parser.add_argument("--fee-bps", type=float, default=7.5, help="Round-trip realism via per-side fee/slippage bps.")
    parser.add_argument("--cache-dir", type=Path, default=Path("scripts/.cache"))
    parser.add_argument("--zone-dataset", type=Path)
    parser.add_argument("--trades-dataset", type=Path)
    parser.add_argument("--output-prefix", type=Path, default=Path("scripts/walkforward_zone_filter"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--model", choices=["sklearn_rf", "sklearn_hgb"], default="sklearn_rf")
    parser.add_argument("--label-rr", type=float, default=1.0)
    parser.add_argument("--label-horizon-bars", type=int, default=288)
    parser.add_argument("--htf-left", type=int, default=5)
    parser.add_argument("--htf-right", type=int, default=5)
    parser.add_argument("--htf-ob-search-bars", type=int, default=50)
    parser.add_argument("--zone-penetration-frac", type=float, default=0.50)
    parser.add_argument("--min-reclaim-pos", type=float, default=0.70)
    parser.add_argument("--mbq-ob-lookback-bars", type=int, default=200)
    parser.add_argument("--mbq-confluence-atr", type=float, default=0.50)
    parser.add_argument("--max-zone-scan", type=int, default=250)
    args = parser.parse_args()

    train_symbols = expand_symbol_args(args.train_symbols, args.train_symbol_set)
    trade_symbols = expand_symbol_args(args.trade_symbols, args.trade_symbol_set)
    warmup_start = parse_utc_datetime(args.warmup_start)
    start = parse_utc_datetime(args.start)
    end = parse_utc_datetime(args.end)
    thresholds = [float(item.strip()) for item in args.thresholds.split(",") if item.strip()]

    if args.zone_dataset and args.zone_dataset.exists():
        zone_dataset = pd.read_csv(args.zone_dataset)
    else:
        zone_dataset = build_zone_dataset(args, train_symbols, start, end)
        if args.zone_dataset:
            args.zone_dataset.parent.mkdir(parents=True, exist_ok=True)
            zone_dataset.to_csv(args.zone_dataset, index=False)

    if args.trades_dataset and args.trades_dataset.exists():
        trades = pd.read_csv(args.trades_dataset)
    else:
        trades = build_trade_dataset(args, trade_symbols, warmup_start, end)
        if args.trades_dataset:
            args.trades_dataset.parent.mkdir(parents=True, exist_ok=True)
            trades.to_csv(args.trades_dataset, index=False)

    for column in ["time", "entry_time", "exit_time", "sweep_time"]:
        if column in zone_dataset.columns:
            zone_dataset[column] = pd.to_datetime(zone_dataset[column], utc=True)
        if column in trades.columns:
            trades[column] = pd.to_datetime(trades[column], utc=True)

    zone_dataset["symbol"] = zone_dataset["symbol"].map(normalize_binance_spot_symbol)
    trades["symbol"] = trades["symbol"].map(normalize_binance_spot_symbol)
    train_symbol_names = {normalize_binance_spot_symbol(symbol) for symbol in train_symbols}
    trade_symbol_names = {normalize_binance_spot_symbol(symbol) for symbol in trade_symbols}
    zone_dataset = zone_dataset[zone_dataset["symbol"].isin(train_symbol_names)].copy()
    trades = trades[trades["symbol"].isin(trade_symbol_names)].copy()
    trades = add_fee_adjusted_r(trades, args.fee_bps)

    fold_rows = []
    threshold_rows = []
    trade_rows = []
    calibration_rows = []
    fold_idx = 1
    test_start = month_add(pd.Timestamp(start), args.train_months + args.val_months)
    while test_start < pd.Timestamp(end):
        train_start = test_start - pd.DateOffset(months=args.train_months + args.val_months)
        train_end = test_start - pd.DateOffset(months=args.val_months)
        val_start = train_end
        val_end = test_start
        test_end = min(month_add(test_start, args.test_months), pd.Timestamp(end))

        train_samples = zone_dataset[(zone_dataset["time"] >= train_start) & (zone_dataset["time"] < train_end)].copy()
        val_samples = zone_dataset[(zone_dataset["time"] >= val_start) & (zone_dataset["time"] < val_end)].copy()
        test_samples = zone_dataset[(zone_dataset["time"] >= test_start) & (zone_dataset["time"] < test_end)].copy()
        if train_samples["hold_label"].nunique() < 2 or val_samples.empty or test_samples.empty:
            print(f"Skipping fold {fold_idx}: insufficient sample labels.", flush=True)
            test_start = test_end
            fold_idx += 1
            continue

        model = fit_sklearn_model(train_samples, FEATURE_COLUMNS, args.model)
        val_samples["hold_prob"] = model.predict_proba(val_samples[FEATURE_COLUMNS].astype(float))[:, 1]
        test_samples["hold_prob"] = model.predict_proba(test_samples[FEATURE_COLUMNS].astype(float))[:, 1]

        prob_lookup = pd.concat([val_samples[["event_key", "hold_prob"]], test_samples[["event_key", "hold_prob"]]], ignore_index=True).drop_duplicates("event_key")
        prob_map = dict(zip(prob_lookup["event_key"], prob_lookup["hold_prob"]))

        val_trades = trades[(trades["sweep_time"] >= val_start) & (trades["sweep_time"] < val_end)].copy()
        test_trades = trades[(trades["sweep_time"] >= test_start) & (trades["sweep_time"] < test_end)].copy()
        val_trades["hold_prob"] = val_trades["event_key"].map(prob_map)
        test_trades["hold_prob"] = test_trades["event_key"].map(prob_map)
        val_trades = val_trades.dropna(subset=["hold_prob"])
        test_trades = test_trades.dropna(subset=["hold_prob"])

        val_table = evaluate_thresholds(val_trades, thresholds, args.min_val_trades)
        eligible = val_table[val_table["eligible"]].copy()
        selected_threshold = args.default_threshold if eligible.empty else float(eligible.sort_values(["score", "net_r"], ascending=[False, False]).iloc[0]["threshold"])
        test_kept = test_trades[test_trades["hold_prob"] >= selected_threshold].copy()
        val_kept = val_trades[val_trades["hold_prob"] >= selected_threshold].copy()

        for _, row in val_table.iterrows():
            threshold_rows.append({
                "fold": fold_idx,
                "threshold": row["threshold"],
                "train_start": train_start,
                "train_end": train_end,
                "val_start": val_start,
                "val_end": val_end,
                **{f"val_{key}": row[key] for key in ["trades", "win_rate", "profit_factor", "net_r", "avg_r", "max_dd_r", "score", "eligible"]},
            })

        fold = {
            "fold": fold_idx,
            "train_start": train_start,
            "train_end": train_end,
            "val_start": val_start,
            "val_end": val_end,
            "test_start": test_start,
            "test_end": test_end,
            "selected_threshold": selected_threshold,
            "train_rows": len(train_samples),
            "val_rows": len(val_samples),
            "test_rows": len(test_samples),
            **{f"val_model_{key}": value for key, value in classifier_metrics(val_samples).items()},
            **{f"test_model_{key}": value for key, value in classifier_metrics(test_samples).items()},
            **{f"val_{key}": value for key, value in metrics(val_kept).items()},
            **{f"test_{key}": value for key, value in metrics(test_kept).items()},
            "test_missing_trade_probs": int(test_trades["hold_prob"].isna().sum()),
        }
        fold_rows.append(fold)
        if not test_kept.empty:
            out = test_kept.copy()
            out["fold"] = fold_idx
            out["selected_threshold"] = selected_threshold
            trade_rows.append(out)

        bins = calibrate_bins(test_samples, [0.0, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 1.0])
        if not bins.empty:
            bins.insert(0, "fold", fold_idx)
            calibration_rows.append(bins)

        print(
            f"fold {fold_idx}: val {val_start.date()}..{val_end.date()} selected {selected_threshold:.2f}; "
            f"test {test_start.date()}..{test_end.date()} {fold['test_trades']} trades {fold['test_net_r']}R pf={fold['test_profit_factor']}",
            flush=True,
        )
        test_start = test_end
        fold_idx += 1

    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    folds = pd.DataFrame(fold_rows)
    threshold_frame = pd.DataFrame(threshold_rows)
    selected_trades = pd.concat(trade_rows, ignore_index=True) if trade_rows else pd.DataFrame()
    calibration = pd.concat(calibration_rows, ignore_index=True) if calibration_rows else pd.DataFrame()
    folds.to_csv(output_prefix.with_name(output_prefix.name + "_folds.csv"), index=False)
    threshold_frame.to_csv(output_prefix.with_name(output_prefix.name + "_thresholds.csv"), index=False)
    selected_trades.to_csv(output_prefix.with_name(output_prefix.name + "_trades.csv"), index=False)
    calibration.to_csv(output_prefix.with_name(output_prefix.name + "_calibration.csv"), index=False)

    print()
    print("Fold Summary:")
    display_cols = [
        "fold",
        "selected_threshold",
        "test_trades",
        "test_win_rate",
        "test_profit_factor",
        "test_net_r",
        "test_avg_r",
        "test_max_dd_r",
        "test_model_auc",
    ]
    print(folds[display_cols].to_string(index=False) if not folds.empty else "No completed folds.")

    print()
    print("Aggregate Selected Test Trades:")
    print(pd.DataFrame([{**metrics(selected_trades), "folds": len(folds)}]).to_string(index=False))
    if not selected_trades.empty:
        rows = []
        for symbol, group in selected_trades.groupby("symbol"):
            rows.append({"symbol": symbol, **metrics(group)})
        print()
        print("By Symbol:")
        print(pd.DataFrame(rows).sort_values("net_r", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
