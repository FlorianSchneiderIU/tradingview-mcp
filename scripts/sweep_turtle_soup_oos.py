from __future__ import annotations

import argparse
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backtest_turtle_soup import Config, fetch_klines, normalize_binance_spot_symbol, parse_utc_datetime, run_backtest, summarize


DATA: pd.DataFrame | None = None


def _init_worker(cache_path: str) -> None:
    global DATA
    DATA = pd.read_pickle(cache_path)


def _to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _cache_path(cache_dir: Path, symbol: str, interval: str, start: datetime, end: datetime) -> Path:
    start_s = start.strftime("%Y%m%d")
    end_s = end.strftime("%Y%m%d")
    return cache_dir / f"{normalize_binance_spot_symbol(symbol).lower()}_{interval}_{start_s}_{end_s}.pkl"


def ensure_cache(symbol: str, interval: str, start: datetime, end: datetime, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Prefer the existing long ETH cache when it fully covers the requested span.
    requested_symbol = normalize_binance_spot_symbol(symbol).lower()
    for candidate in sorted(cache_dir.glob(f"{requested_symbol}_{interval}_*.pkl")):
        try:
            df = pd.read_pickle(candidate)
        except Exception:
            continue
        if df.empty:
            continue
        if df["open_time"].iloc[0].to_pydatetime() <= start and df["close_time"].iloc[-1].to_pydatetime() >= end:
            return candidate

    path = _cache_path(cache_dir, symbol, interval, start, end)
    if path.exists():
        return path

    df = fetch_klines(symbol, interval, _to_ms(start), _to_ms(end))
    df.to_pickle(path)
    return path


def metrics_for_window(trades: list, start: datetime, end: datetime) -> dict[str, Any]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    window_trades = [trade for trade in trades if start_ts <= trade.entry_time < end_ts]
    out = summarize(window_trades)
    out["max_dd_r"] = max_drawdown_r(window_trades)
    return out


def max_drawdown_r(trades: list) -> float:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for trade in sorted(trades, key=lambda item: item.exit_time):
        equity += trade.r_multiple
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)
    return round(max_dd, 3)


def parse_float_list(value: str) -> list[float]:
    values = []
    for item in value.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return values or [0.0]


def config_grid(
    preset: str = "full",
    zone_hold_model: str | None = None,
    zone_hold_min_probs: list[float] | None = None,
    zone_hold_filter_tf: str = "4h",
    min_entry_risk_pcts: list[float] | None = None,
    max_zone_scan: int = 0,
) -> list[tuple[str, dict[str, Any]]]:
    zone_hold_min_probs = zone_hold_min_probs or [0.0]
    min_entry_risk_pcts = min_entry_risk_pcts or [0.0]
    if preset == "probe":
        base_rows = [
            (
                "1d_only|zone_retest|no_dead|choch32",
                dict(exec_tf="5m", structure_tf="15m", entry_mode="zone_retest", tf1="1d", tf2="1w", use_tf1=True, use_tf2=False, block_dead_zone=False, max_structure_bars_to_choch=32),
            ),
            (
                "1d_only|zone_retest|dead|choch48",
                dict(exec_tf="5m", structure_tf="15m", entry_mode="zone_retest", tf1="1d", tf2="1w", use_tf1=True, use_tf2=False, block_dead_zone=True, max_structure_bars_to_choch=48),
            ),
            (
                "1d_only|zone_retest|no_dead|choch48",
                dict(exec_tf="5m", structure_tf="15m", entry_mode="zone_retest", tf1="1d", tf2="1w", use_tf1=True, use_tf2=False, block_dead_zone=False, max_structure_bars_to_choch=48),
            ),
            (
                "4h_1d|retest_close|no_dead|choch16",
                dict(exec_tf="5m", structure_tf="15m", entry_mode="retest_close", tf1="4h", tf2="1d", use_tf1=True, use_tf2=True, block_dead_zone=False, max_structure_bars_to_choch=16),
            ),
            (
                "4h_1d|retest_close|no_dead|choch48",
                dict(exec_tf="5m", structure_tf="15m", entry_mode="retest_close", tf1="4h", tf2="1d", use_tf1=True, use_tf2=True, block_dead_zone=False, max_structure_bars_to_choch=48),
            ),
            (
                "4h_1d|limit_mid|no_dead|choch48",
                dict(exec_tf="5m", structure_tf="15m", entry_mode="limit_mid", tf1="4h", tf2="1d", use_tf1=True, use_tf2=True, block_dead_zone=False, max_structure_bars_to_choch=48),
            ),
        ]
        return expand_ml_grid(base_rows, zone_hold_model, zone_hold_min_probs, zone_hold_filter_tf, min_entry_risk_pcts, max_zone_scan)
    if preset != "full":
        raise ValueError(f"Unknown preset {preset!r}. Use 'full' or 'probe'.")

    rows: list[tuple[str, dict[str, Any]]] = []
    tf_profiles = [
        ("1h_only", dict(tf1="1h", tf2="4h", use_tf1=True, use_tf2=False)),
        ("1h_4h", dict(tf1="1h", tf2="4h", use_tf1=True, use_tf2=True)),
        ("4h_1d", dict(tf1="4h", tf2="1d", use_tf1=True, use_tf2=True)),
        ("1d_only", dict(tf1="1d", tf2="1w", use_tf1=True, use_tf2=False)),
    ]
    entry_modes = ["zone_retest", "retest_close", "limit_mid"]
    dead_zone_options = [True, False]
    choch_windows = [16, 32, 48]

    for tf_name, tf_kwargs in tf_profiles:
        for entry_mode in entry_modes:
            for block_dead_zone in dead_zone_options:
                for max_choch in choch_windows:
                    name = (
                        f"{tf_name}|{entry_mode}|"
                        f"{'dead' if block_dead_zone else 'no_dead'}|choch{max_choch}"
                    )
                    rows.append((
                        name,
                        dict(
                            exec_tf="5m",
                            structure_tf="15m",
                            entry_mode=entry_mode,
                            block_dead_zone=block_dead_zone,
                            max_structure_bars_to_choch=max_choch,
                            **tf_kwargs,
                        ),
                    ))
    return expand_ml_grid(rows, zone_hold_model, zone_hold_min_probs, zone_hold_filter_tf, min_entry_risk_pcts, max_zone_scan)


def expand_ml_grid(
    rows: list[tuple[str, dict[str, Any]]],
    zone_hold_model: str | None,
    zone_hold_min_probs: list[float],
    zone_hold_filter_tf: str,
    min_entry_risk_pcts: list[float],
    max_zone_scan: int,
) -> list[tuple[str, dict[str, Any]]]:
    expanded: list[tuple[str, dict[str, Any]]] = []
    for name, kwargs in rows:
        for min_risk in min_entry_risk_pcts:
            for min_prob in zone_hold_min_probs:
                suffix_parts = []
                next_kwargs = dict(kwargs)
                if max_zone_scan > 0:
                    next_kwargs["max_zone_scan"] = max_zone_scan
                    suffix_parts.append(f"scan{max_zone_scan}")
                if min_risk > 0:
                    next_kwargs["min_entry_risk_pct"] = min_risk
                    suffix_parts.append(f"risk{min_risk:g}")
                if zone_hold_model and min_prob > 0:
                    next_kwargs["zone_hold_model_path"] = zone_hold_model
                    next_kwargs["zone_hold_min_prob"] = min_prob
                    next_kwargs["zone_hold_filter_tf"] = zone_hold_filter_tf
                    suffix_parts.append(f"ml{min_prob:g}")
                expanded_name = name if not suffix_parts else f"{name}|{'|'.join(suffix_parts)}"
                expanded.append((expanded_name, next_kwargs))
    return expanded


def run_one(spec: tuple[str, dict[str, Any]], train_start: datetime, train_end: datetime, oos_end: datetime) -> dict[str, Any]:
    if DATA is None:
        raise RuntimeError("Worker data was not initialized.")
    name, kwargs = spec
    cfg = Config(**kwargs)
    trades = run_backtest(DATA, cfg)
    train = metrics_for_window(trades, train_start, train_end)
    oos = metrics_for_window(trades, train_end, oos_end)
    full = metrics_for_window(trades, train_start, oos_end)

    row: dict[str, Any] = {
        "config": name,
        "tf1": cfg.tf1,
        "tf2": cfg.tf2,
        "use_tf2": cfg.use_tf2,
        "entry_mode": cfg.entry_mode,
        "dead_zone": cfg.block_dead_zone,
        "max_choch": cfg.max_structure_bars_to_choch,
        "zone_hold_min_prob": cfg.zone_hold_min_prob,
        "min_entry_risk_pct": cfg.min_entry_risk_pct,
        "max_zone_scan": cfg.max_zone_scan,
        "train_trades": train["trades"],
        "train_wr": train["win_rate"],
        "train_pf": train["profit_factor"],
        "train_net_r": train["net_r"],
        "train_dd_r": train["max_dd_r"],
        "oos_trades": oos["trades"],
        "oos_wr": oos["win_rate"],
        "oos_pf": oos["profit_factor"],
        "oos_net_r": oos["net_r"],
        "oos_dd_r": oos["max_dd_r"],
        "full_trades": full["trades"],
        "full_pf": full["profit_factor"],
        "full_net_r": full["net_r"],
    }
    row["oos_pass"] = row["oos_trades"] >= 5 and row["oos_net_r"] > 0 and row["oos_pf"] >= 1.05
    row["robust_score"] = robust_score(row)
    return row


def robust_score(row: dict[str, Any]) -> float:
    if row["train_trades"] < 15 or row["oos_trades"] < 5:
        return -9999.0
    train_pf = min(float(row["train_pf"]), 5.0) if math.isfinite(float(row["train_pf"])) else 5.0
    oos_pf = min(float(row["oos_pf"]), 5.0) if math.isfinite(float(row["oos_pf"])) else 5.0
    return round(
        row["train_net_r"] * 0.35
        + row["oos_net_r"] * 0.65
        + (train_pf - 1.0) * 2.0
        + (oos_pf - 1.0) * 4.0
        + row["oos_dd_r"] * 0.15,
        3,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BINANCE:ETHUSDT")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--cache-dir", default="scripts/.cache")
    parser.add_argument("--warmup-start", default="2021-09-01")
    parser.add_argument("--train-start", default="2022-04-20")
    parser.add_argument("--split", default="2025-04-20")
    parser.add_argument("--end", default="2026-04-20")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", default="scripts/turtle_soup_oos_results.csv")
    parser.add_argument("--preset", choices=["full", "probe"], default="full")
    parser.add_argument("--zone-hold-model")
    parser.add_argument("--zone-hold-min-probs", default="0.0")
    parser.add_argument("--zone-hold-filter-tf", default="4h")
    parser.add_argument("--min-entry-risk-pcts", default="0.0")
    parser.add_argument("--max-zone-scan", type=int, default=0)
    args = parser.parse_args()

    warmup_start = parse_utc_datetime(args.warmup_start)
    train_start = parse_utc_datetime(args.train_start)
    split = parse_utc_datetime(args.split)
    end = parse_utc_datetime(args.end)

    cache_path = ensure_cache(args.symbol, args.interval, warmup_start, end, Path(args.cache_dir))
    specs = config_grid(
        args.preset,
        args.zone_hold_model,
        parse_float_list(args.zone_hold_min_probs),
        args.zone_hold_filter_tf,
        parse_float_list(args.min_entry_risk_pcts),
        args.max_zone_scan,
    )
    print(f"symbol={args.symbol} cache={cache_path} configs={len(specs)} workers={args.workers}", flush=True)
    print(f"train={train_start.date()}..{split.date()} oos={split.date()}..{end.date()}", flush=True)

    rows: list[dict[str, Any]] = []
    if args.workers <= 1:
        _init_worker(str(cache_path))
        for idx, spec in enumerate(specs, start=1):
            name = spec[0]
            row = run_one(spec, train_start, split, end)
            rows.append(row)
            print(f"[{idx:02d}/{len(specs)}] {name} train={row['train_net_r']}R oos={row['oos_net_r']}R", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(str(cache_path),)) as pool:
            futures = {pool.submit(run_one, spec, train_start, split, end): spec[0] for spec in specs}
            for idx, future in enumerate(as_completed(futures), start=1):
                name = futures[future]
                row = future.result()
                rows.append(row)
                print(f"[{idx:02d}/{len(specs)}] {name} train={row['train_net_r']}R oos={row['oos_net_r']}R", flush=True)

    out = pd.DataFrame(rows)
    out = out.sort_values(["robust_score", "oos_net_r", "train_net_r"], ascending=[False, False, False])
    output = Path(args.output)
    out.to_csv(output, index=False)

    display_cols = [
        "config",
        "train_trades",
        "train_pf",
        "train_net_r",
        "train_dd_r",
        "oos_trades",
        "oos_pf",
        "oos_net_r",
        "oos_dd_r",
        "full_net_r",
        "robust_score",
    ]
    print("\n=== Top Robust Configs ===")
    print(out[display_cols].head(12).to_string(index=False))
    print(f"\nSaved {len(out)} rows to {output}")


if __name__ == "__main__":
    main()
