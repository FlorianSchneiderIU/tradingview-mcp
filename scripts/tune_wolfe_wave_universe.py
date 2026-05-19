from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backtest_wolfe_wave import (  # noqa: E402
    WolfeConfig,
    bybit_symbol,
    ensure_ohlcv_frame,
    fetch_bybit_klines,
    parse_utc_datetime,
    run_backtest,
    split_trades,
    strategy_metrics,
)


DEFAULT_BASE_URL = "https://api.bybit.com"


def load_top_config_symbols(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [bybit_symbol(key) for key in payload if not str(key).startswith("_")]


def unique_symbols(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        symbol = bybit_symbol(value)
        if symbol and symbol not in seen:
            seen.add(symbol)
            out.append(symbol)
    return out


def fetch_mintick(symbol: str, *, base_url: str) -> float:
    response = requests.get(
        f"{base_url.rstrip('/')}/v5/market/instruments-info",
        params={"category": "linear", "symbol": bybit_symbol(symbol)},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("retCode", 0) not in (0, "0"):
        raise RuntimeError(f"Bybit instruments-info failed: {payload.get('retMsg')}")
    rows = payload.get("result", {}).get("list", [])
    if not rows:
        raise RuntimeError(f"No Bybit linear instrument found for {symbol}")
    return float(rows[0].get("priceFilter", {}).get("tickSize", "0.01"))


def window_frame(frame: pd.DataFrame, *, days: int, end: datetime | None) -> pd.DataFrame:
    out = ensure_ohlcv_frame(frame)
    if out.empty:
        return out
    end_ts = pd.Timestamp(end) if end is not None else pd.Timestamp(out["open_time"].iloc[-1])
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")
    start_ts = end_ts - pd.Timedelta(days=days)
    mask = (pd.to_datetime(out["open_time"], utc=True) >= start_ts) & (
        pd.to_datetime(out["open_time"], utc=True) <= end_ts
    )
    return out.loc[mask].reset_index(drop=True)


def requested_window(*, days: int, end: datetime | None) -> tuple[pd.Timestamp, pd.Timestamp]:
    end_ts = pd.Timestamp(end) if end is not None else pd.Timestamp.now(tz="UTC")
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")
    return end_ts - pd.Timedelta(days=days), end_ts


def cache_covers_request(frame: pd.DataFrame, *, days: int, end: datetime | None, tolerance_days: int = 2) -> bool:
    out = ensure_ohlcv_frame(frame)
    if out.empty:
        return False
    start_ts, end_ts = requested_window(days=days, end=end)
    first = pd.Timestamp(out["open_time"].iloc[0]).tz_convert("UTC")
    last = pd.Timestamp(out["open_time"].iloc[-1]).tz_convert("UTC")
    tolerance = pd.Timedelta(days=tolerance_days)
    return first <= start_ts + tolerance and last >= end_ts - tolerance


def load_or_fetch_data(
    symbol: str,
    *,
    interval: str,
    days: int,
    end: datetime | None,
    cache_dir: Path,
    refresh: bool,
    base_url: str,
) -> pd.DataFrame:
    cache_path = cache_dir / f"{symbol.lower()}_{interval}_bybit.csv"
    if cache_path.exists() and not refresh:
        cached = pd.read_csv(cache_path)
        if cache_covers_request(cached, days=days, end=end):
            return window_frame(cached, days=days, end=end)

    end_dt = end or datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days)
    frame = fetch_bybit_klines(symbol, interval, start_dt, end_dt, base_url=base_url)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_path, index=False)
    return window_frame(frame, days=days, end=end)


def has_min_daily_history(symbol: str, *, min_history_days: int, end: datetime | None, base_url: str) -> tuple[bool, str]:
    if min_history_days <= 0:
        return True, ""
    end_dt = end or datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=min_history_days + 21)
    try:
        daily = fetch_bybit_klines(symbol, "1d", start_dt, end_dt, base_url=base_url)
    except Exception as exc:  # noqa: BLE001
        return False, f"daily history fetch failed: {exc}"
    if daily.empty:
        return False, "no daily history"
    first = pd.Timestamp(daily["open_time"].iloc[0]).tz_convert("UTC")
    required_start = pd.Timestamp(end_dt).tz_convert("UTC") - pd.Timedelta(days=min_history_days)
    if first > required_start + pd.Timedelta(days=7):
        age_days = (pd.Timestamp(end_dt).tz_convert("UTC") - first).days
        return False, f"only ~{age_days} daily-history days available"
    return True, ""


def split_bounds(
    frame: pd.DataFrame,
    *,
    validation_days: int,
    oos_days: int,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(frame["open_time"].iloc[0]).tz_convert("UTC")
    data_end = pd.Timestamp(frame["open_time"].iloc[-1]).tz_convert("UTC")
    if validation_days > 0 or oos_days > 0:
        validation_end = data_end - pd.Timedelta(days=max(oos_days, 0))
        train_end = validation_end - pd.Timedelta(days=max(validation_days, 0))
        if train_end <= start:
            span = data_end - start
            train_end = start + span * 0.60
            validation_end = start + span * 0.80
        return train_end, validation_end
    span = data_end - start
    return start + span * 0.60, start + span * 0.80


def candidate_configs(*, exec_tf: str, mintick: float, extra_samples: int = 180) -> list[WolfeConfig]:
    templates: list[dict[str, Any]] = [
        {
            "pattern_tf": "5m",
            "pivot_method": "fractal",
            "pivot_window": 8,
            "max_time_ratio": 3.0,
            "max_p5_break_atr": 1.4,
            "stop_atr_buffer": 0.50,
            "min_rr": 1.2,
            "min_score": 58.0,
            "target_projection_bars": 18,
            "max_hold_bars": 144,
            "trend_filter": "none",
        },
        {
            "pattern_tf": "5m",
            "pivot_method": "fractal",
            "pivot_window": 5,
            "max_time_ratio": 3.8,
            "max_p5_break_atr": 2.2,
            "stop_atr_buffer": 0.50,
            "min_rr": 1.2,
            "min_score": 64.0,
            "target_projection_bars": 30,
            "max_hold_bars": 288,
            "trend_filter": "rsi",
        },
        {
            "pattern_tf": "5m",
            "pivot_method": "zigzag",
            "pivot_window": 5,
            "zigzag_atr_mult": 1.8,
            "max_time_ratio": 3.0,
            "max_p5_break_atr": 2.2,
            "stop_atr_buffer": 0.75,
            "min_rr": 1.2,
            "min_score": 64.0,
            "target_projection_bars": 18,
            "max_hold_bars": 144,
            "trend_filter": "rsi",
        },
        {
            "pattern_tf": "15m",
            "pivot_method": "fractal",
            "pivot_window": 8,
            "max_time_ratio": 2.2,
            "max_p5_break_atr": 2.2,
            "stop_atr_buffer": 0.30,
            "min_rr": 1.2,
            "min_score": 58.0,
            "target_projection_bars": 12,
            "max_hold_bars": 288,
            "trend_filter": "none",
        },
        {
            "pattern_tf": "15m",
            "pivot_method": "fractal",
            "pivot_window": 8,
            "max_time_ratio": 3.0,
            "max_p5_break_atr": 1.4,
            "stop_atr_buffer": 0.30,
            "min_rr": 1.5,
            "min_score": 48.0,
            "target_projection_bars": 8,
            "max_hold_bars": 288,
            "trend_filter": "rsi",
        },
        {
            "pattern_tf": "15m",
            "pivot_method": "fractal",
            "pivot_window": 8,
            "max_time_ratio": 2.2,
            "max_p5_break_atr": 2.2,
            "stop_atr_buffer": 0.30,
            "min_rr": 1.2,
            "min_score": 48.0,
            "target_projection_bars": 18,
            "max_hold_bars": 96,
            "trend_filter": "rsi",
        },
        {
            "pattern_tf": "15m",
            "pivot_method": "fractal",
            "pivot_window": 8,
            "max_time_ratio": 3.8,
            "max_p5_break_atr": 1.4,
            "stop_atr_buffer": 0.30,
            "min_rr": 2.0,
            "min_score": 48.0,
            "target_projection_bars": 30,
            "max_hold_bars": 288,
            "trend_filter": "rsi",
        },
        {
            "pattern_tf": "15m",
            "pivot_method": "zigzag",
            "pivot_window": 5,
            "zigzag_atr_mult": 2.2,
            "max_time_ratio": 3.0,
            "max_p5_break_atr": 3.0,
            "stop_atr_buffer": 0.75,
            "min_rr": 1.2,
            "min_score": 70.0,
            "target_projection_bars": 30,
            "max_hold_bars": 288,
            "trend_filter": "rsi",
        },
        {
            "pattern_tf": "1h",
            "pivot_method": "fractal",
            "pivot_window": 3,
            "max_time_ratio": 2.2,
            "max_p5_break_atr": 2.2,
            "stop_atr_buffer": 0.30,
            "min_rr": 1.5,
            "min_score": 64.0,
            "target_projection_bars": 18,
            "max_hold_bars": 288,
            "trend_filter": "none",
        },
        {
            "pattern_tf": "1h",
            "pivot_method": "fractal",
            "pivot_window": 3,
            "max_time_ratio": 2.2,
            "max_p5_break_atr": 2.2,
            "stop_atr_buffer": 0.30,
            "min_rr": 1.2,
            "min_score": 64.0,
            "target_projection_bars": 12,
            "max_hold_bars": 432,
            "trend_filter": "none",
        },
        {
            "pattern_tf": "1h",
            "pivot_method": "zigzag",
            "pivot_window": 8,
            "zigzag_atr_mult": 1.4,
            "max_time_ratio": 2.2,
            "max_p5_break_atr": 2.2,
            "stop_atr_buffer": 0.30,
            "min_rr": 2.0,
            "min_score": 48.0,
            "target_projection_bars": 12,
            "max_hold_bars": 144,
            "trend_filter": "rsi",
        },
    ]

    variants: list[WolfeConfig] = []
    for template in templates:
        stop_values = sorted({float(template["stop_atr_buffer"]), 0.30, 0.50})
        score_values = sorted({float(template["min_score"]), max(48.0, float(template["min_score"]) - 6.0)})
        for stop_atr_buffer in stop_values:
            for min_score in score_values:
                values = {
                    **template,
                    "exec_tf": exec_tf,
                    "mintick": mintick,
                    "stop_atr_buffer": stop_atr_buffer,
                    "min_score": min_score,
                    "fee_bps_side": 5.5,
                    "slippage_bps_side": 1.0,
                    "risk_fraction": 0.01,
                }
                variants.append(WolfeConfig.from_mapping(values))

    extra_grid: list[tuple[Any, ...]] = []
    for pattern_tf in ("5m", "15m", "1h"):
        for pivot_method in ("fractal", "zigzag"):
            for pivot_window in (3, 5, 8):
                for zigzag_atr_mult in (1.0, 1.4, 1.8, 2.2):
                    for max_time_ratio in (2.2, 3.0, 3.8):
                        for max_p5_break_atr in (1.4, 2.2, 3.0):
                            for stop_atr_buffer in (0.30, 0.50, 0.75):
                                for min_rr in (1.2, 1.5, 2.0):
                                    for min_score in (48.0, 58.0, 64.0, 70.0):
                                        for target_projection_bars in (8, 12, 18, 30):
                                            for max_hold_bars in (96, 144, 288):
                                                for trend_filter in ("none", "rsi"):
                                                    extra_grid.append(
                                                        (
                                                            pattern_tf,
                                                            pivot_method,
                                                            pivot_window,
                                                            zigzag_atr_mult,
                                                            max_time_ratio,
                                                            max_p5_break_atr,
                                                            stop_atr_buffer,
                                                            min_rr,
                                                            min_score,
                                                            target_projection_bars,
                                                            max_hold_bars,
                                                            trend_filter,
                                                        )
                                                    )
    if extra_samples > 0 and len(extra_grid) > extra_samples:
        rng = np.random.default_rng(20260518)
        keep = np.sort(rng.choice(len(extra_grid), size=extra_samples, replace=False))
        extra_grid = [extra_grid[int(idx)] for idx in keep]
    for (
        pattern_tf,
        pivot_method,
        pivot_window,
        zigzag_atr_mult,
        max_time_ratio,
        max_p5_break_atr,
        stop_atr_buffer,
        min_rr,
        min_score,
        target_projection_bars,
        max_hold_bars,
        trend_filter,
    ) in extra_grid:
        variants.append(
            WolfeConfig(
                exec_tf=exec_tf,
                pattern_tf=pattern_tf,
                pivot_method=pivot_method,
                pivot_window=int(pivot_window),
                zigzag_atr_mult=float(zigzag_atr_mult),
                max_time_ratio=float(max_time_ratio),
                max_p5_break_atr=float(max_p5_break_atr),
                stop_atr_buffer=float(stop_atr_buffer),
                min_rr=float(min_rr),
                min_score=float(min_score),
                target_projection_bars=int(target_projection_bars),
                max_hold_bars=int(max_hold_bars),
                trend_filter=str(trend_filter),
                mintick=mintick,
                fee_bps_side=5.5,
                slippage_bps_side=1.0,
                risk_fraction=0.01,
            )
        )

    out: list[WolfeConfig] = []
    seen: set[str] = set()
    for cfg in variants:
        key = json.dumps(asdict(cfg), sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(cfg)
    return out


def metric_prefix(frame: pd.DataFrame, prefix: str) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in strategy_metrics(frame).items()}


def selection_score(train_m: dict[str, float], val_m: dict[str, float], *, min_train: int, min_validation: int) -> float:
    score = (
        min(train_m["avg_r"], val_m["avg_r"]) * 120.0
        + min(train_m["profit_factor"], val_m["profit_factor"], 5.0) * 5.0
        + min(train_m["net_r"], val_m["net_r"]) * 0.5
        - abs(min(train_m["max_dd_r"], val_m["max_dd_r"])) * 0.25
    )
    if train_m["trades"] < min_train:
        score -= (min_train - train_m["trades"]) * 15.0
    if val_m["trades"] < min_validation:
        score -= (min_validation - val_m["trades"]) * 20.0
    return float(score)


def evaluate_config(
    symbol: str,
    frame: pd.DataFrame,
    cfg: WolfeConfig,
    *,
    train_end: pd.Timestamp,
    validation_end: pd.Timestamp,
    min_train: int,
    min_validation: int,
) -> dict[str, Any]:
    trades = run_backtest(frame, cfg, symbol=symbol)
    buckets = split_trades(trades, train_end=train_end, validation_end=validation_end)
    train_m = strategy_metrics(buckets["train"])
    val_m = strategy_metrics(buckets["validation"])
    oos_m = strategy_metrics(buckets["oos"])
    all_m = strategy_metrics(trades)
    return {
        **asdict(cfg),
        "selection_score": selection_score(train_m, val_m, min_train=min_train, min_validation=min_validation),
        **metric_prefix(buckets["train"], "train"),
        **metric_prefix(buckets["validation"], "validation"),
        **metric_prefix(buckets["oos"], "oos"),
        **metric_prefix(trades, "all"),
    }


def oos_pass(row: pd.Series, *, min_oos: int, min_train: int, min_validation: int) -> bool:
    return (
        float(row.get("train_trades", 0.0)) >= float(min_train)
        and float(row.get("validation_trades", 0.0)) >= float(min_validation)
        and float(row.get("oos_trades", 0.0)) >= float(min_oos)
        and float(row.get("train_net_r", 0.0)) > 0.0
        and float(row.get("validation_net_r", 0.0)) > 0.0
        and float(row.get("oos_net_r", 0.0)) > 0.0
        and float(row.get("oos_profit_factor", 0.0)) >= 1.2
        and float(row.get("oos_avg_r", 0.0)) > 0.05
    )


def evaluate_symbol(task: dict[str, Any]) -> dict[str, Any]:
    symbol = task["symbol"]
    end = parse_utc_datetime(task["end"]) if task.get("end") else None
    history_ok, history_reason = has_min_daily_history(
        symbol,
        min_history_days=int(task["min_history_days"]),
        end=end,
        base_url=task["base_url"],
    )
    if not history_ok:
        return {
            "symbol": symbol,
            "skipped": True,
            "skip_reason": history_reason,
            "selected_pass": False,
            "selected_config": {},
            "configs_tested": 0,
        }

    frame = load_or_fetch_data(
        symbol,
        interval=task["interval"],
        days=int(task["days"]),
        end=end,
        cache_dir=Path(task["cache_dir"]),
        refresh=bool(task["refresh"]),
        base_url=task["base_url"],
    )
    if len(frame) < 5000:
        raise RuntimeError(f"Only {len(frame)} candles available")
    start = pd.Timestamp(frame["open_time"].iloc[0]).tz_convert("UTC")
    data_end = pd.Timestamp(frame["open_time"].iloc[-1]).tz_convert("UTC")
    data_days = (data_end - start).days
    if int(task["min_history_days"]) > 0 and data_days < int(task["min_history_days"]) - 7:
        return {
            "symbol": symbol,
            "skipped": True,
            "skip_reason": f"5m cache/fetch covers only ~{data_days} days",
            "selected_pass": False,
            "selected_config": {},
            "configs_tested": 0,
            "bars": int(len(frame)),
            "data_start": start.isoformat(),
            "data_end": data_end.isoformat(),
        }

    mintick = fetch_mintick(symbol, base_url=task["base_url"])
    configs = candidate_configs(
        exec_tf=task["interval"],
        mintick=mintick,
        extra_samples=max(180, int(task["max_configs"]) * 2) if int(task["max_configs"]) > 0 else 180,
    )
    if int(task["max_configs"]) > 0:
        locked_count = 38
        locked = configs[:locked_count]
        rest = configs[locked_count:]
        wanted = int(task["max_configs"])
        if wanted <= len(locked):
            configs = locked[:wanted]
        elif rest:
            rng = np.random.default_rng(20260518)
            take = min(wanted - len(locked), len(rest))
            keep = np.sort(rng.choice(len(rest), size=take, replace=False))
            configs = [*locked, *[rest[int(idx)] for idx in keep]]

    train_end, validation_end = split_bounds(
        frame,
        validation_days=int(task["validation_days"]),
        oos_days=int(task["oos_days"]),
    )
    rows = [
        evaluate_config(
            symbol,
            frame,
            cfg,
            train_end=train_end,
            validation_end=validation_end,
            min_train=int(task["min_train"]),
            min_validation=int(task["min_validation"]),
        )
        for cfg in configs
    ]
    table = pd.DataFrame(rows)
    eligible = table[
        (table["train_trades"] >= int(task["min_train"]))
        & (table["validation_trades"] >= int(task["min_validation"]))
        & (table["train_net_r"] > 0)
        & (table["validation_net_r"] > 0)
    ].copy()
    ranked = eligible if not eligible.empty else table
    ranked = ranked.sort_values(
        ["selection_score", "validation_net_r", "all_net_r"],
        ascending=[False, False, False],
        na_position="last",
    )
    selected = ranked.iloc[0].copy()

    best_oos = table.sort_values(
        ["oos_net_r", "oos_profit_factor", "all_net_r"],
        ascending=[False, False, False],
        na_position="last",
    ).iloc[0]
    passed = oos_pass(
        selected,
        min_oos=int(task["min_oos"]),
        min_train=int(task["min_train"]),
        min_validation=int(task["min_validation"]),
    )
    symbol_dir = Path(task["output_dir"]) / "per_symbol"
    symbol_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(symbol_dir / f"{symbol.lower()}_wolfe_tuning.csv", index=False)

    fields = set(WolfeConfig.__dataclass_fields__)  # type: ignore[attr-defined]
    selected_config = {
        key: selected[key].item() if hasattr(selected[key], "item") else selected[key]
        for key in fields
        if key in selected.index and pd.notna(selected[key])
    }
    return {
        "symbol": symbol,
        "skipped": False,
        "skip_reason": "",
        "bars": int(len(frame)),
        "data_start": start.isoformat(),
        "data_end": data_end.isoformat(),
        "train_end": train_end.isoformat(),
        "validation_end": validation_end.isoformat(),
        "mintick": mintick,
        "configs_tested": int(len(table)),
        "selected_pass": bool(passed),
        "selected_config": selected_config,
        **{f"selected_{key}": value for key, value in selected.to_dict().items() if key not in fields},
        "best_oos_net_r": float(best_oos["oos_net_r"]),
        "best_oos_trades": float(best_oos["oos_trades"]),
        "best_oos_profit_factor": float(best_oos["oos_profit_factor"]),
        "best_oos_pattern_tf": str(best_oos["pattern_tf"]),
        "best_oos_pivot_method": str(best_oos["pivot_method"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune Wolfe Wave configs over the bot symbol universe.")
    parser.add_argument("--top-config", type=Path, default=Path("bot/configs/top20_configs.json"))
    parser.add_argument("--symbols", nargs="*", default=[])
    parser.add_argument("--only-symbols", action="store_true", help="Use --symbols without adding the top-config universe.")
    parser.add_argument("--include-btc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--end")
    parser.add_argument("--min-history-days", type=int, default=0)
    parser.add_argument("--validation-days", type=int, default=0)
    parser.add_argument("--oos-days", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, default=Path("scripts/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("scripts/wolfe_wave_universe"))
    parser.add_argument("--save-config", type=Path, default=Path("bot/configs/wolfe_wave_universe_configs.json"))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-configs", type=int, default=0)
    parser.add_argument("--min-train", type=int, default=8)
    parser.add_argument("--min-validation", type=int, default=3)
    parser.add_argument("--min-oos", type=int, default=4)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols = [] if args.only_symbols else load_top_config_symbols(args.top_config)
    if args.include_btc:
        symbols = ["BTCUSDT", *symbols]
    symbols = unique_symbols([*symbols, *args.symbols])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Universe symbols={len(symbols)} days={args.days} workers={args.workers} "
        f"min_history_days={args.min_history_days} validation_days={args.validation_days} "
        f"oos_days={args.oos_days}",
        flush=True,
    )
    print(",".join(symbols), flush=True)

    task_base = {
        "interval": args.interval,
        "days": args.days,
        "end": args.end,
        "min_history_days": args.min_history_days,
        "validation_days": args.validation_days,
        "oos_days": args.oos_days,
        "cache_dir": str(args.cache_dir),
        "refresh": args.refresh,
        "base_url": args.base_url,
        "output_dir": str(args.output_dir),
        "max_configs": args.max_configs,
        "min_train": args.min_train,
        "min_validation": args.min_validation,
        "min_oos": args.min_oos,
    }
    tasks = [{**task_base, "symbol": symbol} for symbol in symbols]
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(evaluate_symbol, task): task["symbol"] for task in tasks}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                row = future.result()
                rows.append(row)
                if row.get("skipped"):
                    print(f"SKIP {symbol}: {row.get('skip_reason', 'not enough history')}", flush=True)
                else:
                    status = "PASS" if row["selected_pass"] else "MISS"
                    print(
                        f"{status} {symbol}: sel_oos={row.get('selected_oos_net_r', 0.0):+.2f}R/"
                        f"{row.get('selected_oos_trades', 0.0):.0f} trades "
                        f"all={row.get('selected_all_net_r', 0.0):+.2f}R",
                        flush=True,
                    )
            except Exception as exc:  # noqa: BLE001
                errors.append({"symbol": symbol, "error": str(exc)})
                print(f"ERROR {symbol}: {exc}", flush=True)

    summary = pd.DataFrame(rows)
    sort_cols = [col for col in ["selected_pass", "selected_oos_net_r"] if col in summary.columns]
    if sort_cols:
        summary = summary.sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last")
    summary_path = args.output_dir / "wolfe_universe_summary.csv"
    summary.to_csv(summary_path, index=False)
    if errors:
        pd.DataFrame(errors).to_csv(args.output_dir / "wolfe_universe_errors.csv", index=False)

    configs = {
        row["symbol"]: row["selected_config"]
        for row in rows
        if row.get("selected_pass") and not row.get("skipped")
    }
    args.save_config.parent.mkdir(parents=True, exist_ok=True)
    args.save_config.write_text(json.dumps(configs, indent=2, sort_keys=True, default=str), encoding="utf-8")

    pass_count = int(summary["selected_pass"].sum()) if not summary.empty else 0
    skip_count = int(summary["skipped"].sum()) if "skipped" in summary.columns and not summary.empty else 0
    print(f"\nPASS {pass_count}/{len(symbols)} selected configs  SKIP {skip_count}/{len(symbols)} symbols", flush=True)
    print(f"Summary: {summary_path}", flush=True)
    print(f"Saved passing configs: {args.save_config}", flush=True)
    if not summary.empty:
        display_cols = [
            "symbol",
            "selected_pass",
            "selected_train_trades",
            "selected_train_net_r",
            "selected_validation_trades",
            "selected_validation_net_r",
            "selected_oos_trades",
            "selected_oos_net_r",
            "selected_oos_profit_factor",
            "selected_all_net_r",
            "best_oos_net_r",
            "best_oos_trades",
        ]
        print(summary[[col for col in display_cols if col in summary.columns]].to_string(index=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
