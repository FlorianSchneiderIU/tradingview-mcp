from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backtest_wolfe_wave import (  # noqa: E402
    WolfeConfig,
    bybit_symbol,
    ensure_ohlcv_frame,
    load_ohlcv_csv,
    parse_utc_datetime,
    run_backtest,
    split_trades,
    strategy_metrics,
)
from scripts.tune_wolfe_wave_universe import metric_prefix, oos_pass  # noqa: E402


DEFAULT_SYMBOLS = ("UNIUSDT", "STXUSDT", "DOGEUSDT", "LINKUSDT", "AAVEUSDT")
DEFAULT_ROLLING_ENDS = ("2024-05-18", "2025-05-18", "2026-05-18")


def slug_for_symbol(symbol: str) -> str:
    clean = bybit_symbol(symbol).lower()
    return clean[:-4] if clean.endswith("usdt") else clean


def load_candidate(symbol: str, candidate_dir: Path) -> WolfeConfig:
    slug = slug_for_symbol(symbol)
    path = candidate_dir / f"wolfe_wave_{slug}_refined_candidate.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    symbol_key = bybit_symbol(symbol)
    if symbol_key not in payload:
        raise KeyError(f"{path} does not contain {symbol_key}")
    return WolfeConfig.from_mapping(payload[symbol_key])


def load_symbol_frame(symbol: str, *, data_dir: Path, days: int, data_end: str) -> pd.DataFrame:
    slug = bybit_symbol(symbol).lower()
    path = data_dir / f"{slug}_5m_bybit.csv"
    frame = ensure_ohlcv_frame(load_ohlcv_csv(path))
    end = pd.Timestamp(parse_utc_datetime(data_end))
    start = end - pd.Timedelta(days=days)
    times = pd.to_datetime(frame["open_time"], utc=True)
    return frame[(times >= start) & (times <= end)].reset_index(drop=True)


def frame_until(frame: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    times = pd.to_datetime(frame["open_time"], utc=True)
    return frame[times <= end].reset_index(drop=True)


def window_bounds(end: pd.Timestamp, validation_days: int, oos_days: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    validation_end = end - pd.Timedelta(days=oos_days)
    train_end = validation_end - pd.Timedelta(days=validation_days)
    return train_end, validation_end


def direction_variants(base: WolfeConfig) -> list[tuple[str, WolfeConfig]]:
    return [
        ("both", WolfeConfig.from_mapping({**asdict(base), "allow_longs": True, "allow_shorts": True})),
        ("long_only", WolfeConfig.from_mapping({**asdict(base), "allow_longs": True, "allow_shorts": False})),
        ("short_only", WolfeConfig.from_mapping({**asdict(base), "allow_longs": False, "allow_shorts": True})),
    ]


def evaluate_window(
    frame: pd.DataFrame,
    cfg: WolfeConfig,
    *,
    symbol: str,
    rolling_end: str,
    validation_days: int,
    oos_days: int,
    min_train: int,
    min_validation: int,
    min_oos: int,
) -> dict[str, Any]:
    end = pd.Timestamp(parse_utc_datetime(rolling_end))
    window = frame_until(frame, end)
    train_end, validation_end = window_bounds(end, validation_days, oos_days)
    trades = run_backtest(window, cfg, symbol=symbol)
    buckets = split_trades(trades, train_end=train_end, validation_end=validation_end)
    row: dict[str, Any] = {
        "rolling_end": end.date().isoformat(),
        "bars": int(len(window)),
        "train_end": train_end.isoformat(),
        "validation_end": validation_end.isoformat(),
        **metric_prefix(buckets["train"], "train"),
        **metric_prefix(buckets["validation"], "validation"),
        **metric_prefix(buckets["oos"], "oos"),
        **metric_prefix(trades, "all"),
    }
    row["pass_gate"] = oos_pass(
        pd.Series(row),
        min_train=min_train,
        min_validation=min_validation,
        min_oos=min_oos,
    )
    return row


def evaluate_symbol(task: dict[str, Any]) -> list[dict[str, Any]]:
    symbol = bybit_symbol(task["symbol"])
    base = load_candidate(symbol, Path(task["candidate_dir"]))
    frame = load_symbol_frame(
        symbol,
        data_dir=Path(task["data_dir"]),
        days=int(task["days"]),
        data_end=str(task["data_end"]),
    )
    rows: list[dict[str, Any]] = []
    for variant, cfg in direction_variants(base):
        for rolling_end in task["rolling_ends"]:
            metrics = evaluate_window(
                frame,
                cfg,
                symbol=symbol,
                rolling_end=str(rolling_end),
                validation_days=int(task["validation_days"]),
                oos_days=int(task["oos_days"]),
                min_train=int(task["min_train"]),
                min_validation=int(task["min_validation"]),
                min_oos=int(task["min_oos"]),
            )
            rows.append(
                {
                    "symbol": symbol,
                    "variant": variant,
                    "allow_longs": cfg.allow_longs,
                    "allow_shorts": cfg.allow_shorts,
                    **asdict(cfg),
                    **metrics,
                }
            )
    return rows


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = rows.groupby(["symbol", "variant"], as_index=False)
    summary = grouped.agg(
        windows=("rolling_end", "count"),
        pass_windows=("pass_gate", "sum"),
        total_oos_net_r=("oos_net_r", "sum"),
        min_oos_net_r=("oos_net_r", "min"),
        median_oos_net_r=("oos_net_r", "median"),
        total_oos_trades=("oos_trades", "sum"),
        min_oos_trades=("oos_trades", "min"),
        median_oos_trades=("oos_trades", "median"),
        min_oos_profit_factor=("oos_profit_factor", "min"),
        total_all_net_r=("all_net_r", "sum"),
        median_all_trades=("all_trades", "median"),
    )
    summary["all_oos_positive"] = summary["min_oos_net_r"] > 0.0
    summary["strict_all_pass"] = summary["pass_windows"] == summary["windows"]
    summary["score"] = (
        summary["total_oos_net_r"]
        + summary["pass_windows"] * 10.0
        + summary["min_oos_net_r"].clip(upper=0.0) * 2.0
        + summary["min_oos_trades"].clip(upper=30.0) * 0.2
    )
    return summary.sort_values(
        ["strict_all_pass", "all_oos_positive", "score", "total_oos_net_r"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Wolfe candidate long-only and short-only variants.")
    parser.add_argument("--symbols", nargs="*", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--candidate-dir", type=Path, default=Path("scripts"))
    parser.add_argument("--data-dir", type=Path, default=Path("scripts/data"))
    parser.add_argument("--days", type=int, default=1825)
    parser.add_argument("--data-end", default="2026-05-18")
    parser.add_argument("--rolling-ends", nargs="*", default=list(DEFAULT_ROLLING_ENDS))
    parser.add_argument("--validation-days", type=int, default=365)
    parser.add_argument("--oos-days", type=int, default=365)
    parser.add_argument("--min-train", type=int, default=30)
    parser.add_argument("--min-validation", type=int, default=15)
    parser.add_argument("--min-oos", type=int, default=30)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("scripts/wolfe_wave_direction_variants"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "candidate_dir": str(args.candidate_dir),
        "data_dir": str(args.data_dir),
        "days": args.days,
        "data_end": args.data_end,
        "rolling_ends": args.rolling_ends,
        "validation_days": args.validation_days,
        "oos_days": args.oos_days,
        "min_train": args.min_train,
        "min_validation": args.min_validation,
        "min_oos": args.min_oos,
    }
    tasks = [{**common, "symbol": symbol} for symbol in args.symbols]
    rows: list[dict[str, Any]] = []
    print(f"Evaluating {len(tasks)} symbols x 3 direction variants x {len(args.rolling_ends)} windows", flush=True)
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        future_map = {pool.submit(evaluate_symbol, task): task for task in tasks}
        for future in as_completed(future_map):
            task = future_map[future]
            result = future.result()
            rows.extend(result)
            table = pd.DataFrame(result)
            fixed = table.pivot(index="variant", columns="rolling_end", values="oos_net_r")
            trades = table.pivot(index="variant", columns="rolling_end", values="oos_trades")
            print(f"\n{bybit_symbol(task['symbol'])} OOS R", flush=True)
            print(fixed.round(2).to_string(), flush=True)
            print(f"{bybit_symbol(task['symbol'])} OOS trades", flush=True)
            print(trades.round(0).to_string(), flush=True)

    metrics = pd.DataFrame(rows).sort_values(["symbol", "variant", "rolling_end"])
    summary = summarize(metrics)
    metrics.to_csv(args.output_dir / "direction_fixed_window_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "direction_summary.csv", index=False)
    print("\nDirection summary", flush=True)
    print(
        summary[
            [
                "symbol",
                "variant",
                "strict_all_pass",
                "all_oos_positive",
                "pass_windows",
                "total_oos_net_r",
                "min_oos_net_r",
                "total_oos_trades",
                "min_oos_trades",
                "min_oos_profit_factor",
                "score",
            ]
        ].round(3).to_string(index=False),
        flush=True,
    )
    print(f"\nWrote direction outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
