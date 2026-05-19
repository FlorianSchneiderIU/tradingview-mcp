from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backtest_wolfe_wave import (  # noqa: E402
    WolfeConfig,
    bybit_symbol,
    ensure_ohlcv_frame,
    run_backtest,
    split_trades,
    strategy_metrics,
)
from scripts.tune_wolfe_wave_universe import metric_prefix, oos_pass, selection_score, split_bounds  # noqa: E402


FRAME: pd.DataFrame | None = None
TRAIN_END: pd.Timestamp | None = None
VALIDATION_END: pd.Timestamp | None = None
SYMBOL = "BTCUSDT"


def init_worker(data_path: str, symbol: str, validation_days: int, oos_days: int) -> None:
    global FRAME, TRAIN_END, VALIDATION_END, SYMBOL
    FRAME = ensure_ohlcv_frame(pd.read_csv(data_path))
    SYMBOL = bybit_symbol(symbol)
    TRAIN_END, VALIDATION_END = split_bounds(FRAME, validation_days=validation_days, oos_days=oos_days)


def cfg_key(cfg: WolfeConfig) -> str:
    return json.dumps(asdict(cfg), sort_keys=True)


def dedupe(configs: list[WolfeConfig]) -> list[WolfeConfig]:
    out: list[WolfeConfig] = []
    seen: set[str] = set()
    for cfg in configs:
        key = cfg_key(cfg)
        if key in seen:
            continue
        seen.add(key)
        out.append(cfg)
    return out


def cfg_from_row(row: pd.Series) -> WolfeConfig:
    fields = set(WolfeConfig.__dataclass_fields__)  # type: ignore[attr-defined]
    return WolfeConfig.from_mapping({key: row[key] for key in fields if key in row.index and pd.notna(row[key])})


def locked_configs(tuning_path: Path) -> list[WolfeConfig]:
    if not tuning_path.exists():
        return []
    table = pd.read_csv(tuning_path)
    seeds = pd.concat(
        [
            table.sort_values(["selection_score", "validation_net_r", "all_net_r"], ascending=[False, False, False]).head(12),
            table.sort_values(["oos_net_r", "oos_profit_factor", "all_net_r"], ascending=[False, False, False]).head(12),
        ],
        ignore_index=True,
    )
    out: list[WolfeConfig] = []
    for _, row in seeds.iterrows():
        base = cfg_from_row(row)
        sources = ["wick", "close", "body"] if base.pivot_method == "fractal" else ["wick", "close", "body"]
        confirm_values = [0]
        if base.pivot_method == "fractal":
            confirm_values = sorted({0, 3, 5, int(base.pivot_window), 12})
        for source in sources:
            for confirm in confirm_values:
                out.append(WolfeConfig.from_mapping({**asdict(base), "pivot_source": source, "pivot_confirm_window": confirm}))
    return dedupe(out)


def generate_configs(max_configs: int, tuning_path: Path) -> list[WolfeConfig]:
    base = WolfeConfig(exec_tf="5m", mintick=0.1, fee_bps_side=5.5, slippage_bps_side=1.0, risk_fraction=0.01)
    configs = locked_configs(tuning_path)

    grid: list[dict[str, Any]] = []
    for pattern_tf in ("15m", "1h"):
        for pivot_source in ("wick", "close", "body"):
            for pivot_window in (3, 5, 8, 12):
                for pivot_confirm_window in (2, 3, 5, 8, 12, 16):
                    for max_time_ratio in (2.2, 3.0, 3.8):
                        for max_p5_break_atr in (1.0, 1.4, 2.2, 3.0):
                            for stop_atr_buffer in (0.30, 0.50, 0.75):
                                for min_rr in (1.0, 1.2, 1.5, 2.0):
                                    for min_score in (42.0, 48.0, 52.0, 58.0, 64.0):
                                        for target_projection_bars in (8, 12, 18, 30):
                                            for max_hold_bars in (144, 288, 576):
                                                for trend_filter in ("none", "rsi"):
                                                    grid.append(
                                                        {
                                                            "pattern_tf": pattern_tf,
                                                            "pivot_method": "fractal",
                                                            "pivot_source": pivot_source,
                                                            "pivot_window": pivot_window,
                                                            "pivot_confirm_window": pivot_confirm_window,
                                                            "max_time_ratio": max_time_ratio,
                                                            "max_p5_break_atr": max_p5_break_atr,
                                                            "stop_atr_buffer": stop_atr_buffer,
                                                            "min_rr": min_rr,
                                                            "min_score": min_score,
                                                            "target_projection_bars": target_projection_bars,
                                                            "max_hold_bars": max_hold_bars,
                                                            "trend_filter": trend_filter,
                                                        }
                                                    )
        for pivot_source in ("wick", "close", "body"):
            for pivot_window in (3, 5, 8):
                for zigzag_atr_mult in (0.8, 1.0, 1.4, 1.8, 2.2, 3.0):
                    for max_time_ratio in (2.2, 3.0, 3.8):
                        for max_p5_break_atr in (1.4, 2.2, 3.0):
                            for stop_atr_buffer in (0.30, 0.50, 0.75):
                                for min_rr in (1.0, 1.2, 1.5, 2.0):
                                    for min_score in (42.0, 48.0, 58.0, 64.0):
                                        for target_projection_bars in (8, 12, 18, 30):
                                            for max_hold_bars in (144, 288, 576):
                                                for trend_filter in ("none", "rsi"):
                                                    grid.append(
                                                        {
                                                            "pattern_tf": pattern_tf,
                                                            "pivot_method": "zigzag",
                                                            "pivot_source": pivot_source,
                                                            "pivot_window": pivot_window,
                                                            "pivot_confirm_window": 0,
                                                            "zigzag_atr_mult": zigzag_atr_mult,
                                                            "max_time_ratio": max_time_ratio,
                                                            "max_p5_break_atr": max_p5_break_atr,
                                                            "stop_atr_buffer": stop_atr_buffer,
                                                            "min_rr": min_rr,
                                                            "min_score": min_score,
                                                            "target_projection_bars": target_projection_bars,
                                                            "max_hold_bars": max_hold_bars,
                                                            "trend_filter": trend_filter,
                                                        }
                                                    )

    rng = np.random.default_rng(20260518)
    remaining = max(0, max_configs - len(configs)) if max_configs > 0 else len(grid)
    if remaining > 0 and remaining < len(grid):
        keep = np.sort(rng.choice(len(grid), size=remaining, replace=False))
        grid = [grid[int(idx)] for idx in keep]
    for values in grid[:remaining]:
        configs.append(WolfeConfig.from_mapping({**asdict(base), **values}))
    if max_configs > 0:
        configs = configs[:max_configs]
    return dedupe(configs)


def evaluate_config(payload: dict[str, Any]) -> dict[str, Any]:
    if FRAME is None or TRAIN_END is None or VALIDATION_END is None:
        raise RuntimeError("Worker was not initialized.")
    cfg = WolfeConfig.from_mapping(payload)
    trades = run_backtest(FRAME, cfg, symbol=SYMBOL)
    buckets = split_trades(trades, train_end=TRAIN_END, validation_end=VALIDATION_END)
    train_m = strategy_metrics(buckets["train"])
    val_m = strategy_metrics(buckets["validation"])
    row = {
        **asdict(cfg),
        "selection_score": selection_score(train_m, val_m, min_train=30, min_validation=15),
        **metric_prefix(buckets["train"], "train"),
        **metric_prefix(buckets["validation"], "validation"),
        **metric_prefix(buckets["oos"], "oos"),
        **metric_prefix(trades, "all"),
    }
    row["pass_gate"] = oos_pass(pd.Series(row), min_oos=30, min_train=30, min_validation=15)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wolfe swing-definition sweep.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--tuning", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-configs", type=int, default=240)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--validation-days", type=int, default=365)
    parser.add_argument("--oos-days", type=int, default=365)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbol = bybit_symbol(args.symbol)
    slug = symbol.lower()
    if args.data is None:
        args.data = Path(f"scripts/data/{slug}_5m_bybit.csv")
    if args.tuning is None:
        args.tuning = Path(f"scripts/wolfe_wave_universe_4y_oos1y_stage40_fast/per_symbol/{slug}_wolfe_tuning.csv")
    if args.output_dir is None:
        args.output_dir = Path(f"scripts/wolfe_wave_{slug}_swing_sweep")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configs = generate_configs(args.max_configs, args.tuning)
    (args.output_dir / "candidate_configs.json").write_text(
        json.dumps([asdict(cfg) for cfg in configs], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"{symbol} swing sweep configs={len(configs)} workers={args.workers}", flush=True)
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=max(1, args.workers),
        initializer=init_worker,
        initargs=(str(args.data), symbol, args.validation_days, args.oos_days),
    ) as pool:
        futures = {pool.submit(evaluate_config, asdict(cfg)): idx for idx, cfg in enumerate(configs, start=1)}
        for future in as_completed(futures):
            idx = futures[future]
            row = future.result()
            rows.append(row)
            if row.get("pass_gate") or len(rows) % 20 == 0:
                print(
                    f"{len(rows)}/{len(configs)} idx={idx} pass={row.get('pass_gate')} "
                    f"{row['pattern_tf']} {row['pivot_method']} {row['pivot_source']} "
                    f"w={row['pivot_window']} c={row['pivot_confirm_window']} "
                    f"train={row['train_net_r']:+.1f} val={row['validation_net_r']:+.1f} "
                    f"oos={row['oos_net_r']:+.1f}/{row['oos_trades']:.0f} "
                    f"elapsed={time.perf_counter() - t0:.1f}s",
                    flush=True,
                )

    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / f"{slug}_swing_definition_sweep.csv", index=False)
    sort_cols = ["pass_gate", "selection_score", "validation_net_r", "oos_net_r"]
    ranked = table.sort_values(sort_cols, ascending=[False, False, False, False], na_position="last")
    ranked.to_csv(args.output_dir / f"{slug}_swing_definition_ranked.csv", index=False)
    display_cols = [
        "pass_gate",
        "pattern_tf",
        "pivot_method",
        "pivot_source",
        "pivot_window",
        "pivot_confirm_window",
        "zigzag_atr_mult",
        "max_time_ratio",
        "max_p5_break_atr",
        "stop_atr_buffer",
        "min_rr",
        "min_score",
        "target_projection_bars",
        "max_hold_bars",
        "trend_filter",
        "train_trades",
        "train_net_r",
        "validation_trades",
        "validation_net_r",
        "oos_trades",
        "oos_net_r",
        "oos_profit_factor",
        "all_net_r",
        "selection_score",
    ]
    print("\nTop ranked")
    print(ranked[[col for col in display_cols if col in ranked.columns]].head(25).to_string(index=False), flush=True)
    print("\nTop OOS")
    print(
        table.sort_values(["oos_net_r", "oos_profit_factor", "all_net_r"], ascending=[False, False, False])[
            [col for col in display_cols if col in table.columns]
        ]
        .head(25)
        .to_string(index=False),
        flush=True,
    )
    grouped = (
        table.groupby(["pattern_tf", "pivot_method", "pivot_source"], as_index=False)
        .agg(
            configs=("exec_tf", "size"),
            pass_configs=("pass_gate", "sum"),
            median_train_net_r=("train_net_r", "median"),
            median_validation_net_r=("validation_net_r", "median"),
            median_oos_net_r=("oos_net_r", "median"),
            median_oos_trades=("oos_trades", "median"),
            best_oos_net_r=("oos_net_r", "max"),
        )
        .sort_values(["pass_configs", "median_validation_net_r", "median_oos_net_r"], ascending=[False, False, False])
    )
    grouped.to_csv(args.output_dir / f"{slug}_swing_definition_grouped.csv", index=False)
    print("\nGrouped")
    print(grouped.to_string(index=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
