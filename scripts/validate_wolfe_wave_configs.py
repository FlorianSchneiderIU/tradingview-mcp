from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
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
    parse_utc_datetime,
    run_backtest,
    split_trades,
    strategy_metrics,
)
from scripts.tune_wolfe_wave_universe import (  # noqa: E402
    DEFAULT_BASE_URL,
    load_or_fetch_data,
    metric_prefix,
    oos_pass,
    selection_score,
)


def load_config_map(path: Path, symbols: list[str] | None = None) -> dict[str, WolfeConfig]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = {bybit_symbol(symbol) for symbol in symbols} if symbols else None
    out: dict[str, WolfeConfig] = {}
    for raw_symbol, raw_cfg in payload.items():
        symbol = bybit_symbol(raw_symbol)
        if selected is not None and symbol not in selected:
            continue
        if not isinstance(raw_cfg, dict):
            continue
        out[symbol] = WolfeConfig.from_mapping(raw_cfg)
    return out


def unique_configs(configs: list[tuple[str, WolfeConfig]]) -> list[tuple[str, WolfeConfig]]:
    out: list[tuple[str, WolfeConfig]] = []
    seen: set[str] = set()
    for label, cfg in configs:
        key = json.dumps(asdict(cfg), sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append((label, cfg))
    return out


def config_variant(base: WolfeConfig, label: str, **updates: Any) -> tuple[str, WolfeConfig]:
    return label, WolfeConfig.from_mapping({**asdict(base), **updates})


def neighborhood(base: WolfeConfig) -> list[tuple[str, WolfeConfig]]:
    variants: list[tuple[str, WolfeConfig]] = [("base", base)]
    for value in sorted({base.min_score - 6.0, base.min_score + 6.0}):
        if value >= 36.0:
            variants.append(config_variant(base, f"min_score={value:g}", min_score=float(value)))
    for value in sorted({max(0.1, base.stop_atr_buffer - 0.2), base.stop_atr_buffer + 0.2}):
        variants.append(config_variant(base, f"stop_atr_buffer={value:g}", stop_atr_buffer=float(value)))
    hold_values = [value for value in (96, 144, 288, 432) if value != base.max_hold_bars]
    for value in sorted(hold_values, key=lambda item: abs(item - base.max_hold_bars))[:2]:
        variants.append(config_variant(base, f"max_hold_bars={value}", max_hold_bars=int(value)))
    rr_values = [value for value in (1.2, 1.5, 2.0) if not math.isclose(value, base.min_rr)]
    for value in rr_values:
        variants.append(config_variant(base, f"min_rr={value:g}", min_rr=float(value)))
    p5_values = [value for value in (1.4, 2.2, 3.0) if not math.isclose(value, base.max_p5_break_atr)]
    for value in p5_values:
        variants.append(config_variant(base, f"max_p5_break_atr={value:g}", max_p5_break_atr=float(value)))
    target_values = [value for value in (12, 18, 30) if value != base.target_projection_bars]
    for value in target_values:
        variants.append(config_variant(base, f"target_projection_bars={value}", target_projection_bars=int(value)))
    flipped_trend = "rsi" if base.trend_filter == "none" else "none"
    variants.append(config_variant(base, f"trend_filter={flipped_trend}", trend_filter=flipped_trend))
    return unique_configs(variants)


def load_symbol_frame(task: dict[str, Any]) -> pd.DataFrame:
    end = parse_utc_datetime(task["data_end"]) if task.get("data_end") else None
    return load_or_fetch_data(
        task["symbol"],
        interval=task["interval"],
        days=int(task["days"]),
        end=end,
        cache_dir=Path(task["cache_dir"]),
        refresh=False,
        base_url=task["base_url"],
    )


def frame_until(frame: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    clean = ensure_ohlcv_frame(frame)
    times = pd.to_datetime(clean["open_time"], utc=True, errors="coerce")
    return clean[times <= end].reset_index(drop=True)


def window_bounds(end: pd.Timestamp, validation_days: int, oos_days: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    validation_end = end - pd.Timedelta(days=oos_days)
    train_end = validation_end - pd.Timedelta(days=validation_days)
    return train_end, validation_end


def evaluate_trades(
    trades: pd.DataFrame,
    *,
    train_end: pd.Timestamp,
    validation_end: pd.Timestamp,
    min_train: int,
    min_validation: int,
    min_oos: int,
) -> dict[str, float | bool]:
    buckets = split_trades(trades, train_end=train_end, validation_end=validation_end)
    train_m = strategy_metrics(buckets["train"])
    val_m = strategy_metrics(buckets["validation"])
    row: dict[str, float | bool] = {
        "selection_score": selection_score(train_m, val_m, min_train=min_train, min_validation=min_validation),
        **metric_prefix(buckets["train"], "train"),
        **metric_prefix(buckets["validation"], "validation"),
        **metric_prefix(buckets["oos"], "oos"),
        **metric_prefix(trades, "all"),
    }
    row["pass_gate"] = oos_pass(pd.Series(row), min_oos=min_oos, min_train=min_train, min_validation=min_validation)
    return row


def run_config_window(task: dict[str, Any], cfg: WolfeConfig, end: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = frame_until(load_symbol_frame(task), end)
    return run_config_window_on_frame(task, cfg, end, frame)


def run_config_window_on_frame(
    task: dict[str, Any],
    cfg: WolfeConfig,
    end: pd.Timestamp,
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    train_end, validation_end = window_bounds(
        end,
        validation_days=int(task["validation_days"]),
        oos_days=int(task["oos_days"]),
    )
    trades = run_backtest(frame, cfg, symbol=task["symbol"])
    metrics = evaluate_trades(
        trades,
        train_end=train_end,
        validation_end=validation_end,
        min_train=int(task["min_train"]),
        min_validation=int(task["min_validation"]),
        min_oos=int(task["min_oos"]),
    )
    meta = {
        "data_start": pd.Timestamp(frame["open_time"].iloc[0]).isoformat() if not frame.empty else "",
        "data_end": pd.Timestamp(frame["open_time"].iloc[-1]).isoformat() if not frame.empty else "",
        "train_end": train_end.isoformat(),
        "validation_end": validation_end.isoformat(),
        "oos_end": end.isoformat(),
        "bars": int(len(frame)),
    }
    return trades, {**meta, **metrics}


def fixed_task(task: dict[str, Any]) -> dict[str, Any]:
    cfg = WolfeConfig.from_mapping(task["config"])
    end = parse_utc_datetime(task["rolling_end"])
    _, metrics = run_config_window(task, cfg, pd.Timestamp(end))
    return {
        "kind": "fixed",
        "symbol": task["symbol"],
        "rolling_end": pd.Timestamp(end).date().isoformat(),
        "variant": "base",
        **asdict(cfg),
        **metrics,
    }


def robustness_task(task: dict[str, Any]) -> list[dict[str, Any]]:
    base = WolfeConfig.from_mapping(task["config"])
    end = parse_utc_datetime(task["rolling_end"])
    frame = frame_until(load_symbol_frame(task), pd.Timestamp(end))
    rows: list[dict[str, Any]] = []
    for variant, cfg in neighborhood(base):
        _, metrics = run_config_window_on_frame(task, cfg, pd.Timestamp(end), frame)
        rows.append(
            {
                "kind": "robustness",
                "symbol": task["symbol"],
                "rolling_end": pd.Timestamp(end).date().isoformat(),
                "variant": variant,
                **asdict(cfg),
                **metrics,
            }
        )
    return rows


def reselect_task(task: dict[str, Any]) -> dict[str, Any]:
    base = WolfeConfig.from_mapping(task["config"])
    end = parse_utc_datetime(task["rolling_end"])
    rows = robustness_task(task)
    table = pd.DataFrame(rows)
    eligible = table[
        (table["train_trades"] >= int(task["min_train"]))
        & (table["validation_trades"] >= int(task["min_validation"]))
        & (table["train_net_r"] > 0)
        & (table["validation_net_r"] > 0)
    ].copy()
    ranked = eligible if not eligible.empty else table
    selected = ranked.sort_values(
        ["selection_score", "validation_net_r", "all_net_r"],
        ascending=[False, False, False],
        na_position="last",
    ).iloc[0]
    fields = set(WolfeConfig.__dataclass_fields__)  # type: ignore[attr-defined]
    return {
        "kind": "rolling_reselect",
        "symbol": task["symbol"],
        "rolling_end": pd.Timestamp(end).date().isoformat(),
        "variant_count": int(len(table)),
        "eligible_count": int(len(eligible)),
        "selected_variant": selected["variant"],
        "base_pattern_tf": base.pattern_tf,
        **{f"selected_{key}": selected[key] for key in fields if key in selected.index},
        **{key: selected[key] for key in selected.index if key not in fields and key not in {"kind", "symbol", "rolling_end"}},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Wolfe Wave configs with rolling windows and robustness checks.")
    parser.add_argument("--config", type=Path, default=Path("bot/configs/wolfe_wave_configs.json"))
    parser.add_argument("--symbols", nargs="*", default=[])
    parser.add_argument("--days", type=int, default=1825)
    parser.add_argument("--data-end")
    parser.add_argument("--rolling-ends", nargs="*", default=["2024-05-18", "2025-05-18", "2026-05-18"])
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--validation-days", type=int, default=365)
    parser.add_argument("--oos-days", type=int, default=365)
    parser.add_argument("--min-train", type=int, default=30)
    parser.add_argument("--min-validation", type=int, default=15)
    parser.add_argument("--min-oos", type=int, default=30)
    parser.add_argument("--cache-dir", type=Path, default=Path("scripts/data"))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--output-dir", type=Path, default=Path("scripts/wolfe_wave_validation"))
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = load_config_map(args.config, args.symbols or None)
    if not configs:
        raise RuntimeError(f"No Wolfe configs loaded from {args.config}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    common = {
        "interval": args.interval,
        "days": args.days,
        "data_end": args.data_end,
        "validation_days": args.validation_days,
        "oos_days": args.oos_days,
        "min_train": args.min_train,
        "min_validation": args.min_validation,
        "min_oos": args.min_oos,
        "cache_dir": str(args.cache_dir),
        "base_url": args.base_url,
    }
    base_tasks = [
        {**common, "symbol": symbol, "config": asdict(cfg), "rolling_end": rolling_end}
        for symbol, cfg in configs.items()
        for rolling_end in args.rolling_ends
    ]

    fixed_rows: list[dict[str, Any]] = []
    robust_rows: list[dict[str, Any]] = []
    reselect_rows: list[dict[str, Any]] = []

    print(
        f"Validating {len(configs)} symbols x {len(args.rolling_ends)} rolling windows "
        f"with workers={args.workers}",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        future_map = {pool.submit(fixed_task, task): ("fixed", task) for task in base_tasks}
        latest_tasks = [task for task in base_tasks if task["rolling_end"] == args.rolling_ends[-1]]
        for task in latest_tasks:
            future_map[pool.submit(robustness_task, task)] = ("robustness", task)
        for task in base_tasks:
            future_map[pool.submit(reselect_task, task)] = ("rolling_reselect", task)

        for future in as_completed(future_map):
            kind, task = future_map[future]
            result = future.result()
            if kind == "fixed":
                fixed_rows.append(result)  # type: ignore[arg-type]
                print(
                    f"FIXED {task['symbol']} {task['rolling_end']}: "
                    f"oos={result.get('oos_net_r', 0.0):+.2f}R/{result.get('oos_trades', 0.0):.0f} "
                    f"pass={result.get('pass_gate')}",
                    flush=True,
                )
            elif kind == "robustness":
                robust_rows.extend(result)  # type: ignore[arg-type]
                passed = sum(1 for row in result if row.get("pass_gate"))
                print(f"ROBUST {task['symbol']} {task['rolling_end']}: {passed}/{len(result)} variants pass", flush=True)
            else:
                reselect_rows.append(result)  # type: ignore[arg-type]
                print(
                    f"RESELECT {task['symbol']} {task['rolling_end']}: "
                    f"{result.get('selected_variant')} oos={result.get('oos_net_r', 0.0):+.2f}R/"
                    f"{result.get('oos_trades', 0.0):.0f} pass={result.get('pass_gate')}",
                    flush=True,
                )

    fixed = pd.DataFrame(fixed_rows).sort_values(["symbol", "rolling_end"])
    robust = pd.DataFrame(robust_rows).sort_values(["symbol", "pass_gate", "selection_score"], ascending=[True, False, False])
    reselect = pd.DataFrame(reselect_rows).sort_values(["symbol", "rolling_end"])
    fixed.to_csv(args.output_dir / "fixed_window_metrics.csv", index=False)
    robust.to_csv(args.output_dir / "robustness_metrics.csv", index=False)
    reselect.to_csv(args.output_dir / "rolling_reselect_metrics.csv", index=False)

    robust_summary = (
        robust.groupby("symbol", as_index=False)
        .agg(
            variants=("variant", "count"),
            pass_variants=("pass_gate", "sum"),
            median_oos_net_r=("oos_net_r", "median"),
            best_oos_net_r=("oos_net_r", "max"),
            worst_oos_net_r=("oos_net_r", "min"),
        )
        .sort_values(["pass_variants", "median_oos_net_r"], ascending=[False, False])
    )
    robust_summary.to_csv(args.output_dir / "robustness_summary.csv", index=False)

    print("\nFixed-window summary", flush=True)
    print(fixed[["symbol", "rolling_end", "oos_trades", "oos_net_r", "oos_profit_factor", "pass_gate"]].to_string(index=False), flush=True)
    print("\nRolling reselect summary", flush=True)
    print(reselect[["symbol", "rolling_end", "selected_variant", "oos_trades", "oos_net_r", "oos_profit_factor", "pass_gate"]].to_string(index=False), flush=True)
    print("\nRobustness summary", flush=True)
    print(robust_summary.to_string(index=False), flush=True)
    print(f"\nWrote validation outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
