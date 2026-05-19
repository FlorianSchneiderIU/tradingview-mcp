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
    parse_utc_datetime,
    run_backtest,
    split_trades,
    strategy_metrics,
)
from scripts.tune_wolfe_wave_universe import metric_prefix, oos_pass, selection_score  # noqa: E402


FRAME: pd.DataFrame | None = None
SYMBOL = ""


def init_worker(data_path: str, symbol: str) -> None:
    global FRAME, SYMBOL
    FRAME = ensure_ohlcv_frame(pd.read_csv(data_path))
    SYMBOL = bybit_symbol(symbol)


def frame_until(frame: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    times = pd.to_datetime(frame["open_time"], utc=True, errors="coerce")
    return frame[times <= end].reset_index(drop=True)


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
) -> dict[str, Any]:
    buckets = split_trades(trades, train_end=train_end, validation_end=validation_end)
    train_m = strategy_metrics(buckets["train"])
    val_m = strategy_metrics(buckets["validation"])
    row: dict[str, Any] = {
        "selection_score": selection_score(train_m, val_m, min_train=min_train, min_validation=min_validation),
        **metric_prefix(buckets["train"], "train"),
        **metric_prefix(buckets["validation"], "validation"),
        **metric_prefix(buckets["oos"], "oos"),
        **metric_prefix(trades, "all"),
    }
    row["pass_gate"] = oos_pass(pd.Series(row), min_oos=min_oos, min_train=min_train, min_validation=min_validation)
    return row


def evaluate_candidate(task: dict[str, Any]) -> dict[str, Any]:
    if FRAME is None:
        raise RuntimeError("Worker was not initialized.")
    cfg = WolfeConfig.from_mapping(task["config"])
    rolling_end = parse_utc_datetime(task["rolling_end"])
    sliced = frame_until(FRAME, pd.Timestamp(rolling_end))
    train_end, validation_end = window_bounds(pd.Timestamp(rolling_end), int(task["validation_days"]), int(task["oos_days"]))
    trades = run_backtest(sliced, cfg, symbol=SYMBOL)
    return {
        "symbol": SYMBOL,
        "candidate_rank": task["candidate_rank"],
        "rolling_end": pd.Timestamp(rolling_end).date().isoformat(),
        **asdict(cfg),
        **evaluate_trades(
            trades,
            train_end=train_end,
            validation_end=validation_end,
            min_train=int(task["min_train"]),
            min_validation=int(task["min_validation"]),
            min_oos=int(task["min_oos"]),
        ),
    }


def cfg_from_row(row: pd.Series) -> WolfeConfig:
    fields = set(WolfeConfig.__dataclass_fields__)  # type: ignore[attr-defined]
    return WolfeConfig.from_mapping({key: row[key] for key in fields if key in row.index and pd.notna(row[key])})


def plain_values(values: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in values.items():
        if hasattr(value, "item"):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select Wolfe swing candidates across rolling windows.")
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--sweep-root-template", default="scripts/wolfe_wave_{slug}_swing_sweep")
    parser.add_argument("--data-template", default="scripts/data/{slug}_5m_bybit.csv")
    parser.add_argument("--rolling-ends", nargs="+", default=["2025-05-18", "2026-05-18"])
    parser.add_argument("--validation-days", type=int, default=365)
    parser.add_argument("--oos-days", type=int, default=365)
    parser.add_argument("--min-train", type=int, default=30)
    parser.add_argument("--min-validation", type=int, default=15)
    parser.add_argument("--min-oos", type=int, default=30)
    parser.add_argument("--top-pass-candidates", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("scripts/wolfe_wave_swing_sweep_validation"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    selected: dict[str, dict[str, Any]] = {}
    selected_rows: list[dict[str, Any]] = []

    for raw_symbol in args.symbols:
        symbol = bybit_symbol(raw_symbol)
        slug = symbol.lower()
        sweep_dir = Path(args.sweep_root_template.format(symbol=symbol, slug=slug))
        sweep_path = sweep_dir / f"{slug}_swing_definition_sweep.csv"
        data_path = Path(args.data_template.format(symbol=symbol, slug=slug))
        table = pd.read_csv(sweep_path)
        candidates = (
            table[table["pass_gate"]]
            .sort_values(["selection_score", "validation_net_r", "oos_net_r"], ascending=[False, False, False])
            .head(args.top_pass_candidates)
            .reset_index(drop=True)
        )
        tasks: list[dict[str, Any]] = []
        for rank, (_, row) in enumerate(candidates.iterrows(), start=1):
            cfg = cfg_from_row(row)
            for rolling_end in args.rolling_ends:
                tasks.append(
                    {
                        "candidate_rank": rank,
                        "config": asdict(cfg),
                        "rolling_end": rolling_end,
                        "validation_days": args.validation_days,
                        "oos_days": args.oos_days,
                        "min_train": args.min_train,
                        "min_validation": args.min_validation,
                        "min_oos": args.min_oos,
                    }
                )

        rows: list[dict[str, Any]] = []
        with ProcessPoolExecutor(
            max_workers=max(1, args.workers),
            initializer=init_worker,
            initargs=(str(data_path), symbol),
        ) as pool:
            futures = {pool.submit(evaluate_candidate, task): task for task in tasks}
            for future in as_completed(futures):
                rows.append(future.result())
        all_rows.extend(rows)
        symbol_table = pd.DataFrame(rows)
        grouped = (
            symbol_table.groupby("candidate_rank", as_index=False)
            .agg(
                windows=("rolling_end", "size"),
                pass_windows=("pass_gate", "sum"),
                min_oos_net_r=("oos_net_r", "min"),
                median_oos_net_r=("oos_net_r", "median"),
                total_oos_net_r=("oos_net_r", "sum"),
                min_oos_trades=("oos_trades", "min"),
                median_selection_score=("selection_score", "median"),
            )
            .sort_values(
                ["pass_windows", "min_oos_net_r", "median_selection_score", "total_oos_net_r"],
                ascending=[False, False, False, False],
            )
        )
        winner_rank = int(grouped.iloc[0]["candidate_rank"])
        winner = candidates.iloc[winner_rank - 1]
        cfg = cfg_from_row(winner)
        cfg_values = plain_values(asdict(cfg))
        selected[symbol] = cfg_values
        selected_rows.append(
            {
                "symbol": symbol,
                "candidate_rank": winner_rank,
                **cfg_values,
                **{f"latest_{key}": winner[key] for key in winner.index if key.endswith("_net_r") or key.endswith("_trades") or key in {"selection_score", "oos_profit_factor"}},
                **{f"rolling_{key}": grouped.iloc[0][key] for key in grouped.columns if key != "candidate_rank"},
            }
        )
        print(f"{symbol}: selected candidate {winner_rank}; pass_windows={int(grouped.iloc[0]['pass_windows'])}/{int(grouped.iloc[0]['windows'])}", flush=True)
        print(grouped.head(8).to_string(index=False), flush=True)

    metrics = pd.DataFrame(all_rows).sort_values(["symbol", "candidate_rank", "rolling_end"])
    metrics.to_csv(args.output_dir / "rolling_candidate_metrics.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(args.output_dir / "selected_swing_candidates_summary.csv", index=False)
    (args.output_dir / "selected_swing_candidates.json").write_text(json.dumps(selected, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
