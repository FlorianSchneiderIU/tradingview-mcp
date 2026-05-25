#!/usr/bin/env python3
"""Summarize a Monday-context reward:risk sweep with weekly block bootstrap CIs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_BASE_DIR = Path("scripts/output/bybit_regime_histogram")
DEFAULT_SWEEP_DIR = DEFAULT_BASE_DIR / "monday_rr_sweep"
STAMP = "btcusdt_20210525_20260525"
REQUESTED_CONDITIONS = [
    "monday_high_low_sequence",
    "monday_color",
    "monday_close_vs_pwh_pwl",
]
RR_PATHS = {
    0.5: "rr_0_5",
    1.0: "",
    1.5: "rr_1_5",
    2.0: "rr_2_0",
    3.0: "rr_3_0",
}


def rr_label(rr: float) -> str:
    return f"{rr:g}R"


def condition_column(condition: str) -> str:
    return condition


def trades_path(base_dir: Path, sweep_dir: Path, stamp: str, rr: float) -> Path:
    folder = RR_PATHS[rr]
    if folder:
        return sweep_dir / folder / f"{stamp}_monday_trades.csv"
    return base_dir / f"{stamp}_monday_trades.csv"


def best_path(base_dir: Path, sweep_dir: Path, stamp: str, rr: float) -> Path:
    folder = RR_PATHS[rr]
    if folder:
        return sweep_dir / folder / f"{stamp}_monday_best_hours.csv"
    return base_dir / f"{stamp}_monday_best_hours.csv"


def metric_summary(frame: pd.DataFrame) -> dict[str, Any]:
    r_values = pd.to_numeric(frame["r_multiple"], errors="coerce")
    trades = int(len(frame))
    wins = int((frame["outcome"] == "target").sum())
    losses = int((frame["outcome"] == "stop").sum())
    return {
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / trades if trades else np.nan,
        "avg_r": float(r_values.mean()) if r_values.notna().any() else np.nan,
        "total_r": float(r_values.sum(skipna=True)),
        "median_r": float(r_values.median()) if r_values.notna().any() else np.nan,
    }


def load_trades(base_dir: Path, sweep_dir: Path, stamp: str) -> dict[float, pd.DataFrame]:
    out: dict[float, pd.DataFrame] = {}
    for rr in RR_PATHS:
        path = trades_path(base_dir, sweep_dir, stamp, rr)
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, parse_dates=["entry_time", "week_start"])
        out[rr] = frame
    return out


def load_best(base_dir: Path, sweep_dir: Path, stamp: str) -> dict[float, pd.DataFrame]:
    out: dict[float, pd.DataFrame] = {}
    for rr in RR_PATHS:
        path = best_path(base_dir, sweep_dir, stamp, rr)
        if not path.exists():
            raise FileNotFoundError(path)
        out[rr] = pd.read_csv(path)
    return out


def baseline_setups(best_1r: pd.DataFrame) -> pd.DataFrame:
    setups = best_1r[
        (best_1r["condition"].isin(REQUESTED_CONDITIONS))
        & (best_1r["entry_mode"] == "close")
    ].copy()
    return setups.sort_values(["condition", "condition_value"]).reset_index(drop=True)


def filter_setup(frame: pd.DataFrame, setup: pd.Series) -> pd.DataFrame:
    column = condition_column(str(setup["condition"]))
    mask = (
        (frame[column].astype(str) == str(setup["condition_value"]))
        & (frame["entry_mode"] == str(setup["entry_mode"]))
        & (frame["direction"] == str(setup["direction"]))
        & (frame["entry_hour_utc"] == int(setup["entry_hour_utc"]))
    )
    return frame.loc[mask].copy()


def block_bootstrap_diff(
    base: pd.DataFrame,
    other: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    keys = ["entry_time", "entry_mode", "direction"]
    merged = base[keys + ["week_start", "outcome", "r_multiple"]].merge(
        other[keys + ["outcome", "r_multiple"]],
        on=keys,
        how="inner",
        suffixes=("_base", "_other"),
    )
    if merged.empty:
        return {
            "matched_trades": 0,
            "avg_r_diff": np.nan,
            "avg_r_diff_ci_low": np.nan,
            "avg_r_diff_ci_high": np.nan,
            "win_rate_diff": np.nan,
            "win_rate_diff_ci_low": np.nan,
            "win_rate_diff_ci_high": np.nan,
        }

    weeks = np.array(sorted(merged["week_start"].dropna().unique()))
    if len(weeks) == 0:
        return {
            "matched_trades": int(len(merged)),
            "avg_r_diff": np.nan,
            "avg_r_diff_ci_low": np.nan,
            "avg_r_diff_ci_high": np.nan,
            "win_rate_diff": np.nan,
            "win_rate_diff_ci_low": np.nan,
            "win_rate_diff_ci_high": np.nan,
        }

    weekly = (
        merged.assign(
            wins_base=(merged["outcome_base"] == "target").astype(float),
            wins_other=(merged["outcome_other"] == "target").astype(float),
        )
        .groupby("week_start", as_index=False)
        .agg(
            trades=("entry_time", "count"),
            r_base=("r_multiple_base", "sum"),
            r_other=("r_multiple_other", "sum"),
            wins_base=("wins_base", "sum"),
            wins_other=("wins_other", "sum"),
        )
        .sort_values("week_start")
    )

    trades_arr = weekly["trades"].to_numpy(float)
    r_base_arr = weekly["r_base"].to_numpy(float)
    r_other_arr = weekly["r_other"].to_numpy(float)
    wins_base_arr = weekly["wins_base"].to_numpy(float)
    wins_other_arr = weekly["wins_other"].to_numpy(float)

    def metrics(indices: np.ndarray) -> tuple[float, float]:
        trades_sum = trades_arr[indices].sum()
        if trades_sum <= 0:
            return np.nan, np.nan
        avg_diff = (r_other_arr[indices].sum() - r_base_arr[indices].sum()) / trades_sum
        win_diff = (wins_other_arr[indices].sum() - wins_base_arr[indices].sum()) / trades_sum
        return float(avg_diff), float(win_diff)

    observed_avg_diff = float((merged["r_multiple_other"].sum() - merged["r_multiple_base"].sum()) / len(merged))
    observed_win_diff = float(((merged["outcome_other"] == "target").sum() - (merged["outcome_base"] == "target").sum()) / len(merged))
    rng = np.random.default_rng(seed)
    avg_diffs = np.empty(n_boot, dtype=float)
    win_diffs = np.empty(n_boot, dtype=float)
    week_indices = np.arange(len(weekly))

    for i in range(n_boot):
        sampled_indices = rng.choice(week_indices, size=len(week_indices), replace=True)
        avg_diffs[i], win_diffs[i] = metrics(sampled_indices)

    avg_low, avg_high = np.quantile(avg_diffs, [0.025, 0.975])
    win_low, win_high = np.quantile(win_diffs, [0.025, 0.975])
    return {
        "matched_trades": int(len(merged)),
        "avg_r_diff": observed_avg_diff,
        "avg_r_diff_ci_low": float(avg_low),
        "avg_r_diff_ci_high": float(avg_high),
        "win_rate_diff": observed_win_diff,
        "win_rate_diff_ci_low": float(win_low),
        "win_rate_diff_ci_high": float(win_high),
    }


def fixed_setup_summary(
    trades_by_rr: dict[float, pd.DataFrame],
    setups: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, setup in setups.iterrows():
        base_slice = filter_setup(trades_by_rr[1.0], setup)
        for rr, frame in trades_by_rr.items():
            chunk = filter_setup(frame, setup)
            row = {
                "baseline_condition": setup["condition"],
                "condition_value": setup["condition_value"],
                "baseline_entry_mode": setup["entry_mode"],
                "baseline_direction": setup["direction"],
                "baseline_hour_utc": int(setup["entry_hour_utc"]),
                "rr": rr,
                "rr_label": rr_label(rr),
            }
            row.update(metric_summary(chunk))
            if rr == 1.0:
                row.update(
                    {
                        "matched_trades": int(len(chunk)),
                        "avg_r_diff": 0.0,
                        "avg_r_diff_ci_low": 0.0,
                        "avg_r_diff_ci_high": 0.0,
                        "win_rate_diff": 0.0,
                        "win_rate_diff_ci_low": 0.0,
                        "win_rate_diff_ci_high": 0.0,
                    }
                )
            else:
                row.update(block_bootstrap_diff(base_slice, chunk, n_boot=n_boot, seed=seed + int(rr * 100)))
            rows.append(row)
    return pd.DataFrame(rows)


def reoptimized_best_summary(best_by_rr: dict[float, pd.DataFrame]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for rr, frame in best_by_rr.items():
        chunk = frame[
            (frame["condition"].isin(REQUESTED_CONDITIONS))
            & (frame["entry_mode"] == "close")
        ].copy()
        chunk["rr"] = rr
        chunk["rr_label"] = rr_label(rr)
        rows.append(chunk)
    return pd.concat(rows, ignore_index=True).sort_values(["condition", "condition_value", "rr"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Monday-context RR sweep outputs.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    parser.add_argument("--stamp", default=STAMP)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trades_by_rr = load_trades(args.base_dir, args.sweep_dir, args.stamp)
    best_by_rr = load_best(args.base_dir, args.sweep_dir, args.stamp)
    setups = baseline_setups(best_by_rr[1.0])

    fixed = fixed_setup_summary(trades_by_rr, setups, n_boot=args.n_boot, seed=args.seed)
    reoptimized = reoptimized_best_summary(best_by_rr)

    fixed_path = args.output_dir / f"{args.stamp}_rr_sweep_fixed_1r_setups.csv"
    reoptimized_path = args.output_dir / f"{args.stamp}_rr_sweep_reoptimized_best_hours.csv"
    fixed.to_csv(fixed_path, index=False)
    reoptimized.to_csv(reoptimized_path, index=False)

    print("Fixed 1R setup RR sweep:")
    display = fixed[
        [
            "baseline_condition",
            "condition_value",
            "baseline_direction",
            "baseline_hour_utc",
            "rr_label",
            "trades",
            "win_rate",
            "avg_r",
            "avg_r_diff",
            "avg_r_diff_ci_low",
            "avg_r_diff_ci_high",
            "win_rate_diff",
            "win_rate_diff_ci_low",
            "win_rate_diff_ci_high",
        ]
    ]
    print(display.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nOutput files:")
    print(f"  {fixed_path}")
    print(f"  {reoptimized_path}")


if __name__ == "__main__":
    main()
