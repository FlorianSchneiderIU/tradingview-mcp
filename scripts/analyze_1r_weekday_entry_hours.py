#!/usr/bin/env python3
"""Analyze statistically meaningful 1R entry hours by weekday.

This focuses on symmetric 1:1 RR trades with SL = 2 * hourly ATR. It evaluates
entry weekday + UTC hour + direction cells under:

- no context,
- completed Monday candle context,
- price location relative to the completed Monday candle,
- daily market regime,
- selected Monday-context x daily-regime interactions.

For Monday-derived context, trades are only evaluated after the Monday candle is
complete, so the eligible weekdays are Tuesday through Sunday.
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.bybit_btc_monday_context_study import monday_week_start  # noqa: E402


BASE_DIR = Path("scripts/output/bybit_regime_histogram")
STAMP = "btcusdt_20210525_20260525"
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def day_name(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True).dt.day_name()


def add_time_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["entry_time"] = pd.to_datetime(out["entry_time"], utc=True)
    out["entry_weekday"] = day_name(out["entry_time"])
    out["entry_weekday_num"] = out["entry_time"].dt.weekday
    out["entry_hour_utc"] = out["entry_hour_utc"].astype(int)
    out["week_start"] = monday_week_start(out["entry_time"])
    out["win"] = (out["outcome"] == "target").astype(int)
    out["loss"] = (out["outcome"] == "stop").astype(int)
    return out


def tag_daily_regime(trades: pd.DataFrame, daily_regimes: pd.DataFrame) -> pd.DataFrame:
    if "regime" in trades.columns and trades["regime"].notna().any():
        return trades.copy()
    state = daily_regimes[["regime_available_time", "regime"]].copy()
    state["regime_available_time"] = pd.to_datetime(state["regime_available_time"], utc=True)
    return pd.merge_asof(
        trades.sort_values("entry_time"),
        state.sort_values("regime_available_time"),
        left_on="entry_time",
        right_on="regime_available_time",
        direction="backward",
    )


def monday_relative_labels(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["entry_vs_monday_range"] = np.select(
        [
            out["entry_price"] > out["monday_high"],
            out["entry_price"] < out["monday_low"],
        ],
        ["above_monday_high", "below_monday_low"],
        default="inside_monday_range",
    )
    out["entry_vs_monday_open"] = np.where(
        out["entry_price"] >= out["monday_open"],
        "above_or_at_monday_open",
        "below_monday_open",
    )
    monday_mid = (out["monday_high"] + out["monday_low"]) / 2.0
    out["entry_vs_monday_mid"] = np.where(
        out["entry_price"] >= monday_mid,
        "above_or_at_monday_mid",
        "below_monday_mid",
    )
    return out


def make_condition_rows(frame: pd.DataFrame, family: str, value_series: pd.Series, source: str) -> pd.DataFrame:
    keep = [
        "entry_time",
        "week_start",
        "entry_weekday",
        "entry_weekday_num",
        "entry_hour_utc",
        "entry_mode",
        "direction",
        "outcome",
        "win",
        "loss",
        "r_multiple",
        "holding_hours",
    ]
    out = frame[keep].copy()
    out["context_source"] = source
    out["context_family"] = family
    out["context_value"] = value_series.astype(str).to_numpy()
    return out


def build_analysis_frame(
    hourly_trades: pd.DataFrame,
    monday_trades: pd.DataFrame,
    daily_regimes: pd.DataFrame,
) -> pd.DataFrame:
    hourly = add_time_columns(hourly_trades)
    hourly = hourly[hourly["entry_mode"] == "close"].copy()
    hourly = tag_daily_regime(hourly, daily_regimes)

    monday = add_time_columns(monday_trades)
    monday = monday[monday["entry_mode"] == "close"].copy()
    monday = tag_daily_regime(monday, daily_regimes)
    monday = monday_relative_labels(monday)

    chunks: list[pd.DataFrame] = [
        make_condition_rows(hourly, "all", pd.Series("all", index=hourly.index), "all_weekdays"),
        make_condition_rows(hourly, "daily_regime", hourly["regime"], "all_weekdays"),
        make_condition_rows(monday, "monday_high_low_sequence", monday["monday_high_low_sequence"], "after_monday_close"),
        make_condition_rows(monday, "monday_color", monday["monday_color"], "after_monday_close"),
        make_condition_rows(monday, "monday_close_vs_pwh_pwl", monday["monday_close_vs_pwh_pwl"], "after_monday_close"),
        make_condition_rows(monday, "entry_vs_monday_range", monday["entry_vs_monday_range"], "after_monday_close"),
        make_condition_rows(monday, "entry_vs_monday_open", monday["entry_vs_monday_open"], "after_monday_close"),
        make_condition_rows(monday, "entry_vs_monday_mid", monday["entry_vs_monday_mid"], "after_monday_close"),
        make_condition_rows(
            monday,
            "monday_color_x_daily_regime",
            monday["monday_color"].astype(str) + "|" + monday["regime"].astype(str),
            "after_monday_close",
        ),
        make_condition_rows(
            monday,
            "entry_vs_monday_range_x_daily_regime",
            monday["entry_vs_monday_range"].astype(str) + "|" + monday["regime"].astype(str),
            "after_monday_close",
        ),
        make_condition_rows(
            monday,
            "monday_close_vs_pwh_pwl_x_daily_regime",
            monday["monday_close_vs_pwh_pwl"].astype(str) + "|" + monday["regime"].astype(str),
            "after_monday_close",
        ),
    ]
    return pd.concat(chunks, ignore_index=True)


def aggregate_cells(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        frame.groupby(
            [
                "context_source",
                "context_family",
                "context_value",
                "entry_weekday_num",
                "entry_weekday",
                "entry_hour_utc",
                "direction",
            ],
            as_index=False,
        )
        .agg(
            trades=("r_multiple", "size"),
            weeks=("week_start", "nunique"),
            wins=("win", "sum"),
            losses=("loss", "sum"),
            total_r=("r_multiple", "sum"),
            avg_holding_hours=("holding_hours", "mean"),
        )
    )
    grouped["win_rate"] = grouped["wins"] / grouped["trades"]
    grouped["avg_r"] = grouped["total_r"] / grouped["trades"]
    return grouped


def weekly_cell_stats(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(
            [
                "week_start",
                "context_source",
                "context_family",
                "context_value",
                "entry_weekday_num",
                "entry_weekday",
                "entry_hour_utc",
                "direction",
            ],
            as_index=False,
        )
        .agg(trades=("r_multiple", "size"), wins=("win", "sum"), total_r=("r_multiple", "sum"))
    )


def add_bootstrap_stats(cells: pd.DataFrame, weekly: pd.DataFrame, *, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    key_cols = [
        "context_source",
        "context_family",
        "context_value",
        "entry_weekday_num",
        "entry_weekday",
        "entry_hour_utc",
        "direction",
    ]
    weekly_groups = {key: group for key, group in weekly.groupby(key_cols, dropna=False)}
    for row in cells.itertuples(index=False):
        key = tuple(getattr(row, col) for col in key_cols)
        group = weekly_groups.get(key)
        out = row._asdict()
        if group is None or group.empty:
            out.update(
                {
                    "avg_r_ci_low": np.nan,
                    "avg_r_ci_high": np.nan,
                    "p_value_signflip": np.nan,
                    "win_rate_ci_low": np.nan,
                    "win_rate_ci_high": np.nan,
                }
            )
            rows.append(out)
            continue

        trades_arr = group["trades"].to_numpy(float)
        r_arr = group["total_r"].to_numpy(float)
        wins_arr = group["wins"].to_numpy(float)
        idx_base = np.arange(len(group))
        if len(group) == 1:
            avg_samples = np.array([r_arr.sum() / trades_arr.sum()])
            win_samples = np.array([wins_arr.sum() / trades_arr.sum()])
        else:
            sample_idx = rng.choice(idx_base, size=(n_boot, len(group)), replace=True)
            avg_samples = r_arr[sample_idx].sum(axis=1) / trades_arr[sample_idx].sum(axis=1)
            win_samples = wins_arr[sample_idx].sum(axis=1) / trades_arr[sample_idx].sum(axis=1)

        signs = rng.choice(np.array([-1.0, 1.0]), size=(n_boot, len(group)), replace=True)
        null_avg = (signs * r_arr).sum(axis=1) / trades_arr.sum()
        observed = float(r_arr.sum() / trades_arr.sum())
        p_value = float((np.sum(null_avg >= observed) + 1) / (n_boot + 1))
        avg_low, avg_high = np.quantile(avg_samples, [0.025, 0.975])
        win_low, win_high = np.quantile(win_samples, [0.025, 0.975])
        out.update(
            {
                "avg_r_ci_low": float(avg_low),
                "avg_r_ci_high": float(avg_high),
                "p_value_signflip": p_value,
                "win_rate_ci_low": float(win_low),
                "win_rate_ci_high": float(win_high),
            }
        )
        rows.append(out)
    return pd.DataFrame(rows)


def add_q_values(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["q_value_bh"] = np.nan
    for family, idx in out.groupby("context_family").groups.items():
        p = out.loc[idx, "p_value_signflip"].fillna(1.0).to_numpy(float)
        n = len(p)
        order = np.argsort(p)
        ranked = p[order]
        q = np.empty(n, dtype=float)
        running = 1.0
        for i in range(n - 1, -1, -1):
            running = min(running, ranked[i] * n / (i + 1))
            q[order[i]] = running
        out.loc[idx, "q_value_bh"] = np.clip(q, 0.0, 1.0)
    return out


def best_by_weekday(frame: pd.DataFrame, *, min_trades: int, min_weeks: int) -> pd.DataFrame:
    eligible = frame[(frame["trades"] >= min_trades) & (frame["weeks"] >= min_weeks)].copy()
    if eligible.empty:
        return eligible
    rows = []
    group_cols = ["context_source", "context_family", "context_value", "entry_weekday_num", "entry_weekday"]
    for _, group in eligible.groupby(group_cols, dropna=False):
        rows.append(group.sort_values(["avg_r", "trades"], ascending=[False, False]).iloc[0])
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def meaningful_cells(frame: pd.DataFrame, *, min_trades: int, min_weeks: int, q_threshold: float) -> pd.DataFrame:
    eligible = frame[
        (frame["trades"] >= min_trades)
        & (frame["weeks"] >= min_weeks)
        & (frame["avg_r_ci_low"] > 0)
    ].copy()
    eligible["passes_fdr"] = eligible["q_value_bh"] <= q_threshold
    return eligible.sort_values(["passes_fdr", "avg_r_ci_low", "avg_r"], ascending=[False, False, False]).reset_index(drop=True)


def render_report(output_dir: Path, summary: pd.DataFrame, best: pd.DataFrame, meaningful: pd.DataFrame, args: argparse.Namespace) -> Path:
    report_path = output_dir / f"{STAMP}_1r_weekday_entry_hour_report.html"

    def table(frame: pd.DataFrame, limit: int = 40) -> str:
        if frame.empty:
            return "<p>No rows.</p>"
        return frame.head(limit).to_html(index=False, escape=True, float_format=lambda value: f"{value:.4f}")

    top_meaningful = meaningful[
        [
            "context_family",
            "context_value",
            "entry_weekday_num",
            "entry_weekday",
            "entry_hour_utc",
            "direction",
            "trades",
            "weeks",
            "win_rate",
            "avg_r",
            "avg_r_ci_low",
            "avg_r_ci_high",
            "q_value_bh",
            "passes_fdr",
        ]
    ]
    top_best = best[
        [
            "context_family",
            "context_value",
            "entry_weekday_num",
            "entry_weekday",
            "entry_hour_utc",
            "direction",
            "trades",
            "weeks",
            "win_rate",
            "avg_r",
            "avg_r_ci_low",
            "avg_r_ci_high",
            "q_value_bh",
        ]
    ].sort_values(["context_family", "context_value", "entry_weekday_num"])
    top_best = top_best.drop(columns=["entry_weekday_num"])

    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{STAMP} 1R weekday entry hours</title>
  <style>
    body {{ font-family: Inter, ui-sans-serif, system-ui, sans-serif; margin: 32px; background: #f8fafc; color: #0f172a; }}
    main {{ max-width: 1320px; margin: 0 auto; }}
    section {{ margin: 26px 0; padding: 18px; background: white; border: 1px solid #dbe3ee; border-radius: 10px; overflow-x: auto; }}
    h1 {{ margin-bottom: 4px; }}
    p {{ color: #475569; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ border-bottom: 1px solid #e2e8f0; padding: 7px 8px; text-align: left; white-space: nowrap; }}
    th {{ background: #f1f5f9; }}
  </style>
</head>
<body>
<main>
  <h1>{html.escape(STAMP.upper())} 1R Weekday Entry Hours</h1>
  <p>Close-entry, symmetric 1R, SL = 2 * hourly ATR. Meaningful cells require at least {args.min_trades} trades, {args.min_weeks} weeks, and weekly bootstrap avg-R CI lower bound above zero. FDR pass uses q <= {args.q_threshold:.2f} within each context family.</p>
  <section>
    <h2>Meaningful Cells</h2>
    {table(top_meaningful, 80)}
  </section>
  <section>
    <h2>Best Cell Per Context Value And Weekday</h2>
    {table(top_best, 120)}
  </section>
</main>
</body>
</html>
"""
    report_path.write_text(doc, encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze 1R entry hour significance by weekday, Monday context, and daily regime.")
    parser.add_argument("--base-dir", type=Path, default=BASE_DIR)
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "weekday_1r_eval")
    parser.add_argument("--stamp", default=STAMP)
    parser.add_argument("--min-trades", type=int, default=100)
    parser.add_argument("--min-weeks", type=int, default=40)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--q-threshold", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    hourly_path = args.base_dir / f"{args.stamp}_hourly_trades.csv"
    monday_path = args.base_dir / f"{args.stamp}_monday_trades.csv"
    regime_path = args.base_dir / f"{args.stamp}_daily_regimes.csv"
    print("Loading 1R trade and regime files...")
    hourly_trades = pd.read_csv(hourly_path, parse_dates=["entry_time", "exit_time"])
    monday_trades = pd.read_csv(monday_path, parse_dates=["entry_time", "exit_time", "week_start"])
    daily_regimes = pd.read_csv(regime_path, parse_dates=["regime_available_time"])

    print("Building condition-expanded analysis frame...")
    analysis = build_analysis_frame(hourly_trades, monday_trades, daily_regimes)
    weekly = weekly_cell_stats(analysis)
    cells = aggregate_cells(analysis)

    print("Running weekly bootstrap/sign-flip stats...")
    summary = add_bootstrap_stats(cells, weekly, n_boot=args.n_boot, seed=args.seed)
    summary = add_q_values(summary)
    best = best_by_weekday(summary, min_trades=args.min_trades, min_weeks=args.min_weeks)
    meaningful = meaningful_cells(summary, min_trades=args.min_trades, min_weeks=args.min_weeks, q_threshold=args.q_threshold)

    summary_path = args.output_dir / f"{args.stamp}_1r_weekday_entry_hour_summary.csv"
    best_path = args.output_dir / f"{args.stamp}_1r_weekday_entry_hour_best.csv"
    meaningful_path = args.output_dir / f"{args.stamp}_1r_weekday_entry_hour_meaningful.csv"
    analysis_path = args.output_dir / f"{args.stamp}_1r_weekday_entry_hour_analysis_rows.csv"
    report_path = render_report(args.output_dir, summary, best, meaningful, args)

    summary.to_csv(summary_path, index=False)
    best.to_csv(best_path, index=False)
    meaningful.to_csv(meaningful_path, index=False)
    analysis.to_csv(analysis_path, index=False)

    print("\nMeaningful cells by context family:")
    if meaningful.empty:
        print("No cells passed the meaningful-cell thresholds.")
    else:
        print(meaningful["context_family"].value_counts().to_string())
        display = meaningful[
            [
                "context_family",
                "context_value",
                "entry_weekday",
                "entry_hour_utc",
                "direction",
                "trades",
                "weeks",
                "win_rate",
                "avg_r",
                "avg_r_ci_low",
                "q_value_bh",
                "passes_fdr",
            ]
        ].head(30)
        print("\nTop meaningful cells:")
        print(display.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    print("\nOutput files:")
    for path in [summary_path, best_path, meaningful_path, analysis_path, report_path]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
