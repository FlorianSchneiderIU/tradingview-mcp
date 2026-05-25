#!/usr/bin/env python3
"""Study hourly BTCUSDT trade outcomes conditioned on the completed Monday candle.

The Monday candle is treated as a weekly context signal only after it has closed
at Tuesday 00:00 UTC by default. Each eligible trade for the rest of that week
is tagged with:

- whether Monday's high was made before Monday's low, using 1h candles,
- whether Monday closed green/red/doji,
- whether Monday closed above the previous week high or below the previous week low.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.bybit_btc_regime_hourly_histogram import (  # noqa: E402
    BYBIT_URL,
    DIRECTIONS,
    ENTRY_MODES,
    build_hourly_features,
    bybit_symbol,
    fetch_bybit_klines,
    resolve_period,
    simulate_trades,
    trim_frame,
)


DEFAULT_CACHE_DIR = Path("scripts/.cache/bybit_regime_histogram")
DEFAULT_OUTPUT_DIR = Path("scripts/output/bybit_regime_histogram")
CONDITION_SPECS = {
    "monday_high_low_sequence": "monday_high_low_sequence",
    "monday_color": "monday_color",
    "monday_close_vs_pwh_pwl": "monday_close_vs_pwh_pwl",
    "monday_combo": "monday_combo",
}


SUMMARY_METRICS = [
    "trades",
    "wins",
    "losses",
    "ambiguous",
    "timeouts",
    "unresolved",
    "win_rate",
    "loss_rate",
    "avg_r",
    "median_r",
    "total_r",
    "avg_holding_hours",
]


def monday_week_start(series: pd.Series) -> pd.Series:
    times = pd.to_datetime(series, utc=True)
    return times.dt.floor("D") - pd.to_timedelta(times.dt.weekday, unit="D")


def classify_monday_close_vs_previous_week(close: float, previous_high: float, previous_low: float) -> str:
    if not np.isfinite(previous_high) or not np.isfinite(previous_low):
        return "no_previous_week"
    if close > previous_high:
        return "above_pwh"
    if close < previous_low:
        return "below_pwl"
    return "inside_previous_week_range"


def first_touch_time(frame: pd.DataFrame, column: str, value: float) -> pd.Timestamp:
    matches = frame[np.isclose(frame[column].astype(float), float(value), rtol=0.0, atol=1e-9)]
    if matches.empty:
        if column == "high":
            idx = frame[column].astype(float).idxmax()
        else:
            idx = frame[column].astype(float).idxmin()
        return pd.Timestamp(frame.loc[idx, "open_time"])
    return pd.Timestamp(matches["open_time"].iloc[0])


def build_monday_context(daily: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    daily_frame = daily.sort_values("open_time").reset_index(drop=True).copy()
    hourly_frame = hourly.sort_values("open_time").reset_index(drop=True).copy()
    daily_frame["week_start"] = monday_week_start(daily_frame["open_time"])

    weekly = (
        daily_frame.groupby("week_start", as_index=False)
        .agg(
            week_high=("high", "max"),
            week_low=("low", "min"),
            week_open=("open", "first"),
            week_close=("close", "last"),
            week_days=("open_time", "count"),
        )
        .sort_values("week_start")
    )
    previous_week = weekly[["week_start", "week_high", "week_low"]].copy()
    previous_week["week_start"] = previous_week["week_start"] + pd.Timedelta(days=7)
    previous_week = previous_week.rename(columns={"week_high": "previous_week_high", "week_low": "previous_week_low"})

    mondays = daily_frame[daily_frame["open_time"].dt.weekday == 0].copy()
    mondays = mondays.merge(previous_week, on="week_start", how="left")

    rows: list[dict[str, Any]] = []
    for row in mondays.itertuples(index=False):
        monday_open_time = pd.Timestamp(row.open_time)
        monday_close_time = monday_open_time + pd.Timedelta(days=1)
        week_end_time = monday_open_time + pd.Timedelta(days=7)
        monday_hours = hourly_frame[
            (hourly_frame["open_time"] >= monday_open_time)
            & (hourly_frame["open_time"] < monday_close_time)
        ].copy()
        if monday_hours.empty:
            continue

        monday_high_time = first_touch_time(monday_hours, "high", float(row.high))
        monday_low_time = first_touch_time(monday_hours, "low", float(row.low))
        if monday_high_time < monday_low_time:
            sequence = "high_first"
        elif monday_low_time < monday_high_time:
            sequence = "low_first"
        else:
            sequence = "same_hour"

        if float(row.close) > float(row.open):
            monday_color = "green"
        elif float(row.close) < float(row.open):
            monday_color = "red"
        else:
            monday_color = "doji"

        previous_week_high = float(row.previous_week_high) if pd.notna(row.previous_week_high) else np.nan
        previous_week_low = float(row.previous_week_low) if pd.notna(row.previous_week_low) else np.nan
        close_vs_previous = classify_monday_close_vs_previous_week(float(row.close), previous_week_high, previous_week_low)
        combo = f"{sequence}|{monday_color}|{close_vs_previous}"

        rows.append(
            {
                "week_start": monday_open_time,
                "context_available_time": monday_close_time,
                "context_expires_time": week_end_time,
                "monday_open": float(row.open),
                "monday_high": float(row.high),
                "monday_low": float(row.low),
                "monday_close": float(row.close),
                "monday_return_pct": (float(row.close) / float(row.open) - 1.0) * 100.0,
                "monday_range_pct": (float(row.high) / float(row.low) - 1.0) * 100.0 if float(row.low) else np.nan,
                "monday_high_time": monday_high_time,
                "monday_low_time": monday_low_time,
                "monday_high_low_sequence": sequence,
                "monday_color": monday_color,
                "previous_week_high": previous_week_high,
                "previous_week_low": previous_week_low,
                "monday_close_vs_pwh_pwl": close_vs_previous,
                "monday_combo": combo,
            }
        )

    return pd.DataFrame(rows).sort_values("week_start").reset_index(drop=True)


def assign_monday_context(trades: pd.DataFrame, monday_context: pd.DataFrame, *, context_from: str) -> pd.DataFrame:
    if trades.empty or monday_context.empty:
        return pd.DataFrame()

    if context_from == "week-open":
        trades_with_week = trades.copy()
        trades_with_week["week_start"] = monday_week_start(trades_with_week["entry_time"])
        merged = trades_with_week.merge(monday_context, on="week_start", how="left")
        return merged.dropna(subset=["monday_high_low_sequence"]).reset_index(drop=True)

    state = monday_context.sort_values("context_available_time")
    merged = pd.merge_asof(
        trades.sort_values("entry_time"),
        state,
        left_on="entry_time",
        right_on="context_available_time",
        direction="backward",
    )
    merged = merged[
        (merged["entry_time"] >= merged["context_available_time"])
        & (merged["entry_time"] < merged["context_expires_time"])
    ].copy()
    return merged.dropna(subset=["monday_high_low_sequence"]).reset_index(drop=True)


def expand_conditions(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    expanded: list[pd.DataFrame] = []
    for condition_name, source_column in CONDITION_SPECS.items():
        chunk = trades.copy()
        chunk["condition"] = condition_name
        chunk["condition_value"] = chunk[source_column].astype(str)
        expanded.append(chunk)
    return pd.concat(expanded, ignore_index=True)


def summarize_groups(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    columns = group_cols + SUMMARY_METRICS
    if frame.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        r_values = pd.to_numeric(group["r_multiple"], errors="coerce")
        trades = int(len(group))
        wins = int((group["outcome"] == "target").sum())
        losses = int((group["outcome"] == "stop").sum())
        row.update(
            {
                "trades": trades,
                "wins": wins,
                "losses": losses,
                "ambiguous": int((group["outcome"] == "ambiguous").sum()),
                "timeouts": int((group["outcome"] == "timeout").sum()),
                "unresolved": int((group["outcome"] == "unresolved").sum()),
                "win_rate": wins / trades if trades else np.nan,
                "loss_rate": losses / trades if trades else np.nan,
                "avg_r": float(r_values.mean()) if r_values.notna().any() else np.nan,
                "median_r": float(r_values.median()) if r_values.notna().any() else np.nan,
                "total_r": float(r_values.sum(skipna=True)),
                "avg_holding_hours": float(group["holding_hours"].mean()),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows, columns=columns).sort_values(group_cols).reset_index(drop=True)


def best_hours(hourly_summary: pd.DataFrame, min_trades: int) -> pd.DataFrame:
    if hourly_summary.empty:
        return hourly_summary.copy()
    candidates = hourly_summary[hourly_summary["trades"] >= min_trades].copy()
    if candidates.empty:
        return pd.DataFrame(columns=hourly_summary.columns)
    rows = []
    for _, group in candidates.groupby(["condition", "condition_value"], dropna=False):
        rows.append(group.sort_values(["avg_r", "trades"], ascending=[False, False]).iloc[0])
    return pd.DataFrame(rows).sort_values(["condition", "avg_r"], ascending=[True, False]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch BTCUSDT Bybit candles and summarize hourly ATR RR outcomes by "
            "completed Monday candle conditions."
        )
    )
    parser.add_argument("--symbol", default="BTCUSDT", help="Bybit symbol, e.g. BTCUSDT")
    parser.add_argument("--category", default="linear", help="Bybit product category, default: linear")
    parser.add_argument("--base-url", default=BYBIT_URL, help="Bybit REST base URL")
    parser.add_argument("--years", type=int, default=5, help="Lookback years when --start is omitted")
    parser.add_argument("--start", help="UTC start date/time, e.g. 2021-05-25 or 2021-05-25T00:00:00Z")
    parser.add_argument("--end", help="UTC end date/time, defaults to current UTC hour")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force-refresh", action="store_true", help="Ignore cached Bybit kline CSVs")
    parser.add_argument("--daily-warmup-days", type=int, default=21, help="Extra daily candles for previous-week context")
    parser.add_argument("--hourly-warmup-hours", type=int, default=250, help="Extra hourly candles for ATR")
    parser.add_argument("--atr-length", type=int, default=14, help="Hourly ATR length for trade stops")
    parser.add_argument("--sl-atr-multiple", type=float, default=2.0, help="Stop distance in hourly ATR multiples")
    parser.add_argument("--rr", type=float, default=1.0, help="Reward:risk multiple")
    parser.add_argument("--entry-mode", choices=("open", "close", "both"), default="both")
    parser.add_argument(
        "--context-from",
        choices=("after-close", "week-open"),
        default="after-close",
        help="after-close avoids lookahead and applies Monday context from Tuesday 00:00 UTC",
    )
    parser.add_argument(
        "--tie-policy",
        choices=("stop", "target", "skip"),
        default="stop",
        help="When TP and SL both hit inside one hourly candle, default is conservative stop",
    )
    parser.add_argument("--max-hold-hours", type=int, default=0, help="0 means hold until TP/SL or data end")
    parser.add_argument("--min-trades", type=int, default=30, help="Minimum trades for best-hour selection")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbol = bybit_symbol(args.symbol)
    period = resolve_period(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    hourly_fetch_start = period.start - pd.Timedelta(hours=args.hourly_warmup_hours)
    daily_fetch_start = period.start - pd.Timedelta(days=max(args.daily_warmup_days, 14))
    print(f"Fetching {symbol} 1h candles from {hourly_fetch_start} to {period.end} ...")
    hourly_raw = fetch_bybit_klines(
        symbol,
        "1h",
        hourly_fetch_start,
        period.end,
        category=args.category,
        base_url=args.base_url,
        cache_dir=args.cache_dir,
        force_refresh=args.force_refresh,
    )
    print(f"Fetching {symbol} 1d candles from {daily_fetch_start} to {period.end} ...")
    daily_raw = fetch_bybit_klines(
        symbol,
        "1d",
        daily_fetch_start,
        period.end,
        category=args.category,
        base_url=args.base_url,
        cache_dir=args.cache_dir,
        force_refresh=args.force_refresh,
    )

    hourly = build_hourly_features(hourly_raw, args.atr_length)
    trade_frames: list[pd.DataFrame] = []
    entry_modes = ENTRY_MODES if args.entry_mode == "both" else (args.entry_mode,)
    for entry_mode in entry_modes:
        for direction in DIRECTIONS:
            simulated = simulate_trades(
                hourly,
                entry_mode=entry_mode,
                direction=direction,
                atr_length=args.atr_length,
                sl_atr_multiple=args.sl_atr_multiple,
                rr=args.rr,
                tie_policy=args.tie_policy,
                max_hold_hours=args.max_hold_hours,
            )
            trade_frames.append(trim_frame(simulated, period.start, period.end, time_column="entry_time"))

    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    if args.tie_policy == "skip" and "outcome" in trades.columns:
        trades = trades[trades["outcome"] != "ambiguous"].reset_index(drop=True)

    monday_context = build_monday_context(daily_raw, hourly_raw)
    monday_context = trim_frame(
        monday_context,
        period.start - pd.Timedelta(days=7),
        period.end,
        time_column="week_start",
    )
    monday_trades = assign_monday_context(trades, monday_context, context_from=args.context_from)
    expanded = expand_conditions(monday_trades)

    condition_summary = summarize_groups(expanded, ["condition", "condition_value", "entry_mode", "direction"])
    hourly_summary = summarize_groups(
        expanded,
        ["condition", "condition_value", "entry_mode", "direction", "entry_hour_utc"],
    )
    best = best_hours(hourly_summary, args.min_trades)

    stamp = f"{symbol.lower()}_{period.start:%Y%m%d}_{period.end:%Y%m%d}"
    prefix = args.output_dir / stamp
    context_path = prefix.with_name(f"{stamp}_monday_context.csv")
    trades_path = prefix.with_name(f"{stamp}_monday_trades.csv")
    condition_summary_path = prefix.with_name(f"{stamp}_monday_condition_summary.csv")
    hourly_summary_path = prefix.with_name(f"{stamp}_monday_hourly_histogram.csv")
    best_path = prefix.with_name(f"{stamp}_monday_best_hours.csv")
    run_summary_path = prefix.with_name(f"{stamp}_monday_run_summary.txt")

    monday_context.to_csv(context_path, index=False)
    monday_trades.to_csv(trades_path, index=False)
    condition_summary.to_csv(condition_summary_path, index=False)
    hourly_summary.to_csv(hourly_summary_path, index=False)
    best.to_csv(best_path, index=False)

    run_summary = [
        f"symbol={symbol}",
        f"period_start={period.start}",
        f"period_end={period.end}",
        f"context_from={args.context_from}",
        f"monday_context_rows={len(monday_context)}",
        f"monday_tagged_trades={len(monday_trades)}",
        f"expanded_condition_rows={len(expanded)}",
        f"sl_atr_multiple={args.sl_atr_multiple}",
        f"rr={args.rr}",
        f"tie_policy={args.tie_policy}",
        f"min_trades={args.min_trades}",
        f"monday_context_csv={context_path}",
        f"monday_trades_csv={trades_path}",
        f"condition_summary_csv={condition_summary_path}",
        f"hourly_histogram_csv={hourly_summary_path}",
        f"best_hours_csv={best_path}",
    ]
    run_summary_path.write_text("\n".join(run_summary) + "\n", encoding="utf-8")

    print("\nMonday condition counts:")
    if not monday_context.empty:
        for column in ["monday_high_low_sequence", "monday_color", "monday_close_vs_pwh_pwl"]:
            print(f"\n{column}:")
            print(monday_context[column].value_counts().to_string())

    print("\nBest hour+direction per Monday condition by avg_r:")
    if best.empty:
        print("No rows meet the min-trades threshold.")
    else:
        display = best[
            [
                "condition",
                "condition_value",
                "entry_mode",
                "direction",
                "entry_hour_utc",
                "trades",
                "win_rate",
                "avg_r",
                "total_r",
            ]
        ].sort_values(["condition", "condition_value"])
        print(display.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    print("\nOutput files:")
    for path in [context_path, trades_path, condition_summary_path, hourly_summary_path, best_path, run_summary_path]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
