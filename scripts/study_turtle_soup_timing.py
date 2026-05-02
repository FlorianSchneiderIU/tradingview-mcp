from __future__ import annotations

import argparse
import math
import random
import sys
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backtest_turtle_soup import Config, fetch_klines, run_backtest, summarize

WEEKDAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def build_profiles() -> dict[str, Config]:
    base = dict(exec_tf="5m", structure_tf="15m", entry_mode="limit_mid")
    return {
        "baseline": Config(**base),
        "study_default": Config(**base, min_sweep_reclaim_pos=0.70, block_dead_zone=False),
        "dead_zone": Config(**base, block_dead_zone=True),
        "dead_reclaim70": Config(**base, min_sweep_reclaim_pos=0.70, block_dead_zone=True),
        "htf_4h_dead_reclaim70": Config(
            **base,
            htf_bias_mode="4h_ema",
            min_sweep_reclaim_pos=0.70,
            block_dead_zone=True,
        ),
    }


def describe_config(name: str, cfg: Config) -> str:
    parts = [
        f"profile={name}",
        f"exec_tf={cfg.exec_tf}",
        f"struct_tf={cfg.structure_tf}",
        f"entry_mode={cfg.entry_mode}",
        f"reclaim={cfg.min_sweep_reclaim_pos:.2f}",
        f"dead_zone={'on' if cfg.block_dead_zone else 'off'}",
        f"bias={cfg.htf_bias_mode}",
    ]
    return " | ".join(parts)


def week_of_month(ts: pd.Timestamp) -> int:
    return int((ts.day - 1) // 7 + 1)


def bootstrap_mean_ci(values: list[float], iterations: int, seed: int) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    if len(values) == 1:
        return values[0], values[0]

    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(iterations):
        sample = rng.choices(values, k=n)
        means.append(sum(sample) / n)
    means.sort()
    low_idx = max(0, int(0.025 * (iterations - 1)))
    high_idx = min(iterations - 1, int(0.975 * (iterations - 1)))
    return means[low_idx], means[high_idx]


def bootstrap_delta_ci(values: list[float], rest_values: list[float], iterations: int, seed: int) -> tuple[float, float]:
    if not values or not rest_values:
        return math.nan, math.nan
    if len(values) == 1 and len(rest_values) == 1:
        delta = values[0] - rest_values[0]
        return delta, delta

    rng = random.Random(seed)
    n_group = len(values)
    n_rest = len(rest_values)
    deltas: list[float] = []
    for _ in range(iterations):
        sample_group = rng.choices(values, k=n_group)
        sample_rest = rng.choices(rest_values, k=n_rest)
        deltas.append(sum(sample_group) / n_group - sum(sample_rest) / n_rest)
    deltas.sort()
    low_idx = max(0, int(0.025 * (iterations - 1)))
    high_idx = min(iterations - 1, int(0.975 * (iterations - 1)))
    return deltas[low_idx], deltas[high_idx]


def compute_daily_extremes(exec_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    frame = exec_df.reset_index(drop=True).copy()
    frame["day"] = frame["open_time"].dt.floor("1D")
    rows: list[dict] = []

    for day, day_df in frame.groupby("day", sort=True):
        high_idx = int(day_df["high"].idxmax())
        low_idx = int(day_df["low"].idxmin())
        high_row = frame.iloc[high_idx]
        low_row = frame.iloc[low_idx]
        rows.append({
            "symbol": symbol,
            "day": day,
            "weekday": day.day_name(),
            "weekday_num": int(day.day_of_week),
            "week_of_month": week_of_month(day),
            "day_high": float(day_df["high"].max()),
            "day_low": float(day_df["low"].min()),
            "day_high_hour": int(high_row["open_time"].hour),
            "day_low_hour": int(low_row["open_time"].hour),
            "day_high_time": high_row["open_time"],
            "day_low_time": low_row["open_time"],
        })

    return pd.DataFrame(rows)


def analyze_trades(exec_df: pd.DataFrame, daily_extremes: pd.DataFrame, trades, symbol: str) -> pd.DataFrame:
    frame = exec_df.reset_index(drop=True).copy()
    frame["day"] = frame["open_time"].dt.floor("1D")
    day_slices = {day: day_df for day, day_df in frame.groupby("day", sort=True)}
    extreme_lookup = daily_extremes.set_index("day")[["day_high", "day_low"]].to_dict("index")

    rows: list[dict] = []
    for trade in trades:
        day = trade.entry_time.floor("1D")
        day_df = day_slices.get(day)
        day_extreme = extreme_lookup.get(day)
        if day_df is None or day_extreme is None:
            continue

        future_df = day_df[day_df.index > trade.entry_index]
        future_high = float(future_df["high"].max()) if len(future_df) else math.nan
        future_low = float(future_df["low"].min()) if len(future_df) else math.nan
        day_low_ahead = bool(len(future_df) and future_low <= day_extreme["day_low"] + 1e-9)
        day_high_ahead = bool(len(future_df) and future_high >= day_extreme["day_high"] - 1e-9)
        relevant_extreme_ahead = day_low_ahead if trade.direction == "long" else day_high_ahead

        rows.append({
            "symbol": symbol,
            "day": day,
            "weekday": trade.entry_time.day_name(),
            "weekday_num": int(trade.entry_time.day_of_week),
            "week_of_month": week_of_month(trade.entry_time),
            "direction": trade.direction,
            "entry_time": trade.entry_time,
            "entry_hour": int(trade.entry_time.hour),
            "r_multiple": float(trade.r_multiple),
            "win": 1 if trade.r_multiple > 0 else 0,
            "relevant_extreme_ahead": 1 if relevant_extreme_ahead else 0,
            "day_low_ahead": 1 if day_low_ahead else 0,
            "day_high_ahead": 1 if day_high_ahead else 0,
            "hold_bars": int(trade.hold_bars),
            "exit_reason": trade.exit_reason,
            "zone_tf": trade.zone_tf,
        })

    return pd.DataFrame(rows)


def distribution_by_hour(extremes: pd.DataFrame, hour_col: str, label: str) -> pd.DataFrame:
    out = (
        extremes.groupby(hour_col)
        .size()
        .rename("days")
        .reset_index()
        .sort_values(hour_col)
        .rename(columns={hour_col: "hour"})
    )
    total_days = int(out["days"].sum())
    out["pct_days"] = (out["days"] / total_days * 100.0).round(2) if total_days else 0.0
    out.insert(0, "metric", label)
    return out


def dominant_hour_by_group(extremes: pd.DataFrame, group_cols: list[str], hour_col: str, metric: str) -> pd.DataFrame:
    if extremes.empty:
        return pd.DataFrame(columns=[*group_cols, "metric", "top_hour", "days", "pct_days"])

    rows: list[dict] = []
    for keys, group in extremes.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        counts = group[hour_col].value_counts().sort_values(ascending=False)
        if counts.empty:
            continue
        top_hour = int(counts.index[0])
        top_days = int(counts.iloc[0])
        total_days = int(counts.sum())
        row = {col: value for col, value in zip(group_cols, keys)}
        row.update({
            "metric": metric,
            "top_hour": top_hour,
            "days": top_days,
            "pct_days": round(top_days / total_days * 100.0, 2) if total_days else 0.0,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_entry_hours(trade_rows: pd.DataFrame) -> pd.DataFrame:
    if trade_rows.empty:
        return pd.DataFrame(columns=[
            "direction", "entry_hour", "trades", "win_rate", "avg_r",
            "net_r", "extreme_ahead_pct", "avg_hold_bars",
        ])

    out = (
        trade_rows.groupby(["direction", "entry_hour"])
        .agg(
            trades=("r_multiple", "size"),
            win_rate=("win", lambda s: round(float(s.mean() * 100.0), 2)),
            avg_r=("r_multiple", lambda s: round(float(s.mean()), 3)),
            net_r=("r_multiple", lambda s: round(float(s.sum()), 3)),
            extreme_ahead_pct=("relevant_extreme_ahead", lambda s: round(float(s.mean() * 100.0), 2)),
            avg_hold_bars=("hold_bars", lambda s: round(float(s.mean()), 1)),
        )
        .reset_index()
        .sort_values(["direction", "entry_hour"])
    )
    return out


def summarize_pattern_axis(
    trade_rows: pd.DataFrame,
    axis_name: str,
    sort_cols: list[str],
    display_cols: list[str],
    bootstrap_iterations: int,
    significance_min_trades: int,
) -> pd.DataFrame:
    if trade_rows.empty:
        return pd.DataFrame(columns=[
            "direction", *display_cols, "trades", "win_rate", "avg_r", "net_r",
            "extreme_ahead_pct", "avg_hold_bars", "rest_avg_r", "delta_avg_r",
            "mean_ci_low", "mean_ci_high", "delta_ci_low", "delta_ci_high",
            "symbols_with_trades", "symbols_negative", "symbols_positive",
            "significant_underperf", "significant_outperf",
        ])

    rows: list[dict] = []
    group_cols = ["direction", *display_cols]
    for keys, group in trade_rows.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        direction = keys[0]
        direction_rows = trade_rows[trade_rows["direction"] == direction]

        mask = pd.Series(True, index=direction_rows.index)
        for col, value in zip(display_cols, keys[1:]):
            mask &= direction_rows[col] == value
        rest_rows = direction_rows[~mask]

        values = group["r_multiple"].astype(float).tolist()
        rest_values = rest_rows["r_multiple"].astype(float).tolist()
        trades = len(values)
        avg_r = float(sum(values) / trades)
        rest_avg_r = float(sum(rest_values) / len(rest_values)) if rest_values else math.nan
        delta_avg_r = avg_r - rest_avg_r if not math.isnan(rest_avg_r) else math.nan

        if trades >= significance_min_trades:
            seed_base = 1000 + int(sum(ord(ch) for ch in f"{axis_name}|{keys}"))
            mean_ci_low, mean_ci_high = bootstrap_mean_ci(values, bootstrap_iterations, seed_base)
            delta_ci_low, delta_ci_high = bootstrap_delta_ci(values, rest_values, bootstrap_iterations, seed_base + 17)
        else:
            mean_ci_low = mean_ci_high = math.nan
            delta_ci_low = delta_ci_high = math.nan

        symbol_means = group.groupby("symbol")["r_multiple"].mean()
        row = {
            "direction": direction,
            "trades": trades,
            "win_rate": round(float(group["win"].mean() * 100.0), 2),
            "avg_r": round(avg_r, 3),
            "net_r": round(float(group["r_multiple"].sum()), 3),
            "extreme_ahead_pct": round(float(group["relevant_extreme_ahead"].mean() * 100.0), 2),
            "avg_hold_bars": round(float(group["hold_bars"].mean()), 1),
            "rest_avg_r": round(rest_avg_r, 3) if not math.isnan(rest_avg_r) else math.nan,
            "delta_avg_r": round(delta_avg_r, 3) if not math.isnan(delta_avg_r) else math.nan,
            "mean_ci_low": round(mean_ci_low, 3) if not math.isnan(mean_ci_low) else math.nan,
            "mean_ci_high": round(mean_ci_high, 3) if not math.isnan(mean_ci_high) else math.nan,
            "delta_ci_low": round(delta_ci_low, 3) if not math.isnan(delta_ci_low) else math.nan,
            "delta_ci_high": round(delta_ci_high, 3) if not math.isnan(delta_ci_high) else math.nan,
            "symbols_with_trades": int(len(symbol_means)),
            "symbols_negative": int((symbol_means < 0).sum()),
            "symbols_positive": int((symbol_means > 0).sum()),
        }
        for col, value in zip(display_cols, keys[1:]):
            row[col] = value
        row["significant_underperf"] = bool(
            trades >= significance_min_trades
            and not math.isnan(mean_ci_high)
            and not math.isnan(delta_ci_high)
            and mean_ci_high < 0
            and delta_ci_high < 0
        )
        row["significant_outperf"] = bool(
            trades >= significance_min_trades
            and not math.isnan(mean_ci_low)
            and not math.isnan(delta_ci_low)
            and mean_ci_low > 0
            and delta_ci_low > 0
        )
        rows.append(row)

    out = pd.DataFrame(rows)
    return out.sort_values(sort_cols + ["direction"]).reset_index(drop=True)


def summarize_symbols(symbol_rows: pd.DataFrame) -> pd.DataFrame:
    if symbol_rows.empty:
        return pd.DataFrame(columns=["symbol", "direction", "trades", "win_rate", "avg_r", "net_r"])

    out = (
        symbol_rows.groupby(["symbol", "direction"])
        .agg(
            trades=("r_multiple", "size"),
            win_rate=("win", lambda s: round(float(s.mean() * 100.0), 2)),
            avg_r=("r_multiple", lambda s: round(float(s.mean()), 3)),
            net_r=("r_multiple", lambda s: round(float(s.sum()), 3)),
        )
        .reset_index()
        .sort_values(["symbol", "direction"])
    )
    return out


def pattern_candidates(pattern_rows: pd.DataFrame, min_trades: int) -> pd.DataFrame:
    if pattern_rows.empty:
        return pattern_rows
    filtered = pattern_rows[
        (pattern_rows["trades"] >= min_trades)
        & (
            pattern_rows["significant_underperf"]
            | (
                (pattern_rows["avg_r"] < 0)
                & (pattern_rows["symbols_negative"] >= 2)
            )
        )
    ].copy()
    return filtered


def write_optional_csvs(save_dir: Path | None, tables: Iterable[tuple[str, pd.DataFrame]]) -> None:
    if save_dir is None:
        return
    save_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables:
        table.to_csv(save_dir / f"{name}.csv", index=False)


def print_section(title: str, table: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    if table.empty:
        print("(no rows)")
        return
    print(table.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=["ETHUSDT", "SOLUSDT"])
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--profile", choices=sorted(build_profiles()), default="study_default")
    parser.add_argument("--min-trades", type=int, default=2)
    parser.add_argument("--significance-min-trades", type=int, default=8)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--save-dir")
    args = parser.parse_args()

    profiles = build_profiles()
    cfg = profiles[args.profile]

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=args.days)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    print("=== Study Config ===")
    print(describe_config(args.profile, cfg))
    print(f"symbols={','.join(args.symbols)} | days={args.days} | from={start_dt:%Y-%m-%d %H:%M UTC} | to={end_dt:%Y-%m-%d %H:%M UTC}")
    print(f"candidate_block_min_trades={args.min_trades}")
    print(f"significance_min_trades={args.significance_min_trades} | bootstrap_iterations={args.bootstrap_iterations}")

    daily_extreme_tables: list[pd.DataFrame] = []
    trade_tables: list[pd.DataFrame] = []
    symbol_summaries: list[dict] = []

    for symbol in args.symbols:
        exec_df = fetch_klines(symbol, cfg.exec_tf, start_ms, end_ms)
        trades = run_backtest(exec_df, cfg)
        summary = summarize(trades)
        symbol_summaries.append({
            "symbol": symbol,
            "trades": summary["trades"],
            "win_rate": summary["win_rate"],
            "profit_factor": summary["profit_factor"],
            "net_r": summary["net_r"],
        })

        daily_extremes = compute_daily_extremes(exec_df, symbol)
        trade_rows = analyze_trades(exec_df, daily_extremes, trades, symbol)
        daily_extreme_tables.append(daily_extremes)
        trade_tables.append(trade_rows)

    daily_extremes_all = pd.concat(daily_extreme_tables, ignore_index=True) if daily_extreme_tables else pd.DataFrame()
    trade_rows_all = pd.concat(trade_tables, ignore_index=True) if trade_tables else pd.DataFrame()

    summary_table = pd.DataFrame(symbol_summaries).sort_values("symbol")
    high_dist_all = distribution_by_hour(daily_extremes_all, "day_high_hour", "daily_high")
    low_dist_all = distribution_by_hour(daily_extremes_all, "day_low_hour", "daily_low")
    hourly_combined = summarize_entry_hours(trade_rows_all)
    symbol_direction = summarize_symbols(trade_rows_all)
    weekday_patterns = summarize_pattern_axis(
        trade_rows_all,
        axis_name="weekday",
        sort_cols=["weekday_num"],
        display_cols=["weekday_num", "weekday"],
        bootstrap_iterations=args.bootstrap_iterations,
        significance_min_trades=args.significance_min_trades,
    )
    hour_patterns = summarize_pattern_axis(
        trade_rows_all,
        axis_name="entry_hour",
        sort_cols=["entry_hour"],
        display_cols=["entry_hour"],
        bootstrap_iterations=args.bootstrap_iterations,
        significance_min_trades=args.significance_min_trades,
    )
    wom_patterns = summarize_pattern_axis(
        trade_rows_all,
        axis_name="week_of_month",
        sort_cols=["week_of_month"],
        display_cols=["week_of_month"],
        bootstrap_iterations=args.bootstrap_iterations,
        significance_min_trades=args.significance_min_trades,
    )
    weekday_candidates = pattern_candidates(weekday_patterns, args.min_trades)
    hour_candidates = pattern_candidates(hour_patterns, args.min_trades)
    wom_candidates = pattern_candidates(wom_patterns, args.min_trades)

    high_by_weekday = dominant_hour_by_group(daily_extremes_all, ["weekday_num", "weekday"], "day_high_hour", "daily_high")
    low_by_weekday = dominant_hour_by_group(daily_extremes_all, ["weekday_num", "weekday"], "day_low_hour", "daily_low")
    high_by_wom = dominant_hour_by_group(daily_extremes_all, ["week_of_month"], "day_high_hour", "daily_high")
    low_by_wom = dominant_hour_by_group(daily_extremes_all, ["week_of_month"], "day_low_hour", "daily_low")

    print_section("Symbol Performance", summary_table)
    print_section("Combined Daily High Hour Distribution", high_dist_all)
    print_section("Combined Daily Low Hour Distribution", low_dist_all)
    print_section("Combined Entry Hour Stats", hourly_combined)
    print_section("Symbol Direction Summary", symbol_direction)
    print_section("High Hour Modes By Weekday", high_by_weekday.sort_values("weekday_num").drop(columns=["weekday_num"]))
    print_section("Low Hour Modes By Weekday", low_by_weekday.sort_values("weekday_num").drop(columns=["weekday_num"]))
    print_section("High Hour Modes By Week Of Month", high_by_wom.sort_values("week_of_month"))
    print_section("Low Hour Modes By Week Of Month", low_by_wom.sort_values("week_of_month"))
    print_section("Entry Hour Experiments", hour_patterns)
    print_section("Weekday Experiments", weekday_patterns.drop(columns=["weekday_num"]))
    print_section("Week Of Month Experiments", wom_patterns)
    print_section("Hour Underperformance Candidates", hour_candidates)
    print_section("Weekday Underperformance Candidates", weekday_candidates.drop(columns=["weekday_num"], errors="ignore"))
    print_section("Week Of Month Underperformance Candidates", wom_candidates)

    for symbol in args.symbols:
        symbol_extremes = daily_extremes_all[daily_extremes_all["symbol"] == symbol]
        symbol_trades = trade_rows_all[trade_rows_all["symbol"] == symbol]
        print_section(f"{symbol} Daily High Hours", distribution_by_hour(symbol_extremes, "day_high_hour", "daily_high"))
        print_section(f"{symbol} Daily Low Hours", distribution_by_hour(symbol_extremes, "day_low_hour", "daily_low"))
        print_section(f"{symbol} Entry Hour Stats", summarize_entry_hours(symbol_trades))

    save_dir = Path(args.save_dir) if args.save_dir else None
    write_optional_csvs(save_dir, [
        ("symbol_performance", summary_table),
        ("combined_daily_high_hours", high_dist_all),
        ("combined_daily_low_hours", low_dist_all),
        ("combined_entry_hour_stats", hourly_combined),
        ("symbol_direction_summary", symbol_direction),
        ("high_hour_modes_by_weekday", high_by_weekday),
        ("low_hour_modes_by_weekday", low_by_weekday),
        ("high_hour_modes_by_week_of_month", high_by_wom),
        ("low_hour_modes_by_week_of_month", low_by_wom),
        ("entry_hour_experiments", hour_patterns),
        ("weekday_experiments", weekday_patterns),
        ("week_of_month_experiments", wom_patterns),
        ("hour_underperformance_candidates", hour_candidates),
        ("weekday_underperformance_candidates", weekday_candidates),
        ("week_of_month_underperformance_candidates", wom_candidates),
        ("daily_extremes_raw", daily_extremes_all),
        ("trade_timing_raw", trade_rows_all),
    ])


if __name__ == "__main__":
    main()
