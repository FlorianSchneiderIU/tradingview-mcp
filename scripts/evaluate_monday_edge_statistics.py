#!/usr/bin/env python3
"""Comprehensive statistical checks for the BTC Monday-context edge study.

The script consumes the Monday RR-sweep outputs and produces:

1. walk-forward train/test selections,
2. weekday placebo results,
3. Monday label permutation p-values,
4. candidate multiple-testing stats with BH/FDR q-values,
5. yearly stability,
6. MFE/MAE summaries,
7. time-to-hit summaries,
8. non-overlapping execution summaries,
9. transaction-cost sensitivity,
10. interaction with the daily regime classifier.
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

from scripts.bybit_btc_monday_context_study import (  # noqa: E402
    first_touch_time,
    monday_week_start,
)
from scripts.bybit_btc_regime_hourly_histogram import read_ohlcv_csv  # noqa: E402


BASE_DIR = Path("scripts/output/bybit_regime_histogram")
SWEEP_DIR = BASE_DIR / "monday_rr_sweep"
CACHE_DIR = Path("scripts/.cache/bybit_regime_histogram")
STAMP = "btcusdt_20210525_20260525"
RR_FOLDERS = {
    0.5: SWEEP_DIR / "rr_0_5",
    1.0: BASE_DIR,
    1.5: SWEEP_DIR / "rr_1_5",
    2.0: SWEEP_DIR / "rr_2_0",
    3.0: SWEEP_DIR / "rr_3_0",
}
REQUESTED_CONDITIONS = [
    "monday_high_low_sequence",
    "monday_color",
    "monday_close_vs_pwh_pwl",
]
CONDITION_DISPLAY = {
    "monday_high_low_sequence": "Monday high/low sequence",
    "monday_color": "Monday color",
    "monday_close_vs_pwh_pwl": "Monday close vs PWH/PWL",
}
WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def rr_label(rr: float) -> str:
    return f"{rr:g}R"


def metric_summary(frame: pd.DataFrame, r_column: str = "r_multiple") -> dict[str, Any]:
    r_values = pd.to_numeric(frame[r_column], errors="coerce")
    trades = int(len(frame))
    wins = int((frame["outcome"] == "target").sum()) if "outcome" in frame.columns else int(np.nan)
    losses = int((frame["outcome"] == "stop").sum()) if "outcome" in frame.columns else int(np.nan)
    return {
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / trades if trades else np.nan,
        "avg_r": float(r_values.mean()) if r_values.notna().any() else np.nan,
        "median_r": float(r_values.median()) if r_values.notna().any() else np.nan,
        "total_r": float(r_values.sum(skipna=True)),
    }


def bh_qvalues(p_values: pd.Series) -> pd.Series:
    values = p_values.to_numpy(float)
    n = len(values)
    order = np.argsort(values)
    ranked = values[order]
    q = np.empty(n, dtype=float)
    running = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        running = min(running, ranked[i] * n / rank)
        q[order[i]] = running
    return pd.Series(np.clip(q, 0.0, 1.0), index=p_values.index)


def trades_path(rr: float) -> Path:
    return RR_FOLDERS[rr] / f"{STAMP}_monday_trades.csv"


def best_path(rr: float) -> Path:
    return RR_FOLDERS[rr] / f"{STAMP}_monday_best_hours.csv"


def load_trades() -> dict[float, pd.DataFrame]:
    out: dict[float, pd.DataFrame] = {}
    date_cols = ["entry_time", "exit_time", "week_start"]
    for rr, folder in RR_FOLDERS.items():
        path = trades_path(rr)
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, parse_dates=[col for col in date_cols if col])
        frame["rr"] = rr
        frame["rr_label"] = rr_label(rr)
        frame["entry_hour_utc"] = frame["entry_hour_utc"].astype(int)
        out[rr] = frame
    return out


def load_best_1r_setups() -> pd.DataFrame:
    frame = pd.read_csv(best_path(1.0))
    setups = frame[(frame["condition"].isin(REQUESTED_CONDITIONS)) & (frame["entry_mode"] == "close")].copy()
    return setups.sort_values(["condition", "condition_value"]).reset_index(drop=True)


def expand_conditions(trades_by_rr: dict[float, pd.DataFrame], *, close_only: bool = True) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    keep = [
        "entry_time",
        "exit_time",
        "week_start",
        "entry_mode",
        "direction",
        "entry_hour_utc",
        "outcome",
        "r_multiple",
        "holding_hours",
        "entry_price",
        "risk",
        "rr",
        "rr_label",
    ]
    for rr, frame in trades_by_rr.items():
        base = frame[frame["entry_mode"] == "close"].copy() if close_only else frame.copy()
        for condition in REQUESTED_CONDITIONS:
            chunk = base[keep].copy()
            chunk["condition"] = condition
            chunk["condition_value"] = base[condition].astype(str).to_numpy()
            chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


def candidate_weekly(expanded: pd.DataFrame) -> pd.DataFrame:
    frame = expanded.copy()
    frame["wins"] = (frame["outcome"] == "target").astype(int)
    weekly = (
        frame.groupby(
            ["week_start", "condition", "condition_value", "rr", "rr_label", "direction", "entry_hour_utc"],
            as_index=False,
        )
        .agg(trades=("r_multiple", "size"), wins=("wins", "sum"), r_sum=("r_multiple", "sum"))
        .sort_values("week_start")
    )
    return weekly


def aggregate_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    grouped = (
        frame.groupby(["condition", "condition_value", "rr", "rr_label", "direction", "entry_hour_utc"], as_index=False)
        .agg(trades=("trades", "sum"), wins=("wins", "sum"), r_sum=("r_sum", "sum"))
    )
    grouped["win_rate"] = grouped["wins"] / grouped["trades"]
    grouped["avg_r"] = grouped["r_sum"] / grouped["trades"]
    return grouped.sort_values(["avg_r", "trades"], ascending=[False, False]).reset_index(drop=True)


def candidate_mask(frame: pd.DataFrame, row: pd.Series) -> pd.Series:
    return (
        (frame["condition"] == row["condition"])
        & (frame["condition_value"].astype(str) == str(row["condition_value"]))
        & (frame["rr"] == float(row["rr"]))
        & (frame["direction"] == row["direction"])
        & (frame["entry_hour_utc"] == int(row["entry_hour_utc"]))
    )


def walk_forward(weekly: pd.DataFrame, *, train_weeks: int, test_weeks: int, min_train_trades: int) -> pd.DataFrame:
    weeks = np.array(sorted(weekly["week_start"].dropna().unique()))
    rows: list[dict[str, Any]] = []
    fold = 0
    start = 0
    while start + train_weeks + test_weeks <= len(weeks):
        train_set = set(weeks[start : start + train_weeks])
        test_set = set(weeks[start + train_weeks : start + train_weeks + test_weeks])
        train = weekly[weekly["week_start"].isin(train_set)]
        test = weekly[weekly["week_start"].isin(test_set)]
        ranked = aggregate_candidates(train)
        ranked = ranked[ranked["trades"] >= min_train_trades]
        if ranked.empty:
            start += test_weeks
            continue
        selected = ranked.iloc[0]
        selected_test = test[candidate_mask(test, selected)]
        test_metrics = aggregate_candidates(selected_test)
        row: dict[str, Any] = {
            "fold": fold,
            "train_start": weeks[start],
            "train_end": weeks[start + train_weeks - 1],
            "test_start": weeks[start + train_weeks],
            "test_end": weeks[start + train_weeks + test_weeks - 1],
            "selected_condition": selected["condition"],
            "selected_value": selected["condition_value"],
            "selected_rr": selected["rr"],
            "selected_rr_label": selected["rr_label"],
            "selected_direction": selected["direction"],
            "selected_hour_utc": int(selected["entry_hour_utc"]),
            "train_trades": int(selected["trades"]),
            "train_win_rate": float(selected["win_rate"]),
            "train_avg_r": float(selected["avg_r"]),
        }
        if test_metrics.empty:
            row.update({"test_trades": 0, "test_wins": 0, "test_win_rate": np.nan, "test_avg_r": np.nan, "test_total_r": 0.0})
        else:
            tm = test_metrics.iloc[0]
            row.update(
                {
                    "test_trades": int(tm["trades"]),
                    "test_wins": int(tm["wins"]),
                    "test_win_rate": float(tm["win_rate"]),
                    "test_avg_r": float(tm["avg_r"]),
                    "test_total_r": float(tm["r_sum"]),
                }
            )
        rows.append(row)
        fold += 1
        start += test_weeks
    return pd.DataFrame(rows)


def classify_close_vs_previous_week(close: float, previous_high: float, previous_low: float) -> str:
    if not np.isfinite(previous_high) or not np.isfinite(previous_low):
        return "no_previous_week"
    if close > previous_high:
        return "above_pwh"
    if close < previous_low:
        return "below_pwl"
    return "inside_previous_week_range"


def build_weekday_context(daily: pd.DataFrame, hourly: pd.DataFrame, weekday: int) -> pd.DataFrame:
    daily_frame = daily.sort_values("open_time").reset_index(drop=True).copy()
    hourly_frame = hourly.sort_values("open_time").reset_index(drop=True).copy()
    daily_frame["week_start"] = monday_week_start(daily_frame["open_time"])
    weekly = (
        daily_frame.groupby("week_start", as_index=False)
        .agg(week_high=("high", "max"), week_low=("low", "min"))
        .sort_values("week_start")
    )
    previous_week = weekly.copy()
    previous_week["week_start"] = previous_week["week_start"] + pd.Timedelta(days=7)
    previous_week = previous_week.rename(columns={"week_high": "previous_week_high", "week_low": "previous_week_low"})

    contexts = daily_frame[daily_frame["open_time"].dt.weekday == weekday].merge(previous_week, on="week_start", how="left")
    rows: list[dict[str, Any]] = []
    for row in contexts.itertuples(index=False):
        open_time = pd.Timestamp(row.open_time)
        close_time = open_time + pd.Timedelta(days=1)
        expires = row.week_start + pd.Timedelta(days=7)
        if close_time >= expires:
            continue
        day_hours = hourly_frame[(hourly_frame["open_time"] >= open_time) & (hourly_frame["open_time"] < close_time)].copy()
        if day_hours.empty:
            continue
        high_time = first_touch_time(day_hours, "high", float(row.high))
        low_time = first_touch_time(day_hours, "low", float(row.low))
        sequence = "same_hour"
        if high_time < low_time:
            sequence = "high_first"
        elif low_time < high_time:
            sequence = "low_first"
        color = "doji"
        if float(row.close) > float(row.open):
            color = "green"
        elif float(row.close) < float(row.open):
            color = "red"
        previous_high = float(row.previous_week_high) if pd.notna(row.previous_week_high) else np.nan
        previous_low = float(row.previous_week_low) if pd.notna(row.previous_week_low) else np.nan
        close_vs = classify_close_vs_previous_week(float(row.close), previous_high, previous_low)
        rows.append(
            {
                "context_weekday": weekday,
                "context_weekday_name": WEEKDAY_NAMES[weekday],
                "week_start": row.week_start,
                "context_available_time": close_time,
                "context_expires_time": expires,
                "context_high_low_sequence": sequence,
                "context_color": color,
                "context_close_vs_pwh_pwl": close_vs,
            }
        )
    return pd.DataFrame(rows).sort_values("context_available_time").reset_index(drop=True)


def assign_context_to_trades(trades: pd.DataFrame, contexts: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge_asof(
        trades.sort_values("entry_time"),
        contexts.sort_values("context_available_time"),
        left_on="entry_time",
        right_on="context_available_time",
        direction="backward",
    )
    merged = merged[(merged["entry_time"] >= merged["context_available_time"]) & (merged["entry_time"] < merged["context_expires_time"])]
    return merged.reset_index(drop=True)


def weekday_placebo(trades_by_rr: dict[float, pd.DataFrame], daily: pd.DataFrame, hourly: pd.DataFrame, min_trades: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    trade_cols = [
        "entry_time",
        "exit_time",
        "entry_mode",
        "direction",
        "entry_hour_utc",
        "outcome",
        "r_multiple",
        "holding_hours",
        "entry_price",
        "risk",
        "rr",
        "rr_label",
    ]
    all_trades = pd.concat(
        [frame.loc[frame["entry_mode"] == "close", trade_cols].copy() for frame in trades_by_rr.values()],
        ignore_index=True,
    )
    for weekday in range(5):
        contexts = build_weekday_context(daily, hourly, weekday)
        tagged = assign_context_to_trades(all_trades, contexts)
        if tagged.empty:
            continue
        chunks = []
        mapping = {
            "context_high_low_sequence": "high_low_sequence",
            "context_color": "color",
            "context_close_vs_pwh_pwl": "close_vs_pwh_pwl",
        }
        for source, name in mapping.items():
            chunk = tagged.copy()
            chunk["condition"] = name
            chunk["condition_value"] = chunk[source].astype(str)
            chunks.append(chunk)
        expanded = pd.concat(chunks, ignore_index=True)
        expanded["wins"] = (expanded["outcome"] == "target").astype(int)
        grouped = (
            expanded.groupby(["context_weekday_name", "condition", "condition_value", "rr", "rr_label", "direction", "entry_hour_utc"], as_index=False)
            .agg(trades=("r_multiple", "size"), wins=("wins", "sum"), r_sum=("r_multiple", "sum"))
        )
        grouped = grouped[grouped["trades"] >= min_trades].copy()
        grouped["win_rate"] = grouped["wins"] / grouped["trades"]
        grouped["avg_r"] = grouped["r_sum"] / grouped["trades"]
        best = grouped.sort_values(["avg_r", "trades"], ascending=[False, False]).groupby(["context_weekday_name", "condition"], as_index=False).head(1)
        rows.append(best)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def label_permutation(weekly: pd.DataFrame, *, min_trades: int, n_perm: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for condition in REQUESTED_CONDITIONS:
        cond = weekly[weekly["condition"] == condition].copy()
        actual = aggregate_candidates(cond)
        actual = actual[actual["trades"] >= min_trades]
        actual_max = float(actual["avg_r"].max()) if not actual.empty else np.nan
        week_labels = cond[["week_start", "condition_value"]].drop_duplicates().sort_values("week_start")
        weeks = week_labels["week_start"].to_numpy()
        labels = week_labels["condition_value"].astype(str).to_numpy()
        base = cond.drop(columns=["condition_value"]).copy()
        maxima = np.empty(n_perm, dtype=float)
        for i in range(n_perm):
            shuffled = pd.DataFrame({"week_start": weeks, "condition_value": rng.permutation(labels)})
            perm = base.merge(shuffled, on="week_start", how="left")
            ranked = aggregate_candidates(perm)
            ranked = ranked[ranked["trades"] >= min_trades]
            maxima[i] = ranked["avg_r"].max() if not ranked.empty else np.nan
        valid = maxima[np.isfinite(maxima)]
        rows.append(
            {
                "condition": condition,
                "actual_best_avg_r": actual_max,
                "null_mean_best_avg_r": float(np.nanmean(valid)) if valid.size else np.nan,
                "null_p95_best_avg_r": float(np.nanquantile(valid, 0.95)) if valid.size else np.nan,
                "p_value": float(np.mean(valid >= actual_max)) if valid.size and np.isfinite(actual_max) else np.nan,
                "permutations": int(valid.size),
            }
        )
    return pd.DataFrame(rows)


def multiple_testing_stats(weekly: pd.DataFrame, *, min_trades: int, n_perm: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    candidates = aggregate_candidates(weekly)
    candidates = candidates[candidates["trades"] >= min_trades].copy().reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for idx, row in candidates.iterrows():
        chunk = weekly[candidate_mask(weekly, row)].copy()
        week_agg = chunk.groupby("week_start", as_index=False).agg(trades=("trades", "sum"), r_sum=("r_sum", "sum"))
        trades_arr = week_agg["trades"].to_numpy(float)
        r_arr = week_agg["r_sum"].to_numpy(float)
        observed = float(r_arr.sum() / trades_arr.sum()) if trades_arr.sum() else np.nan
        if len(r_arr) == 0 or not np.isfinite(observed):
            p_value = np.nan
            ci_low = np.nan
            ci_high = np.nan
        else:
            signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, len(r_arr)), replace=True)
            null = (signs * r_arr).sum(axis=1) / trades_arr.sum()
            p_value = float((np.sum(null >= observed) + 1) / (n_perm + 1))
            boot_idx = rng.integers(0, len(r_arr), size=(n_perm, len(r_arr)))
            boot = r_arr[boot_idx].sum(axis=1) / trades_arr[boot_idx].sum(axis=1)
            ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
        out = row.to_dict()
        out.update({"p_value_signflip": p_value, "avg_r_ci_low": ci_low, "avg_r_ci_high": ci_high, "weeks": int(len(r_arr))})
        rows.append(out)
    stats = pd.DataFrame(rows)
    stats["q_value_bh"] = bh_qvalues(stats["p_value_signflip"].fillna(1.0))
    return stats.sort_values(["q_value_bh", "avg_r"], ascending=[True, False]).reset_index(drop=True)


def filter_fixed_setup(frame: pd.DataFrame, setup: pd.Series, rr: float | None = None) -> pd.DataFrame:
    out = frame[
        (frame[str(setup["condition"])].astype(str) == str(setup["condition_value"]))
        & (frame["entry_mode"] == str(setup["entry_mode"]))
        & (frame["direction"] == str(setup["direction"]))
        & (frame["entry_hour_utc"] == int(setup["entry_hour_utc"]))
    ].copy()
    if rr is not None:
        out = out[out["rr"] == rr].copy()
    return out


def yearly_stability(trades_by_rr: dict[float, pd.DataFrame], setups: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, setup in setups.iterrows():
        for rr, frame in trades_by_rr.items():
            chunk = filter_fixed_setup(frame, setup, rr=rr)
            if chunk.empty:
                continue
            chunk["year"] = chunk["entry_time"].dt.year
            for year, group in chunk.groupby("year"):
                row = {
                    "condition": setup["condition"],
                    "condition_value": setup["condition_value"],
                    "direction": setup["direction"],
                    "hour_utc": int(setup["entry_hour_utc"]),
                    "rr": rr,
                    "rr_label": rr_label(rr),
                    "year": int(year),
                }
                row.update(metric_summary(group))
                rows.append(row)
    return pd.DataFrame(rows)


def compute_mfe_mae(trades: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    frame = trades.sort_values("entry_time").copy()
    h = hourly.sort_values("open_time").reset_index(drop=True).copy()
    times = (
        pd.to_datetime(h["open_time"], utc=True)
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .astype("datetime64[ns]")
        .astype("int64")
        .to_numpy()
    )
    highs = h["high"].to_numpy(float)
    lows = h["low"].to_numpy(float)
    mfe_values: list[float] = []
    mae_values: list[float] = []
    for row in frame.itertuples(index=False):
        entry_time = pd.Timestamp(row.entry_time).tz_convert("UTC").value
        exit_time = pd.Timestamp(row.exit_time).tz_convert("UTC").value
        start = int(np.searchsorted(times, entry_time, side="left"))
        end = int(np.searchsorted(times, exit_time, side="left"))
        if end <= start:
            end = min(start + 1, len(times))
        if start >= len(times) or end <= start or float(row.risk) <= 0:
            mfe_values.append(np.nan)
            mae_values.append(np.nan)
            continue
        high_slice = highs[start:end]
        low_slice = lows[start:end]
        if row.direction == "long":
            mfe = (np.max(high_slice) - float(row.entry_price)) / float(row.risk)
            mae = (float(row.entry_price) - np.min(low_slice)) / float(row.risk)
        else:
            mfe = (float(row.entry_price) - np.min(low_slice)) / float(row.risk)
            mae = (np.max(high_slice) - float(row.entry_price)) / float(row.risk)
        mfe_values.append(float(mfe))
        mae_values.append(float(mae))
    frame["mfe_r"] = mfe_values
    frame["mae_r"] = mae_values
    return frame


def mfe_mae_summary(trades_by_rr: dict[float, pd.DataFrame], setups: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, setup in setups.iterrows():
        for rr, frame in trades_by_rr.items():
            chunk = filter_fixed_setup(frame, setup, rr=rr)
            enriched = compute_mfe_mae(chunk, hourly)
            if enriched.empty:
                continue
            row = {
                "condition": setup["condition"],
                "condition_value": setup["condition_value"],
                "direction": setup["direction"],
                "hour_utc": int(setup["entry_hour_utc"]),
                "rr": rr,
                "rr_label": rr_label(rr),
                "trades": len(enriched),
                "mfe_median": float(enriched["mfe_r"].median()),
                "mfe_p75": float(enriched["mfe_r"].quantile(0.75)),
                "mfe_p90": float(enriched["mfe_r"].quantile(0.90)),
                "mae_median": float(enriched["mae_r"].median()),
                "mae_p75": float(enriched["mae_r"].quantile(0.75)),
                "mae_p90": float(enriched["mae_r"].quantile(0.90)),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def time_to_hit_summary(trades_by_rr: dict[float, pd.DataFrame], setups: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, setup in setups.iterrows():
        for rr, frame in trades_by_rr.items():
            chunk = filter_fixed_setup(frame, setup, rr=rr)
            if chunk.empty:
                continue
            winners = chunk[chunk["outcome"] == "target"]
            losers = chunk[chunk["outcome"] == "stop"]
            row = {
                "condition": setup["condition"],
                "condition_value": setup["condition_value"],
                "direction": setup["direction"],
                "hour_utc": int(setup["entry_hour_utc"]),
                "rr": rr,
                "rr_label": rr_label(rr),
                "trades": len(chunk),
                "median_hold_hours": float(chunk["holding_hours"].median()),
                "p75_hold_hours": float(chunk["holding_hours"].quantile(0.75)),
                "winner_median_hours": float(winners["holding_hours"].median()) if not winners.empty else np.nan,
                "loser_median_hours": float(losers["holding_hours"].median()) if not losers.empty else np.nan,
                "target_within_6h": float(((chunk["outcome"] == "target") & (chunk["holding_hours"] <= 6)).mean()),
                "target_within_12h": float(((chunk["outcome"] == "target") & (chunk["holding_hours"] <= 12)).mean()),
                "target_within_24h": float(((chunk["outcome"] == "target") & (chunk["holding_hours"] <= 24)).mean()),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def nonoverlap_summary(trades_by_rr: dict[float, pd.DataFrame], setups: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, setup in setups.iterrows():
        for rr, frame in trades_by_rr.items():
            chunk = filter_fixed_setup(frame, setup, rr=rr).sort_values("entry_time")
            variants = {"all_independent": chunk}
            if not chunk.empty:
                accepted = []
                last_exit = pd.Timestamp.min.tz_localize("UTC")
                for row in chunk.itertuples(index=False):
                    if pd.Timestamp(row.entry_time) >= last_exit:
                        accepted.append(row._asdict())
                        last_exit = pd.Timestamp(row.exit_time)
                variants["one_active_trade"] = pd.DataFrame(accepted)
                variants["first_per_week"] = chunk.sort_values("entry_time").groupby("week_start", as_index=False).head(1)
            for variant, data in variants.items():
                row = {
                    "condition": setup["condition"],
                    "condition_value": setup["condition_value"],
                    "direction": setup["direction"],
                    "hour_utc": int(setup["entry_hour_utc"]),
                    "rr": rr,
                    "rr_label": rr_label(rr),
                    "execution_variant": variant,
                }
                row.update(metric_summary(data))
                rows.append(row)
    return pd.DataFrame(rows)


def cost_sensitivity(trades_by_rr: dict[float, pd.DataFrame], setups: pd.DataFrame, cost_bps_values: list[float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, setup in setups.iterrows():
        for rr, frame in trades_by_rr.items():
            chunk = filter_fixed_setup(frame, setup, rr=rr).copy()
            if chunk.empty:
                continue
            for cost_bps in cost_bps_values:
                adjusted = chunk.copy()
                adjusted["r_after_cost"] = adjusted["r_multiple"] - (adjusted["entry_price"] * (cost_bps / 10_000.0) / adjusted["risk"])
                row = {
                    "condition": setup["condition"],
                    "condition_value": setup["condition_value"],
                    "direction": setup["direction"],
                    "hour_utc": int(setup["entry_hour_utc"]),
                    "rr": rr,
                    "rr_label": rr_label(rr),
                    "round_turn_cost_bps": cost_bps,
                }
                row.update(metric_summary(adjusted, r_column="r_after_cost"))
                rows.append(row)
    return pd.DataFrame(rows)


def daily_regime_interaction(trades_1r: pd.DataFrame, setups: pd.DataFrame, daily_regimes: pd.DataFrame) -> pd.DataFrame:
    state = daily_regimes[["regime_available_time", "regime"]].copy()
    state["regime_available_time"] = pd.to_datetime(state["regime_available_time"], utc=True)
    rows: list[dict[str, Any]] = []
    for _, setup in setups.iterrows():
        chunk = filter_fixed_setup(trades_1r, setup, rr=1.0)
        if chunk.empty:
            continue
        tagged = pd.merge_asof(
            chunk.sort_values("entry_time"),
            state.sort_values("regime_available_time"),
            left_on="entry_time",
            right_on="regime_available_time",
            direction="backward",
        ).dropna(subset=["regime"])
        for regime, group in tagged.groupby("regime"):
            row = {
                "condition": setup["condition"],
                "condition_value": setup["condition_value"],
                "direction": setup["direction"],
                "hour_utc": int(setup["entry_hour_utc"]),
                "regime": regime,
            }
            row.update(metric_summary(group))
            rows.append(row)
    return pd.DataFrame(rows)


def render_html_report(output_dir: Path, tables: dict[str, pd.DataFrame]) -> Path:
    def table_html(frame: pd.DataFrame, limit: int = 12) -> str:
        if frame.empty:
            return "<p>No rows.</p>"
        return frame.head(limit).to_html(index=False, escape=True, float_format=lambda x: f"{x:.4f}")

    report_path = output_dir / f"{STAMP}_statistical_evaluation_report.html"
    sections = []
    for title, frame in tables.items():
        sections.append(f"<section><h2>{html.escape(title)}</h2>{table_html(frame)}</section>")
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{STAMP} statistical evaluation</title>
  <style>
    body {{ font-family: Inter, ui-sans-serif, system-ui, sans-serif; margin: 32px; background: #f8fafc; color: #0f172a; }}
    main {{ max-width: 1280px; margin: 0 auto; }}
    h1 {{ margin-bottom: 4px; }}
    p {{ color: #475569; }}
    section {{ margin: 28px 0; padding: 18px; background: white; border: 1px solid #dbe3ee; border-radius: 10px; overflow-x: auto; }}
    h2 {{ margin-top: 0; font-size: 18px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ border-bottom: 1px solid #e2e8f0; padding: 7px 8px; text-align: left; white-space: nowrap; }}
    th {{ background: #f1f5f9; }}
  </style>
</head>
<body>
<main>
  <h1>{STAMP.upper()} Statistical Evaluation</h1>
  <p>Close-entry Monday context candidates, RR sweep, weekly/statistical robustness checks.</p>
  {''.join(sections)}
</main>
</body>
</html>
"""
    report_path.write_text(document, encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all statistical evaluations for the Monday-context BTC study.")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "monday_stat_eval")
    parser.add_argument("--min-trades", type=int, default=100)
    parser.add_argument("--walk-train-weeks", type=int, default=104)
    parser.add_argument("--walk-test-weeks", type=int, default=26)
    parser.add_argument("--n-perm", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading trade outputs...")
    trades_by_rr = load_trades()
    setups = load_best_1r_setups()
    expanded = expand_conditions(trades_by_rr, close_only=True)
    weekly = candidate_weekly(expanded)

    print("Running walk-forward validation...")
    walk = walk_forward(
        weekly,
        train_weeks=args.walk_train_weeks,
        test_weeks=args.walk_test_weeks,
        min_train_trades=args.min_trades,
    )

    print("Running weekday placebo checks...")
    daily_raw = read_ohlcv_csv(CACHE_DIR / "btcusdt_1d_202105040700_202605250700.csv")
    hourly_raw = read_ohlcv_csv(CACHE_DIR / "btcusdt_1h_202105142100_202605250700.csv")
    placebos = weekday_placebo(trades_by_rr, daily_raw, hourly_raw, args.min_trades)

    print("Running label permutation checks...")
    permutation = label_permutation(weekly, min_trades=args.min_trades, n_perm=args.n_perm, seed=args.seed)

    print("Running multiple-testing correction...")
    mt_stats = multiple_testing_stats(weekly, min_trades=args.min_trades, n_perm=args.n_perm, seed=args.seed + 1)

    print("Computing stability and execution diagnostics...")
    yearly = yearly_stability(trades_by_rr, setups)
    mfe_mae = mfe_mae_summary(trades_by_rr, setups, hourly_raw)
    time_to_hit = time_to_hit_summary(trades_by_rr, setups)
    nonoverlap = nonoverlap_summary(trades_by_rr, setups)
    costs = cost_sensitivity(trades_by_rr, setups, [0.0, 4.0, 8.0, 12.0, 20.0])

    print("Computing daily-regime interaction...")
    daily_regimes = pd.read_csv(BASE_DIR / f"{STAMP}_daily_regimes.csv", parse_dates=["regime_available_time"])
    regime = daily_regime_interaction(trades_by_rr[1.0], setups, daily_regimes)

    outputs = {
        "walk_forward": walk,
        "weekday_placebo": placebos,
        "label_permutation": permutation,
        "multiple_testing": mt_stats,
        "yearly_stability": yearly,
        "mfe_mae": mfe_mae,
        "time_to_hit": time_to_hit,
        "nonoverlap": nonoverlap,
        "cost_sensitivity": costs,
        "daily_regime_interaction": regime,
    }
    paths = []
    for name, frame in outputs.items():
        path = args.output_dir / f"{STAMP}_{name}.csv"
        frame.to_csv(path, index=False)
        paths.append(path)

    report = render_html_report(
        args.output_dir,
        {
            "Walk-Forward Folds": walk,
            "Weekday Placebo Best Cells": placebos.sort_values(["condition", "avg_r"], ascending=[True, False]) if not placebos.empty else placebos,
            "Label Permutation": permutation,
            "Multiple Testing Top Candidates": mt_stats,
            "Yearly Stability": yearly.sort_values(["condition", "condition_value", "rr", "year"]) if not yearly.empty else yearly,
            "MFE/MAE": mfe_mae,
            "Time To Hit": time_to_hit,
            "Non-Overlap": nonoverlap,
            "Cost Sensitivity": costs,
            "Daily Regime Interaction": regime.sort_values(["avg_r"], ascending=False) if not regime.empty else regime,
        },
    )

    print("\nGenerated evaluation files:")
    for path in paths:
        print(f"  {path}")
    print(f"  {report}")

    if not walk.empty:
        print("\nWalk-forward aggregate:")
        print(
            pd.DataFrame(
                [
                    {
                        "folds": len(walk),
                        "test_trades": int(walk["test_trades"].sum()),
                        "test_win_rate": float(walk["test_wins"].sum() / walk["test_trades"].sum()),
                        "test_avg_r": float(walk["test_total_r"].sum() / walk["test_trades"].sum()),
                        "test_total_r": float(walk["test_total_r"].sum()),
                    }
                ]
            ).to_string(index=False, float_format=lambda x: f"{x:.4f}")
        )

    print("\nPermutation p-values:")
    print(permutation.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nMultiple-testing top rows:")
    cols = ["condition", "condition_value", "rr_label", "direction", "entry_hour_utc", "trades", "win_rate", "avg_r", "p_value_signflip", "q_value_bh"]
    print(mt_stats[cols].head(12).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
