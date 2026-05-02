from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def profit_factor(rs: pd.Series) -> float:
    wins = rs[rs > 0]
    losses = rs[rs <= 0]
    gross_loss = abs(float(losses.sum()))
    if gross_loss == 0:
        return float("inf") if len(wins) else 0.0
    return float(wins.sum()) / gross_loss


def max_drawdown(values: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for value in values:
        equity += float(value)
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)
    return max_dd


def metrics(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    if frame.empty:
        return {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "net_r": 0.0, "avg_r": 0.0, "max_dd_r": 0.0}
    ordered = frame.sort_values("exit_time")
    rs = ordered[column].astype(float)
    return {
        "trades": int(len(ordered)),
        "win_rate": round(100.0 * float((rs > 0).mean()), 2),
        "profit_factor": round(profit_factor(rs), 3),
        "net_r": round(float(rs.sum()), 3),
        "avg_r": round(float(rs.mean()), 4),
        "max_dd_r": round(max_drawdown(rs.to_list()), 3),
    }


def parse_timestamp(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(pd.NaT, index=frame.index)
    return pd.to_datetime(frame[column], utc=True, errors="coerce")


def qbucket_from_train(frame: pd.DataFrame, column: str, train_mask: pd.Series, buckets: int = 4) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    train = values[train_mask & values.notna()]
    if train.nunique() < buckets:
        return pd.Series(["all"] * len(frame), index=frame.index)
    probs = [i / buckets for i in range(1, buckets)]
    cuts = sorted(set(float(train.quantile(prob)) for prob in probs))
    if not cuts:
        return pd.Series(["all"] * len(frame), index=frame.index)
    labels = [f"q{i+1}" for i in range(len(cuts) + 1)]
    return pd.cut(values, bins=[-np.inf, *cuts, np.inf], labels=labels).astype(str).fillna("nan")


def enrich(frame: pd.DataFrame, fee_bps_side: float) -> pd.DataFrame:
    out = frame.copy()
    for column in ["entry_time", "exit_time", "sweep_time", "choch_time", "signal_time", "break_time", "retest_time", "confirm_time", "zone_time"]:
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], utc=True, errors="coerce")

    risk = (out["entry_price"].astype(float) - out["stop_price"].astype(float)).abs()
    out["risk_abs"] = risk
    out["risk_pct"] = risk / out["entry_price"].astype(float) * 100.0
    out["target_rr_actual"] = (out["target_price"].astype(float) - out["entry_price"].astype(float)).abs() / risk
    out["zone_width_pct"] = (out["zone_top"].astype(float) - out["zone_bottom"].astype(float)).abs() / out["entry_price"].astype(float) * 100.0
    out["zone_width_to_risk"] = (out["zone_top"].astype(float) - out["zone_bottom"].astype(float)).abs() / risk

    if {"sweep_time", "choch_time"}.issubset(out.columns):
        out["sweep_to_choch_hours"] = (out["choch_time"] - out["sweep_time"]).dt.total_seconds() / 3600.0
    if {"choch_time", "signal_time"}.issubset(out.columns):
        out["choch_to_signal_hours"] = (out["signal_time"] - out["choch_time"]).dt.total_seconds() / 3600.0
    if {"break_time", "retest_time"}.issubset(out.columns):
        out["retest_delay_hours"] = (out["retest_time"] - out["break_time"]).dt.total_seconds() / 3600.0
    if {"retest_time", "confirm_time"}.issubset(out.columns):
        out["confirm_delay_hours"] = (out["confirm_time"] - out["retest_time"]).dt.total_seconds() / 3600.0
    if {"break_time", "zone_time"}.issubset(out.columns):
        out["zone_age_hours"] = (out["break_time"] - out["zone_time"]).dt.total_seconds() / 3600.0

    entry = pd.to_datetime(out["entry_time"], utc=True)
    out["entry_hour"] = entry.dt.hour
    out["entry_dow"] = entry.dt.dayofweek
    out["hour_bucket"] = (out["entry_hour"] // 4 * 4).astype(str) + "-" + (out["entry_hour"] // 4 * 4 + 3).astype(str)
    out["symbol_direction"] = out["symbol"].astype(str) + "_" + out["direction"].astype(str)

    cost_r = (out["entry_price"].abs() + out["exit_price"].abs()) * fee_bps_side / 10000.0 / risk
    out["cost_r"] = cost_r
    out["r_net_cost"] = out["r_multiple"].astype(float) - cost_r
    return out


def window_frames(frame: pd.DataFrame, train_end: pd.Timestamp, val_end: pd.Timestamp, test_end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    return {
        "train": frame[frame["entry_time"] < train_end],
        "validation": frame[(frame["entry_time"] >= train_end) & (frame["entry_time"] < val_end)],
        "oos": frame[(frame["entry_time"] >= val_end) & (frame["entry_time"] < test_end)],
    }


def candidate_rows(
    frame: pd.DataFrame,
    columns: list[str],
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
    test_end: pd.Timestamp,
    min_train: int,
    min_val: int,
    min_oos: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column in columns:
        if column not in frame.columns:
            continue
        for value, group in frame.groupby(column, dropna=False):
            windows = window_frames(group, train_end, val_end, test_end)
            if len(windows["train"]) < min_train or len(windows["validation"]) < min_val or len(windows["oos"]) < min_oos:
                continue
            row: dict[str, Any] = {"filter": column, "value": str(value)}
            for name, window in windows.items():
                row.update({f"{name}_{key}": val for key, val in metrics(window, "r_net_cost").items()})
            row["robust_score"] = robust_score(row)
            rows.append(row)
    return rows


def combo_rows(
    frame: pd.DataFrame,
    columns: list[str],
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
    test_end: pd.Timestamp,
    min_train: int,
    min_val: int,
    min_oos: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    present = [column for column in columns if column in frame.columns]
    for i, left in enumerate(present):
        for right in present[i + 1:]:
            for values, group in frame.groupby([left, right], dropna=False):
                windows = window_frames(group, train_end, val_end, test_end)
                if len(windows["train"]) < min_train or len(windows["validation"]) < min_val or len(windows["oos"]) < min_oos:
                    continue
                row: dict[str, Any] = {"filter": f"{left}+{right}", "value": "|".join(str(v) for v in values)}
                for name, window in windows.items():
                    row.update({f"{name}_{key}": val for key, val in metrics(window, "r_net_cost").items()})
                row["robust_score"] = robust_score(row)
                rows.append(row)
    return rows


def robust_score(row: dict[str, Any]) -> float:
    train = float(row.get("train_net_r", 0.0))
    val = float(row.get("validation_net_r", 0.0))
    oos = float(row.get("oos_net_r", 0.0))
    train_pf = float(row.get("train_profit_factor", 0.0))
    val_pf = float(row.get("validation_profit_factor", 0.0))
    oos_pf = float(row.get("oos_profit_factor", 0.0))
    if not math.isfinite(train_pf):
        train_pf = 5.0
    if not math.isfinite(val_pf):
        val_pf = 5.0
    if not math.isfinite(oos_pf):
        oos_pf = 5.0
    consistency_penalty = 0.0
    for value in [train, val, oos]:
        if value < 0:
            consistency_penalty += abs(value) * 1.5
    return round(train * 0.25 + val * 0.35 + oos * 0.40 + (train_pf + val_pf + oos_pf - 3.0) * 3.0 - consistency_penalty, 3)


def winner_loser_rows(frame: pd.DataFrame, columns: list[str], split: pd.Timestamp) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for window_name, window in [("pre_oos", frame[frame["entry_time"] < split]), ("oos", frame[frame["entry_time"] >= split])]:
        if window.empty:
            continue
        wins = window[window["r_net_cost"] > 0]
        losses = window[window["r_net_cost"] <= 0]
        for column in columns:
            if column not in frame.columns:
                continue
            rows.append({
                "window": window_name,
                "feature": column,
                "win_median": round(float(pd.to_numeric(wins[column], errors="coerce").median()), 6) if not wins.empty else math.nan,
                "loss_median": round(float(pd.to_numeric(losses[column], errors="coerce").median()), 6) if not losses.empty else math.nan,
                "abs_delta": round(abs(float(pd.to_numeric(wins[column], errors="coerce").median()) - float(pd.to_numeric(losses[column], errors="coerce").median())), 6)
                if not wins.empty and not losses.empty else math.nan,
            })
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep entry-edge diagnostics for trade CSVs.")
    parser.add_argument("file", type=Path)
    parser.add_argument("--label", default=None)
    parser.add_argument("--fee-bps-side", type=float, default=5.0)
    parser.add_argument("--train-end", default="2024-04-20")
    parser.add_argument("--validation-end", default="2025-04-20")
    parser.add_argument("--test-end", default="2026-04-20")
    parser.add_argument("--min-train", type=int, default=30)
    parser.add_argument("--min-validation", type=int, default=15)
    parser.add_argument("--min-oos", type=int, default=15)
    parser.add_argument("--out-prefix", type=Path, default=Path("scripts/deep_trade_edge_study"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label = args.label or args.file.stem
    train_end = pd.Timestamp(args.train_end, tz="UTC")
    val_end = pd.Timestamp(args.validation_end, tz="UTC")
    test_end = pd.Timestamp(args.test_end, tz="UTC")
    frame = enrich(pd.read_csv(args.file), args.fee_bps_side)
    train_mask = frame["entry_time"] < train_end

    numeric_bucket_sources = [
        "risk_pct",
        "zone_width_pct",
        "zone_width_to_risk",
        "zone_hold_prob",
        "sweep_to_choch_hours",
        "choch_to_signal_hours",
        "retest_delay_hours",
        "confirm_delay_hours",
        "confirm_fvg_atr",
        "retest_reject_pos",
        "zone_age_hours",
    ]
    for column in numeric_bucket_sources:
        if column in frame.columns:
            frame[f"{column}_q"] = qbucket_from_train(frame, column, train_mask)

    filters = [
        "symbol",
        "direction",
        "symbol_direction",
        "entry_mode",
        "zone_tf",
        "hour_bucket",
        "entry_dow",
        *[f"{column}_q" for column in numeric_bucket_sources if f"{column}_q" in frame.columns],
    ]
    summaries = []
    for name, window in window_frames(frame, train_end, val_end, test_end).items():
        summaries.append({"label": label, "window": name, **metrics(window, "r_multiple")})
        summaries.append({"label": label, "window": f"{name}_net_{args.fee_bps_side:g}bps", **metrics(window, "r_net_cost")})
    summary = pd.DataFrame(summaries)

    candidates = pd.DataFrame(candidate_rows(frame, filters, train_end, val_end, test_end, args.min_train, args.min_validation, args.min_oos))
    combos = pd.DataFrame(combo_rows(frame, filters, train_end, val_end, test_end, args.min_train, args.min_validation, args.min_oos))
    if not candidates.empty:
        candidates = candidates.sort_values(["robust_score", "oos_net_r"], ascending=False)
    if not combos.empty:
        combos = combos.sort_values(["robust_score", "oos_net_r"], ascending=False)

    feature_rows = pd.DataFrame(winner_loser_rows(frame, numeric_bucket_sources, val_end))
    if not feature_rows.empty:
        feature_rows = feature_rows.sort_values(["window", "abs_delta"], ascending=[True, False])

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_prefix.with_name(f"{args.out_prefix.name}_{label}_summary.csv")
    filters_path = args.out_prefix.with_name(f"{args.out_prefix.name}_{label}_filters.csv")
    combos_path = args.out_prefix.with_name(f"{args.out_prefix.name}_{label}_combos.csv")
    features_path = args.out_prefix.with_name(f"{args.out_prefix.name}_{label}_winner_loser_features.csv")
    enriched_path = args.out_prefix.with_name(f"{args.out_prefix.name}_{label}_trades.csv")

    summary.to_csv(summary_path, index=False)
    candidates.to_csv(filters_path, index=False)
    combos.to_csv(combos_path, index=False)
    feature_rows.to_csv(features_path, index=False)
    frame.to_csv(enriched_path, index=False)

    print("Summary:")
    print(summary.to_string(index=False))
    if not candidates.empty:
        print("\nTop single filters:")
        print(candidates.head(15).to_string(index=False))
    if not combos.empty:
        print("\nTop two-factor filters:")
        print(combos.head(15).to_string(index=False))
    if not feature_rows.empty:
        print("\nWinner/loss median deltas:")
        print(feature_rows.head(20).to_string(index=False))
    print(f"\nSaved summary to {summary_path}")
    print(f"Saved filters to {filters_path}")
    print(f"Saved combos to {combos_path}")
    print(f"Saved enriched trades to {enriched_path}")


if __name__ == "__main__":
    main()
