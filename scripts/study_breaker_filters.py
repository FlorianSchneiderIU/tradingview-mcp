from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

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


def metrics(frame: pd.DataFrame, column: str = "r_net_cost") -> dict[str, Any]:
    if frame.empty:
        return {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "net_r": 0.0, "avg_r": 0.0, "max_dd_r": 0.0}
    rs = frame.sort_values("exit_time")[column].astype(float)
    return {
        "trades": int(len(rs)),
        "win_rate": round(100.0 * float((rs > 0).mean()), 2),
        "profit_factor": round(profit_factor(rs), 3),
        "net_r": round(float(rs.sum()), 3),
        "avg_r": round(float(rs.mean()), 4),
        "max_dd_r": round(max_drawdown(rs.to_list()), 3),
    }


def enrich(frame: pd.DataFrame, fee_bps_side: float) -> pd.DataFrame:
    out = frame.copy()
    for column in ["entry_time", "exit_time", "break_time", "retest_time", "zone_time"]:
        out[column] = pd.to_datetime(out[column], utc=True)
    risk = (out["entry_price"].astype(float) - out["stop_price"].astype(float)).abs()
    out["risk_pct"] = risk / out["entry_price"].astype(float) * 100.0
    out["zone_width_pct"] = (out["zone_top"].astype(float) - out["zone_bottom"].astype(float)).abs() / out["entry_price"].astype(float) * 100.0
    out["zone_age_hours"] = (out["break_time"] - out["zone_time"]).dt.total_seconds() / 3600.0
    out["retest_delay_hours"] = (out["retest_time"] - out["break_time"]).dt.total_seconds() / 3600.0
    out["entry_hour"] = out["entry_time"].dt.hour
    out["entry_dow"] = out["entry_time"].dt.dayofweek
    cost_r = (out["entry_price"].abs() + out["exit_price"].abs()) * fee_bps_side / 10000.0 / risk
    out["r_net_cost"] = out["r_multiple"].astype(float) - cost_r
    return out


def qcut_bucket(series: pd.Series, buckets: int = 4) -> pd.Series:
    try:
        return pd.qcut(series.rank(method="first"), buckets, labels=[f"q{i+1}" for i in range(buckets)])
    except ValueError:
        return pd.Series(["all"] * len(series), index=series.index)


def grouped_rows(frame: pd.DataFrame, split: pd.Timestamp, end: pd.Timestamp, column: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value, group in frame.groupby(column, dropna=False):
        train = group[group["entry_time"] < split]
        oos = group[(group["entry_time"] >= split) & (group["entry_time"] < end)]
        rows.append({
            "filter": column,
            "value": str(value),
            **{f"train_{key}": val for key, val in metrics(train).items()},
            **{f"oos_{key}": val for key, val in metrics(oos).items()},
        })
    return rows


def combination_rows(
    frame: pd.DataFrame,
    split: pd.Timestamp,
    end: pd.Timestamp,
    columns: list[str],
    min_train_trades: int,
    min_oos_trades: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left_index, left in enumerate(columns):
        for right in columns[left_index + 1:]:
            for values, group in frame.groupby([left, right], dropna=False):
                train = group[group["entry_time"] < split]
                oos = group[(group["entry_time"] >= split) & (group["entry_time"] < end)]
                if len(train) < min_train_trades or len(oos) < min_oos_trades:
                    continue
                rows.append({
                    "filter": f"{left}+{right}",
                    "value": "|".join(str(value) for value in values),
                    **{f"train_{key}": val for key, val in metrics(train).items()},
                    **{f"oos_{key}": val for key, val in metrics(oos).items()},
                })
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search simple breaker-continuation filters.")
    parser.add_argument("file", type=Path)
    parser.add_argument("--split", default="2025-04-20")
    parser.add_argument("--end", default="2026-04-20")
    parser.add_argument("--fee-bps-side", type=float, default=5.0)
    parser.add_argument("--min-train-trades", type=int, default=120)
    parser.add_argument("--min-oos-trades", type=int, default=30)
    parser.add_argument("--out-prefix", type=Path, default=Path("scripts/breaker_filter_study"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = pd.Timestamp(args.split, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    frame = enrich(pd.read_csv(args.file), args.fee_bps_side)
    frame["risk_pct_q"] = qcut_bucket(frame["risk_pct"])
    frame["zone_width_q"] = qcut_bucket(frame["zone_width_pct"])
    frame["zone_age_q"] = qcut_bucket(frame["zone_age_hours"])
    frame["retest_delay_q"] = qcut_bucket(frame["retest_delay_hours"])
    frame["hour_bucket"] = (frame["entry_hour"] // 4 * 4).astype(str) + "-" + (frame["entry_hour"] // 4 * 4 + 3).astype(str)
    frame["symbol_direction"] = frame["symbol"].astype(str) + "_" + frame["direction"].astype(str)

    filters = [
        "symbol",
        "direction",
        "symbol_direction",
        "risk_pct_q",
        "zone_width_q",
        "zone_age_q",
        "retest_delay_q",
        "hour_bucket",
        "entry_dow",
    ]
    rows: list[dict[str, Any]] = []
    for column in filters:
        rows.extend(grouped_rows(frame, split, end, column))
    summary = pd.DataFrame(rows).sort_values(["oos_net_r", "train_net_r"], ascending=False)
    combo_summary = pd.DataFrame(
        combination_rows(frame, split, end, filters, args.min_train_trades, args.min_oos_trades)
    )
    if not combo_summary.empty:
        combo_summary = combo_summary.sort_values(["oos_net_r", "train_net_r"], ascending=False)
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_prefix.with_suffix(".csv")
    combo_path = args.out_prefix.with_name(args.out_prefix.name + "_combos.csv")
    enriched_path = args.out_prefix.with_name(args.out_prefix.name + "_trades.csv")
    summary.to_csv(summary_path, index=False)
    combo_summary.to_csv(combo_path, index=False)
    frame.to_csv(enriched_path, index=False)

    print("Overall train:", metrics(frame[frame["entry_time"] < split]))
    print("Overall oos:  ", metrics(frame[(frame["entry_time"] >= split) & (frame["entry_time"] < end)]))
    print()
    print("Top simple filters by OOS net R after costs:")
    print(summary.head(20).to_string(index=False))
    if not combo_summary.empty:
        print()
        print("Top two-factor filters by OOS net R after costs:")
        print(combo_summary.head(20).to_string(index=False))
    print(f"\nSaved summary to {summary_path}")
    print(f"Saved combo summary to {combo_path}")
    print(f"Saved enriched trades to {enriched_path}")


if __name__ == "__main__":
    main()
