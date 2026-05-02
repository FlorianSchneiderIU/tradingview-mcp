from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a frozen zone-channel production replay run.")
    parser.add_argument("--prefix", type=Path, required=True, help="Output prefix used by run_zone_channel_production.py")
    parser.add_argument(
        "--signals-prefix",
        type=Path,
        default=None,
        help="Optional alternate prefix to source *_signals.csv and *_config.json from.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def trade_metrics(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {
            "trades": 0.0,
            "net_r": 0.0,
            "hit_rate": 0.0,
            "profit_factor": 0.0,
            "avg_r": 0.0,
            "median_r": 0.0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
        }
    net_r = pd.to_numeric(frame["r_multiple_net"], errors="coerce").fillna(0.0)
    returns = pd.to_numeric(frame["return_pct"], errors="coerce").fillna(0.0)
    wins = net_r[net_r > 0.0]
    losses = net_r[net_r < 0.0]
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    gross_profit = float(wins.sum()) if not wins.empty else 0.0
    gross_loss = float(-losses.sum()) if not losses.empty else 0.0
    pf = float(gross_profit / gross_loss) if gross_loss > 0.0 else (float("inf") if gross_profit > 0.0 else 0.0)
    return {
        "trades": float(len(frame)),
        "net_r": float(net_r.sum()),
        "hit_rate": float((net_r > 0.0).mean()),
        "profit_factor": pf,
        "avg_r": float(net_r.mean()),
        "median_r": float(net_r.median()),
        "total_return": float(equity.iloc[-1] - 1.0),
        "max_drawdown": float(drawdown.min()) if not drawdown.empty else 0.0,
    }


def summarize_group(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = frame.groupby(group_cols, dropna=False, sort=True)
    for key, chunk in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: value for col, value in zip(group_cols, key)}
        row.update(trade_metrics(chunk))
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty and "net_r" in out.columns:
        out = out.sort_values(group_cols).reset_index(drop=True)
    return out


def finite_or_blank(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except Exception:
        return ""
    if np.isnan(number):
        return ""
    if np.isinf(number):
        return "inf" if number > 0 else "-inf"
    return f"{number:.{digits}f}"


def frame_to_markdown(frame: pd.DataFrame, columns: list[str] | None = None, max_rows: int = 20) -> str:
    if frame.empty:
        return "_none_"
    view = frame.copy()
    if columns is not None:
        keep = [column for column in columns if column in view.columns]
        view = view[keep]
    if len(view) > max_rows:
        view = view.head(max_rows)
    try:
        return view.to_markdown(index=False)
    except ImportError:
        return "```text\n" + view.to_string(index=False) + "\n```"


def decide_vol_bucket(series: pd.Series) -> pd.Series:
    usable = pd.to_numeric(series, errors="coerce")
    if usable.notna().sum() < 3:
        out = pd.Series(index=series.index, dtype=object)
        out.loc[usable.notna()] = "all"
        return out
    ranks = usable.rank(method="first", pct=True)
    out = pd.Series(index=series.index, dtype=object)
    valid = usable.notna()
    out.loc[valid & (ranks <= 1.0 / 3.0)] = "low"
    out.loc[valid & (ranks > 1.0 / 3.0) & (ranks <= 2.0 / 3.0)] = "mid"
    out.loc[valid & (ranks > 2.0 / 3.0)] = "high"
    return out


def decide_trend_bucket(series: pd.Series) -> pd.Series:
    usable = pd.to_numeric(series, errors="coerce")
    out = pd.Series(index=series.index, dtype=object)
    valid = usable.notna()
    out.loc[valid & (usable < 0.50)] = "<0.50"
    out.loc[valid & (usable >= 0.50) & (usable < 0.75)] = "0.50-0.75"
    out.loc[valid & (usable >= 0.75) & (usable < 1.00)] = "0.75-1.00"
    out.loc[valid & (usable >= 1.00)] = ">=1.00"
    return out


def decide_rr_bucket(series: pd.Series) -> pd.Series:
    usable = pd.to_numeric(series, errors="coerce")
    out = pd.Series(index=series.index, dtype=object)
    valid = usable.notna()
    out.loc[valid & (usable < 0.5)] = "<0.5"
    out.loc[valid & (usable >= 0.5) & (usable < 1.0)] = "0.5-1.0"
    out.loc[valid & (usable >= 1.0) & (usable < 1.5)] = "1.0-1.5"
    out.loc[valid & (usable >= 1.5)] = ">=1.5"
    return out


def main() -> None:
    args = parse_args()
    prefix = args.prefix
    signals_prefix = args.signals_prefix or prefix
    signals_path = signals_prefix.with_name(signals_prefix.name + "_signals.csv")
    decisions_path = prefix.with_name(prefix.name + "_decisions.csv")
    trades_path = prefix.with_name(prefix.name + "_trades.csv")
    summary_path = prefix.with_name(prefix.name + "_summary.json")
    config_path = signals_prefix.with_name(signals_prefix.name + "_config.json")

    signals = pd.read_csv(signals_path)
    decisions = pd.read_csv(decisions_path)
    trades = pd.read_csv(trades_path)
    summary = load_json(summary_path)
    config = load_json(config_path)

    if trades.empty:
        raise SystemExit("Trade file is empty; nothing to analyze.")

    for column in ["event_time", "entry_time", "exit_time"]:
        if column in signals.columns:
            signals[column] = pd.to_datetime(signals[column], utc=True, errors="coerce")
        if column in trades.columns:
            trades[column] = pd.to_datetime(trades[column], utc=True, errors="coerce")
        if column in decisions.columns:
            decisions[column] = pd.to_datetime(decisions[column], utc=True, errors="coerce")

    merged = trades.merge(
        signals,
        on=["event_key", "symbol", "direction"],
        how="left",
        suffixes=("", "_signal"),
    )
    merged["entry_year"] = merged["entry_time"].dt.year
    merged["entry_month"] = merged["entry_time"].dt.to_period("M").astype(str)
    merged["entry_quarter"] = merged["entry_time"].dt.to_period("Q").astype(str)
    merged["vol_bucket"] = decide_vol_bucket(merged.get("realized_vol_24h", pd.Series(index=merged.index, dtype=float)))
    merged["trend_bucket"] = decide_trend_bucket(merged.get("rolling_trend_strength_1h", pd.Series(index=merged.index, dtype=float)))
    merged["rr_bucket"] = decide_rr_bucket(merged.get("target_rr_planned", pd.Series(index=merged.index, dtype=float)))

    yearly = summarize_group(merged, ["entry_year"])
    quarterly = summarize_group(merged, ["entry_quarter"])
    monthly = summarize_group(merged, ["entry_month"])
    by_direction = summarize_group(merged, ["direction"])
    by_zone_tf = summarize_group(merged, ["zone_tf"])
    by_direction_zone = summarize_group(merged, ["direction", "zone_tf"])
    by_exit = summarize_group(merged, ["exit_reason"])
    by_vol = summarize_group(merged, ["vol_bucket"])
    by_trend = summarize_group(merged, ["trend_bucket"])
    by_rr = summarize_group(merged, ["rr_bucket"])

    decision_counts = (
        decisions["status"]
        .astype(str)
        .value_counts(dropna=False)
        .rename_axis("status")
        .reset_index(name="count")
    )

    worst_stretch = quarterly.sort_values("net_r").head(4).reset_index(drop=True)
    best_stretch = quarterly.sort_values("net_r", ascending=False).head(4).reset_index(drop=True)
    worst_trades = merged.sort_values("r_multiple_net").head(10).copy()
    best_trades = merged.sort_values("r_multiple_net", ascending=False).head(10).copy()

    out_yearly = prefix.with_name(prefix.name + "_analysis_yearly.csv")
    out_quarterly = prefix.with_name(prefix.name + "_analysis_quarterly.csv")
    out_monthly = prefix.with_name(prefix.name + "_analysis_monthly.csv")
    out_direction = prefix.with_name(prefix.name + "_analysis_direction.csv")
    out_zone = prefix.with_name(prefix.name + "_analysis_zone_tf.csv")
    out_direction_zone = prefix.with_name(prefix.name + "_analysis_direction_zone_tf.csv")
    out_exit = prefix.with_name(prefix.name + "_analysis_exit_reason.csv")
    out_vol = prefix.with_name(prefix.name + "_analysis_vol_bucket.csv")
    out_trend = prefix.with_name(prefix.name + "_analysis_trend_bucket.csv")
    out_rr = prefix.with_name(prefix.name + "_analysis_rr_bucket.csv")
    out_decisions = prefix.with_name(prefix.name + "_analysis_decision_counts.csv")
    out_worst = prefix.with_name(prefix.name + "_analysis_worst_trades.csv")
    out_best = prefix.with_name(prefix.name + "_analysis_best_trades.csv")
    out_report = prefix.with_name(prefix.name + "_analysis_report.md")

    yearly.to_csv(out_yearly, index=False)
    quarterly.to_csv(out_quarterly, index=False)
    monthly.to_csv(out_monthly, index=False)
    by_direction.to_csv(out_direction, index=False)
    by_zone_tf.to_csv(out_zone, index=False)
    by_direction_zone.to_csv(out_direction_zone, index=False)
    by_exit.to_csv(out_exit, index=False)
    by_vol.to_csv(out_vol, index=False)
    by_trend.to_csv(out_trend, index=False)
    by_rr.to_csv(out_rr, index=False)
    decision_counts.to_csv(out_decisions, index=False)
    worst_trades.to_csv(out_worst, index=False)
    best_trades.to_csv(out_best, index=False)

    lines = [
        "# Zone-Channel Production Run Analysis",
        "",
        "## Scope",
        "",
        f"- name: `{config.get('name', prefix.name)}`",
        f"- symbol: `{config.get('symbol', '')}`",
        f"- decision timeframe: `{config.get('decision_timeframe', '')}`",
        f"- zone timeframes: `{', '.join(config.get('zone_timeframes', []))}`",
        f"- selection gates: `{', '.join(config.get('selection_gates', []))}`",
        "",
        "## Headline",
        "",
        f"- trades: `{summary.get('trades', 0)}`",
        f"- selected signal rows: `{summary.get('selected_signal_rows', 0)}`",
        f"- net R: `{finite_or_blank(summary.get('net_r'))}`",
        f"- total return: `{finite_or_blank(summary.get('total_return'))}`",
        f"- profit factor: `{finite_or_blank(summary.get('profit_factor'))}`",
        f"- hit rate: `{finite_or_blank(summary.get('hit_rate'))}`",
        f"- max drawdown: `{finite_or_blank(summary.get('max_drawdown'))}`",
        "",
        "## Decision Funnel",
        "",
        frame_to_markdown(decision_counts, ["status", "count"], max_rows=20),
        "",
        "## Yearly Performance",
        "",
        frame_to_markdown(yearly, ["entry_year", "trades", "net_r", "profit_factor", "hit_rate", "avg_r", "total_return", "max_drawdown"], max_rows=10),
        "",
        "## Direction",
        "",
        frame_to_markdown(by_direction, ["direction", "trades", "net_r", "profit_factor", "hit_rate", "avg_r", "total_return"], max_rows=10),
        "",
        "## Zone Timeframe",
        "",
        frame_to_markdown(by_zone_tf, ["zone_tf", "trades", "net_r", "profit_factor", "hit_rate", "avg_r", "total_return"], max_rows=10),
        "",
        "## Direction x Zone Timeframe",
        "",
        frame_to_markdown(by_direction_zone, ["direction", "zone_tf", "trades", "net_r", "profit_factor", "hit_rate", "avg_r"], max_rows=20),
        "",
        "## Exit Reason",
        "",
        frame_to_markdown(by_exit, ["exit_reason", "trades", "net_r", "profit_factor", "hit_rate", "avg_r"], max_rows=10),
        "",
        "## Regime Splits",
        "",
        "### realized_vol_24h terciles",
        "",
        frame_to_markdown(by_vol, ["vol_bucket", "trades", "net_r", "profit_factor", "hit_rate", "avg_r"], max_rows=10),
        "",
        "### rolling_trend_strength_1h buckets",
        "",
        frame_to_markdown(by_trend, ["trend_bucket", "trades", "net_r", "profit_factor", "hit_rate", "avg_r"], max_rows=10),
        "",
        "### target_rr_planned buckets",
        "",
        frame_to_markdown(by_rr, ["rr_bucket", "trades", "net_r", "profit_factor", "hit_rate", "avg_r"], max_rows=10),
        "",
        "## Best/Worst Quarters",
        "",
        "### Worst",
        "",
        frame_to_markdown(worst_stretch, ["entry_quarter", "trades", "net_r", "profit_factor", "hit_rate", "avg_r"], max_rows=10),
        "",
        "### Best",
        "",
        frame_to_markdown(best_stretch, ["entry_quarter", "trades", "net_r", "profit_factor", "hit_rate", "avg_r"], max_rows=10),
        "",
        "## Worst Trades",
        "",
        frame_to_markdown(
            worst_trades,
            ["entry_time", "direction", "zone_tf", "exit_reason", "r_multiple_net", "target_rr_planned", "reclaim_pos", "rolling_trend_strength_1h", "realized_vol_24h"],
            max_rows=10,
        ),
        "",
        "## Best Trades",
        "",
        frame_to_markdown(
            best_trades,
            ["entry_time", "direction", "zone_tf", "exit_reason", "r_multiple_net", "target_rr_planned", "reclaim_pos", "rolling_trend_strength_1h", "realized_vol_24h"],
            max_rows=10,
        ),
        "",
    ]
    out_report.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    print(f"Saved analysis report to {out_report}")
    print(f"Saved yearly summary to {out_yearly}")
    print(f"Saved quarterly summary to {out_quarterly}")
    print(f"Saved monthly summary to {out_monthly}")
    print(f"Saved direction summary to {out_direction}")
    print(f"Saved zone summary to {out_zone}")
    print(f"Saved direction-zone summary to {out_direction_zone}")
    print(f"Saved exit summary to {out_exit}")
    print(f"Saved vol summary to {out_vol}")
    print(f"Saved trend summary to {out_trend}")
    print(f"Saved RR summary to {out_rr}")
    print(f"Saved decision counts to {out_decisions}")
    print(f"Saved worst trades to {out_worst}")
    print(f"Saved best trades to {out_best}")


if __name__ == "__main__":
    main()
