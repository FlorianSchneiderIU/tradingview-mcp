from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_strategy_time_filters import (
    ROOT,
    bh_qvalues,
    fmt_metric,
    load_sources,
    metrics,
    safe_fisher_win_p,
    safe_mannwhitney_p,
    safe_welch_p,
)


DEFAULT_SOURCES = {
    "million_moves_top20_trail",
    "session_orb_deployed_fixed",
    "turtle_core3_bfm_proxy_p050",
}

DIMENSIONS = [
    "day_of_week",
    "weekend",
    "day_of_month",
    "day_of_month_bin",
    "hour_utc",
    "session_state",
    "session",
    "dow_session_state",
    "weekend_session_state",
    "dow_session",
    "weekend_session",
]


def source_min_n(strategy: str) -> int:
    if strategy.startswith("million_moves"):
        return 60
    if strategy.startswith("session_orb"):
        return 50
    if strategy.startswith("turtle"):
        return 15
    return 30


def pooled_bucket_tests(trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for strategy, strategy_frame in trades.groupby("strategy"):
        min_n = source_min_n(strategy)
        all_m = metrics(strategy_frame)
        for dimension in DIMENSIONS:
            if dimension not in strategy_frame.columns:
                continue
            for bucket, group in strategy_frame.groupby(dimension, dropna=False):
                rest = strategy_frame[strategy_frame[dimension] != bucket]
                if group.empty or rest.empty:
                    continue
                group_m = metrics(group)
                rest_m = metrics(rest)
                row = {
                    "strategy": strategy,
                    "dimension": dimension,
                    "bucket": str(bucket),
                    "min_trades": min_n,
                    "trades": group_m["trades"],
                    "avg_r": group_m["avg_r"],
                    "net_r": group_m["net_r"],
                    "profit_factor": group_m["profit_factor"],
                    "win_rate": group_m["win_rate"],
                    "max_dd": group_m["max_dd"],
                    "rest_trades": rest_m["trades"],
                    "rest_avg_r": rest_m["avg_r"],
                    "rest_profit_factor": rest_m["profit_factor"],
                    "rest_win_rate": rest_m["win_rate"],
                    "all_avg_r": all_m["avg_r"],
                    "all_profit_factor": all_m["profit_factor"],
                    "delta_avg_r": group_m["avg_r"] - rest_m["avg_r"],
                    "p_mean_welch": safe_welch_p(group["r"], rest["r"]),
                    "p_rank_mannwhitney": safe_mannwhitney_p(group["r"], rest["r"]),
                    "p_win_fisher": safe_fisher_win_p(group["r"], rest["r"]),
                }
                rows.append(row)
    out = pd.DataFrame(rows)
    for p_col in ["p_mean_welch", "p_rank_mannwhitney", "p_win_fisher"]:
        q_col = p_col.replace("p_", "q_")
        out[q_col] = np.nan
        for _strategy, idx in out.groupby("strategy").groups.items():
            out.loc[idx, q_col] = bh_qvalues(out.loc[idx, p_col])
    out["significant_mean"] = (out["trades"] >= out["min_trades"]) & (out["q_mean_welch"] <= 0.05)
    out["significant_rank"] = (out["trades"] >= out["min_trades"]) & (out["q_rank_mannwhitney"] <= 0.05)
    out["significant_win"] = (out["trades"] >= out["min_trades"]) & (out["q_win_fisher"] <= 0.05)
    out["pattern_direction"] = np.where(out["delta_avg_r"] >= 0, "stronger", "weaker")
    return out


def symbol_consistency(trades: pd.DataFrame, tests: pd.DataFrame) -> pd.DataFrame:
    significant = tests[
        tests["significant_mean"] | tests["significant_rank"] | tests["significant_win"]
    ].copy()
    rows: list[dict] = []
    for candidate in significant.itertuples(index=False):
        if candidate.dimension not in trades.columns:
            continue
        subset = trades[trades["strategy"].eq(candidate.strategy)]
        block_like = candidate.delta_avg_r < 0
        for symbol, group_all in subset.groupby("symbol"):
            group = group_all[group_all[candidate.dimension].astype(str).eq(str(candidate.bucket))]
            rest = group_all[~group_all[candidate.dimension].astype(str).eq(str(candidate.bucket))]
            if group.empty or rest.empty:
                continue
            gm = metrics(group)
            rm = metrics(rest)
            delta = gm["avg_r"] - rm["avg_r"]
            rows.append(
                {
                    "strategy": candidate.strategy,
                    "symbol": symbol,
                    "dimension": candidate.dimension,
                    "bucket": candidate.bucket,
                    "bucket_trades": gm["trades"],
                    "bucket_avg_r": gm["avg_r"],
                    "bucket_net_r": gm["net_r"],
                    "bucket_pf": gm["profit_factor"],
                    "rest_trades": rm["trades"],
                    "rest_avg_r": rm["avg_r"],
                    "rest_pf": rm["profit_factor"],
                    "delta_avg_r": delta,
                    "agrees": delta < 0 if block_like else delta > 0,
                }
            )
    return pd.DataFrame(rows)


def write_report(
    *,
    trades: pd.DataFrame,
    tests: pd.DataFrame,
    symbol_tests: pd.DataFrame,
    output_prefix: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    included_sources: set[str],
) -> None:
    lines: list[str] = []
    lines.append("# Strategy Time Pattern Analysis")
    lines.append("")
    lines.append(
        f"Exploratory pooled window: `{start.date()}` <= entry_time < `{end.date()}` UTC."
    )
    lines.append(
        "This report is for pattern discovery, not live filtering. It intentionally pools the period instead of enforcing train/OOS consistency."
    )
    lines.append("")
    lines.append("## Sources")
    for strategy in sorted(included_sources):
        subset = trades[trades["strategy"].eq(strategy)]
        if subset.empty:
            continue
        first = subset["entry_time"].min()
        last = subset["entry_time"].max()
        lines.append(
            f"- `{strategy}`: {len(subset)} trades, "
            f"{first.date()}..{last.date()}, symbols={subset['symbol'].nunique()}."
        )
    lines.append("")

    lines.append("## Overall")
    for strategy, group in trades.groupby("strategy"):
        m = metrics(group)
        lines.append(
            f"- `{strategy}`: {int(m['trades'])} trades, net {fmt_metric(m['net_r'], 2)}, "
            f"avg {fmt_metric(m['avg_r'])}, PF {fmt_metric(m['profit_factor'])}, "
            f"win {fmt_metric(m['win_rate'] * 100, 1)}%, maxDD {fmt_metric(m['max_dd'], 2)}"
        )
    lines.append("")

    sig = tests[
        tests["significant_mean"] | tests["significant_rank"] | tests["significant_win"]
    ].copy()
    sig = sig.sort_values(["strategy", "q_mean_welch", "q_rank_mannwhitney", "dimension", "bucket"])
    lines.append("## Statistically Visible Patterns")
    if sig.empty:
        lines.append("- No bucket passed q<=0.05 after correction in this pooled pass.")
    else:
        for row in sig.itertuples(index=False):
            q_bits = []
            if row.significant_mean:
                q_bits.append(f"mean q={fmt_metric(row.q_mean_welch)}")
            if row.significant_rank:
                q_bits.append(f"rank q={fmt_metric(row.q_rank_mannwhitney)}")
            if row.significant_win:
                q_bits.append(f"win q={fmt_metric(row.q_win_fisher)}")
            lines.append(
                f"- `{row.strategy}` `{row.dimension}={row.bucket}` is {row.pattern_direction}: "
                f"{int(row.trades)} trades avg {fmt_metric(row.avg_r)} vs rest {fmt_metric(row.rest_avg_r)}, "
                f"PF {fmt_metric(row.profit_factor)} vs {fmt_metric(row.rest_profit_factor)}, "
                f"{', '.join(q_bits)}."
            )
    lines.append("")

    if not sig.empty and not symbol_tests.empty:
        lines.append("## Symbol Consistency")
        for row in sig.itertuples(index=False):
            subset = symbol_tests[
                symbol_tests["strategy"].eq(row.strategy)
                & symbol_tests["dimension"].eq(row.dimension)
                & symbol_tests["bucket"].astype(str).eq(str(row.bucket))
            ].copy()
            if subset.empty:
                continue
            agrees = int(subset["agrees"].sum())
            total = int(len(subset))
            top = subset.sort_values("delta_avg_r", ascending=row.delta_avg_r < 0).head(8)
            detail = ", ".join(
                f"{r.symbol}:{fmt_metric(r.delta_avg_r)}({int(r.bucket_trades)}t)"
                for r in top.itertuples(index=False)
            )
            lines.append(
                f"- `{row.strategy}` `{row.dimension}={row.bucket}`: {agrees}/{total} symbols agree. {detail}"
            )
        lines.append("")

    lines.append("## Top Watch Buckets")
    for strategy, group in tests.groupby("strategy"):
        eligible = group[group["trades"] >= group["min_trades"]].copy()
        if eligible.empty:
            continue
        lines.append(f"### {strategy}")
        for row in eligible.sort_values("delta_avg_r").head(5).itertuples(index=False):
            lines.append(
                f"- weaker `{row.dimension}={row.bucket}`: {int(row.trades)} trades, "
                f"avg {fmt_metric(row.avg_r)} vs rest {fmt_metric(row.rest_avg_r)}, "
                f"q(mean) {fmt_metric(row.q_mean_welch)}"
            )
        for row in eligible.sort_values("delta_avg_r", ascending=False).head(5).itertuples(index=False):
            lines.append(
                f"- stronger `{row.dimension}={row.bucket}`: {int(row.trades)} trades, "
                f"avg {fmt_metric(row.avg_r)} vs rest {fmt_metric(row.rest_avg_r)}, "
                f"q(mean) {fmt_metric(row.q_mean_welch)}"
            )
        lines.append("")

    lines.append("## Interpretation")
    lines.append("- Use q-values to decide whether a pattern is statistically visible, and symbol consistency to decide whether it is broad or coin-specific.")
    lines.append("- Exact day-of-month and exact hour can still be calendar artifacts; even significant values should usually become model features or monitoring tags before hard live filters.")
    lines.append("- If a pattern repeats in the next dry-run window, it becomes a much better candidate for strategy-specific filtering.")
    lines.append("")
    output_prefix.with_name(f"{output_prefix.name}_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Exploratory pooled time-pattern analysis.")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2026-04-20")
    parser.add_argument("--output-prefix", type=Path, default=ROOT / "strategy_time_patterns_2024_2026")
    parser.add_argument(
        "--sources",
        default=",".join(sorted(DEFAULT_SOURCES)),
        help="Comma-separated source names from analyze_strategy_time_filters.py.",
    )
    args = parser.parse_args()

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    wanted = {chunk.strip() for chunk in args.sources.split(",") if chunk.strip()}

    trades, _sources = load_sources()
    trades = trades[trades["strategy"].isin(wanted)].copy()
    trades = trades[(trades["entry_time"] >= start) & (trades["entry_time"] < end)].copy()
    if trades.empty:
        raise SystemExit("No trades found for requested source/date window.")

    trades["sample"] = "pooled"
    tests = pooled_bucket_tests(trades)
    symbol_tests = symbol_consistency(trades, tests)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(args.output_prefix.with_name(f"{args.output_prefix.name}_normalized_trades.csv"), index=False)
    tests.to_csv(args.output_prefix.with_name(f"{args.output_prefix.name}_bucket_tests.csv"), index=False)
    symbol_tests.to_csv(args.output_prefix.with_name(f"{args.output_prefix.name}_symbol_tests.csv"), index=False)
    write_report(
        trades=trades,
        tests=tests,
        symbol_tests=symbol_tests,
        output_prefix=args.output_prefix,
        start=start,
        end=end,
        included_sources=wanted,
    )

    print(f"Loaded {len(trades)} pooled trades from {len(wanted)} requested sources.")
    print(f"Wrote {args.output_prefix.name}_bucket_tests.csv")
    print(f"Wrote {args.output_prefix.name}_symbol_tests.csv")
    print(f"Wrote {args.output_prefix.name}_report.md")


if __name__ == "__main__":
    main()
