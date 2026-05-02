from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backtest_turtle_soup import add_atr, normalize_binance_spot_symbol, resample_ohlc


def profit_factor(rs: pd.Series) -> float:
    wins = rs[rs > 0]
    losses = rs[rs <= 0]
    gross_loss = abs(float(losses.sum()))
    if gross_loss == 0:
        return float("inf") if len(wins) else 0.0
    return float(wins.sum()) / gross_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep-dive ETH breaker behavior.")
    parser.add_argument(
        "--scored-file",
        type=Path,
        default=Path("scripts/symbol_holdout_breaker_majors10_fvg_rf_5bps_scored.csv"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("scripts/.cache"))
    parser.add_argument("--min-prob", type=float, default=0.55)
    parser.add_argument("--direction", default="long")
    parser.add_argument("--min-risk-pct", type=float, default=1.0)
    parser.add_argument("--confirmation-tf", default="15m")
    parser.add_argument("--out-prefix", type=Path, default=Path("scripts/eth_breaker_investigation"))
    return parser.parse_args()


def load_cache(symbol: str, cache_dir: Path) -> pd.DataFrame:
    normalized = normalize_binance_spot_symbol(symbol).lower()
    matches = sorted(cache_dir.glob(f"{normalized}_5m_*.pkl"))
    if not matches:
        raise FileNotFoundError(f"No cached 5m data found for {symbol} in {cache_dir}")
    frame = pd.read_pickle(matches[-1]).sort_values("open_time").reset_index(drop=True)
    frame = add_atr(frame)
    return frame


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out


def first_index(mask: pd.Series) -> float:
    hits = mask[mask].index.to_list()
    if not hits:
        return math.nan
    return float(hits[0])


def classify_failure(row: pd.Series) -> str:
    if row["r_net_cost"] > 0:
        return "winner"
    if bool(row["reentered_zone_3"]) and bool(row["closed_back_below_zone_3"]) and row["bars_to_exit"] <= 6 and row["mfe_r"] < 0.35:
        return "instant_reclaim_failure"
    if row["bars_to_exit"] <= 6 and row["mfe_r"] < 0.25:
        return "instant_fakeout"
    if row["mfe_r"] >= 0.50:
        return "gave_move_back"
    if row["entry_extension_r"] >= 1.0 and row["confirm_gap_r"] >= 0.25:
        return "late_chase_failure"
    return "slow_failure"


def summarize(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "net_r": 0.0}
    rs = frame["r_net_cost"].astype(float)
    return {
        "trades": int(len(frame)),
        "win_rate": round(100.0 * float((rs > 0).mean()), 2),
        "profit_factor": round(profit_factor(rs), 3),
        "net_r": round(float(rs.sum()), 3),
    }


def add_path_metrics(trades: pd.DataFrame, cache_dir: Path, confirmation_tf: str) -> pd.DataFrame:
    out_frames: list[pd.DataFrame] = []
    for symbol, group in trades.groupby("symbol"):
        candles = load_cache(symbol, cache_dir)
        time_to_idx = {pd.Timestamp(ts): idx for idx, ts in enumerate(candles["open_time"])}
        retest_rows = candles.set_index("open_time")[["open", "high", "low", "close", "atr"]]
        confirm = resample_ohlc(candles, confirmation_tf)
        confirm_rows = confirm.set_index("close_time")[["open", "high", "low", "close"]]

        enriched_rows: list[dict[str, Any]] = []
        for _, trade in group.iterrows():
            row = trade.to_dict()
            entry_time = pd.Timestamp(trade["entry_time"])
            exit_time = pd.Timestamp(trade["exit_time"])
            retest_time = pd.Timestamp(trade["retest_time"])
            entry_idx = time_to_idx.get(entry_time)
            exit_idx = time_to_idx.get(exit_time)
            retest_bar = retest_rows.loc[retest_time] if retest_time in retest_rows.index else None
            confirm_time = pd.Timestamp(trade["confirm_time"]) if pd.notna(trade["confirm_time"]) else pd.NaT
            confirm_bar = confirm_rows.loc[confirm_time] if pd.notna(confirm_time) and confirm_time in confirm_rows.index else None

            risk = abs(float(trade["entry_price"]) - float(trade["stop_price"]))
            zone_height = abs(float(trade["zone_top"]) - float(trade["zone_bottom"]))
            if entry_idx is None or exit_idx is None or risk <= 0:
                enriched_rows.append(row)
                continue

            path = candles.iloc[entry_idx : exit_idx + 1].copy()
            first3 = path.head(3)
            first6 = path.head(6)
            favorable = (path["high"] - float(trade["entry_price"])) / risk
            adverse = (path["low"] - float(trade["entry_price"])) / risk
            if trade["direction"] == "short":
                favorable = (float(trade["entry_price"]) - path["low"]) / risk
                adverse = (float(trade["entry_price"]) - path["high"]) / risk

            row["entry_index_resolved"] = entry_idx
            row["exit_index_resolved"] = exit_idx
            row["bars_to_exit"] = int(exit_idx - entry_idx)
            row["mfe_r"] = round(float(favorable.max()), 4)
            row["mae_r"] = round(float(adverse.min()), 4)
            row["hit_0p5r"] = bool((favorable >= 0.5).any())
            row["hit_1r"] = bool((favorable >= 1.0).any())
            row["bars_to_0p5r"] = first_index((favorable >= 0.5).reset_index(drop=True))
            row["bars_to_1r"] = first_index((favorable >= 1.0).reset_index(drop=True))
            row["bars_to_minus0p5r"] = first_index((adverse <= -0.5).reset_index(drop=True))

            if trade["direction"] == "long":
                zone_top = float(trade["zone_top"])
                row["reentered_zone_3"] = bool((first3["low"] <= zone_top).any())
                row["closed_back_below_zone_3"] = bool((first3["close"] < zone_top).any())
                row["closed_back_below_zone_6"] = bool((first6["close"] < zone_top).any())
                row["entry_extension_r"] = round((float(trade["entry_price"]) - zone_top) / risk, 4)
                row["confirm_gap_r"] = round((safe_float(trade["confirm_fvg_bottom"]) - zone_top) / risk, 4)
                row["confirm_break_r"] = round((safe_float(trade["confirm_break_level"]) - zone_top) / risk, 4)
                if retest_bar is not None and zone_height > 0:
                    row["retest_depth_frac"] = round((zone_top - float(retest_bar["low"])) / zone_height, 4)
                    row["retest_close_margin_r"] = round((float(retest_bar["close"]) - zone_top) / risk, 4)
                    row["retest_range_atr"] = round((float(retest_bar["high"]) - float(retest_bar["low"])) / float(retest_bar["atr"]), 4) if float(retest_bar["atr"]) > 0 else math.nan
                else:
                    row["retest_depth_frac"] = math.nan
                    row["retest_close_margin_r"] = math.nan
                    row["retest_range_atr"] = math.nan
                row["early_zone_acceptance_fail"] = bool(row["reentered_zone_3"] and row["closed_back_below_zone_3"])
            else:
                zone_bottom = float(trade["zone_bottom"])
                row["reentered_zone_3"] = bool((first3["high"] >= zone_bottom).any())
                row["closed_back_below_zone_3"] = bool((first3["close"] > zone_bottom).any())
                row["closed_back_below_zone_6"] = bool((first6["close"] > zone_bottom).any())
                row["entry_extension_r"] = round((zone_bottom - float(trade["entry_price"])) / risk, 4)
                row["confirm_gap_r"] = round((zone_bottom - safe_float(trade["confirm_fvg_top"])) / risk, 4)
                row["confirm_break_r"] = round((zone_bottom - safe_float(trade["confirm_break_level"])) / risk, 4)
                if retest_bar is not None and zone_height > 0:
                    row["retest_depth_frac"] = round((float(retest_bar["high"]) - zone_bottom) / zone_height, 4)
                    row["retest_close_margin_r"] = round((zone_bottom - float(retest_bar["close"])) / risk, 4)
                    row["retest_range_atr"] = round((float(retest_bar["high"]) - float(retest_bar["low"])) / float(retest_bar["atr"]), 4) if float(retest_bar["atr"]) > 0 else math.nan
                else:
                    row["retest_depth_frac"] = math.nan
                    row["retest_close_margin_r"] = math.nan
                    row["retest_range_atr"] = math.nan
                row["early_zone_acceptance_fail"] = bool(row["reentered_zone_3"] and row["closed_back_below_zone_3"])

            if pd.notna(confirm_time):
                row["confirm_to_entry_minutes"] = round((entry_time - confirm_time).total_seconds() / 60.0, 2)
            else:
                row["confirm_to_entry_minutes"] = math.nan
            row["retest_to_entry_minutes"] = round((entry_time - retest_time).total_seconds() / 60.0, 2)
            row["break_to_entry_hours"] = round((entry_time - pd.Timestamp(trade["break_time"])).total_seconds() / 3600.0, 3)

            if confirm_bar is not None:
                confirm_range = float(confirm_bar["high"]) - float(confirm_bar["low"])
                row["confirm_close_pos"] = round((float(confirm_bar["close"]) - float(confirm_bar["low"])) / confirm_range, 4) if confirm_range > 0 else math.nan
                row["confirm_body_frac"] = round(abs(float(confirm_bar["close"]) - float(confirm_bar["open"])) / confirm_range, 4) if confirm_range > 0 else math.nan
            else:
                row["confirm_close_pos"] = math.nan
                row["confirm_body_frac"] = math.nan

            enriched_rows.append(row)

        out_frames.append(pd.DataFrame(enriched_rows))

    out = pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame()
    if not out.empty:
        out["failure_archetype"] = out.apply(classify_failure, axis=1)
    return out


def median_table(frame: pd.DataFrame, columns: list[str], label: str) -> pd.DataFrame:
    winners = frame[frame["r_net_cost"] > 0]
    losers = frame[frame["r_net_cost"] <= 0]
    rows: list[dict[str, Any]] = []
    for column in columns:
        rows.append({
            "slice": label,
            "feature": column,
            "winner_median": round(float(pd.to_numeric(winners[column], errors="coerce").median()), 4) if not winners.empty else math.nan,
            "loser_median": round(float(pd.to_numeric(losers[column], errors="coerce").median()), 4) if not losers.empty else math.nan,
            "delta": round(
                float(pd.to_numeric(winners[column], errors="coerce").median()) - float(pd.to_numeric(losers[column], errors="coerce").median()),
                4,
            ) if not winners.empty and not losers.empty else math.nan,
        })
    return pd.DataFrame(rows)


def format_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(empty)"
    return frame.to_string(index=False)


def gate_report(frame: pd.DataFrame, name: str) -> dict[str, Any]:
    row: dict[str, Any] = {"gate": name, **summarize(frame)}
    for fold in [1, 2, 3]:
        subset = frame[frame["fold"] == fold]
        row[f"fold{fold}_trades"] = int(len(subset))
        row[f"fold{fold}_net_r"] = round(float(subset["r_net_cost"].sum()), 3) if not subset.empty else 0.0
        row[f"fold{fold}_pf"] = round(profit_factor(subset["r_net_cost"]), 3) if not subset.empty else 0.0
    return row


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.scored_file)
    for column in ["entry_time", "exit_time", "zone_time", "break_time", "retest_time", "signal_time", "confirm_time"]:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")

    selected = frame[
        (frame["direction"].astype(str) == args.direction)
        & (pd.to_numeric(frame["risk_pct"], errors="coerce") >= args.min_risk_pct)
        & (pd.to_numeric(frame["breaker_prob"], errors="coerce") >= args.min_prob)
    ].copy()

    universe = add_path_metrics(selected[selected["symbol"].astype(str).isin(["BTCUSDT", "ETHUSDT", "SOLUSDT"])].copy(), args.cache_dir, args.confirmation_tf)
    eth = universe[universe["symbol"].astype(str) == "ETHUSDT"].copy()
    btc_sol = universe[universe["symbol"].astype(str).isin(["BTCUSDT", "SOLUSDT"])].copy()

    features = [
        "entry_extension_r",
        "confirm_gap_r",
        "confirm_break_r",
        "confirm_fvg_atr",
        "confirm_close_pos",
        "confirm_body_frac",
        "retest_depth_frac",
        "retest_close_margin_r",
        "retest_range_atr",
        "retest_reject_pos",
        "retest_delay_hours",
        "confirm_delay_hours",
        "mfe_r",
        "bars_to_exit",
    ]

    summary_rows: list[dict[str, Any]] = []
    for label, subset in [
        ("ETH selected", eth),
        ("BTC+SOL selected", btc_sol),
        ("ETH fold1", eth[eth["fold"] == 1]),
        ("ETH fold2", eth[eth["fold"] == 2]),
        ("ETH fold3", eth[eth["fold"] == 3]),
    ]:
        summary_rows.append({"slice": label, **summarize(subset)})
    summary = pd.DataFrame(summary_rows)

    compare_rows: list[dict[str, Any]] = []
    for label, subset in [("ETH", eth), ("BTC+SOL", btc_sol)]:
        compare_rows.append({
            "slice": label,
            "trades": int(len(subset)),
            "early_zone_acceptance_fail_rate": round(100.0 * float(subset["early_zone_acceptance_fail"].mean()), 2) if not subset.empty else 0.0,
            "reentered_zone_3_rate": round(100.0 * float(subset["reentered_zone_3"].mean()), 2) if not subset.empty else 0.0,
            "stop_without_0p5r_rate": round(100.0 * float(((subset["r_net_cost"] <= 0) & (~subset["hit_0p5r"])).mean()), 2) if not subset.empty else 0.0,
            "gave_0p5r_then_failed_rate": round(100.0 * float(((subset["r_net_cost"] <= 0) & (subset["hit_0p5r"])).mean()), 2) if not subset.empty else 0.0,
            "median_entry_extension_r": round(float(pd.to_numeric(subset["entry_extension_r"], errors="coerce").median()), 4) if not subset.empty else math.nan,
            "median_confirm_gap_r": round(float(pd.to_numeric(subset["confirm_gap_r"], errors="coerce").median()), 4) if not subset.empty else math.nan,
            "median_retest_depth_frac": round(float(pd.to_numeric(subset["retest_depth_frac"], errors="coerce").median()), 4) if not subset.empty else math.nan,
        })
    compare = pd.DataFrame(compare_rows)

    feature_summary = pd.concat([
        median_table(eth, features, "ETH"),
        median_table(btc_sol, features, "BTC+SOL"),
    ], ignore_index=True)

    archetypes = (
        eth.groupby(["fold", "failure_archetype"])
        .size()
        .reset_index(name="trades")
        .sort_values(["fold", "trades"], ascending=[True, False])
    )

    eth_losses = eth[eth["r_net_cost"] <= 0].copy()
    loss_examples = eth_losses[
        [
            "entry_time",
            "fold",
            "breaker_prob",
            "r_net_cost",
            "exit_reason",
            "failure_archetype",
            "entry_extension_r",
            "confirm_gap_r",
            "confirm_fvg_atr",
            "confirm_close_pos",
            "retest_depth_frac",
            "retest_close_margin_r",
            "reentered_zone_3",
            "closed_back_below_zone_3",
            "hit_0p5r",
            "mfe_r",
            "bars_to_exit",
        ]
    ].sort_values(["fold", "r_net_cost"])

    candidate_filters = pd.DataFrame([
        gate_report(eth, "ETH base"),
        gate_report(eth[eth["retest_reject_pos"] >= 0.8], "ETH reject>=0.8"),
        gate_report(eth[eth["confirm_delay_hours"] <= 0.5], "ETH confirm_delay<=0.5h"),
        gate_report(eth[eth["confirm_close_pos"] <= 0.8], "ETH confirm_close_pos<=0.8"),
        gate_report(
            eth[(eth["retest_reject_pos"] >= 0.8) & (eth["confirm_delay_hours"] <= 0.5)],
            "ETH reject>=0.8 & delay<=0.5h",
        ),
        gate_report(
            eth[(eth["retest_reject_pos"] >= 0.8) & (eth["entry_extension_r"] <= 0.5)],
            "ETH reject>=0.8 & entry_ext<=0.5R",
        ),
        gate_report(
            universe[(universe["retest_reject_pos"] >= 0.8) & (universe["confirm_delay_hours"] <= 0.5)],
            "Top3 reject>=0.8 & delay<=0.5h",
        ),
        gate_report(
            universe[(universe["confirm_close_pos"] <= 0.8) & (universe["retest_reject_pos"] >= 0.8) & (universe["confirm_delay_hours"] <= 0.5)],
            "Top3 close_pos<=0.8 & reject>=0.8 & delay<=0.5h",
        ),
    ])

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    trades_path = args.out_prefix.with_name(args.out_prefix.name + "_trades.csv")
    feature_path = args.out_prefix.with_name(args.out_prefix.name + "_feature_summary.csv")
    archetype_path = args.out_prefix.with_name(args.out_prefix.name + "_archetypes.csv")
    loss_path = args.out_prefix.with_name(args.out_prefix.name + "_loss_examples.csv")
    candidate_path = args.out_prefix.with_name(args.out_prefix.name + "_candidate_filters.csv")
    report_path = args.out_prefix.with_name(args.out_prefix.name + "_report.md")

    universe.to_csv(trades_path, index=False)
    feature_summary.to_csv(feature_path, index=False)
    archetypes.to_csv(archetype_path, index=False)
    loss_examples.to_csv(loss_path, index=False)
    candidate_filters.to_csv(candidate_path, index=False)

    report_lines = [
        "# ETH Breaker Investigation",
        "",
        f"Gate: direction={args.direction}, min_prob={args.min_prob}, min_risk_pct={args.min_risk_pct}",
        "",
        "## Summary",
        "",
        "```text",
        format_table(summary),
        "```",
        "",
        "## ETH vs BTC+SOL Path Comparison",
        "",
        "```text",
        format_table(compare),
        "```",
        "",
        "## Winner/Loser Feature Medians",
        "",
        "```text",
        format_table(feature_summary),
        "```",
        "",
        "## ETH Failure Archetypes",
        "",
        "```text",
        format_table(archetypes),
        "```",
        "",
        "## ETH Loss Examples",
        "",
        "```text",
        format_table(loss_examples.head(20)),
        "```",
        "",
        "## Candidate Filters",
        "",
        "```text",
        format_table(candidate_filters),
        "```",
        "",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n".join(report_lines))
    print(f"Saved trades to {trades_path}")
    print(f"Saved feature summary to {feature_path}")
    print(f"Saved archetypes to {archetype_path}")
    print(f"Saved loss examples to {loss_path}")
    print(f"Saved candidate filters to {candidate_path}")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
