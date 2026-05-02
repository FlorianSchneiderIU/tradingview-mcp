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

from scripts.backtest_turtle_soup import add_atr, normalize_binance_spot_symbol
from scripts.investigate_eth_breaker import add_path_metrics
from scripts.ml_breaker_continuation_filter import enrich, trade_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep-dive the BTC+SOL breaker gate and test improvements.")
    parser.add_argument(
        "--trades-file",
        type=Path,
        default=Path("scripts/breaker_continuation_majors10_1h_fvg_print15_retest72_2022_2026.csv"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("scripts/.cache"))
    parser.add_argument("--symbols", default="BTCUSDT,SOLUSDT")
    parser.add_argument("--direction", default="long")
    parser.add_argument("--min-risk-pct", type=float, default=1.0)
    parser.add_argument("--min-reject-pos", type=float, default=0.75)
    parser.add_argument("--max-confirm-close-pos-dir", type=float, default=0.90)
    parser.add_argument("--confirmation-tf", default="15m")
    parser.add_argument("--fee-bps-side", type=float, default=5.0)
    parser.add_argument("--start", default="2023-04-20")
    parser.add_argument("--fold1-end", default="2024-04-20")
    parser.add_argument("--fold2-end", default="2025-04-20")
    parser.add_argument("--end", default="2026-04-20")
    parser.add_argument("--out-prefix", type=Path, default=Path("scripts/breaker_btc_sol_gate_investigation"))
    return parser.parse_args()


def profit_factor(rs: pd.Series) -> float:
    wins = rs[rs > 0]
    losses = rs[rs <= 0]
    gross_loss = abs(float(losses.sum()))
    if gross_loss == 0:
        return float("inf") if len(wins) else 0.0
    return float(wins.sum()) / gross_loss


def format_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(empty)"
    return frame.to_string(index=False)


def load_cache(symbol: str, cache_dir: Path) -> pd.DataFrame:
    normalized = normalize_binance_spot_symbol(symbol).lower()
    matches = sorted(cache_dir.glob(f"{normalized}_5m_*.pkl"))
    if not matches:
        raise FileNotFoundError(f"No cached 5m data found for {symbol} in {cache_dir}")
    candles = pd.read_pickle(matches[-1]).sort_values("open_time").reset_index(drop=True)
    candles = add_atr(candles)
    return candles


def tradingview_high_before_low(open_val: float, high_val: float, low_val: float) -> bool:
    return abs(open_val - high_val) < abs(open_val - low_val)


def fold_label(entry_time: pd.Timestamp, fold1_end: pd.Timestamp, fold2_end: pd.Timestamp, end: pd.Timestamp) -> int:
    if entry_time < fold1_end:
        return 1
    if entry_time < fold2_end:
        return 2
    if entry_time < end:
        return 3
    return 0


def base_summary(frame: pd.DataFrame) -> dict[str, Any]:
    metrics = trade_metrics(frame)
    return {
        "trades": metrics["trades"],
        "win_rate": metrics["win_rate"],
        "profit_factor": metrics["profit_factor"],
        "net_r": metrics["net_r"],
        "avg_r": metrics["avg_r"],
        "max_dd_r": metrics["max_dd_r"],
    }


def median_table(frame: pd.DataFrame, columns: list[str], label: str) -> pd.DataFrame:
    winners = frame[frame["r_net_cost"] > 0]
    losers = frame[frame["r_net_cost"] <= 0]
    rows: list[dict[str, Any]] = []
    for column in columns:
        winners_col = pd.to_numeric(winners[column], errors="coerce")
        losers_col = pd.to_numeric(losers[column], errors="coerce")
        winner_med = float(winners_col.median()) if not winners_col.empty else math.nan
        loser_med = float(losers_col.median()) if not losers_col.empty else math.nan
        rows.append({
            "slice": label,
            "feature": column,
            "winner_median": round(winner_med, 4) if math.isfinite(winner_med) else math.nan,
            "loser_median": round(loser_med, 4) if math.isfinite(loser_med) else math.nan,
            "delta": round(winner_med - loser_med, 4) if math.isfinite(winner_med) and math.isfinite(loser_med) else math.nan,
        })
    return pd.DataFrame(rows)


def classify_btc_sol_failure(row: pd.Series) -> str:
    if row["r_net_cost"] > 0:
        return "winner"
    if bool(row.get("reentered_zone_3", False)) and bool(row.get("closed_back_below_zone_3", False)) and row["bars_to_exit"] <= 6 and row["mfe_r"] < 0.35:
        return "instant_reclaim_failure"
    if row["bars_to_exit"] <= 6 and row["mfe_r"] < 0.25:
        return "instant_fakeout"
    if row["mfe_r"] >= 0.50:
        return "gave_move_back"
    return "slow_failure"


def trade_key(row: pd.Series) -> str:
    entry_time = pd.Timestamp(row["entry_time"])
    return f"{row['symbol']}|{row['direction']}|{entry_time.isoformat()}|{float(row['entry_price']):.8f}|{float(row['stop_price']):.8f}"


def simulate_variant(
    trades: pd.DataFrame,
    cache_dir: Path,
    fee_bps_side: float,
    target_rr: float,
    max_hold_bars: int,
    fast_fail_bars: int = 0,
    break_even_trigger_r: float | None = None,
) -> pd.DataFrame:
    out_rows: list[dict[str, Any]] = []

    for symbol, group in trades.groupby("symbol"):
        candles = load_cache(symbol, cache_dir)
        time_to_idx = {pd.Timestamp(ts): idx for idx, ts in enumerate(candles["open_time"])}

        for _, trade in group.iterrows():
            direction = str(trade["direction"])
            entry_time = pd.Timestamp(trade["entry_time"])
            entry_idx = time_to_idx.get(entry_time)
            if entry_idx is None:
                continue

            entry_price = float(trade["entry_price"])
            stop_price = float(trade["stop_price"])
            risk = abs(entry_price - stop_price)
            if not math.isfinite(risk) or risk <= 0:
                continue

            sign = 1.0 if direction == "long" else -1.0
            target_price = entry_price + sign * risk * target_rr
            zone_ref = float(trade["zone_top"]) if direction == "long" else float(trade["zone_bottom"])
            be_price = entry_price + sign * risk * break_even_trigger_r if break_even_trigger_r is not None else math.nan

            active_stop = stop_price
            be_armed = False
            exit_price = float(trade["exit_price"])
            exit_time = pd.Timestamp(trade["exit_time"])
            exit_reason = str(trade["exit_reason"])
            exit_idx = time_to_idx.get(exit_time, entry_idx)

            max_idx = min(entry_idx + max_hold_bars, len(candles) - 1)
            for j in range(entry_idx, max_idx + 1):
                bar = candles.iloc[j]
                open_val = float(bar["open"])
                high_val = float(bar["high"])
                low_val = float(bar["low"])
                close_val = float(bar["close"])

                stop = active_stop
                if direction == "long":
                    if open_val <= stop:
                        exit_price = open_val
                        exit_time = pd.Timestamp(bar["open_time"])
                        exit_reason = "stop_gap"
                        exit_idx = j
                        break
                    if open_val >= target_price:
                        exit_price = open_val
                        exit_time = pd.Timestamp(bar["open_time"])
                        exit_reason = "target_gap"
                        exit_idx = j
                        break

                    stop_hit = low_val <= stop
                    target_hit = high_val >= target_price
                    if stop_hit and target_hit:
                        if tradingview_high_before_low(open_val, high_val, low_val):
                            exit_price = target_price
                            exit_reason = "target_same_bar"
                        else:
                            exit_price = stop
                            exit_reason = "stop_same_bar"
                        exit_time = pd.Timestamp(bar["close_time"])
                        exit_idx = j
                        break
                    if stop_hit:
                        exit_price = stop
                        exit_time = pd.Timestamp(bar["close_time"])
                        exit_reason = "stop"
                        exit_idx = j
                        break
                    if target_hit:
                        exit_price = target_price
                        exit_time = pd.Timestamp(bar["close_time"])
                        exit_reason = "target"
                        exit_idx = j
                        break
                else:
                    if open_val >= stop:
                        exit_price = open_val
                        exit_time = pd.Timestamp(bar["open_time"])
                        exit_reason = "stop_gap"
                        exit_idx = j
                        break
                    if open_val <= target_price:
                        exit_price = open_val
                        exit_time = pd.Timestamp(bar["open_time"])
                        exit_reason = "target_gap"
                        exit_idx = j
                        break

                    stop_hit = high_val >= stop
                    target_hit = low_val <= target_price
                    if stop_hit and target_hit:
                        if tradingview_high_before_low(open_val, high_val, low_val):
                            exit_price = stop
                            exit_reason = "stop_same_bar"
                        else:
                            exit_price = target_price
                            exit_reason = "target_same_bar"
                        exit_time = pd.Timestamp(bar["close_time"])
                        exit_idx = j
                        break
                    if stop_hit:
                        exit_price = stop
                        exit_time = pd.Timestamp(bar["close_time"])
                        exit_reason = "stop"
                        exit_idx = j
                        break
                    if target_hit:
                        exit_price = target_price
                        exit_time = pd.Timestamp(bar["close_time"])
                        exit_reason = "target"
                        exit_idx = j
                        break

                bars_open = j - entry_idx
                if fast_fail_bars > 0 and bars_open < fast_fail_bars:
                    invalidated = close_val < zone_ref if direction == "long" else close_val > zone_ref
                    if invalidated:
                        exit_price = close_val
                        exit_time = pd.Timestamp(bar["close_time"])
                        exit_reason = f"fast_fail_zone_{fast_fail_bars}"
                        exit_idx = j
                        break

                if bars_open >= max_hold_bars:
                    exit_price = close_val
                    exit_time = pd.Timestamp(bar["close_time"])
                    exit_reason = "timeout"
                    exit_idx = j
                    break

                if not be_armed and break_even_trigger_r is not None:
                    trigger_hit = high_val >= be_price if direction == "long" else low_val <= be_price
                    if trigger_hit:
                        active_stop = entry_price
                        be_armed = True

            r_multiple = (exit_price - entry_price) / risk if direction == "long" else (entry_price - exit_price) / risk
            cost_r = (abs(entry_price) + abs(exit_price)) * fee_bps_side / 10000.0 / risk
            out_rows.append({
                "trade_key": trade_key(trade),
                "variant_exit_time": exit_time,
                "variant_exit_index": exit_idx,
                "variant_exit_price": exit_price,
                "variant_exit_reason": exit_reason,
                "variant_r_multiple": r_multiple,
                "variant_r_net_cost": r_multiple - cost_r,
            })

    return pd.DataFrame(out_rows)


def summarize_variant(name: str, trades: pd.DataFrame, variant: pd.DataFrame) -> dict[str, Any]:
    joined = trades.merge(variant, on="trade_key", how="left")
    metrics = trade_metrics(joined, "variant_r_net_cost")
    row: dict[str, Any] = {"variant": name, **metrics}
    for fold in [1, 2, 3]:
        subset = joined[joined["fold"] == fold]
        fold_metrics = trade_metrics(subset, "variant_r_net_cost")
        row[f"fold{fold}_trades"] = fold_metrics["trades"]
        row[f"fold{fold}_net_r"] = fold_metrics["net_r"]
        row[f"fold{fold}_pf"] = fold_metrics["profit_factor"]
    return row


def gate_report(name: str, frame: pd.DataFrame) -> dict[str, Any]:
    row: dict[str, Any] = {"gate": name, **trade_metrics(frame)}
    for fold in [1, 2, 3]:
        subset = frame[frame["fold"] == fold]
        fold_metrics = trade_metrics(subset)
        row[f"fold{fold}_trades"] = fold_metrics["trades"]
        row[f"fold{fold}_net_r"] = fold_metrics["net_r"]
        row[f"fold{fold}_pf"] = fold_metrics["profit_factor"]
    return row


def main() -> None:
    args = parse_args()
    symbols = [item.strip() for item in args.symbols.split(",") if item.strip()]
    start = pd.Timestamp(args.start, tz="UTC")
    fold1_end = pd.Timestamp(args.fold1_end, tz="UTC")
    fold2_end = pd.Timestamp(args.fold2_end, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")

    raw = pd.read_csv(args.trades_file)
    enriched = enrich(raw, args.fee_bps_side)
    enriched["entry_time"] = pd.to_datetime(enriched["entry_time"], utc=True)
    enriched = enriched[(enriched["entry_time"] >= start) & (enriched["entry_time"] < end)].copy()
    enriched["fold"] = enriched["entry_time"].apply(lambda ts: fold_label(ts, fold1_end, fold2_end, end))

    selected = enriched[
        enriched["symbol"].astype(str).isin(symbols)
        & (enriched["direction"].astype(str) == args.direction)
        & (pd.to_numeric(enriched["risk_pct"], errors="coerce") >= args.min_risk_pct)
        & (pd.to_numeric(enriched["retest_reject_pos"], errors="coerce") >= args.min_reject_pos)
        & (pd.to_numeric(enriched["confirm_close_pos_dir"], errors="coerce") <= args.max_confirm_close_pos_dir)
        & (enriched["fold"] > 0)
    ].copy()
    selected["trade_key"] = selected.apply(trade_key, axis=1)

    universe = add_path_metrics(selected.copy(), args.cache_dir, args.confirmation_tf)
    universe["failure_archetype"] = universe.apply(classify_btc_sol_failure, axis=1)

    summary = pd.DataFrame(
        [
            {"slice": "BTC+SOL base gate", **base_summary(universe)},
            {"slice": "BTCUSDT", **base_summary(universe[universe["symbol"] == "BTCUSDT"])},
            {"slice": "SOLUSDT", **base_summary(universe[universe["symbol"] == "SOLUSDT"])},
            {"slice": "fold1", **base_summary(universe[universe["fold"] == 1])},
            {"slice": "fold2", **base_summary(universe[universe["fold"] == 2])},
            {"slice": "fold3", **base_summary(universe[universe["fold"] == 3])},
        ]
    )

    exit_reason = (
        universe.groupby("exit_reason")["r_net_cost"]
        .agg(["count", "sum", "mean"])
        .reset_index()
        .rename(columns={"sum": "net_r", "mean": "avg_r"})
        .sort_values("count", ascending=False)
        .round(4)
    )

    path_flags = pd.DataFrame(
        [
            {
                "metric": "hit_0p5r_rate",
                "value": round(100.0 * float(universe["hit_0p5r"].mean()), 2),
            },
            {
                "metric": "hit_1r_rate",
                "value": round(100.0 * float(universe["hit_1r"].mean()), 2),
            },
            {
                "metric": "early_zone_acceptance_fail_rate",
                "value": round(100.0 * float(universe["early_zone_acceptance_fail"].mean()), 2),
            },
            {
                "metric": "stops_hit_0p5r_first_rate",
                "value": round(100.0 * float(universe.loc[universe["exit_reason"] == "stop", "hit_0p5r"].mean()), 2),
            },
            {
                "metric": "timeouts_hit_0p5r_rate",
                "value": round(100.0 * float(universe.loc[universe["exit_reason"] == "timeout", "hit_0p5r"].mean()), 2),
            },
        ]
    )

    feature_summary = median_table(
        universe,
        [
            "retest_reject_pos",
            "confirm_gap_r",
            "confirm_fvg_atr",
            "confirm_close_pos",
            "confirm_body_frac",
            "entry_extension_r",
            "retest_depth_frac",
            "retest_close_margin_r",
            "retest_range_atr",
            "break_to_entry_hours",
        ],
        "BTC+SOL",
    )

    failure_archetypes = (
        universe[universe["r_net_cost"] <= 0]
        .groupby("failure_archetype")["r_net_cost"]
        .agg(["count", "sum", "mean"])
        .reset_index()
        .rename(columns={"sum": "net_r", "mean": "avg_r"})
        .sort_values("count", ascending=False)
        .round(4)
    )

    implementable_filters = pd.DataFrame(
        [
            gate_report("base", universe),
            gate_report("reject>=0.85", universe[universe["retest_reject_pos"] >= 0.85]),
            gate_report("reject>=0.90", universe[universe["retest_reject_pos"] >= 0.90]),
            gate_report("confirm_close_pos<=0.75", universe[universe["confirm_close_pos"] <= 0.75]),
            gate_report("confirm_gap_r>=-0.05", universe[universe["confirm_gap_r"] >= -0.05]),
            gate_report(
                "confirm_gap_r>=-0.05 & reject>=0.90",
                universe[(universe["confirm_gap_r"] >= -0.05) & (universe["retest_reject_pos"] >= 0.90)],
            ),
        ]
    ).sort_values(["avg_r", "net_r"], ascending=[False, False])

    variant_specs = [
        ("baseline_rr2_hold120", {"target_rr": 2.0, "max_hold_bars": 120}),
        ("fast_fail_3_rr2_hold120", {"target_rr": 2.0, "max_hold_bars": 120, "fast_fail_bars": 3}),
        ("tp1p5_hold120", {"target_rr": 1.5, "max_hold_bars": 120}),
        ("tp1p5_fast_fail_3", {"target_rr": 1.5, "max_hold_bars": 120, "fast_fail_bars": 3}),
        ("tp1p25_hold120", {"target_rr": 1.25, "max_hold_bars": 120}),
        ("hold180_rr2", {"target_rr": 2.0, "max_hold_bars": 180}),
        ("hold240_rr2", {"target_rr": 2.0, "max_hold_bars": 240}),
        ("be0p5_rr2_hold120", {"target_rr": 2.0, "max_hold_bars": 120, "break_even_trigger_r": 0.5}),
        ("be1p0_rr2_hold120", {"target_rr": 2.0, "max_hold_bars": 120, "break_even_trigger_r": 1.0}),
        (
            "be0p5_tp1p5_fast_fail_3",
            {"target_rr": 1.5, "max_hold_bars": 120, "fast_fail_bars": 3, "break_even_trigger_r": 0.5},
        ),
    ]

    variant_frames: list[pd.DataFrame] = []
    variant_rows: list[dict[str, Any]] = []
    for name, kwargs in variant_specs:
        variant = simulate_variant(universe, args.cache_dir, args.fee_bps_side, **kwargs)
        variant["variant"] = name
        variant_rows.append(summarize_variant(name, universe, variant))
        variant_frames.append(variant)

    management_variants = pd.DataFrame(variant_rows).sort_values(["avg_r", "net_r"], ascending=[False, False])
    variant_trades = pd.concat(variant_frames, ignore_index=True) if variant_frames else pd.DataFrame()

    loss_examples = universe[universe["r_net_cost"] <= 0][
        [
            "symbol",
            "entry_time",
            "fold",
            "r_net_cost",
            "exit_reason",
            "failure_archetype",
            "mfe_r",
            "hit_0p5r",
            "hit_1r",
            "bars_to_exit",
            "retest_reject_pos",
            "confirm_close_pos",
            "confirm_gap_r",
        ]
    ].sort_values(["r_net_cost", "entry_time"]).head(25)

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    trades_path = args.out_prefix.with_name(args.out_prefix.name + "_trades.csv")
    summary_path = args.out_prefix.with_name(args.out_prefix.name + "_summary.csv")
    exit_reason_path = args.out_prefix.with_name(args.out_prefix.name + "_exit_reason.csv")
    feature_path = args.out_prefix.with_name(args.out_prefix.name + "_feature_summary.csv")
    failure_path = args.out_prefix.with_name(args.out_prefix.name + "_failure_archetypes.csv")
    filter_path = args.out_prefix.with_name(args.out_prefix.name + "_implementable_filters.csv")
    variant_path = args.out_prefix.with_name(args.out_prefix.name + "_management_variants.csv")
    variant_trades_path = args.out_prefix.with_name(args.out_prefix.name + "_management_variant_trades.csv")
    loss_path = args.out_prefix.with_name(args.out_prefix.name + "_loss_examples.csv")
    report_path = args.out_prefix.with_name(args.out_prefix.name + "_report.md")

    universe.to_csv(trades_path, index=False)
    summary.to_csv(summary_path, index=False)
    exit_reason.to_csv(exit_reason_path, index=False)
    feature_summary.to_csv(feature_path, index=False)
    failure_archetypes.to_csv(failure_path, index=False)
    implementable_filters.to_csv(filter_path, index=False)
    management_variants.to_csv(variant_path, index=False)
    variant_trades.to_csv(variant_trades_path, index=False)
    loss_examples.to_csv(loss_path, index=False)

    report_lines = [
        "# BTC+SOL Breaker Gate Investigation",
        "",
        f"Base gate: symbols={symbols}, direction={args.direction}, min_risk_pct={args.min_risk_pct}, min_reject_pos={args.min_reject_pos}, max_confirm_close_pos_dir={args.max_confirm_close_pos_dir}",
        "",
        "## Summary",
        "",
        "```text",
        format_table(summary),
        "```",
        "",
        "## Exit Reason Breakdown",
        "",
        "```text",
        format_table(exit_reason),
        "```",
        "",
        "## Path Flags",
        "",
        "```text",
        format_table(path_flags),
        "```",
        "",
        "## Failure Archetypes",
        "",
        "```text",
        format_table(failure_archetypes),
        "```",
        "",
        "## Winner vs Loser Feature Medians",
        "",
        "```text",
        format_table(feature_summary),
        "```",
        "",
        "## Implementable Entry Filters",
        "",
        "```text",
        format_table(implementable_filters),
        "```",
        "",
        "## Management Variants",
        "",
        "```text",
        format_table(management_variants),
        "```",
        "",
        "## Loss Examples",
        "",
        "```text",
        format_table(loss_examples),
        "```",
        "",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n".join(report_lines))
    print(f"Saved trades to {trades_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved exit reasons to {exit_reason_path}")
    print(f"Saved feature summary to {feature_path}")
    print(f"Saved failure archetypes to {failure_path}")
    print(f"Saved implementable filters to {filter_path}")
    print(f"Saved management variants to {variant_path}")
    print(f"Saved management variant trades to {variant_trades_path}")
    print(f"Saved loss examples to {loss_path}")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
