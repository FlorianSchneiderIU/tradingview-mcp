from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.sweep_turtle_soup_oos import ensure_cache


@dataclass
class SimTrade:
    symbol: str
    variant: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str
    r_multiple: float
    exit_reason: str


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in str(raw).split(",") if item.strip()]


def tradingview_high_before_low(open_val: float, high_val: float, low_val: float) -> bool:
    return abs(open_val - high_val) < abs(open_val - low_val)


def zone_key(symbol: str, direction: str, time_value: pd.Timestamp, top: float, bottom: float) -> str:
    return f"{symbol}|{direction}|{pd.Timestamp(time_value).isoformat()}|{top:.8f}|{bottom:.8f}"


def summarize(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "net_r": 0.0,
            "avg_r": 0.0,
        }
    wins = frame[frame["r_multiple"] > 0]["r_multiple"]
    losses = frame[frame["r_multiple"] <= 0]["r_multiple"]
    gross_loss = abs(float(losses.sum()))
    pf = (float(wins.sum()) / gross_loss) if gross_loss > 0 else float("inf")
    return {
        "trades": int(len(frame)),
        "win_rate": round(float((frame["r_multiple"] > 0).mean() * 100.0), 2),
        "profit_factor": round(pf, 3) if pf != float("inf") else float("inf"),
        "net_r": round(float(frame["r_multiple"].sum()), 3),
        "avg_r": round(float(frame["r_multiple"].mean()), 3),
    }


def exit_check(
    direction: str,
    stop: float,
    target: float,
    open_val: float,
    high_val: float,
    low_val: float,
) -> tuple[float | None, str | None]:
    high_first = tradingview_high_before_low(open_val, high_val, low_val)
    if direction == "long":
        if open_val <= stop:
            return open_val, "stop_gap"
        if open_val >= target:
            return open_val, "target_gap"
        stop_hit = low_val <= stop
        target_hit = high_val >= target
        if stop_hit and target_hit:
            return (target, "target_same_bar") if high_first else (stop, "stop_same_bar")
        if stop_hit:
            return stop, "stop"
        if target_hit:
            return target, "target"
    else:
        if open_val >= stop:
            return open_val, "stop_gap"
        if open_val <= target:
            return open_val, "target_gap"
        stop_hit = high_val >= stop
        target_hit = low_val <= target
        if stop_hit and target_hit:
            return (stop, "stop_same_bar") if high_first else (target, "target_same_bar")
        if stop_hit:
            return stop, "stop"
        if target_hit:
            return target, "target"
    return None, None


def simulate_single_target(
    *,
    direction: str,
    entry_price: float,
    stop_price: float,
    target_price: float,
    risk: float,
    entry_index: int,
    max_hold_bars: int,
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    close_times: list[pd.Timestamp],
) -> tuple[float, pd.Timestamp, str]:
    last = min(len(opens) - 1, entry_index + max_hold_bars)
    for j in range(entry_index + 1, last + 1):
        price, reason = exit_check(direction, stop_price, target_price, opens[j], highs[j], lows[j])
        if price is None:
            continue
        if direction == "long":
            r = (price - entry_price) / risk
        else:
            r = (entry_price - price) / risk
        return float(r), pd.Timestamp(close_times[j]), str(reason)

    exit_price = closes[last]
    if direction == "long":
        r = (exit_price - entry_price) / risk
    else:
        r = (entry_price - exit_price) / risk
    return float(r), pd.Timestamp(close_times[last]), "time"


def simulate_hybrid(
    *,
    direction: str,
    entry_price: float,
    stop_price: float,
    risk: float,
    entry_index: int,
    max_hold_bars: int,
    rr1: float,
    rr2: float,
    runner_fraction: float,
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    close_times: list[pd.Timestamp],
) -> tuple[float, pd.Timestamp, str]:
    sign = 1.0 if direction == "long" else -1.0
    first_target = entry_price + sign * risk * rr1

    last = min(len(opens) - 1, entry_index + max_hold_bars)
    first_fill_idx = None
    for j in range(entry_index + 1, last + 1):
        price, reason = exit_check(direction, stop_price, first_target, opens[j], highs[j], lows[j])
        if price is None:
            continue
        if "stop" in str(reason):
            return -1.0, pd.Timestamp(close_times[j]), f"hybrid_first_{reason}"
        first_fill_idx = j
        break

    if first_fill_idx is None:
        # Never got the first partial; fall back to standard stop/time behavior.
        return simulate_single_target(
            direction=direction,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=first_target,
            risk=risk,
            entry_index=entry_index,
            max_hold_bars=max_hold_bars,
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
            close_times=close_times,
        )

    # Realize partial at rr1, then run remainder to rr2 with stop moved to breakeven.
    partial_r = (1.0 - runner_fraction) * rr1
    runner_stop = entry_price
    runner_target = entry_price + sign * risk * rr2

    for j in range(first_fill_idx + 1, last + 1):
        price, reason = exit_check(direction, runner_stop, runner_target, opens[j], highs[j], lows[j])
        if price is None:
            continue
        if direction == "long":
            runner_r = (price - entry_price) / risk
        else:
            runner_r = (entry_price - price) / risk
        total_r = partial_r + runner_fraction * runner_r
        return float(total_r), pd.Timestamp(close_times[j]), f"hybrid_runner_{reason}"

    runner_exit = closes[last]
    if direction == "long":
        runner_r = (runner_exit - entry_price) / risk
    else:
        runner_r = (entry_price - runner_exit) / risk
    total_r = partial_r + runner_fraction * runner_r
    return float(total_r), pd.Timestamp(close_times[last]), "hybrid_time"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Counterfactual target simulations for Turtle Soup indexed OOS trades.")
    parser.add_argument(
        "--indexed-trades",
        type=Path,
        default=Path("scripts/turtle_soup_core3_1h_bfm_rf050_oos_trades.csv"),
    )
    parser.add_argument(
        "--feature-dataset",
        type=Path,
        default=Path("scripts/trade_outcome_dataset_core3_1h_bfm_line_channel_ordered_probe.csv"),
    )
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--warmup-start", default="2021-09-01")
    parser.add_argument("--end", default="2026-04-20")
    parser.add_argument("--cache-dir", type=Path, default=Path("scripts/.cache"))
    parser.add_argument("--max-hold-bars", type=int, default=120)
    parser.add_argument("--fixed-rrs", default="1.0,1.5,2.0,2.5,3.0")
    parser.add_argument("--stop-multipliers", default="0.5,0.75,1.0", help="Scale original stop distance by these multipliers.")
    parser.add_argument("--hybrid-first-rr", type=float, default=1.0)
    parser.add_argument("--hybrid-runner-fraction", type=float, default=0.5)
    parser.add_argument("--opp-min-rr", type=float, default=1.0)
    parser.add_argument("--opp-max-rr", type=float, default=4.0)
    parser.add_argument("--out-prefix", type=Path, default=Path("scripts/turtle_soup_target_variant_sim"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.indexed_trades.exists():
        raise SystemExit(f"Indexed trades file not found: {args.indexed_trades}")
    if not args.feature_dataset.exists():
        raise SystemExit(f"Feature dataset not found: {args.feature_dataset}")

    trades = pd.read_csv(args.indexed_trades)
    for col in ["entry_time", "exit_time", "sweep_time", "signal_time"]:
        if col in trades.columns:
            trades[col] = pd.to_datetime(trades[col], utc=True, errors="coerce")
    if "event_key" not in trades.columns:
        trades["event_key"] = [
            zone_key(str(row.symbol).upper(), str(row.direction), row.sweep_time, float(row.zone_top), float(row.zone_bottom))
            for row in trades.itertuples(index=False)
        ]

    features = pd.read_csv(args.feature_dataset, usecols=[
        "event_key",
        "entry_risk_atr",
        "bfm_signal_opp_close_dist_atr",
        "trade_win_prob",
    ])
    features = features.drop_duplicates("event_key")
    trades = trades.merge(features, on="event_key", how="left")

    warmup_start = pd.Timestamp(args.warmup_start, tz="UTC") if pd.Timestamp(args.warmup_start).tzinfo is None else pd.Timestamp(args.warmup_start)
    end = pd.Timestamp(args.end, tz="UTC") if pd.Timestamp(args.end).tzinfo is None else pd.Timestamp(args.end)

    fixed_rrs = parse_float_list(args.fixed_rrs)
    stop_multipliers = parse_float_list(args.stop_multipliers)
    sim_rows: list[dict] = []

    symbol_frames: dict[str, pd.DataFrame] = {}
    symbol_time_to_index: dict[str, dict[pd.Timestamp, int]] = {}
    for symbol in sorted(trades["symbol"].dropna().astype(str).unique()):
        tv_symbol = f"BINANCE:{symbol}"
        cache = ensure_cache(tv_symbol, args.interval, warmup_start.to_pydatetime(), end.to_pydatetime(), args.cache_dir)
        df = pd.read_pickle(cache)
        df = df.reset_index(drop=True)
        symbol_frames[symbol] = df
        open_times = pd.to_datetime(df["open_time"], utc=True, errors="coerce")
        symbol_time_to_index[symbol] = {pd.Timestamp(ts): int(i) for i, ts in enumerate(open_times)}

    for row in trades.itertuples(index=False):
        symbol = str(row.symbol)
        if symbol not in symbol_frames:
            continue
        df = symbol_frames[symbol]
        entry_idx = symbol_time_to_index[symbol].get(pd.Timestamp(row.entry_time), -1)
        if entry_idx < 0 and pd.notna(row.entry_index):
            entry_idx = int(row.entry_index)
        if entry_idx < 0 or entry_idx >= len(df):
            continue

        opens = df["open"].astype(float).to_list()
        highs = df["high"].astype(float).to_list()
        lows = df["low"].astype(float).to_list()
        closes = df["close"].astype(float).to_list()
        close_times = pd.to_datetime(df["close_time"], utc=True).to_list()

        entry_price = float(row.entry_price)
        base_stop_price = float(row.stop_price)
        base_risk = abs(entry_price - base_stop_price)
        if base_risk <= 0:
            continue

        for stop_mult in stop_multipliers:
            if stop_mult <= 0:
                continue
            risk = base_risk * stop_mult
            if row.direction == "long":
                stop_price = entry_price - risk
            else:
                stop_price = entry_price + risk

            for rr in fixed_rrs:
                sign = 1.0 if row.direction == "long" else -1.0
                target = entry_price + sign * risk * rr
                r_val, exit_time, exit_reason = simulate_single_target(
                    direction=str(row.direction),
                    entry_price=entry_price,
                    stop_price=stop_price,
                    target_price=target,
                    risk=risk,
                    entry_index=entry_idx,
                    max_hold_bars=int(args.max_hold_bars),
                    opens=opens,
                    highs=highs,
                    lows=lows,
                    closes=closes,
                    close_times=close_times,
                )
                sim_rows.append({
                    "symbol": symbol,
                    "variant": f"sl_{stop_mult:g}x|fixed_rr_{rr:g}",
                    "stop_multiplier": stop_mult,
                    "entry_time": row.entry_time,
                    "exit_time": exit_time,
                    "direction": row.direction,
                    "r_multiple": r_val,
                    "exit_reason": exit_reason,
                })

            if pd.notna(row.entry_risk_atr) and float(row.entry_risk_atr) > 0 and pd.notna(row.bfm_signal_opp_close_dist_atr):
                # Keep opposite-side target in price space constant across stop multipliers.
                rr_proxy = float(row.bfm_signal_opp_close_dist_atr) / (float(row.entry_risk_atr) * stop_mult)
                rr_proxy = max(float(args.opp_min_rr), min(float(args.opp_max_rr), rr_proxy))
                sign = 1.0 if row.direction == "long" else -1.0
                opp_target = entry_price + sign * risk * rr_proxy

                r_val, exit_time, exit_reason = simulate_single_target(
                    direction=str(row.direction),
                    entry_price=entry_price,
                    stop_price=stop_price,
                    target_price=opp_target,
                    risk=risk,
                    entry_index=entry_idx,
                    max_hold_bars=int(args.max_hold_bars),
                    opens=opens,
                    highs=highs,
                    lows=lows,
                    closes=closes,
                    close_times=close_times,
                )
                sim_rows.append({
                    "symbol": symbol,
                    "variant": f"sl_{stop_mult:g}x|opp_proxy_target",
                    "stop_multiplier": stop_mult,
                    "entry_time": row.entry_time,
                    "exit_time": exit_time,
                    "direction": row.direction,
                    "r_multiple": r_val,
                    "exit_reason": exit_reason,
                })

                hybrid_r, hybrid_exit_time, hybrid_reason = simulate_hybrid(
                    direction=str(row.direction),
                    entry_price=entry_price,
                    stop_price=stop_price,
                    risk=risk,
                    entry_index=entry_idx,
                    max_hold_bars=int(args.max_hold_bars),
                    rr1=float(args.hybrid_first_rr),
                    rr2=rr_proxy,
                    runner_fraction=float(args.hybrid_runner_fraction),
                    opens=opens,
                    highs=highs,
                    lows=lows,
                    closes=closes,
                    close_times=close_times,
                )
                sim_rows.append({
                    "symbol": symbol,
                    "variant": f"sl_{stop_mult:g}x|hybrid_1r_plus_opp_proxy",
                    "stop_multiplier": stop_mult,
                    "entry_time": row.entry_time,
                    "exit_time": hybrid_exit_time,
                    "direction": row.direction,
                    "r_multiple": hybrid_r,
                    "exit_reason": hybrid_reason,
                })

    simulated = pd.DataFrame(sim_rows)
    if simulated.empty:
        raise SystemExit("No simulated rows were generated.")

    summary_rows: list[dict] = []
    for variant, part in simulated.groupby("variant"):
        summary_rows.append({"variant": variant, **summarize(part)})
    summary = pd.DataFrame(summary_rows).sort_values(["profit_factor", "net_r"], ascending=[False, False])

    summary_by_symbol_rows: list[dict] = []
    for (variant, symbol), part in simulated.groupby(["variant", "symbol"]):
        summary_by_symbol_rows.append({
            "variant": variant,
            "symbol": symbol,
            "stop_multiplier": float(part["stop_multiplier"].iloc[0]) if "stop_multiplier" in part.columns else 1.0,
            **summarize(part),
        })
    summary_by_symbol = pd.DataFrame(summary_by_symbol_rows).sort_values(["variant", "net_r"], ascending=[True, False])

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    trades_path = args.out_prefix.with_name(f"{args.out_prefix.name}_trades.csv")
    summary_path = args.out_prefix.with_name(f"{args.out_prefix.name}_summary.csv")
    symbol_path = args.out_prefix.with_name(f"{args.out_prefix.name}_summary_by_symbol.csv")
    simulated.to_csv(trades_path, index=False)
    summary.to_csv(summary_path, index=False)
    summary_by_symbol.to_csv(symbol_path, index=False)

    print("Variant summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved: {trades_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {symbol_path}")


if __name__ == "__main__":
    main()
