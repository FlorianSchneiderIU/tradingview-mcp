"""Test Power Order Blocks as a *confluence filter* on an independent SMC setup.

Standalone, the ported displacement OB had no durable OOS edge (see study_power_order_blocks).
The remaining honest question: does requiring an *aligned, still-active Power OB at the entry*
improve an independent, already-built SMC setup? We use the breaker zone-retest engine
(`simulate_breakers`) as that independent setup, tag every trade post-hoc with whether its
entry sits inside an aligned Power OB that is alive at entry time, then compare the
confluence subset vs the non-confluence subset vs all -- train/OOS.

A Power OB is "alive" from the close of its displacement bar until price closes through its
far edge (the indicator's own deletion rule) or it ages out. Tagging is pure post-processing
on the trade list, so it cannot leak look-ahead into entries -- it only partitions them.

Example:
    python scripts/study_power_ob_confluence.py --zone-tf 4h --ob-tf 1h
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backtest_turtle_soup import normalize_binance_spot_symbol, parse_utc_datetime, resample_ohlc
from scripts.crypto_symbol_sets import SYMBOL_SETS, expand_symbol_args
from scripts.study_breaker_continuation import BreakerConfig, ensure_cache, frame_metrics, simulate_breakers


def power_ob_intervals(
    df: pd.DataFrame,
    timeframe: str,
    disp_thresh: float,
    str_lookback: int,
    max_age_bars: int,
) -> list[dict[str, Any]]:
    """Return live Power-OB intervals: dicts with direction, top, bottom, power, start_time,
    end_time. ``end_time`` is the close_time of the bar that closed through the far edge, or
    None if the OB never died within ``max_age_bars`` (treated as alive to the age cap)."""
    tf = resample_ohlc(df, timeframe).reset_index(drop=True)
    o = tf["open"].astype(float).to_list()
    h = tf["high"].astype(float).to_list()
    l = tf["low"].astype(float).to_list()
    c = tf["close"].astype(float).to_list()
    ct = tf["close_time"].to_list()
    n = len(tf)
    rng = (tf["high"].astype(float) - tf["low"].astype(float))
    max_range = rng.rolling(str_lookback, min_periods=1).max().to_list()

    out: list[dict[str, Any]] = []
    for i in range(1, n):
        prior_range = h[i - 1] - l[i - 1]
        top = h[i - 1]
        bottom = l[i - 1]
        denom = max_range[i]
        power = ((top - bottom) / denom) * 100.0 if denom > 0 else 0.0

        bull = (c[i - 1] < o[i - 1] and c[i] > o[i] and c[i] > h[i - 1]
                and (c[i] - o[i]) > prior_range * disp_thresh)
        bear = (c[i - 1] > o[i - 1] and c[i] < o[i] and c[i] < l[i - 1]
                and (o[i] - c[i]) > prior_range * disp_thresh)
        if not (bull or bear):
            continue
        direction = "long" if bull else "short"

        end_time = None
        last = min(n - 1, i + max_age_bars)
        for j in range(i + 1, last + 1):
            if direction == "long" and c[j] < bottom:
                end_time = ct[j]
                break
            if direction == "short" and c[j] > top:
                end_time = ct[j]
                break
        else:
            # never closed through within the window -> alive until age cap
            end_time = ct[last] if last > i else None

        out.append({
            "direction": direction,
            "top": top,
            "bottom": bottom,
            "power": power,
            "start_time": pd.Timestamp(ct[i]),
            "end_time": pd.Timestamp(end_time) if end_time is not None else None,
        })
    return out


def tag_trades(trades: pd.DataFrame, intervals: list[dict[str, Any]], tol_frac: float) -> pd.DataFrame:
    """Add ob_confluence (bool) and ob_power (max aligned power) to each trade."""
    if trades.empty:
        return trades
    trades = trades.copy()
    entry_times = pd.to_datetime(trades["entry_time"], utc=True)
    confl: list[bool] = []
    powers: list[float] = []
    for et, direction, price in zip(entry_times, trades["direction"], trades["entry_price"].astype(float)):
        best_power = float("nan")
        found = False
        for iv in intervals:
            if iv["direction"] != direction:
                continue
            if iv["start_time"] > et:
                continue
            if iv["end_time"] is not None and et >= iv["end_time"]:
                continue
            tol = tol_frac * (iv["top"] - iv["bottom"])
            if (iv["bottom"] - tol) <= price <= (iv["top"] + tol):
                found = True
                if not (best_power == best_power) or iv["power"] > best_power:  # nan-safe max
                    best_power = iv["power"]
        confl.append(found)
        powers.append(best_power)
    trades["ob_confluence"] = confl
    trades["ob_power"] = powers
    return trades


def windowed(trades: pd.DataFrame, split, end) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = trades.copy()
    t["entry_time"] = pd.to_datetime(t["entry_time"], utc=True)
    train = t[t["entry_time"] < pd.Timestamp(split)]
    oos = t[(t["entry_time"] >= pd.Timestamp(split)) & (t["entry_time"] < pd.Timestamp(end))]
    return train, oos


def main() -> None:
    parser = argparse.ArgumentParser(description="Power OB confluence filter on breaker SMC trades.")
    parser.add_argument("--symbols", nargs="+", default=[])
    parser.add_argument("--symbol-set", choices=["none", *SYMBOL_SETS.keys()], default="core3")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--start", default="2024-04-20")
    parser.add_argument("--split", default="2025-04-20")
    parser.add_argument("--end", default="2026-04-20")
    parser.add_argument("--cache-dir", type=Path, default=Path("scripts/.cache"))
    parser.add_argument("--output", type=Path, default=Path("scripts/power_ob_confluence_trades.csv"))
    # Breaker base setup (defaults match study_breaker_continuation)
    parser.add_argument("--entry-mode", choices=["zone_retest", "structure_fvg", "fvg_print"], default="zone_retest")
    parser.add_argument("--zone-tf", default="4h")
    parser.add_argument("--confirmation-tf", default="15m")
    parser.add_argument("--target-rr", type=float, default=2.0)
    parser.add_argument("--min-reject-pos", type=float, default=0.50)
    # Power OB confluence params
    parser.add_argument("--ob-tf", default="1h")
    parser.add_argument("--disp-thresh", type=float, default=0.5)
    parser.add_argument("--str-lookback", type=int, default=100)
    parser.add_argument("--ob-max-age-bars", type=int, default=300)
    parser.add_argument("--tol-frac", type=float, default=0.5, help="Entry may sit within tol_frac*zone_width outside the OB edges.")
    parser.add_argument("--min-power", type=float, default=0.0, help="Require aligned OB power >= this for confluence.")
    args = parser.parse_args()

    symbols = expand_symbol_args(args.symbols, args.symbol_set)
    start = parse_utc_datetime(args.start)
    split = parse_utc_datetime(args.split)
    end = parse_utc_datetime(args.end)

    cfg = BreakerConfig(
        entry_mode=args.entry_mode, zone_tf=args.zone_tf, confirmation_tf=args.confirmation_tf,
        structure_left=2, structure_right=2, htf_left=5, htf_right=5, htf_ob_search_bars=50,
        max_zone_scan=250, max_retest_bars=288, max_confirm_bars=72, max_hold_bars=120,
        stop_buffer_atr=0.10, target_rr=args.target_rr, min_reject_pos=args.min_reject_pos,
        min_confirm_fvg_atr=0.0, min_entry_risk_pct=0.0,
    )

    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        cache = ensure_cache(symbol, args.interval, start, end, args.cache_dir)
        df = pd.read_pickle(cache)
        before = time.time()
        trades = simulate_breakers(symbol, df, cfg)
        intervals = power_ob_intervals(df, args.ob_tf, args.disp_thresh, args.str_lookback, args.ob_max_age_bars)
        trades = tag_trades(trades, intervals, args.tol_frac)
        if args.min_power > 0 and not trades.empty:
            trades["ob_confluence"] = trades["ob_confluence"] & (trades["ob_power"] >= args.min_power)
        frames.append(trades)
        conf_n = int(trades["ob_confluence"].sum()) if not trades.empty else 0
        print(f"  {normalize_binance_spot_symbol(symbol)}: {len(trades)} breaker trades, "
              f"{len(intervals)} OBs, {conf_n} with confluence ({time.time()-before:.1f}s)")

    trades = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if trades.empty:
        print("No trades generated.")
        return

    train, oos = windowed(trades, split, end)
    rows: list[dict[str, Any]] = []
    for win_name, win in (("train", train), ("oos", oos)):
        rows.append({"bucket": "ALL", "window": win_name, **frame_metrics(win)})
        rows.append({"bucket": "confluence", "window": win_name, **frame_metrics(win[win["ob_confluence"]])})
        rows.append({"bucket": "no_confluence", "window": win_name, **frame_metrics(win[~win["ob_confluence"]])})

    summary = pd.DataFrame(rows)
    print("\n" + "=" * 92)
    print(f"Power OB confluence on breaker {args.entry_mode} (zone {args.zone_tf}, OB {args.ob_tf}, "
          f"tol={args.tol_frac}, min_power={args.min_power})")
    print(f"train < {args.split} <= oos < {args.end}")
    print("=" * 92)
    print(summary.to_string(index=False))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(args.output, index=False)
    summary.to_csv(args.output.with_name(args.output.stem + "_summary.csv"), index=False)
    print(f"\nSaved {len(trades)} tagged trades to {args.output}")


if __name__ == "__main__":
    main()
