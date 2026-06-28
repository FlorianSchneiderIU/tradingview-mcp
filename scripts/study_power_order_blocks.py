"""Evaluate the ported "Power Order Blocks" indicator inside our SMC framework.

The indicator's thesis is the classic order-block bounce: an OB acts as support/resistance,
and price *retesting* it gives a high-quality entry. We test that thesis honestly by running
two different OB *definitions* through one identical bounce engine (retest -> stop beyond the
zone -> fixed-RR target), so the only thing that changes is the zone definition:

    baseline : build_htf_zone_events  -- our current BOS-anchored "extreme candle" OB
    power    : build_power_ob_events  -- the ported single-candle displacement OB

We also sweep the ported indicator's "power %" score as a quality filter to see whether the
indicator's own strength metric separates good OBs from bad ones.

Entry/exit machinery (gap handling, R-multiple, metrics) is reused verbatim from
study_breaker_continuation so results are comparable to the rest of the SMC research.

Example:
    python scripts/study_power_order_blocks.py --interval 5m --zone-tf 1h
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backtest_turtle_soup import (
    add_atr,
    build_htf_zone_events,
    normalize_binance_spot_symbol,
    parse_utc_datetime,
)
from scripts.crypto_symbol_sets import SYMBOL_SETS, expand_symbol_args
from scripts.power_order_blocks import build_power_ob_events
from scripts.study_breaker_continuation import (
    close_trade,
    ensure_cache,
    exit_for_bar,
    frame_metrics,
)


@dataclass
class OBBounceConfig:
    stop_buffer_atr: float
    target_rr: float
    max_hold_bars: int
    max_retest_bars: int
    min_reject_pos: float
    min_entry_risk_pct: float
    max_zone_scan: int
    min_power: float
    invalidate_close_through: bool


ZoneSource = Callable[[pd.DataFrame], tuple[list[dict[str, Any]], list[dict[str, Any]]]]


def simulate_ob_bounce(
    symbol: str,
    df: pd.DataFrame,
    supply_events: list[dict[str, Any]],
    demand_events: list[dict[str, Any]],
    cfg: OBBounceConfig,
) -> pd.DataFrame:
    """Classic OB-bounce simulator: long on a held retest of a demand OB, short on supply.

    Single position at a time; entry fills at the next bar's open (no look-ahead). A zone is
    consumed after it triggers one trade, dropped after ``max_retest_bars``, and (optionally)
    invalidated when price closes through its far edge -- faithful to the indicator's own
    block-deletion rule (bull block dies when close < bottom, bear when close > top).
    """
    prepared = add_atr(df.sort_values("open_time").reset_index(drop=True).copy())
    normalized = normalize_binance_spot_symbol(symbol)
    opens = prepared["open"].astype(float).to_list()
    highs = prepared["high"].astype(float).to_list()
    lows = prepared["low"].astype(float).to_list()
    closes = prepared["close"].astype(float).to_list()
    atrs = prepared["atr"].bfill().ffill().to_list()
    times = prepared["open_time"].to_list()
    close_times = prepared["close_time"].to_list()

    # Power filter only applies when events carry a "power" field (Power OB source).
    def passes_power(ev: dict[str, Any]) -> bool:
        if cfg.min_power <= 0:
            return True
        return float(ev.get("power", math.inf)) >= cfg.min_power

    demand_sorted = sorted((e for e in demand_events if passes_power(e)), key=lambda e: pd.Timestamp(e["time"]))
    supply_sorted = sorted((e for e in supply_events if passes_power(e)), key=lambda e: pd.Timestamp(e["time"]))

    demand_ptr = 0
    supply_ptr = 0
    demand_zones: list[dict[str, Any]] = []
    supply_zones: list[dict[str, Any]] = []
    pending_entry: dict[str, Any] | None = None
    position: dict[str, Any] | None = None
    trades: list[dict[str, Any]] = []

    def add_zone(event: dict[str, Any], side: str, idx: int) -> dict[str, Any]:
        return {**event, "side": side, "used": False, "add_index": idx}

    def candidates(zones: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = [z for z in reversed(zones) if not z["used"]]
        return out[: cfg.max_zone_scan] if cfg.max_zone_scan > 0 else out

    for i in range(len(prepared)):
        visible_time = pd.Timestamp(close_times[i])

        # Fill a queued entry at this bar's open.
        if pending_entry is not None and pending_entry["submitted_index"] < i and position is None:
            entry_price = opens[i]
            direction = pending_entry["direction"]
            risk = entry_price - pending_entry["stop_price"] if direction == "long" else pending_entry["stop_price"] - entry_price
            risk_pct = risk / entry_price * 100.0 if entry_price > 0 else math.nan
            if risk > 0 and risk_pct >= cfg.min_entry_risk_pct:
                target = entry_price + risk * cfg.target_rr if direction == "long" else entry_price - risk * cfg.target_rr
                position = {
                    **pending_entry,
                    "entry_index": i,
                    "entry_time": pd.Timestamp(times[i]),
                    "entry_price": entry_price,
                    "risk": risk,
                    "target_price": target,
                    "max_hold_bars": cfg.max_hold_bars,
                }
            pending_entry = None

        # Manage open position.
        if position is not None and i > position["entry_index"]:
            exit_value = exit_for_bar(position, prepared.iloc[i])
            if exit_value is not None:
                exit_price, reason = exit_value
                trades.append(close_trade(position, i, pd.Timestamp(times[i]), float(exit_price), reason))
                position = None

        # Reveal zones whose displacement candle has closed.
        while demand_ptr < len(demand_sorted) and pd.Timestamp(demand_sorted[demand_ptr]["time"]) <= visible_time:
            demand_zones.append(add_zone(demand_sorted[demand_ptr], "demand", i))
            demand_ptr += 1
        while supply_ptr < len(supply_sorted) and pd.Timestamp(supply_sorted[supply_ptr]["time"]) <= visible_time:
            supply_zones.append(add_zone(supply_sorted[supply_ptr], "supply", i))
            supply_ptr += 1

        # Invalidate / expire zones.
        def keep_demand(z: dict[str, Any]) -> bool:
            if z["used"]:
                return False
            if i - z["add_index"] > cfg.max_retest_bars:
                return False
            if cfg.invalidate_close_through and closes[i] < float(z["bottom"]):
                return False
            return True

        def keep_supply(z: dict[str, Any]) -> bool:
            if z["used"]:
                return False
            if i - z["add_index"] > cfg.max_retest_bars:
                return False
            if cfg.invalidate_close_through and closes[i] > float(z["top"]):
                return False
            return True

        demand_zones = [z for z in demand_zones if keep_demand(z)]
        supply_zones = [z for z in supply_zones if keep_supply(z)]

        # Look for a held retest -> queue entry for next open.
        candle_range = highs[i] - lows[i]
        if position is None and pending_entry is None and candle_range > 0:
            for zone in candidates(demand_zones):
                if i <= zone["add_index"]:
                    continue
                top = float(zone["top"])
                bottom = float(zone["bottom"])
                reject_pos = (closes[i] - lows[i]) / candle_range
                if lows[i] <= top and closes[i] > bottom and reject_pos >= cfg.min_reject_pos:
                    atr = atrs[i]
                    if math.isfinite(atr) and atr > 0:
                        zone["used"] = True
                        pending_entry = {
                            "symbol": normalized,
                            "direction": "long",
                            "entry_mode": "power_ob_bounce",
                            "zone_top": top,
                            "zone_bottom": bottom,
                            "zone_power": float(zone.get("power", math.nan)),
                            "retest_time": pd.Timestamp(times[i]),
                            "retest_reject_pos": reject_pos,
                            "stop_price": bottom - atr * cfg.stop_buffer_atr,
                            "submitted_index": i,
                        }
                    break

            if pending_entry is None:
                for zone in candidates(supply_zones):
                    if i <= zone["add_index"]:
                        continue
                    top = float(zone["top"])
                    bottom = float(zone["bottom"])
                    reject_pos = (highs[i] - closes[i]) / candle_range
                    if highs[i] >= bottom and closes[i] < top and reject_pos >= cfg.min_reject_pos:
                        atr = atrs[i]
                        if math.isfinite(atr) and atr > 0:
                            zone["used"] = True
                            pending_entry = {
                                "symbol": normalized,
                                "direction": "short",
                                "entry_mode": "power_ob_bounce",
                                "zone_top": top,
                                "zone_bottom": bottom,
                                "zone_power": float(zone.get("power", math.nan)),
                                "retest_time": pd.Timestamp(times[i]),
                                "retest_reject_pos": reject_pos,
                                "stop_price": top + atr * cfg.stop_buffer_atr,
                                "submitted_index": i,
                            }
                        break

    return pd.DataFrame(trades)


def run_source(
    label: str,
    symbols: list[str],
    interval: str,
    zone_tf: str,
    source: ZoneSource,
    cfg: OBBounceConfig,
    cache_dir: Path,
    start,
    split,
    end,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        cache = ensure_cache(symbol, interval, start, end, cache_dir)
        df = pd.read_pickle(cache)
        before = time.time()
        supply_events, demand_events = source(df)
        trades = simulate_ob_bounce(symbol, df, supply_events, demand_events, cfg)
        frames.append(trades)
        print(f"  [{label}] {normalize_binance_spot_symbol(symbol)}: "
              f"{len(supply_events)+len(demand_events)} zones -> {len(trades)} trades "
              f"({time.time()-before:.1f}s)")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize(label: str, trades: pd.DataFrame, split, end) -> list[dict[str, Any]]:
    if trades.empty:
        return [{"source": label, "window": "train", "trades": 0},
                {"source": label, "window": "oos", "trades": 0}]
    trades = trades.copy()
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    train = trades[trades["entry_time"] < pd.Timestamp(split)]
    oos = trades[(trades["entry_time"] >= pd.Timestamp(split)) & (trades["entry_time"] < pd.Timestamp(end))]
    return [
        {"source": label, "window": "train", **frame_metrics(train)},
        {"source": label, "window": "oos", **frame_metrics(oos)},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Test ported Power Order Blocks in the SMC context.")
    parser.add_argument("--symbols", nargs="+", default=[])
    parser.add_argument("--symbol-set", choices=["none", *SYMBOL_SETS.keys()], default="core3")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--zone-tf", default="1h")
    parser.add_argument("--start", default="2024-04-20")
    parser.add_argument("--split", default="2025-04-20")
    parser.add_argument("--end", default="2026-04-20")
    parser.add_argument("--cache-dir", type=Path, default=Path("scripts/.cache"))
    parser.add_argument("--output", type=Path, default=Path("scripts/power_ob_trades.csv"))
    # Indicator params
    parser.add_argument("--disp-thresh", type=float, default=0.5)
    parser.add_argument("--str-lookback", type=int, default=100)
    # HTF baseline OB params (matches breaker defaults)
    parser.add_argument("--htf-left", type=int, default=5)
    parser.add_argument("--htf-right", type=int, default=5)
    parser.add_argument("--htf-ob-search-bars", type=int, default=50)
    # Bounce engine params
    parser.add_argument("--stop-buffer-atr", type=float, default=0.10)
    parser.add_argument("--target-rr", type=float, default=2.0)
    parser.add_argument("--max-hold-bars", type=int, default=120)
    parser.add_argument("--max-retest-bars", type=int, default=288)
    parser.add_argument("--min-reject-pos", type=float, default=0.50)
    parser.add_argument("--min-entry-risk-pct", type=float, default=0.0)
    parser.add_argument("--max-zone-scan", type=int, default=250)
    parser.add_argument("--no-invalidate", action="store_true", help="Disable close-through invalidation.")
    parser.add_argument("--power-sweep", default="0,40,60,80",
                        help="Comma list of min-power thresholds to sweep for the Power OB source.")
    args = parser.parse_args()

    symbols = expand_symbol_args(args.symbols, args.symbol_set)
    start = parse_utc_datetime(args.start)
    split = parse_utc_datetime(args.split)
    end = parse_utc_datetime(args.end)

    def base_cfg(min_power: float) -> OBBounceConfig:
        return OBBounceConfig(
            stop_buffer_atr=args.stop_buffer_atr,
            target_rr=args.target_rr,
            max_hold_bars=args.max_hold_bars,
            max_retest_bars=args.max_retest_bars,
            min_reject_pos=args.min_reject_pos,
            min_entry_risk_pct=args.min_entry_risk_pct,
            max_zone_scan=args.max_zone_scan,
            min_power=min_power,
            invalidate_close_through=not args.no_invalidate,
        )

    htf_source: ZoneSource = lambda df: build_htf_zone_events(
        df, args.zone_tf, args.htf_left, args.htf_right, 0.25, args.htf_ob_search_bars, False
    )
    power_source: ZoneSource = lambda df: build_power_ob_events(
        df, args.zone_tf, args.disp_thresh, args.str_lookback
    )

    rows: list[dict[str, Any]] = []
    power_thresholds = [float(x) for x in str(args.power_sweep).split(",") if x.strip() != ""]

    print(f"== Baseline: BOS-anchored SMC OB (build_htf_zone_events) @ {args.zone_tf} ==")
    baseline = run_source("baseline", symbols, args.interval, args.zone_tf, htf_source, base_cfg(0.0),
                          args.cache_dir, start, split, end)
    rows += summarize("baseline_htf_ob", baseline, split, end)

    all_power_trades: list[pd.DataFrame] = []
    for thr in power_thresholds:
        print(f"== Power OB (displacement) @ {args.zone_tf}, min_power={thr} ==")
        pt = run_source(f"power>={thr}", symbols, args.interval, args.zone_tf, power_source, base_cfg(thr),
                        args.cache_dir, start, split, end)
        if not pt.empty:
            pt = pt.copy()
            pt["min_power"] = thr
            all_power_trades.append(pt)
        rows += summarize(f"power_ob_min{int(thr)}", pt, split, end)

    summary = pd.DataFrame(rows)
    print("\n" + "=" * 88)
    print("SUMMARY  (train < {} <= oos < {})".format(args.split, args.end))
    print("=" * 88)
    print(summary.to_string(index=False))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_frames = []
    if not baseline.empty:
        b = baseline.copy(); b["source"] = "baseline_htf_ob"; out_frames.append(b)
    for pt in all_power_trades:
        pt = pt.copy(); pt["source"] = "power_ob"; out_frames.append(pt)
    if out_frames:
        pd.concat(out_frames, ignore_index=True).to_csv(args.output, index=False)
        print(f"\nSaved trades to {args.output}")
    summary.to_csv(args.output.with_name(args.output.stem + "_summary.csv"), index=False)


if __name__ == "__main__":
    main()
