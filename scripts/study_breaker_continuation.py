from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backtest_turtle_soup import (
    add_atr,
    build_confirmed_pivots,
    build_htf_zone_events,
    fetch_klines,
    normalize_binance_spot_symbol,
    parse_utc_datetime,
    resample_ohlc,
)
from scripts.crypto_symbol_sets import SYMBOL_SETS, expand_symbol_args


@dataclass
class BreakerConfig:
    entry_mode: str
    zone_tf: str
    confirmation_tf: str
    structure_left: int
    structure_right: int
    htf_left: int
    htf_right: int
    htf_ob_search_bars: int
    max_zone_scan: int
    max_retest_bars: int
    max_confirm_bars: int
    max_hold_bars: int
    stop_buffer_atr: float
    target_rr: float
    min_reject_pos: float
    min_confirm_fvg_atr: float
    min_entry_risk_pct: float


def _to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def ensure_cache(symbol: str, interval: str, start: datetime, end: datetime, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    requested_symbol = normalize_binance_spot_symbol(symbol).lower()
    for candidate in sorted(cache_dir.glob(f"{requested_symbol}_{interval}_*.pkl")):
        try:
            df = pd.read_pickle(candidate)
        except Exception:
            continue
        if df.empty:
            continue
        if df["open_time"].iloc[0].to_pydatetime() <= start and df["close_time"].iloc[-1].to_pydatetime() >= end:
            return candidate

    path = cache_dir / f"{requested_symbol}_{interval}_{start:%Y%m%d}_{end:%Y%m%d}.pkl"
    if path.exists():
        return path
    df = fetch_klines(symbol, interval, _to_ms(start), _to_ms(end))
    df.to_pickle(path)
    return path


def tradingview_high_before_low(open_val: float, high_val: float, low_val: float) -> bool:
    return abs(open_val - high_val) < abs(open_val - low_val)


def profit_factor(rs: pd.Series) -> float:
    wins = rs[rs > 0]
    losses = rs[rs <= 0]
    gross_loss = abs(float(losses.sum()))
    if gross_loss == 0:
        return float("inf") if len(wins) else 0.0
    return float(wins.sum()) / gross_loss


def max_drawdown_r(values: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)
    return round(max_dd, 3)


def frame_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "net_r": 0.0, "avg_r": 0.0, "max_dd_r": 0.0}
    rs = frame.sort_values("exit_time")["r_multiple"].astype(float)
    return {
        "trades": int(len(frame)),
        "win_rate": round(100.0 * float((rs > 0).mean()), 2),
        "profit_factor": round(profit_factor(rs), 3),
        "net_r": round(float(rs.sum()), 3),
        "avg_r": round(float(rs.mean()), 3),
        "max_dd_r": max_drawdown_r(rs.to_list()),
    }


def build_fvg_events(exec_df: pd.DataFrame, timeframe: str) -> list[dict[str, Any]]:
    candles = resample_ohlc(exec_df, timeframe)
    events: list[dict[str, Any]] = []
    for i in range(2, len(candles)):
        row = candles.iloc[i]
        two_back = candles.iloc[i - 2]
        if float(row["low"]) > float(two_back["high"]):
            events.append({
                "time": row["close_time"],
                "direction": "long",
                "kind": "fvg_print",
                "break_level": float(row["high"]),
                "has_fvg": True,
                "fvg_top": float(row["low"]),
                "fvg_bottom": float(two_back["high"]),
                "fvg_height": float(row["low"]) - float(two_back["high"]),
            })
        if float(row["high"]) < float(two_back["low"]):
            events.append({
                "time": row["close_time"],
                "direction": "short",
                "kind": "fvg_print",
                "break_level": float(row["low"]),
                "has_fvg": True,
                "fvg_top": float(two_back["low"]),
                "fvg_bottom": float(row["high"]),
                "fvg_height": float(two_back["low"]) - float(row["high"]),
            })
    return sorted(events, key=lambda item: item["time"])


def build_structure_break_events(exec_df: pd.DataFrame, timeframe: str, left: int, right: int) -> list[dict[str, Any]]:
    structure = resample_ohlc(exec_df, timeframe)
    ph_conf: list[float | None] = [None] * len(structure)
    pl_conf: list[float | None] = [None] * len(structure)

    for item in build_confirmed_pivots(structure["high"], left, right, "high"):
        confirm_idx = item["pivot_index"] + right
        if confirm_idx < len(structure):
            ph_conf[confirm_idx] = item["value"]

    for item in build_confirmed_pivots(structure["low"], left, right, "low"):
        confirm_idx = item["pivot_index"] + right
        if confirm_idx < len(structure):
            pl_conf[confirm_idx] = item["value"]

    active_high = math.nan
    active_low = math.nan
    high_crossed = False
    low_crossed = False
    closes = structure["close"].to_list()
    events: list[dict[str, Any]] = []

    for i in range(len(structure)):
        if ph_conf[i] is not None:
            active_high = float(ph_conf[i])
            high_crossed = False
        if pl_conf[i] is not None:
            active_low = float(pl_conf[i])
            low_crossed = False

        prev_close = closes[i - 1] if i > 0 else closes[i]
        bull_break = not math.isnan(active_high) and not high_crossed and closes[i] > active_high and prev_close <= active_high
        bear_break = not math.isnan(active_low) and not low_crossed and closes[i] < active_low and prev_close >= active_low

        if bull_break:
            fvg_top = math.nan
            fvg_bottom = math.nan
            has_fvg = False
            if i >= 2 and float(structure.iloc[i]["low"]) > float(structure.iloc[i - 2]["high"]):
                has_fvg = True
                fvg_top = float(structure.iloc[i]["low"])
                fvg_bottom = float(structure.iloc[i - 2]["high"])
            events.append({
                "time": structure.iloc[i]["close_time"],
                "direction": "long",
                "kind": "structure_break",
                "break_level": active_high,
                "has_fvg": has_fvg,
                "fvg_top": fvg_top,
                "fvg_bottom": fvg_bottom,
                "fvg_height": fvg_top - fvg_bottom if has_fvg else 0.0,
            })
            high_crossed = True

        if bear_break:
            fvg_top = math.nan
            fvg_bottom = math.nan
            has_fvg = False
            if i >= 2 and float(structure.iloc[i]["high"]) < float(structure.iloc[i - 2]["low"]):
                has_fvg = True
                fvg_top = float(structure.iloc[i - 2]["low"])
                fvg_bottom = float(structure.iloc[i]["high"])
            events.append({
                "time": structure.iloc[i]["close_time"],
                "direction": "short",
                "kind": "structure_break",
                "break_level": active_low,
                "has_fvg": has_fvg,
                "fvg_top": fvg_top,
                "fvg_bottom": fvg_bottom,
                "fvg_height": fvg_top - fvg_bottom if has_fvg else 0.0,
            })
            low_crossed = True

    return events


def exit_for_bar(position: dict[str, Any], row: pd.Series) -> tuple[float, str] | None:
    direction = position["direction"]
    open_val = float(row["open"])
    high_val = float(row["high"])
    low_val = float(row["low"])
    close_val = float(row["close"])
    stop = float(position["stop_price"])
    target = float(position["target_price"])
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

    if int(row.name) - int(position["entry_index"]) >= int(position["max_hold_bars"]):
        return close_val, "timeout"
    return None


def close_trade(position: dict[str, Any], exit_index: int, exit_time: pd.Timestamp, exit_price: float, exit_reason: str) -> dict[str, Any]:
    risk = float(position["risk"])
    if position["direction"] == "long":
        r_multiple = (exit_price - float(position["entry_price"])) / risk
    else:
        r_multiple = (float(position["entry_price"]) - exit_price) / risk
    base_keys = [
        "symbol",
        "direction",
        "entry_mode",
        "entry_time",
        "entry_price",
        "stop_price",
        "target_price",
        "zone_tf",
        "confirmation_tf",
        "zone_time",
        "zone_top",
        "zone_bottom",
        "break_time",
        "retest_time",
        "signal_time",
        "confirm_kind",
        "confirm_time",
        "confirm_break_level",
        "confirm_has_fvg",
        "confirm_fvg_top",
        "confirm_fvg_bottom",
        "confirm_fvg_height",
        "confirm_fvg_atr",
        "retest_reject_pos",
    ]
    return {
        **{key: position.get(key) for key in base_keys},
        "exit_time": exit_time,
        "exit_price": float(exit_price),
        "exit_reason": exit_reason,
        "exit_index": exit_index,
        "r_multiple": float(r_multiple),
    }


def simulate_breakers(
    symbol: str,
    df: pd.DataFrame,
    cfg: BreakerConfig,
    return_state: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    prepared = add_atr(df.sort_values("open_time").reset_index(drop=True).copy())
    supply_events, demand_events = build_htf_zone_events(
        prepared,
        cfg.zone_tf,
        cfg.htf_left,
        cfg.htf_right,
        0.25,
        cfg.htf_ob_search_bars,
        False,
    )
    if cfg.entry_mode == "structure_fvg":
        confirmation_events = build_structure_break_events(
            prepared,
            cfg.confirmation_tf,
            cfg.structure_left,
            cfg.structure_right,
        )
    elif cfg.entry_mode == "fvg_print":
        confirmation_events = build_fvg_events(prepared, cfg.confirmation_tf)
    else:
        confirmation_events = []

    normalized = normalize_binance_spot_symbol(symbol)
    opens = prepared["open"].to_list()
    highs = prepared["high"].to_list()
    lows = prepared["low"].to_list()
    closes = prepared["close"].to_list()
    atrs = prepared["atr"].bfill().ffill().to_list()
    times = prepared["open_time"].to_list()
    close_times = prepared["close_time"].to_list()

    demand_ptr = 0
    supply_ptr = 0
    demand_zones: list[dict[str, Any]] = []
    supply_zones: list[dict[str, Any]] = []
    pending_long: dict[str, Any] | None = None
    pending_short: dict[str, Any] | None = None
    pending_confirm: dict[str, Any] | None = None
    pending_entry: dict[str, Any] | None = None
    position: dict[str, Any] | None = None
    trades: list[dict[str, Any]] = []
    confirmation_ptr = 0

    def add_zone(event: dict[str, Any], side: str, seq: int) -> dict[str, Any]:
        return {**event, "side": side, "used": False, "id": f"{side}-{seq}-{pd.Timestamp(event['time']).isoformat()}"}

    def candidates(zones: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = [zone for zone in reversed(zones) if not zone["used"]]
        if cfg.max_zone_scan > 0:
            return out[: cfg.max_zone_scan]
        return out

    def make_setup(direction: str, zone: dict[str, Any], break_idx: int) -> dict[str, Any]:
        return {
            "symbol": normalized,
            "direction": direction,
            "entry_mode": cfg.entry_mode,
            "zone_tf": cfg.zone_tf,
            "confirmation_tf": cfg.confirmation_tf,
            "zone_time": pd.Timestamp(zone["time"]),
            "zone_top": float(zone["top"]),
            "zone_bottom": float(zone["bottom"]),
            "break_index": break_idx,
            "break_time": pd.Timestamp(times[break_idx]),
        }

    def mark_retest(setup: dict[str, Any], signal_idx: int, reject_pos: float) -> dict[str, Any]:
        return {
            **setup,
            "retest_index": signal_idx,
            "retest_time": pd.Timestamp(times[signal_idx]),
            "retest_reject_pos": float(reject_pos),
        }

    def submit_entry(setup: dict[str, Any], signal_idx: int, confirm_event: dict[str, Any] | None = None) -> dict[str, Any] | None:
        atr = atrs[signal_idx]
        if not math.isfinite(atr) or atr <= 0:
            return None
        if setup["direction"] == "long":
            stop = float(setup["zone_bottom"]) - atr * cfg.stop_buffer_atr
        else:
            stop = float(setup["zone_top"]) + atr * cfg.stop_buffer_atr
        confirm_fvg_height = float(confirm_event.get("fvg_height", 0.0)) if confirm_event else 0.0
        planned_entry = float(closes[signal_idx])
        planned_risk = planned_entry - stop if setup["direction"] == "long" else stop - planned_entry
        planned_target = (
            planned_entry + planned_risk * cfg.target_rr
            if setup["direction"] == "long"
            else planned_entry - planned_risk * cfg.target_rr
        ) if planned_risk > 0 else math.nan
        return {
            **setup,
            "signal_index": signal_idx,
            "signal_time": pd.Timestamp(times[signal_idx]),
            "planned_entry_price": planned_entry,
            "retest_time": setup.get("retest_time", pd.Timestamp(times[signal_idx])),
            "confirm_kind": confirm_event.get("kind") if confirm_event else "zone_retest",
            "confirm_time": pd.Timestamp(confirm_event["time"]) if confirm_event else pd.NaT,
            "confirm_break_level": float(confirm_event["break_level"]) if confirm_event else math.nan,
            "confirm_has_fvg": bool(confirm_event.get("has_fvg", False)) if confirm_event else False,
            "confirm_fvg_top": float(confirm_event.get("fvg_top", math.nan)) if confirm_event else math.nan,
            "confirm_fvg_bottom": float(confirm_event.get("fvg_bottom", math.nan)) if confirm_event else math.nan,
            "confirm_fvg_height": confirm_fvg_height,
            "confirm_fvg_atr": confirm_fvg_height / atr if atr > 0 else math.nan,
            "stop_price": stop,
            "planned_target_price": planned_target,
            "submitted_index": signal_idx,
        }

    def qualifies_confirmation(event: dict[str, Any], setup: dict[str, Any], signal_idx: int) -> bool:
        if event["direction"] != setup["direction"]:
            return False
        if pd.Timestamp(event["time"]) < pd.Timestamp(setup["retest_time"]):
            return False
        fvg_height = float(event.get("fvg_height", 0.0))
        if cfg.min_confirm_fvg_atr > 0 and atrs[signal_idx] > 0 and fvg_height / atrs[signal_idx] < cfg.min_confirm_fvg_atr:
            return False
        return bool(event.get("has_fvg", False))

    for i in range(len(prepared)):
        visible_time = pd.Timestamp(close_times[i])

        if pending_entry is not None and pending_entry["submitted_index"] < i and position is None:
            entry_price = float(opens[i])
            risk = entry_price - pending_entry["stop_price"] if pending_entry["direction"] == "long" else pending_entry["stop_price"] - entry_price
            risk_pct = risk / entry_price * 100.0 if entry_price > 0 else math.nan
            if risk > 0 and risk_pct >= cfg.min_entry_risk_pct:
                target = entry_price + risk * cfg.target_rr if pending_entry["direction"] == "long" else entry_price - risk * cfg.target_rr
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

        if position is not None and i > position["entry_index"]:
            exit_value = exit_for_bar(position, prepared.iloc[i])
            if exit_value is not None:
                exit_price, reason = exit_value
                trades.append(close_trade(position, i, pd.Timestamp(times[i]), float(exit_price), reason))
                position = None

        while demand_ptr < len(demand_events) and demand_events[demand_ptr]["time"] <= visible_time:
            demand_zones.append(add_zone(demand_events[demand_ptr], "demand", demand_ptr))
            demand_ptr += 1
        while supply_ptr < len(supply_events) and supply_events[supply_ptr]["time"] <= visible_time:
            supply_zones.append(add_zone(supply_events[supply_ptr], "supply", supply_ptr))
            supply_ptr += 1

        demand_zones = [zone for zone in demand_zones if not zone["used"]]
        supply_zones = [zone for zone in supply_zones if not zone["used"]]

        for zone in candidates(demand_zones):
            if closes[i] < float(zone["bottom"]):
                zone["used"] = True
                pending_short = make_setup("short", zone, i)
                break

        for zone in candidates(supply_zones):
            if closes[i] > float(zone["top"]):
                zone["used"] = True
                pending_long = make_setup("long", zone, i)
                break

        if pending_long is not None and i - pending_long["break_index"] > cfg.max_retest_bars:
            pending_long = None
        if pending_short is not None and i - pending_short["break_index"] > cfg.max_retest_bars:
            pending_short = None
        if pending_confirm is not None and i - pending_confirm["retest_index"] > cfg.max_confirm_bars:
            pending_confirm = None

        candle_range = highs[i] - lows[i]
        if position is None and pending_entry is None and pending_confirm is None and candle_range > 0:
            if pending_long is not None and i > pending_long["break_index"]:
                reject_pos = (closes[i] - lows[i]) / candle_range
                if lows[i] <= pending_long["zone_top"] and closes[i] > pending_long["zone_top"] and reject_pos >= cfg.min_reject_pos:
                    setup = mark_retest(pending_long, i, reject_pos)
                    if cfg.entry_mode == "zone_retest":
                        pending_entry = submit_entry(setup, i)
                    else:
                        pending_confirm = setup
                    pending_long = None
            if pending_short is not None and pending_entry is None and i > pending_short["break_index"]:
                reject_pos = (highs[i] - closes[i]) / candle_range
                if highs[i] >= pending_short["zone_bottom"] and closes[i] < pending_short["zone_bottom"] and reject_pos >= cfg.min_reject_pos:
                    setup = mark_retest(pending_short, i, reject_pos)
                    if cfg.entry_mode == "zone_retest":
                        pending_entry = submit_entry(setup, i)
                    else:
                        pending_confirm = setup
                    pending_short = None

        while confirmation_ptr < len(confirmation_events) and confirmation_events[confirmation_ptr]["time"] <= visible_time:
            event = confirmation_events[confirmation_ptr]
            if pending_confirm is not None and pending_entry is None and position is None and qualifies_confirmation(event, pending_confirm, i):
                pending_entry = submit_entry(pending_confirm, i, event)
                pending_confirm = None
            confirmation_ptr += 1

    result = pd.DataFrame(trades)
    if not return_state:
        return result
    state = {
        "pending_long": pending_long,
        "pending_short": pending_short,
        "pending_confirm": pending_confirm,
        "pending_entry": pending_entry,
        "position": position,
        "closed_trades": len(trades),
    }
    return result, state


def main() -> None:
    parser = argparse.ArgumentParser(description="Study breaker continuation after failed SMC zones.")
    parser.add_argument("--symbols", nargs="+", default=[])
    parser.add_argument("--symbol-set", choices=["none", *SYMBOL_SETS.keys()], default="core3")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--start", default="2024-04-20")
    parser.add_argument("--split", default="2025-04-20")
    parser.add_argument("--end", default="2026-04-20")
    parser.add_argument("--cache-dir", type=Path, default=Path("scripts/.cache"))
    parser.add_argument("--output", type=Path, default=Path("scripts/breaker_continuation_trades.csv"))
    parser.add_argument("--entry-mode", choices=["zone_retest", "structure_fvg", "fvg_print"], default="zone_retest")
    parser.add_argument("--zone-tf", default="4h")
    parser.add_argument("--confirmation-tf", default="15m")
    parser.add_argument("--structure-left", type=int, default=2)
    parser.add_argument("--structure-right", type=int, default=2)
    parser.add_argument("--htf-left", type=int, default=5)
    parser.add_argument("--htf-right", type=int, default=5)
    parser.add_argument("--htf-ob-search-bars", type=int, default=50)
    parser.add_argument("--max-zone-scan", type=int, default=250)
    parser.add_argument("--max-retest-bars", type=int, default=288)
    parser.add_argument("--max-confirm-bars", type=int, default=72)
    parser.add_argument("--max-hold-bars", type=int, default=120)
    parser.add_argument("--stop-buffer-atr", type=float, default=0.10)
    parser.add_argument("--target-rr", type=float, default=2.0)
    parser.add_argument("--min-reject-pos", type=float, default=0.50)
    parser.add_argument("--min-confirm-fvg-atr", type=float, default=0.0)
    parser.add_argument("--min-entry-risk-pct", type=float, default=0.0)
    args = parser.parse_args()

    args.symbols = expand_symbol_args(args.symbols, args.symbol_set)
    start = parse_utc_datetime(args.start)
    split = parse_utc_datetime(args.split)
    end = parse_utc_datetime(args.end)
    cfg = BreakerConfig(
        entry_mode=args.entry_mode,
        zone_tf=args.zone_tf,
        confirmation_tf=args.confirmation_tf,
        structure_left=args.structure_left,
        structure_right=args.structure_right,
        htf_left=args.htf_left,
        htf_right=args.htf_right,
        htf_ob_search_bars=args.htf_ob_search_bars,
        max_zone_scan=args.max_zone_scan,
        max_retest_bars=args.max_retest_bars,
        max_confirm_bars=args.max_confirm_bars,
        max_hold_bars=args.max_hold_bars,
        stop_buffer_atr=args.stop_buffer_atr,
        target_rr=args.target_rr,
        min_reject_pos=args.min_reject_pos,
        min_confirm_fvg_atr=args.min_confirm_fvg_atr,
        min_entry_risk_pct=args.min_entry_risk_pct,
    )

    frames: list[pd.DataFrame] = []
    for symbol in args.symbols:
        cache = ensure_cache(symbol, args.interval, start, end, args.cache_dir)
        df = pd.read_pickle(cache)
        before = time.time()
        trades = simulate_breakers(symbol, df, cfg)
        frames.append(trades)
        print(f"{normalize_binance_spot_symbol(symbol)}: {len(trades)} breaker trades in {time.time() - before:.1f}s")

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    if result.empty:
        print("No breaker trades generated.")
        return

    result["entry_time"] = pd.to_datetime(result["entry_time"], utc=True)
    train = result[result["entry_time"] < pd.Timestamp(split)]
    oos = result[(result["entry_time"] >= pd.Timestamp(split)) & (result["entry_time"] < pd.Timestamp(end))]
    rows: list[dict[str, Any]] = []
    for symbol, frame in result.groupby("symbol"):
        rows.append({"symbol": symbol, "window": "train", **frame_metrics(frame[frame["entry_time"] < pd.Timestamp(split)])})
        rows.append({"symbol": symbol, "window": "oos", **frame_metrics(frame[(frame["entry_time"] >= pd.Timestamp(split)) & (frame["entry_time"] < pd.Timestamp(end))])})
    rows.append({"symbol": "AGG", "window": "train", **frame_metrics(train)})
    rows.append({"symbol": "AGG", "window": "oos", **frame_metrics(oos)})

    print()
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nSaved {len(result)} trades to {args.output}")


if __name__ == "__main__":
    main()
