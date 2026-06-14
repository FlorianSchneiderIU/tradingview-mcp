#!/usr/bin/env python3
"""Reconstruct OPI_BRAIN alerts and test their stop/trigger/trailing lifecycle."""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from scripts.investigate_opi_curl_reversals import (
    DEFAULT_OPI_MATRIX_ROOM_ID,
    OpiSignal,
    ensure_symbol_frame,
    fetch_matrix_messages,
    first_index_after,
    parse_opi_signal,
)


@dataclass(frozen=True)
class TrailConfig:
    stop_pct: float
    trigger_pct: float
    trail_pct: float
    move_to_be: bool = True
    use_declared_stop: bool = False
    use_declared_trigger: bool = False

    @property
    def name(self) -> str:
        if self.use_declared_stop or self.use_declared_trigger:
            return (
                f"declared_sl{int(self.use_declared_stop)}_"
                f"target{int(self.use_declared_trigger)}_trail{self.trail_pct:.2f}"
            )
        return (
            f"sl{self.stop_pct:.2f}_trigger{self.trigger_pct:.2f}_"
            f"trail{self.trail_pct:.2f}_be{int(self.move_to_be)}"
        )


def is_opi_brain(signal: OpiSignal) -> bool:
    text = signal.raw_text
    has_brain_regime = "BOTTOMING" in text or "BURNING OUT" in text
    return (
        signal.kind == "opi_full"
        and signal.target is not None
        and signal.sl is not None
        and has_brain_regime
    )


def load_opi_brain_signals(args: argparse.Namespace) -> list[OpiSignal]:
    messages = fetch_matrix_messages(args.matrix_env_path, args.matrix_room_id, args.matrix_limit)
    start = pd.Timestamp(args.signal_start, tz="UTC")
    end = pd.Timestamp(args.signal_end, tz="UTC")
    signals: list[OpiSignal] = []
    for message in messages:
        message_time = pd.Timestamp(message["time"])
        if message_time < start or message_time > end:
            continue
        signals.extend(signal for signal in parse_opi_signal(message) if is_opi_brain(signal))
    signals.sort(key=lambda signal: signal.time)
    return signals


def segment_crosses(start: float, end: float, level: float) -> bool:
    return min(start, end) <= level <= max(start, end)


def simulate_trailing_trade(
    frame: pd.DataFrame,
    *,
    signal: OpiSignal,
    config: TrailConfig,
    fee_rate: float,
    report_end: pd.Timestamp,
) -> dict[str, Any]:
    entry_idx = first_index_after(frame, signal.time)
    if entry_idx is None:
        return {"outcome": "missing_candles"}

    entry = float(signal.entry)
    direction = signal.direction
    stop_frac = config.stop_pct / 100.0
    trigger_frac = config.trigger_pct / 100.0
    trail_frac = config.trail_pct / 100.0
    if direction == "long":
        initial_stop = float(signal.sl) if config.use_declared_stop else entry * (1.0 - stop_frac)
        trigger = float(signal.target) if config.use_declared_trigger else entry * (1.0 + trigger_frac)
    else:
        initial_stop = float(signal.sl) if config.use_declared_stop else entry * (1.0 + stop_frac)
        trigger = float(signal.target) if config.use_declared_trigger else entry * (1.0 - trigger_frac)

    active_stop = initial_stop
    best_price = entry
    armed = False
    armed_time: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_time: pd.Timestamp | None = None
    outcome = "open"

    for idx in range(entry_idx, len(frame)):
        row = frame.iloc[idx]
        bar_time = pd.Timestamp(row["open_time"])
        if bar_time > report_end:
            break

        open_price = float(row["open"])
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        # A deterministic OHLC path avoids assuming that both extremes happened
        # in whichever order is most favorable to the trade.
        points = [open_price, low, high, close] if close >= open_price else [open_price, high, low, close]

        current = points[0]
        for next_price in points[1:]:
            if direction == "long":
                if next_price < current:
                    if segment_crosses(current, next_price, active_stop):
                        exit_price = active_stop
                        exit_time = bar_time
                        outcome = "trail" if armed else "stop"
                        break
                else:
                    best_price = max(best_price, next_price)
                    if not armed and best_price >= trigger:
                        armed = True
                        armed_time = bar_time
                    if armed:
                        ratchet = best_price * (1.0 - trail_frac)
                        if config.move_to_be:
                            ratchet = max(ratchet, entry)
                        active_stop = max(active_stop, ratchet)
            else:
                if next_price > current:
                    if segment_crosses(current, next_price, active_stop):
                        exit_price = active_stop
                        exit_time = bar_time
                        outcome = "trail" if armed else "stop"
                        break
                else:
                    best_price = min(best_price, next_price)
                    if not armed and best_price <= trigger:
                        armed = True
                        armed_time = bar_time
                    if armed:
                        ratchet = best_price * (1.0 + trail_frac)
                        if config.move_to_be:
                            ratchet = min(ratchet, entry)
                        active_stop = min(active_stop, ratchet)
            current = next_price
        if exit_price is not None:
            break

    if exit_price is None:
        available = frame.loc[
            (frame.index >= entry_idx) & (frame["open_time"] <= report_end),
            :,
        ]
        if available.empty:
            return {"outcome": "missing_candles"}
        last = available.iloc[-1]
        exit_price = float(last["close"])
        exit_time = pd.Timestamp(last["open_time"])

    gross_return = (
        (exit_price - entry) / entry
        if direction == "long"
        else (entry - exit_price) / entry
    )
    fee_return = fee_rate * (entry + exit_price) / entry
    net_return = gross_return - fee_return
    return {
        "outcome": outcome,
        "entry_time": signal.time,
        "exit_time": exit_time,
        "armed_time": armed_time,
        "entry": entry,
        "declared_sl": signal.sl,
        "declared_tp": signal.target,
        "initial_stop": initial_stop,
        "trigger": trigger,
        "exit_price": exit_price,
        "armed": armed,
        "gross_pct": gross_return * 100.0,
        "fees_pct": fee_return * 100.0,
        "net_pct": net_return * 100.0,
        "hold_minutes": (exit_time - signal.time).total_seconds() / 60.0,
    }


def float_range(values: Iterable[float]) -> list[float]:
    return sorted({round(float(value), 8) for value in values})


def configs_from_args(args: argparse.Namespace) -> list[TrailConfig]:
    configs = [
        TrailConfig(
            stop_pct=args.report_stop_pct,
            trigger_pct=args.report_trigger_pct,
            trail_pct=args.report_trail_pct,
            move_to_be=True,
        ),
        TrailConfig(
            stop_pct=0.0,
            trigger_pct=0.0,
            trail_pct=args.report_trail_pct,
            move_to_be=True,
            use_declared_stop=True,
            use_declared_trigger=True,
        ),
    ]
    for stop_pct in float_range(args.stop_pcts):
        for trigger_pct in float_range(args.trigger_pcts):
            for trail_pct in float_range(args.trail_pcts):
                if trail_pct >= trigger_pct:
                    continue
                configs.append(TrailConfig(stop_pct, trigger_pct, trail_pct, True))
    unique: dict[str, TrailConfig] = {config.name: config for config in configs}
    return list(unique.values())


def summarize(config: TrailConfig, trades: pd.DataFrame) -> dict[str, Any]:
    closed = trades[trades["outcome"].isin(["stop", "trail"])]
    wins = int((closed["net_pct"] > 0).sum())
    losses = int((closed["net_pct"] <= 0).sum())
    equity = closed["net_pct"].cumsum()
    running_high = equity.cummax()
    max_drawdown = float((equity - running_high).min()) if not equity.empty else math.nan
    return {
        **asdict(config),
        "config": config.name,
        "signals": len(trades),
        "closed": len(closed),
        "open": int((trades["outcome"] == "open").sum()),
        "wins": wins,
        "losses": losses,
        "winrate": wins / len(closed) if len(closed) else math.nan,
        "gross_pct": float(closed["gross_pct"].sum()),
        "net_pct": float(closed["net_pct"].sum()),
        "avg_net_pct": float(closed["net_pct"].mean()) if len(closed) else math.nan,
        "median_net_pct": float(closed["net_pct"].median()) if len(closed) else math.nan,
        "max_drawdown_pct_points": max_drawdown,
        "median_hold_minutes": float(closed["hold_minutes"].median()) if len(closed) else math.nan,
    }


def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    signals = load_opi_brain_signals(args)
    print(
        f"OPI_BRAIN signals={len(signals)} "
        f"window={args.signal_start} -> {args.signal_end}",
        flush=True,
    )
    if not signals:
        return

    signals_frame = pd.DataFrame(
        {
            **signal.to_dict(),
            "declared_stop_pct": abs(float(signal.entry) - float(signal.sl)) / float(signal.entry) * 100.0,
            "declared_target_pct": abs(float(signal.target) - float(signal.entry)) / float(signal.entry) * 100.0,
        }
        for signal in signals
    )
    signals_frame.to_csv(args.out_dir / "opi_brain_signals.csv", index=False)
    print(
        "symbols="
        + str(signals_frame["symbol"].value_counts().to_dict())
        + f" declared stop median={signals_frame['declared_stop_pct'].median():.3f}%"
        + f" target median={signals_frame['declared_target_pct'].median():.3f}%",
        flush=True,
    )

    candle_start = signals[0].time.to_pydatetime() - timedelta(hours=1)
    candle_end = pd.Timestamp(args.signal_end, tz="UTC").to_pydatetime() + timedelta(minutes=2)
    frames: dict[str, pd.DataFrame] = {}
    for symbol in sorted(signals_frame["symbol"].unique()):
        frame = ensure_symbol_frame(symbol, candle_start, candle_end, args.cache_dir)
        frames[symbol] = frame
        print(f"{symbol}: candles={len(frame)}", flush=True)

    report_end = pd.Timestamp(args.signal_end, tz="UTC")
    summaries: list[dict[str, Any]] = []
    report_trades: pd.DataFrame | None = None
    report_name = TrailConfig(
        args.report_stop_pct,
        args.report_trigger_pct,
        args.report_trail_pct,
        True,
    ).name
    for config in configs_from_args(args):
        rows = []
        for signal in signals:
            frame = frames.get(signal.symbol)
            if frame is None or frame.empty:
                continue
            result = simulate_trailing_trade(
                frame,
                signal=signal,
                config=config,
                fee_rate=args.fee_rate,
                report_end=report_end,
            )
            rows.append(
                {
                    "source_id": signal.source_id,
                    "symbol": signal.symbol,
                    "timeframe": signal.timeframe,
                    "direction": signal.direction,
                    "score": signal.score,
                    "config": config.name,
                    **result,
                }
            )
        trades = pd.DataFrame(rows)
        summaries.append(summarize(config, trades))
        if config.name == report_name:
            report_trades = trades

    summary_frame = pd.DataFrame(summaries).sort_values(
        ["net_pct", "winrate", "closed"],
        ascending=[False, False, False],
    )
    summary_frame.to_csv(args.out_dir / "opi_brain_trailing_grid.csv", index=False)
    if report_trades is not None:
        report_trades.to_csv(args.out_dir / "opi_brain_report_config_trades.csv", index=False)

    print("\nAdvertised report configuration", flush=True)
    print(
        summary_frame.loc[summary_frame["config"] == report_name].to_string(index=False),
        flush=True,
    )
    print("\nTop neighboring configurations", flush=True)
    print(summary_frame.head(20).to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-env-path", type=Path, default=Path("bot/.env.matrix"))
    parser.add_argument("--matrix-room-id", default=DEFAULT_OPI_MATRIX_ROOM_ID)
    parser.add_argument("--matrix-limit", type=int, default=25000)
    # May 13 is the first timestamp that makes the report's 117-trade cohort
    # through June 13 18:50 UTC. The UI's global data window begins April 28.
    parser.add_argument("--signal-start", default="2026-05-13 00:00")
    parser.add_argument("--signal-end", default="2026-06-13 18:50")
    parser.add_argument("--report-stop-pct", type=float, default=0.71)
    parser.add_argument("--report-trigger-pct", type=float, default=0.75)
    parser.add_argument("--report-trail-pct", type=float, default=0.40)
    parser.add_argument("--stop-pcts", type=float, nargs="+", default=[0.61, 0.66, 0.71, 0.76, 0.81])
    parser.add_argument("--trigger-pcts", type=float, nargs="+", default=[0.65, 0.70, 0.75, 0.80, 0.85])
    parser.add_argument("--trail-pcts", type=float, nargs="+", default=[0.30, 0.35, 0.40, 0.45, 0.50])
    parser.add_argument("--fee-rate", type=float, default=0.00055)
    parser.add_argument("--cache-dir", type=Path, default=Path("scripts/.cache/opi_brain"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("scripts/opi_brain_trailing_investigation_20260614"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
