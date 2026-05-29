from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_btc_liquidity_sweeps import TF_MINUTES, TF_SETTINGS, build_replay_paths, stop_anchor_for_event
from scripts.backtest_wolfe_wave import ensure_ohlcv_frame, load_ohlcv_csv


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    stop_anchor_mode: str
    stop_mode: str
    stop_buffer_atr: float
    tp1_r: float
    tp1_weight: float
    tp2_r: float | None


PREFERRED_CONFIGS = [
    StrategyConfig("fixed_2p5", "half_wick_body", "close_15m", buffer, 2.5, 1.0, None)
    for buffer in [0.0, 0.05, 0.10, 0.15]
] + [
    StrategyConfig("fixed_5", "half_wick_body", "close_15m", buffer, 5.0, 1.0, None)
    for buffer in [0.0, 0.05, 0.10, 0.15]
] + [
    StrategyConfig("fixed_7p5", "half_wick_body", "close_15m", buffer, 7.5, 1.0, None)
    for buffer in [0.0, 0.05, 0.10, 0.15]
] + [
    StrategyConfig("partial_2p5_5", "half_wick_body", "close_15m", buffer, 2.5, 0.5, 5.0)
    for buffer in [0.0, 0.05, 0.10, 0.15]
] + [
    StrategyConfig("partial_2p5_7p5", "half_wick_body", "close_15m", buffer, 2.5, 0.5, 7.5)
    for buffer in [0.0, 0.05, 0.10, 0.15]
]


def truthy(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def hit_target(direction: str, high: float, low: float, target: float) -> bool:
    return bool(high >= target) if direction == "long" else bool(low <= target)


def stop_hit_on_close(direction: str, close: float, stop: float) -> bool:
    return bool(close <= stop) if direction == "long" else bool(close >= stop)


def r_price(entry: float, risk: float, direction: str, rr: float) -> float:
    return entry + rr * risk if direction == "long" else entry - rr * risk


def price_to_r(price: float, entry: float, risk: float, direction: str) -> float:
    return (price - entry) / risk if direction == "long" else (entry - price) / risk


def simulate_trade(paths: dict[str, Any], event: pd.Series, config: StrategyConfig, candidate_filter: str) -> dict[str, Any] | None:
    timeframe = str(event["timeframe"])
    settings = TF_SETTINGS.get(timeframe)
    if settings is None:
        return None
    direction = str(event["direction"])
    entry = float(event["entry_price"])
    atr = float(event["sweep_atr"])
    anchor = stop_anchor_for_event(event, direction, config.stop_anchor_mode)
    if not all(math.isfinite(value) for value in [entry, atr, anchor]) or atr <= 0.0:
        return None

    stop = anchor - config.stop_buffer_atr * atr if direction == "long" else anchor + config.stop_buffer_atr * atr
    risk = entry - stop if direction == "long" else stop - entry
    if not math.isfinite(risk) or risk <= 0.0:
        return None

    entry_time = pd.Timestamp(event["entry_time"]).tz_convert("UTC")
    start_ns = int(entry_time.value)
    end_ns = int((entry_time + pd.Timedelta(minutes=TF_MINUTES[timeframe] * int(settings["horizon"]))).value)
    base = paths["5m"]
    confirm_tf = config.stop_mode.removeprefix("close_")
    confirm = paths[confirm_tf]

    start_idx = int(np.searchsorted(base.open_time_ns, start_ns, side="left"))
    end_idx = int(np.searchsorted(base.open_time_ns, end_ns, side="right") - 1)
    end_idx = min(max(end_idx, start_idx), len(base.open_time_ns) - 1)
    if start_idx >= len(base.open_time_ns):
        return None

    confirm_idx = int(np.searchsorted(confirm.close_time_ns, start_ns, side="left"))
    confirm_end_idx = int(np.searchsorted(confirm.close_time_ns, end_ns, side="right") - 1)
    confirm_end_idx = min(max(confirm_end_idx, confirm_idx), len(confirm.close_time_ns) - 1)

    tp1 = r_price(entry, risk, direction, config.tp1_r)
    tp2 = r_price(entry, risk, direction, config.tp2_r) if config.tp2_r is not None else None
    remaining = 1.0
    result_r = 0.0
    tp1_hit = False
    tp2_hit = False
    mfe_r = 0.0
    mae_r = 0.0
    exit_reason = "timeout"
    exit_time_ns = int(base.close_time_ns[end_idx])
    exit_price = float(base.closes[end_idx])
    stop_exit_r = math.nan
    bars_to_exit = int(end_idx - start_idx)

    for cursor in range(start_idx, end_idx + 1):
        high = float(base.highs[cursor])
        low = float(base.lows[cursor])
        close_time_ns = int(base.close_time_ns[cursor])
        mfe_r = max(mfe_r, (high - entry) / risk if direction == "long" else (entry - low) / risk)
        mae_r = max(mae_r, (entry - low) / risk if direction == "long" else (high - entry) / risk)

        if remaining > 0.0 and not tp1_hit and hit_target(direction, high, low, tp1):
            fill_weight = min(config.tp1_weight, remaining)
            result_r += fill_weight * config.tp1_r
            remaining -= fill_weight
            tp1_hit = True
            if remaining <= 1e-12:
                exit_reason = "target1"
                exit_time_ns = close_time_ns
                exit_price = tp1
                bars_to_exit = int(cursor - start_idx)
                break

        if remaining > 0.0 and tp2 is not None and hit_target(direction, high, low, tp2):
            result_r += remaining * float(config.tp2_r)
            remaining = 0.0
            tp2_hit = True
            exit_reason = "target2"
            exit_time_ns = close_time_ns
            exit_price = tp2
            bars_to_exit = int(cursor - start_idx)
            break

        while confirm_idx <= confirm_end_idx and int(confirm.close_time_ns[confirm_idx]) <= close_time_ns:
            stop_close = float(confirm.closes[confirm_idx])
            if stop_hit_on_close(direction, stop_close, stop):
                stop_exit_r = price_to_r(stop_close, entry, risk, direction)
                result_r += remaining * stop_exit_r
                remaining = 0.0
                exit_reason = "stop"
                exit_time_ns = int(confirm.close_time_ns[confirm_idx])
                exit_price = stop_close
                bars_to_exit = int(cursor - start_idx)
                break
            confirm_idx += 1
        if remaining <= 1e-12:
            break

    if remaining > 1e-12:
        timeout_r = price_to_r(float(base.closes[end_idx]), entry, risk, direction)
        result_r += remaining * timeout_r
        exit_reason = "timeout_after_tp1" if tp1_hit else "timeout"
        exit_time_ns = int(base.close_time_ns[end_idx])
        exit_price = float(base.closes[end_idx])
        bars_to_exit = int(end_idx - start_idx)

    return {
        "candidate_filter": candidate_filter,
        "config": config.name,
        "stop_anchor_mode": config.stop_anchor_mode,
        "stop_mode": config.stop_mode,
        "stop_buffer_atr": float(config.stop_buffer_atr),
        "tp1_r": float(config.tp1_r),
        "tp1_weight": float(config.tp1_weight),
        "tp2_r": float(config.tp2_r) if config.tp2_r is not None else math.nan,
        "timeframe": timeframe,
        "event_time": event["event_time"],
        "entry_time": entry_time,
        "exit_time": pd.Timestamp(exit_time_ns, tz="UTC"),
        "direction": direction,
        "session": event["session"],
        "trend_side": event["trend_side"],
        "entry_price": entry,
        "stop_price": float(stop),
        "exit_price": float(exit_price),
        "risk_pct": float(risk / entry),
        "result_r": float(result_r),
        "exit_reason": exit_reason,
        "tp1_hit": bool(tp1_hit),
        "tp2_hit": bool(tp2_hit),
        "mfe_r": float(mfe_r),
        "mae_r": float(mae_r),
        "bars_5m_to_exit": bars_to_exit,
        "hours_to_exit": float(bars_to_exit * 5.0 / 60.0),
        "stop_exit_r": float(stop_exit_r) if math.isfinite(stop_exit_r) else math.nan,
        "level_id": event["level_id"],
    }


def max_drawdown_r(results: pd.Series) -> float:
    if results.empty:
        return 0.0
    equity = results.cumsum()
    drawdown = equity - equity.cummax()
    return float(drawdown.min())


def profit_factor(results: pd.Series) -> float:
    gains = results[results > 0.0].sum()
    losses = results[results < 0.0].sum()
    if losses == 0.0:
        return math.inf if gains > 0.0 else 0.0
    return float(gains / abs(losses))


def one_trade_at_a_time(trades: pd.DataFrame) -> pd.DataFrame:
    kept: list[int] = []
    active_until: pd.Timestamp | None = None
    ordered = trades.sort_values(["entry_time", "exit_time"]).reset_index(drop=True)
    for idx, row in ordered.iterrows():
        entry_time = pd.Timestamp(row["entry_time"])
        if active_until is not None and entry_time < active_until:
            continue
        kept.append(idx)
        active_until = pd.Timestamp(row["exit_time"])
    return ordered.loc[kept].reset_index(drop=True)


def summarize_trades(trades: pd.DataFrame, portfolio: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if trades.empty:
        return pd.DataFrame()
    for keys, group in trades.groupby(["candidate_filter", "config", "stop_buffer_atr"], dropna=False):
        candidate_filter, config, stop_buffer_atr = keys
        ordered = group.sort_values("entry_time").reset_index(drop=True)
        result = pd.to_numeric(ordered["result_r"], errors="coerce")
        wins = result[result > 0.0]
        losses = result[result < 0.0]
        start = pd.Timestamp(ordered["entry_time"].min())
        end = pd.Timestamp(ordered["entry_time"].max())
        years = max((end - start).days / 365.25, 1e-9)
        rows.append(
            {
                "portfolio": portfolio,
                "candidate_filter": candidate_filter,
                "config": config,
                "stop_buffer_atr": float(stop_buffer_atr),
                "trades": int(len(ordered)),
                "trades_per_year": float(len(ordered) / years),
                "net_r": float(result.sum()),
                "avg_r": float(result.mean()),
                "median_r": float(result.median()),
                "win_rate": float((result > 0.0).mean()),
                "profit_factor": profit_factor(result),
                "max_dd_r": max_drawdown_r(result),
                "avg_win_r": float(wins.mean()) if not wins.empty else 0.0,
                "avg_loss_r": float(losses.mean()) if not losses.empty else 0.0,
                "median_risk_pct": float(pd.to_numeric(ordered["risk_pct"], errors="coerce").median()),
                "tp1_hit_rate": float(ordered["tp1_hit"].mean()),
                "tp2_hit_rate": float(ordered["tp2_hit"].mean()),
                "stop_exit_rate": float((ordered["exit_reason"] == "stop").mean()),
                "timeout_rate": float(ordered["exit_reason"].astype(str).str.startswith("timeout").mean()),
                "avg_hours_to_exit": float(pd.to_numeric(ordered["hours_to_exit"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["portfolio", "candidate_filter", "avg_r", "net_r"], ascending=[True, True, False, False])


def yearly_summary(trades: pd.DataFrame, portfolio: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    out = trades.copy()
    out["year"] = pd.to_datetime(out["entry_time"], utc=True).dt.year
    rows: list[dict[str, Any]] = []
    for keys, group in out.groupby(["candidate_filter", "config", "stop_buffer_atr", "year"], dropna=False):
        candidate_filter, config, stop_buffer_atr, year = keys
        result = pd.to_numeric(group["result_r"], errors="coerce")
        rows.append(
            {
                "portfolio": portfolio,
                "candidate_filter": candidate_filter,
                "config": config,
                "stop_buffer_atr": float(stop_buffer_atr),
                "year": int(year),
                "trades": int(len(group)),
                "net_r": float(result.sum()),
                "avg_r": float(result.mean()),
                "win_rate": float((result > 0.0).mean()),
                "max_dd_r": max_drawdown_r(result),
            }
        )
    return pd.DataFrame(rows).sort_values(["portfolio", "candidate_filter", "config", "stop_buffer_atr", "year"])


def build_backtest(events: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    paths = build_replay_paths(base)
    rows: list[dict[str, Any]] = []
    filters = {
        "4h_session_trend": truthy(events["candidate_4h_session_trend"]),
        "4h_session_trend_bounded": truthy(events["candidate_4h_session_trend_bounded"]),
    }
    for filter_name, mask in filters.items():
        candidates = events[mask & events["timeframe"].astype(str).eq("4h")].copy()
        for _, event in candidates.iterrows():
            for config in PREFERRED_CONFIGS:
                row = simulate_trade(paths, event, config, filter_name)
                if row is not None:
                    rows.append(row)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest the preferred BTC 4h liquidity sweep configuration.")
    parser.add_argument("--events", type=Path, default=Path("scripts/btc_liquidity_sweep_study/btc_liquidity_sweep_events.csv"))
    parser.add_argument("--input", type=Path, default=Path("scripts/data/btcusdt_5m_bybit.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("scripts/btc_liquidity_sweep_study"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events = pd.read_csv(args.events)
    for column in ["event_time", "entry_time"]:
        if column in events.columns:
            events[column] = pd.to_datetime(events[column], utc=True)
    base = ensure_ohlcv_frame(load_ohlcv_csv(args.input))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    trades = build_backtest(events, base)
    trades.to_csv(args.output_dir / "preferred_backtest_trades.csv", index=False)

    no_overlap_parts = []
    for _, group in trades.groupby(["candidate_filter", "config", "stop_buffer_atr"], dropna=False):
        no_overlap_parts.append(one_trade_at_a_time(group))
    no_overlap = pd.concat(no_overlap_parts, ignore_index=True) if no_overlap_parts else pd.DataFrame()
    no_overlap.to_csv(args.output_dir / "preferred_backtest_trades_one_at_a_time.csv", index=False)

    summary = pd.concat(
        [
            summarize_trades(trades, "all_signals"),
            summarize_trades(no_overlap, "one_trade_at_a_time"),
        ],
        ignore_index=True,
    )
    summary.to_csv(args.output_dir / "preferred_backtest_summary.csv", index=False)

    yearly = pd.concat(
        [
            yearly_summary(trades, "all_signals"),
            yearly_summary(no_overlap, "one_trade_at_a_time"),
        ],
        ignore_index=True,
    )
    yearly.to_csv(args.output_dir / "preferred_backtest_by_year.csv", index=False)

    print(f"Wrote {args.output_dir / 'preferred_backtest_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
