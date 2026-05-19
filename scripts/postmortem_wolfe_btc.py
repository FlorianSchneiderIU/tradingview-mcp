from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backtest_wolfe_wave import (  # noqa: E402
    WolfeConfig,
    add_indicators,
    ensure_ohlcv_frame,
    find_wolfe_signals,
    high_before_low,
    line_params_time,
    line_value,
    run_backtest,
    split_trades,
    strategy_metrics,
    timestamp_seconds,
)
from scripts.tune_wolfe_wave_universe import split_bounds  # noqa: E402


BTC_TUNING = Path("scripts/wolfe_wave_universe_4y_oos1y_stage40_fast/per_symbol/btcusdt_wolfe_tuning.csv")
BTC_DATA = Path("scripts/data/btcusdt_5m_bybit.csv")
OUT_DIR = Path("scripts/wolfe_wave_btc_postmortem")


def cfg_from_row(row: pd.Series) -> WolfeConfig:
    fields = set(WolfeConfig.__dataclass_fields__)  # type: ignore[attr-defined]
    return WolfeConfig.from_mapping({key: row[key] for key in fields if key in row.index and pd.notna(row[key])})


def metric_row(trades: pd.DataFrame, label: str) -> dict[str, Any]:
    return {"bucket": label, **strategy_metrics(trades)}


def summarize_trades(trades: pd.DataFrame, train_end: pd.Timestamp, validation_end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    buckets = split_trades(trades, train_end=train_end, validation_end=validation_end)
    split_summary = pd.DataFrame([metric_row(frame, label) for label, frame in buckets.items()])
    if trades.empty:
        return {
            "split_summary": split_summary,
            "direction_summary": pd.DataFrame(),
            "exit_summary": pd.DataFrame(),
            "monthly_summary": pd.DataFrame(),
        }
    enriched = trades.copy()
    enriched["entry_time"] = pd.to_datetime(enriched["entry_time"], utc=True)
    enriched["split"] = "train"
    enriched.loc[enriched["entry_time"] >= train_end, "split"] = "validation"
    enriched.loc[enriched["entry_time"] >= validation_end, "split"] = "oos"
    enriched["month"] = enriched["entry_time"].dt.to_period("M").astype(str)
    direction_summary = (
        enriched.groupby(["split", "direction"], as_index=False)
        .agg(
            trades=("r_multiple_net", "size"),
            net_r=("r_multiple_net", "sum"),
            avg_r=("r_multiple_net", "mean"),
            win_rate=("r_multiple_net", lambda values: float((values > 0).mean())),
        )
        .sort_values(["split", "direction"])
    )
    exit_summary = (
        enriched.groupby(["split", "exit_reason"], as_index=False)
        .agg(
            trades=("r_multiple_net", "size"),
            net_r=("r_multiple_net", "sum"),
            avg_r=("r_multiple_net", "mean"),
        )
        .sort_values(["split", "net_r"], ascending=[True, False])
    )
    monthly_summary = (
        enriched.groupby(["month", "split"], as_index=False)
        .agg(
            trades=("r_multiple_net", "size"),
            net_r=("r_multiple_net", "sum"),
            avg_r=("r_multiple_net", "mean"),
            wins=("r_multiple_net", lambda values: int((values > 0).sum())),
        )
        .sort_values("month")
    )
    return {
        "split_summary": split_summary,
        "direction_summary": direction_summary,
        "exit_summary": exit_summary,
        "monthly_summary": monthly_summary,
    }


def dynamic_epa_backtest(frame: pd.DataFrame, cfg: WolfeConfig) -> pd.DataFrame:
    exec_frame = add_indicators(ensure_ohlcv_frame(frame), cfg.atr_length, cfg.ema_length, cfg.rsi_length)
    signals = find_wolfe_signals(exec_frame, cfg, symbol="BTCUSDT")
    trades: list[dict[str, Any]] = []
    next_available_idx = 0
    for sig in signals:
        if cfg.one_trade_at_a_time and sig.entry_index < next_available_idx:
            continue
        if sig.entry_index >= len(exec_frame) - 1:
            continue
        entry = float(sig.entry_price)
        stop = float(sig.stop_price)
        risk = abs(entry - stop)
        if risk <= 0:
            continue
        m14, b14 = line_params_time(sig.pivots[0], sig.pivots[3])
        exit_idx = min(len(exec_frame) - 1, sig.entry_index + max(1, int(cfg.max_hold_bars)))
        exit_price = float(exec_frame["close"].iloc[exit_idx])
        exit_reason = "timeout"
        for idx in range(sig.entry_index + 1, exit_idx + 1):
            row = exec_frame.iloc[idx]
            open_value = float(row["open"])
            high_value = float(row["high"])
            low_value = float(row["low"])
            target = line_value(m14, b14, timestamp_seconds(pd.Timestamp(row["close_time"]).tz_convert("UTC")))
            if sig.direction == "long":
                target_hit = high_value >= target > entry
                stop_hit = low_value <= stop
                if target_hit and stop_hit:
                    if high_before_low(open_value, high_value, low_value):
                        exit_idx, exit_price, exit_reason = idx, target, "epa_target_same_bar"
                    else:
                        exit_idx, exit_price, exit_reason = idx, stop, "stop_same_bar"
                    break
                if stop_hit:
                    exit_idx, exit_price, exit_reason = idx, stop, "stop"
                    break
                if target_hit:
                    exit_idx, exit_price, exit_reason = idx, target, "epa_target"
                    break
            else:
                target_hit = low_value <= target < entry
                stop_hit = high_value >= stop
                if target_hit and stop_hit:
                    if high_before_low(open_value, high_value, low_value):
                        exit_idx, exit_price, exit_reason = idx, stop, "stop_same_bar"
                    else:
                        exit_idx, exit_price, exit_reason = idx, target, "epa_target_same_bar"
                    break
                if stop_hit:
                    exit_idx, exit_price, exit_reason = idx, stop, "stop"
                    break
                if target_hit:
                    exit_idx, exit_price, exit_reason = idx, target, "epa_target"
                    break
        gross_r = (exit_price - entry) / risk if sig.direction == "long" else (entry - exit_price) / risk
        cost_r = ((2.0 * cfg.fee_bps_side) + (2.0 * cfg.slippage_bps_side)) / 10_000.0 * entry / risk
        net_r = gross_r - cost_r
        exit_time = pd.Timestamp(exec_frame["close_time"].iloc[exit_idx]).tz_convert("UTC")
        trades.append(
            {
                "symbol": "BTCUSDT",
                "direction": sig.direction,
                "event_time": sig.event_time,
                "entry_time": sig.entry_time,
                "exit_time": exit_time,
                "entry_price": entry,
                "exit_price": float(exit_price),
                "stop_price": stop,
                "target_price": float(line_value(m14, b14, timestamp_seconds(exit_time))),
                "target_rr_planned": sig.target_rr_planned,
                "r_multiple_gross": gross_r,
                "r_multiple_net": net_r,
                "return_pct": net_r * cfg.risk_fraction,
                "hold_bars": int(exit_idx - sig.entry_index),
                "exit_reason": exit_reason,
                "score": sig.score,
                "p5_break_atr": sig.p5_break_atr,
                "symmetry_ratio": sig.symmetry_ratio,
                "epa_slope_atr": sig.epa_slope_atr,
                "volume_ratio": sig.volume_ratio if math.isfinite(sig.volume_ratio) else math.nan,
                "rsi": sig.rsi,
                "pattern_tf": sig.pattern_tf,
                "exec_tf": sig.exec_tf,
                "pivot_method": sig.pivot_method,
            }
        )
        next_available_idx = exit_idx + 1
    return pd.DataFrame(trades)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tuning = pd.read_csv(BTC_TUNING)
    frame = ensure_ohlcv_frame(pd.read_csv(BTC_DATA))
    train_end, validation_end = split_bounds(frame, validation_days=365, oos_days=365)
    fields = set(WolfeConfig.__dataclass_fields__)  # type: ignore[attr-defined]

    selected = tuning.sort_values(
        ["selection_score", "validation_net_r", "all_net_r"],
        ascending=[False, False, False],
    ).iloc[0]
    best_oos = tuning.sort_values(
        ["oos_net_r", "oos_profit_factor", "all_net_r"],
        ascending=[False, False, False],
    ).iloc[0]
    best_strict_like = tuning[
        (tuning["oos_trades"] >= 30)
        & (tuning["oos_net_r"] > 0)
        & (tuning["oos_profit_factor"] >= 1.2)
        & (tuning["oos_avg_r"] > 0.05)
    ].sort_values(["oos_net_r", "oos_profit_factor"], ascending=[False, False])

    configs = {
        "selected_by_train_validation": selected,
        "best_final_oos": best_oos,
    }
    if not best_strict_like.empty:
        configs["best_oos_with_30_trades"] = best_strict_like.iloc[0]

    overview_rows: list[dict[str, Any]] = []
    for name, row in configs.items():
        cfg = cfg_from_row(row)
        trades = run_backtest(frame, cfg, symbol="BTCUSDT")
        trades.to_csv(OUT_DIR / f"{name}_trades_static.csv", index=False)
        dynamic_trades = dynamic_epa_backtest(frame, cfg)
        dynamic_trades.to_csv(OUT_DIR / f"{name}_trades_dynamic_epa.csv", index=False)

        for mode, mode_trades in [("static_projection_target", trades), ("dynamic_epa_line_target", dynamic_trades)]:
            summaries = summarize_trades(mode_trades, train_end, validation_end)
            for key, table in summaries.items():
                table.to_csv(OUT_DIR / f"{name}_{mode}_{key}.csv", index=False)
            for split_name, split_frame in split_trades(mode_trades, train_end=train_end, validation_end=validation_end).items():
                overview_rows.append(
                    {
                        "config_name": name,
                        "target_mode": mode,
                        "split": split_name,
                        **{key: row[key] for key in fields if key in row.index and pd.notna(row[key])},
                        **strategy_metrics(split_frame),
                    }
                )

    overview = pd.DataFrame(overview_rows)
    overview.to_csv(OUT_DIR / "btc_postmortem_overview.csv", index=False)

    group = (
        tuning.groupby(["pattern_tf", "pivot_method"], as_index=False)
        .agg(
            configs=("exec_tf", "size"),
            median_train_net_r=("train_net_r", "median"),
            median_validation_net_r=("validation_net_r", "median"),
            median_oos_net_r=("oos_net_r", "median"),
            median_oos_trades=("oos_trades", "median"),
            median_all_net_r=("all_net_r", "median"),
            best_oos_net_r=("oos_net_r", "max"),
        )
        .sort_values(["median_oos_net_r", "best_oos_net_r"], ascending=[False, False])
    )
    group.to_csv(OUT_DIR / "btc_config_family_summary.csv", index=False)

    top_cols = [
        "pattern_tf",
        "pivot_method",
        "pivot_window",
        "max_time_ratio",
        "max_p5_break_atr",
        "stop_atr_buffer",
        "min_rr",
        "min_score",
        "target_projection_bars",
        "max_hold_bars",
        "trend_filter",
        "train_trades",
        "train_net_r",
        "validation_trades",
        "validation_net_r",
        "oos_trades",
        "oos_net_r",
        "oos_profit_factor",
        "all_net_r",
        "selection_score",
    ]
    tuning.sort_values(["oos_net_r", "oos_profit_factor"], ascending=[False, False]).head(15)[top_cols].to_csv(
        OUT_DIR / "btc_top_oos_configs.csv", index=False
    )
    tuning.sort_values(["selection_score", "validation_net_r"], ascending=[False, False]).head(15)[top_cols].to_csv(
        OUT_DIR / "btc_top_selection_configs.csv", index=False
    )

    print("BTC postmortem overview")
    print(overview[["config_name", "target_mode", "split", "trades", "net_r", "avg_r", "win_rate", "profit_factor", "max_dd_r"]].to_string(index=False))
    print("\nConfig family summary")
    print(group.to_string(index=False))
    print(f"\nWrote BTC postmortem artifacts to {OUT_DIR}")


if __name__ == "__main__":
    main()
