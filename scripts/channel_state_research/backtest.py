from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategyTrade:
    direction: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    stop_price: float
    target_price: float
    p_long: float
    p_short: float
    long_signal_score: float
    short_signal_score: float
    r_multiple_gross: float
    r_multiple_net: float
    return_pct: float
    hold_bars: int
    exit_reason: str


@dataclass(frozen=True)
class SignalGateSpec:
    min_probability_gap: float = 0.0
    allow_longs: bool = True
    allow_shorts: bool = True
    preset: str = "none"


def choose_signal_direction(
    row: pd.Series,
    long_threshold: float,
    short_threshold: float,
    gate_spec: SignalGateSpec | None = None,
    *,
    long_score_column: str = "p_long",
    short_score_column: str = "p_short",
) -> str | None:
    gate = gate_spec or SignalGateSpec()
    p_long = float(row["p_long"])
    p_short = float(row["p_short"])
    long_score = _row_value(row, long_score_column)
    short_score = _row_value(row, short_score_column)
    if not np.isfinite(long_score):
        long_score = p_long
    if not np.isfinite(short_score):
        short_score = p_short
    long_ready = gate.allow_longs and long_score >= long_threshold and (p_long - p_short) >= gate.min_probability_gap
    short_ready = gate.allow_shorts and short_score >= short_threshold and (p_short - p_long) >= gate.min_probability_gap
    if long_ready and not _direction_allowed(row, "long", gate):
        long_ready = False
    if short_ready and not _direction_allowed(row, "short", gate):
        short_ready = False
    if long_ready and not short_ready:
        return "long"
    if short_ready and not long_ready:
        return "short"
    if long_ready and short_ready:
        long_margin = long_score - long_threshold
        short_margin = short_score - short_threshold
        return "long" if long_margin >= short_margin else "short"
    return None


def simulate_threshold_strategy(
    frame: pd.DataFrame,
    *,
    long_threshold: float,
    short_threshold: float,
    alpha: float,
    beta: float,
    fee_bps_side: float,
    slippage_bps_side: float,
    risk_fraction: float,
    gate_spec: SignalGateSpec | None = None,
    long_score_column: str = "p_long",
    short_score_column: str = "p_short",
) -> tuple[pd.DataFrame, dict[str, float]]:
    ordered = frame.sort_values("decision_time").reset_index(drop=True).copy()
    trades: list[StrategyTrade] = []
    cursor = 0
    active_gate = gate_spec or SignalGateSpec()
    atr_column = _infer_atr_column(ordered)
    bar_seconds = _infer_bar_seconds(ordered)

    while cursor < len(ordered):
        row = ordered.iloc[cursor]
        direction = choose_signal_direction(
            row,
            long_threshold,
            short_threshold,
            gate_spec=active_gate,
            long_score_column=long_score_column,
            short_score_column=short_score_column,
        )
        if direction is None:
            cursor += 1
            continue

        entry_price = float(row["decision_close"])
        atr_value = float(row[atr_column])
        if not np.isfinite(atr_value) or atr_value <= 0.0:
            cursor += 1
            continue

        if direction == "long":
            stop_price = entry_price - beta * atr_value
            target_price = entry_price + alpha * atr_value
        else:
            stop_price = entry_price + beta * atr_value
            target_price = entry_price - alpha * atr_value

        risk = _safe_risk(entry_price, stop_price)
        if not np.isfinite(risk) or risk <= 0.0:
            cursor += 1
            continue

        if direction == "long":
            exit_price = target_price if float(row["tb_label"]) == 1.0 else stop_price if float(row["tb_label"]) == -1.0 else float(row["decision_close"]) * (1.0 + float(row["label_realized_return"]))
            gross_r = (exit_price - entry_price) / risk
        else:
            exit_price = target_price if float(row["tb_label"]) == -1.0 else stop_price if float(row["tb_label"]) == 1.0 else float(row["decision_close"]) * (1.0 + float(row["label_realized_return"]))
            gross_r = (entry_price - exit_price) / risk

        cost_r = ((2.0 * fee_bps_side) + (2.0 * slippage_bps_side)) / 10_000.0 * entry_price / risk
        net_r = gross_r - cost_r
        trade_return_pct = risk_fraction * net_r
        hold_bars = int(
            max(
                1,
                round(
                    (pd.Timestamp(row["label_end_time"]) - pd.Timestamp(row["decision_time"])).total_seconds() / bar_seconds
                ),
            )
        )
        trades.append(
            StrategyTrade(
                direction=direction,
                entry_time=pd.Timestamp(row["decision_time"]).tz_convert("UTC"),
                exit_time=pd.Timestamp(row["label_end_time"]).tz_convert("UTC"),
                entry_price=entry_price,
                exit_price=exit_price,
                stop_price=stop_price,
                target_price=target_price,
                p_long=float(row["p_long"]),
                p_short=float(row["p_short"]),
                long_signal_score=float(_row_value(row, long_score_column)),
                short_signal_score=float(_row_value(row, short_score_column)),
                r_multiple_gross=float(gross_r),
                r_multiple_net=float(net_r),
                return_pct=float(trade_return_pct),
                hold_bars=hold_bars,
                exit_reason=str(row["label_hit_reason"]),
            )
        )

        next_index = ordered["decision_time"].searchsorted(pd.Timestamp(row["label_end_time"]), side="right")
        cursor = int(max(cursor + 1, next_index))

    trades_frame = pd.DataFrame([trade.__dict__ for trade in trades])
    metrics = strategy_metrics(trades_frame)
    metrics["long_threshold"] = float(long_threshold)
    metrics["short_threshold"] = float(short_threshold)
    metrics["probability_gap"] = float(active_gate.min_probability_gap)
    metrics["gate_preset"] = active_gate.preset
    return trades_frame, metrics


def strategy_metrics(trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty:
        return {
            "trades": 0.0,
            "total_return": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "hit_rate": 0.0,
            "profit_factor": 0.0,
            "average_trade": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "turnover": 0.0,
            "exposure": 0.0,
            "long_only_return": 0.0,
            "short_only_return": 0.0,
            "net_r": 0.0,
        }

    returns = trades["return_pct"].astype(float)
    wins = returns[returns > 0.0]
    losses = returns[returns <= 0.0]
    equity_curve = (1.0 + returns).cumprod()
    running_peak = equity_curve.cummax()
    drawdown = equity_curve / running_peak - 1.0
    max_drawdown = float(drawdown.min())

    total_years = max(
        1e-9,
        (pd.Timestamp(trades["exit_time"].max()) - pd.Timestamp(trades["entry_time"].min())).total_seconds() / (365.25 * 24.0 * 3600.0),
    )
    trades_per_year = len(trades) / total_years if total_years > 0.0 else 0.0
    mean_return = float(returns.mean())
    std_return = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    downside = returns[returns < 0.0]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    annualized_return = float(equity_curve.iloc[-1] ** (1.0 / total_years) - 1.0) if total_years > 0.0 else 0.0

    gross_profit = float(wins.sum())
    gross_loss = abs(float(losses.sum()))
    long_only_return = float((1.0 + trades.loc[trades["direction"] == "long", "return_pct"].astype(float)).prod() - 1.0)
    short_only_return = float((1.0 + trades.loc[trades["direction"] == "short", "return_pct"].astype(float)).prod() - 1.0)

    return {
        "trades": float(len(trades)),
        "total_return": float(equity_curve.iloc[-1] - 1.0),
        "sharpe": mean_return / std_return * np.sqrt(trades_per_year) if std_return > 0.0 else 0.0,
        "sortino": mean_return / downside_std * np.sqrt(trades_per_year) if downside_std > 0.0 else 0.0,
        "max_drawdown": max_drawdown,
        "calmar": annualized_return / abs(max_drawdown) if max_drawdown < 0.0 else 0.0,
        "hit_rate": float((returns > 0.0).mean()),
        "profit_factor": gross_profit / gross_loss if gross_loss > 0.0 else (float("inf") if gross_profit > 0.0 else 0.0),
        "average_trade": mean_return,
        "average_win": float(wins.mean()) if len(wins) else 0.0,
        "average_loss": float(losses.mean()) if len(losses) else 0.0,
        "turnover": float(len(trades)),
        "exposure": float(trades["hold_bars"].astype(float).sum()) / max(
            1.0,
            (pd.Timestamp(trades["exit_time"].max()) - pd.Timestamp(trades["entry_time"].min())).total_seconds() / 3600.0,
        ),
        "long_only_return": long_only_return,
        "short_only_return": short_only_return,
        "net_r": float(trades["r_multiple_net"].astype(float).sum()),
    }


def _safe_risk(entry_price: float, stop_price: float) -> float:
    return abs(float(entry_price) - float(stop_price))


def _infer_bar_seconds(frame: pd.DataFrame) -> float:
    decision_times = pd.to_datetime(frame["decision_time"], utc=True, errors="coerce").dropna().sort_values()
    if len(decision_times) < 2:
        return 3600.0
    diffs = decision_times.diff().dropna().dt.total_seconds()
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return 3600.0
    return float(diffs.median())


def _infer_decision_timeframe(frame: pd.DataFrame) -> str:
    seconds = _infer_bar_seconds(frame)
    candidates = {
        "5m": 300.0,
        "15m": 900.0,
        "1h": 3600.0,
        "4h": 14400.0,
        "1d": 86400.0,
    }
    return min(candidates, key=lambda key: abs(candidates[key] - seconds))


def _infer_atr_column(frame: pd.DataFrame) -> str:
    decision_tf = _infer_decision_timeframe(frame)
    preferred = f"atr_tf_{decision_tf}"
    if preferred in frame.columns:
        return preferred
    if "atr_tf_1h" in frame.columns:
        return "atr_tf_1h"
    atr_columns = sorted(column for column in frame.columns if column.startswith("atr_tf_"))
    if atr_columns:
        return atr_columns[0]
    raise KeyError("No atr_tf_* column found for channel-state backtest.")


def _direction_allowed(row: pd.Series, direction: str, gate_spec: SignalGateSpec) -> bool:
    preset = (gate_spec.preset or "none").strip().lower()
    if preset == "none":
        return True

    num_bullish = _row_value(row, "num_bullish_timeframes")
    num_bearish = _row_value(row, "num_bearish_timeframes")
    breakout_uptrend = _row_value(row, "1h_breakout_above_body_and_1d_uptrend")
    lower_body_uptrend = _row_value(row, "1h_near_lower_body_and_4h_uptrend")

    if direction == "short":
        if preset == "bearish_3plus":
            return num_bearish >= 3.0
        if preset == "trend_alignment":
            return num_bearish >= 2.0
        if preset == "combo_context":
            return not (num_bullish >= 3.0 or breakout_uptrend >= 1.0 or lower_body_uptrend >= 1.0)
        if preset == "combo_bearish":
            return num_bearish >= 2.0 and not (
                num_bullish >= 3.0 or breakout_uptrend >= 1.0 or lower_body_uptrend >= 1.0
            )
        return True

    if direction == "long":
        if preset == "trend_alignment":
            return num_bullish >= 2.0
        if preset in {"bearish_3plus", "combo_bearish"}:
            return num_bullish >= 2.0
        return True

    return True


def _row_value(row: pd.Series, column: str) -> float:
    if column not in row.index:
        return np.nan
    value = row[column]
    if pd.isna(value):
        return np.nan
    return float(value)
