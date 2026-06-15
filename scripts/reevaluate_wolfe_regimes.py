from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backtest_wolfe_wave import (  # noqa: E402
    WolfeConfig,
    add_indicators,
    ensure_ohlcv_frame,
    fetch_bybit_klines,
    find_wolfe_signals,
    resample_ohlc,
    rma,
    run_backtest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-evaluate deployed Wolfe configs across recent windows and regimes.")
    parser.add_argument("--old-config", type=Path, default=Path("bot/configs/wolfe_wave_configs.json"))
    parser.add_argument("--v2-config", type=Path, default=Path("bot/configs/wolfe_wave_v2_strong_configs.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("scripts/data_wolfe_top100_deep5y"))
    parser.add_argument("--output-dir", type=Path, default=Path("scripts/wolfe_regime_reeval"))
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--warmup-days", type=int, default=90)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--end", help="UTC ISO timestamp; defaults to now.")
    parser.add_argument("--symbols", help="Optional comma-separated symbol subset.")
    parser.add_argument(
        "--context-timeframes",
        default="15m,1h,4h,1d",
        help="Comma-separated completed-bar context timeframes.",
    )
    parser.add_argument("--no-refresh-tail", action="store_true")
    return parser.parse_args()


def utc_timestamp(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def load_configs(path: Path, strategy: str) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a symbol-to-config object")
    return {
        str(symbol).upper(): {"strategy": strategy, "config": config}
        for symbol, config in payload.items()
        if not str(symbol).startswith("_") and isinstance(config, dict)
    }


def load_symbol_frame(
    symbol: str,
    *,
    cache_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    refresh_tail: bool,
) -> pd.DataFrame:
    cache_path = cache_dir / f"{symbol.lower()}_5m_bybit.csv"
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)
    frame = ensure_ohlcv_frame(pd.read_csv(cache_path))
    if frame.empty:
        raise ValueError(f"{symbol}: empty cache")

    last_open = utc_timestamp(frame["open_time"].iloc[-1])
    if refresh_tail and last_open < end - pd.Timedelta(minutes=10):
        fetch_start = max(last_open + pd.Timedelta(minutes=5), start)
        if fetch_start < end:
            tail = fetch_bybit_klines(
                symbol,
                "5m",
                fetch_start.to_pydatetime(),
                end.to_pydatetime(),
            )
            if not tail.empty:
                frame = ensure_ohlcv_frame(
                    pd.concat([frame, tail], ignore_index=True)
                    .drop_duplicates("open_time", keep="last")
                    .sort_values("open_time")
                )

    times = pd.to_datetime(frame["open_time"], utc=True)
    return frame.loc[(times >= start) & (times <= end)].reset_index(drop=True)


def add_context_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    out = add_indicators(frame, 14, 200, 14)
    prev_close = out["close"].shift(1)
    true_range = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    up_move = out["high"].diff()
    down_move = -out["low"].diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0),
        index=out.index,
        dtype=float,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0),
        index=out.index,
        dtype=float,
    )
    smoothed_tr = rma(true_range, 14).replace(0.0, np.nan)
    plus_di = 100.0 * rma(plus_dm, 14) / smoothed_tr
    minus_di = 100.0 * rma(minus_dm, 14) / smoothed_tr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    out["adx"] = rma(dx, 14)
    out["adx_delta_3"] = out["adx"] - out["adx"].shift(3)
    out["atr_percentile_100"] = out["atr"].rolling(100, min_periods=50).rank(pct=True)
    out["ema_distance_atr"] = (out["close"] - out["ema"]) / out["atr"].replace(0.0, np.nan)
    macd = out["close"].ewm(span=12, adjust=False).mean() - out["close"].ewm(span=26, adjust=False).mean()
    out["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    return out


def build_context_frames(frame: pd.DataFrame, timeframes: list[str]) -> dict[str, pd.DataFrame]:
    contexts: dict[str, pd.DataFrame] = {}
    for timeframe in timeframes:
        context = add_context_indicators(resample_ohlc(frame, timeframe))
        context["close_time"] = pd.to_datetime(context["close_time"], utc=True)
        contexts[timeframe] = context
    return contexts


def attach_signal_features(
    trades: pd.DataFrame,
    signals: list[Any],
    pattern: pd.DataFrame,
) -> pd.DataFrame:
    if trades.empty:
        return trades
    rows: list[dict[str, Any]] = []
    for signal in signals:
        p1, _, p3, _, p5 = signal.pivots
        p3_row = pattern.iloc[min(max(int(p3.idx), 0), len(pattern) - 1)]
        p5_row = pattern.iloc[min(max(int(p5.idx), 0), len(pattern) - 1)]
        p3_rsi = float(p3_row.get("rsi", math.nan))
        p5_rsi = float(p5_row.get("rsi", math.nan))
        p3_macd = float(p3_row.get("macd_hist", math.nan))
        p5_macd = float(p5_row.get("macd_hist", math.nan))
        if signal.direction == "long":
            rsi_divergence_delta = p5_rsi - p3_rsi
            macd_divergence_delta = p5_macd - p3_macd
            p5_rsi_exhausted = p5_rsi < 35.0
        else:
            rsi_divergence_delta = p3_rsi - p5_rsi
            macd_divergence_delta = p3_macd - p5_macd
            p5_rsi_exhausted = p5_rsi > 65.0
        rows.append(
            {
                "event_key": signal.event_key,
                "pattern_span_bars": int(p5.idx - p1.idx),
                "leg_13_bars": int(p3.idx - p1.idx),
                "leg_35_bars": int(p5.idx - p3.idx),
                "p3_rsi": p3_rsi,
                "p5_rsi": p5_rsi,
                "rsi_divergence_delta": rsi_divergence_delta,
                "rsi_divergence": bool(math.isfinite(rsi_divergence_delta) and rsi_divergence_delta > 0.0),
                "rsi_divergence_3": bool(math.isfinite(rsi_divergence_delta) and rsi_divergence_delta >= 3.0),
                "p5_rsi_exhausted": bool(math.isfinite(p5_rsi) and p5_rsi_exhausted),
                "p3_macd_hist": p3_macd,
                "p5_macd_hist": p5_macd,
                "macd_divergence_delta": macd_divergence_delta,
                "macd_divergence": bool(
                    math.isfinite(macd_divergence_delta) and macd_divergence_delta > 0.0
                ),
            }
        )
    features = pd.DataFrame(rows).drop_duplicates("event_key", keep="last")
    return trades.merge(features, on="event_key", how="left")


def attach_multitimeframe_context(
    trades: pd.DataFrame,
    contexts: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    if trades.empty:
        return trades
    out = trades.copy()
    out["entry_time"] = pd.to_datetime(out["entry_time"], utc=True)
    for timeframe, context_frame in contexts.items():
        columns = [
            "close_time",
            "close",
            "ema",
            "ema_slope_atr",
            "atr_ratio",
            "atr_percentile_100",
            "adx",
            "adx_delta_3",
            "rsi",
            "volume_ratio",
            "ema_distance_atr",
        ]
        suffix = timeframe.lower()
        context = context_frame[columns].rename(
            columns={column: f"{column}_{suffix}" for column in columns if column != "close_time"}
        )
        out = pd.merge_asof(
            out.sort_values("entry_time"),
            context.sort_values("close_time"),
            left_on="entry_time",
            right_on="close_time",
            direction="backward",
        ).drop(columns=["close_time"], errors="ignore")

        close_col = f"close_{suffix}"
        ema_col = f"ema_{suffix}"
        slope_col = f"ema_slope_atr_{suffix}"
        adx_col = f"adx_{suffix}"
        distance_col = f"ema_distance_atr_{suffix}"
        direction_sign = out["direction"].map({"long": 1.0, "short": -1.0})
        signed_price_side = direction_sign * (out[close_col] - out[ema_col])
        signed_slope = direction_sign * out[slope_col]
        out[f"ema_aligned_{suffix}"] = (signed_price_side >= 0.0) & (signed_slope >= 0.0)
        out[f"ema_opposed_{suffix}"] = (signed_price_side < 0.0) & (signed_slope < 0.0)
        out[f"runaway_against_{suffix}"] = out[f"ema_opposed_{suffix}"] & (
            out[adx_col] >= 25.0
        )
        out[f"adx_falling_{suffix}"] = out[f"adx_delta_3_{suffix}"] < 0.0
        out[f"reversal_stretch_atr_{suffix}"] = -direction_sign * out[distance_col]
        out[f"vol_regime_{suffix}"] = np.where(
            out[f"atr_ratio_{suffix}"] >= 1.0,
            "high_vol",
            "low_vol",
        )
        out[f"directional_regime_{suffix}"] = np.select(
            [out[f"ema_aligned_{suffix}"], out[f"ema_opposed_{suffix}"]],
            ["trend_aligned", "mean_reversion"],
            default="transition",
        )
    return out


def evaluate_symbol(task: dict[str, Any]) -> list[dict[str, Any]]:
    symbol = task["symbol"]
    frame = load_symbol_frame(
        symbol,
        cache_dir=Path(task["cache_dir"]),
        start=utc_timestamp(task["start"]),
        end=utc_timestamp(task["end"]),
        refresh_tail=bool(task["refresh_tail"]),
    )
    contexts = build_context_frames(frame, list(task["context_timeframes"]))
    buckets: list[pd.DataFrame] = []
    for item in task["configs"]:
        cfg = WolfeConfig.from_mapping(item["config"])
        signals = find_wolfe_signals(frame, cfg, symbol=symbol)
        trades = run_backtest(frame, cfg, symbol=symbol, precomputed_signals=signals)
        if trades.empty:
            continue
        pattern = add_context_indicators(resample_ohlc(frame, cfg.pattern_tf))
        trades = attach_signal_features(trades, signals, pattern)
        trades["strategy"] = item["strategy"]
        trades["configured_regime_filter"] = cfg.regime_filter
        trades["configured_min_rr"] = cfg.min_rr
        trades["config_json"] = json.dumps(asdict(cfg), sort_keys=True)
        buckets.append(trades)
    if not buckets:
        return []
    combined = attach_multitimeframe_context(pd.concat(buckets, ignore_index=True), contexts)
    return combined.to_dict("records")


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty or "r_multiple_net" not in frame:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "net_r": 0.0,
            "avg_r": 0.0,
            "avg_win_r": 0.0,
            "avg_loss_r": 0.0,
            "payoff_ratio": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_r": 0.0,
            "avg_hold_bars": 0.0,
            "stop_rate": 0.0,
            "target_rate": 0.0,
            "timeout_rate": 0.0,
            "max_losing_streak": 0,
        }
    ordered = (
        frame.sort_values("entry_time", kind="stable")
        if "entry_time" in frame.columns
        else frame
    )
    r = pd.to_numeric(ordered["r_multiple_net"], errors="coerce").fillna(0.0)
    wins = r > 0
    gains = float(r[wins].sum())
    losses_r = abs(float(r[~wins].sum()))
    avg_win = float(r[wins].mean()) if wins.any() else 0.0
    avg_loss = abs(float(r[~wins].mean())) if (~wins).any() else 0.0
    equity = pd.concat(
        [pd.Series([0.0]), r.cumsum().reset_index(drop=True)],
        ignore_index=True,
    )
    drawdown = equity - equity.cummax()
    reasons = ordered["exit_reason"].fillna("").astype(str).str.lower()
    max_streak = streak = 0
    for won in wins:
        streak = 0 if bool(won) else streak + 1
        max_streak = max(max_streak, streak)
    return {
        "trades": int(len(frame)),
        "wins": int(wins.sum()),
        "losses": int((~wins).sum()),
        "win_rate": float(wins.mean()),
        "net_r": float(r.sum()),
        "avg_r": float(r.mean()),
        "avg_win_r": avg_win,
        "avg_loss_r": avg_loss,
        "payoff_ratio": avg_win / avg_loss if avg_loss > 0 else (math.inf if avg_win > 0 else 0.0),
        "profit_factor": gains / losses_r if losses_r > 0 else (math.inf if gains > 0 else 0.0),
        "max_drawdown_r": abs(float(drawdown.min())) if not drawdown.empty else 0.0,
        "avg_hold_bars": float(pd.to_numeric(ordered["hold_bars"], errors="coerce").mean()),
        "stop_rate": float(reasons.str.startswith("stop").mean()),
        "target_rate": float(reasons.str.startswith("target").mean()),
        "timeout_rate": float(reasons.eq("timeout").mean()),
        "max_losing_streak": int(max_streak),
    }


def aggregate_windows(trades: pd.DataFrame, end: pd.Timestamp, windows: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    entry_times = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce")
    for strategy in sorted(trades["strategy"].dropna().unique()):
        strategy_frame = trades[trades["strategy"] == strategy]
        strategy_times = entry_times.loc[strategy_frame.index]
        for days in windows:
            bucket = strategy_frame[strategy_times >= end - pd.Timedelta(days=days)]
            rows.append({"strategy": strategy, "window_days": days, **metrics(bucket)})
    return pd.DataFrame(rows)


def aggregate_groups(trades: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, bucket in trades.groupby(columns, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        rows.append({**dict(zip(columns, key_values)), **metrics(bucket)})
    return pd.DataFrame(rows)


def hypothesis_filters(
    trades: pd.DataFrame,
    context_timeframes: list[str],
) -> list[dict[str, Any]]:
    filters: list[dict[str, Any]] = []

    def add(
        hypothesis: str,
        family: str,
        name: str,
        mask: pd.Series,
        *,
        timeframe: str = "trade",
        threshold: float = 0.0,
    ) -> None:
        filters.append(
            {
                "hypothesis": hypothesis,
                "family": family,
                "filter_name": name,
                "timeframe": timeframe,
                "threshold": float(threshold),
                "mask": mask.fillna(False).astype(bool),
            }
        )

    direction = trades["direction"].astype(str)
    entry_rsi = pd.to_numeric(trades["rsi"], errors="coerce")
    add(
        "H3 exhaustion",
        "entry_rsi",
        "entry RSI exhausted (35/65)",
        ((direction == "long") & (entry_rsi <= 35.0))
        | ((direction == "short") & (entry_rsi >= 65.0)),
        timeframe="5m",
        threshold=35.0,
    )
    add(
        "H3 exhaustion",
        "p5_rsi_divergence",
        "point-5 RSI divergence",
        trades["rsi_divergence"],
        timeframe="pattern",
        threshold=0.0,
    )
    add(
        "H3 exhaustion",
        "p5_rsi_divergence",
        "point-5 RSI divergence >= 3",
        pd.to_numeric(trades["rsi_divergence_delta"], errors="coerce") >= 3.0,
        timeframe="pattern",
        threshold=3.0,
    )
    add(
        "H3 exhaustion",
        "p5_rsi_level",
        "point-5 RSI exhausted (35/65)",
        trades["p5_rsi_exhausted"],
        timeframe="pattern",
        threshold=35.0,
    )
    add(
        "H3 exhaustion",
        "p5_macd_divergence",
        "point-5 MACD histogram divergence",
        trades["macd_divergence"],
        timeframe="pattern",
        threshold=0.0,
    )

    rr = pd.to_numeric(trades["target_rr_planned"], errors="coerce")
    for threshold in (1.5, 2.0, 2.5):
        add(
            "H6 EPA reward/risk",
            "minimum_rr",
            f"planned EPA/R >= {threshold:.1f}",
            rr >= threshold,
            threshold=threshold,
        )

    symmetry = pd.to_numeric(trades["symmetry_ratio"], errors="coerce")
    for threshold in (1.4, 1.6, 2.0):
        add(
            "H5 geometry",
            "symmetry",
            f"symmetry ratio <= {threshold:.1f}",
            symmetry <= threshold,
            timeframe="pattern",
            threshold=threshold,
        )
    p5_break = pd.to_numeric(trades["p5_break_atr"], errors="coerce")
    for threshold in (1.0, 1.5, 2.0):
        add(
            "H5 geometry",
            "p5_overshoot",
            f"point-5 overshoot <= {threshold:.1f} ATR",
            p5_break <= threshold,
            timeframe="pattern",
            threshold=threshold,
        )

    entry_volume = pd.to_numeric(trades["volume_ratio"], errors="coerce")
    for threshold in (1.0, 1.2, 1.5):
        add(
            "H7 volume confirmation",
            "entry_volume",
            f"entry volume ratio >= {threshold:.1f}",
            entry_volume >= threshold,
            timeframe="5m",
            threshold=threshold,
        )
    p5_volume = pd.to_numeric(trades["p5_volume_ratio"], errors="coerce")
    p5_rejection = pd.to_numeric(trades["p5_rejection_atr"], errors="coerce")
    add(
        "H7 volume confirmation",
        "p5_volume",
        "point-5 volume ratio >= 1.0",
        p5_volume >= 1.0,
        timeframe="pattern",
        threshold=1.0,
    )
    add(
        "H7 volume confirmation",
        "p5_rejection",
        "point-5 rejection >= 0.2 ATR",
        p5_rejection >= 0.2,
        timeframe="pattern",
        threshold=0.2,
    )
    add(
        "H7 volume confirmation",
        "p5_volume_rejection",
        "point-5 volume >= 1.0 and rejection >= 0.2 ATR",
        (p5_volume >= 1.0) & (p5_rejection >= 0.2),
        timeframe="pattern",
        threshold=1.0,
    )

    for timeframe in context_timeframes:
        suffix = timeframe.lower()
        adx = pd.to_numeric(trades[f"adx_{suffix}"], errors="coerce")
        for threshold in (20.0, 25.0, 30.0):
            add(
                "H1 runaway trend",
                "adx_cap",
                f"{timeframe} ADX < {threshold:.0f}",
                adx < threshold,
                timeframe=timeframe,
                threshold=threshold,
            )
        add(
            "H1 runaway trend",
            "adx_falling",
            f"{timeframe} ADX falling over 3 bars",
            trades[f"adx_falling_{suffix}"],
            timeframe=timeframe,
            threshold=3.0,
        )
        add(
            "H1 runaway trend",
            "avoid_runaway_against",
            f"avoid {timeframe} ADX>=25 opposing trend",
            ~trades[f"runaway_against_{suffix}"],
            timeframe=timeframe,
            threshold=25.0,
        )

        add(
            "H2 higher-timeframe context",
            "ema_not_opposed",
            f"{timeframe} EMA structure not opposed",
            ~trades[f"ema_opposed_{suffix}"],
            timeframe=timeframe,
            threshold=0.0,
        )
        direction_sign = direction.map({"long": 1.0, "short": -1.0})
        price_side = direction_sign * (
            pd.to_numeric(trades[f"close_{suffix}"], errors="coerce")
            - pd.to_numeric(trades[f"ema_{suffix}"], errors="coerce")
        )
        signed_slope = direction_sign * pd.to_numeric(
            trades[f"ema_slope_atr_{suffix}"],
            errors="coerce",
        )
        add(
            "H2 higher-timeframe context",
            "ema_price_side",
            f"{timeframe} price on reversal side of EMA200",
            price_side >= 0.0,
            timeframe=timeframe,
            threshold=0.0,
        )
        add(
            "H2 higher-timeframe context",
            "ema_slope",
            f"{timeframe} EMA200 slope not opposed",
            signed_slope >= 0.0,
            timeframe=timeframe,
            threshold=0.0,
        )
        add(
            "H2 higher-timeframe context",
            "ema_aligned",
            f"{timeframe} EMA structure aligned",
            trades[f"ema_aligned_{suffix}"],
            timeframe=timeframe,
            threshold=1.0,
        )
        stretch = pd.to_numeric(trades[f"reversal_stretch_atr_{suffix}"], errors="coerce")
        for threshold in (0.0, 1.0, 2.0):
            add(
                "H2 higher-timeframe context",
                "healthy_stretch",
                f"{timeframe} reversal stretch >= {threshold:.0f} ATR, no runaway",
                (stretch >= threshold) & ~trades[f"runaway_against_{suffix}"],
                timeframe=timeframe,
                threshold=threshold,
            )

        atr_percentile = pd.to_numeric(
            trades[f"atr_percentile_100_{suffix}"],
            errors="coerce",
        )
        add(
            "H4 volatility normalization",
            "atr_middle",
            f"{timeframe} ATR percentile 30-80",
            atr_percentile.between(0.30, 0.80, inclusive="both"),
            timeframe=timeframe,
            threshold=0.30,
        )
        add(
            "H4 volatility normalization",
            "atr_not_extreme",
            f"{timeframe} ATR percentile 10-90",
            atr_percentile.between(0.10, 0.90, inclusive="both"),
            timeframe=timeframe,
            threshold=0.10,
        )
        add(
            "H4 volatility normalization",
            "atr_upper_cap",
            f"{timeframe} ATR percentile <= 90",
            atr_percentile <= 0.90,
            timeframe=timeframe,
            threshold=0.90,
        )
        add(
            "H4 volatility normalization",
            "atr_lower_floor",
            f"{timeframe} ATR percentile >= 10",
            atr_percentile >= 0.10,
            timeframe=timeframe,
            threshold=0.10,
        )
    return filters


def stability_metrics(baseline: pd.DataFrame, selected: pd.DataFrame) -> dict[str, float]:
    symbols, symbols_improved, symbols_positive = compare_groups_from_frames(
        baseline,
        selected,
        "symbol",
        5,
        3,
    )
    base = baseline.copy()
    chosen = selected.copy()
    base_times = pd.to_datetime(base["entry_time"], utc=True)
    chosen_times = pd.to_datetime(chosen["entry_time"], utc=True)
    base["_quarter"] = (
        base_times.dt.year.astype(str) + "-Q" + base_times.dt.quarter.astype(str)
    )
    chosen["_quarter"] = (
        chosen_times.dt.year.astype(str) + "-Q" + chosen_times.dt.quarter.astype(str)
    )
    periods, periods_improved, periods_positive = compare_groups_from_frames(
        base,
        chosen,
        "_quarter",
        20,
        8,
    )
    directions, directions_improved, directions_positive = compare_groups_from_frames(
        baseline,
        selected,
        "direction",
        20,
        8,
    )
    pattern_tfs, pattern_tfs_improved, pattern_tfs_positive = compare_groups_from_frames(
        baseline,
        selected,
        "pattern_tf",
        20,
        8,
    )
    return {
        "symbols_evaluable": symbols,
        "symbols_improved_pct": symbols_improved,
        "symbols_positive_pct": symbols_positive,
        "periods_evaluable": periods,
        "periods_improved_pct": periods_improved,
        "periods_positive_pct": periods_positive,
        "directions_evaluable": directions,
        "directions_improved_pct": directions_improved,
        "directions_positive_pct": directions_positive,
        "pattern_tfs_evaluable": pattern_tfs,
        "pattern_tfs_improved_pct": pattern_tfs_improved,
        "pattern_tfs_positive_pct": pattern_tfs_positive,
    }


def compare_groups_from_frames(
    baseline: pd.DataFrame,
    selected: pd.DataFrame,
    column: str,
    minimum_baseline: int,
    minimum_selected: int,
) -> tuple[int, float, float]:
    baseline_values = baseline[[column, "r_multiple_net"]].copy()
    selected_values = selected[[column, "r_multiple_net"]].copy()
    baseline_values["r_multiple_net"] = pd.to_numeric(
        baseline_values["r_multiple_net"],
        errors="coerce",
    )
    selected_values["r_multiple_net"] = pd.to_numeric(
        selected_values["r_multiple_net"],
        errors="coerce",
    )
    base_stats = baseline_values.groupby(column, dropna=False)["r_multiple_net"].agg(
        baseline_count="count",
        baseline_avg="mean",
    )
    selected_stats = selected_values.groupby(column, dropna=False)["r_multiple_net"].agg(
        selected_count="count",
        selected_avg="mean",
    )
    comparison = base_stats.join(selected_stats, how="left").fillna(
        {"selected_count": 0, "selected_avg": 0.0}
    )
    comparison = comparison[
        (comparison["baseline_count"] >= minimum_baseline)
        & (comparison["selected_count"] >= minimum_selected)
    ]
    improved = comparison["selected_avg"] > comparison["baseline_avg"]
    positive = comparison["selected_avg"] > 0.0
    return (
        int(len(comparison)),
        float(improved.mean()) if len(comparison) else math.nan,
        float(positive.mean()) if len(comparison) else math.nan,
    )


def evaluate_hypotheses(
    trades: pd.DataFrame,
    *,
    end: pd.Timestamp,
    windows: list[int],
    context_timeframes: list[str],
) -> pd.DataFrame:
    candidates = hypothesis_filters(trades, context_timeframes)
    rows: list[dict[str, Any]] = []
    strategies = ["combined", *sorted(trades["strategy"].dropna().unique())]
    entry_times = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce")
    for strategy in strategies:
        strategy_mask = (
            pd.Series(True, index=trades.index)
            if strategy == "combined"
            else trades["strategy"].eq(strategy)
        )
        for window_days in windows:
            window_mask = entry_times >= end - pd.Timedelta(days=window_days)
            baseline = trades[strategy_mask & window_mask]
            if baseline.empty:
                continue
            baseline_metrics = metrics(baseline)
            for candidate in candidates:
                selected = trades[strategy_mask & window_mask & candidate["mask"]]
                excluded = trades[strategy_mask & window_mask & ~candidate["mask"]]
                selected_metrics = metrics(selected)
                excluded_metrics = metrics(excluded)
                stable = stability_metrics(baseline, selected)
                rows.append(
                    {
                        "strategy": strategy,
                        "window_days": window_days,
                        "hypothesis": candidate["hypothesis"],
                        "family": candidate["family"],
                        "filter_name": candidate["filter_name"],
                        "timeframe": candidate["timeframe"],
                        "threshold": candidate["threshold"],
                        **selected_metrics,
                        "retention": len(selected) / len(baseline),
                        "net_r_retention": (
                            selected_metrics["net_r"] / baseline_metrics["net_r"]
                            if baseline_metrics["net_r"] != 0.0
                            else math.nan
                        ),
                        "baseline_trades": len(baseline),
                        "baseline_win_rate": baseline_metrics["win_rate"],
                        "baseline_avg_r": baseline_metrics["avg_r"],
                        "baseline_profit_factor": baseline_metrics["profit_factor"],
                        "excluded_trades": excluded_metrics["trades"],
                        "excluded_win_rate": excluded_metrics["win_rate"],
                        "excluded_net_r": excluded_metrics["net_r"],
                        "excluded_avg_r": excluded_metrics["avg_r"],
                        "excluded_profit_factor": excluded_metrics["profit_factor"],
                        "delta_win_rate": selected_metrics["win_rate"] - baseline_metrics["win_rate"],
                        "delta_avg_r": selected_metrics["avg_r"] - baseline_metrics["avg_r"],
                        "delta_profit_factor": (
                            selected_metrics["profit_factor"] - baseline_metrics["profit_factor"]
                        ),
                        **stable,
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(
        ["strategy", "window_days", "hypothesis", "family", "timeframe", "threshold"]
    ).reset_index(drop=True)
    group_columns = ["strategy", "window_days", "hypothesis", "family", "timeframe"]
    out["neighbor_median_delta_avg_r"] = out.groupby(group_columns, dropna=False)[
        "delta_avg_r"
    ].transform(lambda values: values.rolling(3, center=True, min_periods=1).median())
    symbol_support = out["symbols_improved_pct"].fillna(0.5)
    period_support = out["periods_improved_pct"].fillna(0.5)
    out["robust_score"] = (
        out["neighbor_median_delta_avg_r"] * np.sqrt(out["retention"].clip(lower=0.0))
        + 0.10 * (symbol_support - 0.5)
        + 0.05 * (period_support - 0.5)
    )
    out["verdict"] = np.select(
        [
            (out["trades"] >= 50)
            & (out["retention"] >= 0.20)
            & (out["neighbor_median_delta_avg_r"] >= 0.03)
            & (symbol_support >= 0.55)
            & (period_support >= 0.55)
            & (out["avg_r"] > 0.0),
            (out["trades"] >= 30)
            & (out["neighbor_median_delta_avg_r"] > 0.0)
            & (out["avg_r"] > 0.0),
        ],
        ["supported", "mixed"],
        default="not_supported",
    )
    out["deployment_use"] = np.select(
        [
            (out["verdict"] == "supported")
            & (out["excluded_trades"] >= 30)
            & (out["excluded_avg_r"] <= 0.0),
            (out["verdict"].isin(["supported", "mixed"]))
            & (out["excluded_trades"] >= 30)
            & (out["excluded_avg_r"] > 0.0),
        ],
        ["hard_gate_candidate", "ranking_or_risk_tilt"],
        default="research_only",
    )
    return out


def rr_gate_recommendation(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in summary[summary["window_days"] == summary["window_days"].max()].iterrows():
        trades = int(row["trades"])
        win_rate = float(row["win_rate"])
        if trades < 8 or win_rate >= 0.50:
            required_rr = 0.0
        elif win_rate <= 0:
            required_rr = 2.0
        else:
            required_rr = min(2.0, max(1.5, (1.0 - win_rate) / win_rate + 0.20))
        rows.append(
            {
                "strategy": row["strategy"],
                "sample_trades": trades,
                "sample_win_rate": win_rate,
                "recommended_min_rr": required_rr,
            }
        )
    return pd.DataFrame(rows)


def write_report(
    output_path: Path,
    *,
    summary: pd.DataFrame,
    rr_bands: pd.DataFrame,
    regimes: pd.DataFrame,
    recommendations: pd.DataFrame,
    hypothesis_results: pd.DataFrame,
) -> None:
    maximum_window = int(summary["window_days"].max()) if not summary.empty else 0
    lines = [
        "# Wolfe Multi-Timeframe Hypothesis Re-evaluation",
        "",
        "The deployed entries and exits are held constant. Each research rule is applied",
        "one at a time as a counterfactual gate. `neighbor_median_delta_avg_r` is the",
        "median improvement across adjacent thresholds in the same family, reducing",
        "the influence of a lucky single cutoff.",
        "",
    ]
    for title, table in (
        ("Rolling Summary", summary),
        (f"RR Bands ({maximum_window}d)", rr_bands),
        (f"Regimes ({maximum_window}d)", regimes),
        ("Research RR Threshold", recommendations),
    ):
        lines.extend([f"## {title}", "", "```text", table.to_string(index=False), "```", ""])
    if not hypothesis_results.empty:
        maximum_window = int(hypothesis_results["window_days"].max())
        combined = hypothesis_results[
            (hypothesis_results["strategy"] == "combined")
            & (hypothesis_results["window_days"] == maximum_window)
        ].copy()
        top_by_hypothesis = (
            combined.sort_values("robust_score", ascending=False)
            .groupby("hypothesis", as_index=False)
            .head(3)
        )
        display_columns = [
            "hypothesis",
            "verdict",
            "deployment_use",
            "filter_name",
            "timeframe",
            "trades",
            "retention",
            "net_r_retention",
            "win_rate",
            "avg_win_r",
            "avg_loss_r",
            "avg_r",
            "profit_factor",
            "excluded_trades",
            "excluded_avg_r",
            "delta_avg_r",
            "neighbor_median_delta_avg_r",
            "symbols_improved_pct",
            "periods_improved_pct",
            "robust_score",
        ]
        lines.extend(
            [
                f"## Best Hypothesis Tests ({maximum_window}d combined)",
                "",
                "```text",
                top_by_hypothesis[display_columns].to_string(index=False),
                "```",
                "",
            ]
        )
        mtf = combined[
            combined["hypothesis"].isin(
                [
                    "H1 runaway trend",
                    "H2 higher-timeframe context",
                    "H4 volatility normalization",
                ]
            )
        ]
        mtf = (
            mtf.sort_values("robust_score", ascending=False)
            .groupby(["hypothesis", "timeframe"], as_index=False)
            .head(1)
        )
        lines.extend(
            [
                "## Best Multi-Timeframe Context Rule",
                "",
                "```text",
                mtf[display_columns].to_string(index=False),
                "```",
                "",
            ]
        )
        strategy_best = (
            hypothesis_results[
                hypothesis_results["window_days"] == maximum_window
            ]
            .sort_values("robust_score", ascending=False)
            .groupby(["strategy", "hypothesis"], as_index=False)
            .head(1)
        )
        lines.extend(
            [
                "## Best Rule by Strategy",
                "",
                "```text",
                strategy_best[["strategy", *display_columns]].to_string(index=False),
                "```",
                "",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    end = utc_timestamp(args.end or datetime.now(timezone.utc))
    analysis_start = end - pd.Timedelta(days=max(1, args.days))
    context_timeframes = [
        item.strip().lower()
        for item in args.context_timeframes.split(",")
        if item.strip()
    ]
    if not context_timeframes:
        raise ValueError("At least one context timeframe is required")
    minimum_warmup_days = 400 if "1d" in context_timeframes else 90
    effective_warmup_days = max(1, args.warmup_days, minimum_warmup_days)
    load_start = analysis_start - pd.Timedelta(days=effective_warmup_days)
    old = load_configs(args.old_config, "wolfe_wave")
    v2 = load_configs(args.v2_config, "wolfe_wave_v2")

    by_symbol: dict[str, list[dict[str, Any]]] = {}
    for source in (old, v2):
        for symbol, item in source.items():
            by_symbol.setdefault(symbol, []).append(item)
    if args.symbols:
        selected = {item.strip().upper() for item in args.symbols.split(",") if item.strip()}
        by_symbol = {symbol: configs for symbol, configs in by_symbol.items() if symbol in selected}
    tasks = [
        {
            "symbol": symbol,
            "configs": configs,
            "cache_dir": str(args.cache_dir),
            "start": load_start.isoformat(),
            "end": end.isoformat(),
            "refresh_tail": not args.no_refresh_tail,
            "context_timeframes": context_timeframes,
        }
        for symbol, configs in sorted(by_symbol.items())
    ]
    print(
        f"Wolfe regime re-evaluation symbols={len(tasks)} "
        f"configs={sum(len(item['configs']) for item in tasks)} "
        f"analysis={analysis_start.date()}->{end.date()} "
        f"context={','.join(context_timeframes)} warmup={effective_warmup_days}d "
        f"workers={args.workers}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(evaluate_symbol, task): task["symbol"] for task in tasks}
        for index, future in enumerate(as_completed(futures), start=1):
            symbol = futures[future]
            try:
                rows.extend(future.result())
                print(f"[{index}/{len(tasks)}] {symbol} done", flush=True)
            except Exception as exc:  # noqa: BLE001
                failures.append({"symbol": symbol, "error": str(exc)})
                print(f"[{index}/{len(tasks)}] {symbol} failed: {exc}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if failures:
        pd.DataFrame(failures).to_csv(args.output_dir / "failures.csv", index=False)
    trades = pd.DataFrame(rows)
    if trades.empty or "entry_time" not in trades:
        raise RuntimeError("No Wolfe trades were produced")
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce")
    trades = trades[trades["entry_time"] >= analysis_start].copy()
    trades["rr_band"] = pd.cut(
        pd.to_numeric(trades["target_rr_planned"], errors="coerce"),
        bins=[-math.inf, 1.5, 2.0, math.inf],
        labels=["<=1.5R", "1.5-2.0R", ">2.0R"],
        right=True,
    ).astype(str)
    trades.to_csv(args.output_dir / "trades.csv", index=False)

    windows = sorted({days for days in [30, 60, 90, 180, args.days] if days <= args.days})
    summary = aggregate_windows(trades, end, windows)
    summary.to_csv(args.output_dir / "rolling_summary.csv", index=False)
    per_symbol = aggregate_groups(trades, ["strategy", "symbol"])
    per_symbol.to_csv(args.output_dir / "per_symbol.csv", index=False)
    rr_bands = aggregate_groups(trades, ["strategy", "rr_band"])
    rr_bands.to_csv(args.output_dir / "rr_bands.csv", index=False)
    regime_tables: list[pd.DataFrame] = []
    for timeframe in context_timeframes:
        suffix = timeframe.lower()
        table = aggregate_groups(
            trades,
            [
                "strategy",
                f"vol_regime_{suffix}",
                f"directional_regime_{suffix}",
            ],
        ).rename(
            columns={
                f"vol_regime_{suffix}": "vol_regime",
                f"directional_regime_{suffix}": "directional_regime",
            }
        )
        table.insert(1, "timeframe", timeframe)
        regime_tables.append(table)
    regimes = pd.concat(regime_tables, ignore_index=True)
    regimes.to_csv(args.output_dir / "regimes.csv", index=False)
    recommendations = rr_gate_recommendation(summary)
    recommendations.to_csv(args.output_dir / "rr_gate_recommendation.csv", index=False)
    hypothesis_windows = sorted(
        {days for days in [90, 365, args.days] if days <= args.days}
    )
    hypothesis_results = evaluate_hypotheses(
        trades,
        end=end,
        windows=hypothesis_windows,
        context_timeframes=context_timeframes,
    )
    hypothesis_results.to_csv(args.output_dir / "hypothesis_results.csv", index=False)
    write_report(
        args.output_dir / "report.md",
        summary=summary,
        rr_bands=rr_bands,
        regimes=regimes,
        recommendations=recommendations,
        hypothesis_results=hypothesis_results,
    )
    print(summary.to_string(index=False), flush=True)
    print(recommendations.to_string(index=False), flush=True)
    if not hypothesis_results.empty:
        top = hypothesis_results[
            (hypothesis_results["strategy"] == "combined")
            & (hypothesis_results["window_days"] == max(hypothesis_windows))
        ]
        top = (
            top.sort_values("robust_score", ascending=False)
            .groupby("hypothesis", as_index=False)
            .head(1)
        )
        print(
            top[
                [
                    "hypothesis",
                    "verdict",
                    "filter_name",
                    "timeframe",
                    "trades",
                    "retention",
                    "avg_r",
                    "delta_avg_r",
                    "robust_score",
                ]
            ].to_string(index=False),
            flush=True,
        )
    print(f"Saved Wolfe regime report: {args.output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
