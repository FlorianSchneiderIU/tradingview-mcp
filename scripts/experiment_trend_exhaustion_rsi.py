from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backtest_turtle_soup import (  # noqa: E402
    Config,
    DEFAULT_BFM_ZONE_TF_SETS,
    DEFAULT_BFM_ZONE_TIMEFRAMES,
    add_atr,
    parse_utc_datetime,
)
from scripts.bybit_demo_turtle_soup import bybit_symbol  # noqa: E402
from scripts.ml_trade_outcome_filter import (  # noqa: E402
    BASE_FEATURE_COLUMNS,
    add_engineered_rescue_features,
    classifier_metrics,
    fit_model,
    frame_metrics,
    threshold_table,
    trade_bfm_feature_columns_for_groups,
    trade_feature_rows,
)
from scripts.train_bybit_top_marketcap_turtle_models import (  # noqa: E402
    BYBIT_PUBLIC_URL,
    ensure_bybit_cache,
)


RSI_EXHAUST_FEATURE_COLUMNS = [
    "tex_rsi22_signal",
    "tex_rsi22_sweep",
    "tex_pressure_signal",
    "tex_pressure_sweep",
    "tex_aligned_strength_recent",
    "tex_aligned_age_bars",
    "tex_counter_strength_recent",
    "tex_counter_age_bars",
    "tex_aligned_signal_bar",
    "tex_counter_signal_bar",
    "tex_aligned_between_sweep_signal",
    "tex_counter_between_sweep_signal",
]


@dataclass
class SignalTrade:
    symbol: str
    variant: str
    direction: str
    signal_index: int
    entry_index: int
    exit_index: int
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    stop_price: float
    target_price: float
    r_multiple: float
    exit_reason: str
    strength: int
    rsi: float


def parse_symbols(raw: list[str]) -> list[str]:
    if raw:
        chunks = raw
    else:
        chunks = ["BNBUSDT,LINKUSDT,TAOUSDT,XMRUSDT,BTCUSDT,ETHUSDT"]
    out: list[str] = []
    for item in chunks:
        for chunk in str(item).split(","):
            symbol = chunk.strip().upper()
            if symbol and symbol not in out:
                out.append(symbol)
    return out


def rsi_wilder(close: pd.Series, length: int = 22) -> pd.Series:
    delta = close.astype(float).diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_up = up.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    avg_down = down.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    rs = avg_up / avg_down.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.mask((avg_down == 0.0) & (avg_up > 0.0), 100.0)
    rsi = rsi.mask((avg_up == 0.0) & (avg_down > 0.0), 0.0)
    return rsi


def compute_exhaustion_frame(df: pd.DataFrame, *, rsi_len: int = 22, valid_bars: int = 4) -> pd.DataFrame:
    out = df.sort_values("open_time").reset_index(drop=True).copy()
    out["rsi22"] = rsi_wilder(out["close"], rsi_len)
    rsi = out["rsi22"]
    prev = rsi.shift(1)

    long_strength = pd.Series(0, index=out.index, dtype=int)
    short_strength = pd.Series(0, index=out.index, dtype=int)
    for level, strength in [(30.0, 1), (20.0, 2), (15.0, 3)]:
        long_strength = long_strength.mask((prev <= level) & (rsi > level), np.maximum(long_strength, strength))
    for level, strength in [(70.0, 1), (80.0, 2), (85.0, 3)]:
        short_strength = short_strength.mask((prev >= level) & (rsi < level), np.maximum(short_strength, strength))

    out["tex_long_strength"] = long_strength.fillna(0).astype(int)
    out["tex_short_strength"] = short_strength.fillna(0).astype(int)

    for side in ["long", "short"]:
        strengths = out[f"tex_{side}_strength"].astype(int).to_numpy()
        recent_strength = np.zeros(len(out), dtype=float)
        recent_age = np.full(len(out), np.nan, dtype=float)
        last_strength = 0
        last_index: int | None = None
        for idx, strength in enumerate(strengths):
            if strength > 0:
                last_strength = int(strength)
                last_index = idx
            if last_index is not None:
                age = idx - last_index
                if age <= valid_bars:
                    recent_strength[idx] = float(last_strength)
                    recent_age[idx] = float(age)
        out[f"tex_{side}_recent_strength"] = recent_strength
        out[f"tex_{side}_recent_age"] = recent_age
    return out


def add_trade_rsi_exhaustion_features(
    dataset: pd.DataFrame,
    prepared: pd.DataFrame,
    *,
    valid_bars: int,
) -> pd.DataFrame:
    out = dataset.copy()
    if out.empty:
        for column in RSI_EXHAUST_FEATURE_COLUMNS:
            out[column] = []
        return out

    tex = compute_exhaustion_frame(prepared, valid_bars=valid_bars)
    close_to_idx = {pd.Timestamp(t): i for i, t in enumerate(pd.to_datetime(tex["close_time"], utc=True))}

    rows: list[dict[str, float]] = []
    for _, row in out.iterrows():
        direction = str(row["direction"])
        aligned_side = "long" if direction == "long" else "short"
        counter_side = "short" if direction == "long" else "long"
        signal_idx = close_to_idx.get(pd.Timestamp(row["signal_time"]))
        sweep_idx = close_to_idx.get(pd.Timestamp(row["sweep_time"]))
        if signal_idx is None:
            rows.append({column: math.nan for column in RSI_EXHAUST_FEATURE_COLUMNS})
            continue
        if sweep_idx is None:
            sweep_idx = signal_idx

        signal_rsi = float(tex.at[signal_idx, "rsi22"])
        sweep_rsi = float(tex.at[sweep_idx, "rsi22"])
        if direction == "long":
            signal_pressure = (50.0 - signal_rsi) / 50.0
            sweep_pressure = (50.0 - sweep_rsi) / 50.0
        else:
            signal_pressure = (signal_rsi - 50.0) / 50.0
            sweep_pressure = (sweep_rsi - 50.0) / 50.0

        lo = min(sweep_idx, signal_idx)
        hi = max(sweep_idx, signal_idx)
        aligned_between = float(tex.loc[lo:hi, f"tex_{aligned_side}_strength"].max())
        counter_between = float(tex.loc[lo:hi, f"tex_{counter_side}_strength"].max())
        rows.append(
            {
                "tex_rsi22_signal": signal_rsi / 100.0,
                "tex_rsi22_sweep": sweep_rsi / 100.0,
                "tex_pressure_signal": signal_pressure,
                "tex_pressure_sweep": sweep_pressure,
                "tex_aligned_strength_recent": float(tex.at[signal_idx, f"tex_{aligned_side}_recent_strength"]),
                "tex_aligned_age_bars": float(tex.at[signal_idx, f"tex_{aligned_side}_recent_age"]),
                "tex_counter_strength_recent": float(tex.at[signal_idx, f"tex_{counter_side}_recent_strength"]),
                "tex_counter_age_bars": float(tex.at[signal_idx, f"tex_{counter_side}_recent_age"]),
                "tex_aligned_signal_bar": float(tex.at[signal_idx, f"tex_{aligned_side}_strength"]),
                "tex_counter_signal_bar": float(tex.at[signal_idx, f"tex_{counter_side}_strength"]),
                "tex_aligned_between_sweep_signal": aligned_between,
                "tex_counter_between_sweep_signal": counter_between,
            }
        )
    feature_frame = pd.DataFrame(rows, index=out.index)
    for column in RSI_EXHAUST_FEATURE_COLUMNS:
        out[column] = feature_frame[column]
    return out


def outcome_from_entry(
    df: pd.DataFrame,
    *,
    symbol: str,
    variant: str,
    direction: str,
    signal_index: int,
    strength: int,
    rsi: float,
    rr: float,
    stop_buffer_atr: float,
    max_hold_bars: int,
) -> SignalTrade | None:
    entry_index = signal_index + 1
    if entry_index >= len(df):
        return None
    entry = float(df.at[entry_index, "open"])
    atr = float(df.at[signal_index, "atr"])
    if not math.isfinite(atr) or atr <= 0:
        return None
    if direction == "long":
        stop = float(df.at[signal_index, "low"]) - stop_buffer_atr * atr
        risk = entry - stop
        target = entry + rr * risk
    else:
        stop = float(df.at[signal_index, "high"]) + stop_buffer_atr * atr
        risk = stop - entry
        target = entry - rr * risk
    if risk <= 0 or risk / entry > 0.08:
        return None

    end = min(len(df) - 1, entry_index + max_hold_bars)
    exit_index = end
    exit_price = float(df.at[end, "close"])
    exit_reason = "time"
    for idx in range(entry_index, end + 1):
        high = float(df.at[idx, "high"])
        low = float(df.at[idx, "low"])
        if direction == "long":
            hit_stop = low <= stop
            hit_target = high >= target
            if hit_stop or hit_target:
                exit_index = idx
                if hit_stop and hit_target:
                    exit_price = stop
                    exit_reason = "stop_and_target_stop_first"
                elif hit_stop:
                    exit_price = stop
                    exit_reason = "stop"
                else:
                    exit_price = target
                    exit_reason = "target"
                break
        else:
            hit_stop = high >= stop
            hit_target = low <= target
            if hit_stop or hit_target:
                exit_index = idx
                if hit_stop and hit_target:
                    exit_price = stop
                    exit_reason = "stop_and_target_stop_first"
                elif hit_stop:
                    exit_price = stop
                    exit_reason = "stop"
                else:
                    exit_price = target
                    exit_reason = "target"
                break
    r_multiple = (exit_price - entry) / risk if direction == "long" else (entry - exit_price) / risk
    return SignalTrade(
        symbol=symbol,
        variant=variant,
        direction=direction,
        signal_index=signal_index,
        entry_index=entry_index,
        exit_index=exit_index,
        signal_time=pd.Timestamp(df.at[signal_index, "close_time"]),
        entry_time=pd.Timestamp(df.at[entry_index, "open_time"]),
        exit_time=pd.Timestamp(df.at[exit_index, "close_time"]),
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        r_multiple=float(r_multiple),
        exit_reason=exit_reason,
        strength=int(strength),
        rsi=float(rsi),
    )


def raw_signal_backtest(
    symbol: str,
    df: pd.DataFrame,
    *,
    split: pd.Timestamp,
    end: pd.Timestamp,
    min_strength: int,
    rr: float,
    valid_bars: int,
    max_hold_bars: int,
    stop_buffer_atr: float,
    min_bars_between: int,
) -> list[SignalTrade]:
    prepared = add_atr(df.sort_values("open_time").reset_index(drop=True).copy())
    tex = compute_exhaustion_frame(prepared, valid_bars=valid_bars)
    return raw_signal_backtest_from_tex(
        symbol,
        tex,
        split=split,
        end=end,
        min_strength=min_strength,
        rr=rr,
        max_hold_bars=max_hold_bars,
        stop_buffer_atr=stop_buffer_atr,
        min_bars_between=min_bars_between,
    )


def raw_signal_backtest_from_tex(
    symbol: str,
    tex: pd.DataFrame,
    *,
    split: pd.Timestamp,
    end: pd.Timestamp,
    min_strength: int,
    rr: float,
    max_hold_bars: int,
    stop_buffer_atr: float,
    min_bars_between: int,
) -> list[SignalTrade]:
    opens = tex["open"].astype(float).to_numpy()
    highs = tex["high"].astype(float).to_numpy()
    lows = tex["low"].astype(float).to_numpy()
    closes = tex["close"].astype(float).to_numpy()
    atrs = tex["atr"].astype(float).to_numpy()
    rsis = tex["rsi22"].astype(float).to_numpy()
    open_times = pd.to_datetime(tex["open_time"], utc=True).to_numpy()
    close_times = pd.to_datetime(tex["close_time"], utc=True).to_numpy()
    close_ts = pd.to_datetime(tex["close_time"], utc=True)
    split_ts = pd.Timestamp(split)
    end_ts = pd.Timestamp(end)
    in_window = (close_ts >= split_ts) & (close_ts < end_ts)
    events: list[tuple[int, str, int]] = []
    long_strengths = tex["tex_long_strength"].astype(int).to_numpy()
    short_strengths = tex["tex_short_strength"].astype(int).to_numpy()
    for idx in np.where(in_window.to_numpy() & (long_strengths >= min_strength))[0]:
        events.append((int(idx), "long", int(long_strengths[idx])))
    for idx in np.where(in_window.to_numpy() & (short_strengths >= min_strength))[0]:
        events.append((int(idx), "short", int(short_strengths[idx])))
    events.sort(key=lambda item: (item[0], 0 if item[1] == "long" else 1))

    trades: list[SignalTrade] = []
    next_allowed = 0
    variant = f"raw_te{min_strength}_rr{rr:g}_hold{max_hold_bars}"
    for idx, direction, strength in events:
        if idx < next_allowed:
            continue
        entry_index = idx + 1
        if entry_index >= len(tex):
            continue
        atr = float(atrs[idx])
        entry = float(opens[entry_index])
        if not math.isfinite(atr) or atr <= 0 or not math.isfinite(entry) or entry <= 0:
            continue
        if direction == "long":
            stop = float(lows[idx]) - stop_buffer_atr * atr
            risk = entry - stop
            target = entry + rr * risk
        else:
            stop = float(highs[idx]) + stop_buffer_atr * atr
            risk = stop - entry
            target = entry - rr * risk
        if risk <= 0 or risk / entry > 0.08:
            continue
        end_idx = min(len(tex) - 1, entry_index + max_hold_bars)
        exit_index = end_idx
        exit_price = float(closes[end_idx])
        exit_reason = "time"
        if direction == "long":
            stop_hits = np.where(lows[entry_index : end_idx + 1] <= stop)[0]
            target_hits = np.where(highs[entry_index : end_idx + 1] >= target)[0]
        else:
            stop_hits = np.where(highs[entry_index : end_idx + 1] >= stop)[0]
            target_hits = np.where(lows[entry_index : end_idx + 1] <= target)[0]
        first_stop = int(stop_hits[0]) if len(stop_hits) else None
        first_target = int(target_hits[0]) if len(target_hits) else None
        if first_stop is not None or first_target is not None:
            if first_target is None or (first_stop is not None and first_stop <= first_target):
                exit_index = entry_index + int(first_stop)
                exit_price = stop
                exit_reason = "stop_and_target_stop_first" if first_target is not None and first_stop == first_target else "stop"
            else:
                exit_index = entry_index + int(first_target)
                exit_price = target
                exit_reason = "target"
        r_multiple = (exit_price - entry) / risk if direction == "long" else (entry - exit_price) / risk
        trade = SignalTrade(
            symbol=symbol,
            variant=variant,
            direction=direction,
            signal_index=idx,
            entry_index=entry_index,
            exit_index=exit_index,
            signal_time=pd.Timestamp(close_times[idx]),
            entry_time=pd.Timestamp(open_times[entry_index]),
            exit_time=pd.Timestamp(close_times[exit_index]),
            entry_price=entry,
            stop_price=stop,
            target_price=target,
            r_multiple=float(r_multiple),
            exit_reason=exit_reason,
            strength=int(strength),
            rsi=float(rsis[idx]),
        )
        trades.append(trade)
        next_allowed = max(trade.exit_index + 1, idx + min_bars_between)
    return trades


def signal_trades_frame(trades: list[SignalTrade]) -> pd.DataFrame:
    return pd.DataFrame([trade.__dict__ for trade in trades])


def compare_models(
    dataset: pd.DataFrame,
    *,
    symbol: str,
    split: pd.Timestamp,
    feature_columns: list[str],
    augmented_columns: list[str],
    thresholds: list[float],
    min_oos_trades: int,
    model_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if dataset.empty:
        return rows
    data = dataset.copy()
    data["entry_time"] = pd.to_datetime(data["entry_time"], utc=True, errors="coerce")
    train = data[data["entry_time"] < split].copy()
    oos = data[data["entry_time"] >= split].copy()
    if len(train) < 80 or train["win_label"].nunique() < 2 or oos.empty:
        return rows
    for label, cols in [("baseline", feature_columns), ("rsi_exhaust_features", feature_columns + augmented_columns)]:
        cols = list(dict.fromkeys(cols))
        usable = [column for column in cols if column in train.columns and train[column].notna().any()]
        if not usable:
            continue
        model = fit_model(train, model_name, usable)
        scored = data.copy()
        scored["trade_win_prob"] = model.predict_proba(scored[usable].astype(float))[:, 1]
        oos_scored = scored[scored["entry_time"] >= split].copy()
        table = threshold_table(oos_scored, thresholds)
        table["passes_min_oos_trades"] = table["trades"].astype(int) >= min_oos_trades
        ranked = table[table["passes_min_oos_trades"]].copy()
        if ranked.empty:
            ranked = table.copy()
        best = ranked.sort_values(["profit_factor", "net_r", "trades"], ascending=[False, False, False]).iloc[0]
        rows.append(
            {
                "symbol": symbol,
                "model_variant": label,
                "feature_count": len(usable),
                "oos_auc": classifier_metrics(oos_scored)["auc"],
                "best_threshold": float(best["threshold"]),
                "best_trades": int(best["trades"]),
                "best_win_rate": float(best["win_rate"]),
                "best_pf": float(best["profit_factor"]),
                "best_net_r": float(best["net_r"]),
                "best_max_dd_r": float(best["max_dd_r"]),
            }
        )
    return rows


def gate_rows(dataset: pd.DataFrame, *, symbol: str, split: pd.Timestamp) -> list[dict[str, Any]]:
    if dataset.empty:
        return []
    oos = dataset[pd.to_datetime(dataset["entry_time"], utc=True, errors="coerce") >= split].copy()
    gates = {
        "all_turtle": pd.Series(True, index=oos.index),
        "tex_recent_any": oos["tex_aligned_strength_recent"].fillna(0) >= 1,
        "tex_recent_strong": oos["tex_aligned_strength_recent"].fillna(0) >= 2,
        "tex_between_any": oos["tex_aligned_between_sweep_signal"].fillna(0) >= 1,
        "tex_pressure_positive": oos["tex_pressure_signal"].fillna(-999) > 0.0,
        "tex_no_counter_recent": oos["tex_counter_strength_recent"].fillna(0) <= 0,
        "tex_recent_any_no_counter": (oos["tex_aligned_strength_recent"].fillna(0) >= 1)
        & (oos["tex_counter_strength_recent"].fillna(0) <= 0),
    }
    rows: list[dict[str, Any]] = []
    for gate, mask in gates.items():
        metrics = frame_metrics(oos[mask].copy())
        rows.append({"symbol": symbol, "gate": gate, **metrics})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment with Money Moves RSI trend-exhaustion signals/features.")
    parser.add_argument("--symbols", nargs="*", default=[])
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--warmup-start", default="2021-09-01")
    parser.add_argument("--train-start", default="2022-04-20")
    parser.add_argument("--split", default="2025-04-20")
    parser.add_argument("--end", default="2026-04-20")
    parser.add_argument("--cache-dir", type=Path, default=Path("scripts/.cache/bybit_linear"))
    parser.add_argument("--base-url", default=BYBIT_PUBLIC_URL)
    parser.add_argument("--model", choices=["rf", "logreg", "hgb"], default="rf")
    parser.add_argument("--thresholds", default="0.35,0.40,0.45,0.50,0.55,0.60")
    parser.add_argument("--min-oos-trades", type=int, default=10)
    parser.add_argument("--valid-bars", type=int, default=4)
    parser.add_argument("--no-bfm-features", action="store_true", help="Skip expensive BFM feature projection for the Turtle feature experiment.")
    parser.add_argument("--signal-only", action="store_true", help="Only run raw RSI exhaustion signal backtests.")
    parser.add_argument("--output-prefix", type=Path, default=Path("scripts/trend_exhaustion_rsi_sfp_v3"))
    args = parser.parse_args()

    symbols = parse_symbols(args.symbols)
    warmup_start = parse_utc_datetime(args.warmup_start)
    train_start = parse_utc_datetime(args.train_start)
    split = pd.Timestamp(parse_utc_datetime(args.split))
    end = pd.Timestamp(parse_utc_datetime(args.end))
    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]

    feature_columns = [column for column in BASE_FEATURE_COLUMNS if column not in RSI_EXHAUST_FEATURE_COLUMNS]
    if not args.no_bfm_features:
        feature_columns.extend(trade_bfm_feature_columns_for_groups("line,channel"))

    model_rows: list[dict[str, Any]] = []
    gate_summary_rows: list[dict[str, Any]] = []
    signal_summary_rows: list[dict[str, Any]] = []
    all_signal_trades: list[pd.DataFrame] = []
    all_datasets: list[pd.DataFrame] = []

    cfg = Config(
        exec_tf=args.interval,
        structure_tf="15m",
        entry_mode="zone_retest",
        tf1="1h",
        tf2="4h",
        use_tf1=True,
        use_tf2=False,
        block_dead_zone=False,
        max_structure_bars_to_choch=32,
        min_entry_risk_pct=0.0,
        max_zone_scan=0,
        use_sfp_liquidity_zones=True,
        sfp_timeframes="15m,1h,4h",
        sfp_left=15,
        sfp_right=10,
        sfp_level_width_atr=0.15,
        sfp_strict=True,
        sfp_require_open_reclaim=True,
    )

    for symbol in symbols:
        started = datetime.now(timezone.utc)
        print(f"{symbol}: loading data ...", flush=True)
        cache_path = ensure_bybit_cache(
            bybit_symbol(symbol),
            args.interval,
            warmup_start,
            end.to_pydatetime(),
            args.cache_dir,
            args.base_url,
        )
        df = pd.read_pickle(cache_path)
        dataset = pd.DataFrame()
        if not args.signal_only:
            print(
                f"{symbol}: building Turtle/SFP features"
                f"{' with BFM' if not args.no_bfm_features else ' without BFM'} ...",
                flush=True,
            )
            feature_frame, _ = trade_feature_rows(
                symbol,
                df,
                cfg,
                use_bfm_features=not args.no_bfm_features,
                bfm_timeframes=DEFAULT_BFM_ZONE_TIMEFRAMES,
                bfm_tf_sets=DEFAULT_BFM_ZONE_TF_SETS,
                bfm_invalidation="wick",
                bfm_max_extension_bars=300,
                use_sfp_liquidity_zones=True,
                sfp_timeframes="15m,1h,4h",
                sfp_left=15,
                sfp_right=10,
                sfp_level_width_atr=0.15,
                sfp_strict=True,
                sfp_require_open_reclaim=True,
            )
            dataset = add_engineered_rescue_features(feature_frame)
            dataset = add_trade_rsi_exhaustion_features(dataset, df, valid_bars=args.valid_bars)
            dataset["entry_time"] = pd.to_datetime(dataset["entry_time"], utc=True, errors="coerce")
            dataset = dataset[(dataset["entry_time"] >= pd.Timestamp(train_start)) & (dataset["entry_time"] < end)].copy()
            all_datasets.append(dataset.assign(symbol=symbol))

            model_rows.extend(
                compare_models(
                    dataset,
                    symbol=symbol,
                    split=split,
                    feature_columns=feature_columns,
                    augmented_columns=RSI_EXHAUST_FEATURE_COLUMNS,
                    thresholds=thresholds,
                    min_oos_trades=args.min_oos_trades,
                    model_name=args.model,
                )
            )
            gate_summary_rows.extend(gate_rows(dataset, symbol=symbol, split=split))

        signal_trades: list[SignalTrade] = []
        signal_tex = compute_exhaustion_frame(
            add_atr(df.sort_values("open_time").reset_index(drop=True).copy()),
            valid_bars=args.valid_bars,
        )
        for min_strength in [1, 2, 3]:
            for rr in [1.0, 1.5]:
                signal_trades.extend(
                    raw_signal_backtest_from_tex(
                        symbol,
                        signal_tex,
                        split=split,
                        end=end,
                        min_strength=min_strength,
                        rr=rr,
                        max_hold_bars=48,
                        stop_buffer_atr=0.05,
                        min_bars_between=6,
                    )
                )
        signal_frame = signal_trades_frame(signal_trades)
        if not signal_frame.empty:
            all_signal_trades.append(signal_frame)
            for variant, group in signal_frame.groupby("variant"):
                metrics = frame_metrics(group.rename(columns={"r_multiple": "r_multiple"}))
                signal_summary_rows.append({"symbol": symbol, "variant": variant, **metrics})
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        print(f"{symbol}: dataset={len(dataset)} model_rows={len(model_rows)} raw_signal_trades={len(signal_trades)} elapsed={elapsed:.0f}s", flush=True)

    model_summary = pd.DataFrame(model_rows)
    gate_summary = pd.DataFrame(gate_summary_rows)
    signal_summary = pd.DataFrame(signal_summary_rows)
    dataset_all = pd.concat(all_datasets, ignore_index=True) if all_datasets else pd.DataFrame()
    signals_all = pd.concat(all_signal_trades, ignore_index=True) if all_signal_trades else pd.DataFrame()

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    model_path = args.output_prefix.with_name(f"{args.output_prefix.name}_model_compare.csv")
    gate_path = args.output_prefix.with_name(f"{args.output_prefix.name}_turtle_gates.csv")
    signal_path = args.output_prefix.with_name(f"{args.output_prefix.name}_raw_signal_summary.csv")
    dataset_path = args.output_prefix.with_name(f"{args.output_prefix.name}_dataset.csv")
    trades_path = args.output_prefix.with_name(f"{args.output_prefix.name}_raw_signal_trades.csv")
    model_summary.to_csv(model_path, index=False)
    gate_summary.to_csv(gate_path, index=False)
    signal_summary.to_csv(signal_path, index=False)
    dataset_all.to_csv(dataset_path, index=False)
    signals_all.to_csv(trades_path, index=False)

    print("\nModel comparison")
    if not model_summary.empty:
        print(model_summary.sort_values(["symbol", "model_variant"]).to_string(index=False))
    print("\nTurtle RSI gates")
    if not gate_summary.empty:
        agg = gate_summary.groupby("gate", as_index=False).agg(
            trades=("trades", "sum"),
            net_r=("net_r", "sum"),
            avg_pf=("profit_factor", "mean"),
            avg_wr=("win_rate", "mean"),
        )
        print(agg.sort_values("net_r", ascending=False).to_string(index=False))
    print("\nRaw RSI exhaustion signals")
    if not signal_summary.empty:
        agg_sig = signal_summary.groupby("variant", as_index=False).agg(
            trades=("trades", "sum"),
            net_r=("net_r", "sum"),
            avg_pf=("profit_factor", "mean"),
            avg_wr=("win_rate", "mean"),
        )
        print(agg_sig.sort_values("net_r", ascending=False).to_string(index=False))
    print(f"\nWrote {model_path}")
    print(f"Wrote {gate_path}")
    print(f"Wrote {signal_path}")
    print(f"Wrote {dataset_path}")
    print(f"Wrote {trades_path}")


if __name__ == "__main__":
    main()
