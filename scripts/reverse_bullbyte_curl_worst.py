from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backtest_turtle_soup import parse_utc_datetime  # noqa: E402
from scripts.experiment_pine_strategy_candidates import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    clean_symbol,
    load_frame,
    resample_frame,
)
from scripts.sweep_bullbyte_curl import (  # noqa: E402
    Candidate,
    ExitSpec,
    FullSpec,
    PreparedFrame,
    SignalSpec,
    calc_levels,
    exit_spec_from_dict,
    generate_candidates,
    metrics_from_values,
    signal_spec_from_dict,
    simulate_candidates,
)


DEFAULT_SUMMARY = Path("scripts/bullbyte_curl_postmortem_btc_eth_allconfigs_summary.csv")
DEFAULT_OUT_PREFIX = Path("scripts/bullbyte_curl_reverse_worst")


def metrics_for_trades(trades: list[dict[str, Any]]) -> dict[str, float]:
    return metrics_from_values(
        [float(trade["r_multiple"]) for trade in trades],
        [pd.Timestamp(trade["entry_time"]) for trade in trades],
    )


def split_metrics(trades: list[dict[str, Any]], split: pd.Timestamp) -> dict[str, float]:
    train = [trade for trade in trades if pd.Timestamp(trade["entry_time"]) < split]
    oos = [trade for trade in trades if pd.Timestamp(trade["entry_time"]) >= split]
    return {
        **{f"train_{key}": value for key, value in metrics_for_trades(train).items()},
        **{f"oos_{key}": value for key, value in metrics_for_trades(oos).items()},
        **{f"all_{key}": value for key, value in metrics_for_trades(trades).items()},
    }


def simulate_reversed_candidates(
    symbol: str,
    prep: PreparedFrame,
    signal_spec: SignalSpec,
    exit_spec: ExitSpec,
    candidates: list[Candidate],
    *,
    fee_bps_per_side: float,
    min_risk_pct: float,
) -> list[dict[str, Any]]:
    high = prep.high
    low = prep.low
    close = prep.close
    atr = prep.signal_atr[exit_spec.atr_period]
    n = prep.n
    spec_name = f"{signal_spec.name}__{exit_spec.name}"
    out: list[dict[str, Any]] = []
    blocked_until = -1
    for candidate in candidates:
        i = candidate.signal_index
        if i >= n - 1 or i <= blocked_until + exit_spec.post_outcome_gap:
            continue
        atr_value = float(atr[i])
        if not math.isfinite(atr_value) or atr_value <= 0:
            continue

        original_stop, original_target, original_risk = calc_levels(candidate, atr_value, exit_spec)
        entry = candidate.entry_price
        if original_risk <= 0 or entry <= 0:
            continue
        original_risk_pct = original_risk / entry * 100.0
        if original_risk_pct < min_risk_pct:
            continue

        reverse_direction = -candidate.direction
        reverse_stop = original_target
        reverse_target = original_stop
        reverse_risk = abs(reverse_stop - entry)
        if reverse_risk <= 0:
            continue

        start = i + 1
        exit_idx = n - 1
        exit_price = float(close[-1])
        exit_reason = "open"
        if reverse_direction > 0:
            sl_hits = np.flatnonzero(low[start:n] <= reverse_stop)
            tp_hits = np.flatnonzero(high[start:n] >= reverse_target)
        else:
            sl_hits = np.flatnonzero(high[start:n] >= reverse_stop)
            tp_hits = np.flatnonzero(low[start:n] <= reverse_target)
        first_sl = int(sl_hits[0]) if sl_hits.size else None
        first_tp = int(tp_hits[0]) if tp_hits.size else None
        if first_sl is not None or first_tp is not None:
            if first_sl is not None and (first_tp is None or first_sl <= first_tp):
                exit_idx = start + first_sl
                exit_price = reverse_stop
                exit_reason = "sl"
            else:
                exit_idx = start + int(first_tp)
                exit_price = reverse_target
                exit_reason = "tp_swapped"

        gross_r = reverse_direction * (exit_price - entry) / reverse_risk
        fee_r = (2.0 * fee_bps_per_side / 10000.0) * entry / reverse_risk
        out.append(
            {
                "symbol": symbol,
                "strategy": "bullbyte_curl_reverse",
                "spec_name": spec_name,
                "signal_spec_name": signal_spec.name,
                "exit_spec_name": exit_spec.name,
                "timeframe": signal_spec.timeframe,
                "direction": "long" if reverse_direction > 0 else "short",
                "original_direction": "long" if candidate.direction > 0 else "short",
                "signal_index": i,
                "entry_index": i,
                "exit_index": exit_idx,
                "signal_time": pd.Timestamp(prep.close_time.iloc[i]).isoformat(),
                "entry_time": pd.Timestamp(prep.close_time.iloc[i]).isoformat(),
                "exit_time": pd.Timestamp(prep.close_time.iloc[exit_idx]).isoformat(),
                "entry_price": float(entry),
                "stop_price": float(reverse_stop),
                "target_price": float(reverse_target),
                "original_stop_price": float(original_stop),
                "original_target_price": float(original_target),
                "exit_price": float(exit_price),
                "exit_reason": exit_reason,
                "r_multiple": float(gross_r - fee_r),
                "gross_r": float(gross_r),
                "fee_r": float(fee_r),
                "risk_pct": float(reverse_risk / entry * 100.0),
                "original_risk_pct": float(original_risk_pct),
                "reward_r": float(original_risk / reverse_risk),
                "bars_held": int(exit_idx - i),
            }
        )
        blocked_until = exit_idx
    return out


def load_prepared(
    symbol: str,
    signal_spec: SignalSpec,
    exit_spec: ExitSpec,
    *,
    cache_dir: Path,
    train_start: pd.Timestamp,
    end: pd.Timestamp,
) -> PreparedFrame:
    base = load_frame(symbol, cache_dir, train_start, end)
    frame = resample_frame(base, signal_spec.timeframe)
    frame = frame[frame["open_time"] >= train_start - pd.Timedelta(days=30)].reset_index(drop=True)
    return PreparedFrame(
        frame,
        local_lengths={max(signal_spec.comp_min_bars, 3)},
        bg_lengths={signal_spec.bg_atr_period},
        atr_lengths={exit_spec.atr_period},
        session_lookbacks={signal_spec.session_lookback},
    )


def row_to_specs(row: pd.Series) -> tuple[SignalSpec, ExitSpec, FullSpec]:
    params = json.loads(str(row["params_json"]))
    signal_spec = signal_spec_from_dict(params["signal"])
    exit_spec = exit_spec_from_dict(params["exit"])
    return signal_spec, exit_spec, FullSpec(signal_spec, exit_spec)


def select_worst_rows(summary: pd.DataFrame, symbols: list[str], worst_by: str, per_symbol: bool) -> pd.DataFrame:
    rows = summary.copy()
    if symbols:
        rows = rows[rows["symbol"].astype(str).isin(symbols)].copy()
    metric = f"{worst_by}_net_r"
    rows[metric] = pd.to_numeric(rows[metric], errors="coerce")
    rows = rows.dropna(subset=[metric])
    if per_symbol:
        return rows.sort_values(metric).groupby("symbol", as_index=False).head(1).reset_index(drop=True)
    return rows.sort_values(metric).head(1).reset_index(drop=True)


def analyze_row(
    row: pd.Series,
    *,
    cache_dir: Path,
    train_start: pd.Timestamp,
    split: pd.Timestamp,
    end: pd.Timestamp,
    fee_bps_per_side: float,
    min_risk_pct: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    symbol = clean_symbol(str(row["symbol"]))
    signal_spec, exit_spec, full_spec = row_to_specs(row)
    prep = load_prepared(symbol, signal_spec, exit_spec, cache_dir=cache_dir, train_start=train_start, end=end)
    candidates = generate_candidates(symbol, prep, signal_spec)
    original = [trade.to_dict() for trade in simulate_candidates(
        symbol,
        prep,
        signal_spec,
        exit_spec,
        candidates,
        fee_bps_per_side=fee_bps_per_side,
        min_risk_pct=min_risk_pct,
    )]
    reversed_trades = simulate_reversed_candidates(
        symbol,
        prep,
        signal_spec,
        exit_spec,
        candidates,
        fee_bps_per_side=fee_bps_per_side,
        min_risk_pct=min_risk_pct,
    )
    original_metrics = split_metrics(original, split)
    reversed_metrics = split_metrics(reversed_trades, split)
    result = {
        "symbol": symbol,
        "spec_name": full_spec.name,
        "selected_train_net_r": float(row["train_net_r"]),
        "selected_oos_net_r": float(row["oos_net_r"]),
        "selected_all_net_r": float(row["all_net_r"]),
        **{f"original_{key}": value for key, value in original_metrics.items()},
        **{f"reversed_{key}": value for key, value in reversed_metrics.items()},
        "reversed_train_delta": reversed_metrics["train_net_r"] - original_metrics["train_net_r"],
        "reversed_oos_delta": reversed_metrics["oos_net_r"] - original_metrics["oos_net_r"],
        "reversed_all_delta": reversed_metrics["all_net_r"] - original_metrics["all_net_r"],
        "reverse_reward_r": 1.0 / float(exit_spec.tp3_r),
        "params_json": json.dumps(full_spec.params, sort_keys=True),
    }
    return result, reversed_trades


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    shown = frame[columns].copy()
    for column in shown.columns:
        if pd.api.types.is_numeric_dtype(shown[column]):
            shown[column] = shown[column].map(
                lambda value: f"{float(value):.4f}" if pd.notna(value) and math.isfinite(float(value)) else ""
            )
        else:
            shown[column] = shown[column].map(lambda value: "" if pd.isna(value) else str(value))
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(str(row[column]) for column in shown.columns) + " |" for _, row in shown.iterrows()]
    return "\n".join([header, sep, *body])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reverse-test worst BullByte Curl configs by swapping TP/SL prices.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    parser.add_argument("--worst-by", choices=["train", "oos", "all"], default="train")
    parser.add_argument("--per-symbol", action="store_true")
    parser.add_argument("--train-start", default="2024-01-01")
    parser.add_argument("--split", default="2025-07-01")
    parser.add_argument("--end", default="2026-04-20")
    parser.add_argument("--fee-bps-per-side", type=float, default=6.5)
    parser.add_argument("--min-risk-pct", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols = [clean_symbol(x) for x in str(args.symbols).split(",") if x.strip()]
    train_start = pd.Timestamp(parse_utc_datetime(args.train_start))
    split = pd.Timestamp(parse_utc_datetime(args.split))
    end = pd.Timestamp(parse_utc_datetime(args.end))
    summary = pd.read_csv(args.summary)
    selected = select_worst_rows(summary, symbols, args.worst_by, args.per_symbol)
    rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        result, reversed_trades = analyze_row(
            row,
            cache_dir=args.cache_dir,
            train_start=train_start,
            split=split,
            end=end,
            fee_bps_per_side=args.fee_bps_per_side,
            min_risk_pct=args.min_risk_pct,
        )
        rows.append(result)
        trades.extend(reversed_trades)

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    result_frame = pd.DataFrame(rows)
    trades_frame = pd.DataFrame(trades)
    result_path = args.out_prefix.with_suffix(".csv")
    trades_path = args.out_prefix.with_name(f"{args.out_prefix.name}_trades.csv")
    report_path = args.out_prefix.with_suffix(".md")
    result_frame.to_csv(result_path, index=False)
    trades_frame.to_csv(trades_path, index=False)

    cols = [
        "symbol",
        "original_train_trades",
        "original_train_net_r",
        "reversed_train_trades",
        "reversed_train_net_r",
        "original_oos_trades",
        "original_oos_net_r",
        "reversed_oos_trades",
        "reversed_oos_net_r",
        "original_all_net_r",
        "reversed_all_net_r",
        "reverse_reward_r",
        "spec_name",
    ]
    lines = [
        "# BullByte Curl Reverse Worst Config",
        "",
        f"Worst selection: `{args.worst_by}_net_r` | per-symbol: `{bool(args.per_symbol)}`",
        f"Window: `{args.train_start}` to `{args.end}` | split: `{args.split}`",
        "",
        "Reverse simulation flips the signal direction and swaps price levels: original TP becomes the reverse SL, "
        "and original SL becomes the reverse TP. R is normalized to the new reverse stop distance.",
        "",
        markdown_table(result_frame, cols),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {result_path}")
    print(f"Wrote {trades_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
