from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict
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
    DEFAULT_UNIVERSE,
    SIGNAL_BUILDERS,
    CandidateSpec,
    base_context,
    build_specs,
    clean_symbol,
    load_frame,
    load_universe,
    parse_list,
    resample_frame,
)


DEFAULT_OUT_PREFIX = Path("scripts/pine_atom_per_coin_full_sweep")


def unique_specs(specs: list[CandidateSpec]) -> list[CandidateSpec]:
    out: dict[str, CandidateSpec] = {}
    for spec in specs:
        out.setdefault(spec.name, spec)
    return list(out.values())


def base_key(spec: CandidateSpec) -> tuple[Any, ...]:
    params = tuple(sorted((k, v) for k, v in spec.params.items() if k not in {"rr", "max_hold_bars"}))
    return (spec.strategy, spec.timeframe, params)


def metrics_from_arrays(r_values: list[float], entry_times: list[pd.Timestamp]) -> dict[str, float]:
    if not r_values:
        return {
            "trades": 0,
            "net_r": 0.0,
            "avg_r": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_dd_r": 0.0,
            "trades_per_week": 0.0,
        }
    r = np.asarray(r_values, dtype=float)
    wins = r[r > 0]
    losses = r[r < 0]
    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())
    curve = np.cumsum(r)
    dd = np.maximum.accumulate(curve) - curve
    times = pd.to_datetime(pd.Series(entry_times), utc=True, errors="coerce")
    if times.notna().sum() >= 2:
        weeks = max((times.max() - times.min()).total_seconds() / (86400.0 * 7.0), 1e-9)
    else:
        weeks = 1e-9
    return {
        "trades": int(len(r)),
        "net_r": float(r.sum()),
        "avg_r": float(r.mean()),
        "win_rate": float((r > 0).mean()),
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0),
        "max_dd_r": float(dd.max()) if len(dd) else 0.0,
        "trades_per_week": float(len(r) / weeks),
    }


def prefixed(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{k}": v for k, v in metrics.items()}


def score_metrics(m: dict[str, float]) -> float:
    trades = float(m.get("trades", 0) or 0)
    if trades <= 0:
        return -1e9
    pf = min(float(m.get("profit_factor", 0) or 0), 3.0)
    avg_r = float(m.get("avg_r", 0) or 0)
    dd = float(m.get("max_dd_r", 0) or 0)
    return avg_r * math.sqrt(min(trades, 500.0)) + 0.12 * pf - 0.004 * dd


def simulate_r_values(
    frame: pd.DataFrame,
    signals: list[dict[str, Any]],
    *,
    rr: float,
    max_hold_bars: int,
    fee_bps_per_side: float,
    min_risk_pct: float,
) -> tuple[list[float], list[pd.Timestamp]]:
    if not signals:
        return [], []
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    open_ = frame["open"].to_numpy(dtype=float)
    close = frame["close"].to_numpy(dtype=float)
    open_time = frame["open_time"].tolist()
    n = len(frame)
    r_values: list[float] = []
    entry_times: list[pd.Timestamp] = []
    blocked_until = -1
    for signal in signals:
        signal_idx = int(signal["idx"])
        entry_idx = signal_idx + 1
        if entry_idx >= n or entry_idx <= blocked_until:
            continue
        direction = int(signal["direction"])
        entry = float(open_[entry_idx])
        stop = float(signal["stop"])
        if not (math.isfinite(entry) and math.isfinite(stop) and entry > 0):
            continue
        if direction > 0 and stop >= entry:
            continue
        if direction < 0 and stop <= entry:
            continue
        risk = abs(entry - stop)
        risk_pct = risk / entry * 100.0
        if risk <= 0 or risk_pct < min_risk_pct:
            continue
        target = entry + direction * risk * rr
        last = min(n - 1, entry_idx + max_hold_bars)
        exit_idx = last
        exit_price = float(close[last])
        if direction > 0:
            sl_hits = np.flatnonzero(low[entry_idx : last + 1] <= stop)
            tp_hits = np.flatnonzero(high[entry_idx : last + 1] >= target)
        else:
            sl_hits = np.flatnonzero(high[entry_idx : last + 1] >= stop)
            tp_hits = np.flatnonzero(low[entry_idx : last + 1] <= target)
        first_sl = int(sl_hits[0]) if sl_hits.size else None
        first_tp = int(tp_hits[0]) if tp_hits.size else None
        if first_sl is not None or first_tp is not None:
            if first_sl is not None and (first_tp is None or first_sl <= first_tp):
                exit_idx = entry_idx + first_sl
                exit_price = stop
            else:
                exit_idx = entry_idx + int(first_tp)
                exit_price = target
        gross_r = direction * (exit_price - entry) / risk
        fee_r = (2.0 * fee_bps_per_side / 10000.0) * entry / risk
        r_values.append(float(gross_r - fee_r))
        entry_times.append(pd.Timestamp(open_time[entry_idx]))
        blocked_until = exit_idx
    return r_values, entry_times


def evaluate_symbol_atom(
    symbol: str,
    atom: str,
    specs: list[CandidateSpec],
    args_dict: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    train_start = pd.Timestamp(parse_utc_datetime(args_dict["train_start"]))
    split = pd.Timestamp(parse_utc_datetime(args_dict["split"]))
    end = pd.Timestamp(parse_utc_datetime(args_dict["end"]))
    try:
        base = load_frame(symbol, Path(args_dict["cache_dir"]), train_start, end)
        frames: dict[str, pd.DataFrame] = {}
        contexts: dict[str, dict[str, np.ndarray]] = {}
        for tf in sorted({spec.timeframe for spec in specs}):
            frame = resample_frame(base, tf)
            frame = frame[frame["open_time"] >= train_start - pd.Timedelta(days=10)].reset_index(drop=True)
            frames[tf] = frame
            contexts[tf] = base_context(frame)

        grouped: dict[tuple[Any, ...], list[CandidateSpec]] = defaultdict(list)
        for spec in specs:
            grouped[base_key(spec)].append(spec)

        rows: list[dict[str, Any]] = []
        for group_specs in grouped.values():
            base_spec = group_specs[0]
            frame = frames[base_spec.timeframe]
            ctx = contexts[base_spec.timeframe]
            builder = SIGNAL_BUILDERS[base_spec.strategy]
            signals = builder(frame, ctx, base_spec)
            for spec in group_specs:
                r_values, entry_times = simulate_r_values(
                    frame,
                    signals,
                    rr=float(spec.params["rr"]),
                    max_hold_bars=int(spec.params["max_hold_bars"]),
                    fee_bps_per_side=float(args_dict["fee_bps_per_side"]),
                    min_risk_pct=float(args_dict["min_risk_pct"]),
                )
                if entry_times:
                    mask_train = [t < split for t in entry_times]
                    train_r = [r for r, is_train in zip(r_values, mask_train) if is_train]
                    train_t = [t for t, is_train in zip(entry_times, mask_train) if is_train]
                    oos_r = [r for r, is_train in zip(r_values, mask_train) if not is_train]
                    oos_t = [t for t, is_train in zip(entry_times, mask_train) if not is_train]
                else:
                    train_r = train_t = oos_r = oos_t = []
                train_m = metrics_from_arrays(train_r, train_t)
                oos_m = metrics_from_arrays(oos_r, oos_t)
                all_m = metrics_from_arrays(r_values, entry_times)
                row = {
                    "symbol": symbol,
                    "atom": atom,
                    "strategy": spec.strategy,
                    "timeframe": spec.timeframe,
                    "spec_name": spec.name,
                    "params_json": json.dumps(spec.params, sort_keys=True),
                    **prefixed(train_m, "train"),
                    **prefixed(oos_m, "oos"),
                    **prefixed(all_m, "all"),
                }
                row["train_score"] = score_metrics(train_m)
                row["oos_score"] = score_metrics(oos_m)
                row["train_eligible"] = bool(train_m["trades"] >= int(args_dict["min_train_trades"]))
                row["oos_eligible"] = bool(oos_m["trades"] >= int(args_dict["min_oos_trades"]))
                rows.append(row)
        return rows, None
    except Exception as exc:
        return [], {"symbol": symbol, "atom": atom, "error": f"{type(exc).__name__}: {exc}"}


def select_rows(summary: pd.DataFrame, min_train_trades: int, min_oos_trades: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_rows: list[pd.Series] = []
    oracle_rows: list[pd.Series] = []
    if summary.empty:
        return pd.DataFrame(), pd.DataFrame()
    for (_symbol, _atom), group in summary.groupby(["symbol", "atom"]):
        train_candidates = group[group["train_trades"] >= min_train_trades].copy()
        if train_candidates.empty:
            train_candidates = group.copy()
        selected_rows.append(train_candidates.sort_values(["train_score", "train_net_r"], ascending=[False, False]).iloc[0])
        oos_candidates = group[group["oos_trades"] >= min_oos_trades].copy()
        if oos_candidates.empty:
            oos_candidates = group.copy()
        oracle_rows.append(oos_candidates.sort_values(["oos_score", "oos_net_r"], ascending=[False, False]).iloc[0])
    selected = pd.DataFrame(selected_rows).sort_values(["oos_score", "oos_net_r"], ascending=[False, False])
    oracle = pd.DataFrame(oracle_rows).sort_values(["oos_score", "oos_net_r"], ascending=[False, False])
    return selected, oracle


def write_report(
    path: Path,
    summary: pd.DataFrame,
    selected: pd.DataFrame,
    oracle: pd.DataFrame,
    failures: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    def table(frame: pd.DataFrame, cols: list[str], n: int = 30) -> str:
        if frame.empty:
            return "_No rows._"
        shown = frame.head(n)[cols].copy()
        for col in shown.columns:
            if pd.api.types.is_float_dtype(shown[col]):
                shown[col] = shown[col].map(lambda x: f"{float(x):.4f}" if pd.notna(x) else "")
            else:
                shown[col] = shown[col].map(lambda x: "" if pd.isna(x) else str(x))
        return "\n".join(
            [
                "| " + " | ".join(cols) + " |",
                "| " + " | ".join(["---"] * len(cols)) + " |",
                *["| " + " | ".join(str(row[c]) for c in cols) + " |" for _, row in shown.iterrows()],
            ]
        )

    lines: list[str] = []
    lines.append("# Pine Atom Per-Coin Sweep")
    lines.append("")
    lines.append(f"Grid mode: `{args.grid_mode}` | timeframes: `{args.timeframes}` | atoms: `{args.atoms}`")
    lines.append(f"Window: `{args.train_start}` to `{args.end}` | split: `{args.split}`")
    lines.append("")
    lines.append("Parameter selection uses train score only. OOS is then used to rank coins/atoms. `oracle` tables are diagnostic and should not be used for live selection.")
    lines.append("")
    lines.append("## Selected By Train, Ranked By OOS")
    lines.append("")
    cols = ["symbol", "atom", "timeframe", "oos_trades", "oos_net_r", "oos_avg_r", "oos_win_rate", "oos_profit_factor", "spec_name"]
    lines.append(table(selected, cols, 40))
    lines.append("")
    lines.append("## Best OOS Oracle Diagnostic")
    lines.append("")
    lines.append(table(oracle, cols, 40))
    lines.append("")
    lines.append("## Atom-Level Selected OOS Summary")
    lines.append("")
    agg_rows: list[dict[str, Any]] = []
    if not selected.empty and "atom" in selected.columns:
        groups = selected.groupby("atom")
    else:
        groups = []
    for atom, group in groups:
        agg_rows.append(
            {
                "atom": atom,
                "coin_atoms": int(len(group)),
                "positive_coin_atoms": int((group["oos_net_r"] > 0).sum()),
                "oos_net_r": float(group["oos_net_r"].sum()),
                "median_oos_avg_r": float(group["oos_avg_r"].median()),
                "median_oos_pf": float(group["oos_profit_factor"].median()),
            }
        )
    agg = pd.DataFrame(agg_rows).sort_values("oos_net_r", ascending=False) if agg_rows else pd.DataFrame()
    lines.append(table(agg, ["atom", "coin_atoms", "positive_coin_atoms", "oos_net_r", "median_oos_avg_r", "median_oos_pf"], 20))
    if not failures.empty:
        lines.append("")
        lines.append("## Failures")
        lines.append("")
        lines.append(table(failures, ["symbol", "atom", "error"], 80))
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full per-coin parameter sweep for ported Pine atoms.")
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--max-symbols", type=int, default=50)
    parser.add_argument("--symbols", default="")
    parser.add_argument("--atoms", default="pivot_breakout,melona_trendline,ha_supertrend,liquidity_sweep,melona_pressure,demarker_exhaustion")
    parser.add_argument("--timeframes", default="15m")
    parser.add_argument("--grid-mode", choices=["smoke", "fast", "full"], default="full")
    parser.add_argument("--train-start", default="2024-01-01")
    parser.add_argument("--split", default="2025-07-01")
    parser.add_argument("--end", default="2026-04-20")
    parser.add_argument("--fee-bps-per-side", type=float, default=6.5)
    parser.add_argument("--min-risk-pct", type=float, default=0.15)
    parser.add_argument("--min-train-trades", type=int, default=40)
    parser.add_argument("--min-oos-trades", type=int, default=20)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.symbols.strip():
        symbols = [clean_symbol(x) for x in parse_list(args.symbols)]
    else:
        symbols = load_universe(args.universe, args.max_symbols)
    atoms = parse_list(args.atoms)
    specs_by_atom: dict[str, list[CandidateSpec]] = {}
    for atom in atoms:
        spec_args = argparse.Namespace(grid_mode=args.grid_mode, timeframes=args.timeframes, strategies=atom)
        specs_by_atom[atom] = unique_specs(build_specs(spec_args))
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Sweeping {len(symbols)} symbols x {len(atoms)} atoms "
        f"({sum(len(v) for v in specs_by_atom.values())} specs per symbol total)",
        flush=True,
    )
    for atom, specs in specs_by_atom.items():
        print(f"  {atom}: {len(specs)} specs", flush=True)

    jobs: list[tuple[str, str, list[CandidateSpec], dict[str, Any]]] = []
    args_dict = {
        **vars(args),
        "cache_dir": str(args.cache_dir),
        "universe": str(args.universe),
        "out_prefix": str(args.out_prefix),
    }
    for symbol in symbols:
        for atom in atoms:
            jobs.append((symbol, atom, specs_by_atom[atom], args_dict))

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if args.workers <= 1:
        iterator = ((job, evaluate_symbol_atom(*job)) for job in jobs)
        for i, (job, (job_rows, failure)) in enumerate(iterator, start=1):
            symbol, atom = job[0], job[1]
            if failure:
                failures.append(failure)
                print(f"[{i}/{len(jobs)}] {symbol} {atom}: failed {failure['error']}", flush=True)
            else:
                rows.extend(job_rows)
                print(f"[{i}/{len(jobs)}] {symbol} {atom}: {len(job_rows)} spec rows", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_map = {executor.submit(evaluate_symbol_atom, *job): job for job in jobs}
            for i, future in enumerate(concurrent.futures.as_completed(future_map), start=1):
                symbol, atom = future_map[future][0], future_map[future][1]
                job_rows, failure = future.result()
                if failure:
                    failures.append(failure)
                    print(f"[{i}/{len(jobs)}] {symbol} {atom}: failed {failure['error']}", flush=True)
                else:
                    rows.extend(job_rows)
                    print(f"[{i}/{len(jobs)}] {symbol} {atom}: {len(job_rows)} spec rows", flush=True)

    summary = pd.DataFrame(rows)
    failures_frame = pd.DataFrame(failures)
    selected, oracle = select_rows(summary, args.min_train_trades, args.min_oos_trades)

    summary_path = args.out_prefix.with_name(f"{args.out_prefix.name}_summary.csv")
    selected_path = args.out_prefix.with_name(f"{args.out_prefix.name}_selected_by_train.csv")
    oracle_path = args.out_prefix.with_name(f"{args.out_prefix.name}_best_oos_oracle.csv")
    failures_path = args.out_prefix.with_name(f"{args.out_prefix.name}_failures.csv")
    report_path = args.out_prefix.with_suffix(".md")
    summary.to_csv(summary_path, index=False)
    selected.to_csv(selected_path, index=False)
    oracle.to_csv(oracle_path, index=False)
    failures_frame.to_csv(failures_path, index=False)
    write_report(report_path, summary, selected, oracle, failures_frame, args)

    print(f"Saved summary: {summary_path}", flush=True)
    print(f"Saved selected: {selected_path}", flush=True)
    print(f"Saved oracle: {oracle_path}", flush=True)
    print(f"Saved report: {report_path}", flush=True)
    if not selected.empty:
        cols = ["symbol", "atom", "timeframe", "oos_trades", "oos_net_r", "oos_avg_r", "oos_profit_factor", "spec_name"]
        print(selected[cols].head(30).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
