from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research_btc_astro_cycle_timing import load_bybit_cached  # noqa: E402
from scripts.research_btc_hierarchical_path_walkforward import (  # noqa: E402
    FEATURE_SETS,
    add_path_features,
    cascade_features,
    directional_price_features,
    run_dynamic_walkforward,
    run_walkforward,
    time_features,
)
from scripts.research_btc_hierarchical_reversal import (  # noqa: E402
    add_indicators,
    json_default,
    parse_float_list,
    parse_str_list,
    parse_utc_datetime,
)
from scripts.research_btc_ltf_calendar_probability import DEFAULT_CACHE_DIR  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def prepare_1m_features(frame: pd.DataFrame, range_bars: int) -> pd.DataFrame:
    out = add_path_features(add_indicators(frame))
    out["wyckoff_range_high"] = out["high"].shift(1).rolling(range_bars, min_periods=range_bars).max()
    out["wyckoff_range_low"] = out["low"].shift(1).rolling(range_bars, min_periods=range_bars).min()
    out["volume_sma_prior"] = out["volume"].shift(1).rolling(range_bars, min_periods=range_bars).mean()
    out["wyckoff_volume_ratio"] = out["volume"] / out["volume_sma_prior"].replace(0.0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def spring_metrics(
    frame: pd.DataFrame,
    idx: int,
    direction: str,
    *,
    min_sweep_atr: float,
    min_reclaim_atr: float,
    min_wick_fraction: float,
    min_close_position: float,
    min_volume_ratio: float,
) -> dict[str, float] | None:
    row = frame.iloc[idx]
    atr = float(row["atr"])
    if not math.isfinite(atr) or atr <= 0.0:
        return None
    range_high = float(row["wyckoff_range_high"])
    range_low = float(row["wyckoff_range_low"])
    volume_ratio = float(row["wyckoff_volume_ratio"])
    if not all(math.isfinite(value) for value in [range_high, range_low, volume_ratio]):
        return None
    range_width_atr = (range_high - range_low) / atr
    if range_width_atr <= 0.0:
        return None

    if direction == "long":
        sweep_atr = (range_low - float(row["low"])) / atr
        reclaim_atr = (float(row["close"]) - range_low) / atr
        rejection_wick = float(row["lower_wick_frac"])
        directional_close_pos = float(row["close_pos"])
        directional_body_atr = float(row["body_atr"])
    else:
        sweep_atr = (float(row["high"]) - range_high) / atr
        reclaim_atr = (range_high - float(row["close"])) / atr
        rejection_wick = float(row["upper_wick_frac"])
        directional_close_pos = 1.0 - float(row["close_pos"])
        directional_body_atr = -float(row["body_atr"])

    if (
        sweep_atr < min_sweep_atr
        or reclaim_atr < min_reclaim_atr
        or rejection_wick < min_wick_fraction
        or directional_close_pos < min_close_position
        or volume_ratio < min_volume_ratio
    ):
        return None
    return {
        "spring_sweep_atr": sweep_atr,
        "spring_reclaim_atr": reclaim_atr,
        "spring_range_width_atr": range_width_atr,
        "spring_rejection_wick": rejection_wick,
        "spring_directional_close_pos": directional_close_pos,
        "spring_directional_body_atr": directional_body_atr,
        "spring_volume_ratio": volume_ratio,
    }


def find_sos(
    frame: pd.DataFrame,
    spring_idx: int,
    direction: str,
    confirm_bars: int,
    min_body_atr: float,
) -> int | None:
    spring = frame.iloc[spring_idx]
    spring_extreme = float(spring["high"] if direction == "long" else spring["low"])
    stop = min(len(frame), spring_idx + confirm_bars + 1)
    for idx in range(spring_idx + 1, stop):
        row = frame.iloc[idx]
        atr = float(row["atr"])
        if not math.isfinite(atr) or atr <= 0.0:
            continue
        body_atr = (float(row["close"]) - float(row["open"])) / atr
        if direction == "long":
            valid = float(row["close"]) > spring_extreme and body_atr >= min_body_atr
        else:
            valid = float(row["close"]) < spring_extreme and -body_atr >= min_body_atr
        if valid:
            return idx
    return None


def find_test(
    frame: pd.DataFrame,
    spring_idx: int,
    sos_idx: int,
    direction: str,
    test_bars: int,
    max_test_volume_ratio: float,
) -> tuple[int, float, float] | None:
    spring = frame.iloc[spring_idx]
    spring_atr = float(spring["atr"])
    spring_volume = max(float(spring["volume"]), 1e-12)
    spring_low = float(spring["low"])
    spring_high = float(spring["high"])
    stop = min(len(frame), sos_idx + test_bars + 1)
    for idx in range(sos_idx + 1, stop):
        row = frame.iloc[idx]
        relative_volume = float(row["volume"]) / spring_volume
        if not math.isfinite(relative_volume) or relative_volume > max_test_volume_ratio:
            continue
        if direction == "long":
            holds = float(row["low"]) > spring_low
            depth_atr = max(0.0, spring_high - float(row["low"])) / spring_atr
            confirms = float(row["close"]) > float(row["open"]) and float(row["close_pos"]) >= 0.55
        else:
            holds = float(row["high"]) < spring_high
            depth_atr = max(0.0, float(row["high"]) - spring_low) / spring_atr
            confirms = float(row["close"]) < float(row["open"]) and float(row["close_pos"]) <= 0.45
        if holds and confirms:
            return idx, depth_atr, relative_volume
    return None


def simulate_wyckoff_multi_rr(
    frame: pd.DataFrame,
    *,
    spring_idx: int,
    entry_idx: int,
    direction: str,
    rr_values: list[float],
    max_hold_bars: int,
    stop_buffer_atr: float,
    min_risk_pct: float,
    cost_bps_round_trip: float,
) -> dict[float, dict[str, Any]] | None:
    if entry_idx >= len(frame):
        return None
    spring = frame.iloc[spring_idx]
    entry = float(frame["open"].iloc[entry_idx])
    atr = float(spring["atr"])
    if not math.isfinite(atr) or atr <= 0.0:
        return None
    if direction == "long":
        stop = float(spring["low"]) - stop_buffer_atr * atr
        stop = min(stop, entry * (1.0 - min_risk_pct))
        risk = entry - stop
        targets = {rr: entry + rr * risk for rr in rr_values}
    else:
        stop = float(spring["high"]) + stop_buffer_atr * atr
        stop = max(stop, entry * (1.0 + min_risk_pct))
        risk = stop - entry
        targets = {rr: entry - rr * risk for rr in rr_values}
    if not math.isfinite(risk) or risk <= 0.0:
        return None

    cost_r = (cost_bps_round_trip / 10_000.0) * entry / risk
    end_idx = min(len(frame) - 1, entry_idx + max_hold_bars - 1)
    unresolved = set(rr_values)
    results: dict[float, dict[str, Any]] = {}
    mfe_r = 0.0
    mae_r = 0.0
    for cursor in range(entry_idx, end_idx + 1):
        high = float(frame["high"].iloc[cursor])
        low = float(frame["low"].iloc[cursor])
        if direction == "long":
            mfe_r = max(mfe_r, (high - entry) / risk)
            mae_r = max(mae_r, (entry - low) / risk)
            hit_stop = low <= stop
        else:
            mfe_r = max(mfe_r, (entry - low) / risk)
            mae_r = max(mae_r, (high - entry) / risk)
            hit_stop = high >= stop
        if hit_stop:
            for rr in unresolved:
                results[rr] = {
                    "result_r": -1.0 - cost_r,
                    "exit_idx": cursor,
                    "exit_time": pd.Timestamp(frame["close_time"].iloc[cursor]).tz_convert("UTC"),
                    "exit_reason": "stop",
                    "mfe_r": mfe_r,
                    "mae_r": mae_r,
                    "cost_r": cost_r,
                }
            unresolved.clear()
            break
        for rr in list(unresolved):
            hit_target = high >= targets[rr] if direction == "long" else low <= targets[rr]
            if hit_target:
                results[rr] = {
                    "result_r": rr - cost_r,
                    "exit_idx": cursor,
                    "exit_time": pd.Timestamp(frame["close_time"].iloc[cursor]).tz_convert("UTC"),
                    "exit_reason": "target",
                    "mfe_r": mfe_r,
                    "mae_r": mae_r,
                    "cost_r": cost_r,
                }
                unresolved.remove(rr)
        if not unresolved:
            break

    if unresolved:
        exit_price = float(frame["close"].iloc[end_idx])
        timeout_r = (exit_price - entry) / risk if direction == "long" else (entry - exit_price) / risk
        for rr in unresolved:
            results[rr] = {
                "result_r": timeout_r - cost_r,
                "exit_idx": end_idx,
                "exit_time": pd.Timestamp(frame["close_time"].iloc[end_idx]).tz_convert("UTC"),
                "exit_reason": "timeout",
                "mfe_r": mfe_r,
                "mae_r": mae_r,
                "cost_r": cost_r,
            }
    return results


def add_outcomes(
    candidate: dict[str, Any],
    outcomes: dict[float, dict[str, Any]] | None,
    rr_values: list[float],
) -> None:
    for rr in rr_values:
        key = f"{rr:g}"
        trade = outcomes.get(rr) if outcomes is not None else None
        if trade is None:
            candidate[f"result_r_{key}"] = np.nan
            candidate[f"exit_idx_{key}"] = np.nan
            candidate[f"exit_time_{key}"] = pd.NaT
            candidate[f"exit_reason_{key}"] = ""
            candidate[f"mfe_r_{key}"] = np.nan
            candidate[f"mae_r_{key}"] = np.nan
            candidate[f"cost_r_{key}"] = np.nan
        else:
            candidate[f"result_r_{key}"] = float(trade["result_r"])
            candidate[f"exit_idx_{key}"] = int(trade["exit_idx"])
            candidate[f"exit_time_{key}"] = trade["exit_time"]
            candidate[f"exit_reason_{key}"] = str(trade["exit_reason"])
            candidate[f"mfe_r_{key}"] = float(trade["mfe_r"])
            candidate[f"mae_r_{key}"] = float(trade["mae_r"])
            candidate[f"cost_r_{key}"] = float(trade["cost_r"])


def build_wyckoff_candidates(
    cascade: pd.DataFrame,
    frame_1m: pd.DataFrame,
    *,
    candidate_start: pd.Timestamp,
    minimum_cascade: float,
    styles: list[str],
    search_minutes: int,
    confirm_bars: int,
    test_bars: int,
    min_sweep_atr: float,
    min_reclaim_atr: float,
    min_wick_fraction: float,
    min_close_position: float,
    min_volume_ratio: float,
    min_sos_body_atr: float,
    max_test_volume_ratio: float,
    rr_values: list[float],
    min_risk_pcts: list[float],
    max_hold_bars: int,
    stop_buffer_atr: float,
    cost_bps_round_trip: float,
) -> pd.DataFrame:
    one_minute_ns = pd.DatetimeIndex(pd.to_datetime(frame_1m["open_time"], utc=True)).as_unit("ns").asi8
    eligible = cascade[pd.to_datetime(cascade["target_5m"], utc=True) >= candidate_start]
    if len(one_minute_ns) and not eligible.empty:
        first_target_ns = pd.Timestamp(eligible["target_5m"].iloc[0]).value
        if first_target_ns > one_minute_ns[-1]:
            raise ValueError("The 1m timestamp index does not overlap the cascade target range.")
    rows: list[dict[str, Any]] = []
    for cascade_direction, direction in [("low", "long"), ("high", "short")]:
        selected = eligible[eligible[f"{cascade_direction}_cascade_min"] >= minimum_cascade]
        print(f"  {direction}: scanning {len(selected):,} cascade windows")
        for item in selected.itertuples(index=False):
            window_time = pd.Timestamp(item.target_5m).tz_convert("UTC")
            start_idx = int(np.searchsorted(one_minute_ns, window_time.value, side="left"))
            stop_idx = min(len(frame_1m), start_idx + search_minutes)
            if start_idx < 25 or start_idx >= len(frame_1m):
                continue
            for spring_idx in range(start_idx, stop_idx):
                metrics = spring_metrics(
                    frame_1m,
                    spring_idx,
                    direction,
                    min_sweep_atr=min_sweep_atr,
                    min_reclaim_atr=min_reclaim_atr,
                    min_wick_fraction=min_wick_fraction,
                    min_close_position=min_close_position,
                    min_volume_ratio=min_volume_ratio,
                )
                if metrics is None:
                    continue
                sos_idx = find_sos(frame_1m, spring_idx, direction, confirm_bars, min_sos_body_atr)
                test = (
                    find_test(
                        frame_1m,
                        spring_idx,
                        sos_idx,
                        direction,
                        test_bars,
                        max_test_volume_ratio,
                    )
                    if sos_idx is not None
                    else None
                )
                setups: list[tuple[str, int, float, float]] = []
                if "spring" in styles:
                    setups.append(("spring", spring_idx, 0.0, 0.0))
                if "sos" in styles and sos_idx is not None:
                    setups.append(("sos", sos_idx, 0.0, 0.0))
                if "test" in styles and test is not None:
                    setups.append(("test", test[0], test[1], test[2]))

                for style, signal_idx, test_depth, test_volume_ratio in setups:
                    entry_idx = signal_idx + 1
                    if entry_idx >= len(frame_1m):
                        continue
                    signal = frame_1m.iloc[signal_idx]
                    spring = frame_1m.iloc[spring_idx]
                    spring_atr = max(float(spring["atr"]), 1e-12)
                    confirmation_displacement = (
                        (float(signal["close"]) - float(spring["close"])) / spring_atr
                        if direction == "long"
                        else (float(spring["close"]) - float(signal["close"])) / spring_atr
                    )
                    base = {
                        "signal_idx": signal_idx,
                        "entry_idx": entry_idx,
                        "target_time": window_time,
                        "decision_time": pd.Timestamp(frame_1m["open_time"].iloc[entry_idx]).tz_convert("UTC"),
                        "direction": direction,
                        "cascade_direction": cascade_direction,
                        "nested_truth": bool(getattr(item, f"nested_{cascade_direction}_truth")),
                        **cascade_features(item, cascade_direction),
                        **directional_price_features(signal, direction),
                        **time_features(pd.Timestamp(frame_1m["open_time"].iloc[entry_idx]).tz_convert("UTC")),
                        **metrics,
                        "wyckoff_style_spring": float(style == "spring"),
                        "wyckoff_style_sos": float(style == "sos"),
                        "wyckoff_style_test": float(style == "test"),
                        "window_offset_minutes": float(spring_idx - start_idx),
                        "confirmation_bars": float(signal_idx - spring_idx),
                        "confirmation_displacement_atr": confirmation_displacement,
                        "confirmation_directional_body_atr": (
                            float(signal["body_atr"]) if direction == "long" else -float(signal["body_atr"])
                        ),
                        "confirmation_volume_ratio": float(signal["wyckoff_volume_ratio"]),
                        "test_depth_atr": test_depth,
                        "test_volume_ratio_to_spring": test_volume_ratio,
                        "wyckoff_style": style,
                        "spring_idx": spring_idx,
                    }
                    for min_risk_pct in min_risk_pcts:
                        candidate = {**base, "min_risk_pct": float(min_risk_pct)}
                        outcomes = simulate_wyckoff_multi_rr(
                            frame_1m,
                            spring_idx=spring_idx,
                            entry_idx=entry_idx,
                            direction=direction,
                            rr_values=rr_values,
                            max_hold_bars=max_hold_bars,
                            stop_buffer_atr=stop_buffer_atr,
                            min_risk_pct=min_risk_pct,
                            cost_bps_round_trip=cost_bps_round_trip,
                        )
                        add_outcomes(candidate, outcomes, rr_values)
                        rows.append(candidate)
                break

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["decision_time"] = pd.to_datetime(out["decision_time"], utc=True)
    for rr in rr_values:
        out[f"exit_time_{rr:g}"] = pd.to_datetime(out[f"exit_time_{rr:g}"], utc=True)
    out = out.sort_values(
        ["direction", "signal_idx", "min_risk_pct", "cascade_min", "wyckoff_style"],
        ascending=[True, True, True, False, True],
    )
    out = out.drop_duplicates(["direction", "signal_idx", "min_risk_pct", "wyckoff_style"], keep="first")
    return out.sort_values(["decision_time", "cascade_min"], ascending=[True, False]).reset_index(drop=True)


def combine_cascades(
    main_cache: Path,
    prior_cache: Path | None,
    candidate_start: pd.Timestamp,
    prior_end: pd.Timestamp,
) -> pd.DataFrame:
    main = pd.read_pickle(main_cache)["cascade"]
    if prior_cache is None:
        return main
    prior = pd.read_pickle(prior_cache)["cascade"]
    prior_time = pd.to_datetime(prior["target_5m"], utc=True)
    main_time = pd.to_datetime(main["target_5m"], utc=True)
    return pd.concat(
        [
            prior[(prior_time >= candidate_start) & (prior_time < prior_end)],
            main[main_time >= prior_end],
        ],
        ignore_index=True,
    ).sort_values("target_5m").reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="1m Wyckoff execution inside hierarchical BTC reversal windows.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2026-06-01")
    parser.add_argument("--candidate-start", default="2023-01-01")
    parser.add_argument("--start-month", default="2025-01")
    parser.add_argument("--end-month", default="2026-05")
    parser.add_argument("--validation-months", type=int, default=12)
    parser.add_argument("--minimum-cascade", type=float, default=0.60)
    parser.add_argument("--styles", default="spring,sos,test")
    parser.add_argument("--search-minutes", type=int, default=15)
    parser.add_argument("--range-bars", type=int, default=20)
    parser.add_argument("--confirm-bars", type=int, default=5)
    parser.add_argument("--test-bars", type=int, default=8)
    parser.add_argument("--min-sweep-atr", type=float, default=0.05)
    parser.add_argument("--min-reclaim-atr", type=float, default=0.0)
    parser.add_argument("--min-wick-fraction", type=float, default=0.25)
    parser.add_argument("--min-close-position", type=float, default=0.55)
    parser.add_argument("--min-volume-ratio", type=float, default=1.0)
    parser.add_argument("--min-sos-body-atr", type=float, default=0.10)
    parser.add_argument("--max-test-volume-ratio", type=float, default=0.85)
    parser.add_argument("--rr-values", default="1.5,2,3,5,10")
    parser.add_argument("--min-risk-pcts", default="0.0025,0.005")
    parser.add_argument("--max-hold-bars", type=int, default=1440)
    parser.add_argument("--stop-buffer-atr", type=float, default=0.05)
    parser.add_argument("--cost-bps-round-trip", type=float, default=8.0)
    parser.add_argument("--models", default="logit")
    parser.add_argument("--feature-sets", default="cascade_wyckoff,wyckoff_only")
    parser.add_argument("--direction-scopes", default="both,long,short")
    parser.add_argument("--coverages", default="0.02,0.035,0.05,0.075,0.10,0.15,0.20")
    parser.add_argument("--min-validation-trades", type=int, default=8)
    parser.add_argument("--min-positive-validation-month-fraction", type=float, default=0.50)
    parser.add_argument("--dynamic-rr", action="store_true")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--cascade-cache",
        type=Path,
        default=Path("scripts/.cache/astro_cycle/hierarchical_cascade_hgb_price_time.pkl"),
    )
    parser.add_argument(
        "--prior-cascade-cache",
        type=Path,
        default=Path("scripts/.cache/astro_cycle/hierarchical_cascade_hgb_price_time_train2023.pkl"),
    )
    parser.add_argument("--prior-cascade-end", default="2024-01-01")
    parser.add_argument(
        "--candidate-cache",
        type=Path,
        default=Path("scripts/.cache/astro_cycle/hierarchical_wyckoff_1m_candidates.pkl"),
    )
    parser.add_argument("--refresh-candidates", action="store_true")
    parser.add_argument("--output-json", type=Path, default=Path("scripts/hierarchical_wyckoff_1m.json"))
    parser.add_argument("--summary-csv", type=Path, default=Path("scripts/hierarchical_wyckoff_1m_summary.csv"))
    parser.add_argument("--monthly-csv", type=Path, default=Path("scripts/hierarchical_wyckoff_1m_monthly.csv"))
    parser.add_argument("--trades-csv", type=Path, default=Path("scripts/hierarchical_wyckoff_1m_trades.csv"))
    args = parser.parse_args()

    start = parse_utc_datetime(args.start)
    end = parse_utc_datetime(args.end)
    candidate_start = pd.Timestamp(parse_utc_datetime(args.candidate_start))
    prior_end = pd.Timestamp(parse_utc_datetime(args.prior_cascade_end))
    styles = parse_str_list(args.styles)
    rr_values = parse_float_list(args.rr_values)
    min_risk_pcts = parse_float_list(args.min_risk_pcts)
    models = parse_str_list(args.models)
    feature_sets = parse_str_list(args.feature_sets)
    direction_scopes = parse_str_list(args.direction_scopes)
    coverages = parse_float_list(args.coverages)
    for style in styles:
        if style not in {"spring", "sos", "test"}:
            raise ValueError(f"Unknown Wyckoff style: {style}")
    for feature_set in feature_sets:
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"Unknown feature set {feature_set!r}")

    if args.candidate_cache.exists() and not args.refresh_candidates:
        print(f"Loading Wyckoff candidate cache {args.candidate_cache}...")
        cached = pd.read_pickle(args.candidate_cache)
        candidates = cached["candidates"]
        cached_config = cached["config"]
        cached_rrs = {float(value) for value in cached_config["rr_values"]}
        cached_risks = {float(value) for value in cached_config["min_risk_pcts"]}
        if not set(rr_values).issubset(cached_rrs) or not set(min_risk_pcts).issubset(cached_risks):
            raise ValueError("Wyckoff cache lacks requested RR/min-risk values; use --refresh-candidates.")
        expected = {
            "minimum_cascade": float(args.minimum_cascade),
            "search_minutes": int(args.search_minutes),
            "range_bars": int(args.range_bars),
            "confirm_bars": int(args.confirm_bars),
            "test_bars": int(args.test_bars),
            "max_hold_bars": int(args.max_hold_bars),
            "cost_bps_round_trip": float(args.cost_bps_round_trip),
        }
        if any(cached_config.get(key) != value for key, value in expected.items()):
            raise ValueError("Wyckoff cache configuration differs from this run; use --refresh-candidates.")
    else:
        print(f"Loading/fetching {args.symbol} 1m {start} -> {end}...")
        raw_1m = load_bybit_cached(args.symbol, "1m", start, end, args.cache_dir)
        frame_1m = prepare_1m_features(raw_1m, args.range_bars)
        cascade = combine_cascades(
            args.cascade_cache,
            args.prior_cascade_cache,
            candidate_start,
            prior_end,
        )
        print("Scanning 1m Wyckoff entries...")
        candidates = build_wyckoff_candidates(
            cascade,
            frame_1m,
            candidate_start=candidate_start,
            minimum_cascade=args.minimum_cascade,
            styles=styles,
            search_minutes=args.search_minutes,
            confirm_bars=args.confirm_bars,
            test_bars=args.test_bars,
            min_sweep_atr=args.min_sweep_atr,
            min_reclaim_atr=args.min_reclaim_atr,
            min_wick_fraction=args.min_wick_fraction,
            min_close_position=args.min_close_position,
            min_volume_ratio=args.min_volume_ratio,
            min_sos_body_atr=args.min_sos_body_atr,
            max_test_volume_ratio=args.max_test_volume_ratio,
            rr_values=rr_values,
            min_risk_pcts=min_risk_pcts,
            max_hold_bars=args.max_hold_bars,
            stop_buffer_atr=args.stop_buffer_atr,
            cost_bps_round_trip=args.cost_bps_round_trip,
        )
        args.candidate_cache.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(
            {
                "candidates": candidates,
                "config": {
                    "start": start,
                    "end": end,
                    "candidate_start": candidate_start,
                    "minimum_cascade": args.minimum_cascade,
                    "styles": styles,
                    "search_minutes": args.search_minutes,
                    "range_bars": args.range_bars,
                    "confirm_bars": args.confirm_bars,
                    "test_bars": args.test_bars,
                    "rr_values": rr_values,
                    "min_risk_pcts": min_risk_pcts,
                    "max_hold_bars": args.max_hold_bars,
                    "cost_bps_round_trip": args.cost_bps_round_trip,
                },
            },
            args.candidate_cache,
        )
        print(f"Wrote {args.candidate_cache}")

    if candidates.empty:
        raise RuntimeError("No 1m Wyckoff candidates were found.")
    candidates = candidates[candidates["wyckoff_style"].isin(styles)].copy()
    if candidates.empty:
        raise RuntimeError(f"No cached 1m Wyckoff candidates match requested styles: {styles}")
    candidates["decision_time"] = pd.to_datetime(candidates["decision_time"], utc=True)
    print(f"Wyckoff candidates: {len(candidates):,}")

    summary_rows: list[dict[str, Any]] = []
    monthly_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    for model_name in models:
        for feature_set in feature_sets:
            for min_risk_pct in min_risk_pcts:
                for direction_scope in direction_scopes:
                    if args.dynamic_rr:
                        label = (
                            f"{model_name}/{feature_set}/risk={min_risk_pct:g}/"
                            f"direction={direction_scope}/rr=dynamic"
                        )
                        print(f"Running {label}...")
                        summary, monthly, trades = run_dynamic_walkforward(
                            candidates,
                            rr_values=rr_values,
                            min_risk_pct=min_risk_pct,
                            direction_scope=direction_scope,
                            feature_set=feature_set,
                            model_name=model_name,
                            validation_months=args.validation_months,
                            start_month=args.start_month,
                            end_month=args.end_month,
                            coverages=coverages,
                            min_validation_trades=args.min_validation_trades,
                            min_positive_month_fraction=args.min_positive_validation_month_fraction,
                            seed=args.seed,
                        )
                        summary_rows.append(summary)
                        monthly.insert(0, "experiment", label)
                        trades.insert(0, "experiment", label)
                        monthly_frames.append(monthly)
                        trade_frames.append(trades)
                    else:
                        for rr in rr_values:
                            label = (
                                f"{model_name}/{feature_set}/risk={min_risk_pct:g}/"
                                f"direction={direction_scope}/rr={rr:g}"
                            )
                            print(f"Running {label}...")
                            summary, monthly, trades = run_walkforward(
                                candidates,
                                rr=rr,
                                min_risk_pct=min_risk_pct,
                                direction_scope=direction_scope,
                                feature_set=feature_set,
                                model_name=model_name,
                                validation_months=args.validation_months,
                                start_month=args.start_month,
                                end_month=args.end_month,
                                coverages=coverages,
                                min_validation_trades=args.min_validation_trades,
                                min_positive_month_fraction=args.min_positive_validation_month_fraction,
                                seed=args.seed,
                            )
                            summary_rows.append(summary)
                            monthly.insert(0, "experiment", label)
                            trades.insert(0, "experiment", label)
                            monthly_frames.append(monthly)
                            trade_frames.append(trades)

    summary_table = pd.DataFrame(summary_rows).sort_values(
        ["net_r", "profit_factor", "trades"],
        ascending=[False, False, False],
    )
    monthly_table = pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame()
    trades_table = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    args.summary_csv.write_text(summary_table.to_csv(index=False), encoding="utf-8")
    args.monthly_csv.write_text(monthly_table.to_csv(index=False), encoding="utf-8")
    args.trades_csv.write_text(trades_table.to_csv(index=False), encoding="utf-8")
    result = {
        "config": {
            "symbol": args.symbol,
            "start": start,
            "end": end,
            "candidate_start": candidate_start,
            "start_month": args.start_month,
            "end_month": args.end_month,
            "validation_months": args.validation_months,
            "styles": styles,
            "search_minutes": args.search_minutes,
            "range_bars": args.range_bars,
            "rr_values": rr_values,
            "min_risk_pcts": min_risk_pcts,
            "models": models,
            "feature_sets": feature_sets,
            "direction_scopes": direction_scopes,
            "dynamic_rr": args.dynamic_rr,
            "cost_bps_round_trip": args.cost_bps_round_trip,
        },
        "candidate_rows": len(candidates),
        "style_counts": candidates["wyckoff_style"].value_counts().to_dict(),
        "experiments": summary_table.to_dict(orient="records"),
    }
    args.output_json.write_text(json.dumps(result, indent=2, default=json_default), encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.monthly_csv}")
    print(f"Wrote {args.trades_csv}")
    print("\nTop experiments:")
    print(summary_table.head(15).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
