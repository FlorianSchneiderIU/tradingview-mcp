from __future__ import annotations

import argparse
import math
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backtest_turtle_soup import parse_utc_datetime  # noqa: E402
from scripts.experiment_session_orb import add_htf_context, build_grid, metrics, parse_thresholds, train_ml_ranker  # noqa: E402
from scripts.experiment_session_orb_fast import (  # noqa: E402
    apply_candidate_filter,
    build_contexts,
    generate_trades,
    select_ranked_trades,
    to_arrays,
)
from scripts.train_bybit_top_marketcap_turtle_models import BYBIT_PUBLIC_URL, ensure_bybit_cache  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep the session Judas/FVG ORB candidate strategy across a Bybit top universe "
            "and train one event ranker per symbol."
        )
    )
    parser.add_argument("--universe-csv", type=Path, default=Path("scripts/bybit_top50_turtle_per_symbol_turnover_sfp_tex_v4_universe.csv"))
    parser.add_argument("--symbols", nargs="+", default=[], help="Optional explicit symbol list; bypasses --universe-csv.")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--cache-dir", type=Path, default=Path("scripts/.cache/bybit_linear"))
    parser.add_argument("--fetch-missing", action="store_true", help="Fetch missing Bybit cache files instead of skipping them.")
    parser.add_argument("--base-url", default=BYBIT_PUBLIC_URL)
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--warmup-start", default="2021-09-01")
    parser.add_argument("--train-start", default="2022-04-20")
    parser.add_argument("--split", default="2025-04-20")
    parser.add_argument("--end", default="2026-04-20")
    parser.add_argument("--sessions", default="asia,london,ny")
    parser.add_argument("--or-minutes", default="30,60,90")
    parser.add_argument("--grid-mode", choices=["fast", "full"], default="fast")
    parser.add_argument("--family", choices=["judas"], default="judas")
    parser.add_argument("--entry-mode", choices=["fvg_retest", "level_retest", "immediate"], default="fvg_retest")
    parser.add_argument(
        "--rank-config-scope",
        choices=["all", "strategy"],
        default="all",
        help="Use the original ETH research flow by ranking all ORB variants first, or rank only this strategy family.",
    )
    parser.add_argument("--top-train-variants", type=int, default=24)
    parser.add_argument("--min-config-train-trades", type=int, default=80)
    parser.add_argument("--min-config-oos-trades", type=int, default=15)
    parser.add_argument(
        "--candidate-filter",
        choices=["none", "judas_fvg_risk2", "judas_fvg_risk25", "asia_ny_judas_fvg_risk25"],
        default="judas_fvg_risk2",
    )
    parser.add_argument("--fee-bps-per-side", type=float, default=6.5)
    parser.add_argument("--thresholds", default="0.45,0.50,0.55,0.60,0.65,0.70")
    parser.add_argument("--fixed-threshold", type=float, default=0.50)
    parser.add_argument("--min-train-trades-for-threshold", type=int, default=80)
    parser.add_argument("--min-oos-trades", type=int, default=30)
    parser.add_argument("--min-oos-pf", type=float, default=1.20)
    parser.add_argument("--min-oos-net-r", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--write-selected-trades", action="store_true", help="Write combined selected trades for fixed and train-chosen thresholds.")
    parser.add_argument("--write-candidates", action="store_true", help="Write combined filtered candidate rows. This can be large.")
    parser.add_argument("--output-prefix", type=Path, default=Path("scripts/session_orb_top50_judas_fvg_risk2_v1"))
    return parser.parse_args()


def utc_now_label() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def clean_symbol(value: str) -> str:
    return str(value).strip().upper()


def find_cache_path(symbol: str, cache_dir: Path, interval: str) -> Path | None:
    candidates = sorted(cache_dir.glob(f"{symbol.lower()}_{interval}_*.pkl"), key=lambda p: p.stat().st_size, reverse=True)
    return candidates[0] if candidates else None


def load_universe(args: argparse.Namespace) -> pd.DataFrame:
    if args.symbols:
        rows = [{"symbol": clean_symbol(symbol), "universe_rank": idx + 1} for idx, symbol in enumerate(args.symbols)]
        return pd.DataFrame(rows).head(args.top_n).copy()
    universe = pd.read_csv(args.universe_csv)
    universe["symbol"] = universe["symbol"].astype(str).map(clean_symbol)
    if "universe_rank" not in universe.columns:
        universe["universe_rank"] = np.arange(1, len(universe) + 1)
    return universe.head(args.top_n).copy()


def finite_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return math.nan
    return out if math.isfinite(out) else math.nan


def prefixed_metrics(frame: pd.DataFrame, prefix: str) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metrics(frame).items()}


def threshold_choice(table: pd.DataFrame, *, min_train_trades: int) -> pd.Series | None:
    if table.empty:
        return None
    choices = table[table["train_trades"].astype(int) >= int(min_train_trades)].copy()
    if choices.empty:
        choices = table.copy()
    choices["rank_pf"] = choices["train_profit_factor"].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    choices["rank_net"] = choices["train_net_r"].astype(float)
    choices["rank_trades"] = choices["train_trades"].astype(int)
    return choices.sort_values(["rank_pf", "rank_net", "rank_trades"], ascending=[False, False, False]).iloc[0]


def closest_threshold_row(table: pd.DataFrame, threshold: float) -> pd.Series | None:
    if table.empty:
        return None
    tmp = table.copy()
    tmp["_dist"] = (tmp["threshold"].astype(float) - float(threshold)).abs()
    return tmp.sort_values("_dist").iloc[0]


def summary_from_selected(selected: pd.DataFrame, split: pd.Timestamp, *, prefix: str) -> dict[str, Any]:
    if selected.empty:
        return {
            **prefixed_metrics(selected, f"{prefix}_train"),
            **prefixed_metrics(selected, f"{prefix}_oos"),
            f"{prefix}_oos_asia_net_r": 0.0,
            f"{prefix}_oos_london_net_r": 0.0,
            f"{prefix}_oos_ny_net_r": 0.0,
            f"{prefix}_oos_long_net_r": 0.0,
            f"{prefix}_oos_short_net_r": 0.0,
        }
    out = selected.copy()
    out["entry_time"] = pd.to_datetime(out["entry_time"], utc=True, errors="coerce")
    train = out[out["entry_time"] < split].copy()
    oos = out[out["entry_time"] >= split].copy()
    row: dict[str, Any] = {
        **prefixed_metrics(train, f"{prefix}_train"),
        **prefixed_metrics(oos, f"{prefix}_oos"),
    }
    for session in ("asia", "london", "ny"):
        subset = oos[oos["session"].eq(session)]
        row[f"{prefix}_oos_{session}_trades"] = int(len(subset))
        row[f"{prefix}_oos_{session}_net_r"] = round(float(subset["r_multiple"].sum()), 3) if not subset.empty else 0.0
    for direction in ("long", "short"):
        subset = oos[oos["direction"].eq(direction)]
        row[f"{prefix}_oos_{direction}_trades"] = int(len(subset))
        row[f"{prefix}_oos_{direction}_net_r"] = round(float(subset["r_multiple"].sum()), 3) if not subset.empty else 0.0
    return row


def build_strategy_configs(args_dict: dict[str, Any]) -> list[Any]:
    grid_args = argparse.Namespace(
        sessions=args_dict["sessions"],
        or_minutes=args_dict["or_minutes"],
        grid_mode=args_dict["grid_mode"],
    )
    configs = build_grid(grid_args)
    return [
        cfg
        for cfg in configs
        if cfg.family == args_dict["family"] and cfg.entry_mode == args_dict["entry_mode"]
    ]


def build_all_configs(args_dict: dict[str, Any]) -> list[Any]:
    grid_args = argparse.Namespace(
        sessions=args_dict["sessions"],
        or_minutes=args_dict["or_minutes"],
        grid_mode=args_dict["grid_mode"],
    )
    return build_grid(grid_args)


def config_summary_score(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    pf = out["train_profit_factor"].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    out["train_score"] = out["train_net_r"].astype(float) + 8.0 * np.log1p(pf) + 0.25 * out["train_max_dd_r"].astype(float)
    return out


def select_top_config_trades(
    arrays: Any,
    *,
    symbol: str,
    contexts: list[Any],
    params: dict[str, Any],
) -> tuple[list[Any], pd.DataFrame, dict[str, Any]]:
    scope_configs = build_all_configs(params) if params["rank_config_scope"] == "all" else build_strategy_configs(params)
    frames: list[tuple[Any, pd.DataFrame]] = []
    rows: list[dict[str, Any]] = []
    split = pd.Timestamp(params["split"])
    for cfg in scope_configs:
        trades = generate_trades(arrays, symbol=symbol, cfg=cfg, contexts=contexts, fee_bps_per_side=params["fee_bps_per_side"])
        if not trades.empty:
            trades = trades.copy()
            trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce")
            train = trades[trades["entry_time"] < split]
            oos = trades[trades["entry_time"] >= split]
        else:
            train = trades
            oos = trades
        frames.append((cfg, trades))
        rows.append(
            {
                **asdict(cfg),
                "variant": cfg.variant,
                **{f"train_{key}": value for key, value in metrics(train).items()},
                **{f"oos_{key}": value for key, value in metrics(oos).items()},
            }
        )
    summary = config_summary_score(pd.DataFrame(rows))
    eligible = summary[
        (summary["train_trades"].astype(int) >= int(params["min_config_train_trades"]))
        & (summary["oos_trades"].astype(int) >= int(params["min_config_oos_trades"]))
    ].copy()
    if eligible.empty:
        eligible = summary[summary["train_trades"].astype(int) >= max(1, int(params["min_config_train_trades"]) // 2)].copy()
    if eligible.empty:
        eligible = summary.copy()
    top = eligible.sort_values(["train_score", "train_profit_factor", "train_net_r"], ascending=[False, False, False]).head(int(params["top_train_variants"]))
    top_variants = set(top["variant"].astype(str))
    selected_pairs = [(cfg, trades) for cfg, trades in frames if cfg.variant in top_variants]
    top_frames = [trades for _, trades in selected_pairs if not trades.empty]
    selected_trades = pd.concat(top_frames, ignore_index=True) if top_frames else pd.DataFrame()
    strategy_pairs = [(cfg, trades) for cfg, trades in selected_pairs if cfg.family == params["family"] and cfg.entry_mode == params["entry_mode"]]
    diagnostics = {
        "rank_scope_configs": int(len(scope_configs)),
        "selected_configs": int(len(selected_pairs)),
        "selected_strategy_configs": int(len(strategy_pairs)),
        "selected_strategy_variants": ",".join(cfg.variant for cfg, _ in strategy_pairs),
    }
    return [cfg for cfg, _ in selected_pairs], selected_trades, diagnostics


def selected_with_tag(scored: pd.DataFrame, *, threshold: float, split: pd.Timestamp, selection: str) -> pd.DataFrame:
    selected = select_ranked_trades(scored, threshold=threshold, split=split)
    if not selected.empty:
        selected = selected.copy()
        selected["selection"] = selection
        selected["selection_threshold"] = float(threshold)
    return selected


def run_symbol_job(params: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    symbol = clean_symbol(params["symbol"])
    split = pd.Timestamp(params["split"])
    train_start = pd.Timestamp(params["train_start"])
    end = pd.Timestamp(params["end"])
    started = time.perf_counter()
    base_row: dict[str, Any] = {
        "symbol": symbol,
        "universe_rank": params.get("universe_rank", math.nan),
        "status": "error",
        "skip_reason": "",
    }
    for optional in ("turnover24h", "openInterestValue", "volume24h", "launch_time", "baseCoin"):
        if optional in params:
            base_row[optional] = params[optional]

    try:
        cache_path = find_cache_path(symbol, Path(params["cache_dir"]), params["interval"])
        if cache_path is None:
            if not params["fetch_missing"]:
                base_row.update({"status": "skipped", "skip_reason": "missing_cache", "elapsed_sec": round(time.perf_counter() - started, 2)})
                return base_row, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            cache_path = ensure_bybit_cache(
                symbol,
                params["interval"],
                params["warmup_start"],
                params["end_dt"],
                Path(params["cache_dir"]),
                params["base_url"],
            )

        raw = pd.read_pickle(cache_path)
        raw["open_time"] = pd.to_datetime(raw["open_time"], utc=True, errors="coerce")
        raw["close_time"] = pd.to_datetime(raw["close_time"], utc=True, errors="coerce")
        raw = raw[(raw["open_time"] >= train_start - pd.Timedelta(days=90)) & (raw["open_time"] < end)].copy()
        if len(raw) < 5000:
            base_row.update({"status": "skipped", "skip_reason": "too_few_warmup_bars", "bars": int(len(raw)), "elapsed_sec": round(time.perf_counter() - started, 2)})
            return base_row, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        df = add_htf_context(raw)
        df = df[(df["open_time"] >= train_start) & (df["open_time"] < end)].reset_index(drop=True)
        if len(df) < 3000:
            base_row.update({"status": "skipped", "skip_reason": "too_few_bars", "bars": int(len(df)), "elapsed_sec": round(time.perf_counter() - started, 2)})
            return base_row, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        arrays = to_arrays(df)
        sessions = [x.strip() for x in params["sessions"].split(",") if x.strip()]
        or_minutes = [int(x.strip()) for x in params["or_minutes"].split(",") if x.strip()]
        contexts = build_contexts(df, arrays, sessions=sessions, or_minutes=or_minutes)
        configs, all_trades, config_diagnostics = select_top_config_trades(
            arrays,
            symbol=symbol,
            contexts=contexts,
            params=params,
        )
        candidates = apply_candidate_filter(all_trades, params["candidate_filter"])
        if candidates.empty:
            base_row.update(
                {
                    "status": "skipped",
                    "skip_reason": "no_candidates",
                    "bars": int(len(df)),
                    "contexts": int(len(contexts)),
                    **config_diagnostics,
                    "raw_trades": int(len(all_trades)),
                    "candidates": 0,
                    "elapsed_sec": round(time.perf_counter() - started, 2),
                }
            )
            return base_row, pd.DataFrame(), candidates, pd.DataFrame()

        candidates["entry_time"] = pd.to_datetime(candidates["entry_time"], utc=True, errors="coerce")
        train_candidates = candidates[candidates["entry_time"] < split]
        oos_candidates = candidates[candidates["entry_time"] >= split]
        ml_table, scored, ml_cols = train_ml_ranker(candidates, split=split, thresholds=params["thresholds"])
        if ml_table.empty or scored.empty or "ml_prob" not in scored.columns:
            base_row.update(
                {
                    "status": "skipped",
                    "skip_reason": "ml_not_trainable",
                    "bars": int(len(df)),
                    "contexts": int(len(contexts)),
                    **config_diagnostics,
                    "raw_trades": int(len(all_trades)),
                    "candidates": int(len(candidates)),
                    "train_candidates": int(len(train_candidates)),
                    "oos_candidates": int(len(oos_candidates)),
                    **prefixed_metrics(train_candidates, "candidate_train"),
                    **prefixed_metrics(oos_candidates, "candidate_oos"),
                    "elapsed_sec": round(time.perf_counter() - started, 2),
                }
            )
            return base_row, ml_table, candidates, pd.DataFrame()

        fixed_row = closest_threshold_row(ml_table, params["fixed_threshold"])
        chosen_row = threshold_choice(ml_table, min_train_trades=params["min_train_trades_for_threshold"])
        fixed_threshold = float(fixed_row["threshold"]) if fixed_row is not None else float(params["fixed_threshold"])
        chosen_threshold = float(chosen_row["threshold"]) if chosen_row is not None else fixed_threshold
        fixed_selected = selected_with_tag(scored, threshold=fixed_threshold, split=split, selection="fixed")
        chosen_selected = selected_with_tag(scored, threshold=chosen_threshold, split=split, selection="train_chosen")
        selected = pd.concat([fixed_selected, chosen_selected], ignore_index=True) if (not fixed_selected.empty or not chosen_selected.empty) else pd.DataFrame()

        fixed_summary = summary_from_selected(fixed_selected, split, prefix="fixed")
        chosen_summary = summary_from_selected(chosen_selected, split, prefix="chosen")
        fixed_table_summary = (
            {f"fixed_table_{key}": fixed_row[key] for key in fixed_row.index if key != "_dist"}
            if fixed_row is not None
            else {}
        )
        chosen_table_summary = (
            {f"chosen_table_{key}": chosen_row[key] for key in chosen_row.index if not str(key).startswith("rank_")}
            if chosen_row is not None
            else {}
        )
        row = {
            **base_row,
            "status": "trained",
            "bars": int(len(df)),
            "contexts": int(len(contexts)),
            **config_diagnostics,
            "raw_trades": int(len(all_trades)),
            "candidates": int(len(candidates)),
            "train_candidates": int(len(train_candidates)),
            "oos_candidates": int(len(oos_candidates)),
            "ml_feature_count": int(len(ml_cols)),
            **prefixed_metrics(train_candidates, "candidate_train"),
            **prefixed_metrics(oos_candidates, "candidate_oos"),
            "fixed_threshold": fixed_threshold,
            **fixed_table_summary,
            "chosen_threshold": chosen_threshold,
            **chosen_table_summary,
            **fixed_summary,
            **chosen_summary,
            "elapsed_sec": round(time.perf_counter() - started, 2),
        }
        row["fixed_add_candidate"] = (
            int(row.get("fixed_oos_trades", 0)) >= int(params["min_oos_trades"])
            and finite_float(row.get("fixed_oos_profit_factor")) >= float(params["min_oos_pf"])
            and finite_float(row.get("fixed_oos_net_r")) > float(params["min_oos_net_r"])
            and finite_float(row.get("fixed_train_net_r")) > 0.0
        )
        row["chosen_add_candidate"] = (
            int(row.get("chosen_oos_trades", 0)) >= int(params["min_oos_trades"])
            and finite_float(row.get("chosen_oos_profit_factor")) >= float(params["min_oos_pf"])
            and finite_float(row.get("chosen_oos_net_r")) > float(params["min_oos_net_r"])
            and finite_float(row.get("chosen_train_net_r")) > 0.0
        )
        ml_table = ml_table.copy()
        ml_table.insert(0, "symbol", symbol)
        return row, ml_table, candidates, selected
    except Exception as exc:
        base_row.update(
            {
                "status": "error",
                "skip_reason": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=6),
                "elapsed_sec": round(time.perf_counter() - started, 2),
            }
        )
        return base_row, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def params_for_row(row: pd.Series, args: argparse.Namespace, *, train_start: pd.Timestamp, split: pd.Timestamp, end: pd.Timestamp, warmup_start: datetime) -> dict[str, Any]:
    params: dict[str, Any] = {
        "symbol": clean_symbol(row["symbol"]),
        "universe_rank": int(row.get("universe_rank", 0)) if pd.notna(row.get("universe_rank", np.nan)) else math.nan,
        "cache_dir": str(args.cache_dir),
        "fetch_missing": bool(args.fetch_missing),
        "base_url": args.base_url,
        "interval": args.interval,
        "warmup_start": warmup_start,
        "train_start": train_start,
        "split": split,
        "end": end,
        "end_dt": end.to_pydatetime(),
        "sessions": args.sessions,
        "or_minutes": args.or_minutes,
        "grid_mode": args.grid_mode,
        "family": args.family,
        "entry_mode": args.entry_mode,
        "rank_config_scope": args.rank_config_scope,
        "top_train_variants": args.top_train_variants,
        "min_config_train_trades": args.min_config_train_trades,
        "min_config_oos_trades": args.min_config_oos_trades,
        "candidate_filter": args.candidate_filter,
        "fee_bps_per_side": args.fee_bps_per_side,
        "thresholds": parse_thresholds(args.thresholds),
        "fixed_threshold": args.fixed_threshold,
        "min_train_trades_for_threshold": args.min_train_trades_for_threshold,
        "min_oos_trades": args.min_oos_trades,
        "min_oos_pf": args.min_oos_pf,
        "min_oos_net_r": args.min_oos_net_r,
    }
    for optional in ("turnover24h", "openInterestValue", "volume24h", "launch_time", "baseCoin"):
        if optional in row.index:
            params[optional] = row[optional]
    return params


def write_frames(
    *,
    leaderboard: pd.DataFrame,
    thresholds: list[pd.DataFrame],
    candidates: list[pd.DataFrame],
    selected: list[pd.DataFrame],
    args: argparse.Namespace,
    partial: bool,
) -> None:
    suffix = "_partial" if partial else ""
    leaderboard_path = args.output_prefix.with_name(f"{args.output_prefix.name}{suffix}_leaderboard.csv")
    threshold_path = args.output_prefix.with_name(f"{args.output_prefix.name}{suffix}_thresholds.csv")
    selected_path = args.output_prefix.with_name(f"{args.output_prefix.name}{suffix}_selected_trades.csv")
    candidate_path = args.output_prefix.with_name(f"{args.output_prefix.name}{suffix}_candidates.csv")
    leaderboard.to_csv(leaderboard_path, index=False)
    if thresholds:
        pd.concat(thresholds, ignore_index=True).to_csv(threshold_path, index=False)
    if args.write_selected_trades and selected:
        pd.concat(selected, ignore_index=True).to_csv(selected_path, index=False)
    if args.write_candidates and candidates:
        pd.concat(candidates, ignore_index=True).to_csv(candidate_path, index=False)


def main() -> None:
    args = parse_args()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    train_start = pd.Timestamp(parse_utc_datetime(args.train_start))
    split = pd.Timestamp(parse_utc_datetime(args.split))
    end = pd.Timestamp(parse_utc_datetime(args.end))
    warmup_start = parse_utc_datetime(args.warmup_start)
    universe = load_universe(args)
    universe_path = args.output_prefix.with_name(f"{args.output_prefix.name}_universe.csv")
    config_path = args.output_prefix.with_name(f"{args.output_prefix.name}_config.json")
    universe.to_csv(universe_path, index=False)
    pd.Series({key: str(value) for key, value in vars(args).items()}).to_json(config_path, indent=2)

    jobs = [params_for_row(row, args, train_start=train_start, split=split, end=end, warmup_start=warmup_start) for _, row in universe.iterrows()]
    print(
        f"Session ORB sweep {utc_now_label()} | {len(jobs)} symbols | "
        f"{args.family}/{args.entry_mode}/{args.candidate_filter} | split={split.date()}",
        flush=True,
    )
    print(f"Universe -> {universe_path}", flush=True)

    rows: list[dict[str, Any]] = []
    threshold_frames: list[pd.DataFrame] = []
    candidate_frames: list[pd.DataFrame] = []
    selected_frames: list[pd.DataFrame] = []
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = {pool.submit(run_symbol_job, params): params["symbol"] for params in jobs}
        for idx, future in enumerate(as_completed(futures), 1):
            symbol = futures[future]
            row, table, candidates, selected = future.result()
            rows.append(row)
            if not table.empty:
                threshold_frames.append(table)
            if args.write_candidates and not candidates.empty:
                candidate_frames.append(candidates)
            if args.write_selected_trades and not selected.empty:
                selected_frames.append(selected)
            status = row.get("status", "unknown")
            detail = row.get("skip_reason", "")
            if status == "trained":
                detail = (
                    f"fixed_oos={row.get('fixed_oos_trades', 0)}tr "
                    f"PF={row.get('fixed_oos_profit_factor', 0)} "
                    f"R={row.get('fixed_oos_net_r', 0)} | "
                    f"chosen_oos={row.get('chosen_oos_trades', 0)}tr "
                    f"PF={row.get('chosen_oos_profit_factor', 0)} "
                    f"R={row.get('chosen_oos_net_r', 0)}"
                )
            print(f"[{idx:02d}/{len(jobs):02d}] {symbol}: {status} {detail}", flush=True)
            if idx % 5 == 0 or idx == len(jobs):
                leaderboard = pd.DataFrame(rows)
                write_frames(
                    leaderboard=leaderboard,
                    thresholds=threshold_frames,
                    candidates=candidate_frames,
                    selected=selected_frames,
                    args=args,
                    partial=idx != len(jobs),
                )

    leaderboard = pd.DataFrame(rows)
    if not leaderboard.empty:
        add_score = leaderboard["chosen_add_candidate"].fillna(False).astype(bool) if "chosen_add_candidate" in leaderboard.columns else pd.Series(False, index=leaderboard.index)
        pf = leaderboard.get("chosen_oos_profit_factor", pd.Series(0.0, index=leaderboard.index)).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        net = leaderboard.get("chosen_oos_net_r", pd.Series(0.0, index=leaderboard.index)).fillna(0.0).astype(float)
        trades = leaderboard.get("chosen_oos_trades", pd.Series(0, index=leaderboard.index)).fillna(0).astype(int)
        dd = leaderboard.get("chosen_oos_max_dd_r", pd.Series(0.0, index=leaderboard.index)).fillna(0.0).astype(float)
        leaderboard["deploy_score"] = add_score.astype(int) * 1000.0 + net + 5.0 * np.log1p(pf.clip(lower=0)) + 0.03 * trades + 0.20 * dd
        for column in ("chosen_oos_net_r", "chosen_oos_profit_factor"):
            if column not in leaderboard.columns:
                leaderboard[column] = 0.0
        leaderboard = leaderboard.sort_values(["deploy_score", "chosen_oos_net_r", "chosen_oos_profit_factor"], ascending=[False, False, False])
    write_frames(
        leaderboard=leaderboard,
        thresholds=threshold_frames,
        candidates=candidate_frames,
        selected=selected_frames,
        args=args,
        partial=False,
    )
    elapsed = time.perf_counter() - start
    trained = int((leaderboard.get("status", pd.Series(dtype=str)) == "trained").sum()) if not leaderboard.empty else 0
    candidates_add = int(leaderboard.get("chosen_add_candidate", pd.Series(dtype=bool)).fillna(False).sum()) if not leaderboard.empty else 0
    print(
        f"Done in {elapsed / 60.0:.1f}m | trained={trained}/{len(jobs)} | chosen add-candidates={candidates_add}",
        flush=True,
    )
    print(f"Leaderboard -> {args.output_prefix.with_name(f'{args.output_prefix.name}_leaderboard.csv')}", flush=True)


if __name__ == "__main__":
    main()
