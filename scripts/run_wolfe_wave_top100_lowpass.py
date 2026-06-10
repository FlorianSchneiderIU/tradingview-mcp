from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backtest_wolfe_wave import (  # noqa: E402
    BTC_TUNE_KEYS,
    BTC_TUNE_VALUES,
    WolfeConfig,
    btc_parameter_grid,
    bybit_symbol,
    config_key,
    enable_wolfe_v2_tune_values,
    evaluate_btc_grid,
    fetch_bybit_mintick,
    refine_btc,
    normalize_timeframe,
    parse_utc_datetime,
)
from scripts.tune_wolfe_wave_universe import (  # noqa: E402
    DEFAULT_BASE_URL,
    has_min_daily_history,
    load_or_fetch_data,
)


DEFAULT_EXCLUDED_SYMBOLS = {
    "USDCUSDT",
    "USDEUSDT",
    "FDUSDUSDT",
    "DAIUSDT",
    "USDDUSDT",
}


def parse_csv_values(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def apply_value_filters(pattern_tfs: str | None, regime_filters: str | None, *, enable_v2_grid: bool = False) -> None:
    if enable_v2_grid:
        enable_wolfe_v2_tune_values()

    selected_tfs = parse_csv_values(pattern_tfs)
    if selected_tfs:
        normalized = tuple(normalize_timeframe(value) for value in selected_tfs)
        unknown = [value for value in normalized if value not in BTC_TUNE_VALUES["pattern_tf"]]
        if unknown:
            raise ValueError(f"Unknown pattern timeframe(s): {', '.join(unknown)}")
        BTC_TUNE_VALUES["pattern_tf"] = normalized

    selected_regimes = parse_csv_values(regime_filters)
    if selected_regimes:
        normalized_regimes = tuple(value.lower() for value in selected_regimes)
        unknown = [value for value in normalized_regimes if value not in BTC_TUNE_VALUES["regime_filter"]]
        if unknown:
            raise ValueError(f"Unknown regime filter(s): {', '.join(unknown)}")
        BTC_TUNE_VALUES["regime_filter"] = normalized_regimes


def fetch_linear_usdt_instruments(base_url: str) -> dict[str, dict[str, Any]]:
    session = requests.Session()
    rows: list[dict[str, Any]] = []
    cursor = ""
    while True:
        params = {"category": "linear", "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        response = session.get(f"{base_url.rstrip('/')}/v5/market/instruments-info", params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if payload.get("retCode", 0) not in (0, "0"):
            raise RuntimeError(f"Bybit instruments-info failed: {payload.get('retMsg')}")
        result = payload.get("result", {})
        rows.extend(result.get("list", []))
        cursor = result.get("nextPageCursor") or ""
        if not cursor:
            break
    return {
        str(row.get("symbol", "")).upper(): row
        for row in rows
        if row.get("quoteCoin") == "USDT" and row.get("status") == "Trading"
    }


def fetch_top_symbols(
    *,
    base_url: str,
    limit: int,
    exclude_symbols: set[str],
) -> tuple[list[str], pd.DataFrame]:
    instruments = fetch_linear_usdt_instruments(base_url)
    response = requests.get(f"{base_url.rstrip('/')}/v5/market/tickers", params={"category": "linear"}, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if payload.get("retCode", 0) not in (0, "0"):
        raise RuntimeError(f"Bybit tickers failed: {payload.get('retMsg')}")

    rows: list[dict[str, Any]] = []
    for ticker in payload.get("result", {}).get("list", []):
        symbol = bybit_symbol(str(ticker.get("symbol", "")))
        if not symbol or symbol not in instruments or symbol in exclude_symbols:
            continue
        rows.append(
            {
                "symbol": symbol,
                "turnover24h": float(ticker.get("turnover24h") or 0.0),
                "volume24h": float(ticker.get("volume24h") or 0.0),
                "launch_time": instruments[symbol].get("launchTime"),
            }
        )
    table = pd.DataFrame(rows).sort_values("turnover24h", ascending=False).reset_index(drop=True)
    if limit > 0:
        table = table.head(limit).copy()
    return table["symbol"].tolist(), table


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def truthy(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def candidate_ok(row: pd.Series, *, min_validation: int, min_oos_net_r: float, min_all_net_r: float) -> bool:
    return (
        truthy(row.get("selection_pass", False))
        and truthy(row.get("oos_pass", False))
        and finite(row.get("oos_net_r")) >= min_oos_net_r
        and finite(row.get("all_net_r")) >= min_all_net_r
        and finite(row.get("validation_trades")) >= min_validation
        and finite(row.get("oos_trades")) >= min_validation
        and finite(row.get("optimization_score", row.get("robust_score"))) > 0.0
        and finite(row.get("local_pass_rate", 1.0), 1.0) >= 0.20
    )


def rank_score(row: pd.Series) -> float:
    lowpass_score = finite(row.get("optimization_score", row.get("robust_score")))
    stability = finite(row.get("stability_score", lowpass_score))
    pass_rate = finite(row.get("local_pass_rate", 0.0))
    oos_net_r = finite(row.get("oos_net_r"))
    val_net_r = finite(row.get("validation_net_r"))
    all_net_r = finite(row.get("all_net_r"))
    drawdown = abs(finite(row.get("all_max_dd_r")))
    return float(lowpass_score + 0.35 * stability + 0.25 * oos_net_r + 0.15 * val_net_r + 0.03 * all_net_r + 5.0 * pass_rate - 0.20 * drawdown)


def config_from_row(row: pd.Series) -> dict[str, Any]:
    fields = tuple(WolfeConfig.__dataclass_fields__)  # type: ignore[attr-defined]
    out: dict[str, Any] = {}
    for key in fields:
        if key not in row.index or pd.isna(row[key]):
            continue
        value = row[key]
        out[key] = value.item() if hasattr(value, "item") else value
    return out


def tune_params_from_config(config: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in BTC_TUNE_KEYS:
        if key in config and config[key] is not None:
            out[key] = config[key]
    if "pivot_method" not in out:
        out["pivot_method"] = config.get("pivot_method", "fractal")
    return out


def load_template_grid(path: Path | None, *, candidate_only: bool = True) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    if path.suffix.lower() == ".csv":
        table = pd.read_csv(path)
        if candidate_only and "candidate_ok" in table.columns:
            table = table[table["candidate_ok"].astype(str).str.lower().eq("true")].copy()
        for _, row in table.iterrows():
            raw = row.get("selected_config_json")
            if not isinstance(raw, str) or not raw.strip():
                continue
            try:
                config = json.loads(raw)
            except json.JSONDecodeError:
                continue
            rows.append(tune_params_from_config(config))
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            values = payload.values()
        elif isinstance(payload, list):
            values = payload
        else:
            values = []
        for value in values:
            if isinstance(value, dict):
                rows.append(tune_params_from_config(value))

    out: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        key = config_key(row)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def best_candidate(table: pd.DataFrame, *, min_validation: int, min_oos_net_r: float, min_all_net_r: float) -> pd.Series | None:
    if table.empty:
        return None
    out = table.copy()
    out["candidate_ok"] = out.apply(
        candidate_ok,
        axis=1,
        min_validation=min_validation,
        min_oos_net_r=min_oos_net_r,
        min_all_net_r=min_all_net_r,
    )
    out["rank_score"] = out.apply(rank_score, axis=1)
    passed = out[out["candidate_ok"]].copy()
    ranked = passed if not passed.empty else out
    ranked = ranked.sort_values(
        ["candidate_ok", "rank_score", "optimization_score", "oos_net_r", "validation_net_r"],
        ascending=[False, False, False, False, False],
        na_position="last",
    )
    return ranked.iloc[0]


def evaluate_symbol(task: dict[str, Any]) -> dict[str, Any]:
    apply_value_filters(
        task.get("pattern_tfs"),
        task.get("regime_filters"),
        enable_v2_grid=truthy(task.get("enable_v2_grid", False)),
    )
    symbol = bybit_symbol(task["symbol"])
    end = parse_utc_datetime(task["end"]) if task.get("end") else None
    history_ok, history_reason = has_min_daily_history(
        symbol,
        min_history_days=int(task["min_history_days"]),
        end=end,
        base_url=task["base_url"],
    )
    if not history_ok:
        return {
            "symbol": symbol,
            "status": "skipped",
            "skip_reason": history_reason,
            "candidate_ok": False,
        }

    frame = load_or_fetch_data(
        symbol,
        interval=task["interval"],
        days=int(task["days"]),
        end=end,
        cache_dir=Path(task["cache_dir"]),
        refresh=bool(task["refresh"]),
        base_url=task["base_url"],
    )
    if len(frame) < int(task["min_bars"]):
        return {
            "symbol": symbol,
            "status": "skipped",
            "skip_reason": f"only {len(frame)} bars available",
            "candidate_ok": False,
            "bars": int(len(frame)),
        }
    start = pd.Timestamp(frame["open_time"].iloc[0]).tz_convert("UTC")
    data_end = pd.Timestamp(frame["open_time"].iloc[-1]).tz_convert("UTC")
    mintick = fetch_bybit_mintick(symbol, base_url=task["base_url"])
    base_cfg = WolfeConfig.from_mapping(
        {
            "exec_tf": task["interval"],
            "mintick": mintick,
            "fee_aware_stop": True,
            "max_fee_to_price_risk": float(task["max_fee_to_price_risk"]),
            "fee_bps_side": float(task["fee_bps_side"]),
            "slippage_bps_side": float(task["slippage_bps_side"]),
            "risk_fraction": 0.01,
        }
    )

    grid: list[dict[str, Any]] = []
    max_configs = int(task["max_configs"])
    if max_configs > 0:
        grid.extend(btc_parameter_grid(max_configs, seed=int(task["random_seed"])))
    grid.extend(
        load_template_grid(
            Path(task["template_candidates"]) if task.get("template_candidates") else None,
            candidate_only=truthy(task.get("template_candidate_only", True)),
        )
    )
    deduped_grid: list[dict[str, Any]] = []
    seen_grid: set[tuple[Any, ...]] = set()
    for row in grid:
        key = config_key(row)
        if key in seen_grid:
            continue
        seen_grid.add(key)
        deduped_grid.append(row)
    if not deduped_grid:
        return {
            "symbol": symbol,
            "status": "skipped",
            "skip_reason": "no grid configs",
            "candidate_ok": False,
            "bars": int(len(frame)),
        }

    screen = evaluate_btc_grid(
        frame,
        base_cfg,
        symbol=symbol,
        grid=deduped_grid,
        min_train_trades=int(task["min_train"]),
        min_validation_trades=int(task["min_validation"]),
        lowpass=True,
        lowpass_radius=float(task["lowpass_radius"]),
        lowpass_min_neighbors=int(task["lowpass_min_neighbors"]),
        lowpass_outlier_penalty=float(task["lowpass_outlier_penalty"]),
    )
    symbol_dir = Path(task["output_dir"]) / "per_symbol"
    symbol_dir.mkdir(parents=True, exist_ok=True)
    screen_path = symbol_dir / f"{symbol.lower()}_screen.csv"
    screen.to_csv(screen_path, index=False)

    best_screen = best_candidate(
        screen,
        min_validation=int(task["min_validation"]),
        min_oos_net_r=float(task["min_oos_net_r"]),
        min_all_net_r=float(task["min_all_net_r"]),
    )
    final_table = screen
    seeds_path = ""
    refined_path = ""
    if truthy(task["refine"]) and best_screen is not None and (
        truthy(best_screen.get("candidate_ok", False)) or finite(best_screen.get("optimization_score")) >= float(task["refine_min_score"])
    ):
        refined, seeds = refine_btc(
            frame,
            base_cfg,
            symbol=symbol,
            scores=screen,
            max_seeds=int(task["refine_seeds"]),
            samples_per_seed=int(task["refine_samples_per_seed"]),
            neighbor_width=int(task["refine_neighbor_width"]),
            seed=int(task["random_seed"]) + 1000,
            min_train_trades=int(task["min_train"]),
            min_validation_trades=int(task["min_validation"]),
            lowpass=True,
            lowpass_radius=float(task["lowpass_radius"]),
            lowpass_min_neighbors=int(task["lowpass_min_neighbors"]),
            lowpass_outlier_penalty=float(task["lowpass_outlier_penalty"]),
        )
        refined_path = str(symbol_dir / f"{symbol.lower()}_refined.csv")
        seeds_path = str(symbol_dir / f"{symbol.lower()}_seeds.csv")
        refined.to_csv(refined_path, index=False)
        seeds.to_csv(seeds_path, index=False)
        final_table = pd.concat([screen.assign(stage="screen"), refined.assign(stage="refined")], ignore_index=True)

    selected = best_candidate(
        final_table,
        min_validation=int(task["min_validation"]),
        min_oos_net_r=float(task["min_oos_net_r"]),
        min_all_net_r=float(task["min_all_net_r"]),
    )
    if selected is None:
        return {
            "symbol": symbol,
            "status": "empty",
            "candidate_ok": False,
            "bars": int(len(frame)),
            "data_start": start.isoformat(),
            "data_end": data_end.isoformat(),
            "screen_path": str(screen_path),
        }

    selected = selected.copy()
    selected["candidate_ok"] = candidate_ok(
        selected,
        min_validation=int(task["min_validation"]),
        min_oos_net_r=float(task["min_oos_net_r"]),
        min_all_net_r=float(task["min_all_net_r"]),
    )
    selected["rank_score"] = rank_score(selected)
    selected_config = config_from_row(selected)
    return {
        "symbol": symbol,
        "status": "ok" if bool(selected["candidate_ok"]) else "miss",
        "candidate_ok": bool(selected["candidate_ok"]),
        "rank_score": float(selected["rank_score"]),
        "bars": int(len(frame)),
        "data_start": start.isoformat(),
        "data_end": data_end.isoformat(),
        "mintick": mintick,
        "screen_path": str(screen_path),
        "refined_path": refined_path,
        "seeds_path": seeds_path,
        "configs_tested": int(len(final_table)),
        "selected_config_json": json.dumps(selected_config, sort_keys=True, default=str),
        **{
            key: selected.get(key)
            for key in [
                "stage",
                "pattern_tf",
                "pivot_method",
                "pivot_source",
                "pivot_window",
                "pivot_confirm_window",
                "min_score",
                "min_rr",
                "max_hold_bars",
                "trend_filter",
                "regime_filter",
                "p1_horizontal_mode",
                "p1_horizontal_tolerance_atr",
                "p1_horizontal_max_distance_bars",
                "p4_contrary_mode",
                "p4_contrary_min_swing_atr",
                "min_v2_quality",
                "v2_score_weight",
                "allow_longs",
                "allow_shorts",
                "optimization_score",
                "stability_score",
                "local_pass_rate",
                "local_neighbor_count",
                "robust_score",
                "train_trades",
                "train_net_r",
                "train_avg_r",
                "train_profit_factor",
                "validation_trades",
                "validation_net_r",
                "validation_avg_r",
                "validation_profit_factor",
                "oos_trades",
                "oos_net_r",
                "oos_avg_r",
                "oos_profit_factor",
                "all_trades",
                "all_net_r",
                "all_avg_r",
                "all_profit_factor",
                "all_max_dd_r",
            ]
            if key in selected.index
        },
    }


def write_outputs(output_dir: Path, rows: list[dict[str, Any]], symbol_table: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    symbol_table.to_csv(output_dir / "top_symbols.csv", index=False)
    summary = pd.DataFrame(rows)
    if not summary.empty:
        if "candidate_ok" not in summary.columns:
            summary["candidate_ok"] = False
        if "rank_score" not in summary.columns:
            summary["rank_score"] = math.nan
        summary = summary.sort_values(["candidate_ok", "rank_score"], ascending=[False, False], na_position="last")
    summary.to_csv(output_dir / "candidate_retest.csv", index=False)
    configs: dict[str, Any] = {}
    if not summary.empty and "candidate_ok" in summary.columns and "selected_config_json" in summary.columns:
        for _, row in summary[summary["candidate_ok"].apply(truthy)].iterrows():
            selected_config = row.get("selected_config_json")
            if selected_config is None or pd.isna(selected_config):
                continue
            configs[str(row["symbol"])] = json.loads(str(selected_config))
    (output_dir / "selected_configs.json").write_text(
        json.dumps(configs, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a low-pass Wolfe Wave sweep over top Bybit USDT perps.")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--symbols", nargs="*", default=[])
    parser.add_argument("--only-symbols", action="store_true")
    parser.add_argument("--exclude-symbols", default=",".join(sorted(DEFAULT_EXCLUDED_SYMBOLS)))
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--days", type=int, default=900)
    parser.add_argument("--end")
    parser.add_argument("--min-history-days", type=int, default=365)
    parser.add_argument("--min-bars", type=int, default=25000)
    parser.add_argument("--cache-dir", type=Path, default=Path("scripts/data_wolfe_top100"))
    parser.add_argument("--output-dir", type=Path, default=Path("scripts/wolfe_wave_top100_lowpass"))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse completed rows and top-symbol snapshot from output-dir.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-configs", type=int, default=60)
    parser.add_argument("--template-candidates", type=Path)
    parser.add_argument("--template-all", action="store_true")
    parser.add_argument("--pattern-tfs", default="5m,15m,1h")
    parser.add_argument("--regime-filters", default="none,high_vol,low_vol,mean_reversion")
    parser.add_argument("--enable-v2-grid", action="store_true", help="Include Wolfe v2 structural validation/scoring params.")
    parser.add_argument("--min-train", type=int, default=20)
    parser.add_argument("--min-validation", type=int, default=8)
    parser.add_argument("--min-oos-net-r", type=float, default=1.0)
    parser.add_argument("--min-all-net-r", type=float, default=8.0)
    parser.add_argument("--fee-bps-side", type=float, default=5.5)
    parser.add_argument("--slippage-bps-side", type=float, default=1.0)
    parser.add_argument("--max-fee-to-price-risk", type=float, default=0.25)
    parser.add_argument("--lowpass-radius", type=float, default=0.45)
    parser.add_argument("--lowpass-min-neighbors", type=int, default=9)
    parser.add_argument("--lowpass-outlier-penalty", type=float, default=0.65)
    parser.add_argument("--refine", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--refine-min-score", type=float, default=0.0)
    parser.add_argument("--refine-seeds", type=int, default=6)
    parser.add_argument("--refine-samples-per-seed", type=int, default=36)
    parser.add_argument("--refine-neighbor-width", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=20260601)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.interval = normalize_timeframe(args.interval)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    top_symbols_path = args.output_dir / "top_symbols.csv"
    if args.resume and top_symbols_path.exists() and not args.only_symbols:
        symbol_table = pd.read_csv(top_symbols_path)
        symbols = [bybit_symbol(symbol) for symbol in symbol_table["symbol"].astype(str).tolist()]
        if args.symbols:
            seen = set(symbols)
            for raw in args.symbols:
                symbol = bybit_symbol(raw)
                if symbol and symbol not in seen:
                    seen.add(symbol)
                    symbols.append(symbol)
                    symbol_table = pd.concat(
                        [symbol_table, pd.DataFrame([{"symbol": symbol, "turnover24h": None, "volume24h": None}])],
                        ignore_index=True,
                    )
    elif args.only_symbols:
        symbols = [bybit_symbol(symbol) for symbol in args.symbols]
        symbol_table = pd.DataFrame({"symbol": symbols})
    else:
        exclude = {bybit_symbol(symbol) for symbol in parse_csv_values(args.exclude_symbols)}
        symbols, symbol_table = fetch_top_symbols(base_url=args.base_url, limit=args.limit, exclude_symbols=exclude)
        if args.symbols:
            seen = set(symbols)
            for raw in args.symbols:
                symbol = bybit_symbol(raw)
                if symbol and symbol not in seen:
                    seen.add(symbol)
                    symbols.append(symbol)

    apply_value_filters(args.pattern_tfs, args.regime_filters, enable_v2_grid=args.enable_v2_grid)
    print(
        f"Top Wolfe low-pass universe symbols={len(symbols)} days={args.days} "
        f"max_configs={args.max_configs} refine={args.refine} v2={args.enable_v2_grid} workers={args.workers}",
        flush=True,
    )
    print(",".join(symbols), flush=True)

    common = {
        "interval": args.interval,
        "days": args.days,
        "end": args.end,
        "min_history_days": args.min_history_days,
        "min_bars": args.min_bars,
        "cache_dir": str(args.cache_dir),
        "output_dir": str(args.output_dir),
        "base_url": args.base_url,
        "refresh": args.refresh,
        "max_configs": args.max_configs,
        "template_candidates": str(args.template_candidates) if args.template_candidates else "",
        "template_candidate_only": not args.template_all,
        "pattern_tfs": args.pattern_tfs,
        "regime_filters": args.regime_filters,
        "enable_v2_grid": args.enable_v2_grid,
        "min_train": args.min_train,
        "min_validation": args.min_validation,
        "min_oos_net_r": args.min_oos_net_r,
        "min_all_net_r": args.min_all_net_r,
        "fee_bps_side": args.fee_bps_side,
        "slippage_bps_side": args.slippage_bps_side,
        "max_fee_to_price_risk": args.max_fee_to_price_risk,
        "lowpass_radius": args.lowpass_radius,
        "lowpass_min_neighbors": args.lowpass_min_neighbors,
        "lowpass_outlier_penalty": args.lowpass_outlier_penalty,
        "refine": args.refine,
        "refine_min_score": args.refine_min_score,
        "refine_seeds": args.refine_seeds,
        "refine_samples_per_seed": args.refine_samples_per_seed,
        "refine_neighbor_width": args.refine_neighbor_width,
        "random_seed": args.random_seed,
    }
    rows: list[dict[str, Any]] = []
    completed_symbols: set[str] = set()
    summary_path = args.output_dir / "candidate_retest.csv"
    if args.resume and summary_path.exists():
        previous = pd.read_csv(summary_path)
        if "symbol" in previous.columns:
            rows = previous.to_dict("records")
            completed_symbols = {bybit_symbol(symbol) for symbol in previous["symbol"].dropna().astype(str)}
            print(f"Resume: loaded {len(rows)} completed rows from {summary_path}", flush=True)

    tasks = [{**common, "symbol": symbol} for symbol in symbols if symbol not in completed_symbols]
    if not tasks:
        write_outputs(args.output_dir, rows, symbol_table)
        print(f"DONE pass={sum(1 for row in rows if row.get('candidate_ok'))}/{len(rows)} skipped=0 errors=0 elapsed=0.0s output={args.output_dir}", flush=True)
        return
    started = datetime.now(timezone.utc)

    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(evaluate_symbol, task): task["symbol"] for task in tasks}
        for idx, future in enumerate(as_completed(futures), start=1):
            symbol = futures[future]
            try:
                row = future.result()
            except Exception as exc:  # noqa: BLE001
                row = {"symbol": symbol, "status": "error", "error": f"{type(exc).__name__}: {exc}", "candidate_ok": False}
            rows.append(row)
            write_outputs(args.output_dir, rows, symbol_table)
            if row.get("candidate_ok"):
                print(
                    f"[{len(rows)}/{len(symbols)}] PASS {symbol}: rank={finite(row.get('rank_score')):.2f} "
                    f"all={finite(row.get('all_net_r')):+.2f}R val={finite(row.get('validation_net_r')):+.2f}R "
                    f"oos={finite(row.get('oos_net_r')):+.2f}R trades={finite(row.get('all_trades')):.0f}",
                    flush=True,
                )
            elif row.get("status") == "skipped":
                print(f"[{len(rows)}/{len(symbols)}] SKIP {symbol}: {row.get('skip_reason')}", flush=True)
            elif row.get("status") == "error":
                print(f"[{len(rows)}/{len(symbols)}] ERROR {symbol}: {row.get('error')}", flush=True)
            else:
                print(
                    f"[{len(rows)}/{len(symbols)}] MISS {symbol}: best_rank={finite(row.get('rank_score')):.2f} "
                    f"all={finite(row.get('all_net_r')):+.2f}R val={finite(row.get('validation_net_r')):+.2f}R "
                    f"oos={finite(row.get('oos_net_r')):+.2f}R",
                    flush=True,
                )

    write_outputs(args.output_dir, rows, symbol_table)
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    passed = sum(1 for row in rows if row.get("candidate_ok"))
    skipped = sum(1 for row in rows if row.get("status") == "skipped")
    errors = sum(1 for row in rows if row.get("status") == "error")
    print(
        f"DONE pass={passed}/{len(rows)} skipped={skipped} errors={errors} elapsed={elapsed:.1f}s "
        f"output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
