from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import pickle
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backtest_pyharmonics_strategy import (  # noqa: E402
    HarmonicConfig,
    add_lowpass_scores,
    candle_pattern_summary,
    config_metrics,
    event_filter_matches,
    find_harmonic_events,
    parse_csv_values,
    run_backtest,
)
from scripts.backtest_wolfe_wave import (  # noqa: E402
    bybit_symbol,
    normalize_timeframe,
    parse_utc_datetime,
    resample_ohlc,
)
from scripts.run_wolfe_wave_top100_lowpass import (  # noqa: E402
    DEFAULT_EXCLUDED_SYMBOLS,
    fetch_top_symbols,
    finite,
    truthy,
)
from scripts.tune_wolfe_wave_universe import (  # noqa: E402
    DEFAULT_BASE_URL,
    has_min_daily_history,
    load_or_fetch_data,
    split_bounds,
)


CONFIG_KEY_FIELDS = (
    "pattern_tf",
    "family",
    "pattern_mode",
    "peak_spacing",
    "fib_tolerance",
    "forming_percent_c_to_d",
    "pattern_lookback_bars",
    "pattern_step_bars",
    "search_limit_to",
    "confirm_bars",
    "entry_window_bars",
    "entry_mode",
    "prz_atr_buffer",
    "candle_filter",
    "trigger_candle_filter",
    "pattern_name_filter",
    "direction_filter",
    "stop_atr_buffer",
    "rr",
    "breakeven_trigger_r",
    "min_harmonic_quality_score",
    "max_hold_bars",
    "trend_filter",
    "time_filter",
    "htf_filter",
    "htf_stretch_atr",
    "htf_rsi_extreme",
)


def config_key(config: HarmonicConfig | dict[str, Any]) -> tuple[Any, ...]:
    data = asdict(config) if isinstance(config, HarmonicConfig) else dict(config)
    return tuple(data.get(field) for field in CONFIG_KEY_FIELDS)


def parse_optional_symbols(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        for chunk in str(raw or "").split(","):
            symbol = bybit_symbol(chunk)
            if symbol and symbol not in seen:
                seen.add(symbol)
                out.append(symbol)
    return out


def split_csv_symbols(raw: str | None) -> set[str]:
    return {bybit_symbol(item) for item in parse_csv_values(raw or "", str)}


def stable_sample(items: list[HarmonicConfig], *, count: int, seed: int) -> list[HarmonicConfig]:
    if count <= 0 or len(items) <= count:
        return items
    rng = random.Random(seed)
    return [items[index] for index in sorted(rng.sample(range(len(items)), count))]


def load_template_configs(path: Path | None) -> list[HarmonicConfig]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    if path.suffix.lower() == ".csv":
        table = pd.read_csv(path)
        if "selected_config_json" in table.columns:
            for raw in table["selected_config_json"].dropna().astype(str):
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
        else:
            rows.extend(table.to_dict("records"))
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            rows.extend(value for value in payload.values() if isinstance(value, dict))
        elif isinstance(payload, list):
            rows.extend(value for value in payload if isinstance(value, dict))

    out: list[HarmonicConfig] = []
    for row in rows:
        fields = {key: row[key] for key in HarmonicConfig.__dataclass_fields__ if key in row and pd.notna(row[key])}  # type: ignore[attr-defined]
        if fields:
            out.append(HarmonicConfig(**fields))
    return out


def grid_dimensions(args: argparse.Namespace) -> list[list[Any]]:
    forming_percents = parse_csv_values(args.forming_percents, float)
    heads: list[tuple[str, str, str, float]] = []
    for pattern_tf, family, pattern_mode in itertools.product(
        parse_csv_values(args.pattern_tfs, str),
        parse_csv_values(args.families, str),
        parse_csv_values(args.pattern_modes, str),
    ):
        pattern_mode = pattern_mode.strip().lower()
        mode_forming_percents = forming_percents
        if bool(getattr(args, "collapse_formed_forming_percents", False)) and pattern_mode == "formed":
            mode_forming_percents = [forming_percents[0] if forming_percents else 0.8]
        for forming_percent in mode_forming_percents:
            heads.append((pattern_tf, family, pattern_mode, forming_percent))
    return [
        heads,
        parse_csv_values(args.peak_spacings, int),
        parse_csv_values(args.fib_tolerances, float),
        parse_csv_values(args.confirm_bars, int),
        parse_csv_values(args.entry_window_bars, int),
        parse_csv_values(args.entry_modes, str),
        parse_csv_values(args.prz_buffers, float),
        parse_csv_values(args.candle_filters, str),
        parse_csv_values(args.trigger_candle_filters, str),
        parse_csv_values(args.pattern_name_filters, str),
        parse_csv_values(args.direction_filters, str),
        parse_csv_values(args.stop_buffers, float),
        parse_csv_values(args.rrs, float),
        parse_csv_values(args.breakeven_triggers, float),
        parse_csv_values(args.min_harmonic_quality_scores, float),
        parse_csv_values(args.max_hold_bars, int),
        parse_csv_values(args.trend_filters, str),
        parse_csv_values(args.time_filters, str),
        parse_csv_values(args.htf_filters, str),
        parse_csv_values(args.htf_stretch_atrs, float),
        parse_csv_values(args.htf_rsi_extremes, float),
    ]


def grid_size(dimensions: list[list[Any]]) -> int:
    total = 1
    for dimension in dimensions:
        total *= len(dimension)
    return total


def product_values_at(dimensions: list[list[Any]], index: int) -> list[Any]:
    values: list[Any] = []
    for dimension in reversed(dimensions):
        values.append(dimension[index % len(dimension)])
        index //= len(dimension)
    return list(reversed(values))


def config_from_grid_values(values: list[Any], args: argparse.Namespace) -> HarmonicConfig:
    (
        head,
        peak_spacing,
        fib_tolerance,
        confirm_bars,
        entry_window,
        entry_mode,
        prz_buffer,
        candle_filter,
        trigger_candle_filter,
        pattern_name_filter,
        direction_filter,
        stop_buffer,
        rr,
        breakeven_trigger,
        min_harmonic_quality_score,
        max_hold,
        trend_filter,
        time_filter,
        htf_filter,
        htf_stretch_atr,
        htf_rsi_extreme,
    ) = values
    pattern_tf, family, pattern_mode, forming_percent = head
    return HarmonicConfig(
        pattern_tf=normalize_timeframe(pattern_tf),
        family=family.strip().upper(),
        pattern_mode=pattern_mode,
        peak_spacing=int(peak_spacing),
        fib_tolerance=float(fib_tolerance),
        forming_percent_c_to_d=float(forming_percent),
        pattern_lookback_bars=int(args.pattern_lookback_bars),
        pattern_step_bars=int(args.pattern_step_bars),
        search_limit_to=int(args.search_limit_to),
        confirm_bars=int(confirm_bars),
        entry_window_bars=int(entry_window),
        entry_mode=entry_mode.strip().lower(),
        prz_atr_buffer=float(prz_buffer),
        candle_filter=candle_filter.strip().lower(),
        trigger_candle_filter=trigger_candle_filter.strip().lower(),
        pattern_name_filter=pattern_name_filter.strip().upper(),
        direction_filter=direction_filter.strip().lower(),
        stop_atr_buffer=float(stop_buffer),
        rr=float(rr),
        breakeven_trigger_r=float(breakeven_trigger),
        min_harmonic_quality_score=float(min_harmonic_quality_score),
        max_hold_bars=int(max_hold),
        trend_filter=str(trend_filter).strip().lower(),
        time_filter=str(time_filter).strip().lower(),
        htf_filter=str(htf_filter).strip().lower(),
        htf_stretch_atr=float(htf_stretch_atr),
        htf_rsi_extreme=float(htf_rsi_extreme),
        fee_bps_side=float(args.fee_bps_side),
        slippage_bps_side=float(args.slippage_bps_side),
        max_fee_to_price_risk=float(args.max_fee_to_price_risk),
        min_entry_risk_pct=float(args.min_entry_risk_pct),
        risk_fraction=float(args.risk_fraction),
        one_trade_at_a_time=not bool(args.allow_overlap),
    )


def iter_grid_configs(args: argparse.Namespace) -> Any:
    dimensions = grid_dimensions(args)
    for values in itertools.product(*dimensions):
        yield config_from_grid_values(list(values), args)


def build_grid(args: argparse.Namespace) -> list[HarmonicConfig]:
    dimensions = grid_dimensions(args)
    total = grid_size(dimensions)
    count = int(args.max_configs)
    if count > 0 and total > count:
        rng = random.Random(int(args.random_seed))
        selected = [
            config_from_grid_values(product_values_at(dimensions, index), args)
            for index in sorted(rng.sample(range(total), count))
        ]
    else:
        selected = [config_from_grid_values(list(values), args) for values in itertools.product(*dimensions)]
    selected.extend(load_template_configs(Path(args.template_configs) if args.template_configs else None))

    deduped: list[HarmonicConfig] = []
    seen: set[tuple[Any, ...]] = set()
    for cfg in selected:
        key = config_key(cfg)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cfg)
    return deduped


def candidate_ok(
    row: pd.Series,
    *,
    min_validation: int,
    min_lowpass_score: float,
    min_oos_net_r: float,
    min_validation_net_r: float,
    min_all_net_r: float,
    min_avg_r: float,
    max_all_dd_r: float,
) -> bool:
    drawdown = abs(finite(row.get("all_max_dd_r")))
    return (
        finite(row.get("lowpass_robust_score", row.get("robust_score"))) >= min_lowpass_score
        and finite(row.get("oos_net_r")) >= min_oos_net_r
        and finite(row.get("validation_net_r")) >= min_validation_net_r
        and finite(row.get("all_net_r")) >= min_all_net_r
        and finite(row.get("all_avg_r")) >= min_avg_r
        and finite(row.get("validation_trades")) >= min_validation
        and finite(row.get("oos_trades")) >= min_validation
        and (max_all_dd_r <= 0.0 or drawdown <= max_all_dd_r)
    )


def rank_score(row: pd.Series) -> float:
    lowpass = finite(row.get("lowpass_robust_score", row.get("robust_score")))
    raw = finite(row.get("robust_score"))
    oos = finite(row.get("oos_net_r"))
    validation = finite(row.get("validation_net_r"))
    all_net = finite(row.get("all_net_r"))
    avg = finite(row.get("all_avg_r"))
    drawdown = abs(finite(row.get("all_max_dd_r")))
    trades = finite(row.get("all_trades"))
    return float(lowpass + 0.25 * raw + 0.25 * oos + 0.15 * validation + 0.03 * all_net + 8.0 * avg + min(math.log1p(trades), 5.0) * 0.2 - 0.15 * drawdown)


def best_candidate(table: pd.DataFrame, task: dict[str, Any]) -> pd.Series | None:
    if table.empty:
        return None
    out = table.copy()
    out["candidate_ok"] = out.apply(
        candidate_ok,
        axis=1,
        min_validation=int(task["min_validation"]),
        min_lowpass_score=float(task["min_lowpass_score"]),
        min_oos_net_r=float(task["min_oos_net_r"]),
        min_validation_net_r=float(task["min_validation_net_r"]),
        min_all_net_r=float(task["min_all_net_r"]),
        min_avg_r=float(task["min_avg_r"]),
        max_all_dd_r=float(task["max_all_dd_r"]),
    )
    out["rank_score"] = out.apply(rank_score, axis=1)
    ranked = out.sort_values(
        ["candidate_ok", "rank_score", "lowpass_robust_score", "oos_net_r", "validation_net_r"],
        ascending=[False, False, False, False, False],
        na_position="last",
    )
    return ranked.iloc[0]


def config_from_row(row: pd.Series) -> HarmonicConfig:
    fields: dict[str, Any] = {}
    for key in HarmonicConfig.__dataclass_fields__:  # type: ignore[attr-defined]
        if key not in row.index or pd.isna(row[key]):
            continue
        value = row[key]
        fields[key] = value.item() if hasattr(value, "item") else value
    return HarmonicConfig(**fields)


def short_hash(payload: Any) -> int:
    raw = json.dumps(payload, sort_keys=True, default=str)
    return int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8], 16)


def family_filter(raw: str) -> set[str]:
    return {item.strip().upper() for item in str(raw or "").split("+") if item.strip()}


def event_cache_file(
    *,
    symbol_dir: Path,
    symbol: str,
    cache_key: tuple[Any, ...],
    cache_family: str,
    frame: pd.DataFrame,
) -> Path:
    start = pd.Timestamp(frame["open_time"].iloc[0]).tz_convert("UTC").isoformat()
    end = pd.Timestamp(frame["open_time"].iloc[-1]).tz_convert("UTC").isoformat()
    payload = {
        "symbol": symbol,
        "cache_key": cache_key,
        "cache_family": cache_family,
        "bars": int(len(frame)),
        "start": start,
        "end": end,
    }
    return symbol_dir / "event_cache" / f"{symbol.lower()}_{short_hash(payload):08x}.pkl"


def load_event_cache(path: Path) -> list[Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            events = pickle.load(handle)
    except Exception:
        return None
    return events if isinstance(events, list) else None


def save_event_cache(path: Path, events: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump(events, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)


def evaluate_symbol(task: dict[str, Any]) -> dict[str, Any]:
    symbol = bybit_symbol(task["symbol"])
    end = parse_utc_datetime(task["end"]) if task.get("end") else None
    history_ok, history_reason = has_min_daily_history(
        symbol,
        min_history_days=int(task["min_history_days"]),
        end=end,
        base_url=task["base_url"],
    )
    if not history_ok:
        return {"symbol": symbol, "status": "skipped", "skip_reason": history_reason, "candidate_ok": False}

    frame = load_or_fetch_data(
        symbol,
        interval=task["exec_tf"],
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

    train_end, validation_end = split_bounds(
        frame,
        validation_days=int(task["validation_days"]),
        oos_days=int(task["oos_days"]),
    )
    configs = [HarmonicConfig(**payload) for payload in task["configs"]]
    cache_family = "+".join(sorted({token for cfg in configs for token in family_filter(cfg.family)}))
    progress_every = int(task.get("progress_every_configs") or 0)
    event_progress_every = int(task.get("event_progress_every_chunks") or 0)
    persist_event_cache = bool(task.get("persist_event_cache"))
    if progress_every:
        print(f"[{symbol}] START bars={len(frame)} configs={len(configs)}", flush=True)
    rows: list[dict[str, Any]] = []
    event_cache: dict[tuple[Any, ...], list[Any]] = {}
    symbol_dir = Path(task["output_dir"]) / "per_symbol"
    symbol_dir.mkdir(parents=True, exist_ok=True)

    for number, cfg in enumerate(configs, start=1):
        cache_key = (
            cfg.pattern_tf,
            cfg.pattern_mode,
            cfg.peak_spacing,
            cfg.fib_tolerance,
            cfg.forming_percent_c_to_d,
            cfg.pattern_lookback_bars,
            cfg.pattern_step_bars,
            cfg.search_limit_to,
        )
        if cache_key not in event_cache:
            pattern_df = resample_ohlc(frame, cfg.pattern_tf)
            event_cfg = replace(cfg, family=cache_family or cfg.family)
            disk_path = event_cache_file(
                symbol_dir=symbol_dir,
                symbol=symbol,
                cache_key=cache_key,
                cache_family=cache_family,
                frame=frame,
            )
            cached_events = load_event_cache(disk_path) if persist_event_cache else None
            if cached_events is not None:
                event_cache[cache_key] = cached_events
                if progress_every:
                    print(f"[{symbol}] event cache hit key={len(event_cache)} events={len(cached_events)}", flush=True)
            else:
                if progress_every:
                    print(
                        f"[{symbol}] event scan key={len(event_cache) + 1} tf={cfg.pattern_tf} "
                        f"mode={cfg.pattern_mode} peak={cfg.peak_spacing} fib={cfg.fib_tolerance} "
                        f"forming={cfg.forming_percent_c_to_d}",
                        flush=True,
                    )
                found_events = find_harmonic_events(
                    pattern_df,
                    event_cfg,
                    symbol=symbol,
                    progress_label=f"[{symbol}]",
                    progress_every_chunks=event_progress_every,
                )
                if persist_event_cache:
                    save_event_cache(disk_path, found_events)
                event_cache[cache_key] = found_events
        requested_families = family_filter(cfg.family)
        events = [event for event in event_cache[cache_key] if not requested_families or event.family in requested_families]
        filtered_events = [event for event in events if event_filter_matches(event, cfg)]
        trades = run_backtest(frame, cfg, symbol=symbol, precomputed_events=filtered_events)
        metrics = config_metrics(trades, train_end=train_end, validation_end=validation_end)
        rows.append(
            {
                "symbol": symbol,
                "config_number": number,
                **asdict(cfg),
                "pattern_events": len(filtered_events),
                **metrics,
            }
        )
        if progress_every and (number == 1 or number % progress_every == 0 or number == len(configs)):
            print(
                f"[{symbol}] config {number}/{len(configs)} cache_keys={len(event_cache)} "
                f"last_events={len(filtered_events)} last_trades={len(trades)}",
                flush=True,
            )

    table = pd.DataFrame(rows)
    table = add_lowpass_scores(
        table,
        radius=float(task["lowpass_radius"]),
        min_neighbors=int(task["lowpass_min_neighbors"]),
    )
    table = table.sort_values(
        ["lowpass_robust_score", "robust_score", "oos_net_r", "validation_net_r"],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    tuning_path = symbol_dir / f"{symbol.lower()}_pyharmonics_tuning.csv"
    table.to_csv(tuning_path, index=False)

    selected = best_candidate(table, task)
    if selected is None:
        return {
            "symbol": symbol,
            "status": "empty",
            "candidate_ok": False,
            "bars": int(len(frame)),
            "data_start": pd.Timestamp(frame["open_time"].iloc[0]).tz_convert("UTC").isoformat(),
            "data_end": pd.Timestamp(frame["open_time"].iloc[-1]).tz_convert("UTC").isoformat(),
            "tuning_path": str(tuning_path),
        }

    selected_cfg = config_from_row(selected)
    selected_cache_key = (
        selected_cfg.pattern_tf,
        selected_cfg.pattern_mode,
        selected_cfg.peak_spacing,
        selected_cfg.fib_tolerance,
        selected_cfg.forming_percent_c_to_d,
        selected_cfg.pattern_lookback_bars,
        selected_cfg.pattern_step_bars,
        selected_cfg.search_limit_to,
    )
    selected_events_all = event_cache.get(selected_cache_key)
    if selected_events_all is None:
        selected_events_all = find_harmonic_events(
            resample_ohlc(frame, selected_cfg.pattern_tf),
            selected_cfg,
            symbol=symbol,
            progress_label=f"[{symbol}] selected",
            progress_every_chunks=event_progress_every,
        )
    selected_families = family_filter(selected_cfg.family)
    selected_events = [
        event for event in selected_events_all if not selected_families or event.family in selected_families
    ]
    selected_events = [event for event in selected_events if event_filter_matches(event, selected_cfg)]
    selected_trades = run_backtest(frame, selected_cfg, symbol=symbol, precomputed_events=selected_events)
    trades_path = symbol_dir / f"{symbol.lower()}_pyharmonics_selected_trades.csv"
    candle_path = symbol_dir / f"{symbol.lower()}_pyharmonics_candle_patterns.csv"
    selected_trades.to_csv(trades_path, index=False)
    candle_pattern_summary(selected_trades).to_csv(candle_path, index=False)

    selected_ok = candidate_ok(
        selected,
        min_validation=int(task["min_validation"]),
        min_lowpass_score=float(task["min_lowpass_score"]),
        min_oos_net_r=float(task["min_oos_net_r"]),
        min_validation_net_r=float(task["min_validation_net_r"]),
        min_all_net_r=float(task["min_all_net_r"]),
        min_avg_r=float(task["min_avg_r"]),
        max_all_dd_r=float(task["max_all_dd_r"]),
    )
    selected_config = asdict(selected_cfg)
    selected_dict = selected.to_dict()
    return {
        "symbol": symbol,
        "status": "ok" if selected_ok else "miss",
        "candidate_ok": bool(selected_ok),
        "rank_score": rank_score(selected),
        "bars": int(len(frame)),
        "data_start": pd.Timestamp(frame["open_time"].iloc[0]).tz_convert("UTC").isoformat(),
        "data_end": pd.Timestamp(frame["open_time"].iloc[-1]).tz_convert("UTC").isoformat(),
        "train_end": pd.Timestamp(train_end).tz_convert("UTC").isoformat(),
        "validation_end": pd.Timestamp(validation_end).tz_convert("UTC").isoformat(),
        "configs_tested": int(len(table)),
        "tuning_path": str(tuning_path),
        "trades_path": str(trades_path),
        "candle_summary_path": str(candle_path),
        "selected_config_json": json.dumps(selected_config, sort_keys=True, default=str),
        "selected_config_hash": short_hash(selected_config),
        **{
            key: selected_dict.get(key)
            for key in [
                *CONFIG_KEY_FIELDS,
                "pattern_events",
                "lowpass_neighbors",
                "lowpass_robust_score",
                "robust_score",
                "train_trades",
                "train_net_r",
                "train_avg_r",
                "validation_trades",
                "validation_net_r",
                "validation_avg_r",
                "oos_trades",
                "oos_net_r",
                "oos_avg_r",
                "all_trades",
                "all_net_r",
                "all_avg_r",
                "all_profit_factor",
                "all_max_dd_r",
            ]
            if key in selected_dict
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
            summary["rank_score"] = float("-inf")
        summary["rank_score"] = pd.to_numeric(summary["rank_score"], errors="coerce").fillna(float("-inf"))
        summary = summary.sort_values(["candidate_ok", "rank_score"], ascending=[False, False], na_position="last")
    summary.to_csv(output_dir / "candidate_retest.csv", index=False)
    configs: dict[str, Any] = {}
    if not summary.empty and {"candidate_ok", "selected_config_json", "symbol"}.issubset(summary.columns):
        for _, row in summary[summary["candidate_ok"].astype(bool)].iterrows():
            try:
                configs[str(row["symbol"])] = json.loads(str(row["selected_config_json"]))
            except json.JSONDecodeError:
                continue
    (output_dir / "selected_configs.json").write_text(
        json.dumps(configs, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    trade_frames: list[pd.DataFrame] = []
    if not summary.empty and "trades_path" in summary.columns:
        for raw_path in summary["trades_path"].dropna().astype(str):
            path = Path(raw_path)
            if not path.exists():
                continue
            try:
                frame = pd.read_csv(path)
            except Exception:
                continue
            if not frame.empty:
                trade_frames.append(frame)
    combined = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    candle_pattern_summary(combined).to_csv(output_dir / "candle_pattern_summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a low-pass pyharmonics sweep over top Bybit USDT perps.")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--symbols", nargs="*", default=[])
    parser.add_argument("--only-symbols", action="store_true")
    parser.add_argument("--exclude-symbols", default=",".join(sorted(DEFAULT_EXCLUDED_SYMBOLS)))
    parser.add_argument("--exec-tf", default="15m")
    parser.add_argument("--days", type=int, default=1825)
    parser.add_argument("--end")
    parser.add_argument("--validation-days", type=int, default=365)
    parser.add_argument("--oos-days", type=int, default=365)
    parser.add_argument("--min-history-days", type=int, default=365)
    parser.add_argument("--min-bars", type=int, default=25000)
    parser.add_argument("--cache-dir", type=Path, default=Path("scripts/data_pyharmonics_top100"))
    parser.add_argument("--output-dir", type=Path, default=Path("scripts/pyharmonics_top100_lowpass"))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-configs", type=int, default=360)
    parser.add_argument("--template-configs", type=Path)
    parser.add_argument("--pattern-tfs", default="1h,4h")
    parser.add_argument("--families", default="ABC,ABCD,XABCD")
    parser.add_argument("--pattern-modes", default="formed,forming")
    parser.add_argument("--peak-spacings", default="10,20,28")
    parser.add_argument("--fib-tolerances", default="0.02,0.03,0.05")
    parser.add_argument("--forming-percents", default="0.70,0.80,0.90")
    parser.add_argument("--collapse-formed-forming-percents", action="store_true")
    parser.add_argument("--pattern-lookback-bars", type=int, default=800)
    parser.add_argument("--pattern-step-bars", type=int, default=200)
    parser.add_argument("--search-limit-to", type=int, default=8)
    parser.add_argument("--confirm-bars", default="10,20,30")
    parser.add_argument("--entry-window-bars", default="24,48")
    parser.add_argument("--entry-modes", default="next_open")
    parser.add_argument("--prz-buffers", default="0.10,0.25")
    parser.add_argument("--candle-filters", default="none,any_reversal,engulfing,pinbar,reclaim,strong_close")
    parser.add_argument("--trigger-candle-filters", default="all")
    parser.add_argument("--pattern-name-filters", default="all")
    parser.add_argument("--direction-filters", default="both")
    parser.add_argument("--stop-buffers", default="0.2,0.5,0.8")
    parser.add_argument("--rrs", default="1.25,1.5,2.0")
    parser.add_argument("--breakeven-triggers", default="0")
    parser.add_argument("--min-harmonic-quality-scores", default="0")
    parser.add_argument("--max-hold-bars", default="48,96,192")
    parser.add_argument("--trend-filters", default="none,with_ema,counter_ema")
    parser.add_argument("--time-filters", default="all")
    parser.add_argument("--htf-filters", default="none")
    parser.add_argument("--htf-stretch-atrs", default="0.75")
    parser.add_argument("--htf-rsi-extremes", default="55")
    parser.add_argument("--fee-bps-side", type=float, default=5.5)
    parser.add_argument("--slippage-bps-side", type=float, default=1.0)
    parser.add_argument("--max-fee-to-price-risk", type=float, default=0.25)
    parser.add_argument("--min-entry-risk-pct", type=float, default=0.001)
    parser.add_argument("--risk-fraction", type=float, default=0.01)
    parser.add_argument("--allow-overlap", action="store_true")
    parser.add_argument("--lowpass-radius", type=float, default=1.60)
    parser.add_argument("--lowpass-min-neighbors", type=int, default=7)
    parser.add_argument("--min-validation", type=int, default=12)
    parser.add_argument("--min-lowpass-score", type=float, default=0.0)
    parser.add_argument("--min-oos-net-r", type=float, default=1.0)
    parser.add_argument("--min-validation-net-r", type=float, default=0.0)
    parser.add_argument("--min-all-net-r", type=float, default=5.0)
    parser.add_argument("--min-avg-r", type=float, default=0.0)
    parser.add_argument("--max-all-dd-r", type=float, default=0.0)
    parser.add_argument("--random-seed", type=int, default=20260602)
    parser.add_argument("--progress-every-configs", type=int, default=0)
    parser.add_argument("--event-progress-every-chunks", type=int, default=0)
    parser.add_argument("--persist-event-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.exec_tf = normalize_timeframe(args.exec_tf)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configs = build_grid(args)
    if not configs:
        raise SystemExit("No pyharmonics configs selected.")
    config_payloads = [asdict(cfg) for cfg in configs]

    if args.resume and (args.output_dir / "top_symbols.csv").exists() and not args.only_symbols:
        symbol_table = pd.read_csv(args.output_dir / "top_symbols.csv")
        symbols = [bybit_symbol(symbol) for symbol in symbol_table["symbol"].astype(str).tolist()]
    elif args.only_symbols:
        symbols = parse_optional_symbols(args.symbols)
        symbol_table = pd.DataFrame({"symbol": symbols})
    else:
        exclude = split_csv_symbols(args.exclude_symbols)
        symbols, symbol_table = fetch_top_symbols(base_url=args.base_url, limit=args.limit, exclude_symbols=exclude)
        extra = parse_optional_symbols(args.symbols)
        if extra:
            seen = set(symbols)
            for symbol in extra:
                if symbol not in seen:
                    seen.add(symbol)
                    symbols.append(symbol)
                    symbol_table = pd.concat(
                        [symbol_table, pd.DataFrame([{"symbol": symbol, "turnover24h": None, "volume24h": None}])],
                        ignore_index=True,
                    )

    print(
        f"Top pyharmonics low-pass universe symbols={len(symbols)} configs={len(configs)} "
        f"days={args.days} exec_tf={args.exec_tf} workers={args.workers}",
        flush=True,
    )
    print(",".join(symbols), flush=True)

    common = {
        "exec_tf": args.exec_tf,
        "days": args.days,
        "end": args.end,
        "validation_days": args.validation_days,
        "oos_days": args.oos_days,
        "min_history_days": args.min_history_days,
        "min_bars": args.min_bars,
        "cache_dir": str(args.cache_dir),
        "output_dir": str(args.output_dir),
        "base_url": args.base_url,
        "refresh": args.refresh,
        "configs": config_payloads,
        "lowpass_radius": args.lowpass_radius,
        "lowpass_min_neighbors": args.lowpass_min_neighbors,
        "min_validation": args.min_validation,
        "min_lowpass_score": args.min_lowpass_score,
        "min_oos_net_r": args.min_oos_net_r,
        "min_validation_net_r": args.min_validation_net_r,
        "min_all_net_r": args.min_all_net_r,
        "min_avg_r": args.min_avg_r,
        "max_all_dd_r": args.max_all_dd_r,
        "progress_every_configs": args.progress_every_configs,
        "event_progress_every_chunks": args.event_progress_every_chunks,
        "persist_event_cache": args.persist_event_cache,
    }
    rows: list[dict[str, Any]] = []
    completed_symbols: set[str] = set()
    summary_path = args.output_dir / "candidate_retest.csv"
    if args.resume and summary_path.exists():
        previous = pd.read_csv(summary_path)
        rows = previous.to_dict("records")
        if "symbol" in previous.columns:
            completed_symbols = {bybit_symbol(symbol) for symbol in previous["symbol"].dropna().astype(str)}
        print(f"Resume: loaded {len(rows)} completed rows from {summary_path}", flush=True)

    tasks = [{**common, "symbol": symbol} for symbol in symbols if symbol not in completed_symbols]
    if not tasks:
        write_outputs(args.output_dir, rows, symbol_table)
        print(f"DONE pass={sum(1 for row in rows if row.get('candidate_ok'))}/{len(rows)} skipped=0 errors=0 elapsed=0.0s output={args.output_dir}", flush=True)
        return

    started = datetime.now(timezone.utc)
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = {pool.submit(evaluate_symbol, task): task["symbol"] for task in tasks}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                row = future.result()
            except Exception as exc:  # noqa: BLE001
                row = {"symbol": symbol, "status": "error", "error": f"{type(exc).__name__}: {exc}", "candidate_ok": False}
            rows.append(row)
            write_outputs(args.output_dir, rows, symbol_table)
            done = len(rows)
            if row.get("candidate_ok"):
                print(
                    f"[{done}/{len(symbols)}] PASS {symbol}: rank={finite(row.get('rank_score')):.2f} "
                    f"lowpass={finite(row.get('lowpass_robust_score')):.2f} "
                    f"all={finite(row.get('all_net_r')):+.2f}R val={finite(row.get('validation_net_r')):+.2f}R "
                    f"oos={finite(row.get('oos_net_r')):+.2f}R trades={finite(row.get('all_trades')):.0f}",
                    flush=True,
                )
            elif row.get("status") == "skipped":
                print(f"[{done}/{len(symbols)}] SKIP {symbol}: {row.get('skip_reason')}", flush=True)
            elif row.get("status") == "error":
                print(f"[{done}/{len(symbols)}] ERROR {symbol}: {row.get('error')}", flush=True)
            else:
                print(
                    f"[{done}/{len(symbols)}] MISS {symbol}: rank={finite(row.get('rank_score')):.2f} "
                    f"lowpass={finite(row.get('lowpass_robust_score')):.2f} "
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
