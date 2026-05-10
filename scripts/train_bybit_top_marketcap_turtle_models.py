from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backtest_turtle_soup import Config, DEFAULT_BFM_ZONE_TF_SETS, DEFAULT_BFM_ZONE_TIMEFRAMES, parse_utc_datetime, run_backtest, summarize
from scripts.bybit_demo_turtle_soup import BybitV5Client, bybit_symbol, fetch_bybit_klines
from scripts.ml_trade_outcome_filter import (
    BASE_FEATURE_COLUMNS,
    add_engineered_rescue_features,
    classifier_metrics,
    feature_rank,
    fit_model,
    frame_metrics,
    threshold_table,
    trade_bfm_feature_columns_for_groups,
    trade_feature_rows,
)


BYBIT_PUBLIC_URL = "https://api.bybit.com"
COINGECKO_MARKETS_URL = "https://api.coingecko.com/api/v3/coins/markets"
COINLORE_TICKERS_URL = "https://api.coinlore.net/api/tickers/"
COINPAPRIKA_TICKERS_URL = "https://api.coinpaprika.com/v1/tickers"
STABLE_BASES = {"USDT", "USDC", "USDE", "USDD", "DAI", "FDUSD", "TUSD", "PYUSD", "USD1", "USDP"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one original Turtle Soup outcome model per current top-market-cap "
            "Bybit USDT perpetual coin and rank the OOS results."
        )
    )
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--symbols", nargs="+", default=[], help="Optional explicit Bybit symbols; skips market-cap universe lookup.")
    parser.add_argument("--universe-source", choices=["bybit", "market_cap"], default="bybit")
    parser.add_argument("--bybit-rank-field", choices=["turnover24h", "openInterestValue", "volume24h"], default="turnover24h")
    parser.add_argument("--include-stablecoins", action="store_true")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--warmup-start", default="2021-09-01")
    parser.add_argument("--train-start", default="2022-04-20")
    parser.add_argument("--split", default="2025-04-20")
    parser.add_argument("--end", default="2026-04-20")
    parser.add_argument("--cache-dir", type=Path, default=Path("scripts/.cache/bybit_linear"))
    parser.add_argument("--base-url", default=BYBIT_PUBLIC_URL)
    parser.add_argument("--tf1", default="1h")
    parser.add_argument("--tf2", default="4h")
    parser.add_argument("--use-tf2", action="store_true")
    parser.add_argument("--dead-zone", action="store_true")
    parser.add_argument("--max-zone-scan", type=int, default=0)
    parser.add_argument("--min-entry-risk-pct", type=float, default=0.0)
    parser.add_argument("--use-bfm-features", action="store_true", default=True)
    parser.add_argument("--bfm-feature-groups", default="line,channel")
    parser.add_argument("--bfm-timeframes", default=DEFAULT_BFM_ZONE_TIMEFRAMES)
    parser.add_argument("--bfm-tf-sets", default=DEFAULT_BFM_ZONE_TF_SETS)
    parser.add_argument("--bfm-invalidation", choices=["wick", "close", "none"], default="wick")
    parser.add_argument("--bfm-max-extension-bars", type=int, default=300)
    parser.add_argument("--use-sfp-liquidity-triggers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sfp-timeframes", default="15m,1h,4h")
    parser.add_argument("--sfp-left", type=int, default=15)
    parser.add_argument("--sfp-right", type=int, default=10)
    parser.add_argument("--sfp-level-width-atr", type=float, default=0.15)
    parser.add_argument("--sfp-strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sfp-require-open-reclaim", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model", choices=["rf", "logreg", "hgb"], default="rf")
    parser.add_argument("--thresholds", default="0.35,0.40,0.45,0.50,0.55,0.60")
    parser.add_argument("--min-train-rows", type=int, default=80)
    parser.add_argument("--min-oos-trades", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--write-models", action="store_true")
    parser.add_argument("--write-scored", action="store_true")
    parser.add_argument("--output-prefix", type=Path, default=Path("scripts/bybit_top50_turtle_per_symbol"))
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    return [float(chunk.strip()) for chunk in str(raw).split(",") if chunk.strip()]


def utc_now_label() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def normalized_market_symbol(value: str) -> str:
    text = str(value).strip().upper()
    for prefix in ("10000000", "1000000", "100000", "10000", "1000", "100"):
        if text.startswith(prefix) and len(text) > len(prefix):
            text = text[len(prefix) :]
            break
    return text.lower()


def fetch_bybit_linear_instruments(base_url: str) -> pd.DataFrame:
    client = BybitV5Client(base_url)
    rows: list[dict[str, Any]] = []
    cursor = ""
    while True:
        params = {"category": "linear", "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        result = client.get_public("/v5/market/instruments-info", params)
        rows.extend(result.get("list", []))
        cursor = str(result.get("nextPageCursor") or "")
        if not cursor:
            break
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("Bybit returned no linear instruments.")
    keep = (
        (frame["status"].astype(str) == "Trading")
        & (frame["quoteCoin"].astype(str) == "USDT")
        & (frame["settleCoin"].astype(str) == "USDT")
        & frame["symbol"].astype(str).str.endswith("USDT")
    )
    frame = frame.loc[keep].copy()
    frame["market_symbol"] = frame["baseCoin"].astype(str).map(normalized_market_symbol)
    frame["launch_time"] = pd.to_datetime(pd.to_numeric(frame["launchTime"], errors="coerce"), unit="ms", utc=True, errors="coerce")
    return frame.sort_values("symbol").reset_index(drop=True)


def fetch_bybit_linear_tickers(base_url: str) -> pd.DataFrame:
    client = BybitV5Client(base_url)
    result = client.get_public("/v5/market/tickers", {"category": "linear"})
    frame = pd.DataFrame(result.get("list", []))
    if frame.empty:
        raise RuntimeError("Bybit returned no linear tickers.")
    for column in [
        "lastPrice",
        "indexPrice",
        "markPrice",
        "turnover24h",
        "volume24h",
        "openInterest",
        "openInterestValue",
        "fundingRate",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def build_bybit_universe(args: argparse.Namespace, instruments: pd.DataFrame) -> pd.DataFrame:
    tickers = fetch_bybit_linear_tickers(args.base_url)
    merged = instruments.merge(tickers, on="symbol", how="inner", suffixes=("", "_ticker"))
    if not args.include_stablecoins:
        merged = merged[~merged["baseCoin"].astype(str).str.upper().isin(STABLE_BASES)].copy()
    metric = str(args.bybit_rank_field)
    merged = merged[pd.to_numeric(merged[metric], errors="coerce").fillna(0.0) > 0.0].copy()
    merged = merged.sort_values([metric, "openInterestValue", "turnover24h"], ascending=[False, False, False])
    merged = merged.drop_duplicates("market_symbol", keep="first").head(int(args.top_n)).reset_index(drop=True)
    merged["universe_rank"] = np.arange(1, len(merged) + 1)
    merged["universe_rank_metric"] = metric
    merged["coin_id"] = ""
    merged["coin_name"] = merged["baseCoin"]
    merged["market_cap_rank"] = np.nan
    merged["market_cap"] = np.nan
    merged["market_cap_source"] = f"bybit_{metric}"
    return merged


def fetch_coingecko_market_caps(max_pages: int = 3) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    session = requests.Session()
    for page in range(1, max_pages + 1):
        response = session.get(
            COINGECKO_MARKETS_URL,
            params={
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": 250,
                "page": page,
                "sparkline": "false",
            },
            timeout=30,
        )
        if response.status_code == 429 and rows:
            break
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break
        rows.extend(batch)
        time.sleep(1.3)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("CoinGecko returned no market-cap rows.")
    frame["market_symbol"] = frame["symbol"].astype(str).map(normalized_market_symbol)
    frame = frame.sort_values(["market_cap_rank", "market_cap"], ascending=[True, False])
    return frame.drop_duplicates("market_symbol", keep="first").reset_index(drop=True)


def fetch_coinlore_market_caps(limit: int = 500) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    session = requests.Session()
    for start in range(0, limit, 100):
        response = session.get(COINLORE_TICKERS_URL, params={"start": start, "limit": 100}, timeout=30)
        response.raise_for_status()
        batch = response.json().get("data", [])
        if not batch:
            break
        rows.extend(batch)
        time.sleep(0.2)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("CoinLore returned no market-cap rows.")
    frame["market_symbol"] = frame["symbol"].astype(str).map(normalized_market_symbol)
    frame["market_cap"] = pd.to_numeric(frame["market_cap_usd"], errors="coerce")
    frame["market_cap_rank"] = pd.to_numeric(frame["rank"], errors="coerce")
    frame["id"] = frame["nameid"].astype(str)
    return frame.sort_values(["market_cap_rank", "market_cap"], ascending=[True, False]).drop_duplicates("market_symbol", keep="first").reset_index(drop=True)


def fetch_coinpaprika_market_caps() -> pd.DataFrame:
    response = requests.get(COINPAPRIKA_TICKERS_URL, params={"quotes": "USD"}, timeout=60)
    response.raise_for_status()
    frame = pd.DataFrame(response.json())
    if frame.empty:
        raise RuntimeError("CoinPaprika returned no market-cap rows.")
    frame["market_symbol"] = frame["symbol"].astype(str).map(normalized_market_symbol)
    frame["market_cap"] = frame["quotes"].map(lambda item: (item or {}).get("USD", {}).get("market_cap"))
    frame["market_cap"] = pd.to_numeric(frame["market_cap"], errors="coerce")
    frame["market_cap_rank"] = pd.to_numeric(frame["rank"], errors="coerce")
    return frame.sort_values(["market_cap_rank", "market_cap"], ascending=[True, False]).drop_duplicates("market_symbol", keep="first").reset_index(drop=True)


def fetch_market_caps() -> tuple[pd.DataFrame, str]:
    errors: list[str] = []
    for source, loader in [
        ("coingecko", fetch_coingecko_market_caps),
        ("coinlore", fetch_coinlore_market_caps),
        ("coinpaprika", fetch_coinpaprika_market_caps),
    ]:
        try:
            return loader(), source
        except Exception as exc:
            errors.append(f"{source}: {exc}")
    raise RuntimeError("All market-cap sources failed: " + " | ".join(errors))


def build_universe(args: argparse.Namespace) -> pd.DataFrame:
    instruments = fetch_bybit_linear_instruments(args.base_url)
    if args.symbols:
        wanted = {bybit_symbol(symbol) for symbol in args.symbols}
        out = instruments[instruments["symbol"].isin(wanted)].copy()
        if len(out) != len(wanted):
            missing = sorted(wanted - set(out["symbol"]))
            raise RuntimeError(f"Missing Bybit linear instruments: {missing}")
        out["market_cap_rank"] = np.nan
        out["market_cap"] = np.nan
        out["coin_id"] = ""
        out["coin_name"] = out["baseCoin"]
        out["universe_rank"] = np.arange(1, len(out) + 1)
        out["universe_rank_metric"] = "explicit"
        out["market_cap_source"] = "explicit"
        return out.reset_index(drop=True)

    if args.universe_source == "bybit":
        return build_bybit_universe(args, instruments)

    caps, source = fetch_market_caps()
    merged = instruments.merge(
        caps[["market_symbol", "id", "name", "market_cap_rank", "market_cap"]],
        on="market_symbol",
        how="inner",
    )
    merged = merged.sort_values(["market_cap_rank", "symbol"], ascending=[True, True])
    merged = merged.drop_duplicates("market_symbol", keep="first")
    merged = merged.rename(columns={"id": "coin_id", "name": "coin_name"})
    merged["market_cap_source"] = source
    return merged.head(int(args.top_n)).reset_index(drop=True)


def ensure_bybit_cache(symbol: str, interval: str, start: datetime, end: datetime, cache_dir: Path, base_url: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    normalized = bybit_symbol(symbol).lower()
    for candidate in sorted(cache_dir.glob(f"{normalized}_{interval}_*.pkl")):
        try:
            frame = pd.read_pickle(candidate)
        except Exception:
            continue
        if frame.empty:
            continue
        first = pd.Timestamp(frame["open_time"].iloc[0]).to_pydatetime()
        last = pd.Timestamp(frame["close_time"].iloc[-1]).to_pydatetime()
        if first <= start and last >= end:
            return candidate
    path = cache_dir / f"{normalized}_{interval}_{start:%Y%m%d}_{end:%Y%m%d}.pkl"
    if path.exists():
        return path
    client = BybitV5Client(base_url)
    frame = fetch_bybit_klines(client, bybit_symbol(symbol), interval, start, end)
    frame.to_pickle(path)
    return path


def train_symbol_job(params: dict[str, Any]) -> tuple[str, dict[str, Any], pd.DataFrame, pd.DataFrame, str]:
    symbol = bybit_symbol(params["symbol"])
    cache_path = ensure_bybit_cache(
        symbol,
        params["interval"],
        params["warmup_start"],
        params["end"],
        params["cache_dir"],
        params["base_url"],
    )
    frame = pd.read_pickle(cache_path)
    cfg = Config(
        exec_tf=params["interval"],
        structure_tf="15m",
        entry_mode="zone_retest",
        tf1=params["tf1"],
        tf2=params["tf2"],
        use_tf1=True,
        use_tf2=params["use_tf2"],
        block_dead_zone=params["dead_zone"],
        max_structure_bars_to_choch=32,
        min_entry_risk_pct=params["min_entry_risk_pct"],
        max_zone_scan=params["max_zone_scan"],
        use_sfp_liquidity_zones=params["use_sfp_liquidity_triggers"],
        sfp_timeframes=params["sfp_timeframes"],
        sfp_left=params["sfp_left"],
        sfp_right=params["sfp_right"],
        sfp_level_width_atr=params["sfp_level_width_atr"],
        sfp_strict=params["sfp_strict"],
        sfp_require_open_reclaim=params["sfp_require_open_reclaim"],
    )
    feature_frame, trades = trade_feature_rows(
        symbol,
        frame,
        cfg,
        use_bfm_features=params["use_bfm_features"],
        bfm_timeframes=params["bfm_timeframes"],
        bfm_tf_sets=params["bfm_tf_sets"],
        bfm_invalidation=params["bfm_invalidation"],
        bfm_max_extension_bars=params["bfm_max_extension_bars"],
        use_sfp_liquidity_zones=params["use_sfp_liquidity_triggers"],
        sfp_timeframes=params["sfp_timeframes"],
        sfp_left=params["sfp_left"],
        sfp_right=params["sfp_right"],
        sfp_level_width_atr=params["sfp_level_width_atr"],
        sfp_strict=params["sfp_strict"],
        sfp_require_open_reclaim=params["sfp_require_open_reclaim"],
    )
    if feature_frame.empty:
        return symbol, {"symbol": symbol, "status": "no_trades"}, pd.DataFrame(), pd.DataFrame(), f"{symbol}: no trade rows"

    dataset = add_engineered_rescue_features(feature_frame)
    dataset["entry_time"] = pd.to_datetime(dataset["entry_time"], utc=True, errors="coerce")
    dataset = dataset[(dataset["entry_time"] >= pd.Timestamp(params["train_start"])) & (dataset["entry_time"] < pd.Timestamp(params["end"]))].copy()
    feature_columns = list(BASE_FEATURE_COLUMNS)
    if params["use_bfm_features"]:
        feature_columns.extend(trade_bfm_feature_columns_for_groups(params["bfm_feature_groups"]))
    for column in feature_columns:
        if column not in dataset.columns:
            dataset[column] = math.nan
    train = dataset[dataset["entry_time"] < pd.Timestamp(params["split"])].copy()
    oos = dataset[dataset["entry_time"] >= pd.Timestamp(params["split"])].copy()
    baseline_train = [trade for trade in trades if pd.Timestamp(params["train_start"]) <= trade.entry_time < pd.Timestamp(params["split"])]
    baseline_oos = [trade for trade in trades if pd.Timestamp(params["split"]) <= trade.entry_time < pd.Timestamp(params["end"])]
    base_train_metrics = summarize(baseline_train)
    base_oos_metrics = summarize(baseline_oos)

    summary: dict[str, Any] = {
        "symbol": symbol,
        "status": "skipped",
        "rows": int(len(dataset)),
        "train_rows": int(len(train)),
        "oos_rows": int(len(oos)),
        "baseline_train_trades": int(base_train_metrics["trades"]),
        "baseline_train_pf": float(base_train_metrics["profit_factor"]),
        "baseline_train_net_r": float(base_train_metrics["net_r"]),
        "baseline_oos_trades": int(base_oos_metrics["trades"]),
        "baseline_oos_pf": float(base_oos_metrics["profit_factor"]),
        "baseline_oos_net_r": float(base_oos_metrics["net_r"]),
    }
    if len(train) < params["min_train_rows"] or train["win_label"].nunique() < 2 or oos.empty:
        summary["skip_reason"] = "insufficient_train_or_oos"
        scored = dataset if params["write_scored"] else pd.DataFrame()
        return symbol, summary, scored, pd.DataFrame(), f"{symbol}: skipped ({summary['skip_reason']}); rows={len(dataset)}"

    raw_feature_count = len(feature_columns)
    feature_columns = [column for column in feature_columns if train[column].notna().any()]
    dropped_feature_count = raw_feature_count - len(feature_columns)
    summary["raw_feature_count"] = int(raw_feature_count)
    summary["dropped_all_nan_features"] = int(dropped_feature_count)
    if not feature_columns:
        summary["skip_reason"] = "no_nonempty_features"
        scored = dataset if params["write_scored"] else pd.DataFrame()
        return symbol, summary, scored, pd.DataFrame(), f"{symbol}: skipped ({summary['skip_reason']}); rows={len(dataset)}"

    model = fit_model(train, params["model"], feature_columns)
    dataset["trade_win_prob"] = model.predict_proba(dataset[feature_columns].astype(float))[:, 1]
    train = dataset[dataset["entry_time"] < pd.Timestamp(params["split"])].copy()
    oos = dataset[dataset["entry_time"] >= pd.Timestamp(params["split"])].copy()
    thresholds = threshold_table(oos, params["thresholds"])
    thresholds.insert(0, "symbol", symbol)
    thresholds["passes_min_oos_trades"] = thresholds["trades"].astype(int) >= int(params["min_oos_trades"])
    ranked = thresholds[thresholds["passes_min_oos_trades"]].copy()
    if ranked.empty:
        ranked = thresholds.copy()
    best = ranked.sort_values(["profit_factor", "net_r", "trades"], ascending=[False, False, False]).iloc[0]
    selected = oos[oos["trade_win_prob"] >= float(best["threshold"])].copy()

    summary.update(
        {
            "status": "trained",
            "feature_count": int(len(feature_columns)),
            "train_auc": classifier_metrics(train)["auc"],
            "oos_auc": classifier_metrics(oos)["auc"],
            "best_threshold": float(best["threshold"]),
            "best_trades": int(best["trades"]),
            "best_win_rate": float(best["win_rate"]),
            "best_pf": float(best["profit_factor"]),
            "best_net_r": float(best["net_r"]),
            "best_avg_r": float(best["avg_r"]),
            "best_max_dd_r": float(best["max_dd_r"]),
        }
    )
    if params["write_models"]:
        import joblib

        model_dir = Path(params["model_dir"])
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": model,
                "feature_columns": feature_columns,
                "model_kind": params["model"],
                "symbol": symbol,
                "config": {key: str(value) for key, value in params.items() if key not in {"symbols"}},
            },
            model_dir / f"{symbol.lower()}_model.joblib",
        )
        importance = feature_rank(model, oos if len(oos) >= 20 else train, params["model"], feature_columns).head(30)
        if not importance.empty:
            importance.insert(0, "symbol", symbol)
            importance.to_csv(model_dir / f"{symbol.lower()}_feature_importance.csv", index=False)
    return symbol, summary, dataset if params["write_scored"] else pd.DataFrame(), thresholds, f"{symbol}: trained rows={len(dataset)} best_pf={summary['best_pf']:.3g} best_net={summary['best_net_r']:.3g}"


def main() -> None:
    args = parse_args()
    if args.end.lower() == "now":
        end = datetime.now(timezone.utc)
    else:
        end = parse_utc_datetime(args.end)
    warmup_start = parse_utc_datetime(args.warmup_start)
    train_start = parse_utc_datetime(args.train_start)
    split = parse_utc_datetime(args.split)
    thresholds = parse_float_list(args.thresholds)

    universe = build_universe(args)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    universe_path = args.output_prefix.with_name(f"{args.output_prefix.name}_universe.csv")
    leaderboard_path = args.output_prefix.with_name(f"{args.output_prefix.name}_leaderboard.csv")
    partial_leaderboard_path = args.output_prefix.with_name(f"{args.output_prefix.name}_partial_leaderboard.csv")
    thresholds_path = args.output_prefix.with_name(f"{args.output_prefix.name}_thresholds.csv")
    selected_path = args.output_prefix.with_name(f"{args.output_prefix.name}_selected_trades.csv")
    scored_path = args.output_prefix.with_name(f"{args.output_prefix.name}_scored.csv")
    config_path = args.output_prefix.with_name(f"{args.output_prefix.name}_config.json")
    model_dir = args.output_prefix.with_name(f"{args.output_prefix.name}_models")

    universe.to_csv(universe_path, index=False)
    print(f"Universe snapshot {utc_now_label()}: {len(universe)} symbols -> {universe_path}", flush=True)
    print(", ".join(universe["symbol"].astype(str).tolist()), flush=True)

    common_params = {
        "interval": args.interval,
        "warmup_start": warmup_start,
        "train_start": train_start,
        "split": split,
        "end": end,
        "cache_dir": args.cache_dir,
        "base_url": args.base_url,
        "tf1": args.tf1,
        "tf2": args.tf2,
        "use_tf2": args.use_tf2,
        "dead_zone": args.dead_zone,
        "max_zone_scan": args.max_zone_scan,
        "min_entry_risk_pct": args.min_entry_risk_pct,
        "use_bfm_features": args.use_bfm_features,
        "bfm_feature_groups": args.bfm_feature_groups,
        "bfm_timeframes": args.bfm_timeframes,
        "bfm_tf_sets": args.bfm_tf_sets,
        "bfm_invalidation": args.bfm_invalidation,
        "bfm_max_extension_bars": args.bfm_max_extension_bars,
        "use_sfp_liquidity_triggers": args.use_sfp_liquidity_triggers,
        "sfp_timeframes": args.sfp_timeframes,
        "sfp_left": args.sfp_left,
        "sfp_right": args.sfp_right,
        "sfp_level_width_atr": args.sfp_level_width_atr,
        "sfp_strict": args.sfp_strict,
        "sfp_require_open_reclaim": args.sfp_require_open_reclaim,
        "model": args.model,
        "thresholds": thresholds,
        "min_train_rows": args.min_train_rows,
        "min_oos_trades": args.min_oos_trades,
        "write_models": args.write_models,
        "write_scored": args.write_scored,
        "model_dir": model_dir,
    }
    jobs = [{**common_params, "symbol": symbol} for symbol in universe["symbol"].astype(str)]

    summaries: list[dict[str, Any]] = []
    threshold_frames: list[pd.DataFrame] = []
    scored_frames: list[pd.DataFrame] = []

    def write_partial_summaries() -> None:
        if summaries:
            pd.DataFrame(summaries).to_csv(partial_leaderboard_path, index=False)

    if args.workers <= 1:
        for job in jobs:
            try:
                _, summary, scored, threshold_frame, message = train_symbol_job(job)
            except Exception as exc:
                summary = {"symbol": job["symbol"], "status": "error", "error": repr(exc)}
                scored = pd.DataFrame()
                threshold_frame = pd.DataFrame()
                message = f"{job['symbol']}: error {exc!r}"
            print(message, flush=True)
            summaries.append(summary)
            if not threshold_frame.empty:
                threshold_frames.append(threshold_frame)
            if not scored.empty:
                scored_frames.append(scored)
            write_partial_summaries()
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
            futures = {pool.submit(train_symbol_job, job): job["symbol"] for job in jobs}
            for future in as_completed(futures):
                try:
                    _, summary, scored, threshold_frame, message = future.result()
                except Exception as exc:
                    symbol = futures[future]
                    summary = {"symbol": symbol, "status": "error", "error": repr(exc)}
                    scored = pd.DataFrame()
                    threshold_frame = pd.DataFrame()
                    message = f"{symbol}: error {exc!r}"
                print(message, flush=True)
                summaries.append(summary)
                if not threshold_frame.empty:
                    threshold_frames.append(threshold_frame)
                if not scored.empty:
                    scored_frames.append(scored)
                write_partial_summaries()

    leaderboard = pd.DataFrame(summaries).merge(
        universe[
            [
                column
                for column in [
                    "symbol",
                    "baseCoin",
                    "coin_name",
                    "coin_id",
                    "market_cap_rank",
                    "market_cap",
                    "market_cap_source",
                    "universe_rank",
                    "universe_rank_metric",
                    "turnover24h",
                    "openInterestValue",
                    "volume24h",
                    "launch_time",
                ]
                if column in universe.columns
            ]
        ],
        on="symbol",
        how="left",
    )
    for metric_column in ["best_pf", "best_net_r", "best_trades"]:
        if metric_column not in leaderboard.columns:
            leaderboard[metric_column] = math.nan
    sort_cols = ["status", "best_pf", "best_net_r", "best_trades"]
    leaderboard["status_sort"] = (leaderboard["status"] == "trained").astype(int)
    leaderboard = leaderboard.sort_values(["status_sort", "best_pf", "best_net_r", "best_trades"], ascending=[False, False, False, False]).drop(columns=["status_sort"])
    leaderboard.to_csv(leaderboard_path, index=False)
    thresholds_out = pd.concat(threshold_frames, ignore_index=True) if threshold_frames else pd.DataFrame()
    thresholds_out.to_csv(thresholds_path, index=False)
    if scored_frames:
        scored = pd.concat(scored_frames, ignore_index=True).sort_values(["entry_time", "symbol"])
        scored.to_csv(scored_path, index=False)
        if "trade_win_prob" in scored.columns:
            selected_parts = []
            best_by_symbol = leaderboard.set_index("symbol")["best_threshold"].dropna().to_dict()
            for symbol, threshold in best_by_symbol.items():
                part = scored[(scored["symbol"] == symbol) & (scored["trade_win_prob"] >= float(threshold))].copy()
                selected_parts.append(part)
            if selected_parts:
                pd.concat(selected_parts, ignore_index=True).sort_values(["entry_time", "symbol"]).to_csv(selected_path, index=False)

    config_path.write_text(
        json.dumps(
            {
                "created": utc_now_label(),
                "sources": {
                    "bybit_instruments": "https://api.bybit.com/v5/market/instruments-info",
                    "coingecko_markets": COINGECKO_MARKETS_URL,
                    "coinlore_tickers": COINLORE_TICKERS_URL,
                    "coinpaprika_tickers": COINPAPRIKA_TICKERS_URL,
                },
                "args": {key: str(value) for key, value in vars(args).items()},
                "symbols": universe["symbol"].astype(str).tolist(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    display_cols = [
        "symbol",
        "universe_rank",
        "universe_rank_metric",
        "market_cap_rank",
        "market_cap",
        "turnover24h",
        "openInterestValue",
        "status",
        "rows",
        "train_rows",
        "oos_rows",
        "baseline_oos_pf",
        "baseline_oos_net_r",
        "oos_auc",
        "best_threshold",
        "best_trades",
        "best_win_rate",
        "best_pf",
        "best_net_r",
        "best_max_dd_r",
    ]
    print()
    print("Leaderboard")
    print(leaderboard[[col for col in display_cols if col in leaderboard.columns]].head(30).to_string(index=False))
    print(f"\nWrote {leaderboard_path}")
    print(f"Wrote {thresholds_path}")
    print(f"Wrote {universe_path}")
    if args.write_scored:
        print(f"Wrote {scored_path}")
        print(f"Wrote {selected_path}")
    if args.write_models:
        print(f"Wrote models under {model_dir}")
    print(f"Wrote {config_path}")


if __name__ == "__main__":
    main()
