from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from queue import Empty, Queue
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backtest_turtle_soup import normalize_timeframe
from scripts.breaker_candidate_presets import (
    BreakerCandidatePreset,
    apply_preset_args,
    build_confirmation_lookup,
    evaluate_candidate,
    get_preset,
    preset_names,
    preset_summary,
)
from scripts.bybit_demo_turtle_soup import (
    DEFAULT_BASE_URL,
    DEFAULT_SYMBOLS,
    BybitV5Client,
    append_closed_candle,
    bybit_symbol,
    bybit_websocket_interval,
    candle_row_from_ws,
    fetch_bybit_klines,
    load_env_file,
    utc_now_iso,
    write_json_file,
    write_jsonl,
)
from scripts.study_breaker_continuation import BreakerConfig, simulate_breakers


DEFAULT_BREAKER_MODEL = Path("scripts/breaker_continuation_model_core3_1h_fvg_print15_retest72_rf.joblib")


def build_config(args: argparse.Namespace) -> BreakerConfig:
    return BreakerConfig(
        entry_mode=args.entry_mode,
        zone_tf=args.zone_tf,
        confirmation_tf=args.confirmation_tf,
        structure_left=args.structure_left,
        structure_right=args.structure_right,
        htf_left=args.htf_left,
        htf_right=args.htf_right,
        htf_ob_search_bars=args.htf_ob_search_bars,
        max_zone_scan=args.max_zone_scan,
        max_retest_bars=args.max_retest_bars,
        max_confirm_bars=args.max_confirm_bars,
        max_hold_bars=args.max_hold_bars,
        stop_buffer_atr=args.stop_buffer_atr,
        target_rr=args.target_rr,
        min_reject_pos=args.min_reject_pos,
        min_confirm_fvg_atr=args.min_confirm_fvg_atr,
        min_entry_risk_pct=args.min_entry_risk_pct,
    )


def load_model(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        raise SystemExit(f"Breaker model not found: {path}")
    import joblib

    return joblib.load(path)


def score_pending_entry(
    pending: dict[str, Any],
    model_payload: dict[str, Any] | None,
    frame: pd.DataFrame,
    confirmation_lookup: pd.DataFrame | None,
    confirmation_tf: str,
) -> float | None:
    if model_payload is None:
        return None

    entry_price = float(pending.get("planned_entry_price", math.nan))
    stop_price = float(pending.get("stop_price", math.nan))
    if not math.isfinite(entry_price) or entry_price <= 0 or not math.isfinite(stop_price):
        return None

    risk = abs(entry_price - stop_price)
    if risk <= 0:
        return None

    entry_time = pd.Timestamp(pending.get("signal_time"), tz="UTC")
    break_time = pd.Timestamp(pending.get("break_time"), tz="UTC")
    retest_time = pd.Timestamp(pending.get("retest_time"), tz="UTC")
    zone_time = pd.Timestamp(pending.get("zone_time"), tz="UTC")
    confirm_time_raw = pending.get("confirm_time")
    confirm_time = pd.Timestamp(confirm_time_raw, tz="UTC") if pd.notna(confirm_time_raw) else pd.NaT

    zone_width = abs(float(pending["zone_top"]) - float(pending["zone_bottom"]))
    zone_age_hours = max(0.0, (break_time - zone_time).total_seconds() / 3600.0)
    retest_delay_hours = max(0.0, (retest_time - break_time).total_seconds() / 3600.0)
    confirm_delay_hours = 0.0 if pd.isna(confirm_time) else max(0.0, (confirm_time - retest_time).total_seconds() / 3600.0)
    entry_hour = entry_time.hour + entry_time.minute / 60.0
    entry_dow = float(entry_time.dayofweek)
    break_hour = break_time.hour + break_time.minute / 60.0
    confirm_fvg_height = abs(float(pending.get("confirm_fvg_height", 0.0) or 0.0))
    direction = str(pending["direction"])
    zone_top = float(pending.get("zone_top", math.nan))
    zone_bottom = float(pending.get("zone_bottom", math.nan))
    zone_height = abs(zone_top - zone_bottom)
    confirm_break_level = float(pending.get("confirm_break_level", math.nan))

    gate_pass, context_metrics, _ = evaluate_candidate(
        {
            **pending,
            "planned_entry_price": entry_price,
            "stop_price": stop_price,
        },
        BreakerCandidatePreset(
            name="_scoring_context",
            symbols=(str(pending["symbol"]),),
            directions=(direction,),
            interval="5m",
            entry_mode=str(pending.get("entry_mode", "fvg_print")),
            zone_tf=str(pending.get("zone_tf", "1h")),
            confirmation_tf=confirmation_tf,
            max_retest_bars=0,
            max_confirm_bars=0,
            max_hold_bars=0,
            stop_buffer_atr=0.0,
            target_rr=0.0,
            min_entry_risk_pct=0.0,
            strategy_min_reject_pos=0.0,
            context_min_retest_reject_pos=0.0,
        ),
        confirmation_lookup=confirmation_lookup,
    )
    _ = gate_pass

    retest_lookup = frame.set_index("open_time")[["open", "high", "low", "close", "atr"]]
    retest_depth_frac = math.nan
    retest_close_margin_r = math.nan
    retest_range_atr = math.nan
    if retest_time in retest_lookup.index and zone_height > 0:
        retest_bar = retest_lookup.loc[retest_time]
        atr = float(retest_bar["atr"])
        if direction == "long":
            retest_depth_frac = (zone_top - float(retest_bar["low"])) / zone_height
            retest_close_margin_r = (float(retest_bar["close"]) - zone_top) / risk
        else:
            retest_depth_frac = (float(retest_bar["high"]) - zone_bottom) / zone_height
            retest_close_margin_r = (zone_bottom - float(retest_bar["close"])) / risk
        if math.isfinite(atr) and atr > 0:
            retest_range_atr = (float(retest_bar["high"]) - float(retest_bar["low"])) / atr

    confirm_gap_r = context_metrics.get("confirm_gap_r", math.nan)
    confirm_close_pos_dir = context_metrics.get("confirm_close_pos_dir", math.nan)
    confirm_body_frac = context_metrics.get("confirm_body_frac", math.nan)
    entry_extension_r = (
        (entry_price - zone_top) / risk
        if direction == "long"
        else (zone_bottom - entry_price) / risk
    )
    confirm_break_r = (
        (confirm_break_level - zone_top) / risk
        if direction == "long" and math.isfinite(confirm_break_level)
        else (zone_bottom - confirm_break_level) / risk
        if direction == "short" and math.isfinite(confirm_break_level)
        else math.nan
    )

    row = {feature: math.nan for feature in model_payload["numeric_features"]}
    row.update({
        "symbol": pending["symbol"],
        "direction": direction,
        "direction_long": 1.0 if direction == "long" else 0.0,
        "risk_pct": context_metrics.get("risk_pct", risk / entry_price * 100.0),
        "zone_width_pct": zone_width / entry_price * 100.0,
        "zone_age_hours_log": math.log1p(zone_age_hours),
        "retest_delay_hours_log": math.log1p(retest_delay_hours),
        "confirm_delay_hours_log": math.log1p(confirm_delay_hours),
        "confirm_fvg_atr": float(pending.get("confirm_fvg_atr", 0.0) or 0.0),
        "confirm_fvg_height_pct": confirm_fvg_height / entry_price * 100.0,
        "confirm_fvg_r": confirm_fvg_height / risk,
        "retest_reject_pos": float(pending.get("retest_reject_pos", math.nan)),
        "entry_extension_r": entry_extension_r,
        "confirm_gap_r": confirm_gap_r,
        "confirm_break_r": confirm_break_r,
        "confirm_close_pos_dir": confirm_close_pos_dir,
        "confirm_body_frac": confirm_body_frac,
        "retest_depth_frac": retest_depth_frac,
        "retest_close_margin_r": retest_close_margin_r,
        "retest_range_atr": retest_range_atr,
        "rejection_speed": float(pending.get("retest_reject_pos", 0.0) or 0.0) / (1.0 + max(0.0, confirm_delay_hours)),
        "chase_score": entry_extension_r * confirm_close_pos_dir if math.isfinite(confirm_close_pos_dir) else math.nan,
        "confirm_strength_score": float(pending.get("retest_reject_pos", 0.0) or 0.0) * (confirm_body_frac if math.isfinite(confirm_body_frac) else 0.0) * (1.0 + float(pending.get("confirm_fvg_atr", 0.0) or 0.0)),
        "entry_hour_sin": math.sin(2.0 * math.pi * entry_hour / 24.0),
        "entry_hour_cos": math.cos(2.0 * math.pi * entry_hour / 24.0),
        "entry_dow_sin": math.sin(2.0 * math.pi * entry_dow / 7.0),
        "entry_dow_cos": math.cos(2.0 * math.pi * entry_dow / 7.0),
        "break_hour_sin": math.sin(2.0 * math.pi * break_hour / 24.0),
        "break_hour_cos": math.cos(2.0 * math.pi * break_hour / 24.0),
    })
    numeric_features = model_payload["numeric_features"]
    categorical_features = model_payload["categorical_features"]
    features = pd.DataFrame([row])
    return float(model_payload["model"].predict_proba(features[numeric_features + categorical_features])[:, 1][0])


def compact_pending(pending: dict[str, Any] | None) -> dict[str, Any] | None:
    if pending is None:
        return None
    keys = [
        "symbol",
        "direction",
        "entry_mode",
        "zone_tf",
        "confirmation_tf",
        "zone_time",
        "zone_top",
        "zone_bottom",
        "break_time",
        "retest_time",
        "signal_time",
        "planned_entry_price",
        "stop_price",
        "planned_target_price",
        "confirm_kind",
        "confirm_time",
        "confirm_break_level",
        "confirm_fvg_top",
        "confirm_fvg_bottom",
        "confirm_fvg_height",
        "confirm_fvg_atr",
        "retest_reject_pos",
    ]
    return {key: pending.get(key) for key in keys}


def process_symbol_frame(
    args: argparse.Namespace,
    symbol: str,
    frame: pd.DataFrame,
    cfg: BreakerConfig,
    model_payload: dict[str, Any] | None,
    candidate_preset: BreakerCandidatePreset | None,
) -> dict[str, Any]:
    trades, state = simulate_breakers(symbol, frame, cfg, return_state=True)
    pending_entry = state.get("pending_entry")
    pending_confirm = state.get("pending_confirm")
    status = f"no pending breaker entry after {state.get('closed_trades', 0)} closed historical trades"
    probability = None
    candidate_pass = None
    candidate_failures: list[str] = []
    candidate_metrics: dict[str, Any] | None = None
    confirmation_lookup = build_confirmation_lookup(frame, args.confirmation_tf)

    if pending_entry is not None:
        signal_time = pd.Timestamp(pending_entry["signal_time"])
        signal_age = pd.Timestamp.now(tz="UTC") - signal_time
        allowed = set(args.directions)
        if pending_entry["direction"] not in allowed:
            status = f"fresh breaker entry candidate ignored by direction gate ({pending_entry['direction']})"
        elif signal_age <= pd.Timedelta(minutes=args.max_signal_age_minutes):
            probability = score_pending_entry(pending_entry, model_payload, frame, confirmation_lookup, args.confirmation_tf)
            if candidate_preset is not None:
                candidate_pass, candidate_metrics, candidate_failures = evaluate_candidate(
                    pending_entry,
                    candidate_preset,
                    confirmation_lookup=confirmation_lookup,
                )
                if candidate_pass:
                    if probability is None:
                        status = f"fresh breaker entry candidate passes preset {candidate_preset.name}"
                    elif probability >= args.breaker_prob_threshold:
                        status = f"fresh breaker entry candidate passes preset {candidate_preset.name} and ML threshold {args.breaker_prob_threshold:.2f}"
                    else:
                        status = f"fresh breaker entry candidate passes preset {candidate_preset.name} but is below ML threshold {args.breaker_prob_threshold:.2f}"
                else:
                    failed = ",".join(candidate_failures) if candidate_failures else "context"
                    status = f"fresh breaker entry candidate fails preset {candidate_preset.name} ({failed})"
            elif probability is None:
                status = "fresh breaker entry candidate"
            elif probability >= args.breaker_prob_threshold:
                status = f"fresh breaker entry candidate passes ML threshold {args.breaker_prob_threshold:.2f}"
            else:
                status = f"fresh breaker entry candidate below ML threshold {args.breaker_prob_threshold:.2f}"
        else:
            status = f"stale breaker entry candidate ({signal_age})"
    elif pending_confirm is not None:
        status = "waiting for breaker confirmation"
    elif state.get("position") is not None:
        status = "historical breaker position open"

    return {
        "time": utc_now_iso(),
        "symbol": symbol,
        "bars": len(frame),
        "status": status,
        "entry_mode": args.entry_mode,
        "directions": args.directions,
        "zone_tf": args.zone_tf,
        "confirmation_tf": args.confirmation_tf,
        "closed_trades": state.get("closed_trades", 0),
        "breaker_prob": probability,
        "breaker_prob_threshold": args.breaker_prob_threshold,
        "candidate_preset": candidate_preset.name if candidate_preset else None,
        "candidate_pass": candidate_pass,
        "candidate_failures": candidate_failures or None,
        "candidate_metrics": candidate_metrics,
        "pending_entry": compact_pending(pending_entry),
        "pending_confirm": compact_pending(pending_confirm),
        "recent_trade": trades.tail(1).to_dict("records")[0] if not trades.empty else None,
    }


def initial_heartbeat(args: argparse.Namespace, mode: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "pid": os.getpid(),
        "started_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "symbols": args.symbols,
        "interval": args.interval,
        "entry_mode": args.entry_mode,
        "zone_tf": args.zone_tf,
        "confirmation_tf": args.confirmation_tf,
        "model": None if args.no_breaker_model else str(args.breaker_model) if args.breaker_model else None,
        "candidate_preset": preset_summary(args.candidate_preset_obj) if getattr(args, "candidate_preset_obj", None) else None,
        "last_status_by_symbol": {},
        "last_candle_by_symbol": {},
        "log_jsonl": str(args.log_jsonl),
    }


def update_heartbeat(args: argparse.Namespace, heartbeat: dict[str, Any], event: str, **updates: Any) -> None:
    heartbeat.update(updates)
    heartbeat["event"] = event
    heartbeat["updated_at"] = utc_now_iso()
    write_json_file(args.heartbeat_json, heartbeat)


def bootstrap_frames(args: argparse.Namespace, client: BybitV5Client) -> dict[str, pd.DataFrame]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.lookback_days)
    frames: dict[str, pd.DataFrame] = {}
    for symbol in args.symbols:
        frames[symbol] = fetch_bybit_klines(client, symbol, args.interval, start, end)
        last_open = frames[symbol]["open_time"].iloc[-1] if not frames[symbol].empty else "n/a"
        print(f"{symbol}: bootstrapped {len(frames[symbol])} closed {args.interval} candles through {last_open}")
    return frames


def websocket_callback(args: argparse.Namespace, event_queue: Queue) -> Any:
    def on_message(message: dict[str, Any]) -> None:
        topic = str(message.get("topic", ""))
        topic_symbol = topic.split(".")[-1] if topic else ""
        for candle in message.get("data", []):
            symbol = bybit_symbol(str(candle.get("symbol") or topic_symbol))
            if symbol not in args.symbols:
                continue
            row = candle_row_from_ws(symbol, args.interval, candle)
            if row is not None:
                event_queue.put(row)

    return on_message


def run_scan(
    args: argparse.Namespace,
    client: BybitV5Client,
    frames: dict[str, pd.DataFrame],
    cfg: BreakerConfig,
    model_payload: dict[str, Any] | None,
    candidate_preset: BreakerCandidatePreset | None,
) -> None:
    for symbol in args.symbols:
        row = process_symbol_frame(args, symbol, frames[symbol], cfg, model_payload, candidate_preset)
        print(f"{symbol}: {row['status']}")
        write_jsonl(args.log_jsonl, row)


def run_websocket(
    args: argparse.Namespace,
    client: BybitV5Client,
    cfg: BreakerConfig,
    model_payload: dict[str, Any] | None,
    candidate_preset: BreakerCandidatePreset | None,
) -> None:
    try:
        from pybit.unified_trading import WebSocket
    except ImportError as exc:
        raise SystemExit("pybit is required for websocket mode. Install pybit in the active venv.") from exc

    heartbeat = initial_heartbeat(args, "websocket")
    update_heartbeat(args, heartbeat, "bootstrapping")
    frames = bootstrap_frames(args, client)
    for symbol in args.symbols:
        heartbeat["last_candle_by_symbol"][symbol] = frames[symbol]["open_time"].iloc[-1].isoformat() if not frames[symbol].empty else None
    update_heartbeat(args, heartbeat, "bootstrapped")
    if args.scan_on_start:
        run_scan(args, client, frames, cfg, model_payload, candidate_preset)

    event_queue: Queue = Queue()
    seen_candles: set[tuple[str, pd.Timestamp]] = set()
    ws = WebSocket(
        channel_type="linear",
        testnet=args.websocket_testnet,
        demo=args.websocket_demo,
        ping_interval=args.websocket_ping_interval,
        ping_timeout=args.websocket_ping_timeout,
        retries=args.websocket_retries,
    )
    ws.kline_stream(
        interval=bybit_websocket_interval(args.interval),
        symbol=args.symbols,
        callback=websocket_callback(args, event_queue),
    )
    print(f"Breaker shadow websocket active interval={args.interval} symbols={','.join(args.symbols)}")
    update_heartbeat(args, heartbeat, "websocket_active")

    processed = 0
    try:
        while True:
            try:
                candle = event_queue.get(timeout=args.websocket_idle_timeout)
            except Empty:
                print(f"WebSocket idle: no confirmed {args.interval} candles in {args.websocket_idle_timeout}s")
                update_heartbeat(args, heartbeat, "websocket_idle", last_idle_time=utc_now_iso())
                continue
            key = (str(candle["symbol"]), pd.Timestamp(candle["open_time"]))
            if key in seen_candles:
                continue
            seen_candles.add(key)
            symbol = key[0]
            frames[symbol] = append_closed_candle(frames[symbol], candle, args.lookback_days)
            row = process_symbol_frame(args, symbol, frames[symbol], cfg, model_payload, candidate_preset)
            print(f"{symbol}: confirmed {args.interval} candle {pd.Timestamp(candle['open_time']).isoformat()} - {row['status']}")
            write_jsonl(args.log_jsonl, row)
            heartbeat["last_status_by_symbol"][symbol] = row["status"]
            heartbeat["last_candle_by_symbol"][symbol] = pd.Timestamp(candle["open_time"]).isoformat()
            update_heartbeat(args, heartbeat, f"processed_{symbol}")
            processed += 1
            if args.websocket_stop_after_events > 0 and processed >= args.websocket_stop_after_events:
                break
    finally:
        ws.exit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live shadow monitor for breaker-continuation experiments. Never places orders.")
    parser.add_argument("--candidate-preset", choices=preset_names(), default=None)
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--lookback-days", type=int, default=120)
    parser.add_argument("--entry-mode", choices=["zone_retest", "structure_fvg", "fvg_print"], default="fvg_print")
    parser.add_argument("--directions", nargs="+", choices=["long", "short"], default=["long", "short"])
    parser.add_argument("--zone-tf", default="1h")
    parser.add_argument("--confirmation-tf", default="15m")
    parser.add_argument("--structure-left", type=int, default=2)
    parser.add_argument("--structure-right", type=int, default=2)
    parser.add_argument("--htf-left", type=int, default=5)
    parser.add_argument("--htf-right", type=int, default=5)
    parser.add_argument("--htf-ob-search-bars", type=int, default=50)
    parser.add_argument("--max-zone-scan", type=int, default=250)
    parser.add_argument("--max-retest-bars", type=int, default=72)
    parser.add_argument("--max-confirm-bars", type=int, default=72)
    parser.add_argument("--max-hold-bars", type=int, default=120)
    parser.add_argument("--stop-buffer-atr", type=float, default=0.10)
    parser.add_argument("--target-rr", type=float, default=2.0)
    parser.add_argument("--min-reject-pos", type=float, default=0.50)
    parser.add_argument("--min-confirm-fvg-atr", type=float, default=0.0)
    parser.add_argument("--min-entry-risk-pct", type=float, default=0.0)
    parser.add_argument("--breaker-model", type=Path, default=DEFAULT_BREAKER_MODEL)
    parser.add_argument("--no-breaker-model", action="store_true")
    parser.add_argument("--breaker-prob-threshold", type=float, default=0.55)
    parser.add_argument("--max-signal-age-minutes", type=float, default=15.0)
    parser.add_argument("--base-url", default=os.environ.get("BYBIT_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--env-file", type=Path, default=Path("scripts/bybit_demo.env"))
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--scan-on-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--websocket-demo", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--websocket-testnet", action="store_true")
    parser.add_argument("--websocket-ping-interval", type=int, default=20)
    parser.add_argument("--websocket-ping-timeout", type=int, default=10)
    parser.add_argument("--websocket-retries", type=int, default=10)
    parser.add_argument("--websocket-idle-timeout", type=int, default=120)
    parser.add_argument("--websocket-stop-after-events", type=int, default=0)
    parser.add_argument("--log-jsonl", type=Path, default=Path("scripts/live_logs/bybit_breaker_shadow.jsonl"))
    parser.add_argument("--heartbeat-json", type=Path, default=Path("scripts/live_logs/bybit_breaker_shadow_heartbeat.json"))
    parser.add_argument("--pid-file", type=Path, default=Path("scripts/bybit_breaker_shadow.pid"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    args.candidate_preset_obj = get_preset(args.candidate_preset) if args.candidate_preset else None
    if args.candidate_preset_obj is not None:
        apply_preset_args(args, args.candidate_preset_obj)
    args.symbols = [bybit_symbol(symbol) for symbol in args.symbols]
    args.interval = normalize_timeframe(args.interval)
    args.zone_tf = normalize_timeframe(args.zone_tf)
    args.confirmation_tf = normalize_timeframe(args.confirmation_tf)
    args.pid_file.parent.mkdir(parents=True, exist_ok=True)
    args.pid_file.write_text(str(os.getpid()), encoding="utf-8")

    model_path = None if args.no_breaker_model else args.breaker_model
    model_payload = load_model(model_path)
    cfg = build_config(args)
    client = BybitV5Client(base_url=args.base_url)

    if args.loop:
        run_websocket(args, client, cfg, model_payload, args.candidate_preset_obj)
        return

    heartbeat = initial_heartbeat(args, "single_scan")
    update_heartbeat(args, heartbeat, "starting_scan")
    frames = bootstrap_frames(args, client)
    run_scan(args, client, frames, cfg, model_payload, args.candidate_preset_obj)
    update_heartbeat(args, heartbeat, "finished_scan")


if __name__ == "__main__":
    main()
