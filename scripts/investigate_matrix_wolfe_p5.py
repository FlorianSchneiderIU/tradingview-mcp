#!/usr/bin/env python3
"""Investigate Matrix Wolfe found_p5 lifecycle and early-entry backtests."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest_wolfe_wave import fetch_bybit_klines  # noqa: E402


DEFAULT_ROOM_ID = "!gYMwAkfoJVPngqDULV:thebox.sbs"
SHORT_ALIAS = {
    "XAU": "XAUUSDT",
    "XAG": "XAGUSDT",
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "ADA": "ADAUSDT",
    "XRP": "XRPUSDT",
    "XLM": "XLMUSDT",
}


STAGE_RE = re.compile(r"\[(?P<stage>[A-Za-z0-9_]+)\]", re.IGNORECASE)
MARKER_RE = re.compile(
    r"#(?P<sym>[A-Z0-9]{2,10})\s+(?P<tf>\d+[smhdw]|1D)(?:\s+(?P<dir>BULL|BEAR))?",
    re.IGNORECASE,
)
HEADER_DIR_RE = re.compile(r"\]\s+(?P<dir>BULL|BEAR)\b", re.IGNORECASE)
ID_RE = re.compile(r"\b(?P<wave_id>[A-Za-z0-9]{4})\s*//\s*Regime:\s*(?P<regime>[^\n]+)", re.IGNORECASE)
TRAILING_ID_RE = re.compile(r"\]\s*.*?\b(?P<wave_id>[A-Za-z0-9]{4,})\s*$", re.IGNORECASE)
CURRENT_RE = re.compile(r"\$\s*(?P<price>[\d,]+(?:\.\d+)?)\s*-\s*\d{1,2}/\d{1,2}", re.IGNORECASE)
ENTRY_ZONE_RE = re.compile(
    r"(?P<side>Long|Short)\s+Entry\s+Zone:\s*(?P<low>[\d,]+(?:\.\d+)?)\s*-\s*(?P<high>[\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)
STOP_RE = re.compile(r"\bStop:\s*(?P<sl>[\d,]+(?:\.\d+)?)", re.IGNORECASE)
TP_RE = re.compile(r"\bTP(?P<num>\d{1,2}):\s*(?P<price>[\d,]+(?:\.\d+)?)", re.IGNORECASE)
SCORE_RE = re.compile(r"\((?P<score>[\d.]+)\s*/\s*8\)")
RR_RE = re.compile(r"\bR/R\s+(?P<rr>[\d.]+)", re.IGNORECASE)


def parse_price(text: str) -> float:
    return float(str(text).replace(",", ""))


def normalise_symbol(raw: str) -> str:
    sym = raw.upper().replace(".P", "")
    if sym in SHORT_ALIAS:
        return SHORT_ALIAS[sym]
    if not sym.endswith("USDT"):
        return f"{sym}USDT"
    return sym


def direction_from_text(raw: str | None) -> str | None:
    if str(raw or "").upper() == "BULL":
        return "long"
    if str(raw or "").upper() == "BEAR":
        return "short"
    return None


def load_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def fetch_matrix_messages(env_path: Path, room_id: str, limit: int) -> list[dict[str, Any]]:
    env = load_env(env_path)
    homeserver = env["MATRIX_HOMESERVER"].rstrip("/")
    token = env["MATRIX_ACCESS_TOKEN"]
    out: list[dict[str, Any]] = []
    from_token = None
    while len(out) < limit:
        qs = {"dir": "b", "limit": "100"}
        if from_token:
            qs["from"] = from_token
        url = (
            f"{homeserver}/_matrix/client/v3/rooms/{urllib.parse.quote(room_id, safe='')}/messages"
            f"?{urllib.parse.urlencode(qs)}"
        )
        request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
        chunk = data.get("chunk") or []
        if not chunk:
            break
        out.extend(chunk)
        next_token = data.get("end")
        if not next_token or next_token == from_token:
            break
        from_token = next_token
    return out[:limit]


def parse_lifecycle_event(event: dict[str, Any]) -> dict[str, Any] | None:
    body = str((event.get("content") or {}).get("body") or "")
    if event.get("type") != "m.room.message" or not body.strip():
        return None
    stage_m = STAGE_RE.search(body)
    marker_m = MARKER_RE.search(body)
    if not stage_m or not marker_m:
        return None
    direction_raw = marker_m.group("dir")
    if not direction_raw:
        header_dir_m = HEADER_DIR_RE.search(body)
        direction_raw = header_dir_m.group("dir") if header_dir_m else None
    id_m = ID_RE.search(body)
    trailing_id_m = TRAILING_ID_RE.search(body.strip())
    ts_ms = int(event.get("origin_server_ts") or 0)
    row: dict[str, Any] = {
        "time": datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc),
        "time_ms": ts_ms,
        "event_id": event.get("event_id"),
        "sender": event.get("sender"),
        "stage": stage_m.group("stage").lower(),
        "symbol": normalise_symbol(marker_m.group("sym")),
        "short_symbol": marker_m.group("sym").upper(),
        "timeframe": marker_m.group("tf"),
        "direction": direction_from_text(direction_raw),
        "wave_id": id_m.group("wave_id") if id_m else trailing_id_m.group("wave_id") if trailing_id_m else "",
        "regime": id_m.group("regime").strip() if id_m else "",
        "body": body,
    }
    if current_m := CURRENT_RE.search(body):
        row["signal_price"] = parse_price(current_m.group("price"))
    if entry_m := ENTRY_ZONE_RE.search(body):
        row["entry_zone_low"] = parse_price(entry_m.group("low"))
        row["entry_zone_high"] = parse_price(entry_m.group("high"))
    if stop_m := STOP_RE.search(body):
        row["sl"] = parse_price(stop_m.group("sl"))
    tps = [parse_price(m.group("price")) for m in TP_RE.finditer(body)]
    if tps:
        row["tps"] = tps
        row["tp1"] = tps[0]
    if score_m := SCORE_RE.search(body):
        row["score"] = float(score_m.group("score"))
    if rr_m := RR_RE.search(body):
        row["rr"] = float(rr_m.group("rr"))
    return row


def wave_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("wave_id") or "").lower(),
        str(row.get("symbol") or "").upper(),
        str(row.get("timeframe") or "").lower(),
        str(row.get("direction") or "").lower(),
    )


def group_waves(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in sorted(events, key=lambda item: item["time"]):
        key = wave_key(row)
        if not key[0] or not key[1] or not key[3]:
            continue
        grouped.setdefault(key, []).append(row)

    waves: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        first_by_stage: dict[str, dict[str, Any]] = {}
        for row in rows:
            first_by_stage.setdefault(row["stage"], row)
        full = first_by_stage.get("bona_fide")
        p5 = first_by_stage.get("found_p5")
        wave = {
            "wave_id": key[0],
            "symbol": key[1],
            "timeframe": key[2],
            "direction": key[3],
            "stages": ">".join(row["stage"] for row in rows),
            "found_p5_time": p5["time"] if p5 else pd.NaT,
            "sweet_zone_time": first_by_stage.get("sweet_zone", {}).get("time", pd.NaT),
            "bona_fide_time": full["time"] if full else pd.NaT,
            "entry_time": first_by_stage.get("entry", {}).get("time", pd.NaT),
            "target_hit_time": first_by_stage.get("target_hit", {}).get("time", pd.NaT),
            "stop_out_time": first_by_stage.get("stop_out", {}).get("time", pd.NaT),
            "canceled_time": first_by_stage.get("canceled", {}).get("time", pd.NaT),
            "has_p5": p5 is not None,
            "has_bona_fide": full is not None,
            "has_entry": "entry" in first_by_stage,
            "has_target_hit": "target_hit" in first_by_stage,
            "has_stop_out": "stop_out" in first_by_stage,
            "has_canceled": "canceled" in first_by_stage,
        }
        if full:
            for field in [
                "signal_price",
                "entry_zone_low",
                "entry_zone_high",
                "sl",
                "tp1",
                "tps",
                "score",
                "rr",
                "regime",
            ]:
                if field in full:
                    wave[field] = full[field]
        waves.append(wave)
    return waves


def add_proximity_lifecycle_links(
    events_df: pd.DataFrame,
    waves_df: pd.DataFrame,
    max_minutes: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if events_df.empty or waves_df.empty or "found_p5_time" not in waves_df.columns:
        return waves_df, pd.DataFrame()

    max_delta = pd.Timedelta(minutes=max_minutes)
    event_times = events_df.copy()
    event_times["time"] = pd.to_datetime(event_times["time"], utc=True)
    linked_waves = waves_df.copy()
    linked_waves["found_p5_time"] = pd.to_datetime(linked_waves["found_p5_time"], utc=True, errors="coerce")
    link_rows: list[dict[str, Any]] = []
    linked_fields = [
        "wave_id",
        "time",
        "signal_price",
        "entry_zone_low",
        "entry_zone_high",
        "sl",
        "tp1",
        "score",
        "rr",
        "regime",
    ]

    p5_mask = linked_waves["has_p5"] == True  # noqa: E712
    for idx, wave in linked_waves[p5_mask].iterrows():
        p5_time = wave["found_p5_time"]
        if pd.isna(p5_time):
            continue
        same_context = event_times[
            (event_times["symbol"] == wave["symbol"])
            & (event_times["timeframe"].astype(str).str.lower() == str(wave["timeframe"]).lower())
            & (event_times["direction"] == wave["direction"])
            & (event_times["time"] > p5_time)
            & (event_times["time"] <= p5_time + max_delta)
        ].sort_values("time")
        if same_context.empty:
            continue

        link_row = {
            "p5_wave_id": wave["wave_id"],
            "symbol": wave["symbol"],
            "timeframe": wave["timeframe"],
            "direction": wave["direction"],
            "found_p5_time": p5_time,
        }
        for stage in ["sweet_zone", "bona_fide", "entry", "target_hit", "stop_out", "canceled"]:
            stage_rows = same_context[same_context["stage"] == stage]
            if stage_rows.empty:
                linked_waves.loc[idx, f"prox_has_{stage}"] = False
                continue
            stage_event = stage_rows.iloc[0]
            minutes = (stage_event["time"] - p5_time).total_seconds() / 60.0
            linked_waves.loc[idx, f"prox_has_{stage}"] = True
            linked_waves.loc[idx, f"prox_{stage}_time"] = stage_event["time"]
            linked_waves.loc[idx, f"prox_{stage}_wave_id"] = stage_event["wave_id"]
            linked_waves.loc[idx, f"prox_{stage}_minutes"] = minutes
            link_row[f"{stage}_wave_id"] = stage_event["wave_id"]
            link_row[f"{stage}_time"] = stage_event["time"]
            link_row[f"{stage}_minutes"] = minutes
            if stage == "bona_fide":
                for field in linked_fields:
                    if field not in stage_event.index:
                        continue
                    value = stage_event[field]
                    if isinstance(value, list):
                        linked_waves.at[idx, f"prox_bona_fide_{field}"] = value
                    elif pd.notna(value):
                        linked_waves.at[idx, f"prox_bona_fide_{field}"] = value
        link_rows.append(link_row)

    for stage in ["sweet_zone", "bona_fide", "entry", "target_hit", "stop_out", "canceled"]:
        col = f"prox_has_{stage}"
        if col not in linked_waves.columns:
            linked_waves[col] = False
        linked_waves[col] = linked_waves[col].fillna(False).astype(bool)

    return linked_waves, pd.DataFrame(link_rows)


def first_index_after(frame: pd.DataFrame, ts: pd.Timestamp) -> int | None:
    idx = frame["open_time"].searchsorted(ts, side="right")
    if idx >= len(frame):
        return None
    return int(idx)


def select_final_level_trade(wave: dict[str, Any], entry: float) -> tuple[float, float] | None:
    direction = wave["direction"]
    sl = float(wave.get("sl") or math.nan)
    tps = wave.get("tps")
    if not isinstance(tps, list):
        tps = [wave.get("tp1")]
    tps = [float(tp) for tp in tps if tp is not None and math.isfinite(float(tp))]
    if not math.isfinite(sl) or not tps:
        return None
    if direction == "long":
        if not sl < entry:
            return None
        valid_tps = [tp for tp in tps if tp > entry]
    else:
        if not sl > entry:
            return None
        valid_tps = [tp for tp in tps if tp < entry]
    if not valid_tps:
        return None
    return sl, valid_tps[0]


def select_proximity_level_trade(wave: dict[str, Any], entry: float) -> tuple[float, float] | None:
    level_wave = {
        "direction": wave.get("direction"),
        "sl": wave.get("prox_bona_fide_sl"),
        "tp1": wave.get("prox_bona_fide_tp1"),
    }
    return select_final_level_trade(level_wave, entry)


def backtest_trade(
    frame: pd.DataFrame,
    *,
    entry_idx: int,
    direction: str,
    entry: float,
    sl: float,
    tp: float,
    max_hold_bars: int,
    fee_rate: float,
) -> dict[str, Any]:
    risk = abs(entry - sl)
    if risk <= 0:
        return {"skip_reason": "non_positive_risk"}
    end_idx = min(len(frame) - 1, entry_idx + max_hold_bars)
    exit_price = float(frame.iloc[end_idx]["close"])
    exit_idx = end_idx
    outcome = "timeout"
    for idx in range(entry_idx, end_idx + 1):
        row = frame.iloc[idx]
        if direction == "long":
            stop_hit = float(row["low"]) <= sl
            target_hit = float(row["high"]) >= tp
        else:
            stop_hit = float(row["high"]) >= sl
            target_hit = float(row["low"]) <= tp
        if stop_hit:
            exit_price = sl
            exit_idx = idx
            outcome = "stop"
            break
        if target_hit:
            exit_price = tp
            exit_idx = idx
            outcome = "target"
            break
    gross_r = ((exit_price - entry) / risk) if direction == "long" else ((entry - exit_price) / risk)
    fee_r = fee_rate * (entry + exit_price) / risk
    return {
        "outcome": outcome,
        "exit_time": frame.iloc[exit_idx]["open_time"],
        "exit_price": exit_price,
        "gross_r": gross_r,
        "fee_r": fee_r,
        "net_r": gross_r - fee_r,
        "risk_pct": risk / entry if entry else math.nan,
        "fee_to_stop_risk": fee_rate * (entry + sl) / risk,
        "hold_bars": exit_idx - entry_idx + 1,
    }


def structural_levels(frame: pd.DataFrame, entry_idx: int, direction: str, lookback: int, target_r: float) -> tuple[float, float] | None:
    if entry_idx <= lookback:
        return None
    entry = float(frame.iloc[entry_idx]["open"])
    window = frame.iloc[entry_idx - lookback:entry_idx]
    if direction == "long":
        sl = float(window["low"].min())
        risk = entry - sl
        tp = entry + target_r * risk
    else:
        sl = float(window["high"].max())
        risk = sl - entry
        tp = entry - target_r * risk
    if risk <= 0 or risk / entry < 0.0002:
        return None
    return sl, tp


def ensure_symbol_frame(symbol: str, start: datetime, end: datetime, cache_dir: Path) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe = symbol.lower()
    path = cache_dir / f"{safe}_1m_{start:%Y%m%d%H%M}_{end:%Y%m%d%H%M}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    frame = fetch_bybit_klines(symbol, "1m", start, end)
    frame.to_pickle(path)
    return frame


def summarize_trades(trades: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for key, part in trades.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        wins = int((part["net_r"] > 0).sum())
        losses = int((part["net_r"] <= 0).sum())
        gross_profit = float(part.loc[part["net_r"] > 0, "net_r"].sum())
        gross_loss = -float(part.loc[part["net_r"] <= 0, "net_r"].sum())
        row = {col: value for col, value in zip(group_cols, key)}
        row.update(
            {
                "trades": len(part),
                "net_r": float(part["net_r"].sum()),
                "avg_r": float(part["net_r"].mean()),
                "median_r": float(part["net_r"].median()),
                "winrate": wins / len(part) if len(part) else 0.0,
                "wins": wins,
                "losses": losses,
                "profit_factor": gross_profit / gross_loss if gross_loss > 0 else math.inf,
                "fee_valid_rate": float((part["fee_to_stop_risk"] <= 0.25).mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["avg_r", "trades"], ascending=[False, False])


def write_trade_summaries(trades_df: pd.DataFrame, out_dir: Path, suffix: str = "") -> None:
    summary_variant = summarize_trades(trades_df, ["variant"])
    summary_symbol = summarize_trades(trades_df, ["variant", "symbol"])
    stem = f"_{suffix}" if suffix else ""
    summary_variant.to_csv(out_dir / f"wolfe_p5_backtest_summary_by_variant{stem}.csv", index=False)
    summary_symbol.to_csv(out_dir / f"wolfe_p5_backtest_summary_by_variant_symbol{stem}.csv", index=False)
    print(f"\nTop variants{stem}", flush=True)
    print(summary_variant.head(20).to_string(index=False), flush=True)
    print(f"\nTop variant/symbol{stem}", flush=True)
    print(summary_symbol.head(30).to_string(index=False), flush=True)


def run(args: argparse.Namespace) -> None:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_events = fetch_matrix_messages(args.env_path, args.room_id, args.limit)
    events = [row for ev in raw_events if (row := parse_lifecycle_event(ev)) is not None]
    waves = group_waves(events)
    waves_df = pd.DataFrame(waves).sort_values(["found_p5_time", "bona_fide_time"], na_position="last")
    events_df = pd.DataFrame(events).sort_values("time")
    waves_df, proximity_links_df = add_proximity_lifecycle_links(events_df, waves_df, args.lifecycle_link_minutes)
    events_df.to_csv(out_dir / "wolfe_lifecycle_events.csv", index=False)
    waves_df.to_csv(out_dir / "wolfe_lifecycle_waves.csv", index=False)
    proximity_links_df.to_csv(out_dir / "wolfe_p5_proximity_lifecycle_links.csv", index=False)

    p5 = waves_df[waves_df["has_p5"] == True].copy()  # noqa: E712
    print(f"events={len(events_df)} waves={len(waves_df)} found_p5_waves={len(p5)}", flush=True)
    for col in ["has_bona_fide", "has_entry", "has_target_hit", "has_stop_out", "has_canceled"]:
        if col in p5:
            print(f"{col}: {int(p5[col].sum())}/{len(p5)} ({p5[col].mean():.1%})", flush=True)
    for col in ["prox_has_bona_fide", "prox_has_entry", "prox_has_target_hit", "prox_has_stop_out", "prox_has_canceled"]:
        if col in p5:
            print(f"{col}: {int(p5[col].sum())}/{len(p5)} ({p5[col].mean():.1%})", flush=True)
    if {"has_target_hit", "has_stop_out"}.issubset(p5.columns):
        outcome_known = p5[p5["has_target_hit"] | p5["has_stop_out"]]
        if not outcome_known.empty:
            print(
                f"channel_outcomes known={len(outcome_known)} target={int(outcome_known['has_target_hit'].sum())} "
                f"stop={int(outcome_known['has_stop_out'].sum())}",
                flush=True,
            )

    if p5.empty:
        return

    min_time = pd.to_datetime(p5["found_p5_time"].dropna()).min().to_pydatetime()
    max_time = pd.to_datetime(p5["found_p5_time"].dropna()).max().to_pydatetime()
    fetch_start = min_time - timedelta(days=args.fetch_pad_days)
    fetch_end = max_time + timedelta(days=args.fetch_pad_days)

    symbol_frames: dict[str, pd.DataFrame] = {}
    for symbol in sorted(p5["symbol"].dropna().unique()):
        try:
            frame = ensure_symbol_frame(symbol, fetch_start, fetch_end, args.cache_dir)
            symbol_frames[symbol] = frame
            print(f"{symbol}: candles={len(frame)} {frame['open_time'].min()} -> {frame['open_time'].max()}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"{symbol}: candle fetch failed: {exc}", flush=True)

    trades: list[dict[str, Any]] = []
    for _, wave in p5.iterrows():
        symbol = str(wave["symbol"])
        frame = symbol_frames.get(symbol)
        if frame is None or frame.empty or pd.isna(wave["found_p5_time"]):
            continue
        entry_idx = first_index_after(frame, pd.Timestamp(wave["found_p5_time"]))
        if entry_idx is None:
            continue
        entry_next_open = float(frame.iloc[entry_idx]["open"])
        base = {
            "wave_id": wave["wave_id"],
            "symbol": symbol,
            "timeframe": wave["timeframe"],
            "direction": wave["direction"],
            "found_p5_time": wave["found_p5_time"],
            "has_bona_fide": bool(wave.get("has_bona_fide")),
            "has_entry": bool(wave.get("has_entry")),
            "prox_has_bona_fide": bool(wave.get("prox_has_bona_fide")),
            "prox_has_entry": bool(wave.get("prox_has_entry")),
            "prox_bona_fide_minutes": wave.get("prox_bona_fide_minutes"),
            "prox_bona_fide_wave_id": wave.get("prox_bona_fide_wave_id"),
            "score": wave.get("score"),
            "rr": wave.get("rr"),
            "prox_score": wave.get("prox_bona_fide_score"),
            "prox_rr": wave.get("prox_bona_fide_rr"),
        }

        if bool(wave.get("has_bona_fide")):
            final_levels = select_final_level_trade(wave.to_dict(), entry_next_open)
            if final_levels:
                sl, tp = final_levels
                result = backtest_trade(
                    frame,
                    entry_idx=entry_idx,
                    direction=str(wave["direction"]),
                    entry=entry_next_open,
                    sl=sl,
                    tp=tp,
                    max_hold_bars=args.max_hold_bars,
                    fee_rate=args.fee_rate,
                )
                if "skip_reason" not in result:
                    trades.append({**base, **result, "variant": "p5_next_open_final_levels", "entry": entry_next_open, "sl": sl, "tp": tp})

        if bool(wave.get("prox_has_bona_fide")):
            final_levels = select_proximity_level_trade(wave.to_dict(), entry_next_open)
            if final_levels:
                sl, tp = final_levels
                result = backtest_trade(
                    frame,
                    entry_idx=entry_idx,
                    direction=str(wave["direction"]),
                    entry=entry_next_open,
                    sl=sl,
                    tp=tp,
                    max_hold_bars=args.max_hold_bars,
                    fee_rate=args.fee_rate,
                )
                if "skip_reason" not in result:
                    trades.append(
                        {
                            **base,
                            **result,
                            "variant": "p5_next_open_prox_bona_fide_levels",
                            "entry": entry_next_open,
                            "sl": sl,
                            "tp": tp,
                        }
                    )

        for lookback in args.lookbacks:
            for target_r in args.target_rs:
                levels = structural_levels(frame, entry_idx, str(wave["direction"]), lookback, target_r)
                if not levels:
                    continue
                sl, tp = levels
                result = backtest_trade(
                    frame,
                    entry_idx=entry_idx,
                    direction=str(wave["direction"]),
                    entry=entry_next_open,
                    sl=sl,
                    tp=tp,
                    max_hold_bars=args.max_hold_bars,
                    fee_rate=args.fee_rate,
                )
                if "skip_reason" not in result:
                    trades.append(
                        {
                            **base,
                            **result,
                            "variant": f"p5_next_open_swing{lookback}_tp{target_r:g}R",
                            "entry": entry_next_open,
                            "sl": sl,
                            "tp": tp,
                            "lookback": lookback,
                            "target_r": target_r,
                        }
                    )

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df.to_csv(out_dir / "wolfe_p5_backtest_trades.csv", index=False)
        write_trade_summaries(trades_df, out_dir)
        for threshold in args.fee_valid_thresholds:
            valid = trades_df[trades_df["fee_to_stop_risk"] <= threshold]
            if valid.empty:
                continue
            write_trade_summaries(valid, out_dir, f"fee{int(threshold * 1000):03d}")
    else:
        print("No backtest trades produced.", flush=True)

    print(f"\nWrote outputs to {out_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-path", type=Path, default=Path("bot/.env.matrix"))
    parser.add_argument("--room-id", default=DEFAULT_ROOM_ID)
    parser.add_argument("--limit", type=int, default=3000)
    parser.add_argument("--out-dir", type=Path, default=Path("scripts/matrix_wolfe_p5_investigation"))
    parser.add_argument("--cache-dir", type=Path, default=Path("scripts/.cache/matrix_wolfe_p5"))
    parser.add_argument("--fetch-pad-days", type=float, default=1.0)
    parser.add_argument("--max-hold-bars", type=int, default=360)
    parser.add_argument("--fee-rate", type=float, default=0.00055)
    parser.add_argument("--fee-valid-thresholds", type=float, nargs="+", default=[0.15, 0.25])
    parser.add_argument("--lifecycle-link-minutes", type=float, default=360.0)
    parser.add_argument("--lookbacks", type=int, nargs="+", default=[10, 20, 40, 80])
    parser.add_argument("--target-rs", type=float, nargs="+", default=[0.75, 1.0, 1.5, 2.0, 3.0])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
