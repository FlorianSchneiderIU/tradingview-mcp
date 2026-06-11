#!/usr/bin/env python3
"""Parse and backtest OPI curl reversal Telegram exports."""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import sys
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest_wolfe_wave import fetch_bybit_klines  # noqa: E402


DEFAULT_OPI_MATRIX_ROOM_ID = "!YURVFZgoYWChTYaXbO:thebox.sbs"

SHORT_ALIAS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "ADA": "ADAUSDT",
    "XRP": "XRPUSDT",
    "XLM": "XLMUSDT",
    "BNB": "BNBUSDT",
    "XAU": "XAUUSDT",
    "XAG": "XAGUSDT",
}

OPI_FULL_RE = re.compile(
    r"\bOp\S*\s*//\s*(?P<symbol>[A-Z0-9]{2,16})\s+"
    r"(?P<timeframe>\d+[smhdw])\s+(?P<direction>LONG|SHORT)\s*@\s*\$?(?P<entry>[\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)
TARGET_RE = re.compile(r"\b(?:T|TP)\s*:?\s*\$?(?P<target>[\d,]+(?:\.\d+)?)", re.IGNORECASE)
SL_RE = re.compile(r"\bSL\s*:?\s*\$?(?P<sl>[\d,]+(?:\.\d+)?)", re.IGNORECASE)
SCORE_RE = re.compile(r"(?:\b1C\s*//.*?)?(?:\*|\bscore\b)?\s*(?P<score>[\d.]+)\s*/\s*8", re.IGNORECASE | re.DOTALL)
NEXT_EVENT_RE = re.compile(r"\bNext\s+event:\s*(?P<event>[^\n\r]+)", re.IGNORECASE)
TF_FLOOR_RE = re.compile(r"(?P<floor>\d+[smhdw]\s+TF-floor[^\n\r]*)", re.IGNORECASE)
MULTI_TF_RE = re.compile(r"(?P<multi_tf>\d+[smhdw](?:/\d+[smhdw]){1,})")
BIAS_FLIP_RE = re.compile(
    r"(?P<symbol>[A-Z0-9]{2,16})\s+(?P<timeframe>\d+[smhdw])\s*//\s*Bias\s+Flip\s*"
    r"\[(?P<bias>[^\]]+)\]\s*@\s*\$?(?P<entry>[\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)
DOMINO_RE = re.compile(
    r"(?P<marker>[\u25b2\u25b3\u25bc\u25bd\U0001f53a\U0001f53b])?\s*"
    r"(?P<symbol>[A-Z0-9]{2,16})\s*//\s*DOMINO\s*@\s*\$?(?P<entry>[\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)
TURN_RE = re.compile(r"(?P<timeframes>\d+[smhdw](?:\+\d+[smhdw])*)\s*<(?P<minutes>\d+)min\s*\[Turn\s+Detected\]", re.IGNORECASE)
MOVE_RE = re.compile(
    r"(?P<marker>[\u25b2\u25b3\u25bc\u25bd\U0001f53a\U0001f53b])?\s*"
    r"(?P<symbol>[A-Z0-9]{2,16})\s*\[(?P<timeframe>\d+[smhdw])\]\s*"
    r"(?P<pct>[+-]?\d+(?:\.\d+)?)%\s+Move\s*@\s*\$?(?P<entry>[\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)


@dataclass
class OpiSignal:
    source_id: str
    time: pd.Timestamp
    kind: str
    symbol: str
    timeframe: str
    direction: str
    entry: float
    target: float | None = None
    sl: float | None = None
    score: float | None = None
    from_bias: str = ""
    to_bias: str = ""
    tf_floor: str = ""
    multi_tf: str = ""
    next_event: str = ""
    raw_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["time"] = self.time.isoformat()
        return out


def parse_price(raw: str) -> float:
    return float(str(raw).replace(",", ""))


ALERT_TRANSLATION = str.maketrans(
    {
        "ᴼ": "O",
        "ᵒ": "o",
        "ᵖ": "p",
        "ᶦ": "i",
        "ᴸ": "L",
        "ᵛ": "v",
        "ˡ": "l",
        "ᶻ": "z",
        "⅂": "L",
        "ⅆ": "d",
        "ℤ": "Z",
        "★": "*",
        "×": "x",
        "·": " ",
        "→": " -> ",
        "➜": " -> ",
        "⟶": " -> ",
    }
)


def normalise_alert_text(text: str) -> str:
    return str(text).translate(ALERT_TRANSLATION)


def normalise_symbol(raw: str) -> str:
    symbol = raw.upper().replace(".P", "")
    if symbol in SHORT_ALIAS:
        return SHORT_ALIAS[symbol]
    if not symbol.endswith("USDT"):
        return f"{symbol}USDT"
    return symbol


def direction_from_bias(raw: str) -> str | None:
    value = str(raw or "").upper()
    if "SHORT" in value or "BEAR" in value:
        return "short"
    if "LONG" in value or "BULL" in value:
        return "long"
    return None


def split_bias_flip(raw: str) -> tuple[str, str, str | None]:
    text = str(raw or "").strip()
    states = re.findall(r"(?:LEAN\s+)?(?:SHORT|LONG|BEAR|BULL)", text, flags=re.IGNORECASE)
    to_bias = states[-1] if states else text
    from_bias = states[0] if len(states) > 1 else ""
    return from_bias, to_bias, direction_from_bias(to_bias)


def direction_from_marker(raw: str | None) -> str | None:
    if raw in {"\u25b2", "\u25b3", "\U0001f53a"}:
        return "long"
    if raw in {"\u25bc", "\u25bd", "\U0001f53b"}:
        return "short"
    return None


def flatten_telegram_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or ""))
        return "".join(parts)
    if isinstance(value, dict):
        return str(value.get("text") or "")
    return str(value)


def parse_time(value: Any) -> pd.Timestamp:
    if value is None or value == "":
        return pd.NaT
    if isinstance(value, (int, float)):
        # Telegram export timestamps are sometimes seconds.
        if value > 10_000_000_000:
            return pd.Timestamp(value, unit="ms", tz="UTC")
        return pd.Timestamp(value, unit="s", tz="UTC")
    text = str(value).strip()
    try:
        return pd.Timestamp(text, tz="UTC")
    except Exception:
        return pd.Timestamp(text).tz_localize("UTC")


def iter_export_messages(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(text)
        if isinstance(data, dict) and isinstance(data.get("messages"), list):
            messages = data["messages"]
        elif isinstance(data, list):
            messages = data
        else:
            messages = []
        out: list[dict[str, Any]] = []
        for idx, item in enumerate(messages):
            if not isinstance(item, dict):
                continue
            body = flatten_telegram_text(item.get("text") or item.get("caption"))
            out.append(
                {
                    "id": str(item.get("id") or idx),
                    "time": parse_time(item.get("date") or item.get("date_unixtime")),
                    "text": body,
                }
            )
        return out
    if suffix in {".jsonl", ".ndjson"}:
        out = []
        for idx, raw in enumerate(text.splitlines()):
            if not raw.strip():
                continue
            item = json.loads(raw)
            out.append(
                {
                    "id": str(item.get("id") or idx),
                    "time": parse_time(item.get("date") or item.get("time") or item.get("ts")),
                    "text": flatten_telegram_text(item.get("text") or item.get("body") or item.get("caption")),
                }
            )
        return out

    # Plain text exports: split on blank lines. If no explicit date exists,
    # the user can still use --default-date for parser smoke checks.
    blocks = re.split(r"\n\s*\n", html.unescape(text))
    return [{"id": str(i), "time": pd.NaT, "text": block.strip()} for i, block in enumerate(blocks) if block.strip()]


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
        qs = {"dir": "b", "limit": str(min(200, limit - len(out)))}
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
        for event in chunk:
            if event.get("type") != "m.room.message":
                continue
            body = flatten_telegram_text((event.get("content") or {}).get("body"))
            if not body.strip():
                continue
            ts_ms = int(event.get("origin_server_ts") or 0)
            out.append(
                {
                    "id": str(event.get("event_id") or len(out)),
                    "time": pd.Timestamp(ts_ms, unit="ms", tz="UTC"),
                    "text": body,
                }
            )
        next_token = data.get("end")
        if not next_token or next_token == from_token:
            break
        from_token = next_token
    return out[:limit]


def parse_opi_signal(message: dict[str, Any], default_date: pd.Timestamp | None = None) -> list[OpiSignal]:
    raw_text = html.unescape(str(message.get("text") or "")).replace("\xa0", " ")
    clean = re.sub(r"\r\n?", "\n", raw_text).strip()
    if not clean:
        return []
    match_text = normalise_alert_text(clean)
    ts = parse_time(message.get("time"))
    if pd.isna(ts) and default_date is not None:
        ts = default_date
    source_id = str(message.get("id") or "")
    out: list[OpiSignal] = []

    if full_m := OPI_FULL_RE.search(match_text):
        direction = full_m.group("direction").lower()
        target_m = TARGET_RE.search(match_text)
        sl_m = SL_RE.search(match_text)
        score_m = SCORE_RE.search(match_text)
        event_m = NEXT_EVENT_RE.search(match_text)
        floor_m = TF_FLOOR_RE.search(match_text)
        multi_tf_m = MULTI_TF_RE.search(match_text)
        out.append(
            OpiSignal(
                source_id=source_id,
                time=ts,
                kind="opi_full",
                symbol=normalise_symbol(full_m.group("symbol")),
                timeframe=full_m.group("timeframe").lower(),
                direction=direction,
                entry=parse_price(full_m.group("entry")),
                target=parse_price(target_m.group("target")) if target_m else None,
                sl=parse_price(sl_m.group("sl")) if sl_m else None,
                score=float(score_m.group("score")) if score_m else None,
                tf_floor=floor_m.group("floor").strip() if floor_m else "",
                multi_tf=multi_tf_m.group("multi_tf") if multi_tf_m else "",
                next_event=event_m.group("event").strip() if event_m else "",
                raw_text=clean,
            )
        )

    if bias_m := BIAS_FLIP_RE.search(match_text):
        from_bias, to_bias, direction = split_bias_flip(bias_m.group("bias"))
        if direction:
            out.append(
                OpiSignal(
                    source_id=source_id,
                    time=ts,
                    kind="bias_flip",
                    symbol=normalise_symbol(bias_m.group("symbol")),
                    timeframe=bias_m.group("timeframe").lower(),
                    direction=direction,
                    entry=parse_price(bias_m.group("entry")),
                    from_bias=from_bias,
                    to_bias=to_bias,
                    raw_text=clean,
                )
            )

    if domino_m := DOMINO_RE.search(match_text):
        direction = direction_from_marker(domino_m.group("marker"))
        turn_m = TURN_RE.search(match_text)
        if direction:
            out.append(
                OpiSignal(
                    source_id=source_id,
                    time=ts,
                    kind="domino_turn",
                    symbol=normalise_symbol(domino_m.group("symbol")),
                    timeframe=(turn_m.group("timeframes").split("+")[-1].lower() if turn_m else "1m"),
                    direction=direction,
                    entry=parse_price(domino_m.group("entry")),
                    multi_tf=turn_m.group("timeframes") if turn_m else "",
                    raw_text=clean,
                )
            )

    if move_m := MOVE_RE.search(match_text):
        direction = direction_from_marker(move_m.group("marker"))
        if direction:
            out.append(
                OpiSignal(
                    source_id=source_id,
                    time=ts,
                    kind="move_alert",
                    symbol=normalise_symbol(move_m.group("symbol")),
                    timeframe=move_m.group("timeframe").lower(),
                    direction=direction,
                    entry=parse_price(move_m.group("entry")),
                    raw_text=clean,
                )
            )
    return out


def first_index_after(frame: pd.DataFrame, ts: pd.Timestamp) -> int | None:
    idx = frame["open_time"].searchsorted(ts, side="right")
    if idx >= len(frame):
        return None
    return int(idx)


def validate_levels(direction: str, entry: float, sl: float, tp: float) -> bool:
    if direction == "long":
        return sl < entry < tp
    return tp < entry < sl


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
    exit_idx = end_idx
    exit_price = float(frame.iloc[end_idx]["close"])
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
            exit_idx = idx
            exit_price = sl
            outcome = "stop"
            break
        if target_hit:
            exit_idx = idx
            exit_price = tp
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


def ensure_symbol_frame(symbol: str, start: datetime, end: datetime, cache_dir: Path) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{symbol.lower()}_1m_{start:%Y%m%d%H%M}_{end:%Y%m%d%H%M}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    frame = fetch_bybit_klines(symbol, "1m", start, end)
    frame.to_pickle(path)
    return frame


def profit_factor(values: pd.Series) -> float:
    gains = float(values[values > 0].sum())
    losses = -float(values[values <= 0].sum())
    if losses <= 0:
        return math.inf if gains > 0 else 0.0
    return gains / losses


def summarize(trades: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, part in trades.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: value for col, value in zip(group_cols, key)}
        wins = int((part["net_r"] > 0).sum())
        row.update(
            {
                "trades": len(part),
                "net_r": float(part["net_r"].sum()),
                "avg_r": float(part["net_r"].mean()),
                "median_r": float(part["net_r"].median()),
                "winrate": wins / len(part) if len(part) else 0.0,
                "profit_factor": profit_factor(part["net_r"]),
                "fee_valid_rate": float((part["fee_to_stop_risk"] <= 0.25).mean()),
            }
        )
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["avg_r", "trades"], ascending=[False, False])


def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    default_date = pd.Timestamp(args.default_date, tz="UTC") if args.default_date else None
    signals: list[OpiSignal] = []
    source_messages = 0
    for path in args.input or []:
        messages = iter_export_messages(path)
        source_messages += len(messages)
        for message in messages:
            signals.extend(parse_opi_signal(message, default_date=default_date))
    if args.matrix_room_id:
        messages = fetch_matrix_messages(args.matrix_env_path, args.matrix_room_id, args.matrix_limit)
        source_messages += len(messages)
        for message in messages:
            signals.extend(parse_opi_signal(message, default_date=default_date))
    signals_df = pd.DataFrame([signal.to_dict() for signal in signals])
    signals_path = args.out_dir / "opi_curl_reversal_signals.csv"
    signals_df.to_csv(signals_path, index=False)
    print(f"messages={source_messages} signals={len(signals_df)} wrote={signals_path}", flush=True)
    if signals_df.empty:
        return

    signals_df["time"] = pd.to_datetime(signals_df["time"], utc=True, errors="coerce")
    signals_df = signals_df.dropna(subset=["time"])
    if signals_df.empty:
        print("No timestamped signals. Provide a Telegram JSON export or --default-date for parser smoke checks.", flush=True)
        return
    start = signals_df["time"].min().to_pydatetime() - timedelta(days=args.fetch_pad_days)
    end = signals_df["time"].max().to_pydatetime() + timedelta(days=args.fetch_pad_days)

    frames: dict[str, pd.DataFrame] = {}
    for symbol in sorted(signals_df["symbol"].dropna().unique()):
        try:
            frame = ensure_symbol_frame(symbol, start, end, args.cache_dir)
            frames[symbol] = frame
            print(f"{symbol}: candles={len(frame)} {frame['open_time'].min()} -> {frame['open_time'].max()}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"{symbol}: candle fetch failed: {exc}", flush=True)

    trades: list[dict[str, Any]] = []
    for signal in signals_df.itertuples(index=False):
        frame = frames.get(str(signal.symbol))
        if frame is None or frame.empty:
            continue
        entry_idx = first_index_after(frame, pd.Timestamp(signal.time))
        if entry_idx is None:
            continue
        next_open = float(frame.iloc[entry_idx]["open"])
        base = {
            "source_id": signal.source_id,
            "time": signal.time,
            "kind": signal.kind,
            "symbol": signal.symbol,
            "timeframe": signal.timeframe,
            "direction": signal.direction,
            "score": signal.score,
            "from_bias": signal.from_bias,
            "to_bias": signal.to_bias,
            "multi_tf": signal.multi_tf,
            "tf_floor": signal.tf_floor,
            "next_event": signal.next_event,
        }

        if pd.notna(signal.sl) and pd.notna(signal.target):
            for variant, entry in [("declared_entry_levels", float(signal.entry)), ("next_open_levels", next_open)]:
                sl = float(signal.sl)
                tp = float(signal.target)
                if not validate_levels(str(signal.direction), entry, sl, tp):
                    continue
                result = backtest_trade(
                    frame,
                    entry_idx=entry_idx,
                    direction=str(signal.direction),
                    entry=entry,
                    sl=sl,
                    tp=tp,
                    max_hold_bars=args.max_hold_bars,
                    fee_rate=args.fee_rate,
                )
                if "skip_reason" not in result:
                    trades.append({**base, **result, "variant": variant, "entry": entry, "sl": sl, "tp": tp})

        for lookback in args.lookbacks:
            for target_r in args.target_rs:
                levels = structural_levels(frame, entry_idx, str(signal.direction), lookback, target_r)
                if not levels:
                    continue
                sl, tp = levels
                result = backtest_trade(
                    frame,
                    entry_idx=entry_idx,
                    direction=str(signal.direction),
                    entry=next_open,
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
                            "variant": f"next_open_swing{lookback}_tp{target_r:g}R",
                            "entry": next_open,
                            "sl": sl,
                            "tp": tp,
                            "lookback": lookback,
                            "target_r": target_r,
                        }
                    )

    trades_df = pd.DataFrame(trades)
    trades_path = args.out_dir / "opi_curl_reversal_backtest_trades.csv"
    trades_df.to_csv(trades_path, index=False)
    print(f"trades={len(trades_df)} wrote={trades_path}", flush=True)
    if trades_df.empty:
        return

    for group_cols, name in [
        (["variant"], "summary_by_variant"),
        (["variant", "kind"], "summary_by_variant_kind"),
        (["variant", "symbol"], "summary_by_variant_symbol"),
        (["variant", "symbol", "timeframe"], "summary_by_variant_symbol_timeframe"),
    ]:
        summary_df = summarize(trades_df, group_cols)
        summary_df.to_csv(args.out_dir / f"opi_curl_reversal_{name}.csv", index=False)
        print(f"\n{name}", flush=True)
        print(summary_df.head(20).to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="*", default=[], help="Telegram JSON export, JSONL, or text dump")
    parser.add_argument("--matrix-room-id", default="", help=f"Matrix room ID to fetch directly, e.g. {DEFAULT_OPI_MATRIX_ROOM_ID}")
    parser.add_argument("--matrix-env-path", type=Path, default=Path("bot/.env.matrix"))
    parser.add_argument("--matrix-limit", type=int, default=3000)
    parser.add_argument("--out-dir", type=Path, default=Path("scripts/opi_curl_reversal_investigation"))
    parser.add_argument("--cache-dir", type=Path, default=Path("scripts/.cache/opi_curl_reversals"))
    parser.add_argument("--default-date", default="", help="UTC fallback date for plain text parser smoke tests")
    parser.add_argument("--fetch-pad-days", type=float, default=1.0)
    parser.add_argument("--max-hold-bars", type=int, default=360)
    parser.add_argument("--fee-rate", type=float, default=0.00055)
    parser.add_argument("--lookbacks", type=int, nargs="+", default=[10, 20, 40, 80])
    parser.add_argument("--target-rs", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
