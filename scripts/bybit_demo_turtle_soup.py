from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_FLOOR, ROUND_HALF_UP
from pathlib import Path
from queue import Empty, Queue
from typing import Any
from urllib.parse import urlencode

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backtest_turtle_soup import Config, INTERVAL_MS, normalize_timeframe, run_backtest
from scripts.turtle_soup_candidate_presets import apply_preset_args, get_preset, preset_names, preset_summary


DEFAULT_BASE_URL = "https://api-demo.bybit.com"
BYBIT_INTERVALS = {
    "1m": "1",
    "3m": "3",
    "5m": "5",
    "15m": "15",
    "1h": "60",
    "4h": "240",
    "1d": "D",
    "1w": "W",
}
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


@dataclass(frozen=True)
class Instrument:
    symbol: str
    tick_size: Decimal
    qty_step: Decimal
    min_qty: Decimal
    min_notional: Decimal


@dataclass(frozen=True)
class EntrySignal:
    symbol: str
    direction: str
    order_type: str
    entry_price: Decimal
    stop_price: Decimal
    target_price: Decimal
    signal_time: pd.Timestamp
    submitted_index: int
    zone_tf: str
    zone_top: float
    zone_bottom: float
    zone_hold_prob: float

    @property
    def risk_per_coin(self) -> Decimal:
        return abs(self.entry_price - self.stop_price)


def load_env_file(path: Path | None) -> None:
    if path is None or not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def clean_params(params: dict[str, Any] | None) -> dict[str, Any]:
    if not params:
        return {}
    return {key: value for key, value in params.items() if value is not None}


class BybitV5Client:
    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        api_secret: str | None = None,
        recv_window: int = 20_000,
        timeout: int = 30,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or ""
        self.api_secret = api_secret or ""
        self.recv_window = str(recv_window)
        self.timeout = timeout
        self.session = requests.Session()

    @property
    def has_credentials(self) -> bool:
        return bool(self.api_key and self.api_secret)

    def _request(self, method: str, path: str, params: dict[str, Any] | None = None, signed: bool = False) -> dict[str, Any]:
        method = method.upper()
        params = clean_params(params)
        query = urlencode(sorted(params.items()), doseq=True)
        body = ""
        headers = {"Content-Type": "application/json"}
        url = f"{self.base_url}{path}"

        if method == "GET":
            if query:
                url = f"{url}?{query}"
            signature_payload = query
            data = None
        else:
            body = json.dumps(params, separators=(",", ":"), ensure_ascii=False)
            signature_payload = body
            data = body

        if signed:
            if not self.has_credentials:
                raise RuntimeError("Bybit credentials are required for private endpoints.")
            timestamp = str(int(time.time() * 1000))
            raw_sign = f"{timestamp}{self.api_key}{self.recv_window}{signature_payload}"
            signature = hmac.new(self.api_secret.encode("utf-8"), raw_sign.encode("utf-8"), hashlib.sha256).hexdigest()
            headers.update(
                {
                    "X-BAPI-API-KEY": self.api_key,
                    "X-BAPI-TIMESTAMP": timestamp,
                    "X-BAPI-RECV-WINDOW": self.recv_window,
                    "X-BAPI-SIGN": signature,
                    "X-BAPI-SIGN-TYPE": "2",
                }
            )

        response = self.session.request(method, url, headers=headers, data=data, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        ret_code = payload.get("retCode", 0)
        if ret_code not in (0, "0"):
            ret_msg = payload.get("retMsg", "unknown Bybit error")
            raise RuntimeError(f"Bybit {path} failed: retCode={ret_code} retMsg={ret_msg}")
        return payload.get("result", {})

    def get_public(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._request("GET", path, params=params, signed=False)

    def get_private(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._request("GET", path, params=params, signed=True)

    def post_private(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", path, params=params, signed=True)

    def instrument(self, symbol: str) -> Instrument:
        result = self.get_public("/v5/market/instruments-info", {"category": "linear", "symbol": symbol})
        instruments = result.get("list", [])
        if not instruments:
            raise RuntimeError(f"No Bybit linear instrument found for {symbol}.")
        info = instruments[0]
        lot = info.get("lotSizeFilter", {})
        price = info.get("priceFilter", {})
        return Instrument(
            symbol=symbol,
            tick_size=Decimal(str(price.get("tickSize", "0.01"))),
            qty_step=Decimal(str(lot.get("qtyStep", "0.001"))),
            min_qty=Decimal(str(lot.get("minOrderQty", "0"))),
            min_notional=Decimal(str(lot.get("minNotionalValue", "0"))),
        )

    def open_orders(self, symbol: str, order_link_prefix: str) -> list[dict[str, Any]]:
        result = self.get_private(
            "/v5/order/realtime",
            {"category": "linear", "symbol": symbol, "openOnly": 0, "limit": 50},
        )
        rows = result.get("list", [])
        return [row for row in rows if str(row.get("orderLinkId", "")).startswith(order_link_prefix)]

    def positions(self, symbol: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"category": "linear", "settleCoin": "USDT"}
        if symbol:
            params["symbol"] = symbol
            params.pop("settleCoin", None)
        result = self.get_private("/v5/position/list", params)
        positions = result.get("list", [])
        return [row for row in positions if Decimal(str(row.get("size", "0") or "0")) != 0]

    def wallet_balance(self, account_type: str) -> dict[str, Any]:
        return self.get_private("/v5/account/wallet-balance", {"accountType": account_type})

    def create_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.post_private("/v5/order/create", payload)

    def cancel_order(self, symbol: str, order_link_id: str) -> dict[str, Any]:
        return self.post_private(
            "/v5/order/cancel",
            {"category": "linear", "symbol": symbol, "orderLinkId": order_link_id},
        )


def decimal_to_str(value: Decimal) -> str:
    normalized = value.normalize()
    text = format(normalized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def quantize_to_step(value: Decimal, step: Decimal, rounding=ROUND_FLOOR) -> Decimal:
    if step <= 0:
        return value
    return (value / step).to_integral_value(rounding=rounding) * step


def quantize_price(value: Decimal, instrument: Instrument) -> Decimal:
    return quantize_to_step(value, instrument.tick_size, ROUND_HALF_UP)


def decimal_or_zero(value: Any) -> Decimal:
    if value in (None, ""):
        return Decimal("0")
    return Decimal(str(value))


def bybit_symbol(raw: str) -> str:
    symbol = raw.strip().upper()
    if ":" in symbol:
        symbol = symbol.split(":", 1)[1]
    return symbol.replace("/", "").replace("-", "").replace(".P", "")


def default_model_path() -> Path:
    live = Path("scripts/zone_hold_model_majors20_1h_live.joblib")
    if live.exists():
        return live
    return Path("scripts/zone_hold_model_majors20_1h_pilot.joblib")


def fetch_bybit_klines(
    client: BybitV5Client,
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    interval = normalize_timeframe(interval)
    bybit_interval = BYBIT_INTERVALS[interval]
    interval_ms = INTERVAL_MS[interval]
    start_ms = int(start.timestamp() * 1000)
    cursor_end_ms = int(end.timestamp() * 1000)
    rows: dict[int, list[Any]] = {}

    while cursor_end_ms >= start_ms:
        result = client.get_public(
            "/v5/market/kline",
            {
                "category": "linear",
                "symbol": symbol,
                "interval": bybit_interval,
                "start": start_ms,
                "end": cursor_end_ms,
                "limit": 1000,
            },
        )
        batch = result.get("list", [])
        if not batch:
            break
        oldest = min(int(row[0]) for row in batch)
        for row in batch:
            rows[int(row[0])] = row
        if len(batch) < 1000 or oldest <= start_ms:
            break
        cursor_end_ms = oldest - 1
        time.sleep(0.05)

    if not rows:
        raise RuntimeError(f"No Bybit klines returned for {symbol} {interval}.")

    frame = pd.DataFrame(
        [rows[key] for key in sorted(rows)],
        columns=["open_time_ms", "open", "high", "low", "close", "volume", "turnover"],
    )
    frame["open_time"] = pd.to_datetime(frame["open_time_ms"].astype("int64"), unit="ms", utc=True)
    frame["close_time"] = frame["open_time"] + pd.Timedelta(milliseconds=interval_ms - 1)
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = frame[column].astype(float)

    now = pd.Timestamp.now(tz="UTC")
    closed = frame[frame["open_time"] + pd.Timedelta(milliseconds=interval_ms) <= now].copy()
    return closed[["open_time", "close_time", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def bybit_websocket_interval(interval: str) -> int | str:
    value = BYBIT_INTERVALS[normalize_timeframe(interval)]
    return int(value) if value.isdigit() else value


def candle_row_from_ws(symbol: str, interval: str, candle: dict[str, Any]) -> dict[str, Any] | None:
    if not candle.get("confirm", False):
        return None
    open_time = pd.to_datetime(int(candle["start"]), unit="ms", utc=True)
    close_time = open_time + pd.Timedelta(milliseconds=INTERVAL_MS[normalize_timeframe(interval)] - 1)
    return {
        "symbol": symbol,
        "open_time": open_time,
        "close_time": close_time,
        "open": float(candle["open"]),
        "high": float(candle["high"]),
        "low": float(candle["low"]),
        "close": float(candle["close"]),
        "volume": float(candle["volume"]),
    }


def append_closed_candle(df: pd.DataFrame, row: dict[str, Any], lookback_days: int) -> pd.DataFrame:
    candle = pd.DataFrame([{key: value for key, value in row.items() if key != "symbol"}])
    out = pd.concat([df, candle], ignore_index=True)
    out = out.drop_duplicates(subset=["open_time"], keep="last").sort_values("open_time").reset_index(drop=True)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=lookback_days)
    return out[out["open_time"] >= cutoff].reset_index(drop=True)


def build_strategy_config(args: argparse.Namespace, instrument: Instrument) -> Config:
    return Config(
        exec_tf=args.interval,
        structure_tf=args.structure_tf,
        entry_mode="zone_retest",
        tf1=args.zone_tf,
        tf2="1d",
        use_tf1=True,
        use_tf2=False,
        block_dead_zone=False,
        max_structure_bars_to_choch=32,
        htf_left=args.htf_left,
        htf_right=args.htf_right,
        htf_ob_search_bars=args.htf_ob_search_bars,
        max_zone_scan=args.max_zone_scan,
        zone_hold_model_path=None if args.no_ml_filter else str(args.model),
        zone_hold_min_prob=0.0 if args.no_ml_filter else args.zone_hold_min_prob,
        zone_hold_filter_tf=args.zone_tf,
        zone_hold_reject_unscored=True,
        mintick=float(instrument.tick_size),
    )


def format_prob(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "nan" if math.isnan(numeric) else f"{numeric:.3f}"


def serialize_zone_hold_candidate(candidate: dict[str, Any] | None, max_signal_age: timedelta) -> dict[str, Any] | None:
    if not candidate:
        return None
    out = dict(candidate)
    sweep_time = candidate.get("sweep_time")
    if sweep_time is not None:
        sweep_ts = pd.Timestamp(sweep_time)
        if sweep_ts.tzinfo is None:
            sweep_ts = sweep_ts.tz_localize("UTC")
        age = pd.Timestamp.now(tz="UTC") - sweep_ts
        out["sweep_time"] = sweep_ts.isoformat()
        out["sweep_age_minutes"] = round(age.total_seconds() / 60.0, 3)
        out["sweep_is_recent"] = age <= pd.Timedelta(max_signal_age)
    return out


def describe_zone_hold_candidate(candidate: dict[str, Any] | None, max_signal_age: timedelta) -> str | None:
    details = serialize_zone_hold_candidate(candidate, max_signal_age)
    if not details:
        return None

    direction = str(details.get("direction", "setup")).upper()
    prob = format_prob(details.get("zone_hold_prob"))
    threshold = details.get("zone_hold_threshold")
    threshold_text = "n/a" if threshold is None else f"{float(threshold):.3f}"
    sweep_time = str(details.get("sweep_time", "unknown"))
    stale_suffix = ""
    age_minutes = details.get("sweep_age_minutes")
    if age_minutes is not None and not details.get("sweep_is_recent", False):
        stale_suffix = f" [stale {float(age_minutes):.1f}m]"

    reason = str(details.get("zone_hold_reason", "unknown"))
    if reason == "prob_below_threshold":
        return f"latest raw {direction} setup rejected by ML ({prob} < {threshold_text}) at {sweep_time}{stale_suffix}"
    if reason == "prob_above_threshold":
        return f"latest raw {direction} setup passed ML ({prob} >= {threshold_text}) at {sweep_time}{stale_suffix}"
    if reason == "features_unavailable":
        return f"latest raw {direction} setup rejected because ML features were unavailable at {sweep_time}{stale_suffix}"
    if reason == "filter_tf_mismatch":
        return f"latest raw {direction} setup rejected because zone tf did not match the ML filter tf at {sweep_time}{stale_suffix}"
    if reason == "no_model":
        return f"latest raw {direction} setup observed without ML filtering at {sweep_time}{stale_suffix}"
    return f"latest raw {direction} setup ended with zone-hold status '{reason}' at {sweep_time}{stale_suffix}"


def latest_signal(
    symbol: str,
    df: pd.DataFrame,
    cfg: Config,
    max_signal_age: timedelta,
) -> tuple[EntrySignal | None, str, dict[str, Any]]:
    diagnostics: dict[str, Any] = {
        "closed_historical_trades": 0,
        "has_pending_entry": False,
        "historical_position_open": False,
        "latest_zone_hold_candidate": None,
    }
    if df.empty:
        return None, "no closed candles", diagnostics
    trades, state = run_backtest(df, cfg, return_state=True)
    diagnostics["closed_historical_trades"] = len(trades)
    diagnostics["latest_zone_hold_candidate"] = serialize_zone_hold_candidate(
        state.get("latest_zone_hold_candidate"),
        max_signal_age,
    )
    pending = state.get("pending_entry")
    if state.get("position") is not None:
        diagnostics["historical_position_open"] = True
        return None, "strategy already has an open historical position", diagnostics
    if pending is None:
        reason = f"no pending entry after {len(trades)} closed historical trades"
        candidate_note = describe_zone_hold_candidate(state.get("latest_zone_hold_candidate"), max_signal_age)
        if candidate_note:
            reason = f"{reason}; {candidate_note}"
        return None, reason, diagnostics

    diagnostics["has_pending_entry"] = True
    signal_time = pd.Timestamp(pending["signal_time"])
    signal_age = pd.Timestamp.now(tz="UTC") - signal_time
    if signal_age > pd.Timedelta(max_signal_age):
        return None, f"latest pending entry is stale ({signal_age})", diagnostics

    direction = pending["direction"]
    entry_float = pending["entry_price"]
    if entry_float is None:
        return None, "latest pending entry is a market entry, but this live bridge only arms zone-retest limits", diagnostics
    entry = Decimal(str(entry_float))
    stop = Decimal(str(pending["stop"]))
    risk = abs(entry - stop)
    if risk <= 0:
        return None, "latest pending entry has no positive risk", diagnostics
    target = entry + risk * Decimal(str(cfg.target_rr)) if direction == "long" else entry - risk * Decimal(str(cfg.target_rr))
    setup = pending["setup"]
    return EntrySignal(
        symbol=symbol,
        direction=direction,
        order_type=pending["order_type"],
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        signal_time=signal_time,
        submitted_index=int(pending["submitted_index"]),
        zone_tf=str(setup["zone_tf"]),
        zone_top=float(setup["zone_top"]),
        zone_bottom=float(setup["zone_bottom"]),
        zone_hold_prob=float(setup.get("zone_hold_prob", math.nan)),
    ), "fresh pending entry", diagnostics


def order_link_id(prefix: str, signal: EntrySignal) -> str:
    raw = (
        f"{signal.symbol}|{signal.direction}|{signal.signal_time.isoformat()}|"
        f"{signal.entry_price}|{signal.stop_price}|{signal.target_price}"
    )
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    side = "L" if signal.direction == "long" else "S"
    return f"{prefix}-{signal.symbol[:4]}-{side}-{digest}"[:36]


def position_idx(args: argparse.Namespace, signal: EntrySignal) -> int:
    if args.position_mode == "hedge":
        return 1 if signal.direction == "long" else 2
    return 0


def build_order_payload(
    args: argparse.Namespace,
    client: BybitV5Client,
    signal: EntrySignal,
    instrument: Instrument,
) -> tuple[dict[str, Any] | None, str]:
    entry = quantize_price(signal.entry_price, instrument)
    stop = quantize_price(signal.stop_price, instrument)
    target = quantize_price(signal.target_price, instrument)
    risk = abs(entry - stop)
    if risk <= 0:
        return None, "risk rounded to zero"

    risk_budget, risk_source = resolve_risk_budget(args, client)
    if risk_budget is None:
        return None, risk_source
    if risk_budget <= 0:
        return None, f"risk budget must be positive, got {decimal_to_str(risk_budget)} USDT"

    risk_qty = risk_budget / risk
    max_notional_qty = Decimal(str(args.max_notional_usdt)) / entry if args.max_notional_usdt > 0 else risk_qty
    qty = quantize_to_step(min(risk_qty, max_notional_qty), instrument.qty_step, ROUND_FLOOR)
    if qty < instrument.min_qty:
        return None, f"qty {decimal_to_str(qty)} is below min_qty {decimal_to_str(instrument.min_qty)}"
    notional = qty * entry
    if instrument.min_notional > 0 and notional < instrument.min_notional:
        return None, f"notional {decimal_to_str(notional)} is below min_notional {decimal_to_str(instrument.min_notional)}"

    payload = {
        "category": "linear",
        "symbol": signal.symbol,
        "side": "Buy" if signal.direction == "long" else "Sell",
        "orderType": "Limit",
        "qty": decimal_to_str(qty),
        "price": decimal_to_str(entry),
        "timeInForce": "GTC",
        "positionIdx": position_idx(args, signal),
        "orderLinkId": order_link_id(args.order_link_prefix, signal),
        "takeProfit": decimal_to_str(target),
        "stopLoss": decimal_to_str(stop),
        "tpTriggerBy": "LastPrice",
        "slTriggerBy": "LastPrice",
        "tpslMode": "Full",
        "tpOrderType": "Market",
        "slOrderType": "Market",
        "reduceOnly": False,
        "closeOnTrigger": False,
    }
    actual_risk = qty * risk
    cap_note = " capped by max_notional" if qty < risk_qty else ""
    return (
        payload,
        (
            f"qty={decimal_to_str(qty)} notional={decimal_to_str(notional)} "
            f"risk={decimal_to_str(actual_risk)} USDT budget={decimal_to_str(risk_budget)} USDT ({risk_source}){cap_note}"
        ),
    )


def summarize_signal(signal: EntrySignal) -> str:
    prob = "nan" if math.isnan(signal.zone_hold_prob) else f"{signal.zone_hold_prob:.3f}"
    return (
        f"{signal.symbol} {signal.direction.upper()} limit={signal.entry_price} "
        f"stop={signal.stop_price} target={signal.target_price} prob={prob} "
        f"signal={signal.signal_time.isoformat()}"
    )


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=str, sort_keys=True) + "\n")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_file(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(row, default=str, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def initial_heartbeat(args: argparse.Namespace, mode: str) -> dict[str, Any]:
    now = utc_now_iso()
    return {
        "pid": os.getpid(),
        "started_at": now,
        "last_update": now,
        "last_event": "starting",
        "mode": mode,
        "execute": bool(args.execute),
        "dry_run": not bool(args.execute),
        "symbols": list(args.symbols),
        "interval": args.interval,
        "structure_tf": args.structure_tf,
        "zone_tf": args.zone_tf,
        "risk_pct": args.risk_pct,
        "risk_usdt": args.risk_usdt,
        "max_notional_usdt": args.max_notional_usdt,
        "order_link_prefix": args.order_link_prefix,
        "log_jsonl": str(args.log_jsonl),
        "last_websocket_event_time": None,
        "last_idle_time": None,
        "last_candle_by_symbol": {},
        "last_status_by_symbol": {},
        "last_reconciliation_by_symbol": {},
        "error_count": 0,
        "last_error": None,
    }


def update_heartbeat(args: argparse.Namespace, state: dict[str, Any], event: str, **updates: Any) -> None:
    state["last_update"] = utc_now_iso()
    state["last_event"] = event
    state.update(updates)
    write_json_file(args.heartbeat_json, state)


def has_private_access(args: argparse.Namespace, client: BybitV5Client) -> bool:
    return client.has_credentials and not args.public_only


def account_equity_usdt(client: BybitV5Client, account_type: str) -> Decimal:
    result = client.wallet_balance(account_type)
    accounts = result.get("list", [])
    if not accounts:
        raise RuntimeError("No wallet account returned by Bybit.")
    account = accounts[0]
    for key in ("totalEquity", "totalWalletBalance", "totalMarginBalance"):
        equity = decimal_or_zero(account.get(key))
        if equity > 0:
            return equity
    raise RuntimeError("Could not read a positive account equity from wallet-balance.")


def resolve_risk_budget(args: argparse.Namespace, client: BybitV5Client) -> tuple[Decimal | None, str]:
    if args.risk_pct is None:
        return Decimal(str(args.risk_usdt)), f"fixed {Decimal(str(args.risk_usdt))} USDT"
    if not has_private_access(args, client):
        return None, "--risk-pct requires private account access; use --risk-usdt in public-only mode"
    equity = account_equity_usdt(client, args.account_type)
    risk = equity * Decimal(str(args.risk_pct)) / Decimal("100")
    return risk, f"{Decimal(str(args.risk_pct))}% of {decimal_to_str(equity)} USDT equity"


def stale_order_age_ms(order: dict[str, Any]) -> int | None:
    created = order.get("createdTime") or order.get("updatedTime")
    if created in (None, ""):
        return None
    try:
        return int(time.time() * 1000) - int(created)
    except (TypeError, ValueError):
        return None


def is_stale_order(args: argparse.Namespace, order: dict[str, Any]) -> bool:
    if args.stale_order_candles <= 0:
        return False
    age_ms = stale_order_age_ms(order)
    if age_ms is None:
        return False
    stale_ms = args.stale_order_candles * INTERVAL_MS[args.interval]
    return age_ms >= stale_ms


def reconcile_symbol(args: argparse.Namespace, client: BybitV5Client, symbol: str) -> dict[str, Any]:
    state: dict[str, Any] = {
        "private": has_private_access(args, client),
        "open_positions": [],
        "active_orders": [],
        "stale_orders": [],
        "canceled_orders": [],
        "would_cancel_orders": [],
    }
    if not state["private"]:
        return state

    positions = client.positions(symbol)
    open_orders = client.open_orders(symbol, args.order_link_prefix)
    stale_orders = [order for order in open_orders if is_stale_order(args, order)]
    active_orders = [order for order in open_orders if order not in stale_orders]
    state["open_positions"] = positions
    state["stale_orders"] = stale_orders

    if args.cancel_stale:
        for order in stale_orders:
            link_id = str(order.get("orderLinkId", ""))
            if not link_id:
                continue
            if args.execute:
                client.cancel_order(symbol, link_id)
                state["canceled_orders"].append(link_id)
                print(f"{symbol}: canceled stale {args.order_link_prefix} orderLinkId={link_id}")
            else:
                state["would_cancel_orders"].append(link_id)
                print(f"{symbol}: dry-run would cancel stale {args.order_link_prefix} orderLinkId={link_id}")
    else:
        active_orders = open_orders

    state["active_orders"] = active_orders
    return state


def process_symbol_frame(
    args: argparse.Namespace,
    client: BybitV5Client,
    symbol: str,
    df: pd.DataFrame,
    instrument: Instrument,
) -> dict[str, Any]:
    reconciliation = reconcile_symbol(args, client, symbol)
    cfg = build_strategy_config(args, instrument)
    signal, reason, diagnostics = latest_signal(symbol, df, cfg, timedelta(minutes=args.max_signal_age_minutes))

    result: dict[str, Any] = {
        "time": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "bars": len(df),
        "status": reason,
        "dry_run": not args.execute,
        "reconciliation": {
            "private": reconciliation["private"],
            "open_positions": len(reconciliation["open_positions"]),
            "active_orders": len(reconciliation["active_orders"]),
            "stale_orders": len(reconciliation["stale_orders"]),
            "canceled_orders": reconciliation["canceled_orders"],
            "would_cancel_orders": reconciliation["would_cancel_orders"],
        },
    }
    if diagnostics:
        result["strategy_diagnostics"] = diagnostics
    if reconciliation["open_positions"]:
        result["status"] = "skipped: open exchange position exists"
        print(f"{symbol}: skipped: open exchange position exists")
        return result
    if reconciliation["active_orders"]:
        link_ids = [str(order.get("orderLinkId", "")) for order in reconciliation["active_orders"]]
        result["status"] = "skipped: active strategy order exists"
        result["active_order_link_ids"] = link_ids
        print(f"{symbol}: skipped: active {args.order_link_prefix} order exists ({', '.join(link_ids)})")
        return result

    if signal is None:
        print(f"{symbol}: {reason}")
        return result

    payload, order_note = build_order_payload(args, client, signal, instrument)
    result.update(
        {
            "signal": summarize_signal(signal),
            "order_note": order_note,
            "order_payload": payload,
        }
    )
    print(f"{symbol}: {summarize_signal(signal)}")
    print(f"{symbol}: {order_note}")

    if payload is None:
        result["status"] = f"skipped: {order_note}"
        print(f"{symbol}: skipped: {order_note}")
        return result

    if not client.has_credentials or args.public_only:
        result["status"] = "dry-run public mode; order not sent"
        print(f"{symbol}: dry-run/public mode; order would use orderLinkId={payload['orderLinkId']}")
        return result

    if args.execute:
        response = client.create_order(payload)
        result["status"] = "submitted"
        result["bybit_result"] = response
        print(f"{symbol}: submitted orderLinkId={payload['orderLinkId']}")
    else:
        result["status"] = "dry-run; order not sent"
        print(f"{symbol}: dry-run; order would use orderLinkId={payload['orderLinkId']}")
    return result


def process_symbol(args: argparse.Namespace, client: BybitV5Client, symbol: str) -> dict[str, Any]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.lookback_days)
    instrument = client.instrument(symbol)
    df = fetch_bybit_klines(client, symbol, args.interval, start, end)
    return process_symbol_frame(args, client, symbol, df, instrument)


def bootstrap_symbol_frames(
    args: argparse.Namespace,
    client: BybitV5Client,
) -> tuple[dict[str, Instrument], dict[str, pd.DataFrame]]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.lookback_days)
    instruments: dict[str, Instrument] = {}
    frames: dict[str, pd.DataFrame] = {}
    for symbol in args.symbols:
        instruments[symbol] = client.instrument(symbol)
        frames[symbol] = fetch_bybit_klines(client, symbol, args.interval, start, end)
        last_open = frames[symbol]["open_time"].iloc[-1] if not frames[symbol].empty else "n/a"
        print(f"{symbol}: bootstrapped {len(frames[symbol])} closed {args.interval} candles through {last_open}")
    return instruments, frames


def account_check(args: argparse.Namespace, client: BybitV5Client) -> None:
    if args.check_account and client.has_credentials and not args.public_only:
        balance = client.wallet_balance(args.account_type)
        positions = client.positions()
        print(f"Account check: wallet entries={len(balance.get('list', []))} open_positions={len(positions)}")


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


def run_websocket_loop(args: argparse.Namespace, client: BybitV5Client) -> None:
    try:
        from pybit.unified_trading import WebSocket
    except ImportError as exc:
        raise SystemExit("pybit is required for --loop-mode websocket. Install pybit in the active venv.") from exc

    heartbeat = initial_heartbeat(args, "websocket")
    update_heartbeat(args, heartbeat, "bootstrapping")
    instruments, frames = bootstrap_symbol_frames(args, client)
    heartbeat["last_candle_by_symbol"] = {
        symbol: (frames[symbol]["open_time"].iloc[-1].isoformat() if not frames[symbol].empty else None)
        for symbol in args.symbols
    }
    update_heartbeat(args, heartbeat, "bootstrapped")
    account_check(args, client)
    if args.scan_on_start:
        for symbol in args.symbols:
            row = process_symbol_frame(args, client, symbol, frames[symbol], instruments[symbol])
            heartbeat["last_status_by_symbol"][symbol] = row.get("status")
            heartbeat["last_reconciliation_by_symbol"][symbol] = row.get("reconciliation")
            update_heartbeat(args, heartbeat, f"startup_scan_{symbol}")
            write_jsonl(args.log_jsonl, row)

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
    print(
        f"WebSocket kline stream active interval={args.interval} symbols={','.join(args.symbols)} "
        f"demo={args.websocket_demo} execute={args.execute}"
    )
    update_heartbeat(args, heartbeat, "websocket_active")

    processed = 0
    try:
        while True:
            try:
                candle = event_queue.get(timeout=args.websocket_idle_timeout)
            except Empty:
                print(f"WebSocket idle: no confirmed {args.interval} candles in {args.websocket_idle_timeout}s")
                heartbeat["last_idle_time"] = utc_now_iso()
                update_heartbeat(args, heartbeat, "websocket_idle")
                continue

            key = (str(candle["symbol"]), pd.Timestamp(candle["open_time"]))
            if key in seen_candles:
                continue
            seen_candles.add(key)
            symbol = key[0]
            frames[symbol] = append_closed_candle(frames[symbol], candle, args.lookback_days)
            print(f"{symbol}: confirmed {args.interval} candle {pd.Timestamp(candle['open_time']).isoformat()}")
            heartbeat["last_websocket_event_time"] = utc_now_iso()
            heartbeat["last_candle_by_symbol"][symbol] = pd.Timestamp(candle["open_time"]).isoformat()
            try:
                row = process_symbol_frame(args, client, symbol, frames[symbol], instruments[symbol])
                heartbeat["last_status_by_symbol"][symbol] = row.get("status")
                heartbeat["last_reconciliation_by_symbol"][symbol] = row.get("reconciliation")
                update_heartbeat(args, heartbeat, f"processed_{symbol}")
            except Exception as exc:
                row = {
                    "time": datetime.now(timezone.utc).isoformat(),
                    "symbol": symbol,
                    "status": "error",
                    "error": str(exc),
                    "dry_run": not args.execute,
                }
                heartbeat["error_count"] = int(heartbeat.get("error_count", 0)) + 1
                heartbeat["last_error"] = row
                update_heartbeat(args, heartbeat, f"error_{symbol}")
                print(f"{symbol}: ERROR {exc}")
            write_jsonl(args.log_jsonl, row)
            processed += 1
            if args.websocket_stop_after_events > 0 and processed >= args.websocket_stop_after_events:
                break
    finally:
        ws.exit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bybit demo execution bridge for the Python Turtle Soup strategy.")
    parser.add_argument("--candidate-preset", choices=preset_names())
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--structure-tf", default="15m")
    parser.add_argument("--zone-tf", default="1h")
    parser.add_argument("--lookback-days", type=int, default=120)
    parser.add_argument("--model", type=Path, default=default_model_path())
    parser.add_argument("--no-ml-filter", action="store_true")
    parser.add_argument("--zone-hold-min-prob", type=float, default=0.60)
    parser.add_argument("--max-zone-scan", type=int, default=250)
    parser.add_argument("--max-signal-age-minutes", type=float, default=15.0)
    parser.add_argument("--htf-left", type=int, default=5)
    parser.add_argument("--htf-right", type=int, default=5)
    parser.add_argument("--htf-ob-search-bars", type=int, default=50)
    parser.add_argument("--risk-usdt", type=float, default=10.0)
    parser.add_argument("--risk-pct", type=float, help="Risk this percent of account equity per trade; overrides --risk-usdt.")
    parser.add_argument("--max-notional-usdt", type=float, default=500.0)
    parser.add_argument("--position-mode", choices=["one_way", "hedge"], default="hedge")
    parser.add_argument("--order-link-prefix", default="TSOUP")
    parser.add_argument("--base-url", default=os.environ.get("BYBIT_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--env-file", type=Path, default=Path("scripts/bybit_demo.env"))
    parser.add_argument("--account-type", default="UNIFIED")
    parser.add_argument("--public-only", action="store_true", help="Do not call private endpoints even if credentials are present.")
    parser.add_argument("--check-account", action="store_true", help="Fetch wallet/positions before scanning signals.")
    parser.add_argument("--cancel-stale", action=argparse.BooleanOptionalAction, default=True, help="Cancel stale open orders created by this script prefix.")
    parser.add_argument("--stale-order-candles", type=int, default=60, help="Cancel own open limit orders after this many execution candles; 0 disables stale detection.")
    parser.add_argument("--execute", action="store_true", help="Actually submit demo orders. Default is dry-run only.")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--loop-mode", choices=["websocket", "poll"], default="websocket")
    parser.add_argument("--scan-on-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--websocket-demo", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--websocket-testnet", action="store_true")
    parser.add_argument("--websocket-ping-interval", type=int, default=20)
    parser.add_argument("--websocket-ping-timeout", type=int, default=10)
    parser.add_argument("--websocket-retries", type=int, default=10)
    parser.add_argument("--websocket-idle-timeout", type=int, default=120)
    parser.add_argument("--websocket-stop-after-events", type=int, default=0)
    parser.add_argument("--log-jsonl", type=Path, default=Path("scripts/bybit_demo_orders.jsonl"))
    parser.add_argument("--heartbeat-json", type=Path, default=Path("scripts/live_logs/bybit_turtle_soup_heartbeat.json"))
    parser.add_argument("--pid-file", type=Path, default=Path("scripts/bybit_turtle_soup.pid"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    if args.candidate_preset:
        preset = get_preset(args.candidate_preset)
        args = apply_preset_args(args, preset)
        print(f"Using Turtle Soup preset: {json.dumps(preset_summary(preset), sort_keys=True)}")
    args.symbols = [bybit_symbol(symbol) for symbol in args.symbols]
    args.interval = normalize_timeframe(args.interval)
    args.structure_tf = normalize_timeframe(args.structure_tf)
    args.zone_tf = normalize_timeframe(args.zone_tf)
    args.pid_file.parent.mkdir(parents=True, exist_ok=True)
    args.pid_file.write_text(str(os.getpid()), encoding="utf-8")

    if not args.no_ml_filter and not args.model.exists():
        raise SystemExit(f"Model file not found: {args.model}. Pass --model or --no-ml-filter.")
    if args.risk_pct is not None and args.risk_pct <= 0:
        raise SystemExit("--risk-pct must be positive.")
    if args.risk_usdt <= 0:
        raise SystemExit("--risk-usdt must be positive.")
    if args.max_notional_usdt < 0:
        raise SystemExit("--max-notional-usdt cannot be negative.")

    client = BybitV5Client(
        base_url=args.base_url,
        api_key=os.environ.get("BYBIT_DEMO_API_KEY") or os.environ.get("BYBIT_API_KEY"),
        api_secret=os.environ.get("BYBIT_DEMO_API_SECRET") or os.environ.get("BYBIT_API_SECRET"),
    )

    if args.execute and (args.public_only or not client.has_credentials):
        raise SystemExit("--execute requires Bybit API credentials and cannot be combined with --public-only.")
    if args.execute and args.base_url.rstrip("/") != DEFAULT_BASE_URL:
        raise SystemExit(f"Refusing --execute because base URL is not demo: {args.base_url}")

    if args.loop and args.loop_mode == "websocket":
        print(f"\nBybit base={args.base_url} execute={args.execute} symbols={','.join(args.symbols)}")
        run_websocket_loop(args, client)
        return

    heartbeat = initial_heartbeat(args, "poll" if args.loop else "single_scan")
    update_heartbeat(args, heartbeat, "starting_scan")
    while True:
        print(f"\nBybit base={args.base_url} execute={args.execute} symbols={','.join(args.symbols)}")
        account_check(args, client)

        for symbol in args.symbols:
            try:
                row = process_symbol(args, client, symbol)
                heartbeat["last_status_by_symbol"][symbol] = row.get("status")
                heartbeat["last_reconciliation_by_symbol"][symbol] = row.get("reconciliation")
                update_heartbeat(args, heartbeat, f"processed_{symbol}")
            except Exception as exc:
                row = {
                    "time": datetime.now(timezone.utc).isoformat(),
                    "symbol": symbol,
                    "status": "error",
                    "error": str(exc),
                    "dry_run": not args.execute,
                }
                heartbeat["error_count"] = int(heartbeat.get("error_count", 0)) + 1
                heartbeat["last_error"] = row
                update_heartbeat(args, heartbeat, f"error_{symbol}")
                print(f"{symbol}: ERROR {exc}")
            write_jsonl(args.log_jsonl, row)

        if not args.loop:
            break
        update_heartbeat(args, heartbeat, "poll_sleeping")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
