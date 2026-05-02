from __future__ import annotations

import argparse
import hashlib
import os
import sys
from datetime import datetime, timezone
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.bybit_demo_turtle_soup import (
    DEFAULT_BASE_URL,
    BybitV5Client,
    Instrument,
    bybit_symbol,
    decimal_to_str,
    load_env_file,
    quantize_price,
    quantize_to_step,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Place a guarded manual Bybit demo order.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--side", choices=["long", "short"], required=True)
    parser.add_argument("--order-type", choices=["market", "limit"], default="market")
    parser.add_argument("--limit-price", type=Decimal)
    parser.add_argument("--stop", type=Decimal, required=True)
    parser.add_argument("--take-profit", type=Decimal, required=True)
    parser.add_argument("--risk-pct", type=Decimal, default=Decimal("1.0"))
    parser.add_argument("--risk-usdt", type=Decimal)
    parser.add_argument("--base-url", default=os.environ.get("BYBIT_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--env-file", type=Path, default=Path("scripts/bybit_demo.env"))
    parser.add_argument("--account-type", default="UNIFIED")
    parser.add_argument("--position-mode", choices=["one_way", "hedge"], default="hedge")
    parser.add_argument("--price-source", choices=["last", "mark"], default="last")
    parser.add_argument("--order-link-prefix", default="MANUAL")
    parser.add_argument("--allow-existing-position", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def account_equity_usdt(client: BybitV5Client, account_type: str) -> Decimal:
    result = client.wallet_balance(account_type)
    accounts = result.get("list", [])
    if not accounts:
        raise RuntimeError("No wallet account returned by Bybit.")
    account = accounts[0]
    for key in ("totalEquity", "totalWalletBalance", "totalMarginBalance"):
        value = account.get(key)
        if value not in (None, ""):
            equity = Decimal(str(value))
            if equity > 0:
                return equity
    raise RuntimeError("Could not read a positive account equity from wallet-balance.")


def latest_price(client: BybitV5Client, symbol: str, price_source: str) -> Decimal:
    result = client.get_public("/v5/market/tickers", {"category": "linear", "symbol": symbol})
    rows = result.get("list", [])
    if not rows:
        raise RuntimeError(f"No ticker returned for {symbol}.")
    ticker = rows[0]
    key = "markPrice" if price_source == "mark" else "lastPrice"
    price = Decimal(str(ticker.get(key) or ticker.get("lastPrice") or ticker.get("markPrice")))
    if price <= 0:
        raise RuntimeError(f"Invalid {key} for {symbol}: {price}")
    return price


def order_link_id(prefix: str, symbol: str, side: str, stop: Decimal, take_profit: Decimal) -> str:
    now = datetime.now(timezone.utc).isoformat()
    digest = hashlib.sha1(f"{symbol}|{side}|{stop}|{take_profit}|{now}".encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{symbol[:4]}-{side[0].upper()}-{digest}"[:36]


def position_idx(position_mode: str, side: str) -> int:
    if position_mode == "hedge":
        return 1 if side == "long" else 2
    return 0


def build_market_payload(
    args: argparse.Namespace,
    instrument: Instrument,
    entry_ref: Decimal,
    equity: Decimal,
) -> tuple[dict[str, Any], dict[str, Any]]:
    stop = quantize_price(args.stop, instrument)
    take_profit = quantize_price(args.take_profit, instrument)
    if args.order_type == "limit":
        if args.limit_price is None:
            raise RuntimeError("--limit-price is required for --order-type limit.")
        entry_price = quantize_price(args.limit_price, instrument)
    else:
        entry_price = entry_ref

    if args.side == "long":
        if not (stop < entry_price < take_profit):
            raise RuntimeError(f"Long requires stop < entry < take-profit, got {stop} < {entry_price} < {take_profit}.")
        order_side = "Buy"
    else:
        if not (take_profit < entry_price < stop):
            raise RuntimeError(f"Short requires take-profit < entry < stop, got {take_profit} < {entry_price} < {stop}.")
        order_side = "Sell"

    risk_usdt = args.risk_usdt if args.risk_usdt is not None else equity * args.risk_pct / Decimal("100")
    risk_per_coin = abs(entry_price - stop)
    raw_qty = risk_usdt / risk_per_coin
    qty = quantize_to_step(raw_qty, instrument.qty_step, ROUND_FLOOR)
    if qty < instrument.min_qty:
        raise RuntimeError(f"Computed qty {qty} is below min_qty {instrument.min_qty}.")

    notional = qty * entry_price
    if instrument.min_notional > 0 and notional < instrument.min_notional:
        raise RuntimeError(f"Computed notional {notional} is below min_notional {instrument.min_notional}.")

    payload = {
        "category": "linear",
        "symbol": instrument.symbol,
        "side": order_side,
        "orderType": "Market" if args.order_type == "market" else "Limit",
        "qty": decimal_to_str(qty),
        "timeInForce": "IOC" if args.order_type == "market" else "GTC",
        "positionIdx": position_idx(args.position_mode, args.side),
        "orderLinkId": order_link_id(args.order_link_prefix, instrument.symbol, args.side, stop, take_profit),
        "takeProfit": decimal_to_str(take_profit),
        "stopLoss": decimal_to_str(stop),
        "tpTriggerBy": "LastPrice",
        "slTriggerBy": "LastPrice",
        "tpslMode": "Full",
        "tpOrderType": "Market",
        "slOrderType": "Market",
        "reduceOnly": False,
        "closeOnTrigger": False,
    }
    if args.order_type == "limit":
        payload["price"] = decimal_to_str(entry_price)
    sizing = {
        "equity_usdt": equity,
        "risk_pct": args.risk_pct,
        "risk_usdt": risk_usdt,
        "entry_ref": entry_ref,
        "entry_price": entry_price,
        "risk_per_coin": risk_per_coin,
        "qty": qty,
        "estimated_notional": notional,
        "stop": stop,
        "take_profit": take_profit,
    }
    return payload, sizing


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    args.symbol = bybit_symbol(args.symbol)

    client = BybitV5Client(
        base_url=args.base_url,
        api_key=os.environ.get("BYBIT_DEMO_API_KEY") or os.environ.get("BYBIT_API_KEY"),
        api_secret=os.environ.get("BYBIT_DEMO_API_SECRET") or os.environ.get("BYBIT_API_SECRET"),
    )
    if not client.has_credentials:
        raise SystemExit("Missing Bybit credentials in env.")
    if args.execute and args.base_url.rstrip("/") != DEFAULT_BASE_URL:
        raise SystemExit(f"Refusing --execute because base URL is not demo: {args.base_url}")

    instrument = client.instrument(args.symbol)
    if args.execute and not args.allow_existing_position:
        open_positions = client.positions(args.symbol)
        if open_positions:
            raise SystemExit(f"Refusing --execute because {args.symbol} already has an open position.")

    equity = account_equity_usdt(client, args.account_type)
    entry_ref = latest_price(client, args.symbol, args.price_source)
    payload, sizing = build_market_payload(args, instrument, entry_ref, equity)

    print(f"Base URL: {args.base_url}")
    print(f"Symbol: {args.symbol} {args.side.upper()} {args.order_type}")
    print(f"Equity: {decimal_to_str(sizing['equity_usdt'])} USDT")
    if args.risk_usdt is None:
        print(f"Risk: {decimal_to_str(sizing['risk_pct'])}% = {decimal_to_str(sizing['risk_usdt'])} USDT")
    else:
        print(f"Risk: {decimal_to_str(sizing['risk_usdt'])} USDT")
    print(f"Reference {args.price_source} price: {decimal_to_str(sizing['entry_ref'])}")
    if args.order_type == "limit":
        print(f"Limit price: {decimal_to_str(sizing['entry_price'])}")
    print(f"Qty: {decimal_to_str(sizing['qty'])}")
    print(f"Estimated notional: {decimal_to_str(sizing['estimated_notional'])} USDT")
    print(f"SL: {decimal_to_str(sizing['stop'])}")
    print(f"TP: {decimal_to_str(sizing['take_profit'])}")
    print(f"OrderLinkId: {payload['orderLinkId']}")

    if not args.execute:
        print("DRY RUN ONLY: pass --execute to submit.")
        return

    result = client.create_order(payload)
    print(f"SUBMITTED: {result}")


if __name__ == "__main__":
    main()
