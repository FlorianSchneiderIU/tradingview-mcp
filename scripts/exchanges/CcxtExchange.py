"""Generic CCXT-based futures exchange implementation."""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Literal, Optional

import ccxt  # type: ignore
import pandas as pd

from exchanges.IFuturesExchange import IFuturesExchange
from exchanges.types.common import (
    Balance,
    ClientOid,
    CreateOrderResponse,
    Direction,
    Fill,
    FuturesMarket,
    HistoricPosition,
    LimitOrderRequest,
    MarginMode,
    MarketOrderRequest,
    Order,
    OrderBookData,
    OrderSide,
    OrderType,
    Position,
    Status,
    StopMarketOrderRequest,
    Ticker,
    TimeFrame,
    TimestampMilliseconds,
    TimestampNanoseconds,
    TradingPair,
)
from exchanges.types.exceptions import PositionNotFoundError
from my_types.percentage import Percentage
from utils.SqlManager import SQLiteManager


def _safe_float(*values: Any) -> float:
    for value in values:
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _milliseconds(value: Optional[int]) -> TimestampMilliseconds:
    if value is None or value <= 0:
        value = int(pd.Timestamp.utcnow().timestamp() * 1000)
    return TimestampMilliseconds(value)


def _nanoseconds(value: Optional[int]) -> TimestampNanoseconds:
    if value is None or value <= 0:
        value = int(pd.Timestamp.utcnow().timestamp() * 1_000_000_000)
    return TimestampNanoseconds(value)


class CcxtExchange(IFuturesExchange):
    """Concrete implementation of :class:`IFuturesExchange` using CCXT."""

    def __init__(
        self,
        name: str,
        client: ccxt.Exchange,
        db_manager: SQLiteManager,
    ) -> None:
        self._name = name
        self.client = client
        self.db_manager = db_manager
        self._markets: dict[TradingPair, FuturesMarket] = {}
        self._margin_mode_cache: dict[str, MarginMode] = {}
        self._order_symbol_cache: dict[str, str] = {}
        self.logger = logging.getLogger(name)
        self.load_markets()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_symbol(self, trading_pair: TradingPair | str) -> str:
        if isinstance(trading_pair, TradingPair):
            return str(trading_pair)
        return str(self._to_trading_pair(trading_pair))

    def _to_trading_pair(self, symbol: str) -> TradingPair:
        # Normalize double slashes to single slash (e.g., "SOL//USDT" -> "SOL/USDT")
        symbol = symbol.replace("//", "/")
        
        if ":" not in symbol and symbol.endswith("/USDT"):
            symbol = f"{symbol}:USDT"
        return TradingPair(symbol)

    def _lot_size(self, trading_pair: TradingPair) -> float:
        market = self.markets.get(trading_pair)
        if not market:
            return 1.0
        return market.lot_size or 1.0

    def _amount_to_lots(self, amount: float, trading_pair: TradingPair) -> int:
        """Convert contract amount to lots using market's lot_size.
        
        This ensures consistent conversion across positions, orders, fills, etc.
        Uses the same formula as position conversion: round(amount / lot_size).
        
        Args:
            amount: The contract amount/quantity from the exchange
            trading_pair: The trading pair to get lot_size for
            
        Returns:
            Integer number of lots
            
        Example:
            amount=1.0, lot_size=0.1 → 10 lots
            amount=0.05, lot_size=0.001 → 50 lots
            amount=1000, lot_size=1.0 → 1000 lots
        """
        market = self.markets.get(trading_pair)
        lot_size = market.lot_size if market else 1.0
        if lot_size and lot_size != 0:
            return int(round(amount / lot_size))
        return int(amount)

    def _convert_market(self, symbol: str, data: Dict[str, Any], ticker: Dict[str, Any]) -> FuturesMarket:
        info = data.get("info", {})
        ticker_info = ticker.get("info", {})
        
        # Debug: Log what we're receiving
        if not ticker:
            self.logger.debug(f"No ticker data for {symbol}, using market data only")
        
        # Get mark price - try ticker first, then fall back to info fields
        mark_price = _safe_float(
            ticker.get("markPrice"),
            ticker_info.get("markPrice"),
            ticker.get("last"),
            ticker_info.get("lastPrice"),
            info.get("markPrice"),
            info.get("lastPrice"),
            data.get("mark"),
        )
        
        leverage_limits = data.get("limits", {}).get("leverage", {})
        max_leverage = int(
            leverage_limits.get("max")
            or info.get("leverageFilter", {}).get("maxLeverage", 50)
            or 50
        )
        
        # Get lot size (qtyStep) from data.precision.amount (CCXT normalized) or info.lotSizeFilter.qtyStep (raw)
        contract_size = _safe_float(
            data.get("precision", {}).get("amount"),
            info.get("lotSizeFilter", {}).get("qtyStep"),
            1.0
        )
        
        # Get percentage change - try ticker first, then info
        # Percentage class expects decimal form (0.04233 = 4.233%)
        # ticker.percentage from CCXT is in percentage form (-4.233 means -4.233%)
        # info.price24hPcnt from Bybit is in decimal string form ("-0.04233" means -4.233%)
        price_change_decimal = 0.0
        
        if ticker.get("percentage") is not None:
            # CCXT gives us percentage form, convert to decimal
            price_change_decimal = _safe_float(ticker.get("percentage"), 0.0) / 100
        elif ticker_info.get("price24hPcnt"):
            # Bybit ticker info gives decimal string form, use as-is
            price_change_decimal = _safe_float(ticker_info.get("price24hPcnt"), 0.0)
        elif info.get("price24hPcnt"):
            # Bybit market info gives decimal string form, use as-is
            price_change_decimal = _safe_float(info.get("price24hPcnt"), 0.0)
        
        market = FuturesMarket(
            markPrice=mark_price,
            maxLeverage=max_leverage,
            lot_size=contract_size or 1.0,
            daily_high=_safe_float(
                ticker.get("high"),
                ticker_info.get("highPrice24h"),
                info.get("highPrice24h")
            ),
            daily_low=_safe_float(
                ticker.get("low"),
                ticker_info.get("lowPrice24h"),
                info.get("lowPrice24h")
            ),
            daily_volume=_safe_float(
                ticker.get("baseVolume"),
                ticker_info.get("volume24h"),
                info.get("volume24h")
            ),
            daily_turnover=_safe_float(
                ticker.get("quoteVolume"),
                ticker_info.get("turnover24h"),
                info.get("turnover24h")
            ),
            daily_change=_safe_float(
                ticker.get("change"),
                ticker_info.get("priceChg"),
                info.get("priceChg")
            ),
            daily_change_rate=Percentage(price_change_decimal),
            open_interest=_safe_float(
                ticker_info.get("openInterest"),
                info.get("openInterest")
            ),
            takerFeeRate=_safe_float(data.get("taker"), info.get("takerFee"), 0.0006),
        )
        return market

    def _record_order_symbol(self, order_id: str, symbol: str) -> None:
        if order_id:
            self._order_symbol_cache[order_id] = symbol

    def _resolve_symbol_from_cache(self, order_id: str) -> Optional[str]:
        return self._order_symbol_cache.get(order_id)

    def _amount_from_lots(self, trading_pair: TradingPair, lots: int) -> float:
        return lots * self._lot_size(trading_pair)

    def _create_order(
        self,
        params: LimitOrderRequest | MarketOrderRequest,
        order_type: OrderType,
        side: OrderSide,
        amount_lots: int,
        price: Optional[float] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> CreateOrderResponse:
        trading_pair = params.trading_pair
        symbol = self._to_symbol(trading_pair)
        amount = self._amount_from_lots(trading_pair, amount_lots)
        payload = {"reduceOnly": params.reduceOnly, "leverage": params.leverage}
        if extra_params:
            payload.update(extra_params)
        order = self.client.create_order(symbol, order_type, side, amount, price, payload)
        order_id = str(order.get("id"))
        self._record_order_symbol(order_id, symbol)
        return CreateOrderResponse(orderId=order_id)

    def _convert_order(self, raw: Dict[str, Any]) -> Order:
        symbol = raw.get("symbol") or raw.get("info", {}).get("symbol")
        trading_pair = self._to_trading_pair(symbol)
        client_oid = raw.get("clientOrderId") or raw.get("info", {}).get("orderLinkId")
        client_oid_obj = ClientOid.from_string(client_oid) if ClientOid.is_valid_string(client_oid) else None
        created = _milliseconds(raw.get("timestamp"))
        updated = _milliseconds(raw.get("lastTradeTimestamp"))
        stop_price = _safe_float(raw.get("stopPrice"), raw.get("triggerPrice"))
        status = raw.get("status") or "open"
        margin_mode = self._margin_mode_cache.get(symbol, "CROSS")
        
        # Convert amount (contracts/qty) to lots using consistent conversion
        amount = _safe_float(raw.get("amount"))
        amount_lots = self._amount_to_lots(amount, trading_pair)
        filled = _safe_float(raw.get("filled"))
        filled_lots = self._amount_to_lots(filled, trading_pair)
        
        order = Order(
            id=str(raw.get("id")),
            trading_pair=trading_pair,
            type=str(raw.get("type")),
            side=raw.get("side"),
            price=str(raw.get("price") or "0"),
            amountLots=amount_lots,
            value=str(raw.get("cost") or "0"),
            dealValue=str(raw.get("filled") or "0"),
            dealSize=filled_lots,
            stp="",
            stop="",
            stopPriceType="",
            stopTriggered=bool(raw.get("triggered")),
            stopPrice=str(stop_price),
            timeInForce=str(raw.get("timeInForce") or "GTC"),
            postOnly=bool(raw.get("postOnly")),
            hidden=bool(raw.get("info", {}).get("hidden")),
            iceberg=False,
            leverage=str(raw.get("info", {}).get("leverage", "")),
            forceHold=False,
            closeOrder=bool(raw.get("reduceOnly")),
            visibleSize=int(_safe_float(raw.get("info", {}).get("visibleSize"))),
            clientOid=client_oid_obj,
            remark=None,
            tags="",
            isActive=status in ("open", "active"),
            cancelExist=status == "canceled",
            createdAt=created,
            updatedAt=updated,
            endAt=_milliseconds(raw.get("lastTradeTimestamp")),
            orderTime=int(created),
            settleCurrency=trading_pair.settle(),
            marginMode=margin_mode,
            avgDealPrice=str(raw.get("average")),
            filledLots=filled_lots,
            filledValue=str(raw.get("cost") or "0"),
            status="open" if status in ("open", "active") else "done",
            reduceOnly=bool(raw.get("reduceOnly")),
            species="tp" if raw.get("info", {}).get("stopOrderType") in ("TakeProfit", "PartialTakeProfit") else
                    "sl" if raw.get("info", {}).get("stopOrderType") in ("StopLoss", "PartialStopLoss") else
                    "entry" if (not raw.get("info", {}).get("stopOrderType") and raw.get("info", {}).get("orderStatus") == "Filled") else
                    "unknown",
        )
        return order

    def _generate_position_id(
        self, 
        trading_pair: TradingPair, 
        direction: Direction, 
        entry_price: float,
        timestamp: TimestampMilliseconds
    ) -> str:
        """Generate a deterministic position ID from position characteristics.
        
        Uses trading_pair, direction, entry_price, and opening timestamp to create
        a unique identifier that remains consistent across API calls for the same position.
        
        Args:
            trading_pair: The normalized trading pair
            direction: Position direction (long/short)
            entry_price: Average entry price
            timestamp: Position opening timestamp
            
        Returns:
            A deterministic string ID like "HIPPO/USDT:USDT_long_0.001194_1763273510428"
        """
        # Format entry price to avoid floating point precision issues
        # Use 8 decimal places which should be sufficient for most crypto pairs
        price_str = f"{entry_price:.8f}".rstrip('0').rstrip('.')
        
        # Create composite key: trading_pair_direction_entryPrice_timestamp
        # This uniquely identifies a specific position instance
        return f"{trading_pair}_{direction}_{price_str}_{timestamp}"
    
    def _convert_position(self, raw: Dict[str, Any]) -> Position:
        symbol = raw.get("symbol") or raw.get("info", {}).get("symbol")
        trading_pair = self._to_trading_pair(symbol)
        amount = _safe_float(raw.get("contracts"), raw.get("info", {}).get("size"))
        
        # Convert amount to lots using consistent conversion
        contracts = self._amount_to_lots(amount, trading_pair)
        
        direction: Direction = "long"
        if str(raw.get("side", "")).lower() == "short":
            direction = "short"
        margin_mode = self._margin_mode_cache.get(symbol, "CROSS")
        
        # Extract position characteristics for ID generation
        entry_price = _safe_float(raw.get("entryPrice"))
        timestamp = _milliseconds(raw.get("timestamp"))
        
        # Construct position ID: use exchange-provided ID if available,
        # otherwise generate deterministic ID from position characteristics
        position_id = raw.get("id") or raw.get("info", {}).get("id")
        if not position_id:
            # For exchanges that don't provide position IDs (like Bybit in some cases),
            # generate a deterministic ID from the position's unique characteristics.
            # This ensures the same position always gets the same ID across API calls.
            position_id = self._generate_position_id(
                trading_pair, direction, entry_price, timestamp
            )
        
        return Position(
            avgEntryPrice=entry_price,
            currentLots=int(contracts),
            currentQty=float(amount),
            id=str(position_id),
            isOpen=bool(contracts),
            leverage=int(_safe_float(raw.get("leverage"), raw.get("info", {}).get("leverage"), 1) or 1),
            liquidationPrice=_safe_float(raw.get("liquidationPrice")),
            marginMode=margin_mode,
            markPrice=_safe_float(raw.get("markPrice"), raw.get("info", {}).get("markPrice")),
            openingTimestamp=_milliseconds(raw.get("timestamp")),
            posCost=_safe_float(raw.get("notional"), raw.get("initialMargin")),
            posInit=_safe_float(raw.get("initialMargin")),
            direction=direction,
            realisedPnl=_safe_float(raw.get("realizedPnl")),
            trading_pair=trading_pair,
            unrealisedPnl=_safe_float(raw.get("unrealizedPnl")),
            unrealisedPnlPcnt=_safe_float(raw.get("percentage")),
            unrealisedRoePcnt=_safe_float(raw.get("percentage")),
        )

    def _convert_fill(self, raw: Dict[str, Any]) -> Fill:
        symbol = raw.get("symbol") or raw.get("info", {}).get("symbol")
        trading_pair = self._to_trading_pair(symbol)
        
        # Convert amount to lots using consistent conversion
        amount = _safe_float(raw.get("amount"))
        size_lots = self._amount_to_lots(amount, trading_pair)
        
        return Fill(
            trading_pair=trading_pair,
            tradeId=str(raw.get("id")),
            orderId=str(raw.get("order")),
            side=str(raw.get("side")),
            liquidity=str(raw.get("takerOrMaker", "taker")),
            forceTaker=str(raw.get("takerOrMaker", "taker")) == "taker",
            price=str(raw.get("price")),
            size=size_lots,
            value=str(raw.get("cost")),
            feeRate=str(_safe_float(raw.get("fee",{}).get("rate",""))),
            fixFee="0",
            feeCurrency=str(raw.get("fee",{}).get("currency","") or trading_pair.settle()),
            stop="",
            fee=str(_safe_float(raw.get("fee",{}).get("cost",""))),
            orderType=str(raw.get("type")),
            tradeType=str(raw.get("info", {}).get("tradeType", "")),
            createdAt=_milliseconds(raw.get("timestamp")),
            settleCurrency=trading_pair.settle(),
            tradeTime=_nanoseconds((raw.get("timestamp") or 0) * 1_000_000),
            openFeePay="0",
            closeFeePay="0",
            marginMode=str(self._margin_mode_cache.get(str(trading_pair), "CROSS")),
            subTradeType=None,
            displayType="",
        )

    def _convert_history(self, raw: Dict[str, Any]) -> HistoricPosition:
        symbol = raw.get("symbol") or raw.get("info", {}).get("symbol")
        trading_pair = self._to_trading_pair(symbol)
        timestamp = _milliseconds(raw.get("timestamp"))
        
        # Convert amount to lots using consistent conversion
        amount = _safe_float(raw.get("amount"))
        max_filled_lots = self._amount_to_lots(amount, trading_pair)
        
        # Extract position direction from trade data
        # CCXT trades have 'side' (buy/sell), but we need position direction (long/short)
        # First try to get from info.side (exchange-specific field)
        info = raw.get("info", {})
        position_type = info.get("side", "")  # Some exchanges provide this
        
        # If not found, try to infer from trade side
        # Note: This is NOT always reliable because we don't know if it's opening or closing
        # For Bybit, closing trades are marked with reduceOnly or we need position side from API
        if not position_type or position_type not in ["long", "Long", "short", "Short"]:
            side = str(raw.get("side", "")).lower()
            # Fallback: assume trade side matches position side (opening trade)
            # This is a best-effort approach - ideally exchange should provide position direction
            if side == "buy":
                position_type = "long"  # Assume buy = opening long
            elif side == "sell":
                position_type = "short"  # Assume sell = opening short
            else:
                position_type = "unknown"
                
        position_type = str(position_type).lower()
        
        return HistoricPosition(
            closeId=str(raw.get("id")),
            userId=str(raw.get("info", {}).get("userId", "")),
            trading_pair=trading_pair,
            settleCurrency=trading_pair.settle(),
            leverage=str(raw.get("info", {}).get("leverage", "")),
            type=position_type,  # Position direction: "long", "short", or fallback
            pnl=str(raw.get("info", {}).get("pnl", "0")),
            realisedGrossCost=str(raw.get("info", {}).get("realisedGrossCost", "0")),
            tradeFee=str(raw.get("fee", "0")),
            fundingFee=str(raw.get("info", {}).get("fundingFee", "0")),
            openTime=timestamp,
            closeTime=timestamp,
            openPrice=str(raw.get("price")),
            closePrice=str(raw.get("price")),
            marginMode=str(self._margin_mode_cache.get(str(trading_pair), "CROSS")),
            maxFilledLots=max_filled_lots,
        )

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def get_name(self) -> str:
        return self._name

    def load_markets(self) -> None:
        """Load markets with retry logic for timestamp sync issues."""
        max_retries = 3
        retry_delay = 0.5  # seconds
        
        for attempt in range(max_retries):
            try:
                raw_markets = self.client.load_markets(True)
                tickers = self.client.fetch_tickers()
                markets: dict[TradingPair, FuturesMarket] = {}
                for symbol, market in raw_markets.items():
                    if not market.get("swap"):
                        continue
                    trading_pair = self._to_trading_pair(symbol)
                    markets[trading_pair] = self._convert_market(symbol, market, tickers.get(symbol, {}))
                self._markets = markets
                return
            except ccxt.InvalidNonce as e:
                if attempt < max_retries - 1:
                    self.logger.warning(
                        f"InvalidNonce error on attempt {attempt + 1}/{max_retries}: {e}. "
                        f"Attempting to synchronize time with exchange..."
                    )
                    # Try to synchronize time with exchange before retrying
                    if hasattr(self.client, "load_time_difference"):
                        try:
                            self.client.load_time_difference()  # type: ignore
                            self.logger.info("Successfully synchronized time with exchange")
                        except Exception as sync_error:
                            self.logger.warning(f"Time sync failed: {sync_error}")
                    
                    self.logger.info(f"Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    self.logger.error(f"Failed to load markets after {max_retries} attempts")
                    raise

    def market(self, trading_pair: TradingPair) -> FuturesMarket:
        self.load_markets()
        market = self._markets.get(trading_pair)
        if not market:
            raise ValueError(f"Unknown trading pair {trading_pair}")
        return market

    def fetch_order_book(self, trading_pair: TradingPair, depth: int | Literal["full"] = 20) -> OrderBookData:
        symbol = self._to_symbol(trading_pair)
        limit = None if depth == "full" else depth
        ob = self.client.fetch_order_book(symbol, limit=limit)
        return OrderBookData(
            trading_pair=trading_pair,
            sequence=int(ob.get("nonce") or 0),
            asks=[[str(price), size] for price, size in ob.get("asks", [])],
            bids=[[str(price), size] for price, size in ob.get("bids", [])],
            ts=_nanoseconds((ob.get("timestamp") or 0) * 1_000_000),
        )

    def fetch_ticker(self, trading_pair: TradingPair) -> Ticker:
        symbol = self._to_symbol(trading_pair)
        ticker = self.client.fetch_ticker(symbol)
        return Ticker(
            last_price=_safe_float(ticker.get("last")),
            mark_price=_safe_float(ticker.get("info", {}).get("markPrice"), ticker.get("last")),
            index_price=_safe_float(ticker.get("info", {}).get("indexPrice")),
        )

    def fetch_positions_history(
        self,
        since: Optional[TimestampMilliseconds] = None,
        trading_pair: Optional[TradingPair] = None,
    ) -> List[HistoricPosition]:
        symbol = self._to_symbol(trading_pair) if trading_pair else None
        trades = self.client.fetch_my_trades(symbol=symbol, since=int(since) if since else None)
        return [self._convert_history(trade) for trade in trades]

    def fetch_ohlcv(
        self,
        trading_pair: TradingPair,
        timeframe: TimeFrame,
        since: Optional[TimestampMilliseconds] = None,
        until: Optional[TimestampMilliseconds] = None,
    ) -> pd.DataFrame:
        symbol = self._to_symbol(trading_pair)
        tf = str(timeframe.value)
        ohlcv = self.client.fetch_ohlcv(symbol, timeframe=tf, since=int(since) if since else None)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        if until:
            df = df[df["timestamp"] <= int(until)]
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df

    def create_limit_order(self, params: LimitOrderRequest, test: bool = False) -> CreateOrderResponse:
        extra_params: dict[str, Any] = {}
        if params.takeProfit is not None:
            extra_params["takeProfit"] = params.takeProfit
        if params.stopLoss is not None:
            extra_params["stopLoss"] = params.stopLoss
        if params.tpslMode is not None:
            extra_params["tpslMode"] = params.tpslMode
        if params.tpTriggerBy is not None:
            extra_params["tpTriggerBy"] = params.tpTriggerBy
        if params.slTriggerBy is not None:
            extra_params["slTriggerBy"] = params.slTriggerBy

        return self._create_order(
            params,
            "limit",
            params.side,
            params.size,
            price=params.price,
            extra_params=extra_params or None,
        )

    def create_market_order(
        self,
        params: MarketOrderRequest | StopMarketOrderRequest,
        test: bool = False,
    ) -> CreateOrderResponse:
        extra_params: dict[str, Any] | None = None
        if isinstance(params, StopMarketOrderRequest):
            extra_params = {
                "stop": params.stop,
                "stopPrice": params.stopPrice,
                "triggerPrice": params.stopPrice,
                "stopPriceType": params.stopPriceType,
            }

        return self._create_order(
            params,
            "market",
            params.order_side,
            params.amount_lots,
            extra_params=extra_params,
        )

    def create_take_profit_order(
        self,
        trading_pair: TradingPair,
        position_direction: Direction,
        lots: int,
        price: float,
        leverage: int,
        clientOid: ClientOid,
        margin_mode: MarginMode,
        test: bool = False,
    ) -> CreateOrderResponse:
        side: OrderSide = "sell" if position_direction == "long" else "buy"
        params = LimitOrderRequest(
            type="limit",
            leverage=leverage,
            trading_pair=trading_pair,
            side=side,
            size=lots,
            clientOid=clientOid,
            price=price,
            marginMode=margin_mode,
            reduceOnly=True,
        )
        return self._create_order(
            params,
            "limit",
            side,
            lots,
            price=price,
            extra_params={"takeProfit": price, "reduceOnly": True},
        )

    def create_stop_loss_order(
        self,
        trading_pair: TradingPair,
        position_direction: Direction,
        lots: int,
        price: float,
        leverage: int,
        clientOid: ClientOid,
        margin_mode: MarginMode,
        test: bool = False,
    ) -> CreateOrderResponse:
        side: OrderSide = "sell" if position_direction == "long" else "buy"
        params = LimitOrderRequest(
            type="limit",
            leverage=leverage,
            trading_pair=trading_pair,
            side=side,
            size=lots,
            clientOid=clientOid,
            price=price,
            marginMode=margin_mode,
            reduceOnly=True,
        )
        return self._create_order(
            params,
            "limit",
            side,
            lots,
            price=price,
            extra_params={"stopLoss": price, "reduceOnly": True},
        )

    def fetch_positions(self) -> List[Position]:
        positions = self.client.fetch_positions()
        return [self._convert_position(pos) for pos in positions if pos]

    def adjust_price(self, price: float, trading_pair: TradingPair) -> float:
        market = self.client.market(self._to_symbol(trading_pair))
        tick_size = _safe_float(
            market.get("info", {}).get("priceFilter", {}).get("tickSize"),
            market.get("precision", {}).get("price"),
            0.5,
        )
        if tick_size <= 0:
            return price
        ticks = math.floor(price / tick_size)
        return ticks * tick_size

    def change_auto_deposit_status(self, trading_pair: TradingPair, status: bool) -> bool:
        self.logger.debug("Auto deposit status change is not supported via CCXT; ignoring request.")
        return False

    def change_cross_leverage(self, trading_pair: TradingPair, leverage: float) -> bool:
        symbol = self._to_symbol(trading_pair)
        try:
            # Some exchanges (Bybit) return an error when the leverage is already set to
            # the requested value (e.g. retCode 110043 / "leverage not modified").
            # CCXT raises an exception in that case with the exchange response in the
            # message. Treat that situation as a successful no-op.
            self.client.set_leverage(int(leverage), symbol)
            return True
        except Exception as e:
            msg = str(e)
            # Normalize message for matching
            low = msg.lower()
            if "leverage not modified" in low or '"retcode":110043' in low or 'retcode":110043' in low:
                # Already at requested leverage — that's fine.
                self.logger.info(
                    f"Leverage for {symbol} already set to {leverage}; treating as success."
                )
                return True
            # Unknown error — re-raise to let caller handle/log it
            raise

    def change_margin_mode(self, trading_pair: TradingPair, margin_mode: MarginMode) -> bool:
        symbol = self._to_symbol(trading_pair)
        mode = margin_mode.lower()
        self.client.set_margin_mode(mode, symbol)
        self._margin_mode_cache[symbol] = margin_mode
        return True

    def cancel_order(self, order_id: str, trading_pair: Optional[TradingPair] = None) -> List[str]:
        # If trading_pair is provided, use it directly
        if trading_pair:
            symbol = self._to_symbol(trading_pair)
        else:
            symbol = self._resolve_symbol_from_cache(order_id)
        
        try:
            if symbol:
                self.client.cancel_order(order_id, symbol)
            else:
                # Symbol not in cache - need to fetch the order first to get its symbol
                # This is required by Bybit which mandates the symbol parameter
                try:
                    # Try to fetch order from open orders first (most common case)
                    open_orders = self.client.fetch_open_orders()
                    for order in open_orders:
                        # Check both order ID and client order ID
                        # (order_id might be a ClientOid UUID string)
                        order_order_id = str(order.get("id"))
                        order_client_oid = order.get("clientOrderId") or order.get(
                            "info", {}
                        ).get("orderLinkId")

                        if order_order_id == order_id or order_client_oid == order_id:
                            symbol = order.get("symbol")
                            if symbol:
                                # Cache the actual order ID with its symbol
                                self._record_order_symbol(order_order_id, symbol)
                                if order_client_oid:
                                    self._record_order_symbol(order_client_oid, symbol)

                                # Cancel using the actual order ID, not client order ID
                                self.client.cancel_order(order_order_id, symbol)
                                return [order_id]

                    # If not found in open orders, it might already be closed/canceled
                    self.logger.warning(
                        f"Could not find order {order_id} in open orders. "
                        f"It may already be canceled or filled."
                    )
                    # Try to cancel anyway without symbol - some exchanges might support it
                    self.client.cancel_order(order_id)
                except Exception as e:
                    # Check if this is an "order not exists" error
                    msg = str(e).lower()
                    order_not_exists_indicators = [
                        "order not exists",
                        "too late to cancel",
                        '"retcode":110001',
                        'retcode":110001',
                        "order does not exist",
                        "order_not_exists",
                    ]
                    if any(indicator in msg for indicator in order_not_exists_indicators):
                        self.logger.info(
                            f"Order {order_id} already closed/canceled or doesn't exist "
                            f"(possibly auto-canceled by exchange when setting new TP/SL); treating as success."
                        )
                        return [order_id]
                    # Unknown error - re-raise
                    raise
        except Exception as e:
            # Check if this is an "order not exists" error at the top level too
            msg = str(e).lower()
            order_not_exists_indicators = [
                "order not exists",
                "too late to cancel",
                '"retcode":110001',
                'retcode":110001',
                "order does not exist",
                "order_not_exists",
            ]
            if any(indicator in msg for indicator in order_not_exists_indicators):
                self.logger.info(
                    f"Order {order_id} already closed/canceled or doesn't exist "
                    f"(possibly auto-canceled by exchange when setting new TP/SL); treating as success."
                )
                return [order_id]
            # Unknown error - log and re-raise
            self.logger.error(f"Failed to cancel order {order_id}: {e}")
            raise
            
        return [order_id]

    def close_position(
        self,
        trading_pair: TradingPair,
        margin_mode: MarginMode,
        test: bool = False,
        channel: str = "MC",
        clientOid: Optional[ClientOid] = None,
    ) -> CreateOrderResponse:
        position = self.fetch_position(trading_pair)
        if not position.currentLots:
            raise PositionNotFoundError(str(trading_pair), self.get_name())
        side: OrderSide = "sell" if position.direction == "long" else "buy"
        symbol = self._to_symbol(trading_pair)
        amount = self._amount_from_lots(trading_pair, position.currentLots)
        order = self.client.create_order(
            symbol,
            "market",
            side,
            amount,
            None,
            {"reduceOnly": True},
        )
        order_id = str(order.get("id"))
        self._record_order_symbol(order_id, symbol)
        return CreateOrderResponse(orderId=order_id)

    def fetch_balance(self) -> Balance:
        balance = self.client.fetch_balance(params={"type": "swap"})
        currency = "USDT"
        total_data = balance.get(currency) or balance.get(f"{currency}:{currency}") or {}
        account_equity = _safe_float(balance.get("info", {}).get("totalEquity"), total_data.get("total"))
        available = _safe_float(total_data.get("free"))
        margin_balance = _safe_float(balance.get("info", {}).get("equity"), total_data.get("total"))
        position_margin = _safe_float(balance.get("info", {}).get("positionMargin"))
        order_margin = _safe_float(balance.get("info", {}).get("orderMargin"))
        frozen = _safe_float(balance.get("info", {}).get("frozenBalance"))
        return Balance(
            accountEquity=account_equity,
            unrealisedPNL=_safe_float(balance.get("info", {}).get("unrealisedPnl")),
            marginBalance=margin_balance,
            positionMargin=position_margin,
            orderMargin=order_margin,
            frozenFunds=frozen,
            availableBalance=available,
            currency=currency,
        )

    def fetch_closed_orders(
        self,
        trading_pair: Optional[TradingPair] = None,
        since: Optional[TimestampMilliseconds] = None,
        limit: Optional[int] = None,
        side: Optional[OrderSide] = None,
    ) -> List[Order]:
        symbol = self._to_symbol(trading_pair) if trading_pair else None
        orders = self.client.fetch_closed_orders(symbol=symbol, since=int(since) if since else None, limit=limit)
        filtered = [order for order in orders if (not side or order.get("side") == side)]
        return [self._convert_order(order) for order in filtered]

    def fetch_closed_tpsl(
        self,
        trading_pair: Optional[TradingPair] = None,
        since: Optional[TimestampMilliseconds] = None,
        limit: Optional[int] = None,
        side: Optional[OrderSide] = None,
    ) -> List[Order]:
        orders = self.fetch_closed_orders(trading_pair, since, limit, side)
        return [order for order in orders if order.reduceOnly]

    def fetch_order_by_coid(self, coid: ClientOid) -> Order:
        client_order_id = str(coid)
        open_orders = self.client.fetch_open_orders()
        for order in open_orders:
            if order.get("clientOrderId") == client_order_id:
                return self._convert_order(order)
        closed_orders = self.client.fetch_closed_orders()
        for order in closed_orders:
            if order.get("clientOrderId") == client_order_id:
                return self._convert_order(order)
        raise ValueError(f"Order with client order id {client_order_id} not found")

    def fetch_order_by_id(self, order_id: str) -> Order:
        symbol = self._resolve_symbol_from_cache(order_id)
        
        # Try to fetch directly first
        try:
            if symbol:
                order = self.client.fetch_order(order_id, symbol)
            else:
                order = self.client.fetch_order(order_id)
            self._record_order_symbol(order_id, order.get("symbol"))
            return self._convert_order(order)
        except Exception as e:
            error_msg = str(e).lower()
            # Bybit limits fetch_order to last 500 orders - fall back to searching open/closed
            if "last 500 orders" in error_msg or "acknowledged" in error_msg:
                self.logger.debug(
                    f"Order {order_id} not in recent history, searching open and closed orders..."
                )
            else:
                # Some other error - re-raise
                raise
        
        # Fall back: search in open orders
        try:
            open_orders = self.client.fetch_open_orders()
            for order in open_orders:
                order_order_id = str(order.get("id"))
                order_client_oid = order.get("clientOrderId") or order.get("info", {}).get("orderLinkId")
                
                if order_order_id == order_id or order_client_oid == order_id:
                    symbol = order.get("symbol")
                    if symbol:
                        self._record_order_symbol(order_order_id, symbol)
                        if order_client_oid:
                            self._record_order_symbol(order_client_oid, symbol)
                    return self._convert_order(order)
        except Exception as e:
            self.logger.warning(f"Failed to search open orders for {order_id}: {e}")
        
        # Fall back: search in closed orders (recent)
        try:
            closed_orders = self.client.fetch_closed_orders(limit=500)
            for order in closed_orders:
                order_order_id = str(order.get("id"))
                order_client_oid = order.get("clientOrderId") or order.get("info", {}).get("orderLinkId")
                
                if order_order_id == order_id or order_client_oid == order_id:
                    symbol = order.get("symbol")
                    if symbol:
                        self._record_order_symbol(order_order_id, symbol)
                        if order_client_oid:
                            self._record_order_symbol(order_client_oid, symbol)
                    return self._convert_order(order)
        except Exception as e:
            self.logger.warning(f"Failed to search closed orders for {order_id}: {e}")
        
        # If we get here, order was not found anywhere
        raise ValueError(f"Order {order_id} not found in open or closed orders")

    def fetch_order_by_symbol(
        self,
        trading_pair: TradingPair,
        since: Optional[TimestampMilliseconds] = None,
        until: Optional[TimestampMilliseconds] = None,
        limit: Optional[int] = None,
    ) -> List[Order]:
        symbol = self._to_symbol(trading_pair)
        orders = self.client.fetch_orders(symbol, since=int(since) if since else None, limit=limit)
        if until:
            orders = [order for order in orders if (order.get("timestamp") or 0) <= int(until)]
        return [self._convert_order(order) for order in orders]

    def fetch_orders_by_status(
        self,
        status: Status,
        trading_pair: Optional[TradingPair] = None,
        since: Optional[TimestampMilliseconds] = None,
        limit: Optional[int] = 1000,
    ) -> List[Order]:
        symbol = self._to_symbol(trading_pair) if trading_pair else None
        if status in ["open", "active"]:
            orders = self.client.fetch_open_orders(symbol=symbol, since=int(since) if since else None, limit=limit)
        else:
            orders = self.client.fetch_closed_orders(symbol=symbol, since=int(since) if since else None, limit=limit)
        return [self._convert_order(order) for order in orders]

    def fetch_position(self, trading_pair: TradingPair, side: Optional[OrderSide] = None) -> Position:
        positions = self.fetch_positions()
        for position in positions:
            if position.trading_pair == trading_pair:
                if side is None or (
                    side == "buy" and position.direction == "long"
                ) or (side == "sell" and position.direction == "short"):
                    return position
        raise PositionNotFoundError(str(trading_pair), self.get_name())

    def fetch_untriggered_stop_orders(
        self,
        trading_pair: Optional[TradingPair] = None,
        since: Optional[TimestampMilliseconds] = None,
        limit: Optional[int] = 1000,
    ) -> List[Order]:
        orders = self.fetch_orders_by_status("open", trading_pair, since, limit)
        return [order for order in orders if order.species in ("sl", "tp") and not order.stopTriggered]

    def get_margin_mode(self, trading_pair: TradingPair) -> MarginMode:
        symbol = self._to_symbol(trading_pair)
        return self._margin_mode_cache.get(symbol, "CROSS")

    def allows_cross_mode(self, trading_pair: TradingPair) -> bool:
        return True

    def get_recent_fills(self) -> List[Fill]:
        trades = self.client.fetch_my_trades(limit=50)
        return [self._convert_fill(trade) for trade in trades]

    def get_standardized_symbol(self, exchange: str) -> TradingPair:
        return self._to_trading_pair(exchange)

    def get_symbol_id(self, standardized_trading_pair: TradingPair) -> str:
        return str(standardized_trading_pair)

    @property
    def markets(self) -> Dict[TradingPair, FuturesMarket]:
        return self._markets

    def cancel_tpsl_order(self, order_id: str, trading_pair: TradingPair) -> List[str]:
        # Use the trading_pair to get the symbol and cache it for cancel_order
        symbol = self._to_symbol(trading_pair)
        self._record_order_symbol(order_id, symbol)
        return self.cancel_order(order_id)
