from enum import Enum
from dataclasses import dataclass
from typing import Literal, Optional, List


class SymbolNotFoundError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class SymbolNotSupportedForTradingException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


# Bitunix API Types
BitunixMarginMode = Literal["ISOLATION", "CROSS"]
# Currently Bitunix futures are settled in USDT only.  Use a distinct type to
# make request models clearer.
BitunixMarginCoin = Literal["USDT"]
BitunixOrderSide = Literal["BUY", "SELL"]
BitunixTradeSide = Literal["OPEN", "CLOSE"]
BitunixOrderType = Literal["LIMIT", "MARKET"]
BitunixEffect = Literal["IOC", "FOK", "GTC", "POST_ONLY"]
BitunixStopType = Literal["MARK_PRICE", "LAST_PRICE"]
BitunixPositionMode = Literal["ONE_WAY", "HEDGE"]
BitunixOrderStatus = Literal["INIT", "NEW", "PART_FILLED", "CANCELED", "FILLED"]


class BitunixContract(str):
    """Custom type for Bitunix symbols"""

    def __new__(cls, value: str):
        if not cls.is_valid(value):
            raise ValueError("Not a valid Bitunix contract symbol")
        return str.__new__(cls, value)

    @staticmethod
    def is_valid(symbol: str) -> bool:
        if not symbol:
            return False
        if not symbol.isupper():
            return False
        if not symbol.endswith("USDT"):
            return False
        return True

    def __str__(self):
        return self.upper()


@dataclass
class BitunixTradingPair:
    """Bitunix trading pair information"""

    symbol: BitunixContract
    base: str
    quote: str
    minTradeVolume: str
    minBuyPriceOffset: str
    maxSellPriceOffset: str
    maxLimitOrderVolume: str
    maxMarketOrderVolume: str
    basePrecision: int
    quotePrecision: int
    maxLeverage: int
    minLeverage: int
    defaultLeverage: int
    defaultMarginMode: str
    priceProtectScope: str
    symbolStatus: str


@dataclass
class BitunixAccount:
    """Bitunix account information"""

    marginCoin: str
    available: str
    frozen: str
    margin: str
    transfer: str
    positionMode: str
    crossUnrealizedPNL: str
    isolationUnrealizedPNL: str
    bonus: str


@dataclass
class BitunixPosition:
    """Bitunix position information"""

    positionId: str
    symbol: BitunixContract
    marginCoin: Optional[str]  # Can be None in API response
    qty: str
    entryValue: str
    side: str  # 'BUY' or 'SELL'
    marginMode: BitunixMarginMode
    positionMode: str  # 'HEDGE' or 'ONE_WAY'
    leverage: int
    fee: str
    funding: str
    realizedPNL: str
    margin: str
    unrealizedPNL: str
    liqPrice: str  # Liquidation price
    avgOpenPrice: str
    marginRate: str
    ctime: str  # Creation time as string (milliseconds)
    mtime: str  # Modification time as string (milliseconds)


@dataclass
class BitunixHistoricPosition:
    """Bitunix historic position information from get_history_positions API"""

    positionId: str
    symbol: BitunixContract
    maxQty: str  # Maximum quantity
    qty: str  # Actual quantity (not just max)
    entryPrice: str  # Entry price
    closePrice: str  # Close price
    side: BitunixOrderSide
    marginMode: BitunixMarginMode
    positionMode: BitunixPositionMode
    leverage: str  # Leverage as string (not int)
    fee: str
    funding: str
    realizedPNL: str
    liqPrice: str  # Liquidation price
    ctime: str  # Creation time as string (milliseconds)
    mtime: str  # Modification time as string (milliseconds)
    marginCoin: Optional[str] = None  # Can be None in API response
    liqQty: Optional[str] = None  # Can be None (liquidated quantity)
    margin: Optional[str] = None  # Can be None in API response


@dataclass
class BitunixOrder:
    """Bitunix order information"""

    orderId: str
    marginCoin: Optional[str]  # Can be None in API response
    symbol: BitunixContract
    qty: str
    tradeQty: str
    positionMode: BitunixPositionMode
    marginMode: BitunixMarginMode
    leverage: int
    price: str  # Can be "MARKET" for market orders
    avgPrice: Optional[str]  # Average execution price
    side: BitunixOrderSide
    orderType: Literal["LIMIT", "MARKET"]
    effect: Literal["IOC", "FOK", "GTC", "POST_ONLY"]
    clientId: str
    reduceOnly: bool
    status: Literal["INIT", "NEW", "PART_FILLED", "CANCELED", "FILLED"]
    fee: str
    realizedPNL: str
    tpPrice: Optional[str]  # Can be None
    tpStopType: Optional[Literal["MARK_PRICE", "LAST_PRICE"]]  # Can be None
    tpOrderType: Optional[Literal["LIMIT", "MARKET"]]  # Can be None
    tpOrderPrice: Optional[str]  # Can be None
    slPrice: Optional[str]  # Can be None
    slStopType: Optional[Literal["MARK_PRICE", "LAST_PRICE"]]  # Can be None
    slOrderType: Optional[Literal["LIMIT", "MARKET"]]  # Can be None
    slOrderPrice: Optional[str]  # Can be None
    ctime: str  # String timestamp in milliseconds
    mtime: str  # String timestamp in milliseconds


@dataclass
class BitunixHistoryTrade:
    """Bitunix historical trade/fill information"""

    tradeId: str
    orderId: str
    symbol: BitunixContract
    qty: str
    positionMode: BitunixPositionMode
    marginMode: BitunixMarginMode
    leverage: int
    price: str
    side: BitunixOrderSide
    orderType: BitunixOrderType
    reduceOnly: bool
    fee: str
    realizedPNL: str
    ctime: str  # String timestamp in milliseconds
    roleType: Literal["TAKER", "MAKER"]
    marginCoin: Optional[BitunixMarginCoin] = None  # Can be None in historical trades
    effect: Optional[BitunixEffect] = None  # Can be None in historical trades
    clientId: Optional[str] = None  # Can be None in historical trades
    status: Optional[str] = None  # Can be None in historical trades


# Request/Response Types for API


@dataclass
class PlaceOrderRequest:
    symbol: BitunixContract
    qty: str
    side: BitunixOrderSide
    tradeSide: BitunixTradeSide
    orderType: BitunixOrderType
    price: Optional[str] = None
    positionId: Optional[str] = None
    effect: Optional[BitunixEffect] = None
    clientId: Optional[str] = None
    reduceOnly: Optional[bool] = None
    tpPrice: Optional[str] = None
    tpStopType: Optional[BitunixStopType] = None
    tpOrderType: Optional[BitunixOrderType] = None
    tpOrderPrice: Optional[str] = None
    slPrice: Optional[str] = None
    slStopType: Optional[BitunixStopType] = None
    slOrderType: Optional[BitunixOrderType] = None
    slOrderPrice: Optional[str] = None


@dataclass
class PlaceOrderResponse:
    orderId: str
    clientId: str


@dataclass
class CancelOrderRequest:
    orderId: str


@dataclass
class CancelOrderResponse:
    orderId: str
    status: str


@dataclass
class ModifyOrderRequest:
    orderId: Optional[str] = None
    clientId: Optional[str] = None
    qty: Optional[str] = None
    price: Optional[str] = None
    tpPrice: Optional[str] = None
    tpStopType: Optional[BitunixStopType] = None
    tpOrderType: Optional[BitunixOrderType] = None
    tpOrderPrice: Optional[str] = None
    slPrice: Optional[str] = None
    slStopType: Optional[BitunixStopType] = None
    slOrderType: Optional[BitunixOrderType] = None
    slOrderPrice: Optional[str] = None


@dataclass
class ModifyOrderResponse:
    orderId: str
    status: str


@dataclass
class GetOrderRequest:
    orderId: Optional[str] = None
    clientId: Optional[str] = None


@dataclass
class GetPendingOrdersRequest:
    symbol: Optional[BitunixContract] = None
    orderId: Optional[str] = None
    clientId: Optional[str] = None
    # Only allow statuses relevant for pending orders per API: NEW or PART_FILLED
    status: Optional[Literal["NEW", "PART_FILLED"]] = None
    # Unix timestamps in milliseconds
    startTime: Optional[int] = None
    endTime: Optional[int] = None
    # pagination
    skip: int = 0
    # Number of queries, default 10, maximum 100 (not enforced here)
    limit: Optional[int] = 10


@dataclass
class GetHistoryOrdersRequest:
    """Request model for historical orders (by position create time).

    Fields:
        symbol: Trading pair
        positionId: Position id
        startTime: Unix timestamp in milliseconds (position create time)
        endTime: Unix timestamp in milliseconds (position create time)
        skip: skip order count, default 0
        limit: number of queries, default 10, max 100
    """

    symbol: Optional[BitunixContract] = None
    positionId: Optional[str] = None
    startTime: Optional[int] = None
    endTime: Optional[int] = None
    skip: int = 0
    limit: Optional[int] = None


@dataclass
class GetTradesRequest:
    symbol: Optional[BitunixContract] = None
    orderId: Optional[str] = None
    startTime: Optional[int] = None
    endTime: Optional[int] = None
    limit: Optional[int] = None
    roleType: Optional[Literal["TAKER", "MAKER"]] = None


@dataclass
class GetPositionsRequest:
    symbol: Optional[str] = None
    marginCoin: Optional[BitunixMarginCoin] = None
    startTime: Optional[int] = None
    endTime: Optional[int] = None
    skip: Optional[int] = None
    limit: Optional[int] = None


@dataclass
class ChangeLeverageRequest:
    marginCoin: BitunixMarginCoin
    symbol: BitunixContract
    leverage: int


@dataclass
class ChangeLeverageResponse:
    marginCoin: BitunixMarginCoin
    symbol: BitunixContract
    leverage: int


@dataclass
class ChangeMarginModeRequest:
    marginCoin: BitunixMarginCoin
    symbol: BitunixContract
    marginMode: BitunixMarginMode


@dataclass
class ChangeMarginModeResponse:
    symbol: BitunixContract
    marginMode: BitunixMarginMode
    marginCoin: BitunixMarginCoin


@dataclass
class ClosePositionRequest:
    symbol: BitunixContract
    positionId: Optional[str] = None


@dataclass
class ClosePositionResponse:
    orderId: str


# Additional request/response models for private interface


@dataclass
class AdjustMarginRequest1:
    symbol: BitunixContract
    marginCoin: BitunixMarginCoin
    amount: str
    side: Literal["LONG", "SHORT"]


@dataclass
class AdjustMarginRequest2:
    symbol: BitunixContract
    marginCoin: BitunixMarginCoin
    amount: str
    positionId: str


@dataclass
class AdjustMarginResponse:
    msg: str


@dataclass
class ChangePositionModeRequest:
    positionMode: BitunixPositionMode


@dataclass
class ChangePositionModeResponse:
    positionMode: BitunixPositionMode


@dataclass
class GetLeverageMarginModeResponse:
    symbol: BitunixContract
    marginCoin: BitunixMarginCoin
    leverage: int
    marginMode: BitunixMarginMode


@dataclass
class AssetQueryResponse:
    available: str
    maxTransfer: str


@dataclass
class TransferRequest:
    amount: str
    assetType: Literal["FUTURES", "SPOT"]


@dataclass
class TransferResponse:
    success: bool


@dataclass
class CancelAllOrdersRequest:
    symbol: Optional[str] = None


@dataclass
class OrderIdentifier:
    orderId: Optional[str] = None
    clientId: Optional[str] = None


@dataclass
class CancelOrdersRequest:
    symbol: BitunixContract
    orderList: List[OrderIdentifier]


@dataclass
class CancelAllOrdersResponse:
    successList: List[OrderIdentifier]
    failureList: List[OrderIdentifier]


@dataclass
class FlashClosePositionRequest:
    positionId: str


@dataclass
class FlashClosePositionResponse:
    positionId: str


@dataclass
class BatchOrderRequest:
    symbol: BitunixContract
    orderList: List[PlaceOrderRequest]


@dataclass
class BatchOrderResult:
    id: str
    clientId: str


@dataclass
class BatchOrderResponse:
    successList: List[BatchOrderResult]
    failureList: List[BatchOrderResult]


# Ticker and Market Data Types


@dataclass
class BitunixTicker:
    """Bitunix ticker information"""

    symbol: BitunixContract
    markPrice: str
    lastPrice: str
    open: str
    last: str
    quoteVol: str
    baseVol: str
    high: str
    low: str


@dataclass
class BitunixOrderBook:
    """Bitunix order book data"""

    symbol: BitunixContract
    bids: List[List[str]]
    asks: List[List[str]]
    timestamp: int


@dataclass
class BitunixKline:
    """Bitunix kline/candlestick data"""

    openTime: int
    open: str
    high: str
    low: str
    close: str
    volume: str
    closeTime: int
    quoteVolume: str
    count: int


class BitunixKlineInterval(Enum):
    """Kline intervals"""

    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H2 = "2h"
    H4 = "4h"
    H6 = "6h"
    H8 = "8h"
    H12 = "12h"
    D1 = "1d"
    D3 = "3d"
    W1 = "1w"
    MONTH1 = "1M"


# ---------------------------------------------------------------------------
# Funding rate information
# ---------------------------------------------------------------------------


@dataclass
class BitunixFundingRate:
    symbol: BitunixContract
    markPrice: str
    lastPrice: str
    fundingRate: str


# ---------------------------------------------------------------------------
# TP/SL order management
# ---------------------------------------------------------------------------


@dataclass
class CancelTpslOrderRequest:
    symbol: BitunixContract
    orderId: str


@dataclass
class TpslOrderIdResponse:
    orderId: str


@dataclass
class PositionTpslOrderRequest:
    symbol: BitunixContract
    positionId: str
    tpPrice: Optional[str] = None
    tpStopType: Optional[BitunixStopType] = None
    slPrice: Optional[str] = None
    slStopType: Optional[BitunixStopType] = None


@dataclass
class TpslPlaceOrderRequest(PositionTpslOrderRequest):
    tpOrderType: Optional[BitunixOrderType] = None
    tpOrderPrice: Optional[str] = None
    slOrderType: Optional[BitunixOrderType] = None
    slOrderPrice: Optional[str] = None
    tpQty: Optional[str] = None
    slQty: Optional[str] = None


@dataclass
class TpslModifyOrderRequest(TpslPlaceOrderRequest):
    orderId: str | None = None


@dataclass
class TpslOrder:
    id: str
    positionId: str
    symbol: BitunixContract
    base: str
    quote: str
    tpPrice: Optional[str]
    tpStopType: Optional[BitunixStopType]
    slPrice: Optional[str]
    slStopType: Optional[BitunixStopType]
    tpOrderType: Optional[BitunixOrderType]
    tpOrderPrice: Optional[str]
    slOrderType: Optional[BitunixOrderType]
    slOrderPrice: Optional[str]
    tpQty: Optional[str]
    slQty: Optional[str]


@dataclass
class HistoryTpslOrder(TpslOrder):
    status: str
    ctime: int
    triggerTime: Optional[int]


@dataclass
class GetTpslOrdersRequest:
    symbol: Optional[str] = None
    positionId: Optional[str] = None
    side: Optional[int] = None
    positionMode: Optional[int] = None
    startTime: Optional[int] = None
    endTime: Optional[int] = None
    skip: Optional[int] = None
    limit: Optional[int] = None
