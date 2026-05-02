from decimal import Decimal
from typing import Literal, Optional, List, TypedDict, Union

from exchanges.types.common import OrderSide, OrderType, TimestampNanoseconds, TradingPair
from exchanges.types.exceptions import SymbolNotFoundError

# Type definitions
StopDirection = Literal['down', 'up']
StopPriceType = Literal['TP', 'IP', 'MP']
SelfTradePrevention = Literal['CN', 'CO', 'CB']
MarginMode = Literal['ISOLATED', 'CROSS']
TimeInForce = Literal['GTC', 'IOC']
Num = Union[None, str, float, int, Decimal]
Str = Optional[str]
Bool = Optional[bool]
Int = Optional[int]
MarketType = Literal['spot', 'margin', 'swap', 'future', 'option']
SubType = Literal['linear', 'inverse']

class KucoinFuturesContract(str):
    def __new__(cls, value: str):
        if not value.endswith('USDTM'):
            raise SymbolNotFoundError(f"Symbol {value} does not end with 'USDTM'")
        return super().__new__(cls, value)

    def __str__(self):
        return self

    def __repr__(self):
        return f"KucoinSymbol('{self}')"

# Base order parameters
class OrderRequest(TypedDict):
    clientOid: Optional[str]  # Unique order ID created by users
    symbol: KucoinFuturesContract | TradingPair  # Contract code, e.g. XBTUSDM
    marginMode: Optional[MarginMode]  # Margin mode: ISOLATED, CROSS, default is ISOLATED
    remark: Optional[str]  # Remark, max 100 utf8 characters

class CreateOrderRequest(OrderRequest):
    leverage: str  # Leverage, required if not closing position
    stop: Optional[StopDirection]  # down or up, requires stopPrice and stopPriceType
    stopPriceType: Optional[StopPriceType]  # TP, IP, or MP, required if stop is specified
    stopPrice: Str  # Required if stop is specified
    reduceOnly: Bool  # Reduce position size only, default is False
    forceHold: Bool  # Forcely hold funds, default is False
    stp: Optional[SelfTradePrevention]  # Self-trade prevention, CN, CO, CB
    side: OrderSide  # buy or sell
    size: int  # Order size, must be a positive integer
    type: Optional[OrderType]  # limit or market, default is 'limit'

# Additional parameters for limit orders
class LimitOrderRequest(CreateOrderRequest):
    price: str  # Limit price
    timeInForce: Optional[TimeInForce]  # Time in force, default is GTC
    postOnly: Bool  # Post-only flag, invalid if timeInForce is IOC
    hidden: Bool  # Hidden order flag
    iceberg: Bool  # Iceberg order flag
    visibleSize: Optional[int]  # Visible size for iceberg orders
    type: Literal['limit']  # Order type must be limit for limit orders

class AdvancedOrderParameters(TypedDict, total=False):
    stop: Optional[StopDirection]  # down or up, requires stopPrice and stopPriceType
    stopPriceType: Optional[StopPriceType]  # TP, IP, or MP, required if stop is specified
    stopPrice: Str  # Required if stop is specified
    reduceOnly: Bool  # Reduce position size only, default is False

# Additional parameters for market orders
class MarketOrderRequest(CreateOrderRequest):
    size: int  # Amount of contract to buy or sell, only required for market orders
    type: Literal['market']  # Order type must be market for market orders

class CloseOrderRequest(OrderRequest):
    closeOrder: Bool  # Must be set to True for close order
    type: Literal['market']
    stop: Optional[StopDirection]  # down or up, requires stopPrice and stopPriceType
    stopPriceType: Optional[StopPriceType]  # TP, IP, or MP, required if stop is specified
    stopPrice: Str  # Required if stop is specified
    reduceOnly: Literal[True]  # Reduce position size only, default is False
 
class CreateOrderResponse(TypedDict):
    orderId: str

# Base order parameters
class BaseStOrderRequest(TypedDict):
    clientOid: str  # Required: Unique order ID
    side: OrderSide  # Required: Order side
    symbol: str  # Required: Contract code, e.g., XBTUSDM
    leverage: str  # Required: Leverage level
    type: OrderType  # Required: Order type
    remark: Optional[str]  # Optional: Order remark, max 100 utf8 characters
    triggerStopUpPrice: Optional[str]  # Optional: Take profit price
    stopPriceType: Optional[Literal['TP', 'IP', 'MP']]  # Optional: Stop price type
    triggerStopDownPrice: Optional[str]  # Optional: Stop loss price
    reduceOnly: Optional[bool]  # Optional: Reduce position size only, default False
    closeOrder: Optional[bool]  # Optional: Close position, default False
    forceHold: Optional[bool]  # Optional: Forcely hold funds, default False
    stp: Optional[Literal['CN', 'CO', 'CB']]  # Optional: Self-trade prevention
    marginMode: MarginMode  # Required: Margin mode, ISOLATED or CROSS

# Additional parameters for limit orders
class LimitStOrderRequest(BaseStOrderRequest):
    type: Literal['limit']  # Order type must be 'limit'
    price: str  # Required: Limit price
    size: int  # Required: Order size, must be positive
    timeInForce: Optional[Literal['GTC', 'IOC']]  # Optional: Time in force, default 'GTC'
    postOnly: Optional[bool]  # Optional: Post-only flag
    hidden: Optional[bool]  # Optional: Hidden order flag
    iceberg: Optional[bool]  # Optional: Iceberg order flag
    visibleSize: Optional[int]  # Optional: Visible size for iceberg orders

class TpOrderRequest(TypedDict):
    side: OrderSide
    symbol: KucoinFuturesContract | TradingPair
    leverage: int
    size: int
    triggerStopUpPrice: str
    marginMode: MarginMode
    

class SlOrderRequest(TypedDict):
    side: OrderSide
    symbol: KucoinFuturesContract | TradingPair
    leverage: int
    size: int
    triggerStopDownPrice: str
    marginMode: MarginMode

# Additional parameters for market orders
class MarketStOrderRequest(BaseStOrderRequest):
    type: Literal['market']  # Order type must be 'market'
    size: int  # Optional: Amount of contract to buy or sell

# Union type for StOrderRequest
StOrderRequest = Union[LimitStOrderRequest, MarketStOrderRequest]

PositionDirection = Literal['LONG','SHORT','BOTH']

class Position(TypedDict):
    id: str
    symbol: KucoinFuturesContract
    autoDeposit: bool
    crossMode: bool
    maintMarginReq: float
    riskLimit: int
    realLeverage: float
    delevPercentage: float
    openingTimestamp: int
    currentTimestamp: int
    currentQty: int
    currentCost: float
    currentComm: float
    unrealisedCost: float
    realisedGrossCost: float
    realisedCost: float
    isOpen: bool
    markPrice: float
    markValue: float
    posCost: float
    posCross: float
    posCrossMargin: float
    posInit: float
    posComm: float
    posCommCommon: float
    posLoss: float
    posMargin: float
    posFunding: float
    posMaint: float
    maintMargin: float
    realisedGrossPnl: float
    realisedPnl: float
    unrealisedPnl: float
    unrealisedPnlPcnt: float
    unrealisedRoePcnt: float
    avgEntryPrice: float
    liquidationPrice: float
    bankruptPrice: float
    settleCurrency: str
    isInverse: bool
    maintainMargin: float
    marginMode: MarginMode
    positionSide: PositionDirection
    leverage: float

class HistoricPosition(TypedDict):
    closeId: str
    userId: str
    symbol: KucoinFuturesContract
    settleCurrency: str
    leverage: str
    type: str
    pnl: str
    realisedGrossCost: str
    withdrawPnl: str
    tradeFee: str
    fundingFee: str
    openTime: int
    closeTime: int
    openPrice: str
    closePrice: str
    marginMode: str

class Fill(TypedDict):
    symbol: KucoinFuturesContract
    tradeId: str
    orderId: str
    side: OrderSide
    liquidity: str
    forceTaker: bool
    price: str
    size: int
    value: str
    feeRate: str
    fixFee: str
    feeCurrency: str
    stop: str
    fee: str
    orderType: str
    tradeType: str
    createdAt: int
    settleCurrency: str
    tradeTime: int
    openFeePay: str
    closeFeePay: str
    marginMode: MarginMode
    subTradeType: Optional[str]
    displayType: str

class OrderRequestParams(TypedDict, total=False):
    status: Optional[Literal['active','done']]  # "active" or "done", defaults to "done"
    symbol: Optional[TradingPair | KucoinFuturesContract]  # Symbol of the contract
    side: Optional[OrderSide]  # "buy" or "sell"
    type: Optional[Literal['limit','market','limit_stop','market_stop']]  # "limit", "market", "limit_stop", or "market_stop"
    startAt: Optional[int]  # Start time in milliseconds
    endAt: Optional[int]  # End time in milliseconds
    currentPage: Optional[int]  # Current request page, defaults to 1
    pageSize: Optional[int]  # Page size, defaults to 50, maximum is 1000

class OrderBookData(TypedDict):
    symbol: str  # Symbol
    sequence: int  # Ticker sequence number
    asks: List[List[Union[str, int]]]  # Asks: [Price, quantity]
    bids: List[List[Union[str, int]]]  # Bids: [Price, quantity]
    ts: TimestampNanoseconds  # Timestamp

class Order(TypedDict):
    id: str  # Order ID
    symbol: KucoinFuturesContract  # Symbol of the contract
    type: str  # Order type, market order or limit order
    side: Optional[OrderSide]  # Transaction side
    price: str  # Order price (as a string to preserve precision)
    size: int  # Order quantity
    value: str  # Order value (as a string to preserve precision)
    dealValue: str  # Executed size of funds (as a string)
    dealSize: int  # Executed quantity
    stp: str  # Self-trade prevention
    stop: str  # Stop order type (stop limit or stop market)
    stopPriceType: str  # Trigger price type of stop orders
    stopTriggered: bool  # Indicates if the stop order is triggered
    stopPrice: str  # Trigger price of stop orders (as a string)
    timeInForce: str  # Time in force policy type
    postOnly: bool  # Indicates if the order is post-only
    hidden: bool  # Indicates if the order is hidden
    iceberg: bool  # Indicates if the order is an iceberg order
    leverage: str  # Leverage of the order (as a string)
    forceHold: bool  # Indicates if funds are forcefully held for the order
    closeOrder: bool  # Indicates if the order closes the position
    visibleSize: int  # Visible size of the iceberg order
    clientOid: str  # Unique client order ID
    remark: Optional[str]  # Remark of the order
    tags: str  # Order source tags
    isActive: bool  # Indicates if the order is active
    cancelExist: bool  # Indicates if a cancel request exists
    createdAt: int  # Time the order was created
    updatedAt: int  # Last update time
    endAt: Optional[int]  # End time
    orderTime: int  # Order creation time in nanoseconds
    settleCurrency: str  # Settlement currency
    marginMode: MarginMode  # Margin mode: ISOLATED or CROSS
    avgDealPrice: str  # Average transaction price (as a string)
    filledSize: int  # Executed quantity
    filledValue: str  # Value of executed orders (as a string)
    status: Literal['open', 'done']  # Order status: "open" or "done"
    reduceOnly: bool  # Indicates if the order reduces the position size only

class FuturesMarket(TypedDict):
    symbol: KucoinFuturesContract
    rootSymbol: str
    type: str
    firstOpenDate: int
    expireDate: Optional[int]
    settleDate: Optional[int]
    baseCurrency: str
    quoteCurrency: str
    settleCurrency: str
    maxOrderQty: int
    maxPrice: float
    lotSize: int
    tickSize: float
    indexPriceTickSize: float
    multiplier: float
    initialMargin: float
    maintainMargin: float
    maxRiskLimit: int
    minRiskLimit: int
    riskStep: int
    makerFeeRate: float
    takerFeeRate: float
    takerFixFee: float
    makerFixFee: float
    settlementFee: Optional[float]
    isDeleverage: bool
    isQuanto: bool
    isInverse: bool
    markMethod: str
    fairMethod: str
    fundingBaseSymbol: str
    fundingQuoteSymbol: str
    fundingRateSymbol: str
    indexSymbol: str
    settlementSymbol: str
    status: str
    fundingFeeRate: float
    predictedFundingFeeRate: float
    fundingRateGranularity: int
    openInterest: str
    turnoverOf24h: float
    volumeOf24h: float
    markPrice: float
    indexPrice: float
    lastTradePrice: float
    nextFundingRateTime: int
    maxLeverage: int
    sourceExchanges: List[str]
    premiumsSymbol1M: str
    premiumsSymbol8H: str
    fundingBaseSymbol1M: str
    fundingQuoteSymbol1M: str
    lowPrice: float
    highPrice: float
    priceChgPct: float
    priceChg: float
    k: float
    m: float
    f: float
    mmrLimit: float
    mmrLevConstant: float

class Balance(TypedDict):
    accountEquity: float
    unrealisedPNL: float
    marginBalance: float
    positionMargin: float
    orderMargin: float
    frozenFunds: float
    availableBalance: float
    currency: str
    
class Ticker(TypedDict):
    sequence: int
    symbol: str
    side: str
    size: int
    tradeId: str
    price: str
    bestBidPrice: str
    bestBidSize: int
    bestAskPrice: str
    bestAskSize: int
    ts: int
