
# Copy Trading API Response Types

from dataclasses import dataclass
from enum import Enum
from typing import List, Literal, Optional

class MarginMode(Enum):
    """Margin mode: ISOLATED or CROSS"""
    ISOLATED = "ISOLATED"
    CROSS = "CROSS"


class Side(Enum):
    """Specify if the order is to 'buy' or 'sell'."""
    BUY = "buy"
    SELL = "sell"


class StopPriceType(Enum):
    """Either 'TP' or 'MP'"""
    MP = "MP"
    TP = "TP"


class TimeInForce(Enum):
    """Optional for type is 'limit' order, [Time in force](/docs-new/enums-definitions) is a
    special strategy used during trading, default is GTC
    """
    GTC = "GTC"
    IOC = "IOC"


class StopDirection(Enum):
    """Either 'down' or 'up'.  If stop is used,parameter stopPrice and stopPriceType also need
    to be provieded.
    """
    DOWN = "down"
    UP = "up"


class TypeEnum(Enum):
    """specify if the order is an 'limit' order or 'market' order"""
    LIMIT = "limit"
    MARKET = "market"


# Base order parameters
@dataclass
class OrderRequest():
    symbol: str  # Contract code, e.g. XBTUSDM
    clientOid: Optional[str]  # Unique order ID created by users
    remark: Optional[str]  # Remark, max 100 utf8 characters

@dataclass
class CreateOrderRequest(OrderRequest):
    leverage: int  # Leverage, required if not closing position
    side: Side  # buy or sell
    size: int  # Order size, must be a positive integer
    type: TypeEnum  # limit or market, default is 'limit'
    price: Optional[str] = None  # Required for limit orders, not used for market orders
    stop: Optional[StopDirection]=None  # down or up, requires stopPrice and stopPriceType
    stopPriceType: Optional[StopPriceType]=None  # TP, IP, or MP, required if stop is specified
    stopPrice: Optional[str]=None  # Required if stop is specified
    reduceOnly: bool=False  # Reduce position size only, default is false
    marginMode: MarginMode=MarginMode.ISOLATED  # Margin mode, default is ISOLATED

# Additional parameters for limit orders
@dataclass
class LimitOrderRequest(CreateOrderRequest):
    price: str = ""  # Limit price
    timeInForce: Optional[TimeInForce]=None  # Time in force, default is GTC
    postOnly: bool=False  # Post-only flag, invalid if timeInForce is IOC
    hidden: bool=False  # Hidden order flag
    iceberg: bool=False  # Iceberg order flag
    type: Literal['limit']='limit'  # Order type must be limit for limit orders
    visibleSize: Optional[int]=None  # Visible size for iceberg orders

# Additional parameters for market orders
@dataclass
class MarketOrderRequest(CreateOrderRequest):
    type: Literal['market']='market'  # Order type must be market for market orders

@dataclass
class CloseOrderRequest(OrderRequest):
    """A mark to close the position. Set to true by default. If closeOrder is set to true, the
    system will close the position and the position size will become 0. Side, Size and
    Leverage fields can be left empty and the system will determine the side and size automatically."""
    reduceOnly: Optional[Literal[True]]
    size: Optional[int] = None  # Order size, must be a positive integer
    closeOrder: Literal[True] = True
    type: Literal[TypeEnum.MARKET] = TypeEnum.MARKET
    marginMode: MarginMode=MarginMode.ISOLATED  # Margin mode, default is ISOLATED

@dataclass
class AddTakeProfitAndStopLossOrderRequest:
    clientOid: str
    """Unique order ID created by users to identify their orders. The maximum length cannot
    exceed 40, e.g. UUID only allows numbers, characters, underline(_), and separator (-).
    """
    leverage: int
    """Used to calculate the margin to be frozen for the order. If you are to close the
    position, this parameter is not required.
    """
    side: Side
    """Specify if the order is to 'buy' or 'sell'."""
    size: int
    """Order size (lot), must be a positive integer. The quantity unit of coin-swap contracts is
    size (lot), and other units are not supported.
    """
    symbol: str
    """Symbol of the contract. Please refer to [Get Symbol endpoint:
    symbol](/docs-new/rest/futures-trading/market-data/get-all-symbols)
    """
    type: TypeEnum
    """Specify if the order is a 'limit' order or 'market' order"""
    closeOrder: Optional[bool] = None
    """A mark to close the position. Set to false by default. If closeOrder is set to true, the
    system will close the position and the position size will become 0. Side, Size and
    Leverage fields can be left empty and the system will determine the side and size
    automatically.
    """
    hidden: Optional[bool] = None
    """Optional for type is 'limit' order, orders not displaying in order book. When hidden is
    chosen, choosing postOnly is not allowed.
    """
    iceberg: Optional[bool] = None
    """Optional for type is 'limit' order, Only visible portion of the order is displayed in the
    order book. When iceberg is chosen, choosing postOnly is not allowed.
    """
    marginMode: Optional[MarginMode] = None
    """Margin mode: ISOLATED, default: ISOLATED"""
    postOnly: Optional[bool] = None
    """Optional for type is 'limit' order, post only flag, invalid when timeInForce is IOC. When
    postOnly is true, choosing hidden or iceberg is not allowed. The post-only flag ensures
    that the trader always pays the maker fee and provides liquidity to the order book. If
    any part of the order is going to pay taker fees, the order will be fully rejected.
    """
    price: Optional[str] = None
    """Required for type is 'limit' order, indicating the operating price"""
    reduceOnly: Optional[bool] = None
    """A mark to reduce the position size only. Set to false by default. Need to set the
    position size when reduceOnly is true. If set to true, only the orders reducing the
    position size will be executed. If the reduce-only order size exceeds the position size,
    the extra size will be canceled.
    """
    stopPriceType: Optional[StopPriceType] = None
    """Either 'TP' or 'MP'"""
    timeInForce: Optional[TimeInForce] = None
    """Optional for type is 'limit' order, [Time in force](/docs-new/enums-definitions) is a
    special strategy used during trading, default is GTC
    """
    triggerStopDownPrice: Optional[str] = None
    """Stop loss price"""
    triggerStopUpPrice: Optional[str] = None
    """Take profit price"""
    visibleSize: Optional[str] = None
    """Optional for type is 'limit' order, the maximum visible size of an iceberg order. Please
    place order in size (lots). The units of qty (base currency) and valueQty (value) are not
    supported. Need to be defined if iceberg is specified.
    """

@dataclass
class CancelOrderByOrderIdResponse:
    """The orderId that has been canceled"""
    cancelledOrderIds: List[str]

@dataclass
class CancelOrderByClientOidResponse:
    """The clientOid that has been canceled"""
    clientOid: str

@dataclass
class CreateOrderResponse():
    orderId: str
    clientOid: str

@dataclass
class GetMaxOpenSizeResponse:
    """Response for Get Max Open Size endpoint"""
    maxBuyOpenSize: str
    maxSellOpenSize: str
    symbol: str

@dataclass 
class GetMaxWithdrawMarginResponse:
    """Response for Get Max Withdraw Margin endpoint"""
    maxWithdrawMargin: str
    symbol: str

@dataclass
class AddIsolatedMarginResponse:
    """Response for Add Isolated Margin endpoint"""
    id: str
    symbol: str
    autoDeposit: bool
    
@dataclass
class RemoveIsolatedMarginResponse:
    """Response for Remove Isolated Margin endpoint"""
    id: str
    symbol: str
    autoDeposit: bool

@dataclass
class ModifyIsolatedMarginRiskLimitResponse:
    """Response for Modify Isolated Margin Risk Limit endpoint"""
    symbol: str
    riskLimitLevel: int

@dataclass
class ModifyAutoDepositStatusResponse:
    """Response for Modify Auto-Deposit Status endpoint"""
    symbol: str
    autoDeposit: bool

# Copy Trading API Request Types
@dataclass 
class GetMaxOpenSizeRequest:
    """Request for Get Max Open Size endpoint"""
    symbol: str
    price: str
    leverage: str

@dataclass
class GetMaxWithdrawMarginRequest:
    """Request for Get Max Withdraw Margin endpoint"""
    symbol: str

@dataclass
class AddIsolatedMarginRequest:
    """Request for Add Isolated Margin endpoint"""
    symbol: str
    margin: str
    bizNo: str

@dataclass
class RemoveIsolatedMarginRequest:
    """Request for Remove Isolated Margin endpoint"""
    symbol: str
    withdrawAmount: str

@dataclass
class ModifyIsolatedMarginRiskLimitRequest:
    """Request for Modify Isolated Margin Risk Limit endpoint"""
    symbol: str
    level: int

@dataclass
class ModifyAutoDepositStatusRequest:
    """Request for Modify Auto-Deposit Status endpoint"""
    symbol: str
    status: bool

@dataclass
class SwitchMarginModeRequest:
    """Request for Switch Margin Mode endpoint (Copy Trading)"""
    symbol: str
    marginMode: str  # 'ISOLATED' or 'CROSS'

@dataclass
class SwitchMarginModeResponse:
    """Response for Switch Margin Mode endpoint (Copy Trading)"""
    symbol: str
    marginMode: str
