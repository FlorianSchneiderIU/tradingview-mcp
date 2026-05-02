# Spot

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

class Side(Enum):
    """Specify if the order is to 'buy' or 'sell'."""
    BUY = "buy"
    SELL = "sell"

@dataclass
class Chain:
    """chain id of currency"""
    chainId: str
    """chain name of currency"""
    chainName: str
    """Number of block confirmations"""
    confirms: int
    """Contract address"""
    contractAddress: str
    """Minimum deposit amount"""
    depositMinSize: str
    """Support deposit or not"""
    isDepositEnabled: bool
    """Support withdrawal or not"""
    isWithdrawEnabled: bool
    """Maximum amount of single deposit (only applicable to Lightning Network)"""
    maxDeposit: str
    """Maximum amount of single withdrawal"""
    maxWithdraw: str
    """whether memo/tag is needed"""
    needTag: bool
    """The number of blocks (confirmations) for advance on-chain verification"""
    preConfirms: int
    """Minimum fees charged for withdrawal"""
    withdrawalMinFee: str
    """Minimum withdrawal amount"""
    withdrawalMinSize: str
    """withdraw fee rate"""
    withdrawFeeRate: str
    """Withdrawal precision bit, indicating the maximum supported length after the decimal point
    of the withdrawal amount
    """
    withdrawPrecision: int
    """deposit fee rate (some currencies have this param, the default is empty)"""
    depositFeeRate: Optional[str] = None
    depositTierFee: Optional[str] = None
    """withdraw max fee(some currencies have this param, the default is empty)"""
    withdrawMaxFee: Optional[str] = None

@dataclass
class SpotCurrency:
    """chain list"""
    chains: List[Chain]
    """Number of block confirmations"""
    confirms: int
    """Contract address"""
    contractAddress: str
    """A unique currency code that will never change"""
    currency: str
    """Full name of a currency, will change after renaming"""
    fullName: str
    """Support debit or not"""
    isDebitEnabled: bool
    """Support margin or not"""
    isMarginEnabled: bool
    """Currency name, will change after renaming"""
    name: str
    """Currency precision"""
    precision: int

@dataclass
class SpotSymbol:
    symbol: str
    name: str
    baseCurrency: str
    quoteCurrency: str
    feeCurrency: str
    market: str
    baseMinSize: str
    quoteMinSize: str
    baseMaxSize: str
    quoteMaxSize: str
    baseIncrement: str
    quoteIncrement: str
    priceIncrement: str
    priceLimitRate: str
    minFunds: str
    isMarginEnabled: bool
    enableTrading: bool
    feeCategory: int
    makerFeeCoefficient: str
    takerFeeCoefficient: str
    st: bool
    callauctionIsEnabled: bool
    callauctionPriceFloor: Optional[str] = None
    callauctionPriceCeiling: Optional[str] = None
    callauctionFirstStageStartTime: Optional[str] = None
    callauctionSecondStageStartTime: Optional[str] = None
    callauctionThirdStageStartTime: Optional[str] = None
    tradingStartTime: Optional[str] = None

class MakerCoefficient(Enum):
    """The maker fee coefficient. The actual fee needs to be multiplied by this coefficient to
    get the final fee. Most currencies have a coefficient of 1. If set to 0, it means no fee
    
    The taker fee coefficient. The actual fee needs to be multiplied by this coefficient to
    get the final fee. Most currencies have a coefficient of 1. If set to 0, it means no fee
    """
    THE_0 = "0"
    THE_1 = "1"

@dataclass
class SpotTicker:
    averagePrice: str
    bestAskSize: str
    bestBidSize: str
    buy: str
    changePrice: str
    changeRate: str
    high: str
    last: str
    low: str
    makerCoefficient: MakerCoefficient
    makerFeeRate: str
    sell: str
    symbol: str
    symbolName: str
    takerCoefficient: MakerCoefficient
    takerFeeRate: str
    vol: str
    volValue: str
    open: str
    lastSize: str

class AccountType(Enum):
    """Account type:，main、trade、isolated(abandon)、margin(abandon)"""
    MAIN = "main"
    TRADE = "trade"

@dataclass
class SpotAccount:
    available: str
    balance: str
    currency: str
    holds: str
    id: str
    type: AccountType

@dataclass
class SpotBalance:
    available: str
    balance: str
    currency: str
    holds: str

@dataclass
class SpotAccountEnriched(SpotAccount):
    ticker: SpotTicker
    value: float

class Stp(Enum):
    """[Self Trade Prevention](apidog://link/pages/338146) is divided into four strategies: CN,
    CO, CB , and DC
    """
    CB = "CB"
    CN = "CN"
    CO = "CO"
    DC = "DC"

class TimeInForce(Enum):
    """[Time in force](apidog://link/pages/338146) is a special strategy used during trading"""
    FOK = "FOK"
    GTC = "GTC"
    GTT = "GTT"
    IOC = "IOC"

class TypeEnum(Enum):
    """specify if the order is an 'limit' order or 'market' order.
    
    The type of order you specify when you place your order determines whether or not you
    need to request other parameters and also affects the execution of the matching engine.
    
    When placing a limit order, you must specify a price and size. The system will try to
    match the order according to market price or a price better than market price. If the
    order cannot be immediately matched, it will stay in the order book until it is matched
    or the user cancels.
    
    Unlike limit orders, the price for market orders fluctuates with market prices. When
    placing a market order, you do not need to specify a price, you only need to specify a
    quantity. Market orders are filled immediately and will not enter the order book. All
    market orders are takers and a taker fee will be charged.
    """
    LIMIT = "limit"
    MARKET = "market"

@dataclass
class HFOrderRequest:
    side: Side
    symbol: str
    type: TypeEnum
    cancelAfter: Optional[int] = None
    clientOid: Optional[str] = None
    funds: Optional[str] = None
    hidden: Optional[bool] = None
    iceberg: Optional[bool] = None
    postOnly: Optional[bool] = None
    price: Optional[str] = None
    remark: Optional[str] = None
    size: Optional[str] = None
    stp: Optional[Stp] = None
    tags: Optional[str] = None
    timeInForce: Optional[TimeInForce] = None
    visibleSize: Optional[str] = None

class Status(Enum):
    """Order Status. open：order is active; done：order has been completed"""
    DONE = "done"
    OPEN = "open"

@dataclass
class HFOrderReturn:
    canceledSize: str
    clientOid: str
    dealSize: str
    matchTime: int
    orderId: str
    orderTime: int
    originSize: str
    remainSize: str
    status: Status
    originFunds: Optional[str] = None
    dealFunds: Optional[str] = None
    remainFunds: Optional[str] = None
    canceledFunds: Optional[str] = None
    
@dataclass
class HFOrderTestReturn:
    """The user self-defined order id."""
    clientOid: str
    """The unique order id generated by the trading system,which can be used later for further
    actions such as canceling the order.
    """
    orderId: str

@dataclass
class HFOrder:
    """Order status: true-The status of the order isactive; false-The status of the order is done"""
    active: bool
    """A GTT timeInForce that expires in n seconds"""
    cancelAfter: int
    """Whether there is a cancellation record for the order."""
    cancelExist: bool
    """Funds of canceled transactions"""
    cancelledFunds: str
    """Number of canceled transactions"""
    cancelledSize: str
    channel: str
    """Client Order Id，unique identifier created by the user"""
    clientOid: str
    createdAt: int
    """Funds of filled transactions"""
    dealFunds: str
    """Number of filled transactions"""
    dealSize: str
    """[Handling fees](apidog://link/pages/5327739)"""
    fee: str
    """currency used to calculate trading fee"""
    feeCurrency: str
    """Order Funds"""
    funds: str
    """Whether its a hidden order."""
    hidden: bool
    """Whether its a iceberg order."""
    iceberg: bool
    """The unique order id generated by the trading system"""
    id: str
    """Whether to enter the orderbook: true: enter the orderbook; false: not enter the orderbook"""
    inOrderBook: bool
    lastUpdatedAt: int
    opType: str
    """Whether its a postOnly order."""
    postOnly: bool
    """Order price"""
    price: str
    """Funds of remain transactions"""
    remainFunds: str
    """Number of remain transactions"""
    remainSize: str
    """Buy or sell"""
    side: Side
    """Order size"""
    size: str
    """symbol"""
    symbol: str
    """Users in some regions need query this field"""
    tax: str
    """Time in force"""
    timeInForce: TimeInForce
    """Trade type, redundancy param"""
    tradeType: str
    """Specify if the order is an 'limit' order or 'market' order."""
    type: TypeEnum
    """Visible size of iceberg order in order book."""
    visibleSize: str
    """Order placement remarks"""
    remark: Optional[str] = None
    """[Self Trade Prevention](apidog://link/pages/5176570)"""
    stp: Optional[Stp] = None
    """Order tag"""
    tags: Optional[str] = None
    """Reason for order cancellation"""
    cancelReason: Optional[str] = None

@dataclass
class HFOrderCancelResponse:
    canceledSize: str
    dealSize: str
    orderId: str
    originSize: str
    remainSize: str
    status: Status

@dataclass
class HFOrderCancelByClientOidResponse:
    canceledSize: str
    clientOid: str
    dealSize: str
    originSize: str
    remainSize: str
    status: Status

@dataclass
class ApikeyInfo:
    apiKey: str
    apiVersion: int
    createdAt: int
    isMaster: bool
    permission: str
    remark: str
    uid: int
    ipWhitelist: Optional[str] = None
    subName: Optional[str] = None

@dataclass
class SubAccountItem:
    """Sub-account Permission"""
    access: str
    """Time of event"""
    created_at: int
    hosted_status: str
    """Sub-account active permissions: If you do not have the corresponding permissions, you
    must log in to the sub-account and go to the corresponding web page to activate.
    """
    opened_trade_types: List[str]
    """Remarks"""
    remarks: str
    """Sub-account; 2:Enable, 3:Frozen"""
    status: int
    """Sub-account name"""
    sub_name: str
    """Sub-account Permissions"""
    trade_types: List[str]
    """Sub-account type"""
    type: int
    """Sub-account UID"""
    uid: int
    """Sub-account User ID"""
    user_id: str

@dataclass
class SubAccountSummary:
    """Current request page"""
    current_page: int
    items: List[SubAccountItem]
    """Number of results per request. Minimum is 1, maximum is 100"""
    page_size: int
    """Total number of messages"""
    total_num: int
    """Total number of pages"""
    total_page: int
    
@dataclass
class SymbolsWithOpenOrders:
    """The symbol that has active orders"""
    symbols: List[str]


class Liquidity(Enum):
    """Liquidity type: taker or maker"""
    maker = "maker"
    taker = "taker"


@dataclass
class Item:
    """Counterparty order Id"""
    counterOrderId: str
    createdAt: int
    """[Handling fees](apidog://link/pages/5327739)"""
    fee: str
    """currency used to calculate trading fee"""
    feeCurrency: str
    """Fee rate"""
    feeRate: str
    forceTaker: bool
    """Order Funds"""
    funds: str
    """Id of transaction detail"""
    id: int
    """Liquidity type: taker or maker"""
    liquidity: Liquidity
    """The unique order id generated by the trading system"""
    orderId: str
    """Order price"""
    price: str
    """Buy or sell"""
    side: Side
    """Order size"""
    size: str
    """Take Profit and Stop Loss type, currently HFT does not support the Take Profit and Stop
    Loss type, so it is empty
    """
    stop: str
    """symbol"""
    symbol: str
    """Users in some regions need query this field"""
    tax: str
    """Tax Rate, Users in some regions need query this field"""
    taxRate: str
    """Trade Id, symbol latitude increment"""
    tradeId: int
    """Trade type, redundancy param"""
    tradeType: str
    """Specify if the order is an 'limit' order or 'market' order."""
    type: TypeEnum


@dataclass
class TradeHistoryReturn:
    items: List[Item]
    """The id of the last set of data from the previous batch of data. By default, the latest
    information is given.
    lastId is used to filter data and paginate. If lastId is not entered, the default is a
    maximum of 100 returned data items. The return results include lastId，which can be used
    as a query parameter to look up new data from the next page.
    """
    lastId: int
    

@dataclass
class TradeHistoryParams:
    """symbol"""
    symbol: str
    """End time (milisecond)"""
    endAt: Optional[int] = None
    """The id of the last set of data from the previous batch of data. By default, the latest
    information is given.
    lastId is used to filter data and paginate. If lastId is not entered, the default is a
    maximum of 100 returned data items. The return results include lastId，which can be used
    as a query parameter to look up new data from the next page.
    """
    lastId: Optional[int] = None
    """Default20，Max100"""
    limit: Optional[int] = None
    """The unique order id generated by the trading system
    (If orderId is specified，please ignore the other query parameters)
    """
    orderId: Optional[str] = None
    """specify if the order is to 'buy' or 'sell'"""
    side: Optional[Side] = None
    """Start time (milisecond)"""
    startAt: Optional[int] = None
    """specify if the order is an 'limit' order or 'market' order."""
    type: Optional[TypeEnum] = None
