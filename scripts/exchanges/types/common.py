from dataclasses import dataclass
from enum import Enum
import math
import re
from typing import List, Literal, Optional, Union

from my_types.percentage import Percentage

# Common exchange types
Coin = str
OrderSide = Literal["buy", "sell"]
OrderType = Literal["limit", "market"]
MarginMode = Literal["ISOLATED", "CROSS"]
Direction = Literal["long", "short"]
Status = Literal["active", "done", "open"]


class ClientOrderType(Enum):
    LIMIT = "L"
    MARKET = "M"


class ClientOrderGoal(Enum):
    MANUAL_CLOSE = "MC"
    STOP_LOSS = "SL"
    TARGET = "T"  # TODO: add manual target everywhere
    TRIGGER_TARGET = "TT"
    DCA_TARGET = "DT"
    DCA_TRIGGER_TARGET = "DTT"
    ENTRY = "E"
    DCA = "D"
    TRAILING_STOP = "TS"
    TP_STOP = "TPS"


class TrailingStopConfiguration(Enum):
    FIXED = "F"
    ATR = "A"
    SWING = "S"
    NONE = "N"


class TpStopConfiguration(Enum):
    BREAK_EVEN = "B"
    NONE = "N"


class ExchangeMode(Enum):
    SPOT = "spot"
    MARGIN = "margin"
    FUTURES = "futures"
    OPTION = "option"


class Side(Enum):
    """specify if the order is to 'buy' or 'sell'"""

    BUY = "buy"
    SELL = "sell"


class TimestampNanoseconds(int):
    allowed_digits = 19

    def __new__(cls, value: int | str) -> "TimestampNanoseconds":
        """Create a nanoseconds timestamp from an int or numeric string."""
        try:
            int_value = int(value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive branch
            raise ValueError(f"Invalid nanoseconds timestamp format: {value}") from exc

        if not cls.is_valid(int_value):
            digits = math.ceil(math.log10(int_value)) if int_value > 0 else 1
            raise ValueError(
                f"Invalid nanoseconds timestamp format: {int_value} has {digits}!={cls.allowed_digits} digits",
            )
        return int.__new__(cls, int_value)

    def to_milliseconds(self) -> "TimestampMilliseconds":
        """Convert nanoseconds timestamp to milliseconds timestamp."""
        return TimestampMilliseconds(self // 1_000_000)

    @staticmethod
    def is_valid(value: int | str) -> bool:
        """Return True if *value* is a positive 19-digit integer or string."""
        try:
            int_value = int(value)
        except (TypeError, ValueError):
            return False
        if int_value <= 0:
            return False
        digits = math.ceil(math.log10(int_value))
        return digits == TimestampNanoseconds.allowed_digits


class TimestampMilliseconds(int):
    allowed_digits = 13

    def __new__(cls, value: int | str) -> "TimestampMilliseconds":
        """Create a milliseconds timestamp from an int or numeric string."""
        try:
            int_value = int(value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive branch
            raise ValueError(f"Invalid milliseconds timestamp format: {value}") from exc

        if not cls.is_valid(int_value):
            digits = math.ceil(math.log10(int_value)) if int_value > 0 else 1
            raise ValueError(
                f"Invalid milliseconds timestamp format: {int_value} has {digits}!={cls.allowed_digits} digits"
            )
        return int.__new__(cls, int_value)

    @staticmethod
    def is_valid(value: int | str) -> bool:
        """Return True if *value* is a positive 13-digit integer or string."""
        try:
            int_value = int(value)
        except (TypeError, ValueError):
            return False
        if int_value <= 0:
            return False
        digits = math.ceil(math.log10(int_value))
        return digits == TimestampMilliseconds.allowed_digits


class TradingPairSpot(str):
    def __new__(cls, value: str) -> "TradingPairSpot":
        if not cls.is_valid(value):
            raise ValueError(f"Invalid TradingPairSpot: '{value}' must contain '-'")
        return str.__new__(cls, value)

    def coin(self) -> Coin:
        return self.split("-")[0]

    def quote(self) -> str:
        return self.split("-")[1]

    @staticmethod
    def is_valid(trading_pair: str) -> bool:
        # Ensure that both '-' is present in the string
        return "-" in trading_pair


class TradingPair(str):
    def __new__(cls, value: str) -> "TradingPair":
        if not cls.is_valid(value):
            raise ValueError(f"Invalid TradingPair: '{value}' must contain ':' and '/'")
        return str.__new__(cls, value)

    def coin(self) -> Coin:
        return self.split("/")[0]

    def coin_without_multiplier(self) -> str:
        """Returns the coin part without any multiplier (e.g., '1000PEPE' -> 'PEPE')"""
        match = re.match(r"^(\d+)([KM]?)([A-Z]+)", self.coin())
        if match:
            return match.group(3)
        return self.coin()

    def quote(self) -> str:
        return self.split("/")[1].split(":")[0]

    def settle(self) -> str:
        return self.split("/")[1].split(":")[1]

    def pair_and_multiplier(self):
        match = re.match(r"^(\d+)([KM]?)([A-Z]+)", self.coin())
        if match:
            digits_str, multiplier, symbol = match.groups()
            digits = int(digits_str)
            if multiplier == "K":
                digits *= 1_000
            elif multiplier == "M":
                digits *= 1_000_000
        else:
            digits = 1
            symbol = self.coin()
        pair = TradingPair(f"{symbol}/{self.quote()}:{self.settle()}")
        return pair, str(digits)

    @staticmethod
    def is_valid(trading_pair: str) -> bool:
        # Ensure that both ':' and '/' are present in the string
        return ":" in trading_pair and "/" in trading_pair


@dataclass(frozen=True)
class TimeFrameValue:
    minutes: int
    name: str

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name


class TimeFrame(Enum):
    ONE_MINUTE = TimeFrameValue(1, "1m")
    FIVE_MINUTES = TimeFrameValue(5, "5m")
    FIFTEEN_MINUTES = TimeFrameValue(15, "15m")
    THIRTY_MINUTES = TimeFrameValue(30, "30m")
    ONE_HOUR = TimeFrameValue(60, "1h")
    TWO_HOURS = TimeFrameValue(120, "2h")
    FOUR_HOURS = TimeFrameValue(240, "4h")
    EIGHT_HOURS = TimeFrameValue(480, "8h")
    TWELVE_HOURS = TimeFrameValue(720, "12h")
    ONE_DAY = TimeFrameValue(1440, "1d")
    ONE_WEEK = TimeFrameValue(10080, "1w")
    ONE_MONTH = TimeFrameValue(43200, "1M")


@dataclass()
class Balance:
    accountEquity: float
    unrealisedPNL: float
    marginBalance: float
    positionMargin: float
    orderMargin: float
    frozenFunds: float
    availableBalance: float
    currency: str


class ClientOid:
    def __init__(
        self,
        channel: str,
        order_type: ClientOrderType,
        client_order_type: ClientOrderGoal,
        group_id: str,
        number: Optional[int] = None,
        trailing_config: TrailingStopConfiguration = TrailingStopConfiguration.NONE,
        tp_config: TpStopConfiguration = TpStopConfiguration.BREAK_EVEN,
    ) -> None:
        channel = channel.strip("*")
        self.channel = channel
        self.order_type = order_type
        self.client_order_type = client_order_type
        self.number = number
        self.group_id = group_id
        self.trailing_config = trailing_config
        self.tp_config = tp_config
        if len(self.__str__()) > 40:
            raise ValueError(f"ClientOid exceeds 40 characters: {self.__str__()}")

    def __str__(self) -> str:
        base = f"{self.channel}_{self.group_id}_{self.order_type.value}_{self.trailing_config.value}{self.tp_config.value}_{self.client_order_type.value}"
        if self.number:
            return f"{base}_{self.number}"
        return base

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, ClientOid):
            return NotImplemented
        return self.__str__() == value.__str__()

    @staticmethod
    def is_valid_string(client_oid: Optional[str]) -> bool:
        """Return True if *client_oid* is a valid client order id string.

        A valid client order id is a non-empty string consisting of at least
        four underscore-separated parts. ``None`` or empty strings are
        considered invalid.
        """
        if not client_oid:
            return False

        parts = client_oid.split("_")
        if len(parts) < 4:
            return False
        if len(client_oid) > 40:
            return False
        return True

    @staticmethod
    def from_string(client_oid: str) -> "ClientOid":
        parts = client_oid.split("_")

        # Fix for duplicated channel prefix (e.g., 'FC_FC_1751321544502_M_NB_MC')
        # If the first two parts are the same, drop the first one
        if len(parts) >= 2 and parts[0] == parts[1]:
            parts = parts[1:]

        if not ClientOid.is_valid_string("_".join(parts)):
            raise ValueError(f"Invalid client_oid: {client_oid}")
        channel = parts[0]
        group_id = parts[1]
        order_type = ClientOrderType(parts[2])
        try:
            joined_config = parts[3]
            trailing_config = TrailingStopConfiguration(joined_config[:1])
            tp_config = TpStopConfiguration(joined_config[1:])
            client_order_type = ClientOrderGoal(parts[4])
            number = int(parts[5]) if len(parts) > 5 else None
        except Exception as _:
            trailing_config = TrailingStopConfiguration.FIXED
            tp_config = TpStopConfiguration.BREAK_EVEN
            client_order_type = ClientOrderGoal(parts[3])
            number = int(parts[4]) if len(parts) > 4 else None

        return ClientOid(
            channel,
            order_type,
            client_order_type,
            group_id,
            number,
            trailing_config,
            tp_config,
        )


@dataclass()
class Position:
    avgEntryPrice: float
    # autoDeposit: bool
    # bankruptPrice: float
    # crossMode: bool
    # currentComm: float
    # currentCost: float
    currentLots: int
    currentQty: float
    # currentTimestamp: int
    # delevPercentage: float
    id: str
    # isInverse: bool
    isOpen: bool
    leverage: int
    liquidationPrice: float
    # maintainMargin: float
    # maintMargin: float
    # maintMarginReq: float
    marginMode: MarginMode
    markPrice: float
    # markValue: float
    openingTimestamp: TimestampMilliseconds
    # posComm: float
    # posCommCommon: float
    posCost: float
    # posCross: float
    # posCrossMargin: float
    # posFunding: float
    posInit: float
    direction: Direction  # positionSide
    # posLoss: float
    # posMaint: float
    # posMargin: float
    # realisedCost: float
    # realisedGrossCost: float
    # realisedGrossPnl: float
    realisedPnl: float
    # riskLimit: int
    # realLeverage: float
    # settleCurrency: str
    trading_pair: TradingPair
    # unrealisedCost: float
    unrealisedPnl: float
    unrealisedPnlPcnt: float
    unrealisedRoePcnt: float


@dataclass()
class HistoricPosition:
    closeId: str
    userId: str
    trading_pair: TradingPair
    settleCurrency: str
    leverage: str
    type: str
    pnl: str
    realisedGrossCost: str
    # withdrawPnl: str
    tradeFee: str
    fundingFee: str
    openTime: TimestampMilliseconds
    closeTime: TimestampMilliseconds
    openPrice: str
    closePrice: str
    marginMode: str
    maxFilledLots: int


@dataclass()
class OrderBookData:
    trading_pair: TradingPair  # Symbol
    sequence: int  # Ticker sequence number
    asks: List[List[Union[str, int]]]  # Asks: [Price, quantity]
    bids: List[List[Union[str, int]]]  # Bids: [Price, quantity]
    ts: TimestampNanoseconds  # Timestamp


@dataclass()
class Order:
    id: str
    trading_pair: TradingPair
    type: str
    side: Optional[OrderSide]
    price: str
    amountLots: int
    value: str
    dealValue: str
    dealSize: int
    stp: str
    stop: str
    stopPriceType: str
    stopTriggered: bool
    stopPrice: str
    timeInForce: str
    postOnly: bool
    hidden: bool
    iceberg: bool
    leverage: str
    forceHold: bool
    closeOrder: bool
    visibleSize: int
    clientOid: Optional[ClientOid]
    remark: Optional[str]
    tags: str
    isActive: bool
    cancelExist: bool
    createdAt: TimestampMilliseconds
    updatedAt: TimestampMilliseconds
    endAt: Optional[TimestampMilliseconds]
    orderTime: int
    settleCurrency: str
    marginMode: MarginMode
    avgDealPrice: str
    filledLots: int
    filledValue: str
    status: Literal["open", "done"]
    reduceOnly: bool
    species: Literal["entry", "tp", "sl", "unknown"] = "unknown"


@dataclass()
class Fill:
    trading_pair: TradingPair
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
    createdAt: TimestampMilliseconds
    settleCurrency: str
    tradeTime: TimestampNanoseconds
    openFeePay: str
    closeFeePay: str
    marginMode: str
    subTradeType: Optional[str]
    displayType: str


@dataclass()
class CreateOrderResponse:
    orderId: str


@dataclass()
class FuturesMarket:
    markPrice: float  # ?
    maxLeverage: int
    lot_size: float  # multiplier
    daily_high: float  # market['highPrice'],
    daily_low: float  # market['lowPrice'],
    daily_volume: float  # market['volumeOf24h'],
    daily_turnover: float  # market['turnoverOf24h'],
    daily_change: float  # market['priceChg'],
    daily_change_rate: Percentage  # '24h Change Rate': market['priceChgPct'],
    open_interest: float  #'Open Interest': market['openInterest'],
    takerFeeRate: float  # 'Taker Fee Rate': market['takerFeeRate'],


@dataclass()
class Ticker:
    last_price: float
    mark_price: float
    index_price: Optional[float] = None


@dataclass()
class LimitOrderRequest:
    type: Literal["limit"]
    leverage: int
    trading_pair: TradingPair
    side: OrderSide
    size: int
    clientOid: ClientOid
    price: float
    marginMode: MarginMode
    reduceOnly: bool
    takeProfit: Optional[float] = None
    stopLoss: Optional[float] = None
    tpslMode: Optional[str] = None
    tpTriggerBy: Optional[str] = None
    slTriggerBy: Optional[str] = None


@dataclass()
class MarketOrderRequest:
    type: Literal["market"]
    leverage: int
    trading_pair: TradingPair
    order_side: OrderSide
    amount_lots: int
    marginMode: MarginMode
    reduceOnly: bool
    clientOid: ClientOid


@dataclass()
class StopMarketOrderRequest(MarketOrderRequest):
    stop: Literal["down"] | Literal["up"]
    stopPrice: float
    stopPriceType: Literal["TP"] = "TP"
