from __future__ import annotations
from datetime import datetime, timezone
from abc import ABC, abstractmethod
from typing import Dict, List, Literal, Optional
import pandas as pd

from exchanges.types.common import ClientOid, Direction, FuturesMarket, HistoricPosition, LimitOrderRequest, MarketOrderRequest, OrderBookData, OrderSide, Status, StopMarketOrderRequest, Ticker, TimeFrame, TimestampMilliseconds, TradingPair, Balance, Position, Order, Fill, CreateOrderResponse, MarginMode
from utils import validate_config
from utils.CryptoMarketCap import CryptoMarketCap
from utils.SqlManager import SQLiteManager

"""futures_exchange.py

A **minimal, fully‑typed interface** that every concrete futures‑exchange
adapter in this code‑base MUST implement.  It deliberately contains *no*
implementation logic – only public type definitions and ``@abstractmethod``
contracts that define the shape of the API.

The goal is that trading‑strategy code can depend solely on
:class:`FuturesExchange` while remaining agnostic about the underlying
exchange (Kucoin, Bitunix, …).
"""

###############################################################################
# Abstract interface                                                           #
###############################################################################

class IFuturesExchange(ABC):
    """Abstract base‑class describing the minimum required exchange surface."""
    db_manager: SQLiteManager
    @abstractmethod
    def get_name(self) -> str:
        """Return the name of the exchange."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Market metadata
    # ------------------------------------------------------------------


    @abstractmethod
    def load_markets(self) -> None:
        """Initialise or refresh the *in‑memory* market cache."""
        raise NotImplementedError

    @abstractmethod
    def market(self, trading_pair: TradingPair) -> FuturesMarket:
        """Return the raw market‑definition for *trading_pair*."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Public market‑data endpoints
    # ------------------------------------------------------------------

    @abstractmethod
    def fetch_order_book(
        self,
        trading_pair: TradingPair,
        depth: int | Literal["full"] = 20,
    ) -> OrderBookData:
        raise NotImplementedError

    @abstractmethod
    def fetch_ticker(self, trading_pair: TradingPair) -> Ticker:
        raise NotImplementedError

    @abstractmethod
    def fetch_positions_history(
        self,
        since: Optional[TimestampMilliseconds] = None,
        trading_pair: Optional[TradingPair] = None,
    ) -> List[HistoricPosition]:
        """Fetch historical positions for a trading pair."""
        raise NotImplementedError

    @abstractmethod
    def fetch_ohlcv(
        self,
        trading_pair: TradingPair,
        timeframe: TimeFrame,
        since: Optional[TimestampMilliseconds] = None,
        until: Optional[TimestampMilliseconds] = None,
    ) -> pd.DataFrame:  # Returns DataFrame with columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        """Fetch OHLCV data and return as pandas DataFrame.
        
        Returns:
            pd.DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            where timestamp is converted to datetime using pd.to_datetime(unit='ms')
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Trading endpoints
    # ------------------------------------------------------------------

    @abstractmethod
    def create_limit_order(
        self,
        params: LimitOrderRequest,
        test: bool = False,
    ) -> CreateOrderResponse:
        raise NotImplementedError

    @abstractmethod
    def create_market_order(
        self,
        params: MarketOrderRequest | StopMarketOrderRequest,
        test: bool = False,
    ) -> CreateOrderResponse:
        raise NotImplementedError

    @abstractmethod
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
        """Create a take profit order.
        
        Args:
            trading_pair: The trading pair
            position_direction: Direction of the position to protect ('long' or 'short')
            size: Size in lots/contracts
            price: Take profit trigger price
            leverage: Position leverage
            clientOid: Client order ID
            margin_mode: Margin mode for the order ("CROSS" or "ISOLATED")
            test: Whether this is a test order
            
        Returns:
            CreateOrderResponse with order ID
        """
        raise NotImplementedError

    @abstractmethod
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
        """Create a stop loss order.
        
        Args:
            trading_pair: The trading pair
            position_direction: Direction of the position to protect ('long' or 'short')
            size: Size in lots/contracts
            price: Stop loss trigger price
            leverage: Position leverage
            clientOid: Client order ID
            margin_mode: Margin mode for the order ("CROSS" or "ISOLATED")
            test: Whether this is a test order
            
        Returns:
            CreateOrderResponse with order ID
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Positions / exposure
    # ------------------------------------------------------------------

    @abstractmethod
    def fetch_positions(self) -> List[Position]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # New abstract methods
    # ------------------------------------------------------------------

    @abstractmethod
    def adjust_price(self, price: float, trading_pair: TradingPair) -> float:
        """Adjust price to conform to exchange tick size requirements."""
        raise NotImplementedError

    @abstractmethod
    def change_auto_deposit_status(self, trading_pair: TradingPair, status: bool) -> bool:
        """Change auto deposit status for a symbol."""
        raise NotImplementedError

    @abstractmethod
    def change_cross_leverage(self, trading_pair: TradingPair, leverage: float) -> bool:
        """Change cross margin leverage for a symbol."""
        raise NotImplementedError

    @abstractmethod
    def change_margin_mode(self, trading_pair: TradingPair, margin_mode: MarginMode) -> bool:
        """Change margin mode for a symbol."""
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, order_id: str, trading_pair: Optional[TradingPair] = None) -> List[str]:
        """Cancel an order by order ID. If trading_pair is provided, it will be used to resolve the symbol."""
        raise NotImplementedError

    @abstractmethod
    def close_position(self, trading_pair: TradingPair, margin_mode: MarginMode, test: bool = False, channel: str = 'MC', clientOid: Optional[ClientOid] = None) -> CreateOrderResponse:
        """Close a position."""
        raise NotImplementedError

    @abstractmethod
    def fetch_balance(self) -> Balance:
        """Fetch account balance."""
        raise NotImplementedError

    @abstractmethod
    def fetch_closed_orders(self, trading_pair: Optional[TradingPair] = None, since: Optional[TimestampMilliseconds] = None, limit: Optional[int] = None, side: Optional[OrderSide]=None) -> List[Order]:
        """Fetch closed orders."""
        raise NotImplementedError

    def fetch_closed_tpsl(self, trading_pair: Optional[TradingPair] = None, since: Optional[TimestampMilliseconds] = None, limit: Optional[int] = None, side: Optional[OrderSide] = None) -> List[Order]:
        raise NotImplementedError

    @abstractmethod
    def fetch_order_by_coid(self, coid: ClientOid) -> Order:
        """Fetch order by client order ID."""
        raise NotImplementedError

    @abstractmethod
    def fetch_order_by_id(self, order_id: str) -> Order:
        """Fetch order by order ID."""
        raise NotImplementedError

    @abstractmethod
    def fetch_order_by_symbol(self, trading_pair: TradingPair, since: Optional[TimestampMilliseconds] = None, until: Optional[TimestampMilliseconds] = None, limit: Optional[int] = None) -> List[Order]:
        """Fetch orders by symbol."""
        raise NotImplementedError

    @abstractmethod
    def fetch_orders_by_status(self, status: Status, trading_pair: Optional[TradingPair] = None, since: Optional[TimestampMilliseconds] = None, limit: Optional[int] = 1000) -> List[Order]:
        """Fetch orders by status."""
        raise NotImplementedError

    @abstractmethod
    def fetch_position(self, trading_pair: TradingPair, side: Optional[OrderSide] = None) -> Position:
        """Fetch position for a trading pair.
        
        Args:
            trading_pair: The trading pair to fetch position for
            side: Optional side filter for hedge mode exchanges ('buy' for long, 'sell' for short)
                  If None, returns any available position (for one-way mode exchanges)
        """
        raise NotImplementedError

    @abstractmethod
    def fetch_untriggered_stop_orders(self, trading_pair: Optional[TradingPair] = None, since: Optional[TimestampMilliseconds] = None, limit: Optional[int] = 1000) -> List[Order]:
        """Fetch untriggered stop orders."""
        raise NotImplementedError

    @abstractmethod
    def get_margin_mode(self, trading_pair: TradingPair) -> MarginMode:
        """Get margin mode for a symbol."""
        raise NotImplementedError

    @abstractmethod
    def allows_cross_mode(self, trading_pair: TradingPair) -> bool:
        """Return True if cross margin mode is allowed for the trading pair."""
        raise NotImplementedError

    @abstractmethod
    def get_recent_fills(self) -> List[Fill]:
        """Get recent fills."""
        raise NotImplementedError

    @abstractmethod
    def get_standardized_symbol(self, exchange: str) -> TradingPair:
        """Convert a Kucoin symbol to standardized trading pair format."""
        raise NotImplementedError

    @abstractmethod
    def get_symbol_id(self, standardized_trading_pair: TradingPair) -> str:
        """Convert a standardized trading pair to exchange symbol format."""
        raise NotImplementedError

    @property
    @abstractmethod
    def markets(self) -> Dict[TradingPair, FuturesMarket]:
        """Get market information."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Advanced Orders (TP/SL) - Exchange-specific TPSL order management
    # ------------------------------------------------------------------

    @abstractmethod
    def cancel_tpsl_order(self, order_id: str, trading_pair: TradingPair) -> List[str]:
        """Cancel a take profit or stop loss order by order ID.
        
        This method handles cancellation of advanced orders (TP/SL) which may
        require different endpoints than regular order cancellation on some exchanges.
        
        Args:
            order_id: The ID of the TP/SL order to cancel
            trading_pair: The trading pair (required for some exchanges)
            
        Returns:
            List of cancelled order IDs
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Misc (optionally override in child classes)
    # ------------------------------------------------------------------

    def extend(self, a: dict, b: dict) -> dict:
        result = a.copy()
        result.update(b)
        return result
    
    async def fetch_additional_info(self, trading_pair: TradingPair) -> tuple[str,dict]:
        self.load_markets()
        market = self.market(trading_pair)
        additional_data: dict[str, str | float | None] = {
            '24h High': market.daily_high,
            '24h Low': market.daily_low,
            '24h Volume': market.daily_volume,
            '24h Turnover': market.daily_turnover,
            '24h Change': market.daily_change,
            '24h Change Rate': str(market.daily_change_rate),
            'Open Interest': market.open_interest,
            'Fear and Greed Index': None,
            'Bitcoin Dominance': None,
            'Bitcoin Dominance Yesterday': None,
            'Bitcoin Dominance Change': None,
            'max_supply': None,
            'circulating_supply': None,
            'total_supply': None,
            'cmc_rank': None,
            'market_cap': None,
            'market_cap_dominance': None
        }
        additional_data_str = "\n".join([str(additional_data[key]) for key in additional_data if additional_data[key] is not None])
        try:
            after_march = datetime.now(timezone.utc) >= datetime(2025, 4, 1, tzinfo=timezone.utc)
            if not after_march:
                raise Exception("API tokens empty")
            cmc = CryptoMarketCap(validate_config.get_config())
            for coin in [trading_pair.coin_without_multiplier(), trading_pair.coin()]:
                try:
                    coin_metrics = cmc.get_cryptocurrency_quotes(coin)
                    break
                except Exception as _:
                    coin_metrics = None
            if not coin_metrics:
                raise ValueError(f"Coin metrics not found for {trading_pair.coin_without_multiplier()}")
            # Connect to the SQLite database
        
            # Get the latest record from coin_market_cap table
            row = await self.db_manager.fetch_one("""
                SELECT fear_and_greed, fear_and_greed_classification,
                        btc_dominance, btc_dominance_yesterday, btc_dominance_24h_percentage_change
                FROM coin_market_cap
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            if row:
                fear_and_greed = {'value': row[0], 'value_classification': row[1]}
                global_metrics = {
                    'btc_dominance': row[2],
                    'btc_dominance_yesterday': row[3],
                    'btc_dominance_24h_percentage_change': row[4]
                }
                
                additional_data_str += f"""\n\nFear and Greed Index (%): {fear_and_greed['value']} ({fear_and_greed['value_classification']})
Bitcoin Dominance: {global_metrics['btc_dominance']}%
Bitcoin Dominance Yesterday: {global_metrics['btc_dominance_yesterday']}%
Bitcoin Dominance Change: {global_metrics['btc_dominance_24h_percentage_change']}%

"""
                additional_data['Fear and Greed Index'] = fear_and_greed['value']
                additional_data['Bitcoin Dominance'] = global_metrics['btc_dominance']
                additional_data['Bitcoin Dominance Yesterday'] = global_metrics['btc_dominance_yesterday']
                additional_data['Bitcoin Dominance Change'] = global_metrics['btc_dominance_24h_percentage_change']
        
            additional_data_str += f"""
Max. supply: {coin_metrics['max_supply']}
Circulation supply: {coin_metrics['circulating_supply']}
Total supply: {coin_metrics['total_supply']}
CMC rank: {coin_metrics['cmc_rank']}
Market cap: {coin_metrics['quote']['USD']['market_cap']}
Market cap dominance: {coin_metrics['quote']['USD']['market_cap_dominance']}%
"""
            additional_data['max_supply'] = coin_metrics['max_supply']
            additional_data['circulating_supply'] = coin_metrics['circulating_supply']
            additional_data['total_supply'] = coin_metrics['total_supply']
            additional_data['cmc_rank'] = coin_metrics['cmc_rank']
            additional_data['market_cap'] = coin_metrics['quote']['USD']['market_cap']
            additional_data['market_cap_dominance'] = coin_metrics['quote']['USD']['market_cap_dominance']
        except Exception as e:
          print(f"Error fetching additional data: {str(e)}")
        return additional_data_str, additional_data

    def adjust_price_as_string(self, price: float, trading_pair: TradingPair) -> str:  # optional helper
        """Round *price* to the nearest valid tick‑size for *symbol*.

        Not every exchange needs a helper like this, but most do.  Leaving a
        *concrete* implementation to the child class is fine – the default just
        raises to make sure people don’t forget to override.
        """
        raise NotImplementedError
