import math
import time
from typing import List, Optional
from exchanges.kucoin.KucoinExchange import KucoinBaseExchange
from exchanges.kucoin.spot.types import AccountType, ApikeyInfo, HFOrder, HFOrderCancelByClientOidResponse, HFOrderCancelResponse, HFOrderRequest, HFOrderReturn, HFOrderTestReturn, Item, Side, SpotAccount, SpotAccountEnriched, SpotBalance, SpotCurrency, SpotSymbol, SpotTicker, SubAccountSummary, SymbolsWithOpenOrders, TradeHistoryParams, TradeHistoryReturn, TypeEnum
from exchanges.types.common import Coin, TradingPairSpot
from my_types.config_models import KucoinConfig
from utils.math import closest_integer_multiple


class KucoinSpotExchange(KucoinBaseExchange):
    def __init__(self, config: KucoinConfig):
        super().__init__(config, 'https://api.kucoin.com')
        self.currencies: List[SpotCurrency] = []
        self.symbols: List[SpotSymbol] = []
        self.tickers: List[SpotTicker] = []
        self._last_load_markets = 0
        self.load_markets()

    def _map_to_default_params(self, params: dict) -> dict:
        for key in ['price', 'stopPrice', 'triggerStopUpPrice', 'triggerStopDownPrice']:
            if key in params and params[key] and 'symbol' in params:
                if isinstance(params['symbol'], TradingPairSpot):
                    symbol = params['symbol']
                else:
                    symbol = TradingPairSpot(params['symbol'])
                params[key] = str(self._adjust_price(float(params[key]), symbol))
        filtered_params = {k: v for k, v in params.items() if v is not None}
        return filtered_params

    def _adjust_size(self, size: float, symbol: TradingPairSpot) -> float:
        spot_symbol = next((s for s in self.symbols if s.symbol == str(symbol)), None)
        if not spot_symbol:
            raise ValueError(f"Symbol {symbol} not found")
        tick = spot_symbol.baseIncrement
        tick = float(tick)
        sig_digits = -math.log10(tick)
        return round(closest_integer_multiple(size, tick), math.ceil(sig_digits))

    def _adjust_price(self, price: float, symbol: TradingPairSpot) -> float:
        spot_symbol = next((s for s in self.symbols if s.symbol == str(symbol)), None)
        if not spot_symbol:
            raise ValueError(f"Symbol {symbol} not found")
        tick = spot_symbol.priceIncrement
        tick = float(tick)
        sig_digits = -math.log10(tick)
        return round(closest_integer_multiple(price, tick), math.ceil(sig_digits))

    def load_markets(self) -> None:
        if time.time() - self._last_load_markets < 60 * 60:
            return

        self.currencies = self._fetch_currencies()        
        self.symbols = self._fetch_symbols()
        self.tickers = self._fetch_tickers()
        self.market: dict[TradingPairSpot,SpotTicker] = {
            TradingPairSpot(ticker.symbol): ticker
            for ticker in self.tickers
        }
        self._last_load_markets = time.time()
        
    def _fetch_currencies(self) -> List[SpotCurrency]:
        path = '/api/v3/currencies'
        response = self._request(path, 'public', 'GET')
        currencies_data = response.get('data', [])
        return [SpotCurrency(**currency) for currency in currencies_data]
    
    def _fetch_symbols(self) -> List[SpotSymbol]:
        path = '/api/v2/symbols'
        response = self._request(path, 'public', 'GET')
        symbols_data = response.get('data', [])
        return [SpotSymbol(**symbol) for symbol in symbols_data]
    
    def _fetch_tickers(self) -> List[SpotTicker]:
        path = '/api/v1/market/allTickers'
        response = self._request(path, 'public', 'GET')
        tickers_data = response.get('data', []).get('ticker', [])
        return [SpotTicker(**ticker) for ticker in tickers_data]

    def _fetch_api_key_info(self) -> ApikeyInfo:
        path = '/api/v1/user/api-key'
        response = self._request(path, 'private', 'GET')
        data = response.get('data', {})
        return ApikeyInfo(**data)
    
    def _fetch_subaccount_summary_list(self) -> SubAccountSummary:
        path = '/api/v2/sub/user'
        response = self._request(path, 'private', 'GET')
        data = response.get('data', {})
        return SubAccountSummary(**data)

    def _fetch_balances(self, coin: Optional[Coin] = None, acc_type: Optional[AccountType] = AccountType.TRADE) -> List[SpotAccount]:
        filtering = ''
        if coin or acc_type:
            filtering = '?'
            if coin:
                filtering += f'currency={coin}'
            if acc_type:
                if coin:
                    filtering += '&'
                filtering += f'type={acc_type.value}'
        path = f'/api/v1/accounts{filtering}'
        response = self._request(path, 'private', 'GET')
        accounts_data = response.get('data', [])
        return [SpotAccount(**account) for account in accounts_data]
    
    def _fetch_account_by_id(self, id: str) -> List[SpotBalance]:
        path = f'/api/v1/accounts/{id}'
        response = self._request(path, 'private', 'GET')
        balances_data = response.get('data', [])
        return [SpotBalance(**balance) for balance in balances_data]
    
    def _fetch_enriched_balances(self, trading_pair: Optional[TradingPairSpot] = None, acc_type: Optional[AccountType] = AccountType.TRADE) -> List[SpotAccountEnriched]:
        balances = self._fetch_balances(trading_pair.coin() if trading_pair else None, acc_type)
        self.tickers = self._fetch_tickers()
        enriched_balances: List[SpotAccountEnriched] = []
        for balance in balances:
            if not trading_pair or trading_pair.coin() != balance.currency:
                trading_pair = TradingPairSpot(f"{balance.currency}-USDT")
            if trading_pair.coin() == trading_pair.quote():
                value = float(balance.balance)
            else:
                ticker = next((t for t in self.tickers if t.symbol == str(trading_pair)), None)
                if not ticker:
                    raise ValueError(f"Ticker not found for symbol {trading_pair}")
                value = float(balance.balance) * float(ticker.last)
            enriched_balance = SpotAccountEnriched(
                **balance.__dict__,  # Pass all fields from base SpotAccount
                ticker=ticker,
                value=value
            )
            enriched_balances.append(enriched_balance)
        return enriched_balances

    def _create_hf_order_sync(self, params: HFOrderRequest) -> HFOrderReturn | HFOrderTestReturn:
        """Create a high-frequency order using the sync endpoint."""
        if params.type not in [TypeEnum.LIMIT, TypeEnum.MARKET]:
            raise ValueError("type must be either 'limit' or 'market'")
        
        if not params.symbol:
            raise ValueError("symbol is required")
            
        if params.side not in [Side.BUY, Side.SELL]:
            raise ValueError("side must be either 'buy' or 'sell'")
        
        if params.type == TypeEnum.LIMIT and not params.price:
            raise ValueError("price is required for limit orders")
            
        if params.type == TypeEnum.MARKET and not (params.size or params.funds):
            raise ValueError("either size or funds is required for market orders")
            
        if params.remark and len(params.remark) > 20:
            raise ValueError("remark cannot exceed 20 characters")
            
        if params.tags and len(params.tags) > 20:
            raise ValueError("tags cannot exceed 20 characters")
            
        if params.clientOid and len(params.clientOid) > 40:
            raise ValueError("clientOid cannot exceed 40 characters")
    
        path = '/api/v1/hf/orders/sync'
        response = self._request(path, 'private', 'POST', params.__dict__)
        if 'test' in path:
            return HFOrderTestReturn(**response.get('data', {}))
        else:
            return HFOrderReturn(**response.get('data', {}))

    def _cancel_hf_order_sync(self, order_id: str) -> HFOrderCancelResponse:
        """Cancel a high-frequency order by order ID."""
        path = f'/api/v1/hf/orders/sync/{order_id}'
        response = self._request(path, 'private', 'DELETE')
        return HFOrderCancelResponse(**response.get('data', {}))

    def _cancel_hf_order_by_client_oid_sync(self, client_oid: str) -> HFOrderCancelByClientOidResponse:
        """Cancel a high-frequency order by client order ID."""
        path = f'/api/v1/hf/orders/sync/client-order/{client_oid}'
        response = self._request(path, 'private', 'DELETE')
        return HFOrderCancelByClientOidResponse(**response.get('data', {}))

    def _fetch_hf_order(self, order_id: str, trading_pair: TradingPairSpot) -> HFOrder:
        """Fetch a high-frequency order by order ID."""
        path = f'/api/v1/hf/orders/{order_id}'
        params = {'symbol': str(trading_pair)}
        response = self._request(path, 'private', 'GET', params)
        return HFOrder(**response.get('data', {}))
    
    def _fetch_hf_order_by_client_oid(self, client_oid: str, trading_pair: TradingPairSpot) -> HFOrder:
        """Fetch a high-frequency order by client order ID."""
        path = f'/api/v1/hf/orders/client-order/{client_oid}'
        params = {'symbol': str(trading_pair)}
        response = self._request(path, 'private', 'GET', params)
        return HFOrder(**response.get('data', {}))

    def _fetch_hf_active_symbols(self) -> SymbolsWithOpenOrders:
        """Fetch all trading pairs that have pending high-frequency orders."""
        path = '/api/v1/hf/orders/active/symbols'
        response = self._request(path, 'private', 'GET')
        return SymbolsWithOpenOrders(**response.get('data', {'symbols': []}))

    def _fetch_hf_trade_history(self, params: TradeHistoryParams) -> TradeHistoryReturn:
        """Fetch high-frequency trading history for a symbol."""
        path = '/api/v1/hf/fills'
        response = self._request(path, 'private', 'GET', params.__dict__)
        data = response.get('data', {'items': [], 'lastId': 0})
        # Map each item in the list to an Item object
        items = [Item(**item) for item in data.get('items', [])]
        return TradeHistoryReturn(items=items, lastId=data.get('lastId', 0))
