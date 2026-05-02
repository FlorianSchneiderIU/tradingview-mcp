import math
import time
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

import pandas as pd
from exchanges.IFuturesExchange import IFuturesExchange
from exchanges.kucoin.KucoinExchange import KucoinBaseExchange
import exchanges.kucoin.futures.types as kft
import exchanges.types.common as common_types
from exchanges.types.common import ClientOid, ClientOrderGoal, ClientOrderType, Direction, Order, OrderSide, Status, TimeFrame, TimestampMilliseconds, TimestampNanoseconds, TradingPair
from exchanges.types.exceptions import LeverageError, PositionNotFoundError, SymbolNotFoundError, MarginModeMismatchError
from my_types.config_models import KucoinConfig
from my_types.percentage import Percentage
from utils.SqlManager import SQLiteManager
from utils.math import closest_integer_multiple


class KucoinFuturesExchange(KucoinBaseExchange, IFuturesExchange):
    def get_name(self) -> str:
        """Return the name of the exchange."""
        return "Kucoin Futures"

    def __init__(self, config: KucoinConfig, db_manager: SQLiteManager):
        super().__init__(config, 'https://api-futures.kucoin.com')
        self.db_manager = db_manager
        self._markets: dict[TradingPair, kft.FuturesMarket] = {}
        self._markets_common: dict[TradingPair, common_types.FuturesMarket] = {}
        self._last_load_markets = 0
        self.load_markets()
    
    def _process_market_data(self, item: kft.FuturesMarket) -> TradingPair:
        """Process a single market item and return the standardized trading pair."""
        base_currency = item['baseCurrency']
        quote_currency = item['quoteCurrency']
        settle_currency = item['settleCurrency']

        # Map common currencies to standardized formats
        base_currency_mapped = self.commonCurrencies.get(base_currency, base_currency)
        quote_currency_mapped = self.commonCurrencies.get(quote_currency, quote_currency)
        settle_currency_mapped = self.commonCurrencies.get(settle_currency, settle_currency)

        # Create a standardized symbol like BTC/USDT:USDT
        return TradingPair(f"{base_currency_mapped}/{quote_currency_mapped}:{settle_currency_mapped}")

    def _fetch_contract(self, symbol: kft.KucoinFuturesContract) -> kft.FuturesMarket:
        """Fetch contract data for a specific symbol."""
        path = f'/api/v1/contracts/{symbol}'
        response = self.request(path, 'public', 'GET')
        if response['code'] != '200000':
            raise Exception(f"Error fetching contract: {response.get('msg', 'Unknown error')}")
        return response['data']

    def _load_markets(self) -> None:
        if self._markets and time.time() - self._last_load_markets < 60 * 60:
            return
        self.commonCurrencies = {
            'HOT': 'HOTNOW',
            'EDGE': 'DADI',
            'WAX': 'WAXP',
            'TRY': 'Trias',
            'VAI': 'VAIOT',
            'XBT': 'BTC',
            'NEIROCTO': 'NEIRO',
            'NEIRO': 'NEIROETH',
        }
        
        # Load market data from KuCoin Futures API
        path = '/api/v1/contracts/active'
        response = self.request(path, 'public', 'GET')
        data: List[kft.FuturesMarket] = response.get('data', [])
        for item in data:
            standardized_symbol = self._process_market_data(item)
            # Store the market information with the standardized symbol
            self._markets[standardized_symbol] = item
        self._last_load_markets = time.time()

    def _fetch_order_book(self, symbol: TradingPair | kft.KucoinFuturesContract, depth: int | Literal['full'] = 20) -> kft.OrderBookData:
        if depth not in [20, 100, 'full']:
            raise ValueError("depth must be either 20, 100 or full")
        path = f'/api/v1/level2/depth{depth}'
        if depth == 'full':
            path = '/api/v1/level2/snapshot'
        params = {'symbol': symbol}
        response = self.request(path=path, api='public', method='GET', params=params)
        if response['code'] != '200000':
            raise Exception(f"Error fetching order book: {response.get('msg', 'Unknown error')}")
        return response['data']

    def _get_standardized_symbol(self, kucoin_symbol: kft.KucoinFuturesContract) -> TradingPair:
        # Map KuCoin symbols to standard format if needed
        for key, market in self._markets.items():
            if market['symbol'] == kucoin_symbol:
                return key
        raise ValueError(f"KuCoin symbol {kucoin_symbol} not found in markets.")

    def _get_symbol_id(self, standardized_symbol:TradingPair) -> kft.KucoinFuturesContract:
        """
        Inverse maps a standardized symbol like BTC/USDT:USDT to the original KuCoin symbol.
        """
        if standardized_symbol in self._markets:
            # Directly return the original symbol stored in self._markets
            return self._markets[standardized_symbol]['symbol']
        else:
            raise SymbolNotFoundError(f"Standardized symbol {standardized_symbol} not found in markets.")

    def _get_full_path(self, path, method):
        # Map custom paths to actual endpoints
        path_mapping = {
            'position/getMarginMode': '/api/v1/position/marginMode',
            'position/changeMarginMode': '/api/v1/position/change-margin-mode',
            'changeCrossUserLeverage': '/api/v1/position/leverage',
            # Add other mappings as necessary
        }
        if path in path_mapping:
            return path_mapping[path]
        return path  # If no mapping, return as is

    def _adjust_price(self, price: float, symbol: TradingPair):
        tick = self._markets[symbol]['tickSize']
        sig_digits = -math.log10(tick)
        return round(closest_integer_multiple(price, tick), math.ceil(sig_digits))
    
    def adjust_price_as_string(self, price: float, symbol: TradingPair) -> str:
        tick = self._markets[symbol]['tickSize']
        sig_digits = -math.log10(tick)
        adjusted_price = self.adjust_price(price, symbol)
        return f"{adjusted_price:.{math.ceil(sig_digits)}f}".rstrip('0').rstrip('.')  # Format to the correct number of decimal places
        
    def _map_to_default_params(self, params: dict) -> dict:
        if 'symbol' in params:
            if isinstance(params['symbol'], TradingPair):
                params['symbol'] = self.get_symbol_id(params['symbol'])
            else:
                params['symbol'] = str(params['symbol'])
        for key in ['price', 'stopPrice', 'triggerStopUpPrice', 'triggerStopDownPrice']:
            if key in params and params[key] and 'symbol' in params:
                if isinstance(params['symbol'], TradingPair):
                    symbol = params['symbol']
                else:
                    symbol = self.get_standardized_symbol(params['symbol'])
                params[key] = self.adjust_price_as_string(float(params[key]), symbol)
        filtered_params = {k: v for k, v in params.items() if v is not None}
        return filtered_params

    def _fetch_ohlcv(self, trading_pair: TradingPair | kft.KucoinFuturesContract, timeframe: TimeFrame, since: Optional[TimestampMilliseconds]=None, until: Optional[TimestampMilliseconds]=None) -> pd.DataFrame:
        path = '/api/v1/kline/query'
        api = 'public'
        
        method = 'GET'
        params = {
            'symbol': trading_pair,
            'granularity': timeframe.value.minutes,
        }
        if since:
            params['from'] = since
        if until:
            params['to'] = until
        response = self.request(path, api, method, params)
        data = response.get('data', [])

        # Note: we intentionally raise on unexpected row shapes so callers can debug API changes.

        # Validate rows: KuCoin now returns exactly 7 columns per row: 
        # [timestamp, open, high, low, close, volume(lots), volume_value]
        # We map volume(lots) to 'volume' for backward compatibility
        # If we receive any row with a different number of columns, raise a ValueError showing
        # the exact offending row so the caller can debug the API response.
        validated: list[list[float]] = []
        for i, row in enumerate(data):
            if isinstance(row, (list, tuple)):
                if len(row) != 7:
                    # Fail loudly: show the row index and content to help debugging
                    raise ValueError(f"Unexpected OHLCV row length {len(row)} at index {i}; expected 7 columns [timestamp, open, high, low, close, volume(lots), volume_value]; row={row}")
                validated.append(list(row))
            elif isinstance(row, dict):
                # Convert dict rows into the expected list order, then validate
                ts = row.get('timestamp') or row.get('time') or row.get('ts')
                o = row.get('open')
                h = row.get('high')
                low = row.get('low')
                c = row.get('close')
                v_lots = row.get('volume_lots') or row.get('volume') or row.get('vol')
                v_value = row.get('volume_value') or row.get('turnover')
                if None in (ts, o, h, low, c, v_lots, v_value):
                    raise ValueError(f"Unexpected OHLCV dict row missing fields at index {i}; expected [timestamp, open, high, low, close, volume(lots), volume_value]; row={row}")
                validated.append([ts, o, h, low, c, v_lots, v_value])
            else:
                # Unknown type -- fail loudly
                raise ValueError(f"Unexpected OHLCV row type {type(row)} at index {i}; row={row}")

        # Build DataFrame from validated rows
        if len(validated) == 0:
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        df = pd.DataFrame(validated, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'volume_value'])
        # KuCoin returns timestamps in milliseconds
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        
        # Return only the standard 6 columns to maintain interface compatibility
        # The extra volume_value information is processed internally but not exposed
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
      
    def fetch_closed_tpsl(
        self,
        trading_pair: Optional[TradingPair] = None,
        since: Optional[TimestampMilliseconds] = None,
        limit: Optional[int] = None,
        side: Optional[OrderSide] = None,
    ) -> list[Order]:
        """Fetch all closed TP/SL orders."""
        params: kft.OrderRequestParams = {
            "side": side,
            "type": "market_stop"
        } 
        orders = self._fetch_closed_orders(symbol=trading_pair, since=since, limit=limit, params=params)
        return [self._map_raw_order_to_common(order) for order in orders]

    def _fetch_my_trades(self, since=None) -> List[kft.Fill]:
        path = '/api/v1/fills'
        api = 'private'
        
        method = 'GET'
        params = {}
        if since:
            params['startAt'] = since
        response = self.request(path, api, method, params)
        return response.get('data', {}).get('items', [])

    def _fetch_positions(self) -> List[kft.Position]:
        path = '/api/v1/positions'
        api = 'private'
        
        method = 'GET'
        response = self.request(path, api, method)
        return response.get('data', [])

    def _fetch_positions_history(self, since: Optional[int]=None, trading_pair: Optional[TradingPair]=None) -> List[kft.HistoricPosition]:
        path = '/api/v1/history-positions'
        api = 'private'
        
        method = 'GET'
        params: dict[str, Any] = {
            'limit': 200
        }
        if since:
            params['from'] = since
        if trading_pair:
            params['symbol'] = self._get_symbol_id(trading_pair)
        response = self.request(path, api, method, params)
        return response.get('data', {}).get('items', [])

    def _fetch_position(self, trading_pair: TradingPair | kft.KucoinFuturesContract) -> kft.Position:
        path = '/api/v1/position'
        api = 'private'
        
        method = 'GET'
        params = {'symbol': trading_pair}
        response = self.request(path, api, method, params)
        return response.get('data', {})

    def _close_position(self, trading_pair: TradingPair | kft.KucoinFuturesContract, margin_mode: kft.MarginMode, test: bool = False, channel: str = 'MC', clientOid: Optional[ClientOid] = None) -> kft.CreateOrderResponse:
        # Create the CloseOrderRequest instance
        close_order_request = kft.CloseOrderRequest({
            'symbol': trading_pair,
            'closeOrder': True,
            'type': 'market',
            'marginMode': margin_mode,
            'clientOid': None,  # set later
            'remark': "",  # Optional field
            'reduceOnly': True,
            'stop': None,
            'stopPriceType': None,
            'stopPrice': None
        })
        if clientOid is None:
            clientOid = self._create_client_oid(channel)
        
        # Send the request to close the position
        return self._ct_create_order(close_order_request, clientOid, test=test)

    def _fetch_ticker(self, trading_pair: TradingPair | kft.KucoinFuturesContract) -> kft.Ticker:
        path = '/api/v1/ticker'
        api = 'public'
        
        method = 'GET'
        params = {'symbol': trading_pair}
        response = self.request(path, api, method, params)
        return response.get('data', {})

    def _fetch_orders_by_status(self, status: Status, symbol:Optional[TradingPair | kft.KucoinFuturesContract]=None, since:Optional[TimestampMilliseconds]=None, limit:Optional[int]=None, params: kft.OrderRequestParams={}) -> List[kft.Order]:
        path = '/api/v1/orders'
        api = 'private'
        
        status = "active" if status == "open" else status
        
        method = 'GET'
        paramsD = self.extend(dict(params), {'status': status})
        if symbol:
            paramsD['symbol'] = symbol
        if since:
            paramsD['startAt'] = since
        if limit:
            paramsD['pageSize'] = limit
        response = self.request(path, api, method, paramsD)
        return response.get('data', {}).get('items', [])
    
    def _fetch_untriggered_stop_orders(self, symbol:Optional[TradingPair | kft.KucoinFuturesContract]=None, since:Optional[TimestampMilliseconds]=None, limit:Optional[int]=1000, params: kft.OrderRequestParams={}) -> List[kft.Order]:
        path = '/api/v1/stopOrders'
        api = 'private'
        
        method = 'GET'
        if symbol:
            params['symbol'] = symbol
        if since:
            params['startAt'] = since
        if limit:
            params['pageSize'] = limit
        response = self.request(path, api, method, dict(params))
        return response.get('data', {}).get('items', [])
    

    def _fetch_balance(self) -> kft.Balance:
        path = '/api/v1/account-overview'
        api = 'private'
        
        method = 'GET'
        params = {'currency': 'USDT'}
        response = self.request(path, api, method, params)
        return response.get('data', {})

    def _create_client_oid(self, channel_abbreviation: str) -> ClientOid:
        uid = str(uuid4())
        uid = uid[:20]
        return ClientOid(channel_abbreviation, ClientOrderType.MARKET, ClientOrderGoal.MANUAL_CLOSE, uid)
    
    def _get_channel_abbreviation(self, client_oid: str):
        if '_' not in client_oid:
            return None
        return client_oid.split('_')[0]

    def _market(self, trading_pair: TradingPair | kft.KucoinFuturesContract) -> kft.FuturesMarket:
        if not isinstance(trading_pair, TradingPair):
            trading_pair = self.get_standardized_symbol(trading_pair)
        if trading_pair in self._markets:
            return self._markets[trading_pair]
        raise ValueError(f"Market {trading_pair} not found in markets.")
    
    def _create_limit_order(self, params: kft.LimitOrderRequest, clientOid: ClientOid, test: bool = False):
        # 'clientOid': None,
        #     'stop': None, 
        #     'stopPriceType': None,
        #     'stopPrice': None,
        #     'remark': None,  
        #     'forceHold': False,
        #     'stp': None,
        #     'timeInForce': None,
        #     'postOnly': False,
        #     'hidden': False,
        #     'iceberg': False,
        #     'visibleSize': None,
        return self._ct_create_order(params, clientOid, test=test)
    
    def _create_market_order(self, params: kft.MarketOrderRequest, clientOid: ClientOid, test: bool = False):
        print("DEBUG: Creating market order normal futures with params:", params)
        return self._ct_create_order(params, clientOid, test=test)

    def _create_close_order(self, params: kft.CloseOrderRequest, clientOid: ClientOid, test: bool = False):
        return self._ct_create_order(params, clientOid, test=test)

    def _create_tp_order(self, params: kft.TpOrderRequest, clientOid: ClientOid, test: bool = False):
        if params['side'] not in ['buy', 'sell']:
            raise ValueError(f"Invalid side: {params['side']}")
        
        tp_param: kft.LimitStOrderRequest = {
            'clientOid': clientOid.__str__(),
            'side': params['side'],
            'symbol': params['symbol'],
            'leverage': str(params['leverage']),
            'price': params['triggerStopUpPrice'],
            'size': params['size'],
            'triggerStopUpPrice': params['triggerStopUpPrice'],
            'marginMode': params['marginMode'],
            'type': 'limit',
            'stopPriceType': 'TP',
            'reduceOnly': True,
            'closeOrder': False,
            'forceHold': False,
            'timeInForce': 'GTC',
            'postOnly': False,
            'hidden': False,
            'iceberg': False,
            'visibleSize': None,
            'remark': None,
            'stp': None,
            'triggerStopDownPrice': None
        } 
        return self._create_st_order(tp_param, test=test), tp_param

    def _create_sl_order(self, params: kft.SlOrderRequest, clientOid: ClientOid, test: bool = False):
        if params['side'] not in ['buy', 'sell']:
            raise ValueError(f"Invalid side: {params['side']}")
        
        sl_param: kft.MarketStOrderRequest = {
            'clientOid': clientOid.__str__(),
            'side': params['side'],
            'symbol': params['symbol'],
            'leverage': str(params['leverage']),
            'size': params['size'],
            'triggerStopDownPrice': params['triggerStopDownPrice'],
            'marginMode': params['marginMode'],
            'type': 'market',
            'stopPriceType': 'TP',
            'reduceOnly': True,
            'closeOrder': False,
            'forceHold': False,
            'remark': None,
            'stp': None,
            'triggerStopUpPrice': None
        } 
        return self._create_st_order(sl_param, test=test), sl_param

    def _create_st_order(self, params: kft.StOrderRequest, test: bool = False):
        if test:
            return kft.CreateOrderResponse({'orderId': 'test'})

        path = '/api/v1/st-orders'
        api = 'private'
        
        method = 'POST'

        try:
            response = self.request(path, api, method, dict(params))
        except Exception as e:
            if e.args and isinstance(e.args[0], str) and "margin mode does not match" in e.args[0]:
                raise MarginModeMismatchError(e.args[0]) from e
            raise

        data = response.get('data', {})
        orderId = str(data.get('orderId', ''))
        return kft.CreateOrderResponse({'orderId': orderId})

    def _ct_create_order(self, params: kft.LimitOrderRequest | kft.MarketOrderRequest | kft.CloseOrderRequest, clientOid: ClientOid, test: bool = False) -> kft.CreateOrderResponse:
        path = '/api/v1/orders'
        if test:
            path = '/api/v1/orders/test'
        api = 'private'
        
        method = 'POST'
        params['clientOid'] = clientOid.__str__()
        
        try:
            response = self.request(path, api, method, dict(params))
        except Exception as e:
            if e.args and isinstance(e.args[0], str) and "margin mode does not match" in e.args[0]:
                raise MarginModeMismatchError(e.args[0]) from e
            raise
        data = response.get('data', {})
        orderId = str(data.get('orderId', ''))
        return kft.CreateOrderResponse({'orderId': orderId})

    def _fetch_closed_orders(self, symbol: Optional[TradingPair | kft.KucoinFuturesContract]=None, since:Optional[TimestampMilliseconds]=None, limit:Optional[int]=None, params:Optional[kft.OrderRequestParams]={}) -> List[kft.Order]:
        path = '/api/v1/orders'
        api = 'private'
        
        method = 'GET'
        paramsD = self.extend(dict(params or {}), {'status': 'done'})
        if symbol:
            paramsD['symbol'] = symbol
        if since:
            paramsD['startAt'] = since
        if limit:
            paramsD['pageSize'] = limit
        response = self.request(path, api, method, paramsD)
        return response.get('data', {}).get('items', [])

    def _fetch_order_by_coid(self, coid:str | ClientOid) -> kft.Order:
        path = '/api/v1/orders/byClientOid'
        api = 'private'
        
        method = 'GET'
        params = {'clientOid': str(coid)}
        response = self.request(path, api, method, params)
        return response.get('data', {})

    def _fetch_order_by_symbol(self, symbol: kft.KucoinFuturesContract | TradingPair, since: Optional[int]=None, until: Optional[int]=None, limit:Optional[int]=None) -> List[kft.Order]:
        path = '/api/v1/orders'
        api = 'private'
        
        method = 'GET'
        params: dict[str, Any] = {'symbol': symbol}
        if since:
            params['startAt'] = since
        if until:
            params['endAt'] = until
        if limit:
            params['pageSize'] = limit
        response = self.request(path, api, method, params)
        return response.get('data', {}).get('items', [])

    def _fetch_order_by_id(self, id: str | int) -> kft.Order:
        path = f'/api/v1/orders/{id}'
        api = 'private'
        
        method = 'GET'
        response = self.request(path, api, method, {})
        return response.get('data', {})

    def _cancel_order(self, order_id: str) -> List[str]:
        path = f'/api/v1/orders/{order_id}'
        api = 'private'
        
        method = 'DELETE'
        response = self.request(path, api, method)
        return response.get('data', {}).get('cancelledOrderIds', [])

    def _change_auto_deposit_status(self, symbol: kft.KucoinFuturesContract, status: bool) -> bool:
        path = '/api/v1/position/margin/auto-deposit-status'
        api = 'private'
        
        method = 'POST'
        params = {'symbol': symbol, 'status': status}
        try:
            response = self.request(path, api, method, params)
        except Exception as e:
            if e.args and isinstance(e.args[0], str) and "margin mode does not match" in e.args[0]:
                raise MarginModeMismatchError(e.args[0]) from e
            raise
        return response.get('data', False)

    def _get_recent_fills(self) -> List[kft.Fill]:
        path = '/api/v1/recentFills'
        api = 'private'
        
        method = 'GET'
        response = self.request(path, api, method)
        return response.get('data', {})

    def _get_margin_mode(self, symbol: kft.KucoinFuturesContract) -> kft.MarginMode:
        path = '/api/v2/position/getMarginMode'
        api = 'private'
        
        method = 'GET'
        params = {'symbol': symbol}
        response = self.request(path, api, method, params)
        data = response.get('data', {}).get('marginMode','')
        if data in ['ISOLATED', 'CROSS']:
            return data # type: ignore
        raise ValueError(f"Invalid margin mode: {data}")

    def _change_margin_mode(self, symbol: kft.KucoinFuturesContract, marginMode: kft.MarginMode):
        path = '/api/v2/position/changeMarginMode'
        api = 'private'
        
        method = 'POST'
        params = {'symbol': symbol, 'marginMode': marginMode}
        response = self.request(path, api, method, params)
        data = response.get('data', False)
        return True if data else False

    def _change_cross_leverage(self, symbol: kft.KucoinFuturesContract, leverage: float):
        path = '/api/v2/changeCrossUserLeverage'
        api = 'private'
        
        method = 'POST'
        params = {'symbol': symbol, 'leverage': str(leverage)}
        response = self.request(path, api, method, params)
        data = response.get('data', False)
        if not data:
            raise LeverageError(response.get('msg', 'Could not change leverage'))
        return True

    def _map_raw_order_to_common(self, raw_order: kft.Order) -> common_types.Order:
        """Convert a raw KuCoin order to common types Order."""
        # Convert KuCoin symbol to standardized trading pair
        trading_pair = self._get_standardized_symbol(raw_order['symbol'])
        
        return common_types.Order(
            id=raw_order['id'],
            trading_pair=trading_pair,
            type=raw_order['type'],
            side=raw_order.get('side'),
            price=raw_order['price'],
            amountLots=raw_order['size'],
            value=raw_order['value'],
            dealValue=raw_order['dealValue'],
            dealSize=raw_order['dealSize'],
            stp=raw_order['stp'],
            stop=raw_order['stop'],
            stopPriceType=raw_order['stopPriceType'],
            stopTriggered=raw_order['stopTriggered'],
            stopPrice=raw_order['stopPrice'],
            timeInForce=raw_order['timeInForce'],
            postOnly=raw_order['postOnly'],
            hidden=raw_order['hidden'],
            iceberg=raw_order['iceberg'],
            leverage=raw_order['leverage'],
            forceHold=raw_order['forceHold'],
            closeOrder=raw_order['closeOrder'],
            visibleSize=raw_order['visibleSize'],
            clientOid=ClientOid.from_string(raw_order['clientOid']) if raw_order.get('clientOid') and ClientOid.is_valid_string(raw_order['clientOid']) else None,
            remark=raw_order.get('remark'),
            tags=raw_order['tags'],
            isActive=raw_order['isActive'],
            cancelExist=raw_order['cancelExist'],
            createdAt=common_types.TimestampMilliseconds(raw_order['createdAt']),
            updatedAt=TimestampMilliseconds(raw_order['updatedAt']),
            endAt=TimestampMilliseconds(raw_order['endAt']) if raw_order['endAt'] else None,
            orderTime=raw_order['orderTime'],
            settleCurrency=raw_order['settleCurrency'],
            marginMode=raw_order['marginMode'],
            avgDealPrice=raw_order['avgDealPrice'],
            filledLots=raw_order['filledSize'],
            filledValue=raw_order['filledValue'],
            status=raw_order['status'],
            reduceOnly=raw_order['reduceOnly']
        )

    def _convert_market_to_common(self, raw_market: kft.FuturesMarket) -> common_types.FuturesMarket:
        """Map a raw KuCoin market to the common FuturesMarket dataclass."""
        return common_types.FuturesMarket(
            markPrice=raw_market['markPrice'],
            maxLeverage=raw_market['maxLeverage'],
            lot_size=float(raw_market['multiplier']),
            daily_high=raw_market['highPrice'],
            daily_low=raw_market['lowPrice'],
            daily_volume=raw_market['volumeOf24h'],
            daily_turnover=raw_market['turnoverOf24h'],
            daily_change=raw_market['priceChg'],
            daily_change_rate=Percentage(raw_market['priceChgPct']),
            open_interest=float(raw_market['openInterest']),
            takerFeeRate=raw_market['takerFeeRate'],
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load_markets(self) -> None:
        """Initialise or refresh the *in‑memory* market cache."""
        self._load_markets()
        # Convert the raw KuCoin futures markets to the common FuturesMarket
        self._markets_common = {
            tp: self._convert_market_to_common(mkt)
            for tp, mkt in self._markets.items()
        }

    def market(self, trading_pair: TradingPair) -> common_types.FuturesMarket:
        """Return the raw market‑definition for *trading_pair*."""
        if not self._markets or trading_pair not in self._markets:
            self.load_markets()
        raw = self._market(trading_pair)
        return self._convert_market_to_common(raw)

    def fetch_order_book(
        self,
        trading_pair: TradingPair,
        depth: int | Literal["full"] = 20,
    ) -> common_types.OrderBookData:
        """Fetch the order book for a trading pair."""
        # Call the private method to fetch the raw order book data
        raw_order_book = self._fetch_order_book(trading_pair, depth)
        
        # Map the raw response to the common type
        return common_types.OrderBookData(
            trading_pair=trading_pair,
            sequence=raw_order_book['sequence'],
            asks=raw_order_book['asks'],
            bids=raw_order_book['bids'],            ts=common_types.TimestampNanoseconds(raw_order_book['ts'])
        )

    def fetch_ticker(self, trading_pair: TradingPair) -> common_types.Ticker:
        """Fetch ticker data for a trading pair."""
        # Convert the standardized trading pair to the exchange's symbol format
        exchange_symbol = self._get_symbol_id(trading_pair)
        
        # Call the private method to fetch the raw ticker data
        raw_ticker = self._fetch_ticker(exchange_symbol)
        
        # Get market data to access mark price and index price
        market_data = self._fetch_contract(exchange_symbol)
        
        # Map the raw response to the common type
        # Use ticker's price field as last_price, and market data for mark_price and index_price
        last_price = float(raw_ticker['price'])
        mark_price = market_data['markPrice']
        index_price = market_data['indexPrice']
        
        return common_types.Ticker(
            last_price=last_price,
            mark_price=mark_price,
            index_price=index_price
        )

    def fetch_ohlcv(
        self,
        trading_pair: TradingPair,
        timeframe: TimeFrame,
        since: Optional[TimestampMilliseconds] = None,
        until: Optional[TimestampMilliseconds] = None,
    ):  # return type intentionally left open (e.g. pandas.DataFrame)
        """Fetch OHLCV (candlestick) data for a trading pair."""
        # Convert the standardized trading pair to the exchange's symbol format
        exchange_symbol = self._get_symbol_id(trading_pair)
        
        # Call the private method to fetch the raw OHLCV data
        return self._fetch_ohlcv(exchange_symbol, timeframe, since, until)

    def create_limit_order(
        self,
        params: common_types.LimitOrderRequest,
        test: bool = False,
    ) -> common_types.CreateOrderResponse:
        """Create a limit order using public interface types.
        
        Args:
            params: Common types limit order request
            test: Whether to use test endpoint
            
        Returns:
            CreateOrderResponse with orderId
        """
        # Convert TradingPair to exchange symbol
        exchange_symbol = self._get_symbol_id(params.trading_pair)
          # Map common_types.LimitOrderRequest to kft.LimitOrderRequest
        kft_params: kft.LimitOrderRequest = {
            'clientOid': str(params.clientOid),
            'symbol': exchange_symbol,
            'side': params.side,
            'type': 'limit',
            'leverage': str(params.leverage),
            'size': params.size,
            'price': str(params.price),
            'marginMode': params.marginMode,
            'reduceOnly': params.reduceOnly,
            'postOnly': False,  # Default value
            'hidden': False,   # Default value  
            'iceberg': False,  # Default value
            'timeInForce': 'GTC',  # Default value
            'forceHold': False,  # Default value
            'stop': None,      # Not used for basic limit orders
            'stopPrice': None, # Not used for basic limit orders
            'stopPriceType': None,  # Not used for basic limit orders
            'stp': None,       # Default value
            'remark': None,    # Default value
            'visibleSize': None  # Default value
        }
          # Call the private method with the ClientOid from params
        kft_response = self._create_limit_order(kft_params, params.clientOid, test=test)
        
        # Map kft.CreateOrderResponse to common_types.CreateOrderResponse
        return common_types.CreateOrderResponse(orderId=kft_response['orderId'])

    def create_market_order(
        self,
        params: common_types.MarketOrderRequest | common_types.StopMarketOrderRequest,
        test: bool = False,
    ) -> common_types.CreateOrderResponse:
        """Create a market order using public interface types.
        
        Args:
            params: Common types market order request (MarketOrderRequest or StopMarketOrderRequest)
            test: Whether to use test endpoint
            
        Returns:
            CreateOrderResponse with orderId
        """        # Convert TradingPair to exchange symbol
        exchange_symbol = self._get_symbol_id(params.trading_pair)
        
        # Handle StopMarketOrderRequest specific fields
        if isinstance(params, common_types.StopMarketOrderRequest):
            stop_val = params.stop
            stop_price_val = str(params.stopPrice)
            stop_price_type_val = params.stopPriceType
        else:
            # For regular market orders, we need to provide default values for required fields
            stop_val = None
            stop_price_val = None
            stop_price_type_val = None
            
        # Map common_types request to kft.MarketOrderRequest
        kft_params: kft.MarketOrderRequest = {
            'clientOid': str(params.clientOid),
            'symbol': exchange_symbol,
            'side': params.order_side,
            'type': 'market',
            'leverage': str(params.leverage),
            'size': params.amount_lots,
            'marginMode': params.marginMode,
            'reduceOnly': params.reduceOnly,
            'forceHold': False,  # Default value
            'stp': None,         # Default value
            'remark': None,      # Default value
            'stop': stop_val,
            'stopPrice': stop_price_val,
            'stopPriceType': stop_price_type_val
        }
          # Call the private method with the ClientOid from params
        kft_response = self._create_market_order(kft_params, params.clientOid, test=test)
        
        # Map kft.CreateOrderResponse to common_types.CreateOrderResponse
        return common_types.CreateOrderResponse(orderId=kft_response['orderId'])

    def create_take_profit_order(
        self,
        trading_pair: TradingPair,
        position_direction: Direction,
        lots: int,
        price: float,
        leverage: int,
        clientOid: ClientOid,
        margin_mode: common_types.MarginMode,
        test: bool = False,
    ) -> common_types.CreateOrderResponse:
        """Create a take profit order using StopMarketOrderRequest."""
        # For Kucoin, we create a take profit order using StopMarketOrderRequest
        # For a long position, we sell to take profit (stop="up")
        # For a short position, we buy to take profit (stop="down")
        order_side = "sell" if position_direction == "long" else "buy"
        stop_direction = "up" if position_direction == "long" else "down"

        tp_params = common_types.StopMarketOrderRequest(
            type="market",
            leverage=leverage,
            trading_pair=trading_pair,
            order_side=order_side,  # type: ignore
            amount_lots=lots,
            marginMode=margin_mode,
            reduceOnly=True,     # TP orders should reduce position
            clientOid=clientOid,
            stop=stop_direction,  # type: ignore
            stopPriceType="TP",
            stopPrice=price
        )
        
        return self.create_market_order(tp_params, test=test)

    def create_stop_loss_order(
        self,
        trading_pair: TradingPair,
        position_direction: Direction,
        lots: int,
        price: float,
        leverage: int,
        clientOid: ClientOid,
        margin_mode: common_types.MarginMode,
        test: bool = False,
    ) -> common_types.CreateOrderResponse:
        """Create a stop loss order using StopMarketOrderRequest."""
        # For Kucoin, we create a stop loss order using StopMarketOrderRequest
        # For a long position, we sell to stop loss (stop="down")
        # For a short position, we buy to stop loss (stop="up")
        order_side = "sell" if position_direction == "long" else "buy"
        stop_direction = "down" if position_direction == "long" else "up"

        sl_params = common_types.StopMarketOrderRequest(
            type="market",
            leverage=leverage,
            trading_pair=trading_pair,
            order_side=order_side,  # type: ignore
            amount_lots=lots,
            marginMode=margin_mode,
            reduceOnly=True,     # SL orders should reduce position
            clientOid=clientOid,
            stop=stop_direction,  # type: ignore
            stopPriceType="TP",  # Using TP as stop price type
            stopPrice=price
        )
        
        return self.create_market_order(sl_params, test=test)

    def adjust_price(self, price: float, trading_pair: TradingPair) -> float:
        """Adjust price to conform to exchange tick size requirements."""
        return self._adjust_price(price, trading_pair)

    def change_auto_deposit_status(self, trading_pair: TradingPair, status: bool) -> bool:
        """Change auto deposit status for a symbol."""
        # Map TradingPair to KuCoin futures contract symbol
        symbol_id = self._get_symbol_id(trading_pair)
        return self._change_auto_deposit_status(symbol_id, status)

    def change_cross_leverage(self, trading_pair: TradingPair, leverage: float) -> bool:
        symbol_id = self._get_symbol_id(trading_pair)
        return self._change_cross_leverage(symbol_id, leverage)

    def change_margin_mode(self, trading_pair: TradingPair, margin_mode: common_types.MarginMode) -> bool:
        """Change margin mode for a symbol."""        # Map TradingPair to KuCoin futures contract symbol
        symbol_id = self._get_symbol_id(trading_pair)
        
        # Map common_types.MarginMode to kft.MarginMode
        kft_margin_mode: kft.MarginMode = margin_mode  # type: ignore
        
        return self._change_margin_mode(symbol_id, kft_margin_mode)

    def cancel_order(self, order_id: str, trading_pair: Optional[TradingPair] = None) -> List[str]:
        """Cancel an order by order ID. If trading_pair is provided, it will be used to resolve the symbol."""
        return self._cancel_order(order_id)

    def close_position(self, trading_pair: TradingPair, margin_mode: common_types.MarginMode, test: bool = False, channel: str = 'MC', clientOid: Optional[ClientOid] = None) -> common_types.CreateOrderResponse:
        """Close a position."""
        # Convert TradingPair to exchange symbol
        exchange_symbol = self._get_symbol_id(trading_pair)
        
        # Map common_types.MarginMode to kft.MarginMode
        kft_margin_mode: kft.MarginMode = margin_mode
        
        # Call the private method to close the position
        kft_response = self._close_position(exchange_symbol, kft_margin_mode, test=test, channel=channel, clientOid=clientOid)
        
        return common_types.CreateOrderResponse(orderId=kft_response['orderId'])

    def fetch_balance(self) -> common_types.Balance:
        """Fetch account balance."""
        # Call the private method to fetch the raw balance data
        raw_balance = self._fetch_balance()
        
        # Map the raw response to the common type
        return common_types.Balance(
            accountEquity=raw_balance['accountEquity'],
            unrealisedPNL=raw_balance['unrealisedPNL'],
            marginBalance=raw_balance['marginBalance'],
            positionMargin=raw_balance['positionMargin'],
            orderMargin=raw_balance['orderMargin'],
            frozenFunds=raw_balance['frozenFunds'],
            availableBalance=raw_balance['availableBalance'],
            currency=raw_balance['currency']
        )

    def fetch_closed_orders(self, trading_pair: Optional[TradingPair] = None, since: Optional[TimestampMilliseconds] = None, limit: Optional[int] = None, side: Optional[common_types.OrderSide]=None) -> List[common_types.Order]:
        """Fetch closed orders."""
        # Convert TradingPair to exchange symbol if provided
        exchange_symbol = self._get_symbol_id(trading_pair) if trading_pair else None
        
        # Call the private method to fetch raw closed orders
        raw_orders = self._fetch_closed_orders(exchange_symbol, since, limit, params=None)
        
        # Map each raw order to the common type
        orders = []
        for raw_order in raw_orders:
            # Filter by side if specified
            if side and raw_order.get('side') != side:  # type: ignore
                continue
            
            # Use helper method to map the raw order to common type
            order = self._map_raw_order_to_common(raw_order)
            orders.append(order)
        
        return orders

    def fetch_order_by_coid(self, coid: ClientOid) -> common_types.Order:
        """Fetch order by client order ID."""
        # Call the private method to fetch the raw order data
        raw_order = self._fetch_order_by_coid(coid)
          # Use helper method to map the raw order to common type
        return self._map_raw_order_to_common(raw_order)

    def fetch_order_by_id(self, order_id: str) -> common_types.Order:
        """Fetch order by order ID."""
        # Call the private method to fetch the raw order data
        raw_order = self._fetch_order_by_id(order_id)
        
        # Use helper method to map the raw order to common type
        return self._map_raw_order_to_common(raw_order)

    def fetch_order_by_symbol(self, trading_pair: TradingPair, since: Optional[TimestampMilliseconds] = None, until: Optional[TimestampMilliseconds] = None, limit: Optional[int] = None) -> List[common_types.Order]:
        """Fetch orders by symbol."""
        # Convert TradingPair to exchange symbol
        exchange_symbol = self._get_symbol_id(trading_pair)
        
        # Call the private method to fetch raw orders
        raw_orders = self._fetch_order_by_symbol(exchange_symbol, since, until, limit)        # Map each raw order to the common type
        orders = []
        for raw_order in raw_orders:
            # Use helper method to map the raw order to common type
            order = self._map_raw_order_to_common(raw_order)
            orders.append(order)
        
        return orders

    def fetch_orders_by_status(self, status: Status, trading_pair: Optional[TradingPair] = None, since: Optional[TimestampMilliseconds] = None, limit: Optional[int] = 1000) -> List[common_types.Order]:
        """Fetch orders by status."""
        # Convert TradingPair to exchange symbol if provided
        exchange_symbol = self._get_symbol_id(trading_pair) if trading_pair else None
        
        # Create params dict for the private method
        params = kft.OrderRequestParams()
        
        # Call the private method to fetch raw orders
        raw_orders = self._fetch_orders_by_status(status, exchange_symbol, since, limit, params)        # Map each raw order to the common type
        orders = []
        for raw_order in raw_orders:
            # Use helper method to map the raw order to common type
            order = self._map_raw_order_to_common(raw_order)
            orders.append(order)
        
        return orders

    def fetch_position(self, trading_pair: TradingPair, side: Optional[common_types.OrderSide] = None) -> common_types.Position:
        """Fetch position for a trading pair.
        
        Args:
            trading_pair: The trading pair to fetch position for
            side: Optional side filter (ignored for KuCoin as it doesn't support hedge mode)
        
        Note: KuCoin doesn't support hedge mode, so the side parameter is ignored
        and we always return the single position for the trading pair.
        """
        # Convert TradingPair to exchange symbol
        exchange_symbol = self._get_symbol_id(trading_pair)
        
        # Call the private method to fetch the raw position data
        # Note: KuCoin doesn't support hedge mode, so we ignore the side parameter
        raw_position = self._fetch_position(exchange_symbol)
        
        # Map direction from KuCoin format to common format
        if raw_position['currentQty'] > 0 and (side is None or side == 'buy'):
            direction = 'long'
        elif raw_position['currentQty'] < 0 and (side is None or side == 'sell'):
            direction = 'short'
        else:
            raise PositionNotFoundError(str(trading_pair), "KucoinFuturesExchange")
        # Map the raw response to the common type
        return common_types.Position(
            avgEntryPrice=raw_position['avgEntryPrice'],
            currentLots=abs(raw_position['currentQty']),  # Use absolute value for size
            currentQty=abs(raw_position['currentQty']) * self.markets[trading_pair].lot_size,
            id=raw_position['id'],
            isOpen=raw_position['isOpen'],
            leverage=int(raw_position['leverage'] if 'leverage' in raw_position else raw_position['realLeverage'] if 'realLeverage' in raw_position else 1),
            liquidationPrice=raw_position['liquidationPrice'],
            marginMode=raw_position['marginMode'],
            markPrice=raw_position['markPrice'],
            openingTimestamp=common_types.TimestampMilliseconds(raw_position['openingTimestamp'] if raw_position['openingTimestamp'] > 1 else int(time.time() * 1000)),
            posCost=raw_position['posCost'],
            posInit=raw_position['posInit'],
            direction=direction,
            realisedPnl=raw_position['realisedPnl'],
            trading_pair=trading_pair,
            unrealisedPnl=raw_position['unrealisedPnl'],
            unrealisedPnlPcnt=raw_position['unrealisedPnlPcnt'],
            unrealisedRoePcnt=raw_position['unrealisedRoePcnt']
        )

    def fetch_positions(self) -> List[common_types.Position]:
        """Fetch all positions."""
        # Call the private method to fetch the raw positions data
        raw_positions = self._fetch_positions()
        
        # Map each raw position to the common type
        positions = []
        for raw_position in raw_positions:
            # Convert KuCoin symbol to standardized trading pair
            trading_pair = self._get_standardized_symbol(raw_position['symbol'])
            
            # Map direction from KuCoin format to common format
            if raw_position['currentQty'] > 0:
                direction = 'long'
            elif raw_position['currentQty'] < 0:
                direction = 'short'
            else:
                # No position, skip this one
                continue
                
            position = common_types.Position(
                avgEntryPrice=raw_position['avgEntryPrice'],
                currentLots=abs(raw_position['currentQty']),  # Use absolute value for size
                currentQty=abs(raw_position['currentQty']) * self.markets[trading_pair].lot_size,
                id=raw_position['id'],
                isOpen=raw_position['isOpen'],
                leverage=int(raw_position['leverage']),
                liquidationPrice=raw_position['liquidationPrice'],
                marginMode=raw_position['marginMode'],
                markPrice=raw_position['markPrice'],
                openingTimestamp=common_types.TimestampMilliseconds(raw_position['openingTimestamp']),
                posCost=raw_position['posCost'],
                posInit=raw_position['posInit'],
                direction=direction,
                realisedPnl=raw_position['realisedPnl'],
                trading_pair=trading_pair,
                unrealisedPnl=raw_position['unrealisedPnl'],
                unrealisedPnlPcnt=raw_position['unrealisedPnlPcnt'],
                unrealisedRoePcnt=raw_position['unrealisedRoePcnt']
            )
            positions.append(position)
        
        return positions

    def fetch_positions_history(self, since: Optional[TimestampMilliseconds] = None, trading_pair: Optional[TradingPair] = None) -> List[common_types.HistoricPosition]:
        """Fetch historical positions for a trading pair."""
        # Call the private method to fetch the raw historic positions data
        raw_historic_positions = self._fetch_positions_history(since, trading_pair)

        # Map each raw historic position to the common type
        historic_positions = []
        for raw_position in raw_historic_positions:
            # Convert KuCoin symbol to standardized trading pair
            trading_pair = self._get_standardized_symbol(raw_position['symbol'])
            estimate_qty = float(raw_position['pnl']) / (float(raw_position['closePrice']) - float(raw_position['openPrice'])) if float(raw_position['closePrice']) != float(raw_position['openPrice']) else 0
            lots = round(estimate_qty / self.market(trading_pair).lot_size)
            historic_position = common_types.HistoricPosition(
                closeId=raw_position['closeId'],
                userId=raw_position['userId'],
                trading_pair=trading_pair,
                settleCurrency=raw_position['settleCurrency'],
                leverage=raw_position['leverage'],
                type=raw_position['type'],
                pnl=raw_position['pnl'],
                realisedGrossCost=raw_position['realisedGrossCost'],
                tradeFee=raw_position['tradeFee'],
                fundingFee=raw_position['fundingFee'],
                openTime=common_types.TimestampMilliseconds(raw_position['openTime']),
                closeTime=common_types.TimestampMilliseconds(raw_position['closeTime']),
                openPrice=raw_position['openPrice'],
                closePrice=raw_position['closePrice'],
                marginMode=raw_position['marginMode'],
                maxFilledLots=lots
            )
            historic_positions.append(historic_position)
        
        return historic_positions

    def fetch_untriggered_stop_orders(self, trading_pair: Optional[TradingPair] = None, since: Optional[TimestampMilliseconds] = None, limit: Optional[int] = 1000) -> List[common_types.Order]:
        """Fetch untriggered stop orders."""
        symbol = self._get_symbol_id(trading_pair) if trading_pair else None
        params = kft.OrderRequestParams()
        raw_orders = self._fetch_untriggered_stop_orders(symbol, since, limit, params)
        return [self._map_raw_order_to_common(o) for o in raw_orders]

    def cancel_tpsl_order(self, order_id: str, trading_pair: TradingPair) -> List[str]:
        """Cancel a take profit or stop loss order by order ID.
        
        For KuCoin, TPSL orders are handled as regular stop orders,
        so we use the standard order cancellation endpoint.
        """
        return self.cancel_order(order_id)

    def get_margin_mode(self, trading_pair: TradingPair) -> common_types.MarginMode:
        """Get margin mode for a symbol."""
        symbol = self._get_symbol_id(trading_pair)
        return self._get_margin_mode(symbol)

    def allows_cross_mode(self, trading_pair: TradingPair) -> bool:
        return True

    def get_recent_fills(self) -> List[common_types.Fill]:
        """Get recent fills."""
        raw_fills = self._get_recent_fills()
        fills: List[common_types.Fill] = []
        for raw in raw_fills:
            trading_pair = self._get_standardized_symbol(raw['symbol'])
            fills.append(common_types.Fill(
                trading_pair=trading_pair,
                tradeId=raw['tradeId'],
                orderId=raw['orderId'],
                side=raw['side'],
                liquidity=raw['liquidity'],
                forceTaker=raw['forceTaker'],
                price=raw['price'],
                size=raw['size'],
                value=raw['value'],
                feeRate=raw['feeRate'],
                fixFee=raw['fixFee'],
                feeCurrency=raw['feeCurrency'],
                stop=raw['stop'],
                fee=raw['fee'],
                orderType=raw['orderType'],
                tradeType=raw['tradeType'],
                createdAt=TimestampMilliseconds(raw['createdAt']),
                settleCurrency=raw['settleCurrency'],
                tradeTime=TimestampNanoseconds(raw['tradeTime']),
                openFeePay=raw['openFeePay'],
                closeFeePay=raw['closeFeePay'],
                marginMode=raw['marginMode'],
                subTradeType=raw.get('subTradeType'),
                displayType=raw['displayType']
            ))
        return fills

    def get_standardized_symbol(self, exchange: str) -> TradingPair:
        """Convert a Kucoin symbol to standardized trading pair format."""
        return self._get_standardized_symbol(kft.KucoinFuturesContract(exchange))

    def get_symbol_id(self, standardized_trading_pair: TradingPair) -> str:
        """Convert a standardized trading pair to exchange symbol format."""
        return self._get_symbol_id(standardized_trading_pair)

    def request(self, path: str, api: str = 'public', method: str = 'GET', params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a request to the exchange API."""
        if params is None:
            params = {}
        return self._request(path, api, method, params)

    @property
    def markets(self) -> Dict[TradingPair, common_types.FuturesMarket]:
        """Get market information."""
        self.load_markets()
        return self._markets_common
