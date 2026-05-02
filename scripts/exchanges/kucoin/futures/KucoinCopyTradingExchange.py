import time
from typing import List, Optional

import exchanges.kucoin.futures.ct_types as ct
import exchanges.kucoin.futures.types as kft
from exchanges.types.common import (
    ClientOid,
    ClientOrderGoal,
    ClientOrderType,
    TradingPair,
)
from exchanges.kucoin.futures.KucoinFuturesExchange import KucoinFuturesExchange
from exchanges.types.exceptions import LeverageError, SymbolNotSupportedForCopyTradingException
from my_types.config_models import KucoinConfig
from utils.SqlManager import SQLiteManager

class KucoinFuturesCopyTradingExchange(KucoinFuturesExchange):
    def get_name(self) -> str:
        """Return the name of the exchange."""
        return "Kucoin Futures Copy Trading"

    def __init__(self, config: KucoinConfig, db_manager: SQLiteManager) -> None:
        super().__init__(config, db_manager)
        self._last_load_full_markets = 0
        self.max_leverage = 20
        
    def _load_markets(self) -> None:
        # Check if we need a full reload (no markets or last full load > 1 day ago)
        one_day_ago = time.time() - (24 * 60 * 60)
        needs_full_reload = not self._markets or self._last_load_full_markets < one_day_ago
        if needs_full_reload:
            # Full reload: fetch all markets and filter unsupported ones
            super()._load_markets()
            # TODO: store this in the database to avoid reloading every time
            if False:
                if not self._markets:
                    return
                    # Copy markets to iterate over, as we'll be modifying the original dict
                markets_to_check = self._markets.copy()
                first_request = True
                for key, market in markets_to_check.items():
                    try:
                        response = self._get_max_open_size(key, float(market['lastTradePrice']), 1)
                    except SymbolNotSupportedForCopyTradingException:
                        if key in self._markets:
                            del self._markets[key]
                    except Exception as e:
                        error_msg = str(e)
                        print(f"Error fetching max open size for {key}: {error_msg}")
                        
                        # Check for 403 Forbidden error on first request - indicates no copy trading permissions
                        if first_request and "403 Client Error: Forbidden" in error_msg:
                            print("403 Forbidden error detected on first copy trading API call - API key lacks copy trading permissions")
                            print("Clearing all markets as copy trading is not available")
                            self._markets.clear()
                            break
                        
                        if key in self._markets:
                            del self._markets[key]
                    
                    first_request = False
            
            # Update the last full load timestamp
            self._last_load_full_markets = time.time()
        else:
            # Quick update: only refresh data for existing supported pairs
            if time.time() - self._last_load_markets < 60 * 60:
                return
                
            # Store the current reduced market set
            supported_pairs = set(self._markets.keys())
            
            # Fetch fresh market data from parent
            super()._load_markets()
              # Only keep the markets that were in our supported set
            updated_markets = {}
            for key in supported_pairs:
                if key in self._markets:
                    updated_markets[key] = self._markets[key]
            
            # Replace markets with the updated subset
            self._markets = updated_markets
            
    # Check params
    def _ct_create_order(
        self,
        params: ct.LimitOrderRequest | ct.MarketOrderRequest | ct.CloseOrderRequest,
        clientOid: ClientOid,
        test: bool = False,
    ) -> ct.CreateOrderResponse:
        path = '/api/v1/copy-trade/futures/orders'
        if test:
            path = '/api/v1/copy-trade/futures/orders/test'
        api = 'private'
        
        method = 'POST'
        params.clientOid = clientOid.__str__()
        
        response = self.request(path, api, method, params.__dict__)
        data = response.get('data', {})
        orderId = str(data.get('orderId', ''))
        coid = str(data.get('clientOid', ''))
        return ct.CreateOrderResponse(**{'orderId': orderId, 'clientOid': coid})
    
    def _add_take_profit_stop_loss_order(self, params: ct.AddTakeProfitAndStopLossOrderRequest) -> ct.CreateOrderResponse:
        """Add Take Profit And Stop Loss Order
        
        Args:
            params: Order parameters following KuCoin Copy Trading API specification
            test: Whether to use test endpoint
            
        Returns:
            CreateOrderResponse with orderId
        """
        path = '/api/v1/copy-trade/futures/st-orders'
        api = 'private'
        method = 'POST'
        
        response = self.request(path, api, method, params.__dict__)
        data = response.get('data', {})
        orderId = str(data.get('orderId', ''))
        clientOid = str(data.get('clientOid', ''))
        return ct.CreateOrderResponse(**{'orderId': orderId, 'clientOid': clientOid})
    
    def _create_limit_order(
        self,
        params: kft.LimitOrderRequest,
        clientOid: ClientOid,
        test: bool = False,
    ) -> kft.CreateOrderResponse:
        ct_params = ct.LimitOrderRequest(
            symbol=self.get_symbol_id(params['symbol']) if isinstance(params['symbol'], TradingPair) else params['symbol'],
            clientOid=clientOid.__str__(),
            remark=params.get('remark'),
            leverage=int(round(float(params['leverage']))),
            side=ct.Side.BUY if params['side'] == 'buy' else ct.Side.SELL,
            size=params['size'],
            type='limit',
            price=params['price'],
            stop=ct.StopDirection.DOWN if params.get('stop') == 'down' else (ct.StopDirection.UP if params.get('stop') == 'up' else None),
            stopPriceType=ct.StopPriceType.TP if params.get('stopPriceType') == 'TP' else (ct.StopPriceType.MP if params.get('stopPriceType') == 'MP' else None),
            stopPrice=params.get('stopPrice', None),
            reduceOnly=params.get('reduceOnly', False) or False,
            marginMode=ct.MarginMode.CROSS if params.get('marginMode') == 'CROSS' else ct.MarginMode.ISOLATED,
            timeInForce=ct.TimeInForce.GTC if params.get('timeInForce') == 'GTC' else (ct.TimeInForce.IOC if params.get('timeInForce') == 'IOC' else None),
            postOnly=params.get('postOnly', False) or False,
            hidden=params.get('hidden', False) or False,
            iceberg=params.get('iceberg', False) or False,
            visibleSize=params.get('visibleSize', None)
        )
        ct_COR = self._ct_create_order(ct_params, clientOid, test=test)
        return {'orderId': ct_COR.orderId}
    
    def _create_market_order(
        self,
        params: kft.MarketOrderRequest,
        clientOid: ClientOid,
        test: bool = False,
    ) -> kft.CreateOrderResponse:
        # Map FuturesExchange MarketOrderRequest to CopyTrading MarketOrderRequest
        ct_params = ct.MarketOrderRequest(
            symbol=self.get_symbol_id(params['symbol']) if isinstance(params['symbol'], TradingPair) else params['symbol'],
            clientOid=clientOid.__str__(),
            remark=params.get('remark'),
            leverage=int(round(float(params['leverage']))),
            side=ct.Side.BUY if params['side'] == 'buy' else ct.Side.SELL,
            size=params['size'],
            type='market',
            stop=ct.StopDirection.DOWN if params.get('stop') == 'down' else (ct.StopDirection.UP if params.get('stop') == 'up' else None),
            stopPriceType=ct.StopPriceType.TP if params.get('stopPriceType') == 'TP' else (ct.StopPriceType.MP if params.get('stopPriceType') == 'MP' else None),
            stopPrice=params.get('stopPrice'),
            reduceOnly=params.get('reduceOnly', False) or False,
            marginMode=ct.MarginMode.CROSS if params.get('marginMode') == 'CROSS' else ct.MarginMode.ISOLATED,
        )
        
        # Call the copy trading create_order method
        ct_response = self._ct_create_order(ct_params, clientOid, test=test)

        # Return in FuturesExchange format
        return {'orderId': ct_response.orderId}
    
    def _get_margin_mode(self, symbol: kft.KucoinFuturesContract) -> kft.MarginMode:
        """Delegate to the parent — CT API keys can use the standard read endpoint."""
        return super()._get_margin_mode(symbol)

    def allows_cross_mode(self, trading_pair: TradingPair) -> bool:
        # Copy trading now supports both ISOLATED and CROSS modes.
        return True

    def _switch_margin_mode_ct(
        self, symbol: kft.KucoinFuturesContract, marginMode: Optional[kft.MarginMode]
    ) -> ct.SwitchMarginModeResponse:
        """Call the Copy Trading-specific switch-margin-mode endpoint.

        POST /api/v1/copy-trade/futures/position/changeMarginMode
        """
        path = '/api/v1/copy-trade/futures/position/changeMarginMode'
        api = 'private'
        method = 'POST'
        params: dict = {'symbol': symbol}
        if marginMode is not None:
            params['marginMode'] = marginMode if isinstance(marginMode, str) else marginMode.value
        response = self.request(path, api, method, params)
        data = response.get('data', {})
        return ct.SwitchMarginModeResponse(
            symbol=data.get('symbol', symbol),
            marginMode=data.get('marginMode', ''),
        )

    def _change_margin_mode(
        self, symbol: kft.KucoinFuturesContract, marginMode: kft.MarginMode
    ) -> bool:
        """Switch margin mode using the Copy Trading-specific endpoint."""
        result = self._switch_margin_mode_ct(symbol, marginMode)
        return result.marginMode == (marginMode if isinstance(marginMode, str) else marginMode.value)

    def _change_cross_leverage(self, symbol: kft.KucoinFuturesContract, leverage: float) -> bool:
        """Change cross margin leverage via the Copy Trading-specific endpoint.

        POST /api/v2/copy-trade/futures/changeCrossUserLeverage
        """
        path = '/api/v2/copy-trade/futures/changeCrossUserLeverage'
        api = 'private'
        method = 'POST'
        params = {'symbol': symbol, 'leverage': str(leverage)}
        response = self.request(path, api, method, params)
        data = response.get('data', False)
        if not data:
            raise LeverageError(response.get('msg', 'Could not change CT cross leverage'))
        return True
    
    def _close_position(
        self,
        trading_pair: TradingPair | kft.KucoinFuturesContract,
        margin_mode: kft.MarginMode,
        test: bool = False,
        channel: str = 'MC',
        clientOid: Optional[ClientOid] = None,
    ) -> kft.CreateOrderResponse:
        """Close Position
        
        Args:
            trading_pair: Trading pair to close
            margin_mode: Margin mode (ISOLATED or CROSS)
            test: Whether to use test endpoint
            channel: Channel abbreviation for clientOid
            clientOid: Client order ID
            
        Returns:
            CreateOrderResponse with orderId
        """
        if clientOid is None:
            clientOid = self._create_client_oid(channel)
        
        close_order_request = ct.CloseOrderRequest(
            symbol=trading_pair,
            closeOrder = True,
            type = ct.TypeEnum.MARKET,
            marginMode = ct.MarginMode.CROSS if margin_mode == kft.MarginMode.CROSS else ct.MarginMode.ISOLATED,
            clientOid = str(clientOid),
            reduceOnly=True,
            remark="")
        
        ctCOR = self._ct_create_order(close_order_request, clientOid, test=test)
        return {'orderId': ctCOR.orderId}
        
    
    def _close_position_ct(self, params: ct.CloseOrderRequest, clientOid: ClientOid) -> ct.CreateOrderResponse:
        """Close Position
        
        Args:
            params: Close order parameters
            clientOid: Client order ID
            
        Returns:
            CreateOrderResponse with orderId
        """
        return self._ct_create_order(params, clientOid, test=False)
    
    def _cancel_order(self, order_id: str) -> List[str]:
        """Cancel Order - Shadow method for FuturesExchange compatibility
        
        Args:
            order_id: The order ID to cancel
            
        Returns:
            List of cancelled order IDs (for FuturesExchange compatibility)
        """
        # Call the copy trading cancel_order_by_order_id method
        ct_response = self._cancel_order_by_order_id(order_id)
        
        # Return in FuturesExchange format (List[str])
        return ct_response.cancelledOrderIds
    
    def _cancel_order_by_order_id(self, order_id: str) -> ct.CancelOrderByOrderIdResponse:
        """Cancel Order By OrderId
        
        Args:
            order_id: The order ID to cancel
            
        Returns:
            Response with cancelledOrderIds
        """
        path = '/api/v1/copy-trade/futures/orders'
        api = 'private'
        method = 'DELETE'
        
        response = self.request(path, api, method, {'orderId': order_id})
        return ct.CancelOrderByOrderIdResponse(**response.get('data', {'cancelledOrderIds': []}))
    
    def _cancel_order_by_client_oid(self, client_oid: ClientOid, symbol: kft.KucoinFuturesContract | TradingPair) -> ct.CancelOrderByClientOidResponse:
        """Cancel Order By ClientOid
        
        Args:
            client_oid: The client order ID to cancel
            symbol: Trading symbol
            
        Returns:
            Response with cancelledOrderIds
        """
        path = '/api/v1/copy-trade/futures/orders/client-order'
        api = 'private'
        method = 'DELETE'
        params = {
            'clientOid': str(client_oid),
            'symbol': symbol
        }
        
        response = self.request(path, api, method, params)
        return ct.CancelOrderByClientOidResponse(**response.get('data', {'clientOid': []}))
    
    def _get_max_open_size(self, symbol: kft.KucoinFuturesContract | TradingPair, price: float, leverage: int) -> ct.GetMaxOpenSizeResponse:
        """Get Max Open Size
        
        Args:
            symbol: Trading symbol
            price: Order price
            leverage: Leverage level
            
        Returns:
            GetMaxOpenSizeResponse with max buy/sell open sizes
        """
        path = '/api/v1/copy-trade/futures/get-max-open-size'
        api = 'private'
        method = 'GET'
        params = {
            'symbol': symbol,
            'price': price,
            'leverage': str(leverage)
        }
        
        response = self.request(path, api, method, params)
        return ct.GetMaxOpenSizeResponse(**response.get('data', {}))
    
    def _get_max_withdraw_margin(self, symbol: str) -> ct.GetMaxWithdrawMarginResponse:
        """Get Max Withdraw Margin
        
        Args:
            symbol: Trading symbol
            
        Returns:
            GetMaxWithdrawMarginResponse with max withdraw margin
        """
        path = '/api/v1/copy-trade/futures/position/margin/max-withdraw-margin'
        api = 'private'
        method = 'GET'
        params = {
            'symbol': symbol
        }
        
        response = self.request(path, api, method, params)
        return ct.GetMaxWithdrawMarginResponse(**response.get('data', {}))
    
    def _add_isolated_margin(self, symbol: TradingPair | kft.KucoinFuturesContract, margin: str, biz_no: str) -> ct.AddIsolatedMarginResponse:
        """Add Isolated Margin
        
        Args:
            symbol: Trading symbol
            margin: Margin amount to add
            biz_no: Business number (unique identifier)
            
        Returns:
            AddIsolatedMarginResponse with deposit information
        """
        path = '/api/v1/copy-trade/futures/position/margin/deposit-margin'
        api = 'private'
        method = 'POST'
        params = {
            'symbol': symbol,
            'margin': margin,
            'bizNo': biz_no
        }
        
        response = self.request(path, api, method, params)
        return ct.AddIsolatedMarginResponse(**response.get('data', {}))
    
    def _remove_isolated_margin(self, symbol: TradingPair | kft.KucoinFuturesContract, withdraw_amount: str) -> ct.RemoveIsolatedMarginResponse:
        """Remove Isolated Margin
        
        Args:
            symbol: Trading symbol
            withdraw_amount: Amount to withdraw
            
        Returns:
            RemoveIsolatedMarginResponse with withdrawal information
        """
        path = '/api/v1/copy-trade/futures/position/margin/withdraw-margin'
        api = 'private'
        method = 'POST'
        params = {
            'symbol': symbol,
            'withdrawAmount': withdraw_amount
        }
        
        response = self.request(path, api, method, params)
        return ct.RemoveIsolatedMarginResponse(**response.get('data', {}))
    
    def _modify_isolated_margin_risk_limit(self, symbol: str, level: int) -> ct.ModifyIsolatedMarginRiskLimitResponse:
        """Modify Isolated Margin Risk Limit
        
        Args:
            symbol: Trading symbol
            level: Risk limit level
            
        Returns:
            ModifyIsolatedMarginRiskLimitResponse with updated risk limit
        """
        path = '/api/v1/copy-trade/futures/position/risk-limit-level/change'
        api = 'private'
        method = 'POST'
        params = {
            'symbol': symbol,
            'level': level
        }
        
        response = self.request(path, api, method, params)
        return ct.ModifyIsolatedMarginRiskLimitResponse(**response.get('data', {}))
    
    def _modify_auto_deposit_status(self, symbol: str, status: bool) -> ct.ModifyAutoDepositStatusResponse:
        """Modify Isolated Margin Auto-Deposit Status
        
        Args:
            symbol: Trading symbol
            status: Auto-deposit status (true/false)
            
        Returns:
            ModifyAutoDepositStatusResponse with updated status
        """
        path = '/api/v1/copy-trade/futures/position/margin/auto-deposit-status'
        api = 'private'
        method = 'POST'
        params = {
            'symbol': symbol,
            'status': status
        }
        
        response = self.request(path, api, method, params)
        return ct.ModifyAutoDepositStatusResponse(**response.get('data', {}))
