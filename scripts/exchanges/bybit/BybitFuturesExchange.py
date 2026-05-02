"""Bybit futures exchange powered by CCXT."""

from __future__ import annotations

import ccxt  # type: ignore
import time

from exchanges.CcxtExchange import CcxtExchange
from my_types.config_models import BybitConfig
from utils.SqlManager import SQLiteManager
from typing import Optional, Any, Dict, List
from exchanges.types.common import (
    StopMarketOrderRequest,
    TradingPair,
    Direction,
    ClientOid,
    MarginMode,
    CreateOrderResponse,
    Order,
)


class ByBitFuturesExchange(CcxtExchange):
    """Concrete :class:`CcxtExchange` implementation for Bybit Production.
    
    This exchange connects to Bybit's PRODUCTION environment.
    
    Behavior based on config.testnet:
    - testnet=True: Uses internal mocking, no real API calls
    - testnet=False: Connects to PRODUCTION Bybit endpoints with real API calls
    """

    def __init__(self, config: BybitConfig, db_manager: SQLiteManager, use_demo_trading: bool = False) -> None:
        """Initialize Bybit Futures exchange.
        
        Args:
            config: Bybit configuration with API credentials
            db_manager: SQLite database manager
            use_demo_trading: If True, enables demo trading mode (for prop firm accounts)
        """
        params = {
            "apiKey": config.bybit_api_key,
            "secret": config.bybit_api_secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": "swap",
                "recvWindow": 20000,  # Increase receive window to 20 seconds
                "adjustForTimeDifference": True,  # Auto-adjust for time sync issues
            },
        }
        client = ccxt.bybit(params)
        
        # Configure sandbox and demo trading modes BEFORE any API calls
        if hasattr(client, "set_sandbox_mode"):
            client.set_sandbox_mode(False)
        if hasattr(client, "enable_demo_trading"):
            client.enable_demo_trading(use_demo_trading)

        # CRITICAL: Synchronize with exchange server time BEFORE calling super().__init__()
        if hasattr(client, "load_time_difference"):
            try:
                client.load_time_difference()  # type: ignore
            except Exception:
                pass

        # Now call parent constructor which will call load_markets()
        exchange_name = "Bybit Demo (Demo Trading)" if use_demo_trading else "Bybit Futures"
        super().__init__(exchange_name, client, db_manager)
        self.config = config
        
        # Cache for stop order snapshots to avoid redundant API calls
        # Key: (symbol_id, category), Value: (timestamp, order_ids_set)
        self._stop_order_cache: Dict[tuple[str, str], tuple[float, set[str]]] = {}
        self._stop_order_cache_ttl: float = 2.0  # Cache valid for 2 seconds

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
        """Create TP order using Bybit V5 position trading stop endpoint.
        
        This uses set_position_tpsl instead of creating a separate limit order,
        which is the proper way to set TP/SL on Bybit positions.
        """
        # Convert lots to size (quantity)
        market = self.markets.get(trading_pair)
        if not market:
            raise ValueError(f"Unknown trading pair {trading_pair}")
        size = lots * market.lot_size
        
        # Resolve positionIdx dynamically (supports hedge mode)
        try:
            position_idx = self._resolve_position_idx(trading_pair, position_direction)
        except Exception:
            # Fallback to one-way index
            position_idx = 0

        # Set partial TP for the position
        resp = self.set_position_tpsl(
            trading_pair=trading_pair,
            position_idx=position_idx,
            position_direction=position_direction,
            tpsl_mode="Partial",
            takeProfit=price,
            tpTriggerBy="LastPrice",
            tpSize=size,
        )
        
        # Extract order ID from response - use first new order ID or fallback to clientOid
        order_ids = resp.get("orderIds", [])
        order_id = order_ids[0] if order_ids else ""
        return CreateOrderResponse(orderId=order_id)

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
        """Create SL order using Bybit V5 position trading stop endpoint.
        
        This uses set_position_tpsl instead of creating a separate limit order,
        which is the proper way to set TP/SL on Bybit positions.
        
        NOTE: For Full stop losses, Bybit updates the existing order instead of creating
        a new one. In this case, we return the existing order ID from before the update
        to prevent the FuturesTrader from canceling the updated order.
        """
        # Convert lots to size (quantity)
        market = self.markets.get(trading_pair)
        if not market:
            raise ValueError(f"Unknown trading pair {trading_pair}")
        size = lots * market.lot_size
        
        # Resolve positionIdx dynamically (supports hedge mode)
        try:
            position_idx = self._resolve_position_idx(trading_pair, position_direction)
        except Exception:
            # Fallback to one-way index
            position_idx = 0

        # Set partial SL for the position
        resp = self.set_position_tpsl(
            trading_pair=trading_pair,
            position_idx=position_idx,
            position_direction=position_direction,
            tpsl_mode="Partial",
            stopLoss=price,
            slTriggerBy="LastPrice",
            slSize=size,
        )
        
        # Extract order ID from response
        # If orderIds is empty, it means Bybit updated an existing order instead of creating new one
        # In this case, return the existing order ID that was updated
        order_ids = resp.get("orderIds", [])
        existing_order_id = resp.get("existingOrderId", "")
        
        if order_ids:
            # New order was created
            order_id = order_ids[0]
        elif existing_order_id:
            # Existing order was updated - return its ID so trailing_stop_loss knows not to cancel it
            order_id = existing_order_id
        else:
            # Fallback to empty string
            order_id = ""
            
        return CreateOrderResponse(orderId=order_id)

    def create_market_order(
        self,
        params,
        test: bool = False,
    ) -> CreateOrderResponse:
        """Wrap base market order creation to handle Bybit positionIdx mismatch when opening positions.

        This calls the generic implementation via super() and retries with sensible
        positionIdx candidates (1 for buy/long, 2 for sell/short) if Bybit complains
        about position index not matching position mode.
        """
        try:
            return super().create_market_order(params, test=test)
        except Exception as exc:
            msg = str(exc)
            low = msg.lower()
            if "position idx not match position mode" in low or ("position idx" in low and "position mode" in low):
                # Build fallback payload and retry directly with positionIdx candidates
                trading_pair = params.trading_pair
                symbol = self._to_symbol(trading_pair)
                amount = self._amount_from_lots(trading_pair, params.amount_lots)
                base_payload = {"reduceOnly": params.reduceOnly, "leverage": params.leverage}
                if isinstance(params, StopMarketOrderRequest):
                    # preserve stop params if present
                    base_payload.update({"stop": params.stop, "stopPrice": params.stopPrice, "triggerPrice": params.stopPrice, "stopPriceType": params.stopPriceType})

                # Choose candidates based on side
                if params.order_side == "buy":
                    candidates = [1, 0, 2]
                else:
                    candidates = [2, 0, 1]

                last_exc = exc
                for idx in candidates:
                    try:
                        payload = base_payload.copy()
                        payload["positionIdx"] = idx
                        self.logger.debug(f"Retrying market order for {symbol} with positionIdx={idx}")
                        order = self.client.create_order(symbol, "market", params.order_side, amount, None, payload)
                        order_id = str(order.get("id"))
                        self._record_order_symbol(order_id, symbol)
                        return CreateOrderResponse(orderId=order_id)
                    except Exception as e:
                        last_exc = e
                        continue
                # If none succeeded re-raise original
                raise last_exc
            # Unknown error: re-raise
            raise

    def create_limit_order(self, params, test: bool = False) -> CreateOrderResponse:
        """Wrap base limit order creation to handle Bybit positionIdx mismatch similarly."""
        try:
            return super().create_limit_order(params, test=test)
        except Exception as exc:
            msg = str(exc)
            low = msg.lower()
            if "position idx not match position mode" in low or ("position idx" in low and "position mode" in low):
                trading_pair = params.trading_pair
                symbol = self._to_symbol(trading_pair)
                amount = self._amount_from_lots(trading_pair, params.size)
                base_payload = {"reduceOnly": params.reduceOnly, "leverage": params.leverage}
                if params.takeProfit is not None:
                    base_payload["takeProfit"] = params.takeProfit
                if params.stopLoss is not None:
                    base_payload["stopLoss"] = params.stopLoss
                if params.tpslMode is not None:
                    base_payload["tpslMode"] = params.tpslMode
                if params.tpTriggerBy is not None:
                    base_payload["tpTriggerBy"] = params.tpTriggerBy
                if params.slTriggerBy is not None:
                    base_payload["slTriggerBy"] = params.slTriggerBy
                if params.__dict__.get("price"):
                    price = params.price
                else:
                    price = None

                if params.side == "buy":
                    candidates = [1, 0, 2]
                else:
                    candidates = [2, 0, 1]

                last_exc = exc
                for idx in candidates:
                    try:
                        payload = base_payload.copy()
                        payload["positionIdx"] = idx
                        self.logger.debug(f"Retrying limit order for {symbol} with positionIdx={idx}")
                        order = self.client.create_order(symbol, "limit", params.side, amount, price, payload)
                        order_id = str(order.get("id"))
                        self._record_order_symbol(order_id, symbol)
                        return CreateOrderResponse(orderId=order_id)
                    except Exception as e:
                        last_exc = e
                        continue
                raise last_exc
            raise

    def create_bracket_order(
        self,
        trading_pair: TradingPair,
        position_direction: Direction,
        lots: int,
        take_profit: float,
        stop_loss: float,
        leverage: int,
        tp_clientOid: ClientOid | None,
        sl_clientOid: ClientOid | None,
        margin_mode: MarginMode,
        test: bool = False,
    ) -> Dict[str, Any]:
        """Create TP and SL simultaneously for a position using set_position_tpsl.

        This sets both takeProfit and stopLoss in a single API call to ensure
        the exchange creates a bracket (single atomic TPSL update) rather than
        two separate orders which may appear as two independent orders.
        Returns the raw response from set_position_tpsl for caller convenience.
        """
        market = self.markets.get(trading_pair)
        if not market:
            raise ValueError(f"Unknown trading pair {trading_pair}")
        size = lots * market.lot_size

        # Resolve positionIdx for potential hedge mode
        try:
            position_idx = self._resolve_position_idx(trading_pair, position_direction)
        except Exception:
            position_idx = 0

        resp = self.set_position_tpsl(
            trading_pair=trading_pair,
            position_idx=position_idx,
            position_direction=position_direction,
            tpsl_mode="Partial",
            takeProfit=take_profit,
            tpTriggerBy="LastPrice",
            tpSize=size,
            stopLoss=stop_loss,
            slTriggerBy="LastPrice",
            slSize=size,
        )
        return resp

    def _resolve_position_idx(
        self,
        trading_pair: TradingPair | str,
        position_direction: Direction,
        category: str = "linear",
    ) -> int:
        """Resolve Bybit positionIdx for a trading pair and side.

        Bybit uses `positionIdx` to distinguish positions in hedge mode (1/2) and
        0 for one-way mode. This helper queries the V5 position list and returns
        the matching index for the specified side. Falls back to 0 if the information
        cannot be obtained.
        """
        sym = self._to_symbol(trading_pair)
        symbol_id = sym.split(":")[0].replace("/", "")
        method = getattr(self.client, "privateGetV5PositionList", None)
        if not method:
            return 0

        try:
            resp = method({"category": category, "symbol": symbol_id})
            positions = resp.get("result", {}).get("list", [])
            for pos in positions:
                # Try both 'side' and 'posSide' fields which vary across responses
                side = str(pos.get("side", "")).lower()
                pos_side = str(pos.get("posSide", "")).lower()
                if position_direction == "long" and (side in ("buy", "long") or pos_side == "long"):
                    return int(pos.get("positionIdx", 0) or 0)
                if position_direction == "short" and (side in ("sell", "short") or pos_side == "short"):
                    return int(pos.get("positionIdx", 0) or 0)
        except Exception:
            # If any error occurs, fall back to default 0
            return 0

        return 0

    def set_position_tpsl(
        self,
        trading_pair: TradingPair | str,
        position_idx: Optional[int] = None,
        position_direction: Optional[Direction] = None,
        tpsl_mode: str = "Partial",
        takeProfit: Optional[float] = None,
        tpTriggerBy: str = "LastPrice",
        tpSize: Optional[float] = None,
        stopLoss: Optional[float] = None,
        slTriggerBy: str = "LastPrice",
        slSize: Optional[float] = None,
        category: str = "linear",
        **kwargs
    ) -> Dict[str, Any]:
        """Set TP/SL for a position using Bybit V5 private API endpoint.

        This supports both Partial and Full tpsl modes and accepts sizes for
        partial TP/SLs. Returns the raw exchange response with added 'orderIds' field.

        The method is resilient to position index mismatches (hedge mode).
        If the initial call fails with a "position idx not match position mode" error,
        it will attempt to resolve and retry with the correct index.
        
        Uses caching to avoid redundant API calls when setting multiple TP/SL orders
        in quick succession (within 2 seconds).
        """
        # Convert trading pair to Bybit symbol id like 'BTCUSDT'
        sym = self._to_symbol(trading_pair)
        symbol_id = sym.split(":")[0].replace("/", "")
        cache_key = (symbol_id, category)
        current_time = time.time()

        # Check if we have a recent cached snapshot
        before_order_ids: set[str] = set()
        get_orders_method = getattr(self.client, "privateGetV5OrderRealtime", None)
        
        if cache_key in self._stop_order_cache:
            cache_time, cached_order_ids = self._stop_order_cache[cache_key]
            if current_time - cache_time < self._stop_order_cache_ttl:
                # Cache is still fresh, reuse it
                before_order_ids = cached_order_ids
                self.logger.debug(f"Using cached stop order snapshot for {symbol_id} (age: {current_time - cache_time:.2f}s)")
            else:
                # Cache expired, will fetch fresh data
                self.logger.debug(f"Stop order cache expired for {symbol_id}, fetching fresh snapshot")
        
        # If no valid cache, fetch current stop orders
        if not before_order_ids:
            try:
                if get_orders_method:
                    before_orders_resp = get_orders_method({
                        "category": category,
                        "symbol": symbol_id,
                        "orderFilter": "StopOrder",
                    })
                    if before_orders_resp.get("result", {}).get("list"):
                        before_order_ids = {
                            order.get("orderId") 
                            for order in before_orders_resp["result"]["list"]
                            if order.get("orderId")
                        }
                    # Cache the fresh snapshot
                    self._stop_order_cache[cache_key] = (current_time, before_order_ids)
            except Exception as e:
                self.logger.warning(f"Failed to snapshot stop orders before TPSL update: {e}")
                before_order_ids = set()

        # If position index not provided, attempt to resolve it using current positions
        if position_idx is None:
            if position_direction is not None:
                try:
                    position_idx = self._resolve_position_idx(trading_pair, position_direction, category=category)
                except Exception:
                    position_idx = 0
            else:
                position_idx = 0

        payload: Dict[str, Any] = {
            "category": category,
            "symbol": symbol_id,
            "positionIdx": int(position_idx),
            "tpslMode": tpsl_mode,
        }

        if takeProfit is not None:
            payload["takeProfit"] = str(takeProfit)
            payload["tpTriggerBy"] = tpTriggerBy
        if tpSize is not None:
            payload["tpSize"] = str(tpSize)
        if stopLoss is not None:
            payload["stopLoss"] = str(stopLoss)
            payload["slTriggerBy"] = slTriggerBy
            # Determine Full vs Partial correctly: if neither TP nor SL sizes are provided,
            # this is a Full update. If SL size is provided (or TP size), use the requested mode (usually Partial).
            if tpSize is None and slSize is None:
                payload["tpslMode"] = "Full"
            else:
                payload["tpslMode"] = tpsl_mode
        if slSize is not None:
            payload["slSize"] = str(slSize)

        for key, value in kwargs.items():
            if value is not None:
                payload[key] = str(value)

        # Call the raw private endpoint provided by CCXT for Bybit V5
        method = getattr(self.client, "privatePostV5PositionTradingStop", None)
        if not method:
            raise NotImplementedError("Bybit V5 position trading stop endpoint not available in CCXT client")

        def _call_method(p: Dict[str, Any]) -> Dict[str, Any]:
            try:
                return method(p)
            except Exception as e:
                raise

        try:
            # Log full payload to help debug tpsl behavior in live runs
            self.logger.debug("Calling privatePostV5PositionTradingStop with payload: %s", payload)
            resp = _call_method(payload)
        except Exception as e:
            msg = str(e)
            low = msg.lower()

            # Handle position index / position mode mismatch (hedge vs one-way)
            if "position idx not match position mode" in low or ("position idx" in low and "position mode" in low):
                self.logger.warning(f"Position idx mismatch detected for {symbol_id}: {msg}. Attempting to resolve correct index and retry.")
                # Try to resolve proper index based on position side and retry
                tried_idxs = set([int(payload.get("positionIdx", 0))])
                candidate_idxs = [1, 2, 0]
                success_resp = None
                for idx in candidate_idxs:
                    if idx in tried_idxs:
                        continue
                    try:
                        new_payload = payload.copy()
                        new_payload["positionIdx"] = idx
                        resp = _call_method(new_payload)
                        success_resp = resp
                        payload = new_payload
                        break
                    except Exception:
                        tried_idxs.add(idx)
                        continue
                if success_resp is not None:
                    resp = success_resp
                else:
                    # Unable to recover - re-raise original
                    raise

            else:
                # Check for "not modified" errors - TP/SL already set to requested value
                not_modified_indicators = [
                    "not modified",
                    '"retcode":34040',
                    'retcode":34040',
                ]
                if any(indicator in low for indicator in not_modified_indicators):
                    self.logger.info(
                        f"TP/SL for {symbol_id} already set to requested values; treating as success."
                    )
                    return {
                        "retCode": 0,
                        "retMsg": "OK",
                        "result": {},
                        "orderIds": [],
                    }

                # Check for validation errors where SL/TP is too close to current price
                # retCode 10001 with message like "StopLoss(0):10000 < 10_pcnt of base:118000"
                price_too_close_indicators = [
                    '"retcode":10001',
                    'retcode":10001',
                    '< 10_pcnt of base',
                    'stoploss(0):',
                    'takeprofit(0):',
                ]
                if any(indicator in low for indicator in price_too_close_indicators):
                    # Extract a cleaner error message
                    if "retmsg" in low:
                        # Try to extract the retMsg
                        import json
                        try:
                            error_data = json.loads(msg.split("bybit ")[-1])
                            error_msg = error_data.get("retMsg", "SL/TP too close to current price")
                        except Exception:
                            error_msg = "SL/TP too close to current price (minimum distance requirement)"
                    else:
                        error_msg = "SL/TP too close to current price (minimum distance requirement)"

                    self.logger.warning(
                        f"Cannot set TP/SL for {symbol_id}: {error_msg}. "
                        f"Bybit requires minimum 10% distance from current price."
                    )
                    # Return a success response to avoid crashing, but with empty orderIds
                    # The caller can check for empty orderIds to know no order was created
                    return {
                        "retCode": 0,
                        "retMsg": f"Skipped: {error_msg}",
                        "result": {},
                        "orderIds": [],
                        "warning": error_msg,
                    }

                # Unknown error — re-raise to let caller handle/log it
                raise

        # Fetch updated stop orders AFTER making the change
        new_order_ids = []
        existing_order_id = ""
        try:
            if get_orders_method:
                after_orders_resp = get_orders_method({
                    "category": category,
                    "symbol": symbol_id,
                    "orderFilter": "StopOrder",
                })
                if after_orders_resp.get("result", {}).get("list"):
                    after_orders = after_orders_resp["result"]["list"]
                    after_order_ids = {
                        order.get("orderId") 
                        for order in after_orders
                        if order.get("orderId")
                    }
                    # Find new order IDs that appeared after the call
                    new_order_ids = list(after_order_ids - before_order_ids)
                    
                    # If no new orders were created, an existing order was likely updated
                    # Find the stop loss order specifically (not take profit)
                    if not new_order_ids and after_order_ids:
                        # Look for the stop loss order by checking stopOrderType
                        for order in after_orders:
                            if order.get("stopOrderType") == "StopLoss" and order.get("orderId"):
                                existing_order_id = order["orderId"]
                                self.logger.debug(
                                    f"No new stop orders detected for {symbol_id} - existing StopLoss order updated. "
                                    f"Returning existing order ID: {existing_order_id}"
                                )
                                break
                        
                        # Fallback: if we still don't have an ID but there are orders, use first one
                        if not existing_order_id and after_order_ids:
                            existing_order_id = list(after_order_ids)[0]
                            self.logger.debug(
                                f"No StopLoss order found, using first available order ID: {existing_order_id}"
                            )
                    
                    # Update cache with the new state for subsequent calls
                    self._stop_order_cache[cache_key] = (time.time(), after_order_ids)
                    self.logger.debug(f"Updated stop order cache for {symbol_id} with {len(after_order_ids)} orders")
        except Exception as e:
            self.logger.warning(f"Failed to detect new stop orders after TPSL update: {e}")

        # Add the order IDs to the response for caller convenience
        resp["orderIds"] = new_order_ids
        resp["existingOrderId"] = existing_order_id
        
        return resp

    def fetch_order_by_symbol(
        self,
        trading_pair: TradingPair,
        since: Optional[int] = None,
        until: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[Order]:
        """Override fetch_order_by_symbol to handle Bybit UTA limitation.
        
        Bybit UTA accounts don't support the generic fetch_orders() method.
        This override combines fetchOpenOrders, fetchClosedOrders, and fetchCanceledOrders.
        """
        symbol = self._to_symbol(trading_pair)
        
        # Bybit UTA accounts don't support fetch_orders, so we need to combine multiple calls
        self.logger.debug(
            f"Fetching orders for {symbol} using fetchOpenOrders + fetchClosedOrders + fetchCanceledOrders"
        )
        
        all_orders = []
        
        # Fetch open orders
        try:
            open_orders = self.client.fetch_open_orders(
                symbol, 
                since=int(since) if since else None, 
                limit=limit
            )
            all_orders.extend(open_orders)
            self.logger.debug(f"Fetched {len(open_orders)} open orders for {symbol}")
        except Exception as open_err:
            self.logger.debug(f"Failed to fetch open orders for {symbol}: {open_err}")
        
        # Fetch closed orders
        try:
            closed_orders = self.client.fetch_closed_orders(
                symbol, 
                since=int(since) if since else None, 
                limit=limit
            )
            all_orders.extend(closed_orders)
            self.logger.debug(f"Fetched {len(closed_orders)} closed orders for {symbol}")
        except Exception as closed_err:
            self.logger.debug(f"Failed to fetch closed orders for {symbol}: {closed_err}")
        
        # Fetch canceled orders if available
        fetch_canceled = getattr(self.client, 'fetch_canceled_orders', None)
        if fetch_canceled:
            try:
                canceled_orders = fetch_canceled(
                    symbol, 
                    since=int(since) if since else None, 
                    limit=limit
                )
                all_orders.extend(canceled_orders)
                self.logger.debug(f"Fetched {len(canceled_orders)} canceled orders for {symbol}")
            except Exception as canceled_err:
                self.logger.debug(f"Failed to fetch canceled orders for {symbol}: {canceled_err}")
        
        # Apply until filter if provided
        if until:
            all_orders = [order for order in all_orders if (order.get("timestamp") or 0) <= int(until)]
        
        # Convert to Order objects
        return [self._convert_order(order) for order in all_orders]
