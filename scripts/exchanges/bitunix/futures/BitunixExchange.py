from decimal import ROUND_DOWN, Decimal
import hashlib
import json
import time
import traceback
import uuid
from dataclasses import asdict
from typing import Dict, List, Optional

import pandas as pd
import requests

from exchanges.bitunix.futures.error_codes import ErrorCode
from exchanges.types.exceptions import MarginModeMismatchError, PositionNotFoundError
from exchanges.bitunix.futures.types import (
    AdjustMarginRequest1,
    AdjustMarginRequest2,
    BitunixContract,
    BitunixHistoryTrade,
    BitunixMarginMode,
    BitunixOrderBook,
    BitunixOrderSide,
    BitunixTicker,
    BitunixTradingPair,
    AdjustMarginResponse,
    AssetQueryResponse,
    BatchOrderRequest,
    BatchOrderResponse,
    BatchOrderResult,
    BitunixAccount,
    BitunixHistoricPosition,
    BitunixOrder,
    BitunixPosition,
    CancelAllOrdersRequest,
    CancelAllOrdersResponse,
    CancelOrderRequest,
    CancelOrderResponse,
    CancelOrdersRequest,
    ChangeLeverageRequest,
    ChangeLeverageResponse,
    ChangeMarginModeRequest,
    ChangeMarginModeResponse,
    ChangePositionModeRequest,
    ChangePositionModeResponse,
    ClosePositionRequest,
    ClosePositionResponse,
    FlashClosePositionRequest,
    FlashClosePositionResponse,
    GetHistoryOrdersRequest,
    GetLeverageMarginModeResponse,
    BitunixFundingRate,
    GetOrderRequest,
    GetPendingOrdersRequest,
    GetPositionsRequest,
    GetTradesRequest,
    HistoryTpslOrder,
    ModifyOrderRequest,
    ModifyOrderResponse,
    OrderIdentifier,
    PlaceOrderRequest,
    PlaceOrderResponse,
    CancelTpslOrderRequest,
    TpslOrderIdResponse,
    PositionTpslOrderRequest,
    TpslPlaceOrderRequest,
    TpslModifyOrderRequest,
    TpslOrder,
    GetTpslOrdersRequest,
    TransferRequest,
    TransferResponse,
)
from exchanges.IFuturesExchange import IFuturesExchange
import exchanges.types.common as common
from exchanges.types.common import (
    Balance,
    ClientOid,
    CreateOrderResponse,
    Direction,
    FuturesMarket,
    HistoricPosition,
    LimitOrderRequest,
    MarketOrderRequest,
    OrderBookData,
    Order,
    OrderSide,
    Position,
    Status,
    StopMarketOrderRequest,
    Ticker,
    TimeFrame,
    TimestampMilliseconds,
    TradingPair,
    MarginMode,
    Fill,
)
from my_types.percentage import Percentage
from my_types.config_models import BitunixConfig
from utils.SqlManager import SQLiteManager
from utils.math import ms_to_nano


class _BitunixBase:
    """HTTP and authentication helper"""

    def __init__(self, config: BitunixConfig) -> None:
        self.api_key = config.bitunix_api_key
        self.api_secret = config.bitunix_api_secret
        self.base_url = "https://fapi.bitunix.com"
        self.session = requests.Session()
        self.testmode = config.testnet
        self.options = {'maxRetriesOnFailure': 3, 'maxRetriesOnFailureDelay': 1000}

    def _generate_nonce(self) -> str:
        """Generate a nonce for API requests using UUID as per official implementation."""

        return str(uuid.uuid4()).replace("-", "")

    def _generate_signature(
        self,
        nonce: str,
        timestamp: str,
        query_params: Optional[Dict] = None,
        body: Optional[str] = None,
    ) -> str:
        """Generate signature for Bitunix API requests according to their official implementation."""
        # Official implementation concatenates key-value pairs directly without URL encoding
        query_string = (
            "".join(f"{k}{v}" for k, v in sorted(query_params.items()))
            if query_params
            else ""
        )
        body = body if body else ""
        message = f"{nonce}{timestamp}{self.api_key}{query_string}{body}"

        # First SHA256 encryption
        digest = hashlib.sha256(message.encode()).hexdigest()

        # Second SHA256 encryption with secret
        sign = hashlib.sha256((digest + self.api_secret).encode()).hexdigest()

        return sign

    def _request(
        self,
        path: str,
        method: str = "GET",
        params: Optional[dict] = None,
        body: Optional[dict] = None,
    ) -> dict | list[dict]:
        if params is None:
            params = {}
        
        def make_request(request_params, request_body):
            url = f"{self.base_url}{path}"
            timestamp = str(int(time.time() * 1000))
            nonce = self._generate_nonce()

            if path.startswith("/api/v1/") and not path.startswith(
                "/api/v1/futures/public"
            ):
                # For authenticated requests
                headers = {
                    "api-key": self.api_key,
                    "timestamp": timestamp,
                    "nonce": nonce,
                }

                if method == "GET":
                    # For GET requests - pass query params to signature
                    sign = self._generate_signature(nonce, timestamp, query_params=request_params)
                    headers["sign"] = sign
                    resp = self.session.get(url, params=request_params, headers=headers, timeout=60)
                else:
                    # For POST/DELETE requests - JSON body with Content-Type
                    headers["Content-Type"] = "application/json"
                    body_str = (
                        json.dumps(request_body, separators=(",", ":"), sort_keys=True)
                        if request_body
                        else json.dumps(request_params, separators=(",", ":"), sort_keys=True)
                    )
                    sign = self._generate_signature(nonce, timestamp, body=body_str)
                    headers["sign"] = sign

                    if method == "POST":
                        resp = self.session.post(url, data=body_str, headers=headers, timeout=60)
                    else:  # DELETE
                        resp = self.session.delete(url, data=body_str, headers=headers, timeout=60)
            else:
                # Public endpoints - no authentication
                headers = {"Content-Type": "application/json"}
                if method == "GET":
                    resp = self.session.get(url, params=request_params, headers=headers, timeout=60)
                elif method == "POST":
                    resp = self.session.post(
                        url, json=request_body if request_body else request_params, headers=headers, timeout=60
                    )
                else:
                    resp = self.session.delete(url, json=request_body, headers=headers, timeout=60)

            return resp

        max_retries = self.options.get('maxRetriesOnFailure', 3)
        base_delay = self.options.get('maxRetriesOnFailureDelay', 1000) / 1000  # Convert to seconds

        for attempt in range(max_retries):
            try:
                resp = make_request(params, body)
                resp.raise_for_status()
                
                resp_json = resp.json()
                if "code" in resp_json and resp_json["code"] != ErrorCode.SUCCESS.code:
                    error_code = resp_json.get("code", "UNKNOWN_ERROR")
                    error_message = f"Unknown error code: {error_code} - {resp_json.get('msg', 'No message provided')}"
                    for error in ErrorCode:
                        if error.code == error_code:
                            error_message = error.message
                            break
                    
                    # Check if it's a retryable network error
                    if error_code == 1 or error_code == 10001 or "network" in error_message.lower():
                        if attempt < max_retries - 1:
                            # Use exponential backoff for network errors
                            delay = base_delay * (2 ** attempt)
                            print(f"Network error (code: {error_code}), retrying in {delay:.1f}s, attempt {attempt + 1} of {max_retries}")
                            time.sleep(delay)
                            continue
                    
                    if "margin mode" in error_message.lower():
                        raise MarginModeMismatchError(
                            f"API error for endpoint {path}: {error_message} (code: {error_code})"
                        )
                    raise Exception(
                        f"API error for endpoint {path}: {error_message} (code: {error_code})"
                    )
                data = resp_json.get("data", {})

                # Handle pagination if 'total' is present
                if isinstance(data, dict) and 'total' in data:
                    # Identify list field and page info
                    list_field = None
                    for k, v in data.items():
                        if isinstance(v, list):
                            list_field = k
                            break

                    if list_field:
                        all_items = data[list_field]
                        if all_items:
                            total = int(data.get('total', 0))
                            page_size = len(all_items)
                            seen_ids = set()
                            # Try to pick a stable unique key; fall back to (positionId/id, ctime, mtime) tuple
                            def item_key(x):
                                return x.get('id') or x.get('orderId') or x.get('positionId') or (x.get('symbol'), x.get('ctime'), x.get('mtime'))

                            for it in list(all_items):
                                seen_ids.add(item_key(it))

                            skip = 0
                            limit = params.get('limit', page_size) or page_size

                            # Helper to fetch next page by skip/limit
                            def fetch_by_skip(curr_skip):
                                page_params = params.copy()
                                page_params['skip'] = curr_skip
                                page_params['limit'] = limit
                                nxt = make_request(page_params, body)
                                nxt.raise_for_status()
                                return nxt.json()

                            # Helper to fetch next page by endTime keyset
                            def fetch_by_time(end_ts):
                                page_params = params.copy()
                                page_params['endTime'] = end_ts
                                page_params['limit'] = limit
                                # remove skip if present
                                page_params.pop('skip', None)
                                nxt = make_request(page_params, body)
                                nxt.raise_for_status()
                                return nxt.json()

                            # First try offset pagination; if it makes no progress, fall back to time-based
                            while len(all_items) < total:
                                skip += page_size
                                try:
                                    next_json = fetch_by_skip(skip)
                                    if next_json.get("code", 0) != 0:
                                        break
                                    next_data = next_json.get("data", {})
                                    next_list = next_data.get(list_field, [])
                                    if not next_list:
                                        break

                                    # Check progress
                                    before = len(seen_ids)
                                    for it in next_list:
                                        key = item_key(it)
                                        if key not in seen_ids:
                                            seen_ids.add(key)
                                            all_items.append(it)
                                    after = len(seen_ids)

                                    if after == before:
                                        # No progress → switch to keyset using last item's ctime
                                        last_ctime = int(all_items[-1].get('ctime', 0))
                                        if not last_ctime:
                                            break
                                        while len(all_items) < total:
                                            next_json = fetch_by_time(last_ctime - 1)
                                            if next_json.get("code", 0) != 0:
                                                break
                                            next_data = next_json.get("data", {})
                                            next_list = next_data.get(list_field, [])
                                            if not next_list:
                                                break
                                            progressed = False
                                            for it in next_list:
                                                key = item_key(it)
                                                if key not in seen_ids:
                                                    seen_ids.add(key)
                                                    all_items.append(it)
                                                    progressed = True
                                            if not progressed:
                                                break
                                            last_ctime = int(all_items[-1].get('ctime', 0)) or last_ctime - 1
                                        break
                                except Exception:
                                    break

                            data[list_field] = all_items
                return data

            except requests.exceptions.RequestException as e:
                # Log or print the error message
                print(f"Request failed: {e}, attempt {attempt + 1} of {max_retries}")
                print(f"""Request error occurred:
            URL: {self.base_url}{path}
            Method: {method}
            Params: {params}
            Body: {body}
            error: {str(e)}""")
                
                # If it's the last attempt, re-raise the exception
                if attempt == max_retries - 1 or (e.response is not None and e.response.status_code in [403]):
                    print(traceback.format_exc())
                    raise e
                # Use exponential backoff for request exceptions
                delay = base_delay * (2 ** attempt)
                print(f"Retrying in {delay:.1f}s...")
                time.sleep(delay)
        
        raise Exception("Request failed after all retries")

    def _map_side(self, side: BitunixOrderSide) -> common.OrderSide:
        if side == "BUY":
            return "buy"
        elif side == "SELL":
            return "sell"
        else:
            raise ValueError(f"Unknown order side: {side}")

    def _map_side_inverse(self, side: common.OrderSide) -> BitunixOrderSide:
        if side == "buy":
            return "BUY"
        elif side == "sell":
            return "SELL"
        else:
            raise ValueError(f"Unknown order side: {side}")

    def _map_margin_mode(self, margin_mode: BitunixMarginMode) -> common.MarginMode:
        if margin_mode == "ISOLATION":
            return "ISOLATED"
        elif margin_mode == "CROSS":
            return "CROSS"
        else:
            raise ValueError(f"Unknown margin mode: {margin_mode}")

    def _map_margin_mode_inverse(
        self, margin_mode: common.MarginMode
    ) -> BitunixMarginMode:
        if margin_mode == "ISOLATED":
            return "ISOLATION"
        elif margin_mode == "CROSS":
            return "CROSS"
        else:
            raise ValueError(f"Unknown margin mode: {margin_mode}")


class _BitunixPrivate(_BitunixBase):
    """Low level Bitunix futures private API"""

    def __init__(self, config: BitunixConfig) -> None:
        super().__init__(config)

    # helpers -----------------------------------------------------------------
    def _clean(self, data: dict) -> dict:
        return {k: v for k, v in data.items() if v is not None}

    # account -----------------------------------------------------------------
    def _adjust_position_margin(
        self, req: AdjustMarginRequest1 | AdjustMarginRequest2
    ) -> AdjustMarginResponse:
        if self.testmode:
            return AdjustMarginResponse(msg="Test mode: no action taken")
        data = self._request(
            "/api/v1/futures/account/adjust_position_margin",
            "POST",
            body=self._clean(asdict(req)),
        )
        return AdjustMarginResponse(msg=str(data))

    def _change_leverage(self, req: ChangeLeverageRequest) -> ChangeLeverageResponse:
        if self.testmode:
            return ChangeLeverageResponse(
                marginCoin=req.marginCoin,
                symbol=req.symbol,
                leverage=req.leverage,
            )
        data = self._request(
            "/api/v1/futures/account/change_leverage",
            "POST",
            body=self._clean(asdict(req)),
        )
        if isinstance(data, list):
            raise Exception("Unexpected response format: list instead of dict")
        return ChangeLeverageResponse(
            marginCoin=data.get("marginCoin", req.marginCoin),
            symbol=data.get("symbol", req.symbol),
            leverage=int(data.get("leverage", req.leverage)),
        )

    def _change_margin_mode(
        self, req: ChangeMarginModeRequest
    ) -> ChangeMarginModeResponse:
        if self.testmode:
            return ChangeMarginModeResponse(
                symbol=req.symbol,
                marginMode=req.marginMode,
                marginCoin=req.marginCoin,
            )
        data = self._request(
            "/api/v1/futures/account/change_margin_mode",
            "POST",
            body=self._clean(asdict(req)),
        )
        if isinstance(data, list):
            data = data[0]
        return ChangeMarginModeResponse(
            symbol=data.get("symbol", req.symbol),
            marginMode=data.get("marginMode", req.marginMode),
            marginCoin=data.get("marginCoin", req.marginCoin),
        )

    def _change_position_mode(
        self, req: ChangePositionModeRequest
    ) -> ChangePositionModeResponse:
        if self.testmode:
            return ChangePositionModeResponse(positionMode=req.positionMode)
        data = self._request(
            "/api/v1/futures/account/change_position_mode",
            "POST",
            body=self._clean(asdict(req)),
        )
        if isinstance(data, list):
            raise Exception("Unexpected response format: list instead of dict")
        return ChangePositionModeResponse(
            positionMode=data.get("positionMode", req.positionMode)
        )

    def _get_leverage_margin_mode(
        self, symbol: str, marginCoin: str
    ) -> GetLeverageMarginModeResponse:
        params = {"symbol": symbol, "marginCoin": marginCoin}
        data = self._request(
            "/api/v1/futures/account/get_leverage_margin_mode",
            "GET",
            params=params,
        )
        if isinstance(data, list):
            raise Exception("Unexpected response format: list instead of dict")
        return GetLeverageMarginModeResponse(
            symbol=data["symbol"],
            marginCoin=data["marginCoin"],
            leverage=int(data["leverage"]),
            marginMode=data["marginMode"],
        )

    def _get_account(self, marginCoin: str) -> BitunixAccount:
        data = self._request(
            "/api/v1/futures/account", "GET", params={"marginCoin": marginCoin}
        )
        if isinstance(data, list):
            raise Exception("Unexpected response format: list instead of dict")
        return BitunixAccount(**data)

    def _asset_query(self) -> AssetQueryResponse:
        data = self._request("/api/v1/cp/asset/query", "GET")
        if isinstance(data, list):
            raise Exception("Unexpected response format: list instead of dict")
        return AssetQueryResponse(
            available=data["available"], maxTransfer=data["maxTransfer"]
        )

    def _transfer_to_sub_account(self, req: TransferRequest) -> TransferResponse:
        if self.testmode:
            return TransferResponse(success=True)
        self._request(
            "/api/v1/cp/asset/transfer-to-sub-account",
            "POST",
            body=self._clean(asdict(req)),
        )
        return TransferResponse(success=True)

    def _transfer_to_main_account(self, req: TransferRequest) -> TransferResponse:
        if self.testmode:
            return TransferResponse(success=True)
        self._request(
            "/api/v1/cp/asset/transfer-to-main-account",
            "POST",
            body=self._clean(asdict(req)),
        )
        return TransferResponse(success=True)

    # market -----------------------------------------------------------------
    def _get_funding_rate(self, symbol: str) -> BitunixFundingRate:
        data = self._request(
            "/api/v1/futures/market/funding_rate",
            params={"symbol": symbol},
        )
        if isinstance(data, list):
            data = data[0]
        return BitunixFundingRate(**data)

    def _get_funding_rate_batch(
        self, symbols: Optional[List[str]] = None
    ) -> List[BitunixFundingRate]:
        params = {"symbols": ",".join(symbols)} if symbols else {}
        data = self._request(
            "/api/v1/futures/market/funding_rate/batch",
            params=params,
        )
        # API returns array directly: [{'symbol': 'CGPTUSDT', 'markPrice': '0.09753', 'lastPrice': '0.09751', 'fundingRate': '0.005'}]
        if isinstance(data, dict):
            # Fallback for wrapped response format
            data = data.get("fundingRateList", data.get("data", []))
        return [BitunixFundingRate(**d) for d in data]

    # orders ------------------------------------------------------------------
    def _place_order(self, req: PlaceOrderRequest) -> PlaceOrderResponse:
        if self.testmode:
            return PlaceOrderResponse(
                orderId="test_order_id", clientId="test_client_oid"
            )
        data = self._request(
            "/api/v1/futures/trade/place_order", "POST", body=self._clean(asdict(req))
        )
        if isinstance(data, list):
            raise Exception("Unexpected response format: list instead of dict")
        return PlaceOrderResponse(orderId=data["orderId"], clientId=data["clientId"])

    def _batch_order(self, req: BatchOrderRequest) -> BatchOrderResponse:
        if len(req.orderList) > 20:
            raise ValueError("batch_order accepts at most 20 orders")
        if self.testmode:
            return BatchOrderResponse(
                successList=[
                    BatchOrderResult(id="test_order_id", clientId="test_client_oid")
                ],
                failureList=[],
            )
        payload = self._clean(asdict(req))
        payload["orderList"] = [self._clean(asdict(o)) for o in req.orderList]
        data = self._request("/api/v1/futures/trade/batch_order", "POST", body=payload)
        if isinstance(data, list):
            raise Exception("Unexpected response format: list instead of dict")
        success = [BatchOrderResult(**o) for o in data.get("successList", [])]
        failure = [BatchOrderResult(**o) for o in data.get("failureList", [])]
        return BatchOrderResponse(successList=success, failureList=failure)

    def _cancel_all_orders(
        self, req: CancelAllOrdersRequest
    ) -> CancelAllOrdersResponse:
        if self.testmode:
            return CancelAllOrdersResponse(
                successList=[
                    OrderIdentifier(orderId="test_order_id", clientId="test_client_oid")
                ],
                failureList=[],
            )
        data = self._request(
            "/api/v1/futures/trade/cancel_all_orders",
            "POST",
            body=self._clean(asdict(req)),
        )
        if isinstance(data, list):
            raise Exception("Unexpected response format: list instead of dict")
        success = [OrderIdentifier(**o) for o in data.get("successList", [])]
        failure = [OrderIdentifier(**o) for o in data.get("failureList", [])]
        return CancelAllOrdersResponse(successList=success, failureList=failure)

    def _cancel_orders(self, req: CancelOrdersRequest) -> CancelAllOrdersResponse:
        if self.testmode:
            return CancelAllOrdersResponse(
                successList=[
                    OrderIdentifier(orderId="test_order_id", clientId="test_client_oid")
                ],
                failureList=[],
            )
        payload = self._clean(asdict(req))
        payload["orderList"] = [self._clean(asdict(o)) for o in req.orderList]
        data = self._request(
            "/api/v1/futures/trade/cancel_orders", "POST", body=payload
        )
        if isinstance(data, list):
            raise Exception("Unexpected response format: list instead of dict")
        success = [OrderIdentifier(**o) for o in data.get("successList", [])]
        failure = [OrderIdentifier(**o) for o in data.get("failureList", [])]
        return CancelAllOrdersResponse(successList=success, failureList=failure)

    def _cancel_order(self, req: CancelOrderRequest) -> CancelOrderResponse:
        if self.testmode:
            return CancelOrderResponse(orderId=req.orderId, status="cancelled")
        data = self._request(
            "/api/v1/futures/trade/cancel_orders",
            "POST",
            body={"orderList": [asdict(req)]},
        )
        if isinstance(data, list):
            raise Exception("Unexpected response format: list instead of dict")
        res = data.get("successList", [{}])[0]
        return CancelOrderResponse(orderId=res.get("orderId", req.orderId), status="")

    def _modify_order(self, req: ModifyOrderRequest) -> ModifyOrderResponse:
        if self.testmode:
            return ModifyOrderResponse(
                orderId=req.orderId or "test-id", status="modified"
            )
        data = self._request(
            "/api/v1/futures/trade/modify_order", "POST", body=self._clean(asdict(req))
        )
        if isinstance(data, list):
            raise Exception("Unexpected response format: list instead of dict")
        return ModifyOrderResponse(
            orderId=data["orderId"], status=data.get("status", "")
        )

    def _get_order_detail(self, req: GetOrderRequest) -> BitunixOrder:
        data = self._request(
            "/api/v1/futures/trade/get_order_detail",
            "GET",
            params=self._clean(asdict(req)),
        )
        if isinstance(data, list):
            raise Exception("Unexpected response format: list instead of dict")
        return BitunixOrder(**data)

    def _get_history_orders(self, req: GetHistoryOrdersRequest) -> List[BitunixOrder]:
        data = self._request(
            "/api/v1/futures/trade/get_history_orders",
            "GET",
            params=self._clean(asdict(req)),
        )
        if isinstance(data, list):
            raise Exception("Unexpected response format: list instead of dict")
        return [BitunixOrder(**o) for o in data.get("orderList", [])]

    def _get_pending_orders(self, req: GetPendingOrdersRequest) -> List[BitunixOrder]:
        data = self._request(
            "/api/v1/futures/trade/get_pending_orders",
            "GET",
            params=self._clean(asdict(req)),
        )
        if isinstance(data, list):
            raise Exception("Unexpected response format: list instead of dict")
        return [BitunixOrder(**o) for o in data.get("orderList", [])]

    def _get_history_trades(self, req: GetTradesRequest) -> List[BitunixHistoryTrade]:
        data = self._request(
            "/api/v1/futures/trade/get_history_trades",
            "GET",
            params=self._clean(asdict(req)),
        )
        if isinstance(data, list):
            raise Exception("Unexpected response format: list instead of dict")
        return [BitunixHistoryTrade(**t) for t in data.get("tradeList", [])]

    # positions ----------------------------------------------------------------
    def _get_pending_positions(self, req: GetPositionsRequest) -> List[BitunixPosition]:
        data = self._request(
            "/api/v1/futures/position/get_pending_positions",
            "GET",
            params=self._clean(asdict(req)),
        )
        # Endpoint returns a list of positions directly
        if isinstance(data, dict):
            data = data.get("positionList", [])
        return [BitunixPosition(**p) for p in data]

    def _get_history_positions(
        self, req: GetPositionsRequest
    ) -> List[BitunixHistoricPosition]:
        data = self._request(
            "/api/v1/futures/position/get_history_positions",
            "GET",
            params=self._clean(asdict(req)),
        )
        # API may return list directly or dict with positionList
        if isinstance(data, list):
            positions = data
        else:
            positions = data.get("positionList", [])
        return [BitunixHistoricPosition(**p) for p in positions]

    def _get_position_tiers(self, symbol: str) -> List[dict]:
        data = self._request(
            "/api/v1/futures/position/get_position_tiers",
            "GET",
            params={"symbol": symbol},
        )
        if isinstance(data, dict):
            raise Exception("Unexpected response format: dict instead of list")
        return data

    def _close_all_position(self, req: CancelAllOrdersRequest) -> None:
        if self.testmode:
            return
        self._request(
            "/api/v1/futures/trade/close_all_position",
            "POST",
            body=self._clean(asdict(req)),
        )

    def _flash_close_position(
        self, req: FlashClosePositionRequest
    ) -> FlashClosePositionResponse:
        if self.testmode:
            return FlashClosePositionResponse(positionId="test_position_id")
        data = self._request(
            "/api/v1/futures/trade/flash_close_position",
            "POST",
            body=self._clean(asdict(req)),
        )
        if isinstance(data, list):
            raise Exception("Unexpected response format: list instead of dict")
        return FlashClosePositionResponse(positionId=data["positionId"])

    # tpsl --------------------------------------------------------------------
    def _cancel_tpsl_order(self, req: CancelTpslOrderRequest) -> TpslOrderIdResponse:
        if self.testmode:
            return TpslOrderIdResponse(orderId=req.orderId)
        data = self._request(
            "/api/v1/futures/tpsl/cancel_order",
            "POST",
            body=self._clean(asdict(req)),
        )
        if isinstance(data, list):
            data = data[0] if data else {}
        return TpslOrderIdResponse(orderId=data.get("orderId", req.orderId))

    def _get_history_tpsl_orders(
        self, req: GetTpslOrdersRequest
    ) -> List[HistoryTpslOrder]:
        data = self._request(
            "/api/v1/futures/tpsl/get_history_orders",
            "GET",
            params=self._clean(asdict(req)),
        )
        if isinstance(data, dict):
            orders = data.get("orderList", data.get("data", []))
        else:
            orders = data
        return [HistoryTpslOrder(**o) for o in orders]

    def _get_pending_tpsl_orders(self, req: GetTpslOrdersRequest) -> List[TpslOrder]:
        data = self._request(
            "/api/v1/futures/tpsl/get_pending_orders",
            "GET",
            params=self._clean(asdict(req)),
        )
        if isinstance(data, dict):
            orders = data.get("orderList", data.get("data", []))
        else:
            orders = data
        return [TpslOrder(**o) for o in orders]

    def _modify_position_tpsl_order(
        self, req: PositionTpslOrderRequest
    ) -> TpslOrderIdResponse:
        if self.testmode:
            return TpslOrderIdResponse(orderId="test-id")
        data = self._request(
            "/api/v1/futures/tpsl/position/modify_order",
            "POST",
            body=self._clean(asdict(req)),
        )
        if isinstance(data, list):
            data = data[0] if data else {}
        return TpslOrderIdResponse(orderId=data.get("orderId", ""))

    def _place_position_tpsl_order(
        self, req: PositionTpslOrderRequest
    ) -> TpslOrderIdResponse:
        if self.testmode:
            return TpslOrderIdResponse(orderId="test-id")
        data = self._request(
            "/api/v1/futures/tpsl/position/place_order",
            "POST",
            body=self._clean(asdict(req)),
        )
        if isinstance(data, list):
            data = data[0] if data else {}
        return TpslOrderIdResponse(orderId=data.get("orderId", ""))

    def _place_tpsl_order(self, req: TpslPlaceOrderRequest) -> TpslOrderIdResponse:
        if self.testmode:
            return TpslOrderIdResponse(orderId="test-id")
        data = self._request(
            "/api/v1/futures/tpsl/place_order",
            "POST",
            body=self._clean(asdict(req)),
        )
        if isinstance(data, list):
            data = data[0] if data else {}
        return TpslOrderIdResponse(orderId=data.get("orderId", ""))

    def _modify_tpsl_order(self, req: TpslModifyOrderRequest) -> TpslOrderIdResponse:
        if self.testmode:
            return TpslOrderIdResponse(orderId="test-id")
        data = self._request(
            "/api/v1/futures/tpsl/modify_order",
            "POST",
            body=self._clean(asdict(req)),
        )
        if isinstance(data, list):
            data = data[0] if data else {}
        return TpslOrderIdResponse(
            orderId=data.get("orderId", req.orderId if hasattr(req, "orderId") else "")
        )


class BitunixFuturesExchange(_BitunixPrivate, IFuturesExchange):
    """Public interface implementation using the Bitunix private API"""

    def get_name(self) -> str:
        """Return the name of the exchange."""
        return "Bitunix Futures"

    def __init__(self, config: BitunixConfig, db_manager: SQLiteManager) -> None:
        super().__init__(config)
        self.db_manager = db_manager
        self._markets_raw: Dict[TradingPair, BitunixTradingPair] = {}
        self._markets: Dict[TradingPair, FuturesMarket] = {}
        self.load_markets()

    # ------------------------------------------------------------------
    # market metadata
    # ------------------------------------------------------------------
    def load_markets(self) -> None:
        data = self._request("/api/v1/futures/market/trading_pairs")
        ticker_data = self._request("/api/v1/futures/market/tickers")

        # Create a lookup dictionary for ticker data by symbol
        tickers_by_symbol = {}
        if isinstance(ticker_data, list):
            for ticker in ticker_data:
                tickers_by_symbol[ticker.get("symbol")] = ticker

        self._markets_raw = {}
        self._markets = {}
        for item in data:
            pair = BitunixTradingPair(**item)
            tp = TradingPair(
                f"{pair.symbol.split(pair.quote)[0]}/{pair.quote}:{pair.quote}"
            )
            self._markets_raw[tp] = pair

            # Get ticker data for this symbol
            ticker = tickers_by_symbol.get(item.get("symbol"), {})

            # Extract market data from ticker
            mark_price = float(ticker.get("markPrice", 0.0))
            daily_high = float(ticker.get("high", 0.0))
            daily_low = float(ticker.get("low", 0.0))
            daily_volume = float(ticker.get("baseVol", 0.0))
            daily_turnover = float(ticker.get("quoteVol", 0.0))
            last_price = float(ticker.get("lastPrice", 0.0))
            open_price = float(ticker.get("open", 0.0))

            # Calculate daily change and change rate
            if open_price > 0:
                daily_change = last_price - open_price
                daily_change_rate = Percentage((daily_change / open_price) * 100)
            else:
                daily_change = 0.0
                daily_change_rate = Percentage(0)

            self._markets[tp] = FuturesMarket(
                markPrice=mark_price,
                maxLeverage=int(item.get("maxLeverage", 1)),
                lot_size=float(self._lots_to_qty_precise(1, tp)),
                daily_high=daily_high,
                daily_low=daily_low,
                daily_volume=daily_volume,
                daily_turnover=daily_turnover,
                daily_change=daily_change,
                daily_change_rate=daily_change_rate,
                open_interest=0.0,  # Not provided in ticker endpoint
                takerFeeRate=0.0006,
            )

    def market(self, trading_pair: TradingPair) -> FuturesMarket:
        if not self._markets or trading_pair not in self._markets:
            self.load_markets()
        return self._markets[trading_pair]

    # ------------------------------------------------------------------
    # public market data
    # ------------------------------------------------------------------
    def fetch_order_book(
        self, trading_pair: TradingPair, depth: int | str = 20
    ) -> OrderBookData:
        symbol = self.get_symbol_id(trading_pair)
        allowed_numbers = [1, 5, 15, 50]
        if isinstance(depth, int) and depth not in allowed_numbers:
            closest_depth = min(allowed_numbers, key=lambda x: abs(x - depth))  # type: ignore
            depth = closest_depth
        elif isinstance(depth, str) and depth != "max":
            depth = "max"
        data = self._request(
            "/api/v1/futures/market/depth",
            params={"symbol": symbol, "limit": depth},
        )
        if isinstance(data, list):
            raise Exception("Unexpected response format: list instead of dict")
        book = BitunixOrderBook(
            symbol=symbol,
            bids=data.get("bids", []),
            asks=data.get("asks", []),
            timestamp=int(time.time() * 1000),
        )
        return OrderBookData(
            trading_pair=trading_pair,
            sequence=0,
            asks=[[a[0], a[1]] for a in book.asks],
            bids=[[b[0], b[1]] for b in book.bids],
            ts=common.TimestampNanoseconds(book.timestamp * 1_000_000),
        )

    def fetch_ticker(self, trading_pair: TradingPair) -> Ticker:
        symbol = self.get_symbol_id(trading_pair)
        data = self._request(
            "/api/v1/futures/market/tickers", params={"symbols": symbol}
        )
        if isinstance(data, list):
            if not data:
                raise ValueError(
                    f"No ticker data found for trading pair {trading_pair}"
                )
            data = data[0]
        tick = BitunixTicker(**data)
        return Ticker(
            last_price=float(tick.lastPrice), mark_price=float(tick.lastPrice)
        )

    def fetch_ohlcv(
        self,
        trading_pair: TradingPair,
        timeframe: TimeFrame,
        since: Optional[TimestampMilliseconds] = None,
        until: Optional[TimestampMilliseconds] = None,
    ) -> pd.DataFrame:
        symbol = self.get_symbol_id(trading_pair)
        params: dict[str, str | TimestampMilliseconds | int] = {
            "symbol": symbol,
            "interval": timeframe.value.name,
            "limit": 200,
        }
        if since is not None:
            params["startTime"] = since
        if until is not None:
            params["endTime"] = until
        data = self._request("/api/v1/futures/market/kline", params=params)

        # Convert Bitunix OHLCV data to standardized DataFrame format
        # Bitunix format: [{"open":60000,"high":60001,"close":60000,"low":59989.2,"time":111111,"quoteVol":"1","baseVol":"60000","type":"LAST_PRICE"}]
        df_data = []
        for candle in data:
            df_data.append(
                {
                    "timestamp": candle["time"],
                    "open": float(candle["open"]),
                    "high": float(candle["high"]),
                    "low": float(candle["low"]),
                    "close": float(candle["close"]),
                    "volume": float(candle["baseVol"]),
                }
            )

        df = pd.DataFrame(df_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.sort_values(by="timestamp", inplace=True)

        return df

    # ------------------------------------------------------------------
    # trading endpoints
    # ------------------------------------------------------------------
    def create_limit_order(
        self, params: LimitOrderRequest, test: bool = False
    ) -> CreateOrderResponse:
        symbol = self.get_symbol_id(params.trading_pair)
        # Convert from lot size (contracts) to actual quantity expected by Bitunix
        qty_str = self._lots_to_qty_precise(params.size, params.trading_pair)

        trade_side = "CLOSE" if params.reduceOnly else "OPEN"

        req = PlaceOrderRequest(
            symbol=symbol,
            qty=qty_str,
            side=self._map_side_inverse(params.side),
            tradeSide=trade_side,
            orderType="LIMIT",
            price=str(params.price),
            clientId=str(params.clientOid),
            reduceOnly=params.reduceOnly,
        )
        res = self._place_order(req)
        return CreateOrderResponse(orderId=res.orderId)

    def create_market_order(
        self,
        params: MarketOrderRequest | StopMarketOrderRequest,
        test: bool = False,
    ) -> CreateOrderResponse:
        symbol = self.get_symbol_id(params.trading_pair)
        is_tp = isinstance(params, StopMarketOrderRequest) and (
            (params.stop == "up" and params.order_side == "sell")
            or (params.stop == "down" and params.order_side == "buy")
        )
        is_sl = isinstance(params, StopMarketOrderRequest) and (
            (params.stop == "down" and params.order_side == "sell")
            or (params.stop == "up" and params.order_side == "buy")
        )
        stop_type = (
            params.stopPriceType if isinstance(params, StopMarketOrderRequest) else None
        )
        if stop_type:
            if stop_type == "TP":
                stop_type = "LAST_PRICE"
            else:
                raise ValueError(
                    f"Unsupported stop price type: {stop_type}. Only 'TP' is supported."
                )

        qty_str = self._lots_to_qty_precise(params.amount_lots, params.trading_pair)
        if float(qty_str) <= float(self._markets_raw[params.trading_pair].minTradeVolume):
            raise ValueError(f"Quantity {qty_str} is below the minimum trade volume.")
        trade_side = "CLOSE" if params.reduceOnly else "OPEN"

        req = PlaceOrderRequest(
            symbol=symbol,
            qty=qty_str,
            side=self._map_side_inverse(params.order_side),
            tradeSide=trade_side,
            orderType="MARKET",
            clientId=str(params.clientOid),
            reduceOnly=params.reduceOnly,
            tpPrice=str(params.stopPrice)
            if isinstance(params, StopMarketOrderRequest) and is_tp
            else None,
            tpStopType=stop_type if is_tp else None,
            slPrice=str(params.stopPrice)
            if isinstance(params, StopMarketOrderRequest) and is_sl
            else None,
            slStopType=stop_type if is_sl else None,
        )
        res = self._place_order(req)
        return CreateOrderResponse(orderId=res.orderId)

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
        """Create a take profit order using Bitunix TPSL API."""
        if test or self.testmode:
            order_id = "test_tp_order_id"
            return CreateOrderResponse(orderId=order_id)

        symbol = self.get_symbol_id(trading_pair)

        # Fetch the position to get the position ID
        # For position_direction "long", we need to find the buy-side position
        # For position_direction "short", we need to find the sell-side position
        position_side = "buy" if position_direction == "long" else "sell"
        try:
            position = self.fetch_position(trading_pair, side=position_side)
            position_id = position.id
        except ValueError:
            raise ValueError(
                f"No open {position_direction} position found for {trading_pair}. Cannot create take profit order."
            )

        # Convert size from lots to actual quantity
        qty_str = self._lots_to_qty_precise(lots, trading_pair)

        # For TP orders, we use the TPSL API
        req = TpslPlaceOrderRequest(
            symbol=symbol,
            positionId=position_id,  # Use the actual position ID
            tpPrice=str(price),
            tpStopType="LAST_PRICE",  # Use last price as trigger
            tpOrderType="MARKET",  # Market order when triggered
            tpQty=qty_str,
        )

        res = self._place_tpsl_order(req)
        return CreateOrderResponse(orderId=res.orderId)

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
        """Create a stop loss order using Bitunix TPSL API."""
        if test or self.testmode:
            order_id = "test_sl_order_id"
            return CreateOrderResponse(orderId=order_id)

        symbol = self.get_symbol_id(trading_pair)

        # Fetch the position to get the position ID
        # For position_direction "long", we need to find the buy-side position
        # For position_direction "short", we need to find the sell-side position
        position_side = "buy" if position_direction == "long" else "sell"
        try:
            position = self.fetch_position(trading_pair, side=position_side)
            position_id = position.id
        except ValueError:
            raise ValueError(
                f"No open {position_direction} position found for {trading_pair}. Cannot create stop loss order."
            )

        # Convert size from lots to actual quantity
        qty_str = self._lots_to_qty_precise(lots, trading_pair)

        # For SL orders, we use the TPSL API
        req = TpslPlaceOrderRequest(
            symbol=symbol,
            positionId=position_id,  # Use the actual position ID
            slPrice=str(price),
            slStopType="LAST_PRICE",  # Use last price as trigger
            slOrderType="MARKET",  # Market order when triggered
            slQty=qty_str,
        )

        res = self._place_tpsl_order(req)
        return CreateOrderResponse(orderId=res.orderId)

    def _get_step(self, trading_pair: TradingPair) -> Decimal:
        """Get the step size for the given trading pair."""
        try:
            market_raw = self._markets_raw[trading_pair]
        except KeyError:
            self.load_markets()
            market_raw = self._markets_raw.get(trading_pair)
            if not market_raw:
                raise ValueError(f"Trading pair {trading_pair} not found in markets.")
        base_precision = market_raw.basePrecision
        return Decimal(1).scaleb(-base_precision)

    def _qty_to_lots(self, trading_pair: TradingPair, qty: float) -> int:
        step = self._get_step(trading_pair)
        return int((Decimal(str(qty)) / step).to_integral_value(rounding=ROUND_DOWN))

    def _lots_to_qty_precise(self, lots: int, trading_pair: TradingPair) -> str:
        """
        Convert lots to quantity with precise rounding to avoid floating point precision issues.
        
        Args:
            lots: Number of lots
            trading_pair: The trading pair for which to convert lots

        Returns:
            str: Precisely calculated quantity as string
        """
        from decimal import Decimal

        step = self._get_step(trading_pair)

        # Compute qty in step units and quantize to exactly base_precision decimals
        qty = (Decimal(lots) * step).quantize(step, rounding=ROUND_DOWN)

        # Return plain string (no exponent), preserving the allowed decimals
        return format(qty, 'f')

    # ------------------------------------------------------------------
    # positions
    # ------------------------------------------------------------------
    def fetch_positions(self) -> List[Position]:
        def liquidation_price(p0: float, leverage: float, direction: Direction,
                            target_pct: float = 1000.0) -> float:
            """
            direction: 'long' or 'short'
            Returns None if the target loss can't be reached with the given leverage.
            """
            loss_multiple = target_pct / 100
            if loss_multiple > leverage:
                raise ValueError(
                    f"Target loss percentage {target_pct}% exceeds leverage {leverage}."
                )

            if direction.lower() == 'long':
                return p0 * (1 - loss_multiple / leverage)
            elif direction.lower() == 'short':
                return p0 * (1 + loss_multiple / leverage)
            else:
                raise ValueError("direction must be 'long' or 'short'")

        raw = self._get_pending_positions(GetPositionsRequest())
        positions: List[Position] = []
        for p in raw:
            pair = self.get_standardized_symbol(p.symbol)
            direction = "long" if p.side.lower() == "buy" else "short"
            mark_price = (
                self.fetch_ticker(pair).last_price if pair in self._markets else 0.0
            )
            leverage = int(p.leverage) if p.leverage else 1
            target_pct = leverage * 50  # Target loss percentage for liquidation price calculation
            minLiquidationPrice = liquidation_price(mark_price, leverage, direction, target_pct)
            pos_liquidation = float(p.liqPrice) if p.liqPrice else minLiquidationPrice
            final_liquidation_price = max(
                minLiquidationPrice, pos_liquidation) if direction=="long" else min(minLiquidationPrice, pos_liquidation)
            positions.append(
                Position(
                    avgEntryPrice=float(p.avgOpenPrice),
                    currentLots=self._qty_to_lots(pair, float(p.qty)),
                    currentQty=float(p.qty),
                    id=p.positionId,
                    isOpen=True,
                    leverage=int(p.leverage),
                    liquidationPrice=final_liquidation_price,
                    marginMode=self._map_margin_mode(p.marginMode),
                    markPrice=mark_price,
                    openingTimestamp=common.TimestampNanoseconds(
                        int(p.ctime)
                    ).to_milliseconds()
                    if common.TimestampNanoseconds.is_valid(int(p.ctime))
                    else common.TimestampMilliseconds(int(p.ctime)),
                    posCost=float(p.entryValue),
                    posInit=float(p.margin) / p.leverage,  # Approximation
                    direction=direction,  # type: ignore
                    realisedPnl=float(p.realizedPNL),
                    trading_pair=pair,
                    unrealisedPnl=float(p.unrealizedPNL),
                    unrealisedPnlPcnt=float(p.unrealizedPNL) / float(p.entryValue)
                    if p.entryValue
                    else 0.0,
                    unrealisedRoePcnt=float(p.unrealizedPNL)
                    / (float(p.margin) / p.leverage)
                    if p.margin and p.leverage
                    else 0.0,
                )
            )
        return positions

    def fetch_positions_history(
        self, since: Optional[TimestampMilliseconds] = None,
        trading_pair: Optional[TradingPair] = None
        ) -> List[HistoricPosition]:
        """Fetch historical positions."""
        req = GetPositionsRequest(startTime=since, symbol=self.get_symbol_id(trading_pair) if trading_pair else None)
        raw = self._get_history_positions(req)
        historic_positions: List[HistoricPosition] = []

        for p in raw:
            pair = self.get_standardized_symbol(p.symbol)

            # Map BitunixHistoricPosition to HistoricPosition
            # Note: Some fields like closeId, userId are not available in Bitunix API
            # We'll use reasonable defaults or derived values

            # Convert side from 'BUY'/'SELL' to 'long'/'short'
            position_type = "long" if p.side.upper() == "BUY" else "short"

            # Handle timestamps as strings
            open_time = common.TimestampMilliseconds(int(p.ctime))
            close_time = common.TimestampMilliseconds(int(p.mtime))
            filled_lots = self._qty_to_lots(pair, float(p.maxQty))

            historic_position = HistoricPosition(
                closeId=p.positionId,  # Use positionId as closeId since no separate closeId available
                userId="",  # Not available in Bitunix API
                trading_pair=pair,
                settleCurrency="USDT",  # Historic positions marginCoin can be None, default to USDT
                leverage=p.leverage,  # leverage is already a string
                type=position_type,  # Convert 'BUY'/'SELL' to 'long'/'short'
                pnl=p.realizedPNL,
                realisedGrossCost=p.maxQty,  # Use maxQty as a proxy for realized gross cost
                tradeFee=p.fee,
                fundingFee=p.funding,
                openTime=open_time,  # ctime is now string, convert to int first
                closeTime=close_time,  # mtime is now string, convert to int first
                openPrice=p.entryPrice,  # Use entryPrice from historic position
                closePrice=p.closePrice,  # Use closePrice from historic position
                marginMode=p.marginMode.lower(),  # Convert to lowercase
                maxFilledLots= filled_lots,  # Use filled lots calculated from maxQty
            )
            historic_positions.append(historic_position)

        return historic_positions

    # ------------------------------------------------------------------
    # misc
    # ------------------------------------------------------------------
    def adjust_price(self, price: float, trading_pair: TradingPair) -> float:
        market = self._markets_raw[trading_pair]
        tick = 10 ** (-market.quotePrecision)
        return round(round(price / tick) * tick, market.quotePrecision)

    def adjust_price_as_string(self, price: float, trading_pair: TradingPair) -> str:
        market = self._markets_raw[trading_pair]
        tick = 10 ** (-market.quotePrecision)
        adjusted_price = round(round(price / tick) * tick, market.quotePrecision)
        return f"{adjusted_price:.{market.quotePrecision}f}"

    def change_auto_deposit_status(
        self, trading_pair: TradingPair, status: bool
    ) -> bool:
        # not supported
        return False

    def change_cross_leverage(self, trading_pair: TradingPair, leverage: float) -> bool:
        symbol = self.get_symbol_id(trading_pair)
        self._change_leverage(
            ChangeLeverageRequest(
                symbol=symbol, leverage=int(leverage), marginCoin="USDT"
            )
        )
        return True

    def change_margin_mode(
        self, trading_pair: TradingPair, margin_mode: MarginMode
    ) -> bool:
        symbol = self.get_symbol_id(trading_pair)
        self._change_margin_mode(
            ChangeMarginModeRequest(
                symbol=symbol,
                marginMode=self._map_margin_mode_inverse(margin_mode),
                marginCoin="USDT",
            )
        )
        return True

    def cancel_order(self, order_id: str, trading_pair: Optional[TradingPair] = None) -> List[str]:
        res = self._cancel_order(CancelOrderRequest(orderId=order_id))
        return [res.orderId]

    def close_position(
        self,
        trading_pair: TradingPair,
        margin_mode: MarginMode,
        test: bool = False,
        channel: str = "MC",
        clientOid: Optional[ClientOid] = None,
    ) -> CreateOrderResponse:
        symbol = self.get_symbol_id(trading_pair)
        req = CancelAllOrdersRequest(symbol)
        if not test:
            self._close_all_position(req)
        return CreateOrderResponse(orderId="")

    def fetch_balance(self) -> Balance:
        acct = self._get_account("USDT")
        return Balance(
            accountEquity=float(acct.available)
            + float(acct.frozen)
            + float(acct.crossUnrealizedPNL)
            + float(acct.margin),
            unrealisedPNL=float(acct.crossUnrealizedPNL),
            marginBalance=float(acct.margin) + float(acct.frozen),
            positionMargin=float(acct.margin),
            orderMargin=float(acct.frozen),
            frozenFunds=float(acct.frozen),
            availableBalance=float(acct.available),
            currency=acct.marginCoin,
        )

    def fetch_closed_orders(
        self,
        trading_pair: Optional[TradingPair] = None,
        since: Optional[TimestampMilliseconds] = None,
        limit: Optional[int] = None,
        side: Optional[OrderSide] = None,
    ) -> List[Order]:
        symbol = self.get_symbol_id(trading_pair) if trading_pair else None
        req = GetHistoryOrdersRequest(
            symbol=symbol,
            startTime=since,
            limit=limit,
        )
        raw = self._get_history_orders(req)
        orders = [self._map_order(o) for o in raw]
        if side:
            orders = [o for o in orders if o.side == side]
        return orders

    def fetch_order_by_coid(self, coid: ClientOid) -> Order:
        req = GetOrderRequest(clientId=str(coid))
        try:
            o = self._get_order_detail(req)
            return self._map_order(o)
        except Exception as e:
            open_stop_orders = self.fetch_untriggered_stop_orders()
            for order in open_stop_orders:
                if order.clientOid == coid:
                    return order
            raise e

    def fetch_order_by_id(self, order_id: str) -> Order:
        req = GetOrderRequest(orderId=order_id)
        try:
            o = self._get_order_detail(req)
            return self._map_order(o)
        except Exception as e:
            open_stop_orders = self.fetch_untriggered_stop_orders()
            for order in open_stop_orders:
                if order.id == order_id:
                    return order
            raise e

    def fetch_order_by_symbol(
        self,
        trading_pair: TradingPair,
        since: Optional[TimestampMilliseconds] = None,
        until: Optional[TimestampMilliseconds] = None,
        limit: Optional[int] = None,
    ) -> List[Order]:
        req = GetHistoryOrdersRequest(
            symbol=self.get_symbol_id(trading_pair),
            startTime=since,
            endTime=until,
            limit=limit, #type: ignore
        )
        raw = self._get_history_orders(req)
        return [self._map_order(o) for o in raw]

    def fetch_orders_by_status(
        self,
        status: Status,
        trading_pair: Optional[TradingPair] = None,
        since: Optional[TimestampMilliseconds] = None,
        limit: Optional[int] = 1000,
    ) -> List[Order]:
        if status == "open" or status == "active":
            req = GetPendingOrdersRequest(
                symbol=self.get_symbol_id(trading_pair) if trading_pair else None,
                startTime=since,
                limit=limit, # type: ignore
            )
            raw = self._get_pending_orders(req)
        else:
            req = GetHistoryOrdersRequest(
                symbol=self.get_symbol_id(trading_pair) if trading_pair else None,
                startTime=since,
                limit=limit,
            )
            raw = self._get_history_orders(req)
        return [self._map_order(o) for o in raw]

    def fetch_position(
        self, trading_pair: TradingPair, side: Optional[OrderSide] = None
    ) -> Position:
        """Fetch position for a trading pair, optionally filtered by side for hedge mode."""
        pos = self.fetch_positions()

        # Filter positions by trading pair
        matching_positions = [p for p in pos if p.trading_pair == trading_pair]

        if not matching_positions:
            raise PositionNotFoundError(str(trading_pair), "BitunixFuturesExchange")

        # If no side specified, return the first available position (backward compatibility)
        if side is None:
            return matching_positions[0]

        # Filter by side for hedge mode
        # Convert side to direction: 'buy' -> 'long', 'sell' -> 'short'
        target_direction = "long" if side == "buy" else "short"

        for p in matching_positions:
            if p.direction == target_direction:
                return p

        raise ValueError(f"No {target_direction} position found for {trading_pair}")

    def fetch_untriggered_stop_orders(
        self,
        trading_pair: Optional[TradingPair] = None,
        since: Optional[TimestampMilliseconds] = None,
        limit: Optional[int] = 1000,
    ) -> List[Order]:
        """Fetch untriggered stop orders using TPSL endpoints."""
        symbol = self.get_symbol_id(trading_pair) if trading_pair else None
        req = GetTpslOrdersRequest(symbol=symbol, startTime=since, limit=limit)
        raw = self._get_pending_tpsl_orders(req)
        # Fetch positions once to avoid multiple API calls
        positions = self.fetch_positions()
        positions_dict = {pos.id: pos.direction for pos in positions if pos.trading_pair == trading_pair} if trading_pair else {pos.id: pos.direction for pos in positions}
        return [self._map_tpsl_order(o, positions_dict) for o in raw]

    def cancel_tpsl_order(self, order_id: str, trading_pair: TradingPair) -> List[str]:
        """Cancel a take profit or stop loss order by order ID."""
        symbol = self.get_symbol_id(trading_pair)
        req = CancelTpslOrderRequest(symbol=symbol, orderId=order_id)
        res = self._cancel_tpsl_order(req)
        return [res.orderId]

    def fetch_closed_tpsl(
        self,
        trading_pair: Optional[TradingPair] = None,
        since: Optional[TimestampMilliseconds] = None,
        limit: Optional[int] = None,
        side: Optional[OrderSide] = None,
    ) -> List[Order]:
        """Fetch all closed TP/SL orders."""
        # Get closed TPSL orders
        symbol = self.get_symbol_id(trading_pair) if trading_pair else None
        req = GetTpslOrdersRequest(symbol=symbol, startTime=since, limit=limit)
        closed_tpsl_orders = self._get_history_tpsl_orders(req)
        # Fetch historic positions once to avoid multiple API calls
        historic_positions = self.fetch_positions_history(since=since, trading_pair=trading_pair)
        positions_dict = {pos.closeId: pos.type for pos in historic_positions}
        tpsl_orders = [self._map_tpsl_order(o, positions_dict) for o in closed_tpsl_orders]
        
        # Filter TPSL orders by side if specified
        if side:
            tpsl_orders = [order for order in tpsl_orders if order.side == side]

        # Combine and return
        return tpsl_orders

    def get_margin_mode(self, trading_pair: TradingPair) -> MarginMode:
        symbol = self.get_symbol_id(trading_pair)
        info = self._get_leverage_margin_mode(symbol, "USDT")
        return self._map_margin_mode(info.marginMode)

    def allows_cross_mode(self, trading_pair: TradingPair) -> bool:
        return True

    def get_recent_fills(self) -> List[Fill]:
        historic_trades = self._get_history_trades(GetTradesRequest(limit=50))
        fills: List[Fill] = []
        for trade in historic_trades:
            pair = self.get_standardized_symbol(trade.symbol)

            # Convert market data from BitunixHistoryTrade
            size_in_lots = self._qty_to_lots(
                pair, float(trade.qty)
            )  # Convert qty to lots

            # Calculate trade value (price * quantity)
            trade_value = str(float(trade.price) * float(trade.qty))

            # Map margin mode
            margin_mode = self._map_margin_mode(trade.marginMode)

            # Determine if this was a forced taker (always true for market orders)
            force_taker = trade.orderType == "MARKET" or trade.roleType == "TAKER"

            # Map liquidity based on role type
            liquidity = "taker" if trade.roleType == "TAKER" else "maker"

            # Determine trade type based on reduceOnly flag
            trade_type = "close" if trade.reduceOnly else "open"

            # Handle ctime as string timestamp
            timestamp_ms = TimestampMilliseconds(int(trade.ctime))

            # Create display type, handling None effect
            effect_str = trade.effect if trade.effect else "UNKNOWN"
            display_type = f"{effect_str}_{trade.positionMode}"

            fills.append(
                Fill(
                    trading_pair=pair,
                    tradeId=trade.tradeId,
                    orderId=trade.orderId,
                    side=trade.side.lower(),  # type: ignore
                    liquidity=liquidity,
                    forceTaker=force_taker,
                    price=trade.price,
                    size=size_in_lots,  # Size in lots (contracts)
                    value=trade_value,  # Trade value (price * quantity)
                    feeRate="",  # Fee rate not provided in trade data
                    fixFee="",
                    feeCurrency="USDT",  # Bitunix futures fees are in USDT
                    stop="",
                    fee=trade.fee,
                    orderType=trade.orderType.lower(),
                    tradeType=trade_type,
                    createdAt=timestamp_ms,
                    settleCurrency="USDT",
                    tradeTime=ms_to_nano(timestamp_ms),
                    openFeePay=trade.fee
                    if not trade.reduceOnly
                    else "",  # Use fee for open trades
                    closeFeePay=trade.fee
                    if trade.reduceOnly
                    else "",  # Use fee for close trades
                    marginMode=margin_mode,
                    subTradeType=None,
                    displayType=display_type,  # Combine effect and position mode for display
                )
            )
        return fills

    # ------------------------------------------------------------------
    # symbol helpers
    # ------------------------------------------------------------------
    def get_standardized_symbol(self, exchange: str) -> TradingPair:
        if exchange.endswith("USDT"):
            base = exchange[:-4]
            return TradingPair(f"{base}/USDT:USDT")
        if exchange.endswith("USD"):
            base = exchange[:-3]
            return TradingPair(f"{base}/USD:USD")
        raise ValueError("unknown symbol")

    def get_symbol_id(self, standardized_trading_pair: TradingPair) -> BitunixContract:
        base, rest = standardized_trading_pair.split("/")
        quote = rest.split(":")[0]
        return BitunixContract(f"{base}{quote}")

    @property
    def markets(self) -> Dict[TradingPair, FuturesMarket]:
        self.load_markets()
        return self._markets

    # internal helper ---------------------------------------------------------
    def _map_order(self, o: BitunixOrder) -> Order:
        tp = self.get_standardized_symbol(o.symbol)

        # Handle margin mode conversion (string to enum)
        if o.marginMode == "ISOLATION":
            margin_mode = "ISOLATED"
        elif o.marginMode == "CROSS":
            margin_mode = "CROSS"
        else:
            margin_mode = "CROSS"  # Default fallback

        # Use avgPrice if available, otherwise fall back to price
        avg_price = o.avgPrice if o.avgPrice and o.avgPrice != "0" else o.price
        display_price = avg_price if avg_price != "MARKET" else "0"
        if display_price == "MARKET":
            ticker = self.fetch_ticker(tp)
            display_price = str(ticker.last_price) if ticker else "0"

        return Order(
            id=o.orderId,
            trading_pair=tp,
            type=o.orderType.lower(),
            side=o.side.lower(),  # type: ignore
            price=display_price,
            amountLots=self._qty_to_lots(tp, float(o.qty)),  # Convert qty to lots
            value="",
            dealValue="",
            dealSize=int(float(o.tradeQty)),
            stp="",
            stop="",
            stopPriceType="",
            stopTriggered=False,  # TODO: check if this is correct
            stopPrice="",
            timeInForce="",
            postOnly=False,
            hidden=False,
            iceberg=False,
            leverage=str(o.leverage),
            forceHold=False,
            closeOrder=False,
            visibleSize=0,
            clientOid=ClientOid.from_string(o.clientId)
            if ClientOid.is_valid_string(o.clientId)
            else None,
            remark=None,
            tags="",
            isActive=o.status == "NEW",
            cancelExist=o.status in ["CANCELED"],
            createdAt=common.TimestampMilliseconds(int(o.ctime)),
            updatedAt=common.TimestampMilliseconds(int(o.mtime)),
            endAt=common.TimestampMilliseconds(int(o.mtime)),  # Use mtime as end time
            orderTime=0,
            settleCurrency="USDT",
            marginMode=margin_mode,  # type: ignore
            avgDealPrice=avg_price if avg_price != "MARKET" else display_price,
            filledLots=self._qty_to_lots(tp, float(o.tradeQty)),
            filledValue="",
            status="open" if o.status in ["NEW", "INIT"] else "done",
            reduceOnly=o.reduceOnly,
        )

    def _map_tpsl_order(self, o: TpslOrder | HistoryTpslOrder, positions_dict: Optional[Dict[str, str]] = None) -> Order:
        """Map a TpslOrder to common Order format."""
        tp = self.get_standardized_symbol(o.symbol)

        # Determine order type and price based on whether it's TP or SL
        order_type = "market"  # TPSL orders are typically market orders when triggered
        price = "0"
        stop_price = "0"
        order_side = "unknown"

        # Determine position direction
        if positions_dict is not None:
            position_direction = positions_dict.get(o.positionId)
        else:
            # Fallback: fetch positions (for backward compatibility)
            positions = self.fetch_positions()
            position_direction = None
            for pos in positions:
                if pos.id == o.positionId:
                    position_direction = pos.direction
                    break
        if position_direction == "long":
            order_side = "sell"
        elif position_direction == "short":
            order_side = "buy"
        else:
            order_side = "sell"  # default

        # Check if it's a take profit order
        if o.tpPrice:
            price = o.tpPrice
            stop_price = o.tpPrice
            order_type = o.tpOrderType.lower() if o.tpOrderType else "market"
            # TP orders are usually sell orders to close long positions or buy to close short
        # Check if it's a stop loss order
        elif o.slPrice:
            price = o.slPrice
            stop_price = o.slPrice
            order_type = o.slOrderType.lower() if o.slOrderType else "market"
            # SL orders are usually sell orders to close long positions or buy to close short

        # Determine size - use tpQty or slQty
        qty_str = o.tpQty if o.tpQty else (o.slQty if o.slQty else "0")
        qty = float(qty_str) if qty_str and qty_str != "0" else 0
        filled_lots = self._qty_to_lots(tp, float(qty))
        # Default timestamps and filled size
        current_ms = int(time.time() * 1000)
        created_at = current_ms
        updated_at = current_ms
        endAt = current_ms

        # Map status and timestamps
        is_active = not isinstance(o, HistoryTpslOrder)
        status = "open" if is_active else "done"
        trigger_time = getattr(o, "triggerTime", None)
        if not is_active:
            created_at = o.ctime
            if trigger_time:
                updated_at = trigger_time
                endAt = trigger_time
            else:
                updated_at = o.ctime
                endAt = o.ctime

        client_oid = None

        return Order(
            id=o.id,
            trading_pair=tp,
            type=order_type,
            side=order_side,  # type: ignore
            price=price,
            amountLots=filled_lots,
            value="",
            dealValue="",
            dealSize=0,  # TPSL orders don't show partial fills in the same way
            stp="",
            stop="TP" if o.tpPrice else ("SL" if o.slPrice else ""),
            stopPriceType="LAST_PRICE",  # Bitunix typically uses last price
            stopTriggered=bool(trigger_time and int(trigger_time) > 0)
            if not is_active
            else False,
            stopPrice=stop_price,
            timeInForce="",
            postOnly=False,
            hidden=False,
            iceberg=False,
            leverage="",  # Not available in TPSL order
            forceHold=False,
            closeOrder=True,  # TPSL orders are always close orders
            visibleSize=0,
            clientOid=client_oid,
            remark=None,
            tags="TPSL",
            isActive=is_active,
            cancelExist=o.status in ["CANCELED", "SYSTEM_CANCELLED"] if isinstance(o, HistoryTpslOrder) else False,
            createdAt=common.TimestampMilliseconds(created_at),
            updatedAt=common.TimestampMilliseconds(updated_at),
            endAt=common.TimestampMilliseconds(endAt),
            orderTime=0,
            settleCurrency="USDT",
            marginMode="CROSS",  # Default for TPSL orders
            avgDealPrice=price,
            filledLots=filled_lots,
            filledValue="",
            status=status,
            reduceOnly=True,  # TPSL orders are always reduce only
        )
