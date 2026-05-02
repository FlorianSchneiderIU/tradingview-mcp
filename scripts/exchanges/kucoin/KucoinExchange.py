from enum import Enum
import time
import hashlib
import hmac
import base64
import traceback
import requests
import json
import ntplib

from exchanges.types.exceptions import SymbolNotSupportedForCopyTradingException
from my_types.config_models import KucoinConfig
from utils.SqlManager import SQLiteManager

class KucoinBaseExchange:
    def __init__(self, config: KucoinConfig, base_url: str):
        self.apiKey = config.kucoin_api_key
        self.secret = config.kucoin_api_secret
        self.password = config.kucoin_api_password
        self.enableRateLimit = 'enableRateLimit', True
        self.session = requests.Session()
        self.options = {'maxRetriesOnFailure': 3, 'maxRetriesOnFailureDelay': 1000}
        self.base_url = base_url
        self.db_manager = SQLiteManager('trading_data.db', logger=None)
    

    def _load_markets(self) -> None:
        raise NotImplementedError
    
    def _map_to_default_params(self, params: dict) -> dict:
        return params
    
    def _get_full_path(self, path, method):
        return path
    
    def _get_kucoin_timestamp(self, api_url = "https://api.kucoin.com/api/v1/timestamp"):
        try:
            response = requests.get(api_url, timeout=30)
            response.raise_for_status()

            timestamp = str(response.json()["data"])

            return timestamp
        except Exception as e:
            print(f"Error getting KuCoin timestamp: {e}")
            try:
                ntp_server = 'pool.ntp.org'
                c = ntplib.NTPClient()
                response = c.request(ntp_server)
                ntp_time = response.tx_time
                
                return str(ntp_time)
            except Exception as e:
                print(f"Error getting NTP timestamp: {e}")
                return str(int(time.time() * 1000))
    
    def _serialize_params(self, params: dict) -> dict:
        """Convert enums to their values in the params dictionary"""
        serialized = {}
        for key, value in params.items():
            if value is None:
                continue
            if isinstance(value, Enum):
                serialized[key] = value.value
            else:
                serialized[key] = value
        return serialized

    def _request(self, path: str, api: str='public', method='GET', params=None) -> dict:
        if params is None:
            params = {}
        params = self._map_to_default_params(params)
        params = self._serialize_params(params)  # Add this line to serialize enums
        
        def make_request(request_params):
            headers = {}
            body = ''
            query_string = ''
            if method in ['GET', 'DELETE']:
                if request_params:
                    query_string = '?' + '&'.join([f"{k}={v}" for k, v in request_params.items() if v is not None])
            else:
                if request_params:
                    body = json.dumps(request_params)
                else:
                    body = ''
            full_path = self._get_full_path(path, method) + query_string
            url = self.base_url + full_path
            
            max_retries = self.options.get('maxRetriesOnFailure', 3)  # Number of retries
            delay = self.options.get('maxRetriesOnFailureDelay', 1000) / 1000  # Delay in seconds between retries
            for attempt in range(max_retries):
                if api == 'private':
                    timestamp = self._get_kucoin_timestamp()
                    str_to_sign = timestamp + method.upper() + full_path + body
                    signature = base64.b64encode(
                        hmac.new(self.secret.encode('utf-8'), str_to_sign.encode('utf-8'), hashlib.sha256).digest()
                    ).decode()
                    passphrase = base64.b64encode(
                        hmac.new(self.secret.encode('utf-8'), self.password.encode('utf-8'), hashlib.sha256).digest()
                    ).decode()
                    headers = {
                        'KC-API-SIGN': signature,
                        'KC-API-TIMESTAMP': timestamp,
                        'KC-API-KEY': self.apiKey,
                        'KC-API-PASSPHRASE': passphrase,
                        'KC-API-KEY-VERSION': '3',
                        'Content-Type': 'application/json',
                    }
                try:
                    #print(f"Making {method} request to {url} with body: {body} and headers: {headers}")
                    response = self.session.request(method, url, headers=headers, data=body, timeout=60)
                    #print(f"Response: {response.status_code} - {response.text}")
                    response.raise_for_status()
                    return response.json()
                except requests.exceptions.RequestException as e:
                    # Log or print the error message
                    print(f"Request failed: {e}, attempt {attempt + 1} of {max_retries}")
                    # If it's the last attempt, re-raise the exception
                    if attempt == max_retries - 1 or (e.response is not None and e.response.status_code in [403]):
                        print(traceback.format_exc())
                        raise e
                    # Wait before retrying
                    time.sleep(delay)
            raise Exception("Request failed after all retries")
          # Make the initial request
        body = make_request(params)
        response_code = body.get('code', '')
        if response_code == '200000' or response_code == '200':
            data = body.get('data', {})
            # Check for pagination
            if isinstance(data, dict) and ('currentPage' in data and 'totalPage' in data and 'items' in data):
                all_items = data.get('items', [])
                current_page = data.get('currentPage', 1)
                total_page = data.get('totalPage', 1)
                while current_page < total_page:
                    current_page += 1
                    page_params = params.copy()
                    page_params['currentPage'] = current_page
                    page_params['pageId'] = current_page
                    next_body = make_request(page_params)
                    next_response_code = next_body.get('code', '')
                    if next_response_code != '200000' and next_response_code != '200':
                        raise Exception(f"Request failed with code {next_response_code}: {next_body}\nPath: {path}\nParams: {page_params}")
                    next_data = next_body.get('data', {})
                    next_items = next_data.get('items', [])
                    all_items.extend(next_items)
                # Update data
                data['items'] = all_items
                data['currentPage'] = 1
                data['totalPage'] = 1
                data['totalNum'] = len(all_items)
                body['data'] = data
            return body
        elif response_code == '180011' and 'symbol not supported for copyTrading' in body.get('msg', ''):
            raise SymbolNotSupportedForCopyTradingException(f"Symbol not supported for copy trading: {body}\nPath: {path}\nParams: {params}")
        else:
            raise Exception(f"Request failed with code {response_code}: {body}\nPath: {path}\nParams: {params}")
