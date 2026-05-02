"""Bybit demo/testnet futures exchange powered by CCXT."""

from __future__ import annotations

from exchanges.bybit.BybitFuturesExchange import ByBitFuturesExchange
from my_types.config_models import BybitConfig
from utils.SqlManager import SQLiteManager


class ByBitDemoExchange(ByBitFuturesExchange):
    """Concrete :class:`ByBitFuturesExchange` for Bybit Demo Trading.
    
    This exchange connects to Bybit's PRODUCTION API but enables demo trading mode.
    This is used for prop firm demo accounts (e.g., KleinFunding) that use demo trading
    keys but connect to the production Bybit infrastructure.
    
    The only difference from ByBitFuturesExchange is passing use_demo_trading=True.
    """

    def __init__(self, config: BybitConfig, db_manager: SQLiteManager) -> None:
        # Call parent with use_demo_trading=True to configure client before any API calls
        super().__init__(config, db_manager, use_demo_trading=True)
