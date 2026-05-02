class LeverageError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
        
class SymbolNotFoundError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class SymbolNotSupportedForCopyTradingException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class MarginModeMismatchError(Exception):
    """Raised when an order's margin mode conflicts with the position's margin mode."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)
        
class PositionNotFoundError(Exception):
    """Raised when a position is not found for a given trading pair."""

    def __init__(self, pair: str, exchange: str):
        self.message = f"Position not found for trading pair {pair} on exchange {exchange}."
        super().__init__(self.message)
