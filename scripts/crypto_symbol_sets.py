from __future__ import annotations

from scripts.backtest_turtle_soup import normalize_binance_spot_symbol


SYMBOL_SETS: dict[str, list[str]] = {
    "core3": [
        "BINANCE:BTCUSDT",
        "BINANCE:ETHUSDT",
        "BINANCE:SOLUSDT",
    ],
    "majors4": [
        "BINANCE:BTCUSDT",
        "BINANCE:ETHUSDT",
        "BINANCE:SOLUSDT",
        "BINANCE:BNBUSDT",
    ],
    "liquid6": [
        "BINANCE:BTCUSDT",
        "BINANCE:ETHUSDT",
        "BINANCE:SOLUSDT",
        "BINANCE:BNBUSDT",
        "BINANCE:LINKUSDT",
        "BINANCE:BCHUSDT",
    ],
    "majors10": [
        "BINANCE:BTCUSDT",
        "BINANCE:ETHUSDT",
        "BINANCE:BNBUSDT",
        "BINANCE:SOLUSDT",
        "BINANCE:XRPUSDT",
        "BINANCE:DOGEUSDT",
        "BINANCE:ADAUSDT",
        "BINANCE:TRXUSDT",
        "BINANCE:LINKUSDT",
        "BINANCE:AVAXUSDT",
    ],
    # Current-ish high market-cap Binance spot universe, excluding stables/wrapped
    # assets and keeping the name intentionally approximate.
    "majors20": [
        "BINANCE:BTCUSDT",
        "BINANCE:ETHUSDT",
        "BINANCE:BNBUSDT",
        "BINANCE:SOLUSDT",
        "BINANCE:XRPUSDT",
        "BINANCE:DOGEUSDT",
        "BINANCE:ADAUSDT",
        "BINANCE:TRXUSDT",
        "BINANCE:LINKUSDT",
        "BINANCE:AVAXUSDT",
        "BINANCE:XLMUSDT",
        "BINANCE:BCHUSDT",
        "BINANCE:HBARUSDT",
        "BINANCE:LTCUSDT",
        "BINANCE:TONUSDT",
        "BINANCE:SHIBUSDT",
        "BINANCE:DOTUSDT",
        "BINANCE:UNIUSDT",
        "BINANCE:APTUSDT",
        "BINANCE:NEARUSDT",
    ],
}


def expand_symbol_args(symbols: list[str], symbol_set: str | None) -> list[str]:
    expanded: list[str] = []
    if symbol_set and symbol_set != "none":
        expanded.extend(SYMBOL_SETS[symbol_set])
    expanded.extend(symbols)

    seen: set[str] = set()
    unique: list[str] = []
    for symbol in expanded:
        normalized = normalize_binance_spot_symbol(symbol)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(f"BINANCE:{normalized}")
    return unique
