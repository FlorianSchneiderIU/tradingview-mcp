"""
Million Moves Algo V4.3 — Python Backtester
============================================
Replicates the Pine Script strategy logic exactly:
  - Supertrend(source=open, mult=2.5, ATR len=11)
  - EMA(200) on close
  - SMA(13)  on close
  - ATR(14)  on close (for SL sizing)

Entry signals (Smart Signals):
  Sbull = crossover(close, supertrend) AND close >= sma13 AND close[-1] > ema200 AND close > ema200
  Sbear = crossunder(close, supertrend) AND close <= sma13 AND NOT(close[-1] > ema200 AND close > ema200)

Exit logic (mirrors TV strategy.exit):
  SL  = low  - ATR14 * 2.2   (long)  /  high + ATR14 * 2.2  (short)   — at the signal bar
  TP1 = entry + 1 * risk  (long)  /  entry - 1 * risk  (short)   — close 33%
  TP2 = entry + 2 * risk  (long)  /  entry - 2 * risk  (short)   — close 50% of remaining
  TP3 = entry + 3 * risk  (long)  /  entry - 3 * risk  (short)   — close rest
  Reversal: close immediately when opposite signal fires

Data: BINANCE ETHUSDT 15m fetched via ccxt (Binance)
Output: scripts/million_moves_v43_trades.csv
"""

import sys
import os
import math
import argparse
import numpy as np
import pandas as pd
import ccxt

# ─────────────────────────────────────────────────────────────────────────────
# Parameters (match strategy defaults)
# ─────────────────────────────────────────────────────────────────────────────
ST_MULT       = 2.5   # supertrend multiplier (sigsensiviti)
ST_ATR_LEN    = 11    # supertrend ATR length  (factor)
EMA_LEN       = 200
SMA_LEN       = 13
ATR_SL_LEN    = 14
ATR_SL_MULT   = 2.2
TP_MULT       = 1.0   # TP strength multiplier

SYMBOL        = "ETH/USDT"
TIMEFRAME     = "15m"
SINCE_DATE    = "2021-01-01"   # start of history
EXCHANGE_ID   = "binance"

OUTPUT_CSV    = os.path.join(os.path.dirname(__file__), "million_moves_v43_trades.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Data fetching
# ─────────────────────────────────────────────────────────────────────────────
def fetch_ohlcv(symbol: str, timeframe: str, since_date: str) -> pd.DataFrame:
    exchange = ccxt.binance({"enableRateLimit": True})
    since_ms = exchange.parse8601(f"{since_date}T00:00:00Z")
    all_bars = []
    print(f"Fetching {symbol} {timeframe} from {since_date} …", flush=True)
    while True:
        bars = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=1000)
        if not bars:
            break
        all_bars.extend(bars)
        if len(bars) < 1000:
            break
        since_ms = bars[-1][0] + 1
    df = pd.DataFrame(all_bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    print(f"  → {len(df):,} bars loaded  ({df.index[0]} … {df.index[-1]})")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Indicator helpers (Wilder / RMA style to match Pine Script)
# ─────────────────────────────────────────────────────────────────────────────
def _rma(series: pd.Series, length: int) -> pd.Series:
    """Wilder's smoothed moving average (ta.rma in Pine Script)."""
    alpha  = 1.0 / length
    result = series.copy() * np.nan
    # seed with simple average for first window
    first_valid = series.first_valid_index()
    if first_valid is None:
        return result
    iloc_start = series.index.get_loc(first_valid)
    # find first full window
    start = iloc_start + length - 1
    if start >= len(series):
        return result
    result.iloc[start] = series.iloc[iloc_start:start + 1].mean()
    for i in range(start + 1, len(series)):
        result.iloc[i] = alpha * series.iloc[i] + (1 - alpha) * result.iloc[i - 1]
    return result


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    """Average True Range using Wilder smoothing (matches ta.atr in Pine)."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return _rma(tr, length)


def _ema(series: pd.Series, length: int) -> pd.Series:
    """Exponential moving average (matches ta.ema — uses standard alpha=2/(len+1))."""
    return series.ewm(span=length, adjust=False).mean()


def _sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(window=length).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Supertrend (source = open, direction checks use close)
# ─────────────────────────────────────────────────────────────────────────────
def supertrend(df: pd.DataFrame, mult: float, atr_len: int) -> tuple[pd.Series, pd.Series]:
    """
    Replicates the Pine Script supertrend() exactly:
      atr = ta.atr(atrLen)
      upperBand = _close + factor * atr     # _close = open in original
      lowerBand = _close - factor * atr
      band ratchet and direction checks use close (not open)
    Returns (supertrend_series, direction_series)  — direction: -1 bullish, 1 bearish
    """
    src   = df["open"].values
    close = df["close"].values
    high  = df["high"].values
    low_  = df["low"].values
    n     = len(df)

    atr_vals = _atr(df["high"], df["low"], df["close"], atr_len).values

    upper_raw = src + mult * atr_vals
    lower_raw = src - mult * atr_vals

    upper = upper_raw.copy()
    lower = lower_raw.copy()
    direction = np.full(n, np.nan)
    st        = np.full(n, np.nan)

    for i in range(1, n):
        if np.isnan(atr_vals[i - 1]):
            direction[i] = 2
        else:
            # ratchet lower band
            if lower_raw[i] > lower[i - 1] or close[i - 1] < lower[i - 1]:
                lower[i] = lower_raw[i]
            else:
                lower[i] = lower[i - 1]

            # ratchet upper band
            if upper_raw[i] < upper[i - 1] or close[i - 1] > upper[i - 1]:
                upper[i] = upper_raw[i]
            else:
                upper[i] = upper[i - 1]

            prev_st = st[i - 1]
            if np.isnan(prev_st):
                # initialise direction from previous band
                prev_st = upper[i - 1] if not np.isnan(upper[i - 1]) else lower[i - 1]

            if prev_st == upper[i - 1]:
                direction[i] = -1 if close[i] > upper[i] else 1
            else:
                direction[i] = 1 if close[i] < lower[i] else -1

        st[i] = lower[i] if direction[i] == -1 else upper[i]

    st_series  = pd.Series(st,        index=df.index, name="supertrend")
    dir_series = pd.Series(direction, index=df.index, name="st_dir")
    return st_series, dir_series


# ─────────────────────────────────────────────────────────────────────────────
# Build indicators
# ─────────────────────────────────────────────────────────────────────────────
def build_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema200"], df["sma13"], df["atr14"] = (
        _ema(df["close"], EMA_LEN),
        _sma(df["close"], SMA_LEN),
        _atr(df["high"],  df["low"], df["close"], ATR_SL_LEN),
    )
    df["st"], df["st_dir"] = supertrend(df, ST_MULT, ST_ATR_LEN)

    # Crossover / crossunder  (Pine: a crosses above b ≡ a[1] < b[1] and a > b)
    df["co"] = (df["close"].shift(1) < df["st"].shift(1)) & (df["close"] > df["st"])
    df["cu"] = (df["close"].shift(1) > df["st"].shift(1)) & (df["close"] < df["st"])

    # above EMA200 flag (matches Pine `aboveEma = close[1] > ema200 and close > ema200`)
    df["above_ema"] = (df["close"].shift(1) > df["ema200"].shift(1)) & (df["close"] > df["ema200"])

    # Smart Signals
    df["Sbull"] = df["co"] & (df["close"] >= df["sma13"]) &  df["above_ema"]
    df["Sbear"] = df["cu"] & (df["close"] <= df["sma13"]) & ~df["above_ema"]

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Trade simulation
# ─────────────────────────────────────────────────────────────────────────────
class Trade:
    def __init__(self, entry_time, direction: str, entry_price: float,
                 sl: float, tp1: float, tp2: float, tp3: float):
        self.entry_time  = entry_time
        self.direction   = direction
        self.entry_price = entry_price
        self.sl          = sl
        self.tp1         = tp1
        self.tp2         = tp2
        self.tp3         = tp3
        # fractional position remaining (0-1)
        self.remaining   = 1.0
        self.tp1_hit     = False
        self.tp2_hit     = False
        # partial fills for P&L calculation
        self.fills: list[tuple[float, float]] = []   # (fraction_closed, price)

    def pnl_pct(self) -> float:
        """Weighted average exit P&L as % of entry price (before commission)."""
        if self.direction == "long":
            return sum((p - self.entry_price) / self.entry_price * frac
                       for frac, p in self.fills)
        else:
            return sum((self.entry_price - p) / self.entry_price * frac
                       for frac, p in self.fills)


def simulate_trades(df: pd.DataFrame) -> list[dict]:
    records = []
    active: Trade | None = None

    close  = df["close"].values
    high   = df["high"].values
    low_   = df["low"].values
    sbull  = df["Sbull"].values
    sbear  = df["Sbear"].values
    atr14  = df["atr14"].values
    idx    = df.index

    n = len(df)

    def close_trade(t: Trade, exit_time, exit_price: float, fraction: float, reason: str):
        t.fills.append((fraction, exit_price))
        t.remaining -= fraction
        records.append({
            "trade_id"   : len(records) + 1,
            "entry_time" : t.entry_time,
            "exit_time"  : exit_time,
            "direction"  : t.direction,
            "entry_price": t.entry_price,
            "exit_price" : exit_price,
            "sl"         : t.sl,
            "tp1"        : t.tp1,
            "tp2"        : t.tp2,
            "tp3"        : t.tp3,
            "exit_reason": reason,
            "fraction"   : fraction,
            "pnl_pct"    : ((exit_price - t.entry_price) / t.entry_price
                            if t.direction == "long"
                            else (t.entry_price - exit_price) / t.entry_price) * fraction,
        })

    for i in range(1, n):
        # ── Check open trade exits on this bar ────────────────────────────
        if active is not None:
            bar_high = high[i]
            bar_low  = low_[i]
            direction = active.direction

            if direction == "long":
                # SL check first (conservative — assume lower price hit first)
                sl_hit  = bar_low  <= active.sl
                tp1_hit = bar_high >= active.tp1 and not active.tp1_hit
                tp2_hit = bar_high >= active.tp2 and active.tp1_hit and not active.tp2_hit
                tp3_hit = bar_high >= active.tp3 and active.tp2_hit

                if sl_hit and not active.tp1_hit:
                    # stop takes entire remaining position
                    close_trade(active, idx[i], active.sl, active.remaining, "SL")
                    active = None
                else:
                    if tp1_hit:
                        frac = round(0.33 * 1.0, 10)
                        close_trade(active, idx[i], active.tp1, frac, "TP1")
                        active.tp1_hit = True
                    if active is not None and tp2_hit:
                        # 50% of remaining (after TP1)
                        frac = round(active.remaining * 0.50, 10)
                        close_trade(active, idx[i], active.tp2, frac, "TP2")
                        active.tp2_hit = True
                    if active is not None and tp3_hit:
                        close_trade(active, idx[i], active.tp3, active.remaining, "TP3")
                        active = None
                    # SL after TP1 (if stop still triggers on same bar)
                    if active is not None and sl_hit:
                        close_trade(active, idx[i], active.sl, active.remaining, "SL")
                        active = None
            else:  # short
                sl_hit  = bar_high >= active.sl
                tp1_hit = bar_low  <= active.tp1 and not active.tp1_hit
                tp2_hit = bar_low  <= active.tp2 and active.tp1_hit and not active.tp2_hit
                tp3_hit = bar_low  <= active.tp3 and active.tp2_hit

                if sl_hit and not active.tp1_hit:
                    close_trade(active, idx[i], active.sl, active.remaining, "SL")
                    active = None
                else:
                    if tp1_hit:
                        frac = round(0.33 * 1.0, 10)
                        close_trade(active, idx[i], active.tp1, frac, "TP1")
                        active.tp1_hit = True
                    if active is not None and tp2_hit:
                        frac = round(active.remaining * 0.50, 10)
                        close_trade(active, idx[i], active.tp2, frac, "TP2")
                        active.tp2_hit = True
                    if active is not None and tp3_hit:
                        close_trade(active, idx[i], active.tp3, active.remaining, "TP3")
                        active = None
                    if active is not None and sl_hit:
                        close_trade(active, idx[i], active.sl, active.remaining, "SL")
                        active = None

        # ── Check for new entry signals on this bar ───────────────────────
        # Reverse if opposite signal fires while in a trade
        new_long  = sbull[i]
        new_short = sbear[i]

        if active is not None and active.direction == "long" and new_short:
            close_trade(active, idx[i], close[i], active.remaining, "Reversed")
            active = None

        if active is not None and active.direction == "short" and new_long:
            close_trade(active, idx[i], close[i], active.remaining, "Reversed")
            active = None

        # Enter new trade (only if not already in a trade — matches pyramiding=0)
        atr = atr14[i]
        if active is None and new_long and not math.isnan(atr):
            sl   = low_[i] - atr * ATR_SL_MULT
            risk = max(close[i] - sl, 1e-10)
            active = Trade(
                entry_time  = idx[i],
                direction   = "long",
                entry_price = close[i],
                sl  = sl,
                tp1 = close[i] + TP_MULT * 1 * risk,
                tp2 = close[i] + TP_MULT * 2 * risk,
                tp3 = close[i] + TP_MULT * 3 * risk,
            )

        elif active is None and new_short and not math.isnan(atr):
            sl   = high[i] + atr * ATR_SL_MULT
            risk = max(sl - close[i], 1e-10)
            active = Trade(
                entry_time  = idx[i],
                direction   = "short",
                entry_price = close[i],
                sl  = sl,
                tp1 = close[i] - TP_MULT * 1 * risk,
                tp2 = close[i] - TP_MULT * 2 * risk,
                tp3 = close[i] - TP_MULT * 3 * risk,
            )

    # Close any open trade at end of data (mark as "Open")
    if active is not None:
        last_i = n - 1
        close_trade(active, idx[last_i], close[last_i], active.remaining, "Open")

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Aggregated trade-level summary (one row per entry/signal)
# ─────────────────────────────────────────────────────────────────────────────
def aggregate_trades(records: list[dict]) -> pd.DataFrame:
    """
    Group partial fills from the same entry into a single row,
    showing entry_time, direction, overall PnL%, and composite exit reason.
    """
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    grp = df.groupby(["entry_time", "direction", "entry_price", "sl", "tp1", "tp2", "tp3"],
                     sort=False)

    rows = []
    for (entry_time, direction, entry_price, sl, tp1, tp2, tp3), g in grp:
        reasons   = g["exit_reason"].tolist()
        exit_time = g["exit_time"].iloc[-1]   # last partial fill bar
        total_pnl = g["pnl_pct"].sum()
        rows.append({
            "entry_time"  : entry_time,
            "exit_time"   : exit_time,
            "direction"   : direction,
            "entry_price" : round(entry_price, 6),
            "sl"          : round(sl, 6),
            "tp1"         : round(tp1, 6),
            "tp2"         : round(tp2, 6),
            "tp3"         : round(tp3, 6),
            "exit_reasons": "|".join(reasons),
            "pnl_pct"     : round(total_pnl * 100, 4),   # as percentage
        })

    return pd.DataFrame(rows).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Million Moves V4.3 Python Backtester")
    parser.add_argument("--since",  default=SINCE_DATE,  help="Start date YYYY-MM-DD")
    parser.add_argument("--symbol", default=SYMBOL,      help="Exchange symbol (e.g. ETH/USDT)")
    parser.add_argument("--tf",     default=TIMEFRAME,   help="Timeframe (e.g. 15m)")
    parser.add_argument("--output", default=OUTPUT_CSV,  help="Output CSV path")
    parser.add_argument("--show",   type=int, default=20, help="Print N most recent trades")
    args = parser.parse_args()

    # 1. Fetch data
    df = fetch_ohlcv(args.symbol, args.tf, args.since)

    # 2. Build indicators
    print("Computing indicators …", flush=True)
    df = build_indicators(df)

    sbull_count = df["Sbull"].sum()
    sbear_count = df["Sbear"].sum()
    print(f"  Sbull signals: {sbull_count:,}   Sbear signals: {sbear_count:,}")

    # 3. Simulate trades
    print("Simulating trades …", flush=True)
    records = simulate_trades(df)

    # 4. Aggregate
    trades = aggregate_trades(records)
    print(f"  Total trade entries: {len(trades):,}")

    if trades.empty:
        print("No trades generated.")
        return

    # 5. Print summary
    long_trades  = trades[trades["direction"] == "long"]
    short_trades = trades[trades["direction"] == "short"]
    wins   = trades[trades["pnl_pct"] > 0]
    losses = trades[trades["pnl_pct"] < 0]

    print(f"\n{'─'*60}")
    print(f"  BACKTEST SUMMARY  ({args.symbol} {args.tf})")
    print(f"{'─'*60}")
    print(f"  Total trades  : {len(trades)}")
    print(f"  Long  trades  : {len(long_trades)}")
    print(f"  Short trades  : {len(short_trades)}")
    print(f"  Win rate      : {len(wins) / max(len(trades), 1) * 100:.1f}%")
    print(f"  Avg PnL       : {trades['pnl_pct'].mean():.3f}%")
    print(f"  Total PnL     : {trades['pnl_pct'].sum():.2f}%")
    print(f"  Period        : {trades['entry_time'].iloc[0]}  →  {trades['exit_time'].iloc[-1]}")
    print(f"{'─'*60}\n")

    print(f"Last {args.show} trades:")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    print(trades.tail(args.show).to_string(index=True))

    # 6. Save
    trades.to_csv(args.output, index=True)
    print(f"\nTrades saved → {args.output}")

    # 7. Exit reasons breakdown
    all_reasons = "|".join(trades["exit_reasons"].tolist()).split("|")
    reason_counts = pd.Series(all_reasons).value_counts()
    print("\nExit reason breakdown:")
    print(reason_counts.to_string())


if __name__ == "__main__":
    main()
