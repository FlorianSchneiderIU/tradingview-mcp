# Capitulation Spring — forward-test runbook

A standalone strategy **sidecar** (`bot/capitulation_spring_bot.py`) that forward-tests
the validated "Capitulation Spring" long on a basket of liquid Bybit perps, forwards
accepted signals to the **RL execution sidecar**, and can optionally execute a scaled
exit on a **separate Bybit Demo account**.

Research provenance: `astro_btc_reversal_research/` — 5m deep-sweep Wyckoff spring +
early-week + negative-funding flush; long-only; scaled exit 25%@4R / 50%@12R / 25%@30R,
stop→breakeven after the first partial. Backtest (2022–2026, 18 perps, realistic fees):
~1.3 trades/wk, win ~27%, +0.56R/trade, MAR ~1.84 at 0.5% risk / 4 concurrent, positive
every year incl. the 2022 bear. **Past backtest ≠ live; this is a forward test.**

## What the setup is (per closed 5m bar, per symbol)
1. **Deep sweep**: bar trades below the lowest low of the prior ~15 days (4320 5m bars)
   and **closes back above** it (reclaim).
2. **Rejection**: long lower wick (≥50% of range) and close in the top half.
3. **Early week**: within the first 40% of the week (Mon–Wed; weekly lows cluster early).
4. **Funding flush**: `funding_z ≤ −1` (current funding unusually negative vs its recent
   distribution = shorts crowded → contrarian-bullish capitulation).
   → Long. Stop = spring low − 0.05·ATR. TPs at 4R/12R/30R (25/50/25%), stop→BE after TP1.

## Execution mechanics
- **Bracketed sub-positions (no reduce-only limits)**: the entry is placed as **one
  bracketed market leg per TP share** — 25% / 50% / 25% of qty — each leg carrying its own
  attached **`[stopLoss=SL, takeProfit=TP_i]`** as a native Bybit **Partial bracket**
  (`tpslMode="Partial"`, market triggers). The legs aggregate into a single hedge-mode
  long (`positionIdx=1`); the partial SLs at the same price sum to a full-position stop,
  and the TPs are tiered at 4R / 12R / 30R. Each leg is protected the moment it fills (no
  unprotected window). Sized to `RISK_PCT` of equity (default 0.5%).
  *Bybit aggregates same-symbol/side fills into one position, so the "sub-positions" are
  the bracket legs; this is safe because we never pyramid (one position per symbol).*
- **SL → breakeven**: a background poller (`_be_poller`, every `BE_POLL_SECONDS`=10s)
  watches each position's size; when it drops below the entry qty (first partial/4R leg
  filled) it calls `set_trading_stop(stopLoss=entry, tpslMode="Partial", slSize=remaining)`
  once, then cleans up when size reaches 0.
- **Hedge mode**: accounts here are hedge/BothSide, so the long side uses
  `positionIdx=1` for the entry, the reduce-only TPs, the breakeven `set_trading_stop`,
  and the position-size lookup. The bot **auto-detects** the mode at startup (via
  `get_positions`) and falls back to `CAPITULATION_HEDGE_MODE` (default true). One-way
  accounts use `positionIdx=0`.

## Components
- `bot/capitulation_spring.py` — pure, numpy-only signal logic (unit-tested; matches the
  research implementation 25/25 on real BTC 5m).
- `bot/capitulation_spring_bot.py` — the live sidecar: 5m WS feed for all symbols, funding
  z-score cache, RL forwarding (`rl_signal_v1`), Telegram, ledger, and the demo scaled-exit
  executor with a breakeven poller + concurrency cap.
- `docker-compose.yml` → service `capitulation-spring-bot` (depends on `rl-exec-bot`).
- `bot/.env.example` → `CAPITULATION_*` settings.

## Run it (forward test)

**Demo execution is ON by default** (nothing to lose on demo, and it exercises the real
fill/exit path — which is the whole point of a forward test). Provide a **dedicated demo
Trading account** (not the mm-bot account):
```bash
# bot/.env:
CAPITULATION_TRADING_ENABLED=true      # default
CAPITULATION_BYBIT_DEMO=true           # default
CAPITULATION_BYBIT_API_KEY=...         # demo key
CAPITULATION_BYBIT_API_SECRET=...      # demo secret
CAPITULATION_RISK_PCT=0.005            # 0.5% equity/trade
CAPITULATION_MAX_CONCURRENT=4
docker compose up -d rl-exec-bot capitulation-spring-bot
docker compose logs -f capitulation-spring-bot
```
The bot warms up (~4.5k 5m bars/symbol), then on each closed 5m bar evaluates the setup.
Execution = market entry + stop, three reduce-only limit TPs (4R/12R/30R), and a poller
that moves the stop to breakeven once the first partial fills. It refuses to exceed the
concurrency cap or open two positions in one symbol. Every **accepted** signal is also
POSTed to the RL sidecar and written to `bot/logs/capitulation_spring_signals.jsonl`.

**Graceful degradation / safety:**
- No demo keys present → bot **degrades to signal + RL-forward only** (no crash), so you
  can still watch the cadence; add keys and redeploy to start placing demo orders.
- **Going LIVE is guarded**: real-money trading requires `CAPITULATION_BYBIT_DEMO=false`
  **and** `CAPITULATION_LIVE_CONFIRM=true`; otherwise the bot refuses and runs signal-only.
- To run **signal-only on purpose**, set `CAPITULATION_TRADING_ENABLED=false`.

## "Normal sidecar" + "RL sidecar" mapping
- **Normal sidecar** = this standalone strategy bot trading the normal (demo) account —
  same pattern as `weekday-edge-bot`.
- **Accepted trades are forwarded to the RL sidecar** (`rl-exec-bot:8090`) automatically
  via `RlSidecarClient` with `status:"accepted"` and the full setup/features — so the RL
  agent can learn/shadow-trade alongside.

### Exit handling differs between the two books (by design)
- **Normal sidecar (this bot)** executes the strategy's **scaled exit**: 25%@4R /
  50%@12R / 25%@30R, stop → breakeven after the first partial.
- **RL sidecar** does **not** replicate that. Like every other strategy it applies its
  own learned execution: it sizes by its RL action, picks **one** TP tier from
  `take_profit_levels` (we send the real 4R/12R/30R tiers; single fallback = 12R), sets
  one SL or a software trailing stop, and **exits the full position at the first
  trigger** — no partial scale-outs, no strategy-driven move-to-breakeven. The
  `tp_prices` / `tp_qty_pcts` / `move_sl_to_be_after_tp1` fields are read into the RL
  feature vector but ignored at execution. So the RL shadow book is an independent
  experiment on the same *entries*, not a copy of the normal book's exit.

## Key env knobs
| Var | Default | Meaning |
|---|---|---|
| `CAPITULATION_TRADING_ENABLED` | false | place demo orders (else signal + RL only) |
| `CAPITULATION_RISK_PCT` | 0.005 | risk per trade (fraction of equity); 0 → use `RISK_USDT` |
| `CAPITULATION_MAX_CONCURRENT` | 4 | global open-position cap |
| `CAPITULATION_FUNDING_Z_THR` | −1.0 | funding flush threshold (−1.5 = higher conviction) |
| `CAPITULATION_WEEK_FRAC_MAX` | 0.40 | early-week gate |
| `CAPITULATION_SWEEP_LOOKBACK` | 4320 | 15-day deep-sweep window (5m bars) |
| `CAPITULATION_SYMBOLS` | 18 perps | basket |

## Monitoring & known caveats
- Watch `capitulation_spring_signals.jsonl`, the RL sidecar `decisions.jsonl`/`rewards.jsonl`,
  and Bybit demo positions. Expect **~1–1.5 setups/week** clustered around market-wide flushes.
- Fat-tailed (~27% win) — long droughts are normal; judge over many weeks, not days.
- 5m **slippage** on the spring fill is the main live unknown vs backtest (tiny stop ⇒
  slippage matters proportionally). The funding z-score uses Bybit `get_funding_rate_history`
  (8h cadence) cached ~30 min; it is a coarse live proxy for the research `funding_z`.
- Shorts were tested and rejected — **long-only** by design.
