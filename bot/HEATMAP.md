# Heatmap bot — Bybit market-structure service

Two containers (`heatmap-bot`, `heatmap-telegram-bot`) that turn public Bybit data into
liquidity / liquidation / volume structure for the rest of the fleet. **No API key** — all
public market data. State persists in SQLite on the `heatmap_data` named volume (survives
restarts & rebuilds).

## Layers

| Layer | Source | What it is |
|---|---|---|
| **Order-book liquidity** | `orderbook.200` WS | Resting L2 size binned around mid, spoof-filtered by a lifetime threshold |
| **Actual liquidations** | `allLiquidation` WS | Real force-liquidation prints (Sell = long liq, Buy = short) |
| **Predictive liquidations** | klines + open interest | Coinglass-style ΔOI leverage-cohort model; mark-price liq levels; cohorts decay on OI drop and are consumed when price crosses them. Bayesian (Dirichlet) auto-calibration to actual prints |
| **Volume profile** | `publicTrade` WS | Buy/sell (delta) volume per price bin, hourly; kline-seeded; POC/VA/HVN + delta |

Derived: **consolidated levels with confluence**, **market-structure snapshot**, **CVD**,
**screener**, plus **proactive alerts** (liquidation cascades, level proximity) and a **WS
staleness watchdog**.

## REST API (`http://heatmap-bot:8110`)

```
GET /health
GET /v1/universe
GET /v1/liquidity/{symbol}?limit=                         order-book liquidity snapshots
GET /v1/liquidations/actual/{symbol}?limit=               real liquidation prints
GET /v1/liquidations/estimated/{symbol}?tf=5m|15m|1h      predictive heatmap (+ /latest = magnets)
GET /v1/volume_profile/{symbol}?window=4h|24h|7d|daily|weekly   (+ /heatmap = time×price cells + OHLC)
GET /v1/cvd/{symbol}?window=                              cumulative volume delta series
GET /v1/levels/{symbol}?tf=&window=&n=                    ranked key levels + confluence
GET /v1/structure/{symbol}                                decision-ready snapshot (bias, S/R, skew, funding, OI)
GET /v1/screener?metric=liq|cvd|imbalance|volume&n=       rank the universe
GET /v1/ohlc/{symbol}?interval=&start=                    candles (for overlays)
GET /v1/calibration                                       weights, concentration, hit-rate history
GET /v1/watches?uid=                                      a user's proximity subscriptions
POST /v1/watch  {uid, symbol, add}                        add/remove a proximity subscription
```

Other fleet bots should use [`heatmap_client.py`](heatmap_client.py) rather than raw HTTP:

```python
from heatmap_client import HeatmapClient
hm = HeatmapClient()                      # HEATMAP_API_URL or http://heatmap-bot:8110
s = hm.structure("BTCUSDT")               # None on any error (never raises)
lvl = hm.is_near_level("BTCUSDT", entry_price, pct=0.003)
```

## Telegram

```
/liquidations SYM [5m|15m|1h]      /actual SYM      /liquidity SYM
/volume SYM [4h|24h|7d|daily|weekly]    /cvd SYM [window]
/levels SYM [tf]      /structure SYM      /screener [metric]      /score
/watch SYM   /unwatch SYM   /watches      /coins   /help
```
Cascade alerts fire automatically; proximity alerts fire for `/watch`'d coins. Set
`HEATMAP_TG_BOT_TOKEN` + `HEATMAP_TG_ALLOWED_UIDS` in `bot/.env.heatmap`.

## Config

All knobs live in [`configs/heatmap_configs.json`](configs/heatmap_configs.json), each
overridable by the matching `HEATMAP_*` env var (env wins). The tracked universe is
`HEATMAP_SYMBOLS` in `.env.heatmap`. See [`.env.heatmap.example`](.env.heatmap.example).

## Calibration

Once a day the predictive leverage weights are fit to actual liquidation prints as a
Dirichlet posterior (`alpha ← forget·alpha + counts`; weights = posterior mean). `/score`
and `/v1/calibration` report the **hit_rate** (share of real liquidations the model placed
intensity at) so you can see whether the predictive layer is any good. Weights are global
(pooled across symbols); per-symbol calibration is a future step once per-coin volumes
justify it.

## Run / test

```
docker compose up -d --build heatmap-bot heatmap-telegram-bot
python bot/test_heatmap_bot.py        # pure-function unit checks (stdlib, no pytest)
```
