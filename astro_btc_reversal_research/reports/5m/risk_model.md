# 5m Capitulation Long — Risk Model + BTC.D Regime Gate

337 raw trades | fees: taker entry/stop 5.5bps, maker TP 2bps | risk 0.5%/trade, max 4 concurrent unless noted | BTC.D coverage 100% of trades.

## Regime gates (account metrics, risk 0.5%, K=4)

| Gate | Skipped | Trades | /wk | Win % | CAGR % | MaxDD % | MAR | Total % |
|---|---|---|---|---|---|---|---|---|
| none | 0 | 323 | 1.44 | 26.9 | 22.8 | 12.4 | 1.84 | 142.5 |
| freefall<-0.15 | 11 | 312 | 1.42 | 26.3 | 21.0 | 12.4 | 1.69 | 124.3 |
| freefall<-0.25 | 2 | 321 | 1.43 | 26.8 | 22.0 | 12.4 | 1.77 | 135.8 |
| btcd_14d>+0.05 | 116 | 219 | 0.99 | 25.6 | 13.6 | 12.4 | 1.1 | 71.5 |
| btcd_14d>+0.08 | 84 | 251 | 1.12 | 27.9 | 21.9 | 11.3 | 1.93 | 134.5 |
| freefall<-0.15 & btcd>+0.05 | 122 | 213 | 0.97 | 24.9 | 12.1 | 12.4 | 0.97 | 62.1 |

## Sizing / concurrency grid (no gate)

| Config | Trades | /wk | CAGR % | MaxDD % | MAR | Total % |
|---|---|---|---|---|---|---|
| risk0.5%_K2 | 268 | 1.2 | 19.1 | 11.2 | 1.7 | 112.2 |
| risk0.5%_K4 | 323 | 1.44 | 22.8 | 12.4 | 1.84 | 142.5 |
| risk0.5%_K6 | 334 | 1.49 | 23.5 | 14.8 | 1.59 | 148.1 |
| risk0.75%_K2 | 268 | 1.2 | 28.6 | 16.3 | 1.75 | 195.7 |
| risk0.75%_K4 | 323 | 1.44 | 33.9 | 18.1 | 1.87 | 252.3 |
| risk0.75%_K6 | 334 | 1.49 | 34.8 | 21.5 | 1.62 | 262.6 |
| risk1%_K2 | 268 | 1.2 | 38.0 | 21.2 | 1.79 | 300.8 |
| risk1%_K4 | 323 | 1.44 | 44.7 | 23.5 | 1.9 | 391.3 |
| risk1%_K6 | 334 | 1.49 | 45.7 | 27.7 | 1.65 | 407.3 |

## Reading guide

MAR = CAGR / MaxDD% (higher is better; >0.5 is decent, >1 is strong). The regime gate is worth keeping only if it raises MAR (cuts MaxDD more than it cuts CAGR). BTC.D rising = alts bleeding; BTC freefall = cascade. But capitulation bottoms often occur DURING those, so a gate can also remove the best entries - let the MAR column decide.
