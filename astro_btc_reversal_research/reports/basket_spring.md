# Multi-Symbol Deep-Sweep-Spring Basket

18/18 symbols | 15m | 2022-01-01 -> 2026-06-01 (~230.3 weeks) | sweep lookback 2880 bars | week gate 0.4 | costs 11.0 bps RT.

## Aggregate (all symbols pooled)

| Target | Trades | Trades/week | Win % | Avg R | Net R | PF | MaxDD R | reach20R | Holdout avg R (n) |
|---|---|---|---|---|---|---|---|---|---|
| 20R | 529 | 2.30 | 11.0 | -0.221 | -117.1 | 0.775 | -204.6 | 0.015 | 0.275 (204) |
| 30R | 529 | 2.30 | 10.8 | -0.241 | -127.7 | 0.756 | -225.7 | 0.015 | 0.319 (204) |

## Per-symbol (target 20R)

| Symbol | bars | signals | trades | win % | avg R | net R | PF |
|---|---|---|---|---|---|---|---|
| BTCUSDT | 154752 | 45 | 36 | 11.1 | -0.156 | -5.6 | 0.845 |
| ETHUSDT | 154753 | 26 | 22 | 13.6 | -0.602 | -13.3 | 0.361 |
| SOLUSDT | 154753 | 43 | 36 | 8.3 | -0.256 | -9.2 | 0.745 |
| BNBUSDT | 154753 | 25 | 23 | 13.0 | 0.428 | 9.9 | 1.429 |
| XRPUSDT | 154753 | 27 | 23 | 21.7 | 0.102 | 2.4 | 1.125 |
| ADAUSDT | 154753 | 37 | 31 | 3.2 | -1.012 | -31.4 | 0.059 |
| DOGEUSDT | 154753 | 28 | 23 | 8.7 | -0.655 | -15.1 | 0.350 |
| AVAXUSDT | 154753 | 51 | 43 | 7.0 | -0.890 | -38.3 | 0.131 |
| LINKUSDT | 154753 | 36 | 30 | 10.0 | -0.404 | -12.1 | 0.590 |
| DOTUSDT | 154753 | 48 | 39 | 7.7 | -0.555 | -21.7 | 0.466 |
| LTCUSDT | 154753 | 24 | 21 | 14.3 | -0.249 | -5.2 | 0.740 |
| ATOMUSDT | 154753 | 47 | 43 | 7.0 | -0.583 | -25.1 | 0.445 |
| NEARUSDT | 154753 | 51 | 44 | 9.1 | 0.156 | 6.9 | 1.157 |
| FILUSDT | 154753 | 20 | 18 | 22.2 | 1.208 | 21.7 | 2.412 |
| ARBUSDT | 111878 | 32 | 28 | 10.7 | -0.622 | -17.4 | 0.372 |
| OPUSDT | 140221 | 26 | 23 | 8.7 | -0.819 | -18.8 | 0.179 |
| INJUSDT | 132834 | 23 | 21 | 19.0 | 0.529 | 11.1 | 1.586 |
| SUIUSDT | 107947 | 30 | 25 | 20.0 | 1.764 | 44.1 | 3.029 |

## Setup clustering (excursion signals per quarter)

2022Q1:9, 2022Q2:108, 2022Q3:8, 2022Q4:36, 2023Q1:14, 2023Q2:40, 2023Q3:30, 2023Q4:9, 2024Q1:26, 2024Q2:36, 2024Q3:4, 2024Q4:5, 2025Q1:96, 2025Q2:15, 2025Q3:5, 2025Q4:73, 2026Q1:10, 2026Q2:5

## Reading guide

Goal: ~1 trade/week (trades/week ~ 1) with avg R > 0 net of costs AND positive holdout. Watch the per-quarter clustering - basket deep-sweeps bunch during market-wide crashes, so 'trades/week' is an average, not an even cadence. A few symbols may carry the edge; check the per-symbol table for breadth vs concentration.
