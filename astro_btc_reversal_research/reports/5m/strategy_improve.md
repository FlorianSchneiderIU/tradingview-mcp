# 5m Capitulation Long — Fee & Drawdown Improvements

Median stop distance: **0.758% of price** (mean 1.281%).

## 1) Fee decomposition (buffer 0.05 ATR, funding_z<=-1)

| Fee model | Trades | /wk | Win % | Avg R | Net R | MaxDD R | Holdout avg R (n) |
|---|---|---|---|---|---|---|---|
| gross (0 fees) | 337 | 1.46 | 26.7 | 0.767 | 258.5 | -28.7 | 0.892 (163) |
| net taker 11bps RT | 337 | 1.46 | 26.7 | 0.588 | 198.2 | -34.6 | 0.678 (163) |
| net realistic (taker entry/stop, maker TP) | 337 | 1.46 | 26.7 | 0.597 | 201.1 | -34.6 | 0.688 (163) |

## 2) Stop-width x funding sweep (realistic fees)

| Config | Trades | /wk | Win % | Avg R | Net R | MaxDD R | Holdout avg R (n) |
|---|---|---|---|---|---|---|---|
| buf0.05_fz-1.0 | 337 | 1.46 | 26.7 | 0.597 | 201.1 | -34.6 | 0.688 (163) |
| buf0.05_fz-1.5 | 198 | 0.86 | 27.8 | 0.484 | 95.9 | -29.6 | 0.436 (99) |
| buf0.15_fz-1.0 | 333 | 1.45 | 26.1 | 0.544 | 181.0 | -34.9 | 0.602 (161) |
| buf0.15_fz-1.5 | 196 | 0.85 | 27.0 | 0.419 | 82.2 | -29.9 | 0.397 (98) |
| buf0.3_fz-1.0 | 323 | 1.4 | 26.6 | 0.535 | 172.7 | -34.4 | 0.651 (158) |
| buf0.3_fz-1.5 | 190 | 0.83 | 25.8 | 0.359 | 68.2 | -29.4 | 0.32 (97) |
| buf0.5_fz-1.0 | 318 | 1.38 | 25.5 | 0.48 | 152.5 | -31.4 | 0.687 (156) |
| buf0.5_fz-1.5 | 188 | 0.82 | 25.0 | 0.304 | 57.2 | -28.4 | 0.42 (96) |

## 3) Portfolio concurrency cap (realistic fees, buffer 0.30, fz<=-1)

| Max concurrent | Trades | /wk | Win % | Avg R | Net R | MaxDD R | Holdout avg R (n) |
|---|---|---|---|---|---|---|---|
| K=inf | 323 | 1.4 | 26.6 | 0.535 | 172.7 | -34.4 | 0.651 (158) |
| K=8 | 322 | 1.4 | 26.7 | 0.54 | 173.7 | -33.4 | 0.661 (157) |
| K=6 | 320 | 1.39 | 26.9 | 0.549 | 175.8 | -31.4 | 0.683 (155) |
| K=4 | 308 | 1.34 | 27.3 | 0.56 | 172.6 | -26.1 | 0.657 (148) |
| K=2 | 251 | 1.09 | 27.5 | 0.535 | 134.3 | -23.3 | 0.472 (126) |

## Reading guide

Wider stops cut the fee fraction (fewer R but lower MaxDD); the concurrency cap is the real drawdown lever — capping correlated knife-catches during market-wide crashes shrinks MaxDD while barely touching avg R. Net account DD% = MaxDD_R x risk-per-trade; with a cap of K and total heat H%, risk-per-trade = H/K.
