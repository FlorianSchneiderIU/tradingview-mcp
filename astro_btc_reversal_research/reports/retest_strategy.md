# Weekly-Low Retest Entry Strategy

**BTCUSDT** LTF 5m | 2021-01-01 -> 2026-06-01 (569377 bars)
Establish weekly low after a 1.5% rally; enter on a spring retesting it within 0.5%; stop 0.05 ATR below; costs 11.0 bps RT.

## R available after entry (excursion pass)

| Book | signals | avg MFE R | 5R | 10R | 20R | 30R |
|---|---|---|---|---|---|---|
| retest_earlywk | 295 | 2.859 | 0.118 | 0.055 | 0.027 | 0.018 |
| retest_anytime | 384 | 2.729 | 0.116 | 0.048 | 0.024 | 0.017 |
| first_deep_spring_earlywk | 126 | 4.648 | 0.168 | 0.126 | 0.053 | 0.032 |
| ungated_local_spring | 2089 | 4.063 | 0.169 | 0.096 | 0.050 | 0.035 |

## Target 20R (net of costs)

| Book | Trades | Win % | Avg R | Net R | PF | MaxDD R | Holdout avg R (n) |
|---|---|---|---|---|---|---|---|
| retest_earlywk | 222 | 3.2 | -0.739 | -164.0 | 0.443 | -182.8 | -0.993 (67) |
| retest_anytime | 296 | 2.7 | -0.831 | -246.0 | 0.366 | -269.4 | -1.023 (89) |
| first_deep_spring_earlywk | 95 | 11.6 | 0.488 | 46.3 | 1.451 | -24.2 | 0.920 (33) |
| ungated_local_spring | 1243 | 5.7 | -0.300 | -372.6 | 0.766 | -426.6 | -0.423 (378) |

retest_earlywk per-year: 2021 -0.68R/39t, 2022 -1.21R/39t, 2023 0.16R/29t, 2024 -0.59R/48t, 2025 -0.86R/51t, 2026 -1.42R/16t

## Target 30R (net of costs)

| Book | Trades | Win % | Avg R | Net R | PF | MaxDD R | Holdout avg R (n) |
|---|---|---|---|---|---|---|---|
| retest_earlywk | 222 | 2.7 | -0.678 | -150.6 | 0.491 | -186.3 | -1.157 (67) |
| retest_anytime | 296 | 2.4 | -0.752 | -222.6 | 0.429 | -256.1 | -1.146 (89) |
| first_deep_spring_earlywk | 95 | 10.5 | 0.519 | 49.3 | 1.475 | -44.6 | 1.526 (33) |
| ungated_local_spring | 1176 | 5.1 | -0.133 | -156.8 | 0.897 | -267.7 | -0.178 (354) |

retest_earlywk per-year: 2021 -0.42R/39t, 2022 -1.21R/39t, 2023 0.31R/29t, 2024 -0.38R/48t, 2025 -1.07R/51t, 2026 -1.42R/16t

## Reading guide

Does **retest_earlywk** beat **first_deep_spring_earlywk** on avg R and holdout? The retest should give a tighter stop (entry at a tested level) and thus higher R per win. Watch trade count - the retest is rarer; weigh expectancy against sample size and MaxDD.
