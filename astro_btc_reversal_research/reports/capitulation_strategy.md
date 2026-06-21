# Capitulation Strategy: Longs + Shorts, Fixed vs Scaled Exit

15m | 2022-01-01 -> 2026-06-01 (~230.3 wk) | sweep 1440 bars | early-week | |funding_z|>=1.0 | costs 11.0 bps RT.
Long = spring + funding<=-1.0; Short = upthrust + funding>=+1.0. Fixed = 30R; Scaled = 25%@4R / 50%@12R / 25%@30R, stop->BE after first partial.

| Book | Trades | Trades/wk | Win % | Avg R | Net R | PF | MaxDD R | Dev avg R | Holdout avg R (n) |
|---|---|---|---|---|---|---|---|---|---|
| long_fixed | 242 | 1.05 | 16.5 | 0.531 | 128.5 | 1.571 | -63.0 | -0.286 | 1.235 (130) |
| long_scaled | 243 | 1.06 | 29.6 | 0.593 | 144.1 | 1.756 | -31.3 | 0.261 | 0.877 (131) |
| short_fixed | 252 | 1.09 | 5.2 | -0.190 | -47.8 | 0.830 | -86.3 | -0.236 | -0.063 (68) |
| short_scaled | 254 | 1.10 | 19.7 | -0.387 | -98.3 | 0.589 | -100.8 | -0.318 | -0.576 (68) |
| combined_fixed | 494 | 2.15 | 10.7 | 0.163 | 80.7 | 1.160 | -86.3 | -0.255 | 0.789 (198) |
| combined_scaled | 497 | 2.16 | 24.5 | 0.092 | 45.8 | 1.107 | -103.5 | -0.100 | 0.380 (199) |

## Per-year (combined_scaled)

2022: -0.39R/66t, 2023: 0.14R/104t, 2024: -0.14R/128t, 2025: 0.31R/136t, 2026: 0.54R/63t

## Reading guide

Compare scaled vs fixed: scaled should LIFT win rate and SHRINK MaxDD (banking 4R partials, BE after TP1), trading some avg R for consistency. Do SHORTS add positive, holdout-positive trades? Combined cadence ~ longs + shorts per week. Watch dev vs holdout and the bear year(s).
