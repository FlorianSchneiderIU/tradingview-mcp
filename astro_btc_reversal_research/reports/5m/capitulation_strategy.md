# Capitulation Strategy: Longs + Shorts, Fixed vs Scaled Exit

5m | 2022-01-01 -> 2026-06-01 (~230.3 wk) | sweep 4320 bars | early-week | |funding_z|>=1.0 | costs 11.0 bps RT.
Long = spring + funding<=-1.0; Short = upthrust + funding>=+1.0. Fixed = 30R; Scaled = 25%@4R / 50%@12R / 25%@30R, stop->BE after first partial.

| Book | Trades | Trades/wk | Win % | Avg R | Net R | PF | MaxDD R | Dev avg R | Holdout avg R (n) |
|---|---|---|---|---|---|---|---|---|---|
| long_fixed | 333 | 1.45 | 9.9 | 0.728 | 242.4 | 1.685 | -54.5 | 0.292 | 1.199 (160) |
| long_scaled | 337 | 1.46 | 26.7 | 0.588 | 198.2 | 1.679 | -26.1 | 0.504 | 0.678 (163) |
| short_fixed | 355 | 1.54 | 3.9 | -0.369 | -130.9 | 0.695 | -129.4 | -0.394 | -0.303 (98) |
| short_scaled | 357 | 1.55 | 20.2 | -0.370 | -132.1 | 0.632 | -130.5 | -0.286 | -0.589 (99) |
| combined_fixed | 688 | 2.99 | 6.8 | 0.162 | 111.5 | 1.142 | -139.1 | -0.118 | 0.629 (258) |
| combined_scaled | 694 | 3.01 | 23.3 | 0.095 | 66.1 | 1.102 | -132.1 | 0.032 | 0.199 (262) |

## Per-year (combined_scaled)

2022: 0.01R/116t, 2023: 0.06R/131t, 2024: 0.02R/185t, 2025: 0.47R/172t, 2026: -0.32R/90t

## Reading guide

Compare scaled vs fixed: scaled should LIFT win rate and SHRINK MaxDD (banking 4R partials, BE after TP1), trading some avg R for consistency. Do SHORTS add positive, holdout-positive trades? Combined cadence ~ longs + shorts per week. Watch dev vs holdout and the bear year(s).
