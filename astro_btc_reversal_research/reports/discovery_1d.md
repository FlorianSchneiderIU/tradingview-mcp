# Aspect Discovery (Milestone 2) - 1d

**Symbol:** BTCUSDT | 2021-01-01 -> 2026-06-01 (1978 candles)
**Pivot window:** +/-3 candles (medium) | orb 3.0 deg | pivot ATR threshold 2.0
**Hypotheses tested:** 222 (pair x aspect) | BH FDR alpha 0.05 -> 8 significant

## Top aspect (pair x angle) by pivot-window lift

| Pair | Aspect | In-window bars | Lift | Binom p | Rand p | BH sig | Holdout lift |
|---|---|---|---|---|---|---|---|
| mars-neptune | 45 | 30 | 1.5811 | 0.00000 | 0.00498 | yes | n/a |
| sun-pluto | 150 | 30 | 1.4230 | 0.00103 | 0.00498 | yes | 1.0737 |
| jupiter-saturn | 120 | 69 | 1.4207 | 0.00000 | 0.00498 | yes | 1.4472 |
| mars-uranus | 60 | 75 | 1.4125 | 0.00000 | 0.00498 | yes | 1.4411 |
| sun-uranus | 45 | 33 | 1.3895 | 0.00161 | 0.00498 | yes | 0.9203 |
| sun-neptune | 90 | 32 | 1.3835 | 0.00226 | 0.01493 | no | 1.3422 |
| sun-saturn | 180 | 30 | 1.3703 | 0.00439 | 0.00995 | no | 1.3422 |
| sun-pluto | 0 | 36 | 1.3615 | 0.00229 | 0.01493 | no | 1.6106 |
| mars-saturn | 90 | 34 | 1.3486 | 0.00432 | 0.00498 | no | n/a |
| mercury-jupiter | 120 | 46 | 1.3405 | 0.00118 | 0.00995 | yes | 1.5211 |
| venus-saturn | 90 | 31 | 1.3261 | 0.01080 | 0.00995 | no | 0.8053 |
| mars-jupiter | 45 | 68 | 1.3254 | 0.00017 | 0.00498 | yes | 1.3356 |
| sun-pluto | 45 | 36 | 1.3176 | 0.00745 | 0.01990 | no | 1.3422 |
| venus-jupiter | 90 | 58 | 1.3085 | 0.00099 | 0.00498 | yes | 1.6106 |
| mercury-neptune | 45 | 39 | 1.2973 | 0.00903 | 0.01990 | no | 0.8053 |
| mars-saturn | 150 | 32 | 1.2847 | 0.02299 | 0.02488 | no | 1.0249 |
| mars-neptune | 135 | 32 | 1.2847 | 0.02299 | 0.02488 | no | 0.9395 |
| sun-saturn | 0 | 41 | 1.2726 | 0.01379 | 0.02985 | no | 0.9911 |
| mars-jupiter | 120 | 35 | 1.2649 | 0.02630 | 0.03980 | no | 1.4316 |
| sun-uranus | 135 | 30 | 1.2649 | 0.03909 | 0.03483 | no | 1.6106 |
| moon-venus | 0 | 30 | 1.2649 | 0.03909 | 0.05473 | no | 1.4316 |
| mercury-uranus | 90 | 33 | 1.2457 | 0.04345 | 0.04478 | no | 1.6106 |
| mercury-venus | 60 | 56 | 1.2423 | 0.01043 | 0.01990 | no | 1.2079 |
| mars-saturn | 135 | 65 | 1.2406 | 0.00631 | 0.00995 | no | 1.3009 |
| venus-neptune | 0 | 54 | 1.2298 | 0.01653 | 0.02985 | no | 1.2790 |

Baseline pivot-window rate: 0.6325.
Lift = P(pivot within window | in aspect window) / baseline. Holdout (2025+) is reported, never used to rank. Treat large lifts with tiny in-window-bar counts as noise until they survive the holdout and random-calendar columns.
