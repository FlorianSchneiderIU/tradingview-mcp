# Astro Calendar Search (precision framing) - 1h

**Symbol:** BTCUSDT | 2021-01-01 -> 2026-06-01 (47449 candles)
**Pivots:** ATR threshold 5.0 (960 pivots) | tolerance window +/-6 candles (medium) | orb 1.0 deg
**Baseline hit rate** (random window contains a pivot): 0.2520
**Hypotheses tested:** 238 | BH FDR alpha 0.05 -> 0 significant

Question: when a calendar fires, does a pivot land within the window more often than for random windows of the same count/width? Missing pivots is fine; **hit rate + lift** matter.

## Dark Pivot calendar (Moon-Pluto hard aspects 0/90/180/270)

- firings: 289 | **hit rate: 0.2561** | baseline 0.2520 | lift 1.0163 | coverage 0.0792
- binomial p 0.45862 | random-calendar p 0.46108
- holdout (2025+): hit rate 0.2105 vs baseline 0.2397 (lift 0.8782)
- shifted-calendar controls (real should beat these): +72b=0.2699, +168b=0.2431, +312b=0.2613, +504b=0.2168, +888b=0.2782, +1992b=0.2347

## Top single pair x aspect calendars by hit rate

| Pair | Aspect | Firings | Hit rate | Lift | Coverage | Binom p | Rand p | BH sig | Holdout hit |
|---|---|---|---|---|---|---|---|---|---|
| mars-uranus | 60 | 5 | 0.8000 | 3.1752 | 0.0014 | 0.01609 | 0.01198 | no | 0.5000 |
| sun-neptune | 60 | 6 | 0.6667 | 2.6460 | 0.0016 | 0.03864 | 0.03792 | no | 0.5000 |
| venus-neptune | 30 | 6 | 0.6667 | 2.6460 | 0.0016 | 0.03864 | 0.03792 | no | 0.5000 |
| sun-saturn | 0 | 6 | 0.6667 | 2.6460 | 0.0016 | 0.03864 | 0.03792 | no | 1.0000 |
| sun-saturn | 90 | 5 | 0.6000 | 2.3814 | 0.0014 | 0.10559 | 0.10180 | no | 1.0000 |
| sun-uranus | 120 | 5 | 0.6000 | 2.3814 | 0.0014 | 0.10559 | 0.10180 | no | 1.0000 |
| venus-mars | 45 | 5 | 0.6000 | 2.3814 | 0.0014 | 0.10559 | 0.10180 | no | 0.0000 |
| mercury-pluto | 180 | 5 | 0.6000 | 2.3814 | 0.0014 | 0.10559 | 0.10180 | no | 1.0000 |
| sun-uranus | 60 | 5 | 0.6000 | 2.3814 | 0.0014 | 0.10559 | 0.10180 | no | 0.0000 |
| venus-saturn | 150 | 5 | 0.6000 | 2.3814 | 0.0014 | 0.10559 | 0.10180 | no | 1.0000 |
| mercury-jupiter | 0 | 7 | 0.5714 | 2.2680 | 0.0019 | 0.07237 | 0.07186 | no | 0.0000 |
| mercury-neptune | 90 | 7 | 0.5714 | 2.2680 | 0.0019 | 0.07237 | 0.07186 | no | 1.0000 |
| sun-saturn | 30 | 6 | 0.5000 | 1.9845 | 0.0016 | 0.17254 | 0.14970 | no | 1.0000 |
| sun-venus | 30 | 6 | 0.5000 | 1.9845 | 0.0016 | 0.17254 | 0.14970 | no | 0.5000 |
| venus-pluto | 45 | 6 | 0.5000 | 1.9845 | 0.0016 | 0.17254 | 0.14970 | no | 1.0000 |
| mercury-pluto | 45 | 8 | 0.5000 | 1.9845 | 0.0022 | 0.11654 | 0.09581 | no | 0.7500 |
| venus-pluto | 120 | 6 | 0.5000 | 1.9845 | 0.0016 | 0.17254 | 0.14970 | no | 0.5000 |
| mercury-uranus | 90 | 7 | 0.4286 | 1.7010 | 0.0019 | 0.24766 | 0.23553 | no | 0.0000 |
| venus-neptune | 150 | 7 | 0.4286 | 1.7010 | 0.0019 | 0.24766 | 0.23553 | no | 0.0000 |
| mercury-neptune | 135 | 5 | 0.4000 | 1.5876 | 0.0014 | 0.37131 | 0.36527 | no | 1.0000 |
| mercury-saturn | 120 | 5 | 0.4000 | 1.5876 | 0.0014 | 0.37131 | 0.36527 | no | 0.0000 |
| mercury-saturn | 135 | 5 | 0.4000 | 1.5876 | 0.0014 | 0.37131 | 0.36527 | no | 0.0000 |
| sun-pluto | 180 | 5 | 0.4000 | 1.5876 | 0.0014 | 0.37131 | 0.36527 | no | 1.0000 |
| venus-uranus | 150 | 5 | 0.4000 | 1.5876 | 0.0014 | 0.37131 | 0.36527 | no | 0.0000 |
| mars-saturn | 135 | 5 | 0.4000 | 1.5876 | 0.0014 | 0.37131 | 0.36527 | no | 0.5000 |
| venus-neptune | 180 | 5 | 0.4000 | 1.5876 | 0.0014 | 0.37131 | 0.36527 | no | 0.0000 |
| venus-jupiter | 60 | 5 | 0.4000 | 1.5876 | 0.0014 | 0.37131 | 0.36527 | no | 0.0000 |
| venus-saturn | 180 | 5 | 0.4000 | 1.5876 | 0.0014 | 0.37131 | 0.36527 | no | 0.0000 |
| venus-jupiter | 150 | 5 | 0.4000 | 1.5876 | 0.0014 | 0.37131 | 0.36527 | no | 1.0000 |
| mercury-pluto | 150 | 5 | 0.4000 | 1.5876 | 0.0014 | 0.37131 | 0.36527 | no | 0.0000 |

## Aspect-confluence calendars

| Min simultaneous aspects | Firings | Hit rate | Baseline | Lift | Binom p | Holdout hit |
|---|---|---|---|---|---|---|
| >= 2 | 764 | 0.2736 | 0.2520 | 1.0858 | 0.09195 | 0.2500 |
| >= 3 | 207 | 0.2850 | 0.2520 | 1.1313 | 0.15487 | 0.2333 |
| >= 4 | 56 | 0.2500 | 0.2520 | 0.9922 | 0.56449 | 0.2273 |

## Reading guide

A calendar is credible only if its hit rate clearly exceeds the baseline (lift > 1) with a small binomial/random p, **and** the lift survives on the holdout column and beats the shifted controls. High lift on very few firings is noise - weight the firing count and FDR column.
