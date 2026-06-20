# Astro Calendar Search (precision framing) - 4h

**Symbol:** BTCUSDT | 2021-01-01 -> 2026-06-01 (11863 candles)
**Pivots:** ATR threshold 4.0 (369 pivots) | tolerance window +/-2 candles (medium) | orb 2.0 deg
**Baseline hit rate** (random window contains a pivot): 0.1513
**Hypotheses tested:** 238 | BH FDR alpha 0.05 -> 0 significant

Question: when a calendar fires, does a pivot land within the window more often than for random windows of the same count/width? Missing pivots is fine; **hit rate + lift** matter.

## Dark Pivot calendar (Moon-Pluto hard aspects 0/90/180/270)

- firings: 289 | **hit rate: 0.1696** | baseline 0.1513 | lift 1.1205 | coverage 0.1218
- binomial p 0.21457 | random-calendar p 0.21956
- holdout (2025+): hit rate 0.1447 vs baseline 0.1489 (lift 0.9723)
- shifted-calendar controls (real should beat these): +18b=0.1592, +42b=0.1493, +78b=0.1498, +126b=0.1573, +222b=0.1690, +498b=0.1408

## Top single pair x aspect calendars by hit rate

| Pair | Aspect | Firings | Hit rate | Lift | Coverage | Binom p | Rand p | BH sig | Holdout hit |
|---|---|---|---|---|---|---|---|---|---|
| venus-neptune | 60 | 6 | 0.6667 | 4.4059 | 0.0025 | 0.00608 | 0.00599 | no | 0.5000 |
| mercury-saturn | 90 | 6 | 0.6667 | 4.4059 | 0.0025 | 0.00608 | 0.00599 | no | 1.0000 |
| sun-saturn | 90 | 5 | 0.6000 | 3.9653 | 0.0021 | 0.02726 | 0.04192 | no | 1.0000 |
| mercury-pluto | 90 | 6 | 0.5000 | 3.3045 | 0.0025 | 0.04843 | 0.04192 | no | 0.5000 |
| sun-saturn | 30 | 6 | 0.5000 | 3.3045 | 0.0025 | 0.04843 | 0.04192 | no | 0.5000 |
| sun-saturn | 0 | 6 | 0.5000 | 3.3045 | 0.0025 | 0.04843 | 0.04192 | no | 0.5000 |
| venus-neptune | 45 | 6 | 0.5000 | 3.3045 | 0.0025 | 0.04843 | 0.04192 | no | 0.5000 |
| venus-uranus | 30 | 6 | 0.5000 | 3.3045 | 0.0025 | 0.04843 | 0.04192 | no | 0.0000 |
| mercury-uranus | 90 | 7 | 0.4286 | 2.8324 | 0.0030 | 0.07539 | 0.07385 | no | 0.0000 |
| mercury-uranus | 60 | 5 | 0.4000 | 2.6436 | 0.0021 | 0.16721 | 0.15768 | no | 0.0000 |
| mars-uranus | 30 | 5 | 0.4000 | 2.6436 | 0.0021 | 0.16721 | 0.15768 | no | n/a |
| sun-jupiter | 135 | 5 | 0.4000 | 2.6436 | 0.0021 | 0.16721 | 0.15768 | no | 1.0000 |
| venus-jupiter | 150 | 5 | 0.4000 | 2.6436 | 0.0021 | 0.16721 | 0.15768 | no | 1.0000 |
| venus-mars | 45 | 5 | 0.4000 | 2.6436 | 0.0021 | 0.16721 | 0.15768 | no | 0.0000 |
| mars-saturn | 135 | 5 | 0.4000 | 2.6436 | 0.0021 | 0.16721 | 0.15768 | no | 0.0000 |
| mercury-venus | 30 | 16 | 0.3750 | 2.4783 | 0.0067 | 0.02450 | 0.03194 | no | 0.0000 |
| mercury-venus | 60 | 6 | 0.3333 | 2.2030 | 0.0025 | 0.22660 | 0.20160 | no | 0.0000 |
| mercury-saturn | 135 | 6 | 0.3333 | 2.2030 | 0.0025 | 0.22660 | 0.20160 | no | 0.0000 |
| venus-neptune | 30 | 6 | 0.3333 | 2.2030 | 0.0025 | 0.22660 | 0.20160 | no | 0.5000 |
| mercury-uranus | 0 | 6 | 0.3333 | 2.2030 | 0.0025 | 0.22660 | 0.20160 | no | 0.0000 |
| sun-neptune | 60 | 6 | 0.3333 | 2.2030 | 0.0025 | 0.22660 | 0.20160 | no | 0.5000 |
| sun-neptune | 0 | 6 | 0.3333 | 2.2030 | 0.0025 | 0.22660 | 0.20160 | no | 1.0000 |
| sun-pluto | 30 | 6 | 0.3333 | 2.2030 | 0.0025 | 0.22660 | 0.20160 | no | 0.0000 |
| venus-uranus | 45 | 6 | 0.3333 | 2.2030 | 0.0024 | 0.22660 | 0.20160 | no | 0.5000 |
| venus-pluto | 45 | 6 | 0.3333 | 2.2030 | 0.0025 | 0.22660 | 0.20160 | no | 0.5000 |
| venus-neptune | 150 | 6 | 0.3333 | 2.2030 | 0.0025 | 0.22660 | 0.20160 | no | 0.0000 |
| mercury-pluto | 150 | 6 | 0.3333 | 2.2030 | 0.0025 | 0.22660 | 0.20160 | no | 0.0000 |
| venus-pluto | 90 | 6 | 0.3333 | 2.2030 | 0.0025 | 0.22660 | 0.20160 | no | 0.5000 |
| venus-saturn | 0 | 7 | 0.2857 | 1.8883 | 0.0030 | 0.28708 | 0.28343 | no | 0.3333 |
| mars-saturn | 120 | 7 | 0.2857 | 1.8883 | 0.0030 | 0.28708 | 0.28343 | no | 0.0000 |

## Aspect-confluence calendars

| Min simultaneous aspects | Firings | Hit rate | Baseline | Lift | Binom p | Holdout hit |
|---|---|---|---|---|---|---|
| >= 2 | 759 | 0.1173 | 0.1513 | 0.7750 | 0.99702 | 0.0860 |
| >= 3 | 416 | 0.1755 | 0.1513 | 1.1597 | 0.09729 | 0.1345 |
| >= 4 | 174 | 0.1609 | 0.1513 | 1.0635 | 0.39324 | 0.1071 |

## Reading guide

A calendar is credible only if its hit rate clearly exceeds the baseline (lift > 1) with a small binomial/random p, **and** the lift survives on the holdout column and beats the shifted controls. High lift on very few firings is noise - weight the firing count and FDR column.
