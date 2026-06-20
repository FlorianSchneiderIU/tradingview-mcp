# Astro Calendar Search (precision framing) - 1d

**Symbol:** BTCUSDT | 2021-01-01 -> 2026-06-01 (1978 candles)
**Pivots:** ATR threshold 3.0 (111 pivots) | tolerance window +/-3 candles (medium) | orb 3.0 deg
**Baseline hit rate** (random window contains a pivot): 0.3524
**Hypotheses tested:** 237 | BH FDR alpha 0.05 -> 0 significant

Question: when a calendar fires, does a pivot land within the window more often than for random windows of the same count/width? Missing pivots is fine; **hit rate + lift** matter.

## Dark Pivot calendar (Moon-Pluto hard aspects 0/90/180/270)

- firings: 127 | **hit rate: 0.3858** | baseline 0.3524 | lift 1.0949 | coverage 0.4408
- binomial p 0.24173 | random-calendar p 0.21956
- holdout (2025+): hit rate 0.3529 vs baseline 0.3540 (lift 0.9971)
- shifted-calendar controls (real should beat these): +3b=0.3701, +7b=0.3543, +13b=0.3858, +21b=0.3622, +37b=0.3200, +83b=0.4250

## Top single pair x aspect calendars by hit rate

| Pair | Aspect | Firings | Hit rate | Lift | Coverage | Binom p | Rand p | BH sig | Holdout hit |
|---|---|---|---|---|---|---|---|---|---|
| venus-saturn | 90 | 6 | 0.8333 | 2.3649 | 0.0212 | 0.02303 | 0.02395 | no | 0.5000 |
| venus-jupiter | 45 | 5 | 0.8000 | 2.2703 | 0.0177 | 0.05536 | 0.01996 | no | 1.0000 |
| mars-pluto | 135 | 5 | 0.8000 | 2.2703 | 0.0177 | 0.05536 | 0.04391 | no | n/a |
| sun-saturn | 180 | 5 | 0.8000 | 2.2703 | 0.0177 | 0.05536 | 0.02395 | no | 1.0000 |
| mercury-saturn | 0 | 7 | 0.7143 | 2.0271 | 0.0248 | 0.05721 | 0.07984 | no | 0.6667 |
| venus-neptune | 150 | 6 | 0.6667 | 1.8919 | 0.0212 | 0.12002 | 0.07186 | no | 1.0000 |
| mercury-uranus | 120 | 6 | 0.6667 | 1.8919 | 0.0212 | 0.12002 | 0.08184 | no | 1.0000 |
| mercury-saturn | 90 | 6 | 0.6667 | 1.8919 | 0.0212 | 0.12002 | 0.08583 | no | 1.0000 |
| venus-neptune | 30 | 6 | 0.6667 | 1.8919 | 0.0212 | 0.12002 | 0.10180 | no | 1.0000 |
| sun-saturn | 0 | 6 | 0.6667 | 1.8919 | 0.0212 | 0.12002 | 0.08583 | no | 0.5000 |
| sun-pluto | 45 | 6 | 0.6667 | 1.8919 | 0.0212 | 0.12002 | 0.08583 | no | 0.5000 |
| sun-saturn | 90 | 5 | 0.6000 | 1.7027 | 0.0177 | 0.23887 | 0.15768 | no | 1.0000 |
| sun-saturn | 135 | 5 | 0.6000 | 1.7027 | 0.0177 | 0.23887 | 0.15768 | no | 0.0000 |
| sun-uranus | 120 | 5 | 0.6000 | 1.7027 | 0.0177 | 0.23887 | 0.17365 | no | 1.0000 |
| sun-uranus | 135 | 5 | 0.6000 | 1.7027 | 0.0177 | 0.23887 | 0.17365 | no | 1.0000 |
| sun-neptune | 150 | 5 | 0.6000 | 1.7027 | 0.0177 | 0.23887 | 0.17565 | no | 0.0000 |
| sun-uranus | 90 | 5 | 0.6000 | 1.7027 | 0.0177 | 0.23887 | 0.17365 | no | 0.0000 |
| venus-saturn | 120 | 5 | 0.6000 | 1.7027 | 0.0177 | 0.23887 | 0.11976 | no | 0.0000 |
| mars-saturn | 135 | 5 | 0.6000 | 1.7027 | 0.0177 | 0.23887 | 0.17964 | no | 1.0000 |
| venus-saturn | 150 | 5 | 0.6000 | 1.7027 | 0.0177 | 0.23887 | 0.12176 | no | 1.0000 |
| mercury-saturn | 30 | 7 | 0.5714 | 1.6216 | 0.0248 | 0.20378 | 0.17964 | no | 0.5000 |
| mercury-uranus | 180 | 7 | 0.5714 | 1.6216 | 0.0248 | 0.20378 | 0.17764 | no | 0.6667 |
| sun-moon | 30 | 30 | 0.5667 | 1.6081 | 0.1062 | 0.01330 | 0.00798 | no | 0.6250 |
| mercury-venus | 30 | 16 | 0.5625 | 1.5963 | 0.0566 | 0.06980 | 0.04990 | no | 0.5000 |
| moon-venus | 0 | 30 | 0.5333 | 1.5135 | 0.1062 | 0.03206 | 0.02196 | no | 0.5556 |
| mercury-neptune | 45 | 8 | 0.5000 | 1.4189 | 0.0283 | 0.29871 | 0.31936 | no | 0.5000 |
| mercury-neptune | 120 | 6 | 0.5000 | 1.4189 | 0.0212 | 0.35772 | 0.32136 | no | 0.5000 |
| venus-pluto | 30 | 6 | 0.5000 | 1.4189 | 0.0212 | 0.35772 | 0.38523 | no | 0.5000 |
| mercury-neptune | 180 | 6 | 0.5000 | 1.4189 | 0.0212 | 0.35772 | 0.32934 | no | 1.0000 |
| venus-saturn | 45 | 6 | 0.5000 | 1.4189 | 0.0212 | 0.35772 | 0.38323 | no | 0.5000 |

## Aspect-confluence calendars

| Min simultaneous aspects | Firings | Hit rate | Baseline | Lift | Binom p | Holdout hit |
|---|---|---|---|---|---|---|
| >= 2 | 233 | 0.3133 | 0.3524 | 0.8891 | 0.90701 | 0.2391 |
| >= 3 | 232 | 0.3534 | 0.3524 | 1.0030 | 0.51107 | 0.3571 |
| >= 4 | 119 | 0.3109 | 0.3524 | 0.8824 | 0.85165 | 0.2432 |

## Reading guide

A calendar is credible only if its hit rate clearly exceeds the baseline (lift > 1) with a small binomial/random p, **and** the lift survives on the holdout column and beats the shifted controls. High lift on very few firings is noise - weight the firing count and FDR column.
