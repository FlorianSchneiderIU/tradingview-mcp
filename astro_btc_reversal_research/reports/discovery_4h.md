# Aspect Discovery (Milestone 2) - 4h

**Symbol:** BTCUSDT | 2021-01-01 -> 2026-06-01 (11863 candles)
**Pivot window:** +/-2 candles (medium) | orb 2.0 deg | pivot ATR threshold 2.0
**Hypotheses tested:** 322 (pair x aspect) | BH FDR alpha 0.05 -> 12 significant

## Top aspect (pair x angle) by pivot-window lift

| Pair | Aspect | In-window bars | Lift | Binom p | Rand p | BH sig | Holdout lift |
|---|---|---|---|---|---|---|---|
| venus-mars | 180 | 34 | 1.4337 | 0.00691 | 0.01493 | no | n/a |
| mars-jupiter | 135 | 118 | 1.3880 | 0.00001 | 0.00498 | yes | 1.4735 |
| mercury-uranus | 180 | 123 | 1.3475 | 0.00005 | 0.00498 | yes | 1.6167 |
| sun-uranus | 90 | 125 | 1.3415 | 0.00005 | 0.00498 | yes | 1.2071 |
| mercury-saturn | 180 | 67 | 1.3387 | 0.00296 | 0.00498 | no | 1.7289 |
| sun-mars | 180 | 35 | 1.3371 | 0.02922 | 0.03980 | no | 1.3314 |
| moon-neptune | 90 | 135 | 1.3288 | 0.00005 | 0.00498 | yes | 1.1199 |
| mercury-pluto | 180 | 68 | 1.3190 | 0.00462 | 0.00995 | no | 0.4715 |
| mercury-saturn | 90 | 134 | 1.3096 | 0.00014 | 0.00498 | yes | 1.4146 |
| mars-jupiter | 120 | 138 | 1.2999 | 0.00018 | 0.00498 | yes | 1.2574 |
| mars-uranus | 60 | 294 | 1.2866 | 0.00000 | 0.00498 | yes | 1.4242 |
| venus-saturn | 45 | 131 | 1.2801 | 0.00062 | 0.00498 | yes | 1.2711 |
| mercury-neptune | 60 | 150 | 1.2609 | 0.00064 | 0.00995 | yes | 1.2003 |
| mercury-pluto | 90 | 83 | 1.2451 | 0.01414 | 0.00995 | no | 1.3058 |
| moon-pluto | 150 | 138 | 1.2434 | 0.00208 | 0.00498 | no | 0.8907 |
| mercury-saturn | 0 | 138 | 1.2293 | 0.00354 | 0.00995 | no | 1.3952 |
| mercury-neptune | 30 | 142 | 1.2221 | 0.00408 | 0.00498 | no | 0.8757 |
| venus-saturn | 180 | 100 | 1.2089 | 0.02009 | 0.03980 | no | 1.4670 |
| mars-jupiter | 90 | 207 | 1.2057 | 0.00143 | 0.00498 | yes | 1.3580 |
| moon-saturn | 45 | 120 | 1.2024 | 0.01417 | 0.01493 | no | 1.3273 |
| sun-saturn | 0 | 164 | 1.2008 | 0.00506 | 0.00995 | no | 1.3472 |
| jupiter-saturn | 0 | 44 | 1.1965 | 0.11741 | 0.09950 | no | n/a |
| venus-jupiter | 30 | 119 | 1.1961 | 0.01736 | 0.02985 | no | 0.7380 |
| mars-uranus | 45 | 139 | 1.1924 | 0.01220 | 0.00995 | no | n/a |
| mars-jupiter | 30 | 87 | 1.1879 | 0.04501 | 0.03980 | no | n/a |

Baseline pivot-window rate: 0.5129.
Lift = P(pivot within window | in aspect window) / baseline. Holdout (2025+) is reported, never used to rank. Treat large lifts with tiny in-window-bar counts as noise until they survive the holdout and random-calendar columns.
