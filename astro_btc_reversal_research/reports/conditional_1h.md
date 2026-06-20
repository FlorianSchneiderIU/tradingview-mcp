# Direction-Conditional Calendar Search - 1h

**Symbol:** BTCUSDT | 2021-01-01 -> 2026-06-01 (47449 candles)
**Pivots:** ATR threshold 5.0 | tolerance +/-6 candles | orb 1.0 deg
**Context:** move >= 1.0 ATR over 24 bars into the aspect. Dump->expect BOTTOM, pump->expect TOP.

Tests proposal H5 / the Dark Pivot thesis: does an aspect firing add timing info *beyond* the price context? Baseline = random bars from the SAME context, so lift isolates the astro part.

## dump days -> pivot LOW

Context bars: 16534 | context baseline hit rate 0.2634 | hypotheses 88 | BH-FDR significant: 0

| Pair | Aspect | Firings | Hit rate | Lift | Binom p | Rand p | BH sig | Holdout hit |
|---|---|---|---|---|---|---|---|---|
| moon-venus | 60 | 21 | 0.4762 | 1.8079 | 0.02940 | 0.04396 | no | 0.2000 |
| moon-pluto | 30 | 18 | 0.4444 | 1.6874 | 0.07469 | 0.08092 | no | 0.5000 |
| mercury-venus | 0 | 7 | 0.4286 | 1.6271 | 0.27188 | 0.25974 | no | 0.5000 |
| moon-jupiter | 135 | 24 | 0.4167 | 1.5819 | 0.07466 | 0.07692 | no | 0.3333 |
| moon-saturn | 0 | 17 | 0.4118 | 1.5633 | 0.13394 | 0.12887 | no | 0.3333 |
| moon-mars | 150 | 27 | 0.4074 | 1.5467 | 0.07352 | 0.07493 | no | 0.7143 |
| moon-neptune | 180 | 32 | 0.4062 | 1.5424 | 0.05544 | 0.05495 | no | 0.2500 |
| sun-moon | 90 | 25 | 0.4000 | 1.5186 | 0.09609 | 0.10589 | no | 0.0000 |
| mercury-venus | 30 | 5 | 0.4000 | 1.5186 | 0.39543 | 0.39261 | no | 0.0000 |
| moon-pluto | 0 | 20 | 0.4000 | 1.5186 | 0.13001 | 0.14286 | no | 0.3333 |
| moon-venus | 135 | 24 | 0.3750 | 1.4237 | 0.15603 | 0.14785 | no | 0.2857 |
| moon-mars | 90 | 24 | 0.3750 | 1.4237 | 0.15603 | 0.14785 | no | 0.3750 |
| moon-pluto | 90 | 24 | 0.3750 | 1.4237 | 0.15603 | 0.14785 | no | 0.4286 |
| moon-mercury | 90 | 19 | 0.3684 | 1.3987 | 0.21331 | 0.22278 | no | 0.4286 |
| moon-mars | 0 | 30 | 0.3667 | 1.3921 | 0.14139 | 0.13387 | no | 0.1667 |
| moon-jupiter | 30 | 30 | 0.3667 | 1.3921 | 0.14139 | 0.13387 | no | 0.2500 |
| sun-moon | 45 | 22 | 0.3636 | 1.3806 | 0.20106 | 0.20879 | no | 0.4286 |
| sun-mercury | 0 | 14 | 0.3571 | 1.3559 | 0.29877 | 0.30270 | no | 0.2500 |
| moon-uranus | 135 | 31 | 0.3548 | 1.3472 | 0.16951 | 0.16384 | no | 0.2857 |
| moon-saturn | 120 | 23 | 0.3478 | 1.3205 | 0.24136 | 0.23876 | no | 0.2857 |
| moon-jupiter | 60 | 33 | 0.3333 | 1.2655 | 0.23287 | 0.22278 | no | 0.5000 |
| moon-mercury | 120 | 24 | 0.3333 | 1.2655 | 0.28402 | 0.28671 | no | 0.1250 |
| moon-venus | 120 | 21 | 0.3333 | 1.2655 | 0.30536 | 0.30669 | no | 0.2727 |
| moon-neptune | 120 | 24 | 0.3333 | 1.2655 | 0.28402 | 0.28671 | no | 0.2857 |
| moon-jupiter | 120 | 21 | 0.3333 | 1.2655 | 0.30536 | 0.30669 | no | 0.5000 |

Dark Pivot (Moon-Pluto hard) in this context: firings 92, hit 0.2935 vs baseline 0.2634 (lift 1.1142, binom p 0.29126, holdout lift 1.0389).

## pump days -> pivot HIGH

Context bars: 17680 | context baseline hit rate 0.2392 | hypotheses 87 | BH-FDR significant: 0

| Pair | Aspect | Firings | Hit rate | Lift | Binom p | Rand p | BH sig | Holdout hit |
|---|---|---|---|---|---|---|---|---|
| moon-uranus | 120 | 26 | 0.4615 | 1.9295 | 0.01083 | 0.00999 | no | 0.4000 |
| moon-uranus | 45 | 24 | 0.4583 | 1.9161 | 0.01540 | 0.01099 | no | 0.2857 |
| moon-mercury | 90 | 21 | 0.4286 | 1.7917 | 0.04350 | 0.03896 | no | 0.5714 |
| moon-pluto | 150 | 21 | 0.4286 | 1.7917 | 0.04350 | 0.03896 | no | 0.1429 |
| moon-neptune | 30 | 25 | 0.4000 | 1.6723 | 0.05485 | 0.05694 | no | 0.2857 |
| mercury-uranus | 90 | 5 | 0.4000 | 1.6723 | 0.34441 | 0.33467 | no | 0.0000 |
| venus-pluto | 120 | 5 | 0.4000 | 1.6723 | 0.34441 | 0.33467 | no | 0.0000 |
| moon-jupiter | 0 | 27 | 0.3704 | 1.5484 | 0.08900 | 0.08691 | no | 0.3000 |
| moon-mars | 0 | 22 | 0.3636 | 1.5202 | 0.13301 | 0.12587 | no | 0.4000 |
| sun-moon | 0 | 25 | 0.3600 | 1.5050 | 0.12083 | 0.12088 | no | 0.2500 |
| moon-mars | 120 | 25 | 0.3600 | 1.5050 | 0.12083 | 0.12088 | no | 0.4000 |
| moon-mercury | 30 | 20 | 0.3500 | 1.4632 | 0.18149 | 0.19381 | no | 0.0000 |
| sun-moon | 30 | 20 | 0.3500 | 1.4632 | 0.18149 | 0.19381 | no | 0.3333 |
| moon-jupiter | 45 | 29 | 0.3448 | 1.4416 | 0.13345 | 0.13886 | no | 0.1667 |
| moon-venus | 90 | 27 | 0.3333 | 1.3936 | 0.17661 | 0.18082 | no | 0.2500 |
| moon-pluto | 90 | 31 | 0.3226 | 1.3486 | 0.18754 | 0.19381 | no | 0.1429 |
| moon-saturn | 30 | 31 | 0.3226 | 1.3486 | 0.18754 | 0.19381 | no | 0.5000 |
| moon-mars | 30 | 25 | 0.3200 | 1.3378 | 0.23193 | 0.24076 | no | 0.2000 |
| moon-pluto | 135 | 22 | 0.3182 | 1.3302 | 0.25955 | 0.26074 | no | 0.0000 |
| moon-uranus | 60 | 23 | 0.3043 | 1.2724 | 0.30167 | 0.30969 | no | 0.1667 |
| sun-moon | 60 | 27 | 0.2963 | 1.2387 | 0.30861 | 0.31968 | no | 0.2000 |
| moon-saturn | 180 | 27 | 0.2963 | 1.2387 | 0.30861 | 0.31968 | no | 0.4286 |
| moon-jupiter | 90 | 27 | 0.2963 | 1.2387 | 0.30861 | 0.31968 | no | 0.2000 |
| moon-mercury | 45 | 24 | 0.2917 | 1.2194 | 0.34502 | 0.35964 | no | 0.1667 |
| sun-moon | 120 | 24 | 0.2917 | 1.2194 | 0.34502 | 0.35964 | no | 0.3333 |

Dark Pivot (Moon-Pluto hard) in this context: firings 110, hit 0.1636 vs baseline 0.2392 (lift 0.6841, binom p 0.97925, holdout lift 0.3123).

## Reading guide

Lift > 1 here means the aspect beats *random same-context bars* (e.g. random dumps), not just the unconditional baseline. Credible only if it survives BH-FDR and the holdout. Few firings (aspect AND context) -> expect noise.
