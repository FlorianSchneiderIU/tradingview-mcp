# Direction-Conditional Calendar Search - 1d

**Symbol:** BTCUSDT | 2021-01-01 -> 2026-06-01 (1978 candles)
**Pivots:** ATR threshold 3.0 | tolerance +/-3 candles | orb 3.0 deg
**Context:** move >= 1.0 ATR over 2 bars into the aspect. Dump->expect BOTTOM, pump->expect TOP.

Tests proposal H5 / the Dark Pivot thesis: does an aspect firing add timing info *beyond* the price context? Baseline = random bars from the SAME context, so lift isolates the astro part.

## dump days -> pivot LOW

Context bars: 210 | context baseline hit rate 0.3190 | hypotheses 31 | BH-FDR significant: 0

| Pair | Aspect | Firings | Hit rate | Lift | Binom p | Rand p | BH sig | Holdout hit |
|---|---|---|---|---|---|---|---|---|
| moon-mars | 30 | 5 | 0.6000 | 1.8806 | 0.18918 | 0.17982 | no | 0.0000 |
| moon-jupiter | 120 | 5 | 0.6000 | 1.8806 | 0.18918 | 0.17982 | no | 1.0000 |
| moon-venus | 135 | 7 | 0.5714 | 1.7910 | 0.15206 | 0.13886 | no | 0.5000 |
| moon-mars | 0 | 6 | 0.5000 | 1.5672 | 0.29172 | 0.27872 | no | 0.6667 |
| moon-uranus | 60 | 7 | 0.4286 | 1.3433 | 0.39646 | 0.36663 | no | 0.5000 |
| moon-venus | 120 | 5 | 0.4000 | 1.2537 | 0.51059 | 0.49251 | no | 0.6667 |
| moon-saturn | 150 | 5 | 0.4000 | 1.2537 | 0.51059 | 0.49251 | no | n/a |
| moon-uranus | 45 | 5 | 0.4000 | 1.2537 | 0.51059 | 0.49251 | no | 0.0000 |
| moon-uranus | 135 | 5 | 0.4000 | 1.2537 | 0.51059 | 0.49251 | no | 0.3333 |
| moon-uranus | 150 | 6 | 0.3333 | 1.0448 | 0.62002 | 0.58741 | no | 0.2500 |
| moon-pluto | 150 | 6 | 0.3333 | 1.0448 | 0.62002 | 0.58741 | no | 0.0000 |
| sun-mercury | 0 | 9 | 0.3333 | 1.0448 | 0.58698 | 0.54845 | no | 0.5000 |
| sun-moon | 30 | 6 | 0.3333 | 1.0448 | 0.62002 | 0.58741 | no | 1.0000 |
| moon-venus | 60 | 6 | 0.3333 | 1.0448 | 0.62002 | 0.58741 | no | 0.5000 |
| moon-mars | 45 | 7 | 0.2857 | 0.8955 | 0.70944 | 0.69530 | no | 0.0000 |
| moon-saturn | 90 | 7 | 0.2857 | 0.8955 | 0.70944 | 0.69530 | no | 0.0000 |
| moon-neptune | 45 | 8 | 0.2500 | 0.7836 | 0.78048 | 0.77023 | no | 0.0000 |
| moon-pluto | 90 | 5 | 0.2000 | 0.6269 | 0.85359 | 0.84416 | no | 0.0000 |
| moon-neptune | 60 | 5 | 0.2000 | 0.6269 | 0.85359 | 0.84416 | no | 0.0000 |
| moon-venus | 30 | 5 | 0.2000 | 0.6269 | 0.85359 | 0.84416 | no | 0.0000 |
| moon-neptune | 180 | 5 | 0.2000 | 0.6269 | 0.85359 | 0.84416 | no | 0.0000 |
| moon-uranus | 30 | 6 | 0.1667 | 0.5224 | 0.90030 | 0.90410 | no | 0.0000 |
| moon-mercury | 120 | 6 | 0.1667 | 0.5224 | 0.90030 | 0.90410 | no | 0.2500 |
| moon-jupiter | 30 | 6 | 0.1667 | 0.5224 | 0.90030 | 0.90410 | no | n/a |
| moon-mars | 60 | 6 | 0.1667 | 0.5224 | 0.90030 | 0.90410 | no | 0.0000 |

Dark Pivot (Moon-Pluto hard) in this context: firings 14, hit 0.3571 vs baseline 0.3190 (lift 1.1194, binom p 0.47817, holdout lift 1.6667).

## pump days -> pivot HIGH

Context bars: 227 | context baseline hit rate 0.3612 | hypotheses 22 | BH-FDR significant: 0

| Pair | Aspect | Firings | Hit rate | Lift | Binom p | Rand p | BH sig | Holdout hit |
|---|---|---|---|---|---|---|---|---|
| moon-mercury | 150 | 5 | 0.8000 | 2.2146 | 0.06053 | 0.05694 | no | 0.5000 |
| sun-moon | 135 | 6 | 0.6667 | 1.8455 | 0.13001 | 0.14386 | no | 0.6667 |
| sun-moon | 120 | 5 | 0.6000 | 1.6610 | 0.25286 | 0.25375 | no | 0.0000 |
| moon-mars | 60 | 5 | 0.6000 | 1.6610 | 0.25286 | 0.25375 | no | 0.0000 |
| moon-mars | 120 | 6 | 0.5000 | 1.3841 | 0.37572 | 0.36863 | no | 0.0000 |
| moon-uranus | 120 | 8 | 0.5000 | 1.3841 | 0.31799 | 0.32767 | no | 0.0000 |
| moon-uranus | 135 | 6 | 0.5000 | 1.3841 | 0.37572 | 0.36863 | no | 0.0000 |
| moon-venus | 30 | 7 | 0.4286 | 1.1864 | 0.49343 | 0.49151 | no | 1.0000 |
| moon-venus | 150 | 5 | 0.4000 | 1.1073 | 0.59296 | 0.59540 | no | 1.0000 |
| moon-pluto | 90 | 5 | 0.4000 | 1.1073 | 0.59296 | 0.59540 | no | 0.0000 |
| moon-saturn | 180 | 5 | 0.4000 | 1.1073 | 0.59296 | 0.59540 | no | 0.5000 |
| moon-venus | 120 | 5 | 0.4000 | 1.1073 | 0.59296 | 0.59540 | no | n/a |
| moon-venus | 60 | 5 | 0.4000 | 1.1073 | 0.59296 | 0.59540 | no | n/a |
| moon-jupiter | 135 | 8 | 0.3750 | 1.0381 | 0.59870 | 0.60340 | no | n/a |
| moon-pluto | 0 | 6 | 0.3333 | 0.9228 | 0.70158 | 0.68631 | no | 0.0000 |
| moon-pluto | 45 | 6 | 0.3333 | 0.9228 | 0.70158 | 0.68631 | no | 1.0000 |
| moon-pluto | 30 | 7 | 0.2857 | 0.7909 | 0.78484 | 0.78422 | no | 0.5000 |
| moon-jupiter | 150 | 8 | 0.2500 | 0.6921 | 0.84689 | 0.83816 | no | 0.5000 |
| moon-mercury | 135 | 5 | 0.2000 | 0.5537 | 0.89366 | 0.89311 | no | 0.0000 |
| moon-mercury | 120 | 5 | 0.2000 | 0.5537 | 0.89366 | 0.89311 | no | 0.0000 |
| sun-moon | 180 | 5 | 0.2000 | 0.5537 | 0.89366 | 0.89311 | no | 1.0000 |
| moon-saturn | 45 | 5 | 0.2000 | 0.5537 | 0.89366 | 0.89311 | no | 0.0000 |

Dark Pivot (Moon-Pluto hard) in this context: firings 20, hit 0.3500 vs baseline 0.3612 (lift 0.9689, binom p 0.62408, holdout lift 0.0000).

## Reading guide

Lift > 1 here means the aspect beats *random same-context bars* (e.g. random dumps), not just the unconditional baseline. Credible only if it survives BH-FDR and the holdout. Few firings (aspect AND context) -> expect noise.
