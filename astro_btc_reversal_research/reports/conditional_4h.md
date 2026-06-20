# Direction-Conditional Calendar Search - 4h

**Symbol:** BTCUSDT | 2021-01-01 -> 2026-06-01 (11863 candles)
**Pivots:** ATR threshold 4.0 | tolerance +/-2 candles | orb 2.0 deg
**Context:** move >= 1.0 ATR over 6 bars into the aspect. Dump->expect BOTTOM, pump->expect TOP.

Tests proposal H5 / the Dark Pivot thesis: does an aspect firing add timing info *beyond* the price context? Baseline = random bars from the SAME context, so lift isolates the astro part.

## dump days -> pivot LOW

Context bars: 2602 | context baseline hit rate 0.1864 | hypotheses 84 | BH-FDR significant: 0

| Pair | Aspect | Firings | Hit rate | Lift | Binom p | Rand p | BH sig | Holdout hit |
|---|---|---|---|---|---|---|---|---|
| moon-mercury | 60 | 18 | 0.4444 | 2.3844 | 0.01067 | 0.01299 | no | 0.3333 |
| moon-jupiter | 120 | 12 | 0.4167 | 2.2354 | 0.05585 | 0.05395 | no | 0.2500 |
| moon-pluto | 30 | 12 | 0.4167 | 2.2354 | 0.05585 | 0.05395 | no | 0.0000 |
| moon-venus | 45 | 17 | 0.4118 | 2.2091 | 0.02639 | 0.02597 | no | 0.5000 |
| mercury-venus | 30 | 5 | 0.4000 | 2.1460 | 0.23512 | 0.24076 | no | 0.0000 |
| moon-jupiter | 60 | 23 | 0.3913 | 2.0993 | 0.01770 | 0.01898 | no | 0.4286 |
| moon-venus | 60 | 15 | 0.3333 | 1.7883 | 0.13099 | 0.13986 | no | 0.0000 |
| moon-venus | 180 | 12 | 0.3333 | 1.7883 | 0.17057 | 0.17782 | no | 0.5000 |
| moon-mercury | 120 | 18 | 0.3333 | 1.7883 | 0.10190 | 0.09191 | no | 0.2857 |
| sun-moon | 180 | 13 | 0.3077 | 1.6508 | 0.21206 | 0.22178 | no | 0.0000 |
| moon-venus | 135 | 17 | 0.2941 | 1.5779 | 0.19720 | 0.20679 | no | 0.4000 |
| moon-uranus | 45 | 17 | 0.2941 | 1.5779 | 0.19720 | 0.20679 | no | 0.0000 |
| moon-mercury | 0 | 14 | 0.2857 | 1.5328 | 0.25593 | 0.25974 | no | 0.0000 |
| moon-venus | 30 | 14 | 0.2857 | 1.5328 | 0.25593 | 0.25974 | no | 0.7500 |
| moon-neptune | 135 | 14 | 0.2857 | 1.5328 | 0.25593 | 0.25974 | no | 0.3333 |
| sun-mercury | 0 | 7 | 0.2857 | 1.5328 | 0.38555 | 0.38861 | no | 0.0000 |
| moon-venus | 120 | 18 | 0.2778 | 1.4903 | 0.23385 | 0.23776 | no | 0.5000 |
| moon-mercury | 150 | 18 | 0.2778 | 1.4903 | 0.23385 | 0.23776 | no | 0.5000 |
| moon-saturn | 120 | 18 | 0.2778 | 1.4903 | 0.23385 | 0.23776 | no | 0.2857 |
| sun-moon | 90 | 15 | 0.2667 | 1.4307 | 0.30137 | 0.31868 | no | 0.0000 |
| moon-mars | 0 | 15 | 0.2667 | 1.4307 | 0.30137 | 0.31868 | no | 0.5000 |
| moon-pluto | 0 | 19 | 0.2632 | 1.4118 | 0.27220 | 0.27972 | no | 0.0000 |
| moon-neptune | 30 | 16 | 0.2500 | 1.3412 | 0.34758 | 0.35864 | no | 0.3333 |
| moon-jupiter | 45 | 12 | 0.2500 | 1.3412 | 0.39313 | 0.40659 | no | 0.0000 |
| moon-saturn | 60 | 12 | 0.2500 | 1.3412 | 0.39313 | 0.40659 | no | 0.5000 |

Dark Pivot (Moon-Pluto hard) in this context: firings 77, hit 0.2338 vs baseline 0.1864 (lift 1.2541, binom p 0.17706, holdout lift 0.7722).

## pump days -> pivot HIGH

Context bars: 2782 | context baseline hit rate 0.1776 | hypotheses 83 | BH-FDR significant: 0

| Pair | Aspect | Firings | Hit rate | Lift | Binom p | Rand p | BH sig | Holdout hit |
|---|---|---|---|---|---|---|---|---|
| sun-mercury | 0 | 9 | 0.4444 | 2.5029 | 0.05892 | 0.04995 | no | 0.5000 |
| moon-mercury | 30 | 14 | 0.4286 | 2.4135 | 0.02563 | 0.01798 | no | 0.6000 |
| moon-mars | 0 | 12 | 0.4167 | 2.3465 | 0.04645 | 0.03896 | no | 0.5000 |
| moon-jupiter | 0 | 17 | 0.4118 | 2.3189 | 0.02053 | 0.01698 | no | 0.1429 |
| moon-venus | 30 | 18 | 0.3889 | 2.1901 | 0.02856 | 0.02597 | no | 0.6000 |
| moon-mars | 180 | 13 | 0.3846 | 2.1660 | 0.06475 | 0.04995 | no | 0.2000 |
| sun-moon | 0 | 21 | 0.3810 | 2.1454 | 0.02240 | 0.01998 | no | 0.5000 |
| moon-mercury | 45 | 11 | 0.3636 | 2.0478 | 0.11513 | 0.10689 | no | 0.0000 |
| moon-uranus | 120 | 17 | 0.3529 | 1.9876 | 0.06571 | 0.06294 | no | 0.5000 |
| moon-venus | 60 | 9 | 0.3333 | 1.8772 | 0.20446 | 0.18981 | no | 0.0000 |
| moon-uranus | 150 | 18 | 0.3333 | 1.8772 | 0.08428 | 0.07493 | no | 0.0000 |
| moon-uranus | 45 | 16 | 0.3125 | 1.7599 | 0.13955 | 0.13087 | no | 0.4000 |
| moon-neptune | 30 | 16 | 0.3125 | 1.7599 | 0.13955 | 0.13087 | no | 0.4000 |
| moon-mercury | 60 | 10 | 0.3000 | 1.6895 | 0.25576 | 0.25075 | no | 0.3333 |
| sun-moon | 120 | 20 | 0.3000 | 1.6895 | 0.12905 | 0.12388 | no | 0.4286 |
| moon-saturn | 30 | 20 | 0.3000 | 1.6895 | 0.12905 | 0.12388 | no | 0.6000 |
| moon-mars | 150 | 17 | 0.2941 | 1.6563 | 0.17032 | 0.15684 | no | 0.6667 |
| moon-pluto | 135 | 17 | 0.2941 | 1.6563 | 0.17032 | 0.15684 | no | 0.2000 |
| moon-jupiter | 90 | 17 | 0.2941 | 1.6563 | 0.17032 | 0.15684 | no | 0.0000 |
| moon-mars | 120 | 17 | 0.2941 | 1.6563 | 0.17032 | 0.15684 | no | 0.3333 |
| moon-saturn | 90 | 21 | 0.2857 | 1.6090 | 0.15494 | 0.15485 | no | 0.0000 |
| moon-neptune | 180 | 18 | 0.2778 | 1.5643 | 0.20341 | 0.19381 | no | 0.5000 |
| moon-venus | 135 | 15 | 0.2667 | 1.5018 | 0.26951 | 0.27073 | no | 0.0000 |
| moon-pluto | 90 | 15 | 0.2667 | 1.5018 | 0.26951 | 0.27073 | no | 0.3333 |
| sun-moon | 45 | 16 | 0.2500 | 1.4079 | 0.31283 | 0.32268 | no | 0.1667 |

Dark Pivot (Moon-Pluto hard) in this context: firings 71, hit 0.1690 vs baseline 0.1776 (lift 0.9518, binom p 0.62303, holdout lift 0.6728).

## Reading guide

Lift > 1 here means the aspect beats *random same-context bars* (e.g. random dumps), not just the unconditional baseline. Credible only if it survives BH-FDR and the holdout. Few firings (aspect AND context) -> expect noise.
