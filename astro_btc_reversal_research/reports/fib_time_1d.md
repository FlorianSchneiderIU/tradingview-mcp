# Fibonacci-TIME event study - 1d

**BTCUSDT** | 1978 bars | 111 pivots (ATR 3.0) | tolerance +/-3 bars | baseline pivot-window rate 0.352.

Question: do pivots land at Fibonacci TIME projections more than at ordinary (non-Fib) ratios of the same prior swing, and more than random? **Decisive = fib_ratio_ex1 beats nonfib_ratio AND random** (and is more than the repeat_1x persistence benchmark).

| Calendar | Firings | Hit rate | Baseline | Lift | Binom p | Rand p | Holdout lift |
|---|---|---|---|---|---|---|---|
| fib_ratio | 466 | 0.380 | 0.352 | 1.078 | 0.1170 | 0.0979 | 1.130 |
| fib_ratio_ex1 | 379 | 0.393 | 0.352 | 1.116 | 0.0548 | 0.0380 | 1.136 |
| repeat_1x | 108 | 0.398 | 0.352 | 1.130 | 0.1849 | 0.1888 | 1.256 |
| nonfib_ratio | 816 | 0.401 | 0.352 | 1.137 | 0.0023 | 0.0010 | 1.117 |
| fib_zone | 1029 | 0.467 | 0.352 | 1.327 | 0.0000 | 0.0010 | 1.281 |
| nonfib_zone | 1072 | 0.310 | 0.352 | 0.879 | 0.9986 | 1.0000 | 0.948 |
| random | 466 | 0.345 | 0.352 | 0.980 | 0.6388 | 0.6613 | 0.826 |

## Fib(ex-1.0) vs placebo / persistence (two-proportion p-values)

- fib_ex1_vs_nonfib_ratio: p = 0.8029
- fib_ex1_vs_repeat_1x: p = 0.9251
- fib_ex1_vs_random: p = 0.1529

## Reading guide

If `fib_ratio_ex1` lift ~= `nonfib_ratio` lift, the clustering is just swing-duration persistence, not Fibonacci. If `fib_ratio_ex1` ~ `random` (lift ~1), there is no time effect at all. A real Fib-time edge shows fib_ratio_ex1 lift clearly > nonfib and > random with small two-proportion p-values and a holdout lift > 1.
