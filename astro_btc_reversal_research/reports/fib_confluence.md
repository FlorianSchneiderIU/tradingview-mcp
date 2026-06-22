# Fibonacci TIME + PRICE confluence on the 5m spring

**basket(18)** | 8879704 bars | 4745 deep-sweep springs (lookback 4320) | 192542 pivots | fib-time +/-12 bars | fib-price tol 0.05 of swing range.
Baseline (all springs) reach-20R: **0.398**. Profitability = scaled exit 4/12/30R, stop->BE, realistic fees (taker entry/stop, maker TP).

| Book | Springs | Avg R | Win % | PF | Net R | reach-20R | Holdout avg R (n) |
|---|---|---|---|---|---|---|---|
| all | 4745 | -0.161 | 0.190 | 0.83 | -765.4 | 0.398 | -0.216 (2008) |
| fib_time | 3512 | -0.191 | 0.187 | 0.80 | -672.4 | 0.387 | -0.270 (1417) |
| fib_price | 967 | -0.073 | 0.204 | 0.92 | -70.6 | 0.395 | -0.027 (392) |
| fib_both | 748 | -0.026 | 0.205 | 0.97 | -19.7 | 0.392 | 0.041 (299) |
| nonfib_both | 973 | 0.112 | 0.215 | 1.12 | 108.6 | 0.430 | 0.343 (378) |
| neither | 1014 | -0.042 | 0.196 | 0.96 | -42.2 | 0.434 | -0.056 (498) |

## Reading guide (trader's framing)

Useful as a filter only if **fib_both** (Fib time AND price) has a clearly higher **avg R** (net of costs) than **all** springs (the no-filter baseline) AND than **neither** (springs that fail the filter), and it holds on the holdout. We are NOT asking whether it beats a random/non-Fib set - only whether the confluence selects more profitable entries than trading every spring. Weight the sample size: the confluence is a deep filter, so n is small.
