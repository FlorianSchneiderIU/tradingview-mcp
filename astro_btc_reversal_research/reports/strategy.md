# High-RR Reversal Strategy (Milestone 4)

**BTCUSDT** LTF 1h / HTF 1d | 2021-01-01 -> 2026-06-01
Setup: daily dump (>= 1.0 ATR over 2d) opens a 4-day long-alert window; enter on a 1h sweep+reclaim of the prior-range low; stop 0.1 ATR below the sweep; costs 8.0 bps RT.
LTF bars 47449 | sweep signals 229 | gated signals 103 | active window 0.296 of time.

## R available after the trigger (excursion pass, big target)

Gated avg MFE 1.567 R | avg MAE 1.294 R

| Book | reach 1R | 2R | 3R | 5R | 10R |
|---|---|---|---|---|---|
| gated | 0.474 | 0.268 | 0.165 | 0.052 | 0.010 |
| ungated | 0.528 | 0.311 | 0.193 | 0.061 | 0.005 |

## Fixed-RR P&L (net of costs)

### target 2R

| Book | Trades | Win rate | Avg R | Net R | PF | MaxDD R |
|---|---|---|---|---|---|---|
| gated (dump+sweep) | 99 | 0.303 | -0.204 | -20.2 | 0.726 | -26.6 |
| ungated (sweep only) | 216 | 0.347 | -0.049 | -10.5 | 0.928 | -22.9 |
| gated DEV (<holdout) | 66 | 0.303 | -0.180 | -11.9 | 0.756 | -15.7 |
| gated HOLDOUT (2025+) | 33 | 0.303 | -0.251 | -8.3 | 0.666 | -10.4 |

Per-year (gated): 2021 -3.6R (18t, 0.278wr), 2022 -4.2R (19t, 0.316wr), 2023 -1.8R (10t, 0.300wr), 2024 -2.3R (19t, 0.316wr), 2025 -4.5R (24t, 0.333wr), 2026 -3.8R (9t, 0.222wr)

### target 3R

| Book | Trades | Win rate | Avg R | Net R | PF | MaxDD R |
|---|---|---|---|---|---|---|
| gated (dump+sweep) | 99 | 0.263 | -0.195 | -19.3 | 0.752 | -27.0 |
| ungated (sweep only) | 216 | 0.273 | -0.096 | -20.7 | 0.872 | -38.6 |
| gated DEV (<holdout) | 66 | 0.242 | -0.226 | -14.9 | 0.718 | -18.8 |
| gated HOLDOUT (2025+) | 33 | 0.303 | -0.131 | -4.3 | 0.826 | -10.4 |

Per-year (gated): 2021 -4.7R (18t, 0.222wr), 2022 -3.1R (19t, 0.316wr), 2023 -2.8R (10t, 0.200wr), 2024 -4.3R (19t, 0.211wr), 2025 -2.5R (24t, 0.333wr), 2026 -1.8R (9t, 0.222wr)

### target 5R

| Book | Trades | Win rate | Avg R | Net R | PF | MaxDD R |
|---|---|---|---|---|---|---|
| gated (dump+sweep) | 97 | 0.216 | -0.354 | -34.3 | 0.577 | -46.5 |
| ungated (sweep only) | 213 | 0.216 | -0.271 | -57.7 | 0.666 | -68.2 |
| gated DEV (<holdout) | 65 | 0.200 | -0.435 | -28.3 | 0.487 | -31.8 |
| gated HOLDOUT (2025+) | 32 | 0.250 | -0.188 | -6.0 | 0.768 | -13.6 |

Per-year (gated): 2021 -8.3R (18t, 0.167wr), 2022 -7.8R (19t, 0.263wr), 2023 -0.7R (10t, 0.200wr), 2024 -11.5R (18t, 0.167wr), 2025 -8.2R (23t, 0.261wr), 2026 2.2R (9t, 0.222wr)

## Reading guide

Profitable only if avg R (expectancy) > 0 net of costs and it holds on DEV *and* HOLDOUT. Win rate needed ~ 1/(RR+1) (e.g. >33% at 2R, >17% at 5R). Compare gated vs ungated to see if the dump gate helps; compare the astro overlay to confirm it does not. High RR with thin win rate => fat-tailed equity; weight MaxDD and per-year stability, not just net R.
