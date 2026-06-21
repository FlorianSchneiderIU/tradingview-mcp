# Capitulation-Filtered Basket Spring (target 20R)

15m | 2022-01-01 -> 2026-06-01 (~230.3 wk) | sweep 1440 bars | week gate 0.4 | funding_z<=-1.0 | oi_z<=-1.0 | coverage funding 18 / OI 18 symbols.

Capitulation = deep-sweep spring coinciding with unusually negative funding (shorts crowded) and/or an OI flush (forced long liquidations).

| Book | Trades | Trades/wk | Win % | Avg R | Net R | PF | MaxDD R | reach20R | Holdout avg R (n) |
|---|---|---|---|---|---|---|---|---|---|
| all | 1017 | 4.42 | 9.3 | -0.084 | -85.6 | 0.917 | -247.2 | 0.031 | 0.200 (430) |
| cap_funding | 242 | 1.05 | 16.9 | 0.475 | 115.0 | 1.514 | -58.8 | 0.045 | 0.909 (130) |
| cap_oiflush | 165 | 0.72 | 15.2 | -0.040 | -6.5 | 0.958 | -48.3 | 0.024 | 0.250 (61) |
| cap_either | 354 | 1.54 | 14.4 | 0.331 | 117.2 | 1.348 | -70.6 | 0.042 | 0.744 (162) |
| cap_both | 53 | 0.23 | 28.3 | -0.165 | -8.8 | 0.793 | -17.1 | 0.000 | 0.446 (29) |

## Reading guide

Does a capitulation book beat **all** on avg R / reach20R / holdout while keeping trades/week near 1? That would be the high-conviction weekly setup: deep-sweep spring + funding/OI flush. If capitulation books are no better, the flush adds nothing.
