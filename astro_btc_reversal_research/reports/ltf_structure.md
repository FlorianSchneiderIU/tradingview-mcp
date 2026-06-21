# Multi-TF Structure Edge: 5m Wyckoff Springs at Weekly Lows

**BTCUSDT** LTF 5m / HTF 1d | 2021-01-01 -> 2026-06-01
Weekly lows (daily zigzag >= 3.0 ATR): 55 | 5m springs: 333 | stop 0.05 ATR below spring | costs 11.0 bps RT | 'near low' = +/-288 5m bars.

## Max-R available after a spring (excursion pass)

| Book | signals | avg MFE R | reach 5R | 10R | 20R | 30R |
|---|---|---|---|---|---|---|
| all_springs | 333 | 3.813 | 0.144 | 0.082 | 0.037 | 0.029 |
| near_weekly_low | 66 | 8.625 | 0.240 | 0.200 | 0.140 | 0.100 |
| away_from_low | 267 | 2.980 | 0.127 | 0.061 | 0.020 | 0.015 |

## Fixed-RR P&L net of costs (expectancy = avg R)

### Target 10R

| Book | Trades | Win rate | Avg R | Net R | PF | MaxDD R | Holdout avg R |
|---|---|---|---|---|---|---|---|
| all_springs | 246 | 0.089 | -0.331 | -81.4 | 0.710 | -80.3 | -0.361 |
| near_weekly_low | 50 | 0.220 | 1.034 | 51.7 | 2.069 | -11.3 | 2.488 |
| away_from_low | 199 | 0.065 | -0.579 | -115.2 | 0.506 | -114.1 | -0.750 |

### Target 20R

| Book | Trades | Win rate | Avg R | Net R | PF | MaxDD R | Holdout avg R |
|---|---|---|---|---|---|---|---|
| all_springs | 245 | 0.078 | -0.040 | -9.8 | 0.966 | -54.2 | 0.280 |
| near_weekly_low | 50 | 0.220 | 2.719 | 135.9 | 3.812 | -11.3 | 5.818 |
| away_from_low | 198 | 0.051 | -0.545 | -107.9 | 0.543 | -109.4 | -0.471 |

### Target 30R

| Book | Trades | Win rate | Avg R | Net R | PF | MaxDD R | Holdout avg R |
|---|---|---|---|---|---|---|---|
| all_springs | 245 | 0.073 | 0.123 | 30.2 | 1.106 | -56.4 | 0.621 |
| near_weekly_low | 50 | 0.220 | 3.419 | 170.9 | 4.536 | -11.3 | 7.318 |
| away_from_low | 198 | 0.045 | -0.499 | -98.9 | 0.583 | -116.9 | -0.215 |

## Reading guide

At 20R breakeven win rate is ~1/(20+1) ~ 4.8% (before costs); the tight 5m stop makes costs a real fraction of risk, so read **avg R** (net). The decisive question: does **near_weekly_low** beat **away_from_low** on reach-20R and avg R? If yes, the asymmetric opportunity concentrates at weekly lows -> a learnable gate (predict the weekly-low zone on the HTF, confirm with the 5m spring). If near and away look the same, the spring alone has no structural edge.
