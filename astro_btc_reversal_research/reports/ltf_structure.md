# Multi-TF Structure Edge: 5m Wyckoff Springs at Weekly Lows

**BTCUSDT** LTF 5m / HTF 1d | 2021-01-01 -> 2026-06-01
Weekly lows (daily zigzag >= 3.0 ATR): 55 | 5m springs: 78 | stop 0.05 ATR below spring | costs 11.0 bps RT | 'near low' = +/-288 5m bars.

## Max-R available after a spring (excursion pass)

| Book | signals | avg MFE R | reach 5R | 10R | 20R | 30R |
|---|---|---|---|---|---|---|
| all_springs | 78 | 4.900 | 0.193 | 0.140 | 0.053 | 0.035 |
| near_weekly_low | 17 | 11.590 | 0.308 | 0.308 | 0.154 | 0.154 |
| away_from_low | 61 | 2.870 | 0.156 | 0.089 | 0.022 | 0.000 |

## Fixed-RR P&L net of costs (expectancy = avg R)

### Target 10R

| Book | Trades | Win rate | Avg R | Net R | PF | MaxDD R | Holdout avg R |
|---|---|---|---|---|---|---|---|
| all_springs | 57 | 0.140 | 0.329 | 18.8 | 1.318 | -14.7 | 0.624 |
| near_weekly_low | 13 | 0.308 | 2.037 | 26.5 | 3.234 | -5.1 | 3.893 |
| away_from_low | 45 | 0.089 | -0.198 | -8.9 | 0.816 | -19.1 | -0.382 |

### Target 20R

| Book | Trades | Win rate | Avg R | Net R | PF | MaxDD R | Holdout avg R |
|---|---|---|---|---|---|---|---|
| all_springs | 57 | 0.123 | 0.760 | 43.3 | 1.720 | -24.2 | 1.616 |
| near_weekly_low | 13 | 0.308 | 4.125 | 53.6 | 5.524 | -5.1 | 8.049 |
| away_from_low | 45 | 0.067 | -0.255 | -11.5 | 0.768 | -30.1 | -0.363 |

### Target 30R

| Book | Trades | Win rate | Avg R | Net R | PF | MaxDD R | Holdout avg R |
|---|---|---|---|---|---|---|---|
| all_springs | 57 | 0.105 | 0.743 | 42.3 | 1.690 | -27.8 | 2.205 |
| near_weekly_low | 13 | 0.308 | 5.663 | 73.6 | 7.211 | -5.1 | 10.549 |
| away_from_low | 45 | 0.044 | -0.722 | -32.5 | 0.360 | -31.3 | -0.363 |

## Reading guide

At 20R breakeven win rate is ~1/(20+1) ~ 4.8% (before costs); the tight 5m stop makes costs a real fraction of risk, so read **avg R** (net). The decisive question: does **near_weekly_low** beat **away_from_low** on reach-20R and avg R? If yes, the asymmetric opportunity concentrates at weekly lows -> a learnable gate (predict the weekly-low zone on the HTF, confirm with the 5m spring). If near and away look the same, the spring alone has no structural edge.
