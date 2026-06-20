# Dark Pivot claim - replication + baseline

Moon-Pluto hard aspects as Dark Pivots. 289 firings over 2021-01-01 -> 2026-06-01.
Advertised next date 2026-06-24 is in the computed calendar: 2026-06-24 16:36, 2026-07-02 05:11, 2026-07-09 04:34, 2026-07-15 05:58 ...

## Their loose rule reproduced, with a baseline beside it

Rule: dumped into the activation day, then a bullish expansion within `horizon` days (x_atr = expansion size beyond the activation high). **lift_vs_dump** compares to the SAME rule on ordinary (non-Dark-Pivot) dump days.

| lookback | horizon | x_atr | signals | DP hit | base(dump) | base(all) | lift vs dump | binom p |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.0 | 151 | 0.265 | 0.335 | 0.472 | 0.791 | 0.9738 |
| 1 | 1 | 0.5 | 151 | 0.079 | 0.096 | 0.146 | 0.830 | 0.7893 |
| 1 | 2 | 0.0 | 151 | 0.417 | 0.467 | 0.600 | 0.894 | 0.9027 |
| 1 | 2 | 0.5 | 151 | 0.172 | 0.196 | 0.262 | 0.878 | 0.7992 |
| 1 | 3 | 0.0 | 151 | 0.517 | 0.528 | 0.660 | 0.979 | 0.6377 |
| 1 | 3 | 0.5 | 151 | 0.245 | 0.289 | 0.359 | 0.846 | 0.9038 |
| 1 | 5 | 0.0 | 151 | 0.642 | 0.623 | 0.730 | 1.031 | 0.3459 |
| 1 | 5 | 0.5 | 151 | 0.377 | 0.401 | 0.464 | 0.942 | 0.7461 |
| 1 | 7 | 0.0 | 151 | 0.695 | 0.685 | 0.774 | 1.015 | 0.4344 |
| 1 | 7 | 0.5 | 151 | 0.464 | 0.478 | 0.537 | 0.969 | 0.6724 |
| 2 | 1 | 0.0 | 132 | 0.288 | 0.382 | 0.472 | 0.754 | 0.9905 |
| 2 | 1 | 0.5 | 132 | 0.061 | 0.089 | 0.146 | 0.678 | 0.9116 |
| 2 | 2 | 0.0 | 132 | 0.432 | 0.516 | 0.600 | 0.837 | 0.9781 |
| 2 | 2 | 0.5 | 132 | 0.189 | 0.200 | 0.262 | 0.945 | 0.6586 |
| 2 | 3 | 0.0 | 132 | 0.538 | 0.579 | 0.660 | 0.930 | 0.8492 |
| 2 | 3 | 0.5 | 132 | 0.273 | 0.293 | 0.359 | 0.929 | 0.7293 |
| 2 | 5 | 0.0 | 132 | 0.644 | 0.665 | 0.730 | 0.968 | 0.7331 |
| 2 | 5 | 0.5 | 132 | 0.386 | 0.411 | 0.464 | 0.941 | 0.7428 |
| 2 | 7 | 0.0 | 132 | 0.705 | 0.720 | 0.774 | 0.979 | 0.6907 |
| 2 | 7 | 0.5 | 132 | 0.500 | 0.484 | 0.537 | 1.032 | 0.3918 |
| 3 | 1 | 0.0 | 131 | 0.359 | 0.405 | 0.472 | 0.885 | 0.8796 |
| 3 | 1 | 0.5 | 131 | 0.084 | 0.099 | 0.146 | 0.852 | 0.7534 |
| 3 | 2 | 0.0 | 131 | 0.489 | 0.530 | 0.600 | 0.923 | 0.8480 |
| 3 | 2 | 0.5 | 131 | 0.198 | 0.214 | 0.262 | 0.926 | 0.7030 |
| 3 | 3 | 0.0 | 131 | 0.573 | 0.590 | 0.660 | 0.971 | 0.6904 |
| 3 | 3 | 0.5 | 131 | 0.305 | 0.297 | 0.359 | 1.029 | 0.4478 |
| 3 | 5 | 0.0 | 131 | 0.672 | 0.674 | 0.730 | 0.997 | 0.5595 |
| 3 | 5 | 0.5 | 131 | 0.405 | 0.416 | 0.464 | 0.972 | 0.6388 |
| 3 | 7 | 0.0 | 131 | 0.725 | 0.729 | 0.774 | 0.995 | 0.5841 |
| 3 | 7 | 0.5 | 131 | 0.511 | 0.483 | 0.537 | 1.059 | 0.2843 |

## Literal 'marked a local bottom' (+/-3 d)

- Dark-Pivot dump days that are local lows: **0.167** (132 signals)
- ordinary dump days that are local lows: 0.159 | all days: 0.101
- **lift vs ordinary dump days: 1.045**

## '50% window' (midpoint between Dark Pivots) marks the opposite extreme (top)

- midpoints that are local highs: **0.094** (288 midpoints)
- all days that are local highs: 0.100 | **lift: 0.941**

## Verdict

The advertised ~77% is the **base rate**, not an edge. With horizon 7 (the gap to the next pivot), the unconditional probability of a higher high within 7 days is ~0.774 - essentially the claimed 77.27%. Dark-Pivot dump days score ~0.70-0.73, i.e. AT OR BELOW that base rate, and ~equal to ordinary dump days (lift vs dump ~1.0, binomial p ~0.4-0.7). The loose success rule ('a bullish expansion within a week of a dump') is near-universal on daily BTC, so a high hit rate is guaranteed regardless of astrology. The literal 'local bottom' and '50% top' readings show lift ~1.0 too. No edge over baseline.
