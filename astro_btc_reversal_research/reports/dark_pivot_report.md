# Dark Pivot Candidate Test (Milestone 1)

**Candidate:** moon-pluto hard aspects [0, 90, 180, 270]
**Symbol/timeframe:** BTCUSDT 1d | 2021-01-01 -> 2026-06-01 (1978 candles)
**Dump definition:** move <= -1.0 ATR over 2 bars
**Expansion:** bullish, horizon 3 candles, target 1.0 ATR, buffer 0.1 ATR
**Event window:** +/-1 candles around each aspect event

## Headline result

| Set | N | Bullish-expansion hit rate | 95% CI |
|---|---|---|---|
| A: dump days on/into aspect window | 92 | 0.2065 | [0.1304, 0.2935] |
| B: ordinary dump days (baseline) | 118 | 0.1949 | [0.1271, 0.2712] |

- **Lift (A / B):** 1.0595
- **Rate difference (A - B):** 0.0116 (95% CI [-0.1008, 0.1205])
- **Binomial p-value** (A beats baseline rate): 0.43031
- **Random-calendar p-value** (1000 draws): 0.42258 (null mean 0.1995)

## Shifted-calendar baseline (real events should beat these)

| Offset (days) | N (dump in window) | Hit rate |
|---|---|---|
| +3 | 89 | 0.1910 |
| +7 | 97 | 0.2268 |
| +13 | 87 | 0.2069 |
| +21 | 97 | 0.2371 |
| +37 | 87 | 0.1724 |
| +83 | 99 | 0.2424 |

## MFE / MAE / max-R for set A (event dump days)

- mean MFE: 1.6621 R | mean MAE: 1.9253 R
- median max-R available: 1.2631 R
- share reaching >=1R: 0.5978 | >=2R: 0.3370

## Out-of-sample (holdout) check

Holdout start: 2025-01-01 00:00:00+00:00
- dev set A hit rate: 0.2143 (N 56)
- holdout set A hit rate: 0.1944 (N 36)

## Pivot proximity (secondary)

- aspect candles within +/-1 of an ATR pivot: 0.1730
- baseline candles near a pivot: 0.1764 (lift 0.9806)

## Interpretation guard

A positive lift is only interesting if it (a) beats the random-calendar null, (b) beats shifted calendars, and (c) holds out-of-sample. Small N inflates noise; read the CIs, not the point estimates.
