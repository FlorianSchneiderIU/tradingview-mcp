# Weekly reversal-window predictability from prior weeks

**BTCUSDT** 15m | 282 weeks (279 usable after 3 lags). 'early low' = first 40% of the week.

## Does the low-timing persist week to week?

- base rate early: 0.553
- P(early | prev week early): 0.564
- P(early | prev week late): 0.536
- transition chi-square p: low_early 0.7263 | 3-bucket 0.7973 | 7-dow 0.8214

## Autocorrelation of within-week extreme timing (95% noise band +/-0.117)

| lag (weeks) | low_frac ac | high_frac ac |
|---|---|---|
| 1 | 0.048 | 0.006 |
| 2 | -0.070 | 0.012 |
| 3 | -0.013 | 0.020 |
| 4 | -0.050 | -0.007 |

Spacing between consecutive weekly lows: median 7.0d | share 7+/-1d 0.295 | share <=3d 0.114.

## Predictive test (walk-forward, OOS) vs baseline

Baseline: predict base rate -> log-loss 0.688, majority accuracy 0.552. A model is useful only if OOS ROC-AUC > 0.5, log-loss < baseline, and the top-20% slice's early-rate beats the base rate.

| Features | Model | OOS n | ROC-AUC | log-loss | accuracy | top-20% early rate |
|---|---|---|---|---|---|---|
| lag1 | logistic_l2 | 164 | 0.446 | 0.718 | 0.457 | 0.485 |
| lag1 | hgb | 164 | 0.418 | 0.731 | 0.476 | 0.458 |
| lags123 | logistic_l2 | 164 | 0.471 | 0.742 | 0.488 | 0.576 |
| lags123 | hgb | 164 | 0.429 | 0.735 | 0.476 | 0.467 |
| lags+price | logistic_l2 | 164 | 0.474 | 0.745 | 0.476 | 0.545 |
| lags+price | hgb | 164 | 0.460 | 0.713 | 0.494 | 0.491 |

## Reading guide

If chi-square p > 0.05, autocorrelations sit inside the noise band, spacing is just the trivial ~7d (Monday-to-Monday) peak, and OOS ROC-AUC ~ 0.5 with log-loss ~ baseline, then prior weeks carry no extra information beyond the standing Monday/early-week prior. A genuine cycle shows up as significant persistence AND an OOS model that beats baseline.
