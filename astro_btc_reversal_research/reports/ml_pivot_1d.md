# ML Pivot-Window Model (Milestone 3)

**Target:** pivot_within_3 | **Symbol/timeframe:** BTCUSDT 1d | 2021-01-01 -> 2026-06-01 (1978 candles)
**Pivots:** ATR threshold 2.0 (235 pivots) | base rate 0.3251
**Validation:** 4-fold expanding walk-forward, embargo 3 candles | holdout 2025-01-01 00:00:00+00:00

## Pooled out-of-sample (walk-forward) by feature set x model

| Feature set | Model | PR-AUC | ROC-AUC | Brier | Prec@5% | Lift@5% | Holdout PR-AUC |
|---|---|---|---|---|---|---|---|
| price_only | logistic_l2 | 0.5080 | 0.6326 | 0.2248 | 0.7414 | 2.2728 | 0.5022 |
| price_only | hgb | 0.4566 | 0.6182 | 0.2186 | 0.5690 | 1.7442 | 0.5002 |
| calendar_only | logistic_l2 | 0.3692 | 0.5633 | 0.2520 | 0.4138 | 1.2685 | 0.4475 |
| calendar_only | hgb | 0.3773 | 0.5341 | 0.2367 | 0.5172 | 1.5857 | 0.4063 |
| lunar_only | logistic_l2 | 0.3316 | 0.5012 | 0.2642 | 0.2931 | 0.8985 | 0.3569 |
| lunar_only | hgb | 0.3500 | 0.5350 | 0.2366 | 0.3448 | 1.0571 | 0.3289 |
| astro_only | logistic_l2 | 0.3294 | 0.5104 | 0.3273 | 0.3448 | 1.0571 | 0.2811 |
| astro_only | hgb | 0.3082 | 0.4560 | 0.2661 | 0.3448 | 1.0571 | 0.2877 |
| astro_cycle | logistic_l2 | 0.3316 | 0.5170 | 0.3214 | 0.2586 | 0.7928 | 0.2943 |
| astro_cycle | hgb | 0.3146 | 0.4746 | 0.2597 | 0.3448 | 1.0571 | 0.2850 |
| astro_plus_price | logistic_l2 | 0.3730 | 0.5535 | 0.3084 | 0.4483 | 1.3742 | 0.3144 |
| astro_plus_price | hgb | 0.3744 | 0.5398 | 0.2459 | 0.4655 | 1.4271 | 0.4007 |
| full | logistic_l2 | 0.3885 | 0.5674 | 0.3030 | 0.5000 | 1.5328 | 0.3315 |
| full | hgb | 0.3814 | 0.5581 | 0.2429 | 0.4828 | 1.4800 | 0.4122 |

## Shifted-placebo control (astro/cycle shifted ~37d)

| Feature set | Model | Real OOS PR-AUC | Placebo OOS PR-AUC |
|---|---|---|---|
| astro_cycle | logistic_l2 | 0.3316 | 0.3457 |
| astro_cycle | hgb | 0.3146 | 0.3464 |
| astro_plus_price | logistic_l2 | 0.3730 | 0.4113 |
| astro_plus_price | hgb | 0.3744 | 0.4247 |
| full | logistic_l2 | 0.3885 | 0.4365 |
| full | hgb | 0.3814 | 0.4410 |

## Reading guide

PR-AUC above the base rate and lift@K > 1 indicate signal; pivots are rare so PR-AUC/precision@K matter more than ROC-AUC. A feature set is only credible if it beats both `price_only` and its shifted placebo, and holds on the holdout column.
