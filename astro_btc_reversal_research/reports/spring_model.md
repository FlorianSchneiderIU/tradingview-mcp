# Learnable weekly-low timing: ranking 5m springs by reach-20R

2089 springs | reach-20R base rate 0.048 | base avg R @20 -0.292 | dev 1515 / holdout 574.
Top-frac selected: 0.1. A tradeable edge = top-slice avg R > 0 out-of-sample AND on holdout.

| Model | OOS PR-AUC | top-slice n | top reach20 | top avg R | all avg R | holdout top avg R |
|---|---|---|---|---|---|---|
| hgb | 0.046 | 122 | 0.041 | -0.318 | -0.339 | -0.733 |
| logistic_l2 | 0.044 | 122 | 0.041 | -0.385 | -0.339 | -0.456 |

## Top logistic structural drivers (standardized coef, +=more likely 20R)

- l_atr_norm: -0.982
- l_vol48: 0.578
- l_ret288: -0.571
- d_dd30: -0.489
- d_rsi14: 0.459
- l_ema200_dist: 0.397
- l_ret12: 0.272
- l_ret48: -0.144
- d_ema50_dist: 0.139
- l_rsi14: -0.133

## Reading guide

base avg R @20 is what a blind spring earns (negative). If a model's **top avg R** is clearly positive OOS and on the holdout, it has learned to find the weekly-low springs in real time - a tradeable high-RR edge. If top avg R ~ base, the structure at the spring does not reveal the weekly low ahead of time, and the edge stays hindsight-only.
