# Wolfe Multi-Timeframe Hypothesis Re-evaluation

The deployed entries and exits are held constant. Each research rule is applied
one at a time as a counterfactual gate. `neighbor_median_delta_avg_r` is the
median improvement across adjacent thresholds in the same family, reducing
the influence of a lucky single cutoff.

## Rolling Summary

```text
  strategy  window_days  trades  wins  losses  win_rate     net_r     avg_r  avg_win_r  avg_loss_r  payoff_ratio  profit_factor  max_drawdown_r  avg_hold_bars  stop_rate  target_rate  timeout_rate  max_losing_streak
wolfe_wave           30       2     1       1  0.500000 -0.013364 -0.006682   1.068056    1.081420      0.987643       0.987643        0.000000          123.0   0.500000     0.500000           0.0                  1
wolfe_wave           60       5     4       1  0.800000  6.325016  1.265003   1.851609    1.081420      1.712202       6.848806        1.081420           83.0   0.200000     0.800000           0.0                  1
wolfe_wave           90       7     4       3  0.571429  3.882711  0.554673   1.851609    1.174575      1.576408       2.101877        2.273738           83.0   0.428571     0.571429           0.0                  2
```

## RR Bands (365d)

```text
  strategy  rr_band  trades  wins  losses  win_rate     net_r     avg_r  avg_win_r  avg_loss_r  payoff_ratio  profit_factor  max_drawdown_r  avg_hold_bars  stop_rate  target_rate  timeout_rate  max_losing_streak
wolfe_wave 1.5-2.0R       1     0       1       0.0 -1.081420 -1.081420   0.000000    1.081420      0.000000       0.000000        0.000000           59.0        1.0          0.0           0.0                  1
wolfe_wave   <=1.5R       1     1       0       1.0  1.068056  1.068056   1.068056    0.000000           inf            inf        0.000000          187.0        0.0          1.0           0.0                  0
wolfe_wave    >2.0R       5     3       2       0.6  3.896075  0.779215   2.112793    1.221152      1.730163       2.595245        1.192318           67.0        0.4          0.6           0.0                  1
```

## Regimes (365d)

```text
  strategy timeframe vol_regime directional_regime  trades  wins  losses  win_rate     net_r     avg_r  avg_win_r  avg_loss_r  payoff_ratio  profit_factor  max_drawdown_r  avg_hold_bars  stop_rate  target_rate  timeout_rate  max_losing_streak
wolfe_wave       15m   high_vol     mean_reversion       6     3       3  0.500000  1.528218  0.254703   1.683981    1.174575      1.433694       1.433694        2.273738      91.500000   0.500000     0.500000           0.0                  3
wolfe_wave       15m   high_vol         transition       1     1       0  1.000000  2.354493  2.354493   2.354493    0.000000           inf            inf        0.000000      32.000000   0.000000     1.000000           0.0                  0
wolfe_wave        1h   high_vol     mean_reversion       2     1       1  0.500000 -0.013364 -0.006682   1.068056    1.081420      0.987643       0.987643        0.000000     123.000000   0.500000     0.500000           0.0                  1
wolfe_wave        1h   high_vol         transition       1     1       0  1.000000  1.970974  1.970974   1.970974    0.000000           inf            inf        0.000000      98.000000   0.000000     1.000000           0.0                  0
wolfe_wave        1h   high_vol      trend_aligned       1     1       0  1.000000  2.354493  2.354493   2.354493    0.000000           inf            inf        0.000000      32.000000   0.000000     1.000000           0.0                  0
wolfe_wave        1h    low_vol     mean_reversion       3     1       2  0.333333 -0.429393 -0.143131   2.012912    1.221152      1.648371       0.824186        1.192318      68.333333   0.666667     0.333333           0.0                  2
wolfe_wave        4h   high_vol     mean_reversion       2     1       1  0.500000 -0.013364 -0.006682   1.068056    1.081420      0.987643       0.987643        0.000000     123.000000   0.500000     0.500000           0.0                  1
wolfe_wave        4h    low_vol     mean_reversion       3     1       2  0.333333 -0.429393 -0.143131   2.012912    1.221152      1.648371       0.824186        1.192318      68.333333   0.666667     0.333333           0.0                  2
wolfe_wave        4h    low_vol      trend_aligned       2     2       0  1.000000  4.325467  2.162734   2.162734    0.000000           inf            inf        0.000000      65.000000   0.000000     1.000000           0.0                  0
wolfe_wave        1d    low_vol     mean_reversion       6     3       3  0.500000  1.869799  0.311633   1.797841    1.174575      1.530631       1.530631        2.273738      90.333333   0.500000     0.500000           0.0                  2
wolfe_wave        1d    low_vol      trend_aligned       1     1       0  1.000000  2.012912  2.012912   2.012912    0.000000           inf            inf        0.000000      39.000000   0.000000     1.000000           0.0                  0
```

## Live RR Gate

```text
  strategy  sample_trades  sample_win_rate  recommended_min_rr
wolfe_wave              7         0.571429                 0.0
```

## Best Hypothesis Tests (90d combined)

```text
                 hypothesis       verdict                                    filter_name timeframe  trades  retention  win_rate  avg_win_r  avg_loss_r    avg_r  profit_factor  delta_avg_r  neighbor_median_delta_avg_r  symbols_improved_pct  periods_improved_pct  robust_score
           H1 runaway trend not_supported                avoid 4h ADX>=25 opposing trend        4h       3   0.428571  1.000000   2.112793    0.000000 2.112793            inf     1.558120                     1.558120                   NaN                   NaN      1.020029
H2 higher-timeframe context not_supported                   4h EMA structure not opposed        4h       2   0.285714  1.000000   2.162734    0.000000 2.162734            inf     1.608061                     1.608061                   NaN                   NaN      0.859545
H2 higher-timeframe context not_supported                       4h EMA structure aligned        4h       2   0.285714  1.000000   2.162734    0.000000 2.162734            inf     1.608061                     1.608061                   NaN                   NaN      0.859545
H2 higher-timeframe context not_supported                   1h EMA structure not opposed        1h       2   0.285714  1.000000   2.162734    0.000000 2.162734            inf     1.608061                     1.608061                   NaN                   NaN      0.859545
H4 volatility normalization not_supported                        4h ATR percentile 30-80        4h       3   0.428571  1.000000   1.797841    0.000000 1.797841            inf     1.243168                     1.243168                   NaN                   NaN      0.813845
           H1 runaway trend not_supported                                    4h ADX < 25        4h       2   0.285714  1.000000   1.991943    0.000000 1.991943            inf     1.437270                     1.437270                   NaN                   NaN      0.768253
           H1 runaway trend not_supported                                    4h ADX < 30        4h       2   0.285714  1.000000   1.991943    0.000000 1.991943            inf     1.437270                     1.437270                   NaN                   NaN      0.768253
H4 volatility normalization not_supported                        4h ATR percentile 10-90        4h       4   0.571429  0.750000   1.797841    1.249987 1.035884       4.314864     0.481211                     0.481211                   NaN                   NaN      0.363761
     H7 volume confirmation not_supported point-5 volume >= 1.0 and rejection >= 0.2 ATR   pattern       4   0.571429  0.750000   1.683981    1.192318 0.964906       4.237077     0.410233                     0.410233                   NaN                   NaN      0.310107
     H7 volume confirmation not_supported                   point-5 rejection >= 0.2 ATR   pattern       4   0.571429  0.750000   1.683981    1.192318 0.964906       4.237077     0.410233                     0.410233                   NaN                   NaN      0.310107
H4 volatility normalization not_supported                        1h ATR percentile 10-90        1h       4   0.571429  0.750000   1.683981    1.249987 0.950489       4.041597     0.395816                     0.395816                   NaN                   NaN      0.299209
     H7 volume confirmation not_supported                    point-5 volume ratio >= 1.0   pattern       6   0.857143  0.666667   1.851609    1.136869 0.855450       3.257383     0.300777                     0.300777                   NaN                   NaN      0.278465
         H6 EPA reward/risk not_supported                           planned EPA/R >= 1.5     trade       6   0.857143  0.500000   2.112793    1.174575 0.469109       1.798773    -0.085564                     0.069489                   NaN                   NaN      0.064334
         H6 EPA reward/risk not_supported                           planned EPA/R >= 2.5     trade       2   0.285714  0.500000   2.354493    1.249987 0.552253       1.883614    -0.002420                     0.111061                   NaN                   NaN      0.059365
                H5 geometry not_supported                   point-5 overshoot <= 1.5 ATR   pattern       7   1.000000  0.571429   1.851609    1.174575 0.554673       2.101877     0.000000                     0.000000                   NaN                   NaN      0.000000
              H3 exhaustion not_supported                    entry RSI exhausted (35/65)        5m       0   0.000000  0.000000   0.000000    0.000000 0.000000       0.000000    -0.554673                    -0.554673                   NaN                   NaN      0.000000
                H5 geometry not_supported                   point-5 overshoot <= 2.0 ATR   pattern       7   1.000000  0.571429   1.851609    1.174575 0.554673       2.101877     0.000000                     0.000000                   NaN                   NaN      0.000000
         H6 EPA reward/risk not_supported                           planned EPA/R >= 2.0     trade       5   0.714286  0.600000   2.112793    1.221152 0.779215       2.595245     0.224542                    -0.002420                   NaN                   NaN     -0.002045
              H3 exhaustion not_supported                  point-5 RSI exhausted (35/65)   pattern       4   0.571429  0.500000   2.183703    1.136869 0.523417       1.920804    -0.031256                    -0.031256                   NaN                   NaN     -0.023627
                H5 geometry not_supported                   point-5 overshoot <= 1.0 ATR   pattern       6   0.857143  0.500000   1.683981    1.174575 0.254703       1.433694    -0.299970                    -0.149985                   NaN                   NaN     -0.138859
              H3 exhaustion not_supported                    point-5 RSI divergence >= 3   pattern       4   0.571429  0.500000   1.991943    1.165703 0.413120       1.708791    -0.141553                    -0.302097                   NaN                   NaN     -0.228364
```

## Best Multi-Timeframe Context Rule

```text
                 hypothesis       verdict                     filter_name timeframe  trades  retention  win_rate  avg_win_r  avg_loss_r    avg_r  profit_factor  delta_avg_r  neighbor_median_delta_avg_r  symbols_improved_pct  periods_improved_pct  robust_score
           H1 runaway trend not_supported avoid 4h ADX>=25 opposing trend        4h       3   0.428571  1.000000   2.112793    0.000000 2.112793            inf     1.558120                     1.558120                   NaN                   NaN      1.020029
H2 higher-timeframe context not_supported    1h EMA structure not opposed        1h       2   0.285714  1.000000   2.162734    0.000000 2.162734            inf     1.608061                     1.608061                   NaN                   NaN      0.859545
H2 higher-timeframe context not_supported    4h EMA structure not opposed        4h       2   0.285714  1.000000   2.162734    0.000000 2.162734            inf     1.608061                     1.608061                   NaN                   NaN      0.859545
H4 volatility normalization not_supported         4h ATR percentile 30-80        4h       3   0.428571  1.000000   1.797841    0.000000 1.797841            inf     1.243168                     1.243168                   NaN                   NaN      0.813845
H2 higher-timeframe context not_supported   15m EMA structure not opposed       15m       1   0.142857  1.000000   2.354493    0.000000 2.354493            inf     1.799820                     1.799820                   NaN                   NaN      0.680268
           H1 runaway trend not_supported                    15m ADX < 20       15m       2   0.285714  1.000000   2.162734    0.000000 2.162734            inf     1.608061                     1.051844                   NaN                   NaN      0.562235
H2 higher-timeframe context not_supported    1d EMA structure not opposed        1d       1   0.142857  1.000000   2.012912    0.000000 2.012912            inf     1.458239                     1.458239                   NaN                   NaN      0.551163
           H1 runaway trend not_supported                     1h ADX < 30        1h       5   0.714286  0.800000   1.851609    1.192318 1.242824       6.211795     0.688151                     0.591889                   NaN                   NaN      0.500238
H4 volatility normalization not_supported         1h ATR percentile 10-90        1h       4   0.571429  0.750000   1.683981    1.249987 0.950489       4.041597     0.395816                     0.395816                   NaN                   NaN      0.299209
           H1 runaway trend not_supported avoid 1d ADX>=25 opposing trend        1d       6   0.857143  0.666667   1.851609    1.221152 0.827355       3.032560     0.272682                     0.272682                   NaN                   NaN      0.252455
H4 volatility normalization not_supported        15m ATR percentile 30-80       15m       1   0.142857  1.000000   1.068056    0.000000 1.068056            inf     0.513383                     0.513383                   NaN                   NaN      0.194041
H4 volatility normalization not_supported         1d ATR percentile 10-90        1d       5   0.714286  0.600000   1.797841    1.165703 0.612423       2.313420     0.057750                     0.057750                   NaN                   NaN      0.048808
```

## Best Rule by Strategy

```text
  strategy                  hypothesis       verdict                                    filter_name timeframe  trades  retention  win_rate  avg_win_r  avg_loss_r    avg_r  profit_factor  delta_avg_r  neighbor_median_delta_avg_r  symbols_improved_pct  periods_improved_pct  robust_score
wolfe_wave            H1 runaway trend not_supported                avoid 4h ADX>=25 opposing trend        4h       3   0.428571  1.000000   2.112793    0.000000 2.112793            inf     1.558120                     1.558120                   NaN                   NaN      1.020029
  combined            H1 runaway trend not_supported                avoid 4h ADX>=25 opposing trend        4h       3   0.428571  1.000000   2.112793    0.000000 2.112793            inf     1.558120                     1.558120                   NaN                   NaN      1.020029
  combined H2 higher-timeframe context not_supported                   4h EMA structure not opposed        4h       2   0.285714  1.000000   2.162734    0.000000 2.162734            inf     1.608061                     1.608061                   NaN                   NaN      0.859545
wolfe_wave H2 higher-timeframe context not_supported                       4h EMA structure aligned        4h       2   0.285714  1.000000   2.162734    0.000000 2.162734            inf     1.608061                     1.608061                   NaN                   NaN      0.859545
wolfe_wave H4 volatility normalization not_supported                        4h ATR percentile 30-80        4h       3   0.428571  1.000000   1.797841    0.000000 1.797841            inf     1.243168                     1.243168                   NaN                   NaN      0.813845
  combined H4 volatility normalization not_supported                        4h ATR percentile 30-80        4h       3   0.428571  1.000000   1.797841    0.000000 1.797841            inf     1.243168                     1.243168                   NaN                   NaN      0.813845
  combined      H7 volume confirmation not_supported point-5 volume >= 1.0 and rejection >= 0.2 ATR   pattern       4   0.571429  0.750000   1.683981    1.192318 0.964906       4.237077     0.410233                     0.410233                   NaN                   NaN      0.310107
wolfe_wave      H7 volume confirmation not_supported                   point-5 rejection >= 0.2 ATR   pattern       4   0.571429  0.750000   1.683981    1.192318 0.964906       4.237077     0.410233                     0.410233                   NaN                   NaN      0.310107
wolfe_wave          H6 EPA reward/risk not_supported                           planned EPA/R >= 1.5     trade       6   0.857143  0.500000   2.112793    1.174575 0.469109       1.798773    -0.085564                     0.069489                   NaN                   NaN      0.064334
  combined          H6 EPA reward/risk not_supported                           planned EPA/R >= 1.5     trade       6   0.857143  0.500000   2.112793    1.174575 0.469109       1.798773    -0.085564                     0.069489                   NaN                   NaN      0.064334
  combined                 H5 geometry not_supported                   point-5 overshoot <= 2.0 ATR   pattern       7   1.000000  0.571429   1.851609    1.174575 0.554673       2.101877     0.000000                     0.000000                   NaN                   NaN      0.000000
wolfe_wave                 H5 geometry not_supported                   point-5 overshoot <= 1.5 ATR   pattern       7   1.000000  0.571429   1.851609    1.174575 0.554673       2.101877     0.000000                     0.000000                   NaN                   NaN      0.000000
wolfe_wave               H3 exhaustion not_supported                    entry RSI exhausted (35/65)        5m       0   0.000000  0.000000   0.000000    0.000000 0.000000       0.000000    -0.554673                    -0.554673                   NaN                   NaN      0.000000
  combined               H3 exhaustion not_supported                    entry RSI exhausted (35/65)        5m       0   0.000000  0.000000   0.000000    0.000000 0.000000       0.000000    -0.554673                    -0.554673                   NaN                   NaN      0.000000
```
