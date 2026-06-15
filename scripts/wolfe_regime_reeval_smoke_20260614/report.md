# Wolfe Recent Regime Re-evaluation

## Rolling Summary

```text
     strategy  window_days  trades  wins  losses  win_rate     net_r     avg_r  profit_factor  stop_rate  target_rate  timeout_rate  max_losing_streak
   wolfe_wave           30       3     1       2  0.333333 -2.188696 -0.729565       0.058014   0.666667     0.000000      0.333333                  2
   wolfe_wave           60       4     2       2  0.500000  0.165797  0.041449       1.071357   0.500000     0.250000      0.250000                  2
   wolfe_wave           90       5     2       3  0.400000 -1.084190 -0.216838       0.696601   0.600000     0.200000      0.200000                  2
   wolfe_wave          180       5     2       3  0.400000 -1.084190 -0.216838       0.696601   0.600000     0.200000      0.200000                  2
wolfe_wave_v2           30       5     4       1  0.800000  9.440599  1.888120      26.995161   0.000000     0.800000      0.200000                  1
wolfe_wave_v2           60       9     6       3  0.666667 11.267643  1.251960       5.243322   0.222222     0.555556      0.222222                  1
wolfe_wave_v2           90      14     9       5  0.642857 17.826239  1.273303       4.483219   0.285714     0.500000      0.214286                  2
wolfe_wave_v2          180      14     9       5  0.642857 17.826239  1.273303       4.483219   0.285714     0.500000      0.214286                  2
```

## RR Bands (365d)

```text
     strategy  rr_band  trades  wins  losses  win_rate     net_r     avg_r  profit_factor  stop_rate  target_rate  timeout_rate  max_losing_streak
   wolfe_wave 1.5-2.0R       2     1       1  0.500000 -0.946624 -0.473312       0.124647   0.500000     0.000000      0.500000                  1
   wolfe_wave    >2.0R       3     1       2  0.333333 -0.137566 -0.045855       0.944798   0.666667     0.333333      0.000000                  1
wolfe_wave_v2    >2.0R      14     9       5  0.642857 17.826239  1.273303       4.483219   0.285714     0.500000      0.214286                  2
```

## Regimes (365d)

```text
     strategy vol_regime directional_regime  trades  wins  losses  win_rate     net_r     avg_r  profit_factor  stop_rate  target_rate  timeout_rate  max_losing_streak
   wolfe_wave   high_vol     mean_reversion       4     1       3  0.250000 -3.438683 -0.859671       0.037721   0.750000     0.000000      0.250000                  3
   wolfe_wave   high_vol         transition       1     1       0  1.000000  2.354493  2.354493            inf   0.000000     1.000000      0.000000                  0
wolfe_wave_v2   high_vol     mean_reversion       7     7       0  1.000000 18.661959  2.665994            inf   0.000000     0.714286      0.285714                  0
wolfe_wave_v2    low_vol     mean_reversion       6     1       5  0.166667 -2.958592 -0.493099       0.421896   0.666667     0.166667      0.166667                  3
wolfe_wave_v2    low_vol      trend_aligned       1     1       0  1.000000  2.122873  2.122873            inf   0.000000     1.000000      0.000000                  0
```

## Live RR Gate

```text
     strategy  sample_trades  sample_win_rate  recommended_min_rr
   wolfe_wave              5         0.400000                 0.0
wolfe_wave_v2             14         0.642857                 0.0
```
