# Wolfe Recent Regime Re-evaluation

## Rolling Summary

```text
     strategy  window_days  trades  wins  losses  win_rate       net_r    avg_r  profit_factor  stop_rate  target_rate  timeout_rate  max_losing_streak
   wolfe_wave           30     173    84      89  0.485549   89.472152 0.517180       1.912323   0.497110     0.427746      0.075145                  5
   wolfe_wave           60     332   167     165  0.503012  189.100097 0.569579       2.029812   0.481928     0.424699      0.093373                  5
   wolfe_wave           90     457   231     226  0.505470  263.266678 0.576076       2.039018   0.479212     0.435449      0.085339                 10
   wolfe_wave          180     922   464     458  0.503254  496.886297 0.538922       1.964472   0.485900     0.427332      0.086768                 10
   wolfe_wave          365    2010   999    1011  0.497015 1065.532703 0.530116       1.941560   0.490050     0.411443      0.098507                 10
wolfe_wave_v2           30      38    20      18  0.526316   26.152150 0.688214       2.440186   0.394737     0.447368      0.157895                  3
wolfe_wave_v2           60      86    48      38  0.558140   73.052627 0.849449       2.828662   0.395349     0.465116      0.139535                  4
wolfe_wave_v2           90     120    66      54  0.550000   95.093970 0.792450       2.638378   0.416667     0.458333      0.125000                  6
wolfe_wave_v2          180     236   128     108  0.542373  183.192614 0.776240       2.539690   0.440678     0.457627      0.101695                  5
wolfe_wave_v2          365     484   256     228  0.528926  345.747944 0.714355       2.357934   0.462810     0.450413      0.086777                  6
```

## RR Bands (365d)

```text
     strategy  rr_band  trades  wins  losses  win_rate      net_r    avg_r  profit_factor  stop_rate  target_rate  timeout_rate  max_losing_streak
   wolfe_wave 1.5-2.0R     345   189     156  0.547826 116.853597 0.338706       1.676651   0.434783     0.492754      0.072464                  9
   wolfe_wave   <=1.5R     109    53      56  0.486239   2.393843 0.021962       1.038470   0.513761     0.486239      0.000000                  5
   wolfe_wave    >2.0R    1556   757     799  0.486504 946.285263 0.608152       2.055241   0.500643     0.388175      0.111183                  9
wolfe_wave_v2 1.5-2.0R      65    41      24  0.630769  37.377859 0.575044       2.436435   0.353846     0.553846      0.092308                  3
wolfe_wave_v2   <=1.5R      19    10       9  0.526316   3.140920 0.165312       1.323211   0.473684     0.526316      0.000000                  2
wolfe_wave_v2    >2.0R     400   205     195  0.512500 305.229164 0.763073       2.394542   0.480000     0.430000      0.090000                  5
```

## Regimes (365d)

```text
     strategy vol_regime directional_regime  trades  wins  losses  win_rate      net_r     avg_r  profit_factor  stop_rate  target_rate  timeout_rate  max_losing_streak
   wolfe_wave   high_vol     mean_reversion    1248   622     626  0.498397 627.338090  0.502675       1.903716   0.487179     0.398237      0.114583                  8
   wolfe_wave   high_vol         transition      62    24      38  0.387097  10.225511  0.164928       1.238093   0.612903     0.338710      0.048387                  8
   wolfe_wave   high_vol      trend_aligned     105    60      45  0.571429  78.984246  0.752231       2.585745   0.409524     0.495238      0.095238                  5
   wolfe_wave    low_vol     mean_reversion     493   239     254  0.484787 279.065764  0.566056       1.959951   0.507099     0.432049      0.060852                 13
   wolfe_wave    low_vol         transition      23     9      14  0.391304   1.931915  0.083996       1.118665   0.608696     0.260870      0.130435                  3
   wolfe_wave    low_vol      trend_aligned      79    45      34  0.569620  67.987177  0.860597       2.801167   0.405063     0.481013      0.113924                  5
wolfe_wave_v2   high_vol     mean_reversion     289   163     126  0.564014 245.843258  0.850669       2.772412   0.429066     0.463668      0.107266                  4
wolfe_wave_v2   high_vol         transition      11     4       7  0.363636  -2.013513 -0.183047       0.740585   0.636364     0.272727      0.090909                  3
wolfe_wave_v2   high_vol      trend_aligned      16    10       6  0.625000  11.252464  0.703279       2.722223   0.375000     0.500000      0.125000                  2
wolfe_wave_v2    low_vol     mean_reversion     142    64      78  0.450704  66.954482  0.471510       1.752566   0.535211     0.422535      0.042254                  7
wolfe_wave_v2    low_vol         transition       5     2       3  0.400000   1.280874  0.256175       1.366069   0.600000     0.400000      0.000000                  2
wolfe_wave_v2    low_vol      trend_aligned      21    13       8  0.619048  22.430379  1.068113       3.452714   0.380952     0.523810      0.095238                  2
```

## Live RR Gate

```text
     strategy  sample_trades  sample_win_rate  recommended_min_rr
   wolfe_wave           2010         0.497015                 1.5
wolfe_wave_v2            484         0.528926                 0.0
```
