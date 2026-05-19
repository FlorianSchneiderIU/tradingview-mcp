# BTC Wolfe Wave Postmortem

Generated on 2026-05-18.

## Executive Summary

BTC did not fail because the final OOS year was bad. It failed because the same intraday Wolfe definitions that worked in the final year were structurally negative in train and validation.

The strongest intraday BTC family was `15m + fractal pivots`. Its median final-year OOS was positive, but its train and validation medians were negative:

| Pattern Family | Median Train R | Median Validation R | Median OOS R | Median OOS Trades |
|---|---:|---:|---:|---:|
| `15m fractal` | -14.84R | -16.63R | +15.68R | 29 |
| `1h fractal` | -1.21R | -1.35R | +8.25R | 6 |
| `15m zigzag` | -54.45R | +7.51R | -2.31R | 40 |
| `5m fractal` | -207.37R | -63.56R | -91.26R | 161 |
| `5m zigzag` | -485.83R | -227.85R | -188.20R | 384 |

This is not a small parameter miss. It is a regime/definition problem.

## Key Findings

### 1. The final BTC year is real but not robust enough

Best final-OOS config:

| Split | Trades | Net R | PF | Win Rate |
|---|---:|---:|---:|---:|
| Train | 65 | -12.08R | 0.79 | 27.7% |
| Validation | 42 | -19.33R | 0.50 | 26.2% |
| OOS | 29 | +19.11R | 2.37 | 62.1% |

The final year works because hit rate jumps, not because the geometry produces better planned reward.

### 2. Dynamic EPA target does not fix BTC

I compared the current fixed projected EPA target against a dynamic 1-4 EPA-line exit on the same detected signals.

For the best final-OOS config:

| Target Mode | Train | Validation | OOS |
|---|---:|---:|---:|
| Fixed projection target | -12.08R | -19.33R | +19.11R |
| Dynamic EPA line target | -13.15R | -22.32R | +15.69R |

The fixed-target implementation is not the primary reason BTC fails. The problem is upstream: admitted patterns stop out too often in earlier regimes.

### 3. Simple entry filters do not rescue it

I tested entry-time filters over:

- score
- planned RR
- p5 break in ATR
- symmetry ratio
- EPA slope
- RSI by direction

No filter with enough train, validation, and OOS trades produced positive train and validation simultaneously.

That means the current geometry score is not separating good BTC Wolfe waves from bad BTC Wolfe waves.

### 4. 5m BTC Wolfe is toxic

The 5m configurations generate plenty of trades, but they are consistently negative. This is likely microstructure noise, not Wolfe Wave behavior.

The statistical sample size is good, but the pattern definition is too permissive at that resolution.

### 5. 4h BTC Wolfe is under-modeled in the current universe runner

The universe runner searched `5m`, `15m`, and `1h` pattern timeframes, but not `4h`.

The older BTC tuner did include `4h`, but `max_hold_bars` is measured in 5m execution bars. That means a 4h Wolfe could be forced out after only 1-2 days even when the 1-4 EPA target naturally needs multiple 4h bars.

A small hand-picked 4h check with scaled holds found some positive configs, but only 3 OOS trades. That is not statistically usable yet.

## Diagnosis

The current BTC problem is not "Wolfe Waves do not work on BTC."

It is:

1. The statistically relevant implementation is mostly an intraday 15m/5m Wolfe system.
2. The 5m version is noise.
3. The 15m version works in the final OOS year but fails badly in prior regimes.
4. The classical BTC swing-Wolfe hypothesis probably belongs on `4h` or higher, but the current research runner does not model those holds correctly and cannot generate enough trades from 5 years of Bybit data.

## Recommended Fixes

1. Add `max_hold_pattern_bars` and convert it to execution bars internally.
2. Add `4h` and possibly `1d` pattern timeframes to a separate swing-Wolfe runner, not the intraday universe runner.
3. Keep the dynamic EPA line as a supported exit mode, but do not expect it to fix BTC by itself.
4. Replace the BTC pivot engine with a volatility-scaled market-structure swing detector instead of 5m/15m fractals.
5. Add a higher-timeframe regime filter before scoring BTC Wolfe waves.
6. Treat BTC separately from the alt universe. It likely needs fewer, larger swing setups rather than more intraday setups.

## Swing Definition Follow-up

Follow-up sweep generated on 2026-05-18.

The failure was primarily in the swing definition. The old fractal implementation used the same `pivot_window` for structure size and right-side confirmation. On BTC that made the signal either too late or too wick-sensitive.

I added two swing-definition controls:

- `pivot_source`: `wick`, `close`, or `body`
- `pivot_confirm_window`: separate right-side confirmation for fractal pivots

BTC improved materially when using 15m fractal swings with a short 3-bar right-side confirmation. Zigzag did not help, and close-only pivots were not robust.

Top validated config:

| Field | Value |
|---|---:|
| `pattern_tf` | `15m` |
| `pivot_method` | `fractal` |
| `pivot_source` | `body` |
| `pivot_window` | `12` |
| `pivot_confirm_window` | `3` |
| `max_time_ratio` | `3.8` |
| `max_p5_break_atr` | `3.0` |
| `stop_atr_buffer` | `0.5` |
| `min_rr` | `1.5` |
| `min_score` | `48` |
| `target_projection_bars` | `18` |
| `max_hold_bars` | `576` |
| `trend_filter` | `rsi` |

Fixed rolling validation with 1-year OOS windows:

| Rolling End | Train Trades | Train R | Validation Trades | Validation R | OOS Trades | OOS R | OOS PF | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 2025-05-18 | 89 | +3.73R | 33 | +30.17R | 49 | +16.24R | 1.53 | yes |
| 2026-05-18 | 122 | +33.90R | 49 | +16.24R | 48 | +17.70R | 1.55 | yes |

Latest-window robustness:

| Variants | Passing Variants | Median OOS R | Best OOS R | Worst OOS R |
|---:|---:|---:|---:|---:|
| 14 | 13 | +15.85R | +25.08R | +4.55R |

Trade distribution for the top config:

| Bucket | Trades | Net R | Avg R | Win Rate | PF |
|---|---:|---:|---:|---:|---:|
| Train | 122 | +33.90R | +0.28R | 42.6% | 1.39 |
| Validation | 49 | +16.24R | +0.33R | 49.0% | 1.53 |
| OOS | 48 | +17.70R | +0.37R | 50.0% | 1.55 |

The current BTC candidate has been added to `bot/configs/wolfe_wave_configs.json`.

## Artifacts

- `btc_postmortem_overview.csv`
- `btc_config_family_summary.csv`
- `btc_top_oos_configs.csv`
- `btc_trade_level_filter_grid.csv`
- `btc_4h_handpicked.csv`
- `../wolfe_wave_btc_swing_sweep/btc_swing_definition_sweep.csv`
- `../wolfe_wave_btc_swing_sweep/btc_swing_definition_grouped.csv`
- `../wolfe_wave_btc_swing_sweep/validation_candidate_1/`
