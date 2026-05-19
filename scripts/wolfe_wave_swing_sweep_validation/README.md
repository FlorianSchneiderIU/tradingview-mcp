# Wolfe Swing Definition Validation

Generated on 2026-05-18.

This run extended the BTC swing-definition fix to ETH, XRP, and BNB. The search reused the same discipline:

- At least 4 years of 5m data.
- 365-day validation slice.
- 365-day OOS slice.
- Minimum 30 train trades, 15 validation trades, and 30 OOS trades.
- Fixed rolling checks ending on 2025-05-18 and 2026-05-18.

## Result

All three selected configs pass both fixed rolling windows. The shared structural finding is the same as BTC: `15m fractal` swings with a short `pivot_confirm_window=3`. Zigzag and 5m variants were not competitive.

| Symbol | Source | Window | Confirm | Min Score | Min RR | Target Bars | Max Hold | Trend | Min OOS R | Median OOS R |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| ETHUSDT | close | 8 | 3 | 48 | 1.2 | 18 | 96 | rsi | +14.13R | +15.07R |
| XRPUSDT | body | 8 | 3 | 48 | 1.5 | 8 | 288 | rsi | +7.96R | +18.23R |
| BNBUSDT | close | 8 | 3 | 52 | 1.2 | 12 | 288 | none | +12.06R | +13.84R |

## Fixed Rolling Validation

| Symbol | Rolling End | Train Trades | Train R | Validation Trades | Validation R | OOS Trades | OOS R | OOS PF | Pass |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| BNBUSDT | 2025-05-18 | 113 | +26.77R | 62 | +26.51R | 68 | +12.06R | 1.26 | yes |
| BNBUSDT | 2026-05-18 | 175 | +53.28R | 68 | +12.06R | 54 | +15.62R | 1.42 | yes |
| ETHUSDT | 2025-05-18 | 95 | +21.91R | 68 | +11.37R | 63 | +14.13R | 1.38 | yes |
| ETHUSDT | 2026-05-18 | 163 | +33.28R | 63 | +13.25R | 61 | +16.01R | 1.49 | yes |
| XRPUSDT | 2025-05-18 | 93 | +16.21R | 53 | +4.20R | 53 | +7.96R | 1.23 | yes |
| XRPUSDT | 2026-05-18 | 146 | +20.41R | 53 | +7.96R | 62 | +28.50R | 1.76 | yes |

## Latest-Window Robustness

| Symbol | Variants | Passing Variants | Median OOS R | Best OOS R | Worst OOS R |
|---|---:|---:|---:|---:|---:|
| XRPUSDT | 15 | 15 | +29.02R | +37.96R | +23.74R |
| ETHUSDT | 14 | 14 | +15.34R | +18.96R | +9.90R |
| BNBUSDT | 14 | 10 | +14.88R | +22.77R | +3.55R |

## Artifacts

- `rolling_candidate_selection/selected_swing_candidates.json`
- `rolling_candidate_selection/rolling_candidate_metrics.csv`
- `selected_candidates_validation/fixed_window_metrics.csv`
- `selected_candidates_validation/robustness_summary.csv`
- Per-symbol sweep folders:
  - `../wolfe_wave_ethusdt_swing_sweep/`
  - `../wolfe_wave_xrpusdt_swing_sweep/`
  - `../wolfe_wave_bnbusdt_swing_sweep/`

The selected ETH, XRP, and BNB configs have been added to `bot/configs/wolfe_wave_configs.json`.
