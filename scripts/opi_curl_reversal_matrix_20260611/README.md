# OPI Curl Reversal Matrix Investigation

Run date: 2026-06-11

Source:
- Matrix room: `!YURVFZgoYWChTYaXbO:thebox.sbs`
- Room name: `Opi`
- Messages fetched: 3000
- Parsed signals: 2135
- Window: 2026-06-06 to 2026-06-11

Parsed signal types:
- `domino_turn`: 1079
- `bias_flip`: 692
- `move_alert`: 270
- `opi_full`: 94

Main result:
- Broad OPI curl/reversal signals are not profitable as a raw strategy.
- Explicit OPI SL/TP setups are also slightly negative overall:
  - `declared_entry_levels`: 93 trades, -7.38R, -0.079R avg, PF 0.84
  - `next_open_levels`: 93 trades, -7.00R, -0.075R avg, PF 0.85
- The room is very chatty, so raw independent trade counting overstates clustered signal opportunities.

Explicit OPI setup pockets:
- BTC 5m: 15 trades, +3.14R, +0.209R avg, PF 1.76
- XLM 5m: 9 trades, +2.79R, +0.310R avg, PF 1.48
- XAU 5m: 7 trades, +1.06R, +0.152R avg, PF 1.91

Structural alert pockets after 30-minute cooldown:
- XLM 30m, swing80 TP3R: 62 trades, +18.25R, +0.294R avg, PF 1.51
- XLM 30m, swing40 TP3R: 62 trades, +17.99R, +0.290R avg, PF 1.45
- XLM 30m, swing20 TP3R: 62 trades, +17.21R, +0.278R avg, PF 1.41
- XRP 3m, swing80 TP3R: 127 trades, +29.74R, +0.234R avg, PF 1.40
- XLM 3m, swing80 TP2R/TP3R: positive, but weaker than XLM 30m.

Interpretation:
- Do not deploy all OPI messages.
- The best current hypothesis is a shadow-only OPI strategy using alert-only entries with structural stops, fee-valid filtering, symbol/timeframe filters, and a cooldown.
- Candidate filters to shadow first:
  - `XLMUSDT` 30m alerts, `swing80_tp3R` or neighboring `swing40/swing20_tp3R`
  - `XRPUSDT` 3m alerts, `swing80_tp3R`
  - Explicit `opi_full` BTC/XLM/XAU 5m setups as a separate small sample watchlist

Files:
- `opi_curl_reversal_signals.csv`
- `opi_curl_reversal_backtest_trades.csv`
- `opi_curl_reversal_cooldown30_summary_by_variant_symbol_timeframe.csv`
- `opi_curl_reversal_nonoverlap_symbol_summary_by_variant_symbol_timeframe.csv`

