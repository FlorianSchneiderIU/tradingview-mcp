# OPI Brain Matrix Investigation

Run date: 2026-06-14

## Cohort

- Matrix room: `!YURVFZgoYWChTYaXbO:thebox.sbs`
- Window: 2026-05-13 00:00 UTC through 2026-06-13 18:50 UTC
- Refined `opi_brain` alerts: 117
- Symbols: XAG 23, BTC 22, XAU 22, ETH 20, SOL 12, XRP 9, XLM 6, ADA 3
- Median declared target distance: 0.710%
- Median declared stop distance: 2.000%

The May 13 start is the point at which the room history reproduces the
report's exact 117-alert count. The report UI's April 28 date is the wider
dashboard data window, not the start of this OPI-brain cohort.

## Replay

One-minute Bybit candles, immediate signal entry, deterministic intrabar
ordering, 0.055% taker fee per side:

| Exit model | Closed | Open | Wins | Losses | Win rate | Gross sum | Net sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Report labels: 0.71% SL, 0.75% trigger, 0.40% trail | 112 | 5 | 40 | 72 | 35.7% | -18.09% | -30.39% |
| Alert levels: declared 2% SL, declared 0.71% target trigger, 0.40% trail | 111 | 6 | 69 | 42 | 62.2% | -32.74% | -44.92% |

The external `99/18` and 84.6% win-rate result is therefore not reproducible
from the parameters visible in the screenshot. Timestamp alignment is not the
cause: the median difference between the alert entry and next one-minute open
is approximately zero.

## Live Audit

- Matrix breakeven updates were sent through an unauthenticated local executor.
  Every recent update failed with `Authenticated endpoints require keys`.
- The Matrix RL executor converted trailing requests to an attached fixed
  partial TP/SL. OPI positions closed fully at the alert target instead of
  arming a runner.
- Exit notices were matched to the newest symbol/direction reference, so
  overlapping trades could target the wrong entry.
- The parser labeled refined OPI-brain alerts as `opi_full` and only accepted
  BTC, XLM, and XAU 5m.

## Action

- Keep the existing BTC/XLM/XAU 5m candidate scope.
- Classify refined alerts as `opi_brain`.
- Treat the alert target as the trailing activation price.
- Keep the alert's declared initial stop.
- Use a decision-specific partial stop order, move it to breakeven at
  activation, then ratchet it by 0.40%.
- Route channel TP/ratchet notices to the authenticated Matrix RL executor.
- Do not expand to the full eight-symbol cohort until the external report's
  outcome semantics can be reproduced independently.

Artifacts:

- `opi_brain_signals.csv`
- `opi_brain_report_config_trades.csv`
- `opi_brain_trailing_grid.csv`
