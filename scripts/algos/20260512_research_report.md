# 2026-05-12 Pine Strategy Candidate Research

## TradingView MCP Check

All staged Pine files in `scripts/algos/20260512` were analyzed and compile-checked with the local TradingView MCP CLI. The server-side check returned `compiled=true` and `error_count=0` for every file.

Warnings:

| File | Warnings | Notes |
| --- | ---: | --- |
| `Indi 170.txt` | 4 | Liquidity Matrix visual/sweep dashboard. |
| `Indi 172.txt` | 10 | S/R, ICT candles, FVG, HA-supertrend signal. |
| `Indi 177.txt` | 0 | Clean pivot trendline breakout indicator. |
| `MELONA CONFIRMER U2 (by 3dots).txt` | 0 | Visual confirmer, no actionable alert. |
| `MELONA CONFIRMER U3 R1 (by 3dots).txt` | 0 | HA-supertrend signal variant. |
| `MELONA OB STRTEGY - Update 9 R1/R2/R3` | 26/26/23 | Large visual OB dashboard; many Pine consistency warnings. |

## Extracted Python Atoms

Implemented in `scripts/experiment_pine_strategy_candidates.py`:

| Python atom | Pine source | Signal definition |
| --- | --- | --- |
| `pivot_breakout` | `Indi 177` | Connect confirmed pivot highs/lows, enter on close breaking active trendline. |
| `ha_supertrend` | `Indi 172`, `MELONA CONFIRMER U3` | Heikin-Ashi ATR supertrend direction flip. |
| `liquidity_sweep` | `Indi 170` | Wick sweep of recent swing high/low with reclaim and volume/wick filters. |
| `melona_pressure` | MELONA OB | Single-candle order-block pressure pattern. |
| `demarker_exhaustion` | MELONA OB | Major TP-point/exhaustion reversal counter. |
| `melona_trendline` | MELONA OB | EzTrendline break atom. |

Backtests enter at the next candle open after the Pine signal bar and resolve same-bar TP/SL collisions conservatively as SL first.

## Main Top-50 Smoke Backtest

Universe: top Bybit USDT perpetual candidates from `bybit_top50_turtle_per_symbol_turnover_sfp_tex_v4_universe.csv`.

Data: cached 5m data resampled to 15m, `2024-01-01` to `2026-04-20`, OOS split `2025-07-01`.

Representative raw 15m settings:

| Strategy | OOS trades | OOS net R | Avg R | Win rate | PF | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `pivot_breakout` | 21,081 | -1,969.88 | -0.0934 | 35.28% | 0.874 | Best raw, still negative. |
| `demarker_exhaustion` | 22,767 | -5,292.71 | -0.2325 | 38.10% | 0.682 | Reject. |
| `melona_pressure` | 42,583 | -6,912.69 | -0.1623 | 39.39% | 0.765 | Raw reject. |
| `liquidity_sweep` | 48,893 | -11,368.54 | -0.2325 | 38.61% | 0.683 | Reject; worse than current Turtle Soup. |
| `ha_supertrend` | 59,784 | -12,518.99 | -0.2094 | 39.13% | 0.710 | Reject. |

Artifacts:

- `scripts/pine_strategy_candidate_research_top50_smoke15m_summary.csv`
- `scripts/pine_strategy_candidate_research_top50_smoke15m_trades.csv`
- `scripts/pine_strategy_candidate_research_top50_smoke15m.md`

## ML Rescue Test

Random-forest filters were trained on pre-split trades and tested OOS. Only `melona_pressure` showed an apparent rescue on the fixed `2025-07-01` split:

| Strategy | Threshold | OOS trades | OOS net R | Avg R | Win rate | PF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `melona_pressure` | 0.55 | 630 | +60.14 | +0.0955 | 48.89% | 1.167 |
| `pivot_breakout` | 0.50 | 9,249 | -822.33 | -0.0889 | 36.23% | 0.881 |
| `demarker_exhaustion` | 0.50 | 9,667 | -1,872.95 | -0.1937 | 38.80% | 0.727 |
| `ha_supertrend` | 0.50 | 25,457 | -3,923.02 | -0.1541 | 41.38% | 0.779 |
| `liquidity_sweep` | 0.50 | 24,181 | -4,707.88 | -0.1947 | 39.94% | 0.727 |

The pressure model relied mostly on trend/position features (`close_vs_ema200`, direction, `close_vs_ema50`, ATR%, risk%, hour/day).

## Robustness Check

The promising `melona_pressure` result is not stable enough:

| Split | Threshold | OOS selected trades | OOS net R | Avg R | PF |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2025-01-01 | 0.55 | 778 | +0.80 | +0.0010 | 1.002 |
| 2025-04-01 | 0.55 | 1,339 | -11.84 | -0.0088 | 0.986 |
| 2025-07-01 | 0.55 | 176 | +37.04 | +0.2104 | 1.405 |
| 2025-10-01 | 0.55 | 310 | -40.02 | -0.1291 | 0.809 |

Additional pressure parameter tuning on the top-12 subset was also weak:

| Best tuned pressure variant | OOS trades | OOS net R | Avg R | PF |
| --- | ---: | ---: | ---: | ---: |
| `rr=1.5`, `sl_atr=0.2`, `hold=96`, ML threshold 0.55 | 172 | +0.31 | +0.0018 | 1.003 |

## Decision

Do not wire these Pine candidates into the live bot yet.

The only potentially interesting atom is MELONA OB `melona_pressure`, but the edge is split-sensitive and disappears under a smaller tuning universe. I kept the Python implementation as a reproducible research harness, but did not add a live strategy engine or Docker/env wiring.

Most useful follow-up if we revisit this:

- Treat `melona_pressure` as a feature inside the existing Turtle Soup/ORB models, not as a standalone signal.
- Test symbol-specific pressure models only on symbols where the pooled OOS breakdown was positive across at least 10 trades.
- Add pressure/FVG/order-block context to the current Turtle Soup feature set before replacing any existing working strategy.
