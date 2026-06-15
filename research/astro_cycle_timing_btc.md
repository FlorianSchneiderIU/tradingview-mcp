# BTC Astrology/Cycle Reversal Timing Research

Date: 2026-06-14

This is research, not financial advice. The useful question is not whether a chart can be made to look magical after the fact, but whether a mechanically defined timing model improves out-of-sample reversal-window density and trade selection versus calendar baselines and placebo calendars.

## Executive Conclusion

I do **not** validate the strong claim that astrology/cycle-only features can consistently predict BTC MTF/HTF reversal windows within 15 minutes.

The best result is narrower and more practical:

1. Astrology-only and astrology+cycle-only models did **not** beat a 37-day shifted placebo on direct 15m reversal-window prediction.
2. A trade-level selector using LTF sweep quality, calendar features, and real astrology/cycle timing-head scores produced a positive 10R walk-forward candidate on 2025-01 through 2026-05.
3. The current candidate is not a pure astrology strategy. It is a 5m liquidity-sweep strategy with a real astro/cycle timing overlay.
4. The edge is moderate, target-sparse, and still research-grade until it survives more assets, more placebo calendars, and live/paper walk-forward.

My current view: astrology/cycle timing is worth researching as a **trade-selection feature**, not as a standalone reversal oracle.

Update after the LTF probability reframing: the best-supported claim is now even more specific. Time/calendar features do contain a small but real-looking OOS signal for **probability of a 5m reversal within the next N candles**. The signal is not strong enough to trade by itself, but it is strong enough to justify a calendar-probability head inside the LTF setup selector.

Update after completed-candle confirmation: the hierarchy is more useful as a
two-step system than as a direct entry oracle. The pre-candle hierarchy supplies
the timing prior. After a candidate candle closes, completed price structure
strongly updates the probability that it contained the parent high or low.
Calendar features do not improve that posterior. A small short-side 5R retest
candidate survives validation and the locked test, but no 10R version is
validated and the sample remains too small for live use.

## Sources And Context

Relevant external reference points:

- Yuan, Zheng, and Zhu's 2006 Journal of Empirical Finance paper found lower global stock returns around full moons than new moons, but the effect is broad/daily and not the same as intraday BTC reversal timing: https://ideas.repec.org/a/eee/empfin/v13y2006i1p1-23.html
- The paper itself warns about spurious patterns from historical return mining, which is exactly the danger here: https://www.bus.umich.edu/pdf/mitsui/workshopdocs/ZhengMoonstruck.pdf
- A finance astrology test, "Today is a 7", reports limited economic usefulness from simple astrological stock-sign effects: https://jfi-aof.org/index.php/jfi/article/download/2622/2160
- Market data was pulled from Bybit's official V5 kline endpoint: https://bybit-exchange.github.io/docs/v5/market/kline
- Planetary positions were generated with Skyfield/JPL ephemerides: https://rhodesmill.org/skyfield/planets.html and https://ssd.jpl.nasa.gov/planets/eph_export.html
- The classifiers use regularized scikit-learn models: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

## First-Pass Timing Test

Data:

- Symbol: `BTCUSDT` Bybit USDT perpetual.
- Timing bars: 15m candles from 2021-01-01 to 2026-06-01.
- Execution bars: 5m candles from 2025-01-01 to 2026-06-01.
- Splits:
  - Train/discovery: 2021-01-01 through 2023-12-31.
  - Validation: 2024-01-01 through 2024-12-31.
  - Test: 2025-01-01 through 2026-06-01.

Swing label:

- 15m directional-change pivots using an 8 ATR reversal threshold.
- Total pivots: 1,670, balanced 835 highs and 835 lows.
- Base 15m reversal-window rate in test: 0.8256%.
- Target: `1` when an 8 ATR pivot occurs inside the current/next 15m window.

Feature sets:

- `calendar_only`: time of day, day of week, month, day of month, weekend flag.
- `cycle_only`: fixed harmonic cycles including lunar synodic/draconic/anomalistic, Mercury/Venus/Mars synodic, Jupiter/Saturn cycles, and BTC halving-like period.
- `astro_only`: geocentric Sun/Moon/planet longitudes, latitude, declination, distance, apparent speed, retrograde flags, Moon phase harmonics, pairwise planetary angles, and aspect-cluster counts.
- `astro_cycle`: cycle + astrology.
- `all_real`: calendar + cycle + astrology.
- `placebo_shift_37d`: same feature dimensionality, but astro/cycle features shifted by 37 days while calendar controls remain real.

Out-of-sample direct reversal-window results:

| Model | AP | ROC AUC | Top 2% precision | Lift vs base | Event recall |
|---|---:|---:|---:|---:|---:|
| calendar_only | 0.01032 | 0.571 | 1.413% | 1.71x | 3.42% |
| astro_only | 0.00811 | 0.485 | 0.908% | 1.10x | 2.20% |
| astro_cycle | 0.00812 | 0.488 | 0.706% | 0.86x | 1.71% |
| all_real | 0.00951 | 0.544 | 1.110% | 1.34x | 2.69% |
| placebo_shift_37d | 0.00971 | 0.542 | 1.514% | 1.83x | 3.67% |

Interpretation:

- The direct timing model does **not** establish an astrology edge.
- Calendar-only is more stable than astrology-only.
- The shifted-placebo calendar beats `all_real` on top-2% direct reversal-window precision in the untouched test period.
- Without placebo calendars, the `all_real` model would look mildly promising and could easily be overinterpreted.

## LTF Calendar Probability Test

This follow-up steps back from full trade construction and asks a cleaner question:

> Given only time-derived features, how well can we predict `P(reversal within the next N 5m candles)`?

Data and labels:

- Symbol: `BTCUSDT` Bybit USDT perpetual.
- Bars: 5m candles from 2021-01-01 through 2026-06-01.
- Train: 2021-01-01 through 2023-12-31.
- Validation: 2024-01-01 through 2024-12-31.
- Test: 2025-01-01 through 2026-06-01.
- Reversal event: 5m ATR directional-change pivot.
- Swing thresholds tested: 6 ATR and 8 ATR.
- Horizons tested: next 3, 6, 12, and 24 candles, meaning 15m, 30m, 60m, and 120m.

Feature/model families:

- `civil`: time of day, minute, day of week, day/month/year seasonality, weekend flag, and session flags.
- `civil_cycle`: civil features plus fixed lunar/planetary/BTC-cycle harmonics.
- `astro_core`: cached Skyfield geocentric planetary features, reduced to a core set.
- `all_time`: civil + fixed cycles + astro core.
- Calendar-bin models: smoothed empirical rates for recurring slots such as hour-of-week, 5m slot-of-week, month-hour, lunar phase bins, and mixed civil/lunar bins. Smoothing is selected on 2024 validation data only.

Linear/logistic probability results for the cleaner 8 ATR swing label:

| Horizon | Test base rate | Civil AP | Civil AUC | Astro-core AP | Astro-core AUC | All-time AP | All-time AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 bars / 15m | 2.548% | 0.02748 | 0.524 | 0.02558 | 0.500 | 0.02583 | 0.506 |
| 6 bars / 30m | 5.095% | 0.05508 | 0.525 | 0.05121 | 0.500 | 0.05147 | 0.505 |
| 12 bars / 60m | 10.185% | 0.10934 | 0.527 | 0.10224 | 0.501 | 0.10303 | 0.506 |
| 24 bars / 120m | 20.280% | 0.21266 | 0.524 | 0.20494 | 0.506 | 0.20709 | 0.512 |

Interpretation of the linear test:

- Civil calendar features do have OOS signal, but it is modest: AUC is roughly 0.52 to 0.53.
- Astro-core features alone are near random in this setup.
- Adding fixed cycles/astro to the civil model usually dilutes the civil signal rather than improving it.
- A nonlinear HGB run on civil features did not beat the simpler civil/logistic or calendar-bin models.

Best calendar-bin probability models on the 8 ATR label:

| Horizon | Test base rate | Best AP model | Test AP | AUC | Best actionable window model | Test coverage | Precision | Lift |
|---:|---:|---|---:|---:|---|---:|---:|---:|
| 3 bars / 15m | 2.548% | `ensemble_civil` | 0.02750 | 0.527 | `hour_week_moon8` | 0.534% | 4.786% | 1.88x |
| 6 bars / 30m | 5.095% | `hour_week` | 0.05614 | 0.539 | `hour_week_moon8` | 0.525% | 9.103% | 1.79x |
| 12 bars / 60m | 10.185% | `ensemble_top8_val_ap` | 0.11338 | 0.539 | `month_hour` | 0.501% | 17.070% | 1.68x |
| 24 bars / 120m | 20.280% | `ensemble_top8_val_ap` | 0.22164 | 0.543 | `month_hour` | 1.001% | 31.183% | 1.54x |

For the 6 ATR label, the broad best-AP models reached AUC around 0.529 to 0.539. The highest-lift tiny windows were often cycle-derived and reached roughly 1.7x to 1.9x OOS lift at about 0.4% to 0.5% test coverage, but those cycle windows had near-flat global AUC and very low event recall. I would treat them as candidate filters, not as proof.

What this means:

- Yes, the model can infer something useful from calendar/time features OOS.
- No, this is not an accurate standalone reversal clock. The broad probability edge is small.
- The most useful shape is a **rare-window filter**: top calendar windows can nearly double reversal density, but they catch only a small fraction of all reversal events.
- For strategy building, the best next use is to feed `P(reversal in next 6 to 12 candles)` into the LTF sweep/meta selector, not to open trades purely because a calendar window is active.
- The most practical horizons for the existing 5m sweep strategy are likely 6 and 12 candles, because they align with a 30 to 60 minute setup/entry window without making the label too common.

## Hierarchical Parent-Extreme Model

The stronger formulation is a top-down fractal model:

1. 4h predicts whether the next 4h candle prints the 1d high or low.
2. 1h predicts whether the next 1h candle prints the containing 4h high or low.
3. 15m predicts whether the next 15m candle prints the containing 1h high or low.
4. 5m predicts whether the next 5m candle prints the containing 15m high or low.

This is different from the earlier ATR-pivot question. Here the labels are parent-candle extremes, separated into high and low heads. The model uses only information available up to the current child candle close: parent slot, running parent high/low so far, ATR-normalized candle/body/wick/return features, EMA/RSI/volume context, and time/cycle features.

Strict next-child-candle HGB results, test 2025-01 through 2026-05:

| Layer | Direction | Base | AP | AUC | Validation-threshold coverage | Precision | Lift |
|---|---|---:|---:|---:|---:|---:|---:|
| 4h -> 1d | low | 16.672% | 0.3803 | 0.776 | 0.743% | 69.565% | 4.17x |
| 4h -> 1d | high | 16.672% | 0.3739 | 0.764 | 1.519% | 76.596% | 4.59x |
| 1h -> 4h | low | 24.994% | 0.4936 | 0.761 | 2.229% | 83.333% | 3.33x |
| 1h -> 4h | high | 24.994% | 0.4895 | 0.754 | 0.614% | 85.526% | 3.42x |
| 15m -> 1h | low | 24.998% | 0.5123 | 0.771 | 0.876% | 87.788% | 3.51x |
| 15m -> 1h | high | 25.001% | 0.5018 | 0.769 | 1.042% | 83.527% | 3.34x |
| 5m -> 15m | low | 33.334% | 0.5767 | 0.755 | 0.448% | 88.138% | 2.64x |
| 5m -> 15m | high | 33.333% | 0.5749 | 0.751 | 0.792% | 87.935% | 2.64x |

This validates the architecture much more strongly than the flat calendar-only reversal model. The heads can localize parent-candle extremes OOS with useful lift at every layer.

Important caveat: wider labels become almost trivial. For example, asking whether a 4h extreme appears within the next 4 hourly candles has an 82% base rate, and within 8 hourly candles it is effectively 100%. Those wide windows are useful as search regions, but they should not be the main binary target. The actionable heads should remain strict next-child-candle localizers.

Feature ablation with strict logistic models:

| Feature set | Mean AP | Mean AUC | Mean validation-threshold lift |
|---|---:|---:|---:|
| price/parent geometry | 0.4275 | 0.743 | 2.03x |
| price + time/cycles | 0.4241 | 0.741 | 2.00x |
| time/cycles only | 0.2957 | 0.577 | 1.54x |

Interpretation:

- The hierarchy edge mostly comes from parent-candle microstructure and position, not astrology/calendar features.
- Time/cycle features are not useless, but they are secondary in this formulation.
- The most promising strategy architecture is now:
  1. Use the 4h head to identify candidate next-4h daily high/low windows.
  2. Inside selected 4h windows, rank 1h bars with the strict 1h -> 4h head.
  3. Inside selected 1h windows, rank 15m bars with the strict 15m -> 1h head.
  4. Inside selected 15m windows, rank 5m bars with the strict 5m -> 15m head.
  5. Execute only when this cascade agrees with an LTF sweep/displacement entry model.

## Causal Cascade Backtest

I combined the four strict heads into one causal score. For a candidate 5m candle:

- The 4h score was made by the preceding 4h candle for the containing 4h target.
- The 1h score was made by the preceding 1h candle for the containing 1h target.
- The 15m score was made by the preceding 15m candle for the containing 15m target.
- The 5m score was made by the preceding 5m candle for the target 5m candle.
- Each raw score was converted to a percentile using the frozen 2024 validation distribution.
- The cascade score was the minimum of the four percentiles, so every layer had to pass.

The exact nested event is rare: the target 5m candle must be the correct 15m extreme, its 15m candle the correct 1h extreme, its 1h candle the correct 4h extreme, and its 4h candle the correct daily extreme. Its test base rate is 0.347%.

Test localization, 2025-01 through 2026-05:

| Direction | Cascade percentile | Coverage | Exact-event precision | Lift | Event recall |
|---|---:|---:|---:|---:|---:|
| high | 0.80 | 0.704% | 5.163% | 14.87x | 10.47% |
| high | 0.90 | 0.158% | 10.213% | 29.41x | 4.65% |
| high | 0.93 | 0.030% | 13.333% | 38.40x | 1.16% |
| low | 0.80 | 0.928% | 4.931% | 14.20x | 13.18% |
| low | 0.90 | 0.215% | 6.250% | 18.00x | 3.88% |
| low | 0.93 | 0.042% | 3.226% | 9.29x | 0.39% |

These figures are from the corrected pipeline: missing values are imputed from training data only, the future-dependent complete-parent flag is removed, cascade percentiles use the training-era score distribution, and the holding period is exactly 288 bars.

At the 0.80 threshold, high-side lift progressed from 2.17x at the 4h layer to 4.21x for 4h + 1h, 8.79x after adding 15m, and 14.87x for the exact four-layer event. The low side showed the same compounding pattern, ending at 14.20x lift.

This is the strongest result in the project so far: the hierarchy genuinely concentrates nested swing timing OOS. It is not merely four individually decent classifiers placed next to each other.

Execution tests used the predicted 5m candle as an anchor, required a rejection, 12-bar sweep, or displacement confirmation, entered at the next 5m open, charged 8 bps round trip, and resolved same-bar stop/target collisions stop-first. A minimum 0.5% to 1.0% stop distance prevented fees from dominating R.

| Variant | Validation | Test | Interpretation |
|---|---:|---:|---|
| Validation-selected: long rejection, 1.0% risk, 2R, 0.90 cascade | 54 trades, +13.11R, PF 1.42 | 67 trades, -26.83R, PF 0.46 | failed OOS |
| Both directions: 12-bar sweep, 0.5% risk, 1.5R, 0.93 cascade | 35 trades, +9.72R, PF 1.56 | 36 trades, -9.11R, PF 0.66 | failed OOS |
| Short rejection, 0.5% risk, 1.5R, 0.93 cascade | 12 trades, +1.25R, PF 1.18 | 12 trades, +1.09R, PF 1.16 | positive but far too small |
| Long 12-bar sweep, 1.0% risk, 2R, 0.93 cascade | 16 trades, +9.07R, PF 2.20 | 19 trades, +0.06R, PF 1.01 | edge vanished |

The first 5R pilot also failed badly because one-candle invalidation stops made 8 bps of costs worth roughly 0.3R to 0.7R per trade. Wider fee-aware stops fixed that mechanical problem, but not the regime instability.

Conclusion:

- The hierarchy is validated as a reversal-window localizer.
- The tested candle-pattern entries are not validated as a strategy.
- There is currently no evidence supporting consistent 1:10 to 1:30 trades from this cascade.
- High-side localization was more stable than low-side localization, but direction selection itself overfit validation.
- The next execution model should be trained walk-forward on post-window path outcomes instead of selecting a fixed rejection/sweep rule from one validation year.

## Walk-Forward Path Outcome Model

I implemented the next execution layer as a leakage-controlled monthly walk-forward experiment.

Dataset and labels:

- One candidate per predicted 5m reversal window and direction.
- The predicted 5m candle is allowed to close; decision and entry occur at the next 5m open.
- Structural stop is beyond the predicted candle extreme, with a minimum 0.5% or 1.0% distance.
- Separate heads predict whether 1.5R, 2R, 3R, 5R, or 10R is reached before the stop within exactly 288 bars.
- A target is positive only when the target is actually touched before the stop. Profitable timeouts are not positive labels.
- Training and validation rows are included only when their complete label path ended before the next fold begins.
- Validation chooses the score coverage using realized net R after fees and one-position-at-a-time simulation. The month is skipped if validation has no positive configuration.

To provide enough history for a fair 12-month validator, cascade predictions are prequential:

- 2023 predictions come from hierarchical models trained through 2022.
- 2024 onward predictions come from models trained through 2023.
- Only these OOS cascade predictions enter the execution learner.

The attractive intermediate result was:

| Configuration | Validation history | Trades | Net R | PF | Max DD |
|---|---:|---:|---:|---:|---:|
| Combined cascade + price, both directions, 3R, 0.5% stop | 3 months | 192 | +21.80R | 1.15 | -21.64R |
| Combined cascade + price, long only, 3R, 0.5% stop | 3 months | 131 | +22.83R | 1.23 | -20.36R |
| Combined cascade + price, long only, 3R, 0.5% stop | 6 months | 141 | +25.49R | 1.24 | -16.28R |

This did not survive the fair 12-month prequential test:

| Configuration | Trades | Target hits | Net R | PF | Max DD |
|---|---:|---:|---:|---:|---:|
| Combined cascade + price, long, 3R, 0.5% stop | 62 | 10 | -28.77R | 0.50 | -32.65R |
| Combined cascade + price, long, 3R, 1.0% stop | 113 | 16 | -28.61R | 0.69 | -36.52R |
| Price only inside the cascade-gated universe, long, 3R, 1.0% stop | 97 | 13 | -27.81R | 0.65 | -32.16R |

Model and target robustness checks also failed:

- ExtraTrees at 3R lost 13.73R for combined features and 26.91R for price-only features.
- A calibrated multi-head model that selected dynamically among 1.5R, 2R, 3R, 5R, and 10R lost 11.46R for combined features.
- The dynamic cascade-only variant made only +6.52R with PF 1.03 and was not stable enough to treat as evidence.
- No 5R or 10R variant produced credible target-hit-based profitability. The few superficially positive 10R rows depended on timeout PnL, not 10R targets.

Interpretation:

- Short validation windows adapted favorably to part of 2025 but promoted a regime that failed later.
- The hierarchy improves the density of exact extremes, but an exact extreme is not automatically a good trade after waiting for the candle close, paying costs, and placing a feasible stop.
- The model can identify **when an extreme is likely**, but it has not learned **whether price will depart from that extreme far enough to support stable positive R**.
- The current evidence supports the hierarchy as an alert/localization system, not as an autonomous trading strategy.

## 1m Wyckoff Execution

I replaced the 5m direct entry with a causal 1m Wyckoff entry model. The test uses 1,795,681 Bybit BTCUSDT minute bars from 2023-01-01 through 2026-06-01.

Mechanical setup:

1. A hierarchical high/low cascade window reaches at least the 0.60 percentile on all four layers.
2. Starting at the predicted 5m candle open, search the next 15 one-minute bars.
3. Define the local trading range from the preceding 20 completed one-minute bars.
4. Long spring: price trades below the range low, closes back above it, leaves at least a 25% lower wick, closes in the upper 45% of the candle, and volume is at least the prior 20-bar average.
5. Short upthrust: the exact mirrored condition above the range high.
6. Three entry stages are tested:
   - `spring`: next 1m open after the reclaim candle.
   - `sos`: next 1m open after price closes beyond the spring/upthrust candle in the reversal direction with at least a 0.10 ATR body.
   - `test`: next 1m open after a lower-volume retest holds the spring/upthrust extreme.
7. Stop is beyond the spring/upthrust extreme plus 0.05 ATR, with minimum risk of 0.25% or 0.50%.
8. Targets are 1.5R, 2R, 3R, 5R, and 10R; maximum hold is 1,440 one-minute bars.
9. Same-bar collisions are resolved stop-first and costs remain 8 bps round trip.

The candidate cache contains 27,096 setup/stop-policy rows:

- 5,838 immediate spring/upthrust entries.
- 4,097 SOS/SOW confirmations.
- 3,613 low-volume tests.
- Each setup is represented under both stop floors.

Fair 12-month prequential walk-forward results:

| Variant | Trades | Net R | PF | Max DD | Status |
|---|---:|---:|---:|---:|---|
| Short, all Wyckoff stages, cascade + Wyckoff, 1.5R, 0.50% stop | 31 | +3.95R | 1.24 | -3.12R | small positive |
| Short SOS only, cascade + Wyckoff, 1.5R, 0.50% stop | 33 | +2.66R | 1.14 | -6.86R | small positive |
| Short SOS only, Wyckoff-only ExtraTrees, 1.5R | 28 | +0.02R | 1.00 | -5.70R | flat |
| Short SOS only, cascade + Wyckoff, 2R | 14 | -8.69R | 0.32 | -7.53R | failed |
| Short SOS only, cascade + Wyckoff, 3R | 26 | -9.39R | 0.60 | -20.11R | failed |

The small 1.5R result is not robust:

- The matching SOS model lost 5.40R with a 3-month validator.
- It lost 13.58R with a 6-month validator.
- ExtraTrees lost 6.09R on the combined 1.5R SOS setup.
- Dynamic selection across 1.5R to 10R lost 13.66R.
- Rules-only variants selected on 2024 usually failed badly in 2025-2026.

The rules-only 10R short-test variant at a 0.90 cascade threshold finished test at +6.71R, but only one trade actually reached 10R; most positive contribution came from timeout exits. It does not validate a 10R strategy.

Conclusion:

- Entering on 1m preserves more excursion than waiting for the 5m candle to close, but the tested Wyckoff rules still do not produce stable high-RR expectancy.
- The only surviving pocket is short-side 1.5R, and its model/validation sensitivity is too high for live use.
- The implementation is useful as a research and alert layer. It should remain disabled for autonomous trading.

## Richer 1m Context And Excursion Policies

This section tested the wrong execution formulation for the final strategy:
it tried to improve immediate Wyckoff entries and their exits. It is retained
as a negative control, but it is superseded by the hot-candle retest experiment
below.

I tested whether the weak 1m execution result could be rescued by predicting the
distribution of favorable excursion instead of a single fixed-R target.

New causal features available at the signal close:

- Running daily range, directional position, and daily VWAP distance.
- Prior-day and prior-week high/low distance and sweep flags.
- Asia, London, New York, and late-session range, position, and VWAP distance.
- Rolling 15m, 1h, 4h, and 24h range, return, realized volatility, and trend efficiency.
- Range/ATR compression and short-term volume impulse.
- The existing hierarchical cascade, local candle, and Wyckoff-quality features.

Separate heads estimate the probability of reaching 1R, 1.5R, 2R, 3R, 5R,
and 10R before the structural stop. Probabilities are forced to be monotone.
Validation then selects either no trade, a fixed 1.5R/2R target, or a partial
exit with a 5R/10R runner. Runner stops move to breakeven after the partial.
All folds purge paths whose labels were unresolved at the fold boundary.

The richer feature set did not improve excursion ranking. Mean test AUC was
approximately 0.48 to 0.51 for the 1R to 5R heads. The sparse 10R head reached
about 0.55, but its training base rate was only about 3% and it did not produce
a robust trading result.

Feature comparison, short side, fixed 1.5R, 0.50% minimum stop, 12-month validator:

| Features | Trades | Net R | PF | Max DD | Positive trading months |
|---|---:|---:|---:|---:|---:|
| Cascade + local Wyckoff/price | 24 | +4.57R | 1.39 | -2.32R | 8 |
| Cascade + local + rich context | 12 | +2.23R | 1.38 | -3.48R | 4 |
| Local Wyckoff/price only | 16 | -1.00R | 0.90 | -3.12R | 3 |
| Local + rich context, no cascade | 9 | -2.94R | 0.58 | -4.64R | 3 |

The apparently best fixed-target result was not robust:

| Robustness check | Trades | Net R | PF | Max DD |
|---|---:|---:|---:|---:|
| Logistic, 12-month validation | 24 | +4.57R | 1.39 | -2.32R |
| Logistic, 6-month validation | 49 | -11.79R | 0.66 | -17.87R |
| Logistic, 3-month validation | 27 | -11.46R | 0.45 | -14.88R |
| ExtraTrees, 12-month validation | 15 | -7.36R | 0.42 | -8.70R |
| Spring/upthrust entries only, 12-month validation | 45 | +2.31R | 1.09 | -6.81R |

Partial runners degraded the result:

| Policy, rich combined features | Trades | Net R | PF |
|---|---:|---:|---:|
| Adaptive policy choice | 15 | +2.11R | 1.20 |
| Fixed 2R | 13 | +1.07R | 1.13 |
| 50% at 1.5R, runner to 5R | 16 | -5.55R | 0.52 |
| 75% at 1.5R, runner to 5R | 9 | -6.19R | 0.24 |
| 50% at 2R, runner to 10R | 8 | -5.28R | 0.24 |

Conclusion:

- The hierarchical timing cascade still adds information relative to local price action alone.
- More rolling OHLCV context does not rescue the execution layer and often dilutes the small cascade signal.
- Fixed 1.5R is less bad than high-RR runners, but its sensitivity to validation length and model family rules out live use.
- The next meaningful data expansion is event-level microstructure: bid/ask spread and imbalance, aggressive trade delta, liquidation bursts, open-interest change, perp basis, and cross-venue lead/lag. Adding more transformations of the same candles is unlikely to solve the problem.

## Hierarchy-Gated Hot-Candle Retest

The corrected execution hypothesis is:

1. The 4h, 1h, 15m, and 5m hierarchy identifies a specific future 5m reversal interval.
2. Every completed 1m candle inside that interval is scored for reversal-direction hotness.
3. Hotness uses only information available at that candle close: displacement, body efficiency, close location, range/body expansion, volume, local structure break, local returns, and the hierarchy state.
4. The first candle whose inferred probability clears the validation threshold becomes the setup candle. Later hotter candles in the same 5m interval cannot replace it retrospectively.
5. A limit order is placed at the midpoint of the hot candle body after it closes.
6. The order expires after 10 one-minute bars. A pending order blocks later orders, even if it never fills.
7. If filled, the stop is beyond the hot candle extreme plus 0.50 ATR. The maximum hold is 240 minutes.
8. Fill-bar and later stop/target collisions are resolved stop-first. Targets are not credited on the fill bar because one-minute OHLC does not reveal whether the target occurred before the retest.
9. Labels are purged using order expiry for unfilled orders and trade exit for filled orders.

The cascade fields are training-era percentiles, not calibrated probabilities.
For example, `q70` means every hierarchy layer clears its own 70th-percentile
training score. The monthly validator selects the hierarchy percentile and the
model-score coverage using only the preceding validation window.

Strict execution geometry was too sparse:

- With `q80`, a 0.02 ATR stop buffer, 8 bps round-trip cost, and maximum friction of 0.25R, only 64 valid order candidates survived the full history.
- No validation fold had enough resolved fills to support a strategy conclusion.

The discovery pass therefore used `q70` as the minimum hierarchy gate, a 0.50
ATR buffer, and allowed up to 0.50R modeled friction. Those are research
settings, not production assumptions.

The best current pocket is a reversal-direction candle, body-midpoint retest,
logistic hotness model, and fixed 5R target:

| Variant | Trades | Net R | PF | Max DD | Target hits | Positive months |
|---|---:|---:|---:|---:|---:|---:|
| Hierarchy + hotness, 6m validator | 38 | +4.49R | 1.13 | -19.65R | 8 | 3 |
| Hierarchy + hotness, 12m validator | 17 | +1.68R | 1.10 | -11.86R | 4 | 2 |
| ExtraTrees, 12m validator | 11 | -8.89R | 0.34 | -10.85R | 1 | 1 |

The feature ablation is the most useful finding:

| 5R model, 6m validator | Trades | Net R | PF |
|---|---:|---:|---:|
| Hierarchy + hot-candle features | 38 | +4.49R | 1.13 |
| Hierarchy only | 21 | -8.98R | 0.59 |
| Hot-candle features only | 29 | -13.55R | 0.56 |

Interpretation:

- The corrected architecture behaves as a layered model: neither timing nor candle hotness is sufficient alone, while their interaction creates a small positive pocket.
- This is much closer to the intended strategy than the immediate-entry Wyckoff tests.
- The result is still provisional. Model-family sensitivity, high friction in R, sparse target hits, and drawdown much larger than net profit prevent live use.
- The next iteration should improve the candle-hotness measurement with 30s/tick sequencing, aggressive trade delta, order-book imbalance, liquidations, open-interest change, and cross-venue confirmation. The hierarchy/retest architecture itself should remain fixed while those features are tested.

## Completed-Candle Hierarchy Confirmation

The preceding hot-candle experiment interpreted the hierarchy as a prediction
of one future 5m interval. That was not the intended architecture. The corrected
test evaluates every hierarchy level independently:

1. Before the child candle opens, use the existing hierarchy score as the prior.
2. After the child candle closes, estimate whether it contained the corresponding parent high or low.
3. If confirmed, identify a reversal-direction lower-timeframe displacement candle formed inside the completed child.
4. Place a causal limit retest only after both the hierarchy candle and setup candle have closed.

The four posterior heads are:

| Completed child | Parent extreme | Entry-zone timeframe |
|---|---|---|
| 4h | 1d high/low | 15m |
| 1h | 4h high/low | 5m |
| 15m | 1h high/low | 1m |
| 5m | 15m high/low | 1m |

Models train on 2023, gates are selected on 2024, and 2025-01-01 through
2026-05-31 remains locked test data.

### Posterior discrimination

The full-sample test metrics initially looked almost perfect at the lower
levels because the final child closes simultaneously with its parent. At that
point the parent extreme is known, not predicted. The honest comparison
therefore reports both all children and only children whose parent remains
open.

| Layer/direction | Prior AP, all | Price posterior AP, all | Prior AP, parent open | Price posterior AP, parent open |
|---|---:|---:|---:|---:|
| 4h to 1d low | 0.405 | 0.746 | 0.345 | 0.619 |
| 4h to 1d high | 0.392 | 0.742 | 0.311 | 0.624 |
| 1h to 4h low | 0.511 | 0.844 | 0.401 | 0.699 |
| 1h to 4h high | 0.505 | 0.824 | 0.394 | 0.657 |
| 15m to 1h low | 0.521 | 0.838 | 0.393 | 0.686 |
| 15m to 1h high | 0.514 | 0.832 | 0.401 | 0.677 |
| 5m to 15m low | 0.584 | 0.870 | 0.467 | 0.724 |
| 5m to 15m high | 0.582 | 0.869 | 0.470 | 0.723 |

This is the strongest classification result in the project, including when the
parent is still open.

### Feature ablation

The useful posterior update comes from completed price structure:

- `prior scores` alone approximately reproduce the original hierarchy.
- `completed time/calendar + prior` is flat or worse OOS.
- `completed price + prior` is best at every layer.
- Adding completed time/calendar features to completed price usually reduces AP.

The operational model therefore uses the calendar/cycle hierarchy only as the
pre-candle prior, then updates it with completed-candle price features:
range/body efficiency, wick structure, close location, volatility and volume,
returns, trend context, whether a new running parent extreme was made, distance
from the running parent high/low, and the causal child slot.

### Retest execution

The posterior classification edge does not automatically become a high-RR
strategy. With 8 bps round-trip cost, stop-first OHLC handling, pending-order
blocking, and purged labels:

- Most layer/direction/RR combinations either fail validation or fail the locked test.
- A body-midpoint 15m-to-1h short result looked strong on test, but its
  parent-open validation subset was negative. It is not retained.
- Body-open and range-mid retests fail.
- The only repeatable pocket is parent-closed confirmation on the short side,
  using the proximal edge of a bearish 1m displacement candle.

At the 1h close, whether the final 15m child contained the hourly high is
causally known. Replacing the ML posterior with that exact fact produces the
simpler candidate:

| Variant | Validation trades | Validation net/PF | Test trades | Test net/PF | Max test DD |
|---|---:|---:|---:|---:|---:|
| Exact hourly-high confirmation, short, 2R | 17 | +7.12R / 1.75 | 13 | +0.74R / 1.08 | -7.65R |
| Exact hourly-high confirmation, short, 5R | 14 | +15.35R / 2.44 | 13 | +12.44R / 2.16 | -5.21R |

The 5R rule is:

1. At an hourly close, require the final 15m child to contain the completed hour high.
2. Require the pre-candle high hierarchy prior to clear the validation-selected gate.
3. Inside that 15m child, find a sufficiently hot bearish 1m displacement candle.
4. After it closes, sell a retest of its proximal body edge.
5. Stop beyond its high plus 0.25 ATR, subject to the 0.50R friction cap.
6. Target 5R; expire the pending order after 60 minutes; maximum hold is 24 hours.

This is not yet a finished strategy. It has only 13 locked-test trades, all
test profit is concentrated in 2025, 2026 is slightly negative, the long side
does not validate, and 10R does not pass validation. The correct conclusion is
that completed-candle price confirmation is validated as a hierarchy layer,
while the current retest rule is only a narrow research candidate.

## Baseline 5m Gate

Simple entry model:

- Gate: active 15m windows selected from the validation threshold.
- LTF trigger: 5m candle sweeps the previous 12-bar high/low and closes back inside.
- Direction: high sweep = short, low sweep = long.
- Entry: next 5m open.
- Stop: sweep extreme plus/minus 0.05 ATR.
- Targets: 10R, 20R, 30R.
- Max hold: 288 five-minute bars, about 24 hours.
- Cost assumption: 8 bps round trip.

Out-of-sample 2025-01-01 to 2026-06-01:

| Gate | RR | Trades | Win rate | Avg R | Net R | PF | Max DD |
|---|---:|---:|---:|---:|---:|---:|---:|
| ungated | 10 | 4,724 | 10.9% | -0.925 | -4,371R | 0.48 | -4,372R |
| calendar gate | 10 | 180 | 11.1% | -0.948 | -171R | 0.46 | -174R |
| placebo gate | 10 | 356 | 14.0% | -0.784 | -279R | 0.55 | -304R |
| all_real gate | 10 | 228 | 17.1% | +0.068 | +15.6R | 1.05 | -48R |
| all_real gate | 20 | 215 | 12.6% | -0.053 | -11.3R | 0.96 | -62R |
| all_real gate | 30 | 210 | 12.9% | +0.024 | +5.1R | 1.02 | -63R |

This showed that timing helps trade selection, but the threshold gate alone is too weak.

## Meta-Selector Development

I then treated every 5m sweep as a candidate trade and trained a trade-level selector.

Candidate features:

- LTF sweep quality: sweep depth, rejection fraction, reclaim distance, body direction, range/ATR, close from extreme.
- LTF context: RSI, EMA slope, ATR ratio, volume ratio, EMA distance.
- Calendar context: time of day and day of week.
- Timing heads: separate `any`, `high`, and `low` scores from calendar-only, real astrology/cycle, and shifted-placebo models.

Models tested:

- Logistic regression.
- Histogram gradient boosting.
- ExtraTrees with shallow depth and large leaf size.

Static selector, 12-bar sweep, 10R target, 24h max hold:

| Selector | Test trades | Net R | PF | Max DD | Notes |
|---|---:|---:|---:|---:|---|
| Raw 12-bar sweeps | 4,713 | -4,327R | 0.48 | -4,327R | unusable |
| ExtraTrees `price_calendar` | 157 | +49.2R | 1.32 | -34.2R | best static result, no astro required |
| ExtraTrees `price_real` | 79 | -14.6R | 0.80 | -29.9R | static real timing did not generalize |
| ExtraTrees `price_placebo` | 82 | -4.1R | 0.94 | -23.5R | placebo not enough |

Static training selected a useful price/calendar filter, but the real astro/cycle score only became useful when handled adaptively.

## Walk-Forward Candidate

Walk-forward process:

1. For each test month, train the trade selector on older history.
2. Use only the immediately prior 12 months to choose the probability/coverage threshold.
3. Trade the next month.
4. Repeat monthly.

12-bar sweep, 10R target, 24h max hold:

| Feature set | Validation window | Test months | Net R | Trades | Positive months |
|---|---:|---:|---:|---:|---:|
| `price_real` | 12 months | 17 | +40.9R | 155 | 11 |
| `price_calendar` | 3 months | 17 | +9.3R | 207 | 9 |
| `price_calendar` | 12 months | 17 | +5.0R | 153 | 9 |
| `price_placebo` | 12 months | 17 | +0.6R | 152 | 8 |
| `price_real_placebo` | 12 months | 17 | -15.7R | 141 | 8 |

Best current candidate:

| Variant | Trades | Win rate | Avg R | Net R | PF | Max DD | Targets | Stops | Timeouts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Walk-forward `price_real` | 155 | 27.1% | +0.264R | +40.9R | 1.29 | -36.4R | 9 | 111 | 35 |
| Same + pause after -6R monthly loss | 144 | 26.4% | +0.306R | +44.1R | 1.33 | -33.3R | 9 | 104 | 31 |

What failed:

- 20R was weak. Most positive outcomes were timeouts, not actual 20R target hits.
- 30R is not supported by the evidence from the earlier pass and the 20R degradation.
- 6-bar sweeps generated too many low-quality signals and were strongly negative.
- 24-bar sweeps overfit validation and failed the untouched test.
- 6-hour max hold killed the 10R edge; the 24-hour max hold is currently necessary.

## Legacy Strategy Candidate

The earlier walk-forward 10R candidate used:

1. Each month, train the trade selector on older data.
2. Use the prior 12 months only to choose the probability/coverage threshold.
3. Candidate setup: 5m sweep of the previous 12 bars that closes back inside the swept level.
4. Meta features: LTF sweep-quality features plus calendar and real astrology/cycle timing-head scores.
5. Model: ExtraTrees classifier with shallow depth and large leaf size.
6. Direction: high sweep = short, low sweep = long.
7. Entry: next 5m open.
8. Stop: sweep extreme plus/minus 0.05 ATR.
9. Target: 10R.
10. Max hold: 288 five-minute bars, about 24 hours.
11. Skip overlapping trades.
12. Pause new trades for the rest of the month after -6R realized monthly strategy loss.

That result is now considered provisional. The later audit found that the older walk-forward framework did not purge unresolved path labels at every fold boundary and did not carry all execution state across monthly boundaries. It should not be treated as the current validated strategy until rebuilt on the corrected prequential framework.

## Why This Is Still Not Proven

The positive trade result could still come from:

- A real planetary/cycle effect.
- A regime-specific interaction between calendar/time features and BTC market structure.
- The selector finding favorable volatility/liquidity clusters.
- Overfitting to the 2024 validation period and 2025-2026 design cycle.
- The 5m sweep model benefiting from reduced trade frequency rather than accurate HTF reversal timing.

The direct reversal-window evidence argues against strong astrology-only predictability. The walk-forward execution evidence says real astro/cycle timing scores may add value when used as one feature group inside a trade selector.

## Next Validation Steps

Before considering live use:

1. Treat cascade windows as alerts and collect 1m or 30s execution data around them.
2. Test market-structure shift, FVG retest, volume impulse, and order-flow entries inside the predicted window.
3. Refit the hierarchical heads prequentially at every monthly or quarterly boundary instead of using yearly folds.
4. Run placebo ensembles: shifted calendars, randomized cycle periods, and equal-frequency random event calendars.
5. Test ETH and SOL as validation assets, not discovery assets.
6. Add spread, latency, partial fills, funding, and exact exchange maker/taker fee assumptions.
7. Test whether entering before the predicted 5m candle closes can retain more excursion without unacceptable false starts.
8. Test alternate parent-extreme definitions that require post-extreme displacement, not merely being the absolute candle high or low.
9. Add event-level trade/order-book data, open interest, liquidations, perp basis, and cross-venue lead/lag before expanding the OHLCV feature set again.

## Artifacts

- Timing validation script: `scripts/research_btc_astro_cycle_timing.py`
- Meta selector script: `scripts/research_btc_astro_meta_strategy.py`
- Walk-forward script: `scripts/research_btc_astro_walkforward.py`
- LTF probability script: `scripts/research_btc_ltf_calendar_probability.py`
- LTF calendar-bin script: `scripts/research_btc_ltf_calendar_bins.py`
- Hierarchical parent-extreme script: `scripts/research_btc_hierarchical_reversal.py`
- Hierarchical cascade backtest: `scripts/research_btc_hierarchical_cascade_backtest.py`
- Hierarchical path walk-forward: `scripts/research_btc_hierarchical_path_walkforward.py`
- Hierarchical 1m Wyckoff execution: `scripts/research_btc_hierarchical_wyckoff_1m.py`
- Hierarchical 1m excursion/runner research: `scripts/research_btc_hierarchical_excursion_runner.py`
- Hierarchy-gated 1m hot-candle retest: `scripts/research_btc_hierarchical_hot_retest_1m.py`
- Completed-candle hierarchy confirmation: `scripts/research_btc_hierarchical_completed_confirmation.py`
- Full timing JSON: `scripts/astro_cycle_timing_results.json`
- Top windows CSV: `scripts/astro_cycle_top_windows.csv`
- LTF probability JSON/CSV:
  - `scripts/ltf_calendar_probability_logit_full.json`
  - `scripts/ltf_calendar_probability_logit_full.csv`
  - `scripts/ltf_calendar_probability_civil_hgb.json`
  - `scripts/ltf_calendar_probability_civil_hgb.csv`
- LTF calendar-bin JSON/CSV:
  - `scripts/ltf_calendar_bin_results.json`
  - `scripts/ltf_calendar_bin_summary.csv`
- Hierarchical model JSON/CSV:
  - `scripts/hierarchical_reversal_hgb.json`
  - `scripts/hierarchical_reversal_hgb.csv`
  - `scripts/hierarchical_reversal_strict_hgb.json`
  - `scripts/hierarchical_reversal_strict_hgb.csv`
  - `scripts/hierarchical_reversal_strict_logit_feature_ablation.json`
  - `scripts/hierarchical_reversal_strict_logit_feature_ablation.csv`
- Hierarchical cascade outputs:
  - `scripts/hierarchical_cascade_corrected.json`
  - `scripts/hierarchical_cascade_corrected_localization.csv`
  - `scripts/hierarchical_cascade_corrected_variants.csv`
  - `scripts/hierarchical_cascade_corrected_trades.csv`
- Hierarchical path outputs:
  - `scripts/hierarchical_path_walkforward_primary_logit.json`
  - `scripts/hierarchical_path_walkforward_rr3_direction_6m.json`
  - `scripts/hierarchical_path_walkforward_dynamic_logit_3m_corrected.json`
  - `scripts/hierarchical_path_walkforward_long_12m_prequential.json`
- 1m Wyckoff outputs:
  - `scripts/hierarchical_wyckoff_1m_lowrr_12m.json`
  - `scripts/hierarchical_wyckoff_1m_highrr_12m.json`
  - `scripts/hierarchical_wyckoff_1m_sos_short_12m.json`
  - `scripts/hierarchical_wyckoff_1m_static_variants.csv`
- 1m excursion/runner outputs:
  - `scripts/hierarchical_excursion_runner_fixed15_features_summary.csv`
  - `scripts/hierarchical_excursion_runner_policy_compare_summary.csv`
  - `scripts/hierarchical_excursion_runner_basefixed15_v6_summary.csv`
  - `scripts/hierarchical_excursion_runner_basefixed15_v3_summary.csv`
  - `scripts/hierarchical_excursion_runner_basefixed15_et_summary.csv`
  - `scripts/hierarchical_excursion_runner_basefixed15_spring_summary.csv`
- Hot-candle retest outputs:
  - `scripts/hierarchical_hot_retest_1m_reversal_model_focused_summary.csv`
  - `scripts/hierarchical_hot_retest_1m_reversal_model_v6_summary.csv`
  - `scripts/hierarchical_hot_retest_1m_reversal_model_et_summary.csv`
  - `scripts/hierarchical_hot_retest_1m_reversal_ablation_summary.csv`
- Completed-candle confirmation outputs:
  - `scripts/hierarchical_completed_confirmation_posteriors_diagnostics.csv`
  - `scripts/hierarchical_completed_confirmation_l15_states_mid_summary.csv`
  - `scripts/hierarchical_completed_confirmation_l15_states_proximal_summary.csv`
  - `scripts/hierarchical_completed_confirmation_l15_closed_proximal_rr_summary.csv`
- Meta result JSONs:
  - `scripts/astro_meta_strategy_results_12_10r.json`
  - `scripts/astro_meta_strategy_results_12_20r.json`
  - `scripts/astro_meta_strategy_results_6_10r.json`
  - `scripts/astro_meta_strategy_results_24_10r.json`
- Walk-forward summary: `scripts/astro_walkforward_price_real_12m_10r_summary.json`
- Walk-forward monthly CSV: `scripts/astro_walkforward_monthly_12_10r.csv`
- Walk-forward trades: `scripts/astro_walkforward_price_real_12m_10r_trades.csv`
