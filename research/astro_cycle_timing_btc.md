# BTC Astrology/Cycle Reversal Timing Research

Date: 2026-06-13

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
| high | 0.80 | 0.694% | 5.233% | 15.07x | 10.47% |
| high | 0.90 | 0.148% | 8.636% | 24.87x | 3.68% |
| high | 0.93 | 0.032% | 12.766% | 36.77x | 1.16% |
| low | 0.80 | 0.997% | 4.723% | 13.60x | 13.57% |
| low | 0.90 | 0.219% | 5.828% | 16.79x | 3.68% |
| low | 0.93 | 0.042% | 3.175% | 9.14x | 0.39% |

At the 0.80 threshold, the high-side precision progressed from 36.87% at the 4h layer to 17.81% for 4h + 1h, 9.27% after adding 15m, and 5.23% for the exact four-layer event. Relative to each stage's shrinking base rate, the lift compounded from 2.21x to 4.27x, 8.90x, and 15.07x. The low side showed the same pattern, ending at 13.60x lift.

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

## Current Strategy Spec

Use this as the current research candidate:

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

The current candidate is a **walk-forward, time-aware liquidity-sweep selector**, not a pure astrology strategy.

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

1. Replace static cascade execution selection with monthly walk-forward training and threshold selection.
2. Train a post-window path model: probability of reaching 1.5R, 2R, 3R, 5R, and 10R before invalidation, conditional on the cascade.
3. Feed the new `P(reversal in next 6/12 candles)` calendar heads into the execution selector and compare them against the parent-extreme cascade.
4. Run placebo ensembles: 20 shifted calendars, randomized cycle periods, and random event calendars with equal frequency.
5. Test ETH and SOL as validation assets, not discovery assets.
6. Replace 5m execution with 1m and then 30s if data is available.
7. Test market-structure shift, FVG retest, volume impulse, and order-flow entries inside the predicted 5m window.
8. Add funding, spread, latency, partial fills, and exact exchange maker/taker fee assumptions.
9. Verify the monthly walk-forward process in paper/live forward mode.
10. Test alternate pivot definitions: fractal pivots, ATR directional-change pivots, liquidity-sweep pivots, and HTF swing-failure pivots.

## Artifacts

- Timing validation script: `scripts/research_btc_astro_cycle_timing.py`
- Meta selector script: `scripts/research_btc_astro_meta_strategy.py`
- Walk-forward script: `scripts/research_btc_astro_walkforward.py`
- LTF probability script: `scripts/research_btc_ltf_calendar_probability.py`
- LTF calendar-bin script: `scripts/research_btc_ltf_calendar_bins.py`
- Hierarchical parent-extreme script: `scripts/research_btc_hierarchical_reversal.py`
- Hierarchical cascade backtest: `scripts/research_btc_hierarchical_cascade_backtest.py`
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
  - `scripts/hierarchical_cascade_feeaware_directional.json`
  - `scripts/hierarchical_cascade_feeaware_directional_localization.csv`
  - `scripts/hierarchical_cascade_feeaware_directional_variants.csv`
  - `scripts/hierarchical_cascade_feeaware_directional_trades.csv`
- Meta result JSONs:
  - `scripts/astro_meta_strategy_results_12_10r.json`
  - `scripts/astro_meta_strategy_results_12_20r.json`
  - `scripts/astro_meta_strategy_results_6_10r.json`
  - `scripts/astro_meta_strategy_results_24_10r.json`
- Walk-forward summary: `scripts/astro_walkforward_price_real_12m_10r_summary.json`
- Walk-forward monthly CSV: `scripts/astro_walkforward_monthly_12_10r.csv`
- Walk-forward trades: `scripts/astro_walkforward_price_real_12m_10r_trades.csv`
