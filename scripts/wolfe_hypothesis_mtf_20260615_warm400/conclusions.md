# Wolfe Multi-Timeframe Hypothesis Conclusions

## Scope

- 67 symbols, 83 exact deployed old/v2 configurations
- 5,083 trades over 730 days
- Context measured on completed 15m, 1h, 4h, and 1d candles
- 400-day warmup for the daily EMA200
- Fees and slippage included
- Each hypothesis tested one at a time
- Adjacent thresholds evaluated by neighborhood-median improvement
- Stability checked across symbols and calendar quarters

## Baseline

| Window | Trades | Win rate | Avg win | Avg loss | Expectancy | Profit factor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 30d | 213 | 51.2% | +2.24R | -1.12R | +0.598R | 2.09 |
| 90d | 577 | 52.2% | +2.26R | -1.13R | +0.640R | 2.19 |
| 365d | 2,491 | 50.5% | +2.23R | -1.12R | +0.570R | 2.03 |
| 730d | 5,083 | 50.6% | +2.26R | -1.11R | +0.596R | 2.09 |

The attached premise is confirmed: the roughly 50% win rate is not a weakness
in the research population. Winners are approximately twice as large as losses.

## Hypotheses

### H1: Avoid strong runaway trends

**Verdict: not supported as a general ADX gate.**

ADX caps of 20, 25, or 30 on 15m, 1h, 4h, and 1d did not improve the combined
population consistently. Avoiding an ADX>=25 opposing trend also failed.

Daily ADX falling is a useful v2 ranking feature: over 730 days it retained 577
v2 trades with +0.821R expectancy versus +0.705R baseline. The effect was also
positive over 365 days, but small over the latest 90 days. Do not hard-gate it.

### H2: Higher-timeframe EMA context

**Verdict: mixed; useful as context, not as a gate.**

The strongest combined long-run rule was daily EMA structure not opposed, but it
was harmful in the latest 90 days. Local 15m EMA alignment improved average
quality but retained only 12% of trades. A 4h reversal stretch was strong for v2
over 730 days but failed in the latest 90 days.

No EMA-context rule is stable enough for production rejection.

### H3: Point-5 exhaustion

**Verdict: absolute point-5 RSI exhaustion is supported; divergence is rejected.**

Point-5 RSI below 35 for longs or above 65 for shorts retained 69% of trades and
raised 730-day expectancy from +0.596R to +0.650R. The effect was particularly
strong for v2: +0.815R versus +0.705R, with support across 90/365/730 days.

RSI divergence between points 3 and 5, MACD histogram divergence, and entry-candle
RSI exhaustion all reduced expectancy. They should not be added as filters.

### H4: Volatility normalization

**Verdict: 1h volatility context is the most reliable timeframe.**

A 1h ATR percentile between 10 and 90 retained 68% of trades and raised
expectancy to +0.645R. It improved all evaluated quarters and remained positive
over 90, 365, and 730 days. A 1d ATR floor above the bottom decile was also
helpful, while 15m volatility filters were not.

The excluded volatility buckets still earned +0.491R per trade, so this is a
ranking/risk feature rather than a hard gate.

### H5: Pattern quality and symmetry

**Verdict: moderate point-5 overshoot matters; generic symmetry does not.**

Point-5 overshoot no greater than 1 ATR retained 62% of trades and raised
expectancy to +0.659R. This held across 90/365/730 days and 66% of evaluable
symbols over the full window.

Symmetry thresholds did not generalize for old Wolfe. A ratio <=1.4 was useful
for v2 only, but the advantage decayed from +0.248R over 90 days to +0.047R over
730 days. Keep symmetry as a v2 score feature, not a universal gate.

### H6: EPA reward/risk

**Verdict: strongest universal quality feature.**

Planned EPA/R >=2.0 retained 79% of trades and raised expectancy to +0.662R.
EPA/R >=2.5 retained 56% and raised expectancy to +0.689R. Both effects were
stable over 90/365/730 days.

The excluded buckets remained profitable (+0.340R below 2R and +0.476R below
2.5R). Therefore a permanent 2R or 2.5R hard gate would reduce total expected R.
The existing conditional rule that raises minimum RR only when live win rate is
below 50% remains the better implementation.

### H7: Volume/liquidity confirmation

**Verdict: point-5 volume is mildly useful; entry volume confirmation is harmful.**

Point-5 volume above its 20-period average retained 78% and raised expectancy
to +0.623R. The effect was stronger and more stable for v2.

Entry volume thresholds, point-5 rejection alone, and volume-plus-rejection
combinations reduced expectancy. Funding and open-interest hypotheses were not
tested because the research cache contains OHLCV only.

## Timeframe Answer

- **15m:** local EMA alignment identifies high-quality trades but is too selective.
- **1h:** most reliable general context; ATR percentile is the best regime feature.
- **4h:** no robust combined gate; reversal stretch may help v2 ranking.
- **1d:** ADX direction is useful for v2 ranking, but absolute ADX and EMA gates are unstable.

## Recommendation

Do not add another hard production gate. All excluded buckets in the supported
tests still had positive expectancy.

Add the following as continuous shadow-score or ML/RL features:

1. Planned EPA/R
2. Point-5 RSI and exhaustion flag
3. Point-5 overshoot in ATR
4. 1h ATR percentile
5. Point-5 volume ratio
6. For v2 only: daily ADX change and symmetry ratio

Validate an equal-weight score with chronological walk-forward replay before it
affects sizing. This preserves the broad edge while testing whether higher-ranked
setups deserve more risk.
