# Astro BTC Reversal-Window Research

A reproducible research framework testing whether planetary aspects identify
**probabilistic BTC reversal windows** (not price predictions). Implements
Milestones 1 and 2 of the project proposal.

> The goal is honest validation. A result is only "interesting" if it beats
> random-calendar and shifted-calendar baselines **and** holds out-of-sample.
> A negative conclusion is a successful outcome if it is statistically honest.

## ⭐ The result: the "Capitulation Spring" (~1.5 high-RR trades/week, +EV every year)

After exhausting astro (all negative), the one edge that holds out-of-sample is a pure
price-structure + positioning setup, run across a **basket of ~18 liquid Bybit perps**
for breadth, on **5m** with a **scaled exit** (`run_capitulation_strategy.py`):

> **Deep-sweep 5m Wyckoff spring** (sweep of a ~15-day low + reclaim with rejection),
> in the **early week** (Mon-Wed), **filtered to unusually negative funding** (`funding_z
> ≤ −1` = shorts crowded after a flush → contrarian-bullish capitulation). Tiny stop
> below the spring; **scaled exit 25% @ 4R / 50% @ 12R / 25% @ 30R**, stop → breakeven
> after the first partial. **Long-only** (shorts tested, rejected).

| Version | Trades/wk | Win % | Avg R (net) | PF | MaxDD R | Dev | Holdout | Years + |
|---|---|---|---|---|---|---|---|---|
| 15m, fixed 30R | 1.05 | 16.5 | +0.53 | 1.57 | −63 | −0.29 | +1.24 | 3/5 |
| 15m, scaled | 1.06 | 29.6 | +0.59 | 1.76 | −31 | +0.26 | +0.88 | 4/5 |
| **5m, scaled** ⭐ | **1.46** | 26.7 | **+0.59** | 1.68 | **−26** | **+0.50** | +0.68 | **5/5** |

5m, long_scaled is **positive in all five years** (2022 bear +0.39, 2023 +0.32, 2024
+0.77, 2025 +0.57, 2026 +1.28). The 5m tighter stop adds frequency and quadruples
reach-20R (~3%→~13%); the scaled exit adds cross-year robustness and halves drawdown.

Why I trust it: **monotonic dose-response** in the funding threshold (−0.5→+0.28,
−1.0→+0.48, −1.5→+0.72R on 15m); a **falsification that passes** (funding_z ≥ +1, longs
crowded, *loses*: −0.37R — directional contrarian signal, not a noise filter); 11/18
symbols positive; positive on dev AND holdout AND every calendar year. Free data only
(Bybit klines + funding history). Reproduce: `run_capitulation_strategy.py --interval 5m
--spring-lookback 4320 --max-hold 4032 --fz 1.0 --rr 30` (the `long_scaled` book).

### Scaled exit makes it far more robust (`run_capitulation_strategy.py`)

Taking partials instead of a single 30R target — **25% @ 4R, 50% @ 12R, 25% @ 30R,
stop → breakeven after the first partial** — is a large improvement for the longs:

| Long exit | Win % | Avg R | MaxDD R | Dev avg R | Holdout avg R | Years positive |
|---|---|---|---|---|---|---|
| fixed 30R | 16.5 | +0.53 | −63 | −0.29 | +1.24 | 3/5 |
| **scaled 4/12/30** | **29.6** | **+0.59** | **−31** | **+0.26** | +0.88 | **4/5** |

The scaled exit nearly doubles the win rate, halves the drawdown, keeps avg R slightly
higher, and turns the dev period and 4/5 years positive (even the 2022 deep-bear is only
−0.19R vs −1.10R). This is the version to trade.

**Shorts were tested and rejected.** The mirror setup (deep-sweep *upthrust* + crowded-long
funding `z ≥ +1` → short) is **negative** both fixed (−0.19R, holdout −0.06) and scaled
(−0.39R, holdout −0.58). Crypto's upward drift fights reversal-shorts and tops are rounded
(no sharp V for a tight-stop short). **The edge is long-only.**

### Fees and drawdown — measured and addressed (`run_strategy_improve.py`)

Two fair objections — tested directly:

**Fees don't eat the edge.** The 5m spring stop is **0.76% of price (median)**, not the
~0.2% one might fear, so fees are a modest fraction of risk: gross +0.77R → **net +0.59R**
after fees (~0.18R drag, ~23% of gross). Modelling realistic execution (taker entry/stop,
**maker limit TPs**) barely moves it (+0.597 vs +0.588 all-taker) because the taker *entry*
dominates. Widening the stop does **not** help (avg R falls, MaxDD flat) — keep it tight.

**The −26R drawdown is a *cumulative-R* number, not account %.** A fixed-fractional
**equity simulator with a concurrency cap** (`portfolio.py`, `run_risk_model.py`) gives
the real account metrics over 2022-2026, net of realistic fees:

| Risk/trade · cap | CAGR | MaxDD % | **MAR** | Total |
|---|---|---|---|---|
| **0.5% · 4** | **+22.8%** | **−12.4%** | **1.84** | +142% |
| 0.75% · 4 | +33.9% | −18.1% | 1.87 | +252% |
| 1.0% · 4 | +44.7% | −23.5% | 1.90 | +391% |

So the headline drawdown is **−12.4% of account** at 0.5%/trade with a 4-position cap —
not 26%. MAR ≈ 1.85 (CAGR ≈ 1.85× the max drawdown) is a strong, tradeable profile; the
user picks the operating point along the line.

**BTC dominance (BTC.D) regime gate — tested, mostly doesn't help.** Using Binance
BTCDOMUSDT (`regime.py`), gating on BTC freefall *lowers* returns (capitulation bottoms
happen *during* freefalls), and a strong BTC.D-up gate cuts too many good alt-bounces. A
*light* "skip when BTC.D rose >8%/14d" gate gives only a marginal MAR bump (1.93 vs 1.84).
The funding-flush filter already captures the regime that matters; an explicit BTC.D gate
is optional at best.

**Remaining caveats:** still fat-tailed (~27% win), somewhat symbol-concentrated (11/18+),
5m fills assume next-bar open (slippage on the spring is real — size for it), 2026 partial.
The 5m+scaled version is positive every year incl. 2022 bear, so the bear-regime worry is
largely resolved; a regime off-switch is still prudent. Lighter variants: 15m
(`run_capitulation_strategy.py --fz 1.0 --rr 30`) or fixed-exit (`run_basket_capitulation.py`).

## Can we build a high-RR strategy from this? (Milestone 4)

`run_reversal_strategy.py` builds the honest version of the request: HTF daily dump
(the "bottom zone") opens a long-alert window; on the LTF (1h) we wait for a
sweep+reclaim of a significant prior low with a displacement candle (the "reaction"),
enter with a tight stop below the swept low, and target a high RR. It reports the full
R-distribution, fixed-RR P&L net of costs, gated-vs-ungated, per-year + holdout.

That 1h/2-5R version finds **no edge** (net-negative every year; the dump gate makes it
worse). But that frontier was wrong: 2-5R needs a ~33% win rate. The real approach is
**1m/5m precision** — a tiny stop and a *weekly-swing* target, i.e. 1:20-1:30 RR, which
only needs a ~5-8% win rate.

`run_ltf_structure.py` tests that: HTF weekly pivots (daily zigzag) + **5m Wyckoff
springs** (sweep of a significant low + reclaim with rejection), tight stop below the
spring, measured to 10/20/30R. Findings (2021-2026, costs incl.):

- Springs in general have **no edge** (avg R < 0 at every RR) — same as the 1h test.
- Springs **at a weekly low** run to 20R far more often (9.7% vs 5.0%) and are strongly
  positive (+1.5R avg @20R, positive in 4/5 years) — but "weekly low" there is hindsight.
- The **tradeable, real-time version**: require the spring to sweep a **deep
  (15-30 day) low** — a major liquidity grab that *is* the weekly-low proxy. This is
  net-positive with a clean monotonic dose-response in sweep depth: 30R avg R goes
  −0.20 (4d) → +0.12 (15d) → +0.44 (30d), with holdout (2025+) avg R +0.5-0.6.
  `run_ltf_structure.py --spring-lookback 8640`.

So there **is** a high-RR edge, and it matches the trader: time the weekly low via a
deep liquidity sweep, confirm with a 5m Wyckoff spring, tiny stop, 20-30R target. It is
fat-tailed (~8% win rate, PF 1.2-1.4, modest sample), so size small and expect long
droughts. A generic ML model over structural features (`run_spring_model.py`) did NOT
beat random at finding these springs — **sweep depth itself is the one feature that
carries the signal.** The astro/Dark-Pivot timing remains useless and is not part of this.

### Weekly timing cycle + the combined edge

`run_weekly_timing.py` asks *when* (at 15m resolution) the weekly low/high print and
whether they get retested. On 2021-2026 (282 weeks, Mon 00:00 UTC weeks):

- The **weekly LOW clusters on Monday: ~36% of weeks** (vs 14.3% uniform), modal cell
  Monday Asia session; Friday is secondary. Robust in **every year** (31-43%). The
  weekly high is also Monday-heavy (~29%). Both extremes form in the first ~40% of the
  week (median ~33-37%). Low precedes high in exactly 50% of weeks.
- The weekly low is **retested** later the same week in ~30% of weeks (median ~16h
  after); the high in ~34% (~21h after).

**Combining the cycle with the spring** is the strongest result: gating the deep-sweep
springs to the early-week window stacks cleanly (two a-priori, independent filters):

| 30-day sweep, target 30R | Trades | Win % | Avg R | PF | Holdout avg R |
|---|---|---|---|---|---|
| any time | 146 | 8.2% | +0.44 | 1.40 | +0.46 |
| first 40% of week | 57 | 10.5% | +0.74 | 1.69 | +2.20 |
| Mon + Fri | 67 | 10.4% | +0.81 | 1.74 | +2.58 |

Reproduce: `run_ltf_structure.py --spring-lookback 8640 --week-frac-max 0.4`
(or `--dows 0,4`). Sample is small (size accordingly), but the gain is consistent
across sweep depths, RR targets, and the 2025+ holdout.

### Retest entry — tested, it underperforms (`run_retest_strategy.py`)

The intuitive follow-up — wait for the mid-week **retest** of the established weekly
low instead of taking the first spring — is **worse**, robustly. Real-time logic
(track the running weekly low, require a rally away to 'establish' it, then enter on a
5m spring retesting that level): win rate collapses to ~3% and avg R is negative at
20R/30R, **negative on the holdout under every parameter combination** tried. The
deep-sweep *first touch* (a V-bottom: sweep + immediate reclaim → run) is the edge;
waiting for a retest filters those out and keeps the weaker lows that break. On BTC
2021-2026, weekly-low retests break more often than they hold at tight-stop precision.

### Fibonacci in time + price (no edge)

`run_fib_time.py` / `run_fib_confluence.py` test whether a pivot/spring at a Fibonacci
*time* projection (swing-ratio level) and/or *price* level (retracement/extension of the
prior swing) outperforms — with the decisive control being a **matched non-Fibonacci
placebo** (ordinary ratios). Result: **no Fibonacci-specific effect**. Swing-ratio
Fib-time lift (~1.12) is *equal to* the non-Fib placebo (~1.14, fib-vs-nonfib p≈0.80);
on the 5m spring, `fib_both` (time AND price) reach-20R = `nonfib_both` (0.40 vs 0.40).
The apparent `fib_zone` lift is an artifact of small offsets (1,2,3,5,8) hugging the
anchor pivot.

Re-tested on the trader's terms (absolute profitability, scaled exit net of costs, no
placebo as the criterion — just "is the double-Fib subset profitable and better than
trading every spring?"): on BTC alone fib_both looked good in-sample (+0.52R) but its
holdout was breakeven on 15 trades. Across the 18-symbol basket (4,745 springs, 299
fib_both holdout trades) fib_both is **breakeven, not profitable** — avg R −0.03,
holdout +0.04. It is the best of the Fib books and modestly better than unfiltered
springs (−0.16), so the confluence is a mild quality nudge, but it does not reach
profitability on its own, and the BTC positive was small-sample luck. Conclusion holds
whether you use the placebo control or pure profitability: Fib time/price/confluence is
not a tradeable edge; the reversal edge stays the deep-sweep spring + funding flush.

### Do prior weeks predict this week's window? (no)

`run_weekly_sequence.py` tests whether conditioning on the previous 1-3 weeks'
extreme timing (and this week's open vs prior levels) sharpens the distribution of
this week's low. It does **not**: transition chi-square p = 0.7-0.8 (no persistence),
within-week-timing autocorrelation sits inside the noise band at all lags, inter-low
spacing is just the trivial ~7d Monday-to-Monday peak, and a walk-forward classifier
scores **OOS ROC-AUC 0.42-0.47 (below 0.5)** with log-loss worse than baseline.
Weekly extreme timing is effectively memoryless week-to-week — the only structure is
the unconditional Monday/early-week prior (already exploited above).

## The core question (precision, not classification)

The most useful framing is **not** "classify every candle as pivot/not-pivot"
(that optimizes recall over all pivots and is dominated by ordinary price/volatility
structure). It is:

> Is there a sparse astro **calendar** whose firings reliably contain a pivot within
> a tolerance window — at a hit rate well above random windows of the same width and
> count? Missing most pivots is fine; **high hit rate + lift** is the goal.

`run_calendar_search.py` answers this directly via an event study on each calendar's
firings. The random-window baseline automatically accounts for how frequent pivots
are, so it also sidesteps the pivot-definition sensitivity. Treat this as the primary
analysis; the M3 classifier is kept for completeness but is the weaker framing.

`run_conditional_calendar.py` pushes the strongest remaining idea (proposal H5 / the
Dark Pivot thesis): fire only when price **dumped into** the aspect (expect a bottom)
or **pumped into** it (expect a top), with the baseline drawn from the *same* price
context so lift isolates the astro contribution beyond the move itself.

**Finding so far (2021–2026 Bybit, 1D/4H/1H):** across unconditional and direction-
conditional searches, **zero** calendars survive Benjamini–Hochberg FDR, and the
Moon–Pluto Dark Pivot lift stays ~1.0–1.25 (never significant, fails holdout). The
high-hit-rate calendars are all 5–8-firing small-sample artifacts. No astro calendar
reliably predicts BTC pivot windows above the random-window baseline.

**Replicating the public "Dark Pivot" 77% claim** (`run_dark_pivot_replica.py`):
the Dark Pivot calendar *is* Moon–Pluto hard aspects (the advertised next date,
2026-06-24, matches the engine to the minute). Reproducing their loose rule ("dumped
into the activation day, then a bullish expansion within ~a week → local bottom")
gives ~70–77% — but that is the **base rate**: the unconditional probability of a
higher high within 7 days of *any* day is 0.774. Dark-Pivot dump days score at or
slightly *below* that, and the same as ordinary dump days (lift ≈ 1.0). The literal
"local bottom" and "50% top" readings also show lift ≈ 1.0. The headline % is a loose
success rule meeting a high base rate, not an edge.

## What it does

- **Milestone 1 — Dark Pivot candidate test** (`run_dark_pivot.py`): builds the exact
  Moon–Pluto hard-aspect calendar and asks whether BTC dumps *into/on* those days are
  followed by bullish expansion more often than on ordinary dump days. Reports hit
  rate, lift, binomial test, bootstrap CIs, random-calendar p-value, shifted-calendar
  comparison, MFE/MAE/max-R, and an out-of-sample holdout check.
- **Milestone 2 — Aspect library discovery** (`run_aspect_discovery.py`): scores
  pivot-window **lift** for every body-pair × aspect-angle on 1H/4H/1D, with
  Benjamini–Hochberg FDR control, random-subset nulls, and a 2025+ holdout column.
- **Milestone 3 — ML pivot-window model** (`run_ml_pivot.py`): predicts
  bottom/top/pivot within N candles across feature-set ablations (price-only,
  calendar-only, lunar-only, astro-only, astro+cycle, astro+price, full) and models
  (logistic, HistGradientBoosting), under expanding walk-forward validation with
  embargo, a final holdout, and a shifted-placebo astro control. Reports PR-AUC,
  ROC-AUC, Brier, precision@K and lift@K.

## Design: reuse, don't reinvent

The ephemeris, Bybit data loading, ATR indicators, and ATR-based zigzag pivots already
exist in [`scripts/research_btc_astro_cycle_timing.py`](../scripts/research_btc_astro_cycle_timing.py).
[`src/astro_reversal/reuse.py`](src/astro_reversal/reuse.py) re-exports them. New code:
exact aspect-event root-finding, dump/expansion labels, random/shifted baselines,
binomial/bootstrap/FDR stats, and the discovery harness.

Data is loaded from the shared 15m Bybit cache (2021–2026) and resampled up to the
target timeframe — fully offline and deterministic. The Skyfield DE421 kernel and
computed positions are cached under `scripts/.cache/astro_cycle/`.

## Usage

```bash
# Milestone 1 (defaults read from configs/aspects.yaml)
python astro_btc_reversal_research/run_dark_pivot.py --timeframe 1d

# Milestone 2 (run per timeframe)
python astro_btc_reversal_research/run_aspect_discovery.py --timeframe 1d
python astro_btc_reversal_research/run_aspect_discovery.py --timeframe 4h
python astro_btc_reversal_research/run_aspect_discovery.py --timeframe 1h

# Milestone 3 (target = bottom | top | pivot)
python astro_btc_reversal_research/run_ml_pivot.py --timeframe 1d --target pivot

# Calendar search (precision framing - the primary question)
python astro_btc_reversal_research/run_pivot_diagnostic.py          # pick reversal-grade threshold
python astro_btc_reversal_research/run_calendar_search.py --timeframe 1d   # (4h, 1h too)

# Direction-conditional (astro + price context: dump->bottom, pump->top)
python astro_btc_reversal_research/run_conditional_calendar.py --timeframe 1d   # (4h, 1h too)

# Replicate the public "Dark Pivot 77%" claim and show the baseline
python astro_btc_reversal_research/run_dark_pivot_replica.py
```

Outputs land in `reports/` (`*.csv`, `*.json`, `*.md`). Tune bodies, aspects, orbs,
windows, pivot thresholds, holdout, and baseline counts in
[`configs/aspects.yaml`](configs/aspects.yaml).

## Tests

```bash
python -m pytest astro_btc_reversal_research/tests -q
```

Covers the aspect-event engine (anchored to the proposal's known 2026-06-24 Dark Pivot
date), pivot labelling, the statistical helpers, and anti-leakage guards (features use
only past/current data; expansion labels are confirmed forward-looking).

## Anti-leakage rules

- Aspect calendars are deterministic and known in advance → usable as features.
- Pivot + expansion labels use future candles → **labels only**, never features.
- `dump_into_event` uses only past returns.
- The 2025+ holdout is reported, never used to select the "best" aspect.
- Random + shifted controls and BH-FDR guard the large search space.

## Out of scope (later milestones)

High-RR execution analysis (M4); a Binance fetcher; pyswisseph extended bodies
(Lilith/Chiron/Nodes).

## Layout

```
configs/aspects.yaml          # bodies, aspects, orbs, windows, dark_pivot, baselines
src/astro_reversal/
  reuse.py                    # re-exports of existing scripts/ utilities
  data.py                     # config + OHLCV loading (resample base interval)
  ephemeris_events.py         # exact aspect event calendar (root-finding)
  pivots.py                   # ATR directional-change + fractal pivot labels
  event_labels.py             # dump/pump-into-event + expansion (MFE/MAE/max-R)
  baselines.py                # random + shifted calendars
  stats.py                    # binomial, bootstrap, BH-FDR, empirical p
  discovery.py                # M2 cross-pair lift discovery
  calendar_search.py          # PRIMARY: event-study calendar precision + conditional search
  dark_pivot_replica.py       # faithful replication of the public 'Dark Pivot' claim
  strategy.py                 # M4 dump gate + LTF sweep/reclaim long backtest engine
  ltf_structure.py            # weekly pivots + 5m Wyckoff springs (the EDGE lives here)
  spring_model.py             # real-time features + reach-20R labels for the spring model
  weekly_timing.py            # weekly low/high time-of-week + intra-week retest
  weekly_sequence.py          # prior-weeks conditioning / cycle test (memoryless)
  retest_strategy.py          # real-time weekly-low retest entry (tested: underperforms)
  basket.py                   # 18-symbol liquid-perp basket loader
  capitulation.py             # free Bybit funding-rate + open-interest features
  exits.py                    # scaled-exit simulator (partial TPs, stop->BE, maker/taker)
  regime.py                   # Binance BTC.D (BTCDOMUSDT) + BTC-freefall regime features
  portfolio.py                # fixed-fractional equity sim + concurrency cap (CAGR/DD/MAR)
  features_ml.py              # M3 price features + ablation feature-sets
  labels_ml.py                # M3 forward bottom/top/pivot-within-N targets
  walk_forward.py             # M3 expanding folds + embargo + holdout split
  models_ml.py                # M3 logistic (L1/L2/EN) + HistGradientBoosting
  evaluate_ml.py              # M3 PR-AUC / ROC / Brier / precision@K / lift@K
  report.py                   # markdown / json / csv writers
run_dark_pivot.py             # M1 CLI
run_aspect_discovery.py       # M2 CLI
run_ml_pivot.py               # M3 CLI
run_calendar_search.py        # calendar precision search CLI (primary)
run_conditional_calendar.py   # astro + price-context (dump->bottom / pump->top) search
run_dark_pivot_replica.py     # replicate public "Dark Pivot 77%" claim vs base rate
run_reversal_strategy.py      # M4 high-RR dump->sweep long backtest (costs, holdout)
run_ltf_structure.py          # 5m Wyckoff springs at weekly lows (+ --week-frac-max gate)
run_spring_model.py           # walk-forward model to find weekly-low springs in real time
run_weekly_timing.py          # when weekly low/high print within the week + retest stats
run_weekly_sequence.py        # do prior weeks predict this week's window? (memoryless: no)
run_retest_strategy.py        # weekly-low retest entry vs first-deep-spring (retest loses)
fetch_basket.py               # cache 15m klines for the 18-perp basket
fetch_capitulation.py         # cache Bybit funding + OI history for the basket
run_basket_spring.py          # multi-symbol deep-sweep-spring (breadth -> ~4 setups/wk)
run_basket_capitulation.py    # + funding-flush filter -> ~1 hi-conv trade/wk (fixed exit)
run_capitulation_strategy.py  # *** longs+shorts x fixed/scaled exit (long_scaled = best)
run_strategy_improve.py       # fee decomposition + stop sweep + portfolio concurrency cap
run_risk_model.py             # *** account-level risk model: CAGR/MaxDD%/MAR + BTC.D gate
run_pivot_diagnostic.py       # pivot count/cadence/base-rate sweep
tests/                        # aspect events, pivots, stats, walk-forward, calendar, no-leakage
reports/                      # generated artifacts
```
