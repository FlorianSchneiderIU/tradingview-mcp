# Astro BTC Reversal-Window Research

A reproducible research framework testing whether planetary aspects identify
**probabilistic BTC reversal windows** (not price predictions). Implements
Milestones 1 and 2 of the project proposal.

> The goal is honest validation. A result is only "interesting" if it beats
> random-calendar and shifted-calendar baselines **and** holds out-of-sample.
> A negative conclusion is a successful outcome if it is statistically honest.

## Can we build a high-RR strategy from this? (Milestone 4)

`run_reversal_strategy.py` builds the honest version of the request: HTF daily dump
(the "bottom zone") opens a long-alert window; on the LTF (1h) we wait for a
sweep+reclaim of a significant prior low with a displacement candle (the "reaction"),
enter with a tight stop below the swept low, and target a high RR. It reports the full
R-distribution, fixed-RR P&L net of costs, gated-vs-ungated, per-year + holdout.

**Result: no tradeable edge.** Net-negative at every RR and every year; the dump gate
makes it *worse* than sweeps alone (knife-catching), and the Moon-Pluto overlay adds
nothing. The reason is structural, not a tuning artifact: after the trigger only ~31%
of setups reach 2R and ~19% reach 3R (avg MFE ≈ avg MAE), so expectancy
≈ 0.31·2 − 0.69 ≈ −0.07 at 2R *before* costs — negative at any target/stop. BTC daily
bottom-catching via dumps + LTF sweeps did not carry positive expectancy in 2021–2026.
The harness (and its R-distribution diagnostic) is reusable to vet any future trigger;
the one component with measured lift remains the M3 **price-only** probability model.

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
run_pivot_diagnostic.py       # pivot count/cadence/base-rate sweep
tests/                        # aspect events, pivots, stats, walk-forward, calendar, no-leakage
reports/                      # generated artifacts
```
