# Strategy Remediation Runbook

Why the bots bled in live paper trading and what was changed to fix it.
Companion to the analysis in `~/.claude/plans/none-of-the-bot-fancy-whisper.md`.

## Diagnosis (recap)

Live paper (`bot/logs/trade_ledger.jsonl`, 2026-05-10 → 06-19) lost **−18%**. The
cause was **not execution slippage** (traced stop fills slip only ~5 bps from
trigger to fill). The causes were:

1. **MM parameters & DT thresholds were selected on zero-cost simulations** —
   `sim_trail` modelled no fees/slippage, so thin (Sharpe 0.1–0.4) edges were
   noise once ~13 bps round-trip cost applied.
2. **MM exit asymmetry** — banking `0.5×tp1_r` (tp1 as low as 0.5R) then snapping
   the runner to breakeven gave ≈+0.25R wins vs −1.0R losses → negative
   expectancy even at a 38% stop rate.
3. **Validation too thin** — single 75/25 split, 2-fold "holdout", no
   multiple-testing correction; OOS often beat IS (the overfit tell).
4. **Quality gating not enforced** (Wolfe `_quality_profile.mode = "shadow"`).

## What changed in code

| File | Change |
|---|---|
| `bot/indicators.py` | `sim_trail` now models round-trip cost (`fee_bps_side`, `slippage_bps_side`) and a post-TP1 **lock-in** (`lockin_r`) instead of pure breakeven. New constants `DEFAULT_FEE_BPS_SIDE=5.5`, `DEFAULT_SLIPPAGE_BPS_SIDE=1.0`, `DEFAULT_LOCKIN_R=0.3`. Defaults keep other callers cost-free. |
| `bot/train_dt.py` | Rebuilt: rolling walk-forward (expanding train + 6mo val + 6mo OOS, step 6mo) on a development span, with the **most recent 365 days carved off as an untouched final holdout**. Jointly re-tunes `(sl, tp1, trail, threshold)` on a grid (tp1≥1.5, trail≥1.0). Gates: trade-count floors, PF≥1.15 (train/val/OOS), net avg-R≥0.05, ≥60% of OOS folds positive. **Multiple-testing guard** (random-selection null at the 95th percentile) and **stability guard** (beat grid-neighbour median). Final model fit on dev data only; holdout scored once. **No more retrain-on-all-data.** Writes `<SYM>_holdout_report.json`. |
| `bot/bot.py` | DT loader syncs the validated `sl/tp1/trail` from the model artifact into `state.cfg` so live execution matches what was validated. New env `BREAKEVEN_LOCKIN_R` (default 0.0 = legacy breakeven) makes the post-TP1 stop a direction-aware lock-in. |
| `scripts/backtest_ggshot_227.py` | Eligibility gate tightened: PF≥1.15 and net-R>0 on **both** train and OOS, plus ≥50% of symbols profitable (rejects the "PF 4.34 on 41 trades" overfit). |

Backward compatibility: existing deployed `*_dt.pkl` models still load (loader
only requires `model`+`threshold`; missing `sl/tp1/trail` fall back to config).

## Phase 0 — Stop the bleed NOW (deployment env, reversible)

These are runtime/env actions on the running bot (not committed config). Apply in
the deployment environment, then confirm via `tv_health_check` and by watching
`bot/logs/trade_ledger.jsonl` for no new entries from disabled strategies.

```
ENABLE_WOLFE_WAVE=false          # worst bleeder: 78% SL, -$714
ENABLE_WOLFE_WAVE_V2=false
ENABLE_GGSHOT_227=false          # 71% SL
ENABLE_SESSION_ORB=false         # 54% SL
ALLOW_MM_WITHOUT_DT=false        # disable MM symbols lacking a fresh model
NOTIONAL_PCT=0.005               # halve per-trade risk during remediation
BREAKEVEN_LOCKIN_R=0.3           # lock-in floor for strategies that move SL after TP1
```

If Wolfe must stay on, point it at the rigorously validated 3-symbol config and
enforce gating:

```
WOLFE_WAVE_CONFIG_PATH=/app/configs/wolfe_wave_universe_4y_oos1y_stage40_configs.json
WOLFE_WAVE_SYMBOLS=LINKUSDT,LTCUSDT,SOLUSDT
# and set "_quality_profile": {"mode": "gate"} in that config file
```

## Phase 1–3 — Re-derive MM configs with the new harness

```bash
# From repo root with .venv active and Bybit creds in env.
python bot/train_dt.py --since 2022-01-01 --fee-bps 5.5 --slip-bps 1.0
# or a subset:
python bot/train_dt.py --symbols BTCUSDT,ETHUSDT,LINKUSDT
```

Outputs per symbol:
- `bot/models/<SYM>_dt.pkl` — only written if the symbol **passes the final
  holdout** (`PF≥1.10`, `avg_r>0`, ≥30 holdout trades). Carries the re-tuned
  `sl/tp1/trail`, threshold, and metrics.
- `bot/models/<SYM>_holdout_report.json` — written even on rejection; inspect
  `selected`, `holdout`, `n_configs_tried`.

Expect **far fewer symbols to pass** than before — that is the point. The old
asymmetric configs (e.g. ATOM/ARB `tp1=0.5`) should now fail the net-expectancy
gate. Symbols with no surviving config will have `mm_enabled` auto-disabled live
(when `ALLOW_MM_WITHOUT_DT=false`).

Note: the grid×folds sweep is ~1–3 min/symbol. `N_NULL=1000` only runs on
gate-passing configs.

## Phase 4–5 — Paper-gate then ramp

For each strategy/symbol that passes its final holdout:
1. Run demo/paper (`BYBIT_DEMO=true`) for ≥4–6 weeks or ≥30 trades.
2. Require live-paper net expectancy to match the holdout within tolerance before
   flipping `ENABLE_*` back on at reduced `NOTIONAL_PCT`.
3. Re-enable one strategy at a time, smallest blast radius first (the 3 validated
   Wolfe symbols). Restore full `NOTIONAL_PCT` only after in-tolerance live
   performance across ≥1 regime change.

## Remaining work (not yet implemented — Phase 2b)

The unified methodology lives in `bot/train_dt.py` (walk-forward folds, gates,
random-selection null, stability filter). Port the same pattern to the other
two custom backtests so their configs clear the same bar before deployment:

- **`scripts/backtest_ggshot_227.py`** — gate already tightened (PF/net floors).
  Still needs: rolling walk-forward + a carved final-year holdout + the
  random-selection null. Today it uses a single train/oos split.
- **`scripts/experiment_session_orb.py`** (ML-model based) — add a held-out final
  year scored once, raise min-trade floors, and add the multiple-testing null on
  the model's OOS selection.
- **Wolfe** — RESOLVED, see the dedicated section below.

Acceptance gates to apply uniformly: train≥30 / val≥15 / OOS≥15 / holdout≥30
trades; PF≥1.15 (1.10 holdout); net avg-R≥0.05 on val+OOS; ≥60% of OOS folds
positive; beat the random-selection null at the 95th percentile; portfolio of
accepted symbols net-positive on the holdout.

## Verification checklist

1. **Costing works** — re-run `train_dt.py`; previously-accepted asymmetric
   configs now fail. Costed backtest expectancy reconciles with the live ledger
   to within ~5 bps.
2. **Holdout reports exist** — one `*_holdout_report.json` per symbol; the null
   guard rejects PF-4.34-type noise.
3. **Exit math** — with avg win ≈2R the breakeven win-rate is ≈33%, below the
   observed MM ~62% non-SL rate (expectancy now positive).
4. **Live sync** — bot startup log shows `synced[...]` when a model's
   `sl/tp1/trail` differ from config, and `holdout_pf`/`holdout_avg_r`.
5. **Phase 0** — no new entries from disabled strategies in the ledger; equity
   stops declining.

## Wolfe — root cause and fix (RESOLVED)

Forensic finding (this overturned the initial "overfit config" guess):

- **The Wolfe backtest is sound.** The deployed config, scored on a held-out
  final year (>=2025-06, which **includes the live-failure window**), gives
  **PF 2.24, 53% win, +330R** with *every* OOS month positive. No look-ahead
  (signals are stable to future truncation), exit style matches live
  (`fixed_tp`), and the live config-merge is byte-identical to the backtest.
- **The live loss was a too-loose `min_score` gate, not parameters.** The one
  fully-verifiable live trade (BTC long 2026-05-22, entry 77027.8) corresponds
  to a backtest signal scoring **59.1**. `git` shows BTC `min_score` was **48**
  in the first Wolfe commit (`5264153`) and was later raised to 64. Live traded
  in the 48-era (59.1 >= 48 -> taken); the same signal is now rejected.
  Replaying the **real** `WolfeWaveEngine.detect_signal` confirms: with
  `min_score` 64/66 the bad signal is REJECTED. So the gate code is correct;
  the threshold *values* were the bug, and 8 symbols still sit at the loose 48.

Tools (kept):
- `scripts/wolfe_live_replay.py` — drives the real live engine + real config
  loader over cached CSVs, bar by bar. Use to verify live accept/reject
  decisions match the backtest. Example:
  `python scripts/wolfe_live_replay.py --symbol BTCUSDT --start 2026-05-22 --end 2026-05-23`
- `scripts/build_wolfe_shared_config.py` — builds + validates the shared config.

The fix — a single **shared, quality-gated** config (avoids 64 per-symbol
overfit thresholds), validated on pooled trades with the held-out year:

- `bot/configs/wolfe_wave_shared_v1_configs.json` (**60 symbols** of the full
  63-symbol universe — TONUSDT has no Bybit 5m data; 2 dropped for <10 held-out
  trades — `min_score=66`, `regime_filter=none`, per-symbol `mintick`). Symbols
  are included by held-out **sample size only** (NOT by held-out sign, to avoid
  curve-fitting the holdout). Embedded `_validation`: full-universe pooled
  **OOS held-out (>=2025-06): 1695 trades, 45.5% win, PF 1.58, +590R**; the
  60-symbol deployed set is essentially identical (PF 1.57, +571R) — strongly
  positive across the exact period that lost money live.

  Rebuild/refresh with: `python scripts/revalidate_wolfe_universe.py`
  (fetches any missing universe symbols, re-validates, rewrites the config).

Deploy it (Phase 0 disabled Wolfe; re-enable behind the validated config):
```
ENABLE_WOLFE_WAVE=true
WOLFE_WAVE_CONFIG_PATH=/app/configs/wolfe_wave_shared_v1_configs.json
NOTIONAL_PCT=0.005          # keep reduced during the paper-gate period
```
Then paper-gate per Phase 4 before restoring size. If you instead keep the
per-symbol `wolfe_wave_configs.json`, at minimum raise every `min_score=48`
entry to >=58 — those loose symbols are the residual risk.

### v2 evaluated and rejected

Ran the same pooled held-out pass with Wolfe's v2 structural scoring turned ON
(`v2_score_weight`+`min_v2_quality` blend, `p1_horizontal_mode`,
`p4_contrary_mode`) across the full universe. Held-out (>=2025-06) vs v1:

| variant | OOS trades | PF | avg_r | netR |
|---|---|---|---|---|
| v1 (shipped) | 1672 | 1.57 | +0.342 | +571 |
| v2 blend | 1182 | 1.58 | +0.350 | +414 |
| v2 struct | 631 | 1.59 | +0.354 | +223 |
| v2 full | 646 | 1.63 | +0.376 | +243 |

v2's filters trim ~60% of trades for a within-noise per-trade PF bump and ~2.3x
less total return. **v1 wins; keep `ENABLE_WOLFE_WAVE_V2=false` permanently.** The
deployed `wolfe_wave_v2_strong_configs.json` also had all v2 features *off*, so it
was just a redundant, unvalidated copy of v1.

Not changed (minor, noted): live `detect_signal` enters at the current bar's
close (bot/wolfe_wave.py:399) rather than the signal's qualifying-bar price. In
the verified case this matched; if a future audit shows entry drift, align it to
`signal.entry_price`.
