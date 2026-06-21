# Wolfe DCA Strategy — Implementation Plan

> **Status:** Steps 1–2 DONE (backtest + validation). The DCA is implemented in
> `run_backtest` (`WolfeConfig.dca_enabled`, `dca_stop_frac_k`) and validated across
> the full universe — the official engine reproduces the research prototype exactly:
> **held-out OOS PF 1.85, WR 64%, avg_r +0.506, net +608** (vs gated no-DCA 1.64 /
> 46% / +457). Config written to `bot/configs/wolfe_wave_shared_v1_dca_configs.json`
> (53 symbols, k=0.25). **The live config is unchanged** — live still runs the gated,
> no-DCA strategy. Steps 3–5 (live state machine + parity replay + paper-gate) await
> a separate go-ahead.
>
> **UPDATE — Step 3 is now a STANDALONE bot.** The earlier in-`bot.py` hooks were
> REVERTED (main bot has zero DCA code again) so the DCA position-monitoring can never
> perturb production. The live build is `bot/wolfe_dca_bot.py` — a self-contained runner
> (own process, own Bybit Demo account, own ledger) that reuses the shared Wolfe
> detection and POLL-monitors the DCA structure itself. Compose service `wolfe-dca-bot`
> (profile `dca`, separate account via `bot/.env.dca`). See "Standalone build" below.
> The backtest DCA (`run_backtest`, `WolfeConfig.dca_enabled/dca_stop_frac_k`) is kept.

## What & why
Add an optional **DCA second leg at the stop level** to the gated Wolfe strategy.
Research (`scripts/wolfe_dca_strategy_research.py`) shows that, conditional on
price reaching the original SL, it mean-reverts back to the entry ~58–60% of the
time. Exploiting this with a tight DCA stop (`k=0.25`) improves the held-out year
on scale-invariant metrics: **PF 1.64→1.85, ret/DD 12.2→14.2, 12/13 months ≥
original**, and it is **robust to slippage under a limit-based execution design**
(beats original at +0/3/7/15 bps).

## Trade logic (long; short mirrored)
Given a gated Wolfe signal: entry `E`, stop `SL`, target `T`, `R=|E-SL|`,
`k=0.25`, `SL2 = SL - k*R`.
- **Leg1** enters at `E` (market, as today). Protective stop for the position is
  `SL2` (NOT `SL` — `SL` is the DCA trigger, not a stop). Leg1 take-profit limit at `T`.
- A **leg2 limit buy** rests at `SL`.
- **Path A** — price hits `T` first: leg1 TPs (full RR). Cancel the leg2 limit. Done.
- **Path B** — price hits `SL` first: leg2 fills (now 2 units, avg `E-R/2`). Then:
  cancel the `T` target, place a **combined TP limit at `E`**, keep the hard stop at `SL2`.
    - bounce to `E`: combined exit (limit) → leg1 0 + leg2 +1R = **+1R** ("0.5R avg").
    - continue to `SL2`: hard stop (market) → leg1 −(1+k)R + leg2 −k*R = **−(1+2k)R = −1.5R**.

Only the `SL2` hard stop is a market order; leg2 entry and the `E` exit are limits.

## Changes required

### 1. Backtest / shared validation (`scripts/backtest_wolfe_wave.py`)
The trade simulator (`run_backtest`) must model the DCA exactly as the live bot
will, so config validation reflects reality (the `min_score`-divergence lesson).
- Add `WolfeConfig` fields: `dca_enabled: bool=False`, `dca_stop_frac_k: float=0.25`,
  `dca_target: str="first_entry"`.
- Port the DCA branch from `wolfe_dca_strategy_research.py::sim` into `run_backtest`
  (leg2 fill at SL, combined target E, combined stop SL2, 2-leg P/L, per-leg cost).
- Reuse `_cost_r`; cost = 1 round-trip on Path A, 2 on Path B.

### 2. Re-validate + regenerate config
- Re-run `scripts/revalidate_wolfe_universe.py` with `dca_enabled=true, k=0.25`.
- Regenerate `wolfe_wave_shared_v1_configs.json` (or a `_dca` variant) with the DCA
  params and held-out provenance (`_validation` block: DCA PF/ret-DD).

### 3. Live execution (`bot/wolfe_wave.py` + `bot/bot.py`)
This is the substantial, highest-risk part — a small state machine replacing the
current single `fixed_tp` bracket. New `exit_style="wolfe_dca"`:
- On signal: market leg1 @E; limit TP @T; resting **limit buy leg2 @SL**; hard stop @SL2.
- On leg2 fill (private WS): cancel TP@T; place combined limit TP @E; (stop already @SL2).
- On TP@T fill (Path A): cancel resting leg2 limit; close.
- On TP@E fill (Path B bounce) or stop@SL2 (Path B continue): close; cancel siblings.
- State to track per position: `phase ∈ {leg1_only, combined}`, leg2 order id, sizes.
- Position-mode/`positionIdx`, qty rounding, and reduce-only flags as per existing code.

### 4. Live↔backtest parity check
- Extend `scripts/wolfe_live_replay.py` (or a new replay) to drive the live DCA
  decision path over cached data and confirm it reproduces the backtest trades.

### 5. Paper-gate (NON-NEGOTIABLE before sizing up)
The backtest assumes the **limit fills happen** at `E` and `SL`. That is the one
thing it cannot verify. Run on demo and measure:
- **leg2 limit fill rate** at SL (does it fill when price tags SL?).
- **bounce-exit limit fill rate** at E (does it fill, or does price reverse first?).
- **real `SL2` stop slippage** (market, in fast moves).
- Compare live DCA realized PF/avg-R to the backtest. Only ramp size if they match
  within tolerance over ≥30 Path-B trades.

## Risks / open items
- **Limit-fill uncertainty (top risk):** if the `E` exit limit misses (price reverses
  before filling), the trade rides to `SL2` or timeout — worse than backtest. Paper-gate
  measures this; consider a fallback (e.g., market-exit a fraction if price stalls near E).
- **Order-management complexity:** multi-leg conditional state is the most bug-prone code
  in the system; build behind a flag, default off, and unit-test the state transitions.
- **Position count / margin:** Path B doubles size on ~50% of trades → interacts with
  `MAX_OPEN_POSITIONS`, cluster guard, and margin. Size leg1 so the combined worst case
  (≈1.5R) fits the per-trade risk budget.
- **Regime:** tight stop already fixed the k=1 fragility (12/13 months), but it remains a
  mean-reversion bet; keep monitoring the checkpoint's `Accepted (gated)` line.

## Recommendation
Worth building — it's the strongest validated improvement after the HTF gate, and
robust to slippage under the (natural) limit-based design. But it is a multi-step,
carefully-tested build, not a quick deploy. Sequence: (1) backtest DCA + re-validate
→ (2) live state machine behind a flag → (3) parity replay → (4) paper-gate to verify
limit fills → (5) ramp. Do not skip the paper-gate.

## Live build — as implemented (Step 3)

Code (gated entirely by the config's `dca_enabled`; default off → no effect on the
deployed gated strategy):

- **`bot/wolfe_wave.py` `detect_signal`**: when `cfg.dca_enabled`, sets
  `payload["exit_style"]="wolfe_dca"` and `payload["dca_k"]`. (Override is before the
  min_score reject, so the score gate is unaffected.)
- **`bot/bot.py` `_execute_trade`**: for `exit_style=="wolfe_dca"`, leg1 market entry
  carries the HARD stop at `SL2 = SL - k*R` (not SL) and TP at `T`; active_trade gains
  `dca_phase="leg1_only"`, `dca_k`, `dca_leg2_price(=SL)`, `dca_sl2`, `dca_target(=E)`,
  `dca_leg2_qty`. After entry, `_place_dca_leg2` rests a **limit add** (not reduce-only)
  at SL with a `DCAL2-…` orderLinkId.
- **`_on_private_order`**: a fill whose orderLinkId starts `DCAL2` →
  `_on_dca_leg2_filled` (amends position TP `T→E` via `set_trading_stop`, keeps stop
  SL2, sets `dca_phase="combined"`, logs a `DCA LEG2 FILLED` ledger event). On a full
  position close while still `leg1_only`, `_cancel_dca_leg2` cancels the resting leg2.
- Helpers: `_place_dca_leg2`, `_on_dca_leg2_filled`, `_cancel_dca_leg2`.

Paths: A (T before SL) → leg1 full TP, leg2 cancelled. B (SL first) → leg2 fills,
target → E; bounce = +1R combined, continue → SL2 = −(1+2k)R.

### Known edge cases for the paper-gate to watch
- **Limit-fill reality** (top risk): backtest assumes leg2@SL and exit@E fill when
  price tags them. Measure real fill rates on demo.
- **Fast-gap race**: if price gaps through SL and SL2 in one bar, leg2 add and the SL2
  stop can both trigger; or a close-then-leg2-fill race could leave a stray position.
  `_cancel_dca_leg2` fires on close, and `_sync_positions` reconciles strays, but verify
  on demo. Consider a post-close sweep that cancels any lingering `DCAL2` order.
- **Reconciliation**: a `wolfe_dca` position carries a resting `DCAL2` order; ensure
  startup/`_sync_positions` doesn't treat it as stale while leg1 is open (it isn't in
  `tp_orders`, so the existing stale-TP cancel won't touch it).

### How to paper-gate (demo only — do NOT point production at the DCA config yet)
On a demo/paper mm-bot instance set `WOLFE_WAVE_CONFIG_PATH=/app/configs/wolfe_wave_shared_v1_dca_configs.json`
(+ `BYBIT_DEMO=true`, reduced `NOTIONAL_PCT`). Watch the checkpoint `Accepted (gated)`
line and the ledger for `DCA LEG2 FILLED` / `dca_*` exits; compare realized PF/WR to the
backtest (≥30 Path-B trades) before considering production.

## Standalone build (`bot/wolfe_dca_bot.py`) — supersedes the in-bot.py approach

Why standalone: the production mm-bot already handles a lot (MM/turtle/session_orb/
ggshot/wolfe + RL forwarding + reconciliation of all positions). Embedding a DCA
state machine there risks perturbing that. So the DCA runs as its **own process on its
own demo account**, and the main `bot.py`/`wolfe_wave.py` were reverted to zero DCA code.

- **Detection:** reuses `WolfeWaveEngine.detect_signal` (same gate, min_score, HTF
  over-extension) — parity with the backtest, no duplicated pattern logic.
- **Execution + monitoring (self-contained, POLL-based, no private WS):**
  `DcaExecutor` places leg1 (market, Full bracket: stop SL2, TP T) + a resting limit
  leg2 add at SL, then a poller (`WOLFE_DCA_POLL_SECONDS`, default 10s) on
  `get_positions` drives the state machine:
    - size grew past leg1 → leg2 filled → `set_trading_stop` retargets TP T→E (keep SL2),
      phase `combined`, logs `DCA_LEG2_FILLED`.
    - size==0 → closed (Path A TP@T, bounce TP@E, or hard stop SL2) → cancel resting leg2
      if still `leg1_only`, finalize, free slot.
    - timeout (`max_hold_bars`) → reduce-only market close + cancel leg2.
  Handles both directions (hedge long_idx=1 / short_idx=2, auto-detected; one-way=0).
- **Isolation:** own keys (`WOLFE_DCA_BYBIT_API_KEY/SECRET`), own ledger
  (`bot/logs_dca/wolfe_dca_ledger.jsonl`), own state in-process. Touches nothing of the
  production bot.

### Run it (paper-gate)
1. Create a SECOND Bybit Demo account/API key.
2. `cp bot/.env.dca.example bot/.env.dca` and set `WOLFE_DCA_BYBIT_API_KEY/SECRET`.
3. `docker compose up -d --build wolfe-dca-bot`  (profile-gated; a plain `up` ignores it).
4. Watch `docker logs wolfe-dca-bot` (startup: N symbols, exec on) and
   `bot/logs_dca/wolfe_dca_ledger.jsonl` for `signal` rows, `DCA_LEG2_FILLED`, `DCA_EXIT`.
5. Compare realized PF/WR to the backtest (1.85 / 64%) over ≥30 Path-B trades.

### Edge cases to watch on the paper-gate
- **Poll latency:** after leg1 TPs at T, the resting leg2 stays live until the next poll
  (~10s) cancels it; in a fast reversal price could tag SL and fill leg2 first. The
  poller then sees a `combined` position with no matching leg1 and manages it to E/SL2
  — acceptable, but verify no stray positions. Lower `WOLFE_DCA_POLL_SECONDS` if needed.
- **Limit-fill reality** (the one thing the backtest assumes): does leg2@SL / exit@E
  actually fill? Measure from the ledger.
- **Dockerfile** now COPYies `wolfe_dca_bot.py` (and `capitulation_spring_bot.py`, which
  was previously missing) — rebuild required.
