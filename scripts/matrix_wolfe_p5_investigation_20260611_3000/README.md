# Matrix Wolfe p5 Backtest Investigation

Run date: 2026-06-11

Scope:
- Matrix room `!gYMwAkfoJVPngqDULV:thebox.sbs`
- Parsed Wolfe lifecycle messages from roughly 2026-06-05 to 2026-06-11
- Symbols seen: ADAUSDT, BTCUSDT, ETHUSDT, SOLUSDT, XAGUSDT, XAUUSDT, XLMUSDT, XRPUSDT
- Entry model: next 1m open after `[found_p5]`
- Holding limit: 360 minutes
- Fee model: 0.055% per side equivalent in R via entry/exit notional

Lifecycle:
- Parsed lifecycle events: 1134
- Found p5 waves: 512
- Exact wave-id p5 to bona_fide links: 0
- Proximity p5 to bona_fide links: 111 / 512 (21.7%)
- Proximity p5 to entry links: 4 / 512 (0.8%)

Important caveat:
- `[found_p5]` messages do not include entry zone, stop, or targets.
- Exact `wave_id` does not link p5 to bona_fide/entry in this sample.
- Proximity links are useful diagnostically, but they can introduce survivor bias and are not a clean live-entry rule.

Main result:
- Raw p5 early entries with generic swing stops are negative overall.
- The problem is mostly very tight stops: fees become too large in R.
- If we require fee-valid stops (`fee_to_stop_risk <= 0.25R`), 3R variants turn positive:
  - swing20 TP3R: 130 trades, +20.84R, +0.160R avg, PF 1.22
  - swing40 TP3R: 162 trades, +20.72R, +0.128R avg, PF 1.18
  - swing80 TP3R: 202 trades, +25.74R, +0.127R avg, PF 1.19

Best fee-valid symbol pockets:
- ADAUSDT swing80 TP3R: 38 trades, +26.26R, +0.691R avg, PF 2.30
- ADAUSDT swing20 TP3R: 30 trades, +22.88R, +0.763R avg, PF 2.37
- XRPUSDT swing40 TP3R: 18 trades, +15.27R, +0.848R avg, PF 2.86
- XRPUSDT swing80 TP3R: 25 trades, +14.67R, +0.587R avg, PF 2.14

Interpretation:
- Do not deploy all p5 messages raw.
- The promising live hypothesis is p5 early entry only when the structural stop is wide enough to be fee-valid, preferably with a 3R target and symbol/timeframe filters.
- Start paper/shadow with ADA/XRP first; XLM is mixed in this window, especially long vs short.

