"""Pivot-definition diagnostic.

Answers "is our pivot definition sensible?" by reporting, per timeframe and per
ATR directional-change threshold (and a couple of fractal settings), how many
pivots result, their cadence, and the forward-label base rate. Use it to pick
thresholds that yield *rare, significant* reversal windows rather than ordinary
swing churn.

Outputs reports/pivot_diagnostic.json and prints a table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PKG_SRC = Path(__file__).resolve().parent / "src"
if str(PKG_SRC) not in sys.path:
    sys.path.insert(0, str(PKG_SRC))

from astro_reversal import data, pivots, report  # noqa: E402

DEFAULT_HORIZONS = {"1d": 3, "4h": 6, "1h": 12}
DEFAULT_THRESHOLDS = [1.5, 2.0, 3.0, 4.0, 5.0, 7.0]
DEFAULT_FRACTALS = {"1d": [3, 5, 10], "4h": [6, 12], "1h": [12, 24]}


def main() -> int:
    p = argparse.ArgumentParser(description="Pivot-definition diagnostic.")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--timeframes", nargs="+", default=["1d", "4h", "1h"])
    p.add_argument("--outdir", type=Path, default=data.REPORTS_DIR)
    args = p.parse_args()

    cfg = data.load_config(args.config)
    out: dict = {"config": {"symbol": cfg["data"]["symbol"], "start": cfg["data"]["start"],
                            "end": cfg["data"]["end"]}, "timeframes": {}}

    header = f"{'tf':>3} {'def':>10} {'npiv':>5} {'gap_bars':>8} {'gap_days':>8} {'br_any':>7} {'br_low':>7} {'br_high':>7}"
    print(header)
    print("-" * len(header))
    for tf in args.timeframes:
        H = DEFAULT_HORIZONS.get(tf, 3)
        frame = data.load_ohlcv(cfg["data"]["symbol"], tf, cfg["data"]["start"], cfg["data"]["end"],
                                base_interval=cfg["data"]["base_interval"])
        rows = []
        for thr in DEFAULT_THRESHOLDS:
            st = pivots.pivot_stats(frame, pivots.atr_directional_pivots(frame, thr), H)
            st.update({"definition": f"atr{thr:g}", "threshold": thr})
            rows.append(st)
        for lr in DEFAULT_FRACTALS.get(tf, []):
            st = pivots.pivot_stats(frame, pivots.fractal_pivots(frame, lr, lr), H)
            st.update({"definition": f"fractal{lr}", "left_right": lr})
            rows.append(st)
        out["timeframes"][tf] = {"horizon": H, "rows": rows}
        for r in rows:
            print(f"{tf:>3} {r['definition']:>10} {r['n_pivots']:>5} {r['median_gap_bars']:>8.1f} "
                  f"{r['median_gap_days']:>8.2f} {r['base_rate_any']:>7.3f} {r['base_rate_low']:>7.3f} "
                  f"{r['base_rate_high']:>7.3f}")
        print()

    report.write_json(args.outdir / "pivot_diagnostic.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
