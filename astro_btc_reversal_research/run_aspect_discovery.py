"""Milestone 2 - full aspect library discovery (CLI).

Runs cross-pair x aspect pivot-window lift discovery for one timeframe.
Outputs (reports/): discovery_<tf>.csv, discovery_<tf>.json, discovery_<tf>.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PKG_SRC = Path(__file__).resolve().parent / "src"
if str(PKG_SRC) not in sys.path:
    sys.path.insert(0, str(PKG_SRC))

from astro_reversal import data, discovery, report  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Aspect library discovery (Milestone 2).")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--timeframe", default="1d", choices=["1h", "4h", "1d"])
    p.add_argument("--symbol", default=None)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--window-kind", default=None, choices=["tight", "medium", "wide"])
    p.add_argument("--pivot-threshold-atr", type=float, default=None)
    p.add_argument("--outdir", type=Path, default=data.REPORTS_DIR)
    args = p.parse_args()

    cfg = data.load_config(args.config)
    symbol = args.symbol or cfg["data"]["symbol"]
    start = args.start or cfg["data"]["start"]
    end = args.end or cfg["data"]["end"]
    tf = args.timeframe
    disc = cfg["discovery"]
    window_kind = args.window_kind or disc["window_kind"]

    frame = data.load_ohlcv(symbol, tf, start, end, base_interval=cfg["data"]["base_interval"])
    holdout_start = pd.Timestamp(cfg["holdout_start"], tz="UTC")

    out = discovery.run_discovery(
        frame=frame,
        timeframe=tf,
        bodies=cfg["bodies"],
        aspects=cfg["aspects"]["discovery"],
        orb_deg=float(cfg["orb"]["discovery_orb_degrees"][tf]),
        window_bars=int(cfg["windows"][tf][window_kind]),
        window_kind=window_kind,
        pivot_threshold_atr=(args.pivot_threshold_atr if args.pivot_threshold_atr is not None
                             else float(disc["pivot_threshold_atr"][tf])),
        holdout_start=holdout_start,
        min_in_window_bars=int(disc["min_in_window_bars"]),
        random_draws=int(disc["random_draws"]),
        fdr_alpha=float(disc["fdr_alpha"]),
        seed=int(cfg.get("random_seed", 42)),
        symbol=symbol,
        start=start,
        end=end,
    )

    outdir = args.outdir
    report.write_csv(outdir / f"discovery_{tf}.csv", out["table"])
    report.write_json(outdir / f"discovery_{tf}.json", out["results"])
    (outdir / f"discovery_{tf}.md").write_text(report.discovery_markdown(out["results"]), encoding="utf-8")
    print(report.discovery_markdown(out["results"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
