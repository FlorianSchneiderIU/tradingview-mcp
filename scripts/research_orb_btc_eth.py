#!/usr/bin/env python3
"""Base-rate scan of the ORB/Judas concept on BTC/ETH — NO ML, selection-free.

A beginner bolts ML on top hoping it finds an edge. First question: does the RAW setup
have any pulse? We pool ALL trades within each (family, entry_mode) across the whole
config grid (no config cherry-picking) and compare the dev period vs an untouched 365d
holdout. If a whole family is <1.0 PF in both, the concept doesn't work and ML was noise.
Cost = fees+slippage (8 bps/side)."""
from __future__ import annotations

import os
import sys
import warnings
from argparse import Namespace

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.experiment_session_orb import add_htf_context, build_grid  # noqa: E402
from scripts.experiment_session_orb_fast import build_contexts, generate_trades, to_arrays  # noqa: E402

FEE = 8.0
DATA = "scripts/data/{}_5m_bybit.csv"
GRID = Namespace(sessions="asia,london,ny", or_minutes="30,60,90", grid_mode="full")


def pooled(df: pd.DataFrame) -> dict:
    if df.empty:
        return dict(trades=0, pf=0.0, net_r=0.0, avg_r=0.0, win=0.0)
    r = df["r_multiple"].to_numpy(float)
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    return dict(trades=len(r), pf=(gp / gl if gl > 0 else np.inf), net_r=r.sum(),
                avg_r=r.mean(), win=(r > 0).mean())


def load_all_trades(symbol: str) -> pd.DataFrame:
    raw = pd.read_csv(DATA.format(symbol.lower()))
    raw["open_time"] = pd.to_datetime(raw["open_time"], utc=True)
    raw["close_time"] = pd.to_datetime(raw["close_time"], utc=True)
    for c in ("open", "high", "low", "close", "volume"):
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = raw.dropna(subset=["open", "high", "low", "close"]).sort_values("open_time").reset_index(drop=True)
    df = add_htf_context(raw).reset_index(drop=True)
    a = to_arrays(df)
    ctx = build_contexts(df, a, sessions=["asia", "london", "ny"], or_minutes=[30, 60, 90])
    frames = []
    for cfg in build_grid(GRID):
        t = generate_trades(a, symbol=symbol, cfg=cfg, contexts=ctx, fee_bps_per_side=FEE)
        if t.empty:
            continue
        t = t.copy()
        t["family"], t["entry_mode"], t["cfg_session"], t["rr_"] = cfg.family, cfg.entry_mode, cfg.session, cfg.rr
        t["variant"] = cfg.variant
        frames.append(t)
    cand = pd.concat(frames, ignore_index=True)
    cand["entry_time"] = pd.to_datetime(cand["entry_time"], utc=True, errors="coerce")
    return cand.dropna(subset=["entry_time", "r_multiple"])


def show(title: str, groups) -> None:
    print(f"\n-- {title} --")
    print(f"{'group':<34}{'dev_n':>7}{'dev_PF':>8}{'dev_R':>8}{'hold_n':>8}{'hold_PF':>9}{'hold_R':>8}{'h_win%':>8}")
    for key, d, h in groups:
        print(f"{str(key):<34}{d['trades']:>7}{d['pf']:>8.2f}{d['net_r']:>8.1f}"
              f"{h['trades']:>8}{h['pf']:>9.2f}{h['net_r']:>8.1f}{h['win']*100:>7.0f}%")


def main() -> None:
    for symbol in ("BTCUSDT", "ETHUSDT"):
        print(f"\n{'='*90}\n{symbol}\n{'='*90}", flush=True)
        cand = load_all_trades(symbol)
        hstart = cand["entry_time"].max() - pd.Timedelta(days=365)
        dev = cand[cand.entry_time < hstart]
        hold = cand[cand.entry_time >= hstart]
        print(f"{len(cand)} trades  {cand.entry_time.min().date()} -> {cand.entry_time.max().date()}  "
              f"holdout>={hstart.date()}  (dev {len(dev)} / hold {len(hold)})")

        def grp(cols):
            out = []
            for key, g in cand.groupby(cols):
                d = pooled(g[g.entry_time < hstart]); h = pooled(g[g.entry_time >= hstart])
                out.append((key, d, h))
            return sorted(out, key=lambda x: x[2]["pf"], reverse=True)

        show("by family x entry_mode (selection-free)", grp(["family", "entry_mode"]))
        show("by family x entry_mode x direction", grp(["family", "entry_mode", "direction"]))
        show("by family x entry_mode x session", grp(["family", "entry_mode", "cfg_session"]))

        # per-config: pick the BEST configs by DEV avg_r (min trades), then see holdout.
        rows = []
        for variant, g in cand.groupby("variant"):
            d = pooled(g[g.entry_time < hstart]); h = pooled(g[g.entry_time >= hstart])
            if d["trades"] >= 150 and h["trades"] >= 50:
                rows.append((variant, d, h))
        rows.sort(key=lambda x: x[1]["avg_r"], reverse=True)
        print("\n-- TOP 12 configs by DEV avg_r -> holdout (single-config, no ML) --")
        print(f"{'variant':<40}{'dev_n':>6}{'dev_avgR':>9}{'dev_PF':>7}{'h_n':>6}{'h_avgR':>9}{'h_PF':>7}")
        for v, d, h in rows[:12]:
            print(f"{str(v)[:39]:<40}{d['trades']:>6}{d['avg_r']:>9.3f}{d['pf']:>7.2f}{h['trades']:>6}{h['avg_r']:>9.3f}{h['pf']:>7.2f}")
        pos = [r for r in rows if r[1]["avg_r"] > 0 and r[2]["avg_r"] > 0]
        print(f"configs positive in BOTH dev and holdout: {len(pos)} / {len(rows)}")


if __name__ == "__main__":
    main()
