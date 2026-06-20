#!/usr/bin/env python3
"""
wolfe_live_replay.py — Faithfully replay the LIVE Wolfe engine over cached data.
================================================================================
Drives the real bot.wolfe_wave.WolfeWaveEngine.detect_signal (and the real
config loader) bar-by-bar, exactly like bot.py does in production, against the
cached <symbol>_5m_bybit.csv files. This isolates live/backtest divergence:
which signals the LIVE path would actually trade (accepted) vs reject, with the
score vs the configured min_score gate.

Usage:
    python scripts/wolfe_live_replay.py --symbol BTCUSDT --start 2026-05-20 --end 2026-05-24
    python scripts/wolfe_live_replay.py --symbol BTCUSDT --start 2026-05-20 --end 2026-05-24 --config bot/configs/wolfe_wave_configs.json
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

# pybit is pulled in by bot.wolfe_wave -> turtle_soup; not needed for CSV replay.
# Stub it if absent so this tool runs offline. In deployment pybit exists and
# this block is a no-op.
try:  # noqa: SIM105
    import pybit  # noqa: F401
except ModuleNotFoundError:
    import types
    _m = types.ModuleType("pybit")
    _ut = types.ModuleType("pybit.unified_trading")
    _ut.HTTP = type("HTTP", (), {"__init__": lambda self, *a, **k: None})
    _m.unified_trading = _ut
    sys.modules["pybit"] = _m
    sys.modules["pybit.unified_trading"] = _ut

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "bot"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import pandas as pd  # noqa: E402

import wolfe_wave as ww  # noqa: E402


def load_bars(symbol: str, data_dir: str) -> list[dict]:
    path = os.path.join(data_dir, f"{symbol.lower()}_5m_bybit.csv")
    df = pd.read_csv(path, usecols=["open_time", "open", "high", "low", "close", "volume"])
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.sort_values("open_time").reset_index(drop=True)
    bars = []
    for _, r in df.iterrows():
        bars.append({
            "ts": int(r["open_time"].timestamp() * 1000),
            "open": float(r["open"]), "high": float(r["high"]),
            "low": float(r["low"]), "close": float(r["close"]),
            "volume": float(r["volume"]),
            "_t": r["open_time"],
        })
    return bars


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay live Wolfe engine over cached data")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--start", required=True, help="window start (UTC date/time)")
    ap.add_argument("--end", required=True, help="window end (UTC date/time)")
    ap.add_argument("--warmup", type=int, default=8000, help="bars of context before --start")
    ap.add_argument("--data-dir", default=os.path.join(REPO, "scripts", "data"))
    ap.add_argument("--config", default=os.path.join(REPO, "bot", "configs", "wolfe_wave_configs.json"))
    ap.add_argument("--strategy-name", default="wolfe_wave")
    args = ap.parse_args()

    sym = args.symbol.upper()
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")

    cfg = load_wolfe_cfg(sym, args.config)
    print(f"{sym}: min_score={cfg.min_score:g} pattern_tf={cfg.pattern_tf} exec_tf={cfg.exec_tf} "
          f"quality={cfg.research_quality_profile}/{cfg.research_quality_mode} "
          f"min_rr={cfg.min_rr:g} max_rr={cfg.max_rr:g}")

    bars = load_bars(sym, args.data_dir)
    win_start_i = next((i for i, b in enumerate(bars) if b["_t"] >= start), None)
    if win_start_i is None:
        print("no bars at/after --start"); return
    warm_from = max(0, win_start_i - args.warmup)

    engine = ww.WolfeWaveEngine(strategy_name=args.strategy_name)
    state = ww.WolfeWaveState(sym, cfg, max_bars=20000)
    # Pre-fill warmup context (as bot.py does on startup), then step the window.
    for b in bars[warm_from:win_start_i]:
        state.push_bar({k: b[k] for k in ("ts", "open", "high", "low", "close", "volume")})

    accepted = rejected = 0
    print(f"\nreplaying {sym} {start.date()}..{end.date()} (warmup {win_start_i - warm_from} bars, "
          f"engine.min_bars={engine.min_bars})\n")
    for b in bars[win_start_i:]:
        if b["_t"] > end:
            break
        state.push_bar({k: b[k] for k in ("ts", "open", "high", "low", "close", "volume")})
        sig = engine.detect_signal(state)
        if sig is None:
            continue
        is_rej = bool(sig.get("rejected"))
        score = sig.get("score")
        decision = "REJECT" if is_rej else "ACCEPT(would trade)"
        if is_rej:
            rejected += 1
        else:
            accepted += 1
        print(f"  {str(b['_t'])[:16]}  {sig.get('signal'):5} entry={sig.get('entry'):g} "
              f"score={score if score is None else round(float(score),1)} vs min_score={cfg.min_score:g}  "
              f"rr={sig.get('target_rr_planned')}  -> {decision}"
              + (f"  [{sig.get('reject_reason')}]" if is_rej else ""))

    print(f"\nSUMMARY {sym}: accepted(would-trade)={accepted}  rejected={rejected}")


def load_wolfe_cfg(symbol: str, config_path: str):
    cfgs = ww.load_wolfe_wave_configs(symbols=[symbol], config_path=config_path)
    if symbol not in cfgs:
        raise SystemExit(f"{symbol} not in config {config_path}")
    return cfgs[symbol]


if __name__ == "__main__":
    main()
