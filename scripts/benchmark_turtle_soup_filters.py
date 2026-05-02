from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backtest_turtle_soup import Config, fetch_klines, run_backtest, summarize


def profile_configs() -> list[tuple[str, Config]]:
    base = dict(exec_tf="5m", structure_tf="15m", entry_mode="limit_mid")
    return [
        ("baseline", Config(**base)),
        ("dead_zone", Config(**base, block_dead_zone=True)),
        ("htf_4h", Config(**base, htf_bias_mode="4h_ema")),
        ("htf_4h_1d", Config(**base, htf_bias_mode="4h_1d_ema")),
        ("htf_4h_dead", Config(**base, htf_bias_mode="4h_ema", block_dead_zone=True)),
        ("dead_reclaim70", Config(**base, block_dead_zone=True, min_sweep_reclaim_pos=0.70)),
        ("htf_4h_dead_reclaim70", Config(**base, htf_bias_mode="4h_ema", block_dead_zone=True, min_sweep_reclaim_pos=0.70)),
        ("dead_reclaim70_vol1_2", Config(**base, block_dead_zone=True, min_sweep_reclaim_pos=0.70, min_sweep_volume_mult=1.0, max_sweep_volume_mult=2.0)),
        ("htf_4h_first4", Config(**base, htf_bias_mode="4h_ema", use_first4_return_bias=True, first4_return_threshold=0.5)),
        ("htf_4h_prevday", Config(**base, htf_bias_mode="4h_ema", use_prev_day_reversion_bias=True, prev_day_reversion_threshold=1.0)),
        ("htf_4h_dead_first4", Config(**base, htf_bias_mode="4h_ema", block_dead_zone=True, use_first4_return_bias=True, first4_return_threshold=0.5)),
        ("htf_4h_dead_prevday", Config(**base, htf_bias_mode="4h_ema", block_dead_zone=True, use_prev_day_reversion_bias=True, prev_day_reversion_threshold=1.0)),
        ("htf_4h_dead_allbias", Config(
            **base,
            htf_bias_mode="4h_ema",
            block_dead_zone=True,
            use_first4_return_bias=True,
            first4_return_threshold=0.5,
            use_prev_day_reversion_bias=True,
            prev_day_reversion_threshold=1.0,
            use_thursday_bearish_bias=True,
        )),
    ]


def main() -> None:
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=180)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    symbols = ["ETHUSDT", "SOLUSDT"]
    data = {symbol: fetch_klines(symbol, "5m", start_ms, end_ms) for symbol in symbols}

    rows: list[dict] = []
    for profile, cfg in profile_configs():
        combined_net = 0.0
        combined_pf = []
        min_trades = None
        profitable_symbols = 0
        row: dict[str, object] = {"profile": profile}
        for symbol in symbols:
            trades = run_backtest(data[symbol], cfg)
            summary = summarize(trades)
            row[f"{symbol}_trades"] = summary["trades"]
            row[f"{symbol}_wr"] = summary["win_rate"]
            row[f"{symbol}_pf"] = summary["profit_factor"]
            row[f"{symbol}_net_r"] = summary["net_r"]
            combined_net += summary["net_r"]
            combined_pf.append(summary["profit_factor"])
            profitable_symbols += 1 if summary["net_r"] > 0 else 0
            min_trades = summary["trades"] if min_trades is None else min(min_trades, summary["trades"])

        finite_pfs = [pf for pf in combined_pf if pf != float("inf")]
        row["combined_net_r"] = round(combined_net, 3)
        row["avg_pf"] = round(sum(finite_pfs) / len(finite_pfs), 3) if finite_pfs else float("inf")
        row["min_trades"] = min_trades if min_trades is not None else 0
        row["profitable_symbols"] = profitable_symbols
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(
        ["profitable_symbols", "combined_net_r", "avg_pf", "min_trades"],
        ascending=[False, False, False, False],
    )
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
