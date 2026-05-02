from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ml_breaker_continuation_filter import enrich


BASKETS: dict[str, list[str]] = {
    "top3": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    "btc_sol": ["BTCUSDT", "SOLUSDT"],
    "all": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "TRXUSDT", "LINKUSDT", "AVAXUSDT"],
    "no_doge": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "TRXUSDT", "LINKUSDT", "AVAXUSDT"],
    "no_doge_avax": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "TRXUSDT", "LINKUSDT"],
    "quality4": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "LINKUSDT"],
    "quality5": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "LINKUSDT", "BNBUSDT"],
    "quality6": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "LINKUSDT", "BNBUSDT", "XRPUSDT"],
    "trend5": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "LINKUSDT", "TRXUSDT"],
    "liq6": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "LINKUSDT", "TRXUSDT"],
    "big4": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"],
}


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def fold_profit_factor(frame: pd.DataFrame) -> float:
    wins = frame.loc[frame["r_net_cost"] > 0, "r_net_cost"].sum()
    losses = abs(frame.loc[frame["r_net_cost"] <= 0, "r_net_cost"].sum())
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return float(wins) / float(losses)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Study breaker ML gates on curated symbol baskets.")
    parser.add_argument("scored_file", type=Path)
    parser.add_argument("--fees", default="5,8,10,12")
    parser.add_argument("--prob-values", default="0.55,0.575")
    parser.add_argument("--min-risk-pct", type=float, default=1.0)
    parser.add_argument("--direction", default="long")
    parser.add_argument("--min-fold-trades", type=int, default=20)
    parser.add_argument("--baskets", default="quality6,liq6,no_doge_avax,big4,all")
    parser.add_argument("--out", type=Path, default=Path("scripts/breaker_basket_study.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = pd.read_csv(args.scored_file)
    baskets = [name.strip() for name in args.baskets.split(",") if name.strip()]
    fees = parse_float_list(args.fees)
    probs = parse_float_list(args.prob_values)

    rows: list[dict[str, Any]] = []
    for fee in fees:
        frame = enrich(raw.copy(), fee)
        for prob in probs:
            gated = frame[pd.to_numeric(frame["breaker_prob"], errors="coerce") >= prob].copy()
            if args.direction != "ALL":
                gated = gated[gated["direction"].astype(str) == args.direction].copy()
            if args.min_risk_pct > 0:
                gated = gated[pd.to_numeric(gated["risk_pct"], errors="coerce") >= args.min_risk_pct].copy()

            for basket in baskets:
                symbols = BASKETS[basket]
                selected = gated[gated["symbol"].astype(str).isin(symbols)].copy()
                row: dict[str, Any] = {
                    "fee_bps_side": fee,
                    "min_prob": prob,
                    "direction": args.direction,
                    "min_risk_pct": args.min_risk_pct,
                    "basket": basket,
                    "symbols": " ".join(symbols),
                    "aggregate_trades": int(len(selected)),
                    "aggregate_net_r": round(float(selected["r_net_cost"].sum()), 3) if not selected.empty else 0.0,
                    "aggregate_win_rate": round(100.0 * float(selected["r_net_cost"].gt(0).mean()), 2) if not selected.empty else 0.0,
                }
                fold_positive = True
                for fold in [1, 2, 3]:
                    fold_frame = selected[selected["fold"] == fold].copy()
                    pf = fold_profit_factor(fold_frame)
                    net_r = round(float(fold_frame["r_net_cost"].sum()), 3) if not fold_frame.empty else 0.0
                    row[f"fold{fold}_trades"] = int(len(fold_frame))
                    row[f"fold{fold}_net_r"] = net_r
                    row[f"fold{fold}_pf"] = round(pf, 3)
                    if len(fold_frame) < args.min_fold_trades or net_r <= 0 or pf <= 1.0:
                        fold_positive = False
                row["passes"] = fold_positive
                rows.append(row)

    out = pd.DataFrame(rows).sort_values(["fee_bps_side", "passes", "aggregate_net_r"], ascending=[True, False, False])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))
    print(f"\nSaved {len(out)} rows to {args.out}")


if __name__ == "__main__":
    main()
