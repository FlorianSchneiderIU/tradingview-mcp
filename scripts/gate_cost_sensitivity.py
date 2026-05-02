from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ml_breaker_continuation_filter import enrich, trade_metrics


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_gate(value: str) -> dict[str, Any]:
    parts = [part.strip() for part in value.split(":")]
    if len(parts) != 5:
        raise ValueError("Gate must be name:min_prob:direction:min_risk_pct:symbol")
    return {
        "name": parts[0],
        "min_prob": float(parts[1]),
        "direction": parts[2],
        "min_risk_pct": float(parts[3]),
        "symbol": parts[4],
    }


def apply_gate(frame: pd.DataFrame, gate: dict[str, Any]) -> pd.DataFrame:
    out = frame[pd.to_numeric(frame["breaker_prob"], errors="coerce") >= gate["min_prob"]].copy()
    if gate["direction"] != "ALL":
        out = out[out["direction"].astype(str) == gate["direction"]].copy()
    if gate["symbol"] != "ALL":
        out = out[out["symbol"].astype(str) == gate["symbol"]].copy()
    if gate["min_risk_pct"] > 0:
        out = out[pd.to_numeric(out["risk_pct"], errors="coerce") >= gate["min_risk_pct"]].copy()
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cost sensitivity for probability/risk/symbol gates.")
    parser.add_argument("file", type=Path)
    parser.add_argument(
        "--gates",
        required=True,
        help="Comma-separated gates as name:min_prob:direction:min_risk_pct:symbol.",
    )
    parser.add_argument("--fees", default="1,3,5,8,10,12")
    parser.add_argument("--out", type=Path, default=Path("scripts/gate_cost_sensitivity.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = pd.read_csv(args.file)
    gates = [parse_gate(item) for item in args.gates.split(",") if item.strip()]
    fees = parse_float_list(args.fees)

    rows: list[dict[str, Any]] = []
    for fee in fees:
        frame = enrich(raw, fee)
        folds = sorted(frame["fold"].dropna().unique()) if "fold" in frame.columns else []
        for gate in gates:
            selected = apply_gate(frame, gate)
            aggregate = {
                "fee_bps_side": fee,
                "gate": gate["name"],
                "fold": "aggregate",
                "min_prob": gate["min_prob"],
                "direction": gate["direction"],
                "min_risk_pct": gate["min_risk_pct"],
                "symbol": gate["symbol"],
                **trade_metrics(selected),
            }
            rows.append(aggregate)
            for fold in folds:
                fold_frame = selected[selected["fold"] == fold]
                rows.append({
                    "fee_bps_side": fee,
                    "gate": gate["name"],
                    "fold": int(fold),
                    "min_prob": gate["min_prob"],
                    "direction": gate["direction"],
                    "min_risk_pct": gate["min_risk_pct"],
                    "symbol": gate["symbol"],
                    **trade_metrics(fold_frame),
                })

    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))
    print(f"\nSaved {len(out)} rows to {args.out}")


if __name__ == "__main__":
    main()
