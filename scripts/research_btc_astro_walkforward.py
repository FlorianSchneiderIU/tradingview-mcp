from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research_btc_astro_cycle_timing import json_default  # noqa: E402
from scripts.research_btc_astro_meta_strategy import make_meta_model, trade_summary  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)


BASE_COLUMNS = [
    "lookback",
    "is_long",
    "sweep_depth_atr",
    "rejection_frac",
    "reclaim_atr",
    "body_dir_atr",
    "body_abs_atr",
    "range_atr",
    "close_from_extreme_atr",
    "ltf_rsi",
    "ltf_ema_slope_atr",
    "ltf_atr_ratio",
    "ltf_volume_ratio",
    "ltf_dist_ema_atr",
]
CALENDAR_COLUMNS = ["tod_sin", "tod_cos", "dow_sin", "dow_cos", "calendar_any_score", "calendar_dir_score"]
REAL_COLUMNS = ["real_any_score", "real_dir_score", "real_opp_score", "real_edge_score"]
PLACEBO_COLUMNS = ["placebo_any_score", "placebo_dir_score", "placebo_opp_score", "placebo_edge_score"]
FEATURE_SETS = {
    "price_only": BASE_COLUMNS,
    "price_calendar": BASE_COLUMNS + CALENDAR_COLUMNS,
    "price_real": BASE_COLUMNS + CALENDAR_COLUMNS + REAL_COLUMNS,
    "price_placebo": BASE_COLUMNS + CALENDAR_COLUMNS + PLACEBO_COLUMNS,
    "price_real_placebo": BASE_COLUMNS + CALENDAR_COLUMNS + REAL_COLUMNS + PLACEBO_COLUMNS,
}


def period_start(period: pd.Period) -> pd.Timestamp:
    return pd.Timestamp(period.to_timestamp()).tz_localize("UTC")


def selected_rows(frame: pd.DataFrame, threshold: float, score_col: str, rr_key: str) -> list[dict[str, object]]:
    selected = frame[frame[score_col] >= threshold].sort_values(["signal_idx", score_col], ascending=[True, False])
    rows: list[dict[str, object]] = []
    blocked_until = -1
    for row in selected.itertuples(index=False):
        signal_idx = int(getattr(row, "signal_idx"))
        if signal_idx <= blocked_until:
            continue
        rows.append(
            {
                "signal_time": getattr(row, "signal_time"),
                "result_r": float(getattr(row, f"result_r_{rr_key}")),
                "exit_reason": str(getattr(row, f"exit_reason_{rr_key}")),
                "mfe_r": float(getattr(row, f"mfe_r_{rr_key}")),
                "direction": str(getattr(row, "direction")),
                "score": float(getattr(row, score_col)),
            }
        )
        blocked_until = max(blocked_until, int(getattr(row, f"exit_idx_{rr_key}")))
    return rows


def summary_from_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    return trade_summary(
        [
            {
                "result_r": float(row["result_r"]),
                "exit_reason": str(row["exit_reason"]),
                "mfe_r": float(row["mfe_r"]),
            }
            for row in rows
        ]
    )


def select_threshold(validation: pd.DataFrame, rr_key: str, coverages: list[float], min_trades: int) -> tuple[float, float, dict[str, object]]:
    best: tuple[float, float, dict[str, object], float] | None = None
    for coverage in coverages:
        threshold = float(np.quantile(validation["score"], 1.0 - coverage))
        rows = selected_rows(validation, threshold, "score", rr_key)
        summary = summary_from_rows(rows)
        if int(summary["trades"]) < min_trades:
            continue
        objective = float(summary["net_r"]) / max(abs(float(summary["max_drawdown_r"])), 10.0) + 0.25 * float(summary["avg_r"])
        if best is None or objective > best[3]:
            best = (threshold, coverage, summary, objective)
    if best is None:
        threshold = float(np.quantile(validation["score"], 0.99))
        rows = selected_rows(validation, threshold, "score", rr_key)
        return threshold, 0.01, summary_from_rows(rows)
    return best[0], best[1], best[2]


def apply_monthly_loss_stop(rows: list[dict[str, object]], stop_r: float | None) -> list[dict[str, object]]:
    if stop_r is None:
        return rows
    out: list[dict[str, object]] = []
    by_month: dict[str, float] = {}
    for row in rows:
        month = str(pd.Timestamp(row["signal_time"]).to_period("M"))
        current = by_month.get(month, 0.0)
        if current <= -abs(stop_r):
            continue
        out.append(row)
        by_month[month] = current + float(row["result_r"])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Monthly walk-forward selector for BTC astro meta candidates.")
    parser.add_argument("--candidate-cache", type=Path, default=Path("scripts/.cache/astro_cycle/meta_candidates_12_10r.pkl"))
    parser.add_argument("--output-json", type=Path, default=Path("scripts/astro_walkforward_price_real_12m_10r_summary.json"))
    parser.add_argument("--monthly-csv", type=Path, default=Path("scripts/astro_walkforward_monthly_12_10r.csv"))
    parser.add_argument("--trades-csv", type=Path, default=Path("scripts/astro_walkforward_price_real_12m_10r_trades.csv"))
    parser.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default="price_real")
    parser.add_argument("--model", choices=["extra_trees"], default="extra_trees")
    parser.add_argument("--rr", type=float, default=10.0)
    parser.add_argument("--validation-months", type=int, default=12)
    parser.add_argument("--start-month", default="2025-01")
    parser.add_argument("--end-month", default="2026-05")
    parser.add_argument("--monthly-loss-stop-r", type=float, default=6.0)
    args = parser.parse_args()

    candidates = pd.read_pickle(args.candidate_cache)
    candidates["signal_time"] = pd.to_datetime(candidates["signal_time"], utc=True)
    rr_key = f"{args.rr:g}"
    columns = FEATURE_SETS[args.feature_set]
    coverages = [0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.035, 0.05]
    months = pd.period_range(args.start_month, args.end_month, freq="M")

    all_rows: list[dict[str, object]] = []
    monthly_rows: list[dict[str, object]] = []
    for period in months:
        start = period_start(period)
        end = period_start(period + 1)
        validation_start = start - DateOffset(months=args.validation_months)
        train = candidates[candidates["signal_time"] < validation_start].copy()
        validation = candidates[(candidates["signal_time"] >= validation_start) & (candidates["signal_time"] < start)].copy()
        test = candidates[(candidates["signal_time"] >= start) & (candidates["signal_time"] < end)].copy()
        if train.empty or validation.empty or test.empty:
            continue
        model = make_meta_model(args.model)
        model.fit(train[columns], (train[f"result_r_{rr_key}"] > 0.0).astype(int))
        validation["score"] = model.predict_proba(validation[columns])[:, 1]
        test["score"] = model.predict_proba(test[columns])[:, 1]
        threshold, coverage, validation_summary = select_threshold(validation, rr_key, coverages, min_trades=10)
        test_rows = selected_rows(test, threshold, "score", rr_key)
        for row in test_rows:
            row["month"] = str(period)
            row["threshold"] = threshold
            row["coverage"] = coverage
        all_rows.extend(test_rows)
        test_summary = summary_from_rows(test_rows)
        monthly_rows.append(
            {
                "month": str(period),
                "threshold": threshold,
                "coverage": coverage,
                **{f"validation_{key}": value for key, value in validation_summary.items()},
                **{f"test_{key}": value for key, value in test_summary.items()},
            }
        )

    risk_rows = apply_monthly_loss_stop(all_rows, args.monthly_loss_stop_r)
    summary = {
        "config": {
            "candidate_cache": args.candidate_cache,
            "feature_set": args.feature_set,
            "model": args.model,
            "rr": args.rr,
            "validation_months": args.validation_months,
            "start_month": args.start_month,
            "end_month": args.end_month,
            "monthly_loss_stop_r": args.monthly_loss_stop_r,
        },
        "raw": summary_from_rows(all_rows),
        "monthly_loss_stop": summary_from_rows(risk_rows),
        "positive_months": int(sum(row["test_net_r"] > 0 for row in monthly_rows)),
        "months": len(monthly_rows),
    }
    args.output_json.write_text(json.dumps(summary, indent=2, default=json_default), encoding="utf-8")
    pd.DataFrame(monthly_rows).to_csv(args.monthly_csv, index=False)
    pd.DataFrame(all_rows).to_csv(args.trades_csv, index=False)
    print(json.dumps(summary, indent=2, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
