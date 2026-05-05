from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def profit_factor(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    wins = float(frame.loc[frame["r_multiple"] > 0, "r_multiple"].sum())
    losses = abs(float(frame.loc[frame["r_multiple"] <= 0, "r_multiple"].sum()))
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return wins / losses


def max_drawdown_r(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    ordered = frame.sort_values("exit_time")
    equity = ordered["r_multiple"].astype(float).cumsum()
    drawdown = equity - equity.cummax()
    return round(float(drawdown.min()), 3)


def trades_per_week(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    start = pd.Timestamp(frame["entry_time"].min())
    end = pd.Timestamp(frame["entry_time"].max())
    weeks = max((end - start).total_seconds() / (7 * 24 * 3600), 1e-9)
    return float(len(frame) / weeks)


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "net_r": 0.0,
            "avg_r": 0.0,
            "max_dd_r": 0.0,
            "trades_per_week": 0.0,
        }
    wins = int((frame["r_multiple"] > 0).sum())
    total = int(len(frame))
    return {
        "trades": total,
        "wins": wins,
        "losses": total - wins,
        "win_rate": round(100.0 * wins / total, 2),
        "profit_factor": round(profit_factor(frame), 3),
        "net_r": round(float(frame["r_multiple"].sum()), 3),
        "avg_r": round(float(frame["r_multiple"].mean()), 3),
        "max_dd_r": max_drawdown_r(frame),
        "trades_per_week": round(trades_per_week(frame), 3),
    }


def bucket_channel_pos(series: pd.Series) -> pd.Categorical:
    bins = [-np.inf, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, np.inf]
    labels = ["<0", "0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0", ">1"]
    return pd.cut(series.astype(float), bins=bins, labels=labels)


def bucket_quintile(series: pd.Series, prefix: str) -> pd.Series:
    clean = series.astype(float)
    ranks = clean.rank(method="average", pct=True)
    bins = pd.cut(ranks, bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], include_lowest=True, labels=False)
    out = bins.map(lambda idx: f"{prefix}_q{int(idx) + 1}" if pd.notna(idx) else "nan")
    return out.astype(str)


def probability_bins(frame: pd.DataFrame, prob_col: str) -> pd.DataFrame:
    bins = [0.0, 0.35, 0.45, 0.50, 0.55, 0.60, 0.65, 1.0]
    rows: list[dict[str, float | str]] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi < 1.0:
            part = frame[(frame[prob_col] >= lo) & (frame[prob_col] < hi)]
        else:
            part = frame[(frame[prob_col] >= lo) & (frame[prob_col] <= hi)]
        rows.append({"bin": f"{lo:.2f}-{hi:.2f}", **metrics(part)})
    return pd.DataFrame(rows)


def grouped_metrics(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for value, part in frame.groupby(group_col, dropna=False):
        row = {"group": str(value), **metrics(part)}
        # Per-group entry windows can be tiny and make cadence explode; keep cadence at dataset level only.
        row["trades_per_week"] = math.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("net_r", ascending=False)


def evaluate_gates(
    frame: pd.DataFrame,
    *,
    base_mask: pd.Series,
    min_trades: int,
) -> pd.DataFrame:
    candidates: list[tuple[str, pd.Series]] = []

    if "opp_target_rr_proxy" in frame.columns:
        for cut in [1.0, 1.5, 2.0, 2.5, 3.0]:
            candidates.append((f"opp_rr_proxy>={cut:g}", frame["opp_target_rr_proxy"] >= cut))

    for tf in ["1h", "4h", "1d"]:
        pos_col = f"bfm_signal_{tf}_channel_pos"
        width_col = f"bfm_signal_{tf}_channel_width_atr"
        align_col = f"bfm_signal_{tf}_trend_aligned"
        if pos_col in frame.columns:
            pos = frame[pos_col].astype(float)
            candidates.extend([
                (f"{tf}_pos_0_0.4", (pos >= 0.0) & (pos <= 0.4)),
                (f"{tf}_pos_0.4_0.8", (pos > 0.4) & (pos <= 0.8)),
                (f"{tf}_pos_outside", (pos < 0.0) | (pos > 1.0)),
            ])
        if width_col in frame.columns:
            width = frame[width_col].astype(float)
            q50 = width.quantile(0.5)
            q75 = width.quantile(0.75)
            candidates.extend([
                (f"{tf}_width_ge_q50", width >= q50),
                (f"{tf}_width_ge_q75", width >= q75),
            ])
        if align_col in frame.columns:
            align = frame[align_col].astype(float)
            candidates.extend([
                (f"{tf}_trend_aligned_pos", align > 0),
                (f"{tf}_trend_aligned_nonneg", align >= 0),
                (f"{tf}_trend_counter", align < 0),
            ])

    rows: list[dict[str, float | str]] = []
    base_frame = frame[base_mask]
    base = metrics(base_frame)
    for gate_name, gate_mask in candidates:
        keep = frame[base_mask & gate_mask.fillna(False)]
        if len(keep) < min_trades:
            continue
        m = metrics(keep)
        rows.append(
            {
                "gate": gate_name,
                "kept_trades": m["trades"],
                "kept_win_rate": m["win_rate"],
                "kept_profit_factor": m["profit_factor"],
                "kept_net_r": m["net_r"],
                "kept_avg_r": m["avg_r"],
                "delta_pf_vs_base": round(float(m["profit_factor"] - base["profit_factor"]), 3),
                "delta_net_r_vs_base": round(float(m["net_r"] - base["net_r"]), 3),
                "keep_pct_of_base": round(100.0 * len(keep) / max(len(base_frame), 1), 2),
            }
        )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.sort_values(["kept_profit_factor", "kept_net_r", "kept_trades"], ascending=[False, False, False])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-mortem Turtle Soup BFM RR behavior and target headroom.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("scripts/trade_outcome_dataset_core3_1h_bfm_line_channel_ordered_probe.csv"),
    )
    parser.add_argument(
        "--zone-dataset",
        type=Path,
        default=Path("scripts/zone_hold_dataset_core3_1h_bfm_rf_probe_v2.csv"),
    )
    parser.add_argument("--split", default="2025-04-20")
    parser.add_argument("--prob-col", default="trade_win_prob")
    parser.add_argument("--base-threshold", type=float, default=0.50)
    parser.add_argument("--min-gate-trades", type=int, default=20)
    parser.add_argument("--out-prefix", type=Path, default=Path("scripts/turtle_soup_bfm_rr_postmortem"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        raise SystemExit(f"Dataset not found: {args.dataset}")

    frame = pd.read_csv(args.dataset)
    for col in ["entry_time", "exit_time", "sweep_time", "signal_time"]:
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col], utc=True, errors="coerce")

    if args.prob_col not in frame.columns:
        raise SystemExit(f"Missing probability column {args.prob_col!r} in {args.dataset}")

    split = pd.Timestamp(args.split, tz="UTC") if pd.Timestamp(args.split).tzinfo is None else pd.Timestamp(args.split)
    frame = frame.sort_values(["entry_time", "symbol"]).reset_index(drop=True)
    train = frame[frame["entry_time"] < split].copy()
    oos = frame[frame["entry_time"] >= split].copy()

    oos_keep = oos[oos[args.prob_col] >= args.base_threshold].copy()
    oos_reject = oos[oos[args.prob_col] < args.base_threshold].copy()

    baseline = pd.DataFrame(
        [
            {"window": "train_all", **metrics(train)},
            {"window": "oos_all", **metrics(oos)},
            {"window": f"oos_keep_p_ge_{args.base_threshold:.2f}", **metrics(oos_keep)},
            {"window": f"oos_reject_p_lt_{args.base_threshold:.2f}", **metrics(oos_reject)},
        ]
    )

    oos_prob_bins = probability_bins(oos, args.prob_col)

    channel_rows: list[pd.DataFrame] = []
    for tf in ["1h", "4h", "1d"]:
        pos_col = f"bfm_signal_{tf}_channel_pos"
        width_col = f"bfm_signal_{tf}_channel_width_atr"
        slope_col = f"bfm_signal_{tf}_trend_slope_atr"
        if pos_col not in oos.columns:
            continue
        work = oos.copy()
        work[f"{tf}_pos_bucket"] = bucket_channel_pos(work[pos_col])
        if width_col in work.columns:
            work[f"{tf}_width_bucket"] = bucket_quintile(work[width_col], f"{tf}_width")
        if slope_col in work.columns:
            work[f"{tf}_trend_bucket"] = pd.cut(
                work[slope_col].astype(float),
                bins=[-np.inf, -0.003, -0.001, 0.001, 0.003, np.inf],
                labels=["strong_down", "down", "flat", "up", "strong_up"],
            )
        for group_col in [f"{tf}_pos_bucket", f"{tf}_width_bucket", f"{tf}_trend_bucket"]:
            if group_col not in work.columns:
                continue
            sliced = grouped_metrics(work[work[args.prob_col] >= args.base_threshold], group_col)
            sliced.insert(0, "slice", group_col)
            sliced.insert(1, "timeframe", tf)
            channel_rows.append(sliced)
    channel_slices = pd.concat(channel_rows, ignore_index=True) if channel_rows else pd.DataFrame()

    feasibility = pd.DataFrame()
    gate_experiments = pd.DataFrame()
    if {"entry_risk_atr", "bfm_signal_opp_close_dist_atr"}.issubset(oos.columns):
        work = oos.copy()
        risk = work["entry_risk_atr"].astype(float)
        opp_atr = work["bfm_signal_opp_close_dist_atr"].astype(float)
        work["opp_target_rr_proxy"] = np.where(risk > 0, opp_atr / risk, np.nan)
        work["opp_target_rr_bucket"] = pd.cut(
            work["opp_target_rr_proxy"],
            bins=[-np.inf, 0.75, 1.0, 1.5, 2.0, 3.0, np.inf],
            labels=["<0.75", "0.75-1.0", "1.0-1.5", "1.5-2.0", "2.0-3.0", ">=3.0"],
        )
        feasibility = grouped_metrics(work[work[args.prob_col] >= args.base_threshold], "opp_target_rr_bucket")
        feasibility.insert(0, "slice", "opp_target_rr_bucket")
        gate_experiments = evaluate_gates(
            work,
            base_mask=work[args.prob_col] >= args.base_threshold,
            min_trades=int(args.min_gate_trades),
        )

    exit_ceiling = pd.DataFrame()
    if args.zone_dataset.exists() and "event_key" in oos.columns:
        zone = pd.read_csv(args.zone_dataset)
        for col in ["time", "close_time"]:
            if col in zone.columns:
                zone[col] = pd.to_datetime(zone[col], utc=True, errors="coerce")
        zone_cols = [
            "event_key",
            "hold_prob",
            "hold_label",
            "future_r",
            "mfe_r",
            "mae_r",
            "bars_to_outcome",
            "outcome",
        ]
        available = [col for col in zone_cols if col in zone.columns]
        merged = oos.merge(zone[available], on="event_key", how="left")
        merged = merged[merged[args.prob_col] >= args.base_threshold].copy()
        if "mfe_r" in merged.columns:
            merged["headroom_r_to_mfe"] = merged["mfe_r"].astype(float) - merged["r_multiple"].astype(float)
            merged["mfe_bucket"] = pd.cut(
                merged["mfe_r"].astype(float),
                bins=[-np.inf, 0.5, 1.0, 1.5, 2.0, 3.0, np.inf],
                labels=["<0.5", "0.5-1.0", "1.0-1.5", "1.5-2.0", "2.0-3.0", ">=3.0"],
            )
            exit_ceiling = grouped_metrics(merged, "mfe_bucket")
            exit_ceiling.insert(0, "slice", "mfe_bucket")
            if "headroom_r_to_mfe" in merged.columns:
                headroom_summary = {
                    "slice": "headroom_summary",
                    "group": "all",
                    **metrics(merged),
                    "avg_headroom_r": round(float(merged["headroom_r_to_mfe"].mean()), 3),
                    "p50_headroom_r": round(float(merged["headroom_r_to_mfe"].median()), 3),
                    "pct_with_2r_mfe": round(float((merged["mfe_r"] >= 2.0).mean() * 100.0), 2),
                }
                exit_ceiling = pd.concat([pd.DataFrame([headroom_summary]), exit_ceiling], ignore_index=True)

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    baseline_path = args.out_prefix.with_name(f"{args.out_prefix.name}_baseline.csv")
    prob_bins_path = args.out_prefix.with_name(f"{args.out_prefix.name}_oos_prob_bins.csv")
    channel_path = args.out_prefix.with_name(f"{args.out_prefix.name}_channel_slices.csv")
    feas_path = args.out_prefix.with_name(f"{args.out_prefix.name}_opp_target_feasibility.csv")
    gate_path = args.out_prefix.with_name(f"{args.out_prefix.name}_gate_experiments.csv")
    ceiling_path = args.out_prefix.with_name(f"{args.out_prefix.name}_exit_ceiling.csv")

    baseline.to_csv(baseline_path, index=False)
    oos_prob_bins.to_csv(prob_bins_path, index=False)
    channel_slices.to_csv(channel_path, index=False)
    feasibility.to_csv(feas_path, index=False)
    gate_experiments.to_csv(gate_path, index=False)
    exit_ceiling.to_csv(ceiling_path, index=False)

    print("Baseline:")
    print(baseline.to_string(index=False))
    print("\nOOS probability bins:")
    print(oos_prob_bins.to_string(index=False))
    print(f"\nSaved: {baseline_path}")
    print(f"Saved: {prob_bins_path}")
    print(f"Saved: {channel_path}")
    print(f"Saved: {feas_path}")
    print(f"Saved: {gate_path}")
    print(f"Saved: {ceiling_path}")


if __name__ == "__main__":
    main()
