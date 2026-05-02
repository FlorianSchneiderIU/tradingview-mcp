from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


NONLEAK_FEATURES = [
    "direction_long",
    "entry_risk_pct",
    "entry_risk_atr",
    "risk_to_zone_width",
    "risk_to_ob_width",
    "target_distance_atr",
    "zone_width_pct_signal",
    "zone_width_atr_signal",
    "zone_age_hours_sweep",
    "zone_age_hours_signal",
    "sweep_penetration_frac",
    "sweep_reclaim_pos",
    "sweep_range_atr",
    "sweep_same_bar_reaction_atr",
    "sweep_same_bar_close_reaction_atr",
    "sweep_same_bar_adverse_atr",
    "sweep_reclaim_body_atr",
    "sweep_vol_mult",
    "choch_wait_bars",
    "signal_after_choch_bars",
    "ob_width_atr_signal",
    "entry_to_ob_mid_atr",
    "entry_to_zone_mid_atr",
    "stop_beyond_zone_atr",
    "entry_vs_signal_close_atr",
    "signal_close_vs_zone_atr",
    "ret_1h_dir",
    "ret_4h_dir",
    "ret_24h_dir",
    "range_1h_pct",
    "range_4h_pct",
    "bias_4h_aligned",
    "bias_1d_aligned",
    "htf_sma50_aligned",
    "first4_ret_dir",
    "first4_range_pos",
    "prev_day_ret_dir",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]


def zone_key(symbol: str, direction: str, time_value: pd.Timestamp, top: float, bottom: float) -> str:
    return f"{symbol}|{direction}|{pd.Timestamp(time_value).isoformat()}|{top:.8f}|{bottom:.8f}"


def profit_factor(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    wins = float(frame.loc[frame["r_multiple"] > 0, "r_multiple"].sum())
    losses = abs(float(frame.loc[frame["r_multiple"] <= 0, "r_multiple"].sum()))
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return wins / losses


def metrics(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "net_r": 0.0, "avg_r": 0.0, "pf": 0.0}
    wins = int((frame["r_multiple"] > 0).sum())
    losses = int(len(frame) - wins)
    return {
        "trades": int(len(frame)),
        "wins": wins,
        "losses": losses,
        "win_rate": round(100.0 * wins / len(frame), 2),
        "net_r": round(float(frame["r_multiple"].sum()), 3),
        "avg_r": round(float(frame["r_multiple"].mean()), 3),
        "pf": round(profit_factor(frame), 3),
    }


def make_full_trade_frame(trades_path: Path, zone_path: Path) -> pd.DataFrame:
    trades = pd.read_csv(
        trades_path,
        parse_dates=["entry_time", "exit_time", "sweep_time", "choch_time", "signal_time"],
    )
    zones = pd.read_csv(zone_path)
    trades["event_key"] = [
        zone_key(row.symbol, row.direction, row.sweep_time, row.zone_top, row.zone_bottom)
        for row in trades.itertuples()
    ]
    zone_cols = ["event_key", "hold_prob", "hold_label", "future_r", "outcome", "mfe_r", "mae_r"]
    merged = trades.merge(zones[zone_cols], on="event_key", how="left")
    merged = merged.rename(columns={"hold_prob": "zone_hold_prob", "hold_label": "zone_hold_label", "future_r": "zone_future_r"})
    return merged


def threshold_diagnostics(oos: pd.DataFrame, threshold: float) -> dict:
    kept = oos[oos["zone_hold_prob"] >= threshold]
    rejected = oos[oos["zone_hold_prob"] < threshold]
    winners = oos[oos["r_multiple"] > 0]
    losses = oos[oos["r_multiple"] <= 0]
    false_negatives = rejected[rejected["r_multiple"] > 0]
    false_positives = kept[kept["r_multiple"] <= 0]
    return {
        "threshold": threshold,
        **{f"kept_{k}": v for k, v in metrics(kept).items()},
        "winner_recall": round(100.0 * len(kept[kept["r_multiple"] > 0]) / len(winners), 2) if len(winners) else 0.0,
        "loss_rejection": round(100.0 * len(rejected[rejected["r_multiple"] <= 0]) / len(losses), 2) if len(losses) else 0.0,
        "false_negative_winners": int(len(false_negatives)),
        "false_negative_net_r": round(float(false_negatives["r_multiple"].sum()), 3),
        "false_positive_losses": int(len(false_positives)),
        "false_positive_net_r": round(float(false_positives["r_multiple"].sum()), 3),
    }


def probability_bins(frame: pd.DataFrame, prob_col: str) -> pd.DataFrame:
    bins = [0.0, 0.35, 0.45, 0.50, 0.55, 0.60, 0.65, 1.0]
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        part = frame[(frame[prob_col] >= lo) & (frame[prob_col] < hi if hi < 1.0 else frame[prob_col] <= hi)]
        rows.append({"prob_col": prob_col, "bin": f"{lo:.2f}-{hi:.2f}", **metrics(part)})
    return pd.DataFrame(rows)


def group_feature_summary(oos: pd.DataFrame, threshold: float) -> pd.DataFrame:
    frame = oos.copy()
    frame["group"] = "rejected_loss"
    frame.loc[(frame["zone_hold_prob"] >= threshold) & (frame["r_multiple"] > 0), "group"] = "kept_win"
    frame.loc[(frame["zone_hold_prob"] >= threshold) & (frame["r_multiple"] <= 0), "group"] = "kept_loss"
    frame.loc[(frame["zone_hold_prob"] < threshold) & (frame["r_multiple"] > 0), "group"] = "rejected_win"

    rows = []
    for feature in NONLEAK_FEATURES:
        if feature not in frame.columns:
            continue
        row = {"feature": feature}
        for group_name, group in frame.groupby("group"):
            row[f"{group_name}_median"] = round(float(group[feature].median()), 4) if len(group) else math.nan
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["rej_win_vs_rej_loss_abs_delta"] = (
        out.get("rejected_win_median", math.nan) - out.get("rejected_loss_median", math.nan)
    ).abs()
    return out.sort_values("rej_win_vs_rej_loss_abs_delta", ascending=False)


def rescue_gate_search(oos: pd.DataFrame, base_threshold: float, min_added_trades: int = 3) -> pd.DataFrame:
    base_keep = oos["zone_hold_prob"] >= base_threshold
    base_metrics = metrics(oos[base_keep])
    rows = []
    for feature in NONLEAK_FEATURES:
        if feature not in oos.columns:
            continue
        values = oos[feature].dropna()
        if values.nunique() < 4:
            continue
        quantiles = sorted(set(float(q) for q in values.quantile([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]).dropna()))
        for cut in quantiles:
            for op in [">=", "<="]:
                condition = oos[feature] >= cut if op == ">=" else oos[feature] <= cut
                added = oos[(~base_keep) & condition]
                if len(added) < min_added_trades:
                    continue
                keep = base_keep | condition
                kept = oos[keep]
                kept_metrics = metrics(kept)
                added_metrics = metrics(added)
                rows.append({
                    "feature": feature,
                    "op": op,
                    "cut": round(cut, 5),
                    "base_trades": base_metrics["trades"],
                    "base_net_r": base_metrics["net_r"],
                    "kept_trades": kept_metrics["trades"],
                    "kept_wins": kept_metrics["wins"],
                    "kept_net_r": kept_metrics["net_r"],
                    "kept_pf": kept_metrics["pf"],
                    "added_trades": added_metrics["trades"],
                    "added_wins": added_metrics["wins"],
                    "added_losses": added_metrics["losses"],
                    "added_net_r": added_metrics["net_r"],
                    "added_pf": added_metrics["pf"],
                })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["net_r_improvement"] = out["kept_net_r"] - out["base_net_r"]
    return out.sort_values(["kept_net_r", "net_r_improvement", "kept_pf"], ascending=False)


def apply_rescue_gate(frame: pd.DataFrame, base_threshold: float, feature: str, op: str, cut: float) -> pd.DataFrame:
    base_keep = frame["zone_hold_prob"] >= base_threshold
    condition = frame[feature] >= cut if op == ">=" else frame[feature] <= cut
    return frame[base_keep | condition]


def train_selected_rescue_eval(train: pd.DataFrame, oos: pd.DataFrame, base_threshold: float) -> pd.DataFrame:
    train_search = rescue_gate_search(train, base_threshold, min_added_trades=8)
    if train_search.empty:
        return train_search

    rows = []
    for candidate in train_search.head(40).itertuples(index=False):
        train_kept = apply_rescue_gate(train, base_threshold, candidate.feature, candidate.op, candidate.cut)
        oos_kept = apply_rescue_gate(oos, base_threshold, candidate.feature, candidate.op, candidate.cut)
        oos_added = oos[(oos["zone_hold_prob"] < base_threshold) & (
            oos[candidate.feature] >= candidate.cut if candidate.op == ">=" else oos[candidate.feature] <= candidate.cut
        )]
        train_m = metrics(train_kept)
        oos_m = metrics(oos_kept)
        added_m = metrics(oos_added)
        rows.append({
            "feature": candidate.feature,
            "op": candidate.op,
            "cut": candidate.cut,
            "train_trades": train_m["trades"],
            "train_net_r": train_m["net_r"],
            "train_pf": train_m["pf"],
            "oos_trades": oos_m["trades"],
            "oos_wins": oos_m["wins"],
            "oos_net_r": oos_m["net_r"],
            "oos_pf": oos_m["pf"],
            "oos_added_trades": added_m["trades"],
            "oos_added_wins": added_m["wins"],
            "oos_added_net_r": added_m["net_r"],
            "oos_added_pf": added_m["pf"],
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["oos_net_r", "oos_pf", "train_net_r"], ascending=False)


def stage_two_diagnostics(two_stage_path: Path, split: pd.Timestamp, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    two_stage = pd.read_csv(two_stage_path, parse_dates=["entry_time", "exit_time", "sweep_time", "signal_time"])
    oos = two_stage[two_stage["entry_time"] >= split].copy()
    if "trade_win_prob" not in oos.columns:
        return pd.DataFrame(), pd.DataFrame()
    rows = []
    kept = oos[oos["trade_win_prob"] >= threshold]
    rejected = oos[oos["trade_win_prob"] < threshold]
    rows.append({"stage": "stage2_kept", **metrics(kept)})
    rows.append({"stage": "stage2_rejected", **metrics(rejected)})
    rows.append({"stage": "stage2_rejected_winners", **metrics(rejected[rejected["r_multiple"] > 0])})
    return pd.DataFrame(rows), probability_bins(oos, "trade_win_prob")


def main() -> None:
    parser = argparse.ArgumentParser(description="Study ML false negatives / rejected winners.")
    parser.add_argument("--trades", type=Path, default=Path("scripts/trade_outcome_dataset.csv"))
    parser.add_argument("--zones", type=Path, default=Path("scripts/zone_hold_dataset_mbq.csv"))
    parser.add_argument("--two-stage", type=Path, default=Path("scripts/trade_outcome_dataset_2stage.csv"))
    parser.add_argument("--split", default="2025-04-20")
    parser.add_argument("--zone-threshold", type=float, default=0.55)
    parser.add_argument("--trade-threshold", type=float, default=0.60)
    parser.add_argument("--out-prefix", type=Path, default=Path("scripts/ml_rejection_study"))
    args = parser.parse_args()

    split = pd.Timestamp(args.split).tz_localize("UTC") if pd.Timestamp(args.split).tzinfo is None else pd.Timestamp(args.split)
    full = make_full_trade_frame(args.trades, args.zones)
    oos = full[full["entry_time"] >= split].copy()
    train = full[full["entry_time"] < split].copy()

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    threshold_rows = pd.DataFrame([threshold_diagnostics(oos, th) for th in [0.35, 0.45, 0.50, 0.55, 0.60, 0.65]])
    zone_bins = probability_bins(oos, "zone_hold_prob")
    group_summary = group_feature_summary(oos, args.zone_threshold)
    rescue = rescue_gate_search(oos, args.zone_threshold)
    train_rescue_eval = train_selected_rescue_eval(train, oos, args.zone_threshold)
    stage2_summary, stage2_bins = stage_two_diagnostics(args.two_stage, split, args.trade_threshold)

    rejected_winners = oos[(oos["zone_hold_prob"] < args.zone_threshold) & (oos["r_multiple"] > 0)].sort_values("r_multiple", ascending=False)
    kept_losers = oos[(oos["zone_hold_prob"] >= args.zone_threshold) & (oos["r_multiple"] <= 0)].sort_values("r_multiple")

    outputs = {
        "thresholds": threshold_rows,
        "zone_bins": zone_bins,
        "feature_groups": group_summary,
        "rescue_gates": rescue,
        "train_selected_rescue_eval": train_rescue_eval,
        "stage2_summary": stage2_summary,
        "stage2_bins": stage2_bins,
        "rejected_winners": rejected_winners,
        "kept_losers": kept_losers,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.out_prefix.with_name(f"{args.out_prefix.name}_{name}.csv"), index=False)

    print(f"Train trades: {len(train)} | OOS trades: {len(oos)}")
    print("\nStage-one zone probability diagnostics:")
    print(threshold_rows.to_string(index=False))
    print("\nZone probability bins:")
    print(zone_bins.to_string(index=False))
    print(f"\nRejected OOS winners at zone p < {args.zone_threshold}:")
    display_cols = [
        "symbol", "entry_time", "direction", "r_multiple", "zone_hold_prob", "zone_hold_label",
        "entry_risk_pct", "sweep_reclaim_body_atr", "sweep_reclaim_pos", "ret_4h_dir",
        "bias_4h_aligned", "prev_day_ret_dir", "hour_sin",
    ]
    print(rejected_winners[display_cols].head(20).to_string(index=False))
    print(f"\nKept OOS losses at zone p >= {args.zone_threshold}:")
    print(kept_losers[display_cols].head(20).to_string(index=False))
    print("\nTop feature median deltas: rejected winners vs rejected losses")
    print(group_summary.head(14).to_string(index=False))
    print("\nTop simple rescue gates:")
    print(rescue.head(14).to_string(index=False))
    print("\nTrain-selected rescue gates evaluated on OOS:")
    print(train_rescue_eval.head(14).to_string(index=False))
    print("\nStage-two diagnostics:")
    print(stage2_summary.to_string(index=False))
    print("\nStage-two probability bins:")
    print(stage2_bins.to_string(index=False))
    print(f"\nSaved study CSVs with prefix {args.out_prefix}")


if __name__ == "__main__":
    main()
