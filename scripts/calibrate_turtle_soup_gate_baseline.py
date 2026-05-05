from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    RandomForestClassifier = None
    SimpleImputer = None
    LogisticRegression = None
    make_pipeline = None
    StandardScaler = None
    SKLEARN_AVAILABLE = False


FEATURE_COLUMNS = [
    "ret_1h_dir",
    "ret_4h_dir",
    "sweep_same_bar_adverse_atr",
    "sweep_same_bar_close_reaction_atr",
    "bfm_signal_4h_trend_aligned",
    "bfm_signal_1d_trend_aligned",
    "bfm_signal_1d_channel_pos",
    "bfm_signal_4h_channel_pos",
    "bfm_signal_4h_channel_width_atr",
]


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


def add_relative_metrics(part: pd.DataFrame, base: pd.DataFrame) -> dict[str, float]:
    base_wins = int((base["r_multiple"] > 0).sum())
    base_losses = int((base["r_multiple"] <= 0).sum())
    keep_wins = int((part["r_multiple"] > 0).sum())
    keep_losses = int((part["r_multiple"] <= 0).sum())
    return {
        "winner_recall_pct": round(100.0 * keep_wins / max(base_wins, 1), 2),
        "loss_rejection_pct": round(100.0 * (base_losses - keep_losses) / max(base_losses, 1), 2),
        "keep_pct_of_base": round(100.0 * len(part) / max(len(base), 1), 2),
    }


def fit_meta_model(train: pd.DataFrame, model_name: str):
    x = train[FEATURE_COLUMNS].astype(float)
    y = train["win_label"].astype(int)

    if model_name == "logreg":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2500, class_weight="balanced", C=0.1),
        ).fit(x, y)

    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=500,
            max_depth=4,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            random_state=17,
            n_jobs=1,
        ),
    ).fit(x, y)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate rule policies and optional meta-ML gate for Turtle Soup p>=0.50 cohort.")
    parser.add_argument("--dataset", type=Path, default=Path("scripts/trade_outcome_dataset_core3_1h_bfm_line_channel_ordered_probe.csv"))
    parser.add_argument("--split", default="2025-04-20")
    parser.add_argument("--prob-col", default="trade_win_prob")
    parser.add_argument("--base-threshold", type=float, default=0.50)
    parser.add_argument("--ret1h-floor", type=float, default=-0.8009)
    parser.add_argument("--ret4h-floor", type=float, default=-0.6539)
    parser.add_argument("--enable-meta-model", action="store_true")
    parser.add_argument("--meta-model", choices=["rf", "logreg"], default="rf")
    parser.add_argument("--meta-thresholds", default="0.45,0.50,0.55,0.60")
    parser.add_argument("--out-prefix", type=Path, default=Path("scripts/turtle_soup_gate_policy_baseline"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        raise SystemExit(f"Dataset not found: {args.dataset}")

    frame = pd.read_csv(args.dataset)
    for col in ["entry_time", "exit_time"]:
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col], utc=True, errors="coerce")

    if args.prob_col not in frame.columns:
        raise SystemExit(f"Missing probability column {args.prob_col!r} in {args.dataset}")

    split = pd.Timestamp(args.split, tz="UTC") if pd.Timestamp(args.split).tzinfo is None else pd.Timestamp(args.split)
    frame = frame.sort_values(["entry_time", "symbol"]).reset_index(drop=True)
    oos = frame[frame["entry_time"] >= split].copy()
    train = frame[frame["entry_time"] < split].copy()

    base_oos = oos[oos[args.prob_col] >= args.base_threshold].copy()
    base_train = train[train[args.prob_col] >= args.base_threshold].copy()

    if base_oos.empty:
        raise SystemExit("Base OOS cohort is empty. Adjust --base-threshold.")

    rule_4h_nonneg = base_oos["bfm_signal_4h_trend_aligned"].astype(float) >= 0.0
    rule_1d_nonneg = base_oos["bfm_signal_1d_trend_aligned"].astype(float) >= 0.0
    rule_1d_mid = (base_oos["bfm_signal_1d_channel_pos"].astype(float) >= 0.4) & (base_oos["bfm_signal_1d_channel_pos"].astype(float) <= 0.8)
    rule_ret1h = base_oos["ret_1h_dir"].astype(float) >= float(args.ret1h_floor)
    rule_ret4h = base_oos["ret_4h_dir"].astype(float) >= float(args.ret4h_floor)

    policies: list[tuple[str, pd.Series]] = [
        ("base_p_ge_0.50", base_oos[args.prob_col] >= -math.inf),
        ("aggressive_veto_4h_counter", rule_4h_nonneg),
        ("balanced_veto_1d_counter", rule_1d_nonneg),
        ("balanced_1d_counter_plus_ret1h_floor", rule_1d_nonneg & rule_ret1h),
        ("conservative_dual_trend_nonneg", rule_4h_nonneg & rule_1d_nonneg),
        ("precision_1d_mid_and_dual_trend", rule_1d_mid & rule_4h_nonneg & rule_1d_nonneg),
        ("momentum_guard_ret1h_and_ret4h", rule_ret1h & rule_ret4h),
    ]

    rows: list[dict[str, float | str]] = []
    selected_trades: list[pd.DataFrame] = []
    for policy_name, mask in policies:
        kept = base_oos[mask.fillna(False)].copy()
        row = {"policy": policy_name, **metrics(kept), **add_relative_metrics(kept, base_oos)}
        rows.append(row)
        if not kept.empty:
            kept = kept.copy()
            kept["policy"] = policy_name
            selected_trades.append(kept)

    summary = pd.DataFrame(rows).sort_values(["profit_factor", "net_r", "winner_recall_pct"], ascending=[False, False, False])

    meta_table = pd.DataFrame()
    meta_scored = pd.DataFrame()
    if args.enable_meta_model:
        if not SKLEARN_AVAILABLE:
            raise SystemExit("scikit-learn is not available. Install it in the active environment or run without --enable-meta-model.")
        missing = [col for col in FEATURE_COLUMNS if col not in frame.columns]
        if missing:
            raise SystemExit(f"Missing feature columns required for meta model: {missing}")
        if base_train["win_label"].nunique() < 2:
            raise SystemExit("Base train cohort has a single class; cannot train meta model.")

        model = fit_meta_model(base_train, args.meta_model)
        oos_scored = base_oos.copy()
        oos_scored["meta_gate_prob"] = model.predict_proba(oos_scored[FEATURE_COLUMNS].astype(float))[:, 1]

        thr_rows: list[dict[str, float | str]] = []
        for thr in [float(item.strip()) for item in str(args.meta_thresholds).split(",") if item.strip()]:
            kept = oos_scored[oos_scored["meta_gate_prob"] >= thr].copy()
            thr_rows.append({
                "policy": f"meta_{args.meta_model}_p_ge_{thr:.2f}",
                **metrics(kept),
                **add_relative_metrics(kept, base_oos),
            })
        meta_table = pd.DataFrame(thr_rows).sort_values(["profit_factor", "net_r", "winner_recall_pct"], ascending=[False, False, False])
        meta_scored = oos_scored.sort_values("meta_gate_prob", ascending=False)

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_prefix.with_name(f"{args.out_prefix.name}_summary.csv")
    trades_path = args.out_prefix.with_name(f"{args.out_prefix.name}_selected_trades.csv")
    summary.to_csv(summary_path, index=False)
    if selected_trades:
        pd.concat(selected_trades, ignore_index=True).to_csv(trades_path, index=False)
    else:
        pd.DataFrame().to_csv(trades_path, index=False)

    print("Rule policy summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {trades_path}")

    if args.enable_meta_model:
        meta_summary_path = args.out_prefix.with_name(f"{args.out_prefix.name}_meta_summary.csv")
        meta_scored_path = args.out_prefix.with_name(f"{args.out_prefix.name}_meta_scored_oos.csv")
        meta_table.to_csv(meta_summary_path, index=False)
        meta_scored.to_csv(meta_scored_path, index=False)
        print("\nMeta-model policy summary:")
        print(meta_table.to_string(index=False))
        print(f"\nSaved: {meta_summary_path}")
        print(f"Saved: {meta_scored_path}")


if __name__ == "__main__":
    main()
