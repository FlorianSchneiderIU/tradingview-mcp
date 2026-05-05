from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    ExtraTreesClassifier = None
    RandomForestClassifier = None
    SimpleImputer = None
    LogisticRegression = None
    make_pipeline = None
    StandardScaler = None
    SKLEARN_AVAILABLE = False


CORE_FEATURES = [
    "ret_1h_dir",
    "ret_4h_dir",
    "ret_24h_dir",
    "sweep_same_bar_adverse_atr",
    "sweep_same_bar_close_reaction_atr",
    "sweep_reclaim_pos",
    "entry_risk_atr",
    "zone_width_atr_signal",
    "entry_to_zone_mid_atr",
    "bfm_signal_4h_trend_aligned",
    "bfm_signal_1d_trend_aligned",
    "bfm_signal_4h_channel_pos",
    "bfm_signal_1d_channel_pos",
    "bfm_signal_4h_channel_width_atr",
    "bfm_signal_1d_channel_width_atr",
]

BFM_SIGNAL_FEATURES = [
    "bfm_signal_4h_trend_aligned",
    "bfm_signal_1d_trend_aligned",
    "bfm_signal_4h_channel_pos",
    "bfm_signal_1d_channel_pos",
    "bfm_signal_4h_channel_width_atr",
    "bfm_signal_1d_channel_width_atr",
    "bfm_signal_4h_width_slope_atr",
    "bfm_signal_1d_width_slope_atr",
    "bfm_signal_4h_widening",
    "bfm_signal_4h_closing",
    "bfm_signal_1d_widening",
    "bfm_signal_1d_closing",
]

RESCUE_STYLE_FEATURES = [
    "ret_1h_dir",
    "ret_4h_dir",
    "first4_ret_dir",
    "first4_range_pos",
    "prev_day_ret_dir",
    "sweep_same_bar_adverse_atr",
    "sweep_same_bar_close_reaction_atr",
    "sweep_reclaim_pos",
    "entry_to_ob_mid_atr",
    "stop_beyond_zone_atr",
    "bfm_signal_4h_trend_aligned",
    "bfm_signal_1d_trend_aligned",
]


@dataclass
class ExperimentSpec:
    model_name: str
    feature_set: str


def profit_factor(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    wins = float(frame.loc[frame["r_multiple"] > 0, "r_multiple"].sum())
    losses = abs(float(frame.loc[frame["r_multiple"] <= 0, "r_multiple"].sum()))
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return wins / losses


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
    }


def relative_metrics(kept: pd.DataFrame, base: pd.DataFrame) -> dict[str, float]:
    base_wins = int((base["r_multiple"] > 0).sum())
    base_losses = int((base["r_multiple"] <= 0).sum())
    kept_wins = int((kept["r_multiple"] > 0).sum())
    kept_losses = int((kept["r_multiple"] <= 0).sum())
    return {
        "winner_recall_pct": round(100.0 * kept_wins / max(base_wins, 1), 2),
        "loss_rejection_pct": round(100.0 * (base_losses - kept_losses) / max(base_losses, 1), 2),
        "keep_pct_of_base": round(100.0 * len(kept) / max(len(base), 1), 2),
    }


def make_candidate_id(
    scenario: str,
    symbol_specific: bool,
    model: str,
    feature_set: str,
    chosen_threshold: float,
) -> str:
    return f"{scenario}|sym={int(bool(symbol_specific))}|{model}|{feature_set}|thr={chosen_threshold:.4f}"


def build_robustness_views(
    leaderboard: pd.DataFrame,
    scored_oos: pd.DataFrame,
    top_k: int,
    threshold_stress: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if leaderboard.empty or scored_oos.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty

    top_candidates = leaderboard.head(max(1, int(top_k))).copy()
    stress_rows: list[dict[str, float | str]] = []
    by_symbol_rows: list[dict[str, float | str]] = []
    by_quarter_rows: list[dict[str, float | str]] = []

    for _, cand in top_candidates.iterrows():
        cid = str(cand["candidate_id"])
        scoped = scored_oos[scored_oos["candidate_id"] == cid].copy()
        if scoped.empty:
            continue

        symbol_specific = bool(cand["symbol_specific"])
        chosen_threshold = float(cand["chosen_threshold"])

        if symbol_specific:
            kept = scoped[scoped["meta_prob"] >= scoped["local_threshold"]].copy()
            m = metrics(kept)
            rel = relative_metrics(kept, scoped)
            stress_rows.append(
                {
                    "candidate_id": cid,
                    "mode": "local_symbol_threshold",
                    "applied_threshold": np.nan,
                    **m,
                    **rel,
                }
            )
        else:
            for thr in [
                max(0.0, chosen_threshold - threshold_stress),
                chosen_threshold,
                min(1.0, chosen_threshold + threshold_stress),
            ]:
                kept = scoped[scoped["meta_prob"] >= thr].copy()
                m = metrics(kept)
                rel = relative_metrics(kept, scoped)
                stress_rows.append(
                    {
                        "candidate_id": cid,
                        "mode": "global_threshold",
                        "applied_threshold": round(float(thr), 4),
                        **m,
                        **rel,
                    }
                )

            kept = scoped[scoped["meta_prob"] >= chosen_threshold].copy()

        if kept.empty:
            continue

        for symbol, grp in kept.groupby("symbol", dropna=False):
            by_symbol_rows.append(
                {
                    "candidate_id": cid,
                    "symbol": str(symbol),
                    **metrics(grp),
                }
            )

        kept = kept.copy()
        quarter_ts = pd.to_datetime(kept["entry_time"], utc=True, errors="coerce")
        kept["quarter"] = quarter_ts.dt.year.astype("Int64").astype(str) + "Q" + quarter_ts.dt.quarter.astype("Int64").astype(str)
        for quarter, grp in kept.groupby("quarter", dropna=False):
            by_quarter_rows.append(
                {
                    "candidate_id": cid,
                    "quarter": str(quarter),
                    **metrics(grp),
                }
            )

    return (
        top_candidates,
        pd.DataFrame(stress_rows),
        pd.DataFrame(by_symbol_rows),
        pd.DataFrame(by_quarter_rows),
    )


def build_model(name: str):
    if name == "logreg":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2500, class_weight="balanced", C=0.2),
        )
    if name == "rf_d3":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=500,
                max_depth=3,
                min_samples_leaf=8,
                class_weight="balanced_subsample",
                random_state=31,
                n_jobs=1,
            ),
        )
    if name == "rf_d5":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=600,
                max_depth=5,
                min_samples_leaf=6,
                class_weight="balanced_subsample",
                random_state=31,
                n_jobs=1,
            ),
        )
    if name == "et_d5":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=600,
                max_depth=5,
                min_samples_leaf=6,
                class_weight="balanced_subsample",
                random_state=31,
                n_jobs=1,
            ),
        )
    raise ValueError(f"Unknown model {name!r}")


def feature_sets(columns: list[str]) -> dict[str, list[str]]:
    out = {
        "core": [c for c in CORE_FEATURES if c in columns],
        "bfm_signal": [c for c in BFM_SIGNAL_FEATURES if c in columns],
        "rescue_style": [c for c in RESCUE_STYLE_FEATURES if c in columns],
    }
    # Add a wider set by excluding obvious non-causal or target columns.
    exclude_prefixes = (
        "symbol_",
    )
    exclude_exact = {
        "symbol",
        "direction",
        "entry_time",
        "exit_time",
        "sweep_time",
        "choch_time",
        "signal_time",
        "event_key",
        "exit_reason",
        "win_label",
        "r_multiple",
        "trade_win_prob",
    }
    wide = []
    for col in columns:
        if col in exclude_exact:
            continue
        if any(col.startswith(pref) for pref in exclude_prefixes):
            continue
        # Keep numeric-like predictors only.
        if col.endswith("_time"):
            continue
        wide.append(col)
    out["wide"] = wide
    return out


def evaluate_thresholds(
    scored: pd.DataFrame,
    base_ref: pd.DataFrame,
    thresholds: list[float],
) -> pd.DataFrame:
    rows = []
    for thr in thresholds:
        kept = scored[scored["meta_prob"] >= thr].copy()
        kept_wins = int((kept["r_multiple"] > 0).sum())
        kept_losses = int((kept["r_multiple"] <= 0).sum())
        rows.append({
            "threshold": thr,
            "kept_wins": kept_wins,
            "kept_losses": kept_losses,
            **metrics(kept),
            **relative_metrics(kept, base_ref),
        })
    return pd.DataFrame(rows)


def threshold_objective(
    row: pd.Series,
    *,
    min_trades: int,
    min_winner_recall: float,
    fp_cost: float,
    fn_cost: float,
    pf_weight: float,
    net_r_weight: float,
    loss_rej_weight: float,
) -> float:
    if int(row["trades"]) < min_trades:
        return -1e9
    if float(row["winner_recall_pct"]) < min_winner_recall:
        return -1e9
    pf = float(row["profit_factor"])
    net_r = float(row["net_r"])
    loss_rej = float(row["loss_rejection_pct"])
    winner_recall = float(row["winner_recall_pct"])
    kept_loss_pct = 100.0 - loss_rej
    dropped_winner_pct = 100.0 - winner_recall
    return (
        pf * pf_weight
        + net_r * net_r_weight
        + loss_rej * loss_rej_weight
        - fp_cost * kept_loss_pct
        - fn_cost * dropped_winner_pct
    )


def apply_rule_policy(frame: pd.DataFrame, policy: str, ret1h_floor: float) -> pd.Series:
    if policy == "none":
        return pd.Series(True, index=frame.index)

    rule_4h_nonneg = frame["bfm_signal_4h_trend_aligned"].astype(float) >= 0.0
    rule_1d_nonneg = frame["bfm_signal_1d_trend_aligned"].astype(float) >= 0.0
    rule_1d_mid = (frame["bfm_signal_1d_channel_pos"].astype(float) >= 0.4) & (frame["bfm_signal_1d_channel_pos"].astype(float) <= 0.8)
    rule_ret1h = frame["ret_1h_dir"].astype(float) >= ret1h_floor
    rule_ret4h = frame["ret_4h_dir"].astype(float) >= -0.6539

    if policy == "aggressive_veto_4h_counter":
        return rule_4h_nonneg
    if policy == "balanced_veto_1d_counter":
        return rule_1d_nonneg
    if policy == "balanced_1d_counter_plus_ret1h_floor":
        return rule_1d_nonneg & rule_ret1h
    if policy == "conservative_dual_trend_nonneg":
        return rule_4h_nonneg & rule_1d_nonneg
    if policy == "precision_1d_mid_and_dual_trend":
        return rule_1d_mid & rule_4h_nonneg & rule_1d_nonneg
    if policy == "momentum_guard_ret1h_and_ret4h":
        return rule_ret1h & rule_ret4h

    raise ValueError(f"Unknown rule policy {policy!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run richer ML experiments for Turtle Soup meta-gating.")
    parser.add_argument("--dataset", type=Path, default=Path("scripts/trade_outcome_dataset_core3_1h_bfm_line_channel_ordered_probe.csv"))
    parser.add_argument("--split", default="2025-04-20")
    parser.add_argument("--prob-col", default="trade_win_prob")
    parser.add_argument("--base-threshold", type=float, default=0.50)
    parser.add_argument("--val-start", default="2024-04-20")
    parser.add_argument("--models", default="logreg,rf_d3,rf_d5,et_d5")
    parser.add_argument("--feature-sets", default="core,bfm_signal,rescue_style,wide")
    parser.add_argument("--enable-symbol-specific", action="store_true")
    parser.add_argument("--enable-stacked-rule", action="store_true")
    parser.add_argument("--stack-rule-policy", default="balanced_1d_counter_plus_ret1h_floor")
    parser.add_argument("--stack-ret1h-floor", type=float, default=-0.8009)
    parser.add_argument("--min-trades", type=int, default=35)
    parser.add_argument("--min-winner-recall", type=float, default=70.0)
    parser.add_argument("--fp-cost", type=float, default=0.08, help="Penalty weight for keeping losers (false positives).")
    parser.add_argument("--fn-cost", type=float, default=0.12, help="Penalty weight for dropping winners (false negatives).")
    parser.add_argument("--pf-weight", type=float, default=10.0)
    parser.add_argument("--net-r-weight", type=float, default=0.5)
    parser.add_argument("--loss-rej-weight", type=float, default=0.1)
    parser.add_argument("--threshold-grid", default="0.40,0.45,0.50,0.55,0.60,0.65")
    parser.add_argument("--top-k-robustness", type=int, default=3)
    parser.add_argument("--threshold-stress", type=float, default=0.03)
    parser.add_argument("--out-prefix", type=Path, default=Path("scripts/turtle_soup_meta_ml_experiments"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not SKLEARN_AVAILABLE:
        raise SystemExit("scikit-learn is required for this experiment script.")
    if not args.dataset.exists():
        raise SystemExit(f"Dataset not found: {args.dataset}")

    df = pd.read_csv(args.dataset)
    for col in ["entry_time", "exit_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    split = pd.Timestamp(args.split, tz="UTC") if pd.Timestamp(args.split).tzinfo is None else pd.Timestamp(args.split)
    val_start = pd.Timestamp(args.val_start, tz="UTC") if pd.Timestamp(args.val_start).tzinfo is None else pd.Timestamp(args.val_start)

    train_all = df[df["entry_time"] < split].copy()
    oos_all = df[df["entry_time"] >= split].copy()
    base_train = train_all[train_all[args.prob_col] >= args.base_threshold].copy()
    base_oos = oos_all[oos_all[args.prob_col] >= args.base_threshold].copy()

    feat_map = feature_sets(df.columns.tolist())
    models = [m.strip() for m in str(args.models).split(",") if m.strip()]
    set_names = [s.strip() for s in str(args.feature_sets).split(",") if s.strip()]
    threshold_grid = [float(x.strip()) for x in str(args.threshold_grid).split(",") if x.strip()]

    scenarios: list[tuple[str, pd.DataFrame, pd.DataFrame, bool]] = [
        ("global_base", base_train, base_oos, False),
    ]
    if args.enable_symbol_specific:
        scenarios.append(("symbol_base", base_train, base_oos, True))

    if args.enable_stacked_rule:
        train_mask = apply_rule_policy(base_train, args.stack_rule_policy, float(args.stack_ret1h_floor))
        oos_mask = apply_rule_policy(base_oos, args.stack_rule_policy, float(args.stack_ret1h_floor))
        stack_train = base_train[train_mask.fillna(False)].copy()
        stack_oos = base_oos[oos_mask.fillna(False)].copy()
        scenarios.append((f"global_stacked_{args.stack_rule_policy}", stack_train, stack_oos, False))
        if args.enable_symbol_specific:
            scenarios.append((f"symbol_stacked_{args.stack_rule_policy}", stack_train, stack_oos, True))

    leaderboard_rows: list[dict[str, float | str]] = []
    threshold_rows: list[dict[str, float | str]] = []
    scored_oos_rows: list[pd.DataFrame] = []
    scored_oos_all_rows: list[pd.DataFrame] = []

    for scenario_name, scenario_train, scenario_oos, symbol_specific in scenarios:
        if scenario_train.empty or scenario_oos.empty:
            continue
        fit_train = scenario_train[scenario_train["entry_time"] < val_start].copy()
        val = scenario_train[scenario_train["entry_time"] >= val_start].copy()
        if fit_train.empty or val.empty:
            continue

        for model_name in models:
            for set_name in set_names:
                cols = feat_map.get(set_name, [])
                if len(cols) < 5:
                    continue

                if not symbol_specific:
                    if fit_train["win_label"].nunique() < 2:
                        continue
                    estimator = build_model(model_name)
                    estimator.fit(fit_train[cols].astype(float), fit_train["win_label"].astype(int))

                    val_scored = val.copy()
                    val_scored["meta_prob"] = estimator.predict_proba(val_scored[cols].astype(float))[:, 1]
                    oos_scored = scenario_oos.copy()
                    oos_scored["meta_prob"] = estimator.predict_proba(oos_scored[cols].astype(float))[:, 1]

                    q_thresholds = [float(val_scored["meta_prob"].quantile(q)) for q in [0.55, 0.65, 0.75, 0.85]]
                    thresholds = sorted(set(threshold_grid + q_thresholds))
                    val_table = evaluate_thresholds(val_scored, val, thresholds)
                    val_table["objective"] = val_table.apply(
                        threshold_objective,
                        axis=1,
                        min_trades=int(args.min_trades),
                        min_winner_recall=float(args.min_winner_recall),
                        fp_cost=float(args.fp_cost),
                        fn_cost=float(args.fn_cost),
                        pf_weight=float(args.pf_weight),
                        net_r_weight=float(args.net_r_weight),
                        loss_rej_weight=float(args.loss_rej_weight),
                    )

                    best_val = val_table.sort_values(["objective", "profit_factor", "net_r"], ascending=[False, False, False]).iloc[0]
                    chosen_thr = float(best_val["threshold"])
                    oos_kept = oos_scored[oos_scored["meta_prob"] >= chosen_thr].copy()
                    oos_scored = oos_scored.copy()
                    oos_scored["local_threshold"] = chosen_thr
                else:
                    # Train and threshold per symbol, then aggregate.
                    val_parts = []
                    oos_scored_parts = []
                    oos_parts = []
                    chosen_map: dict[str, float] = {}
                    for symbol in sorted(set(scenario_oos["symbol"].dropna().astype(str).tolist())):
                        fit_sym = fit_train[fit_train["symbol"].astype(str) == symbol].copy()
                        val_sym = val[val["symbol"].astype(str) == symbol].copy()
                        oos_sym = scenario_oos[scenario_oos["symbol"].astype(str) == symbol].copy()
                        if fit_sym.empty or val_sym.empty or oos_sym.empty:
                            continue
                        if fit_sym["win_label"].nunique() < 2:
                            continue

                        estimator = build_model(model_name)
                        estimator.fit(fit_sym[cols].astype(float), fit_sym["win_label"].astype(int))
                        val_sym["meta_prob"] = estimator.predict_proba(val_sym[cols].astype(float))[:, 1]
                        oos_sym["meta_prob"] = estimator.predict_proba(oos_sym[cols].astype(float))[:, 1]

                        q_thresholds = [float(val_sym["meta_prob"].quantile(q)) for q in [0.55, 0.65, 0.75, 0.85]]
                        thresholds = sorted(set(threshold_grid + q_thresholds))
                        val_table_sym = evaluate_thresholds(val_sym, val_sym, thresholds)
                        val_table_sym["objective"] = val_table_sym.apply(
                            threshold_objective,
                            axis=1,
                            min_trades=max(8, int(args.min_trades // 3)),
                            min_winner_recall=float(args.min_winner_recall),
                            fp_cost=float(args.fp_cost),
                            fn_cost=float(args.fn_cost),
                            pf_weight=float(args.pf_weight),
                            net_r_weight=float(args.net_r_weight),
                            loss_rej_weight=float(args.loss_rej_weight),
                        )
                        best_sym = val_table_sym.sort_values(["objective", "profit_factor", "net_r"], ascending=[False, False, False]).iloc[0]
                        thr_sym = float(best_sym["threshold"])
                        chosen_map[symbol] = thr_sym
                        oos_sym = oos_sym.copy()
                        oos_sym["local_threshold"] = thr_sym
                        oos_scored_parts.append(oos_sym)
                        val_parts.append(val_sym[val_sym["meta_prob"] >= thr_sym].copy())
                        oos_parts.append(oos_sym[oos_sym["meta_prob"] >= thr_sym].copy())

                    if not oos_parts:
                        continue
                    oos_scored = pd.concat(oos_scored_parts, ignore_index=True)
                    val_kept = pd.concat(val_parts, ignore_index=True) if val_parts else pd.DataFrame()
                    oos_kept = pd.concat(oos_parts, ignore_index=True)
                    chosen_thr = float(np.mean(list(chosen_map.values()))) if chosen_map else 0.5

                    # Build a single summary table row for symbol-specific selection.
                    val_table = pd.DataFrame([
                        {
                            "threshold": chosen_thr,
                            **metrics(val_kept),
                            **relative_metrics(val_kept, val),
                        }
                    ])
                    val_table["objective"] = val_table.apply(
                        threshold_objective,
                        axis=1,
                        min_trades=int(args.min_trades),
                        min_winner_recall=float(args.min_winner_recall),
                        fp_cost=float(args.fp_cost),
                        fn_cost=float(args.fn_cost),
                        pf_weight=float(args.pf_weight),
                        net_r_weight=float(args.net_r_weight),
                        loss_rej_weight=float(args.loss_rej_weight),
                    )
                    best_val = val_table.iloc[0]

                oos_result = {
                    **metrics(oos_kept),
                    **relative_metrics(oos_kept, scenario_oos),
                }

                candidate_id = make_candidate_id(scenario_name, symbol_specific, model_name, set_name, chosen_thr)

                leaderboard_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "scenario": scenario_name,
                        "symbol_specific": symbol_specific,
                        "model": model_name,
                        "feature_set": set_name,
                        "n_features": len(cols),
                        "chosen_threshold": round(chosen_thr, 4),
                        "val_objective": round(float(best_val["objective"]), 4),
                        "val_pf": float(best_val["profit_factor"]),
                        "val_net_r": float(best_val["net_r"]),
                        "val_winner_recall": float(best_val["winner_recall_pct"]),
                        "oos_trades": oos_result["trades"],
                        "oos_pf": oos_result["profit_factor"],
                        "oos_net_r": oos_result["net_r"],
                        "oos_win_rate": oos_result["win_rate"],
                        "oos_winner_recall": oos_result["winner_recall_pct"],
                        "oos_loss_rejection": oos_result["loss_rejection_pct"],
                        "oos_keep_pct": oos_result["keep_pct_of_base"],
                    }
                )

                val_table = val_table.copy()
                val_table.insert(0, "candidate_id", candidate_id)
                val_table.insert(1, "scenario", scenario_name)
                val_table.insert(2, "symbol_specific", symbol_specific)
                val_table.insert(3, "model", model_name)
                val_table.insert(4, "feature_set", set_name)
                threshold_rows.append(val_table)

                kept_scored = oos_kept.copy()
                kept_scored["candidate_id"] = candidate_id
                kept_scored["scenario"] = scenario_name
                kept_scored["symbol_specific"] = symbol_specific
                kept_scored["model"] = model_name
                kept_scored["feature_set"] = set_name
                kept_scored["chosen_threshold"] = chosen_thr
                scored_oos_rows.append(kept_scored)

                scored_all = oos_scored.copy()
                keep_cols = ["symbol", "entry_time", "event_key", "r_multiple", "win_label", "meta_prob", "local_threshold"]
                keep_cols = [c for c in keep_cols if c in scored_all.columns]
                scored_all = scored_all[keep_cols]
                scored_all["candidate_id"] = candidate_id
                scored_all["scenario"] = scenario_name
                scored_all["symbol_specific"] = symbol_specific
                scored_all["model"] = model_name
                scored_all["feature_set"] = set_name
                scored_all["chosen_threshold"] = chosen_thr
                scored_oos_all_rows.append(scored_all)

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        ["oos_pf", "oos_net_r", "oos_winner_recall", "oos_trades"],
        ascending=[False, False, False, False],
    )
    threshold_table = pd.concat(threshold_rows, ignore_index=True) if threshold_rows else pd.DataFrame()
    kept_oos = pd.concat(scored_oos_rows, ignore_index=True) if scored_oos_rows else pd.DataFrame()
    scored_oos_all = pd.concat(scored_oos_all_rows, ignore_index=True) if scored_oos_all_rows else pd.DataFrame()

    top_candidates, stress_table, by_symbol, by_quarter = build_robustness_views(
        leaderboard=leaderboard,
        scored_oos=scored_oos_all,
        top_k=int(args.top_k_robustness),
        threshold_stress=float(args.threshold_stress),
    )

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    leaderboard_path = args.out_prefix.with_name(f"{args.out_prefix.name}_leaderboard.csv")
    threshold_path = args.out_prefix.with_name(f"{args.out_prefix.name}_thresholds.csv")
    kept_oos_path = args.out_prefix.with_name(f"{args.out_prefix.name}_kept_oos.csv")
    scored_oos_path = args.out_prefix.with_name(f"{args.out_prefix.name}_scored_oos.csv")
    top_candidates_path = args.out_prefix.with_name(f"{args.out_prefix.name}_top_candidates.csv")
    stress_path = args.out_prefix.with_name(f"{args.out_prefix.name}_robustness_stress.csv")
    by_symbol_path = args.out_prefix.with_name(f"{args.out_prefix.name}_robustness_by_symbol.csv")
    by_quarter_path = args.out_prefix.with_name(f"{args.out_prefix.name}_robustness_by_quarter.csv")
    leaderboard.to_csv(leaderboard_path, index=False)
    threshold_table.to_csv(threshold_path, index=False)
    kept_oos.to_csv(kept_oos_path, index=False)
    scored_oos_all.to_csv(scored_oos_path, index=False)
    top_candidates.to_csv(top_candidates_path, index=False)
    stress_table.to_csv(stress_path, index=False)
    by_symbol.to_csv(by_symbol_path, index=False)
    by_quarter.to_csv(by_quarter_path, index=False)

    print("Top experiments:")
    print(leaderboard.head(12).to_string(index=False))
    print(f"\nSaved: {leaderboard_path}")
    print(f"Saved: {threshold_path}")
    print(f"Saved: {kept_oos_path}")
    print(f"Saved: {scored_oos_path}")
    print(f"Saved: {top_candidates_path}")
    print(f"Saved: {stress_path}")
    print(f"Saved: {by_symbol_path}")
    print(f"Saved: {by_quarter_path}")


if __name__ == "__main__":
    main()
