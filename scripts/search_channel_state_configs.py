from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_channel_state_study import load_dataset
from scripts.channel_state_research.modeling import WalkForwardSpec, run_walkforward_study


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search a small set of channel-state model configurations.")
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--feature-groups-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("scripts/channel_state_config_search.csv"))
    parser.add_argument("--model", choices=["logreg", "rf", "hgb"], default="logreg")
    parser.add_argument("--train-months", type=int, default=6)
    parser.add_argument("--val-months", type=int, default=3)
    parser.add_argument("--test-months", type=int, default=3)
    parser.add_argument("--threshold-mode", choices=["absolute", "percentile"], default="percentile")
    parser.add_argument("--long-score-mode", choices=["probability", "edge", "logit", "logit_edge"], default="probability")
    parser.add_argument("--short-score-mode", choices=["probability", "edge", "logit", "logit_edge"], default="logit_edge")
    parser.add_argument("--long-thresholds", default="0.80,0.90,0.95")
    parser.add_argument("--short-thresholds", default="0.80,0.90,0.95")
    parser.add_argument("--probability-gap-values", default="0.0,0.5,0.9")
    parser.add_argument("--gate-presets", default="none,trend_alignment,combo_bearish")
    return parser.parse_args()


def candidate_configs() -> list[dict[str, Any]]:
    return [
        {
            "name": "baseline_full",
            "channel_family": "both",
            "timeframes": ("1h", "4h", "1d", "1w"),
            "feature_group_names": (
                "structural",
                "position",
                "excursion_acceptance",
                "touch_interaction",
                "swing_state",
                "channel_evolution",
                "confluence",
                "regime",
            ),
        },
        {
            "name": "no_week",
            "channel_family": "both",
            "timeframes": ("1h", "4h", "1d"),
            "feature_group_names": (
                "structural",
                "position",
                "excursion_acceptance",
                "touch_interaction",
                "swing_state",
                "channel_evolution",
                "confluence",
                "regime",
            ),
        },
        {
            "name": "wick_no_week",
            "channel_family": "wick",
            "timeframes": ("1h", "4h", "1d"),
            "feature_group_names": (
                "structural",
                "position",
                "excursion_acceptance",
                "touch_interaction",
                "swing_state",
                "channel_evolution",
                "confluence",
                "regime",
            ),
        },
        {
            "name": "wick_geom_no_week",
            "channel_family": "wick",
            "timeframes": ("1h", "4h", "1d"),
            "feature_group_names": (
                "structural",
                "position",
                "excursion_acceptance",
                "touch_interaction",
                "swing_state",
                "confluence",
            ),
        },
        {
            "name": "structure_position_only",
            "channel_family": "wick",
            "timeframes": ("1h", "4h", "1d"),
            "feature_group_names": (
                "structural",
                "position",
                "touch_interaction",
                "confluence",
            ),
        },
    ]


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset_path)
    feature_groups = json.loads(args.feature_groups_path.read_text(encoding="utf-8"))

    long_thresholds = tuple(float(item.strip()) for item in args.long_thresholds.split(",") if item.strip())
    short_thresholds = tuple(float(item.strip()) for item in args.short_thresholds.split(",") if item.strip())
    probability_gap_values = tuple(float(item.strip()) for item in args.probability_gap_values.split(",") if item.strip())
    gate_presets = tuple(item.strip() for item in args.gate_presets.split(",") if item.strip())

    rows: list[dict[str, Any]] = []
    for config in candidate_configs():
        spec = WalkForwardSpec(
            model_name=args.model,
            train_months=args.train_months,
            val_months=args.val_months,
            test_months=args.test_months,
            threshold_mode=args.threshold_mode,
            long_score_mode=args.long_score_mode,
            short_score_mode=args.short_score_mode,
            long_thresholds=long_thresholds,
            short_thresholds=short_thresholds,
            probability_gap_values=probability_gap_values,
            gate_presets=gate_presets,
            compute_feature_importance=False,
            channel_family=config["channel_family"],
            timeframes=config["timeframes"],
            feature_group_names=config["feature_group_names"],
        )
        results = run_walkforward_study(dataset, feature_groups, spec, all_timeframes=["1h", "4h", "1d", "1w"])
        folds = results["folds"]
        aggregate = folds[folds["fold"].astype(str) == "aggregate"]
        row: dict[str, Any] = {
            "config": config["name"],
            "channel_family": config["channel_family"],
            "timeframes": ",".join(config["timeframes"]),
            "feature_groups": ",".join(config["feature_group_names"]),
        }
        if aggregate.empty:
            row["status"] = "no_aggregate"
        else:
            agg = aggregate.iloc[0]
            row["status"] = "ok"
            for column in [
                "trade_total_return",
                "trade_trades",
                "trade_profit_factor",
                "trade_hit_rate",
                "trade_max_drawdown",
                "trade_probability_gap",
                "trade_gate_preset",
                "trade_long_threshold",
                "trade_short_threshold",
                "threshold_mode",
                "long_score_mode",
                "short_score_mode",
                "long_test_auc",
                "short_test_auc",
            ]:
                row[column] = agg.get(column)
        print(row, flush=True)
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(["status", "trade_total_return"], ascending=[True, False])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print()
    print(summary.to_string(index=False))
    print(f"\nSaved config search to {args.output}")


if __name__ == "__main__":
    main()
