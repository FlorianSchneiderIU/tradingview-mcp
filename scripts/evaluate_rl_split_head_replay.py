#!/usr/bin/env python3
"""Walk-forward replay for the normal RL sidecar split-head policy.

The evaluator replays decisions and rewards in timestamp order. At each signal
it records what a legacy single-head policy and the current accepted/rejected
split-head policy would have done, then applies rewards only after they appear.
The reward comparison is a sizing proxy: it scales the realized default-risk
reward by candidate action / actual historical action.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_log_dir() -> Path:
    root = repo_root()
    env_path = os.environ.get("RL_LOG_DIR")
    if env_path:
        return Path(env_path)
    local = root / "bot" / "logs" / "rl"
    if local.exists():
        return local
    container = Path("/app/rl")
    if container.exists():
        return container
    return local


def import_runtime(log_dir: Path):
    root = repo_root()
    os.environ.setdefault("RL_LOG_DIR", str(log_dir))
    os.environ.setdefault("RL_STATE_PATH", str(log_dir / "runtime_state.json"))
    os.environ.setdefault("RL_MODEL_PATH", str(log_dir / "agent_state.json"))
    os.environ.setdefault("RL_DECISIONS_PATH", str(log_dir / "decisions.jsonl"))
    os.environ.setdefault("RL_REWARDS_PATH", str(log_dir / "rewards.jsonl"))
    os.environ.setdefault("RL_TRAINING_EXAMPLES_PATH", str(log_dir / "training_examples.jsonl"))
    os.environ.setdefault("RL_TRADING_ENABLED", "false")
    for candidate in (root / "bot", root):
        text = str(candidate)
        if text not in sys.path:
            sys.path.insert(0, text)
    import rl_execution_bot as runtime  # noqa: PLC0415

    return runtime


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def ts_of(row: dict[str, Any]) -> str:
    payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
    return str(
        row.get("received_at")
        or row.get("queued_at")
        or row.get("completed_at")
        or row.get("sent_at")
        or payload.get("sent_at")
        or ""
    )


def make_single_head_agent(runtime):
    class SingleHeadRiskAgent(runtime.ContextualRiskAgent):
        def _policy_weights_unlocked(self, status_key: str):  # noqa: ARG002
            return self.weights, self.tp_weights, self.side_weights

        def learn(self, decision: dict[str, Any], reward: float) -> None:
            agent_data = decision.get("agent") if isinstance(decision.get("agent"), dict) else {}
            vector = agent_data.get("feature_vector") if isinstance(agent_data.get("feature_vector"), dict) else {}
            action = runtime.to_float(decision.get("action"))
            if not vector or action is None:
                return
            risk_action = abs(action)
            with self.lock:
                old_baseline = self.reward_baseline
                self.reward_updates += 1
                baseline_alpha = min(0.20, 2.0 / (self.reward_updates + 10.0))
                self.reward_baseline = (1.0 - baseline_alpha) * self.reward_baseline + baseline_alpha * reward
                advantage = runtime.clamp(reward - old_baseline, -5.0, 5.0)
                gradient_scale = runtime.LEARNING_RATE * advantage * max(0.05, risk_action * (1.0 - risk_action))
                tp_scale = runtime.to_float(decision.get("tp_scale"))
                tp_fraction = runtime.clamp((tp_scale or 0.0) / float(runtime.TP_SCALE_MAX), 0.0, 1.0)
                gradient_scale_tp = runtime.LEARNING_RATE * advantage * max(0.05, tp_fraction * (1.0 - tp_fraction))
                for key, value in vector.items():
                    x = runtime.to_float(value)
                    if x is None:
                        continue
                    old_weight = self.weights.get(str(key), 0.0)
                    self.weights[str(key)] = runtime.clamp(
                        old_weight * (1.0 - runtime.WEIGHT_DECAY) + gradient_scale * x,
                        -12.0,
                        12.0,
                    )
                    old_tp_weight = self.tp_weights.get(str(key), 0.0)
                    self.tp_weights[str(key)] = runtime.clamp(
                        old_tp_weight * (1.0 - runtime.WEIGHT_DECAY) + gradient_scale_tp * x,
                        -12.0,
                        12.0,
                    )

    return SingleHeadRiskAgent()


def metric_bucket() -> dict[str, Any]:
    return {
        "closed": 0,
        "would_trade": 0,
        "proxy_r": 0.0,
        "actions": [],
        "tp_scales": [],
        "tp_abs_diff": [],
        "positive_actions": [],
        "negative_actions": [],
        "action_reward_pairs": [],
    }


def add_metric(
    bucket: dict[str, Any],
    *,
    action: float,
    tp_scale: int,
    actual_tp_scale: float | None,
    per_action_reward: float,
    proxy_reward: float,
    min_action: float,
) -> None:
    bucket["closed"] += 1
    if abs(action) >= min_action:
        bucket["would_trade"] += 1
    bucket["proxy_r"] += proxy_reward
    bucket["actions"].append(action)
    bucket["tp_scales"].append(float(tp_scale))
    if actual_tp_scale is not None:
        bucket["tp_abs_diff"].append(abs(float(tp_scale) - actual_tp_scale))
    if per_action_reward > 0:
        bucket["positive_actions"].append(action)
    elif per_action_reward < 0:
        bucket["negative_actions"].append(action)
    bucket["action_reward_pairs"].append((action, per_action_reward))


def corr(values: list[tuple[float, float]]) -> float:
    if len(values) < 2:
        return 0.0
    xs = [x for x, _y in values]
    ys = [y for _x, y in values]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    num = sum((x - x_mean) * (y - y_mean) for x, y in values)
    den_x = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    if den_x <= 0 or den_y <= 0:
        return 0.0
    return num / (den_x * den_y)


def finalize_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    closed = int(bucket["closed"])

    def avg(name: str) -> float:
        values = bucket[name]
        return sum(values) / len(values) if values else 0.0

    return {
        "closed": closed,
        "would_trade": int(bucket["would_trade"]),
        "would_trade_rate": (float(bucket["would_trade"]) / closed) if closed else 0.0,
        "proxy_r": float(bucket["proxy_r"]),
        "proxy_avg_r": (float(bucket["proxy_r"]) / closed) if closed else 0.0,
        "avg_action": avg("actions"),
        "avg_tp_scale": avg("tp_scales"),
        "avg_tp_abs_diff": avg("tp_abs_diff"),
        "avg_action_on_positive": avg("positive_actions"),
        "avg_action_on_negative": avg("negative_actions"),
        "action_reward_corr": corr(bucket["action_reward_pairs"]),
    }


def replay(args: argparse.Namespace) -> dict[str, Any]:
    log_dir = Path(args.log_dir)
    runtime = import_runtime(log_dir)
    decisions_path = Path(args.decisions) if args.decisions else log_dir / "decisions.jsonl"
    rewards_path = Path(args.rewards) if args.rewards else log_dir / "rewards.jsonl"

    decision_first: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(decisions_path):
        decision_id = str(row.get("decision_id") or "")
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else None
        if not decision_id or not payload:
            continue
        decision_first.setdefault(decision_id, row)

    reward_rows: list[tuple[str, dict[str, Any], float]] = []
    for row in load_jsonl(rewards_path):
        decision_id = str(row.get("decision_id") or "")
        if not decision_id:
            continue
        reward = runtime.to_float(row.get("reward_default_r"))
        if reward is None:
            reward = runtime.to_float(row.get("reward_actual_r"))
        if reward is None:
            reward = runtime.to_float(row.get("reward"))
        if reward is None:
            continue
        reward_rows.append((decision_id, row, float(reward)))

    events: list[tuple[str, int, str, str, Any]] = []
    counter = 0
    for decision_id, row in decision_first.items():
        events.append((ts_of(row), counter, "decision", decision_id, row))
        counter += 1
    for decision_id, row, reward in reward_rows:
        events.append((ts_of(row), counter, "reward", decision_id, (row, reward)))
        counter += 1
    events.sort(key=lambda item: (item[0], item[1]))

    single = make_single_head_agent(runtime)
    split = runtime.ContextualRiskAgent()
    predictions: dict[str, dict[str, Any]] = {}
    metrics = {
        "actual": metric_bucket(),
        "single_head": metric_bucket(),
        "split_head": metric_bucket(),
    }
    by_status: dict[str, dict[str, dict[str, Any]]] = {}
    matched_rewards = 0
    unmatched_rewards = 0

    for _ts, _counter, kind, decision_id, data in events:
        if kind == "decision":
            row = data
            payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
            if not payload:
                continue
            single_action, single_info = single.decide(payload)
            split_action, split_info = split.decide(payload)
            predictions[decision_id] = {
                "payload": payload,
                "source_status": row.get("source_status", payload.get("status")),
                "actual_action": runtime.to_float(row.get("action")) or 0.0,
                "actual_risk_action": abs(runtime.to_float(row.get("action")) or 0.0),
                "actual_tp_scale": runtime.to_float(row.get("tp_scale")),
                "single_action": single_action,
                "single_tp_scale": int(single_info.get("tp_scale") or 0),
                "single_info": single_info,
                "split_action": split_action,
                "split_tp_scale": int(split_info.get("tp_scale") or 0),
                "split_info": split_info,
            }
            continue

        row, reward = data
        prediction = predictions.get(decision_id)
        if prediction is None:
            unmatched_rewards += 1
            continue
        matched_rewards += 1
        actual_action = float(prediction.get("actual_action") or 0.0)
        actual_risk_action = float(prediction.get("actual_risk_action") or abs(actual_action))
        if actual_risk_action <= 1e-9:
            per_action_reward = 0.0
        else:
            per_action_reward = float(reward) / actual_risk_action
        actual_tp_scale = prediction.get("actual_tp_scale")
        status = runtime.ContextualRiskAgent._status_key(prediction.get("source_status"))
        status_metrics = by_status.setdefault(
            status,
            {
                "actual": metric_bucket(),
                "single_head": metric_bucket(),
                "split_head": metric_bucket(),
            },
        )

        policy_values = {
            "actual": (actual_action, int(actual_tp_scale or 0), reward),
            "single_head": (
                float(prediction["single_action"]),
                int(prediction["single_tp_scale"]),
                per_action_reward * abs(float(prediction["single_action"])),
            ),
            "split_head": (
                float(prediction["split_action"]),
                int(prediction["split_tp_scale"]),
                per_action_reward * abs(float(prediction["split_action"])),
            ),
        }
        for name, (action, tp_scale, proxy_reward) in policy_values.items():
            add_metric(
                metrics[name],
                action=action,
                tp_scale=tp_scale,
                actual_tp_scale=actual_tp_scale,
                per_action_reward=per_action_reward,
                proxy_reward=float(proxy_reward),
                min_action=runtime.MIN_ACTION_TO_TRADE,
            )
            add_metric(
                status_metrics[name],
                action=action,
                tp_scale=tp_scale,
                actual_tp_scale=actual_tp_scale,
                per_action_reward=per_action_reward,
                proxy_reward=float(proxy_reward),
                min_action=runtime.MIN_ACTION_TO_TRADE,
            )

        closed_pnl = runtime.to_float(row.get("closed_pnl")) or 0.0
        entry_slippage_bps = runtime.to_float(row.get("entry_slippage_bps"))
        for agent, action_key, tp_key, info_key in (
            (single, "single_action", "single_tp_scale", "single_info"),
            (split, "split_action", "split_tp_scale", "split_info"),
        ):
            payload = prediction["payload"]
            agent.add_trade_history(
                payload,
                closed_pnl=closed_pnl,
                reward_default_r=float(reward),
                entry_slippage_bps=entry_slippage_bps,
            )
            agent.learn(
                {
                    "decision_id": decision_id,
                    "action": prediction[action_key],
                    "tp_scale": prediction[tp_key],
                    "source_status": prediction.get("source_status"),
                    "payload": payload,
                    "agent": prediction[info_key],
                },
                float(reward),
            )

    finalized = {name: finalize_bucket(bucket) for name, bucket in metrics.items()}
    finalized_status = {
        status: {name: finalize_bucket(bucket) for name, bucket in buckets.items()}
        for status, buckets in by_status.items()
    }
    return {
        "schema_version": "rl_split_head_replay_v1",
        "decisions": len(decision_first),
        "rewards": len(reward_rows),
        "matched_rewards": matched_rewards,
        "unmatched_rewards": unmatched_rewards,
        "decisions_path": str(decisions_path),
        "rewards_path": str(rewards_path),
        "note": (
            "proxy_r scales realized reward_default_r by candidate action / actual historical action; "
            "this evaluates sizing quality, not a full counterfactual fill simulation."
        ),
        "policies": finalized,
        "by_status": finalized_status,
        "split_minus_single_proxy_r": finalized["split_head"]["proxy_r"] - finalized["single_head"]["proxy_r"],
        "split_agent_updates": split.reward_updates,
        "single_agent_updates": single.reward_updates,
        "split_status_updates": split.reward_updates_by_status,
    }


def print_summary(report: dict[str, Any]) -> None:
    print("RL split-head walk-forward replay")
    print(f"decisions={report['decisions']} rewards={report['rewards']} matched={report['matched_rewards']}")
    print(f"split_minus_single_proxy_r={report['split_minus_single_proxy_r']:+.4f}R")
    for name, bucket in report["policies"].items():
        print(
            f"{name:12s} proxy={bucket['proxy_r']:+.4f}R "
            f"avg={bucket['proxy_avg_r']:+.4f}R "
            f"action={bucket['avg_action']:.3f} "
            f"trade={bucket['would_trade_rate']:.1%} "
            f"tp_diff={bucket['avg_tp_abs_diff']:.2f} "
            f"corr={bucket['action_reward_corr']:+.3f}"
        )
    for status, buckets in sorted(report["by_status"].items()):
        single = buckets.get("single_head", {})
        split = buckets.get("split_head", {})
        print(
            f"{status:8s} split-single="
            f"{float(split.get('proxy_r', 0.0)) - float(single.get('proxy_r', 0.0)):+.4f}R "
            f"closed={int(split.get('closed', 0))}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", default=str(default_log_dir()), help="RL log directory")
    parser.add_argument("--decisions", default="", help="Path to decisions.jsonl")
    parser.add_argument("--rewards", default="", help="Path to rewards.jsonl")
    parser.add_argument(
        "--output",
        default=str(repo_root() / "scripts" / "output" / "rl_split_head_replay_report.json"),
        help="Where to write the JSON report",
    )
    args = parser.parse_args()
    report = replay(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print_summary(report)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
