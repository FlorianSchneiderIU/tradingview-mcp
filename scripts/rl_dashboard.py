#!/usr/bin/env python3
"""Local dashboard for the normal and Matrix RL execution sidecars."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


UTC = timezone.utc


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_log_root() -> Path:
    local = repo_root() / "bot" / "logs"
    if local.exists():
        return local
    container = Path("/app/logs")
    if container.exists():
        return container
    return local


def now_utc() -> datetime:
    return datetime.now(UTC)


def to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def parse_ts(raw: Any) -> datetime | None:
    if raw is None or raw == "":
        return None
    if isinstance(raw, (int, float)) and math.isfinite(float(raw)):
        value = float(raw)
        if value > 10_000_000_000:
            value /= 1000.0
        try:
            return datetime.fromtimestamp(value, UTC)
        except (OSError, OverflowError, ValueError):
            return None
    text = str(raw).strip()
    if not text:
        return None
    if text.isdigit():
        return parse_ts(float(text))
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def iso_or_none(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def ts_of(row: dict[str, Any]) -> datetime | None:
    payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
    reward = row.get("reward") if isinstance(row.get("reward"), dict) else {}
    for key in (
        "received_at",
        "completed_at",
        "queued_at",
        "execution_finished_at",
        "execution_started_at",
        "sent_at",
    ):
        parsed = parse_ts(row.get(key))
        if parsed is not None:
            return parsed
    return parse_ts(payload.get("sent_at") or reward.get("received_at"))


def age_seconds(value: datetime | None) -> float | None:
    if value is None:
        return None
    return max(0.0, (now_utc() - value).total_seconds())


def status_key(*rows: dict[str, Any]) -> str:
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw = row.get("source_status") or row.get("status")
        if raw:
            text = str(raw).strip().lower()
            if text:
                return text
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        raw = payload.get("status")
        if raw:
            text = str(raw).strip().lower()
            if text:
                return text
    return "unknown"


def strategy_symbol_key(*rows: dict[str, Any]) -> str:
    strategy = "unknown"
    symbol = "unknown"
    for row in rows:
        if not isinstance(row, dict):
            continue
        if strategy == "unknown" and row.get("strategy"):
            strategy = str(row.get("strategy"))
        if symbol == "unknown" and row.get("symbol"):
            symbol = str(row.get("symbol")).upper()
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        if strategy == "unknown" and payload.get("strategy"):
            strategy = str(payload.get("strategy"))
        if symbol == "unknown" and payload.get("symbol"):
            symbol = str(payload.get("symbol")).upper()
    return f"{strategy}/{symbol}"


def get_reward_r(row: dict[str, Any]) -> float | None:
    reward = row.get("reward") if isinstance(row.get("reward"), dict) else {}
    for value in (
        row.get("reward_actual_r"),
        reward.get("reward_actual_r"),
        row.get("reward_default_r"),
        reward.get("reward_default_r"),
        row.get("reward"),
    ):
        number = to_float(value)
        if number is not None:
            return number
    return None


def get_pnl(row: dict[str, Any]) -> float | None:
    reward = row.get("reward") if isinstance(row.get("reward"), dict) else {}
    for value in (row.get("closed_pnl"), reward.get("closed_pnl")):
        number = to_float(value)
        if number is not None:
            return number
    return None


def get_action(row: dict[str, Any], decision: dict[str, Any] | None = None) -> float | None:
    reward = row.get("reward") if isinstance(row.get("reward"), dict) else {}
    decision = decision if isinstance(decision, dict) else {}
    for value in (row.get("action"), reward.get("action"), decision.get("action")):
        number = to_float(value)
        if number is not None:
            return number
    return None


def get_risk_action(row: dict[str, Any], decision: dict[str, Any] | None = None) -> float | None:
    reward = row.get("reward") if isinstance(row.get("reward"), dict) else {}
    decision = decision if isinstance(decision, dict) else {}
    for value in (row.get("risk_action"), reward.get("risk_action"), decision.get("risk_action")):
        number = to_float(value)
        if number is not None:
            return number
    action = get_action(row, decision)
    return abs(action) if action is not None else None


def get_tp_scale(row: dict[str, Any], decision: dict[str, Any] | None = None) -> float | None:
    reward = row.get("reward") if isinstance(row.get("reward"), dict) else {}
    decision = decision if isinstance(decision, dict) else {}
    for value in (row.get("tp_scale"), reward.get("tp_scale"), decision.get("tp_scale")):
        number = to_float(value)
        if number is not None:
            return number
    return None


def avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


class FileCache:
    def __init__(self) -> None:
        self._jsonl: dict[str, tuple[int, int, list[dict[str, Any]]]] = {}
        self._json: dict[str, tuple[int, int, Any]] = {}
        self._compose_cache: tuple[float, dict[str, dict[str, Any]]] = (0.0, {})

    @staticmethod
    def _stat(path: Path) -> tuple[int, int] | None:
        try:
            stat = path.stat()
        except OSError:
            return None
        return stat.st_mtime_ns, stat.st_size

    def jsonl(self, path: Path) -> list[dict[str, Any]]:
        stat = self._stat(path)
        if stat is None:
            return []
        key = str(path)
        cached = self._jsonl.get(key)
        if cached and cached[0] == stat[0] and cached[1] == stat[1]:
            return cached[2]
        rows: list[dict[str, Any]] = []
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
        self._jsonl[key] = (stat[0], stat[1], rows)
        return rows

    def json(self, path: Path) -> Any:
        stat = self._stat(path)
        if stat is None:
            return None
        key = str(path)
        cached = self._json.get(key)
        if cached and cached[0] == stat[0] and cached[1] == stat[1]:
            return cached[2]
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            data = None
        self._json[key] = (stat[0], stat[1], data)
        return data

    def compose_services(self, cwd: Path) -> dict[str, dict[str, Any]]:
        ts, cached = self._compose_cache
        if time.time() - ts < 5.0:
            return cached
        services: dict[str, dict[str, Any]] = {}
        try:
            proc = subprocess.run(
                ["docker", "compose", "ps", "--format", "json"],
                cwd=str(cwd),
                text=True,
                capture_output=True,
                timeout=3.0,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            self._compose_cache = (time.time(), {})
            return {}
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            service = str(row.get("Service") or row.get("Name") or "")
            if service:
                services[service] = row
        self._compose_cache = (time.time(), services)
        return services


class DashboardData:
    def __init__(self, *, log_root: Path, project_root: Path) -> None:
        self.log_root = log_root
        self.project_root = project_root
        self.cache = FileCache()
        self.bots = [
            {
                "key": "normal",
                "name": "Normal RL",
                "service": "rl-exec-bot",
                "log_dir": log_root / "rl",
            },
            {
                "key": "matrix",
                "name": "Matrix RL",
                "service": "matrix-rl-exec-bot",
                "log_dir": log_root / "rl_matrix",
            },
        ]

    def summary(self) -> dict[str, Any]:
        compose = self.cache.compose_services(self.project_root)
        return {
            "generated_at": now_utc().isoformat(),
            "log_root": str(self.log_root),
            "bots": [self._bot_summary(bot, compose.get(bot["service"], {})) for bot in self.bots],
        }

    def _load_bot_files(self, log_dir: Path) -> dict[str, Any]:
        return {
            "decisions": self.cache.jsonl(log_dir / "decisions.jsonl"),
            "rewards": self.cache.jsonl(log_dir / "rewards.jsonl"),
            "examples": self.cache.jsonl(log_dir / "training_examples.jsonl"),
            "agent": self.cache.json(log_dir / "agent_state.json") or {},
            "runtime": self.cache.json(log_dir / "runtime_state.json") or {},
            "heartbeat": self.cache.json(log_dir / "heartbeat_state.json") or {},
            "replay": self.cache.json(log_dir / "rl_split_head_replay_report.json") or None,
        }

    @staticmethod
    def _latest_decisions(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for row in rows:
            decision_id = str(row.get("decision_id") or "")
            if decision_id:
                latest[decision_id] = row
        return latest

    def _bot_summary(self, bot: dict[str, Any], compose_row: dict[str, Any]) -> dict[str, Any]:
        log_dir = Path(bot["log_dir"])
        files = self._load_bot_files(log_dir)
        decisions = files["decisions"]
        rewards = files["rewards"]
        examples = files["examples"]
        agent = files["agent"] if isinstance(files["agent"], dict) else {}
        runtime = files["runtime"] if isinstance(files["runtime"], dict) else {}
        heartbeat = files["heartbeat"] if isinstance(files["heartbeat"], dict) else {}
        latest = self._latest_decisions(decisions)
        latest_values = list(latest.values())
        decision_lookup = latest
        latest_decision_ts = max((ts_of(row) for row in latest_values), default=None)
        latest_reward_ts = max((ts_of(row) for row in rewards), default=None)
        active_trades = runtime.get("active_trades") if isinstance(runtime.get("active_trades"), dict) else {}
        active_count = len(active_trades)
        if not active_count:
            active_count = sum(1 for row in latest_values if row.get("executed") and not row.get("completed"))

        status_counts: dict[str, int] = defaultdict(int)
        execution_counts: dict[str, int] = defaultdict(int)
        actions: list[float] = []
        risk_actions: list[float] = []
        tp_scales: list[float] = []
        reversed_decisions = 0
        for row in latest_values:
            status_counts[status_key(row)] += 1
            execution_counts[str(row.get("execution_status") or "unknown")] += 1
            if row.get("reversed_trade"):
                reversed_decisions += 1
            action = to_float(row.get("action"))
            if action is not None:
                actions.append(action)
                risk_actions.append(abs(action))
            tp_scale = to_float(row.get("tp_scale"))
            if tp_scale is not None:
                tp_scales.append(tp_scale)

        reward_metrics = self._reward_metrics(rewards, decision_lookup)
        status_diagnostics = self._status_diagnostics(latest_values, rewards, decision_lookup, agent)
        pockets = self._strategy_pockets(rewards, decision_lookup)
        safety = self._safety(examples, rewards, decision_lookup)
        shadow = self._shadow_policies(latest_values)
        latest_features = self._latest_feature_info(latest_values)
        agent_summary = self._agent_summary(agent)
        file_summary = self._file_summary(log_dir)
        replay = files["replay"] if isinstance(files["replay"], dict) else None

        state = str(compose_row.get("State") or "").lower()
        status = str(compose_row.get("Status") or "")
        inferred_status = False
        if not status:
            inferred_status = True
            decision_age = age_seconds(latest_decision_ts)
            reward_age = age_seconds(latest_reward_ts)
            freshest = min(
                [value for value in (decision_age, reward_age) if value is not None],
                default=None,
            )
            if freshest is not None and freshest < 2.0 * 3600.0:
                status = "logs active"
                state = "logs"
            elif log_dir.exists():
                status = "logs quiet"
                state = "logs"
            else:
                status = "unavailable"
        return {
            "key": bot["key"],
            "name": bot["name"],
            "service": bot["service"],
            "log_dir": str(log_dir),
            "container": {
                "state": state or "unknown",
                "status": status or "unavailable",
                "running": state == "running",
                "inferred": inferred_status,
            },
            "freshness": {
                "latest_decision_at": iso_or_none(latest_decision_ts),
                "latest_decision_age_s": age_seconds(latest_decision_ts),
                "latest_reward_at": iso_or_none(latest_reward_ts),
                "latest_reward_age_s": age_seconds(latest_reward_ts),
                "model_saved_at": agent.get("saved_at"),
                "heartbeat_day": heartbeat.get("last_daily_heartbeat_key"),
            },
            "counts": {
                "decisions": len(latest),
                "decision_events": len(decisions),
                "rewards": len(rewards),
                "training_examples": len(examples),
                "active_trades": active_count,
                "reversed_decisions": reversed_decisions,
                "status": dict(sorted(status_counts.items())),
                "execution": dict(sorted(execution_counts.items())),
            },
            "actions": {
                "avg_action": avg(actions),
                "avg_risk_action": avg(risk_actions),
                "avg_tp_scale": avg(tp_scales),
            },
            "agent": agent_summary,
            "rewards": reward_metrics,
            "status_diagnostics": status_diagnostics,
            "pockets": pockets,
            "safety": safety,
            "shadow": shadow,
            "latest_features": latest_features,
            "files": file_summary,
            "replay": replay,
        }

    @staticmethod
    def _file_summary(log_dir: Path) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name in (
            "decisions.jsonl",
            "rewards.jsonl",
            "training_examples.jsonl",
            "agent_state.json",
            "runtime_state.json",
            "rl_execution_bot.log",
            "rl_split_head_replay_report.json",
        ):
            path = log_dir / name
            try:
                stat = path.stat()
            except OSError:
                continue
            out[name] = {
                "bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
            }
        return out

    @staticmethod
    def _agent_summary(agent: dict[str, Any]) -> dict[str, Any]:
        weights_by_status = agent.get("weights_by_status") if isinstance(agent.get("weights_by_status"), dict) else {}
        tp_weights_by_status = (
            agent.get("tp_weights_by_status") if isinstance(agent.get("tp_weights_by_status"), dict) else {}
        )
        side_weights_by_status = (
            agent.get("side_weights_by_status") if isinstance(agent.get("side_weights_by_status"), dict) else {}
        )
        baselines = (
            agent.get("reward_baselines_by_status")
            if isinstance(agent.get("reward_baselines_by_status"), dict)
            else {}
        )
        updates_by_status = (
            agent.get("reward_updates_by_status")
            if isinstance(agent.get("reward_updates_by_status"), dict)
            else {}
        )
        signal_history = agent.get("signal_history") if isinstance(agent.get("signal_history"), dict) else {}
        trade_history = agent.get("trade_history") if isinstance(agent.get("trade_history"), dict) else {}
        return {
            "reward_updates": int(agent.get("reward_updates") or 0),
            "reward_baseline": to_float(agent.get("reward_baseline")) or 0.0,
            "global_weight_count": len(agent.get("weights") or {}),
            "global_tp_weight_count": len(agent.get("tp_weights") or {}),
            "global_side_weight_count": len(agent.get("side_weights") or {}),
            "stat_count": len(agent.get("stats") or {}),
            "status_weight_counts": {
                str(status): len(weights) for status, weights in weights_by_status.items() if isinstance(weights, dict)
            },
            "status_tp_weight_counts": {
                str(status): len(weights)
                for status, weights in tp_weights_by_status.items()
                if isinstance(weights, dict)
            },
            "status_side_weight_counts": {
                str(status): len(weights)
                for status, weights in side_weights_by_status.items()
                if isinstance(weights, dict)
            },
            "status_baselines": {str(k): to_float(v) or 0.0 for k, v in baselines.items()},
            "status_updates": {str(k): int(v or 0) for k, v in updates_by_status.items()},
            "signal_history_scopes": len(signal_history),
            "signal_history_rows": sum(len(v) for v in signal_history.values() if isinstance(v, list)),
            "trade_history_scopes": len(trade_history),
            "trade_history_rows": sum(len(v) for v in trade_history.values() if isinstance(v, list)),
            "saved_at": agent.get("saved_at"),
        }

    @staticmethod
    def _reward_metrics(
        rewards: list[dict[str, Any]],
        decision_lookup: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        now = now_utc()
        windows = {
            "all": None,
            "today": now.date().isoformat(),
            "24h": 24.0 * 3600.0,
            "7d": 7.0 * 24.0 * 3600.0,
        }
        out: dict[str, Any] = {}
        enriched: list[tuple[dict[str, Any], dict[str, Any], datetime | None, float, float]] = []
        for row in rewards:
            decision = decision_lookup.get(str(row.get("decision_id") or ""), {})
            reward_r = get_reward_r(row)
            pnl = get_pnl(row)
            if reward_r is None:
                continue
            enriched.append((row, decision, ts_of(row), float(reward_r), float(pnl or 0.0)))

        for name, window in windows.items():
            selected = []
            for row in enriched:
                ts = row[2]
                if name == "all":
                    selected.append(row)
                elif name == "today":
                    if ts and ts.date().isoformat() == window:
                        selected.append(row)
                elif ts and (now - ts).total_seconds() <= float(window):
                    selected.append(row)
            out[name] = DashboardData._reward_metric_bucket(selected)

        curve_rows = sorted(enriched, key=lambda row: row[2] or datetime.min.replace(tzinfo=UTC))[-120:]
        cumulative = 0.0
        curve = []
        for row, decision, ts, reward_r, pnl in curve_rows:
            cumulative += reward_r
            curve.append(
                {
                    "ts": iso_or_none(ts),
                    "r": reward_r,
                    "pnl": pnl,
                    "cumulative_r": cumulative,
                    "status": status_key(row, decision),
                    "pocket": strategy_symbol_key(row, decision),
                }
            )
        out["curve"] = curve

        daily: dict[str, dict[str, float]] = defaultdict(lambda: {"closed": 0.0, "r": 0.0, "pnl": 0.0})
        for row, decision, ts, reward_r, pnl in enriched:
            if ts is None:
                continue
            key = ts.date().isoformat()
            daily[key]["closed"] += 1.0
            daily[key]["r"] += reward_r
            daily[key]["pnl"] += pnl
        out["daily"] = [
            {"day": day, "closed": int(value["closed"]), "r": value["r"], "pnl": value["pnl"]}
            for day, value in sorted(daily.items())[-21:]
        ]
        return out

    @staticmethod
    def _reward_metric_bucket(
        rows: list[tuple[dict[str, Any], dict[str, Any], datetime | None, float, float]]
    ) -> dict[str, Any]:
        closed = len(rows)
        wins = sum(1 for _row, _decision, _ts, reward_r, pnl in rows if pnl > 0 or (pnl == 0 and reward_r > 0))
        losses = sum(1 for _row, _decision, _ts, reward_r, pnl in rows if pnl < 0 or (pnl == 0 and reward_r < 0))
        reward_sum = sum(row[3] for row in rows)
        pnl_sum = sum(row[4] for row in rows)
        actions = [get_action(row, decision) for row, decision, _ts, _reward_r, _pnl in rows]
        actions = [value for value in actions if value is not None]
        risk_actions = [get_risk_action(row, decision) for row, decision, _ts, _reward_r, _pnl in rows]
        risk_actions = [value for value in risk_actions if value is not None]
        tp_scales = [get_tp_scale(row, decision) for row, decision, _ts, _reward_r, _pnl in rows]
        tp_scales = [value for value in tp_scales if value is not None]
        slippage = []
        for row, _decision, _ts, _reward_r, _pnl in rows:
            value = to_float(row.get("entry_slippage_bps"))
            if value is not None:
                slippage.append(value)
        return {
            "closed": closed,
            "wins": wins,
            "losses": losses,
            "winrate": (wins / closed) if closed else 0.0,
            "pnl": pnl_sum,
            "r": reward_sum,
            "avg_r": (reward_sum / closed) if closed else 0.0,
            "avg_action": avg(actions),
            "avg_risk_action": avg(risk_actions),
            "avg_tp_scale": avg(tp_scales),
            "avg_slippage_bps": avg(slippage),
            "avg_abs_slippage_bps": avg([abs(value) for value in slippage]),
        }

    @staticmethod
    def _status_diagnostics(
        decisions: list[dict[str, Any]],
        rewards: list[dict[str, Any]],
        decision_lookup: dict[str, dict[str, Any]],
        agent: dict[str, Any],
    ) -> list[dict[str, Any]]:
        statuses = set()
        for row in decisions:
            statuses.add(status_key(row))
        for row in rewards:
            statuses.add(status_key(row, decision_lookup.get(str(row.get("decision_id") or ""), {})))
        weights_by_status = agent.get("weights_by_status") if isinstance(agent.get("weights_by_status"), dict) else {}
        statuses.update(str(key) for key in weights_by_status)
        statuses.discard("")
        buckets: dict[str, dict[str, Any]] = {
            status: {
                "status": status,
                "decisions": 0,
                "executed": 0,
                "closed": 0,
                "wins": 0,
                "losses": 0,
                "r": 0.0,
                "pnl": 0.0,
                "actions": [],
                "risk_actions": [],
                "tp_scales": [],
                "reversed": 0,
            }
            for status in statuses
        }
        for row in decisions:
            bucket = buckets.setdefault(status_key(row), {"status": status_key(row), "actions": [], "risk_actions": [], "tp_scales": [], "reversed": 0})
            bucket["decisions"] = int(bucket.get("decisions", 0)) + 1
            if row.get("executed"):
                bucket["executed"] = int(bucket.get("executed", 0)) + 1
            if row.get("reversed_trade"):
                bucket["reversed"] = int(bucket.get("reversed", 0)) + 1
            action = to_float(row.get("action"))
            if action is not None:
                bucket.setdefault("actions", []).append(action)
                bucket.setdefault("risk_actions", []).append(abs(action))
            tp_scale = to_float(row.get("tp_scale"))
            if tp_scale is not None:
                bucket.setdefault("tp_scales", []).append(tp_scale)
        for row in rewards:
            decision = decision_lookup.get(str(row.get("decision_id") or ""), {})
            bucket = buckets.setdefault(
                status_key(row, decision),
                {"status": status_key(row, decision), "actions": [], "risk_actions": [], "tp_scales": [], "reversed": 0},
            )
            reward_r = get_reward_r(row) or 0.0
            pnl = get_pnl(row) or 0.0
            bucket["closed"] = int(bucket.get("closed", 0)) + 1
            bucket["r"] = float(bucket.get("r", 0.0)) + reward_r
            bucket["pnl"] = float(bucket.get("pnl", 0.0)) + pnl
            if pnl > 0 or (pnl == 0 and reward_r > 0):
                bucket["wins"] = int(bucket.get("wins", 0)) + 1
            elif pnl < 0 or (pnl == 0 and reward_r < 0):
                bucket["losses"] = int(bucket.get("losses", 0)) + 1
        baselines = agent.get("reward_baselines_by_status") if isinstance(agent.get("reward_baselines_by_status"), dict) else {}
        updates = agent.get("reward_updates_by_status") if isinstance(agent.get("reward_updates_by_status"), dict) else {}
        tp_weights = agent.get("tp_weights_by_status") if isinstance(agent.get("tp_weights_by_status"), dict) else {}
        rows = []
        for status, bucket in buckets.items():
            closed = int(bucket.get("closed", 0))
            rows.append(
                {
                    "status": status,
                    "decisions": int(bucket.get("decisions", 0)),
                    "executed": int(bucket.get("executed", 0)),
                    "closed": closed,
                    "wins": int(bucket.get("wins", 0)),
                    "losses": int(bucket.get("losses", 0)),
                    "winrate": (float(bucket.get("wins", 0)) / closed) if closed else 0.0,
                    "pnl": float(bucket.get("pnl", 0.0)),
                    "r": float(bucket.get("r", 0.0)),
                    "avg_r": (float(bucket.get("r", 0.0)) / closed) if closed else 0.0,
                    "avg_action": avg(bucket.get("actions", [])),
                    "avg_risk_action": avg(bucket.get("risk_actions", [])),
                    "avg_tp_scale": avg(bucket.get("tp_scales", [])),
                    "reversed": int(bucket.get("reversed", 0)),
                    "updates": int(updates.get(status, 0) or 0),
                    "baseline": to_float(baselines.get(status)) or 0.0,
                    "weight_count": len(weights_by_status.get(status, {})) if isinstance(weights_by_status.get(status), dict) else 0,
                    "tp_weight_count": len(tp_weights.get(status, {})) if isinstance(tp_weights.get(status), dict) else 0,
                }
            )
        return sorted(rows, key=lambda row: (row["status"] != "accepted", row["status"]))

    @staticmethod
    def _strategy_pockets(
        rewards: list[dict[str, Any]],
        decision_lookup: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        buckets: dict[str, dict[str, Any]] = {}
        for row in rewards:
            decision = decision_lookup.get(str(row.get("decision_id") or ""), {})
            key = f"{status_key(row, decision)}/{strategy_symbol_key(row, decision)}"
            bucket = buckets.setdefault(
                key,
                {
                    "pocket": key,
                    "closed": 0,
                    "wins": 0,
                    "losses": 0,
                    "r": 0.0,
                    "pnl": 0.0,
                    "actions": [],
                    "risk_actions": [],
                    "tp_scales": [],
                },
            )
            reward_r = get_reward_r(row) or 0.0
            pnl = get_pnl(row) or 0.0
            bucket["closed"] += 1
            bucket["r"] += reward_r
            bucket["pnl"] += pnl
            action = get_action(row, decision)
            if action is not None:
                bucket["actions"].append(action)
            risk_action = get_risk_action(row, decision)
            if risk_action is not None:
                bucket["risk_actions"].append(risk_action)
            tp_scale = get_tp_scale(row, decision)
            if tp_scale is not None:
                bucket["tp_scales"].append(tp_scale)
            if pnl > 0 or (pnl == 0 and reward_r > 0):
                bucket["wins"] += 1
            elif pnl < 0 or (pnl == 0 and reward_r < 0):
                bucket["losses"] += 1
        rows = []
        for bucket in buckets.values():
            closed = int(bucket["closed"])
            rows.append(
                {
                    "pocket": bucket["pocket"],
                    "closed": closed,
                    "wins": int(bucket["wins"]),
                    "losses": int(bucket["losses"]),
                    "r": float(bucket["r"]),
                    "pnl": float(bucket["pnl"]),
                    "winrate": (float(bucket["wins"]) / closed) if closed else 0.0,
                    "avg_r": (float(bucket["r"]) / closed) if closed else 0.0,
                    "avg_action": avg(bucket["actions"]),
                    "avg_risk_action": avg(bucket["risk_actions"]),
                    "avg_tp_scale": avg(bucket["tp_scales"]),
                }
            )
        return {
            "best": sorted(rows, key=lambda row: row["r"], reverse=True)[:10],
            "worst": sorted(rows, key=lambda row: row["r"])[:10],
            "all_count": len(rows),
        }

    @staticmethod
    def _example_session(row: dict[str, Any]) -> str:
        market = row.get("market_context") if isinstance(row.get("market_context"), dict) else {}
        regime = market.get("regime") if isinstance(market.get("regime"), dict) else {}
        session = str(regime.get("session_tag") or "").strip().lower()
        return session or "unknown"

    def _safety(
        self,
        examples: list[dict[str, Any]],
        rewards: list[dict[str, Any]],
        decision_lookup: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        sorted_rewards = sorted(rewards, key=lambda row: ts_of(row) or datetime.min.replace(tzinfo=UTC))
        streaks: dict[str, int] = defaultdict(int)
        for row in sorted_rewards:
            decision = decision_lookup.get(str(row.get("decision_id") or ""), {})
            key = f"{status_key(row, decision)}/{strategy_symbol_key(row, decision)}"
            reward_r = get_reward_r(row) or 0.0
            pnl = get_pnl(row) or 0.0
            if pnl < 0 or (pnl == 0 and reward_r < 0):
                streaks[key] += 1
            else:
                streaks[key] = 0
        loss_streaks = [
            {"pocket": key, "loss_streak": value}
            for key, value in streaks.items()
            if value > 0
        ]
        loss_streaks.sort(key=lambda row: row["loss_streak"], reverse=True)

        recent_rewards = sorted_rewards[-80:]
        degraded = []
        pocket_recent: dict[str, list[float]] = defaultdict(list)
        for row in recent_rewards:
            decision = decision_lookup.get(str(row.get("decision_id") or ""), {})
            pocket_recent[f"{status_key(row, decision)}/{strategy_symbol_key(row, decision)}"].append(get_reward_r(row) or 0.0)
        for key, values in pocket_recent.items():
            losses = sum(1 for value in values if value < 0)
            total = sum(values)
            if len(values) >= 3 and (total <= -0.5 or losses >= 3):
                degraded.append(
                    {
                        "pocket": key,
                        "closed": len(values),
                        "r": total,
                        "avg_r": total / len(values),
                        "losses": losses,
                    }
                )
        degraded.sort(key=lambda row: row["r"])

        slippage_values = []
        for row in recent_rewards[-20:]:
            value = to_float(row.get("entry_slippage_bps"))
            if value is not None:
                slippage_values.append(value)

        session_values: dict[str, list[float]] = defaultdict(list)
        for row in examples[-160:]:
            reward_r = get_reward_r(row)
            if reward_r is not None:
                session_values[self._example_session(row)].append(reward_r)
        session_decay = []
        for session, values in session_values.items():
            if len(values) < 8:
                continue
            split = len(values) // 2
            prev = values[:split]
            recent = values[split:]
            session_decay.append(
                {
                    "session": session,
                    "count": len(values),
                    "previous_avg_r": avg(prev),
                    "recent_avg_r": avg(recent),
                    "delta_r": avg(recent) - avg(prev),
                }
            )
        session_decay.sort(key=lambda row: row["delta_r"])

        return {
            "loss_streaks": loss_streaks[:8],
            "degraded_pockets": degraded[:8],
            "recent_slippage_bps_avg": avg(slippage_values),
            "recent_slippage_bps_abs_avg": avg([abs(value) for value in slippage_values]),
            "session_decay": session_decay[:8],
        }

    @staticmethod
    def _shadow_policies(decisions: list[dict[str, Any]]) -> dict[str, Any]:
        rows = sorted(decisions, key=lambda row: ts_of(row) or datetime.min.replace(tzinfo=UTC), reverse=True)
        buckets: dict[str, dict[str, Any]] = {}
        latest = None
        for row in rows[:120]:
            agent = row.get("agent") if isinstance(row.get("agent"), dict) else {}
            shadows = agent.get("shadow_policies") if isinstance(agent.get("shadow_policies"), dict) else {}
            if shadows and latest is None:
                latest = {
                    "decision_id": row.get("decision_id"),
                    "symbol": row.get("symbol"),
                    "strategy": row.get("strategy"),
                    "status": status_key(row),
                    "feature_hash": agent.get("feature_hash"),
                    "policies": shadows,
                    "top_contributors": agent.get("top_contributors") if isinstance(agent.get("top_contributors"), list) else [],
                }
            for name, policy in shadows.items():
                if not isinstance(policy, dict):
                    continue
                bucket = buckets.setdefault(name, {"policy": name, "count": 0, "actions": [], "tp_scales": []})
                bucket["count"] += 1
                action = to_float(policy.get("action"))
                if action is not None:
                    bucket["actions"].append(action)
                    bucket.setdefault("risk_actions", []).append(abs(action))
                tp_scale = to_float(policy.get("tp_scale"))
                if tp_scale is not None:
                    bucket["tp_scales"].append(tp_scale)
        summary = []
        for bucket in buckets.values():
            summary.append(
                {
                    "policy": bucket["policy"],
                    "count": int(bucket["count"]),
                    "avg_action": avg(bucket["actions"]),
                    "avg_risk_action": avg(bucket.get("risk_actions", [])),
                    "avg_tp_scale": avg(bucket["tp_scales"]),
                }
            )
        return {
            "summary": sorted(summary, key=lambda row: row["policy"]),
            "latest": latest,
        }

    @staticmethod
    def _latest_feature_info(decisions: list[dict[str, Any]]) -> dict[str, Any] | None:
        rows = sorted(decisions, key=lambda row: ts_of(row) or datetime.min.replace(tzinfo=UTC), reverse=True)
        for row in rows:
            agent = row.get("agent") if isinstance(row.get("agent"), dict) else {}
            feature_hash = agent.get("feature_hash")
            top = agent.get("top_contributors") if isinstance(agent.get("top_contributors"), list) else []
            if feature_hash or top:
                return {
                    "decision_id": row.get("decision_id"),
                    "symbol": row.get("symbol"),
                    "strategy": row.get("strategy"),
                    "status": status_key(row),
                    "feature_hash": feature_hash,
                    "top_contributors": top[:10],
                    "tp_top_contributors": (
                        agent.get("tp_top_contributors")[:10]
                        if isinstance(agent.get("tp_top_contributors"), list)
                        else []
                    ),
                }
        return None


DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RL Sidecar Dashboard</title>
  <style>
    :root {
      --bg: #f6f7f9;
      --surface: #ffffff;
      --surface-2: #eef2f6;
      --line: #d8dee8;
      --text: #18202a;
      --muted: #667085;
      --green: #13795b;
      --green-bg: #dff5ec;
      --red: #b42318;
      --red-bg: #fde7e5;
      --amber: #9a5b00;
      --amber-bg: #fff0cc;
      --blue: #1f5fbf;
      --blue-bg: #e4efff;
      --violet: #6941c6;
      --violet-bg: #eee8ff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--text);
      background: var(--bg);
      font: 13px/1.45 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header.topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 16px 20px;
      border-bottom: 1px solid var(--line);
      background: var(--surface);
      position: sticky;
      top: 0;
      z-index: 5;
    }
    h1, h2, h3 { margin: 0; letter-spacing: 0; }
    h1 { font-size: 20px; font-weight: 750; }
    h2 { font-size: 16px; font-weight: 700; }
    h3 { font-size: 13px; font-weight: 700; color: var(--muted); text-transform: uppercase; }
    button, select {
      height: 32px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--surface);
      color: var(--text);
      padding: 0 10px;
      font: inherit;
    }
    button { cursor: pointer; }
    .controls { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    .muted { color: var(--muted); }
    .mono { font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace; }
    main { padding: 16px; max-width: 2040px; margin: 0 auto; }
    .compare-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 14px;
    }
    .panel {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    .panel-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfe;
    }
    .panel-body { padding: 14px; }
    .bot-layout {
      display: grid;
      grid-template-columns: minmax(390px, 0.92fr) minmax(520px, 1.08fr);
      gap: 14px;
    }
    .stack { display: grid; gap: 14px; }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px;
      min-height: 64px;
      background: #fff;
    }
    .metric label {
      display: block;
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      margin-bottom: 5px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .metric strong { font-size: 18px; font-weight: 750; }
    .chips { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
    .chip {
      display: inline-flex;
      align-items: center;
      height: 24px;
      padding: 0 8px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: var(--surface-2);
      white-space: nowrap;
      font-size: 12px;
    }
    .chip.good { color: var(--green); background: var(--green-bg); border-color: #9dddc6; }
    .chip.bad { color: var(--red); background: var(--red-bg); border-color: #f3b0aa; }
    .chip.warn { color: var(--amber); background: var(--amber-bg); border-color: #f2cb75; }
    .chip.info { color: var(--blue); background: var(--blue-bg); border-color: #aacbff; }
    .section-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 8px;
    }
    table {
      width: max-content;
      min-width: 100%;
      border-collapse: collapse;
    }
    th, td {
      padding: 7px 8px;
      border-bottom: 1px solid var(--line);
      text-align: right;
      vertical-align: middle;
      white-space: nowrap;
    }
    th:first-child, td:first-child {
      text-align: left;
      min-width: 220px;
      max-width: 520px;
      white-space: nowrap;
    }
    th { color: var(--muted); font-size: 11px; text-transform: uppercase; font-weight: 700; background: #fbfcfe; }
    tr:last-child td { border-bottom: 0; }
    .table-wrap {
      max-width: 100%;
      overflow-x: auto;
      overflow-y: hidden;
      border: 1px solid var(--line);
      border-radius: 6px;
      scrollbar-gutter: stable;
    }
    .spark {
      width: 100%;
      height: 98px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfcfe;
      display: block;
    }
    .bar-grid {
      display: grid;
      grid-template-columns: repeat(21, minmax(8px, 1fr));
      gap: 3px;
      height: 72px;
      align-items: end;
      padding: 8px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfcfe;
    }
    .bar {
      min-height: 2px;
      border-radius: 4px 4px 0 0;
      background: var(--blue);
    }
    .bar.neg { background: var(--red); }
    .split {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 12px;
    }
    .split > div, .stack > div, .panel-body > div { min-width: 0; }
    .small-list {
      display: grid;
      gap: 8px;
    }
    .rowline {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      border-bottom: 1px solid var(--line);
      padding-bottom: 7px;
    }
    .rowline:last-child { border-bottom: 0; padding-bottom: 0; }
    .pos { color: var(--green); }
    .neg { color: var(--red); }
    .empty {
      color: var(--muted);
      border: 1px dashed var(--line);
      border-radius: 6px;
      padding: 14px;
      text-align: center;
      background: #fbfcfe;
    }
    .tabs {
      display: flex;
      gap: 4px;
      background: var(--surface-2);
      padding: 3px;
      border-radius: 7px;
      border: 1px solid var(--line);
    }
    .tab {
      border: 0;
      background: transparent;
      height: 28px;
      border-radius: 5px;
    }
    .tab.active { background: var(--surface); box-shadow: 0 1px 2px rgba(16, 24, 40, 0.08); }
    @media (max-width: 1200px) {
      .compare-grid, .bot-layout, .split { grid-template-columns: 1fr; }
      .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 700px) {
      header.topbar { align-items: flex-start; flex-direction: column; }
      main { padding: 10px; }
      .metric-grid { grid-template-columns: 1fr; }
      th, td { padding: 6px; }
    }
  </style>
</head>
<body>
  <header class="topbar">
    <div>
      <h1>RL Sidecar Dashboard</h1>
      <div class="muted">Normal and Matrix execution agents</div>
    </div>
    <div class="controls">
      <div class="tabs" aria-label="reward window">
        <button class="tab active" data-window="24h">24h</button>
        <button class="tab" data-window="7d">7d</button>
        <button class="tab" data-window="all">All</button>
      </div>
      <button id="refreshBtn" title="Refresh dashboard">Refresh</button>
      <span id="lastRefresh" class="muted mono">loading</span>
    </div>
  </header>
  <main>
    <div id="error"></div>
    <section class="panel" style="margin-bottom: 14px;">
      <div class="panel-header">
        <h2>Sidecar Compare</h2>
        <div id="logRoot" class="muted mono"></div>
      </div>
      <div class="panel-body" id="compare"></div>
    </section>
    <div id="bots" class="stack"></div>
  </main>
  <script>
    const state = { data: null, window: '24h' };
    const $ = (id) => document.getElementById(id);
    const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (ch) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
    const num = (value, digits = 2) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : '-';
    const pct = (value) => Number.isFinite(Number(value)) ? (Number(value) * 100).toFixed(1) + '%' : '-';
    const money = (value) => (Number(value) >= 0 ? '+' : '') + num(value, 2);
    const rfmt = (value) => (Number(value) >= 0 ? '+' : '') + num(value, 3) + 'R';
    const clsNum = (value) => Number(value) < 0 ? 'neg' : 'pos';
    const age = (seconds) => {
      if (!Number.isFinite(Number(seconds))) return '-';
      const s = Math.max(0, Number(seconds));
      if (s < 60) return Math.round(s) + 's';
      if (s < 3600) return Math.round(s / 60) + 'm';
      if (s < 86400) return Math.round(s / 3600) + 'h';
      return Math.round(s / 86400) + 'd';
    };
    function freshnessChip(bot) {
      const s = Number(bot.freshness?.latest_decision_age_s);
      if (!Number.isFinite(s)) return '<span class="chip warn">no signals</span>';
      if (s < 900) return '<span class="chip good">fresh ' + age(s) + '</span>';
      if (s < 7200) return '<span class="chip warn">quiet ' + age(s) + '</span>';
      return '<span class="chip bad">stale ' + age(s) + '</span>';
    }
    function serviceChip(bot) {
      const running = bot.container?.running;
      const label = running ? 'running' : (bot.container?.status || 'unknown');
      let kind = running ? 'good' : 'warn';
      if (!running && String(label).startsWith('logs active')) kind = 'info';
      return '<span class="chip ' + kind + '">' + esc(label) + '</span>';
    }
    function metric(label, value, sub = '') {
      return '<div class="metric"><label>' + esc(label) + '</label><strong>' + value + '</strong><div class="muted">' + sub + '</div></div>';
    }
    function sparkline(points) {
      if (!points || points.length < 2) return '<div class="empty">No reward curve yet</div>';
      const values = points.map(p => Number(p.cumulative_r)).filter(Number.isFinite);
      if (values.length < 2) return '<div class="empty">No reward curve yet</div>';
      const min = Math.min(...values);
      const max = Math.max(...values);
      const span = Math.max(0.000001, max - min);
      const coords = values.map((v, i) => {
        const x = (i / Math.max(1, values.length - 1)) * 100;
        const y = 86 - ((v - min) / span) * 72;
        return x.toFixed(2) + ',' + y.toFixed(2);
      }).join(' ');
      const last = values[values.length - 1];
      const color = last >= values[0] ? '#13795b' : '#b42318';
      return '<svg class="spark" viewBox="0 0 100 100" preserveAspectRatio="none" role="img">' +
        '<line x1="0" y1="86" x2="100" y2="86" stroke="#d8dee8" stroke-width="1"/>' +
        '<polyline fill="none" stroke="' + color + '" stroke-width="2.4" vector-effect="non-scaling-stroke" points="' + coords + '"/>' +
        '</svg>';
    }
    function dailyBars(rows) {
      if (!rows || !rows.length) return '<div class="empty">No daily rewards yet</div>';
      const maxAbs = Math.max(...rows.map(r => Math.abs(Number(r.r) || 0)), 0.000001);
      return '<div class="bar-grid">' + rows.map(row => {
        const value = Number(row.r) || 0;
        const h = Math.max(2, Math.abs(value) / maxAbs * 58);
        return '<div class="bar ' + (value < 0 ? 'neg' : '') + '" title="' + esc(row.day + ' ' + rfmt(value)) + '" style="height:' + h.toFixed(1) + 'px"></div>';
      }).join('') + '</div>';
    }
    function table(headers, rows, emptyText = 'No rows', minWidth = 760) {
      if (!rows || !rows.length) return '<div class="empty">' + esc(emptyText) + '</div>';
      return '<div class="table-wrap"><table style="min-width:' + Number(minWidth || 760) + 'px"><thead><tr>' + headers.map(h => '<th>' + esc(h.label) + '</th>').join('') +
        '</tr></thead><tbody>' + rows.map(row => '<tr>' + headers.map(h => '<td>' + h.render(row) + '</td>').join('') + '</tr>').join('') +
        '</tbody></table></div>';
    }
    function renderCompare(data) {
      $('logRoot').textContent = data.log_root || '';
      const rows = data.bots.map(bot => {
        const w = bot.rewards?.[state.window] || {};
        return {
          bot: bot.name,
          status: serviceChip(bot) + ' ' + freshnessChip(bot),
          decisions: bot.counts?.decisions || 0,
          active: bot.counts?.active_trades || 0,
          closed: w.closed || 0,
          r: w.r || 0,
          pnl: w.pnl || 0,
          winrate: w.winrate || 0,
          action: w.avg_action || 0,
          riskAction: w.avg_risk_action || 0,
          tpScale: w.avg_tp_scale || 0,
          reversed: bot.counts?.reversed_decisions || 0,
          updates: bot.agent?.reward_updates || 0,
        };
      });
      $('compare').innerHTML = table([
        {label:'Bot', render:r => esc(r.bot)},
        {label:'State', render:r => r.status},
        {label:'Signals', render:r => String(r.decisions)},
        {label:'Active', render:r => String(r.active)},
        {label:'Closed', render:r => String(r.closed)},
        {label:'Actual R', render:r => '<span class="' + clsNum(r.r) + '">' + rfmt(r.r) + '</span>'},
        {label:'PnL', render:r => '<span class="' + clsNum(r.pnl) + '">' + money(r.pnl) + '</span>'},
        {label:'Winrate', render:r => pct(r.winrate)},
        {label:'Avg action', render:r => num(r.action, 3)},
        {label:'Avg size', render:r => num(r.riskAction, 3)},
        {label:'Avg TP', render:r => num(r.tpScale, 1)},
        {label:'Reverse', render:r => String(r.reversed)},
        {label:'Updates', render:r => String(r.updates)},
      ], rows, 'No bots', 1320);
    }
    function renderBot(bot) {
      const w = bot.rewards?.[state.window] || {};
      const all = bot.rewards?.all || {};
      const statusRows = bot.status_diagnostics || [];
      const agent = bot.agent || {};
      const replay = bot.replay;
      const execution = bot.counts?.execution || {};
      const statusCounts = bot.counts?.status || {};
      const recentCurve = bot.rewards?.curve || [];
      const shadowRows = bot.shadow?.summary || [];
      const latestFeatures = bot.latest_features;
      const safety = bot.safety || {};
      const headerChips = [
        serviceChip(bot),
        freshnessChip(bot),
        '<span class="chip info">active ' + esc(bot.counts?.active_trades || 0) + '</span>',
        '<span class="chip">model ' + esc(agent.reward_updates || 0) + ' updates</span>'
      ].join('');
      const metrics = [
        metric('Window actual R', '<span class="' + clsNum(w.r) + '">' + rfmt(w.r || 0) + '</span>', (w.closed || 0) + ' closed'),
        metric('Window PnL', '<span class="' + clsNum(w.pnl) + '">' + money(w.pnl || 0) + '</span>', pct(w.winrate || 0) + ' winrate'),
        metric('Window sizing', num(w.avg_risk_action || 0, 3), 'signed ' + num(w.avg_action || 0, 3) + ' / TP ' + num(w.avg_tp_scale || 0, 1)),
        metric('All-decision sizing', num(bot.actions?.avg_risk_action || 0, 3), 'signed ' + num(bot.actions?.avg_action || 0, 3)),
        metric('All actual R', '<span class="' + clsNum(all.r) + '">' + rfmt(all.r || 0) + '</span>', (all.closed || 0) + ' closed'),
        metric('Signals', String(bot.counts?.decisions || 0), Object.entries(statusCounts).map(([k,v]) => esc(k) + ' ' + v).join(', ')),
        metric('Execution', String(execution.executed || 0), 'reverse ' + (bot.counts?.reversed_decisions || 0)),
        metric('Weights', String(
          (agent.global_weight_count || 0) +
          (agent.global_tp_weight_count || 0) +
          (agent.global_side_weight_count || 0) +
          Object.values(agent.status_weight_counts || {}).reduce((a,b)=>a+Number(b||0),0) +
          Object.values(agent.status_tp_weight_counts || {}).reduce((a,b)=>a+Number(b||0),0) +
          Object.values(agent.status_side_weight_counts || {}).reduce((a,b)=>a+Number(b||0),0)
        ), 'stats ' + (agent.stat_count || 0)),
        metric('History', String(agent.trade_history_rows || 0), 'signals ' + (agent.signal_history_rows || 0)),
      ].join('');
      const statusTable = table([
        {label:'Status', render:r => esc(r.status)},
        {label:'Sig', render:r => String(r.decisions)},
        {label:'Exe', render:r => String(r.executed)},
        {label:'Closed', render:r => String(r.closed)},
        {label:'R', render:r => '<span class="' + clsNum(r.r) + '">' + rfmt(r.r) + '</span>'},
        {label:'PnL', render:r => '<span class="' + clsNum(r.pnl) + '">' + money(r.pnl) + '</span>'},
        {label:'Avg R', render:r => '<span class="' + clsNum(r.avg_r) + '">' + rfmt(r.avg_r) + '</span>'},
        {label:'W/L', render:r => esc((r.wins || 0) + '/' + (r.losses || 0))},
        {label:'Action', render:r => num(r.avg_action, 3)},
        {label:'Size', render:r => num(r.avg_risk_action, 3)},
        {label:'Rev', render:r => String(r.reversed || 0)},
        {label:'TP', render:r => num(r.avg_tp_scale, 1)},
        {label:'Updates', render:r => String(r.updates || 0)},
        {label:'Base', render:r => '<span class="' + clsNum(r.baseline) + '">' + rfmt(r.baseline) + '</span>'},
      ], statusRows, 'No status diagnostics', 1120);
      const pocketHeaders = [
        {label:'Pocket', render:r => esc(r.pocket)},
        {label:'Closed', render:r => String(r.closed)},
        {label:'R', render:r => '<span class="' + clsNum(r.r) + '">' + rfmt(r.r) + '</span>'},
        {label:'PnL', render:r => '<span class="' + clsNum(r.pnl) + '">' + money(r.pnl) + '</span>'},
        {label:'Avg', render:r => '<span class="' + clsNum(r.avg_r) + '">' + rfmt(r.avg_r) + '</span>'},
        {label:'W/L', render:r => esc((r.wins || 0) + '/' + (r.losses || 0))},
        {label:'Action', render:r => num(r.avg_action, 3)},
        {label:'Size', render:r => num(r.avg_risk_action, 3)},
        {label:'TP', render:r => num(r.avg_tp_scale, 1)},
      ];
      const shadowTable = table([
        {label:'Policy', render:r => esc(r.policy)},
        {label:'Count', render:r => String(r.count)},
        {label:'Action', render:r => num(r.avg_action, 3)},
        {label:'Risk', render:r => num(r.avg_risk_action, 3)},
        {label:'TP', render:r => num(r.avg_tp_scale, 1)},
      ], shadowRows, 'Shadow policy rows will appear on new decisions', 620);
      const replayHtml = replay ? table([
        {label:'Policy', render:r => esc(r.name)},
        {label:'Proxy R', render:r => '<span class="' + clsNum(r.proxy_r) + '">' + rfmt(r.proxy_r) + '</span>'},
        {label:'Avg R', render:r => '<span class="' + clsNum(r.proxy_avg_r) + '">' + rfmt(r.proxy_avg_r) + '</span>'},
        {label:'Action', render:r => num(r.avg_action, 3)},
        {label:'Corr', render:r => num(r.action_reward_corr, 3)},
      ], Object.entries(replay.policies || {}).map(([name, row]) => ({name, ...row})), 'No replay report', 620) : '<div class="empty">No replay report</div>';
      const safetyHtml = [
        '<div class="small-list">',
        '<div class="rowline"><span>Recent slippage</span><strong>' + num(safety.recent_slippage_bps_avg || 0, 2) + ' bps / abs ' + num(safety.recent_slippage_bps_abs_avg || 0, 2) + '</strong></div>',
        (safety.loss_streaks || []).slice(0, 4).map(r => '<div class="rowline"><span>' + esc(r.pocket) + '</span><strong class="neg">x' + esc(r.loss_streak) + '</strong></div>').join('') || '<div class="empty">No active loss streaks</div>',
        '</div>'
      ].join('');
      const featuresHtml = latestFeatures ? [
        '<div class="small-list">',
        '<div class="rowline"><span>' + esc(latestFeatures.strategy) + ' / ' + esc(latestFeatures.symbol) + '</span><strong class="mono">' + esc(latestFeatures.feature_hash || '-') + '</strong></div>',
        ...(latestFeatures.top_contributors || []).slice(0, 7).map(row => '<div class="rowline"><span class="mono">' + esc(row.feature) + '</span><strong class="' + clsNum(row.contribution) + '">' + num(row.contribution, 4) + '</strong></div>'),
        '</div>'
      ].join('') : '<div class="empty">Feature contributors will appear on new decisions</div>';
      return '<section class="panel">' +
        '<div class="panel-header"><div><h2>' + esc(bot.name) + '</h2><div class="muted mono">' + esc(bot.service) + '</div></div><div class="chips">' + headerChips + '</div></div>' +
        '<div class="panel-body bot-layout">' +
          '<div class="stack">' +
            '<div class="metric-grid">' + metrics + '</div>' +
            '<div><div class="section-title"><h3>Cumulative actual R</h3><span class="muted">' + esc(state.window) + ' selected</span></div>' + sparkline(recentCurve) + '</div>' +
            '<div><div class="section-title"><h3>Daily reward</h3><span class="muted">last 21 days</span></div>' + dailyBars(bot.rewards?.daily || []) + '</div>' +
            '<div><div class="section-title"><h3>Status heads</h3></div>' + statusTable + '</div>' +
          '</div>' +
          '<div class="stack">' +
            '<div class="split"><div><div class="section-title"><h3>Best pockets</h3></div>' + table(pocketHeaders, bot.pockets?.best || [], 'No pockets', 980) + '</div><div><div class="section-title"><h3>Worst pockets</h3></div>' + table(pocketHeaders, bot.pockets?.worst || [], 'No pockets', 980) + '</div></div>' +
            '<div class="split"><div><div class="section-title"><h3>Safety</h3></div>' + safetyHtml + '</div><div><div class="section-title"><h3>Shadow policies</h3></div>' + shadowTable + '</div></div>' +
            '<div class="split"><div><div class="section-title"><h3>Feature drivers</h3></div>' + featuresHtml + '</div><div><div class="section-title"><h3>Replay</h3></div>' + replayHtml + '</div></div>' +
          '</div>' +
        '</div></section>';
    }
    function render(data) {
      state.data = data;
      $('lastRefresh').textContent = new Date(data.generated_at).toLocaleTimeString();
      renderCompare(data);
      $('bots').innerHTML = data.bots.map(renderBot).join('');
    }
    async function load() {
      try {
        $('error').innerHTML = '';
        const res = await fetch('/api/summary', {cache: 'no-store'});
        if (!res.ok) throw new Error('HTTP ' + res.status);
        render(await res.json());
      } catch (err) {
        $('error').innerHTML = '<div class="panel" style="margin-bottom:14px"><div class="panel-body"><span class="chip bad">Load failed</span> <span class="muted">' + esc(err.message) + '</span></div></div>';
      }
    }
    document.querySelectorAll('.tab').forEach(btn => {
      btn.addEventListener('click', () => {
        state.window = btn.dataset.window;
        document.querySelectorAll('.tab').forEach(x => x.classList.toggle('active', x === btn));
        if (state.data) render(state.data);
      });
    });
    $('refreshBtn').addEventListener('click', load);
    load();
    setInterval(load, 15000);
  </script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    data: DashboardData

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status.value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self._send(HTTPStatus.OK, DASHBOARD_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/summary":
            payload = self.data.summary()
            body = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
            self._send(HTTPStatus.OK, body, "application/json; charset=utf-8")
            return
        if parsed.path == "/api/bot":
            query = parse_qs(parsed.query)
            key = (query.get("key") or [""])[0]
            payload = self.data.summary()
            bot = next((row for row in payload["bots"] if row["key"] == key), None)
            if bot is None:
                self._send(HTTPStatus.NOT_FOUND, b'{"error":"bot not found"}', "application/json; charset=utf-8")
                return
            body = json.dumps(bot, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
            self._send(HTTPStatus.OK, body, "application/json; charset=utf-8")
            return
        self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain; charset=utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("RL_DASHBOARD_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("RL_DASHBOARD_PORT", "8765")))
    parser.add_argument("--log-root", type=Path, default=Path(os.environ.get("RL_DASHBOARD_LOG_ROOT", str(default_log_root()))))
    parser.add_argument("--project-root", type=Path, default=repo_root())
    args = parser.parse_args()

    DashboardHandler.data = DashboardData(
        log_root=args.log_root.resolve(),
        project_root=args.project_root.resolve(),
    )
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"RL dashboard listening on http://{args.host}:{args.port}")
    print(f"Reading logs from {args.log_root.resolve()}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
