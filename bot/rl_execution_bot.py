#!/usr/bin/env python3
"""
Reinforcement-learning execution sidecar for the trading bot.

The normal bot remains the primary execution path.  This service receives every
accepted and rejected signal over REST, chooses one continuous action a in
[0, 1], and optionally opens the same setup on a separate Bybit demo account
with risk a * RL_DEFAULT_RISK_USDT.
"""
from __future__ import annotations

import json
import logging
import math
import os
import queue
import random
import signal
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from pybit.unified_trading import HTTP, WebSocket


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


HOST = os.environ.get("RL_HOST", "0.0.0.0")
PORT = env_int("RL_PORT", 8090)
LOG_DIR = os.environ.get("RL_LOG_DIR", "/app/rl")
STATE_PATH = os.environ.get("RL_STATE_PATH", os.path.join(LOG_DIR, "runtime_state.json"))
MODEL_PATH = os.environ.get("RL_MODEL_PATH", os.path.join(LOG_DIR, "agent_state.json"))
DECISIONS_PATH = os.environ.get("RL_DECISIONS_PATH", os.path.join(LOG_DIR, "decisions.jsonl"))
REWARDS_PATH = os.environ.get("RL_REWARDS_PATH", os.path.join(LOG_DIR, "rewards.jsonl"))
TRAINING_EXAMPLES_PATH = os.environ.get(
    "RL_TRAINING_EXAMPLES_PATH",
    os.path.join(LOG_DIR, "training_examples.jsonl"),
)
PRETRAIN_PATH = os.environ.get("RL_PRETRAIN_PATH", "").strip()
PRETRAIN_ON_START = env_bool("RL_PRETRAIN_ON_START", False)
EXECUTION_QUEUE_SIZE = env_int("RL_EXECUTION_QUEUE_SIZE", 2000)

BYBIT_DEMO = env_bool("RL_BYBIT_DEMO", True)
TRADING_ENABLED = env_bool("RL_TRADING_ENABLED", False)
API_KEY = os.environ.get("RL_BYBIT_API_KEY", "").strip()
API_SECRET = os.environ.get("RL_BYBIT_API_SECRET", "").strip()
ENABLE_PRIVATE_ORDER_WS = env_bool("RL_ENABLE_PRIVATE_ORDER_WS", True)
REWARD_POLL_SECONDS = env_float("RL_REWARD_POLL_SECONDS", 30.0)

DEFAULT_RISK_USDT = env_float("RL_DEFAULT_RISK_USDT", 100.0)
MIN_ACTION_TO_TRADE = env_float("RL_MIN_ACTION_TO_TRADE", 0.05)
INITIAL_ACTION = env_float("RL_INITIAL_ACTION", 0.10)
EXPLORATION_RATE = env_float("RL_EXPLORATION_RATE", 0.02)
EXPLORATION_MAX_ACTION = env_float("RL_EXPLORATION_MAX_ACTION", 0.25)
LEARNING_RATE = env_float("RL_LEARNING_RATE", 0.03)
WEIGHT_DECAY = env_float("RL_WEIGHT_DECAY", 0.0001)

TAKER_FEE_RATE = env_float("RL_TAKER_FEE_RATE", env_float("TAKER_FEE_RATE", 0.00055))
MIN_STOP_DISTANCE_PCT = env_float("RL_MIN_STOP_DISTANCE_PCT", env_float("MIN_STOP_DISTANCE_PCT", 0.001))
MAX_FEE_TO_PRICE_RISK = env_float("RL_MAX_FEE_TO_PRICE_RISK", env_float("MAX_FEE_TO_PRICE_RISK", 0.25))
ALLOW_MIN_QTY_OVERRISK = env_bool("RL_ALLOW_MIN_QTY_OVERRISK", False)


os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-5s] [rl-exec] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, "rl_execution_bot.log")),
    ],
)
log = logging.getLogger("rl_exec")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def now_ms() -> int:
    return int(time.time() * 1000)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, deque)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def clamp(value: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, value))


def logit(probability: float) -> float:
    p = clamp(probability, 1e-4, 1.0 - 1e-4)
    return math.log(p / (1.0 - p))


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def round_to_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    precision = max(0, round(-math.log10(step)))
    return round(round(value / step) * step, precision)


def floor_to_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    precision = max(0, round(-math.log10(step)))
    return round(math.floor(value / step) * step, precision)


def ceil_to_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    precision = max(0, round(-math.log10(step)))
    return round(math.ceil(value / step) * step, precision)


def qty_to_str(value: float, step: float = 0.0) -> str:
    if step > 0:
        precision = max(0, round(-math.log10(step)))
        text = f"{value:.{precision}f}"
    else:
        text = f"{round(value, 8):.8f}".rstrip("0").rstrip(".")
    return text if text else "0"


def get_balance_metrics(http_client: HTTP) -> dict[str, float]:
    resp = http_client.get_wallet_balance(accountType="UNIFIED")
    row = resp.get("result", {}).get("list", [{}])[0]
    total_equity = float(row.get("totalEquity", 0) or 0)
    total_available = float(row.get("totalAvailableBalance", 0) or 0)

    usdt_equity = 0.0
    usdt_available = 0.0
    for coin in row.get("coin", []):
        if coin.get("coin") != "USDT":
            continue
        usdt_equity = float(coin.get("equity", 0) or 0)
        usdt_available = float(
            coin.get("availableToWithdraw")
            or coin.get("availableToBorrow")
            or 0
        )
        break

    return {
        "equity": usdt_equity if usdt_equity > 0 else total_equity,
        "available": min(usdt_available, total_available)
        if (usdt_available > 0 and total_available > 0)
        else (usdt_available if usdt_available > 0 else total_available),
    }


def get_instrument_info(http_client: HTTP, symbol: str) -> dict[str, Any]:
    resp = http_client.get_instruments_info(category="linear", symbol=symbol)
    items = resp.get("result", {}).get("list", [])
    if not items:
        return {}
    lot = items[0].get("lotSizeFilter", {})
    price = items[0].get("priceFilter", {})
    leverage = items[0].get("leverageFilter", {})
    min_leverage_raw = str(leverage.get("minLeverage", "1") or "1")
    max_leverage_raw = str(leverage.get("maxLeverage", "1") or "1")
    leverage_step_raw = str(leverage.get("leverageStep", "0.01") or "0.01")
    return {
        "status": str(items[0].get("status", "")),
        "qty_step": float(lot.get("qtyStep", "0.001")),
        "min_qty": float(lot.get("minOrderQty", "0.001")),
        "tick_size": float(price.get("tickSize", "0.01")),
        "min_leverage": float(min_leverage_raw),
        "min_leverage_raw": min_leverage_raw,
        "max_leverage": float(max_leverage_raw),
        "max_leverage_raw": max_leverage_raw,
        "leverage_step": float(leverage_step_raw),
        "leverage_step_raw": leverage_step_raw,
    }


def append_jsonl(path: str, row: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(jsonable(row), sort_keys=True) + "\n")


class ContextualRiskAgent:
    """Small online contextual policy for the continuous risk action."""

    def __init__(self) -> None:
        self.weights: dict[str, float] = {"__bias__": logit(INITIAL_ACTION)}
        self.stats: dict[str, dict[str, float]] = {}
        self.reward_baseline = 0.0
        self.reward_updates = 0
        self.lock = threading.Lock()

    @classmethod
    def load(cls, path: str) -> "ContextualRiskAgent":
        agent = cls()
        if not os.path.exists(path):
            return agent
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data.get("weights"), dict):
                agent.weights = {str(k): float(v) for k, v in data["weights"].items()}
            if isinstance(data.get("stats"), dict):
                agent.stats = {
                    str(k): {
                        "n": float(v.get("n", 0)),
                        "mean": float(v.get("mean", 0)),
                        "m2": float(v.get("m2", 0)),
                    }
                    for k, v in data["stats"].items()
                    if isinstance(v, dict)
                }
            agent.reward_baseline = float(data.get("reward_baseline", 0.0) or 0.0)
            agent.reward_updates = int(data.get("reward_updates", 0) or 0)
            log.info(
                "Loaded RL agent state from %s  weights=%d updates=%d baseline=%.4f",
                path,
                len(agent.weights),
                agent.reward_updates,
                agent.reward_baseline,
            )
        except Exception as exc:
            log.warning("Could not load RL agent state %s: %s", path, exc)
        return agent

    def save(self, path: str) -> None:
        tmp_path = f"{path}.tmp"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "weights": self.weights,
                    "stats": self.stats,
                    "reward_baseline": self.reward_baseline,
                    "reward_updates": self.reward_updates,
                    "saved_at": now_iso(),
                },
                fh,
                indent=2,
                sort_keys=True,
            )
        os.replace(tmp_path, path)

    @staticmethod
    def _sanitize_name(name: Any) -> str:
        text = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(name))
        return text[:96] or "unknown"

    @staticmethod
    def _parse_time(raw: Any) -> datetime | None:
        if not raw:
            return None
        try:
            text = str(raw).replace("Z", "+00:00")
            parsed = datetime.fromisoformat(text)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            return None

    def _update_stat(self, name: str, value: float) -> None:
        stat = self.stats.setdefault(name, {"n": 0.0, "mean": 0.0, "m2": 0.0})
        n = stat["n"] + 1.0
        delta = value - stat["mean"]
        mean = stat["mean"] + delta / n
        stat["m2"] += delta * (value - mean)
        stat["n"] = n
        stat["mean"] = mean

    def _normalise(self, name: str, value: float) -> float:
        stat = self.stats.get(name)
        if stat and stat.get("n", 0.0) >= 10:
            variance = stat["m2"] / max(1.0, stat["n"] - 1.0)
            std = math.sqrt(max(variance, 1e-12))
            return clamp((value - stat["mean"]) / std, -6.0, 6.0)
        if abs(value) <= 10.0:
            return clamp(value, -6.0, 6.0)
        return math.tanh(value / 10.0)

    def _numeric_features(self, payload: dict[str, Any]) -> dict[str, float]:
        out: dict[str, float] = {}

        def add_number(name: str, value: Any, *, log1p_abs: bool = False) -> None:
            number = to_float(value)
            if number is None:
                return
            if log1p_abs:
                sign = -1.0 if number < 0 else 1.0
                number = sign * math.log1p(abs(number))
            out[name] = number

        raw_features = payload.get("features")
        if isinstance(raw_features, dict):
            for name, value in raw_features.items():
                number = to_float(value)
                if number is not None:
                    out[f"src.{self._sanitize_name(name)}"] = number

        market = payload.get("market_context")
        if isinstance(market, dict):
            basis = market.get("basis") if isinstance(market.get("basis"), dict) else {}
            add_number("market.basis_pct", basis.get("basis_pct"))

            orderbook = market.get("orderbook") if isinstance(market.get("orderbook"), dict) else {}
            add_number("orderbook.spread_bps", orderbook.get("spread_bps"))
            add_number("orderbook.imbalance_top10", orderbook.get("imbalance_top10"))
            bid_depth = to_float(orderbook.get("bid_notional_top10"))
            ask_depth = to_float(orderbook.get("ask_notional_top10"))
            if bid_depth is not None:
                add_number("orderbook.bid_notional_top10_log", bid_depth, log1p_abs=True)
            if ask_depth is not None:
                add_number("orderbook.ask_notional_top10_log", ask_depth, log1p_abs=True)
            if bid_depth is not None and ask_depth is not None:
                add_number("orderbook.total_notional_top10_log", bid_depth + ask_depth, log1p_abs=True)

            flow = market.get("trade_flow_60s") if isinstance(market.get("trade_flow_60s"), dict) else {}
            add_number("flow.trade_count_60s", flow.get("trade_count"), log1p_abs=True)
            add_number("flow.notional_imbalance_60s", flow.get("notional_imbalance"))
            add_number("flow.buy_notional_60s_log", flow.get("buy_notional"), log1p_abs=True)
            add_number("flow.sell_notional_60s_log", flow.get("sell_notional"), log1p_abs=True)

            oi = market.get("open_interest") if isinstance(market.get("open_interest"), dict) else {}
            oi_current = to_float(oi.get("current"))
            add_number("oi.current_log", oi_current, log1p_abs=True)
            add_number("oi.zscore", oi.get("zscore"))
            for window in ("5m", "15m", "1h"):
                delta = to_float(oi.get(f"delta_{window}"))
                add_number(f"oi.delta_{window}_log", delta, log1p_abs=True)
                if oi_current not in (None, 0.0) and delta is not None:
                    add_number(f"oi.delta_{window}_pct", delta / oi_current)

            funding = market.get("funding") if isinstance(market.get("funding"), dict) else {}
            add_number("funding.current", funding.get("current"))
            add_number("funding.zscore", funding.get("zscore"))
            mins_to_funding = to_float(funding.get("minutes_to_next_funding"))
            if mins_to_funding is not None:
                add_number("funding.minutes_to_next", mins_to_funding)
                add_number("funding.hours_to_next", mins_to_funding / 60.0)

            regime = market.get("regime") if isinstance(market.get("regime"), dict) else {}
            for name in (
                "minutes_to_session_open",
                "minutes_to_session_close",
                "atr_pctile",
                "volume_ratio",
                "realized_vol_20",
                "realized_vol_96",
                "volume_percentile_200",
            ):
                add_number(f"regime.{name}", regime.get(name))

            plan = market.get("execution_plan") if isinstance(market.get("execution_plan"), dict) else {}
            add_number("execution.stop_distance_ticks_log", plan.get("stop_distance_ticks"), log1p_abs=True)
            add_number("execution.target_distance_ticks_log", plan.get("target_distance_ticks"), log1p_abs=True)

        setup = payload.get("setup") if isinstance(payload.get("setup"), dict) else {}
        entry = to_float(setup.get("entry"))
        stop = to_float(setup.get("stop_loss"))
        target = to_float(setup.get("take_profit"))
        trail_dist = to_float(setup.get("trail_dist"))
        if entry and entry > 0 and stop is not None:
            risk = abs(entry - stop)
            out["setup.stop_distance_pct"] = risk / entry
            if target is not None:
                out["setup.target_distance_pct"] = abs(target - entry) / entry
                if risk > 0:
                    out["setup.reward_risk"] = abs(target - entry) / risk
            if trail_dist is not None and risk > 0:
                out["setup.trail_to_risk"] = trail_dist / risk

        probability = to_float(payload.get("ml_probability"))
        threshold = to_float(payload.get("ml_threshold"))
        if probability is not None:
            out["ml_probability"] = probability
        if threshold is not None:
            out["ml_threshold"] = threshold
            if probability is not None:
                out["ml_edge_vs_threshold"] = probability - threshold

        out["status.accepted"] = 1.0 if payload.get("status") == "accepted" else 0.0
        out["status.rejected"] = 1.0 if payload.get("status") == "rejected" else 0.0

        entry_time = self._parse_time(setup.get("entry_time")) or self._parse_time(payload.get("sent_at"))
        if entry_time is not None:
            hour_angle = 2.0 * math.pi * (entry_time.hour + entry_time.minute / 60.0) / 24.0
            dow_angle = 2.0 * math.pi * entry_time.weekday() / 7.0
            out["time.hour_sin"] = math.sin(hour_angle)
            out["time.hour_cos"] = math.cos(hour_angle)
            out["time.dow_sin"] = math.sin(dow_angle)
            out["time.dow_cos"] = math.cos(dow_angle)

        return out

    def build_vector(self, payload: dict[str, Any]) -> dict[str, float]:
        vector = {"__bias__": 1.0}
        numeric = self._numeric_features(payload)
        for name, value in numeric.items():
            key = f"n:{name}"
            vector[key] = self._normalise(key, value)
            self._update_stat(key, value)

        for field in ("symbol", "strategy", "direction", "status"):
            value = payload.get(field)
            if value:
                vector[f"c:{field}:{self._sanitize_name(value).lower()}"] = 1.0

        setup = payload.get("setup") if isinstance(payload.get("setup"), dict) else {}
        exit_style = setup.get("exit_style")
        if exit_style:
            vector[f"c:exit_style:{self._sanitize_name(exit_style).lower()}"] = 1.0
        reason = payload.get("reason")
        if reason:
            reason_key = str(reason).split(":", 1)[0].lower()[:48]
            vector[f"c:reject_reason:{self._sanitize_name(reason_key)}"] = 1.0
        market = payload.get("market_context")
        regime = market.get("regime") if isinstance(market, dict) and isinstance(market.get("regime"), dict) else {}
        session_tag = regime.get("session_tag") if isinstance(regime, dict) else None
        if session_tag:
            vector[f"c:session:{self._sanitize_name(session_tag).lower()}"] = 1.0
        return vector

    def decide(self, payload: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        with self.lock:
            vector = self.build_vector(payload)
            score = sum(self.weights.get(k, 0.0) * v for k, v in vector.items())
            policy_action = sigmoid(score)
            action = policy_action
            explored = False
            if random.random() < EXPLORATION_RATE:
                explored = True
                action = random.uniform(0.0, clamp(EXPLORATION_MAX_ACTION, 0.0, 1.0))
            action = clamp(action, 0.0, 1.0)
            return action, {
                "policy_action": policy_action,
                "score": score,
                "explored": explored,
                "feature_vector": vector,
            }

    def learn(self, decision: dict[str, Any], reward: float) -> None:
        agent_data = decision.get("agent") if isinstance(decision.get("agent"), dict) else {}
        vector = agent_data.get("feature_vector") if isinstance(agent_data.get("feature_vector"), dict) else {}
        action = to_float(decision.get("action"))
        if not vector or action is None:
            return
        with self.lock:
            old_baseline = self.reward_baseline
            self.reward_updates += 1
            baseline_alpha = min(0.20, 2.0 / (self.reward_updates + 10.0))
            self.reward_baseline = (1.0 - baseline_alpha) * self.reward_baseline + baseline_alpha * reward
            advantage = clamp(reward - old_baseline, -5.0, 5.0)
            gradient_scale = LEARNING_RATE * advantage * max(0.05, action * (1.0 - action))
            for key, value in vector.items():
                x = to_float(value)
                if x is None:
                    continue
                old_weight = self.weights.get(str(key), 0.0)
                decayed = old_weight * (1.0 - WEIGHT_DECAY)
                self.weights[str(key)] = clamp(decayed + gradient_scale * x, -12.0, 12.0)

    def pretrain_from_path(self, path: str) -> int:
        if not path or not os.path.exists(path):
            return 0
        updates = 0
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                reward = to_float(row.get("reward_default_r", row.get("reward")))
                action = to_float(row.get("action"))
                payload = row.get("payload") if isinstance(row.get("payload"), dict) else row
                if reward is None or action is None:
                    continue
                _action, agent_info = self.decide(payload)
                decision = {"action": action, "agent": agent_info}
                self.learn(decision, reward)
                updates += 1
        return updates


class RLExecutionService:
    def __init__(self) -> None:
        self.agent = ContextualRiskAgent.load(MODEL_PATH)
        self.http: HTTP | None = None
        self.private_ws: WebSocket | None = None
        self.lock = threading.RLock()
        self.decisions: dict[str, dict[str, Any]] = {}
        self.event_to_decision: dict[str, str] = {}
        self.active_trades: dict[str, str] = {}
        self.order_to_decision: dict[str, str] = {}
        self.max_leverage_symbols: set[str] = set()
        self.claimed_closed_pnl_ids: set[str] = set()
        self.instrument_cache: dict[str, dict[str, Any]] = {}
        self.notified_exit_ids: set[str] = set()
        self.execution_queue: queue.Queue[str] = queue.Queue(maxsize=max(1, EXECUTION_QUEUE_SIZE))
        self.started_at = now_iso()
        self._load_runtime_state()
        threading.Thread(target=self._execution_worker, daemon=True, name="rl-execution-worker").start()
        self._requeue_pending_executions()

        if API_KEY and API_SECRET:
            self.http = HTTP(testnet=False, demo=BYBIT_DEMO, api_key=API_KEY, api_secret=API_SECRET)
            if ENABLE_PRIVATE_ORDER_WS:
                self._open_private_ws()
            threading.Thread(target=self._reward_poll_loop, daemon=True, name="rl-reward-poll").start()
        elif TRADING_ENABLED:
            log.error("RL_TRADING_ENABLED=true but RL_BYBIT_API_KEY/SECRET are missing")
        else:
            log.info("No RL Bybit credentials configured; sidecar will record decisions only")

        if PRETRAIN_ON_START and PRETRAIN_PATH:
            updates = self.agent.pretrain_from_path(PRETRAIN_PATH)
            if updates:
                self.agent.save(MODEL_PATH)
            log.info("Optional RL pretrain complete  path=%s updates=%d", PRETRAIN_PATH, updates)

    def _requeue_pending_executions(self) -> None:
        pending: list[str] = []
        with self.lock:
            for decision_id, decision in self.decisions.items():
                if decision.get("executed") or decision.get("completed") or decision.get("skip_reason"):
                    continue
                if decision.get("execution_status") in {"queued", "running"}:
                    decision["execution_status"] = "queued"
                    pending.append(decision_id)
        for decision_id in pending:
            try:
                self.execution_queue.put_nowait(decision_id)
            except queue.Full:
                log.warning("Execution queue full while requeueing pending decision %s", decision_id)
                break
        if pending:
            log.info("Requeued %d pending RL execution decisions", len(pending))

    def _load_runtime_state(self) -> None:
        if not os.path.exists(STATE_PATH):
            return
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data.get("decisions"), dict):
                self.decisions = {
                    str(k): v for k, v in data["decisions"].items()
                    if isinstance(v, dict)
                }
            if isinstance(data.get("event_to_decision"), dict):
                self.event_to_decision = {str(k): str(v) for k, v in data["event_to_decision"].items()}
            if isinstance(data.get("active_trades"), dict):
                raw_active = {str(k): str(v) for k, v in data["active_trades"].items()}
                migrated: dict[str, str] = {}
                for key, value in raw_active.items():
                    if key in self.decisions:
                        migrated[key] = value
                    elif value in self.decisions:
                        migrated[value] = key
                self.active_trades = migrated
            if isinstance(data.get("order_to_decision"), dict):
                self.order_to_decision = {str(k): str(v) for k, v in data["order_to_decision"].items()}
            if isinstance(data.get("claimed_closed_pnl_ids"), list):
                self.claimed_closed_pnl_ids = {str(v) for v in data["claimed_closed_pnl_ids"]}
            log.info(
                "Loaded RL runtime state from %s  decisions=%d active=%d",
                STATE_PATH,
                len(self.decisions),
                len(self.active_trades),
            )
        except Exception as exc:
            log.warning("Could not load RL runtime state %s: %s", STATE_PATH, exc)

    def _save_runtime_state_locked(self) -> None:
        pending_or_recent: dict[str, dict[str, Any]] = {}
        for decision_id, decision in list(self.decisions.items())[-1000:]:
            if not decision.get("completed") or decision.get("executed"):
                pending_or_recent[decision_id] = decision
        recent_events = {
            event_id: decision_id
            for event_id, decision_id in self.event_to_decision.items()
            if decision_id in pending_or_recent
        }
        tmp_path = f"{STATE_PATH}.tmp"
        os.makedirs(os.path.dirname(STATE_PATH) or ".", exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(
                jsonable({
                    "saved_at": now_iso(),
                    "decisions": pending_or_recent,
                    "event_to_decision": recent_events,
                    "active_trades": self.active_trades,
                    "order_to_decision": self.order_to_decision,
                    "claimed_closed_pnl_ids": sorted(self.claimed_closed_pnl_ids)[-2000:],
                }),
                fh,
                indent=2,
                sort_keys=True,
            )
        os.replace(tmp_path, STATE_PATH)
        self.event_to_decision = recent_events

    def _open_private_ws(self) -> None:
        try:
            self.private_ws = WebSocket(
                testnet=False,
                demo=BYBIT_DEMO,
                channel_type="private",
                api_key=API_KEY,
                api_secret=API_SECRET,
            )
            self.private_ws.order_stream(self._on_private_order)
            log.info("Private Bybit order WebSocket opened  demo=%s", BYBIT_DEMO)
        except Exception as exc:
            self.private_ws = None
            log.warning("Private order WebSocket unavailable; reward polling remains active: %s", exc)

    def status(self) -> dict[str, Any]:
        with self.lock:
            return {
                "ok": True,
                "started_at": self.started_at,
                "trading_enabled": TRADING_ENABLED,
                "bybit_demo": BYBIT_DEMO,
                "has_credentials": bool(API_KEY and API_SECRET),
                "active_trades": len(self.active_trades),
                "decisions": len(self.decisions),
                "execution_queue_depth": self.execution_queue.qsize(),
                "agent_updates": self.agent.reward_updates,
                "agent_reward_baseline": self.agent.reward_baseline,
            }

    def handle_signal(self, payload: dict[str, Any]) -> dict[str, Any]:
        event_id = str(payload.get("event_id") or uuid.uuid4().hex)
        with self.lock:
            existing_id = self.event_to_decision.get(event_id)
            if existing_id and existing_id in self.decisions:
                return self._decision_response(self.decisions[existing_id])

        action, agent_info = self.agent.decide(payload)
        decision_id = uuid.uuid4().hex
        setup = payload.get("setup") if isinstance(payload.get("setup"), dict) else {}
        decision: dict[str, Any] = {
            "decision_id": decision_id,
            "event_id": event_id,
            "received_at": now_iso(),
            "source_status": payload.get("status"),
            "source_reason": payload.get("reason"),
            "symbol": str(payload.get("symbol") or "").upper(),
            "strategy": payload.get("strategy"),
            "direction": payload.get("direction"),
            "action": action,
            "risk_mode": "fixed_usdt",
            "default_risk_usdt": DEFAULT_RISK_USDT,
            "setup": setup,
            "payload": payload,
            "agent": agent_info,
            "executed": False,
            "completed": False,
            "execution_status": "queued",
        }

        with self.lock:
            self.decisions[decision_id] = decision
            self.event_to_decision[event_id] = decision_id
        try:
            self.execution_queue.put_nowait(decision_id)
            decision["queued_at"] = now_iso()
        except queue.Full:
            decision["execution_status"] = "queue_full"
            decision["skip_reason"] = f"execution queue full ({EXECUTION_QUEUE_SIZE})"

        with self.lock:
            self._save_runtime_state_locked()
        append_jsonl(DECISIONS_PATH, decision)
        return self._decision_response(decision)

    @staticmethod
    def _decision_response(decision: dict[str, Any]) -> dict[str, Any]:
        return {
            "decision_id": decision.get("decision_id"),
            "symbol": decision.get("symbol"),
            "strategy": decision.get("strategy"),
            "action": decision.get("action"),
            "executed": decision.get("executed", False),
            "execution_status": decision.get("execution_status"),
            "skip_reason": decision.get("skip_reason"),
            "order_id": decision.get("entry_order_id"),
            "order_link_id": decision.get("entry_order_link_id"),
            "reward": decision.get("reward"),
        }

    def _execution_worker(self) -> None:
        while True:
            decision_id = self.execution_queue.get()
            try:
                with self.lock:
                    decision = self.decisions.get(decision_id)
                    if decision is not None and not decision.get("completed") and not decision.get("executed"):
                        decision["execution_status"] = "running"
                        decision["execution_started_at"] = now_iso()
                if decision is None:
                    continue
                if decision.get("completed") or decision.get("executed"):
                    continue

                try:
                    self._maybe_execute(decision)
                    decision["execution_status"] = "executed" if decision.get("executed") else "skipped"
                except Exception as exc:
                    decision["execution_status"] = "failed"
                    decision["execution_error"] = str(exc)
                    log.exception(
                        "[%s] RL execution failed decision=%s",
                        decision.get("symbol"),
                        decision_id,
                    )
                finally:
                    decision["execution_finished_at"] = now_iso()
                    with self.lock:
                        self._save_runtime_state_locked()
                    append_jsonl(DECISIONS_PATH, decision)
            finally:
                self.execution_queue.task_done()

    def _instrument_info(self, symbol: str) -> dict[str, Any]:
        if self.http is None:
            raise RuntimeError("Bybit HTTP client unavailable")
        with self.lock:
            cached = self.instrument_cache.get(symbol)
            if cached:
                return cached
        info = get_instrument_info(self.http, symbol)
        if not info:
            raise RuntimeError(f"instrument info missing for {symbol}")
        if info.get("status") and info.get("status") != "Trading":
            raise RuntimeError(f"instrument {symbol} status={info.get('status')}")
        with self.lock:
            self.instrument_cache[symbol] = info
        return info

    @staticmethod
    def _order_link_id(strategy: Any, symbol: str, direction: Any) -> str:
        ts = datetime.now(timezone.utc).strftime("%y%m%d%H%M%S")
        suffix = uuid.uuid4().hex[:4].upper()
        strat = "".join(ch for ch in str(strategy or "rl").upper() if ch.isalnum())[:8] or "RL"
        base = symbol.replace("USDT", "")[:8]
        side = "L" if str(direction).lower() == "long" else "S"
        return f"RL-{strat}-{base}-{side}-{ts}-{suffix}"[:36]

    def _ensure_max_leverage(self, symbol: str, info: dict[str, Any]) -> float:
        if self.http is None:
            raise RuntimeError("Bybit HTTP client unavailable")
        max_leverage = max(float(info.get("max_leverage", 1.0)), 1.0)
        if symbol in self.max_leverage_symbols:
            return max_leverage
        target_text = qty_to_str(max_leverage)
        try:
            pos_resp = self.http.get_positions(category="linear", symbol=symbol)
            rows = pos_resp.get("result", {}).get("list", []) if isinstance(pos_resp, dict) else []
            leverage_values = [
                to_float(row.get("leverage"))
                for row in rows
                if isinstance(row, dict) and str(row.get("symbol") or "").upper() == symbol
            ]
            known = [value for value in leverage_values if value is not None]
            if known and all(abs(value - max_leverage) < 1e-9 for value in known):
                self.max_leverage_symbols.add(symbol)
                return max_leverage
        except Exception as exc:
            log.debug("[%s] could not inspect current leverage before max-leverage sync: %s", symbol, exc)

        try:
            resp = self.http.set_leverage(
                category="linear",
                symbol=symbol,
                buyLeverage=target_text,
                sellLeverage=target_text,
            )
        except Exception as exc:
            if "leverage not modified" in str(exc).lower():
                self.max_leverage_symbols.add(symbol)
                return max_leverage
            raise
        if resp.get("retCode", 0) != 0 and "not modified" not in str(resp.get("retMsg", "")).lower():
            raise RuntimeError(f"set_leverage failed retCode={resp.get('retCode')} retMsg={resp.get('retMsg')}")
        self.max_leverage_symbols.add(symbol)
        return max_leverage

    def _maybe_execute(self, decision: dict[str, Any]) -> None:
        action = float(decision["action"])
        symbol = str(decision.get("symbol") or "").upper()
        direction = str(decision.get("direction") or "").lower()
        setup = decision.get("setup") if isinstance(decision.get("setup"), dict) else {}
        entry = to_float(setup.get("entry"))
        stop = to_float(setup.get("stop_loss"))
        target = to_float(setup.get("take_profit"))
        trail_dist = to_float(setup.get("trail_dist"))
        exit_style = str(setup.get("exit_style") or "fixed_tp")

        if action < MIN_ACTION_TO_TRADE:
            decision["skip_reason"] = f"action {action:.4f} below RL_MIN_ACTION_TO_TRADE={MIN_ACTION_TO_TRADE:.4f}"
            return
        if not TRADING_ENABLED:
            decision["skip_reason"] = "RL_TRADING_ENABLED is false"
            return
        if self.http is None:
            decision["skip_reason"] = "Bybit client unavailable"
            return
        if not symbol or direction not in {"long", "short"}:
            decision["skip_reason"] = "missing symbol or direction"
            return
        if entry is None or stop is None or entry <= 0:
            decision["skip_reason"] = "invalid entry or stop"
            return
        if target is None:
            decision["skip_reason"] = "missing take-profit; hedge-mode RL requires paired partial TP/SL"
            return

        unit_risk = abs(entry - stop)
        if unit_risk <= 0:
            decision["skip_reason"] = "zero stop distance"
            return
        stop_distance_pct = unit_risk / entry
        decision["stop_distance_pct"] = stop_distance_pct
        if MIN_STOP_DISTANCE_PCT > 0 and stop_distance_pct < MIN_STOP_DISTANCE_PCT:
            decision["skip_reason"] = (
                f"stop distance {stop_distance_pct:.4%} below minimum "
                f"{MIN_STOP_DISTANCE_PCT:.4%}"
            )
            return

        fee_risk_per_unit = max(TAKER_FEE_RATE, 0.0) * (entry + stop)
        fee_to_price_risk = fee_risk_per_unit / unit_risk if unit_risk > 0 else 0.0
        decision["fee_to_price_risk"] = fee_to_price_risk
        if MAX_FEE_TO_PRICE_RISK > 0 and fee_to_price_risk > MAX_FEE_TO_PRICE_RISK:
            decision["skip_reason"] = (
                f"fee/risk ratio {fee_to_price_risk:.2%} above limit "
                f"{MAX_FEE_TO_PRICE_RISK:.2%}"
            )
            return

        info = self._instrument_info(symbol)
        balances = get_balance_metrics(self.http)
        equity = float(balances.get("equity", 0.0))
        available = float(balances.get("available", 0.0))
        if equity <= 0:
            decision["skip_reason"] = f"invalid equity {equity}"
            return

        default_risk_usdt = DEFAULT_RISK_USDT
        if default_risk_usdt <= 0:
            decision["skip_reason"] = f"invalid RL_DEFAULT_RISK_USDT={default_risk_usdt}"
            return
        risk_budget = default_risk_usdt * action
        if risk_budget <= 0:
            decision["skip_reason"] = f"risk budget is zero for action={action:.4f}"
            return
        q_step = float(info["qty_step"])
        min_qty = float(info["min_qty"])
        tick = float(info["tick_size"])
        max_leverage = max(float(info.get("max_leverage", 1.0)), 1.0)
        unit_risk_with_fees = unit_risk + fee_risk_per_unit
        raw_qty = risk_budget / unit_risk_with_fees
        margin_basis = available if available > 0 else equity
        max_qty_by_margin = (margin_basis * max_leverage * 0.95) / entry
        if raw_qty > max_qty_by_margin:
            raw_qty = max_qty_by_margin
            decision["qty_capped_by_margin"] = True
        qty = floor_to_step(raw_qty, q_step)
        if qty < min_qty:
            if not ALLOW_MIN_QTY_OVERRISK:
                decision["skip_reason"] = (
                    f"risk action too small for min qty: qty={qty_to_str(qty, q_step)} "
                    f"min_qty={qty_to_str(min_qty, q_step)}"
                )
                return
            qty = min_qty

        sl_price = round_to_step(stop, tick)
        tp_price = round_to_step(target, tick) if target is not None else None
        trail_price_dist = round_to_step(trail_dist, tick) if trail_dist is not None else 0.0
        side = "Buy" if direction == "long" else "Sell"
        position_idx = 1 if side == "Buy" else 2
        order_link_id = self._order_link_id(decision.get("strategy"), symbol, direction)
        order_leverage = self._ensure_max_leverage(symbol, info)

        order_kwargs: dict[str, Any] = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "orderType": "Market",
            "qty": qty_to_str(qty, q_step),
            "positionIdx": position_idx,
            "orderLinkId": order_link_id,
        }
        if tp_price is not None:
            order_kwargs["takeProfit"] = str(tp_price)
            order_kwargs["stopLoss"] = str(sl_price)
            order_kwargs["tpslMode"] = "Partial"
            order_kwargs["tpTriggerBy"] = "LastPrice"
            order_kwargs["slTriggerBy"] = "LastPrice"
            order_kwargs["tpOrderType"] = "Market"
            order_kwargs["slOrderType"] = "Market"
        else:
            decision["protection_warning"] = "No take-profit supplied; entry uses no attached partial TP/SL"

        try:
            resp = self.http.place_order(**order_kwargs)
        except Exception as exc:
            if "position idx" in str(exc).lower():
                raise RuntimeError(
                    f"hedge-mode positionIdx={position_idx} rejected for {symbol}; "
                    "verify the RL Bybit account is in hedge mode"
                ) from exc
            raise

        ret_code = int(resp.get("retCode", -1))
        if ret_code != 0:
            message = str(resp.get("retMsg", "?"))
            if ret_code == 10001 and "position idx" in message.lower():
                raise RuntimeError(
                    f"hedge-mode positionIdx={position_idx} rejected for {symbol}; "
                    "verify the RL Bybit account is in hedge mode"
                )
            raise RuntimeError(f"place_order failed retCode={ret_code} retMsg={resp.get('retMsg')}")

        order_id = str(resp.get("result", {}).get("orderId") or "")
        notional = qty * entry
        expected_price_sl_loss = qty * unit_risk
        expected_fee_loss = qty * fee_risk_per_unit
        expected_sl_loss = expected_price_sl_loss + expected_fee_loss
        decision.update({
            "executed": True,
            "entry_order_id": order_id,
            "entry_order_link_id": order_link_id,
            "position_idx": position_idx,
            "entry_side": side,
            "exit_side": "Sell" if side == "Buy" else "Buy",
            "qty": qty_to_str(qty, q_step),
            "qty_float": qty,
            "notional": notional,
            "risk_mode": "fixed_usdt",
            "default_risk_usdt": default_risk_usdt,
            "risk_budget_usdt": risk_budget,
            "expected_sl_loss_usdt": expected_sl_loss,
            "expected_price_sl_loss_usdt": expected_price_sl_loss,
            "expected_fee_loss_usdt": expected_fee_loss,
            "order_leverage": order_leverage,
            "opened_at_ms": now_ms(),
            "order_request": order_kwargs,
            "order_response": resp,
            "partial_tpsl_mode": bool(tp_price is not None),
            "trailing_disabled_reason": (
                "RL hedge-mode execution uses attached Partial TP/SL sized to the entry qty; "
                "full-position trailing stops are disabled"
                if exit_style != "fixed_tp" or trail_price_dist > 0 else None
            ),
        })
        with self.lock:
            decision_id = str(decision["decision_id"])
            self.active_trades[decision_id] = symbol
            if order_id:
                self.order_to_decision[order_id] = decision_id
            self.order_to_decision[order_link_id] = decision_id
        log.info(
            "[%s] RL order accepted action=%.3f qty=%s risk~%.2f default_risk=%.2f orderId=%s",
            symbol,
            action,
            qty_to_str(qty, q_step),
            expected_sl_loss,
            default_risk_usdt,
            order_id or "-",
        )

    @staticmethod
    def _private_items(msg: dict[str, Any]) -> list[dict[str, Any]]:
        data = msg.get("data") if isinstance(msg, dict) else None
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if isinstance(data, dict):
            return [data]
        return []

    @staticmethod
    def _truthy(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes"}

    @staticmethod
    def _is_exit_order(order: dict[str, Any]) -> bool:
        stop_type = str(order.get("stopOrderType") or "").strip()
        create_type = str(order.get("createType") or "").lower()
        return (
            bool(stop_type)
            or RLExecutionService._truthy(order.get("reduceOnly"))
            or RLExecutionService._truthy(order.get("closeOnTrigger"))
            or any(token in create_type for token in ("takeprofit", "stoploss", "trailing"))
        )

    def _find_decision_for_order(self, order: dict[str, Any]) -> dict[str, Any] | None:
        symbol = str(order.get("symbol") or "").upper()
        keys = [
            order.get("orderId"),
            order.get("orderLinkId"),
            order.get("parentOrderId"),
            order.get("parentOrderLinkId"),
            order.get("blockTradeId"),
        ]
        with self.lock:
            for key in keys:
                decision_id = self.order_to_decision.get(str(key or ""))
                if decision_id and decision_id in self.decisions:
                    return self.decisions[decision_id]

            side = str(order.get("side") or "")
            position_idx = int(to_float(order.get("positionIdx")) or 0)
            qty = to_float(order.get("cumExecQty") or order.get("qty") or order.get("leavesQty"))
            candidates = [
                self.decisions[decision_id]
                for decision_id, active_symbol in self.active_trades.items()
                if active_symbol == symbol and decision_id in self.decisions
            ]
        matching: list[dict[str, Any]] = []
        for decision in candidates:
            if decision.get("completed"):
                continue
            if int(to_float(decision.get("position_idx")) or 0) != position_idx:
                continue
            if side and str(decision.get("exit_side") or "") != side and self._is_exit_order(order):
                continue
            if side and str(decision.get("entry_side") or "") != side and not self._is_exit_order(order):
                continue
            expected_qty = to_float(decision.get("qty_float") or decision.get("qty"))
            if qty is not None and expected_qty is not None:
                tolerance = max(abs(expected_qty) * 1e-6, 1e-12)
                if abs(qty - expected_qty) > tolerance:
                    continue
            matching.append(decision)
        if not matching:
            return None
        return sorted(matching, key=lambda item: int(to_float(item.get("opened_at_ms")) or 0))[0]

    def _on_private_order(self, msg: dict[str, Any]) -> None:
        try:
            for order in self._private_items(msg):
                if order.get("category") not in (None, "", "linear"):
                    continue
                if str(order.get("orderStatus", "")).lower() != "filled":
                    continue
                symbol = str(order.get("symbol") or "").upper()
                if not symbol:
                    continue
                order_id = str(order.get("orderId") or "")
                order_link_id = str(order.get("orderLinkId") or "")
                decision = self._find_decision_for_order(order)
                if not decision:
                    continue
                decision_id = str(decision["decision_id"])
                for key in (order_id, order_link_id, order.get("parentOrderId"), order.get("parentOrderLinkId")):
                    if key:
                        with self.lock:
                            self.order_to_decision[str(key)] = decision_id
                if not self._is_exit_order(order):
                    if order_id == decision.get("entry_order_id") or order_link_id == decision.get("entry_order_link_id"):
                        decision["entry_filled_at_ms"] = int(to_float(order.get("updatedTime")) or now_ms())
                        decision["entry_fill_raw"] = order
                    continue

                dedupe_key = order_id or f"{symbol}:{order.get('updatedTime')}:{order.get('avgPrice')}"
                if dedupe_key in self.notified_exit_ids:
                    continue
                closed_pnl = to_float(order.get("closedPnl"))
                if closed_pnl is None:
                    log.info("[%s] exit fill seen but closedPnl missing; waiting for REST poll", symbol)
                    continue
                self.notified_exit_ids.add(dedupe_key)
                self._complete_trade(
                    str(decision["decision_id"]),
                    closed_pnl=closed_pnl,
                    source="private_order_ws",
                    exit_order_id=order_id,
                    raw=order,
                )
        except Exception:
            log.exception("Error in private order callback")

    def _reward_poll_loop(self) -> None:
        while True:
            time.sleep(max(5.0, REWARD_POLL_SECONDS))
            try:
                self._poll_rewards_once()
            except Exception:
                log.exception("Reward poll failed")

    def _closed_pnl_row_matches_decision(self, row: dict[str, Any], decision: dict[str, Any]) -> bool:
        row_keys = [
            row.get("orderId"),
            row.get("orderLinkId"),
            row.get("parentOrderId"),
            row.get("parentOrderLinkId"),
        ]
        with self.lock:
            for key in row_keys:
                mapped = self.order_to_decision.get(str(key or ""))
                if mapped and mapped == decision.get("decision_id"):
                    return True

        opened_at = int(to_float(decision.get("opened_at_ms")) or 0)
        ts = int(to_float(row.get("updatedTime")) or to_float(row.get("createdTime")) or 0)
        if opened_at and ts < opened_at - 300_000:
            return False
        if str(row.get("symbol") or "").upper() != str(decision.get("symbol") or "").upper():
            return False
        row_side = str(row.get("side") or "")
        if row_side and row_side != str(decision.get("exit_side") or ""):
            return False
        row_qty = to_float(row.get("qty") or row.get("closedSize") or row.get("cumExecQty"))
        expected_qty = to_float(decision.get("qty_float") or decision.get("qty"))
        if row_qty is not None and expected_qty is not None:
            tolerance = max(abs(expected_qty) * 1e-6, 1e-12)
            if abs(row_qty - expected_qty) > tolerance:
                return False
        return True

    def _poll_rewards_once(self) -> None:
        if self.http is None:
            return
        with self.lock:
            active_items = list(self.active_trades.items())
        symbols = sorted({symbol for _decision_id, symbol in active_items})
        closed_rows_by_symbol: dict[str, list[dict[str, Any]]] = {}
        for symbol in symbols:
            try:
                pnl_resp = self.http.get_closed_pnl(category="linear", symbol=symbol, limit=50)
                rows = pnl_resp.get("result", {}).get("list", []) if isinstance(pnl_resp, dict) else []
                closed_rows_by_symbol[symbol] = [row for row in rows if isinstance(row, dict)]
            except Exception as exc:
                log.debug("[%s] closed pnl poll failed: %s", symbol, exc)
        for decision_id, symbol in active_items:
            with self.lock:
                decision = self.decisions.get(decision_id)
            if not decision or decision.get("completed"):
                continue
            opened_at = int(to_float(decision.get("opened_at_ms")) or 0)
            if opened_at and now_ms() - opened_at < 15_000:
                continue
            best_row = None
            best_ts = 0
            for row in closed_rows_by_symbol.get(symbol, []):
                row_id = str(row.get("orderId") or row.get("orderLinkId") or "")
                if row_id and row_id in self.claimed_closed_pnl_ids:
                    continue
                ts = int(to_float(row.get("updatedTime")) or to_float(row.get("createdTime")) or 0)
                if (
                    ts >= best_ts
                    and to_float(row.get("closedPnl")) is not None
                    and self._closed_pnl_row_matches_decision(row, decision)
                ):
                    best_row = row
                    best_ts = ts
            if best_row is None:
                continue
            row_id = str(best_row.get("orderId") or best_row.get("orderLinkId") or "")
            if row_id:
                self.claimed_closed_pnl_ids.add(row_id)
            self._complete_trade(
                decision_id,
                closed_pnl=float(best_row.get("closedPnl")),
                source="closed_pnl_poll",
                exit_order_id=str(best_row.get("orderId") or ""),
                raw=best_row,
            )

    def _complete_trade(
        self,
        decision_id: str,
        *,
        closed_pnl: float,
        source: str,
        exit_order_id: str,
        raw: dict[str, Any],
    ) -> None:
        with self.lock:
            decision = self.decisions.get(decision_id)
            if not decision or decision.get("completed"):
                return
            default_risk = to_float(decision.get("default_risk_usdt")) or 0.0
            actual_risk = to_float(decision.get("risk_budget_usdt")) or to_float(decision.get("expected_sl_loss_usdt")) or 0.0
            reward_default_r = closed_pnl / default_risk if default_risk > 0 else 0.0
            reward_actual_r = closed_pnl / actual_risk if actual_risk > 0 else 0.0
            reward_payload = {
                "decision_id": decision_id,
                "symbol": decision.get("symbol"),
                "strategy": decision.get("strategy"),
                "action": decision.get("action"),
                "closed_pnl": closed_pnl,
                "reward_default_r": reward_default_r,
                "reward_actual_r": reward_actual_r,
                "source": source,
                "exit_order_id": exit_order_id,
                "received_at": now_iso(),
                "raw": raw,
            }
            payload = decision.get("payload") if isinstance(decision.get("payload"), dict) else {}
            agent = decision.get("agent") if isinstance(decision.get("agent"), dict) else {}
            training_example = {
                "schema_version": "rl_training_example_v1",
                "decision_id": decision_id,
                "event_id": decision.get("event_id"),
                "symbol": decision.get("symbol"),
                "strategy": decision.get("strategy"),
                "direction": decision.get("direction"),
                "source_status": decision.get("source_status"),
                "source_reason": decision.get("source_reason"),
                "received_at": decision.get("received_at"),
                "completed_at": reward_payload["received_at"],
                "action": decision.get("action"),
                "policy_action": agent.get("policy_action"),
                "policy_score": agent.get("score"),
                "explored": agent.get("explored"),
                "risk_mode": decision.get("risk_mode"),
                "default_risk_usdt": decision.get("default_risk_usdt"),
                "risk_budget_usdt": decision.get("risk_budget_usdt"),
                "expected_sl_loss_usdt": decision.get("expected_sl_loss_usdt"),
                "setup": decision.get("setup"),
                "source_features": payload.get("features"),
                "feature_columns": payload.get("feature_columns"),
                "agent_feature_vector": agent.get("feature_vector"),
                "market_context": payload.get("market_context"),
                "reward": reward_payload,
            }
            decision["completed"] = True
            decision["completed_at"] = reward_payload["received_at"]
            decision["reward"] = reward_payload
            self.active_trades.pop(decision_id, None)
            if exit_order_id:
                self.order_to_decision[str(exit_order_id)] = decision_id
                self.claimed_closed_pnl_ids.add(str(exit_order_id))
            self.agent.learn(decision, reward_default_r)
            self.agent.save(MODEL_PATH)
            self._save_runtime_state_locked()
        append_jsonl(REWARDS_PATH, reward_payload)
        append_jsonl(TRAINING_EXAMPLES_PATH, training_example)
        log.info(
            "[%s] RL reward decision=%s pnl=%.4f reward_default_r=%.4f source=%s",
            decision.get("symbol"),
            decision_id,
            closed_pnl,
            reward_default_r,
            source,
        )

    def handle_manual_reward(self, payload: dict[str, Any]) -> dict[str, Any]:
        decision_id = str(payload.get("decision_id") or "")
        if not decision_id:
            order_link_id = str(payload.get("order_link_id") or "")
            with self.lock:
                for candidate_id, decision in self.decisions.items():
                    if decision.get("entry_order_link_id") == order_link_id:
                        decision_id = candidate_id
                        break
        if not decision_id:
            raise ValueError("decision_id or order_link_id is required")
        closed_pnl = to_float(payload.get("closed_pnl"))
        reward = to_float(payload.get("reward"))
        with self.lock:
            decision = self.decisions.get(decision_id)
            default_risk = to_float((decision or {}).get("default_risk_usdt")) or 0.0
        if closed_pnl is None:
            if reward is None:
                raise ValueError("closed_pnl or reward is required")
            closed_pnl = reward * default_risk if default_risk > 0 else reward
        self._complete_trade(
            decision_id,
            closed_pnl=closed_pnl,
            source="manual_reward",
            exit_order_id=str(payload.get("exit_order_id") or ""),
            raw=payload,
        )
        with self.lock:
            return self._decision_response(self.decisions[decision_id])


class RequestHandler(BaseHTTPRequestHandler):
    service: RLExecutionService

    def log_message(self, fmt: str, *args: Any) -> None:
        log.debug("http: " + fmt, *args)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(jsonable(payload), sort_keys=True).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            log.debug("HTTP client disconnected before response could be written: %s", exc)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:
        if self.path.rstrip("/") in {"", "/health"}:
            self._send_json(HTTPStatus.OK, self.service.status())
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            if self.path.rstrip("/") == "/v1/signals":
                response = self.service.handle_signal(payload)
                self._send_json(HTTPStatus.ACCEPTED, response)
                return
            if self.path.rstrip("/") == "/v1/rewards":
                response = self.service.handle_manual_reward(payload)
                self._send_json(HTTPStatus.OK, response)
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except json.JSONDecodeError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"invalid json: {exc}"})
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:
            log.exception("HTTP request failed")
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})


def main() -> None:
    service = RLExecutionService()
    RequestHandler.service = service
    server = ThreadingHTTPServer((HOST, PORT), RequestHandler)

    def _shutdown(_sig_num: int, _frame: Any) -> None:
        log.info("Shutdown signal received")
        server.shutdown()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    log.info(
        "RL execution sidecar listening on %s:%d  trading_enabled=%s demo=%s "
        "default_risk=%.2f USDT min_action=%.3f",
        HOST,
        PORT,
        TRADING_ENABLED,
        BYBIT_DEMO,
        DEFAULT_RISK_USDT,
        MIN_ACTION_TO_TRADE,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
