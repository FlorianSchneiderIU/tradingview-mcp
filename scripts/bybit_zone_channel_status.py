from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show status for the Bybit zone-channel shadow runner.")
    parser.add_argument("--tag", default="broad_v2")
    parser.add_argument("--current", type=Path)
    parser.add_argument("--heartbeat", type=Path)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--pid-file", type=Path)
    parser.add_argument("--tail", type=int, default=8)
    return parser.parse_args()


def resolve_default_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    live_logs = Path("scripts/live_logs")
    current = args.current or live_logs / f"bybit_zone_channel_{args.tag}_current.json"
    heartbeat = args.heartbeat or live_logs / f"bybit_zone_channel_{args.tag}_heartbeat.json"
    state = args.state or live_logs / f"bybit_zone_channel_{args.tag}_state.json"
    pid_file = args.pid_file or Path("scripts") / f"bybit_zone_channel_{args.tag}.pid"
    return current, heartbeat, state, pid_file


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        return {"error": f"invalid JSON in {path}: {exc}"}


def tail_lines(path: Path | None, count: int) -> list[str]:
    if path is None:
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return []
    return lines[-count:]


def tail_jsonl(path: Path | None, count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in tail_lines(path, count):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"raw": line})
    return rows


def process_running(pid: Any) -> bool:
    if pid in (None, ""):
        return False
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return False
    if os.name == "nt":
        ps_result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"if (Get-Process -Id {pid_int} -ErrorAction SilentlyContinue) {{ exit 0 }} else {{ exit 1 }}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return ps_result.returncode == 0
    try:
        os.kill(pid_int, 0)
        return True
    except OSError:
        return False


def parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def age_text(value: Any) -> str:
    dt = parse_time(value)
    if dt is None:
        return "unknown"
    seconds = (datetime.now(timezone.utc) - dt).total_seconds()
    if seconds < 90:
        return f"{seconds:.0f}s ago"
    return f"{seconds / 60:.1f}m ago"


def main() -> None:
    args = parse_args()
    current_path, heartbeat_path, state_path, pid_file = resolve_default_paths(args)
    current = read_json(current_path) or {}
    heartbeat = read_json(heartbeat_path) or {}
    state = read_json(state_path) or {}

    stdout = Path(current["stdout"]) if current.get("stdout") else None
    stderr = Path(current["stderr"]) if current.get("stderr") else None
    jsonl = Path(current["jsonl"]) if current.get("jsonl") else None

    pid_file_pid = None
    try:
        pid_file_pid = pid_file.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        pass
    pid = heartbeat.get("pid") or pid_file_pid or current.get("pid")

    print("Bybit Zone-Channel Shadow Status")
    print(f"Tag: {args.tag}")
    print(f"PID: active={pid} running={process_running(pid)}")
    print(f"Started: {current.get('started_at') or heartbeat.get('started_at')}")
    print(f"Config: {current.get('config') or heartbeat.get('config_name')}")
    print(f"Heartbeat: {heartbeat.get('updated_at', heartbeat.get('last_update', 'missing'))} ({age_text(heartbeat.get('updated_at', heartbeat.get('last_update')))})")
    print(f"Last event: {heartbeat.get('event', heartbeat.get('last_event', 'unknown'))}")
    print(f"Symbol: {heartbeat.get('symbol', state.get('symbol'))}")
    print(f"Decision TF: {heartbeat.get('decision_timeframe', state.get('decision_timeframe'))}")
    print(f"Logs: stdout={stdout} stderr={stderr} jsonl={jsonl}")
    print()

    summary = state.get("summary") or heartbeat.get("summary") or {}
    print("Summary")
    for key in [
        "signal_rows",
        "selected_signal_rows",
        "decision_rows",
        "accepted_orders",
        "filled_trades",
        "expired_orders",
        "filtered_by_gate",
        "blocked_active_trade",
        "blocked_loss_streak",
        "open_pending_orders",
        "open_trades",
        "consecutive_losses",
        "trades",
        "total_return",
        "profit_factor",
        "net_r",
    ]:
        if key in summary:
            print(f"{key}: {summary[key]}")

    if state:
        print()
        print("Live State")
        print(f"last_base_candle_time: {state.get('last_base_candle_time')}")
        print(f"last_decision_candle_time: {state.get('last_decision_candle_time')}")
        print(f"pending_order: {json.dumps(state.get('pending_order'), default=str)}")
        print(f"active_trade: {json.dumps(state.get('active_trade'), default=str)}")

    stderr_tail = tail_lines(stderr, args.tail)
    if stderr_tail:
        print()
        print("stderr tail")
        for line in stderr_tail:
            print(line)

    print()
    print("decision tail")
    for row in tail_jsonl(jsonl, args.tail):
        if "raw" in row:
            print(row["raw"])
            continue
        print(
            f"{row.get('time')} {row.get('kind')} {row.get('status')} "
            f"symbol={row.get('symbol')} direction={row.get('direction')} "
            f"rr={row.get('target_rr_planned')}"
        )

    print()
    print("stdout tail")
    for line in tail_lines(stdout, args.tail):
        print(line)


if __name__ == "__main__":
    main()
