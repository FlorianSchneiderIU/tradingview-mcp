from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LIVE_LOG_DIR = REPO_ROOT / "scripts" / "live_logs"
DEFAULT_CONFIG = REPO_ROOT / "scripts" / "zone_channel_candidate_btc_15m_broad_v2.json"
DEFAULT_RUNNER = REPO_ROOT / "scripts" / "bybit_zone_channel_shadow.py"
DEFAULT_ENV_FILE = REPO_ROOT / "scripts" / "bybit_demo.env"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the Bybit zone-channel shadow runner in the background.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--python", type=Path, default=REPO_ROOT / ".venv" / "Scripts" / "python.exe")
    parser.add_argument("--lookback-days", type=int, default=120)
    parser.add_argument("--tag", default="broad_v2")
    parser.add_argument("--websocket-demo", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--websocket-testnet", action="store_true")
    parser.add_argument("--scan-on-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--websocket-idle-timeout", type=int, default=120)
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    args = parse_args()
    LIVE_LOG_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    stem = f"bybit_zone_channel_{args.tag}_{timestamp}"
    stdout_path = LIVE_LOG_DIR / f"{stem}.out.log"
    stderr_path = LIVE_LOG_DIR / f"{stem}.err.log"
    jsonl_path = LIVE_LOG_DIR / f"{stem}.jsonl"
    heartbeat_path = LIVE_LOG_DIR / f"bybit_zone_channel_{args.tag}_heartbeat.json"
    state_path = LIVE_LOG_DIR / f"bybit_zone_channel_{args.tag}_state.json"
    decisions_path = LIVE_LOG_DIR / f"bybit_zone_channel_{args.tag}_decisions.csv"
    trades_path = LIVE_LOG_DIR / f"bybit_zone_channel_{args.tag}_trades.csv"
    summary_path = LIVE_LOG_DIR / f"bybit_zone_channel_{args.tag}_summary.json"
    report_path = LIVE_LOG_DIR / f"bybit_zone_channel_{args.tag}_report.md"
    config_copy_path = LIVE_LOG_DIR / f"bybit_zone_channel_{args.tag}_config.json"
    current_path = LIVE_LOG_DIR / f"bybit_zone_channel_{args.tag}_current.json"
    pid_path = REPO_ROOT / "scripts" / f"bybit_zone_channel_{args.tag}.pid"

    command = [
        str(args.python),
        str(args.runner),
        "--config",
        str(args.config),
        "--env-file",
        str(args.env_file),
        "--lookback-days",
        str(args.lookback_days),
        "--log-jsonl",
        str(jsonl_path),
        "--heartbeat-json",
        str(heartbeat_path),
        "--state-json",
        str(state_path),
        "--decisions-csv",
        str(decisions_path),
        "--trades-csv",
        str(trades_path),
        "--summary-json",
        str(summary_path),
        "--report-md",
        str(report_path),
        "--pid-file",
        str(pid_path),
        "--config-copy",
        str(config_copy_path),
        "--loop",
        "--websocket-idle-timeout",
        str(args.websocket_idle_timeout),
    ]
    if not args.scan_on_start:
        command.append("--no-scan-on-start")
    if args.websocket_demo:
        command.append("--websocket-demo")
    if args.websocket_testnet:
        command.append("--websocket-testnet")

    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
        popen_kwargs: dict[str, object] = {
            "stdout": stdout_handle,
            "stderr": stderr_handle,
            "cwd": str(REPO_ROOT),
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS  # type: ignore[attr-defined]
        process = subprocess.Popen(command, **popen_kwargs)

    payload = {
        "started_at": utc_now_iso(),
        "tag": args.tag,
        "config": str(args.config),
        "runner": str(args.runner),
        "python": str(args.python),
        "command": command,
        "pid": process.pid,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "jsonl": str(jsonl_path),
        "heartbeat": str(heartbeat_path),
        "state": str(state_path),
        "decisions_csv": str(decisions_path),
        "trades_csv": str(trades_path),
        "summary_json": str(summary_path),
        "report_md": str(report_path),
        "config_copy": str(config_copy_path),
        "pid_file": str(pid_path),
    }
    current_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
