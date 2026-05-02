from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.bybit_demo_turtle_soup import DEFAULT_BASE_URL, BybitV5Client, load_env_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show status for the Bybit demo Turtle Soup bot.")
    parser.add_argument("--current", type=Path, default=Path("scripts/live_logs/bybit_turtle_soup_current.json"))
    parser.add_argument("--heartbeat", type=Path, default=Path("scripts/live_logs/bybit_turtle_soup_heartbeat.json"))
    parser.add_argument("--pid-file", type=Path, default=Path("scripts/bybit_turtle_soup.pid"))
    parser.add_argument("--env-file", type=Path, default=Path("scripts/bybit_demo.env"))
    parser.add_argument("--tail", type=int, default=8)
    parser.add_argument("--no-exchange", action="store_true")
    return parser.parse_args()


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
        if ps_result.returncode == 0:
            return True
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid_int}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            check=False,
        )
        return str(pid_int) in result.stdout and "INFO:" not in result.stdout.upper()

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


def exchange_summary(symbols: list[str], prefix: str, env_file: Path) -> dict[str, Any]:
    load_env_file(env_file)
    client = BybitV5Client(
        base_url=os.environ.get("BYBIT_BASE_URL", DEFAULT_BASE_URL),
        api_key=os.environ.get("BYBIT_DEMO_API_KEY") or os.environ.get("BYBIT_API_KEY"),
        api_secret=os.environ.get("BYBIT_DEMO_API_SECRET") or os.environ.get("BYBIT_API_SECRET"),
    )
    if not client.has_credentials:
        return {"error": "missing Bybit credentials"}

    out: dict[str, Any] = {}
    for symbol in symbols:
        positions = client.positions(symbol)
        orders = client.open_orders(symbol, prefix)
        out[symbol] = {
            "positions": [
                {
                    "side": row.get("side"),
                    "size": row.get("size"),
                    "avgPrice": row.get("avgPrice"),
                    "takeProfit": row.get("takeProfit"),
                    "stopLoss": row.get("stopLoss"),
                    "unrealisedPnl": row.get("unrealisedPnl"),
                }
                for row in positions
            ],
            "orders": [
                {
                    "side": row.get("side"),
                    "qty": row.get("qty"),
                    "price": row.get("price"),
                    "orderStatus": row.get("orderStatus"),
                    "takeProfit": row.get("takeProfit"),
                    "stopLoss": row.get("stopLoss"),
                    "orderLinkId": row.get("orderLinkId"),
                }
                for row in orders
            ],
        }
    return out


def main() -> None:
    args = parse_args()
    current = read_json(args.current) or {}
    heartbeat = read_json(args.heartbeat) or {}
    stdout = Path(current["stdout"]) if current.get("stdout") else None
    stderr = Path(current["stderr"]) if current.get("stderr") else None
    jsonl = Path(current["jsonl"]) if current.get("jsonl") else None
    pid_file_pid = None
    try:
        pid_file_pid = args.pid_file.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        pass
    current_pid = current.get("pid")
    heartbeat_pid = heartbeat.get("pid")
    pid = heartbeat_pid or pid_file_pid or current_pid
    symbols = list(heartbeat.get("symbols") or ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    prefix = str(heartbeat.get("order_link_prefix") or "TSOUP")

    print("Bybit Turtle Soup Bot Status")
    print(f"PID: active={pid} running={process_running(pid)}")
    print(f"PID detail: current={current_pid} heartbeat={heartbeat_pid} pid_file={pid_file_pid}")
    print(f"Started: {current.get('started_at') or heartbeat.get('started_at')}")
    print(f"Command: {current.get('command', 'unknown')}")
    print(f"Heartbeat: {heartbeat.get('last_update', 'missing')} ({age_text(heartbeat.get('last_update'))})")
    print(f"Last event: {heartbeat.get('last_event', 'unknown')}")
    print(f"Mode: {heartbeat.get('mode', 'unknown')} execute={heartbeat.get('execute')} interval={heartbeat.get('interval')}")
    print(f"Logs: stdout={stdout} stderr={stderr} jsonl={jsonl}")
    print()

    print("Per Symbol")
    last_status = heartbeat.get("last_status_by_symbol", {})
    last_candle = heartbeat.get("last_candle_by_symbol", {})
    last_recon = heartbeat.get("last_reconciliation_by_symbol", {})
    for symbol in symbols:
        recon = last_recon.get(symbol) or {}
        print(
            f"{symbol}: candle={last_candle.get(symbol)} "
            f"status={last_status.get(symbol)} "
            f"positions={recon.get('open_positions')} active_orders={recon.get('active_orders')}"
        )

    if not args.no_exchange:
        print()
        print("Exchange")
        try:
            summary = exchange_summary(symbols, prefix, args.env_file)
            print(json.dumps(summary, indent=2))
        except Exception as exc:
            print(f"exchange query failed: {exc}")

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
        recon = row.get("reconciliation") or {}
        print(
            f"{row.get('time')} {row.get('symbol')} {row.get('status')} "
            f"pos={recon.get('open_positions')} orders={recon.get('active_orders')} "
            f"canceled={recon.get('canceled_orders')}"
        )

    print()
    print("stdout tail")
    for line in tail_lines(stdout, args.tail):
        print(line)


if __name__ == "__main__":
    main()
