from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.channel_state_research.production import (
    build_heartbeat,
    build_production_inputs,
    build_production_report,
    latest_selected_signal,
    load_production_config,
    replay_production_strategy,
    save_production_config,
    update_heartbeat,
    write_json_file,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay/shadow runner for the frozen zone-channel production config.")
    parser.add_argument("--config", type=Path, default=Path("scripts/zone_channel_production_width_rr_v1.json"))
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default=datetime.now(timezone.utc).date().isoformat())
    parser.add_argument("--cache-dir", type=Path, default=Path("scripts/.cache"))
    parser.add_argument("--mode", choices=["replay", "latest"], default="replay")
    parser.add_argument("--max-signal-age-hours", type=float, default=12.0)
    parser.add_argument("--output-prefix", type=Path, default=Path("scripts/zone_channel_production"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_production_config(args.config)
    output_prefix = _resolve_output_prefix(args.output_prefix, config.name)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    heartbeat_path = output_prefix.with_name(output_prefix.name + "_heartbeat.json")
    log_jsonl_path = output_prefix.with_name(output_prefix.name + "_decisions.jsonl")
    if log_jsonl_path.exists():
        log_jsonl_path.unlink()

    heartbeat = build_heartbeat(
        mode=args.mode,
        config=config,
        start=args.start,
        end=args.end,
        log_jsonl=log_jsonl_path,
    )
    update_heartbeat(heartbeat_path, heartbeat, "building_inputs")

    market, signals, feature_groups = build_production_inputs(
        config,
        start=args.start,
        end=args.end,
        cache_dir=args.cache_dir,
    )
    update_heartbeat(
        heartbeat_path,
        heartbeat,
        "inputs_ready",
        signal_rows=int(len(signals)),
        selected_signal_rows=int(signals["gate_pass"].astype(bool).sum()) if not signals.empty and "gate_pass" in signals.columns else 0,
    )

    config_copy_path = output_prefix.with_name(output_prefix.name + "_config.json")
    save_production_config(config_copy_path, config)
    feature_groups_path = output_prefix.with_name(output_prefix.name + "_feature_groups.json")
    feature_groups_path.write_text(json.dumps(feature_groups, indent=2), encoding="utf-8")
    signals_path = output_prefix.with_name(output_prefix.name + "_signals.csv")
    signals.to_csv(signals_path, index=False)

    if args.mode == "latest":
        snapshot = latest_selected_signal(
            signals,
            as_of=pd.Timestamp(market.bars_by_timeframe[config.decision_timeframe]["close_time"].iloc[-1]).tz_convert("UTC"),
            max_age_hours=args.max_signal_age_hours,
        )
        latest_path = output_prefix.with_name(output_prefix.name + "_latest_signal.json")
        if snapshot is None:
            payload = {
                "status": "no_recent_selected_signal",
                "as_of": str(pd.Timestamp(market.bars_by_timeframe[config.decision_timeframe]["close_time"].iloc[-1]).tz_convert("UTC")),
                "max_signal_age_hours": args.max_signal_age_hours,
            }
        else:
            payload = {
                "status": "selected_signal",
                **{key: _json_safe(snapshot[key]) for key in snapshot.index},
            }
            write_jsonl(log_jsonl_path, payload)
        write_json_file(latest_path, payload)
        update_heartbeat(heartbeat_path, heartbeat, "latest_complete", last_status=payload["status"])
        print(json.dumps(payload, indent=2, default=str))
        return

    results = replay_production_strategy(
        config=config,
        exec_frame=market.bars_by_timeframe[config.decision_timeframe],
        signals=signals,
    )
    decisions = results["decisions"]
    trades = results["trades"]
    summary = results["summary"]

    decisions_path = output_prefix.with_name(output_prefix.name + "_decisions.csv")
    trades_path = output_prefix.with_name(output_prefix.name + "_trades.csv")
    summary_path = output_prefix.with_name(output_prefix.name + "_summary.json")
    report_path = output_prefix.with_name(output_prefix.name + "_report.md")

    decisions.to_csv(decisions_path, index=False)
    trades.to_csv(trades_path, index=False)
    write_json_file(summary_path, summary)
    report_path.write_text(
        build_production_report(
            config=config,
            start=args.start,
            end=args.end,
            summary=summary,
        ),
        encoding="utf-8",
    )

    for _, row in decisions.iterrows():
        write_jsonl(log_jsonl_path, {key: _json_safe(value) for key, value in row.to_dict().items()})

    update_heartbeat(
        heartbeat_path,
        heartbeat,
        "replay_complete",
        last_status="ok",
        summary=summary,
    )

    print(json.dumps(summary, indent=2, default=str))
    print(f"Saved signals to {signals_path}")
    print(f"Saved decisions to {decisions_path}")
    print(f"Saved trades to {trades_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved report to {report_path}")


def _resolve_output_prefix(requested: Path, config_name: str) -> Path:
    if requested == Path("scripts/zone_channel_production"):
        return requested.with_name(config_name)
    return requested


def _json_safe(value: object) -> object:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


if __name__ == "__main__":
    main()
