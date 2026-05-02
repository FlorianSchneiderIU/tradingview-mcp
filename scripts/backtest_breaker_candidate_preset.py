from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.breaker_candidate_presets import candidate_mask, get_preset, preset_names, preset_summary
from scripts.ml_breaker_continuation_filter import enrich, trade_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a named breaker candidate preset to the offline breaker trade set.")
    parser.add_argument("--preset", choices=preset_names(), required=True)
    parser.add_argument(
        "--trades-file",
        type=Path,
        default=Path("scripts/breaker_continuation_majors10_1h_fvg_print15_retest72_2022_2026.csv"),
    )
    parser.add_argument("--fee-bps-side", type=float, default=5.0)
    parser.add_argument("--start", default="2023-04-20")
    parser.add_argument("--fold1-end", default="2024-04-20")
    parser.add_argument("--fold2-end", default="2025-04-20")
    parser.add_argument("--end", default="2026-04-20")
    parser.add_argument("--out-prefix", type=Path, default=Path("scripts/breaker_candidate_preset"))
    return parser.parse_args()


def format_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(empty)"
    return frame.to_string(index=False)


def summary_row(name: str, frame: pd.DataFrame) -> dict[str, Any]:
    metrics = trade_metrics(frame)
    return {"slice": name, **metrics}


def main() -> None:
    args = parse_args()
    preset = get_preset(args.preset)
    start = pd.Timestamp(args.start, tz="UTC")
    fold1_end = pd.Timestamp(args.fold1_end, tz="UTC")
    fold2_end = pd.Timestamp(args.fold2_end, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")

    frame = enrich(pd.read_csv(args.trades_file), args.fee_bps_side)
    frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True)
    frame = frame[(frame["entry_time"] >= start) & (frame["entry_time"] < end)].copy()
    frame["fold"] = 0
    frame.loc[(frame["entry_time"] >= start) & (frame["entry_time"] < fold1_end), "fold"] = 1
    frame.loc[(frame["entry_time"] >= fold1_end) & (frame["entry_time"] < fold2_end), "fold"] = 2
    frame.loc[(frame["entry_time"] >= fold2_end) & (frame["entry_time"] < end), "fold"] = 3

    selected = frame[candidate_mask(frame, preset)].copy()

    summary = pd.DataFrame(
        [
            summary_row("selected", selected),
            summary_row("fold1", selected[selected["fold"] == 1]),
            summary_row("fold2", selected[selected["fold"] == 2]),
            summary_row("fold3", selected[selected["fold"] == 3]),
        ]
    )

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    stem = f"{args.out_prefix.name}_{preset.name}_{int(args.fee_bps_side)}bps"
    selected_path = args.out_prefix.with_name(stem + "_selected.csv")
    summary_path = args.out_prefix.with_name(stem + "_summary.csv")
    report_path = args.out_prefix.with_name(stem + "_report.md")

    selected.to_csv(selected_path, index=False)
    summary.to_csv(summary_path, index=False)

    report_lines = [
        f"# Breaker Candidate Preset: {preset.name}",
        "",
        "## Preset",
        "",
        "```text",
        format_table(pd.DataFrame([preset_summary(preset)])),
        "```",
        "",
        "## Summary",
        "",
        "```text",
        format_table(summary),
        "```",
        "",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n".join(report_lines))
    print(f"Saved selected trades to {selected_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
