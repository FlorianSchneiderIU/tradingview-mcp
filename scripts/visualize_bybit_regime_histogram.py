#!/usr/bin/env python3
"""Render SVG/HTML visualizations for Bybit regime hourly histograms."""

from __future__ import annotations

import argparse
import html
import math
from pathlib import Path

import numpy as np
import pandas as pd


REGIMES = ("long_trending", "short_trending", "sideways", "chop")
ENTRY_MODES = ("open", "close")
DIRECTIONS = ("long", "short")
DEFAULT_OUTPUT_DIR = Path("scripts/output/bybit_regime_histogram")


def slug(value: str) -> str:
    return value.lower().replace(" ", "_").replace("/", "_").replace("\\", "_")


def latest_histogram_csv(output_dir: Path) -> Path:
    candidates = sorted(output_dir.glob("*_hourly_histogram.csv"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No *_hourly_histogram.csv files found in {output_dir}")
    return candidates[-1]


def parse_stamp(histogram_path: Path) -> str:
    suffix = "_hourly_histogram.csv"
    name = histogram_path.name
    return name[: -len(suffix)] if name.endswith(suffix) else histogram_path.stem


def metric_label(metric: str) -> str:
    labels = {
        "avg_r": "Average R",
        "win_rate": "Win rate",
        "total_r": "Total R",
        "trades": "Trade count",
    }
    return labels.get(metric, metric)


def format_metric(value: float, metric: str) -> str:
    if not np.isfinite(value):
        return "n/a"
    if metric == "win_rate":
        return f"{value * 100:.1f}%"
    if metric == "trades":
        return f"{value:.0f}"
    return f"{value:.3f}"


def color_for(value: float, metric: str) -> str:
    if not np.isfinite(value):
        return "#94a3b8"
    if metric == "win_rate":
        if value >= 0.55:
            return "#0f766e"
        if value <= 0.45:
            return "#b91c1c"
        return "#64748b"
    if metric == "trades":
        return "#2563eb"
    if value > 0:
        return "#0f766e"
    if value < 0:
        return "#b91c1c"
    return "#64748b"


def panel_domain(values: list[float], metric: str) -> tuple[float, float, float]:
    finite = [float(value) for value in values if np.isfinite(value)]
    if metric == "win_rate":
        return 0.0, 1.0, 0.5
    if metric == "trades":
        top = max(finite) if finite else 1.0
        return 0.0, max(1.0, top), 0.0
    extent = max([abs(value) for value in finite] + [0.05])
    extent = math.ceil(extent * 20.0) / 20.0
    return -extent, extent, 0.0


def svg_text(x: float, y: float, text: str, *, size: int = 12, weight: int = 400, anchor: str = "start", fill: str = "#0f172a") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
        f'text-anchor="{anchor}" fill="{fill}">{html.escape(text)}</text>'
    )


def best_hour_label(group: pd.DataFrame, metric: str) -> str:
    valid = group[pd.to_numeric(group[metric], errors="coerce").notna()].copy()
    if valid.empty:
        return "best hour: n/a"
    valid[metric] = pd.to_numeric(valid[metric], errors="coerce")
    best = valid.sort_values([metric, "trades"], ascending=[False, False]).iloc[0]
    hour = int(best["entry_hour_utc"])
    return (
        f"best {hour:02d}:00 UTC, {metric_label(metric).lower()} "
        f"{format_metric(float(best[metric]), metric)}, n={int(best['trades'])}"
    )


def render_regime_svg(histogram: pd.DataFrame, regime: str, metric: str, output_path: Path) -> None:
    width = 1180
    panel_h = 170
    gap = 34
    top = 86
    bottom = 54
    height = top + (panel_h + gap) * 4 - gap + bottom
    left = 72
    right = 34
    plot_w = width - left - right
    bar_slot = plot_w / 24.0
    bar_w = bar_slot * 0.68
    title = regime.replace("_", " ").title()
    metric_name = metric_label(metric)

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        svg_text(32, 38, f"{title} - Hourly Histogram", size=26, weight=700),
        svg_text(32, 62, f"Bars show {metric_name.lower()} by UTC entry hour. Green is favorable, red is unfavorable.", size=13, fill="#475569"),
    ]

    regime_df = histogram[histogram["regime"] == regime].copy()
    panel_idx = 0
    for entry_mode in ENTRY_MODES:
        for direction in DIRECTIONS:
            y0 = top + panel_idx * (panel_h + gap)
            group = regime_df[
                (regime_df["entry_mode"] == entry_mode)
                & (regime_df["direction"] == direction)
            ].copy()
            group = group.set_index("entry_hour_utc").reindex(range(24)).reset_index()
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(float)
            trades = pd.to_numeric(group["trades"], errors="coerce").fillna(0).to_numpy(int)
            y_min, y_max, baseline = panel_domain(values.tolist(), metric)

            def y_scale(value: float) -> float:
                if y_max == y_min:
                    return y0 + panel_h / 2.0
                return y0 + 18 + (y_max - value) / (y_max - y_min) * (panel_h - 46)

            base_y = y_scale(baseline)
            parts.append(f'<rect x="24" y="{y0 - 22:.1f}" width="{width - 48}" height="{panel_h + 34}" rx="8" fill="#ffffff" stroke="#dbe3ee"/>')
            parts.append(svg_text(42, y0 - 1, f"{entry_mode.title()} entry - {direction.title()}", size=16, weight=700))
            parts.append(svg_text(width - 42, y0 - 1, best_hour_label(group, metric), size=12, anchor="end", fill="#475569"))
            parts.append(f'<line x1="{left}" y1="{base_y:.1f}" x2="{width - right}" y2="{base_y:.1f}" stroke="#94a3b8" stroke-width="1"/>')
            parts.append(svg_text(left - 10, y_scale(y_max) + 4, format_metric(y_max, metric), size=10, anchor="end", fill="#64748b"))
            parts.append(svg_text(left - 10, base_y + 4, format_metric(baseline, metric), size=10, anchor="end", fill="#64748b"))
            parts.append(svg_text(left - 10, y_scale(y_min) + 4, format_metric(y_min, metric), size=10, anchor="end", fill="#64748b"))

            for hour, value in enumerate(values):
                if not np.isfinite(value):
                    continue
                x = left + hour * bar_slot + (bar_slot - bar_w) / 2.0
                y_value = y_scale(value)
                y = min(y_value, base_y)
                h = max(1.0, abs(base_y - y_value))
                parts.append(
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" '
                    f'rx="2" fill="{color_for(float(value), metric)}">'
                    f'<title>{hour:02d}:00 UTC - {metric_name}: {format_metric(float(value), metric)}; trades: {trades[hour]}</title>'
                    "</rect>"
                )
                if hour % 2 == 0:
                    parts.append(svg_text(x + bar_w / 2.0, y0 + panel_h - 4, f"{hour:02d}", size=9, anchor="middle", fill="#64748b"))

            parts.append(svg_text(left, y0 + panel_h + 16, "UTC hour", size=11, fill="#64748b"))
            panel_idx += 1

    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def top_cells_table(histogram: pd.DataFrame, metric: str, limit: int) -> str:
    filtered = histogram[histogram["trades"] > 0].copy()
    filtered[metric] = pd.to_numeric(filtered[metric], errors="coerce")
    filtered = filtered.dropna(subset=[metric]).sort_values([metric, "trades"], ascending=[False, False]).head(limit)
    rows = [
        "<tr><th>Regime</th><th>Entry</th><th>Side</th><th>Hour UTC</th><th>Trades</th><th>Win rate</th><th>Avg R</th><th>Total R</th></tr>"
    ]
    for _, row in filtered.iterrows():
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['regime']).replace('_', ' '))}</td>"
            f"<td>{html.escape(str(row['entry_mode']))}</td>"
            f"<td>{html.escape(str(row['direction']))}</td>"
            f"<td>{int(row['entry_hour_utc']):02d}:00</td>"
            f"<td>{int(row['trades'])}</td>"
            f"<td>{format_metric(float(row['win_rate']), 'win_rate')}</td>"
            f"<td>{format_metric(float(row['avg_r']), 'avg_r')}</td>"
            f"<td>{format_metric(float(row['total_r']), 'total_r')}</td>"
            "</tr>"
        )
    return "<table>" + "\n".join(rows) + "</table>"


def render_html(histogram_path: Path, svg_paths: list[Path], metric: str, output_path: Path) -> None:
    histogram = pd.read_csv(histogram_path)
    stamp = parse_stamp(histogram_path)
    total_trades = int(histogram["trades"].sum())
    images = "\n".join(
        f'<section><img src="{html.escape(path.name)}" alt="{html.escape(path.stem)}"></section>'
        for path in svg_paths
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(stamp)} regime histograms</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #0f172a;
      background: #e5edf5;
    }}
    body {{
      margin: 0;
      padding: 32px;
    }}
    main {{
      max-width: 1240px;
      margin: 0 auto;
    }}
    header {{
      margin-bottom: 22px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 30px;
      line-height: 1.15;
    }}
    p {{
      margin: 0;
      color: #475569;
    }}
    section {{
      margin: 18px 0;
    }}
    img {{
      width: 100%;
      height: auto;
      display: block;
      border: 1px solid #cbd5e1;
      border-radius: 10px;
      background: #f8fafc;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 18px 0 28px;
      background: #ffffff;
      border: 1px solid #cbd5e1;
      border-radius: 10px;
      overflow: hidden;
    }}
    th, td {{
      padding: 9px 10px;
      border-bottom: 1px solid #e2e8f0;
      text-align: left;
      font-size: 13px;
    }}
    th {{
      background: #f1f5f9;
      font-weight: 700;
    }}
    tr:last-child td {{
      border-bottom: 0;
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>{html.escape(stamp.upper())} Regime Hourly Histograms</h1>
      <p>Metric: {html.escape(metric_label(metric))}. Hours are UTC. Total histogram cells sum to {total_trades:,} trade observations across long/short and open/close variants.</p>
    </header>
    {top_cells_table(histogram, metric, 16)}
    {images}
  </main>
</body>
</html>
"""
    output_path.write_text(document, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render SVG and HTML charts from a Bybit regime hourly histogram CSV.")
    parser.add_argument("--histogram", type=Path, help="Path to *_hourly_histogram.csv. Defaults to newest output file.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for report outputs.")
    parser.add_argument("--metric", choices=("avg_r", "win_rate", "total_r", "trades"), default="avg_r")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    histogram_path = args.histogram or latest_histogram_csv(args.output_dir)
    histogram = pd.read_csv(histogram_path)
    missing = {"regime", "entry_mode", "direction", "entry_hour_utc", args.metric, "trades"} - set(histogram.columns)
    if missing:
        raise ValueError(f"Histogram file is missing columns: {sorted(missing)}")

    stamp = parse_stamp(histogram_path)
    svg_paths: list[Path] = []
    for regime in REGIMES:
        svg_path = args.output_dir / f"{stamp}_{slug(regime)}_{args.metric}.svg"
        render_regime_svg(histogram, regime, args.metric, svg_path)
        svg_paths.append(svg_path)

    html_path = args.output_dir / f"{stamp}_{args.metric}_report.html"
    render_html(histogram_path, svg_paths, args.metric, html_path)
    print("Rendered visualization files:")
    print(f"  {html_path}")
    for path in svg_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
