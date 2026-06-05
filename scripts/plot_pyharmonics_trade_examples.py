from __future__ import annotations

import argparse
import html
import pickle
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyharmonics.patterns import ABCDPattern, ABCPattern, XABCDPattern
from pyharmonics.plotter import HarmonicPlotter
from pyharmonics.technicals import OHLCTechnicals

from scripts.backtest_pyharmonics_strategy import (  # noqa: E402
    HarmonicPatternEvent,
    to_pyharmonics_frame,
)
from scripts.backtest_wolfe_wave import ensure_ohlcv_frame  # noqa: E402


DEFAULT_RUN_DIR = Path("scripts/pyharmonics_focus_filters_v2_inj_link_ltc_15m_abcd_20260603_141333")


@dataclass(frozen=True)
class TradeExample:
    symbol: str
    label: str
    row: pd.Series
    event: HarmonicPatternEvent
    html_path: Path


def parse_command_args(run_dir: Path) -> dict[str, str]:
    command_path = run_dir / "command.txt"
    if not command_path.exists():
        return {}
    tokens = shlex.split(command_path.read_text(encoding="utf-8"), posix=False)
    out: dict[str, str] = {}
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token.startswith("--"):
            key = token[2:]
            value = "true"
            if idx + 1 < len(tokens) and not tokens[idx + 1].startswith("--"):
                value = tokens[idx + 1]
                idx += 1
            out[key] = value
        idx += 1
    return out


def parse_csv_values(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def load_symbol_frame(cache_dir: Path, symbol: str, timeframe: str) -> pd.DataFrame:
    path = cache_dir / f"{symbol.lower()}_{timeframe}_bybit.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing cached OHLCV for {symbol}: {path}")
    return ensure_ohlcv_frame(pd.read_csv(path))


def load_events(run_dir: Path, symbol: str) -> dict[str, HarmonicPatternEvent]:
    cache_dir = run_dir / "per_symbol" / "event_cache"
    paths = sorted(cache_dir.glob(f"{symbol.lower()}_*.pkl"))
    if not paths:
        raise FileNotFoundError(f"Missing event cache for {symbol} in {cache_dir}")
    events: dict[str, HarmonicPatternEvent] = {}
    for path in paths:
        payload = pickle.loads(path.read_bytes())
        for event in payload:
            events[event.event_key] = event
    return events


def event_to_pyharmonics_pattern(event: HarmonicPatternEvent) -> ABCPattern:
    points = len(event.x)
    cls: type[ABCPattern]
    if points == 5:
        cls = XABCDPattern
    elif points == 4:
        cls = ABCDPattern
    else:
        cls = ABCPattern
    return cls(
        event.symbol,
        event.pattern_tf,
        tuple(event.x),
        list(event.y),
        event.name,
        dict(event.retraces),
        bool(event.formed),
        event.direction == "long",
    )


def select_examples(trades: pd.DataFrame, max_per_symbol: int) -> list[tuple[str, pd.Series]]:
    rows: list[tuple[str, pd.Series]] = []
    seen: set[tuple[str, str]] = set()

    def add(label: str, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        row = frame.iloc[0]
        key = (str(row["event_key"]), str(row["entry_time"]))
        if key in seen:
            return
        seen.add(key)
        rows.append((label, row))

    exit_reason = trades["exit_reason"].astype(str)
    add("best_target", trades[exit_reason.str.contains("target", case=False, na=False)].nlargest(1, "r_multiple_net"))
    add("best_timeout", trades[exit_reason.eq("timeout")].nlargest(1, "r_multiple_net"))
    add("worst_stop", trades[exit_reason.str.contains("stop", case=False, na=False)].nsmallest(1, "r_multiple_net"))

    if len(rows) < max_per_symbol:
        for _, row in trades.sort_values("r_multiple_net", ascending=False).iterrows():
            add("extra_winner", pd.DataFrame([row]))
            if len(rows) >= max_per_symbol:
                break
    return rows[:max_per_symbol]


def clean_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value).strip("_").lower()


def as_ts(value: Any) -> pd.Timestamp:
    return pd.Timestamp(value).tz_convert("UTC")


def add_price_line(
    fig: go.Figure,
    *,
    x0: pd.Timestamp,
    x1: pd.Timestamp,
    y: float,
    label: str,
    color: str,
    dash: str = "solid",
) -> None:
    fig.add_shape(
        type="line",
        x0=x0,
        x1=x1,
        y0=y,
        y1=y,
        line={"color": color, "width": 2, "dash": dash},
        row=1,
        col=1,
    )
    fig.add_annotation(
        x=x1,
        y=y,
        text=label,
        showarrow=False,
        xanchor="left",
        yanchor="middle",
        font={"size": 11, "color": color},
        bgcolor="rgba(255,255,255,0.75)",
        row=1,
        col=1,
    )


def plot_trade(
    *,
    symbol: str,
    label: str,
    trade: pd.Series,
    event: HarmonicPatternEvent,
    frame: pd.DataFrame,
    output_path: Path,
    pre_bars: int,
    post_bars: int,
) -> TradeExample:
    open_times = pd.to_datetime(frame["open_time"], utc=True)
    entry_time = as_ts(trade["entry_time"])
    exit_time = as_ts(trade["exit_time"])
    event_start = min(event.x[0], event.completion_time, entry_time)
    event_end = max(event.completion_time, entry_time, exit_time)

    start_idx = max(0, int(open_times.searchsorted(event_start, side="left")) - pre_bars)
    end_idx = min(len(frame), int(open_times.searchsorted(event_end, side="right")) + post_bars)
    plot_frame = frame.iloc[start_idx:end_idx].copy()
    if plot_frame.empty:
        raise ValueError(f"No candles selected for {symbol} {label} {entry_time}")

    technicals = OHLCTechnicals(to_pyharmonics_frame(plot_frame), symbol, str(trade["exec_tf"]), peak_spacing=int(trade["peak_spacing"]))
    plotter = HarmonicPlotter(technicals, plot_ema=False, plot_sma=False)
    plotter.add_harmonic_pattern(event_to_pyharmonics_pattern(event))
    fig = plotter.main_plot

    pattern_low = min(event.completion_min_price, event.completion_max_price, event.completion_price)
    pattern_high = max(event.completion_min_price, event.completion_max_price, event.completion_price)
    if pattern_high > pattern_low:
        fig.add_shape(
            type="rect",
            x0=event.completion_time,
            x1=entry_time,
            y0=pattern_low,
            y1=pattern_high,
            line={"color": "rgba(245, 158, 11, 0.65)", "width": 1, "dash": "dot"},
            fillcolor="rgba(245, 158, 11, 0.10)",
            row=1,
            col=1,
        )

    entry = float(trade["entry_price"])
    stop = float(trade["stop_price"])
    structural_stop = float(trade["structural_stop_price"])
    target = float(trade["target_price"])
    exit_price = float(trade["exit_price"])
    net_r = float(trade["r_multiple_net"])
    direction = str(trade["direction"]).lower()
    marker_symbol = "triangle-up" if direction == "long" else "triangle-down"
    exit_color = "#15803d" if net_r > 0 else "#b91c1c"

    add_price_line(fig, x0=entry_time, x1=exit_time, y=entry, label="entry", color="#2563eb")
    add_price_line(fig, x0=entry_time, x1=exit_time, y=structural_stop, label="structure", color="#f97316", dash="dash")
    add_price_line(fig, x0=entry_time, x1=exit_time, y=stop, label="SL", color="#dc2626")
    add_price_line(fig, x0=entry_time, x1=exit_time, y=target, label=f"TP {float(trade['target_rr_planned']):.1f}R", color="#16a34a")
    breakeven_trigger_price = pd.to_numeric(pd.Series([trade.get("breakeven_trigger_price")]), errors="coerce").iloc[0]
    if pd.notna(breakeven_trigger_price):
        add_price_line(
            fig,
            x0=entry_time,
            x1=exit_time,
            y=float(breakeven_trigger_price),
            label=f"BE {float(trade.get('breakeven_trigger_r', 0.0)):.2g}R",
            color="#7c3aed",
            dash="dot",
        )

    fig.add_trace(
        go.Scatter(
            x=[entry_time],
            y=[entry],
            mode="markers+text",
            marker={"symbol": marker_symbol, "size": 13, "color": "#2563eb", "line": {"width": 1, "color": "#111827"}},
            text=["ENTRY"],
            textposition="top center",
            name="entry",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[exit_time],
            y=[exit_price],
            mode="markers+text",
            marker={"symbol": "x", "size": 13, "color": exit_color, "line": {"width": 2, "color": exit_color}},
            text=[f"EXIT {net_r:+.2f}R"],
            textposition="bottom center",
            name="exit",
        ),
        row=1,
        col=1,
    )

    trigger_time = pd.to_datetime(trade.get("trigger_time"), utc=True, errors="coerce")
    if pd.notna(trigger_time):
        fig.add_vline(
            x=trigger_time,
            line={"color": "#64748b", "width": 1, "dash": "dot"},
            row=1,
            col=1,
        )
        fig.add_annotation(
            x=trigger_time,
            y=entry,
            text=f"trigger: {trade.get('trigger_candle_primary', 'none')}",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-45,
            font={"size": 11, "color": "#334155"},
            bgcolor="rgba(255,255,255,0.78)",
            row=1,
            col=1,
        )
    breakeven_activation_time = pd.to_datetime(trade.get("breakeven_activation_time"), utc=True, errors="coerce")
    if pd.notna(breakeven_activation_time):
        fig.add_vline(
            x=breakeven_activation_time,
            line={"color": "#7c3aed", "width": 1, "dash": "dash"},
            row=1,
            col=1,
        )

    quality = pd.to_numeric(pd.Series([trade.get("harmonic_quality_score")]), errors="coerce").iloc[0]
    quality_text = f" | quality {float(quality):.0f}" if pd.notna(quality) else ""
    title = (
        f"{symbol} {trade['exec_tf']} {event.name} {direction.upper()} | {label} | "
        f"{trade['exit_reason']} {net_r:+.2f}R{quality_text}"
    )
    subtitle = (
        f"Entry {entry:g} | Structure {structural_stop:g} | SL {stop:g} | "
        f"TP {target:g} ({float(trade['target_rr_planned']):.1f}R) | "
        f"stop buffer {float(trade['stop_atr_buffer']):.2f} ATR | "
        f"BE after {float(trade.get('breakeven_trigger_r', 0.0)):.2g}R"
    )
    fig.update_layout(
        title={"text": f"{html.escape(title)}<br><sup>{html.escape(subtitle)}</sup>", "x": 0.01, "xanchor": "left"},
        template="plotly_white",
        hovermode="x unified",
        width=1400,
        height=900,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"l": 60, "r": 120, "t": 90, "b": 45},
    )
    fig.update_xaxes(rangeslider_visible=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="directory", full_html=True)
    return TradeExample(symbol=symbol, label=label, row=trade, event=event, html_path=output_path)


def write_index(output_dir: Path, examples: list[TradeExample], run_dir: Path) -> None:
    rows = []
    for example in examples:
        trade = example.row
        rel = example.html_path.name
        rows.append(
            "<tr>"
            f"<td><a href=\"{html.escape(rel)}\">{html.escape(example.symbol)}</a></td>"
            f"<td>{html.escape(example.label)}</td>"
            f"<td>{html.escape(str(trade['direction']))}</td>"
            f"<td>{html.escape(str(trade['pattern_name']))}</td>"
            f"<td>{html.escape(str(trade['entry_time']))}</td>"
            f"<td>{html.escape(str(trade['exit_reason']))}</td>"
            f"<td>{float(trade['r_multiple_net']):+.3f}R</td>"
            f"<td>{float(trade['entry_price']):g}</td>"
            f"<td>{float(trade['structural_stop_price']):g}</td>"
            f"<td>{float(trade['stop_price']):g}</td>"
            f"<td>{float(trade['target_price']):g}</td>"
            "</tr>"
        )
    body = "\n".join(rows)
    index = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Pyharmonics Trade Examples</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #111827; }}
    table {{ border-collapse: collapse; min-width: 1100px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; white-space: nowrap; }}
    th {{ font-size: 12px; color: #475569; text-transform: uppercase; }}
    .note {{ max-width: 980px; color: #475569; line-height: 1.45; }}
  </style>
</head>
<body>
  <h1>Pyharmonics Trade Examples</h1>
  <p class="note">
    Source run: {html.escape(str(run_dir))}. The orange dashed line is structural invalidation.
    The red SL is structural invalidation plus the configured ATR buffer. TP is entry plus/minus
    configured RR times entry risk. Net R includes the backtest fee/slippage model.
  </p>
  <table>
    <thead>
      <tr>
        <th>Symbol</th><th>Sample</th><th>Side</th><th>Pattern</th><th>Entry Time</th>
        <th>Exit</th><th>Net R</th><th>Entry</th><th>Structure</th><th>SL</th><th>TP</th>
      </tr>
    </thead>
    <tbody>
      {body}
    </tbody>
  </table>
</body>
</html>
"""
    (output_dir / "index.html").write_text(index, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot selected pyharmonics trades from a low-pass research run.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--symbols", default="")
    parser.add_argument("--exec-tf", default="")
    parser.add_argument("--max-per-symbol", type=int, default=3)
    parser.add_argument("--pre-bars", type=int, default=220)
    parser.add_argument("--post-bars", type=int, default=80)
    args = parser.parse_args()

    run_dir = args.run_dir
    command_args = parse_command_args(run_dir)
    cache_dir_value = args.cache_dir or command_args.get("cache-dir")
    if not cache_dir_value:
        raise SystemExit("Pass --cache-dir or use a run directory with command.txt containing --cache-dir.")
    cache_dir = Path(cache_dir_value)
    exec_tf = args.exec_tf or command_args.get("exec-tf", "15m")
    output_dir = args.output_dir or (run_dir / "trade_plots")
    per_symbol = run_dir / "per_symbol"

    symbols = parse_csv_values(args.symbols)
    if not symbols:
        symbols = sorted(path.name.split("_pyharmonics_selected_trades.csv")[0].upper() for path in per_symbol.glob("*_pyharmonics_selected_trades.csv"))

    examples: list[TradeExample] = []
    for symbol in symbols:
        trades_path = per_symbol / f"{symbol.lower()}_pyharmonics_selected_trades.csv"
        if not trades_path.exists():
            print(f"[{symbol}] missing selected trades: {trades_path}", file=sys.stderr)
            continue
        trades = pd.read_csv(trades_path)
        if trades.empty:
            print(f"[{symbol}] no selected trades", file=sys.stderr)
            continue
        events = load_events(run_dir, symbol)
        frame = load_symbol_frame(cache_dir, symbol, exec_tf)
        for label, row in select_examples(trades, max(1, args.max_per_symbol)):
            event = events.get(str(row["event_key"]))
            if event is None:
                print(f"[{symbol}] event not found for {row['event_key']}", file=sys.stderr)
                continue
            filename = (
                f"{clean_name(symbol)}_{clean_name(label)}_{clean_name(str(row['direction']))}_"
                f"{clean_name(str(row['pattern_name']))}_{clean_name(str(row['entry_time'])[:16])}.html"
            )
            example = plot_trade(
                symbol=symbol,
                label=label,
                trade=row,
                event=event,
                frame=frame,
                output_path=output_dir / filename,
                pre_bars=max(20, args.pre_bars),
                post_bars=max(20, args.post_bars),
            )
            examples.append(example)
            print(f"[{symbol}] wrote {example.html_path}")

    if not examples:
        raise SystemExit("No examples were plotted.")
    write_index(output_dir, examples, run_dir)
    print(f"Index: {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
