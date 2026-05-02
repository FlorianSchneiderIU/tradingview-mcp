from __future__ import annotations

import argparse
import html
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backtest_turtle_soup import Config, Trade, fetch_klines, run_backtest


COLORS = {
    "bg": "#f7f4ea",
    "panel": "#fffdf7",
    "grid": "#e7dfca",
    "text": "#2a241b",
    "muted": "#746655",
    "bull": "#0f9d58",
    "bear": "#d93025",
    "zone": "#d7b66f",
    "ob": "#ef8f5a",
    "entry": "#0b57d0",
    "stop": "#b3261e",
    "target": "#188038",
    "sweep": "#8e24aa",
    "choch": "#ff8f00",
    "signal": "#00897b",
    "exit": "#5f6368",
}


def fmt_ts(ts) -> str:
    return ts.strftime("%Y-%m-%d %H:%M UTC")


def pick_trade(trades: list[Trade], trade_index: int | None) -> tuple[int, Trade]:
    if not trades:
        raise ValueError("No trades found for the requested configuration.")
    if trade_index is not None:
        if trade_index < 0:
            trade_index = len(trades) + trade_index
        if trade_index < 0 or trade_index >= len(trades):
            raise IndexError(f"Trade index {trade_index} is out of range for {len(trades)} trades.")
        return trade_index, trades[trade_index]

    for idx in range(len(trades) - 1, -1, -1):
        trade = trades[idx]
        if trade.exit_reason == "target":
            return idx, trade
    for idx in range(len(trades) - 1, -1, -1):
        trade = trades[idx]
        if trade.r_multiple > 0:
            return idx, trade
    return len(trades) - 1, trades[-1]


def build_svg(exec_df, trade: Trade, trade_index: int, symbol: str, out_path: Path) -> None:
    left_bars = 36
    right_bars = 12
    start_idx = max(0, trade.sweep_index - left_bars)
    end_idx = min(len(exec_df) - 1, trade.exit_index + right_bars)
    window = exec_df.iloc[start_idx:end_idx + 1].reset_index(drop=True)
    trade_local_entry = trade.entry_index - start_idx
    trade_local_exit = trade.exit_index - start_idx
    local_sweep = trade.sweep_index - start_idx
    local_choch = trade.choch_index - start_idx
    local_signal = trade.signal_index - start_idx

    width = 1600
    height = 980
    chart_left = 90
    chart_right = 1220
    chart_top = 110
    chart_bottom = 820
    chart_width = chart_right - chart_left
    chart_height = chart_bottom - chart_top
    side_x = 1260
    side_w = 290

    bars = len(window)
    bar_step = chart_width / max(1, bars)
    candle_body = max(3.0, bar_step * 0.62)

    price_min = min(
        float(window["low"].min()),
        trade.stop_price,
        trade.zone_bottom,
        trade.ob_bottom,
        trade.exit_price,
        trade.entry_price,
    )
    price_max = max(
        float(window["high"].max()),
        trade.target_price,
        trade.zone_top,
        trade.ob_top,
        trade.exit_price,
        trade.entry_price,
    )
    pad = (price_max - price_min) * 0.08 or trade.entry_price * 0.01
    price_min -= pad
    price_max += pad

    def x_pos(local_idx: int) -> float:
        return chart_left + (local_idx + 0.5) * bar_step

    def y_pos(price: float) -> float:
        return chart_bottom - ((price - price_min) / (price_max - price_min)) * chart_height

    def price_line(price: float, color: str, label: str, dash: str = "8 6") -> str:
        y = y_pos(price)
        return (
            f'<line x1="{chart_left}" y1="{y:.2f}" x2="{chart_right}" y2="{y:.2f}" '
            f'stroke="{color}" stroke-width="1.6" stroke-dasharray="{dash}" opacity="0.95" />'
            f'<text x="{chart_right + 10}" y="{y + 4:.2f}" font-size="14" fill="{color}" '
            f'font-family="Segoe UI, Arial">{html.escape(label)} {price:.2f}</text>'
        )

    def vertical_marker(local_idx: int, color: str, label: str) -> str:
        x = x_pos(local_idx)
        return (
            f'<line x1="{x:.2f}" y1="{chart_top}" x2="{x:.2f}" y2="{chart_bottom}" '
            f'stroke="{color}" stroke-width="1.8" stroke-dasharray="6 5" opacity="0.9" />'
            f'<text x="{x + 6:.2f}" y="{chart_top - 10}" font-size="13" fill="{color}" '
            f'font-family="Segoe UI, Arial">{html.escape(label)}</text>'
        )

    svg_parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{COLORS["bg"]}" />',
        f'<rect x="40" y="40" width="{width - 80}" height="{height - 80}" rx="24" fill="{COLORS["panel"]}" stroke="{COLORS["grid"]}" stroke-width="2" />',
        f'<text x="90" y="78" font-size="28" font-weight="700" fill="{COLORS["text"]}" font-family="Segoe UI, Arial">Turtle Soup Sample Trade</text>',
        f'<text x="90" y="100" font-size="16" fill="{COLORS["muted"]}" font-family="Segoe UI, Arial">{html.escape(symbol)} | exec {trade.exec_tf} | structure {trade.structure_tf} | entry {trade.entry_mode} | trade #{trade_index}</text>',
    ]

    for step in range(6):
        price = price_min + (price_max - price_min) * step / 5
        y = y_pos(price)
        svg_parts.append(
            f'<line x1="{chart_left}" y1="{y:.2f}" x2="{chart_right}" y2="{y:.2f}" '
            f'stroke="{COLORS["grid"]}" stroke-width="1" />'
        )
        svg_parts.append(
            f'<text x="58" y="{y + 5:.2f}" font-size="14" fill="{COLORS["muted"]}" '
            f'font-family="Segoe UI, Arial">{price:.2f}</text>'
        )

    for step in range(7):
        idx = round((bars - 1) * step / 6)
        x = x_pos(idx)
        label = fmt_ts(window.iloc[idx]["open_time"])
        svg_parts.append(
            f'<line x1="{x:.2f}" y1="{chart_top}" x2="{x:.2f}" y2="{chart_bottom}" '
            f'stroke="{COLORS["grid"]}" stroke-width="1" opacity="0.9" />'
        )
        svg_parts.append(
            f'<text x="{x - 38:.2f}" y="{chart_bottom + 24}" font-size="12" fill="{COLORS["muted"]}" '
            f'font-family="Segoe UI, Arial">{html.escape(label[5:16])}</text>'
        )

    zone_y1 = y_pos(trade.zone_top)
    zone_y2 = y_pos(trade.zone_bottom)
    svg_parts.append(
        f'<rect x="{chart_left}" y="{min(zone_y1, zone_y2):.2f}" width="{chart_width}" '
        f'height="{abs(zone_y2 - zone_y1):.2f}" fill="{COLORS["zone"]}" opacity="0.18" />'
    )
    svg_parts.append(
        f'<text x="{chart_left + 8}" y="{min(zone_y1, zone_y2) - 8:.2f}" font-size="14" '
        f'fill="{COLORS["zone"]}" font-family="Segoe UI, Arial">{html.escape(trade.zone_tf.upper())} key zone</text>'
    )

    ob_x1 = x_pos(local_choch) - bar_step * 0.5
    ob_x2 = x_pos(local_signal) + bar_step * 0.5
    ob_y1 = y_pos(trade.ob_top)
    ob_y2 = y_pos(trade.ob_bottom)
    svg_parts.append(
        f'<rect x="{ob_x1:.2f}" y="{min(ob_y1, ob_y2):.2f}" width="{max(4.0, ob_x2 - ob_x1):.2f}" '
        f'height="{abs(ob_y2 - ob_y1):.2f}" fill="{COLORS["ob"]}" opacity="0.18" stroke="{COLORS["ob"]}" stroke-width="1.4" />'
    )
    svg_parts.append(
        f'<text x="{ob_x1 + 8:.2f}" y="{min(ob_y1, ob_y2) - 8:.2f}" font-size="14" '
        f'fill="{COLORS["ob"]}" font-family="Segoe UI, Arial">OB</text>'
    )

    for i, row in window.iterrows():
        x = x_pos(i)
        open_price = float(row["open"])
        close_price = float(row["close"])
        high_price = float(row["high"])
        low_price = float(row["low"])
        bullish = close_price >= open_price
        color = COLORS["bull"] if bullish else COLORS["bear"]
        body_top = y_pos(max(open_price, close_price))
        body_bottom = y_pos(min(open_price, close_price))
        wick_top = y_pos(high_price)
        wick_bottom = y_pos(low_price)
        body_h = max(1.8, body_bottom - body_top)
        svg_parts.append(
            f'<line x1="{x:.2f}" y1="{wick_top:.2f}" x2="{x:.2f}" y2="{wick_bottom:.2f}" '
            f'stroke="{color}" stroke-width="1.5" />'
        )
        svg_parts.append(
            f'<rect x="{x - candle_body / 2:.2f}" y="{body_top:.2f}" width="{candle_body:.2f}" '
            f'height="{body_h:.2f}" fill="{color}" opacity="0.88" />'
        )

    svg_parts.extend([
        price_line(trade.entry_price, COLORS["entry"], "Entry"),
        price_line(trade.stop_price, COLORS["stop"], "Stop"),
        price_line(trade.target_price, COLORS["target"], "Target"),
        vertical_marker(local_sweep, COLORS["sweep"], "Sweep"),
        vertical_marker(local_choch, COLORS["choch"], "CHoCH"),
        vertical_marker(local_signal, COLORS["signal"], "Signal"),
        vertical_marker(trade_local_entry, COLORS["entry"], "Entry"),
        vertical_marker(trade_local_exit, COLORS["exit"], "Exit"),
    ])

    entry_x = x_pos(trade_local_entry)
    entry_y = y_pos(trade.entry_price)
    exit_x = x_pos(trade_local_exit)
    exit_y = y_pos(trade.exit_price)
    svg_parts.append(
        f'<circle cx="{entry_x:.2f}" cy="{entry_y:.2f}" r="6.5" fill="{COLORS["entry"]}" stroke="white" stroke-width="2" />'
    )
    svg_parts.append(
        f'<circle cx="{exit_x:.2f}" cy="{exit_y:.2f}" r="6.5" fill="{COLORS["exit"]}" stroke="white" stroke-width="2" />'
    )

    summary_lines = [
        ("Direction", trade.direction.upper()),
        ("Result", f"{trade.r_multiple:.3f}R via {trade.exit_reason}"),
        ("Sweep", fmt_ts(trade.sweep_time)),
        ("CHoCH", fmt_ts(trade.choch_time)),
        ("Signal", fmt_ts(trade.signal_time)),
        ("Entry", fmt_ts(trade.entry_time)),
        ("Exit", fmt_ts(trade.exit_time)),
        ("Zone", f"{trade.zone_bottom:.2f} - {trade.zone_top:.2f} ({trade.zone_tf.upper()})"),
        ("OB", f"{trade.ob_bottom:.2f} - {trade.ob_top:.2f}"),
        ("Entry / Stop", f"{trade.entry_price:.2f} / {trade.stop_price:.2f}"),
        ("Target / Exit", f"{trade.target_price:.2f} / {trade.exit_price:.2f}"),
        ("Hold", f"{trade.hold_bars} bars"),
    ]

    svg_parts.append(
        f'<rect x="{side_x}" y="{chart_top}" width="{side_w}" height="{chart_bottom - chart_top}" rx="18" fill="#fbf7ec" stroke="{COLORS["grid"]}" stroke-width="1.5" />'
    )
    svg_parts.append(
        f'<text x="{side_x + 20}" y="{chart_top + 34}" font-size="22" font-weight="700" fill="{COLORS["text"]}" font-family="Segoe UI, Arial">Trade Details</text>'
    )
    y = chart_top + 74
    for label, value in summary_lines:
        svg_parts.append(
            f'<text x="{side_x + 20}" y="{y:.2f}" font-size="13" fill="{COLORS["muted"]}" font-family="Segoe UI, Arial">{html.escape(label)}</text>'
        )
        svg_parts.append(
            f'<text x="{side_x + 20}" y="{y + 22:.2f}" font-size="16" fill="{COLORS["text"]}" font-family="Segoe UI, Arial">{html.escape(value)}</text>'
        )
        y += 52

    svg_parts.append(
        f'<text x="{chart_left}" y="{chart_bottom + 52}" font-size="14" fill="{COLORS["muted"]}" font-family="Segoe UI, Arial">Interpretation: price taps the higher-timeframe zone, reverses, confirms CHoCH on {trade.structure_tf}, forms the last opposite-candle OB, and then fills the {trade.entry_mode} entry on {trade.exec_tf}.</text>'
    )
    svg_parts.append("</svg>")

    out_path.write_text("\n".join(svg_parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--exec-tf", default="5m")
    parser.add_argument("--structure-tf", default="15m")
    parser.add_argument("--entry-mode", default="limit_mid")
    parser.add_argument("--trade-index", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=args.days)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    exec_df = fetch_klines(args.symbol, args.exec_tf, start_ms, end_ms)
    cfg = Config(exec_tf=args.exec_tf, structure_tf=args.structure_tf, entry_mode=args.entry_mode)
    trades = run_backtest(exec_df, cfg)
    trade_index, trade = pick_trade(trades, args.trade_index)

    output = Path(args.output) if args.output else Path("screenshots") / (
        f"turtle_soup_trade_{args.symbol.lower()}_{args.exec_tf}_{args.structure_tf}_{args.entry_mode}_{trade_index}.svg"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    build_svg(exec_df, trade, trade_index, args.symbol, output)

    print(f"Wrote {output}")
    print(
        f"Trade #{trade_index}: {trade.direction} {trade.entry_time} -> {trade.exit_time} | "
        f"{trade.r_multiple:.3f}R | {trade.exit_reason}"
    )


if __name__ == "__main__":
    main()
