from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageColor, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.channel_state_research.data import load_base_candles, prepare_timeframe_bars


Image.MAX_IMAGE_PIXELS = None


TIMEFRAME_MINUTES = {
    "5m": 5,
    "15m": 15,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
    "1w": 10080,
}

CHANNEL_STYLES = {
    "15m": {"fill": "#F4A26133", "line": "#E76F51", "wick": "#F4A261"},
    "1h": {"fill": "#4D96FF26", "line": "#2D6CDF", "wick": "#6FA8FF"},
    "4h": {"fill": "#9B5DE526", "line": "#7B2CBF", "wick": "#B48CFF"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a single-image trade contact sheet for zone-channel runs.")
    parser.add_argument("--prefix", type=Path, required=True, help="Trade/decision prefix to load.")
    parser.add_argument(
        "--signals-prefix",
        type=Path,
        default=None,
        help="Optional alternate prefix for *_signals.csv.",
    )
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--panel-width", type=int, default=380)
    parser.add_argument("--panel-height", type=int, default=220)
    parser.add_argument("--pre-bars", type=int, default=16)
    parser.add_argument("--post-bars", type=int, default=8)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0, help="Optional trade limit for testing; 0 means all.")
    return parser.parse_args()


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "arialbd.ttf" if bold else "arial.ttf",
        "segoeuib.ttf" if bold else "segoeui.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def rgba(color: str) -> tuple[int, int, int, int]:
    return ImageColor.getcolor(color, "RGBA")


def timeframe_minutes(name: str) -> int:
    if name not in TIMEFRAME_MINUTES:
        raise KeyError(f"Unsupported timeframe {name!r}")
    return TIMEFRAME_MINUTES[name]


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def project_channel_line(
    trade_row: pd.Series,
    timeframe: str,
    family: str,
    side: str,
    local_indices: list[int],
    signal_index: int,
    decision_tf: str,
) -> list[float | None]:
    boundary_col = f"{side}_{family}_boundary_{timeframe}"
    slope_col = f"{side}_{family}_slope_{timeframe}"
    boundary = safe_float(trade_row.get(boundary_col))
    slope = safe_float(trade_row.get(slope_col))
    if boundary is None or slope is None:
        return [None] * len(local_indices)
    step_ratio = timeframe_minutes(decision_tf) / timeframe_minutes(timeframe)
    values: list[float | None] = []
    for idx in local_indices:
        delta = (idx - signal_index) * step_ratio
        values.append(boundary + slope * delta)
    return values


def family_is_valid(trade_row: pd.Series, timeframe: str, family: str) -> bool:
    flag_col = f"{family}_valid_flag_{timeframe}"
    flag = safe_float(trade_row.get(flag_col))
    return bool(flag is not None and flag >= 0.5)


def draw_polyline(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], fill: tuple[int, int, int, int], width: int = 1) -> None:
    if len(points) < 2:
        return
    draw.line(points, fill=fill, width=width)


def adjust_label_positions(items: list[dict[str, Any]], y_min: float, y_max: float, gap: float = 11.0) -> list[dict[str, Any]]:
    ordered = sorted(items, key=lambda item: item["y"])
    for idx, item in enumerate(ordered):
        if idx == 0:
            item["adj_y"] = max(y_min, min(y_max, item["y"]))
            continue
        item["adj_y"] = max(item["y"], ordered[idx - 1]["adj_y"] + gap)
    for idx in range(len(ordered) - 2, -1, -1):
        ordered[idx]["adj_y"] = min(ordered[idx]["adj_y"], ordered[idx + 1]["adj_y"] - gap)
        ordered[idx]["adj_y"] = max(y_min, ordered[idx]["adj_y"])
    for item in ordered:
        item["adj_y"] = max(y_min, min(y_max, item["adj_y"]))
    return ordered


def year_color(value: float) -> tuple[int, int, int]:
    if value > 0.0:
        return (25, 135, 84)
    if value < 0.0:
        return (171, 28, 51)
    return (120, 120, 120)


def render_panel(
    sheet: Image.Image,
    top_left: tuple[int, int],
    trade_row: pd.Series,
    exec_frame: pd.DataFrame,
    decision_tf: str,
    fonts: dict[str, ImageFont.ImageFont],
    *,
    pre_bars: int,
    post_bars: int,
    panel_width: int,
    panel_height: int,
) -> None:
    panel = Image.new("RGBA", (panel_width, panel_height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(panel, "RGBA")

    net_r = safe_float(trade_row.get("r_multiple_net")) or 0.0
    exit_reason = str(trade_row.get("exit_reason", ""))
    border = rgba("#2E8B57") if net_r > 0.0 else rgba("#B22222")
    if exit_reason == "timeout":
        border = rgba("#8A6D1D")
    draw.rounded_rectangle((0, 0, panel_width - 1, panel_height - 1), radius=8, outline=border, width=2, fill=(250, 251, 252, 255))

    header_h = 34
    footer_h = 6
    plot_left = 12
    plot_right = panel_width - 68
    plot_top = header_h + 6
    plot_bottom = panel_height - footer_h - 12
    plot_w = plot_right - plot_left
    plot_h = plot_bottom - plot_top

    entry_index = int(trade_row["entry_index"])
    exit_index = int(trade_row["exit_index"])
    signal_time = pd.Timestamp(trade_row["event_time"]).tz_convert("UTC")
    signal_matches = exec_frame.index[exec_frame["close_time"] == signal_time]
    signal_index = int(signal_matches[0]) if len(signal_matches) else entry_index

    start_idx = max(0, min(signal_index, entry_index) - pre_bars)
    end_idx = min(len(exec_frame) - 1, max(exit_index, entry_index) + post_bars)
    local = exec_frame.iloc[start_idx : end_idx + 1].copy().reset_index()
    local_indices = local["index"].astype(int).tolist()
    n = len(local)
    if n <= 1:
        return

    raw_price_candidates = local["high"].astype(float).tolist() + local["low"].astype(float).tolist()
    for key in ["zone_top", "zone_bottom", "entry_price", "stop_price", "target_price"]:
        value = safe_float(trade_row.get(key))
        if value is not None:
            raw_price_candidates.append(value)
    raw_low = min(raw_price_candidates)
    raw_high = max(raw_price_candidates)
    raw_range = max(1.0, raw_high - raw_low)
    decision_atr = safe_float(trade_row.get(f"atr_tf_{decision_tf}")) or safe_float(trade_row.get("ATR_1h")) or (raw_range / 6.0)
    decision_close = safe_float(trade_row.get("signal_price")) or safe_float(trade_row.get("entry_price")) or ((raw_low + raw_high) / 2.0)
    relevance_radius = max(decision_atr * 6.0, raw_range * 0.4)
    channel_band_low = decision_close - relevance_radius
    channel_band_high = decision_close + relevance_radius

    channel_values: list[float] = []
    projected: dict[tuple[str, str, str], list[float | None]] = {}
    matched_boundary_value = safe_float(trade_row.get("matched_boundary_value"))
    matched_tf_flags = {tf: bool(safe_float(trade_row.get(f"matched_boundary_tf_{tf}")) or 0.0 >= 0.5) for tf in ["15m", "1h", "4h"]}
    boundary_proximity: dict[tuple[str, str, str], float] = {}
    for timeframe in ["15m", "1h", "4h"]:
        if timeframe not in CHANNEL_STYLES:
            continue
        for family in ["body", "wick"]:
            if not family_is_valid(trade_row, timeframe, family):
                for side in ["upper", "lower"]:
                    projected[(timeframe, family, side)] = [None] * len(local_indices)
                continue
            for side in ["upper", "lower"]:
                arr = project_channel_line(trade_row, timeframe, family, side, local_indices, signal_index, decision_tf)
                finite = [value for value in arr if value is not None]
                if matched_tf_flags.get(timeframe) and matched_boundary_value is not None:
                    boundary_col = f"{side}_{family}_boundary_{timeframe}"
                    boundary_value = safe_float(trade_row.get(boundary_col))
                    proximity = abs(boundary_value - matched_boundary_value) if boundary_value is not None else float("inf")
                    boundary_proximity[(timeframe, family, side)] = proximity
                else:
                    boundary_proximity[(timeframe, family, side)] = min(
                        [abs(value - decision_close) for value in finite] or [float("inf")]
                    )
                visible = [value for value in finite if channel_band_low <= value <= channel_band_high]
                if len(visible) >= 2:
                    filtered = [value if value is not None and channel_band_low <= value <= channel_band_high else None for value in arr]
                    projected[(timeframe, family, side)] = filtered
                    channel_values.extend(visible)
                else:
                    projected[(timeframe, family, side)] = [None] * len(arr)

    highs = local["high"].astype(float).tolist()
    lows = local["low"].astype(float).tolist()
    price_candidates = highs + lows + channel_values
    for key in ["zone_top", "zone_bottom", "entry_price", "stop_price", "target_price"]:
        value = safe_float(trade_row.get(key))
        if value is not None:
            price_candidates.append(value)
    price_low = min(price_candidates)
    price_high = max(price_candidates)
    if price_high <= price_low:
        price_high = price_low + 1.0
    pad = (price_high - price_low) * 0.08
    price_low -= pad
    price_high += pad

    def y_of(price: float) -> float:
        pct = (price_high - price) / (price_high - price_low)
        return plot_top + pct * plot_h

    def x_of(local_pos: int) -> float:
        if n == 1:
            return plot_left + plot_w / 2
        return plot_left + (local_pos / (n - 1)) * plot_w

    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline=(220, 225, 230, 255), fill=(255, 255, 255, 255))
    for frac in [0.25, 0.5, 0.75]:
        y = plot_top + frac * plot_h
        draw.line((plot_left, y, plot_right, y), fill=(235, 238, 241, 255), width=1)

    zone_top = safe_float(trade_row.get("zone_top"))
    zone_bottom = safe_float(trade_row.get("zone_bottom"))
    zone_tf = str(trade_row.get("zone_tf", ""))
    direction = str(trade_row.get("direction", "")).lower()
    if zone_top is not None and zone_bottom is not None:
        zone_fill = rgba("#D8F3DC") if direction == "long" else rgba("#FDE2E4")
        zone_outline = rgba(CHANNEL_STYLES.get(zone_tf, CHANNEL_STYLES["1h"])["line"]) if zone_tf in CHANNEL_STYLES else rgba("#7A7A7A")
        draw.rectangle((plot_left, y_of(zone_top), plot_right, y_of(zone_bottom)), fill=zone_fill, outline=zone_outline, width=2)

    for timeframe in ["4h", "1h", "15m"]:
        style = CHANNEL_STYLES[timeframe]
        upper_body = projected.get((timeframe, "body", "upper"), [])
        lower_body = projected.get((timeframe, "body", "lower"), [])
        fill_ok = (
            upper_body
            and lower_body
            and any(value is not None for value in upper_body + lower_body)
            and boundary_proximity.get((timeframe, "body", "upper"), float("inf")) <= relevance_radius
            and boundary_proximity.get((timeframe, "body", "lower"), float("inf")) <= relevance_radius
        )
        if fill_ok:
            polygon: list[tuple[float, float]] = []
            for i, value in enumerate(upper_body):
                if value is not None:
                    polygon.append((x_of(i), y_of(value)))
            for i in reversed(range(len(lower_body))):
                value = lower_body[i]
                if value is not None:
                    polygon.append((x_of(i), y_of(value)))
            if len(polygon) >= 3:
                draw.polygon(polygon, fill=rgba(style["fill"]))
        for family, width in [("wick", 1), ("body", 2)]:
            for side in ["upper", "lower"]:
                arr = projected.get((timeframe, family, side), [])
                if boundary_proximity.get((timeframe, family, side), float("inf")) > relevance_radius:
                    continue
                pts = [(x_of(i), y_of(value)) for i, value in enumerate(arr) if value is not None]
                if not pts:
                    continue
                line_color = rgba(style["wick"] if family == "wick" else style["line"])
                line_width = width
                if matched_tf_flags.get(timeframe) and matched_boundary_value is not None:
                    if boundary_proximity.get((timeframe, family, side), float("inf")) < max(decision_atr * 1.25, raw_range * 0.08):
                        line_width = max(line_width, 3)
                draw_polyline(draw, pts, fill=line_color, width=line_width)

    candle_step = plot_w / max(1, n - 1)
    candle_body_w = max(2, int(candle_step * 0.55))
    for i, row in local.iterrows():
        x = x_of(i)
        open_price = float(row["open"])
        high_price = float(row["high"])
        low_price = float(row["low"])
        close_price = float(row["close"])
        color = (25, 135, 84, 255) if close_price >= open_price else (200, 56, 56, 255)
        draw.line((x, y_of(high_price), x, y_of(low_price)), fill=color, width=1)
        y_open = y_of(open_price)
        y_close = y_of(close_price)
        top = min(y_open, y_close)
        bottom = max(y_open, y_close)
        if bottom - top < 1.5:
            bottom = top + 1.5
        draw.rectangle((x - candle_body_w / 2, top, x + candle_body_w / 2, bottom), fill=color, outline=color)

    entry_local = max(0, min(n - 1, entry_index - start_idx))
    exit_local = max(0, min(n - 1, exit_index - start_idx))
    x_entry = x_of(entry_local)
    x_exit = x_of(exit_local)
    draw.line((x_entry, plot_top, x_entry, plot_bottom), fill=(35, 35, 35, 160), width=2)
    draw.line((x_exit, plot_top, x_exit, plot_bottom), fill=border, width=2)

    label_specs: list[dict[str, Any]] = []
    for name, key, color in [
        ("E", "entry_price", (20, 20, 20, 255)),
        ("TP", "target_price", (25, 135, 84, 255)),
        ("SL", "stop_price", (171, 28, 51, 255)),
    ]:
        value = safe_float(trade_row.get(key))
        if value is None:
            continue
        y = y_of(value)
        draw.line((plot_left, y, plot_right, y), fill=color, width=2 if name == "E" else 1)
        label_specs.append({"name": name, "price": value, "y": y, "color": color})

    for item in adjust_label_positions(label_specs, plot_top + 4, plot_bottom - 12):
        text = f"{item['name']} {item['price']:.0f}"
        draw.text((plot_right + 4, item["adj_y"] - 6), text, fill=item["color"], font=fonts["tiny"])

    trade_number = int(trade_row.get("_sheet_index", 0))
    title = f"#{trade_number:03d} {signal_time:%Y-%m-%d} {'L' if direction == 'long' else 'S'} {zone_tf} {exit_reason} {net_r:+.2f}R"
    subtitle = f"{pd.Timestamp(trade_row['entry_time']).tz_convert('UTC'):%H:%M} -> {pd.Timestamp(trade_row['exit_time']).tz_convert('UTC'):%H:%M} UTC"
    draw.text((12, 8), title, fill=(28, 28, 30, 255), font=fonts["title"])
    draw.text((12, 21), subtitle, fill=(90, 95, 100, 255), font=fonts["tiny"])

    low_txt = f"{price_low:.0f}"
    high_txt = f"{price_high:.0f}"
    draw.text((plot_right + 4, plot_bottom - 10), low_txt, fill=(110, 110, 110, 255), font=fonts["tiny"])
    draw.text((plot_right + 4, plot_top), high_txt, fill=(110, 110, 110, 255), font=fonts["tiny"])

    sheet.alpha_composite(panel, top_left)


def draw_sheet_header(
    image: Image.Image,
    prefix_name: str,
    trade_count: int,
    config_name: str,
    note: str,
    fonts: dict[str, ImageFont.ImageFont],
) -> int:
    draw = ImageDraw.Draw(image, "RGBA")
    title = f"Zone-Channel Trade Sheet: {prefix_name}"
    subtitle = f"{trade_count} trades | {config_name} | {note}"
    draw.text((18, 12), title, fill=(20, 24, 28, 255), font=fonts["header"])
    draw.text((18, 34), subtitle, fill=(70, 74, 79, 255), font=fonts["title"])

    legend_y = 60
    items = [
        ("15m channel", CHANNEL_STYLES["15m"]["line"]),
        ("1h channel", CHANNEL_STYLES["1h"]["line"]),
        ("4h channel", CHANNEL_STYLES["4h"]["line"]),
        ("Zone", "#2F6F91"),
        ("Entry", "#141414"),
        ("TP", "#198754"),
        ("SL", "#AB1C33"),
        ("Exit bar", "#B22222"),
    ]
    x = 18
    for label, color in items:
        draw.rectangle((x, legend_y, x + 10, legend_y + 10), fill=rgba(color), outline=rgba(color))
        draw.text((x + 16, legend_y - 3), label, fill=(55, 60, 65, 255), font=fonts["tiny"])
        x += 16 + draw.textlength(label, font=fonts["tiny"]) + 26
    return 82


def main() -> None:
    args = parse_args()
    prefix = args.prefix
    signals_prefix = args.signals_prefix or prefix

    trades_path = prefix.with_name(prefix.name + "_trades.csv")
    signals_path = signals_prefix.with_name(signals_prefix.name + "_signals.csv")
    summary_path = prefix.with_name(prefix.name + "_summary.json")
    config_path = signals_prefix.with_name(signals_prefix.name + "_config.json")

    trades = pd.read_csv(trades_path)
    signals = pd.read_csv(signals_path)
    summary = pd.read_json(summary_path, typ="series")
    config = pd.read_json(config_path, typ="series")

    if args.limit > 0:
        trades = trades.head(args.limit).copy()

    merged = trades.merge(
        signals,
        on=["event_key", "symbol", "direction"],
        how="left",
        suffixes=("", "_signal"),
    ).sort_values("entry_time").reset_index(drop=True)
    merged["_sheet_index"] = range(1, len(merged) + 1)

    decision_tf = str(config.get("decision_timeframe", "15m"))
    base_interval = str(config.get("base_interval", "5m"))
    symbol = str(config.get("symbol", "BTCUSDT"))
    start = pd.to_datetime(merged["event_time"], utc=True, errors="coerce").min().floor("D")
    end = pd.to_datetime(merged["exit_time"], utc=True, errors="coerce").max().ceil("D")

    base = load_base_candles(symbol, start, end, cache_dir=Path("scripts/.cache"), interval=base_interval)
    exec_frame = prepare_timeframe_bars(base, decision_tf, atr_length=int(config.get("atr_length", 14)))

    columns = max(1, args.columns)
    rows = math.ceil(len(merged) / columns)
    header_h = 92
    sheet_w = 20 + columns * args.panel_width + (columns - 1) * 12 + 20
    sheet_h = header_h + rows * args.panel_height + max(0, rows - 1) * 12 + 20
    image = Image.new("RGBA", (sheet_w, sheet_h), (245, 247, 250, 255))

    fonts = {
        "header": load_font(20, bold=True),
        "title": load_font(12, bold=True),
        "tiny": load_font(10, bold=False),
    }
    note = "loss-streak disabled for full-flow inspection" if "nostreak" in prefix.name.lower() else "as-configured replay"
    draw_sheet_header(image, prefix.name, len(merged), str(config.get("name", prefix.name)), note, fonts)

    for idx, (_, trade_row) in enumerate(merged.iterrows()):
        row = idx // columns
        col = idx % columns
        x = 20 + col * (args.panel_width + 12)
        y = header_h + row * (args.panel_height + 12)
        render_panel(
            image,
            (x, y),
            trade_row,
            exec_frame,
            decision_tf,
            fonts,
            pre_bars=args.pre_bars,
            post_bars=args.post_bars,
            panel_width=args.panel_width,
            panel_height=args.panel_height,
        )

    output = args.output or prefix.with_name(prefix.name + "_trade_sheet.png")
    image.convert("RGB").save(output, format="PNG", optimize=True)
    print(f"Saved trade sheet to {output}")


if __name__ == "__main__":
    main()
