from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from scripts.channel_state_research.data import load_base_candles, prepare_timeframe_bars
from scripts.render_zone_channel_trade_sheet import draw_sheet_header, load_font, render_panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a one-page-per-trade PDF for zone-channel runs.")
    parser.add_argument("--prefix", type=Path, required=True, help="Trade/decision prefix to load.")
    parser.add_argument(
        "--signals-prefix",
        type=Path,
        default=None,
        help="Optional alternate prefix for *_signals.csv and *_config.json.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0, help="Optional trade limit for testing; 0 means all.")
    parser.add_argument("--pre-bars", type=int, default=20)
    parser.add_argument("--post-bars", type=int, default=12)
    parser.add_argument("--image-width", type=int, default=1900)
    parser.add_argument("--image-height", type=int, default=1120)
    parser.add_argument("--jpeg-quality", type=int, default=88)
    return parser.parse_args()


def build_trade_frame(prefix: Path, signals_prefix: Path, limit: int) -> tuple[pd.DataFrame, pd.Series]:
    trades_path = prefix.with_name(prefix.name + "_trades.csv")
    signals_path = signals_prefix.with_name(signals_prefix.name + "_signals.csv")
    config_path = signals_prefix.with_name(signals_prefix.name + "_config.json")

    trades = pd.read_csv(trades_path)
    signals = pd.read_csv(signals_path)
    config = pd.read_json(config_path, typ="series")
    if limit > 0:
        trades = trades.head(limit).copy()
    merged = trades.merge(
        signals,
        on=["event_key", "symbol", "direction"],
        how="left",
        suffixes=("", "_signal"),
    ).sort_values("entry_time").reset_index(drop=True)
    merged["_sheet_index"] = range(1, len(merged) + 1)
    return merged, config


def render_page_image(
    trade_row: pd.Series,
    exec_frame: pd.DataFrame,
    config: pd.Series,
    prefix_name: str,
    trade_count: int,
    image_width: int,
    image_height: int,
    pre_bars: int,
    post_bars: int,
) -> Image.Image:
    image = Image.new("RGBA", (image_width, image_height), (245, 247, 250, 255))
    fonts = {
        "header": load_font(30, bold=True),
        "title": load_font(18, bold=True),
        "tiny": load_font(14, bold=False),
    }
    note = f"trade {int(trade_row['_sheet_index'])}/{trade_count} | {config.get('name', prefix_name)}"
    header_h = draw_sheet_header(image, prefix_name, trade_count, str(config.get("name", prefix_name)), note, fonts)

    panel_margin_x = 36
    panel_margin_bottom = 28
    panel_y = header_h + 14
    panel_w = image_width - panel_margin_x * 2
    panel_h = image_height - panel_y - panel_margin_bottom

    render_panel(
        image,
        (panel_margin_x, panel_y),
        trade_row,
        exec_frame,
        str(config.get("decision_timeframe", "15m")),
        fonts,
        pre_bars=pre_bars,
        post_bars=post_bars,
        panel_width=panel_w,
        panel_height=panel_h,
    )

    draw = ImageDraw.Draw(image, "RGBA")
    footer = (
        f"event {pd.Timestamp(trade_row['event_time']).tz_convert('UTC'):%Y-%m-%d %H:%M} UTC"
        f" | entry {safe_round(trade_row.get('entry_price'))}"
        f" | stop {safe_round(trade_row.get('stop_price'))}"
        f" | target {safe_round(trade_row.get('target_price'))}"
        f" | planned RR {safe_float_text(trade_row.get('target_rr_planned'))}"
    )
    draw.text((40, image_height - 22), footer, fill=(80, 84, 88, 255), font=fonts["tiny"])
    return image


def safe_round(value: object) -> str:
    try:
        return f"{float(value):.0f}"
    except Exception:
        return "n/a"


def safe_float_text(value: object) -> str:
    try:
        return f"{float(value):.2f}"
    except Exception:
        return "n/a"


def main() -> None:
    args = parse_args()
    prefix = args.prefix
    signals_prefix = args.signals_prefix or prefix
    merged, config = build_trade_frame(prefix, signals_prefix, args.limit)

    decision_tf = str(config.get("decision_timeframe", "15m"))
    base_interval = str(config.get("base_interval", "5m"))
    symbol = str(config.get("symbol", "BTCUSDT"))
    start = pd.to_datetime(merged["event_time"], utc=True, errors="coerce").min().floor("D")
    end = pd.to_datetime(merged["exit_time"], utc=True, errors="coerce").max().ceil("D")

    base = load_base_candles(symbol, start, end, cache_dir=Path("scripts/.cache"), interval=base_interval)
    exec_frame = prepare_timeframe_bars(base, decision_tf, atr_length=int(config.get("atr_length", 14)))

    output = args.output or prefix.with_name(prefix.name + "_trade_book.pdf")
    page_size = landscape(A4)
    pdf = canvas.Canvas(str(output), pagesize=page_size)

    trade_count = len(merged)
    for _, trade_row in merged.iterrows():
        page_image = render_page_image(
            trade_row,
            exec_frame,
            config,
            prefix.name,
            trade_count,
            args.image_width,
            args.image_height,
            args.pre_bars,
            args.post_bars,
        )
        rgb = page_image.convert("RGB")
        buffer = io.BytesIO()
        rgb.save(buffer, format="JPEG", quality=args.jpeg_quality, optimize=True)
        buffer.seek(0)
        reader = ImageReader(buffer)
        pdf.drawImage(reader, 0, 0, width=page_size[0], height=page_size[1], preserveAspectRatio=False, mask="auto")
        pdf.showPage()
        buffer.close()

    pdf.save()
    print(f"Saved trade book PDF to {output}")


if __name__ == "__main__":
    main()
