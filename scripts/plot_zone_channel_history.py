from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from PIL import Image, ImageColor, ImageDraw, ImageFont
except ImportError:  # HTML output does not need Pillow.
    Image = None
    ImageColor = None
    ImageDraw = None
    ImageFont = None

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.channel_state_research.channels import (
    BoundaryPoint,
    LineFit,
    build_body_envelope_points,
    fit_boundary_line,
    pivot_points,
)
from scripts.channel_state_research.data import load_base_candles, prepare_timeframe_bars
from scripts.channel_state_research.production import ZoneChannelProductionConfig, load_production_config
from scripts.channel_state_research.swings import extract_causal_swings


TIMEFRAME_MINUTES = {
    "5m": 5,
    "15m": 15,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
    "1w": 10080,
}


@dataclass(frozen=True)
class ChannelSegment:
    segment_id: int
    timeframe: str
    family: str
    start_index: int
    end_index: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    upper: LineFit
    lower: LineFit
    upper_points: tuple[BoundaryPoint, ...]
    lower_points: tuple[BoundaryPoint, ...]

    def upper_value(self, index: float) -> float:
        return self.upper.evaluate(index)

    def lower_value(self, index: float) -> float:
        return self.lower.evaluate(index)

    def width_value(self, index: float) -> float:
        return self.upper_value(index) - self.lower_value(index)


@dataclass(frozen=True)
class TradeContext:
    row: pd.Series
    trade_number: int
    timeframe: str | None
    family: str | None
    side: str | None


@dataclass(frozen=True)
class BfmPivot:
    set_number: int
    side: str
    pivot_index: int
    confirm_index: int
    pivot_time: pd.Timestamp
    confirm_time: pd.Timestamp
    price: float
    leftbars: int
    rightbars: int


@dataclass(frozen=True)
class BfmTrendline:
    set_number: int
    side: str
    start_pivot: BfmPivot
    end_pivot: BfmPivot
    line_start_index: int
    line_end_index: int
    invalidated: bool
    invalidation_index: int | None
    slope: float
    intercept: float

    def evaluate(self, index: float) -> float:
        return float(self.slope * index + self.intercept)


@dataclass(frozen=True)
class BfmTimeframeResult:
    timeframe: str
    bars: pd.DataFrame
    lines: list[BfmTrendline]
    pivots: list[BfmPivot]
    pivot_sets: tuple[tuple[int, int], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render full-window piecewise zone-channel boundaries for one timeframe/family."
    )
    parser.add_argument("--config", type=Path, default=Path("scripts/channel_15m_broad_v2_full5y_config.json"))
    parser.add_argument("--start", default="2021-04-30")
    parser.add_argument("--end", default="2026-04-30")
    parser.add_argument("--cache-dir", type=Path, default=Path("scripts/.cache"))
    parser.add_argument("--timeframe", default="1h", help="One of 15m/1h/4h, or auto when --trade-number is used.")
    parser.add_argument("--family", choices=["wick", "body", "auto"], default="wick")
    parser.add_argument(
        "--logic",
        choices=["bfm", "channel"],
        default="bfm",
        help="bfm mimics the supplied Pine pivot trendlines; channel keeps the prior channel-regression view.",
    )
    parser.add_argument(
        "--bfm-sets",
        default="300:200,240:160,192:128,154:102",
        help="Comma-separated left:right pivot sets for --logic bfm.",
    )
    parser.add_argument(
        "--bfm-tf-sets",
        default=None,
        help=(
            "Optional per-timeframe BFM sets, e.g. "
            "'1h=330:220,264:176;4h=180:120,144:96;1d=105:70,84:56'. "
            "Timeframes not listed use --bfm-sets."
        ),
    )
    parser.add_argument(
        "--bfm-invalidation",
        choices=["wick", "close", "none"],
        default="wick",
        help="How BFM lines are extended after the second pivot. Pine itself does not invalidate old line objects.",
    )
    parser.add_argument(
        "--bfm-max-extension-bars",
        type=int,
        default=300,
        help="Maximum bars to extend a BFM trendline after the latest defining pivot. Use 0 to disable the cap.",
    )
    parser.add_argument(
        "--bfm-timeframes",
        default=None,
        help="Comma-separated timeframes to overlay for --logic bfm, e.g. 1h,4h,1d. Defaults to --timeframe only.",
    )
    parser.add_argument("--trades-csv", type=Path, default=None)
    parser.add_argument("--signals-csv", type=Path, default=None)
    parser.add_argument(
        "--trade-number",
        type=int,
        default=0,
        help="1-based trade-sheet number after sorting trades by entry_time. Used for auto timeframe/family and markers.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--segments-output", type=Path, default=None)
    parser.add_argument(
        "--format",
        choices=["auto", "png", "html"],
        default="auto",
        help="auto uses the output suffix; .html writes an interactive Plotly HTML file.",
    )
    parser.add_argument(
        "--plotly-js",
        choices=["cdn"],
        default="cdn",
        help="How to load plotly.js for HTML output. cdn keeps the file small.",
    )
    parser.add_argument(
        "--line-mode",
        choices=["anchors", "fit"],
        default="anchors",
        help="anchors draws contact lines through selected boundary points; fit draws the original fitted regressions.",
    )
    parser.add_argument("--width", type=int, default=2200)
    parser.add_argument("--height", type=int, default=1280)
    parser.add_argument(
        "--scale-mode",
        choices=["channel", "price"],
        default="channel",
        help="channel includes segment endpoints in the y-scale; price keeps the chart scaled to candle highs/lows.",
    )
    return parser.parse_args()


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if ImageFont is None:
        raise RuntimeError("Pillow is required for PNG output. Use --format html or install Pillow.")
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
    if ImageColor is None:
        raise RuntimeError("Pillow is required for PNG output. Use --format html or install Pillow.")
    return ImageColor.getcolor(color, "RGBA")


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number):
        return None
    return number


def config_reversal(config: ZoneChannelProductionConfig, timeframe: str) -> float:
    values = {
        "5m": config.reversal_5m,
        "15m": config.reversal_15m,
        "1h": config.reversal_1h,
        "4h": config.reversal_4h,
        "1d": config.reversal_1d,
        "1w": config.reversal_1w,
    }
    if timeframe not in values:
        raise KeyError(f"Unsupported timeframe {timeframe!r}")
    return float(values[timeframe])


def load_trade_context(args: argparse.Namespace, config: ZoneChannelProductionConfig) -> TradeContext | None:
    if args.trade_number <= 0:
        return None
    if args.trades_csv is None or args.signals_csv is None:
        raise ValueError("--trade-number requires --trades-csv and --signals-csv.")
    trades = pd.read_csv(args.trades_csv)
    signals = pd.read_csv(args.signals_csv)
    merged = (
        trades.merge(signals, on=["event_key", "symbol", "direction"], how="left", suffixes=("", "_signal"))
        .sort_values("entry_time")
        .reset_index(drop=True)
    )
    if args.trade_number > len(merged):
        raise IndexError(f"Trade #{args.trade_number} requested, but only {len(merged)} trades were found.")
    row = merged.iloc[args.trade_number - 1]
    matched_timeframe = None
    for timeframe in config.timeframes:
        flag = safe_float(row.get(f"matched_boundary_tf_{timeframe}"))
        if flag is not None and flag >= 0.5:
            matched_timeframe = timeframe
            break
    matched_family = None
    if (safe_float(row.get("matched_boundary_is_wick")) or 0.0) >= 0.5:
        matched_family = "wick"
    elif (safe_float(row.get("matched_boundary_is_body")) or 0.0) >= 0.5:
        matched_family = "body"
    direction = str(row.get("direction", "")).lower()
    matched_side = "lower" if direction == "long" else "upper" if direction == "short" else None
    return TradeContext(
        row=row,
        trade_number=int(args.trade_number),
        timeframe=matched_timeframe,
        family=matched_family,
        side=matched_side,
    )


def resolve_timeframe_and_family(
    args: argparse.Namespace,
    config: ZoneChannelProductionConfig,
    trade_context: TradeContext | None,
) -> tuple[str, str]:
    timeframe = str(args.timeframe)
    if timeframe == "auto":
        if trade_context is None or trade_context.timeframe is None:
            raise ValueError("--timeframe auto requires a trade with a matched boundary timeframe.")
        timeframe = trade_context.timeframe
    if timeframe not in TIMEFRAME_MINUTES:
        raise KeyError(f"Unsupported timeframe {timeframe!r}. Supported: {', '.join(TIMEFRAME_MINUTES)}")
    if timeframe not in config.timeframes:
        raise ValueError(f"Timeframe {timeframe!r} is not in config.timeframes: {config.timeframes!r}")

    family = str(args.family)
    if family == "auto":
        if trade_context is None or trade_context.family is None:
            raise ValueError("--family auto requires a trade with a matched boundary family.")
        family = trade_context.family
    return timeframe, family


def parse_bfm_sets(raw: str) -> list[tuple[int, int]]:
    sets: list[tuple[int, int]] = []
    for chunk in raw.split(","):
        text = chunk.strip()
        if not text:
            continue
        if ":" not in text:
            raise ValueError(f"Invalid BFM set {text!r}; expected left:right.")
        left_raw, right_raw = text.split(":", 1)
        left = int(left_raw)
        right = int(right_raw)
        if left <= 0 or right <= 0:
            raise ValueError(f"Invalid BFM set {text!r}; left/right must be positive.")
        sets.append((left, right))
    if not sets:
        raise ValueError("At least one BFM pivot set is required.")
    return sets


def parse_timeframes(raw: str | None, default_timeframe: str) -> list[str]:
    if raw is None or not str(raw).strip():
        return [default_timeframe]
    out: list[str] = []
    for chunk in str(raw).split(","):
        timeframe = chunk.strip()
        if not timeframe:
            continue
        if timeframe not in TIMEFRAME_MINUTES:
            raise KeyError(f"Unsupported timeframe {timeframe!r}. Supported: {', '.join(TIMEFRAME_MINUTES)}")
        if timeframe not in out:
            out.append(timeframe)
    return out or [default_timeframe]


def parse_bfm_sets_by_timeframe(
    raw: str | None,
    timeframes: list[str],
    default_sets: list[tuple[int, int]],
) -> dict[str, list[tuple[int, int]]]:
    out = {timeframe: list(default_sets) for timeframe in timeframes}
    if raw is None or not str(raw).strip():
        return out
    for chunk in str(raw).split(";"):
        text = chunk.strip()
        if not text:
            continue
        if "=" not in text:
            raise ValueError(f"Invalid --bfm-tf-sets chunk {text!r}; expected timeframe=left:right,...")
        timeframe, raw_sets = text.split("=", 1)
        timeframe = timeframe.strip()
        if timeframe not in out:
            raise ValueError(f"--bfm-tf-sets references {timeframe!r}, not in --bfm-timeframes.")
        out[timeframe] = parse_bfm_sets(raw_sets)
    return out


def bfm_set_text(pivot_sets: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> str:
    return ", ".join(f"S{idx}:{left}/{right}" for idx, (left, right) in enumerate(pivot_sets, start=1))


def bfm_result_set_text(results: list[BfmTimeframeResult]) -> str:
    unique_sets = {tuple(result.pivot_sets) for result in results}
    if len(unique_sets) == 1:
        return bfm_set_text(results[0].pivot_sets)
    return "; ".join(f"{result.timeframe} [{bfm_set_text(result.pivot_sets)}]" for result in results)


def detect_bfm_pivots(
    bars: pd.DataFrame,
    *,
    set_number: int,
    leftbars: int,
    rightbars: int,
) -> tuple[list[BfmPivot], list[BfmPivot]]:
    highs = pd.to_numeric(bars["high"], errors="coerce").to_list()
    lows = pd.to_numeric(bars["low"], errors="coerce").to_list()
    times = [pd.Timestamp(value).tz_convert("UTC") for value in pd.to_datetime(bars["close_time"], utc=True, errors="coerce")]
    high_pivots: list[BfmPivot] = []
    low_pivots: list[BfmPivot] = []
    for pivot_index in range(leftbars, len(bars) - rightbars):
        confirm_index = pivot_index + rightbars
        high_value = safe_float(highs[pivot_index])
        low_value = safe_float(lows[pivot_index])
        if high_value is None or low_value is None:
            continue
        left = pivot_index - leftbars
        right = pivot_index + rightbars + 1
        high_window = [safe_float(value) for value in highs[left:right]]
        low_window = [safe_float(value) for value in lows[left:right]]
        high_finite = [value for value in high_window if value is not None]
        low_finite = [value for value in low_window if value is not None]
        if high_finite and high_value >= max(high_finite):
            high_pivots.append(
                BfmPivot(
                    set_number=set_number,
                    side="resistance",
                    pivot_index=pivot_index,
                    confirm_index=confirm_index,
                    pivot_time=times[pivot_index],
                    confirm_time=times[confirm_index],
                    price=high_value,
                    leftbars=leftbars,
                    rightbars=rightbars,
                )
            )
        if low_finite and low_value <= min(low_finite):
            low_pivots.append(
                BfmPivot(
                    set_number=set_number,
                    side="support",
                    pivot_index=pivot_index,
                    confirm_index=confirm_index,
                    pivot_time=times[pivot_index],
                    confirm_time=times[confirm_index],
                    price=low_value,
                    leftbars=leftbars,
                    rightbars=rightbars,
                )
            )
    return high_pivots, low_pivots


def build_bfm_magic_lines(
    bars: pd.DataFrame,
    pivot_sets: list[tuple[int, int]],
    *,
    invalidation: str,
    max_extension_bars: int,
) -> tuple[list[BfmTrendline], list[BfmPivot]]:
    all_lines: list[BfmTrendline] = []
    all_pivots: list[BfmPivot] = []
    for set_index, (leftbars, rightbars) in enumerate(pivot_sets, start=1):
        high_pivots, low_pivots = detect_bfm_pivots(
            bars,
            set_number=set_index,
            leftbars=leftbars,
            rightbars=rightbars,
        )
        all_pivots.extend(high_pivots)
        all_pivots.extend(low_pivots)
        all_lines.extend(
            _bfm_lines_for_side(
                bars,
                high_pivots,
                side="resistance",
                invalidation=invalidation,
                max_extension_bars=max_extension_bars,
            )
        )
        all_lines.extend(
            _bfm_lines_for_side(
                bars,
                low_pivots,
                side="support",
                invalidation=invalidation,
                max_extension_bars=max_extension_bars,
            )
        )
    all_lines.sort(key=lambda line: (line.set_number, line.side, line.end_pivot.confirm_index))
    all_pivots.sort(key=lambda pivot: (pivot.pivot_index, pivot.set_number, pivot.side))
    return all_lines, all_pivots


def _bfm_lines_for_side(
    bars: pd.DataFrame,
    pivots: list[BfmPivot],
    *,
    side: str,
    invalidation: str,
    max_extension_bars: int,
) -> list[BfmTrendline]:
    lines: list[BfmTrendline] = []
    if len(pivots) < 2:
        return lines
    highs = pd.to_numeric(bars["high"], errors="coerce").to_list()
    lows = pd.to_numeric(bars["low"], errors="coerce").to_list()
    closes = pd.to_numeric(bars["close"], errors="coerce").to_list()
    for index in range(1, len(pivots)):
        previous = pivots[index - 1]
        current = pivots[index]
        dx = current.pivot_index - previous.pivot_index
        if dx == 0:
            continue
        slope = (current.price - previous.price) / dx
        intercept = current.price - slope * current.pivot_index
        invalidation_index: int | None = None
        cap_index = len(bars) - 1
        if max_extension_bars > 0:
            cap_index = min(cap_index, current.pivot_index + int(max_extension_bars))
        if invalidation != "none":
            for cursor in range(current.confirm_index + 1, cap_index + 1):
                line_value = slope * cursor + intercept
                if side == "resistance":
                    probe = closes[cursor] if invalidation == "close" else highs[cursor]
                    if safe_float(probe) is not None and float(probe) > line_value:
                        invalidation_index = cursor
                        break
                else:
                    probe = closes[cursor] if invalidation == "close" else lows[cursor]
                    if safe_float(probe) is not None and float(probe) < line_value:
                        invalidation_index = cursor
                        break
        line_end_index = invalidation_index if invalidation_index is not None else cap_index
        line_end_index = max(current.pivot_index, line_end_index)
        lines.append(
            BfmTrendline(
                set_number=current.set_number,
                side=side,
                start_pivot=previous,
                end_pivot=current,
                line_start_index=previous.pivot_index,
                line_end_index=int(line_end_index),
                invalidated=invalidation_index is not None,
                invalidation_index=invalidation_index,
                slope=float(slope),
                intercept=float(intercept),
            )
        )
    return lines


def build_channel_points(
    bars: pd.DataFrame,
    config: ZoneChannelProductionConfig,
    timeframe: str,
    family: str,
) -> tuple[list[BoundaryPoint], list[BoundaryPoint]]:
    if family == "wick":
        pivots = extract_causal_swings(bars, config_reversal(config, timeframe))
        return pivot_points(pivots, "high"), pivot_points(pivots, "low")
    return (
        build_body_envelope_points(
            bars,
            side="upper",
            lookback=config.body_envelope_lookback,
            min_separation=config.body_envelope_min_separation,
            min_move_atr=config.body_envelope_min_move_atr,
        ),
        build_body_envelope_points(
            bars,
            side="lower",
            lookback=config.body_envelope_lookback,
            min_separation=config.body_envelope_min_separation,
            min_move_atr=config.body_envelope_min_move_atr,
        ),
    )


def build_channel_segments(
    bars: pd.DataFrame,
    config: ZoneChannelProductionConfig,
    timeframe: str,
    family: str,
) -> list[ChannelSegment]:
    upper_points, lower_points = build_channel_points(bars, config, timeframe, family)
    events: list[tuple[int, str, BoundaryPoint]] = []
    events.extend((point.confirm_index, "upper", point) for point in upper_points)
    events.extend((point.confirm_index, "lower", point) for point in lower_points)
    events.sort(key=lambda item: (item[0], item[1]))

    active_points: dict[str, list[BoundaryPoint]] = {"upper": [], "lower": []}
    active_recent: dict[str, tuple[BoundaryPoint, ...]] = {"upper": tuple(), "lower": tuple()}
    active_fit: dict[str, LineFit | None] = {"upper": None, "lower": None}
    times = pd.to_datetime(bars["close_time"], utc=True, errors="coerce").to_list()
    bar_indices = pd.to_numeric(bars["bar_index"], errors="coerce").to_list()

    segments: list[ChannelSegment] = []
    event_cursor = 0
    current_start: int | None = None
    current_definition: tuple[Any, ...] | None = None
    current_upper: LineFit | None = None
    current_lower: LineFit | None = None
    current_upper_points: tuple[BoundaryPoint, ...] = tuple()
    current_lower_points: tuple[BoundaryPoint, ...] = tuple()

    def fit_side(side: str) -> None:
        recent = tuple(active_points[side][-config.point_count :])
        active_recent[side] = recent
        if len(recent) < config.min_points:
            active_fit[side] = None
            return
        active_fit[side] = fit_boundary_line(recent, method=config.channel_estimator)

    def make_definition() -> tuple[Any, ...] | None:
        upper = active_fit["upper"]
        lower = active_fit["lower"]
        if upper is None or lower is None:
            return None
        return (
            round(upper.slope, 12),
            round(upper.intercept, 8),
            _point_signature(active_recent["upper"]),
            round(lower.slope, 12),
            round(lower.intercept, 8),
            _point_signature(active_recent["lower"]),
        )

    def is_valid_at(index: int) -> bool:
        upper = active_fit["upper"]
        lower = active_fit["lower"]
        if upper is None or lower is None:
            return False
        x_value = safe_float(bar_indices[index])
        if x_value is None:
            return False
        upper_value = upper.evaluate(x_value)
        lower_value = lower.evaluate(x_value)
        return math.isfinite(upper_value) and math.isfinite(lower_value) and upper_value > lower_value

    def close_current(end_index: int) -> None:
        nonlocal current_start, current_definition, current_upper, current_lower, current_upper_points, current_lower_points
        if (
            current_start is None
            or current_upper is None
            or current_lower is None
            or end_index < current_start
        ):
            current_start = None
            current_definition = None
            current_upper = None
            current_lower = None
            current_upper_points = tuple()
            current_lower_points = tuple()
            return
        segments.append(
            ChannelSegment(
                segment_id=len(segments) + 1,
                timeframe=timeframe,
                family=family,
                start_index=int(current_start),
                end_index=int(end_index),
                start_time=pd.Timestamp(times[current_start]).tz_convert("UTC"),
                end_time=pd.Timestamp(times[end_index]).tz_convert("UTC"),
                upper=current_upper,
                lower=current_lower,
                upper_points=current_upper_points,
                lower_points=current_lower_points,
            )
        )
        current_start = None
        current_definition = None
        current_upper = None
        current_lower = None
        current_upper_points = tuple()
        current_lower_points = tuple()

    for index in range(len(bars)):
        while event_cursor < len(events) and events[event_cursor][0] <= index:
            _, side, point = events[event_cursor]
            active_points[side].append(point)
            fit_side(side)
            event_cursor += 1

        definition = make_definition()
        if definition is None or not is_valid_at(index):
            close_current(index - 1)
            continue

        if current_definition is None:
            current_start = index
            current_definition = definition
            current_upper = active_fit["upper"]
            current_lower = active_fit["lower"]
            current_upper_points = active_recent["upper"]
            current_lower_points = active_recent["lower"]
        elif definition != current_definition:
            close_current(index - 1)
            current_start = index
            current_definition = definition
            current_upper = active_fit["upper"]
            current_lower = active_fit["lower"]
            current_upper_points = active_recent["upper"]
            current_lower_points = active_recent["lower"]

    close_current(len(bars) - 1)
    return segments


def _point_signature(points: tuple[BoundaryPoint, ...]) -> tuple[tuple[int, int, float], ...]:
    return tuple((point.index, point.confirm_index, round(float(point.price), 8)) for point in points)


def display_line_for_segment(
    segment: ChannelSegment,
    side: str,
    line_mode: str,
) -> tuple[LineFit, BoundaryPoint | None, BoundaryPoint | None]:
    if side == "upper":
        fit = segment.upper
        points = segment.upper_points
    elif side == "lower":
        fit = segment.lower
        points = segment.lower_points
    else:
        raise ValueError(f"Unsupported channel side {side!r}.")

    if line_mode == "fit":
        return fit, None, None
    if line_mode != "anchors":
        raise ValueError(f"Unsupported line mode {line_mode!r}.")
    if len(points) < 2:
        return fit, None, None

    start = points[0]
    end = points[-1]
    dx = float(end.index - start.index)
    if abs(dx) <= 1e-12:
        return fit, None, None
    slope = (float(end.price) - float(start.price)) / dx
    intercept = float(start.price) - slope * float(start.index)
    anchor_fit = LineFit(
        method="anchor_endpoints",
        slope=float(slope),
        intercept=float(intercept),
        points_used=2,
        median_abs_residual=0.0,
    )
    return anchor_fit, start, end


def display_segment_end_index(
    segment: ChannelSegment,
    upper_line: LineFit,
    lower_line: LineFit,
) -> float:
    end_index = float(segment.end_index)
    slope_gap = upper_line.slope - lower_line.slope
    intercept_gap = upper_line.intercept - lower_line.intercept
    if abs(slope_gap) <= 1e-12:
        return end_index
    cross_index = -intercept_gap / slope_gap
    start_index = float(segment.start_index)
    if start_index < cross_index < end_index:
        return float(cross_index)
    return end_index


def timestamp_at_index(bars: pd.DataFrame, index_value: float) -> str:
    if index_value <= 0:
        return _iso(pd.Timestamp(bars["close_time"].iloc[0]).tz_convert("UTC"))
    last_index = len(bars) - 1
    if index_value >= last_index:
        return _iso(pd.Timestamp(bars["close_time"].iloc[-1]).tz_convert("UTC"))
    lower = int(math.floor(index_value))
    upper = int(math.ceil(index_value))
    lower_ts = pd.Timestamp(bars["close_time"].iloc[lower]).tz_convert("UTC")
    upper_ts = pd.Timestamp(bars["close_time"].iloc[upper]).tz_convert("UTC")
    if upper == lower:
        return _iso(lower_ts)
    frac = index_value - lower
    interpolated = lower_ts + (upper_ts - lower_ts) * frac
    return _iso(pd.Timestamp(interpolated).tz_convert("UTC"))


def write_segments_csv(path: Path, segments: list[ChannelSegment], bars: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bar_indices = pd.to_numeric(bars["bar_index"], errors="coerce").to_list()
    fields = [
        "segment_id",
        "timeframe",
        "family",
        "start_time",
        "end_time",
        "start_index",
        "end_index",
        "upper_slope",
        "upper_intercept",
        "lower_slope",
        "lower_intercept",
        "upper_start",
        "upper_end",
        "lower_start",
        "lower_end",
        "width_start",
        "width_end",
        "upper_points",
        "lower_points",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for segment in segments:
            x0 = float(bar_indices[segment.start_index])
            x1 = float(bar_indices[segment.end_index])
            upper_start = segment.upper_value(x0)
            upper_end = segment.upper_value(x1)
            lower_start = segment.lower_value(x0)
            lower_end = segment.lower_value(x1)
            writer.writerow(
                {
                    "segment_id": segment.segment_id,
                    "timeframe": segment.timeframe,
                    "family": segment.family,
                    "start_time": segment.start_time.isoformat(),
                    "end_time": segment.end_time.isoformat(),
                    "start_index": segment.start_index,
                    "end_index": segment.end_index,
                    "upper_slope": segment.upper.slope,
                    "upper_intercept": segment.upper.intercept,
                    "lower_slope": segment.lower.slope,
                    "lower_intercept": segment.lower.intercept,
                    "upper_start": upper_start,
                    "upper_end": upper_end,
                    "lower_start": lower_start,
                    "lower_end": lower_end,
                    "width_start": upper_start - lower_start,
                    "width_end": upper_end - lower_end,
                    "upper_points": _format_points(segment.upper_points),
                    "lower_points": _format_points(segment.lower_points),
                }
            )


def _format_points(points: tuple[BoundaryPoint, ...]) -> str:
    return ";".join(
        f"{point.index}@{point.confirm_time.isoformat()}:{point.price:.8f}" for point in points
    )


def render_plot(
    *,
    output: Path,
    bars: pd.DataFrame,
    segments: list[ChannelSegment],
    config: ZoneChannelProductionConfig,
    timeframe: str,
    family: str,
    trade_context: TradeContext | None,
    width: int,
    height: int,
    scale_mode: str,
) -> None:
    if Image is None or ImageDraw is None:
        raise RuntimeError("Pillow is required for PNG output. Use --format html or install Pillow.")
    output.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGBA", (width, height), (247, 249, 252, 255))
    draw = ImageDraw.Draw(image, "RGBA")
    fonts = {
        "title": load_font(30, bold=True),
        "subtitle": load_font(17, bold=False),
        "label": load_font(15, bold=False),
        "label_bold": load_font(15, bold=True),
        "tiny": load_font(12, bold=False),
    }

    times = pd.to_datetime(bars["close_time"], utc=True, errors="coerce").to_list()
    timestamps = [pd.Timestamp(value).timestamp() for value in times]
    t_min = min(timestamps)
    t_max = max(timestamps)
    bar_indices = pd.to_numeric(bars["bar_index"], errors="coerce").to_list()

    plot_left = 92
    plot_right = width - 72
    plot_top = 112
    metric_h = 170
    gap = 32
    plot_bottom = height - 96 - metric_h - gap
    metric_top = plot_bottom + gap
    metric_bottom = height - 82
    plot_w = plot_right - plot_left
    plot_h = plot_bottom - plot_top

    price_low, price_high = price_scale(bars, segments, bar_indices, scale_mode)

    def x_of_timestamp(timestamp: pd.Timestamp | str) -> float:
        ts = pd.Timestamp(timestamp).tz_convert("UTC").timestamp()
        return plot_left + ((ts - t_min) / max(1.0, t_max - t_min)) * plot_w

    def x_of_seconds(seconds: float) -> float:
        return plot_left + ((seconds - t_min) / max(1.0, t_max - t_min)) * plot_w

    def y_of_price(price: float) -> float:
        pct = (price_high - price) / max(1e-9, price_high - price_low)
        y_value = plot_top + pct * plot_h
        return max(plot_top - plot_h, min(plot_bottom + plot_h, y_value))

    draw.text((30, 24), f"{config.symbol} {timeframe} {family} channel history", fill=(22, 26, 32, 255), font=fonts["title"])
    subtitle_parts = [
        f"{pd.Timestamp(times[0]).date()} to {pd.Timestamp(times[-1]).date()}",
        f"{len(bars):,} bars",
        f"{len(segments):,} channel definitions",
        f"estimator {config.channel_estimator}",
    ]
    if trade_context is not None:
        row = trade_context.row
        subtitle_parts.append(
            f"trade #{trade_context.trade_number:03d} {str(row.get('direction')).upper()} matched {trade_context.timeframe}/{trade_context.family}"
        )
    draw.text((32, 63), " | ".join(subtitle_parts), fill=(84, 91, 101, 255), font=fonts["subtitle"])

    draw_plot_frame(draw, plot_left, plot_top, plot_right, plot_bottom)
    draw_price_grid(
        draw,
        price_low,
        price_high,
        plot_left,
        plot_top,
        plot_right,
        plot_bottom,
        fonts["tiny"],
    )
    draw_time_grid(
        draw,
        pd.Timestamp(times[0]).tz_convert("UTC"),
        pd.Timestamp(times[-1]).tz_convert("UTC"),
        x_of_timestamp,
        plot_top,
        plot_bottom,
        plot_bottom,
        fonts["tiny"],
    )

    channel_overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    channel_draw = ImageDraw.Draw(channel_overlay, "RGBA")
    draw_channel_segments(
        channel_draw,
        segments,
        bar_indices,
        x_of_timestamp,
        y_of_price,
        matched_side=trade_context.side if trade_context is not None and trade_context.timeframe == timeframe and trade_context.family == family else None,
    )
    image.alpha_composite(channel_overlay)
    draw = ImageDraw.Draw(image, "RGBA")

    draw_price_line(draw, bars, timestamps, x_of_seconds, y_of_price)
    if trade_context is not None:
        draw_trade_markers(
            draw,
            trade_context,
            x_of_timestamp,
            y_of_price,
            plot_left,
            plot_top,
            plot_right,
            plot_bottom,
            fonts,
        )

    width_points = build_width_points(bars, segments, bar_indices, timestamps)
    draw_metric_panel(
        draw,
        width_points,
        x_of_seconds,
        metric_top,
        metric_bottom,
        plot_left,
        plot_right,
        fonts,
    )
    draw_footer(draw, segments, plot_left, height - 54, fonts["tiny"])

    image.convert("RGB").save(output, "PNG", optimize=True)


def render_html_plot(
    *,
    output: Path,
    bars: pd.DataFrame,
    segments: list[ChannelSegment],
    config: ZoneChannelProductionConfig,
    timeframe: str,
    family: str,
    trade_context: TradeContext | None,
    scale_mode: str,
    plotly_js: str,
    line_mode: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    times = [pd.Timestamp(value).tz_convert("UTC") for value in pd.to_datetime(bars["close_time"], utc=True, errors="coerce")]
    x_values = [_iso(value) for value in times]
    bar_indices = pd.to_numeric(bars["bar_index"], errors="coerce").to_list()
    closes = [_json_number(value) for value in pd.to_numeric(bars["close"], errors="coerce").to_list()]
    price_low, price_high = price_scale(bars, segments, bar_indices, scale_mode)

    upper_x, upper_y, upper_text = _segment_line_arrays(segments, bars, bar_indices, side="upper", line_mode=line_mode)
    lower_x, lower_y, lower_text = _segment_line_arrays(segments, bars, bar_indices, side="lower", line_mode=line_mode)
    band_x, band_y = _segment_band_arrays(segments, bars, bar_indices, line_mode=line_mode)
    width_x, width_y = _width_atr_arrays(bars, segments, bar_indices)
    point_traces = _definition_point_traces(segments)

    matched_side = (
        trade_context.side
        if trade_context is not None and trade_context.timeframe == timeframe and trade_context.family == family
        else None
    )

    traces: list[dict[str, Any]] = [
        {
            "type": "scatter",
            "mode": "lines",
            "name": "Channel band",
            "x": band_x,
            "y": band_y,
            "fill": "toself",
            "fillcolor": "rgba(45,108,223,0.12)",
            "line": {"color": "rgba(45,108,223,0)", "width": 0},
            "hoverinfo": "skip",
            "showlegend": True,
            "xaxis": "x",
            "yaxis": "y",
        },
        {
            "type": "scattergl",
            "mode": "lines",
            "name": "Close",
            "x": x_values,
            "y": closes,
            "line": {"color": "rgba(31,41,55,0.88)", "width": 1.4},
            "hovertemplate": "%{x}<br>close %{y:,.2f}<extra></extra>",
            "xaxis": "x",
            "yaxis": "y",
        },
        {
            "type": "scatter",
            "mode": "lines",
            "name": "Upper boundary" if line_mode == "fit" else "Upper contact line",
            "x": upper_x,
            "y": upper_y,
            "text": upper_text,
            "line": {"color": "rgba(185,74,90,0.72)", "width": 3 if matched_side == "upper" else 1.4},
            "hovertemplate": "%{text}<br>upper %{y:,.2f}<extra></extra>",
            "xaxis": "x",
            "yaxis": "y",
        },
        {
            "type": "scatter",
            "mode": "lines",
            "name": "Lower boundary" if line_mode == "fit" else "Lower contact line",
            "x": lower_x,
            "y": lower_y,
            "text": lower_text,
            "line": {"color": "rgba(17,24,39,0.9)" if matched_side == "lower" else "rgba(31,111,235,0.72)", "width": 3 if matched_side == "lower" else 1.4},
            "hovertemplate": "%{text}<br>lower %{y:,.2f}<extra></extra>",
            "xaxis": "x",
            "yaxis": "y",
        },
        {
            "type": "scattergl",
            "mode": "lines",
            "name": "Width / ATR",
            "x": width_x,
            "y": width_y,
            "line": {"color": "rgba(75,85,99,0.9)", "width": 1.2},
            "hovertemplate": "%{x}<br>width / ATR %{y:.2f}<extra></extra>",
            "xaxis": "x2",
            "yaxis": "y2",
        },
    ]
    traces.extend(point_traces)

    shapes, annotations, trade_summary = _html_trade_overlays(
        trade_context=trade_context,
        bars=bars,
        price_low=price_low,
        price_high=price_high,
        segments=segments,
    )
    title = f"{config.symbol} {timeframe} {family} channel history"
    subtitle = (
        f"{times[0].date()} to {times[-1].date()} | {len(bars):,} bars | "
        f"{len(segments):,} channel definitions | estimator {config.channel_estimator} | line mode {line_mode}"
    )
    lengths = [segment.end_index - segment.start_index + 1 for segment in segments]
    segment_summary = (
        f"Definition segments: {len(segments):,} | median life {statistics.median(lengths):.0f} bars | max life {max(lengths):,} bars"
        if lengths
        else "No valid channel definitions found."
    )

    layout = {
        "template": "plotly_white",
        "height": 920,
        "margin": {"l": 72, "r": 48, "t": 96, "b": 56},
        "hovermode": "x unified",
        "legend": {"orientation": "h", "x": 0, "y": 1.08},
        "title": {
            "text": f"{escape(title)}<br><sup>{escape(subtitle)}</sup>",
            "x": 0.02,
            "xanchor": "left",
        },
        "xaxis": {
            "domain": [0.0, 1.0],
            "anchor": "y",
            "rangeslider": {"visible": False},
            "rangeselector": {
                "buttons": [
                    {"count": 7, "label": "1w", "step": "day", "stepmode": "backward"},
                    {"count": 1, "label": "1m", "step": "month", "stepmode": "backward"},
                    {"count": 6, "label": "6m", "step": "month", "stepmode": "backward"},
                    {"count": 1, "label": "1y", "step": "year", "stepmode": "backward"},
                    {"step": "all", "label": "All"},
                ],
                "y": 1.02,
            },
        },
        "yaxis": {
            "title": "Price",
            "domain": [0.28, 1.0],
            "range": [price_low, price_high],
            "fixedrange": False,
            "tickformat": ",.0f",
        },
        "xaxis2": {
            "domain": [0.0, 1.0],
            "anchor": "y2",
            "matches": "x",
            "showticklabels": True,
        },
        "yaxis2": {
            "title": "Width / ATR",
            "domain": [0.0, 0.2],
            "rangemode": "tozero",
            "fixedrange": False,
        },
        "shapes": shapes,
        "annotations": annotations,
    }
    config_payload = {
        "responsive": True,
        "displaylogo": False,
        "scrollZoom": True,
        "modeBarButtonsToAdd": ["drawline", "drawrect", "eraseshape"],
    }
    plot_payload = {
        "traces": traces,
        "layout": layout,
        "config": config_payload,
    }
    html = _plotly_html_document(
        plot_payload=plot_payload,
        title=title,
        segment_summary=segment_summary,
        trade_summary=trade_summary,
        plotly_js=plotly_js,
    )
    output.write_text(html, encoding="utf-8")


def render_bfm_html_plot(
    *,
    output: Path,
    bars: pd.DataFrame,
    lines: list[BfmTrendline],
    pivots: list[BfmPivot],
    config: ZoneChannelProductionConfig,
    timeframe: str,
    trade_context: TradeContext | None,
    pivot_sets: list[tuple[int, int]],
    invalidation: str,
    max_extension_bars: int,
    scale_mode: str,
    plotly_js: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    times = [pd.Timestamp(value).tz_convert("UTC") for value in pd.to_datetime(bars["close_time"], utc=True, errors="coerce")]
    x_values = [_iso(value) for value in times]
    price_low, price_high = price_scale(bars, [], pd.to_numeric(bars["bar_index"], errors="coerce").to_list(), scale_mode)

    traces: list[dict[str, Any]] = [
        {
            "type": "ohlc",
            "name": "OHLC",
            "x": x_values,
            "open": [_json_number(value) for value in pd.to_numeric(bars["open"], errors="coerce").to_list()],
            "high": [_json_number(value) for value in pd.to_numeric(bars["high"], errors="coerce").to_list()],
            "low": [_json_number(value) for value in pd.to_numeric(bars["low"], errors="coerce").to_list()],
            "close": [_json_number(value) for value in pd.to_numeric(bars["close"], errors="coerce").to_list()],
            "increasing": {"line": {"color": "rgba(25,135,84,0.48)", "width": 1}},
            "decreasing": {"line": {"color": "rgba(171,28,51,0.48)", "width": 1}},
            "xaxis": "x",
            "yaxis": "y",
        }
    ]
    traces.extend(_bfm_line_traces(lines, bars))
    traces.extend(_bfm_pivot_traces(pivots))

    shapes, annotations, trade_summary = _html_trade_overlays(
        trade_context=trade_context,
        bars=bars,
        price_low=price_low,
        price_high=price_high,
        segments=[],
    )
    title = f"{config.symbol} {timeframe} BFM Magic Trendlines"
    set_text = bfm_set_text(pivot_sets)
    subtitle = (
        f"{times[0].date()} to {times[-1].date()} | {len(bars):,} bars | "
        f"{len(lines):,} trendlines | pivots {set_text} | invalidation {invalidation} | "
        f"max extension {max_extension_bars if max_extension_bars > 0 else 'none'} bars"
    )
    cap_text = f"the fixed {max_extension_bars}-bar cap" if max_extension_bars > 0 else "the end of available data"
    segment_summary = (
        f"BFM mimic: pivots use ta.pivothigh/ta.pivotlow-style left/right windows. "
        f"Each trendline is drawn through the latest two same-side pivots and extended until "
        f"{invalidation} invalidation or {cap_text}."
    )
    layout = {
        "template": "plotly_white",
        "height": 920,
        "margin": {"l": 72, "r": 48, "t": 106, "b": 56},
        "hovermode": "x unified",
        "legend": {"orientation": "h", "x": 0, "y": 1.08},
        "title": {
            "text": f"{escape(title)}<br><sup>{escape(subtitle)}</sup>",
            "x": 0.02,
            "xanchor": "left",
        },
        "xaxis": {
            "domain": [0.0, 1.0],
            "anchor": "y",
            "rangeslider": {"visible": True, "thickness": 0.07},
            "rangeselector": {
                "buttons": [
                    {"count": 7, "label": "1w", "step": "day", "stepmode": "backward"},
                    {"count": 1, "label": "1m", "step": "month", "stepmode": "backward"},
                    {"count": 6, "label": "6m", "step": "month", "stepmode": "backward"},
                    {"count": 1, "label": "1y", "step": "year", "stepmode": "backward"},
                    {"step": "all", "label": "All"},
                ],
                "y": 1.02,
            },
        },
        "yaxis": {
            "title": "Price",
            "range": [price_low, price_high],
            "fixedrange": False,
            "tickformat": ",.0f",
        },
        "shapes": shapes,
        "annotations": annotations,
    }
    config_payload = {
        "responsive": True,
        "displaylogo": False,
        "scrollZoom": True,
        "modeBarButtonsToAdd": ["drawline", "drawrect", "eraseshape"],
    }
    html = _plotly_html_document(
        plot_payload={"traces": traces, "layout": layout, "config": config_payload},
        title=title,
        segment_summary=segment_summary,
        trade_summary=trade_summary,
        plotly_js=plotly_js,
    )
    output.write_text(html, encoding="utf-8")


def render_bfm_multitimeframe_html_plot(
    *,
    output: Path,
    base_bars: pd.DataFrame,
    results: list[BfmTimeframeResult],
    config: ZoneChannelProductionConfig,
    base_timeframe: str,
    trade_context: TradeContext | None,
    pivot_sets: list[tuple[int, int]],
    invalidation: str,
    max_extension_bars: int,
    scale_mode: str,
    plotly_js: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    times = [pd.Timestamp(value).tz_convert("UTC") for value in pd.to_datetime(base_bars["close_time"], utc=True, errors="coerce")]
    x_values = [_iso(value) for value in times]
    price_low, price_high = price_scale(base_bars, [], pd.to_numeric(base_bars["bar_index"], errors="coerce").to_list(), scale_mode)

    traces: list[dict[str, Any]] = [
        {
            "type": "ohlc",
            "name": f"{base_timeframe} OHLC",
            "x": x_values,
            "open": [_json_number(value) for value in pd.to_numeric(base_bars["open"], errors="coerce").to_list()],
            "high": [_json_number(value) for value in pd.to_numeric(base_bars["high"], errors="coerce").to_list()],
            "low": [_json_number(value) for value in pd.to_numeric(base_bars["low"], errors="coerce").to_list()],
            "close": [_json_number(value) for value in pd.to_numeric(base_bars["close"], errors="coerce").to_list()],
            "increasing": {"line": {"color": "rgba(25,135,84,0.42)", "width": 1}},
            "decreasing": {"line": {"color": "rgba(171,28,51,0.42)", "width": 1}},
            "xaxis": "x",
            "yaxis": "y",
        }
    ]
    for result in results:
        traces.extend(_bfm_line_traces(result.lines, result.bars, timeframe=result.timeframe))
    for result in results:
        traces.extend(_bfm_pivot_traces(result.pivots, timeframe=result.timeframe))

    shapes, annotations, trade_summary = _html_trade_overlays(
        trade_context=trade_context,
        bars=base_bars,
        price_low=price_low,
        price_high=price_high,
        segments=[],
    )
    title = f"{config.symbol} multi-timeframe BFM Magic Trendlines"
    set_text = bfm_result_set_text(results)
    tf_text = ", ".join(
        f"{result.timeframe}: {len(result.lines):,} lines / {len(result.pivots):,} pivots"
        for result in results
    )
    subtitle = (
        f"{times[0].date()} to {times[-1].date()} | base {base_timeframe} | "
        f"{tf_text} | pivots {set_text} | invalidation {invalidation} | "
        f"max extension {max_extension_bars if max_extension_bars > 0 else 'none'} bars"
    )
    cap_text = f"the fixed {max_extension_bars}-bar cap" if max_extension_bars > 0 else "the end of available data"
    segment_summary = (
        f"Multi-timeframe BFM mimic. Each timeframe is resampled independently, pivots use "
        f"ta.pivothigh/ta.pivotlow-style left/right windows, and trendlines extend until "
        f"{invalidation} invalidation or {cap_text}."
    )
    layout = {
        "template": "plotly_white",
        "height": 920,
        "margin": {"l": 72, "r": 48, "t": 118, "b": 56},
        "hovermode": "x unified",
        "legend": {"orientation": "h", "x": 0, "y": 1.1},
        "title": {
            "text": f"{escape(title)}<br><sup>{escape(subtitle)}</sup>",
            "x": 0.02,
            "xanchor": "left",
        },
        "xaxis": {
            "domain": [0.0, 1.0],
            "anchor": "y",
            "rangeslider": {"visible": True, "thickness": 0.07},
            "rangeselector": {
                "buttons": [
                    {"count": 7, "label": "1w", "step": "day", "stepmode": "backward"},
                    {"count": 1, "label": "1m", "step": "month", "stepmode": "backward"},
                    {"count": 6, "label": "6m", "step": "month", "stepmode": "backward"},
                    {"count": 1, "label": "1y", "step": "year", "stepmode": "backward"},
                    {"step": "all", "label": "All"},
                ],
                "y": 1.02,
            },
        },
        "yaxis": {
            "title": "Price",
            "range": [price_low, price_high],
            "fixedrange": False,
            "tickformat": ",.0f",
        },
        "shapes": shapes,
        "annotations": annotations,
    }
    config_payload = {
        "responsive": True,
        "displaylogo": False,
        "scrollZoom": True,
        "modeBarButtonsToAdd": ["drawline", "drawrect", "eraseshape"],
    }
    html = _plotly_html_document(
        plot_payload={"traces": traces, "layout": layout, "config": config_payload},
        title=title,
        segment_summary=segment_summary,
        trade_summary=trade_summary,
        plotly_js=plotly_js,
    )
    output.write_text(html, encoding="utf-8")


def _bfm_line_traces(
    lines: list[BfmTrendline],
    bars: pd.DataFrame,
    *,
    timeframe: str | None = None,
) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    colors = {
        1: "rgba(107,114,128,0.95)",
        2: "rgba(220,38,38,0.95)",
        3: "rgba(22,163,74,0.95)",
        4: "rgba(37,99,235,0.95)",
    }
    timeframe_width = {
        "5m": 0.9,
        "15m": 1.0,
        "1h": 1.35,
        "4h": 2.0,
        "1d": 2.6,
        "1w": 3.0,
    }
    timeframe_opacity = {
        "5m": 0.45,
        "15m": 0.55,
        "1h": 0.72,
        "4h": 0.90,
        "1d": 0.96,
        "1w": 1.0,
    }
    prefix = f"{timeframe} " if timeframe else ""
    width = timeframe_width.get(str(timeframe), 1.6)
    opacity = timeframe_opacity.get(str(timeframe), 0.95)
    for set_number in sorted({line.set_number for line in lines}):
        for side in ["resistance", "support"]:
            side_lines = [line for line in lines if line.set_number == set_number and line.side == side]
            if not side_lines:
                continue
            xs: list[str | None] = []
            ys: list[float | None] = []
            texts: list[str | None] = []
            for line in side_lines:
                end_time = timestamp_at_index(bars, float(line.line_end_index))
                x0 = _iso(line.start_pivot.pivot_time)
                x1 = _iso(line.end_pivot.pivot_time)
                x2 = end_time
                y0 = line.start_pivot.price
                y1 = line.end_pivot.price
                y2 = line.evaluate(float(line.line_end_index))
                text = (
                    f"{prefix}Set {line.set_number} {side}<br>"
                    f"pivots {line.start_pivot.pivot_index} -> {line.end_pivot.pivot_index}<br>"
                    f"{line.start_pivot.price:,.2f} -> {line.end_pivot.price:,.2f}<br>"
                    f"slope {line.slope:.4f}<br>"
                    f"{'invalidated' if line.invalidated else 'not invalidated'}"
                )
                xs.extend([x0, x1, x2, None])
                ys.extend([_json_number(y0), _json_number(y1), _json_number(y2), None])
                texts.extend([text, text, text, None])
            traces.append(
                {
                    "type": "scatter",
                    "mode": "lines",
                    "name": f"{prefix}S{set_number} {side}",
                    "x": xs,
                    "y": ys,
                    "text": texts,
                    "line": {
                        "color": colors.get(set_number, "rgba(17,24,39,0.9)"),
                        "width": width,
                        "dash": "dot",
                    },
                    "opacity": opacity,
                    "hovertemplate": "%{text}<br>%{y:,.2f}<extra></extra>",
                    "xaxis": "x",
                    "yaxis": "y",
                }
            )
    return traces


def _bfm_pivot_traces(
    pivots: list[BfmPivot],
    *,
    timeframe: str | None = None,
) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    colors = {
        1: "rgba(107,114,128,0.7)",
        2: "rgba(220,38,38,0.7)",
        3: "rgba(22,163,74,0.7)",
        4: "rgba(37,99,235,0.7)",
    }
    prefix = f"{timeframe} " if timeframe else ""
    for set_number in sorted({pivot.set_number for pivot in pivots}):
        for side, symbol in [("resistance", "circle"), ("support", "circle-open")]:
            side_pivots = [pivot for pivot in pivots if pivot.set_number == set_number and pivot.side == side]
            if not side_pivots:
                continue
            traces.append(
                {
                    "type": "scattergl",
                    "mode": "markers",
                    "name": f"{prefix}S{set_number} {side} pivots",
                    "x": [_iso(pivot.pivot_time) for pivot in side_pivots],
                    "y": [float(pivot.price) for pivot in side_pivots],
                    "text": [
                        (
                            f"Set {pivot.set_number} {side} pivot<br>"
                            f"timeframe {timeframe or ''}<br>"
                            f"pivot {pivot.pivot_time:%Y-%m-%d %H:%M}<br>"
                            f"confirmed {pivot.confirm_time:%Y-%m-%d %H:%M}<br>"
                            f"left/right {pivot.leftbars}/{pivot.rightbars}<br>"
                            f"price {pivot.price:,.2f}"
                        )
                        for pivot in side_pivots
                    ],
                    "marker": {
                        "size": 7,
                        "color": colors.get(set_number, "rgba(17,24,39,0.7)"),
                        "symbol": symbol,
                    },
                    "hovertemplate": "%{text}<extra></extra>",
                    "xaxis": "x",
                    "yaxis": "y",
                    "visible": "legendonly",
                }
            )
    return traces


def _segment_line_arrays(
    segments: list[ChannelSegment],
    bars: pd.DataFrame,
    bar_indices: list[float],
    *,
    side: str,
    line_mode: str,
) -> tuple[list[str | None], list[float | None], list[str | None]]:
    xs: list[str | None] = []
    ys: list[float | None] = []
    texts: list[str | None] = []
    for segment in segments:
        line, anchor_start, anchor_end = display_line_for_segment(segment, side, line_mode)
        upper_line, _, _ = display_line_for_segment(segment, "upper", line_mode)
        lower_line, _, _ = display_line_for_segment(segment, "lower", line_mode)
        end_index = display_segment_end_index(segment, upper_line, lower_line)
        if line_mode == "anchors" and anchor_start is not None and anchor_end is not None:
            points_x = [
                _iso(anchor_start.time),
                _iso(anchor_end.time),
                timestamp_at_index(bars, end_index),
                None,
            ]
            points_y = [
                _json_number(anchor_start.price),
                _json_number(anchor_end.price),
                _json_number(line.evaluate(end_index)),
                None,
            ]
            text = (
                f"segment {segment.segment_id}<br>{segment.start_time:%Y-%m-%d %H:%M} -> "
                f"{segment.end_time:%Y-%m-%d %H:%M}<br>{side} anchor "
                f"{anchor_start.index} -> {anchor_end.index}<br>slope {line.slope:.4f}"
            )
            xs.extend(points_x)
            ys.extend(points_y)
            texts.extend([text, text, text, None])
        else:
            x0 = float(bar_indices[segment.start_index])
            x1 = float(bar_indices[segment.end_index])
            y0 = line.evaluate(x0)
            y1 = line.evaluate(x1)
            text = (
                f"segment {segment.segment_id}<br>{segment.start_time:%Y-%m-%d %H:%M} -> "
                f"{segment.end_time:%Y-%m-%d %H:%M}<br>{side} slope {line.slope:.4f}"
            )
            xs.extend([_iso(segment.start_time), _iso(segment.end_time), None])
            ys.extend([_json_number(y0), _json_number(y1), None])
            texts.extend([text, text, None])
    return xs, ys, texts


def _segment_band_arrays(
    segments: list[ChannelSegment],
    bars: pd.DataFrame,
    bar_indices: list[float],
    *,
    line_mode: str,
) -> tuple[list[str | None], list[float | None]]:
    xs: list[str | None] = []
    ys: list[float | None] = []
    for segment in segments:
        x0 = float(bar_indices[segment.start_index])
        upper_line, _, _ = display_line_for_segment(segment, "upper", line_mode)
        lower_line, _, _ = display_line_for_segment(segment, "lower", line_mode)
        end_index = display_segment_end_index(segment, upper_line, lower_line)
        x1 = float(end_index)
        upper_start = upper_line.evaluate(x0)
        upper_end = upper_line.evaluate(x1)
        lower_start = lower_line.evaluate(x0)
        lower_end = lower_line.evaluate(x1)
        if upper_start <= lower_start or upper_end <= lower_end:
            continue
        start = _iso(segment.start_time)
        end = timestamp_at_index(bars, x1)
        xs.extend([start, end, end, start, None])
        ys.extend(
            [
                _json_number(upper_start),
                _json_number(upper_end),
                _json_number(lower_end),
                _json_number(lower_start),
                None,
            ]
        )
    return xs, ys


def _definition_point_traces(segments: list[ChannelSegment]) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    for side, color, symbol in [
        ("upper", "rgba(185,74,90,0.72)", "triangle-down"),
        ("lower", "rgba(31,111,235,0.72)", "triangle-up"),
    ]:
        seen: set[tuple[int, float]] = set()
        xs: list[str] = []
        ys: list[float] = []
        texts: list[str] = []
        for segment in segments:
            points = segment.upper_points if side == "upper" else segment.lower_points
            for point in points:
                key = (point.index, round(float(point.price), 8))
                if key in seen:
                    continue
                seen.add(key)
                xs.append(_iso(point.time))
                ys.append(float(point.price))
                texts.append(
                    f"{side} defining point<br>pivot index {point.index}<br>"
                    f"pivot {point.time:%Y-%m-%d %H:%M}<br>"
                    f"confirmed {point.confirm_time:%Y-%m-%d %H:%M}<br>"
                    f"price {point.price:,.2f}"
                )
        traces.append(
            {
                "type": "scattergl",
                "mode": "markers",
                "name": f"{side.title()} defining points",
                "x": xs,
                "y": ys,
                "text": texts,
                "marker": {"size": 5, "color": color, "symbol": symbol},
                "hovertemplate": "%{text}<extra></extra>",
                "xaxis": "x",
                "yaxis": "y",
            }
        )
    return traces


def _width_atr_arrays(
    bars: pd.DataFrame,
    segments: list[ChannelSegment],
    bar_indices: list[float],
) -> tuple[list[str], list[float | None]]:
    times = [pd.Timestamp(value).tz_convert("UTC") for value in pd.to_datetime(bars["close_time"], utc=True, errors="coerce")]
    atrs = pd.to_numeric(bars["atr"], errors="coerce").to_list()
    xs: list[str] = []
    ys: list[float | None] = []
    for segment in segments:
        for index in range(segment.start_index, segment.end_index + 1):
            atr = safe_float(atrs[index])
            if atr is None or atr <= 0.0:
                continue
            xs.append(_iso(times[index]))
            ys.append(_json_number(segment.width_value(float(bar_indices[index])) / atr))
    return xs, ys


def _html_trade_overlays(
    *,
    trade_context: TradeContext | None,
    bars: pd.DataFrame,
    price_low: float,
    price_high: float,
    segments: list[ChannelSegment],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    if trade_context is None:
        return [], [], ""

    row = trade_context.row
    start_time = _iso(pd.Timestamp(bars["close_time"].iloc[0]).tz_convert("UTC"))
    end_time = _iso(pd.Timestamp(bars["close_time"].iloc[-1]).tz_convert("UTC"))
    shapes: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []

    zone_top = safe_float(row.get("zone_top"))
    zone_bottom = safe_float(row.get("zone_bottom"))
    if zone_top is not None and zone_bottom is not None:
        shapes.append(
            {
                "type": "rect",
                "xref": "x",
                "yref": "y",
                "x0": start_time,
                "x1": end_time,
                "y0": min(zone_top, zone_bottom),
                "y1": max(zone_top, zone_bottom),
                "fillcolor": "rgba(49,151,93,0.16)",
                "line": {"color": "rgba(49,151,93,0.55)", "width": 1},
                "layer": "below",
            }
        )

    for name, key, color in [
        ("E", "entry_price", "#111827"),
        ("TP", "target_price", "#198754"),
        ("SL", "stop_price", "#AB1C33"),
    ]:
        price = safe_float(row.get(key))
        if price is None:
            continue
        shapes.append(
            {
                "type": "line",
                "xref": "x",
                "yref": "y",
                "x0": start_time,
                "x1": end_time,
                "y0": price,
                "y1": price,
                "line": {"color": color, "width": 1.2},
                "layer": "above",
            }
        )
        annotations.append(
            {
                "xref": "paper",
                "yref": "y",
                "x": 0.002,
                "y": price,
                "text": f"{name} {price:,.0f}",
                "showarrow": False,
                "font": {"size": 11, "color": color},
                "bgcolor": "rgba(255,255,255,0.75)",
            }
        )

    for label, key, color in [
        ("event", "event_time", "#6B7280"),
        ("entry", "entry_time", "#111827"),
        ("exit", "exit_time", "#B22222"),
    ]:
        raw_time = row.get(key)
        if pd.isna(raw_time):
            continue
        timestamp = _iso(pd.Timestamp(raw_time).tz_convert("UTC"))
        shapes.append(
            {
                "type": "line",
                "xref": "x",
                "yref": "paper",
                "x0": timestamp,
                "x1": timestamp,
                "y0": 0.22,
                "y1": 1.0,
                "line": {"color": color, "width": 1.4},
                "layer": "above",
            }
        )
        annotations.append(
            {
                "xref": "x",
                "yref": "paper",
                "x": timestamp,
                "y": 1.0,
                "text": label,
                "showarrow": False,
                "yshift": 12,
                "font": {"size": 11, "color": color},
                "bgcolor": "rgba(255,255,255,0.75)",
            }
        )

    event_time = pd.Timestamp(row.get("event_time")).tz_convert("UTC")
    active_segment = next(
        (
            segment
            for segment in segments
            if segment.start_time <= event_time <= segment.end_time
        ),
        None,
    )
    active_text = ""
    if active_segment is not None:
        active_text = (
            f"Active segment {active_segment.segment_id}: "
            f"{active_segment.start_time:%Y-%m-%d %H:%M} -> {active_segment.end_time:%Y-%m-%d %H:%M}"
        )
    net_r = safe_float(row.get("r_multiple_net")) or 0.0
    matched_value = safe_float(row.get("matched_boundary_value"))
    trade_summary = (
        f"Trade #{trade_context.trade_number:03d}: {event_time:%Y-%m-%d %H:%M UTC} "
        f"{str(row.get('direction')).upper()} {net_r:+.2f}R | matched "
        f"{trade_context.timeframe}/{trade_context.family}/{trade_context.side}"
    )
    if matched_value is not None:
        trade_summary += f" at {matched_value:,.2f}"
    if active_text:
        trade_summary += f" | {active_text}"
    return shapes, annotations, trade_summary


def _plotly_html_document(
    *,
    plot_payload: dict[str, Any],
    title: str,
    segment_summary: str,
    trade_summary: str,
    plotly_js: str,
) -> str:
    if plotly_js != "cdn":
        raise ValueError(f"Unsupported plotly JS mode: {plotly_js}")
    payload_json = json.dumps(plot_payload, allow_nan=False, separators=(",", ":"))
    trade_html = f"<p>{escape(trade_summary)}</p>" if trade_summary else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    html, body {{ margin: 0; padding: 0; background: #f6f8fb; color: #111827; font-family: Arial, sans-serif; }}
    #wrap {{ max-width: 1800px; margin: 0 auto; padding: 18px 18px 28px; }}
    #chart {{ width: 100%; height: 920px; background: white; border: 1px solid #d7dde7; }}
    .note {{ color: #4b5563; font-size: 13px; line-height: 1.45; margin: 10px 0 0; }}
    .note p {{ margin: 4px 0; }}
  </style>
</head>
<body>
  <div id="wrap">
    <div id="chart"></div>
    <div class="note">
      <p>{escape(segment_summary)}</p>
      {trade_html}
      <p>Use the modebar or mouse wheel to zoom, drag to pan, and double-click to reset.</p>
    </div>
  </div>
  <script>
    const spec = {payload_json};
    Plotly.newPlot("chart", spec.traces, spec.layout, spec.config);
  </script>
</body>
</html>
"""


def _iso(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).tz_convert("UTC").isoformat()


def _json_number(value: Any) -> float | None:
    number = safe_float(value)
    return float(number) if number is not None else None


def price_scale(
    bars: pd.DataFrame,
    segments: list[ChannelSegment],
    bar_indices: list[float],
    scale_mode: str,
) -> tuple[float, float]:
    values = (
        pd.to_numeric(bars["low"], errors="coerce").dropna().to_list()
        + pd.to_numeric(bars["high"], errors="coerce").dropna().to_list()
    )
    if scale_mode == "channel":
        for segment in segments:
            for index in [segment.start_index, segment.end_index]:
                x_value = float(bar_indices[index])
                values.append(segment.upper_value(x_value))
                values.append(segment.lower_value(x_value))
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return 0.0, 1.0
    low = min(finite)
    high = max(finite)
    if high <= low:
        high = low + 1.0
    pad = (high - low) * 0.055
    return low - pad, high + pad


def draw_plot_frame(
    draw: ImageDraw.ImageDraw,
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> None:
    draw.rectangle((left, top, right, bottom), fill=(255, 255, 255, 255), outline=(210, 216, 225, 255), width=1)


def draw_price_grid(
    draw: ImageDraw.ImageDraw,
    low: float,
    high: float,
    left: int,
    top: int,
    right: int,
    bottom: int,
    font: ImageFont.ImageFont,
) -> None:
    for tick in nice_ticks(low, high, 7):
        y = top + ((high - tick) / max(1e-9, high - low)) * (bottom - top)
        draw.line((left, y, right, y), fill=(233, 237, 242, 255), width=1)
        label = f"{tick:,.0f}"
        draw.text((right + 8, y - 7), label, fill=(93, 101, 112, 255), font=font)


def draw_time_grid(
    draw: ImageDraw.ImageDraw,
    start: pd.Timestamp,
    end: pd.Timestamp,
    x_of_timestamp: Any,
    top: int,
    bottom: int,
    label_y: int,
    font: ImageFont.ImageFont,
) -> None:
    years = range(start.year + 1, end.year + 1)
    for year in years:
        tick = pd.Timestamp(f"{year}-01-01", tz="UTC")
        if tick < start or tick > end:
            continue
        x = x_of_timestamp(tick)
        draw.line((x, top, x, bottom), fill=(235, 239, 244, 255), width=1)
        draw.text((x - 15, label_y + 8), str(year), fill=(93, 101, 112, 255), font=font)


def draw_channel_segments(
    draw: ImageDraw.ImageDraw,
    segments: list[ChannelSegment],
    bar_indices: list[float],
    x_of_timestamp: Any,
    y_of_price: Any,
    *,
    matched_side: str | None,
) -> None:
    fill_color = rgba("#2D6CDF22")
    upper_color = rgba("#B94A5A99")
    lower_color = rgba("#1F6FEBB0")
    matched_color = rgba("#111827E6")
    for segment in segments:
        x0 = x_of_timestamp(segment.start_time)
        x1 = x_of_timestamp(segment.end_time)
        if x1 < x0:
            continue
        start_x = float(bar_indices[segment.start_index])
        end_x = float(bar_indices[segment.end_index])
        upper_start = segment.upper_value(start_x)
        upper_end = segment.upper_value(end_x)
        lower_start = segment.lower_value(start_x)
        lower_end = segment.lower_value(end_x)
        y_upper_start = y_of_price(upper_start)
        y_upper_end = y_of_price(upper_end)
        y_lower_start = y_of_price(lower_start)
        y_lower_end = y_of_price(lower_end)
        if x1 - x0 >= 0.4:
            draw.polygon(
                [(x0, y_upper_start), (x1, y_upper_end), (x1, y_lower_end), (x0, y_lower_start)],
                fill=fill_color,
            )
        upper_width = 1
        lower_width = 1
        upper_line = upper_color
        lower_line = lower_color
        if matched_side == "upper":
            upper_width = 3
            upper_line = matched_color
        elif matched_side == "lower":
            lower_width = 3
            lower_line = matched_color
        draw.line((x0, y_upper_start, x1, y_upper_end), fill=upper_line, width=upper_width)
        draw.line((x0, y_lower_start, x1, y_lower_end), fill=lower_line, width=lower_width)


def draw_price_line(
    draw: ImageDraw.ImageDraw,
    bars: pd.DataFrame,
    timestamps: list[float],
    x_of_seconds: Any,
    y_of_price: Any,
) -> None:
    closes = pd.to_numeric(bars["close"], errors="coerce").to_list()
    points: list[tuple[float, float]] = []
    last_x: int | None = None
    for timestamp, close in zip(timestamps, closes):
        if not math.isfinite(float(close)):
            continue
        x = x_of_seconds(float(timestamp))
        y = y_of_price(float(close))
        x_int = int(round(x))
        if last_x == x_int and points:
            points[-1] = (x, y)
        else:
            points.append((x, y))
            last_x = x_int
    if len(points) >= 2:
        draw.line(points, fill=(31, 41, 55, 205), width=2)


def draw_trade_markers(
    draw: ImageDraw.ImageDraw,
    trade_context: TradeContext,
    x_of_timestamp: Any,
    y_of_price: Any,
    left: int,
    top: int,
    right: int,
    bottom: int,
    fonts: dict[str, ImageFont.ImageFont],
) -> None:
    row = trade_context.row
    marker_specs = [
        ("event", row.get("event_time"), "#6B7280"),
        ("entry", row.get("entry_time"), "#111827"),
        ("exit", row.get("exit_time"), "#8A6D1D" if str(row.get("exit_reason")) == "timeout" else "#B22222"),
    ]
    for label, raw_time, color in marker_specs:
        if pd.isna(raw_time):
            continue
        x = x_of_timestamp(pd.Timestamp(raw_time).tz_convert("UTC"))
        draw.line((x, top, x, bottom), fill=rgba(color + "CC"), width=2)
        draw.text((x + 5, top + 8), label, fill=rgba(color), font=fonts["tiny"])

    zone_top = safe_float(row.get("zone_top"))
    zone_bottom = safe_float(row.get("zone_bottom"))
    if zone_top is not None and zone_bottom is not None:
        y_top = y_of_price(max(zone_top, zone_bottom))
        y_bottom = y_of_price(min(zone_top, zone_bottom))
        draw.rectangle((left, y_top, right, y_bottom), fill=(49, 151, 93, 28), outline=(49, 151, 93, 150), width=1)

    labels = [
        ("E", safe_float(row.get("entry_price")), "#111827"),
        ("TP", safe_float(row.get("target_price")), "#198754"),
        ("SL", safe_float(row.get("stop_price")), "#AB1C33"),
    ]
    for name, price, color in labels:
        if price is None:
            continue
        y = y_of_price(price)
        draw.line((left, y, right, y), fill=rgba(color + "AA"), width=1)
        draw.text((left + 8, y - 17), f"{name} {price:,.0f}", fill=rgba(color), font=fonts["tiny"])

    direction = str(row.get("direction", "")).upper()
    net_r = safe_float(row.get("r_multiple_net")) or 0.0
    event_time = pd.Timestamp(row.get("event_time")).tz_convert("UTC")
    note = f"#{trade_context.trade_number:03d} {event_time:%Y-%m-%d %H:%M} {direction} {net_r:+.2f}R"
    draw.rounded_rectangle((left + 14, top + 14, left + 430, top + 52), radius=6, fill=(255, 255, 255, 225), outline=(210, 216, 225, 255))
    draw.text((left + 26, top + 22), note, fill=(17, 24, 39, 255), font=fonts["label_bold"])


def build_width_points(
    bars: pd.DataFrame,
    segments: list[ChannelSegment],
    bar_indices: list[float],
    timestamps: list[float],
) -> list[tuple[float, float]]:
    atrs = pd.to_numeric(bars["atr"], errors="coerce").to_list()
    points: list[tuple[float, float]] = []
    for segment in segments:
        for index in range(segment.start_index, segment.end_index + 1):
            atr = safe_float(atrs[index])
            if atr is None or atr <= 0.0:
                continue
            width = segment.width_value(float(bar_indices[index]))
            if math.isfinite(width):
                points.append((float(timestamps[index]), width / atr))
    return points


def draw_metric_panel(
    draw: ImageDraw.ImageDraw,
    width_points: list[tuple[float, float]],
    x_of_seconds: Any,
    top: int,
    bottom: int,
    left: int,
    right: int,
    fonts: dict[str, ImageFont.ImageFont],
) -> None:
    draw.rectangle((left, top, right, bottom), fill=(255, 255, 255, 255), outline=(210, 216, 225, 255), width=1)
    draw.text((left, top - 23), "Channel width / ATR", fill=(55, 65, 81, 255), font=fonts["label_bold"])
    if not width_points:
        return
    values = sorted(value for _, value in width_points if math.isfinite(value))
    if not values:
        return
    max_value = percentile(values, 0.98)
    max_value = max(1.0, max_value)

    for frac in [0.25, 0.5, 0.75, 1.0]:
        y = bottom - frac * (bottom - top)
        draw.line((left, y, right, y), fill=(233, 237, 242, 255), width=1)
        draw.text((right + 8, y - 7), f"{max_value * frac:.1f}", fill=(93, 101, 112, 255), font=fonts["tiny"])

    points: list[tuple[float, float]] = []
    last_x: int | None = None
    for timestamp, value in width_points:
        clipped = max(0.0, min(max_value, value))
        x = x_of_seconds(timestamp)
        y = bottom - (clipped / max_value) * (bottom - top)
        x_int = int(round(x))
        if last_x == x_int and points:
            points[-1] = (x, y)
        else:
            points.append((x, y))
            last_x = x_int
    if len(points) >= 2:
        draw.line(points, fill=(75, 85, 99, 210), width=2)


def draw_footer(
    draw: ImageDraw.ImageDraw,
    segments: list[ChannelSegment],
    left: int,
    y: int,
    font: ImageFont.ImageFont,
) -> None:
    lengths = [segment.end_index - segment.start_index + 1 for segment in segments]
    if lengths:
        median_length = statistics.median(lengths)
        max_length = max(lengths)
        text = f"Definition segments: {len(segments):,} | median life {median_length:.0f} bars | max life {max_length:,} bars"
    else:
        text = "No valid channel definitions found."
    draw.text((left, y), text, fill=(93, 101, 112, 255), font=font)


def nice_ticks(low: float, high: float, count: int) -> list[float]:
    if high <= low:
        return [low]
    raw_step = (high - low) / max(1, count - 1)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    residual = raw_step / magnitude
    if residual >= 5:
        nice_step = 5 * magnitude
    elif residual >= 2:
        nice_step = 2 * magnitude
    else:
        nice_step = magnitude
    first = math.ceil(low / nice_step) * nice_step
    ticks = []
    value = first
    while value <= high + nice_step * 0.5:
        ticks.append(value)
        value += nice_step
    return ticks


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    position = (len(values) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def default_output_path(
    config: ZoneChannelProductionConfig,
    timeframe: str,
    family: str,
    trade_context: TradeContext | None,
    output_format: str = "png",
) -> Path:
    stem = f"{config.symbol.lower()}_{timeframe}_{family}_channel_history"
    if trade_context is not None:
        stem += f"_trade{trade_context.trade_number:03d}"
    suffix = "html" if output_format == "html" else "png"
    return Path("scripts") / f"{stem}.{suffix}"


def default_bfm_output_path(
    config: ZoneChannelProductionConfig,
    timeframe: str,
    trade_context: TradeContext | None,
) -> Path:
    stem = f"{config.symbol.lower()}_{timeframe}_bfm_magic_trendlines"
    if trade_context is not None:
        stem += f"_trade{trade_context.trade_number:03d}"
    return Path("scripts") / f"{stem}.html"


def resolve_output_format(args: argparse.Namespace) -> str:
    if args.format != "auto":
        return str(args.format)
    if args.output is not None and args.output.suffix.lower() in {".html", ".htm"}:
        return "html"
    if args.logic == "bfm":
        return "html"
    return "png"


def main() -> None:
    args = parse_args()
    config = load_production_config(args.config)
    trade_context = load_trade_context(args, config)
    timeframe, family = resolve_timeframe_and_family(args, config, trade_context)
    output_format = resolve_output_format(args)
    output = args.output or (
        default_bfm_output_path(config, timeframe, trade_context)
        if args.logic == "bfm"
        else default_output_path(config, timeframe, family, trade_context, output_format)
    )

    base = load_base_candles(config.symbol, args.start, args.end, cache_dir=args.cache_dir, interval=config.base_interval)
    bars = prepare_timeframe_bars(base, timeframe, atr_length=config.atr_length)

    if args.logic == "bfm":
        pivot_sets = parse_bfm_sets(args.bfm_sets)
        bfm_timeframes = parse_timeframes(args.bfm_timeframes, timeframe)
        pivot_sets_by_timeframe = parse_bfm_sets_by_timeframe(args.bfm_tf_sets, bfm_timeframes, pivot_sets)
        results: list[BfmTimeframeResult] = []
        for bfm_timeframe in bfm_timeframes:
            bfm_bars = bars if bfm_timeframe == timeframe else prepare_timeframe_bars(base, bfm_timeframe, atr_length=config.atr_length)
            timeframe_pivot_sets = pivot_sets_by_timeframe[bfm_timeframe]
            lines, pivots = build_bfm_magic_lines(
                bfm_bars,
                timeframe_pivot_sets,
                invalidation=args.bfm_invalidation,
                max_extension_bars=args.bfm_max_extension_bars,
            )
            results.append(
                BfmTimeframeResult(
                    timeframe=bfm_timeframe,
                    bars=bfm_bars,
                    lines=lines,
                    pivots=pivots,
                    pivot_sets=tuple(timeframe_pivot_sets),
                )
            )
        if len(results) == 1:
            result = results[0]
            render_bfm_html_plot(
                output=output,
                bars=result.bars,
                lines=result.lines,
                pivots=result.pivots,
                config=config,
                timeframe=result.timeframe,
                trade_context=trade_context,
                pivot_sets=list(result.pivot_sets),
                invalidation=args.bfm_invalidation,
                max_extension_bars=args.bfm_max_extension_bars,
                scale_mode=args.scale_mode,
                plotly_js=args.plotly_js,
            )
            print(f"Saved BFM Magic Trendlines html to {output}")
            print(f"{result.timeframe} BFM: {len(result.bars):,} bars, {len(result.pivots):,} pivots, {len(result.lines):,} trendlines")
        else:
            render_bfm_multitimeframe_html_plot(
                output=output,
                base_bars=bars,
                results=results,
                config=config,
                base_timeframe=timeframe,
                trade_context=trade_context,
                pivot_sets=pivot_sets,
                invalidation=args.bfm_invalidation,
                max_extension_bars=args.bfm_max_extension_bars,
                scale_mode=args.scale_mode,
                plotly_js=args.plotly_js,
            )
            print(f"Saved multi-timeframe BFM Magic Trendlines html to {output}")
            for result in results:
                print(f"{result.timeframe} BFM: {len(result.bars):,} bars, {len(result.pivots):,} pivots, {len(result.lines):,} trendlines")
        return

    segments = build_channel_segments(bars, config, timeframe, family)

    segments_output = args.segments_output
    if segments_output is not None:
        write_segments_csv(segments_output, segments, bars)

    if output_format == "html":
        render_html_plot(
            output=output,
            bars=bars,
            segments=segments,
            config=config,
            timeframe=timeframe,
            family=family,
            trade_context=trade_context,
            scale_mode=args.scale_mode,
            plotly_js=args.plotly_js,
            line_mode=args.line_mode,
        )
    else:
        render_plot(
            output=output,
            bars=bars,
            segments=segments,
            config=config,
            timeframe=timeframe,
            family=family,
            trade_context=trade_context,
            width=args.width,
            height=args.height,
            scale_mode=args.scale_mode,
        )

    print(f"Saved channel history {output_format} to {output}")
    if segments_output is not None:
        print(f"Saved channel definition segments to {segments_output}")
    if trade_context is not None:
        row = trade_context.row
        print(
            f"Trade #{trade_context.trade_number:03d}: matched {trade_context.timeframe}/{trade_context.family} "
            f"{trade_context.side} boundary at {safe_float(row.get('matched_boundary_value')):.2f}"
        )
    print(f"{timeframe} {family}: {len(bars):,} bars, {len(segments):,} valid piecewise channel definitions")


if __name__ == "__main__":
    main()
