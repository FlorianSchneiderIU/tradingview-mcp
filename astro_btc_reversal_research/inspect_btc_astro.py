"""Visual inspector: BTCUSDT OHLCV (1d / 4h / 1h) with a filterable astro overlay.

Loads candles from the offline Bybit cache (resampled up from 15m) and overlays
exact planetary-aspect events as vertical lines on the price panel. Each
``pair x aspect`` is its own Plotly legend entry, so you can click to toggle any
aspect family on/off and eyeball whether reversals line up with it -- that is the
"filterable astro calendar over the candles" the task asks for.

One self-contained interactive HTML file is written per timeframe (zoom / pan /
hover for the exact UTC time + orb), plus a CSV of every event mapped to its bar.

Examples
--------
# Default: last 2 years, curated slow-pair hard aspects, all three timeframes.
python inspect_btc_astro.py

# Just the Jupiter-Saturn + Saturn-Uranus cycle, daily only, opening the chart.
python inspect_btc_astro.py --timeframes 1d --pairs jupiter-saturn,saturn-uranus --open

# Every aspect family for a custom pair set on the 4h chart.
python inspect_btc_astro.py --timeframes 4h --pairs sun-saturn,mars-uranus --aspects major

Outputs (reports/): btc_astro_<tf>.html, btc_astro_events_<tf>.csv
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from itertools import combinations
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PKG_SRC = Path(__file__).resolve().parent / "src"
if str(PKG_SRC) not in sys.path:
    sys.path.insert(0, str(PKG_SRC))

from astro_reversal import data, ephemeris_events  # noqa: E402

# Curated default: slow / personal-to-outer pairs whose hard aspects are sparse
# enough to spot patterns against, plus the Moon-Pluto "Dark Pivot". Moon pairs
# fire often (~monthly), so they crowd the chart -- double-click their legend
# entries to isolate, or drop them with an explicit --pairs list.
DEFAULT_PAIRS = [
    ("jupiter", "saturn"),
    ("saturn", "uranus"),
    ("saturn", "pluto"),
    ("jupiter", "uranus"),
    ("mars", "saturn"),
    ("mars", "uranus"),
    ("sun", "saturn"),
    ("sun", "uranus"),
    ("moon", "pluto"),
]

# Stable color per pair (legend grouped by pair); aspect distinguished by dash.
PAIR_COLORS = [
    "#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
    "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff", "#9A6324",
    "#808000", "#000075", "#a9a9a9", "#ffe119",
]
ASPECT_DASH = {0: "solid", 90: "dash", 180: "dot", 270: "dashdot"}


def parse_pairs(spec: str, bodies: list[str]) -> list[tuple[str, str]]:
    """Parse a --pairs spec into (body_1, body_2) tuples.

    "all" -> every unique pair of the configured bodies; otherwise a comma list of
    "body_a-body_b" tokens. Bodies are validated against the ephemeris config.
    """
    valid = set(bodies)
    if spec.strip().lower() == "all":
        return list(combinations(bodies, 2))
    out: list[tuple[str, str]] = []
    for token in spec.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if "-" not in token:
            raise SystemExit(f"Bad pair '{token}': expected 'body_a-body_b'.")
        a, b = (part.strip() for part in token.split("-", 1))
        for body in (a, b):
            if body not in valid:
                raise SystemExit(f"Unknown body '{body}'. Valid: {', '.join(sorted(valid))}")
        out.append((a, b))
    return out


def parse_aspects(spec: str, cfg: dict) -> list[float]:
    """Parse --aspects: a named group from the config (hard/major/...) or a list."""
    groups = cfg.get("aspects", {})
    key = spec.strip().lower()
    if key in groups:
        return [float(a) for a in groups[key]]
    try:
        return [float(a.strip()) for a in spec.split(",") if a.strip()]
    except ValueError:
        raise SystemExit(
            f"Bad --aspects '{spec}'. Use a group ({', '.join(groups)}) or angles like 0,90,180."
        )


def default_window() -> tuple[str, str]:
    """Last 2 years, ending today (UTC)."""
    end = pd.Timestamp.now("UTC").normalize()
    start = end - pd.DateOffset(years=2)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def build_events(
    pairs: list[tuple[str, str]],
    aspects: list[float],
    start: str,
    end: str,
    frame: pd.DataFrame,
    grid_minutes: int,
) -> pd.DataFrame:
    """Compute aspect events for every pair, mapped to this timeframe's candles."""
    rows = []
    for body_1, body_2 in pairs:
        ev = ephemeris_events.compute_aspect_events(
            body_1, body_2, aspects, start, end, grid_minutes=grid_minutes
        )
        if ev.empty:
            continue
        ev = ephemeris_events.map_events_to_candles(ev, frame)
        ev["pair"] = f"{body_1}-{body_2}"
        rows.append(ev)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    # Keep only events that fall inside the visible candle span.
    return out[out["bar_index"] >= 0].sort_values("timestamp_utc").reset_index(drop=True)


def make_figure(
    symbol: str,
    timeframe: str,
    frame: pd.DataFrame,
    events: pd.DataFrame,
    pairs: list[tuple[str, str]],
) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.78, 0.22],
        subplot_titles=(f"{symbol} {timeframe} -- candles + astro aspect overlay", "Volume"),
    )

    x = pd.to_datetime(frame["open_time"], utc=True)
    fig.add_trace(
        go.Candlestick(
            x=x, open=frame["open"], high=frame["high"],
            low=frame["low"], close=frame["close"], name="price",
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
            showlegend=False,
        ),
        row=1, col=1,
    )
    vol_color = ["#26a69a" if c >= o else "#ef5350"
                 for o, c in zip(frame["open"], frame["close"])]
    fig.add_trace(
        go.Bar(x=x, y=frame["volume"], marker_color=vol_color, name="volume", showlegend=False),
        row=2, col=1,
    )

    # Vertical aspect lines span the price panel; one trace per (pair, aspect) so
    # each is independently toggleable from the legend.
    y_lo = float(frame["low"].min())
    y_hi = float(frame["high"].max())
    pad = (y_hi - y_lo) * 0.02
    y_lo, y_hi = y_lo - pad, y_hi + pad

    pair_color = {f"{a}-{b}": PAIR_COLORS[i % len(PAIR_COLORS)] for i, (a, b) in enumerate(pairs)}

    if not events.empty:
        grouped = events.groupby(["pair", "aspect_angle", "aspect_name"], sort=False)
        for (pair, angle, aname), grp in grouped:
            xs: list = []
            ys: list = []
            custom: list = []
            for ts in grp["timestamp_utc"]:
                xs += [ts, ts, None]
                ys += [y_lo, y_hi, None]
                custom += [ts, ts, None]
            fig.add_trace(
                go.Scatter(
                    x=xs, y=ys, mode="lines",
                    line=dict(color=pair_color.get(pair, "#888"), width=1,
                              dash=ASPECT_DASH.get(int(angle), "solid")),
                    opacity=0.55,
                    name=f"{pair} {aname} ({int(angle)}°)",
                    legendgroup=pair,
                    customdata=custom,
                    hovertemplate=f"{pair}<br>{aname} {int(angle)}°<br>%{{customdata|%Y-%m-%d %H:%M}} UTC<extra></extra>",
                ),
                row=1, col=1,
            )

    fig.update_layout(
        template="plotly_dark",
        height=820,
        margin=dict(l=50, r=30, t=60, b=40),
        legend=dict(title="aspects (click to filter)", groupclick="toggleitem",
                    font=dict(size=10), orientation="v"),
        hovermode="x unified",
    )
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_yaxes(title_text="price", row=1, col=1)
    fig.update_yaxes(title_text="vol", row=2, col=1)
    return fig


def main() -> int:
    p = argparse.ArgumentParser(description="BTCUSDT multi-timeframe OHLCV + filterable astro overlay.")
    p.add_argument("--symbol", default=None, help="Ticker (default from config: BTCUSDT).")
    p.add_argument("--timeframes", default="1d,4h,1h", help="Comma list, e.g. '1d,4h,1h'.")
    p.add_argument("--start", default=None, help="ISO date. Default: 2 years ago.")
    p.add_argument("--end", default=None, help="ISO date. Default: today (UTC).")
    p.add_argument("--pairs", default=None,
                   help="Comma list of 'body_a-body_b', or 'all'. Default: curated slow pairs.")
    p.add_argument("--aspects", default="hard",
                   help="Group (hard/major/gann_extended/discovery) or angles like '0,90,180,270'.")
    p.add_argument("--grid-minutes", type=int, default=60,
                   help="Ephemeris sampling step for event root-finding (default 60).")
    p.add_argument("--outdir", type=Path, default=None, help="Output dir (default: reports/).")
    p.add_argument("--open", action="store_true", help="Open each chart in the browser when done.")
    args = p.parse_args()

    cfg = data.load_config()
    symbol = args.symbol or cfg["data"]["symbol"]
    base_interval = cfg["data"]["base_interval"]
    bodies = list(cfg["bodies"])

    win_start, win_end = default_window()
    start = args.start or win_start
    end = args.end or win_end

    pairs = DEFAULT_PAIRS if args.pairs is None else parse_pairs(args.pairs, bodies)
    aspects = parse_aspects(args.aspects, cfg)
    timeframes = [tf.strip() for tf in args.timeframes.split(",") if tf.strip()]
    outdir = args.outdir or data.REPORTS_DIR
    outdir.mkdir(parents=True, exist_ok=True)

    pair_str = ", ".join(f"{a}-{b}" for a, b in pairs)
    print(f"Symbol      : {symbol}")
    print(f"Window      : {start} -> {end}")
    print(f"Timeframes  : {', '.join(timeframes)}")
    print(f"Pairs ({len(pairs):>2}) : {pair_str}")
    print(f"Aspects     : {[int(a) for a in aspects]}")
    print("-" * 64)

    for tf in timeframes:
        frame = data.load_ohlcv(symbol, tf, start, end, base_interval=base_interval)
        if frame.empty:
            print(f"[{tf}] no candle data in range -- skipped.")
            continue
        events = build_events(pairs, aspects, start, end, frame, args.grid_minutes)

        first = pd.Timestamp(frame["open_time"].iloc[0]).strftime("%Y-%m-%d")
        last = pd.Timestamp(frame["close_time"].iloc[-1]).strftime("%Y-%m-%d")
        n_ev = 0 if events.empty else len(events)
        print(f"[{tf}] {len(frame):>6} candles ({first} -> {last}), {n_ev:>4} aspect events")
        if not events.empty:
            counts = events.groupby("pair").size().sort_values(ascending=False)
            for pair, cnt in counts.items():
                print(f"        {pair:<18} {cnt:>4}")

        fig = make_figure(symbol, tf, frame, events, pairs)
        html_path = outdir / f"btc_astro_{tf}.html"
        fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)

        if not events.empty:
            csv_cols = ["timestamp_utc", "pair", "aspect_angle", "aspect_name",
                        "orb_resid_deg", "bar_index", "candle_open_time"]
            events[csv_cols].to_csv(outdir / f"btc_astro_events_{tf}.csv", index=False)

        print(f"        -> {html_path}")
        if args.open:
            webbrowser.open(html_path.resolve().as_uri())

    print("-" * 64)
    print("Tip: click legend entries to toggle each pair/aspect; double-click to isolate one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
