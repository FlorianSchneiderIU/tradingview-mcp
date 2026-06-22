#!/usr/bin/env python3
"""Heatmap Telegram bot — on-demand liquidity / liquidation heatmap images.

Long-polls Telegram. When a WHITELISTED user asks for a universe coin, it fetches the
requested layer from heatmap-bot's REST API, renders it to a PNG, and replies via sendPhoto.

Three layers (pick with a keyword), all for our universe coins only:
    liquidity  — order-book resting liquidity heatmap         (/v1/liquidity)
    actual     — real liquidation prints (ground truth)        (/v1/liquidations/actual)
    estimated  — predictive liquidation heatmap (default)      (/v1/liquidations/estimated?tf=)

Commands:
    /heatmap SYM [layer] [tf]   e.g. "/heatmap BTC", "/heatmap ETH liquidity", "/heatmap SOL estimated 15m"
    SYM [layer] [tf]            shorthand
    /coins                      list tracked universe
    /help

Access control: only HEATMAP_TG_ALLOWED_UIDS are served. Talks only to Telegram + heatmap-bot.
"""
from __future__ import annotations

import io
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import requests  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)sZ [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%dT%H:%M:%S")
log = logging.getLogger("heatmap-tg")


def _csv_ints(name: str) -> set[int]:
    out: set[int] = set()
    for tok in os.environ.get(name, "").split(","):
        tok = tok.strip()
        if tok:
            try:
                out.add(int(tok))
            except ValueError:
                log.warning("ignoring non-integer uid %r in %s", tok, name)
    return out


TG_TOKEN = os.environ.get("HEATMAP_TG_BOT_TOKEN", os.environ.get("TELEGRAM_BOT_TOKEN", "")).strip()
ALLOWED_UIDS = _csv_ints("HEATMAP_TG_ALLOWED_UIDS")
API_URL = os.environ.get("HEATMAP_API_URL", "http://heatmap-bot:8110").rstrip("/")

DEFAULT_TF = os.environ.get("HEATMAP_TG_DEFAULT_TF", "1h").strip()
DEFAULT_LAYER = os.environ.get("HEATMAP_TG_DEFAULT_LAYER", "estimated").strip()
SERIES_LIMIT = int(os.environ.get("HEATMAP_TG_SERIES_LIMIT", "400"))
GRID_ROWS = int(os.environ.get("HEATMAP_TG_GRID_ROWS", "160"))
GRID_COLS = int(os.environ.get("HEATMAP_TG_GRID_COLS", "240"))
POLL_TIMEOUT = int(os.environ.get("HEATMAP_TG_POLL_TIMEOUT", "30"))
UNIVERSE_TTL = int(os.environ.get("HEATMAP_TG_UNIVERSE_TTL", "300"))
HTTP_TIMEOUT = float(os.environ.get("HEATMAP_TG_HTTP_TIMEOUT", "30"))

_TG_API = f"https://api.telegram.org/bot{TG_TOKEN}"
TF_TOKENS = {"5m", "15m", "1h"}
VP_WINDOW_TOKENS = {"4h", "24h", "7d", "daily", "weekly"}
DEFAULT_VP_WINDOW = os.environ.get("HEATMAP_TG_DEFAULT_WINDOW", "24h").strip()
LAYER_ALIASES = {
    "liquidity": "liquidity", "liq": "liquidity", "ob": "liquidity", "orderbook": "liquidity",
    "book": "liquidity", "depth": "liquidity",
    "actual": "actual", "real": "actual", "prints": "actual", "history": "actual", "hist": "actual",
    "estimated": "estimated", "est": "estimated", "pred": "estimated", "predicted": "estimated",
    "predictive": "estimated", "liquidation": "estimated", "liquidations": "estimated", "heatmap": "estimated",
    "volume": "volume", "vp": "volume", "profile": "volume", "vpvr": "volume", "delta": "volume",
}
# Dedicated slash commands -> layer (clearer than overloading /heatmap)
CMD_TO_LAYER = {
    "liquidations": "estimated", "liq": "estimated", "liqs": "estimated", "liquidation": "estimated",
    "magnets": "estimated",
    "actual": "actual", "prints": "actual", "reals": "actual", "realliq": "actual",
    "liquidity": "liquidity", "book": "liquidity", "ob": "liquidity", "depth": "liquidity",
    "volume": "volume", "vp": "volume", "footprint": "volume", "profile": "volume",
}

# ── universe cache ──────────────────────────────────────────────────────────────
_universe: list[str] = []
_universe_at = 0.0


def get_universe() -> list[str]:
    global _universe, _universe_at
    if _universe and (time.time() - _universe_at) < UNIVERSE_TTL:
        return _universe
    try:
        resp = requests.get(f"{API_URL}/v1/universe", timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        syms = [r["symbol"] for r in resp.json().get("universe", [])]
        if syms:
            _universe, _universe_at = syms, time.time()
    except Exception as exc:  # noqa: BLE001
        log.warning("universe fetch failed: %s", exc)
    return _universe


def resolve_symbol(raw: str) -> Optional[str]:
    sym = raw.strip().upper().replace("/", "").replace(":", "")
    uni = get_universe()
    if sym in uni:
        return sym
    if not sym.endswith("USDT") and f"{sym}USDT" in uni:
        return f"{sym}USDT"
    return None


TF_INTERVAL = {"5m": "5", "15m": "15", "1h": "60"}


def fetch_ohlc(symbol: str, interval: str, start_ms: int) -> list:
    """OHLC candles [[ts,o,h,l,c],...] for a price overlay (best-effort)."""
    try:
        r = requests.get(f"{API_URL}/v1/ohlc/{symbol}", params={"interval": interval, "start": int(start_ms)},
                         timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        return r.json().get("ohlc", []) or []
    except Exception as exc:  # noqa: BLE001
        log.warning("ohlc fetch failed %s: %s", symbol, exc)
        return []


def _interval_for_span(span_ms: int) -> str:
    h = span_ms / 3_600_000
    if h <= 6:
        return "5"
    if h <= 24:
        return "15"
    if h <= 24 * 7:
        return "60"
    return "240"


# ── Telegram I/O ────────────────────────────────────────────────────────────────
def tg_send_message(chat_id: int, text: str, reply_to: Optional[int] = None) -> None:
    payload: dict[str, Any] = {"chat_id": chat_id, "text": text, "disable_web_page_preview": True}
    if reply_to is not None:
        payload.update(reply_to_message_id=reply_to, allow_sending_without_reply=True)
    try:
        requests.post(f"{_TG_API}/sendMessage", json=payload, timeout=HTTP_TIMEOUT)
    except Exception as exc:  # noqa: BLE001
        log.warning("sendMessage failed: %s", exc)


def tg_send_photo(chat_id: int, png: bytes, caption: str, reply_to: Optional[int] = None) -> None:
    data: dict[str, Any] = {"chat_id": chat_id, "caption": caption}
    if reply_to is not None:
        data.update(reply_to_message_id=reply_to, allow_sending_without_reply=True)
    try:
        resp = requests.post(f"{_TG_API}/sendPhoto", data=data,
                             files={"photo": ("heatmap.png", png, "image/png")}, timeout=HTTP_TIMEOUT * 2)
        if not resp.ok:
            log.warning("sendPhoto failed: %s %s", resp.status_code, resp.text[:200])
    except Exception as exc:  # noqa: BLE001
        log.warning("sendPhoto error: %s", exc)


def tg_send_action(chat_id: int) -> None:
    try:
        requests.post(f"{_TG_API}/sendChatAction", json={"chat_id": chat_id, "action": "upload_photo"},
                      timeout=HTTP_TIMEOUT)
    except Exception:  # noqa: BLE001
        pass


# ── rendering helpers ────────────────────────────────────────────────────────────
def _new_fig(title: str):
    fig, ax = plt.subplots(figsize=(11, 6), dpi=110)
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.set_title(title, color="#eee", fontsize=12)
    return fig, ax


def _style(ax) -> None:
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M", tz=timezone.utc))
    for sp in ax.spines.values():
        sp.set_color("#444")
    ax.tick_params(colors="#bbb", labelsize=8)
    ax.set_ylabel("Price", color="#ddd")


def _finish(fig, ax, im, cbar_label: str) -> bytes:
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label(cbar_label, color="#ccc")
    plt.setp(cbar.ax.get_yticklabels(), color="#bbb", fontsize=7)
    fig.autofmt_xdate(rotation=30)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _dt(ms_list) -> list[datetime]:
    return [datetime.fromtimestamp(t / 1000.0, tz=timezone.utc) for t in ms_list]


def _fmt_ts(ms) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime("%m-%d %H:%M")


def _draw_candles(ax, ohlc, pspan):
    """Overlay OHLC candlesticks (cyan up / grey down) on a date×price axis.
    Returns (xmin, xmax) in date-num coords, or (None, None) if nothing drawn."""
    if not ohlc:
        return None, None
    xs = mdates.date2num(_dt([o[0] for o in ohlc]))
    w = 0.6 * float(np.median(np.diff(xs))) if len(xs) > 1 else 0.0006
    body_floor = max(pspan * 1e-4, 1e-9)
    for x, (_t, o, hi, lo, c) in zip(xs, ohlc):
        col = "#58e0ff" if c >= o else "#9aa4ad"
        ax.vlines(x, lo, hi, color=col, lw=0.5, alpha=0.95)
        ax.add_patch(Rectangle((x - w / 2, min(o, c)), w, max(abs(c - o), body_floor),
                               facecolor=col, edgecolor=col, lw=0.3, alpha=0.95))
    return float(xs[0]), float(xs[-1])


def render_liquidity(symbol: str, snaps: list[dict[str, Any]]) -> Optional[bytes]:
    snaps = [s for s in snaps if s.get("bins")]
    if not snaps:
        return None
    times = [s["ts"] for s in snaps]
    mids = np.array([s.get("mid") or np.nan for s in snaps], dtype=float)
    lows, highs = [], []
    for s in snaps:
        for b in s["bins"]:
            lows.append(b["price_low"]); highs.append(b["price_high"])
    pmin, pmax = min(lows), max(highs)
    if not (pmax > pmin):
        return None
    rows = GRID_ROWS
    grid = np.zeros((rows, len(snaps)))
    for t, s in enumerate(snaps):
        for b in s["bins"]:
            c = 0.5 * (b["price_low"] + b["price_high"])
            r = min(max(int((c - pmin) / (pmax - pmin) * rows), 0), rows - 1)
            grid[r, t] += (b.get("bid_notional") or 0.0) + (b.get("ask_notional") or 0.0)
    masked = np.ma.masked_less_equal(grid, 0.0)
    if masked.count() == 0:
        return None
    vmax = float(masked.max()); vmin = max(float(masked.min()), vmax / 1e4)
    tnum = mdates.date2num(_dt(times))
    fig, ax = _new_fig(f"{symbol}  order-book liquidity   {len(snaps)} snapshots (UTC)")
    cmap = plt.get_cmap("inferno").copy(); cmap.set_bad("#0e1117")
    im = ax.imshow(masked, origin="lower", aspect="auto", extent=[tnum[0], tnum[-1], pmin, pmax],
                   cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax), interpolation="nearest")
    ax.plot(tnum, mids, color="#39c5cf", lw=1.0, alpha=0.9)
    _style(ax)
    return _finish(fig, ax, im, "resting notional (USDT)")


def render_estimated(symbol: str, tf: str, data: dict[str, Any]) -> Optional[bytes]:
    levels = data.get("levels") or []
    series = data.get("price_series") or []
    if not levels or len(series) < 2:
        return None
    times = np.array([t for t, _ in series], dtype=np.int64)
    closes = np.array([c for _, c in series], dtype=float)
    lows = [l["price_low"] for l in levels]; highs = [l["price_high"] for l in levels]
    pmin = min(min(lows), float(closes.min())); pmax = max(max(highs), float(closes.max()))
    if not (pmax > pmin):
        return None
    rows, cols = GRID_ROWS, len(times)
    grid = np.zeros((rows, cols))
    for l in levels:
        c0 = int(np.searchsorted(times, max(int(l["created_ts"]), int(times[0])), "left"))
        end_ts = l["consumed_ts"] if l["consumed_ts"] is not None else int(times[-1])
        c1 = int(np.searchsorted(times, int(end_ts), "right"))
        if c1 <= c0:
            c1 = min(c0 + 1, cols)
        r = min(max(int((0.5 * (l["price_low"] + l["price_high"]) - pmin) / (pmax - pmin) * rows), 0), rows - 1)
        grid[r, c0:c1] += l["magnitude"]
    masked = np.ma.masked_less_equal(grid, 0.0)
    if masked.count() == 0:
        return None
    vmax = float(masked.max()); vmin = max(float(masked.min()), vmax / 1e4)
    tnum = mdates.date2num(_dt(times.tolist()))
    fig, ax = _new_fig(f"{symbol}  predictive liquidation heatmap   tf={tf}   (UTC)")
    cmap = plt.get_cmap("inferno").copy(); cmap.set_bad("#0e1117")
    im = ax.imshow(masked, origin="lower", aspect="auto", extent=[tnum[0], tnum[-1], pmin, pmax],
                   cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax), interpolation="nearest")
    # OHLC candles (mark price); fall back to a close line if unavailable
    ohlc = fetch_ohlc(symbol, TF_INTERVAL.get(tf, "60"), int(data.get("window_start") or times[0]))
    cx0, cx1 = _draw_candles(ax, ohlc, pmax - pmin)
    if cx0 is None:
        ax.plot(tnum, closes, color="#58e0ff", lw=1.0, alpha=0.9)
    else:
        ax.set_xlim(min(tnum[0], cx0), max(tnum[-1], cx1))
    ax.set_ylim(pmin, pmax)
    _style(ax)
    return _finish(fig, ax, im, "liquidation notional at risk (USDT)")


def render_actual(symbol: str, events: list[dict[str, Any]]) -> Optional[bytes]:
    if len(events) < 2:
        return None
    ts = np.array([e["ts"] for e in events], dtype=np.int64)
    price = np.array([e["price"] for e in events], dtype=float)
    notional = np.array([e.get("notional") or 0.0 for e in events], dtype=float)
    pmin, pmax = float(price.min()), float(price.max())
    tmin, tmax = int(ts.min()), int(ts.max())
    if not (pmax > pmin) or not (tmax > tmin):
        return None
    pedges = np.linspace(pmin, pmax, GRID_ROWS + 1)
    tedges = np.linspace(tmin, tmax, min(GRID_COLS, max(2, len(events))) + 1)
    H, _, _ = np.histogram2d(price, ts, bins=[pedges, tedges], weights=notional)
    masked = np.ma.masked_less_equal(H, 0.0)
    if masked.count() == 0:
        return None
    vmax = float(masked.max()); vmin = max(float(masked.min()), vmax / 1e4)
    tnum0, tnum1 = mdates.date2num(_dt([tmin, tmax]))
    fig, ax = _new_fig(f"{symbol}  actual liquidations   {len(events)} prints (UTC)")
    cmap = plt.get_cmap("inferno").copy(); cmap.set_bad("#0e1117")
    im = ax.imshow(masked, origin="lower", aspect="auto", extent=[tnum0, tnum1, pmin, pmax],
                   cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax), interpolation="nearest")
    # OHLC candles so prints read against price
    ohlc = fetch_ohlc(symbol, _interval_for_span(tmax - tmin), tmin)
    cx0, cx1 = _draw_candles(ax, ohlc, pmax - pmin)
    if ohlc:
        ylo = min(pmin, min(o[3] for o in ohlc))
        yhi = max(pmax, max(o[2] for o in ohlc))
        ax.set_ylim(ylo, yhi)
        ax.set_xlim(min(tnum0, cx0), max(tnum1, cx1))
    _style(ax)
    return _finish(fig, ax, im, "liquidated notional (USDT)")


def render_volume(symbol: str, window: str, prof: dict[str, Any], heat: dict[str, Any]) -> Optional[bytes]:
    """Footprint view sharing the price axis:
      left  — time x price VOLUME heatmap (log USDT notional) with OHLC candles overlaid;
      right — volume-at-price profile (bar length = total USDT volume) colored by the REAL
              taker imbalance (green=net buy, red=net sell, grey=balanced/seed-only),
              with POC / VAH / VAL marked.
    Bar length is honest total volume; colour only shows imbalance where it was measured,
    so kline-seeded (delta-unknown) volume reads grey rather than fake-symmetric red/green."""
    from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap  # local import keeps top tidy

    bins = prof.get("bins") or []
    if not bins:
        return None
    lows = [b["price_low"] for b in bins]
    highs = [b["price_high"] for b in bins]
    pmin, pmax = min(lows), max(highs)
    if not (pmax > pmin):
        return None
    rows = GRID_ROWS

    def prow(p):
        return min(max(int((p - pmin) / (pmax - pmin) * rows), 0), rows - 1)

    # right panel: total volume + measured imbalance, aggregated per price row
    tot = np.zeros(rows)
    dlt = np.zeros(rows)
    for b in bins:
        r = prow(0.5 * (b["price_low"] + b["price_high"]))
        tot[r] += b.get("total") or 0.0
        dlt[r] += b.get("delta") or 0.0

    cells = (heat or {}).get("cells") or []
    ohlc = (heat or {}).get("ohlc") or []
    fig = plt.figure(figsize=(12, 6), dpi=110)
    fig.patch.set_facecolor("#0e1117")
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.02)
    ax_hm = fig.add_subplot(gs[0, 0])
    ax_vp = fig.add_subplot(gs[0, 1], sharey=ax_hm)
    for ax in (ax_hm, ax_vp):
        ax.set_facecolor("#0e1117")

    # left: time x price VOLUME footprint heatmap (log notional) — populated by seed + live
    im = None
    xmin = xmax = None
    if cells:
        hours = sorted({c["ts"] for c in cells})
        h0, h1 = hours[0], hours[-1]
        cols = list(range(h0, h1 + 3_600_000, 3_600_000)) or [h0]
        cidx = {t: i for i, t in enumerate(cols)}
        grid = np.zeros((rows, len(cols)))
        for c in cells:
            r = prow(0.5 * (c["price_low"] + c["price_high"]))
            grid[r, cidx.get(c["ts"], 0)] += c.get("total") or 0.0
        masked = np.ma.masked_less_equal(grid, 0.0)
        if masked.count():
            vmax = float(masked.max()); vmin = max(float(masked.min()), vmax / 1e4)
            x0 = mdates.date2num(_dt(cols))[0]
            x1 = mdates.date2num(_dt([h1 + 3_600_000]))[0]
            cmap = plt.get_cmap("inferno").copy(); cmap.set_bad("#0e1117")
            im = ax_hm.imshow(masked, origin="lower", aspect="auto", extent=[x0, x1, pmin, pmax],
                              cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax), interpolation="nearest", alpha=0.9)
            xmin, xmax = x0, x1

    # OHLC candlesticks over the warm volume heatmap
    cx0, cx1 = _draw_candles(ax_hm, ohlc, pmax - pmin)
    if cx0 is not None:
        xmin = min(xmin, cx0) if xmin is not None else cx0
        xmax = max(xmax, cx1) if xmax is not None else cx1

    if xmin is not None:
        ax_hm.set_xlim(xmin, xmax)
    ax_hm.set_ylim(pmin, pmax)
    ax_hm.xaxis_date()
    ax_hm.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M", tz=timezone.utc))
    ax_hm.set_title(f"{symbol}  price + volume footprint ({window})  (UTC)", color="#eee", fontsize=11)
    ax_hm.set_ylabel("Price", color="#ddd")
    ax_hm.tick_params(colors="#bbb", labelsize=8)
    for sp in ax_hm.spines.values():
        sp.set_color("#444")

    # right: total-volume bars coloured by measured imbalance delta/total in [-1, 1]
    yc = np.array([pmin + (i + 0.5) * (pmax - pmin) / rows for i in range(rows)])
    h = (pmax - pmin) / rows
    imb = np.divide(dlt, tot, out=np.zeros_like(dlt), where=tot > 0)
    # gray-centred diverging map: balanced/seed-only -> grey (recedes), net buy -> green,
    # net sell -> red; clamped to +/-0.4 so realistic imbalances are clearly visible.
    cmap_imb = LinearSegmentedColormap.from_list("imb", ["#c01c28", "#4b4f55", "#26a269"])
    norm_imb = TwoSlopeNorm(vmin=-0.4, vcenter=0.0, vmax=0.4)
    colors = [cmap_imb(norm_imb(float(np.clip(v, -0.4, 0.4)))) if t > 0 else (0.3, 0.3, 0.3, 1.0)
              for v, t in zip(imb, tot)]
    ax_vp.barh(yc, tot, height=h, color=colors)
    poc, vah, val = prof.get("poc"), prof.get("vah"), prof.get("val")
    if poc:
        ax_vp.axhline(poc, color="#f6d32d", lw=1.2)
        ax_hm.axhline(poc, color="#f6d32d", lw=0.9, alpha=0.85)
    for lvl_p in (vah, val):
        if lvl_p:
            ax_vp.axhline(lvl_p, color="#9aa4ad", lw=0.7, ls="--", alpha=0.8)
    if vah and val:
        ax_vp.axhspan(val, vah, color="#ffffff", alpha=0.05)
    ax_vp.set_title("volume@price\n(colour = buy/sell imbalance)", color="#ccc", fontsize=8)
    ax_vp.set_xlabel("USDT notional", color="#999", fontsize=7)
    ax_vp.tick_params(axis="y", labelleft=False)
    ax_vp.tick_params(axis="x", colors="#888", labelsize=6)
    for sp in ax_vp.spines.values():
        sp.set_color("#444")

    fig.autofmt_xdate(rotation=30)
    if im is not None:
        cbar = fig.colorbar(im, ax=ax_vp, pad=0.28, fraction=0.07)
        cbar.set_label("volume (USDT notional, log)", color="#ccc", fontsize=8)
        plt.setp(cbar.ax.get_yticklabels(), color="#bbb", fontsize=6)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def render_cvd(symbol: str, window: str, data: dict[str, Any]) -> Optional[bytes]:
    """Price (candles) above, cumulative volume delta (filled green/red) below, with a
    simple price/CVD divergence flag in the title."""
    series = data.get("series") or []
    if len(series) < 3:
        return None
    ts = [s["ts"] for s in series]
    cvd = np.array([s["cvd"] for s in series], dtype=float)
    tnum = mdates.date2num(_dt(ts))
    ohlc = data.get("ohlc") or []

    fig = plt.figure(figsize=(11, 6), dpi=110)
    fig.patch.set_facecolor("#0e1117")
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.06)
    axp = fig.add_subplot(gs[0])
    axc = fig.add_subplot(gs[1], sharex=axp)
    for ax in (axp, axc):
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="#bbb", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#444")

    if ohlc:
        pmin = min(o[3] for o in ohlc); pmax = max(o[2] for o in ohlc)
        _draw_candles(axp, ohlc, pmax - pmin)
        axp.set_ylim(pmin, pmax)
    else:
        axp.plot(tnum, [s["close"] for s in series], color="#58e0ff", lw=1.0)

    axc.plot(tnum, cvd, color="#e6edf3", lw=1.0)
    axc.fill_between(tnum, cvd, 0, where=cvd >= 0, color="#26a269", alpha=0.4, interpolate=True)
    axc.fill_between(tnum, cvd, 0, where=cvd < 0, color="#c01c28", alpha=0.4, interpolate=True)
    axc.axhline(0, color="#666", lw=0.6)

    valid = [(i, s["close"]) for i, s in enumerate(series) if s["close"]]
    div = ""
    if len(valid) >= 4:
        mid, last = valid[len(valid) // 2], valid[-1]
        if last[1] > mid[1] and cvd[last[0]] < cvd[mid[0]]:
            div = "  ⚠ bearish divergence (price↑ CVD↓)"
        elif last[1] < mid[1] and cvd[last[0]] > cvd[mid[0]]:
            div = "  ⚠ bullish divergence (price↓ CVD↑)"

    axp.set_title(f"{symbol}  price + CVD ({window}){div}  (UTC)", color="#eee", fontsize=11)
    axp.set_ylabel("Price", color="#ddd")
    axc.set_ylabel("CVD", color="#ddd")
    plt.setp(axp.get_xticklabels(), visible=False)
    axc.xaxis_date()
    axc.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M", tz=timezone.utc))
    fig.autofmt_xdate(rotation=30)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ── command handling ─────────────────────────────────────────────────────────────
HELP = (
    "Bybit market-structure bot — for our universe coins.\n"
    "\n"
    "  /liquidations SYM [5m|15m|1h]   predictive liquidation heatmap (default 1h)\n"
    "  /actual SYM                     real liquidation prints\n"
    "  /liquidity SYM                  order-book resting liquidity\n"
    "  /volume SYM [4h|24h|7d|daily|weekly]   volume footprint + profile (default 24h)\n"
    "  /levels SYM [tf]                key levels across all layers (text, ⭐ = confluence)\n"
    "  /structure SYM                  market-structure snapshot (bias, S/R, skew, funding, OI)\n"
    "  /cvd SYM [window]               price + cumulative volume delta (divergence flag)\n"
    "  /screener [liq|cvd|imbalance|volume]   rank the universe\n"
    "  /score                          predictive calibration / hit-rate history\n"
    "  /watch SYM · /unwatch SYM · /watches   proximity alerts for a coin\n"
    "  /coins                          list tracked universe\n"
    "\n"
    "  Cascade alerts fire automatically for big liquidation bursts.\n"
    "  examples:  /liquidations BTC 15m   ·   /levels ETH   ·   /watch SOL   ·   /volume SOL 7d"
)


def parse_args(args: list[str]) -> tuple[Optional[str], str, str, str]:
    """-> (symbol_raw, layer, tf, window) from tokens after the command."""
    sym = args[0] if args else None
    layer, tf, window = DEFAULT_LAYER, DEFAULT_TF, DEFAULT_VP_WINDOW
    for tok in args[1:]:
        low = tok.lower()
        if low in LAYER_ALIASES:
            layer = LAYER_ALIASES[low]
        elif low in TF_TOKENS:
            tf = low
        elif low in VP_WINDOW_TOKENS:
            window = low
    return sym, layer, tf, window


def fetch_and_render(symbol: str, layer: str, tf: str, window: str):
    """-> (png_bytes_or_None, caption, error_or_None)."""
    try:
        if layer == "volume":
            pr = requests.get(f"{API_URL}/v1/volume_profile/{symbol}", params={"window": window}, timeout=HTTP_TIMEOUT)
            pr.raise_for_status()
            prof = pr.json()
            if not prof.get("success") or not prof.get("bins"):
                return None, "", "no volume-profile data yet"
            hm = requests.get(f"{API_URL}/v1/volume_profile/{symbol}/heatmap", params={"window": window},
                              timeout=HTTP_TIMEOUT)
            heat = hm.json() if hm.ok else {}
            png = render_volume(symbol, window, prof, heat)
            poc = prof.get("poc")
            cap = f"{symbol} — volume profile ({window}) — POC {poc:g}" if poc else f"{symbol} — volume profile ({window})"
            return png, cap, None
        if layer == "liquidity":
            r = requests.get(f"{API_URL}/v1/liquidity/{symbol}", params={"limit": SERIES_LIMIT}, timeout=HTTP_TIMEOUT)
            r.raise_for_status()
            snaps = r.json().get("snapshots", [])
            if not snaps:
                return None, "", "no order-book data yet"
            png = render_liquidity(symbol, snaps)
            return png, f"{symbol} — order-book liquidity — {len(snaps)} snapshots", None
        if layer == "actual":
            r = requests.get(f"{API_URL}/v1/liquidations/actual/{symbol}", params={"limit": SERIES_LIMIT * 5},
                             timeout=HTTP_TIMEOUT)
            r.raise_for_status()
            events = r.json().get("events", [])
            if len(events) < 2:
                return None, "", "not enough liquidation prints yet (stream is live; data accrues over time)"
            png = render_actual(symbol, events)
            return png, f"{symbol} — actual liquidations — {len(events)} prints", None
        # estimated
        r = requests.get(f"{API_URL}/v1/liquidations/estimated/{symbol}", params={"tf": tf}, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        if not data.get("success"):
            return None, "", data.get("error", "no data")
        png = render_estimated(symbol, tf, data)
        last = data.get("last_price")
        cap = f"{symbol} — predictive liquidation heatmap — tf={tf}" + (f" — mark {last:g}" if last else "")
        return png, cap, None
    except Exception as exc:  # noqa: BLE001
        return None, "", str(exc)


def handle_text(chat_id: int, uid: int, text: str, message_id: int) -> None:
    text = text.strip()
    if not text:
        return
    parts = text.split()
    cmd = parts[0].lower().lstrip("/").split("@", 1)[0]

    if cmd in {"start", "help"}:
        return tg_send_message(chat_id, HELP, reply_to=message_id)
    if cmd == "coins":
        uni = get_universe()
        body = f"Tracked universe ({len(uni)}):\n{', '.join(uni)}" if uni else "Universe unavailable."
        return tg_send_message(chat_id, body, reply_to=message_id)

    # text commands (no image)
    if cmd == "levels":
        sym_raw, _l, tf, window = parse_args(parts[1:])
        if not sym_raw:
            return tg_send_message(chat_id, "usage: /levels SYM [tf]", reply_to=message_id)
        symbol = resolve_symbol(sym_raw)
        if symbol is None:
            return tg_send_message(chat_id, f"'{sym_raw}' is not tracked.", reply_to=message_id)
        try:
            data = requests.get(f"{API_URL}/v1/levels/{symbol}", params={"tf": tf, "window": window},
                                timeout=HTTP_TIMEOUT).json()
        except Exception as exc:  # noqa: BLE001
            return tg_send_message(chat_id, f"levels error: {exc}", reply_to=message_id)
        price = data.get("last_price")
        out = [f"{symbol} key levels" + (f" (price {price:g})" if price else "") + ":"]
        for L in data.get("levels", [])[:10]:
            d = L.get("distance_pct")
            tag = "x%d" % L["layers"] + ("⭐" if L["layers"] >= 2 else "")
            out.append(f"  {L['price']:g}  {('%+.2f%%' % d) if d is not None else '':>8}  {tag}  "
                       f"[{','.join(L['types'])}]  s{L['score']}")
        return tg_send_message(chat_id, "\n".join(out) if len(out) > 1 else f"no levels for {symbol} yet",
                               reply_to=message_id)

    if cmd in {"watch", "unwatch"}:
        sym_raw = parts[1] if len(parts) > 1 else None
        if not sym_raw:
            return tg_send_message(chat_id, f"usage: /{cmd} SYM", reply_to=message_id)
        symbol = resolve_symbol(sym_raw)
        if symbol is None:
            return tg_send_message(chat_id, f"'{sym_raw}' is not tracked.", reply_to=message_id)
        try:
            w = requests.post(f"{API_URL}/v1/watch", json={"uid": uid, "symbol": symbol, "add": cmd == "watch"},
                              timeout=HTTP_TIMEOUT).json().get("watches", [])
        except Exception as exc:  # noqa: BLE001
            return tg_send_message(chat_id, f"watch error: {exc}", reply_to=message_id)
        verb = "👁 watching" if cmd == "watch" else "removed"
        return tg_send_message(chat_id, f"{verb} {symbol}. Watching: {', '.join(w) or '(none)'}",
                               reply_to=message_id)

    if cmd == "watches":
        try:
            w = requests.get(f"{API_URL}/v1/watches", params={"uid": uid}, timeout=HTTP_TIMEOUT).json().get("watches", [])
        except Exception as exc:  # noqa: BLE001
            return tg_send_message(chat_id, f"watches error: {exc}", reply_to=message_id)
        return tg_send_message(chat_id, f"Watching: {', '.join(w) or '(none)'}\n"
                               "Cascade alerts fire for all coins; proximity alerts only for watched ones.",
                               reply_to=message_id)

    if cmd == "structure":
        sym_raw = parts[1] if len(parts) > 1 else None
        if not sym_raw:
            return tg_send_message(chat_id, "usage: /structure SYM", reply_to=message_id)
        symbol = resolve_symbol(sym_raw)
        if symbol is None:
            return tg_send_message(chat_id, f"'{sym_raw}' is not tracked.", reply_to=message_id)
        try:
            d = requests.get(f"{API_URL}/v1/structure/{symbol}", timeout=HTTP_TIMEOUT).json()
        except Exception as exc:  # noqa: BLE001
            return tg_send_message(chat_id, f"structure error: {exc}", reply_to=message_id)
        if not d.get("success"):
            return tg_send_message(chat_id, f"no structure for {symbol} yet", reply_to=message_id)
        price = d.get("last_price")
        sk = d.get("liquidation_skew") or {}

        def fl(L):
            return f"{L['price']:g} ({L['distance_pct']:+.2f}%, {','.join(L['types'])})" if L else "—"

        oi = (d.get("open_interest_usd") or 0) / 1e6
        lines = [f"{symbol} structure" + (f"  (price {price:g})" if price else ""),
                 f"  bias: {d.get('bias')}",
                 f"  resistance: {fl(d.get('nearest_resistance'))}",
                 f"  support:    {fl(d.get('nearest_support'))}",
                 f"  liq skew: {sk.get('skew')} — {sk.get('lean')}",
                 f"  vol imbalance 24h: {d.get('volume_imbalance')}  · CVD24h: {d.get('cvd_24h')}",
                 f"  funding: {d.get('funding_rate')}  · OI: ${oi:.0f}M"]
        return tg_send_message(chat_id, "\n".join(lines), reply_to=message_id)

    if cmd == "screener":
        metric = parts[1].lower() if len(parts) > 1 else "liq"
        try:
            d = requests.get(f"{API_URL}/v1/screener", params={"metric": metric, "n": 12}, timeout=HTTP_TIMEOUT).json()
        except Exception as exc:  # noqa: BLE001
            return tg_send_message(chat_id, f"screener error: {exc}", reply_to=message_id)
        res = d.get("results", [])
        lines = [f"Screener — top by {d.get('metric')} (liq=1h liquidations, imb/cvd=24h):"]
        for r in res[:12]:
            lines.append(f"  {r['symbol']:<11} liq ${r['liq_1h']/1e6:>5.1f}M  imb {r['vol_imbalance']:+.2f}  "
                         f"cvd ${r['cvd_24h']/1e6:+.1f}M")
        return tg_send_message(chat_id, "\n".join(lines) if res else "no screener data yet", reply_to=message_id)

    if cmd == "score":
        try:
            d = requests.get(f"{API_URL}/v1/calibration", timeout=HTTP_TIMEOUT).json()
        except Exception as exc:  # noqa: BLE001
            return tg_send_message(chat_id, f"score error: {exc}", reply_to=message_id)
        hist = d.get("history", [])
        lines = [f"Predictive calibration ({d.get('method')}, concentration {d.get('concentration')}):"]
        if not hist:
            lines.append("  no runs yet — accrues daily as liquidation prints build up")
        for h in hist[:8]:
            lines.append(f"  {_fmt_ts(h['ts'])}  hit_rate {h.get('hit_rate')}  capture {h.get('explained_frac')}  "
                         f"n={h.get('n_events')}  applied={h.get('applied')}")
        lev = d.get("leverages", []); cw = d.get("current_weights", [])
        lines.append("  weights: " + ", ".join(f"{int(l)}x:{w}" for l, w in zip(lev, cw)))
        return tg_send_message(chat_id, "\n".join(lines), reply_to=message_id)

    if cmd == "cvd":
        sym_raw, _l, _tf, window = parse_args(parts[1:])
        if not sym_raw:
            return tg_send_message(chat_id, "usage: /cvd SYM [window]", reply_to=message_id)
        symbol = resolve_symbol(sym_raw)
        if symbol is None:
            return tg_send_message(chat_id, f"'{sym_raw}' is not tracked.", reply_to=message_id)
        tg_send_action(chat_id)
        try:
            d = requests.get(f"{API_URL}/v1/cvd/{symbol}", params={"window": window}, timeout=HTTP_TIMEOUT).json()
        except Exception as exc:  # noqa: BLE001
            return tg_send_message(chat_id, f"cvd error: {exc}", reply_to=message_id)
        png = render_cvd(symbol, window, d) if d.get("success") else None
        if png is None:
            return tg_send_message(chat_id, f"not enough CVD data for {symbol} yet", reply_to=message_id)
        tg_send_photo(chat_id, png, f"{symbol} — CVD ({window})", reply_to=message_id)
        log.info("served %s/cvd window=%s to uid=%s", symbol, window, uid)
        return

    if cmd in CMD_TO_LAYER:                       # dedicated per-layer command
        sym_raw, _lyr, tf, window = parse_args(parts[1:])
        layer = CMD_TO_LAYER[cmd]
    elif cmd == "heatmap":                         # back-compat: layer keyword in args
        sym_raw, layer, tf, window = parse_args(parts[1:])
    else:                                          # bare "SYM [layer] [tf|window]"
        sym_raw, layer, tf, window = parse_args(parts)
    if not sym_raw:
        return tg_send_message(chat_id, HELP, reply_to=message_id)

    symbol = resolve_symbol(sym_raw)
    if symbol is None:
        uni = get_universe()
        return tg_send_message(chat_id, f"'{sym_raw}' is not a tracked universe coin.\nTracked: {', '.join(uni) or '(unavailable)'}",
                               reply_to=message_id)

    tg_send_action(chat_id)
    png, caption, err = fetch_and_render(symbol, layer, tf, window)
    if png is None:
        return tg_send_message(chat_id, f"Could not render {symbol} ({layer}): {err}", reply_to=message_id)
    tg_send_photo(chat_id, png, caption, reply_to=message_id)
    log.info("served %s/%s (tf=%s window=%s) to uid=%s", symbol, layer, tf, window, uid)


def process_update(update: dict[str, Any]) -> None:
    message = update.get("message") or update.get("edited_message")
    if not message:
        return
    text = message.get("text")
    if not text:
        return
    from_user = message.get("from") or {}
    uid = from_user.get("id")
    chat_id = (message.get("chat") or {}).get("id")
    if uid is None or chat_id is None:
        return
    if uid not in ALLOWED_UIDS:
        log.info("ignoring message from non-whitelisted uid=%s (%s)", uid, from_user.get("username"))
        return
    handle_text(chat_id, uid, text, message.get("message_id"))


def main() -> None:
    if not TG_TOKEN:
        log.error("HEATMAP_TG_BOT_TOKEN (or TELEGRAM_BOT_TOKEN) not set — idling; set it in bot/.env.heatmap")
        while True:
            time.sleep(3600)
    if not ALLOWED_UIDS:
        log.warning("HEATMAP_TG_ALLOWED_UIDS is empty — no one is whitelisted, all messages ignored")
    log.info("heatmap telegram bot started; api=%s allowed_uids=%s", API_URL, sorted(ALLOWED_UIDS))
    get_universe()

    offset: Optional[int] = None
    while True:
        try:
            params = {"timeout": POLL_TIMEOUT}
            if offset:
                params["offset"] = offset
            resp = requests.get(f"{_TG_API}/getUpdates", params=params, timeout=POLL_TIMEOUT + 10)
            resp.raise_for_status()
            for update in resp.json().get("result", []):
                offset = update["update_id"] + 1
                try:
                    process_update(update)
                except Exception:  # noqa: BLE001
                    log.exception("failed to process update")
        except requests.exceptions.ReadTimeout:
            continue
        except Exception as exc:  # noqa: BLE001
            log.warning("getUpdates error: %s", exc)
            time.sleep(3)


if __name__ == "__main__":
    main()
