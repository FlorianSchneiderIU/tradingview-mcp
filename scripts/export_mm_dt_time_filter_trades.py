from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pybit.unified_trading import HTTP
from sklearn.tree import DecisionTreeClassifier


REPO_ROOT = Path(__file__).resolve().parents[1]
BOT_DIR = REPO_ROOT / "bot"
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))

from indicators import (  # noqa: E402
    ATR_HI,
    ATR_LO,
    ATR_LEN,
    ATR_PCTILE_WIN,
    EMA_LEN,
    N_FEATURES,
    SMA_LEN,
    ST_ATR_LEN,
    VOL_THR,
    VOL_WIN,
    build_signals,
    compute_rsi,
    compute_rsi_htf,
    extract_features_batch,
    ind_atr,
    ind_ema,
    ind_sma,
    metrics,
    rolling_atr_pctile,
    rolling_vol_ratio,
    sim_trail,
)
from train_dt import fetch_all_bars, instrument_status  # noqa: E402


IS_FRACTION = 0.75
DT_DEPTH = 2
DT_MIN_LEAF = 15
DT_MIN_IS_SH = 0.5
OOS_MIN_SH = 0.0
OOS_MAX_DEGRADATION = 0.20


def bars_to_arrays(bars: list[dict]) -> dict[str, np.ndarray | pd.DatetimeIndex | list[datetime]]:
    open_ = np.array([b["open"] for b in bars], dtype=np.float64)
    high = np.array([b["high"] for b in bars], dtype=np.float64)
    low = np.array([b["low"] for b in bars], dtype=np.float64)
    close = np.array([b["close"] for b in bars], dtype=np.float64)
    volume = np.array([b["volume"] for b in bars], dtype=np.float64)
    ts_ms = np.array([b["ts"] for b in bars], dtype=np.int64)
    ts_idx = pd.DatetimeIndex(pd.to_datetime(ts_ms, unit="ms", utc=True))
    py_times = [datetime.fromtimestamp(int(ts) / 1000, tz=timezone.utc) for ts in ts_ms]
    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "ts_ms": ts_ms,
        "ts_idx": ts_idx,
        "py_times": py_times,
    }


def trade_rows(
    *,
    symbol: str,
    sample: str,
    variant: str,
    split_offset: int,
    timestamps: pd.DatetimeIndex,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    sbull: np.ndarray,
    trades: list[dict],
    cfg: dict,
    dt_threshold: float | None = None,
    dt_accepted: bool | None = None,
) -> list[dict]:
    rows: list[dict] = []
    for trade in trades:
        entry_i = split_offset + int(trade["entry_i"])
        exit_i = split_offset + int(trade["exit_i"])
        direction = "long" if bool(sbull[entry_i]) else "short"
        rows.append(
            {
                "symbol": symbol,
                "sample": sample,
                "variant": variant,
                "entry_index": entry_i,
                "exit_index": exit_i,
                "entry_time": timestamps[entry_i],
                "exit_time": timestamps[exit_i],
                "direction": direction,
                "entry_price": float(close[entry_i]),
                "exit_price": float(close[exit_i]),
                "entry_high": float(high[entry_i]),
                "entry_low": float(low[entry_i]),
                "exit_reason": trade.get("reason"),
                "r": float(trade["r"]),
                "sl_mult": float(cfg["sl"]),
                "tp1_r": float(cfg["tp1"]),
                "trail_mult": float(cfg["trail"]),
                "dt_threshold": dt_threshold,
                "dt_accepted": dt_accepted,
            }
        )
    return rows


def valid_signal_indices(sbull, sbear, atr_pct, vol_rat) -> list[int]:
    return [
        i
        for i in range(len(sbull))
        if (sbull[i] or sbear[i]) and ATR_LO < atr_pct[i] < ATR_HI and vol_rat[i] >= VOL_THR
    ]


def train_dt_mask(
    *,
    split_i: int,
    sig_idx: list[int],
    close,
    high,
    low,
    open_,
    volume,
    atr14,
    atr_pct,
    vol_rat,
    ema200,
    sma13,
    sbull,
    sbear,
    timestamps,
    rsi14,
    rsi4h,
    cfg: dict,
) -> tuple[np.ndarray | None, dict]:
    is_sig = [i for i in sig_idx if i < split_i]
    oos_sig = [i for i in sig_idx if i >= split_i]
    if len(is_sig) < 20:
        return None, {"accepted": False, "reason": "too_few_is_signals"}

    sl_m = cfg["sl"]
    tp1_r = cfg["tp1"]
    trail_m = cfg["trail"]
    r_is, td_is = sim_trail(
        close[:split_i],
        high[:split_i],
        low[:split_i],
        sbull[:split_i],
        sbear[:split_i],
        atr14[:split_i],
        atr_pct[:split_i],
        vol_rat[:split_i],
        sl_mult=sl_m,
        tp1_r=tp1_r,
        trail_mult=trail_m,
    )
    if len(td_is) < 20:
        return None, {"accepted": False, "reason": "too_few_is_trades"}

    is_label_map = {t["entry_i"]: (1 if t["r"] > 0 else 0) for t in td_is}
    feat_rows_is = extract_features_batch(
        is_sig,
        close,
        high,
        low,
        open_,
        volume,
        atr14,
        atr_pct,
        vol_rat,
        ema200,
        sma13,
        sbull,
        sbear,
        timestamps,
        rsi14,
        rsi4h,
    )
    x_list: list[list[float]] = []
    y_list: list[int] = []
    valid_idx: list[int] = []
    for k, gi in enumerate(is_sig):
        row = feat_rows_is[k]
        label = is_label_map.get(gi)
        if row is not None and label is not None:
            x_list.append(row)
            y_list.append(label)
            valid_idx.append(gi)
    if len(x_list) < 20 or len(set(y_list)) < 2:
        return None, {"accepted": False, "reason": "insufficient_feature_rows"}

    clf = DecisionTreeClassifier(
        max_depth=DT_DEPTH,
        min_samples_leaf=DT_MIN_LEAF,
        class_weight="balanced",
        random_state=42,
    )
    x_is = np.array(x_list, dtype=np.float64)
    clf.fit(x_is, np.array(y_list, dtype=int))

    is_r_map = {t["entry_i"]: t["r"] for t in td_is}
    is_probas = clf.predict_proba(x_is)[:, 1]
    best_thr = 0.55
    best_is_sh = -99.0
    for thr in np.arange(0.40, 0.91, 0.05):
        selected = [is_r_map[gi] for gi, p in zip(valid_idx, is_probas) if p >= thr and gi in is_r_map]
        if len(selected) < 8:
            continue
        sh = metrics(np.array(selected), min_n=5)["sharpe"]
        if sh > best_is_sh:
            best_is_sh = sh
            best_thr = float(thr)
    if best_is_sh < DT_MIN_IS_SH:
        return None, {"accepted": False, "reason": "is_sharpe_gate", "is_sh": best_is_sh, "threshold": best_thr}

    r_oos_raw, _ = sim_trail(
        close[split_i:],
        high[split_i:],
        low[split_i:],
        sbull[split_i:],
        sbear[split_i:],
        atr14[split_i:],
        atr_pct[split_i:],
        vol_rat[split_i:],
        sl_mult=sl_m,
        tp1_r=tp1_r,
        trail_mult=trail_m,
    )
    raw_sh = metrics(r_oos_raw, min_n=3)["sharpe"]

    mask = np.zeros(len(close) - split_i, dtype=bool)
    if oos_sig:
        feat_rows_oos = extract_features_batch(
            oos_sig,
            close,
            high,
            low,
            open_,
            volume,
            atr14,
            atr_pct,
            vol_rat,
            ema200,
            sma13,
            sbull,
            sbear,
            timestamps,
            rsi14,
            rsi4h,
        )
        x_oos: list[list[float]] = []
        valid_oos: list[int] = []
        for k, gi in enumerate(oos_sig):
            row = feat_rows_oos[k]
            if row is not None:
                x_oos.append(row)
                valid_oos.append(gi)
        if x_oos:
            probas = clf.predict_proba(np.array(x_oos, dtype=np.float64))[:, 1]
            for gi, prob in zip(valid_oos, probas):
                if prob >= best_thr:
                    mask[gi - split_i] = True

    r_oos_dt, _ = sim_trail(
        close[split_i:],
        high[split_i:],
        low[split_i:],
        sbull[split_i:],
        sbear[split_i:],
        atr14[split_i:],
        atr_pct[split_i:],
        vol_rat[split_i:],
        sl_mult=sl_m,
        tp1_r=tp1_r,
        trail_mult=trail_m,
        signal_mask=mask,
    )
    dt_sh = metrics(r_oos_dt, min_n=3)["sharpe"]
    raw_floor = raw_sh - abs(raw_sh) * OOS_MAX_DEGRADATION
    accepted = dt_sh >= OOS_MIN_SH and dt_sh >= raw_floor
    info = {
        "accepted": bool(accepted),
        "threshold": best_thr,
        "is_sh": float(best_is_sh),
        "oos_raw_sh": float(raw_sh),
        "oos_dt_sh": float(dt_sh),
    }
    return mask, info


def export_symbol(symbol: str, cfg: dict, bars: list[dict]) -> tuple[list[dict], dict]:
    n = len(bars)
    if n < 500:
        return [], {"symbol": symbol, "status": "too_few_bars", "bars": n}

    arrays = bars_to_arrays(bars)
    open_ = arrays["open"]
    high = arrays["high"]
    low = arrays["low"]
    close = arrays["close"]
    volume = arrays["volume"]
    timestamps = arrays["ts_idx"]
    py_times = arrays["py_times"]

    atr_st = ind_atr(high, low, close, ST_ATR_LEN)
    atr14 = ind_atr(high, low, close, ATR_LEN)
    ema200 = ind_ema(close, EMA_LEN)
    sma13 = ind_sma(close, SMA_LEN)
    atr_pct = rolling_atr_pctile(atr14, ATR_PCTILE_WIN)
    vol_rat = rolling_vol_ratio(volume, VOL_WIN)
    rsi14 = compute_rsi(close, 14)
    rsi4h = compute_rsi_htf(timestamps, close, "4h", 14)
    sbull, sbear = build_signals(close, open_, sma13, ema200, atr_st)

    split_i = int(n * IS_FRACTION)
    rows: list[dict] = []
    summary = {"symbol": symbol, "status": "ok", "bars": n, "split_time": str(timestamps[split_i])}

    for sample, start, end in [("train", 0, split_i), ("oos", split_i, n)]:
        r_raw, td_raw = sim_trail(
            close[start:end],
            high[start:end],
            low[start:end],
            sbull[start:end],
            sbear[start:end],
            atr14[start:end],
            atr_pct[start:end],
            vol_rat[start:end],
            sl_mult=cfg["sl"],
            tp1_r=cfg["tp1"],
            trail_mult=cfg["trail"],
        )
        rows.extend(
            trade_rows(
                symbol=symbol,
                sample=sample,
                variant="trail",
                split_offset=start,
                timestamps=timestamps,
                close=close,
                high=high,
                low=low,
                sbull=sbull,
                trades=td_raw,
                cfg=cfg,
            )
        )
        summary[f"{sample}_trail_trades"] = len(td_raw)
        summary[f"{sample}_trail_sh"] = metrics(r_raw, min_n=3)["sharpe"]

    if bool(cfg.get("use_dt", False)):
        sig_idx = valid_signal_indices(sbull, sbear, atr_pct, vol_rat)
        mask, info = train_dt_mask(
            split_i=split_i,
            sig_idx=sig_idx,
            close=close,
            high=high,
            low=low,
            open_=open_,
            volume=volume,
            atr14=atr14,
            atr_pct=atr_pct,
            vol_rat=vol_rat,
            ema200=ema200,
            sma13=sma13,
            sbull=sbull,
            sbear=sbear,
            timestamps=py_times,
            rsi14=rsi14,
            rsi4h=rsi4h,
            cfg=cfg,
        )
        summary.update({f"dt_{key}": value for key, value in info.items()})
        if mask is not None:
            r_dt, td_dt = sim_trail(
                close[split_i:],
                high[split_i:],
                low[split_i:],
                sbull[split_i:],
                sbear[split_i:],
                atr14[split_i:],
                atr_pct[split_i:],
                vol_rat[split_i:],
                sl_mult=cfg["sl"],
                tp1_r=cfg["tp1"],
                trail_mult=cfg["trail"],
                signal_mask=mask,
            )
            rows.extend(
                trade_rows(
                    symbol=symbol,
                    sample="oos",
                    variant="dt",
                    split_offset=split_i,
                    timestamps=timestamps,
                    close=close,
                    high=high,
                    low=low,
                    sbull=sbull,
                    trades=td_dt,
                    cfg=cfg,
                    dt_threshold=info.get("threshold"),
                    dt_accepted=bool(info.get("accepted", False)),
                )
            )
            summary["oos_dt_trades"] = len(td_dt)
            summary["oos_dt_sh"] = metrics(r_dt, min_n=3)["sharpe"]
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Million Moves train/OOS trades for time-filter tests.")
    parser.add_argument("--configs", type=Path, default=REPO_ROOT / "bot" / "configs" / "top20_configs.json")
    parser.add_argument("--symbols", default=None, help="Comma-separated Bybit symbols, e.g. ETHUSDT,WIFUSDT.")
    parser.add_argument("--since", default="2022-04-20")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--output-prefix", type=Path, default=REPO_ROOT / "scripts" / "million_moves_dt_time_filter")
    args = parser.parse_args()

    with args.configs.open(encoding="utf-8") as fh:
        all_configs = {k: v for k, v in json.load(fh).items() if not k.startswith("_")}
    if args.symbols:
        wanted = {s.strip().upper() for s in args.symbols.split(",") if s.strip()}
        configs = {k: v for k, v in all_configs.items() if k in wanted}
    else:
        configs = {k: v for k, v in all_configs.items() if v.get("enable_mm", True)}

    demo = args.demo or os.environ.get("BYBIT_DEMO", "true").lower() in {"1", "true", "yes"}
    http = HTTP(
        testnet=False,
        demo=demo,
        api_key=os.environ.get("BYBIT_API_KEY", ""),
        api_secret=os.environ.get("BYBIT_API_SECRET", ""),
    )

    all_rows: list[dict] = []
    summaries: list[dict] = []
    for symbol, cfg in configs.items():
        try:
            status = instrument_status(symbol, http)
            if status and status != "Trading":
                summaries.append({"symbol": symbol, "status": f"instrument_{status}"})
                print(f"{symbol}: status={status}, skipping")
                continue
            print(f"{symbol}: fetching bars ...", end=" ", flush=True)
            bars = fetch_all_bars(symbol, args.since, http)
            if bars:
                first = datetime.fromtimestamp(bars[0]["ts"] / 1000, tz=timezone.utc).date()
                last = datetime.fromtimestamp(bars[-1]["ts"] / 1000, tz=timezone.utc).date()
                print(f"{len(bars)} bars {first}..{last}")
            else:
                print("0 bars")
            rows, summary = export_symbol(symbol, cfg, bars)
            all_rows.extend(rows)
            summaries.append(summary)
            print(f"  exported {len(rows)} rows")
        except Exception as exc:
            summaries.append({"symbol": symbol, "status": "error", "detail": str(exc)})
            print(f"{symbol}: ERROR {exc}")

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    trades_path = args.output_prefix.with_name(f"{args.output_prefix.name}_trades.csv")
    summary_path = args.output_prefix.with_name(f"{args.output_prefix.name}_summary.csv")
    pd.DataFrame(all_rows).to_csv(trades_path, index=False)
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    print(f"Saved trades -> {trades_path}")
    print(f"Saved summary -> {summary_path}")


if __name__ == "__main__":
    main()
