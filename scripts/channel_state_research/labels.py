from __future__ import annotations

import numpy as np
import pandas as pd


def high_before_low(open_value: float, high_value: float, low_value: float) -> bool:
    return abs(open_value - high_value) < abs(open_value - low_value)


def add_triple_barrier_labels(
    frame: pd.DataFrame,
    *,
    close_column: str = "close_tf_1h",
    open_column: str = "open_tf_1h",
    high_column: str = "high_tf_1h",
    low_column: str = "low_tf_1h",
    atr_column: str = "atr_tf_1h",
    alpha: float = 1.5,
    beta: float = 1.5,
    horizon_bars: int = 24,
) -> pd.DataFrame:
    out = frame.copy()
    opens = out[open_column].astype(float).to_numpy()
    highs = out[high_column].astype(float).to_numpy()
    lows = out[low_column].astype(float).to_numpy()
    closes = out[close_column].astype(float).to_numpy()
    atrs = out[atr_column].astype(float).to_numpy()
    decision_times = pd.to_datetime(out["decision_time"], utc=True).to_list()

    tb_label = np.full(len(out), np.nan)
    long_label = np.full(len(out), np.nan)
    short_label = np.full(len(out), np.nan)
    upper_barrier = np.full(len(out), np.nan)
    lower_barrier = np.full(len(out), np.nan)
    timeout_time = np.full(len(out), np.datetime64("NaT"), dtype="datetime64[ns]")
    hit_time = np.full(len(out), np.datetime64("NaT"), dtype="datetime64[ns]")
    hit_reason = np.full(len(out), "", dtype=object)
    realized_return = np.full(len(out), np.nan)

    for index in range(len(out)):
        atr_value = atrs[index]
        if not np.isfinite(atr_value) or atr_value <= 0.0:
            continue
        if index + 1 >= len(out):
            continue
        upper = closes[index] + alpha * atr_value
        lower = closes[index] - beta * atr_value
        upper_barrier[index] = upper
        lower_barrier[index] = lower

        final_index = min(len(out) - 1, index + int(horizon_bars))
        timeout_stamp = pd.Timestamp(decision_times[final_index]).tz_convert("UTC")
        timeout_time[index] = np.datetime64(timeout_stamp.tz_localize(None))
        label = 0.0
        reason = "timeout"
        label_time = timeout_stamp
        realized = closes[final_index] / closes[index] - 1.0

        for cursor in range(index + 1, final_index + 1):
            upper_hit = highs[cursor] >= upper
            lower_hit = lows[cursor] <= lower
            if upper_hit and lower_hit:
                upper_first = high_before_low(opens[cursor], highs[cursor], lows[cursor])
                label = 1.0 if upper_first else -1.0
                reason = "upper_same_bar" if upper_first else "lower_same_bar"
                label_time = pd.Timestamp(decision_times[cursor]).tz_convert("UTC")
                realized = upper / closes[index] - 1.0 if upper_first else lower / closes[index] - 1.0
                break
            if upper_hit:
                label = 1.0
                reason = "upper"
                label_time = pd.Timestamp(decision_times[cursor]).tz_convert("UTC")
                realized = upper / closes[index] - 1.0
                break
            if lower_hit:
                label = -1.0
                reason = "lower"
                label_time = pd.Timestamp(decision_times[cursor]).tz_convert("UTC")
                realized = lower / closes[index] - 1.0
                break

        tb_label[index] = label
        long_label[index] = 1.0 if label == 1.0 else 0.0
        short_label[index] = 1.0 if label == -1.0 else 0.0
        hit_time[index] = np.datetime64(label_time.tz_localize(None))
        hit_reason[index] = reason
        realized_return[index] = realized

    out["tb_label"] = tb_label
    out["long_label"] = long_label
    out["short_label"] = short_label
    out["upper_barrier"] = upper_barrier
    out["lower_barrier"] = lower_barrier
    out["label_timeout_time"] = pd.to_datetime(timeout_time, utc=True)
    out["label_hit_time"] = pd.to_datetime(hit_time, utc=True)
    out["label_end_time"] = out["label_hit_time"].fillna(out["label_timeout_time"])
    out["label_hit_reason"] = hit_reason
    out["label_realized_return"] = realized_return
    return out


def add_raw_future_return_labels(
    frame: pd.DataFrame,
    *,
    close_column: str = "close_tf_1h",
    horizon_bars: int = 24,
) -> pd.DataFrame:
    out = frame.copy()
    future_close = out[close_column].astype(float).shift(-int(horizon_bars))
    future_return = future_close / out[close_column].astype(float) - 1.0
    out["future_return_h"] = future_return
    out["future_return_sign"] = np.sign(future_return)
    return out
