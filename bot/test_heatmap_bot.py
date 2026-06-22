#!/usr/bin/env python3
"""Lightweight unit checks for heatmap_bot pure logic (stdlib asserts, no pytest).

Run inside the image (it needs pybit/requests installed):
    docker compose exec heatmap-bot python /app/test_heatmap_bot.py
"""
import heatmap_bot as h


def test_bin_roundtrip():
    for p in (64123.4, 1.5, 0.083455, 3200.0):
        i = h.bin_index(p)
        lo, hi = h.bin_low(i), h.bin_low(i + 1)
        assert lo <= p < hi, (p, lo, hi)
        assert hi > lo
    # geometric width ~ BIN_PCT
    assert abs(h.bin_low(1001) / h.bin_low(1000) - (1 + h.PRED_BIN_PCT)) < 1e-9


def test_vp_window_start():
    now = 1_700_000_000_000  # fixed epoch ms
    assert h.vp_window_start("24h", now) == now - 24 * 3_600_000
    assert h.vp_window_start("4h", now) == now - 4 * 3_600_000
    daily = h.vp_window_start("daily", now)
    assert daily == now - (now % 86_400_000)
    weekly = h.vp_window_start("weekly", now)
    assert weekly <= daily and (now - weekly) <= 7 * 86_400_000
    # weekly anchor lands on a Monday 00:00 UTC
    import datetime as _dt
    assert _dt.datetime.fromtimestamp(weekly / 1000, _dt.timezone.utc).weekday() == 0


def test_fmt_price():
    assert h._fmt_price(64122.0) == "64,122.00"
    assert h._fmt_price(1.2345) == "1.2345"
    assert h._fmt_price(0.083455) == "0.083455"
    assert h._fmt_price(None) == "?"


def test_mapstate_add_and_consume():
    st = h.MapState("TESTUSDT", "1h")
    # seed prior price/OI, then a +OI candle opens cohorts
    st.apply_candle(1, 100.5, 99.5, 100.0, oi=1000.0, turnover=0.0)
    st.apply_candle(2, 100.5, 99.5, 100.0, oi=1100.0, turnover=0.0)  # dOI +100 -> opens longs & shorts
    longs = [k for k in st.alive if k[0] == "long"]
    shorts = [k for k in st.alive if k[0] == "short"]
    assert longs and shorts, "cohorts should be created on positive dOI"
    n_before = len(st.alive)
    # a deep down-candle should consume long-liq levels it trades through
    st.apply_candle(3, 100.0, 80.0, 85.0, oi=1100.0, turnover=0.0)
    assert any(l.side == "long" and l.consumed_ts == 3 for l in st.closed), "down move consumes long magnets"
    assert len(st.alive) < n_before


def test_dirichlet_direction():
    # responsibilities all on one bucket -> posterior mean shifts toward it
    nlev = len(h.LEVERAGES)
    alpha = h._prior_alpha()
    counts = [0.0] * nlev
    counts[-1] = 50.0  # all evidence on the last (highest-leverage) bucket
    before = alpha[-1] / sum(alpha)
    alpha2 = [alpha[i] + counts[i] for i in range(nlev)]
    after = alpha2[-1] / sum(alpha2)
    assert after > before, (before, after)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\nALL {len(tests)} TESTS PASSED")
