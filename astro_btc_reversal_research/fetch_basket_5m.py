"""Fetch + cache 5m klines for the whole basket (run in background, ~45 min)."""
import sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from astro_reversal import basket, reuse

INTERVAL, START, END = "5m", "2022-01-01", "2026-06-01"
s = reuse.parse_utc_datetime(START); e = reuse.parse_utc_datetime(END)
for sym in basket.BASKET:
    t0 = time.time()
    try:
        df = reuse.load_bybit_cached(sym, INTERVAL, s, e, reuse.DEFAULT_CACHE_DIR)
        print(f"OK {sym}: {len(df)} bars in {time.time()-t0:.0f}s", flush=True)
    except Exception as ex:
        print(f"FAIL {sym}: {ex!r}", flush=True)
print("DONE", flush=True)
