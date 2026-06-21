"""Fetch + cache funding + open-interest history for the basket (run in background)."""
import sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from astro_reversal import basket, capitulation as cap

START = "2022-01-01"
for sym in basket.BASKET:
    t0 = time.time()
    try:
        f = cap.fetch_funding(sym, START, None)
        o = cap.fetch_oi(sym, "4h", START, None)
        fr = (f["time"].min().date() if len(f) else None)
        orr = (o["time"].min().date() if len(o) else None)
        print(f"OK {sym}: funding {len(f)} (from {fr}), OI {len(o)} (from {orr}) in {time.time()-t0:.0f}s", flush=True)
    except Exception as ex:
        print(f"FAIL {sym}: {ex!r}", flush=True)
print("DONE", flush=True)
