#!/usr/bin/env python3
"""Fetch Bybit funding-rate history for the Wolfe universe (public, no auth).

Funding posts every 8h on Bybit. Caches one CSV per symbol
(scripts/data/funding/<symbol>.csv: ts_ms, funding_rate) for offline research.

Usage: python scripts/fetch_funding.py [--since 2022-01-01]
"""
from __future__ import annotations
import argparse, json, os, time
from datetime import datetime, timezone
import requests

BASE = "https://api.bybit.com"
OUT = os.path.join("scripts", "data", "funding")
CFG = "bot/configs/wolfe_wave_shared_v1_configs.json"


def fetch_symbol(symbol: str, start_ms: int, session: requests.Session) -> list[tuple[int, float]]:
    rows: dict[int, float] = {}
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    while True:
        params = {"category": "linear", "symbol": symbol, "limit": 200, "endTime": end_ms}
        batch = None
        for attempt in range(5):
            try:
                resp = session.get(f"{BASE}/v5/market/funding/history", params=params, timeout=30)
                resp.raise_for_status()
                payload = resp.json()
                if payload.get("retCode") not in (0, "0"):
                    raise RuntimeError(payload.get("retMsg"))
                batch = payload.get("result", {}).get("list", [])
                break
            except Exception:
                time.sleep(0.5 * (attempt + 1))
        if not batch:
            break
        for it in batch:
            ts = int(it["fundingRateTimestamp"])
            rows[ts] = float(it["fundingRate"])
        oldest = min(int(it["fundingRateTimestamp"]) for it in batch)
        if oldest <= start_ms or len(batch) < 200:
            break
        end_ms = oldest - 1
        time.sleep(0.05)
    return sorted((ts, r) for ts, r in rows.items() if ts >= start_ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2022-01-01")
    args = ap.parse_args()
    start_ms = int(datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    os.makedirs(OUT, exist_ok=True)
    universe = [k for k in json.load(open(CFG)) if not k.startswith("_")]
    session = requests.Session()
    ok = fail = 0
    for i, sym in enumerate(universe, 1):
        path = os.path.join(OUT, f"{sym.lower()}.csv")
        if os.path.exists(path):
            continue
        try:
            t0 = time.time()
            data = fetch_symbol(sym, start_ms, session)
            if not data:
                raise RuntimeError("no funding rows")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("ts_ms,funding_rate\n")
                for ts, r in data:
                    fh.write(f"{ts},{r}\n")
            ok += 1
            first = datetime.fromtimestamp(data[0][0] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
            print(f"  [{i}/{len(universe)}] {sym}: {len(data)} funding pts since {first} [{time.time()-t0:.0f}s]", flush=True)
        except Exception as exc:  # noqa: BLE001
            fail += 1
            print(f"  [{i}/{len(universe)}] {sym}: FAILED {exc}", flush=True)
    print(f"done: {ok} fetched, {fail} failed -> {OUT}")


if __name__ == "__main__":
    main()
