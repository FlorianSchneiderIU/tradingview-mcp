from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scripts.backtest_turtle_soup import normalize_timeframe


@dataclass(frozen=True)
class TurtleSoupCandidatePreset:
    name: str
    symbols: tuple[str, ...]
    interval: str
    structure_tf: str
    zone_tf: str
    zone_hold_min_prob: float
    max_structure_bars_to_choch: int = 32
    max_zone_scan: int = 250
    lookback_days: int = 120
    max_signal_age_minutes: float = 15.0
    model_path: Path | None = None
    order_link_prefix: str | None = None
    notes: str = ""


PRESETS: dict[str, TurtleSoupCandidatePreset] = {
    "current_live_v1": TurtleSoupCandidatePreset(
        name="current_live_v1",
        symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT"),
        interval="5m",
        structure_tf="15m",
        zone_tf="1h",
        zone_hold_min_prob=0.60,
        max_structure_bars_to_choch=32,
        max_zone_scan=250,
        order_link_prefix="TSOUP",
        notes="Current conservative live/demo Turtle Soup candidate.",
    ),
    "fast_eval_btc_sol_v1": TurtleSoupCandidatePreset(
        name="fast_eval_btc_sol_v1",
        symbols=("BTCUSDT", "SOLUSDT"),
        interval="5m",
        structure_tf="5m",
        zone_tf="1h",
        zone_hold_min_prob=0.45,
        max_structure_bars_to_choch=32,
        max_zone_scan=250,
        order_link_prefix="TSFAST",
        notes=(
            "Higher-cadence evaluation sibling: same 1h zones, 5m structure, looser 0.45 ML gate, "
            "BTC+SOL only."
        ),
    ),
}


def preset_names() -> list[str]:
    return sorted(PRESETS)


def get_preset(name: str) -> TurtleSoupCandidatePreset:
    try:
        return PRESETS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown Turtle Soup preset {name!r}. Available: {', '.join(preset_names())}") from exc


def preset_summary(preset: TurtleSoupCandidatePreset) -> dict[str, Any]:
    data = asdict(preset)
    data["symbols"] = list(preset.symbols)
    data["model_path"] = str(preset.model_path) if preset.model_path else None
    return data


def apply_preset_args(args: Any, preset: TurtleSoupCandidatePreset) -> Any:
    args.symbols = list(preset.symbols)
    args.interval = normalize_timeframe(preset.interval)
    args.structure_tf = normalize_timeframe(preset.structure_tf)
    args.zone_tf = normalize_timeframe(preset.zone_tf)
    args.zone_hold_min_prob = preset.zone_hold_min_prob
    args.max_structure_bars_to_choch = preset.max_structure_bars_to_choch
    args.max_zone_scan = preset.max_zone_scan
    args.lookback_days = preset.lookback_days
    args.max_signal_age_minutes = preset.max_signal_age_minutes
    if preset.model_path is not None:
        args.model = preset.model_path
        args.no_ml_filter = False
    if preset.order_link_prefix:
        args.order_link_prefix = preset.order_link_prefix
    return args
