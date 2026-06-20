"""Markdown / JSON / CSV report writers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from . import reuse


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=reuse.json_default), encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, float):
        if value != value:  # NaN
            return "n/a"
        return f"{value:.{digits}f}"
    return str(value)


def dark_pivot_markdown(r: dict) -> str:
    cfg = r["config"]
    test = r["expansion_test"]
    lines = [
        "# Dark Pivot Candidate Test (Milestone 1)",
        "",
        f"**Candidate:** {cfg['body_1']}-{cfg['body_2']} hard aspects {cfg['aspects']}",
        f"**Symbol/timeframe:** {cfg['symbol']} {cfg['timeframe']} | "
        f"{cfg['start']} -> {cfg['end']} ({r['data']['bars']} candles)",
        f"**Dump definition:** move <= -{cfg['dump_threshold_atr']} ATR over {cfg['dump_lookback']} bars",
        f"**Expansion:** bullish, horizon {cfg['expansion_horizon']} candles, "
        f"target {cfg['expansion_target_atr']} ATR, buffer {cfg['expansion_buffer_atr']} ATR",
        f"**Event window:** +/-{cfg['event_window_bars']} candles around each aspect event",
        "",
        "## Headline result",
        "",
        "| Set | N | Bullish-expansion hit rate | 95% CI |",
        "|---|---|---|---|",
        f"| A: dump days on/into aspect window | {test['n_event']} | "
        f"{_fmt(test['rate_event'])} | [{_fmt(test['ci_event'][0])}, {_fmt(test['ci_event'][1])}] |",
        f"| B: ordinary dump days (baseline) | {test['n_baseline']} | "
        f"{_fmt(test['rate_baseline'])} | [{_fmt(test['ci_baseline'][0])}, {_fmt(test['ci_baseline'][1])}] |",
        "",
        f"- **Lift (A / B):** {_fmt(test['lift'])}",
        f"- **Rate difference (A - B):** {_fmt(test['rate_diff'])} "
        f"(95% CI [{_fmt(test['diff_ci'][0])}, {_fmt(test['diff_ci'][1])}])",
        f"- **Binomial p-value** (A beats baseline rate): {_fmt(test['binomial_p'], 5)}",
        f"- **Random-calendar p-value** ({r['random_calendar']['n_draws']} draws): "
        f"{_fmt(r['random_calendar']['empirical_p'], 5)} "
        f"(null mean {_fmt(r['random_calendar']['null_mean'])})",
        "",
        "## Shifted-calendar baseline (real events should beat these)",
        "",
        "| Offset (days) | N (dump in window) | Hit rate |",
        "|---|---|---|",
    ]
    for row in r["shifted_calendar"]:
        lines.append(f"| +{row['offset_days']} | {row['n']} | {_fmt(row['rate'])} |")

    mfe = test["mfe_mae"]
    lines += [
        "",
        "## MFE / MAE / max-R for set A (event dump days)",
        "",
        f"- mean MFE: {_fmt(mfe['mean_mfe_r'])} R | mean MAE: {_fmt(mfe['mean_mae_r'])} R",
        f"- median max-R available: {_fmt(mfe['median_max_r'])} R",
        f"- share reaching >=1R: {_fmt(mfe['share_ge_1r'])} | >=2R: {_fmt(mfe['share_ge_2r'])}",
        "",
        "## Out-of-sample (holdout) check",
        "",
        f"Holdout start: {cfg['holdout_start']}",
        f"- dev set A hit rate: {_fmt(r['holdout']['rate_event_dev'])} (N {r['holdout']['n_event_dev']})",
        f"- holdout set A hit rate: {_fmt(r['holdout']['rate_event_holdout'])} "
        f"(N {r['holdout']['n_event_holdout']})",
        "",
        "## Pivot proximity (secondary)",
        "",
        f"- aspect candles within +/-{cfg['event_window_bars']} of an ATR pivot: "
        f"{_fmt(r['pivot_proximity']['event_pivot_share'])}",
        f"- baseline candles near a pivot: {_fmt(r['pivot_proximity']['baseline_pivot_share'])} "
        f"(lift {_fmt(r['pivot_proximity']['lift'])})",
        "",
        "## Interpretation guard",
        "",
        "A positive lift is only interesting if it (a) beats the random-calendar null, "
        "(b) beats shifted calendars, and (c) holds out-of-sample. Small N inflates noise; "
        "read the CIs, not the point estimates.",
        "",
    ]
    return "\n".join(lines)


def calendar_markdown(r: dict) -> str:
    cfg = r["config"]
    dp = r["dark_pivot"]
    lines = [
        f"# Astro Calendar Search (precision framing) - {cfg['timeframe']}",
        "",
        f"**Symbol:** {cfg['symbol']} | {cfg['start']} -> {cfg['end']} ({r['data']['bars']} candles)",
        f"**Pivots:** ATR threshold {cfg['pivot_threshold_atr']} ({r['data']['n_pivots']} pivots) | "
        f"tolerance window +/-{cfg['window_bars']} candles ({cfg['window_kind']}) | orb {cfg['orb_deg']} deg",
        f"**Baseline hit rate** (random window contains a pivot): {_fmt(r['data']['baseline_hit'])}",
        f"**Hypotheses tested:** {r['n_hypotheses']} | BH FDR alpha {cfg['fdr_alpha']} -> "
        f"{r['n_significant']} significant",
        "",
        "Question: when a calendar fires, does a pivot land within the window more often than "
        "for random windows of the same count/width? Missing pivots is fine; **hit rate + lift** matter.",
        "",
        "## Dark Pivot calendar (Moon-Pluto hard aspects 0/90/180/270)",
        "",
        f"- firings: {dp['n_events']} | **hit rate: {_fmt(dp['hit_rate'])}** | baseline "
        f"{_fmt(dp['baseline_hit'])} | lift {_fmt(dp['lift'])} | coverage {_fmt(dp['coverage'])}",
        f"- binomial p {_fmt(dp['binomial_p'], 5)} | random-calendar p {_fmt(dp['random_p'], 5)}",
        f"- holdout (2025+): hit rate {_fmt(dp['holdout_hit_rate'])} vs baseline "
        f"{_fmt(dp['holdout_baseline'])} (lift {_fmt(dp['holdout_lift'])})",
    ]
    if r.get("dark_pivot_shifted"):
        lines.append("- shifted-calendar controls (real should beat these): " + ", ".join(
            f"+{s['offset_bars']}b={_fmt(s['hit_rate'])}" for s in r["dark_pivot_shifted"]))
    lines += [
        "",
        "## Top single pair x aspect calendars by hit rate",
        "",
        "| Pair | Aspect | Firings | Hit rate | Lift | Coverage | Binom p | Rand p | BH sig | Holdout hit |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in r["top"]:
        lines.append(
            f"| {row['pair']} | {row['aspect']:g} | {row['n_events']} | {_fmt(row['hit_rate'])} | "
            f"{_fmt(row['lift'])} | {_fmt(row['coverage'])} | {_fmt(row['binomial_p'], 5)} | "
            f"{_fmt(row['random_p'], 5)} | {'yes' if row['bh_significant'] else 'no'} | "
            f"{_fmt(row['holdout_hit_rate'])} |"
        )
    lines += ["", "## Aspect-confluence calendars", "",
              "| Min simultaneous aspects | Firings | Hit rate | Baseline | Lift | Binom p | Holdout hit |",
              "|---|---|---|---|---|---|---|"]
    for row in r["confluence"]:
        lines.append(
            f"| >= {row['min_count']} | {row['n_events']} | {_fmt(row['hit_rate'])} | "
            f"{_fmt(row['baseline_hit'])} | {_fmt(row['lift'])} | {_fmt(row['binomial_p'], 5)} | "
            f"{_fmt(row['holdout_hit_rate'])} |"
        )
    lines += [
        "",
        "## Reading guide",
        "",
        "A calendar is credible only if its hit rate clearly exceeds the baseline (lift > 1) with a "
        "small binomial/random p, **and** the lift survives on the holdout column and beats the shifted "
        "controls. High lift on very few firings is noise - weight the firing count and FDR column.",
        "",
    ]
    return "\n".join(lines)


def conditional_calendar_markdown(r: dict) -> str:
    cfg = r["config"]
    lines = [
        f"# Direction-Conditional Calendar Search - {cfg['timeframe']}",
        "",
        f"**Symbol:** {cfg['symbol']} | {cfg['start']} -> {cfg['end']} ({r['data']['bars']} candles)",
        f"**Pivots:** ATR threshold {cfg['pivot_threshold_atr']} | tolerance +/-{cfg['window_bars']} "
        f"candles | orb {cfg['orb_deg']} deg",
        f"**Context:** move >= {cfg['dump_threshold_atr']} ATR over {cfg['dump_lookback']} bars into the "
        f"aspect. Dump->expect BOTTOM, pump->expect TOP.",
        "",
        "Tests proposal H5 / the Dark Pivot thesis: does an aspect firing add timing info *beyond* the "
        "price context? Baseline = random bars from the SAME context, so lift isolates the astro part.",
        "",
    ]
    for direction in ("dump_bottom", "pump_top"):
        block = r[direction]
        ctx = "dump days -> pivot LOW" if direction == "dump_bottom" else "pump days -> pivot HIGH"
        lines += [
            f"## {ctx}",
            "",
            f"Context bars: {block['n_context_bars']} | context baseline hit rate "
            f"{_fmt(block['baseline_hit'])} | hypotheses {block['n_hypotheses']} | "
            f"BH-FDR significant: {block['n_significant']}",
            "",
            "| Pair | Aspect | Firings | Hit rate | Lift | Binom p | Rand p | BH sig | Holdout hit |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for row in block["top"]:
            lines.append(
                f"| {row['pair']} | {row['aspect']:g} | {row['n_events']} | {_fmt(row['hit_rate'])} | "
                f"{_fmt(row['lift'])} | {_fmt(row['binomial_p'], 5)} | {_fmt(row['random_p'], 5)} | "
                f"{'yes' if row['bh_significant'] else 'no'} | {_fmt(row['holdout_hit_rate'])} |"
            )
        dp = block.get("dark_pivot")
        if dp:
            lines += ["",
                      f"Dark Pivot (Moon-Pluto hard) in this context: firings {dp['n_events']}, hit "
                      f"{_fmt(dp['hit_rate'])} vs baseline {_fmt(dp['baseline_hit'])} (lift "
                      f"{_fmt(dp['lift'])}, binom p {_fmt(dp['binomial_p'], 5)}, holdout lift "
                      f"{_fmt(dp['holdout_lift'])})."]
        lines.append("")
    lines += [
        "## Reading guide",
        "",
        "Lift > 1 here means the aspect beats *random same-context bars* (e.g. random dumps), not just "
        "the unconditional baseline. Credible only if it survives BH-FDR and the holdout. Few firings "
        "(aspect AND context) -> expect noise.",
        "",
    ]
    return "\n".join(lines)


def ml_markdown(r: dict) -> str:
    cfg = r["config"]
    lines = [
        "# ML Pivot-Window Model (Milestone 3)",
        "",
        f"**Target:** {cfg['target']}_within_{cfg['horizon']} | **Symbol/timeframe:** "
        f"{cfg['symbol']} {cfg['timeframe']} | {cfg['start']} -> {cfg['end']} ({r['data']['bars']} candles)",
        f"**Pivots:** ATR threshold {cfg['pivot_threshold_atr']} ({r['data']['n_pivots']} pivots, "
        f"~1 per {_fmt(r['data'].get('pivot_stats', {}).get('median_gap_days'), 1)}d) | "
        f"base rate {_fmt(r['data']['base_rate'])}",
        f"**Validation:** {cfg['n_folds']}-fold expanding walk-forward, embargo {cfg['embargo']} candles | "
        f"holdout {cfg['holdout_start']}",
        "",
        "## Pooled out-of-sample (walk-forward) by feature set x model",
        "",
        "| Feature set | Model | PR-AUC | ROC-AUC | Brier | Prec@5% | Lift@5% | Holdout PR-AUC |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in r["results"]:
        oos = row["oos"]
        hold = row.get("holdout", {})
        p5 = oos["precision_at"].get("top_0.05", {})
        lines.append(
            f"| {row['feature_set']} | {row['model']} | {_fmt(oos['pr_auc'])} | "
            f"{_fmt(oos['roc_auc'])} | {_fmt(oos['brier'])} | {_fmt(p5.get('precision'))} | "
            f"{_fmt(p5.get('lift'))} | {_fmt(hold.get('pr_auc'))} |"
        )
    if r.get("placebo"):
        lines += [
            "",
            "## Shifted-placebo control (astro/cycle shifted ~37d)",
            "",
            "| Feature set | Model | Real OOS PR-AUC | Placebo OOS PR-AUC |",
            "|---|---|---|---|",
        ]
        for row in r["placebo"]:
            lines.append(
                f"| {row['feature_set']} | {row['model']} | {_fmt(row['real_pr_auc'])} | "
                f"{_fmt(row['placebo_pr_auc'])} |"
            )
    lines += [
        "",
        "## Reading guide",
        "",
        "PR-AUC above the base rate and lift@K > 1 indicate signal; pivots are rare so "
        "PR-AUC/precision@K matter more than ROC-AUC. A feature set is only credible if it "
        "beats both `price_only` and its shifted placebo, and holds on the holdout column.",
        "",
    ]
    return "\n".join(lines)


def discovery_markdown(r: dict) -> str:
    cfg = r["config"]
    lines = [
        f"# Aspect Discovery (Milestone 2) - {cfg['timeframe']}",
        "",
        f"**Symbol:** {cfg['symbol']} | {cfg['start']} -> {cfg['end']} ({r['data']['bars']} candles)",
        f"**Pivot window:** +/-{cfg['window_bars']} candles ({cfg['window_kind']}) | "
        f"orb {cfg['orb_deg']} deg | pivot ATR threshold {cfg['pivot_threshold_atr']}",
        f"**Hypotheses tested:** {r['n_hypotheses']} (pair x aspect) | "
        f"BH FDR alpha {cfg['fdr_alpha']} -> {r['n_significant']} significant",
        "",
        "## Top aspect (pair x angle) by pivot-window lift",
        "",
        "| Pair | Aspect | In-window bars | Lift | Binom p | Rand p | BH sig | Holdout lift |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in r["top"]:
        lines.append(
            f"| {row['pair']} | {row['aspect']:g} | {row['in_window_bars']} | "
            f"{_fmt(row['lift'])} | {_fmt(row['binomial_p'], 5)} | {_fmt(row['random_p'], 5)} | "
            f"{'yes' if row['bh_significant'] else 'no'} | {_fmt(row['holdout_lift'])} |"
        )
    lines += [
        "",
        f"Baseline pivot-window rate: {_fmt(r['baseline_rate'])}.",
        "Lift = P(pivot within window | in aspect window) / baseline. Holdout (2025+) "
        "is reported, never used to rank. Treat large lifts with tiny in-window-bar counts "
        "as noise until they survive the holdout and random-calendar columns.",
        "",
    ]
    return "\n".join(lines)
