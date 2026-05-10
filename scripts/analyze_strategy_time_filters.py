from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent
OUT_PREFIX = ROOT / "strategy_time_filters"
SPLIT = pd.Timestamp("2025-04-20", tz="UTC")

DEPLOYED_ORB_SYMBOLS = {"ETHUSDT", "WIFUSDT", "NEARUSDT", "ENAUSDT", "OPUSDT", "ONDOUSDT"}


@dataclass(frozen=True)
class TradeSource:
    name: str
    path: Path
    r_col: str
    time_col: str = "entry_time"
    sample_col: str | None = None
    min_oos_bucket: int = 20
    loader: Callable[[pd.DataFrame], pd.DataFrame] | None = None
    notes: str = ""


def profit_factor(values: pd.Series) -> float:
    wins = values[values > 0].sum()
    losses = -values[values < 0].sum()
    if losses <= 0:
        return math.inf if wins > 0 else 0.0
    return float(wins / losses)


def max_drawdown(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    curve = values.astype(float).cumsum()
    dd = curve - curve.cummax()
    return float(dd.min())


def safe_welch_p(group: pd.Series, rest: pd.Series) -> float:
    x = pd.to_numeric(group, errors="coerce").dropna().astype(float).values
    y = pd.to_numeric(rest, errors="coerce").dropna().astype(float).values
    if len(x) < 2 or len(y) < 2:
        return math.nan
    result = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
    return float(result.pvalue) if np.isfinite(result.pvalue) else math.nan


def safe_mannwhitney_p(group: pd.Series, rest: pd.Series) -> float:
    x = pd.to_numeric(group, errors="coerce").dropna().astype(float).values
    y = pd.to_numeric(rest, errors="coerce").dropna().astype(float).values
    if len(x) < 2 or len(y) < 2:
        return math.nan
    try:
        result = stats.mannwhitneyu(x, y, alternative="two-sided")
    except ValueError:
        return math.nan
    return float(result.pvalue) if np.isfinite(result.pvalue) else math.nan


def safe_fisher_win_p(group: pd.Series, rest: pd.Series) -> float:
    x = pd.to_numeric(group, errors="coerce").dropna().astype(float)
    y = pd.to_numeric(rest, errors="coerce").dropna().astype(float)
    if len(x) == 0 or len(y) == 0:
        return math.nan
    table = [
        [int((x > 0).sum()), int((x <= 0).sum())],
        [int((y > 0).sum()), int((y <= 0).sum())],
    ]
    try:
        _odds, p_value = stats.fisher_exact(table, alternative="two-sided")
    except ValueError:
        return math.nan
    return float(p_value) if np.isfinite(p_value) else math.nan


def bootstrap_diff_ci(group: pd.Series, rest: pd.Series, *, n_boot: int = 2000, seed: int = 7) -> tuple[float, float]:
    x = pd.to_numeric(group, errors="coerce").dropna().astype(float).values
    y = pd.to_numeric(rest, errors="coerce").dropna().astype(float).values
    if len(x) < 2 or len(y) < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    x_idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    y_idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    diffs = x[x_idx].mean(axis=1) - y[y_idx].mean(axis=1)
    low, high = np.quantile(diffs, [0.025, 0.975])
    return float(low), float(high)


def bh_qvalues(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce")
    q = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna()
    m = len(valid)
    if m == 0:
        return q
    ordered = valid.sort_values()
    ranks = np.arange(1, m + 1, dtype=float)
    raw_q = ordered.values * m / ranks
    raw_q = np.minimum.accumulate(raw_q[::-1])[::-1]
    raw_q = np.clip(raw_q, 0.0, 1.0)
    q.loc[ordered.index] = raw_q
    return q


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    r = pd.to_numeric(frame["r"], errors="coerce").dropna()
    n = int(len(r))
    if n == 0:
        return {
            "trades": 0,
            "win_rate": math.nan,
            "avg_r": math.nan,
            "net_r": 0.0,
            "profit_factor": math.nan,
            "max_dd": 0.0,
        }
    return {
        "trades": n,
        "win_rate": float((r > 0).mean()),
        "avg_r": float(r.mean()),
        "net_r": float(r.sum()),
        "profit_factor": profit_factor(r),
        "max_dd": max_drawdown(r),
    }


def session_state(hour: int) -> str:
    active: list[str] = []
    if 0 <= hour < 8:
        active.append("asia")
    if 7 <= hour < 16:
        active.append("london")
    if 13 <= hour < 22:
        active.append("ny")
    return "+".join(active) if active else "no_session"


def add_time_columns(frame: pd.DataFrame, time_col: str) -> pd.DataFrame:
    out = frame.copy()
    out["entry_time"] = pd.to_datetime(out[time_col], utc=True, errors="coerce")
    out = out[out["entry_time"].notna()].copy()
    out["day_of_week"] = out["entry_time"].dt.day_name()
    out["day_of_week_num"] = out["entry_time"].dt.dayofweek
    out["weekend"] = np.where(out["day_of_week_num"] >= 5, "weekend", "weekday")
    out["day_of_month"] = out["entry_time"].dt.day
    out["day_of_month_bin"] = pd.cut(
        out["day_of_month"],
        bins=[0, 7, 14, 21, 31],
        labels=["01-07", "08-14", "15-21", "22-EOM"],
        include_lowest=True,
    ).astype(str)
    out["hour_utc"] = out["entry_time"].dt.hour
    out["session_state"] = out["hour_utc"].map(session_state)
    out["dow_session_state"] = out["day_of_week"] + "|" + out["session_state"]
    out["weekend_session_state"] = out["weekend"] + "|" + out["session_state"]
    return out


def normalize_source(source: TradeSource) -> pd.DataFrame:
    raw = pd.read_csv(source.path)
    if source.loader:
        raw = source.loader(raw)
    if raw.empty:
        return pd.DataFrame()
    if source.r_col not in raw.columns:
        raise ValueError(f"{source.path} does not contain {source.r_col}")

    frame = add_time_columns(raw, source.time_col)
    frame["strategy"] = source.name
    frame["r"] = pd.to_numeric(frame[source.r_col], errors="coerce")
    frame = frame[frame["r"].notna()].copy()
    if source.sample_col and source.sample_col in frame.columns:
        frame["sample"] = frame[source.sample_col].astype(str).str.lower()
    else:
        frame["sample"] = np.where(frame["entry_time"] < SPLIT, "train", "oos")
    if "symbol" not in frame.columns:
        frame["symbol"] = source.name
    if "session" not in frame.columns:
        frame["session"] = frame["session_state"]
    frame["dow_session"] = frame["day_of_week"] + "|" + frame["session"].astype(str)
    frame["weekend_session"] = frame["weekend"] + "|" + frame["session"].astype(str)
    frame["source_notes"] = source.notes
    return frame


def load_sources() -> tuple[pd.DataFrame, dict[str, TradeSource]]:
    def orb_loader(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out = out[out["selection"].astype(str).eq("fixed")]
        out = out[out["symbol"].astype(str).isin(DEPLOYED_ORB_SYMBOLS)]
        return out

    def turtle_proxy_loader(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "trade_win_prob" in out.columns:
            out = out[pd.to_numeric(out["trade_win_prob"], errors="coerce") >= 0.50]
        return out

    def mm_top20_trail_loader(df: pd.DataFrame) -> pd.DataFrame:
        return df[df["variant"].astype(str).eq("trail")].copy()

    def mm_top20_dt_loader(df: pd.DataFrame) -> pd.DataFrame:
        out = df[df["variant"].astype(str).eq("dt")].copy()
        if "dt_accepted" in out.columns:
            out = out[out["dt_accepted"].astype(str).str.lower().isin({"true", "1"})]
        return out

    sources = [
        TradeSource(
            name="million_moves_top20_trail",
            path=ROOT / "million_moves_dt_time_filter_top20_trades.csv",
            r_col="r",
            sample_col="sample",
            min_oos_bucket=50,
            loader=mm_top20_trail_loader,
            notes="Exact configured MM top20 trail-only train/OOS export from pybit full history.",
        ),
        TradeSource(
            name="million_moves_top20_dt_oos",
            path=ROOT / "million_moves_dt_time_filter_top20_trades.csv",
            r_col="r",
            sample_col="sample",
            min_oos_bucket=25,
            loader=mm_top20_dt_loader,
            notes="Exact configured MM top20 accepted-DT OOS export; train-side DT rows are not persisted in this export.",
        ),
        TradeSource(
            name="session_orb_deployed_fixed",
            path=ROOT / "session_orb_top50_judas_fvg_risk2_v1_selected_trades.csv",
            r_col="r_multiple",
            sample_col="sample",
            min_oos_bucket=50,
            loader=orb_loader,
            notes="Exact deployed ORB symbols, fixed threshold 0.50.",
        ),
        TradeSource(
            name="turtle_core3_bfm_proxy_p050",
            path=ROOT / "advanced_feature_study_core3_bfm_v2_trade_event_features.csv",
            r_col="r_multiple",
            min_oos_bucket=15,
            loader=turtle_proxy_loader,
            notes=(
                "Proxy for Turtle timing: persisted core3 BFM feature set filtered at p>=0.50. "
                "Current top50 Turtle v4 selected trades were not persisted."
            ),
        ),
        TradeSource(
            name="turtle_core3_oos_selected_p050",
            path=ROOT / "trade_outcome_core3_1h_bfm_line_channel_p050_oos_trades.csv",
            r_col="r_multiple_net",
            min_oos_bucket=10,
            notes="OOS-only persisted Turtle selected trades for BTC/ETH/SOL p=0.50.",
        ),
        TradeSource(
            name="million_moves_wf_oos_reference",
            path=ROOT / "million_moves_v43_wf_oos_trades.csv",
            r_col="pnl_pct",
            min_oos_bucket=20,
            notes=(
                "Persisted Million Moves WFO OOS trade artifact. "
                "Uses pnl_pct because multi-coin per-trade R rows are not currently persisted."
            ),
        ),
    ]

    frames: list[pd.DataFrame] = []
    used: dict[str, TradeSource] = {}
    for source in sources:
        if not source.path.exists():
            continue
        frame = normalize_source(source)
        if not frame.empty:
            frames.append(frame)
            used[source.name] = source
    result = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    return result, used


def grouped_metrics(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows: list[dict] = []
    for (strategy, sample, bucket), group in frame.groupby(["strategy", "sample", group_col], dropna=False):
        row = {"strategy": strategy, "sample": sample, "dimension": group_col, "bucket": str(bucket)}
        row.update(metrics(group))
        rows.append(row)
    return pd.DataFrame(rows)


def diagnostics(frame: pd.DataFrame, group_col: str, sources: dict[str, TradeSource]) -> pd.DataFrame:
    rows: list[dict] = []
    for strategy, strategy_frame in frame.groupby("strategy"):
        min_n = sources[strategy].min_oos_bucket
        oos_all = strategy_frame[strategy_frame["sample"].eq("oos")]
        train_all = strategy_frame[strategy_frame["sample"].eq("train")]
        all_oos_m = metrics(oos_all)
        for bucket, group in oos_all.groupby(group_col, dropna=False):
            rest = oos_all[oos_all[group_col] != bucket]
            train_bucket = train_all[train_all[group_col] == bucket]
            train_rest = train_all[train_all[group_col] != bucket]
            oos_m = metrics(group)
            rest_m = metrics(rest)
            train_m = metrics(train_bucket)
            train_rest_m = metrics(train_rest)
            row = {
                "strategy": strategy,
                "dimension": group_col,
                "bucket": str(bucket),
                "min_oos_bucket": min_n,
                "oos_trades": oos_m["trades"],
                "oos_avg_r": oos_m["avg_r"],
                "oos_net_r": oos_m["net_r"],
                "oos_pf": oos_m["profit_factor"],
                "oos_win_rate": oos_m["win_rate"],
                "oos_max_dd": oos_m["max_dd"],
                "rest_oos_trades": rest_m["trades"],
                "rest_oos_avg_r": rest_m["avg_r"],
                "rest_oos_pf": rest_m["profit_factor"],
                "train_trades": train_m["trades"],
                "train_avg_r": train_m["avg_r"],
                "train_pf": train_m["profit_factor"],
                "train_rest_trades": train_rest_m["trades"],
                "train_rest_avg_r": train_rest_m["avg_r"],
                "train_rest_pf": train_rest_m["profit_factor"],
                "all_oos_avg_r": all_oos_m["avg_r"],
                "all_oos_pf": all_oos_m["profit_factor"],
                "p_mean_oos_welch": safe_welch_p(group["r"], rest["r"]),
                "p_rank_oos_mannwhitney": safe_mannwhitney_p(group["r"], rest["r"]),
                "p_win_oos_fisher": safe_fisher_win_p(group["r"], rest["r"]),
                "p_mean_train_welch": safe_welch_p(train_bucket["r"], train_rest["r"]),
                "p_win_train_fisher": safe_fisher_win_p(train_bucket["r"], train_rest["r"]),
            }
            row["delta_vs_rest_avg_r"] = (
                row["oos_avg_r"] - row["rest_oos_avg_r"]
                if pd.notna(row["oos_avg_r"]) and pd.notna(row["rest_oos_avg_r"])
                else math.nan
            )

            has_train = row["train_trades"] >= max(8, min_n // 2)
            reliable_n = row["oos_trades"] >= min_n
            if not reliable_n:
                row["verdict"] = "low_n"
            elif not has_train:
                row["verdict"] = "oos_only_watch"
            elif row["oos_avg_r"] < row["rest_oos_avg_r"] and row["train_avg_r"] < row["train_rest_avg_r"]:
                row["verdict"] = "candidate_block_unadjusted"
            elif row["oos_avg_r"] > row["rest_oos_avg_r"] and row["train_avg_r"] > row["train_rest_avg_r"]:
                row["verdict"] = "candidate_prefer_unadjusted"
            else:
                row["verdict"] = "watch"
            rows.append(row)
    return pd.DataFrame(rows)


def apply_multiple_testing(diag: pd.DataFrame) -> pd.DataFrame:
    out = diag.copy()
    for p_col in [
        "p_mean_oos_welch",
        "p_rank_oos_mannwhitney",
        "p_win_oos_fisher",
        "p_mean_train_welch",
        "p_win_train_fisher",
    ]:
        q_col = p_col.replace("p_", "q_")
        out[q_col] = np.nan
        for strategy, idx in out.groupby("strategy").groups.items():
            out.loc[idx, q_col] = bh_qvalues(out.loc[idx, p_col])

    block_mask = (
        out["verdict"].eq("candidate_block_unadjusted")
        & (out["q_mean_oos_welch"] <= 0.05)
        & (out["p_mean_train_welch"] <= 0.10)
    )
    prefer_mask = (
        out["verdict"].eq("candidate_prefer_unadjusted")
        & (out["q_mean_oos_welch"] <= 0.05)
        & (out["p_mean_train_welch"] <= 0.10)
    )
    out.loc[block_mask, "verdict"] = "candidate_block"
    out.loc[prefer_mask, "verdict"] = "candidate_prefer"
    return out


def symbol_drilldown(trades: pd.DataFrame, diag: pd.DataFrame) -> pd.DataFrame:
    candidates = diag[
        diag["verdict"].isin(["candidate_block", "candidate_prefer"])
        | ((diag["q_mean_oos_welch"] <= 0.05) & (diag["oos_trades"] >= diag["min_oos_bucket"]))
    ].copy()
    rows: list[dict] = []
    for _, candidate in candidates.iterrows():
        strategy = candidate["strategy"]
        dimension = candidate["dimension"]
        bucket = candidate["bucket"]
        if dimension not in trades.columns:
            continue
        subset = trades[trades["strategy"].eq(strategy)]
        for (sample, symbol), group_all in subset.groupby(["sample", "symbol"]):
            group = group_all[group_all[dimension].astype(str).eq(str(bucket))]
            rest = group_all[~group_all[dimension].astype(str).eq(str(bucket))]
            if group.empty or rest.empty:
                continue
            row = {
                "strategy": strategy,
                "sample": sample,
                "symbol": symbol,
                "dimension": dimension,
                "bucket": bucket,
            }
            gm = metrics(group)
            rm = metrics(rest)
            row.update({
                "bucket_trades": gm["trades"],
                "bucket_avg_r": gm["avg_r"],
                "bucket_net_r": gm["net_r"],
                "bucket_pf": gm["profit_factor"],
                "rest_trades": rm["trades"],
                "rest_avg_r": rm["avg_r"],
                "rest_pf": rm["profit_factor"],
                "delta_avg_r": gm["avg_r"] - rm["avg_r"],
                "p_mean_welch": safe_welch_p(group["r"], rest["r"]),
                "p_win_fisher": safe_fisher_win_p(group["r"], rest["r"]),
            })
            rows.append(row)
    return pd.DataFrame(rows)


def fmt_metric(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "n/a"
    if math.isinf(value):
        return "inf"
    return f"{value:.{digits}f}"


def write_report(
    trades: pd.DataFrame,
    overall: pd.DataFrame,
    diag: pd.DataFrame,
    symbol_tests: pd.DataFrame,
    sources: dict[str, TradeSource],
) -> None:
    lines: list[str] = []
    lines.append("# Strategy Time Filter Analysis")
    lines.append("")
    lines.append(f"Split used for train/OOS where not already labelled: `{SPLIT.date()}` UTC.")
    lines.append("")
    lines.append("## Sources")
    for name, source in sources.items():
        subset = trades[trades["strategy"].eq(name)]
        lines.append(
            f"- `{name}`: {len(subset)} rows from `{source.path.name}`. {source.notes}"
        )
    lines.append("")
    lines.append("## Overall")
    for _, row in overall.sort_values(["strategy", "sample"]).iterrows():
        lines.append(
            f"- `{row['strategy']}` {row['sample']}: "
            f"{int(row['trades'])} trades, net {fmt_metric(row['net_r'], 2)}, "
            f"avg {fmt_metric(row['avg_r'])}, PF {fmt_metric(row['profit_factor'])}, "
            f"win {fmt_metric(row['win_rate'] * 100, 1)}%, maxDD {fmt_metric(row['max_dd'], 2)}"
        )
    lines.append("")

    candidates = diag[diag["verdict"].isin(["candidate_block", "candidate_prefer"])].copy()
    candidates = candidates.sort_values(
        ["verdict", "strategy", "dimension", "delta_vs_rest_avg_r"],
        ascending=[True, True, True, True],
    )
    lines.append("## Strong Candidates")
    if candidates.empty:
        lines.append("- No cross-checked time bucket met the conservative candidate rule.")
    else:
        for _, row in candidates.iterrows():
            lines.append(
                f"- `{row['strategy']}` {row['verdict']} on `{row['dimension']}={row['bucket']}`: "
                f"OOS {int(row['oos_trades'])} trades avg {fmt_metric(row['oos_avg_r'])} "
                f"PF {fmt_metric(row['oos_pf'])}, q(mean) {fmt_metric(row['q_mean_oos_welch'])}; "
                f"train {int(row['train_trades'])} trades avg {fmt_metric(row['train_avg_r'])} "
                f"PF {fmt_metric(row['train_pf'])}, p(mean) {fmt_metric(row['p_mean_train_welch'])}."
            )
    lines.append("")

    rejected_sig = diag[
        (diag["q_mean_oos_welch"] <= 0.05)
        & (diag["oos_trades"] >= diag["min_oos_bucket"])
        & ~diag["verdict"].isin(["candidate_block", "candidate_prefer"])
    ].copy()
    rejected_sig = rejected_sig.sort_values(["strategy", "q_mean_oos_welch", "dimension", "bucket"])
    lines.append("## Significant But Rejected")
    if rejected_sig.empty:
        lines.append("- No OOS-significant bucket was rejected by the train/OOS consistency rule.")
    else:
        for _, row in rejected_sig.iterrows():
            if pd.isna(row["train_avg_r"]):
                reason = "no train-side sample for this source"
            elif (row["oos_avg_r"] - row["rest_oos_avg_r"]) * (row["train_avg_r"] - row["train_rest_avg_r"]) < 0:
                reason = "train and OOS point in opposite directions"
            else:
                reason = "train-side effect was not significant enough"
            lines.append(
                f"- `{row['strategy']}` `{row['dimension']}={row['bucket']}`: "
                f"OOS {int(row['oos_trades'])} trades avg {fmt_metric(row['oos_avg_r'])} "
                f"vs rest {fmt_metric(row['rest_oos_avg_r'])}, q(mean) {fmt_metric(row['q_mean_oos_welch'])}; "
                f"train avg {fmt_metric(row['train_avg_r'])} vs rest {fmt_metric(row['train_rest_avg_r'])}. Rejected: {reason}."
            )
    lines.append("")

    consistency_rows = pd.concat([candidates, rejected_sig], ignore_index=True) if not rejected_sig.empty else candidates
    if not consistency_rows.empty and not symbol_tests.empty:
        lines.append("## Symbol Consistency For Significant / Near-Significant Buckets")
        for _, row in consistency_rows.iterrows():
            subset = symbol_tests[
                symbol_tests["strategy"].eq(row["strategy"])
                & symbol_tests["dimension"].eq(row["dimension"])
                & symbol_tests["bucket"].astype(str).eq(str(row["bucket"]))
                & symbol_tests["sample"].eq("oos")
            ].copy()
            if subset.empty:
                continue
            block_like = row["oos_avg_r"] < row["rest_oos_avg_r"]
            if block_like:
                favorable = int((subset["delta_avg_r"] < 0).sum())
            else:
                favorable = int((subset["delta_avg_r"] > 0).sum())
            total = int(len(subset))
            top = subset.sort_values("delta_avg_r", ascending=block_like).head(5)
            detail = ", ".join(
                f"{r.symbol}:{fmt_metric(r.delta_avg_r)}({int(r.bucket_trades)}t)"
                for r in top.itertuples()
            )
            lines.append(
                f"- `{row['strategy']}` `{row['dimension']}={row['bucket']}`: "
                f"{favorable}/{total} OOS symbols agree on direction. {detail}"
            )
        lines.append("")

    lines.append("## OOS-Only Watchlist")
    watch = diag[
        (diag["verdict"].eq("oos_only_watch"))
        & (diag["oos_trades"] >= diag["min_oos_bucket"])
    ].copy()
    watch["badness"] = watch["oos_avg_r"] - watch["rest_oos_avg_r"]
    watch = watch.sort_values(["strategy", "badness"]).head(20)
    if watch.empty:
        lines.append("- No OOS-only time bucket passed the minimum sample threshold.")
    else:
        for _, row in watch.iterrows():
            lines.append(
                f"- `{row['strategy']}` `{row['dimension']}={row['bucket']}`: "
                f"OOS {int(row['oos_trades'])} trades avg {fmt_metric(row['oos_avg_r'])}, "
                f"rest avg {fmt_metric(row['rest_oos_avg_r'])}, PF {fmt_metric(row['oos_pf'])}."
            )
    lines.append("")

    lines.append("## Caveats")
    lines.append("- Day-of-month and exact-hour buckets are easy to overfit; prefer only buckets that repeat in train and OOS or across symbols.")
    lines.append("- The current Turtle v4 top50 selected-trade CSV was not available because the training run did not use `--write-scored`; the Turtle rows above include a persisted core3 proxy plus an OOS selected core3 artifact.")
    lines.append("- Million Moves top20 trail rows are now exact train/OOS exports. The accepted-DT source is OOS-only in this export, so it is watchlist material rather than a hard-filter source.")
    lines.append("")

    (OUT_PREFIX.with_name(f"{OUT_PREFIX.name}_report.md")).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    trades, sources = load_sources()
    if trades.empty:
        raise SystemExit("No trade sources found.")

    dimensions = [
        "day_of_week",
        "weekend",
        "day_of_month",
        "day_of_month_bin",
        "hour_utc",
        "session_state",
        "session",
        "dow_session_state",
        "weekend_session_state",
        "dow_session",
        "weekend_session",
    ]
    grouped = pd.concat([grouped_metrics(trades, d) for d in dimensions], ignore_index=True)
    diag = pd.concat([diagnostics(trades, d, sources) for d in dimensions], ignore_index=True)
    diag = apply_multiple_testing(diag)
    symbol_tests = symbol_drilldown(trades, diag)
    overall_rows = []
    for (strategy, sample), group in trades.groupby(["strategy", "sample"]):
        row = {"strategy": strategy, "sample": sample}
        row.update(metrics(group))
        overall_rows.append(row)
    overall = pd.DataFrame(overall_rows)

    trades.to_csv(OUT_PREFIX.with_name(f"{OUT_PREFIX.name}_normalized_trades.csv"), index=False)
    overall.to_csv(OUT_PREFIX.with_name(f"{OUT_PREFIX.name}_overall.csv"), index=False)
    grouped.to_csv(OUT_PREFIX.with_name(f"{OUT_PREFIX.name}_bucket_metrics.csv"), index=False)
    diag.to_csv(OUT_PREFIX.with_name(f"{OUT_PREFIX.name}_diagnostics.csv"), index=False)
    symbol_tests.to_csv(OUT_PREFIX.with_name(f"{OUT_PREFIX.name}_symbol_tests.csv"), index=False)
    write_report(trades, overall, diag, symbol_tests, sources)

    print(f"Loaded {len(trades)} normalized trades across {len(sources)} sources.")
    print(f"Wrote {OUT_PREFIX.name}_overall.csv")
    print(f"Wrote {OUT_PREFIX.name}_bucket_metrics.csv")
    print(f"Wrote {OUT_PREFIX.name}_diagnostics.csv")
    print(f"Wrote {OUT_PREFIX.name}_symbol_tests.csv")
    print(f"Wrote {OUT_PREFIX.name}_report.md")


if __name__ == "__main__":
    main()
