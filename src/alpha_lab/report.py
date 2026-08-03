"""Reads metrics.json, leaky_twin_metrics.json, shap_stability.json and
docs/badges.json, recomputes the time-series the dashboard needs from the
cached parquet outputs (never triggers a fresh pipeline run), fills
REPORT.md from a single Python template string, and writes docs/results.json.

Run as: python -m alpha_lab.report
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pandas as pd

from alpha_lab.config import (
    CV,
    DATA_PROCESSED_DIR,
    DOCS_DIR,
    EVAL_START_DATE,
    FEATURE_COLUMNS,
    IC_SUCCESS_THRESHOLD,
    IC_TSTAT_SUCCESS_THRESHOLD,
    INGEST_START_DATE,
    LGBM,
    REQUIRED_ENCODING,
    UNIVERSE,
)
from alpha_lab.evaluate import (
    METRICS_PATH,
    daily_rank_ic,
    rebalance_dates,
    top_bottom_period_returns,
)
from alpha_lab.features import FEATURES_PATH
from alpha_lab.leaky_twin import LEAKY_TWIN_METRICS_PATH
from alpha_lab.model import OOS_PREDICTIONS_PATH
from alpha_lab.explain import SHAP_IMPORTANCE_OVER_TIME_PATH, SHAP_STABILITY_PATH
from alpha_lab.neutralize import NEUTRALIZATION_PATH
from alpha_lab.regime import REGIME_BREAKDOWN_PATH
from alpha_lab.cpcv import CPCV_PATH
from alpha_lab.diagnose import DIAGNOSIS_PATH
from alpha_lab.util import atomic_write_json, atomic_write_text

RESULTS_PATH = DOCS_DIR / "results.json"
REPORT_PATH = DOCS_DIR.parent / "REPORT.md"
BADGES_PATH = DOCS_DIR / "badges.json"
PROVENANCE_PATH = DOCS_DIR.parent / "data" / "provenance.json"

ROLLING_IC_WINDOW = 63


def _load_json(path) -> dict | list | None:
    if not os.path.exists(path):
        return None
    with open(path, encoding=REQUIRED_ENCODING) as f:
        return json.load(f)


def _fmt(value, digits: int = 4, pct: bool = False, tbd: str = "TBD") -> str:
    if value is None:
        return tbd
    try:
        f = float(value)
    except (TypeError, ValueError):
        return tbd
    if f != f:  # NaN
        return tbd
    if pct:
        return f"{f * 100:.2f}%"
    return f"{f:.{digits}f}"


def _momentum_baseline_note(mean_ic: float | None, tstat: float | None) -> str:
    if mean_ic is None or tstat is None:
        return ""
    clears = mean_ic > IC_SUCCESS_THRESHOLD and tstat > IC_TSTAT_SUCCESS_THRESHOLD
    if clears:
        return (
            f"**The no-model 12-1 momentum baseline itself clears the "
            f"pre-registered bar** (mean rank IC {_fmt(mean_ic)} > {IC_SUCCESS_THRESHOLD}, "
            f"NW t-stat {_fmt(tstat, digits=2)} > {IC_TSTAT_SUCCESS_THRESHOLD}) while the "
            f"LightGBM model does not. This is the honest headline of the primary "
            f"result, not a footnote: on this universe and horizon, a single "
            f"engineered feature outperforms the full model."
        )
    return (
        f"The no-model 12-1 momentum baseline does not clear the pre-registered "
        f"bar either (mean rank IC {_fmt(mean_ic)}, NW t-stat {_fmt(tstat, digits=2)})."
    )


def _cum_curve(period_returns: pd.DataFrame, ret_col: str) -> tuple[list[str], list[float]]:
    if period_returns.empty:
        return [], []
    ordered = period_returns.sort_values("date")
    dates = [str(pd.Timestamp(d).date()) for d in ordered["date"]]
    cum = (1.0 + ordered[ret_col]).cumprod().tolist()
    return dates, [float(v) for v in cum]


def _rolling_ic_series(ic_series: pd.Series, window: int = ROLLING_IC_WINDOW) -> tuple[list[str], list[float | None]]:
    if ic_series.empty:
        return [], []
    ordered = ic_series.sort_index()
    rolling = ordered.rolling(window, min_periods=max(5, window // 4)).mean()
    dates = [str(pd.Timestamp(d).date()) for d in ordered.index]
    values = [None if v != v else float(v) for v in rolling.tolist()]
    return dates, values


def _per_fold_ic(oos: pd.DataFrame) -> list[dict]:
    rows = []
    for fold_id, group in oos.groupby("fold_id"):
        ic = daily_rank_ic(group, "y_pred")
        rows.append(
            {
                "fold_id": int(fold_id),
                "mean_ic": float(ic.mean()) if len(ic) else None,
                "n_days": int(len(ic)),
                "test_start": str(group["date"].min().date()),
                "test_end": str(group["date"].max().date()),
            }
        )
    rows.sort(key=lambda r: r["fold_id"])
    return rows


def build_dashboard_series() -> dict:
    """Recomputes chart-ready series from cached parquet outputs only.

    Never calls run_features/run_model with a cache miss: if the parquet
    files are not already on disk this returns an empty/unavailable series
    block instead of triggering ingestion or training as a side effect of
    building a report.
    """
    empty = {
        "available": False,
        "equity_curves": {"dates": [], "model_gross": [], "model_net": [], "momentum_gross": [], "buy_hold_gross": []},
        "rolling_ic": {"dates": [], "values": []},
        "per_fold_ic": [],
        "n_oos_rows": 0,
    }
    if not (os.path.exists(FEATURES_PATH) and os.path.exists(OOS_PREDICTIONS_PATH)):
        return empty

    features = pd.read_parquet(FEATURES_PATH)
    oos = pd.read_parquet(OOS_PREDICTIONS_PATH)

    momentum_col = "ret_252d_skip21"
    joined = oos.merge(features[["date", "ticker", momentum_col]], on=["date", "ticker"], how="left")

    rebal_ds = rebalance_dates(joined["date"])
    model_periods = top_bottom_period_returns(joined, pred_col="y_pred", rebalance_ds=rebal_ds)
    momentum_periods = top_bottom_period_returns(joined, pred_col=momentum_col, rebalance_ds=rebal_ds)

    dates_g, model_gross = _cum_curve(model_periods, "gross_ret")
    _, model_net = _cum_curve(model_periods, "net_ret")
    _, momentum_gross = _cum_curve(momentum_periods, "gross_ret")

    bh_rows = []
    for d in rebal_ds:
        day = joined[joined["date"] == d]
        if day.empty:
            continue
        bh_rows.append({"date": d, "gross_ret": float(day["fwd_ret_5d"].mean())})
    bh_df = pd.DataFrame(bh_rows)
    _, bh_gross = _cum_curve(bh_df, "gross_ret") if not bh_df.empty else ([], [])
    # summary stats (sharpe/drawdown/etc.) for buy-hold live in metrics.json / leaky_twin_metrics.json;
    # this block only rebuilds the cumulative curve for the dashboard chart.

    ic_series = daily_rank_ic(joined, "y_pred")
    roll_dates, roll_values = _rolling_ic_series(ic_series)

    per_fold_ic = _per_fold_ic(oos)

    return {
        "available": True,
        "equity_curves": {
            "dates": dates_g,
            "model_gross": model_gross,
            "model_net": model_net,
            "momentum_gross": momentum_gross,
            "buy_hold_gross": bh_gross,
        },
        "rolling_ic": {"dates": roll_dates, "values": roll_values},
        "per_fold_ic": per_fold_ic,
        "n_oos_rows": int(len(oos)),
    }


def build_results_json() -> dict:
    metrics = _load_json(METRICS_PATH)
    leaky = _load_json(LEAKY_TWIN_METRICS_PATH)
    shap_stability = _load_json(SHAP_STABILITY_PATH)
    badges = _load_json(BADGES_PATH) or []
    series = build_dashboard_series()

    # Post-hoc addendum artifacts. Each may not exist yet (this can run
    # before all supplementary analyses have been computed), so every read
    # here must degrade to None/empty rather than raising.
    neutralization = _load_json(NEUTRALIZATION_PATH)
    regime_breakdown = _load_json(REGIME_BREAKDOWN_PATH)
    shap_importance_over_time = _load_json(SHAP_IMPORTANCE_OVER_TIME_PATH)
    cpcv = _load_json(CPCV_PATH)
    provenance = _load_json(PROVENANCE_PATH)
    diagnosis = _load_json(DIAGNOSIS_PATH)

    run_completed = metrics is not None
    verdict_met = metrics["success_criterion"]["met"] if run_completed else None
    verdict_text = "TBD (pipeline not yet run)"
    if verdict_met is True:
        verdict_text = "PASS"
    elif verdict_met is False:
        verdict_text = "NULL"

    mean_abs_shap_sorted = []
    if shap_stability:
        per_fold = shap_stability["per_fold_mean_abs_shap"]
        final_key = str(shap_stability["final_fold_id"])
        final_mean_abs = per_fold.get(final_key) or per_fold[list(per_fold.keys())[-1]]
        ordered = sorted(final_mean_abs.items(), key=lambda kv: kv[1], reverse=True)
        mean_abs_shap_sorted = [{"feature": k, "mean_abs_shap": v} for k, v in ordered]

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_completed": run_completed,
        "data_span": {
            "ingest_start": INGEST_START_DATE,
            "eval_start": EVAL_START_DATE,
            "oos_date_range": metrics["oos_date_range"] if run_completed else None,
        },
        "verdict": {
            "met": verdict_met,
            "text": verdict_text,
            "ic_threshold": IC_SUCCESS_THRESHOLD,
            "tstat_threshold": IC_TSTAT_SUCCESS_THRESHOLD,
        },
        "model_metrics": metrics["model"] if run_completed else None,
        "momentum_baseline": metrics["momentum_baseline"] if run_completed else None,
        "buy_hold_baseline": metrics["buy_hold_baseline"] if run_completed else None,
        "leaky_twin": leaky,
        "ic_comparison": {
            "honest_mean_ic": metrics["model"]["rank_ic"]["mean"] if run_completed else None,
            "honest_tstat": metrics["model"]["rank_ic"]["newey_west_tstat"] if run_completed else None,
            "leaky_mean_ic": leaky["rank_ic"]["mean"] if leaky else None,
            "leaky_tstat": leaky["rank_ic"]["newey_west_tstat"] if leaky else None,
        },
        "momentum_clears_bar": (
            metrics["momentum_baseline"]["rank_ic"]["mean"] > IC_SUCCESS_THRESHOLD
            and metrics["momentum_baseline"]["rank_ic"]["newey_west_tstat"] > IC_TSTAT_SUCCESS_THRESHOLD
        ) if run_completed else None,
        "equity_curves": series["equity_curves"],
        "rolling_ic": series["rolling_ic"],
        "per_fold_ic": series["per_fold_ic"],
        "feature_importance": {
            "mean_abs_shap_final_fold": mean_abs_shap_sorted,
            "per_fold_rank": shap_stability["per_fold_rank"] if shap_stability else None,
            "top_feature_churns": shap_stability["top_feature_churns"] if shap_stability else None,
            "final_fold_id": shap_stability["final_fold_id"] if shap_stability else None,
        },
        "audits": badges,
        "cv_config": {
            "horizon_days": CV.horizon_days,
            "purge_days": CV.purge_days,
            "embargo_fraction": CV.embargo_fraction,
            "train_start": CV.train_start,
            "train_end": CV.train_end,
            "test_start": CV.test_start,
            "test_block_months": CV.test_block_months,
        },
        "lgbm_params": LGBM.as_params(),
        "feature_columns": FEATURE_COLUMNS,
        "addendum": {
            "note": (
                "Post-hoc supplementary analyses, computed after the primary "
                "pre-registered verdict above. See specs/methodology.md "
                "addendum section. None of these alter the primary verdict."
            ),
            "neutralization": neutralization,
            "regime_breakdown": regime_breakdown,
            "shap_importance_over_time": shap_importance_over_time,
            "cpcv": cpcv,
            "provenance": provenance,
            "diagnosis": diagnosis,
        },
    }
    return results


def _audit_receipt_table(badges: list[dict]) -> str:
    if not badges:
        return "No audit run recorded yet. Run `pytest tests/test_leakage_audit.py` to populate `docs/badges.json`."
    lines = ["| Check | Status | Description |", "|---|---|---|"]
    for b in badges:
        status = "PASS" if b["passed"] else "FAIL"
        lines.append(f"| {b['test']} | {status} | {b['description']} |")
    return "\n".join(lines)


def _render_addendum_section(addendum: dict) -> str:
    neut = addendum.get("neutralization")
    regime = addendum.get("regime_breakdown")
    shap_time = addendum.get("shap_importance_over_time")
    cpcv = addendum.get("cpcv")
    prov = addendum.get("provenance")

    parts = [
        "## Addendum: post-hoc supplementary analyses",
        "",
        "Everything in this section was computed after the primary verdict "
        "above was sealed. These are diagnostic/robustness checks on top of "
        "the already-generated OOS predictions, not a re-run or re-framing "
        "of the primary result; see the addendum in "
        "[`specs/methodology.md`](specs/methodology.md) for the full framing.",
        "",
    ]

    parts.append("### Feature neutralization vs 12-1 momentum")
    if neut:
        raw = neut["raw"]
        neutd = neut["neutralized"]
        still_null = neutd["mean"] is None or neutd["mean"] <= IC_SUCCESS_THRESHOLD or (
            neutd["newey_west_tstat"] is None or neutd["newey_west_tstat"] <= IC_TSTAT_SUCCESS_THRESHOLD
        )
        parts.append(
            f"\n| | Mean rank IC | Newey-West t-stat |\n|---|---|---|\n"
            f"| Raw predictions | {_fmt(raw['mean'])} | {_fmt(raw['newey_west_tstat'], digits=2)} |\n"
            f"| Neutralized vs momentum | {_fmt(neutd['mean'])} | {_fmt(neutd['newey_west_tstat'], digits=2)} |\n"
        )
        verdict_line = (
            "Residualizing against 12-1 momentum still leaves the signal below "
            "the pre-registered bar: this is still a NULL result after removing "
            "momentum overlap, not a different one."
            if still_null
            else "Residualizing against 12-1 momentum leaves a mean IC above the "
            "pre-registered bar. Reported here as a secondary finding worth a "
            "proper pre-registered follow-up study, not as an amendment to the "
            "primary NULL verdict, which stands as computed."
        )
        parts.append(verdict_line)
        ic_clears = neutd["mean"] is not None and neutd["mean"] > IC_SUCCESS_THRESHOLD
        t_clears = neutd["newey_west_tstat"] is not None and neutd["newey_west_tstat"] > IC_TSTAT_SUCCESS_THRESHOLD
        if still_null and t_clears and not ic_clears:
            parts.append(
                f"Noted for transparency: the neutralized NW t-stat "
                f"({_fmt(neutd['newey_west_tstat'], digits=2)}) crosses the "
                f"{IC_TSTAT_SUCCESS_THRESHOLD} bar on its own, but the IC "
                f"criterion ({_fmt(neutd['mean'])}) does not clear "
                f"{IC_SUCCESS_THRESHOLD}. The pre-registered rule requires "
                f"both (AND), so this remains NULL under that rule."
            )
    else:
        parts.append("TBD, run `python -m alpha_lab.neutralize`.")

    parts.append("\n### Regime and per-ticker breakdown")
    if regime:
        ric = regime["regime_rank_ic"]
        lines = ["\n| Regime | Mean rank IC | Newey-West t-stat | N days |", "|---|---|---|---|"]
        for name, stats in ric.items():
            lines.append(
                f"| {name} | {_fmt(stats['mean'])} | {_fmt(stats['newey_west_tstat'], digits=2)} | {stats['n_days']} |"
            )
        parts.append("\n".join(lines))
        ts = regime["per_ticker_ic_summary"]
        n_tickers = ts["n_tickers_total"]
        parts.append(
            f"\nAcross {n_tickers} names, per-ticker time-series rank IC has mean "
            f"{_fmt(ts['mean_across_tickers'])} (std {_fmt(ts['std_across_tickers'])}), "
            f"with {ts['n_tickers_with_positive_ic']}/{ts['n_tickers_total']} tickers "
            f"showing a positive IC. With only {n_tickers} names and no correction for "
            "multiple comparisons, this spread is consistent with noise around a "
            "small or zero true effect, and is reported as such rather than as a "
            "\"the signal works on ticker X\" claim."
        )
    else:
        parts.append("TBD, run `python -m alpha_lab.regime`.")

    parts.append("\n### Rolling per-fold SHAP importance over time")
    if shap_time and shap_time.get("rows"):
        parts.append(
            f"\n{len(shap_time['rows'])} folds x {len(shap_time['feature_columns'])} features written to "
            "`data/processed/shap_importance_over_time.json` for the dashboard's "
            "rolling-importance chart (full series omitted from this table for "
            "brevity)."
        )
    else:
        parts.append("TBD, run `python -m alpha_lab.explain`.")

    parts.append("\n### CPCV (Combinatorial Purged Cross-Validation) + simplified single-config PBO")
    if cpcv and cpcv.get("paths"):
        dist = cpcv["oos_ic_distribution"]
        pbo = cpcv["pbo"]
        ci_n_lo, ci_n_hi = dist["ci_95_normal"]
        ci_p_lo, ci_p_hi = dist["ci_95_percentile"]
        normal_excludes_zero = ci_n_lo is not None and ci_n_hi is not None and not (ci_n_lo <= 0.0 <= ci_n_hi)
        percentile_includes_zero = ci_p_lo is not None and ci_p_hi is not None and ci_p_lo <= 0.0 <= ci_p_hi
        mean_below_bar = dist["mean"] is not None and dist["mean"] <= IC_SUCCESS_THRESHOLD
        parts.append(
            f"\n{cpcv['n_paths_sampled']} of {cpcv['n_combinations_total']} possible "
            f"{cpcv['n_test_blocks_per_path']}-block combinations sampled (seed="
            f"{cpcv['seed']}), same frozen hyperparameters refit per path, runtime "
            f"{_fmt(cpcv['runtime_seconds'], digits=1)}s.\n\n"
            f"| | Value |\n|---|---|\n"
            f"| OOS rank IC across paths (mean) | {_fmt(dist['mean'])} |\n"
            f"| OOS rank IC across paths (std) | {_fmt(dist['std'])} |\n"
            f"| 95% CI (normal approx.) | [{_fmt(ci_n_lo)}, {_fmt(ci_n_hi)}] |\n"
            f"| 95% CI (percentile) | [{_fmt(ci_p_lo)}, {_fmt(ci_p_hi)}] |\n"
            f"| Simplified logit-PBO | {_fmt(pbo.get('logit_pbo'), digits=3)} (PBO={_fmt(pbo.get('pbo'), digits=3)}) |\n"
        )
        interval_note = (
            "The two 95% CIs disagree on whether zero is excluded (normal "
            "approximation excludes it, the percentile interval does not), "
            "which with only 20 paths and a right-skewed path distribution is "
            "itself a sign the interval estimate is not precise enough to lean "
            "on either way."
            if (normal_excludes_zero and percentile_includes_zero)
            else (
                "Both interval estimates exclude zero."
                if normal_excludes_zero and not percentile_includes_zero
                else "Both interval estimates include zero."
            )
        )
        parts.append(
            f"{interval_note} What both agree on: the mean OOS rank IC across "
            f"CPCV paths ({_fmt(dist['mean'])}) is "
            f"{'still below' if mean_below_bar else 'above'} the pre-registered "
            f"{IC_SUCCESS_THRESHOLD} bar. "
            + (
                "This is read as a stronger form of the same primary NULL "
                "finding: the single walk-forward path was not an unlucky low "
                "draw from a real positive distribution, since a distribution "
                "of comparable, equally valid splits does not clear the bar "
                "either. It is not a different or contradicting result."
                if mean_below_bar
                else "Reported here as a secondary finding for a proper "
                "pre-registered follow-up study; the primary pre-registered "
                "verdict above stands unchanged since it was computed on the "
                "one frozen walk-forward path specified before any result was "
                "seen."
            )
        )
        parts.append(f"\n{pbo.get('note', '')}")
    else:
        parts.append("TBD, run `python -m alpha_lab.cpcv`.")

    parts.append("\n### Why did the model underperform the momentum baseline? (diagnostic)")
    diag = addendum.get("diagnosis")
    if diag:
        corr = diag["feature_correlation"]
        shap_top = diag["shap_top_feature_per_fold"]
        mom_agree = diag["momentum_agreement"]
        ic_stab = diag["ic_stability"]
        tail_div = diag["tail_divergence"]
        turnover = diag["turnover_comparison"]
        hp_binding = diag["hyperparameter_binding"]

        pairs = corr.get("redundant_pairs") or []
        if pairs:
            pair_lines = ["\n| Feature A | Feature B | Spearman rho |", "|---|---|---|"]
            for p in pairs:
                pair_lines.append(f"| {p['feature_a']} | {p['feature_b']} | {_fmt(p['correlation'], 3)} |")
            parts.append("\n".join(pair_lines))
        else:
            parts.append("\nNo feature pairs at or above the redundancy threshold.")

        counts = shap_top.get("counts") or {}
        n_folds = len(shap_top.get("per_fold") or [])
        if counts and n_folds:
            tally_lines = ["\n| Feature | Folds as top SHAP feature |", "|---|---|"]
            for feat, cnt in sorted(counts.items(), key=lambda kv: -kv[1]):
                if cnt:
                    tally_lines.append(f"| {feat} | {cnt}/{n_folds} |")
            parts.append("\n".join(tally_lines))

        parts.append(
            f"\nModel-vs-momentum daily rank agreement (Spearman): mean "
            f"{_fmt(mom_agree['mean_daily_spearman'])}, median {_fmt(mom_agree['median_daily_spearman'])}, "
            f"std {_fmt(mom_agree['std_daily_spearman'])}, across {mom_agree['n_days']} days."
        )

        parts.append(
            f"\nFold-level IC stability: model std/|mean| {_fmt(ic_stab['model']['std_over_abs_mean'], 2)} vs "
            f"momentum's {_fmt(ic_stab['momentum']['std_over_abs_mean'], 2)}; cross-fold IC correlation "
            f"{_fmt(ic_stab['cross_fold_correlation'], 3)} across {ic_stab['n_folds']} folds."
        )

        parts.append(
            f"\nTail divergence: {_fmt(tail_div['pct_days_diverged'], digits=1, pct=True)} of "
            f"{tail_div['n_rebalance_days']} rebalance days diverge from momentum's tail picks "
            f"({tail_div['n_diverged_days']} days); on those days the model's realized tail spread is "
            f"{_fmt(tail_div['spread_ratio'], 1)}x smaller than momentum's, with a "
            f"{_fmt(tail_div['losing_swap_rate'], digits=1, pct=True)} losing-swap rate."
        )

        parts.append(
            f"\nTurnover: model {_fmt(turnover['model_mean_turnover'], digits=1, pct=True)} per rebalance vs "
            f"momentum's {_fmt(turnover['momentum_mean_turnover'], digits=1, pct=True)}."
        )

        parts.append(f"\n{hp_binding['note']}")

        summary = diag.get("summary") or []
        if summary:
            parts.append("\n" + "\n".join(f"- {line}" for line in summary))
    else:
        parts.append("\nTBD, run `python -m alpha_lab.diagnose`.")

    parts.append("\n### References")
    parts.append(
        "\n1. de Prado, *Advances in Financial Machine Learning* (2018): consistent with the "
        "substitution effect, where correlated features dilute each other's split importance in "
        "tree ensembles, as observed in the SHAP concentration above.\n"
        "2. Gu, Kelly, Xiu, \"Empirical Asset Pricing via Machine Learning\" (*Review of Financial "
        "Studies*, 2020): documents that ML gains over simple baselines typically require far "
        "larger cross-sections than the 31 names used in this project.\n"
        "3. Avramov, Cheng, Metzker, \"Machine Learning vs. Economic Restrictions\" (*Management "
        "Science*, 2023): documents that ML equity-return predictability concentrates in "
        "illiquid/small names and tends to evaporate on large caps, consistent with a large-cap "
        "universe being a harder setting for a model edge to show up in.\n"
        "4. Kelly, Malamud, Zhou, \"The Virtue of Complexity in Return Prediction\" (*Journal of "
        "Finance*, 2024), together with the Nagel (2025) and Buncic (2025) critiques: consistent "
        "with complex models often rediscovering a momentum-like signal with extra noise rather "
        "than something cleanly orthogonal to it.\n"
        "5. Alphalens/QuantRocket factor-evaluation documentation on positive-IC-but-negative-"
        "decile-spread as a recognized diagnostic pattern, consistent with the tail-spread "
        "divergence reported above.\n\n"
        "These are corroborating context for the pattern seen in this pipeline's own numbers "
        "above, not proof of the mechanism in this specific instance; this project's own "
        "diagnostics are the primary evidence."
    )

    parts.append("\n### Reproducibility manifest")
    if prov:
        pkgs = ", ".join(f"{k}={v}" for k, v in (prov.get("package_versions") or {}).items())
        fp = prov.get("run_fingerprint_sha256") or "N/A"
        parts.append(
            f"\nPython {prov.get('python_version', 'N/A').split()[0]}, packages: {pkgs}. "
            f"Random seed: {prov.get('random_seed')}. Run fingerprint (sha256 over "
            f"concatenated raw-input file hashes): `{fp[:16]}...`. Full manifest in "
            "`data/provenance.json`."
        )
    else:
        parts.append("TBD, written by `python -m alpha_lab.run_all` at the end of a full run.")

    return "\n".join(parts) + "\n"


def render_report_md(results: dict) -> str:
    run_completed = results["run_completed"]
    verdict = results["verdict"]
    model = results["model_metrics"]
    momentum = results["momentum_baseline"]
    buy_hold = results["buy_hold_baseline"]
    leaky = results["leaky_twin"]
    ic_cmp = results["ic_comparison"]
    fi = results["feature_importance"]

    if run_completed:
        model_rank_ic = model["rank_ic"]
        results_section = f"""
### Model (honest, PurgedWalkForward OOS)

| Metric | Value |
|---|---|
| Mean daily rank IC | {_fmt(model_rank_ic["mean"])} |
| Std daily rank IC | {_fmt(model_rank_ic["std"])} |
| Newey-West t-stat (lag=5) | {_fmt(model_rank_ic["newey_west_tstat"], digits=2)} |
| Hit rate (days with IC > 0) | {_fmt(model_rank_ic["hit_rate"], pct=True)} |
| N OOS days | {model_rank_ic["n_days"]} |
| Decile spread (annualized) | {_fmt(model["decile_spread"]["annualized_spread"], pct=True)} |
| Decile spread t-stat | {_fmt(model["decile_spread"]["tstat"], digits=2)} |
| Sharpe (gross) | {_fmt(model["equity_curve_gross"]["sharpe"], digits=2)} |
| Sharpe (net of costs) | {_fmt(model["equity_curve_net"]["sharpe"], digits=2)} |
| Max drawdown (gross) | {_fmt(model["equity_curve_gross"]["max_drawdown"], pct=True)} |
| Max drawdown (net) | {_fmt(model["equity_curve_net"]["max_drawdown"], pct=True)} |
| Mean turnover per rebalance | {_fmt(model["equity_curve_gross"]["mean_turnover"], pct=True)} |
| OOS date range | {results["data_span"]["oos_date_range"][0]} .. {results["data_span"]["oos_date_range"][1]} |

### 12-1 momentum baseline

| Metric | Value |
|---|---|
| Mean daily rank IC | {_fmt(momentum["rank_ic"]["mean"])} |
| Newey-West t-stat | {_fmt(momentum["rank_ic"]["newey_west_tstat"], digits=2)} |
| Decile spread (annualized) | {_fmt(momentum["decile_spread"]["annualized_spread"], pct=True)} |
| Sharpe (gross) | {_fmt(momentum["equity_curve_gross"]["sharpe"], digits=2)} |

{_momentum_baseline_note(momentum["rank_ic"]["mean"], momentum["rank_ic"]["newey_west_tstat"])}

### Equal-weight buy-and-hold baseline

| Metric | Value |
|---|---|
| Annualized return | {_fmt(buy_hold["equity_curve_gross"]["annualized_return"], pct=True)} |
| Sharpe | {_fmt(buy_hold["equity_curve_gross"]["sharpe"], digits=2)} |
| Max drawdown | {_fmt(buy_hold["equity_curve_gross"]["max_drawdown"], pct=True)} |

### Leaky-twin comparison (naive shuffled 5-fold CV, no purge/embargo, same OOS date range as the honest model)

| Metric | Honest (PurgedWalkForward) | Leaky twin |
|---|---|---|
| Mean daily rank IC | {_fmt(ic_cmp["honest_mean_ic"])} | {_fmt(ic_cmp["leaky_mean_ic"])} |
| Newey-West t-stat | {_fmt(ic_cmp["honest_tstat"], digits=2)} | {_fmt(ic_cmp["leaky_tstat"], digits=2)} |

The gap between these two rows is the headline finding: it is the amount of
apparent edge that a naive, non-purged backtest would report purely from
leakage, with everything else (features, model, target, hyperparameters)
held identical.
"""
    else:
        results_section = "\nPipeline has not been run in this session yet. All numeric fields are TBD; run `python -m alpha_lab.run_all` to populate them.\n"

    if fi["mean_abs_shap_final_fold"]:
        shap_lines = ["| Feature | Mean |SHAP| (final fold, OOS) |", "|---|---|"]
        for row in fi["mean_abs_shap_final_fold"]:
            shap_lines.append(f"| {row['feature']} | {_fmt(row['mean_abs_shap'], digits=5)} |")
        shap_table = "\n".join(shap_lines)
        churn_note = (
            "The top feature by mean |SHAP| churns across CV folds. This is reported "
            "as-is, not smoothed over: fold-to-fold instability in feature ranking is "
            "itself informative about how robust the signal is."
            if fi["top_feature_churns"]
            else "The top feature by mean |SHAP| is stable across CV folds."
        )
    else:
        shap_table = "TBD, run `python -m alpha_lab.explain`."
        churn_note = "TBD."

    audits = _audit_receipt_table(results["audits"])
    addendum_section = _render_addendum_section(results.get("addendum") or {})

    template = f"""# REPORT: ML Alpha Signal Research Pipeline

Auto-generated by `src/alpha_lab/report.py`. Do not hand-edit; rerun the
orchestrator (`python -m alpha_lab.run_all`) and this file is rewritten.

Generated at: {results["generated_at"]}

## Question

Does gradient boosting (LightGBM) on 12 standard price/volume features add
predictive power over plain momentum and buy-and-hold, on 5-day
cross-sectional excess returns for {len(UNIVERSE)} large-cap US equities, under
leakage-proof purged walk-forward evaluation?

## Pre-registration

The full frozen methodology, written before any model was trained or result
seen, is in [`specs/methodology.md`](specs/methodology.md). This report does
not deviate from it; any post-hoc finding is logged here as a limitation, not
folded back into the spec.

## Pre-registered success criterion

```
mean(daily rank IC) > {IC_SUCCESS_THRESHOLD}   AND   Newey-West t-stat(daily rank IC) > {IC_TSTAT_SUCCESS_THRESHOLD}
```

Both conditions must hold on the honest OOS predictions. Anything short of
that is reported as a NULL result, not reframed as a partial success.

## Verdict

**{verdict["text"]}**

## Data & known biases

- Universe: {len(UNIVERSE)} hand-picked US large/mega-cap names, liquid since 2010,
  plus `^SPX` as benchmark context only (not traded). (The frozen spec's prose
  says "30 names"; the actual list has always had {len(UNIVERSE)}. This is a
  descriptive miscount in the spec text, not a change to the universe itself,
  see the addendum erratum in `specs/methodology.md`.)
- Source: Stooq primary, yfinance fallback, 3-ticker cross-check logged in
  `data/manifest.json`.
- Ingest from {INGEST_START_DATE}, evaluation from {EVAL_START_DATE} through
  the latest complete month at run time.
- Survivorship bias is explicit: "liquid since 2010" is an outcome we can
  only observe today.

## Method

PurgedWalkForward, expanding window, purge = {CV.purge_days} trading days,
embargo = 1% of sample length, first train window
{CV.train_start}..{CV.train_end}, {CV.test_block_months}-month walk-forward
test blocks from {CV.test_start}. LightGBM regressor with fixed
hyperparameters (`num_leaves={LGBM.num_leaves}`, `max_depth={LGBM.max_depth}`,
`learning_rate={LGBM.learning_rate}`, `n_estimators={LGBM.n_estimators}`,
`min_child_samples={LGBM.min_child_samples}`,
`feature_fraction={LGBM.feature_fraction}`,
`bagging_fraction={LGBM.bagging_fraction}`, `seed={LGBM.seed}`), one model
fit per fold, no hyperparameter tuning.

## Results vs baselines
{results_section}
## Feature importance (SHAP, final fold, OOS test rows only)

{shap_table}

{churn_note}

## Leakage audit receipt

{audits}

All eight checks are implemented in `tests/test_leakage_audit.py` and run
against a committed synthetic fixture (`tests/fixtures/`), no network calls,
in CI (`.github/workflows/ci.yml`).

## Known limitations

- Survivorship bias in the hand-picked {len(UNIVERSE)}-name universe.
- Single asset class (US large-cap equities); no cross-asset generalization
  claim.
- Fixed hyperparameters, not tuned: avoids an obvious leakage vector at the
  cost of not representing a best-effort model.
- Flat 10 bps per side transaction cost model; real costs vary by name, size,
  and market regime.
- {len(UNIVERSE)} names is a small cross-section for daily rank IC and
  decile-spread statistics.

## What would change the answer

- A materially larger, point-in-time-correct universe (removing survivorship
  bias) with the same audit harness.
- Multiple horizons (1-day, 10-day, 21-day) tested with the same
  pre-registration discipline, to see if H=5 is a cherry-picked horizon.
- CPCV (combinatorial purged cross-validation) instead of a single
  walk-forward path, to get a distribution of OOS Sharpe rather than one
  realization.
- A more realistic cost model (size-dependent slippage, borrow cost for
  shorts) instead of the flat 10 bps assumption.

{addendum_section}
## Disclaimer

This is research into a methodology and audit harness, not a trading system
or investment recommendation. Nothing here constitutes financial advice.
"""
    return template


def _sanitize_for_json(obj):
    """NaN/Infinity are valid Python float literals but not valid JSON;
    json.dump's default allow_nan=True emits them anyway, which breaks
    JSON.parse in the browser and would crash the dashboard on any
    degenerate metric. Recursively convert them to None instead."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
        return None
    return obj


def run_report(force: bool = False) -> dict:
    if not force and os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, encoding=REQUIRED_ENCODING) as f:
            return json.load(f)

    results = build_results_json()

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    sanitized_results = _sanitize_for_json(results)
    atomic_write_json(RESULTS_PATH, sanitized_results, encoding=REQUIRED_ENCODING, indent=2, allow_nan=False)
    print(f"wrote {RESULTS_PATH}")

    report_md = render_report_md(results)
    atomic_write_text(REPORT_PATH, report_md, encoding=REQUIRED_ENCODING)
    print(f"wrote {REPORT_PATH}")
    print(f"verdict: {results['verdict']['text']}")
    return results


def main() -> None:
    run_report(force=True)


if __name__ == "__main__":
    main()
