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
)
from alpha_lab.evaluate import (
    METRICS_PATH,
    buy_hold_baseline,
    daily_rank_ic,
    rebalance_dates,
    top_bottom_period_returns,
)
from alpha_lab.features import FEATURES_PATH
from alpha_lab.leaky_twin import LEAKY_TWIN_METRICS_PATH
from alpha_lab.model import OOS_PREDICTIONS_PATH
from alpha_lab.explain import SHAP_STABILITY_PATH

RESULTS_PATH = DOCS_DIR / "results.json"
REPORT_PATH = DOCS_DIR.parent / "REPORT.md"
BADGES_PATH = DOCS_DIR / "badges.json"

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

    bh = buy_hold_baseline(joined)
    bh_rows = []
    for d in rebal_ds:
        day = joined[joined["date"] == d]
        if day.empty:
            continue
        bh_rows.append({"date": d, "gross_ret": float(day["fwd_ret_5d"].mean())})
    bh_df = pd.DataFrame(bh_rows)
    _, bh_gross = _cum_curve(bh_df, "gross_ret") if not bh_df.empty else ([], [])
    del bh  # summary stats already in metrics.json / leaky_twin_metrics.json

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

### Equal-weight buy-and-hold baseline

| Metric | Value |
|---|---|
| Annualized return | {_fmt(buy_hold["equity_curve_gross"]["annualized_return"], pct=True)} |
| Sharpe | {_fmt(buy_hold["equity_curve_gross"]["sharpe"], digits=2)} |
| Max drawdown | {_fmt(buy_hold["equity_curve_gross"]["max_drawdown"], pct=True)} |

### Leaky-twin comparison (naive shuffled 5-fold CV, no purge/embargo)

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

    template = f"""# REPORT: ML Alpha Signal Research Pipeline

Auto-generated by `src/alpha_lab/report.py`. Do not hand-edit; rerun the
orchestrator (`python -m alpha_lab.run_all`) and this file is rewritten.

Generated at: {results["generated_at"]}

## Question

Does gradient boosting (LightGBM) on 12 standard price/volume features add
predictive power over plain momentum and buy-and-hold, on 5-day
cross-sectional excess returns for 30 large-cap US equities, under
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

- Universe: 30 hand-picked US large/mega-cap names, liquid since 2010, plus
  `^SPX` as benchmark context only (not traded).
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

- Survivorship bias in the hand-picked 30-name universe.
- Single asset class (US large-cap equities); no cross-asset generalization
  claim.
- Fixed hyperparameters, not tuned: avoids an obvious leakage vector at the
  cost of not representing a best-effort model.
- Flat 10 bps per side transaction cost model; real costs vary by name, size,
  and market regime.
- 30 names is a small cross-section for daily rank IC and decile-spread
  statistics.

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

## Disclaimer

This is research into a methodology and audit harness, not a trading system
or investment recommendation. Nothing here constitutes financial advice.
"""
    return template


def run_report(force: bool = False) -> dict:
    if not force and os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, encoding=REQUIRED_ENCODING) as f:
            return json.load(f)

    results = build_results_json()

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding=REQUIRED_ENCODING) as f:
        json.dump(results, f, indent=2)
    print(f"wrote {RESULTS_PATH}")

    report_md = render_report_md(results)
    with open(REPORT_PATH, "w", encoding=REQUIRED_ENCODING) as f:
        f.write(report_md)
    print(f"wrote {REPORT_PATH}")
    print(f"verdict: {results['verdict']['text']}")
    return results


def main() -> None:
    run_report(force=True)


if __name__ == "__main__":
    main()
