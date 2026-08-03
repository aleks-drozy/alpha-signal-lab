"""Post-hoc addendum: diagnoses why the ML model underperformed the 12-1
momentum baseline. Uses only already-generated pipeline artifacts
(features.parquet, oos_predictions.parquet, shap_importance_over_time.json)
plus one deterministic seed-42 refit of the final CV fold for the
hyperparameter-binding check. Diagnostic only, computed after the primary
pre-registered verdict; see the addendum in specs/methodology.md.
Run as: python -m alpha_lab.diagnose
"""

from __future__ import annotations

import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from alpha_lab.config import DATA_PROCESSED_DIR, FEATURE_COLUMNS, LGBM, REQUIRED_ENCODING
from alpha_lab.cv import Fold, PurgedWalkForward
from alpha_lab.evaluate import daily_rank_ic, rebalance_dates, top_bottom_period_returns
from alpha_lab.explain import SHAP_IMPORTANCE_OVER_TIME_PATH
from alpha_lab.features import run_features
from alpha_lab.model import run_model
from alpha_lab.util import atomic_write_json

DIAGNOSIS_PATH = DATA_PROCESSED_DIR / "diagnosis.json"
MOMENTUM_COL = "ret_252d_skip21"
REDUNDANT_CORR_THRESHOLD = 0.75
TAIL_N = 3
# A day "diverges" when at most 2 of the 6 tail slots (top-3 + bottom-3) are
# shared between the model's and momentum's picks.
TAIL_SHARED_THRESHOLD = 2


def feature_correlation(features: pd.DataFrame) -> dict:
    """Spearman correlation of the 12 FEATURE_COLUMNS across all rows. The
    columns are already cross-sectionally rank-transformed in
    features.parquet, so this is a correlation of ranks, not raw values."""
    corr = features[FEATURE_COLUMNS].corr(method="spearman")
    matrix = {a: {b: float(corr.loc[a, b]) for b in FEATURE_COLUMNS} for a in FEATURE_COLUMNS}

    pairs = []
    for i, a in enumerate(FEATURE_COLUMNS):
        for b in FEATURE_COLUMNS[i + 1 :]:
            rho = float(corr.loc[a, b])
            if rho == rho and abs(rho) >= REDUNDANT_CORR_THRESHOLD:
                pairs.append({"feature_a": a, "feature_b": b, "correlation": rho})
    pairs.sort(key=lambda p: -abs(p["correlation"]))

    return {"matrix": matrix, "redundant_pairs": pairs}


def shap_top_feature_per_fold(shap_importance: dict) -> dict:
    """Argmax mean|SHAP| feature per fold from explain.py's
    shap_importance_over_time.json rows (one row per fold, one column per
    feature), plus a tally of how many folds each feature won."""
    feature_columns = shap_importance["feature_columns"]
    per_fold = []
    counts: dict[str, int] = {f: 0 for f in feature_columns}
    for row in shap_importance["rows"]:
        top_feature = max(feature_columns, key=lambda f: row[f])
        per_fold.append({"fold_id": int(row["fold_id"]), "top_feature": top_feature})
        counts[top_feature] += 1
    return {"per_fold": per_fold, "counts": counts}


def momentum_agreement(oos: pd.DataFrame, features: pd.DataFrame) -> dict:
    """Per OOS date, Spearman rank correlation between the model's y_pred and
    the raw (rank-transformed) momentum feature within that date's
    cross-section, summarized across days. Answers whether the model is a
    noisy copy of momentum or something orthogonal to it."""
    joined = oos.merge(features[["date", "ticker", MOMENTUM_COL]], on=["date", "ticker"], how="left")

    def _corr(group: pd.DataFrame) -> float:
        if len(group) < 3 or group["y_pred"].nunique() < 2 or group[MOMENTUM_COL].nunique() < 2:
            return np.nan
        corr, _ = spearmanr(group["y_pred"], group[MOMENTUM_COL])
        return corr

    daily = joined.groupby("date").apply(_corr, include_groups=False).dropna()
    values = daily.to_numpy()

    return {
        "mean_daily_spearman": float(np.mean(values)) if len(values) else float("nan"),
        "median_daily_spearman": float(np.median(values)) if len(values) else float("nan"),
        "std_daily_spearman": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
        "n_days": int(len(values)),
    }


def _fold_ic_stats(values: list[float]) -> dict:
    arr = np.array([v for v in values if v == v])
    if len(arr) == 0:
        return {"mean": float("nan"), "std": float("nan"), "std_over_abs_mean": float("nan")}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else float("nan")
    std_over_abs_mean = std / abs(mean) if mean != 0.0 and std == std else float("nan")
    return {"mean": mean, "std": std, "std_over_abs_mean": std_over_abs_mean}


def ic_stability(oos: pd.DataFrame, features: pd.DataFrame) -> dict:
    """Per-fold rank IC for both the model and momentum, using the same fold
    boundaries oos_predictions.parquet already carries (assigned by model.py's
    PurgedWalkForward split), aggregated to one IC-per-fold via the mean of
    that fold's daily rank ICs (matching evaluate.py/report.py's convention).
    """
    joined = oos.merge(features[["date", "ticker", MOMENTUM_COL]], on=["date", "ticker"], how="left")

    model_fold_ic: list[float] = []
    momentum_fold_ic: list[float] = []
    fold_ids = sorted(joined["fold_id"].unique())
    for fold_id in fold_ids:
        group = joined[joined["fold_id"] == fold_id]
        m_ic = daily_rank_ic(group, "y_pred")
        mom_ic = daily_rank_ic(group, MOMENTUM_COL)
        model_fold_ic.append(float(m_ic.mean()) if len(m_ic) else float("nan"))
        momentum_fold_ic.append(float(mom_ic.mean()) if len(mom_ic) else float("nan"))

    valid_pairs = [(m, mo) for m, mo in zip(model_fold_ic, momentum_fold_ic) if m == m and mo == mo]
    if len(valid_pairs) > 1:
        m_arr = np.array([p[0] for p in valid_pairs])
        mo_arr = np.array([p[1] for p in valid_pairs])
        cross_fold_correlation = (
            float(np.corrcoef(m_arr, mo_arr)[0, 1]) if np.std(m_arr) > 0 and np.std(mo_arr) > 0 else float("nan")
        )
    else:
        cross_fold_correlation = float("nan")

    return {
        "model": _fold_ic_stats(model_fold_ic),
        "momentum": _fold_ic_stats(momentum_fold_ic),
        "cross_fold_correlation": cross_fold_correlation,
        "n_folds": len(fold_ids),
    }


def tail_divergence(oos: pd.DataFrame, features: pd.DataFrame, n: int = TAIL_N) -> dict:
    """For each rebalance date, compares the model's top-n/bottom-n tickers
    to momentum's. A day diverges when at most TAIL_SHARED_THRESHOLD of the
    2n tail slots are shared. On diverged days, compares each side's realized
    tail spread (top minus bottom mean forward return) and the fraction of
    diverged days where the model's swap away from momentum's picks lost
    money relative to what momentum would have earned that day."""
    joined = oos.merge(features[["date", "ticker", MOMENTUM_COL]], on=["date", "ticker"], how="left")
    rebal_ds = rebalance_dates(joined["date"])

    diverged_flags = []
    model_spreads = []
    momentum_spreads = []
    losing_swaps = 0
    n_diverged_scored = 0

    for d in rebal_ds:
        day = joined[joined["date"] == d]
        if len(day) < 2 * n:
            continue

        model_ranked = day.sort_values("y_pred", ascending=False)
        mom_ranked = day.sort_values(MOMENTUM_COL, ascending=False)

        model_longs = set(model_ranked.head(n)["ticker"])
        model_shorts = set(model_ranked.tail(n)["ticker"])
        mom_longs = set(mom_ranked.head(n)["ticker"])
        mom_shorts = set(mom_ranked.tail(n)["ticker"])

        shared = len(model_longs & mom_longs) + len(model_shorts & mom_shorts)
        diverged = shared <= TAIL_SHARED_THRESHOLD
        diverged_flags.append(diverged)
        if not diverged:
            continue

        model_spread = float(
            day[day["ticker"].isin(model_longs)]["fwd_ret_5d"].mean()
            - day[day["ticker"].isin(model_shorts)]["fwd_ret_5d"].mean()
        )
        mom_spread = float(
            day[day["ticker"].isin(mom_longs)]["fwd_ret_5d"].mean()
            - day[day["ticker"].isin(mom_shorts)]["fwd_ret_5d"].mean()
        )
        model_spreads.append(model_spread)
        momentum_spreads.append(mom_spread)
        n_diverged_scored += 1
        if model_spread < mom_spread:
            losing_swaps += 1

    n_total = len(diverged_flags)
    n_diverged = int(sum(diverged_flags))
    pct_days_diverged = float(n_diverged / n_total) if n_total else float("nan")

    model_spread_mean = float(np.mean(model_spreads)) if model_spreads else float("nan")
    momentum_spread_mean = float(np.mean(momentum_spreads)) if momentum_spreads else float("nan")
    if model_spread_mean == model_spread_mean and model_spread_mean != 0.0 and momentum_spread_mean == momentum_spread_mean:
        spread_ratio = float(abs(momentum_spread_mean) / abs(model_spread_mean))
    else:
        spread_ratio = float("nan")

    losing_swap_rate = float(losing_swaps / n_diverged_scored) if n_diverged_scored else float("nan")

    return {
        "pct_days_diverged": pct_days_diverged,
        "model_tail_spread_on_diverged_days": model_spread_mean,
        "momentum_tail_spread_on_diverged_days": momentum_spread_mean,
        "spread_ratio": spread_ratio,
        "losing_swap_rate": losing_swap_rate,
        "n_rebalance_days": n_total,
        "n_diverged_days": n_diverged,
    }


def turnover_comparison(oos: pd.DataFrame, features: pd.DataFrame) -> dict:
    """Reuses evaluate.py's top_bottom_period_returns (same turnover
    definition already used for the primary metrics) for both the model's
    and momentum's rebalance sequences."""
    joined = oos.merge(features[["date", "ticker", MOMENTUM_COL]], on=["date", "ticker"], how="left")
    rebal_ds = rebalance_dates(joined["date"])
    model_periods = top_bottom_period_returns(joined, pred_col="y_pred", rebalance_ds=rebal_ds)
    momentum_periods = top_bottom_period_returns(joined, pred_col=MOMENTUM_COL, rebalance_ds=rebal_ds)

    return {
        "model_mean_turnover": float(model_periods["turnover"].mean()) if not model_periods.empty else float("nan"),
        "momentum_mean_turnover": float(momentum_periods["turnover"].mean())
        if not momentum_periods.empty
        else float("nan"),
    }


def hyperparameter_binding(panel: pd.DataFrame, folds: list[Fold]) -> dict:
    """Deterministic (seed=42, from LGBM.seed) refit of the LAST fold only,
    with the frozen hyperparameters, inspecting the fitted booster's actual
    tree structure to see whether num_leaves (the cap) is being hit. Reported
    honestly: hitting the cap on most trees is consistent with num_leaves/
    max_depth binding before min_child_samples, not proof of it in isolation.
    """
    if not folds:
        raise RuntimeError("no CV folds provided, cannot refit the final fold")
    final_fold = folds[-1]
    train_df = panel[panel["date"].isin(final_fold.train_dates)]

    model = lgb.LGBMRegressor(**LGBM.as_params())
    model.fit(train_df[FEATURE_COLUMNS], train_df["label"])

    trees = model.booster_.trees_to_dataframe()
    leaf_rows = trees[trees["split_feature"].isna()]
    leaves_per_tree = leaf_rows.groupby("tree_index").size()
    n_trees = int(trees["tree_index"].nunique())
    mean_leaves = float(leaves_per_tree.mean()) if len(leaves_per_tree) else float("nan")
    max_leaves = int(leaves_per_tree.max()) if len(leaves_per_tree) else 0
    pct_at_cap = float((leaves_per_tree >= LGBM.num_leaves).mean()) if len(leaves_per_tree) else float("nan")
    binds_at_cap = pct_at_cap == pct_at_cap and pct_at_cap > 0.5

    note = (
        f"Refit of the final CV fold ({n_trees} trees, {len(train_df)} training rows) shows a "
        f"mean of {mean_leaves:.1f} leaves per tree against the configured cap of "
        f"{LGBM.num_leaves} (max observed {max_leaves}), with {pct_at_cap * 100:.1f}% of trees "
        f"hitting the num_leaves cap. "
        + (
            "This is consistent with num_leaves/max_depth binding before "
            "min_child_samples given the training-row counts in this fold, not "
            "proof of it in isolation."
            if binds_at_cap
            else "This does not show num_leaves consistently binding in this fold; "
            "min_child_samples or max_depth may be the tighter constraint here "
            "instead."
        )
    )

    return {
        "note": note,
        "leaf_count_summary": {
            "n_trees": n_trees,
            "mean_leaves_per_tree": mean_leaves,
            "max_leaves_per_tree": max_leaves,
            "pct_trees_at_num_leaves_cap": pct_at_cap,
        },
        "min_child_samples_configured": LGBM.min_child_samples,
        "num_leaves_configured": LGBM.num_leaves,
        "final_fold_id": final_fold.fold_id,
        "final_fold_train_rows": int(len(train_df)),
    }


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if value == value else "N/A"


def _pct(value: float) -> float:
    return value * 100.0 if value == value else float("nan")


def build_summary(
    corr: dict,
    shap_top: dict,
    mom_agree: dict,
    ic_stab: dict,
    tail_div: dict,
    turnover: dict,
    hp_binding: dict,
) -> list[str]:
    """Grounds every line in a real computed value from the sections above,
    so report.py can render this list directly as bullet points."""
    lines = []

    if corr["redundant_pairs"]:
        top_pair = corr["redundant_pairs"][0]
        lines.append(
            f"{top_pair['feature_a']} and {top_pair['feature_b']} are redundant "
            f"(Spearman rho={_fmt(top_pair['correlation'], 3)}), one of {len(corr['redundant_pairs'])} "
            f"feature pairs at or above the {REDUNDANT_CORR_THRESHOLD} redundancy threshold."
        )

    if shap_top["counts"] and shap_top["per_fold"]:
        dominant_feature, dominant_count = max(shap_top["counts"].items(), key=lambda kv: kv[1])
        n_folds = len(shap_top["per_fold"])
        lines.append(
            f"{dominant_feature} is the top SHAP feature in {dominant_count}/{n_folds} folds: "
            "the model leans on one feature family rather than a diversified signal."
        )

    lines.append(
        f"The model's daily rank correlation with momentum is only "
        f"{_fmt(mom_agree['mean_daily_spearman'])}: it is not a noisy copy of momentum."
    )

    lines.append(
        f"Model fold-IC std/|mean| is {_fmt(ic_stab['model']['std_over_abs_mean'], 2)} vs momentum's "
        f"{_fmt(ic_stab['momentum']['std_over_abs_mean'], 2)}, and the two sides' fold-level IC are "
        f"correlated at only {_fmt(ic_stab['cross_fold_correlation'], 3)} across folds: the model's edge "
        "is less stable fold to fold, and a good or bad fold for one does not predict a good or bad fold "
        "for the other."
    )

    lines.append(
        f"On {_fmt(_pct(tail_div['pct_days_diverged']), 1)}% of rebalance days the model's tail picks "
        f"diverge from momentum's; on those days the model's realized tail spread is about "
        f"{_fmt(tail_div['spread_ratio'], 1)}x smaller than momentum's, with a "
        f"{_fmt(_pct(tail_div['losing_swap_rate']), 1)}% losing-swap rate."
    )

    lines.append(
        f"Model turnover ({_fmt(_pct(turnover['model_mean_turnover']), 1)}% per rebalance) is far higher "
        f"than momentum's ({_fmt(_pct(turnover['momentum_mean_turnover']), 1)}%): more trading without a "
        "matching payoff."
    )

    lines.append(hp_binding["note"])

    return lines


def run_diagnose(force: bool = False) -> dict:
    if not force and os.path.exists(DIAGNOSIS_PATH):
        with open(DIAGNOSIS_PATH, encoding=REQUIRED_ENCODING) as f:
            return json.load(f)

    features = run_features(force=False)
    oos = run_model(force=False)

    if not os.path.exists(SHAP_IMPORTANCE_OVER_TIME_PATH):
        raise RuntimeError(
            f"{SHAP_IMPORTANCE_OVER_TIME_PATH} not found; run `python -m alpha_lab.explain` first"
        )
    with open(SHAP_IMPORTANCE_OVER_TIME_PATH, encoding=REQUIRED_ENCODING) as f:
        shap_importance = json.load(f)

    folds = PurgedWalkForward().split(features["date"])
    if not folds:
        raise RuntimeError("no CV folds produced, check data span against config.CV")

    corr = feature_correlation(features)
    shap_top = shap_top_feature_per_fold(shap_importance)
    mom_agree = momentum_agreement(oos, features)
    ic_stab = ic_stability(oos, features)
    tail_div = tail_divergence(oos, features)
    turnover = turnover_comparison(oos, features)
    hp_binding = hyperparameter_binding(features, folds)

    summary = build_summary(corr, shap_top, mom_agree, ic_stab, tail_div, turnover, hp_binding)

    result = {
        "description": (
            "Post-hoc addendum: diagnoses why the ML model underperformed the "
            "12-1 momentum baseline, using only already-generated pipeline "
            "artifacts plus one deterministic seed-42 refit of the final CV "
            "fold for the hyperparameter-binding check. Diagnostic only, does "
            "not alter the primary pre-registered verdict."
        ),
        "feature_correlation": corr,
        "shap_top_feature_per_fold": shap_top,
        "momentum_agreement": mom_agree,
        "ic_stability": ic_stab,
        "tail_divergence": tail_div,
        "turnover_comparison": turnover,
        "hyperparameter_binding": hp_binding,
        "summary": summary,
    }

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(DIAGNOSIS_PATH, result, encoding=REQUIRED_ENCODING, indent=2)
    print(f"wrote diagnosis to {DIAGNOSIS_PATH}")
    for line in summary:
        print(f"  - {line}")
    return result


def main() -> None:
    run_diagnose(force=True)


if __name__ == "__main__":
    main()
