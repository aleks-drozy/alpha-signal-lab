"""SHAP explainability on the honest PurgedWalkForward pipeline.

Global beeswarm and mean|SHAP| bar use the final fold's model only, computed
on that fold's OOS test rows only (never train rows: fitting an explainer on
train data would show what the model memorized, not what it actually used
out of sample). Per-fold rank stability retrains every fold to see whether
the top feature is stable or churns, which must be reported honestly either
way. Run as: python -m alpha_lab.explain
"""

from __future__ import annotations

import json
import os

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from alpha_lab.config import (
    DATA_PROCESSED_DIR,
    DOCS_IMG_DIR,
    FEATURE_COLUMNS,
    LGBM,
    REQUIRED_ENCODING,
    SHAP_N_LOCAL_EXAMPLES,
)
from alpha_lab.cv import Fold, PurgedWalkForward
from alpha_lab.features import run_features
from alpha_lab.util import atomic_write_json

SHAP_STABILITY_PATH = DATA_PROCESSED_DIR / "shap_stability.json"
SHAP_IMPORTANCE_OVER_TIME_PATH = DATA_PROCESSED_DIR / "shap_importance_over_time.json"
BEESWARM_PATH = DOCS_IMG_DIR / "shap_beeswarm.png"
IMPORTANCE_BAR_PATH = DOCS_IMG_DIR / "shap_importance_bar.png"
STABILITY_HEATMAP_PATH = DOCS_IMG_DIR / "shap_stability_heatmap.png"


def _fit_fold_model(panel: pd.DataFrame, fold: Fold) -> tuple[lgb.LGBMRegressor, pd.DataFrame]:
    train_df = panel[panel["date"].isin(fold.train_dates)]
    test_df = panel[panel["date"].isin(fold.test_dates)]
    model = lgb.LGBMRegressor(**LGBM.as_params())
    model.fit(train_df[FEATURE_COLUMNS], train_df["label"])
    return model, test_df


def _mean_abs_shap(shap_values: np.ndarray) -> dict[str, float]:
    mean_abs = np.abs(shap_values).mean(axis=0)
    return {col: float(v) for col, v in zip(FEATURE_COLUMNS, mean_abs)}


def plot_beeswarm(shap_values: np.ndarray, test_df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(9, 6))
    shap.summary_plot(
        shap_values, test_df[FEATURE_COLUMNS], feature_names=FEATURE_COLUMNS, show=False
    )
    fig.tight_layout()
    fig.savefig(BEESWARM_PATH, dpi=150)
    plt.close(fig)


def plot_importance_bar(mean_abs: dict[str, float]) -> None:
    ordered = sorted(mean_abs.items(), key=lambda kv: kv[1], reverse=True)
    names = [k for k, _ in ordered]
    values = [v for _, v in ordered]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(names[::-1], values[::-1], color="#4C78A8")
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("Global feature importance (final fold, OOS test rows)")
    fig.tight_layout()
    fig.savefig(IMPORTANCE_BAR_PATH, dpi=150)
    plt.close(fig)


def pick_local_examples(test_df: pd.DataFrame, n: int = SHAP_N_LOCAL_EXAMPLES) -> pd.DataFrame:
    dates = sorted(test_df["date"].unique())
    if not dates:
        return test_df.iloc[0:0]
    march_2020 = [d for d in dates if pd.Timestamp("2020-03-01") <= pd.Timestamp(d) <= pd.Timestamp("2020-03-31")]
    if march_2020:
        picked_dates = [march_2020[len(march_2020) // 2]]
        remaining_dates = [d for d in dates if d not in picked_dates]
        step = max(len(remaining_dates) // (n - 1), 1) if n > 1 else 1
        picked_dates += remaining_dates[::step][: n - 1]
    else:
        step = max(len(dates) // n, 1)
        picked_dates = dates[::step][:n]

    rows = []
    for d in picked_dates:
        day = test_df[test_df["date"] == d]
        top_row = day.sort_values("fwd_ret_5d", ascending=False).iloc[[0]]
        rows.append(top_row)
    return pd.concat(rows, ignore_index=True) if rows else test_df.iloc[0:0]


def plot_local_waterfalls(explainer: shap.TreeExplainer, examples: pd.DataFrame) -> None:
    for i, (_, row) in enumerate(examples.iterrows()):
        x_row = row[FEATURE_COLUMNS].to_frame().T.astype(float)
        explanation = explainer(x_row)
        fig = plt.figure(figsize=(8, 5))
        shap.plots.waterfall(explanation[0], show=False)
        date_str = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
        fig.suptitle(f"{row['ticker']} {date_str}")
        fig.tight_layout()
        fig.savefig(DOCS_IMG_DIR / f"shap_waterfall_{i + 1}.png", dpi=150)
        plt.close(fig)


def compute_fold_stability(panel: pd.DataFrame, folds: list[Fold]) -> dict:
    per_fold_mean_abs: dict[int, dict[str, float]] = {}
    for fold in folds:
        model, test_df = _fit_fold_model(panel, fold)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(test_df[FEATURE_COLUMNS])
        per_fold_mean_abs[fold.fold_id] = _mean_abs_shap(shap_values)

    per_fold_rank: dict[int, dict[str, int]] = {}
    for fold_id, mean_abs in per_fold_mean_abs.items():
        ordered = sorted(mean_abs.items(), key=lambda kv: kv[1], reverse=True)
        per_fold_rank[fold_id] = {feat: rank + 1 for rank, (feat, _) in enumerate(ordered)}

    top_feature_per_fold = {
        fold_id: min(ranks, key=ranks.get) for fold_id, ranks in per_fold_rank.items()
    }
    top_feature_churns = len(set(top_feature_per_fold.values())) > 1

    return {
        "per_fold_mean_abs_shap": per_fold_mean_abs,
        "per_fold_rank": per_fold_rank,
        "top_feature_per_fold": top_feature_per_fold,
        "top_feature_churns": top_feature_churns,
        "n_folds": len(folds),
    }


def plot_stability_heatmap(stability: dict) -> None:
    per_fold_rank = stability["per_fold_rank"]
    fold_ids = sorted(per_fold_rank.keys())
    features = FEATURE_COLUMNS

    matrix = np.array([[per_fold_rank[fid][feat] for fid in fold_ids] for feat in features])

    fig, ax = plt.subplots(figsize=(max(6, len(fold_ids) * 0.9), 6))
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=1, vmax=len(features))
    ax.set_xticks(range(len(fold_ids)))
    ax.set_xticklabels([f"fold {fid}" for fid in fold_ids])
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.set_title("Feature rank by mean |SHAP| per fold (1 = most important)")
    for i in range(len(features)):
        for j in range(len(fold_ids)):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="rank")
    fig.tight_layout()
    fig.savefig(STABILITY_HEATMAP_PATH, dpi=150)
    plt.close(fig)


def build_importance_over_time(stability: dict, folds: list[Fold]) -> dict:
    """Post-hoc addendum: reshapes the per-fold mean |SHAP| values already
    computed by compute_fold_stability (for the rank-stability heatmap) into
    a fold-indexed time series -- one row per fold, one column per feature --
    so a rolling-importance chart can be drawn on the dashboard. No new SHAP
    computation: reuses per_fold_mean_abs_shap as-is."""
    fold_by_id = {f.fold_id: f for f in folds}
    per_fold_mean_abs = stability["per_fold_mean_abs_shap"]
    rows = []
    for fold_key in sorted(per_fold_mean_abs, key=lambda k: int(k)):
        fold_id = int(fold_key)
        fold = fold_by_id.get(fold_id)
        row = {
            "fold_id": fold_id,
            "test_start": str(fold.test_start.date()) if fold else None,
            "test_end": str(fold.test_end.date()) if fold else None,
        }
        row.update(per_fold_mean_abs[fold_key])
        rows.append(row)
    return {"feature_columns": list(FEATURE_COLUMNS), "rows": rows}


def run_explain(force: bool = False) -> dict:
    if not force and os.path.exists(SHAP_STABILITY_PATH):
        with open(SHAP_STABILITY_PATH, encoding=REQUIRED_ENCODING) as f:
            return json.load(f)

    panel = run_features(force=False)
    folds = PurgedWalkForward().split(panel["date"])
    if not folds:
        raise RuntimeError("no CV folds produced, check data span against config.CV")

    DOCS_IMG_DIR.mkdir(parents=True, exist_ok=True)

    final_fold = folds[-1]
    final_model, final_test_df = _fit_fold_model(panel, final_fold)
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(final_test_df[FEATURE_COLUMNS])

    plot_beeswarm(shap_values, final_test_df)
    plot_importance_bar(_mean_abs_shap(shap_values))

    local_examples = pick_local_examples(final_test_df)
    plot_local_waterfalls(explainer, local_examples)

    stability = compute_fold_stability(panel, folds)
    plot_stability_heatmap(stability)

    importance_over_time = build_importance_over_time(stability, folds)

    stability["final_fold_id"] = final_fold.fold_id
    stability["final_fold_test_range"] = [
        str(final_fold.test_start.date()),
        str(final_fold.test_end.date()),
    ]

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(SHAP_STABILITY_PATH, stability, encoding=REQUIRED_ENCODING, indent=2)
    print(f"wrote SHAP stability data to {SHAP_STABILITY_PATH}")
    print(f"top feature churns across folds: {stability['top_feature_churns']}")

    atomic_write_json(SHAP_IMPORTANCE_OVER_TIME_PATH, importance_over_time, encoding=REQUIRED_ENCODING, indent=2)
    print(f"wrote SHAP importance-over-time (addendum) to {SHAP_IMPORTANCE_OVER_TIME_PATH}")
    return stability


def main() -> None:
    run_explain(force=True)


if __name__ == "__main__":
    main()
