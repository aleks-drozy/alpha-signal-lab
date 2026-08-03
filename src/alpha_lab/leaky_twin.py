"""Leaky twin: identical features/model/target, naive random 5-fold shuffled
CV instead of PurgedWalkForward. No purge, no embargo, no time ordering.

This is the headline "leaky vs honest" comparison: the inflated IC here is
the reference ceiling that PurgedWalkForward eliminates. Run as:
python -m alpha_lab.leaky_twin
"""

from __future__ import annotations

import json
import os

import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import KFold

from alpha_lab.config import (
    DATA_PROCESSED_DIR,
    FEATURE_COLUMNS,
    LEAKY_TWIN_N_FOLDS,
    LEAKY_TWIN_SEED,
    LGBM,
    REQUIRED_ENCODING,
)
from alpha_lab.evaluate import evaluate_strategy
from alpha_lab.features import run_features

LEAKY_TWIN_METRICS_PATH = DATA_PROCESSED_DIR / "leaky_twin_metrics.json"


def fit_predict_shuffled_kfold(panel: pd.DataFrame) -> pd.DataFrame:
    kf = KFold(n_splits=LEAKY_TWIN_N_FOLDS, shuffle=True, random_state=LEAKY_TWIN_SEED)
    X = panel[FEATURE_COLUMNS].reset_index(drop=True)
    y = panel["label"].reset_index(drop=True)
    meta = panel[["date", "ticker", "fwd_ret_5d"]].reset_index(drop=True)

    preds = pd.Series(index=X.index, dtype=float)
    for fold_id, (train_idx, test_idx) in enumerate(kf.split(X)):
        model = lgb.LGBMRegressor(**LGBM.as_params())
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds.iloc[test_idx] = model.predict(X.iloc[test_idx])

    out = meta.copy()
    out["label"] = y
    out["y_pred"] = preds
    return out


def run_leaky_twin(force: bool = False) -> dict:
    if not force and os.path.exists(LEAKY_TWIN_METRICS_PATH):
        with open(LEAKY_TWIN_METRICS_PATH, encoding=REQUIRED_ENCODING) as f:
            return json.load(f)

    panel = run_features(force=False)
    preds = fit_predict_shuffled_kfold(panel)
    metrics = evaluate_strategy(preds, pred_col="y_pred")
    metrics["n_rows"] = int(len(preds))
    metrics["n_folds"] = LEAKY_TWIN_N_FOLDS
    metrics["cv_type"] = "naive_shuffled_kfold_no_purge_no_embargo"

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(LEAKY_TWIN_METRICS_PATH, "w", encoding=REQUIRED_ENCODING) as f:
        json.dump(metrics, f, indent=2)
    print(f"wrote leaky twin metrics to {LEAKY_TWIN_METRICS_PATH}")
    print(f"leaky twin mean rank IC: {metrics['rank_ic']['mean']:.4f}")
    return metrics


def main() -> None:
    run_leaky_twin(force=True)


if __name__ == "__main__":
    main()
