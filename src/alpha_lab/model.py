"""Per-fold LightGBM fit/predict on the honest PurgedWalkForward splitter.

Fixed hyperparameters from config.py, no tuning. One model per fold, trained
on that fold's expanding train window only, predicting only that fold's held
out test block. Run as: python -m alpha_lab.model
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import lightgbm as lgb
import pandas as pd

from alpha_lab.config import DATA_PROCESSED_DIR, FEATURE_COLUMNS, LGBM
from alpha_lab.cv import Fold, PurgedWalkForward
from alpha_lab.features import FEATURES_PATH, run_features
from alpha_lab.util import atomic_to_parquet, atomic_write_json, sha256_of_file

OOS_PREDICTIONS_PATH = DATA_PROCESSED_DIR / "oos_predictions.parquet"
OOS_PREDICTIONS_FINGERPRINT_PATH = DATA_PROCESSED_DIR / "oos_predictions.fingerprint.json"


def fit_predict_fold(panel: pd.DataFrame, fold: Fold) -> pd.DataFrame:
    train_df = panel[panel["date"].isin(fold.train_dates)]
    test_df = panel[panel["date"].isin(fold.test_dates)]

    model = lgb.LGBMRegressor(**LGBM.as_params())
    model.fit(train_df[FEATURE_COLUMNS], train_df["label"])
    preds = model.predict(test_df[FEATURE_COLUMNS])

    out = test_df[["date", "ticker", "label", "fwd_ret_5d"]].copy()
    out["fold_id"] = fold.fold_id
    out["y_pred"] = preds
    return out


def run_model(force: bool = False) -> pd.DataFrame:
    if not force and os.path.exists(OOS_PREDICTIONS_PATH):
        return pd.read_parquet(OOS_PREDICTIONS_PATH)

    panel = run_features(force=False)
    folds = PurgedWalkForward().split(panel["date"])
    if not folds:
        raise RuntimeError("no CV folds produced, check data span against config.CV")

    fold_predictions = []
    for fold in folds:
        preds = fit_predict_fold(panel, fold)
        fold_predictions.append(preds)
        print(f"fold {fold.fold_id}: fit on {len(fold.train_dates)}d, predicted {len(preds)} rows")

    oos = pd.concat(fold_predictions, ignore_index=True).sort_values(["date", "ticker"]).reset_index(
        drop=True
    )
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    atomic_to_parquet(oos, OOS_PREDICTIONS_PATH)
    if os.path.exists(FEATURES_PATH):
        fingerprint = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "features_parquet_sha256": sha256_of_file(FEATURES_PATH),
        }
        atomic_write_json(OOS_PREDICTIONS_FINGERPRINT_PATH, fingerprint, indent=2)
    print(f"wrote {len(oos)} OOS predictions to {OOS_PREDICTIONS_PATH}")
    return oos


def main() -> None:
    run_model(force=True)


if __name__ == "__main__":
    main()
