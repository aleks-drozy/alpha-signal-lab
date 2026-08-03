"""The 8 pre-registered leakage audits (see specs/methodology.md). Each test
wraps its assertions in `audit_check(name, description)`, which records a
pass/fail record regardless of outcome; a module-scoped autouse fixture
dumps every record collected in this module to docs/badges.json once all
tests here have run. Uses the committed fixture panel only, no network.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"
BADGES_PATH = DOCS_DIR / "badges.json"

_RESULTS: list[dict] = []


@contextmanager
def audit_check(name: str, description: str):
    try:
        yield
    except Exception:
        _RESULTS.append({"test": name, "passed": False, "description": description})
        raise
    else:
        _RESULTS.append({"test": name, "passed": True, "description": description})


@pytest.fixture(scope="module", autouse=True)
def write_badges_after_module():
    yield
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    with open(BADGES_PATH, "w", encoding="utf-8") as f:
        json.dump(_RESULTS, f, indent=2)


# 1. Shuffled-label destruction ----------------------------------------------


def test_shuffled_label_destruction(panel_with_folds):
    from alpha_lab.config import FEATURE_COLUMNS, LGBM
    from alpha_lab.evaluate import daily_rank_ic
    import lightgbm as lgb

    with audit_check(
        "shuffled_label_destruction",
        "permuting training labels destroys any regression edge over a train-mean baseline and leaves no "
        "systematic OOS rank IC averaged over repeated shuffles",
    ):
        panel, folds, _ = panel_with_folds
        fold = folds[0]
        train_df = panel[panel["date"].isin(fold.train_dates)].copy()
        test_df = panel[panel["date"].isin(fold.test_dates)].copy()

        # A single shuffle on a 3-name fixture is too coarse for a tight
        # absolute rank-IC bound (spearman rho on n=3 only takes values in
        # {-1, -0.5, 0.5, 1}), so average over repeated shuffles instead of
        # relying on one draw.
        shuffled_ics = []
        mse_ratios = []
        for seed in range(10):
            rng = np.random.default_rng(seed)
            shuffled_label = train_df["label"].to_numpy().copy()
            rng.shuffle(shuffled_label)

            model = lgb.LGBMRegressor(**LGBM.as_params())
            model.fit(train_df[FEATURE_COLUMNS], shuffled_label)
            preds = model.predict(test_df[FEATURE_COLUMNS])
            test_df["y_pred"] = preds

            shuffled_ics.append(daily_rank_ic(test_df, "y_pred").mean())

            model_mse = float(np.mean((preds - test_df["label"].to_numpy()) ** 2))
            baseline_pred = float(shuffled_label.mean())
            baseline_mse = float(np.mean((baseline_pred - test_df["label"].to_numpy()) ** 2))
            mse_ratios.append(model_mse / baseline_mse)

        # Regression analog of "accuracy within 2pp of base rate": with
        # destroyed labels, the model should on average do no better than
        # predicting the train mean, within a small relative tolerance.
        assert np.median(mse_ratios) >= 0.98
        assert abs(np.mean(shuffled_ics)) < 0.10


# 2. Split-boundary assertions ------------------------------------------------


def test_split_boundary_assertions(panel_with_folds):
    with audit_check(
        "split_boundary_assertions",
        "every fold's train label window ends strictly before its test block, "
        "the gap meets purge+embargo, and test blocks are disjoint and ordered",
    ):
        panel, folds, cv_config = panel_with_folds
        assert len(folds) > 0

        unique_dates = pd.DatetimeIndex(sorted(pd.unique(panel["date"])))
        embargo = int(round(cv_config.embargo_fraction * len(unique_dates)))
        min_gap = cv_config.purge_days + embargo

        for fold in folds:
            assert fold.train_end < fold.test_start
            idx_train_end = unique_dates.get_loc(fold.train_end)
            idx_test_start = unique_dates.get_loc(fold.test_start)
            assert (idx_test_start - idx_train_end - 1) >= min_gap
            assert set(fold.train_dates).isdisjoint(set(fold.test_dates))

        for prev_fold, next_fold in zip(folds, folds[1:]):
            assert prev_fold.test_end < next_fold.test_start
            assert set(prev_fold.test_dates).isdisjoint(set(next_fold.test_dates))


# 3. Point-in-time feature recomputation --------------------------------------


def test_point_in_time_feature_recomputation(raw_panel, fixture_universe, patch_raw_dir):
    from alpha_lab.config import FEATURE_COLUMNS
    from alpha_lab.features import compute_raw_features, load_raw

    with audit_check(
        "point_in_time_feature_recomputation",
        "recomputing features from raw data sliced to date<=t exactly matches the pipeline's feature matrix",
    ):
        rng = np.random.default_rng(7)
        sample_idx = rng.choice(len(raw_panel), size=min(20, len(raw_panel)), replace=False)

        raw_cache = {t: load_raw(t) for t in fixture_universe}

        for idx in sample_idx:
            row = raw_panel.iloc[idx]
            ticker = row["ticker"]
            target_date = row["date"]

            raw = raw_cache[ticker]
            sliced = raw[raw["Date"] <= target_date].reset_index(drop=True)
            recomputed = compute_raw_features(sliced)
            recomputed_row = recomputed.iloc[-1]

            assert recomputed_row["Date"] == target_date
            for col in FEATURE_COLUMNS:
                assert recomputed_row[col] == pytest.approx(row[col], rel=1e-9, abs=1e-12)


# 4. Shift-sensitivity control -------------------------------------------------


def test_shift_sensitivity_control(panel_with_folds):
    from alpha_lab.config import FEATURE_COLUMNS, LGBM
    from alpha_lab.evaluate import daily_rank_ic
    import lightgbm as lgb

    with audit_check(
        "shift_sensitivity_control",
        "lagging every feature by one extra day does not materially improve OOS rank IC over the honest pipeline",
    ):
        panel, folds, _ = panel_with_folds

        lagged_panel = panel.sort_values(["ticker", "date"]).copy()
        lagged_panel[FEATURE_COLUMNS] = lagged_panel.groupby("ticker")[FEATURE_COLUMNS].shift(1)
        lagged_panel = lagged_panel.dropna(subset=FEATURE_COLUMNS)

        # Concatenate OOS predictions across all folds (as evaluate.py does)
        # rather than comparing a single fold, since a single 3-name test
        # block is too small for a stable rank-IC estimate.
        honest_oos, lagged_oos = [], []
        for fold in folds:
            honest_train = panel[panel["date"].isin(fold.train_dates)]
            honest_test = panel[panel["date"].isin(fold.test_dates)].copy()
            honest_model = lgb.LGBMRegressor(**LGBM.as_params())
            honest_model.fit(honest_train[FEATURE_COLUMNS], honest_train["label"])
            honest_test["y_pred"] = honest_model.predict(honest_test[FEATURE_COLUMNS])
            honest_oos.append(honest_test[["date", "y_pred", "label"]])

            lagged_train = lagged_panel[lagged_panel["date"].isin(fold.train_dates)]
            lagged_test = lagged_panel[lagged_panel["date"].isin(fold.test_dates)].copy()
            if lagged_train.empty or lagged_test.empty:
                continue
            lagged_model = lgb.LGBMRegressor(**LGBM.as_params())
            lagged_model.fit(lagged_train[FEATURE_COLUMNS], lagged_train["label"])
            lagged_test["y_pred"] = lagged_model.predict(lagged_test[FEATURE_COLUMNS])
            lagged_oos.append(lagged_test[["date", "y_pred", "label"]])

        honest_ic = daily_rank_ic(pd.concat(honest_oos, ignore_index=True), "y_pred").mean()
        lagged_ic = daily_rank_ic(pd.concat(lagged_oos, ignore_index=True), "y_pred").mean()

        assert lagged_ic <= honest_ic + 0.01


# 5. Label-window purity -------------------------------------------------------


def test_label_window_purity(fixture_universe, patch_raw_dir):
    from alpha_lab.config import LABEL_HORIZON_DAYS
    from alpha_lab.features import compute_forward_return, load_raw

    with audit_check(
        "label_window_purity",
        "labels cannot be produced without a full forward window; rows lacking one are NaN, never filled",
    ):
        raw = load_raw(fixture_universe[0])
        truncated = raw.iloc[: len(raw) - 20].reset_index(drop=True)

        fwd_full = compute_forward_return(raw, LABEL_HORIZON_DAYS)
        fwd_truncated = compute_forward_return(truncated, LABEL_HORIZON_DAYS)

        assert fwd_truncated.iloc[-LABEL_HORIZON_DAYS:].isna().all()
        assert fwd_full.iloc[: len(truncated) - LABEL_HORIZON_DAYS].equals(
            fwd_truncated.iloc[: len(truncated) - LABEL_HORIZON_DAYS]
        )


# 6. No-full-series-statistics -------------------------------------------------


def test_no_full_series_statistics(panel_with_folds, raw_panel, fixture_universe, patch_raw_dir):
    from alpha_lab.config import FEATURE_COLUMNS
    from alpha_lab.features import compute_raw_features, load_raw

    with audit_check(
        "no_full_series_statistics",
        "a naive scaler fit on train+test differs materially from one fit on train-only, and the pipeline's own "
        "rolling-window features for train-period dates are unchanged whether or not future rows are appended",
    ):
        _, folds, _ = panel_with_folds
        fold = folds[0]
        train_dates = set(fold.train_dates)

        # Raw (pre rank-transform) features, so this is a genuine "fit a
        # global mean/std" comparison, not an artifact of the per-date rank
        # transform (whose cross-sectional mean is ~constant regardless of
        # which dates are included).
        train_only = raw_panel[raw_panel["date"].isin(train_dates)]
        train_and_test = raw_panel[raw_panel["date"].isin(train_dates | set(fold.test_dates))]

        naive_train_mean = train_only["ret_5d"].mean()
        naive_full_mean = train_and_test["ret_5d"].mean()
        assert naive_train_mean != pytest.approx(naive_full_mean, rel=1e-6)

        ticker = fixture_universe[0]
        raw = load_raw(ticker)
        cutoff = raw["Date"].max() - pd.Timedelta(days=200)
        train_slice = raw[raw["Date"] <= cutoff].reset_index(drop=True)

        feats_train_only = compute_raw_features(train_slice)
        feats_full = compute_raw_features(raw)
        feats_full_aligned = feats_full[feats_full["Date"] <= cutoff].reset_index(drop=True)

        for col in FEATURE_COLUMNS:
            a = feats_train_only[col].to_numpy()
            b = feats_full_aligned[col].to_numpy()
            np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True)


# 7. Adjusted-price sanity ------------------------------------------------------


def test_adjusted_price_sanity(fixture_universe, patch_raw_dir):
    from alpha_lab.features import load_raw

    with audit_check(
        "adjusted_price_sanity",
        "no negative prices, no long zero-volume runs, no unflagged >40% daily moves, dates strictly increasing",
    ):
        for ticker in fixture_universe:
            raw = load_raw(ticker)

            assert (raw["Open"] > 0).all()
            assert (raw["High"] > 0).all()
            assert (raw["Low"] > 0).all()
            assert (raw["Close"] > 0).all()

            assert raw["Date"].is_monotonic_increasing
            assert raw["Date"].is_unique

            zero_volume = (raw["Volume"] == 0).astype(int)
            if zero_volume.any():
                run_lengths = (
                    zero_volume.groupby((zero_volume != zero_volume.shift()).cumsum()).cumsum()
                )
                assert run_lengths.max() <= 10

            daily_ret = raw["Close"].pct_change().dropna()
            assert (daily_ret.abs() <= 0.40).all()


# 8. Determinism -----------------------------------------------------------------


def test_determinism_same_seed_same_predictions(panel_with_folds):
    from alpha_lab.model import fit_predict_fold

    with audit_check(
        "determinism",
        "two fold fits/predicts with the same seed and inputs produce hash-identical OOS predictions",
    ):
        panel, folds, _ = panel_with_folds
        fold = folds[0]

        run_a = fit_predict_fold(panel, fold)
        run_b = fit_predict_fold(panel, fold)

        pd.testing.assert_frame_equal(
            run_a.reset_index(drop=True), run_b.reset_index(drop=True), check_exact=True
        )
        hash_a = pd.util.hash_pandas_object(run_a["y_pred"]).sum()
        hash_b = pd.util.hash_pandas_object(run_b["y_pred"]).sum()
        assert hash_a == hash_b
