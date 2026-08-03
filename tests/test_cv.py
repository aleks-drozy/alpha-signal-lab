"""Boundary-math unit tests for PurgedWalkForward (purge, embargo, ordering,
no overlap). No network calls; uses the committed fixture panel.
"""

from __future__ import annotations

import pandas as pd


def test_folds_are_non_empty(panel_with_folds):
    _, folds, _ = panel_with_folds
    assert len(folds) >= 2


def test_train_end_before_test_start(panel_with_folds):
    _, folds, _ = panel_with_folds
    for fold in folds:
        assert fold.train_end < fold.test_start


def test_gap_at_least_purge_plus_embargo(panel_with_folds):
    panel, folds, cv_config = panel_with_folds
    unique_dates = pd.DatetimeIndex(sorted(pd.unique(panel["date"])))
    embargo = int(round(cv_config.embargo_fraction * len(unique_dates)))
    min_gap_days = cv_config.purge_days + embargo

    for fold in folds:
        idx_train_end = unique_dates.get_loc(fold.train_end)
        idx_test_start = unique_dates.get_loc(fold.test_start)
        trading_day_gap = idx_test_start - idx_train_end - 1
        assert trading_day_gap >= min_gap_days


def test_zero_index_intersection_train_test(panel_with_folds):
    _, folds, _ = panel_with_folds
    for fold in folds:
        train_set = set(fold.train_dates)
        test_set = set(fold.test_dates)
        assert train_set.isdisjoint(test_set)


def test_test_blocks_disjoint_and_ordered(panel_with_folds):
    _, folds, _ = panel_with_folds
    for prev_fold, next_fold in zip(folds, folds[1:]):
        assert prev_fold.test_end < next_fold.test_start
        assert set(prev_fold.test_dates).isdisjoint(set(next_fold.test_dates))


def test_train_window_expands_fold_to_fold(panel_with_folds):
    _, folds, _ = panel_with_folds
    for prev_fold, next_fold in zip(folds, folds[1:]):
        assert next_fold.train_end > prev_fold.train_end
        assert len(next_fold.train_dates) > len(prev_fold.train_dates)
        assert next_fold.train_start == prev_fold.train_start


def test_embargo_days_scales_with_sample_length():
    from alpha_lab.config import CVConfig
    from alpha_lab.cv import PurgedWalkForward

    splitter = PurgedWalkForward(CVConfig(embargo_fraction=0.01))
    assert splitter.embargo_days(1000) == 10
    assert splitter.embargo_days(3900) == 39


def test_pathologically_small_train_window_yields_zero_folds_not_garbage():
    """A degenerate CVConfig whose test block starts almost immediately after
    train_start must not silently produce a fold trained on a handful of
    days; the MIN_TRAIN_DATES guard should drop it, leaving zero folds."""
    from alpha_lab.config import CVConfig
    from alpha_lab.cv import PurgedWalkForward

    dates = pd.bdate_range("2020-01-01", periods=40)
    cfg = CVConfig(
        horizon_days=5,
        purge_days=5,
        embargo_fraction=0.01,
        train_start="2020-01-01",
        train_end="2020-01-01",
        test_start=str(dates[15].date()),
        test_block_months=1,
    )
    folds = PurgedWalkForward(cfg).split(pd.Series(dates))
    assert folds == []


def test_min_train_dates_guard_drops_thin_fold_but_keeps_adequate_ones():
    """A longer synthetic range where the first possible test block would be
    backed by a too-thin train window: that fold is dropped, but a later
    fold with an adequate (expanded) train window survives."""
    from alpha_lab.config import CVConfig
    from alpha_lab.cv import MIN_TRAIN_DATES, PurgedWalkForward

    dates = pd.bdate_range("2020-01-01", periods=400)
    cfg = CVConfig(
        horizon_days=5,
        purge_days=5,
        embargo_fraction=0.01,
        train_start="2020-01-01",
        train_end=str(dates[20].date()),
        test_start=str(dates[25].date()),
        test_block_months=1,
    )
    folds = PurgedWalkForward(cfg).split(pd.Series(dates))
    for fold in folds:
        assert len(fold.train_dates) >= MIN_TRAIN_DATES
