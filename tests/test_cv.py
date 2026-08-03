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
