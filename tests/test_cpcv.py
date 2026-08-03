"""Unit tests for the pure-math pieces of cpcv.py: block partitioning,
purge/embargo exclusion on both sides of an interior test block, and the
distribution/PBO statistics. No network, no LightGBM fitting, no real
pipeline data (run_cpcv_paths itself fits models and is exercised only
indirectly via the pipeline; these tests target the leakage-relevant math).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lab.config import CVConfig
from alpha_lab.cpcv import compute_pbo, distribution_stats, six_month_blocks, train_mask_excluding_blocks


def test_six_month_blocks_partitions_from_test_start():
    dates = pd.bdate_range("2016-01-01", periods=800)
    cfg = CVConfig(test_start="2016-01-01", test_block_months=6, train_start="2010-01-01", train_end="2015-12-31")

    blocks = six_month_blocks(dates, cfg)

    assert len(blocks) > 0
    for b_start, b_end in blocks:
        assert b_start <= b_end
        assert b_start in dates
    # Blocks should be ordered and non-overlapping.
    for (a_start, a_end), (b_start, b_end) in zip(blocks, blocks[1:]):
        assert a_end < b_start


def test_six_month_blocks_drops_incomplete_trailing_block():
    # Only enough dates for one full 6-month block plus a partial second one.
    dates = pd.bdate_range("2016-01-01", periods=170)  # ~8 months of business days
    cfg = CVConfig(test_start="2016-01-01", test_block_months=6, train_start="2010-01-01", train_end="2015-12-31")

    blocks = six_month_blocks(dates, cfg)
    assert len(blocks) == 1


def test_train_mask_excluding_blocks_excludes_purge_and_embargo_both_sides():
    dates = pd.bdate_range("2016-01-01", periods=100)
    # A test block interior to the range, at index ~50.
    test_block = (dates[50], dates[55])
    purge_days = 3
    embargo_days = 2

    mask = train_mask_excluding_blocks(
        dates, [test_block], purge_days=purge_days, embargo_days=embargo_days, train_start=dates[0]
    )

    excluded_idx = set(np.where(~mask)[0])
    # Indices strictly within purge+embargo distance on both sides of the
    # block (indices 45..60 inclusive: 50-5..55+5) must all be excluded.
    for i in range(45, 61):
        assert i in excluded_idx, f"index {i} should be excluded (interior purge/embargo)"
    # Far away indices must remain in the training mask.
    assert 0 in np.where(mask)[0]
    assert 90 in np.where(mask)[0]


def test_train_mask_excluding_blocks_respects_train_start():
    dates = pd.bdate_range("2016-01-01", periods=50)
    test_block = (dates[40], dates[42])
    mask = train_mask_excluding_blocks(dates, [test_block], purge_days=1, embargo_days=1, train_start=dates[10])

    included_idx = np.where(mask)[0]
    assert included_idx.min() >= 10


def test_distribution_stats_hand_computed():
    values = [0.01, 0.02, 0.03, 0.04]
    stats = distribution_stats(values)
    assert stats["mean"] == pytest.approx(0.025)
    assert stats["n"] == 4
    lo, hi = stats["ci_95_normal"]
    assert lo < stats["mean"] < hi


def test_distribution_stats_ignores_nan_values():
    values = [0.01, float("nan"), 0.03, float("nan")]
    stats = distribution_stats(values)
    assert stats["n"] == 2
    assert stats["mean"] == pytest.approx(0.02)


def test_distribution_stats_empty_returns_none_fields_no_crash():
    stats = distribution_stats([])
    assert stats["mean"] is None
    assert stats["n"] == 0
    assert stats["ci_95_normal"] == [None, None]


def test_compute_pbo_hand_computed_overfit_case():
    # 4 paths: 2 clearly overfit (IS above median, OOS below median), 2 not.
    is_ic = [0.05, 0.04, 0.01, 0.02]
    oos_ic = [-0.01, -0.02, 0.03, 0.04]
    result = compute_pbo(is_ic, oos_ic)
    assert result["n_paths"] == 4
    assert result["pbo"] == pytest.approx(0.5)


def test_compute_pbo_no_valid_paths_returns_none_no_crash():
    result = compute_pbo([float("nan"), float("nan")], [float("nan"), float("nan")])
    assert result["n_paths"] == 0
    assert result["pbo"] is None
    assert result["logit_pbo"] is None
