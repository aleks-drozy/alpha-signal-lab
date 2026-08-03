"""Unit tests for point-in-time features against hand-computed values, plus
pipeline-level sanity checks on the committed fixture.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _toy_df(closes: list[float]) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-02", periods=len(closes))
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": closes,
            "High": [c * 1.01 for c in closes],
            "Low": [c * 0.99 for c in closes],
            "Close": closes,
            "Volume": [1_000_000] * len(closes),
        }
    )


def test_ret_1d_hand_computed():
    from alpha_lab.features import compute_raw_features

    df = _toy_df([100.0, 102.0, 101.0, 105.0])
    out = compute_raw_features(df)

    assert np.isnan(out["ret_1d"].iloc[0])
    assert out["ret_1d"].iloc[1] == pytest.approx(0.02)
    assert out["ret_1d"].iloc[2] == pytest.approx(101.0 / 102.0 - 1.0)
    assert out["ret_1d"].iloc[3] == pytest.approx(105.0 / 101.0 - 1.0)


def test_ret_5d_hand_computed():
    from alpha_lab.features import compute_raw_features

    closes = [100.0, 101.0, 102.0, 103.0, 104.0, 110.0]
    df = _toy_df(closes)
    out = compute_raw_features(df)

    assert out["ret_5d"].iloc[:5].isna().all()
    assert out["ret_5d"].iloc[5] == pytest.approx(110.0 / 100.0 - 1.0)


def test_rsi_all_gains_is_100():
    from alpha_lab.features import compute_raw_features

    closes = [100.0 + i for i in range(20)]
    df = _toy_df(closes)
    out = compute_raw_features(df)

    assert out["rsi_14"].iloc[14:].eq(100.0).all()


def test_forward_return_hand_computed_and_tail_is_nan():
    from alpha_lab.features import compute_forward_return

    closes = [100.0, 102.0, 104.0, 108.0, 110.0, 120.0, 121.0]
    df = _toy_df(closes)
    fwd = compute_forward_return(df, horizon=5)

    assert fwd.iloc[0] == pytest.approx(120.0 / 100.0 - 1.0)
    assert fwd.iloc[3:].isna().all()


def test_cross_sectional_rank_transform_is_within_unit_interval(ranked_panel):
    from alpha_lab.config import FEATURE_COLUMNS

    for col in FEATURE_COLUMNS:
        values = ranked_panel[col]
        assert values.min() >= 0.0
        assert values.max() <= 1.0


def test_build_feature_matrix_has_no_missing_features_or_labels(ranked_panel):
    from alpha_lab.config import FEATURE_COLUMNS

    assert not ranked_panel[FEATURE_COLUMNS + ["label", "fwd_ret_5d"]].isna().any().any()
    assert len(ranked_panel) > 0


def test_excess_label_is_zero_mean_per_date(raw_panel):
    per_date_mean = raw_panel.groupby("date")["label"].mean()
    assert np.allclose(per_date_mean.to_numpy(), 0.0, atol=1e-9)
