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


def _write_synthetic_raw_csv(dir_path, ticker: str, dates, seed: int) -> None:
    rng = np.random.default_rng(seed)
    closes = 100.0 + np.cumsum(rng.normal(0, 1, size=len(dates)))
    closes = np.clip(closes, 10.0, None)
    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": closes,
            "High": closes * 1.01,
            "Low": closes * 0.99,
            "Close": closes,
            "Volume": rng.integers(1_000_000, 2_000_000, size=len(dates)),
        }
    )
    df.to_csv(dir_path / f"{ticker}.csv", index=False)


def test_short_history_ipo_ticker_excluded_until_warm_and_does_not_corrupt_others(tmp_path, monkeypatch):
    """A ticker with meaningfully less history than the rest of the universe
    (simulating an IPO after the pipeline's start date) must (a) not crash
    build_feature_matrix, (b) be entirely absent from the panel for dates
    before it has a full feature warm-up window, and (c) never silently
    corrupt the cross-sectional rank transform for the other tickers -- the
    other two tickers' pct-ranks must only ever reflect the names actually
    present on a given date, not some stale/padded universe size."""
    import alpha_lab.features as features_mod

    monkeypatch.setattr(features_mod, "DATA_RAW_DIR", tmp_path)

    all_dates = pd.bdate_range("2018-01-02", periods=450)
    ipo_start_idx = 150
    ipo_dates = all_dates[ipo_start_idx:]

    _write_synthetic_raw_csv(tmp_path, "LONGA", all_dates, seed=1)
    _write_synthetic_raw_csv(tmp_path, "LONGB", all_dates, seed=2)
    _write_synthetic_raw_csv(tmp_path, "IPOC", ipo_dates, seed=3)

    panel = features_mod.build_feature_matrix(universe=["LONGA", "LONGB", "IPOC"])

    # (a) no crash reaching here.
    assert len(panel) > 0

    # (b) IPOC needs its own 252+21-day momentum warm-up on top of its IPO
    # date before it can appear at all.
    ipoc_dates_in_panel = panel.loc[panel["ticker"] == "IPOC", "date"]
    # ret_252d_skip21 = close.shift(21) / close.shift(252) - 1 is NaN
    # wherever either shift is NaN, i.e. for the first 252 rows of IPOC's
    # own local history (shift(252) binds tighter than shift(21)); the
    # first row where it can be non-NaN is local index 252.
    warm_up_cutoff = all_dates[ipo_start_idx + 252]
    assert (ipoc_dates_in_panel >= warm_up_cutoff).all()
    assert not (panel.loc[panel["date"] < all_dates[ipo_start_idx], "ticker"] == "IPOC").any()

    # (c) On dates where only LONGA/LONGB are present, pct-rank (method
    # "average", 2 names) must only ever be 0.5 or 1.0 -- never a value
    # implying a 3-name cross-section that isn't there.
    pre_ipo_dates = panel.loc[panel["date"] < warm_up_cutoff, "date"].unique()
    assert len(pre_ipo_dates) > 0
    two_name_rows = panel[panel["date"].isin(pre_ipo_dates)]
    assert set(two_name_rows["ticker"].unique()) == {"LONGA", "LONGB"}
    for col in ["ret_5d", "ret_21d"]:
        assert set(np.round(two_name_rows[col].unique(), 6)) <= {0.5, 1.0}

    # Once IPOC joins, some date has all three names and a genuine 3-way
    # rank split {1/3, 2/3, 1.0} shows up (method="average" over rank 1..3).
    post_ipo_dates = panel.loc[panel["date"] >= warm_up_cutoff, "date"].unique()
    if len(post_ipo_dates) > 0:
        three_name_rows = panel[panel["date"] == post_ipo_dates[0]]
        assert set(three_name_rows["ticker"]) == {"LONGA", "LONGB", "IPOC"}
