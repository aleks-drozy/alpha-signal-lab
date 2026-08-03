"""Unit tests for leaky_twin.py: the OOS-date-range restriction (the M1 fix
restricting leaky-twin evaluation to the honest model's date range) and the
shuffled-K-fold fit/predict shape, on tiny synthetic panels. No network, no
real pipeline data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lab.leaky_twin import fit_predict_shuffled_kfold, restrict_to_oos_date_range


def test_restrict_to_oos_date_range_filters_wider_raw_panel():
    # Raw panel spans a much wider calendar range than the honest model's
    # OOS predictions; only rows inside [oos.date.min(), oos.date.max()]
    # must survive.
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2010-01-01", "2015-06-01", "2016-01-04", "2018-06-15", "2020-12-31"]
            ),
            "ticker": ["A", "A", "A", "A", "A"],
            "value": [1, 2, 3, 4, 5],
        }
    )
    oos = pd.DataFrame({"date": pd.to_datetime(["2016-01-04", "2018-06-15"])})

    restricted = restrict_to_oos_date_range(panel, oos)

    assert restricted["date"].min() == pd.Timestamp("2016-01-04")
    assert restricted["date"].max() == pd.Timestamp("2018-06-15")
    assert set(restricted["value"]) == {3, 4}
    assert len(restricted) == 2


def test_restrict_to_oos_date_range_inclusive_boundaries():
    panel = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]), "value": [1, 2, 3]})
    oos = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-01-03"])})

    restricted = restrict_to_oos_date_range(panel, oos)

    # Boundary dates themselves (equal to min/max) must be included.
    assert len(restricted) == 3


def test_restrict_to_oos_date_range_empty_when_no_overlap():
    panel = pd.DataFrame({"date": pd.to_datetime(["2005-01-01", "2005-06-01"]), "value": [1, 2]})
    oos = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-01-02"])})

    restricted = restrict_to_oos_date_range(panel, oos)
    assert restricted.empty


def test_fit_predict_shuffled_kfold_predicts_every_row_exactly_once(monkeypatch):
    from alpha_lab.config import FEATURE_COLUMNS

    rng = np.random.default_rng(0)
    n = 60
    panel = pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-01", periods=n),
            "ticker": ["A"] * n,
            "fwd_ret_5d": rng.normal(size=n),
            "label": rng.normal(size=n),
        }
    )
    for col in FEATURE_COLUMNS:
        panel[col] = rng.normal(size=n)

    out = fit_predict_shuffled_kfold(panel)

    # Every row gets exactly one OOS prediction (KFold with shuffle=True
    # partitions all indices across folds with no overlap and no omission).
    assert len(out) == n
    assert out["y_pred"].notna().all()
    assert set(out.columns) >= {"date", "ticker", "fwd_ret_5d", "label", "y_pred"}


def test_fit_predict_shuffled_kfold_is_deterministic_given_seed():
    from alpha_lab.config import FEATURE_COLUMNS

    rng = np.random.default_rng(1)
    n = 40
    panel = pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-01", periods=n),
            "ticker": ["A"] * n,
            "fwd_ret_5d": rng.normal(size=n),
            "label": rng.normal(size=n),
        }
    )
    for col in FEATURE_COLUMNS:
        panel[col] = rng.normal(size=n)

    out_a = fit_predict_shuffled_kfold(panel)
    out_b = fit_predict_shuffled_kfold(panel)

    pd.testing.assert_series_equal(out_a["y_pred"], out_b["y_pred"])
