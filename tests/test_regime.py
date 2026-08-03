"""Unit tests for the regime median-split and per-ticker IC logic in
regime.py, on tiny synthetic examples (no network, no real pipeline data).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.regime import assign_regime, per_ticker_ic, regime_ic_breakdown


def test_assign_regime_median_split():
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"])
    # Vol series: two clearly low, two clearly high, straddling the median.
    vol_series = pd.Series([0.01, 0.02, 0.05, 0.06], index=dates)
    oos = pd.DataFrame({"date": dates, "ticker": ["A", "B", "C", "D"], "y_pred": [1, 2, 3, 4], "label": [1, 2, 3, 4]})

    out = assign_regime(oos, vol_series)
    regimes = dict(zip(out["date"], out["regime"]))
    assert regimes[dates[0]] == "low_vol"
    assert regimes[dates[1]] == "low_vol"
    assert regimes[dates[2]] == "high_vol"
    assert regimes[dates[3]] == "high_vol"


def test_assign_regime_forward_fills_missing_vol_dates():
    # OOS date with no matching vol_series entry must ffill from the prior
    # known date, not silently drop into an unlabeled regime.
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    vol_series = pd.Series([0.01, 0.09], index=pd.to_datetime(["2020-01-01", "2020-01-03"]))
    oos = pd.DataFrame({"date": dates, "ticker": ["A", "B", "C"], "y_pred": [1, 2, 3], "label": [1, 2, 3]})

    out = assign_regime(oos, vol_series)
    assert out.loc[out["date"] == dates[1], "regime"].iloc[0] == "low_vol"


def test_regime_ic_breakdown_groups_by_regime_label():
    # daily_rank_ic needs >= 3 rows per date to compute a correlation, so
    # each date here has 3 tickers, not 2.
    dates = pd.to_datetime(["2020-01-01"] * 3 + ["2020-01-02"] * 3)
    oos = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["A", "B", "C", "A", "B", "C"],
            "y_pred": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
            "label": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
            "regime": ["low_vol"] * 3 + ["high_vol"] * 3,
        }
    )
    out = regime_ic_breakdown(oos)
    assert set(out.keys()) == {"low_vol", "high_vol"}
    assert out["low_vol"]["n_days"] == 1
    assert out["high_vol"]["n_days"] == 1


def test_per_ticker_ic_perfect_positive_correlation():
    oos = pd.DataFrame(
        {
            "ticker": ["A"] * 5,
            "y_pred": [1.0, 2.0, 3.0, 4.0, 5.0],
            "label": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    result = per_ticker_ic(oos)
    assert len(result) == 1
    assert np.isclose(result[0]["spearman_ic"], 1.0)


def test_per_ticker_ic_none_when_too_few_observations():
    oos = pd.DataFrame({"ticker": ["A", "A"], "y_pred": [1.0, 2.0], "label": [1.0, 2.0]})
    result = per_ticker_ic(oos)
    assert result[0]["spearman_ic"] is None
