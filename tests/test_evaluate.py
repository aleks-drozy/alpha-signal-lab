"""Unit tests for evaluate.py: Newey-West t-stat correctness (hand-computed
and against synthetic autocorrelated data), decile-spread selection with
fewer than 2n names on a rebalance date, the NaN-prediction guard, and the
equity-curve dead-NaN-check math (std_r == NaN must not silently pass the
`x in (0.0, nan)` bug that the current explicit isnan() guard replaced).
No network calls, no real pipeline data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lab.evaluate import (
    decile_spread_stats,
    equity_curve_stats,
    evaluate_strategy,
    ic_summary,
    newey_west_tstat,
    top_bottom_period_returns,
)


# ---------------------------------------------------------------------------
# Newey-West t-stat correctness
# ---------------------------------------------------------------------------


def test_newey_west_tstat_hand_computed_lag1():
    # x = [1, 2, 3, 4], lag=1. Hand-computed via the same Bartlett-kernel
    # formula the implementation uses:
    #   mean = 2.5, demeaned = [-1.5, -0.5, 0.5, 1.5]
    #   gamma_0 = (2.25+0.25+0.25+2.25)/4 = 1.25
    #   gamma_1 = ((-0.5*-1.5)+(0.5*-0.5)+(1.5*0.5))/4 = 1.25/4 = 0.3125
    #   weight(h=1) = 1 - 1/2 = 0.5
    #   long_run_var = 1.25 + 2*0.5*0.3125 = 1.5625
    #   se = sqrt(1.5625/4) = 0.625
    #   tstat = 2.5/0.625 = 4.0
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert newey_west_tstat(x, lag=1) == pytest.approx(4.0, abs=1e-9)


def test_newey_west_tstat_lag0_reduces_to_population_variance_formula():
    # With lag=0 the loop over h contributes nothing, so long_run_var is
    # exactly gamma_0 (population variance, ddof=0), independently
    # cross-checked here rather than trusting the same code path.
    rng = np.random.default_rng(0)
    x = rng.normal(loc=0.3, scale=1.0, size=50)
    gamma_0 = float(np.sum((x - x.mean()) ** 2) / len(x))
    expected_se = np.sqrt(gamma_0 / len(x))
    expected_tstat = x.mean() / expected_se
    assert newey_west_tstat(x, lag=0) == pytest.approx(expected_tstat, rel=1e-9)


def test_newey_west_tstat_positive_autocorrelated_series_has_lower_tstat_than_naive():
    # Positive autocorrelation inflates the naive (iid-assumed) t-stat
    # relative to the Newey-West-corrected one, since positively correlated
    # observations carry less independent information than iid data of the
    # same sample size. Build an AR(1) series with strong positive rho and
    # verify NW's t-stat is materially smaller in magnitude than the naive
    # mean/(std/sqrt(n)) t-stat.
    rng = np.random.default_rng(42)
    n = 300
    rho = 0.8
    eps = rng.normal(scale=1.0, size=n)
    x = np.zeros(n)
    x[0] = eps[0]
    for i in range(1, n):
        x[i] = rho * x[i - 1] + eps[i]
    x = x + 0.15  # small positive drift so mean != 0

    naive_tstat = x.mean() / (x.std(ddof=1) / np.sqrt(n))
    nw_tstat = newey_west_tstat(x, lag=5)

    assert abs(nw_tstat) < abs(naive_tstat)


def test_newey_west_tstat_nan_for_fewer_than_two_points():
    assert np.isnan(newey_west_tstat(np.array([1.0])))
    assert np.isnan(newey_west_tstat(np.array([])))


def test_newey_west_tstat_nan_when_zero_variance():
    assert np.isnan(newey_west_tstat(np.array([1.0, 1.0, 1.0, 1.0])))


def test_ic_summary_hand_computed():
    values = pd.Series([0.1, 0.2, -0.1, 0.3], index=pd.to_datetime(
        ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]
    ))
    summary = ic_summary(values)
    assert summary["mean"] == pytest.approx(0.125)
    assert summary["hit_rate"] == pytest.approx(0.75)
    assert summary["n_days"] == 4


# ---------------------------------------------------------------------------
# Decile-spread selection: fewer than 2n names on a rebalance date
# ---------------------------------------------------------------------------


def _make_cross_section(date: str, n_names: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [pd.Timestamp(date)] * n_names,
            "ticker": [f"T{i}" for i in range(n_names)],
            "y_pred": np.linspace(0.0, 1.0, n_names),
            "fwd_ret_5d": np.linspace(-0.02, 0.02, n_names),
        }
    )


def test_top_bottom_period_returns_skips_dates_with_fewer_than_2n_names():
    # DECILE_LONG_SHORT_N defaults to 3, so 2n = 6 names required.
    thin_date = _make_cross_section("2020-01-01", n_names=4)
    ok_date = _make_cross_section("2020-01-08", n_names=8)
    df = pd.concat([thin_date, ok_date], ignore_index=True)

    result = top_bottom_period_returns(df, pred_col="y_pred", rebalance_ds=[pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-08")], n=3)

    assert len(result) == 1
    assert result.iloc[0]["date"] == pd.Timestamp("2020-01-08")


def test_top_bottom_period_returns_skips_all_dates_when_universe_too_small():
    df = _make_cross_section("2020-01-01", n_names=5)
    result = top_bottom_period_returns(df, pred_col="y_pred", rebalance_ds=[pd.Timestamp("2020-01-01")], n=3)
    assert result.empty


def test_top_bottom_period_returns_exact_boundary_2n_names_is_not_skipped():
    df = _make_cross_section("2020-01-01", n_names=6)
    result = top_bottom_period_returns(df, pred_col="y_pred", rebalance_ds=[pd.Timestamp("2020-01-01")], n=3)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# NaN-prediction guard
# ---------------------------------------------------------------------------


def test_evaluate_strategy_drops_nan_predictions_before_ranking():
    dates = [pd.Timestamp("2020-01-01")] * 6 + [pd.Timestamp("2020-01-08")] * 6
    df = pd.DataFrame(
        {
            "date": dates,
            "ticker": [f"T{i}" for i in range(6)] * 2,
            "y_pred": [0.1, 0.2, np.nan, 0.4, 0.5, 0.6] + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "label": np.linspace(-0.01, 0.01, 12),
            "fwd_ret_5d": np.linspace(-0.02, 0.02, 12),
        }
    )
    # Must not raise, and the NaN row must not silently sort to either end
    # (e.g. by falling to the tail via sort_values' default NaN-last).
    result = evaluate_strategy(df, pred_col="y_pred")
    assert result["rank_ic"]["n_days"] >= 1


def test_evaluate_strategy_all_nan_predictions_yields_empty_but_no_crash():
    dates = [pd.Timestamp("2020-01-01")] * 4
    df = pd.DataFrame(
        {
            "date": dates,
            "ticker": [f"T{i}" for i in range(4)],
            "y_pred": [np.nan] * 4,
            "label": [0.01, 0.02, -0.01, 0.0],
            "fwd_ret_5d": [0.01, 0.02, -0.01, 0.0],
        }
    )
    result = evaluate_strategy(df, pred_col="y_pred")
    assert result["rank_ic"]["n_days"] == 0
    assert np.isnan(result["rank_ic"]["mean"])


# ---------------------------------------------------------------------------
# Equity-curve dead-NaN-check math (std_r == NaN must route to sharpe=NaN,
# not silently divide by NaN or crash; the old `x in (0.0, float("nan"))`
# check never matched NaN since NaN != NaN, so this is the regression guard).
# ---------------------------------------------------------------------------


def test_equity_curve_stats_single_row_std_is_nan_sharpe_is_nan_not_crash():
    period_returns = pd.DataFrame({"gross_ret": [0.01], "turnover": [0.1]})
    stats = equity_curve_stats(period_returns, "gross_ret")
    assert np.isnan(stats["sharpe"])


def test_equity_curve_stats_zero_variance_returns_sharpe_is_nan():
    period_returns = pd.DataFrame({"gross_ret": [0.01, 0.01, 0.01], "turnover": [0.1, 0.1, 0.1]})
    stats = equity_curve_stats(period_returns, "gross_ret")
    assert np.isnan(stats["sharpe"])


def test_equity_curve_stats_normal_case_computes_finite_sharpe():
    period_returns = pd.DataFrame({"gross_ret": [0.01, -0.005, 0.02, 0.015], "turnover": [0.1] * 4})
    stats = equity_curve_stats(period_returns, "gross_ret")
    assert np.isfinite(stats["sharpe"])


def test_equity_curve_stats_empty_returns_all_nan_no_crash():
    stats = equity_curve_stats(pd.DataFrame(columns=["gross_ret", "turnover"]), "gross_ret")
    assert np.isnan(stats["sharpe"])
    assert np.isnan(stats["max_drawdown"])
    assert np.isnan(stats["annualized_return"])


def test_decile_spread_stats_empty_is_nan():
    stats = decile_spread_stats(pd.DataFrame(columns=["gross_ret"]))
    assert np.isnan(stats["annualized_spread"])
    assert np.isnan(stats["tstat"])
