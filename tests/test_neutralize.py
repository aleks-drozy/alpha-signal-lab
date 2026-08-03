"""Unit tests for the per-date OLS residualization math in neutralize.py,
on tiny synthetic examples (no network, no real pipeline data).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.neutralize import neutralize_predictions, per_date_residual


def test_per_date_residual_perfect_linear_fit_is_zero():
    # y = 1 + 2*x exactly: an OLS fit through it must leave zero residual.
    group = pd.DataFrame({"momentum": [0.0, 1.0, 2.0, 3.0], "pred": [1.0, 3.0, 5.0, 7.0]})
    resid = per_date_residual(group, pred_col="pred", momentum_col="momentum")
    assert np.allclose(resid.to_numpy(), 0.0, atol=1e-8)


def test_per_date_residual_removes_linear_component_leaves_noise():
    # y = 2*x + noise: OLS must remove the linear-in-x component, so the
    # residual is left uncorrelated with x and mean-zero, even though it is
    # not bitwise equal to (noise - mean(noise)) since OLS also removes any
    # linear-in-x component the noise itself happens to carry.
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    noise = np.array([0.1, -0.2, 0.05, 0.15, -0.1])
    y = 2.0 * x + noise
    group = pd.DataFrame({"momentum": x, "pred": y})
    resid = per_date_residual(group, pred_col="pred", momentum_col="momentum").to_numpy()
    assert np.isclose(resid.mean(), 0.0, atol=1e-8)
    assert np.isclose(np.corrcoef(x, resid)[0, 1], 0.0, atol=1e-8)


def test_per_date_residual_nan_when_momentum_degenerate():
    # All-equal momentum has zero spread: cannot fit a slope, must be NaN,
    # not silently zero (a zero here would falsely claim "no signal").
    group = pd.DataFrame({"momentum": [0.5, 0.5, 0.5], "pred": [1.0, 2.0, 3.0]})
    resid = per_date_residual(group, pred_col="pred", momentum_col="momentum")
    assert resid.isna().all()


def test_per_date_residual_nan_when_too_few_points():
    group = pd.DataFrame({"momentum": [0.0, 1.0], "pred": [1.0, 2.0]})
    resid = per_date_residual(group, pred_col="pred", momentum_col="momentum")
    assert resid.isna().all()


def test_neutralize_predictions_operates_per_date_independently():
    # Two dates, each perfectly linear in momentum but with different
    # slopes; the per-date grouping must fit each date separately, not pool.
    df = pd.DataFrame(
        {
            "date": ["2020-01-01"] * 3 + ["2020-01-02"] * 3,
            "ret_252d_skip21": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            "y_pred": [0.0, 1.0, 2.0, 0.0, 5.0, 10.0],
        }
    )
    out = neutralize_predictions(df, pred_col="y_pred", momentum_col="ret_252d_skip21")
    assert np.allclose(out["y_pred_neutralized"].to_numpy(), 0.0, atol=1e-8)
