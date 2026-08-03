"""Unit tests for diagnose.py's pure-math pieces on tiny synthetic examples,
plus an integration/determinism test of run_diagnose against the committed
fixture universe (via panel_with_folds) with real (small) LightGBM fits. No
network calls, no real pipeline data.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from alpha_lab.config import FEATURE_COLUMNS
from alpha_lab.diagnose import (
    build_summary,
    feature_correlation,
    hyperparameter_binding,
    ic_stability,
    momentum_agreement,
    shap_top_feature_per_fold,
    tail_divergence,
    turnover_comparison,
)
from alpha_lab.model import fit_predict_fold


# ---------------------------------------------------------------------------
# feature_correlation
# ---------------------------------------------------------------------------


def test_feature_correlation_matrix_symmetric_with_unit_diagonal():
    rng = np.random.default_rng(0)
    n = 200
    data = {col: rng.normal(size=n) for col in FEATURE_COLUMNS}
    features = pd.DataFrame(data)

    result = feature_correlation(features)
    matrix = result["matrix"]

    for a in FEATURE_COLUMNS:
        assert matrix[a][a] == pytest.approx(1.0, abs=1e-8)
        for b in FEATURE_COLUMNS:
            assert matrix[a][b] == pytest.approx(matrix[b][a], abs=1e-8)


def test_feature_correlation_flags_redundant_pair_above_threshold():
    n = 100
    base = np.linspace(0.0, 1.0, n)
    data = {col: base + 0.0001 * i for i, col in enumerate(FEATURE_COLUMNS)}
    # Two columns built to be exactly rank-identical (perfectly correlated),
    # the rest spread out with tiny per-column jitter that still ranks the
    # same order, so every pair here is highly correlated -- pick two known
    # feature names to check explicitly.
    data["vol_21d"] = base
    data["vol_63d"] = base
    features = pd.DataFrame(data)

    result = feature_correlation(features)
    pairs = result["redundant_pairs"]
    found = [p for p in pairs if {p["feature_a"], p["feature_b"]} == {"vol_21d", "vol_63d"}]
    assert len(found) == 1
    assert found[0]["correlation"] == pytest.approx(1.0, abs=1e-8)


def test_feature_correlation_no_redundant_pairs_when_independent():
    rng = np.random.default_rng(1)
    n = 500
    data = {col: rng.normal(size=n) for col in FEATURE_COLUMNS}
    features = pd.DataFrame(data)

    result = feature_correlation(features)
    for pair in result["redundant_pairs"]:
        assert abs(pair["correlation"]) >= 0.75


def test_feature_correlation_redundant_pairs_sorted_descending_by_abs_corr():
    rng = np.random.default_rng(2)
    n = 300
    base = rng.normal(size=n)
    data = {col: rng.normal(size=n) for col in FEATURE_COLUMNS}
    data["ret_5d"] = base
    data["ret_21d"] = base + rng.normal(scale=0.01, size=n)
    data["ret_63d"] = base + rng.normal(scale=0.3, size=n)
    features = pd.DataFrame(data)

    result = feature_correlation(features)
    abs_corrs = [abs(p["correlation"]) for p in result["redundant_pairs"]]
    assert abs_corrs == sorted(abs_corrs, reverse=True)


# ---------------------------------------------------------------------------
# shap_top_feature_per_fold
# ---------------------------------------------------------------------------


def test_shap_top_feature_per_fold_argmax_and_counts():
    shap_importance = {
        "feature_columns": ["a", "b", "c"],
        "rows": [
            {"fold_id": 0, "a": 0.1, "b": 0.5, "c": 0.2},
            {"fold_id": 1, "a": 0.4, "b": 0.1, "c": 0.05},
            {"fold_id": 2, "a": 0.05, "b": 0.6, "c": 0.02},
        ],
    }
    result = shap_top_feature_per_fold(shap_importance)

    assert result["per_fold"] == [
        {"fold_id": 0, "top_feature": "b"},
        {"fold_id": 1, "top_feature": "a"},
        {"fold_id": 2, "top_feature": "b"},
    ]
    assert result["counts"] == {"a": 1, "b": 2, "c": 0}


def test_shap_top_feature_per_fold_single_dominant_feature_all_folds():
    shap_importance = {
        "feature_columns": ["x", "y"],
        "rows": [{"fold_id": i, "x": 0.9, "y": 0.1} for i in range(5)],
    }
    result = shap_top_feature_per_fold(shap_importance)
    assert result["counts"]["x"] == 5
    assert result["counts"]["y"] == 0


# ---------------------------------------------------------------------------
# momentum_agreement
# ---------------------------------------------------------------------------


def test_momentum_agreement_perfect_agreement_gives_spearman_one():
    dates = pd.to_datetime(["2020-01-01"] * 3 + ["2020-01-02"] * 3)
    oos = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["A", "B", "C"] * 2,
            "y_pred": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
        }
    )
    features = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["A", "B", "C"] * 2,
            "ret_252d_skip21": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
        }
    )
    result = momentum_agreement(oos, features)
    assert result["mean_daily_spearman"] == pytest.approx(1.0)
    assert result["n_days"] == 2


def test_momentum_agreement_perfect_disagreement_gives_spearman_negative_one():
    dates = pd.to_datetime(["2020-01-01"] * 3)
    oos = pd.DataFrame({"date": dates, "ticker": ["A", "B", "C"], "y_pred": [1.0, 2.0, 3.0]})
    features = pd.DataFrame(
        {"date": dates, "ticker": ["A", "B", "C"], "ret_252d_skip21": [3.0, 2.0, 1.0]}
    )
    result = momentum_agreement(oos, features)
    assert result["mean_daily_spearman"] == pytest.approx(-1.0)


def test_momentum_agreement_skips_degenerate_days_no_crash():
    # Fewer than 3 names on a date must be skipped, not crash.
    dates = pd.to_datetime(["2020-01-01", "2020-01-01"])
    oos = pd.DataFrame({"date": dates, "ticker": ["A", "B"], "y_pred": [1.0, 2.0]})
    features = pd.DataFrame({"date": dates, "ticker": ["A", "B"], "ret_252d_skip21": [1.0, 2.0]})
    result = momentum_agreement(oos, features)
    assert result["n_days"] == 0
    assert result["mean_daily_spearman"] != result["mean_daily_spearman"]  # NaN


# ---------------------------------------------------------------------------
# ic_stability
# ---------------------------------------------------------------------------


def test_ic_stability_perfect_model_beats_random_momentum():
    dates = pd.to_datetime(["2020-01-01"] * 3 + ["2020-01-08"] * 3)
    oos = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["A", "B", "C"] * 2,
            "fold_id": [0, 0, 0, 1, 1, 1],
            "y_pred": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "label": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        }
    )
    features = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["A", "B", "C"] * 2,
            "ret_252d_skip21": [3.0, 1.0, 2.0, 2.0, 3.0, 1.0],
        }
    )
    result = ic_stability(oos, features)
    assert result["model"]["mean"] == pytest.approx(1.0)
    assert result["n_folds"] == 2


def test_ic_stability_cross_fold_correlation_hand_computed():
    # Two folds; model fold-IC = [1.0, -1.0], momentum fold-IC = [1.0, -1.0]
    # (perfectly aligned), so cross_fold_correlation must be 1.0.
    dates = pd.to_datetime(["2020-01-01"] * 3 + ["2020-01-08"] * 3)
    oos = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["A", "B", "C"] * 2,
            "fold_id": [0, 0, 0, 1, 1, 1],
            "y_pred": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
            "label": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        }
    )
    features = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["A", "B", "C"] * 2,
            "ret_252d_skip21": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
        }
    )
    result = ic_stability(oos, features)
    assert result["cross_fold_correlation"] == pytest.approx(1.0, abs=1e-8)


# ---------------------------------------------------------------------------
# tail_divergence
# ---------------------------------------------------------------------------


def _six_name_day(date: str, model_order, momentum_order, fwd_rets) -> tuple[pd.DataFrame, pd.DataFrame]:
    tickers = [f"T{i}" for i in range(6)]
    oos = pd.DataFrame(
        {
            "date": [pd.Timestamp(date)] * 6,
            "ticker": tickers,
            "y_pred": [model_order.index(t) for t in tickers],
            "fwd_ret_5d": [fwd_rets[t] for t in tickers],
        }
    )
    features = pd.DataFrame(
        {
            "date": [pd.Timestamp(date)] * 6,
            "ticker": tickers,
            "ret_252d_skip21": [momentum_order.index(t) for t in tickers],
        }
    )
    return oos, features


def test_tail_divergence_identical_picks_never_diverge():
    tickers = [f"T{i}" for i in range(6)]
    fwd_rets = {t: float(i) * 0.01 for i, t in enumerate(tickers)}
    oos, features = _six_name_day("2020-01-01", tickers, tickers, fwd_rets)

    result = tail_divergence(oos, features, n=3)
    assert result["n_rebalance_days"] == 1
    assert result["n_diverged_days"] == 0
    assert result["pct_days_diverged"] == pytest.approx(0.0)


def test_tail_divergence_fully_reversed_picks_diverge_every_day():
    tickers = [f"T{i}" for i in range(6)]
    reversed_order = list(reversed(tickers))
    fwd_rets = {t: float(i) * 0.01 for i, t in enumerate(tickers)}
    oos, features = _six_name_day("2020-01-01", tickers, reversed_order, fwd_rets)

    result = tail_divergence(oos, features, n=3)
    assert result["n_diverged_days"] == 1
    assert result["pct_days_diverged"] == pytest.approx(1.0)


def test_tail_divergence_fields_within_unit_interval():
    tickers = [f"T{i}" for i in range(6)]
    reversed_order = list(reversed(tickers))
    fwd_rets = {t: float(i) * 0.01 for i, t in enumerate(tickers)}
    oos, features = _six_name_day("2020-01-01", tickers, reversed_order, fwd_rets)

    result = tail_divergence(oos, features, n=3)
    assert 0.0 <= result["pct_days_diverged"] <= 1.0
    if result["losing_swap_rate"] == result["losing_swap_rate"]:
        assert 0.0 <= result["losing_swap_rate"] <= 1.0


def test_tail_divergence_skips_dates_with_too_few_names():
    oos = pd.DataFrame(
        {"date": [pd.Timestamp("2020-01-01")] * 4, "ticker": ["A", "B", "C", "D"], "y_pred": [1, 2, 3, 4]}
    )
    features = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-01")] * 4,
            "ticker": ["A", "B", "C", "D"],
            "ret_252d_skip21": [1, 2, 3, 4],
        }
    )
    result = tail_divergence(oos, features, n=3)
    assert result["n_rebalance_days"] == 0
    assert result["pct_days_diverged"] != result["pct_days_diverged"]  # NaN


# ---------------------------------------------------------------------------
# turnover_comparison
# ---------------------------------------------------------------------------


def test_turnover_comparison_within_unit_interval_and_reuses_evaluate_logic():
    rng = np.random.default_rng(3)
    n_dates = 20
    tickers = [f"T{i}" for i in range(8)]
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    rows = []
    for d in dates:
        for t in tickers:
            rows.append(
                {
                    "date": d,
                    "ticker": t,
                    "y_pred": rng.normal(),
                    "fwd_ret_5d": rng.normal(scale=0.01),
                }
            )
    oos = pd.DataFrame(rows)
    features = oos[["date", "ticker"]].copy()
    features["ret_252d_skip21"] = rng.normal(size=len(features))

    result = turnover_comparison(oos, features)
    assert 0.0 <= result["model_mean_turnover"] <= 1.0
    assert 0.0 <= result["momentum_mean_turnover"] <= 1.0


# ---------------------------------------------------------------------------
# build_summary
# ---------------------------------------------------------------------------


def test_build_summary_returns_between_5_and_7_grounded_lines():
    corr = {
        "matrix": {},
        "redundant_pairs": [{"feature_a": "vol_21d", "feature_b": "vol_63d", "correlation": 0.807}],
    }
    shap_top = {
        "per_fold": [{"fold_id": i, "top_feature": "vol_ratio_21_63"} for i in range(3)],
        "counts": {"vol_ratio_21_63": 3},
    }
    mom_agree = {"mean_daily_spearman": 0.0005, "median_daily_spearman": 0.0, "std_daily_spearman": 0.01, "n_days": 100}
    ic_stab = {
        "model": {"mean": 0.01, "std": 0.03, "std_over_abs_mean": 3.32},
        "momentum": {"mean": 0.02, "std": 0.03, "std_over_abs_mean": 1.61},
        "cross_fold_correlation": 0.18,
        "n_folds": 21,
    }
    tail_div = {
        "pct_days_diverged": 0.916,
        "model_tail_spread_on_diverged_days": 0.01,
        "momentum_tail_spread_on_diverged_days": 0.056,
        "spread_ratio": 5.6,
        "losing_swap_rate": 0.524,
        "n_rebalance_days": 100,
        "n_diverged_days": 91,
    }
    turnover = {"model_mean_turnover": 0.682, "momentum_mean_turnover": 0.154}
    hp_binding = {"note": "some honest note about leaf counts"}

    lines = build_summary(corr, shap_top, mom_agree, ic_stab, tail_div, turnover, hp_binding)

    assert 5 <= len(lines) <= 7
    assert all(isinstance(line, str) and line for line in lines)
    assert "0.0005" in lines[2]
    assert "vol_21d" in lines[0] and "vol_63d" in lines[0]
    assert lines[-1] == hp_binding["note"]


def test_build_summary_handles_nan_fields_without_crashing():
    corr = {"matrix": {}, "redundant_pairs": []}
    shap_top = {"per_fold": [], "counts": {}}
    mom_agree = {"mean_daily_spearman": float("nan"), "median_daily_spearman": float("nan"), "std_daily_spearman": float("nan"), "n_days": 0}
    ic_stab = {
        "model": {"mean": float("nan"), "std": float("nan"), "std_over_abs_mean": float("nan")},
        "momentum": {"mean": float("nan"), "std": float("nan"), "std_over_abs_mean": float("nan")},
        "cross_fold_correlation": float("nan"),
        "n_folds": 0,
    }
    tail_div = {
        "pct_days_diverged": float("nan"),
        "model_tail_spread_on_diverged_days": float("nan"),
        "momentum_tail_spread_on_diverged_days": float("nan"),
        "spread_ratio": float("nan"),
        "losing_swap_rate": float("nan"),
        "n_rebalance_days": 0,
        "n_diverged_days": 0,
    }
    turnover = {"model_mean_turnover": float("nan"), "momentum_mean_turnover": float("nan")}
    hp_binding = {"note": "no data"}

    lines = build_summary(corr, shap_top, mom_agree, ic_stab, tail_div, turnover, hp_binding)
    assert len(lines) >= 5


# ---------------------------------------------------------------------------
# hyperparameter_binding (real, small LightGBM fit on the fixture universe)
# ---------------------------------------------------------------------------


def test_hyperparameter_binding_reports_leaf_counts_on_fixture_data(panel_with_folds):
    panel, folds, _ = panel_with_folds
    result = hyperparameter_binding(panel, folds)

    assert result["num_leaves_configured"] == 31
    summary = result["leaf_count_summary"]
    assert summary["n_trees"] > 0
    assert summary["mean_leaves_per_tree"] > 0
    assert 1 <= summary["max_leaves_per_tree"] <= 31
    assert 0.0 <= summary["pct_trees_at_num_leaves_cap"] <= 1.0
    assert result["final_fold_id"] == folds[-1].fold_id
    assert isinstance(result["note"], str) and len(result["note"]) > 0


def test_hyperparameter_binding_raises_on_empty_folds():
    with pytest.raises(RuntimeError):
        hyperparameter_binding(pd.DataFrame(), [])


# ---------------------------------------------------------------------------
# run_diagnose: schema, bounds, and determinism against the fixture universe
# ---------------------------------------------------------------------------


@pytest.fixture()
def diagnose_env(panel_with_folds, tmp_path, monkeypatch):
    """Real fixture-universe panel and real OOS predictions (built with
    model.py's own fit_predict_fold, same as run_model would produce),
    plus a synthetic shap_importance_over_time.json shaped like explain.py's
    real output. All diagnose.py artifact paths are pointed at tmp_path."""
    import alpha_lab.diagnose as diagnose_mod

    panel, folds, _ = panel_with_folds
    fold_predictions = [fit_predict_fold(panel, fold) for fold in folds]
    oos = pd.concat(fold_predictions, ignore_index=True)

    shap_importance = {
        "feature_columns": list(FEATURE_COLUMNS),
        "rows": [
            {"fold_id": fold.fold_id, **{col: float(abs(hash((fold.fold_id, col))) % 100) / 100.0 for col in FEATURE_COLUMNS}}
            for fold in folds
        ],
    }
    shap_path = tmp_path / "shap_importance_over_time.json"
    with open(shap_path, "w", encoding="utf-8") as f:
        json.dump(shap_importance, f)

    diagnosis_path = tmp_path / "diagnosis.json"

    monkeypatch.setattr(diagnose_mod, "run_features", lambda force=False: panel)
    monkeypatch.setattr(diagnose_mod, "run_model", lambda force=False: oos)
    monkeypatch.setattr(diagnose_mod, "SHAP_IMPORTANCE_OVER_TIME_PATH", shap_path)
    monkeypatch.setattr(diagnose_mod, "DIAGNOSIS_PATH", diagnosis_path)

    return diagnose_mod, diagnosis_path


def test_run_diagnose_writes_expected_schema(diagnose_env):
    diagnose_mod, diagnosis_path = diagnose_env

    result = diagnose_mod.run_diagnose(force=True)

    assert diagnosis_path.exists()
    for key in [
        "feature_correlation",
        "shap_top_feature_per_fold",
        "momentum_agreement",
        "ic_stability",
        "tail_divergence",
        "turnover_comparison",
        "hyperparameter_binding",
        "summary",
    ]:
        assert key in result

    matrix = result["feature_correlation"]["matrix"]
    for col in FEATURE_COLUMNS:
        assert matrix[col][col] == pytest.approx(1.0, abs=1e-6)
        for other in FEATURE_COLUMNS:
            assert matrix[col][other] == pytest.approx(matrix[other][col], abs=1e-8)

    for field in ("pct_days_diverged", "losing_swap_rate"):
        value = result["tail_divergence"][field]
        if value == value:
            assert 0.0 <= value <= 1.0

    turnover = result["turnover_comparison"]
    for field in ("model_mean_turnover", "momentum_mean_turnover"):
        value = turnover[field]
        if value == value:
            assert 0.0 <= value <= 1.0

    assert 5 <= len(result["summary"]) <= 7


def _dumps_nan_safe(obj) -> str:
    # NaN != NaN under normal dict equality, but json.dumps renders it as the
    # literal token `NaN` deterministically, so a string comparison is a
    # valid stand-in for structural equality here (the fixture universe is
    # only 3 tickers, so several fields are legitimately NaN).
    return json.dumps(obj, sort_keys=True)


def test_run_diagnose_is_deterministic(diagnose_env):
    diagnose_mod, _ = diagnose_env

    result_a = diagnose_mod.run_diagnose(force=True)
    result_b = diagnose_mod.run_diagnose(force=True)

    assert _dumps_nan_safe(result_a) == _dumps_nan_safe(result_b)


def test_run_diagnose_skips_recompute_without_force(diagnose_env, monkeypatch):
    diagnose_mod, diagnosis_path = diagnose_env

    first = diagnose_mod.run_diagnose(force=True)

    def _boom(force=False):
        raise AssertionError("run_features should not be called when the cache is valid")

    monkeypatch.setattr(diagnose_mod, "run_features", _boom)
    second = diagnose_mod.run_diagnose(force=False)

    assert _dumps_nan_safe(first) == _dumps_nan_safe(second)
