"""Tests for report.py:

1. A dashboard schema round-trip test: every field path the dashboard JS
   (docs/index.html) actually reads from results.json must be present in a
   real build_results_json() output, so a renamed/removed key in Python that
   forgot to update the JS (or vice versa) fails CI instead of silently
   breaking the live dashboard.
2. The PASS-verdict branch of render_report_md/build_results_json, exercised
   explicitly on synthetic PASS-shaped input (previously only ever exercised
   as NULL for real).
3. Graceful degradation when each addendum artifact (badges.json,
   shap_stability.json, neutralization.json, regime_breakdown.json,
   cpcv.json) is individually missing: report.py must not crash.

No network calls; all upstream artifacts are monkeypatched to point at
tmp_path files written directly by the test, and build_dashboard_series is
stubbed so this does not require real parquet pipeline outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import alpha_lab.report as report_mod

INDEX_HTML_PATH = Path(__file__).resolve().parents[1] / "docs" / "index.html"

# Field paths the dashboard JS (docs/index.html) reads off the top-level
# results object and off results.addendum's sub-objects, derived by reading
# renderHeader/renderHero/renderEquityCurve/renderRollingIC/renderFoldIC/
# renderFeatureImportance/renderBadges/renderNeutralization/renderRegime/
# renderShapOverTime/renderCPCV/renderReproducibility in docs/index.html.
# If the JS gains a new field access, extend this list.
SCALAR_PATHS = [
    "verdict.met",
    "verdict.ic_threshold",
    "verdict.tstat_threshold",
    "ic_comparison.honest_mean_ic",
    "ic_comparison.honest_tstat",
    "ic_comparison.leaky_mean_ic",
    "ic_comparison.leaky_tstat",
    "data_span.ingest_start",
    "data_span.oos_date_range",
    "generated_at",
    "equity_curves.dates",
    "equity_curves.model_gross",
    "equity_curves.model_net",
    "equity_curves.momentum_gross",
    "equity_curves.buy_hold_gross",
    "rolling_ic.dates",
    "rolling_ic.values",
    "feature_importance.mean_abs_shap_final_fold",
    "feature_importance.top_feature_churns",
    "addendum.note",
    "addendum.neutralization",
    "addendum.regime_breakdown",
    "addendum.shap_importance_over_time",
    "addendum.cpcv",
    "addendum.provenance",
]

LIST_PATHS = ["audits", "per_fold_ic"]

# Sub-fields read off the first item of each list-valued path.
LIST_ITEM_FIELDS = {
    "audits": ["test", "passed", "description"],
    "per_fold_ic": ["fold_id", "mean_ic", "test_start", "test_end", "n_days"],
    "feature_importance.mean_abs_shap_final_fold": ["feature", "mean_abs_shap"],
}

# Sub-fields read off addendum.<key> when that artifact is present.
ADDENDUM_SUB_PATHS = {
    "neutralization": ["description", "raw.mean", "raw.newey_west_tstat", "neutralized.mean", "neutralized.newey_west_tstat"],
    "regime_breakdown": ["description", "regime_rank_ic", "per_ticker_ic", "per_ticker_ic_summary.n_tickers_total",
                          "per_ticker_ic_summary.mean_across_tickers", "per_ticker_ic_summary.std_across_tickers",
                          "per_ticker_ic_summary.n_tickers_with_positive_ic"],
    "shap_importance_over_time": ["rows", "feature_columns"],
    "cpcv": ["description", "oos_ic_distribution.mean", "oos_ic_distribution.std",
             "oos_ic_distribution.ci_95_normal", "oos_ic_distribution.ci_95_percentile",
             "paths", "n_paths_sampled", "n_combinations_total", "pbo.pbo", "pbo.logit_pbo", "pbo.note"],
    "provenance": ["run_fingerprint_sha256", "random_seed", "python_version", "package_versions"],
}


def _resolve(d: dict, dotted_path: str):
    node = d
    for part in dotted_path.split("."):
        assert isinstance(node, dict), f"expected dict while resolving '{dotted_path}' at '{part}'"
        assert part in node, f"missing key '{part}' while resolving '{dotted_path}'"
        node = node[part]
    return node


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def _synthetic_metrics(criterion_met: bool) -> dict:
    rank_ic = {
        "mean": 0.05 if criterion_met else 0.001,
        "std": 0.1,
        "newey_west_tstat": 3.0 if criterion_met else 0.5,
        "hit_rate": 0.55,
        "n_days": 100,
    }
    equity = {"sharpe": 1.2, "max_drawdown": -0.1, "annualized_return": 0.08, "mean_turnover": 0.3}
    strategy = {
        "rank_ic": rank_ic,
        "decile_spread": {"annualized_spread": 0.1, "tstat": 2.5},
        "equity_curve_gross": equity,
        "equity_curve_net": equity,
    }
    return {
        "model": strategy,
        "momentum_baseline": strategy,
        "buy_hold_baseline": {"equity_curve_gross": equity, "equity_curve_net": equity},
        "success_criterion": {"ic_threshold": 0.02, "tstat_threshold": 2.0, "met": criterion_met},
        "n_oos_rows": 500,
        "oos_date_range": ["2016-01-04", "2020-12-31"],
    }


def _synthetic_leaky() -> dict:
    return {
        "rank_ic": {"mean": 0.15, "std": 0.2, "newey_west_tstat": 6.0, "hit_rate": 0.6, "n_days": 100},
        "decile_spread": {"annualized_spread": 0.3, "tstat": 5.0},
        "equity_curve_gross": {"sharpe": 2.0, "max_drawdown": -0.05, "annualized_return": 0.2, "mean_turnover": 0.3},
        "equity_curve_net": {"sharpe": 1.8, "max_drawdown": -0.06, "annualized_return": 0.18, "mean_turnover": 0.3},
        "n_rows": 500,
        "n_folds": 5,
        "cv_type": "naive_shuffled_kfold_no_purge_no_embargo",
        "date_range": ["2016-01-04", "2020-12-31"],
    }


def _synthetic_shap_stability() -> dict:
    return {
        "per_fold_mean_abs_shap": {"0": {"ret_5d": 0.02, "ret_21d": 0.01}, "1": {"ret_5d": 0.03, "ret_21d": 0.015}},
        "per_fold_rank": {0: {"ret_5d": 1, "ret_21d": 2}, 1: {"ret_5d": 1, "ret_21d": 2}},
        "top_feature_churns": False,
        "final_fold_id": 1,
    }


def _synthetic_neutralization() -> dict:
    return {
        "description": "desc",
        "momentum_feature": "ret_252d_skip21",
        "raw": {"mean": 0.02, "std": 0.1, "newey_west_tstat": 2.5, "hit_rate": 0.55, "n_days": 100},
        "neutralized": {"mean": 0.01, "std": 0.1, "newey_west_tstat": 1.0, "hit_rate": 0.5, "n_days": 100},
        "n_oos_rows": 500,
        "n_rows_dropped_degenerate_date": 0,
    }


def _synthetic_regime() -> dict:
    return {
        "description": "desc",
        "vol_window_days": 21,
        "regime_rank_ic": {
            "high_vol": {"mean": 0.01, "std": 0.1, "newey_west_tstat": 1.0, "hit_rate": 0.5, "n_days": 50},
            "low_vol": {"mean": 0.03, "std": 0.1, "newey_west_tstat": 2.0, "hit_rate": 0.6, "n_days": 50},
        },
        "per_ticker_ic": [{"ticker": "AAPL", "spearman_ic": 0.1, "n_obs": 100}],
        "per_ticker_ic_summary": {
            "mean_across_tickers": 0.05,
            "std_across_tickers": 0.1,
            "n_tickers_with_positive_ic": 18,
            "n_tickers_total": 30,
        },
    }


def _synthetic_shap_over_time() -> dict:
    return {
        "feature_columns": ["ret_5d", "ret_21d"],
        "rows": [{"fold_id": 0, "test_end": "2016-06-30", "ret_5d": 0.02, "ret_21d": 0.01}],
    }


def _synthetic_cpcv() -> dict:
    return {
        "description": "desc",
        "n_blocks_available": 10,
        "n_test_blocks_per_path": 2,
        "n_combinations_total": 45,
        "n_paths_sampled": 20,
        "seed": 42,
        "purge_days": 5,
        "lgbm_params": {},
        "paths": [{"path_id": 0, "oos_rank_ic": {"mean": 0.02}, "is_rank_ic": {"mean": 0.03}}],
        "oos_ic_distribution": {"mean": 0.02, "std": 0.05, "n": 20, "ci_95_normal": [0.0, 0.04], "ci_95_percentile": [-0.01, 0.05]},
        "is_ic_distribution": {"mean": 0.03, "std": 0.05, "n": 20, "ci_95_normal": [0.01, 0.05], "ci_95_percentile": [0.0, 0.06]},
        "pbo": {"n_paths": 20, "is_median_ic": 0.03, "oos_median_ic": 0.02, "pbo": 0.3, "logit_pbo": -0.85, "note": "note"},
        "runtime_seconds": 12.3,
    }


def _synthetic_provenance() -> dict:
    return {
        "python_version": "3.11.5 (main)",
        "package_versions": {"pandas": "2.2.0", "lightgbm": "4.3.0"},
        "random_seed": 42,
        "run_fingerprint_sha256": "a" * 64,
    }


def _synthetic_badges() -> list:
    return [{"test": "determinism", "passed": True, "description": "desc"}]


def _synthetic_series() -> dict:
    return {
        "available": True,
        "equity_curves": {
            "dates": ["2016-01-04", "2016-01-11"],
            "model_gross": [1.0, 1.01],
            "model_net": [1.0, 1.005],
            "momentum_gross": [1.0, 0.99],
            "buy_hold_gross": [1.0, 1.02],
        },
        "rolling_ic": {"dates": ["2016-01-04"], "values": [0.02]},
        "per_fold_ic": [{"fold_id": 0, "mean_ic": 0.02, "n_days": 50, "test_start": "2016-01-04", "test_end": "2016-06-30"}],
        "n_oos_rows": 500,
    }


@pytest.fixture()
def full_synthetic_env(tmp_path, monkeypatch):
    """Points every artifact path report.py reads at tmp_path files, writes
    a complete, PASS-shaped set of upstream artifacts, and stubs
    build_dashboard_series so this doesn't require real parquet outputs."""
    metrics_path = tmp_path / "metrics.json"
    leaky_path = tmp_path / "leaky_twin_metrics.json"
    shap_stability_path = tmp_path / "shap_stability.json"
    neutralization_path = tmp_path / "neutralization.json"
    regime_path = tmp_path / "regime_breakdown.json"
    shap_time_path = tmp_path / "shap_importance_over_time.json"
    cpcv_path = tmp_path / "cpcv.json"
    provenance_path = tmp_path / "provenance.json"

    _write_json(metrics_path, _synthetic_metrics(criterion_met=True))
    _write_json(leaky_path, _synthetic_leaky())
    _write_json(shap_stability_path, _synthetic_shap_stability())
    _write_json(neutralization_path, _synthetic_neutralization())
    _write_json(regime_path, _synthetic_regime())
    _write_json(shap_time_path, _synthetic_shap_over_time())
    _write_json(cpcv_path, _synthetic_cpcv())
    _write_json(provenance_path, _synthetic_provenance())

    monkeypatch.setattr(report_mod, "METRICS_PATH", metrics_path)
    monkeypatch.setattr(report_mod, "LEAKY_TWIN_METRICS_PATH", leaky_path)
    monkeypatch.setattr(report_mod, "SHAP_STABILITY_PATH", shap_stability_path)
    monkeypatch.setattr(report_mod, "NEUTRALIZATION_PATH", neutralization_path)
    monkeypatch.setattr(report_mod, "REGIME_BREAKDOWN_PATH", regime_path)
    monkeypatch.setattr(report_mod, "SHAP_IMPORTANCE_OVER_TIME_PATH", shap_time_path)
    monkeypatch.setattr(report_mod, "CPCV_PATH", cpcv_path)
    monkeypatch.setattr(report_mod, "PROVENANCE_PATH", provenance_path)
    monkeypatch.setattr(report_mod, "build_dashboard_series", lambda: _synthetic_series())

    badges_path = tmp_path / "badges.json"
    _write_json(badges_path, _synthetic_badges())
    monkeypatch.setattr(report_mod, "BADGES_PATH", badges_path)

    return tmp_path


# ---------------------------------------------------------------------------
# 1. Dashboard schema round-trip
# ---------------------------------------------------------------------------


def test_index_html_still_reads_only_documented_field_paths():
    """Guard against silently missing a field the JS started reading: every
    dotted `results.x.y` / `addendum.x.y` access this test knows about must
    still appear (or a superset) in docs/index.html. This does not prove
    completeness on its own (see docstring on SCALAR_PATHS), but it catches
    the JS being edited without SCALAR_PATHS being updated to match, since
    a genuinely new nested access elsewhere in this file will simply not be
    checked -- the round-trip test below is the real regression guard."""
    text = INDEX_HTML_PATH.read_text(encoding="utf-8")
    for path in SCALAR_PATHS:
        top_level = path.split(".")[0]
        assert top_level in text, f"index.html no longer seems to reference '{top_level}' at all"


def test_build_results_json_has_every_field_the_dashboard_reads(full_synthetic_env):
    results = report_mod.build_results_json()

    for path in SCALAR_PATHS:
        _resolve(results, path)

    for path in LIST_PATHS:
        value = _resolve(results, path)
        assert isinstance(value, list)

    for list_path, fields in LIST_ITEM_FIELDS.items():
        items = _resolve(results, list_path)
        assert len(items) > 0, f"{list_path} synthetic fixture must be non-empty to check item fields"
        for field in fields:
            _resolve(items[0], field)

    addendum = results["addendum"]
    for key, sub_paths in ADDENDUM_SUB_PATHS.items():
        block = addendum[key]
        assert block is not None, f"addendum.{key} should be populated in the full synthetic env"
        for sub_path in sub_paths:
            _resolve(block, sub_path)
        # paths/regime_rank_ic/per_ticker_ic are lists/dicts of items with
        # their own sub-fields the JS reads; spot-check the first item.
        if key == "cpcv":
            assert "mean" in block["paths"][0]["oos_rank_ic"]
        if key == "regime_breakdown":
            first_regime = next(iter(block["regime_rank_ic"].values()))
            assert "mean" in first_regime
            assert "ticker" in block["per_ticker_ic"][0]
            assert "spearman_ic" in block["per_ticker_ic"][0]
        if key == "shap_importance_over_time":
            assert "fold_id" in block["rows"][0] or "test_end" in block["rows"][0]


# ---------------------------------------------------------------------------
# 2. PASS-verdict branch
# ---------------------------------------------------------------------------


def test_build_results_json_pass_verdict(full_synthetic_env):
    results = report_mod.build_results_json()
    assert results["verdict"]["met"] is True
    assert results["verdict"]["text"] == "PASS"


def test_render_report_md_pass_verdict_renders_pass_text(full_synthetic_env):
    results = report_mod.build_results_json()
    md = report_mod.render_report_md(results)
    assert "**PASS**" in md
    assert "TBD" not in md.split("## Verdict")[1].split("## Data")[0]


def test_build_results_json_null_verdict_still_works(full_synthetic_env, tmp_path, monkeypatch):
    null_metrics_path = tmp_path / "metrics_null.json"
    _write_json(null_metrics_path, _synthetic_metrics(criterion_met=False))
    monkeypatch.setattr(report_mod, "METRICS_PATH", null_metrics_path)

    results = report_mod.build_results_json()
    assert results["verdict"]["met"] is False
    assert results["verdict"]["text"] == "NULL"

    md = report_mod.render_report_md(results)
    assert "**NULL**" in md


# ---------------------------------------------------------------------------
# 3. Graceful handling of individually-missing addendum artifacts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "attr",
    ["BADGES_PATH", "SHAP_STABILITY_PATH", "NEUTRALIZATION_PATH", "REGIME_BREAKDOWN_PATH", "CPCV_PATH"],
)
def test_build_results_json_survives_each_missing_addendum_artifact(full_synthetic_env, tmp_path, monkeypatch, attr):
    missing_path = tmp_path / "does_not_exist.json"
    monkeypatch.setattr(report_mod, attr, missing_path)

    # Must not raise.
    results = report_mod.build_results_json()
    md = report_mod.render_report_md(results)
    assert isinstance(md, str) and len(md) > 0


def test_build_results_json_survives_no_metrics_at_all(tmp_path, monkeypatch):
    """run_completed=False path: nothing has been run yet."""
    monkeypatch.setattr(report_mod, "METRICS_PATH", tmp_path / "missing_metrics.json")
    monkeypatch.setattr(report_mod, "LEAKY_TWIN_METRICS_PATH", tmp_path / "missing_leaky.json")
    monkeypatch.setattr(report_mod, "SHAP_STABILITY_PATH", tmp_path / "missing_shap.json")
    monkeypatch.setattr(report_mod, "NEUTRALIZATION_PATH", tmp_path / "missing_neut.json")
    monkeypatch.setattr(report_mod, "REGIME_BREAKDOWN_PATH", tmp_path / "missing_regime.json")
    monkeypatch.setattr(report_mod, "SHAP_IMPORTANCE_OVER_TIME_PATH", tmp_path / "missing_shaptime.json")
    monkeypatch.setattr(report_mod, "CPCV_PATH", tmp_path / "missing_cpcv.json")
    monkeypatch.setattr(report_mod, "PROVENANCE_PATH", tmp_path / "missing_prov.json")
    monkeypatch.setattr(report_mod, "BADGES_PATH", tmp_path / "missing_badges.json")
    monkeypatch.setattr(report_mod, "build_dashboard_series", lambda: {
        "available": False,
        "equity_curves": {"dates": [], "model_gross": [], "model_net": [], "momentum_gross": [], "buy_hold_gross": []},
        "rolling_ic": {"dates": [], "values": []},
        "per_fold_ic": [],
        "n_oos_rows": 0,
    })

    results = report_mod.build_results_json()
    assert results["run_completed"] is False
    assert results["verdict"]["text"] == "TBD (pipeline not yet run)"

    md = report_mod.render_report_md(results)
    assert isinstance(md, str) and len(md) > 0
