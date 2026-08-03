from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pytest  # noqa: E402

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
FIXTURE_UNIVERSE = ["FIXA", "FIXB", "FIXC"]


@pytest.fixture()
def fixture_universe() -> list[str]:
    return list(FIXTURE_UNIVERSE)


@pytest.fixture()
def patch_raw_dir(monkeypatch: pytest.MonkeyPatch):
    """Point alpha_lab.features (and ingest) at the committed fixture CSVs
    instead of data/raw, so tests never touch real cached data or the
    network."""
    import alpha_lab.features as features_mod
    import alpha_lab.ingest as ingest_mod

    monkeypatch.setattr(features_mod, "DATA_RAW_DIR", FIXTURES_DIR)
    monkeypatch.setattr(ingest_mod, "DATA_RAW_DIR", FIXTURES_DIR)
    return FIXTURES_DIR


@pytest.fixture()
def raw_panel(patch_raw_dir, fixture_universe):
    """Concatenated per-ticker raw (pre rank-transform) feature panel, with
    rows lacking a full feature or forward-return window dropped."""
    from alpha_lab.config import FEATURE_COLUMNS
    from alpha_lab.features import add_excess_label, build_ticker_panel
    import pandas as pd

    panels = [build_ticker_panel(t) for t in fixture_universe]
    panel = pd.concat(panels, ignore_index=True)
    panel = panel.dropna(subset=FEATURE_COLUMNS + ["fwd_ret_5d"]).reset_index(drop=True)
    panel = add_excess_label(panel)
    return panel.rename(columns={"Date": "date"}).sort_values(["date", "ticker"]).reset_index(drop=True)


@pytest.fixture()
def ranked_panel(patch_raw_dir, fixture_universe):
    """Final feature matrix: cross-sectionally rank-transformed, as consumed
    by cv.py / model.py / evaluate.py."""
    from alpha_lab.features import build_feature_matrix

    return build_feature_matrix(universe=fixture_universe)


@pytest.fixture()
def small_cv_config():
    from alpha_lab.config import CVConfig

    return CVConfig(
        horizon_days=5,
        purge_days=5,
        embargo_fraction=0.01,
        train_start="2018-01-02",
        train_end="2019-06-28",
        test_start="2019-08-01",
        test_block_months=3,
    )


@pytest.fixture()
def panel_with_folds(ranked_panel, small_cv_config):
    from alpha_lab.cv import PurgedWalkForward

    folds = PurgedWalkForward(small_cv_config).split(ranked_panel["date"])
    return ranked_panel, folds, small_cv_config
