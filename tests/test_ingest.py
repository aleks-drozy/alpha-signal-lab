"""Ingest tests against the committed fixture only, no network calls."""

from __future__ import annotations

import pandas as pd
import pytest


def test_raw_path_replaces_caret_for_index_tickers(patch_raw_dir):
    from alpha_lab.ingest import _raw_path

    path = _raw_path("^SPX")
    assert "idx_SPX.csv" in path
    assert path.startswith(str(patch_raw_dir))


def test_ingest_ticker_reads_from_cache_without_network(patch_raw_dir, fixture_universe):
    from alpha_lab.ingest import ingest_ticker

    result = ingest_ticker(fixture_universe[0], refresh=False)

    assert result.ticker == fixture_universe[0]
    assert result.source == "cache"
    assert result.rows == 756
    assert result.first_date == "2018-01-02"


def test_ingest_ticker_missing_cache_falls_back_without_hitting_network(patch_raw_dir, monkeypatch):
    import alpha_lab.ingest as ingest_mod

    monkeypatch.setattr(ingest_mod, "_fetch_stooq", lambda ticker: None)
    monkeypatch.setattr(ingest_mod, "_fetch_yfinance", lambda ticker: None)

    with pytest.raises(RuntimeError):
        ingest_mod.ingest_ticker("NOT_A_REAL_TICKER", refresh=False)


def test_sha256_is_deterministic_and_content_sensitive(patch_raw_dir, fixture_universe):
    from alpha_lab.ingest import _raw_path, _sha256_of_df
    from alpha_lab.config import REQUIRED_ENCODING

    df = pd.read_csv(_raw_path(fixture_universe[0]), encoding=REQUIRED_ENCODING, parse_dates=["Date"])
    hash_a = _sha256_of_df(df)
    hash_b = _sha256_of_df(df)
    assert hash_a == hash_b

    mutated = df.copy()
    mutated.loc[0, "Close"] = mutated.loc[0, "Close"] + 1.0
    hash_mutated = _sha256_of_df(mutated)
    assert hash_mutated != hash_a


def test_fixture_columns_and_types(patch_raw_dir, fixture_universe):
    from alpha_lab.ingest import _raw_path, RAW_COLUMNS
    from alpha_lab.config import REQUIRED_ENCODING

    for ticker in fixture_universe:
        df = pd.read_csv(_raw_path(ticker), encoding=REQUIRED_ENCODING, parse_dates=["Date"])
        assert list(df.columns) == RAW_COLUMNS
        assert df["Date"].is_monotonic_increasing
