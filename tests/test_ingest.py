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


def test_dedupe_and_validate_dates_drops_duplicate_rows_keeps_unique_sorted():
    from alpha_lab.ingest import _dedupe_and_validate_dates

    # Out-of-order with a duplicate date (a repeated 2020-01-02 row, later
    # occurrence has the "corrected" close that should survive via keep="last").
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-03", "2020-01-01", "2020-01-02", "2020-01-02"]),
            "Open": [3.0, 1.0, 2.0, 2.5],
            "High": [3.1, 1.1, 2.1, 2.6],
            "Low": [2.9, 0.9, 1.9, 2.4],
            "Close": [3.0, 1.0, 2.0, 2.5],
            "Volume": [300, 100, 200, 250],
        }
    )

    cleaned = _dedupe_and_validate_dates(df)

    assert list(cleaned["Date"]) == list(pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]))
    assert cleaned["Date"].is_unique
    assert cleaned["Date"].is_monotonic_increasing
    # keep="last": the second 2020-01-02 row (Close=2.5) must be the one kept.
    assert cleaned.loc[cleaned["Date"] == pd.Timestamp("2020-01-02"), "Close"].iloc[0] == pytest.approx(2.5)


def test_dedupe_and_validate_dates_noop_on_already_clean_input():
    from alpha_lab.ingest import _dedupe_and_validate_dates

    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "Open": [1.0, 2.0, 3.0],
            "High": [1.1, 2.1, 3.1],
            "Low": [0.9, 1.9, 2.9],
            "Close": [1.0, 2.0, 3.0],
            "Volume": [100, 200, 300],
        }
    )
    cleaned = _dedupe_and_validate_dates(df)
    pd.testing.assert_frame_equal(cleaned, df)


def test_fetch_one_output_has_unique_sorted_dates_when_source_returns_duplicates(patch_raw_dir, monkeypatch):
    """End-to-end: a source (Stooq/yfinance) hiccup that returns a raw frame
    with duplicate/out-of-order dates must not survive into what
    ingest_ticker caches -- _fetch_one runs dedupe after either source."""
    import alpha_lab.ingest as ingest_mod

    dirty = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-02", "2020-01-01", "2020-01-02"]),
            "Open": [2.0, 1.0, 2.2],
            "High": [2.1, 1.1, 2.3],
            "Low": [1.9, 0.9, 2.1],
            "Close": [2.0, 1.0, 2.2],
            "Volume": [200, 100, 220],
        }
    )
    monkeypatch.setattr(ingest_mod, "_fetch_stooq", lambda ticker: dirty.copy())

    df, source = ingest_mod._fetch_one("DUPTEST")

    assert source == "stooq"
    assert df["Date"].is_unique
    assert df["Date"].is_monotonic_increasing
    assert len(df) == 2


def test_price_sanity_violations_clean_data_has_none(patch_raw_dir, fixture_universe):
    from alpha_lab.ingest import _raw_path, price_sanity_violations
    from alpha_lab.config import REQUIRED_ENCODING

    df = pd.read_csv(_raw_path(fixture_universe[0]), encoding=REQUIRED_ENCODING, parse_dates=["Date"])
    assert price_sanity_violations(df) == []


def test_price_sanity_violations_flags_negative_price():
    from alpha_lab.ingest import price_sanity_violations

    df = pd.DataFrame(
        {
            "Date": pd.bdate_range("2020-01-01", periods=5),
            "Open": [1.0, 2.0, -3.0, 4.0, 5.0],
            "High": [1.1, 2.1, 3.1, 4.1, 5.1],
            "Low": [0.9, 1.9, 2.9, 3.9, 4.9],
            "Close": [1.0, 2.0, 3.0, 4.0, 5.0],
            "Volume": [100] * 5,
        }
    )
    violations = price_sanity_violations(df)
    assert any("Open" in v for v in violations)


def test_price_sanity_violations_flags_long_zero_volume_run():
    from alpha_lab.ingest import price_sanity_violations

    n = 20
    df = pd.DataFrame(
        {
            "Date": pd.bdate_range("2020-01-01", periods=n),
            "Open": [10.0] * n,
            "High": [10.1] * n,
            "Low": [9.9] * n,
            "Close": [10.0] * n,
            "Volume": [0] * 15 + [100] * 5,
        }
    )
    violations = price_sanity_violations(df)
    assert any("zero-volume" in v for v in violations)


def test_price_sanity_violations_flags_unflagged_big_move():
    from alpha_lab.ingest import price_sanity_violations

    closes = [10.0, 10.1, 10.2, 20.0, 20.1]  # ~96% jump
    df = pd.DataFrame(
        {
            "Date": pd.bdate_range("2020-01-01", periods=5),
            "Open": closes,
            "High": [c * 1.01 for c in closes],
            "Low": [c * 0.99 for c in closes],
            "Close": closes,
            "Volume": [1000] * 5,
        }
    )
    violations = price_sanity_violations(df)
    assert any("40%" in v for v in violations)


def test_price_sanity_violations_flags_non_monotonic_and_duplicate_dates():
    from alpha_lab.ingest import price_sanity_violations

    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-02", "2020-01-01", "2020-01-01"]),
            "Open": [1.0, 2.0, 2.0],
            "High": [1.1, 2.1, 2.1],
            "Low": [0.9, 1.9, 1.9],
            "Close": [1.0, 2.0, 2.0],
            "Volume": [100, 200, 200],
        }
    )
    violations = price_sanity_violations(df)
    assert any("monotonic" in v for v in violations)
    assert any("duplicate" in v for v in violations)


def test_check_real_data_sanity_warns_and_flags_manifest_without_raising(
    tmp_path, monkeypatch, fixture_universe, patch_raw_dir, capsys
):
    import alpha_lab.ingest as ingest_mod

    # Copy the committed fixture CSVs into an isolated tmp dir and corrupt
    # the copy there -- never write to the committed tests/fixtures/ files.
    for ticker in fixture_universe:
        src = ingest_mod._raw_path(ticker)
        df = pd.read_csv(src, encoding="utf-8", parse_dates=["Date"])
        df.to_csv(tmp_path / f"{ticker}.csv", index=False)

    monkeypatch.setattr(ingest_mod, "DATA_RAW_DIR", tmp_path)

    bad_ticker = fixture_universe[0]
    bad_path = ingest_mod._raw_path(bad_ticker)
    df = pd.read_csv(bad_path, encoding="utf-8", parse_dates=["Date"])
    df.loc[0, "Close"] = -1.0
    df.to_csv(bad_path, index=False)

    report = ingest_mod.check_real_data_sanity(fixture_universe)

    assert bad_ticker in report
    assert any("Close" in v for v in report[bad_ticker])
    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert bad_ticker in captured.err
    # Never a hard failure: clean tickers are simply absent from the report.
    for ticker in fixture_universe:
        if ticker != bad_ticker:
            assert ticker not in report
