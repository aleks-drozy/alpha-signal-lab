"""Stooq-primary, yfinance-fallback OHLCV ingestion.

Caches one CSV per ticker to data/raw/, writes data/manifest.json with
source, row count, date range, sha256 per ticker, plus a 3-ticker
Stooq-vs-Yahoo cross-check. Run as: python -m alpha_lab.ingest [--refresh]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd
import requests
import yfinance as yf

from alpha_lab.config import (
    BENCHMARK_TICKER,
    CROSS_CHECK_TICKERS,
    CROSS_CHECK_TOLERANCE_PCT,
    DATA_RAW_DIR,
    INGEST_START_DATE,
    MANIFEST_PATH,
    REQUIRED_ENCODING,
    STOOQ_BENCHMARK_SYMBOL,
    STOOQ_REQUEST_DELAY_SECONDS,
    STOOQ_URL_TEMPLATE,
    UNIVERSE,
)
from alpha_lab.util import atomic_to_csv, atomic_write_json

RAW_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


@dataclass
class TickerResult:
    ticker: str
    source: str
    rows: int
    first_date: str
    last_date: str
    sha256: str


@dataclass
class IngestManifest:
    generated_at: str
    tickers: dict = field(default_factory=dict)
    cross_check: dict = field(default_factory=dict)


def _stooq_url(ticker: str) -> str:
    if ticker == BENCHMARK_TICKER:
        return f"https://stooq.com/q/d/l/?s={STOOQ_BENCHMARK_SYMBOL}&i=d"
    return STOOQ_URL_TEMPLATE.format(symbol=ticker.lower())


def _fetch_stooq(ticker: str) -> pd.DataFrame | None:
    url = _stooq_url(ticker)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    text = resp.content.decode(REQUIRED_ENCODING, errors="replace")
    if "Date" not in text or text.strip().lower().startswith("exceeded"):
        return None
    df = pd.read_csv(io.StringIO(text))
    if df.empty or "Date" not in df.columns:
        return None
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[RAW_COLUMNS].sort_values("Date").reset_index(drop=True)
    return df


def _fetch_yfinance(ticker: str) -> pd.DataFrame | None:
    yf_ticker = ticker if ticker != BENCHMARK_TICKER else "^GSPC"
    raw = yf.download(
        yf_ticker,
        start=INGEST_START_DATE,
        progress=False,
        auto_adjust=True,
        multi_level_index=False,
    )
    if raw is None or raw.empty:
        return None
    raw = raw.reset_index()
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(raw["Date"]),
            "Open": raw["Open"],
            "High": raw["High"],
            "Low": raw["Low"],
            "Close": raw["Close"],
            "Volume": raw["Volume"],
        }
    )
    return df.sort_values("Date").reset_index(drop=True)


def _sha256_of_df(df: pd.DataFrame) -> str:
    payload = df.to_csv(index=False).encode(REQUIRED_ENCODING)
    return hashlib.sha256(payload).hexdigest()


def _raw_path(ticker: str) -> str:
    safe_name = ticker.replace("^", "idx_")
    return str(DATA_RAW_DIR / f"{safe_name}.csv")


def _dedupe_and_validate_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate (date) rows (keep last) and enforce sorted-unique dates
    before anything gets cached, so a source hiccup that returns a repeated
    or out-of-order date never silently corrupts downstream point-in-time
    assumptions."""
    df = df.sort_values("Date").drop_duplicates(subset="Date", keep="last").reset_index(drop=True)
    assert df["Date"].is_monotonic_increasing, "dates not sorted after dedupe"
    assert df["Date"].is_unique, "duplicate dates survived dedupe"
    return df


def _fetch_one(ticker: str) -> tuple[pd.DataFrame, str]:
    df = None
    source = "stooq"
    try:
        df = _fetch_stooq(ticker)
    except requests.RequestException:
        df = None
    if df is None or df.empty:
        df = _fetch_yfinance(ticker)
        source = "yfinance"
    if df is None or df.empty:
        raise RuntimeError(f"could not ingest {ticker} from stooq or yfinance")
    df = df[df["Date"] >= pd.Timestamp(INGEST_START_DATE)].reset_index(drop=True)
    df = _dedupe_and_validate_dates(df)
    return df, source


def ingest_ticker(ticker: str, refresh: bool) -> TickerResult:
    path = _raw_path(ticker)
    if not refresh and DATA_RAW_DIR.exists():
        if os.path.exists(path):
            df = pd.read_csv(path, encoding=REQUIRED_ENCODING, parse_dates=["Date"])
            source = "cache"
            return TickerResult(
                ticker=ticker,
                source=source,
                rows=len(df),
                first_date=str(df["Date"].min().date()),
                last_date=str(df["Date"].max().date()),
                sha256=_sha256_of_df(df),
            )

    df, source = _fetch_one(ticker)
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    atomic_to_csv(df, path, encoding=REQUIRED_ENCODING)
    time.sleep(STOOQ_REQUEST_DELAY_SECONDS)
    return TickerResult(
        ticker=ticker,
        source=source,
        rows=len(df),
        first_date=str(df["Date"].min().date()),
        last_date=str(df["Date"].max().date()),
        sha256=_sha256_of_df(df),
    )


def cross_check_stooq_vs_yahoo(tickers: list[str]) -> dict:
    results = {}
    for ticker in tickers:
        try:
            stooq_df = _fetch_stooq(ticker)
            yahoo_df = _fetch_yfinance(ticker)
        except requests.RequestException as exc:
            results[ticker] = {"ok": False, "error": str(exc)}
            continue
        if stooq_df is None or yahoo_df is None:
            results[ticker] = {"ok": False, "error": "missing data from one source"}
            continue
        merged = pd.merge(
            stooq_df[["Date", "Close"]],
            yahoo_df[["Date", "Close"]],
            on="Date",
            suffixes=("_stooq", "_yahoo"),
        )
        if merged.empty:
            results[ticker] = {"ok": False, "error": "no overlapping dates"}
            continue
        rel_diff = (merged["Close_stooq"] - merged["Close_yahoo"]).abs() / merged["Close_yahoo"]
        max_rel_diff = float(rel_diff.max())
        ok = max_rel_diff <= CROSS_CHECK_TOLERANCE_PCT
        results[ticker] = {
            "ok": ok,
            "max_rel_diff": max_rel_diff,
            "overlapping_dates": int(len(merged)),
        }
    return results


def _enforce_cross_check(manifest: IngestManifest) -> None:
    """A failed Stooq-vs-Yahoo cross-check means the two sources disagree
    beyond tolerance for that ticker; that must not pass silently, so it is
    logged loudly and flagged on the ticker's manifest entry for downstream
    reporting to surface."""
    for ticker, result in manifest.cross_check.items():
        if result.get("ok", False):
            continue
        print(
            f"WARNING: Stooq-vs-Yahoo cross-check failed for {ticker}: {result}",
            file=sys.stderr,
        )
        if ticker in manifest.tickers:
            manifest.tickers[ticker]["cross_check_failed"] = True


def price_sanity_violations(df: pd.DataFrame) -> list[str]:
    """Same sanity checks as tests/test_leakage_audit.py's
    test_adjusted_price_sanity (which only ever runs against the committed
    synthetic fixture), made callable against real cached data so the
    pipeline itself enforces the data-quality guarantee the dashboard's
    adjusted_price_sanity badge advertises, not just the fixture. Returns a
    list of human-readable violation descriptions (empty if clean)."""
    violations: list[str] = []
    for col in ("Open", "High", "Low", "Close"):
        n_bad = int((df[col] <= 0).sum())
        if n_bad:
            violations.append(f"{n_bad} non-positive {col} values")

    if not df["Date"].is_monotonic_increasing:
        violations.append("dates not monotonically increasing")
    if not df["Date"].is_unique:
        violations.append("duplicate dates present")

    zero_volume = (df["Volume"] == 0).astype(int)
    if zero_volume.any():
        run_lengths = zero_volume.groupby((zero_volume != zero_volume.shift()).cumsum()).cumsum()
        max_run = int(run_lengths.max())
        if max_run > 10:
            violations.append(f"zero-volume run of {max_run} consecutive trading days (> 10)")

    daily_ret = df["Close"].pct_change().dropna()
    big_moves = int((daily_ret.abs() > 0.40).sum())
    if big_moves:
        violations.append(f"{big_moves} unflagged single-day move(s) > 40%")

    return violations


def check_real_data_sanity(universe: list[str] | None = None) -> dict[str, list[str]]:
    """Runs price_sanity_violations against every cached data/raw/*.csv for
    `universe` (defaults to UNIVERSE + BENCHMARK_TICKER). Logs a loud
    stderr WARNING per violating ticker but never raises: real market data
    legitimately has circuit-breaker days, dividend/split-adjustment
    artifacts, and other edge cases that a hard failure here would block
    ingestion over, so this is a visibility guarantee, not a gate. Returns a
    dict of ticker -> violation list (only for tickers with at least one
    violation) for callers that want to act on it programmatically."""
    universe = universe if universe is not None else UNIVERSE + [BENCHMARK_TICKER]
    report: dict[str, list[str]] = {}
    for ticker in universe:
        path = _raw_path(ticker)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, encoding=REQUIRED_ENCODING, parse_dates=["Date"])
        violations = price_sanity_violations(df)
        if violations:
            report[ticker] = violations
            print(
                f"WARNING: real-data price sanity check failed for {ticker}: {'; '.join(violations)}",
                file=sys.stderr,
            )
    return report


def run_ingest(refresh: bool = False) -> IngestManifest:
    manifest = IngestManifest(generated_at=datetime.now(timezone.utc).isoformat())
    all_tickers = UNIVERSE + [BENCHMARK_TICKER]
    for ticker in all_tickers:
        result = ingest_ticker(ticker, refresh=refresh)
        manifest.tickers[ticker] = {
            "source": result.source,
            "rows": result.rows,
            "first_date": result.first_date,
            "last_date": result.last_date,
            "sha256": result.sha256,
        }
        print(f"ingested {ticker}: source={result.source} rows={result.rows}")
    manifest.cross_check = cross_check_stooq_vs_yahoo(CROSS_CHECK_TICKERS)
    _enforce_cross_check(manifest)

    sanity_report = check_real_data_sanity(all_tickers)
    for ticker, violations in sanity_report.items():
        if ticker in manifest.tickers:
            manifest.tickers[ticker]["price_sanity_violations"] = violations

    _write_manifest(manifest)
    return manifest


def _write_manifest(manifest: IngestManifest) -> None:
    payload = {
        "generated_at": manifest.generated_at,
        "tickers": manifest.tickers,
        "cross_check": manifest.cross_check,
    }
    atomic_write_json(MANIFEST_PATH, payload, encoding=REQUIRED_ENCODING, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest OHLCV data from Stooq/yfinance.")
    parser.add_argument("--refresh", action="store_true", help="re-download even if cached")
    args = parser.parse_args()
    run_ingest(refresh=args.refresh)


if __name__ == "__main__":
    main()
