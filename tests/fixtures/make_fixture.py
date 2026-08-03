"""Generates deterministic, synthetic-but-realistic OHLCV fixtures for tests.

Not downloaded data: a fixed-seed random walk per ticker so tests never hit
the network. Regenerate with: python tests/fixtures/make_fixture.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FIXTURES_DIR = Path(__file__).resolve().parent
N_DAYS = 756
START_DATE = "2018-01-02"

TICKERS = {
    "FIXA": {"seed": 101, "start_price": 100.0, "drift": 0.0004, "vol": 0.015, "base_volume": 2_000_000},
    "FIXB": {"seed": 202, "start_price": 50.0, "drift": 0.0001, "vol": 0.020, "base_volume": 800_000},
    "FIXC": {"seed": 303, "start_price": 200.0, "drift": -0.0002, "vol": 0.012, "base_volume": 1_500_000},
}


def _business_days(start: str, n: int) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


def generate_ticker_ohlcv(seed: int, start_price: float, drift: float, vol: float, base_volume: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = _business_days(START_DATE, N_DAYS)

    daily_ret = rng.normal(loc=drift, scale=vol, size=N_DAYS)
    close = start_price * np.cumprod(1.0 + daily_ret)

    intraday_range = np.abs(rng.normal(loc=0.006, scale=0.003, size=N_DAYS))
    open_gap = rng.normal(loc=0.0, scale=0.003, size=N_DAYS)

    prev_close = np.roll(close, 1)
    prev_close[0] = start_price
    open_ = prev_close * (1.0 + open_gap)
    high = np.maximum(open_, close) * (1.0 + intraday_range / 2.0)
    low = np.minimum(open_, close) * (1.0 - intraday_range / 2.0)

    volume_noise = rng.lognormal(mean=0.0, sigma=0.35, size=N_DAYS)
    volume = (base_volume * volume_noise).astype(np.int64)

    return pd.DataFrame(
        {
            "Date": dates,
            "Open": np.round(open_, 4),
            "High": np.round(high, 4),
            "Low": np.round(low, 4),
            "Close": np.round(close, 4),
            "Volume": volume,
        }
    )


def main() -> None:
    for ticker, params in TICKERS.items():
        df = generate_ticker_ohlcv(**params)
        out_path = FIXTURES_DIR / f"{ticker}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
