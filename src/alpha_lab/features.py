"""Point-in-time features and labels, cross-sectionally rank-transformed.

Every feature at row (ticker, t) uses only data with date <= t. Labels use
data through t+H and rows without a full forward window are dropped, never
filled. Run as: python -m alpha_lab.features
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from alpha_lab.config import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    DOLLAR_VOL_WINDOW,
    DOLLAR_VOL_ZSCORE_WINDOW,
    FEATURE_COLUMNS,
    LABEL_HORIZON_DAYS,
    MOMENTUM_LOOKBACK_DAYS,
    MOMENTUM_SKIP_DAYS,
    REQUIRED_ENCODING,
    RSI_WINDOW,
    SMA_WINDOW,
    UNIVERSE,
    VOL_WINDOW_LONG,
    VOL_WINDOW_SHORT,
    VOLUME_SURGE_LONG,
    VOLUME_SURGE_SHORT,
)

FEATURES_PATH = DATA_PROCESSED_DIR / "features.parquet"


def _raw_path(ticker: str) -> str:
    safe_name = ticker.replace("^", "idx_")
    return str(DATA_RAW_DIR / f"{safe_name}.csv")


def load_raw(ticker: str) -> pd.DataFrame:
    path = _raw_path(ticker)
    df = pd.read_csv(path, encoding=REQUIRED_ENCODING, parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.where(avg_loss != 0.0, 100.0)


def compute_raw_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    volume = df["Volume"]
    daily_ret = close.pct_change(1)

    out = pd.DataFrame({"Date": df["Date"]})
    out["ret_1d"] = daily_ret
    out["ret_5d"] = close.pct_change(5)
    out["ret_21d"] = close.pct_change(21)
    out["ret_63d"] = close.pct_change(63)
    out["ret_252d_skip21"] = close.shift(MOMENTUM_SKIP_DAYS) / close.shift(
        MOMENTUM_LOOKBACK_DAYS
    ) - 1.0

    out["vol_21d"] = daily_ret.rolling(VOL_WINDOW_SHORT).std()
    out["vol_63d"] = daily_ret.rolling(VOL_WINDOW_LONG).std()
    out["vol_ratio_21_63"] = out["vol_21d"] / out["vol_63d"]

    dollar_vol = close * volume
    dollar_vol_21d = dollar_vol.rolling(DOLLAR_VOL_WINDOW).mean()
    roll_mean_63 = dollar_vol_21d.rolling(DOLLAR_VOL_ZSCORE_WINDOW).mean()
    roll_std_63 = dollar_vol_21d.rolling(DOLLAR_VOL_ZSCORE_WINDOW).std()
    out["dollar_vol_21d_z"] = (dollar_vol_21d - roll_mean_63) / roll_std_63

    vol_5 = volume.rolling(VOLUME_SURGE_SHORT).mean()
    vol_63 = volume.rolling(VOLUME_SURGE_LONG).mean()
    out["vol_surge_5_63"] = vol_5 / vol_63

    out["rsi_14"] = _rsi(close, RSI_WINDOW)

    sma200 = close.rolling(SMA_WINDOW).mean()
    out["dist_from_sma200"] = (close - sma200) / out["vol_63d"]

    out["Close"] = close.values
    return out


def compute_forward_return(df: pd.DataFrame, horizon: int) -> pd.Series:
    close = df["Close"]
    n = len(close)
    fwd = pd.Series(np.nan, index=close.index)
    valid_upto = n - horizon
    fwd.iloc[:valid_upto] = close.shift(-horizon).iloc[:valid_upto] / close.iloc[:valid_upto] - 1.0
    return fwd


def build_ticker_panel(ticker: str) -> pd.DataFrame:
    raw = load_raw(ticker)
    feats = compute_raw_features(raw)
    feats["fwd_ret_5d"] = compute_forward_return(raw, LABEL_HORIZON_DAYS)
    feats["ticker"] = ticker
    return feats


def cross_sectional_rank_transform(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    for col in FEATURE_COLUMNS:
        panel[col] = panel.groupby("Date")[col].rank(pct=True, method="average")
    return panel


def add_excess_label(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    universe_mean = panel.groupby("Date")["fwd_ret_5d"].transform("mean")
    panel["label"] = panel["fwd_ret_5d"] - universe_mean
    return panel


def build_feature_matrix(universe: list[str] | None = None) -> pd.DataFrame:
    universe = universe or UNIVERSE
    panels = [build_ticker_panel(t) for t in universe]
    panel = pd.concat(panels, ignore_index=True)

    panel = panel.dropna(subset=FEATURE_COLUMNS + ["fwd_ret_5d"]).reset_index(drop=True)
    panel = add_excess_label(panel)
    panel = cross_sectional_rank_transform(panel)

    panel = panel.rename(columns={"Date": "date"})
    ordered_cols = ["date", "ticker"] + FEATURE_COLUMNS + ["fwd_ret_5d", "label"]
    return panel[ordered_cols].sort_values(["date", "ticker"]).reset_index(drop=True)


def run_features(force: bool = False) -> pd.DataFrame:
    if not force and os.path.exists(FEATURES_PATH):
        return pd.read_parquet(FEATURES_PATH)

    panel = build_feature_matrix()
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(FEATURES_PATH, index=False)
    print(f"wrote {len(panel)} rows to {FEATURES_PATH}")
    return panel


def main() -> None:
    run_features(force=True)


if __name__ == "__main__":
    main()
