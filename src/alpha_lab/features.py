"""Point-in-time features and labels, cross-sectionally rank-transformed.

Every feature at row (ticker, t) uses only data with date <= t. Labels use
data through t+H and rows without a full forward window are dropped, never
filled. Run as: python -m alpha_lab.features
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from alpha_lab.config import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    DOLLAR_VOL_WINDOW,
    DOLLAR_VOL_ZSCORE_WINDOW,
    FEATURE_COLUMNS,
    LABEL_HORIZON_DAYS,
    MANIFEST_PATH,
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
from alpha_lab.util import atomic_to_parquet, atomic_write_json, combined_hash

FEATURES_PATH = DATA_PROCESSED_DIR / "features.parquet"
FEATURES_FINGERPRINT_PATH = DATA_PROCESSED_DIR / "features.fingerprint.json"


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
    rsi = rsi.where(avg_loss != 0.0, 100.0)
    # Flat/stale stretch (avg_gain == avg_loss == 0): neutral, not bullish.
    # NaN here so it is dropped by the same dropna policy as any other
    # incomplete feature, instead of forcing a maximally bullish 100.
    flat = (avg_gain == 0.0) & (avg_loss == 0.0)
    return rsi.where(~flat, np.nan)


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


def _drop_non_finite_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Division by near-zero volatility/volume windows can produce +/-inf,
    which survives a NaN-only dropna and would otherwise reach the
    cross-sectional rank transform and the model silently. Treat non-finite
    the same as NaN: drop the row."""
    n_before = len(panel)
    finite_mask = np.isfinite(panel[FEATURE_COLUMNS].to_numpy(dtype=float)).all(axis=1)
    panel = panel[finite_mask].reset_index(drop=True)
    n_dropped = n_before - len(panel)
    if n_dropped:
        print(f"dropped {n_dropped} rows with non-finite feature values")
    return panel


def build_feature_matrix(universe: list[str] | None = None) -> pd.DataFrame:
    universe = universe or UNIVERSE
    panels = [build_ticker_panel(t) for t in universe]
    panel = pd.concat(panels, ignore_index=True)

    panel = panel.dropna(subset=FEATURE_COLUMNS + ["fwd_ret_5d"]).reset_index(drop=True)
    panel = _drop_non_finite_features(panel)
    panel = add_excess_label(panel)
    panel = cross_sectional_rank_transform(panel)

    panel = panel.rename(columns={"Date": "date"})
    ordered_cols = ["date", "ticker"] + FEATURE_COLUMNS + ["fwd_ret_5d", "label"]
    return panel[ordered_cols].sort_values(["date", "ticker"]).reset_index(drop=True)


def _write_features_fingerprint(universe: list[str]) -> None:
    """Records the sha256 (from data/manifest.json) of every raw ticker file
    this feature matrix was built from, so model.py/evaluate.py can later
    detect a stale rebuild."""
    if not os.path.exists(MANIFEST_PATH):
        return
    with open(MANIFEST_PATH, encoding=REQUIRED_ENCODING) as f:
        raw_manifest = json.load(f)
    input_hashes = {
        t: raw_manifest["tickers"][t]["sha256"]
        for t in universe
        if t in raw_manifest.get("tickers", {})
    }
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_hashes": input_hashes,
        "combined_input_hash": combined_hash(input_hashes),
    }
    atomic_write_json(FEATURES_FINGERPRINT_PATH, payload, encoding=REQUIRED_ENCODING, indent=2)


def run_features(force: bool = False) -> pd.DataFrame:
    if not force and os.path.exists(FEATURES_PATH):
        return pd.read_parquet(FEATURES_PATH)

    universe = UNIVERSE
    panel = build_feature_matrix(universe)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    atomic_to_parquet(panel, FEATURES_PATH)
    _write_features_fingerprint(universe)
    print(f"wrote {len(panel)} rows to {FEATURES_PATH}")
    return panel


def main() -> None:
    run_features(force=True)


if __name__ == "__main__":
    main()
