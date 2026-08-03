"""Post-hoc addendum: regime and per-ticker slice breakdown of the OOS
predictions already computed by the primary pipeline. Splits the OOS period
into high/low realized-volatility regimes (median split on the trailing
21-day realized volatility of the equal-weight universe mean daily return,
computed from raw pre rank-transform returns) and reports rank IC separately
per regime, plus per-ticker time-series rank IC reported honestly even if it
looks like noise. Diagnostic only, computed after the primary pre-registered
verdict; see the addendum in specs/methodology.md.
Run as: python -m alpha_lab.regime
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from alpha_lab.config import DATA_PROCESSED_DIR, REQUIRED_ENCODING, UNIVERSE
from alpha_lab.evaluate import daily_rank_ic, ic_summary
from alpha_lab.features import build_ticker_panel
from alpha_lab.model import run_model
from alpha_lab.util import atomic_write_json

REGIME_BREAKDOWN_PATH = DATA_PROCESSED_DIR / "regime_breakdown.json"
VOL_WINDOW = 21


def trailing_universe_vol(universe: list[str] | None = None, window: int = VOL_WINDOW) -> pd.Series:
    """Trailing realized volatility of the equal-weight universe mean daily
    return. Deliberately recomputed from raw (pre rank-transform) daily
    returns rather than reading features.parquet's vol_21d column, since
    that column is a cross-sectional percentile rank, not a volatility
    level, and cannot be used as a market-wide regime signal."""
    universe = universe or UNIVERSE
    panels = []
    for t in universe:
        raw = build_ticker_panel(t)[["Date", "ret_1d"]].rename(columns={"Date": "date"})
        panels.append(raw)
    combined = pd.concat(panels, ignore_index=True)
    universe_daily_ret = combined.groupby("date")["ret_1d"].mean().sort_index()
    return universe_daily_ret.rolling(window).std().dropna()


def assign_regime(oos: pd.DataFrame, vol_series: pd.Series) -> pd.DataFrame:
    oos = oos.copy()
    dates = pd.DatetimeIndex(sorted(oos["date"].unique()))
    vol_on_oos_dates = vol_series.reindex(dates).ffill()
    median_vol = vol_on_oos_dates.median()
    regime_by_date = (vol_on_oos_dates > median_vol).map({True: "high_vol", False: "low_vol"})
    mapping = dict(zip(dates, regime_by_date))
    oos["regime"] = oos["date"].map(mapping)
    return oos


def regime_ic_breakdown(oos: pd.DataFrame) -> dict:
    out = {}
    for regime, group in oos.groupby("regime"):
        if pd.isna(regime):
            continue
        ic = daily_rank_ic(group, "y_pred")
        summary = ic_summary(ic)
        summary["n_rows"] = int(len(group))
        out[str(regime)] = summary
    return out


def per_ticker_ic(oos: pd.DataFrame) -> list[dict]:
    rows = []
    for ticker, group in oos.groupby("ticker"):
        group = group.dropna(subset=["y_pred", "label"])
        if len(group) < 5 or group["y_pred"].nunique() < 2:
            rows.append({"ticker": ticker, "spearman_ic": None, "n_obs": int(len(group))})
            continue
        corr, _ = spearmanr(group["y_pred"], group["label"])
        rows.append(
            {
                "ticker": ticker,
                "spearman_ic": float(corr) if corr == corr else None,
                "n_obs": int(len(group)),
            }
        )
    rows.sort(key=lambda r: (r["spearman_ic"] is None, -(r["spearman_ic"] or 0.0)))
    return rows


def run_regime(force: bool = False) -> dict:
    if not force and os.path.exists(REGIME_BREAKDOWN_PATH):
        with open(REGIME_BREAKDOWN_PATH, encoding=REQUIRED_ENCODING) as f:
            return json.load(f)

    oos = run_model(force=False)
    vol_series = trailing_universe_vol()
    oos_with_regime = assign_regime(oos, vol_series)

    regime_ic = regime_ic_breakdown(oos_with_regime)
    ticker_ic = per_ticker_ic(oos)

    spreads = [r["spearman_ic"] for r in ticker_ic if r["spearman_ic"] is not None]
    ticker_ic_summary = {
        "mean_across_tickers": float(np.mean(spreads)) if spreads else None,
        "std_across_tickers": float(np.std(spreads, ddof=1)) if len(spreads) > 1 else None,
        "n_tickers_with_positive_ic": int(sum(1 for s in spreads if s > 0)),
        "n_tickers_total": len(ticker_ic),
    }

    result = {
        "description": (
            "Post-hoc addendum: rank IC split by realized-volatility regime "
            "(median split on trailing 21-day realized volatility of the "
            "equal-weight universe mean daily return) and per-ticker "
            "time-series Spearman IC. Diagnostic only, does not alter the "
            "primary pre-registered verdict."
        ),
        "vol_window_days": VOL_WINDOW,
        "regime_rank_ic": regime_ic,
        "per_ticker_ic": ticker_ic,
        "per_ticker_ic_summary": ticker_ic_summary,
    }
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(REGIME_BREAKDOWN_PATH, result, encoding=REQUIRED_ENCODING, indent=2)
    print(f"wrote regime breakdown to {REGIME_BREAKDOWN_PATH}")
    if "high_vol" in regime_ic and "low_vol" in regime_ic:
        print(
            f"high_vol mean IC: {regime_ic['high_vol']['mean']:.4f}  "
            f"low_vol mean IC: {regime_ic['low_vol']['mean']:.4f}"
        )
    return result


def main() -> None:
    run_regime(force=True)


if __name__ == "__main__":
    main()
