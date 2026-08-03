"""Metrics on OOS concatenated predictions only: rank IC, Newey-West t-stat,
decile spread, long-short equity curve (gross/net), turnover, and the
12-1 momentum / equal-weight buy-hold baselines. Writes metrics.json.

Run as: python -m alpha_lab.evaluate
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from alpha_lab.config import (
    DATA_PROCESSED_DIR,
    DECILE_LONG_SHORT_N,
    IC_SUCCESS_THRESHOLD,
    IC_TSTAT_SUCCESS_THRESHOLD,
    NEWEY_WEST_LAG,
    REBALANCE_DAYS,
    REQUIRED_ENCODING,
    TRANSACTION_COST_BPS_PER_SIDE,
)
from alpha_lab.features import run_features
from alpha_lab.model import OOS_PREDICTIONS_PATH, run_model
from alpha_lab.util import atomic_write_json, sha256_of_file

METRICS_PATH = DATA_PROCESSED_DIR / "metrics.json"
TRADING_DAYS_PER_YEAR = 252


def daily_rank_ic(df: pd.DataFrame, pred_col: str, label_col: str = "label") -> pd.Series:
    def _ic(group: pd.DataFrame) -> float:
        if group[pred_col].nunique() < 2 or len(group) < 3:
            return np.nan
        corr, _ = spearmanr(group[pred_col], group[label_col])
        return corr

    return df.groupby("date").apply(_ic, include_groups=False).dropna()


def newey_west_tstat(x: np.ndarray, lag: int = NEWEY_WEST_LAG) -> float:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return float("nan")
    mean_x = x.mean()
    demeaned = x - mean_x
    gamma_0 = float(np.sum(demeaned**2) / n)
    long_run_var = gamma_0
    for h in range(1, min(lag, n - 1) + 1):
        gamma_h = float(np.sum(demeaned[h:] * demeaned[:-h]) / n)
        weight = 1.0 - h / (lag + 1)
        long_run_var += 2.0 * weight * gamma_h
    se = np.sqrt(max(long_run_var, 0.0) / n)
    if se == 0.0:
        return float("nan")
    return float(mean_x / se)


def ic_summary(ic_series: pd.Series) -> dict:
    values = ic_series.to_numpy()
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
        "newey_west_tstat": newey_west_tstat(values, NEWEY_WEST_LAG),
        "hit_rate": float(np.mean(values > 0)),
        "n_days": int(len(values)),
    }


def rebalance_dates(all_dates: pd.DatetimeIndex, step: int = REBALANCE_DAYS) -> list[pd.Timestamp]:
    sorted_dates = pd.DatetimeIndex(sorted(pd.unique(all_dates)))
    return list(sorted_dates[::step])


def top_bottom_period_returns(
    df: pd.DataFrame, pred_col: str, rebalance_ds: list[pd.Timestamp], n: int = DECILE_LONG_SHORT_N
) -> pd.DataFrame:
    rows = []
    skipped = 0
    prev_longs: set[str] = set()
    prev_shorts: set[str] = set()
    for d in rebalance_ds:
        day = df[df["date"] == d]
        if len(day) < 2 * n:
            skipped += 1
            continue
        ranked = day.sort_values(pred_col, ascending=False)
        longs = ranked.head(n)
        shorts = ranked.tail(n)
        long_set = set(longs["ticker"])
        short_set = set(shorts["ticker"])

        long_ret = float(longs["fwd_ret_5d"].mean())
        short_ret = float(shorts["fwd_ret_5d"].mean())
        gross_ret = long_ret - short_ret

        long_turnover = 1.0 - (len(long_set & prev_longs) / n if prev_longs else 0.0)
        short_turnover = 1.0 - (len(short_set & prev_shorts) / n if prev_shorts else 0.0)
        turnover = (long_turnover + short_turnover) / 2.0

        # Full round-trip cost charged proportional to actual turnover, on both books.
        cost = turnover * 2.0 * (TRANSACTION_COST_BPS_PER_SIDE / 10000.0)
        net_ret = gross_ret - cost

        rows.append(
            {
                "date": d,
                "gross_ret": gross_ret,
                "net_ret": net_ret,
                "turnover": turnover,
                "long_ret": long_ret,
                "short_ret": short_ret,
            }
        )
        prev_longs, prev_shorts = long_set, short_set
    if skipped:
        print(
            f"evaluate: skipped {skipped}/{len(rebalance_ds)} rebalance dates "
            f"with fewer than {2 * n} names in the cross-section"
        )
    return pd.DataFrame(rows)


def equity_curve_stats(period_returns: pd.DataFrame, ret_col: str) -> dict:
    if period_returns.empty:
        return {"sharpe": float("nan"), "max_drawdown": float("nan"), "annualized_return": float("nan")}
    r = period_returns[ret_col].to_numpy()
    periods_per_year = TRADING_DAYS_PER_YEAR / REBALANCE_DAYS
    mean_r = r.mean()
    std_r = r.std(ddof=1) if len(r) > 1 else float("nan")
    # `x in (0.0, float("nan"))` never matches NaN (NaN != NaN), so the old
    # guard let a NaN std_r through into the division below; check explicitly.
    can_compute = not (np.isnan(std_r) or std_r == 0.0)
    sharpe = float(mean_r / std_r * np.sqrt(periods_per_year)) if can_compute else float("nan")

    equity = np.cumprod(1.0 + r)
    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1.0
    max_dd = float(drawdown.min())

    n_years = len(r) / periods_per_year
    annualized_return = float(equity[-1] ** (1.0 / n_years) - 1.0) if n_years > 0 else float("nan")

    return {
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "annualized_return": annualized_return,
        "mean_turnover": float(period_returns["turnover"].mean()) if "turnover" in period_returns else None,
    }


def decile_spread_stats(period_returns: pd.DataFrame) -> dict:
    if period_returns.empty:
        return {"annualized_spread": float("nan"), "tstat": float("nan")}
    r = period_returns["gross_ret"].to_numpy()
    periods_per_year = TRADING_DAYS_PER_YEAR / REBALANCE_DAYS
    annualized_spread = float(r.mean() * periods_per_year)
    tstat = newey_west_tstat(r, min(NEWEY_WEST_LAG, len(r) - 1)) if len(r) > 1 else float("nan")
    return {"annualized_spread": annualized_spread, "tstat": tstat}


def evaluate_strategy(df: pd.DataFrame, pred_col: str, label_col: str = "label") -> dict:
    # A NaN prediction must never silently ride sort_values to one end of the
    # ranking and land in the long or short book; drop it instead.
    n_before = len(df)
    df = df[df[pred_col].notna()].copy()
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"evaluate: dropped {n_dropped} rows with NaN {pred_col} before ranking")

    ic_series = daily_rank_ic(df, pred_col, label_col)
    rebal_ds = rebalance_dates(df["date"])
    period_returns = top_bottom_period_returns(df, pred_col, rebal_ds)
    return {
        "rank_ic": ic_summary(ic_series),
        "decile_spread": decile_spread_stats(period_returns),
        "equity_curve_gross": equity_curve_stats(period_returns, "gross_ret"),
        "equity_curve_net": equity_curve_stats(period_returns, "net_ret"),
    }


def buy_hold_baseline(df: pd.DataFrame) -> dict:
    rebal_ds = rebalance_dates(df["date"])
    rows = []
    for d in rebal_ds:
        day = df[df["date"] == d]
        if day.empty:
            continue
        rows.append({"date": d, "gross_ret": float(day["fwd_ret_5d"].mean()), "turnover": 0.0})
    period_returns = pd.DataFrame(rows)
    period_returns["net_ret"] = period_returns["gross_ret"]
    return {
        "equity_curve_gross": equity_curve_stats(period_returns, "gross_ret"),
        "equity_curve_net": equity_curve_stats(period_returns, "net_ret"),
    }


def run_evaluate(force: bool = False) -> dict:
    if not force and os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, encoding=REQUIRED_ENCODING) as f:
            return json.load(f)

    oos = run_model(force=False)
    features = run_features(force=False)

    momentum_col = "ret_252d_skip21"
    joined = oos.merge(
        features[["date", "ticker", momentum_col]], on=["date", "ticker"], how="left"
    )

    universe_sizes = joined.groupby("date").size()
    print(
        "evaluate: per-date cross-sectional universe size "
        f"min={universe_sizes.min()} max={universe_sizes.max()} mean={universe_sizes.mean():.1f}"
    )

    model_metrics = evaluate_strategy(joined, pred_col="y_pred")
    momentum_metrics = evaluate_strategy(joined, pred_col=momentum_col)
    buy_hold_metrics = buy_hold_baseline(joined)

    criterion_met = bool(
        model_metrics["rank_ic"]["mean"] > IC_SUCCESS_THRESHOLD
        and model_metrics["rank_ic"]["newey_west_tstat"] > IC_TSTAT_SUCCESS_THRESHOLD
    )

    metrics = {
        "model": model_metrics,
        "momentum_baseline": momentum_metrics,
        "buy_hold_baseline": buy_hold_metrics,
        "success_criterion": {
            "ic_threshold": IC_SUCCESS_THRESHOLD,
            "tstat_threshold": IC_TSTAT_SUCCESS_THRESHOLD,
            "met": criterion_met,
        },
        "n_oos_rows": int(len(oos)),
        "oos_date_range": [str(oos["date"].min().date()), str(oos["date"].max().date())],
    }
    if os.path.exists(OOS_PREDICTIONS_PATH):
        metrics["upstream_hashes"] = {"oos_predictions_parquet_sha256": sha256_of_file(OOS_PREDICTIONS_PATH)}

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(METRICS_PATH, metrics, encoding=REQUIRED_ENCODING, indent=2)
    print(f"wrote metrics to {METRICS_PATH}")
    print(f"success criterion met: {criterion_met}")
    return metrics


def main() -> None:
    run_evaluate(force=True)


if __name__ == "__main__":
    main()
