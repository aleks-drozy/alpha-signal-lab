# Methodology (Pre-Registered)

Status: FROZEN as of 2026-08-03. This document is written before any model is
trained or any result is seen. It is the proof-of-pre-registration for this
project and MUST NOT be edited based on results. If a mistake is found after
results exist, it is logged in `docs/img` or `REPORT.md` as a deviation, not
silently corrected here.

## Question

Does a LightGBM regressor on 12 standard price/volume features predict 5-day
cross-sectional excess returns better than naive baselines (12-1 momentum,
equal-weight buy-hold), under leakage-proof, purged walk-forward evaluation?

## Universe

30 US large/mega-cap liquid names, chosen for continuous liquid trading history
since 2010 (survivorship bias acknowledged, see README limitations):

AAPL, MSFT, AMZN, GOOGL, JPM, JNJ, XOM, PG, KO, WMT, META, NVDA, V, MA, UNH,
HD, DIS, BAC, PFE, CSCO, INTC, VZ, ADBE, CRM, NFLX, PEP, T, CVX, ABT, MRK, ORCL

Plus `^SPX` (SPX index) as benchmark/context only. SPX is never part of the
trading universe, the cross-sectional ranking, or the label computation.

## Data span

- Ingestion starts 2009-01-01 (warm-up buffer so the longest rolling feature,
  the 252-trading-day momentum window, is fully populated before the
  evaluation period begins).
- Evaluation span: 2016-01 through the latest complete calendar month
  available at run time.

## Target / label definition

5-trading-day forward return per ticker, minus the equal-weight mean of that
same 5-day forward return across the full universe on the same date (t):

```
label(ticker, t) = ret_fwd_5d(ticker, t) - mean_over_universe(ret_fwd_5d(*, t))
```

This is a cross-sectional excess return, not an absolute return. Modeled as a
continuous regression target (LightGBM regressor, MSE objective), not a
classification problem. Rows lacking a full forward 5-day window (end of
series) are dropped, never filled or imputed.

## Features (12, frozen)

All features are computed using only information available at the close of
day t (point-in-time), then cross-sectionally rank-transformed to [0, 1]
across the universe for each date before being handed to the model.

| Feature | Definition | Economic rationale |
|---|---|---|
| `ret_5d` | 5-day trailing return ending at t | short-horizon momentum/continuation |
| `ret_21d` | 21-day trailing return ending at t | 1-month momentum |
| `ret_63d` | 63-day trailing return ending at t | quarterly momentum |
| `ret_252d_skip21` | return from t-252 to t-21 (12-1 momentum) | classic momentum factor, skips the most recent month to avoid short-term reversal contamination |
| `ret_1d` | 1-day trailing return ending at t | short-term reversal / microstructure noise |
| `vol_21d` | realized volatility (stdev of daily returns) over trailing 21 days | risk regime, low-vol anomaly proxy |
| `vol_63d` | realized volatility over trailing 63 days | slower-moving risk regime |
| `vol_ratio_21_63` | `vol_21d / vol_63d` | volatility regime shift (short vs long vol) |
| `dollar_vol_21d_z` | 21-day mean dollar volume, z-scored against its own trailing 63-day history | liquidity/attention shock, own-history normalized so it is comparable across large-cap names of different baseline turnover |
| `vol_surge_5_63` | 5-day mean share volume / 63-day mean share volume | short-term volume surge, proxy for news/attention shock |
| `rsi_14` | 14-day relative strength index | overbought/oversold technical pressure |
| `dist_from_sma200` | `(close - SMA_200) / vol_63d` | trend distance normalized by volatility, mean-reversion/trend-strength signal |

Rank-transform: for each date t, each feature is rank-transformed to [0, 1]
across the cross-section of tickers with valid data on that date. This keeps
the model invariant to absolute price/volume scale and focuses it purely on
relative cross-sectional positioning, consistent with the cross-sectional
excess-return target.

## Cross-validation

`PurgedWalkForward`, expanding window, implemented in `src/alpha_lab/cv.py`.

- Horizon: H = 5 trading days (matches the label window).
- Purge: 5 trading days (= H) removed from the end of the training window,
  immediately before every test block, so no training label overlaps the
  test block's forward-return window.
- Embargo: 1% of total sample length (~39 trading days on a ~3900 trading-day
  sample), applied after the purge gap, before the test block starts.
- First training window: 2010-01 through 2015-12 (6 years minimum).
- Test blocks: 6-month walk-forward blocks, 2016-01 through the last complete
  6-month block available in the data. The training window expands to
  include each prior test block once it has been scored (walk-forward,
  expanding, not rolling).
- Every fold enforces: `max(train label end date) < test_start`, gap between
  train and test `>= purge + embargo`, zero index intersection between train
  and test, and test blocks are disjoint and time-ordered. These are asserted
  directly in `tests/test_leakage_audit.py`.

## Model

LightGBM regressor, fixed hyperparameters, no tuning of any kind:

```
num_leaves = 31
max_depth = 5
learning_rate = 0.05
n_estimators = 400
min_child_samples = 50
feature_fraction = 0.8
bagging_fraction = 0.8
bagging_freq = 1
seed = 42
```

One model is fit per fold on that fold's expanding training window, then used
to predict the corresponding held-out test block only.

## Baselines

1. **12-1 momentum rank alone**: rank tickers by `ret_252d_skip21` each date,
   no model.
2. **Equal-weight buy-and-hold**: the universe's equal-weight return over the
   same window, no ranking/selection.
3. **Leaky-twin naive-shuffled-kfold model** (`src/alpha_lab/leaky_twin.py`):
   identical features, target, and LightGBM hyperparameters, but standard
   random 5-fold CV (no purge, no embargo, no time ordering). Reported as a
   floor/ceiling reference showing how much of a naive backtest's apparent
   edge is pure leakage, not as a competing "real" model.

## Metrics

All metrics computed on OOS concatenated predictions only (never on training
predictions):

- Daily cross-sectional Spearman rank IC: mean, standard deviation,
  Newey-West t-statistic (lag = 5, matching the label horizon), and hit rate
  (fraction of days with positive IC).
- Top-3-minus-bottom-3 decile spread: annualized return and t-stat.
- Long-short equity curve: 5-day rebalance, equal-weight top-3 long /
  bottom-3 short, reported both gross and net of 10 bps per side per trade.
  Sharpe ratio, max drawdown, turnover.
- Same metric suite computed for both baselines and for the leaky-twin
  control, for direct side-by-side comparison.

## Pre-registered success criterion

The model is judged to have found a real, non-trivial signal if and only if,
on the honest (purged walk-forward) OOS predictions:

```
mean(daily rank IC) > 0.02   AND   Newey-West t-stat(daily rank IC) > 2.0
```

Both conditions must hold simultaneously. If either fails, the result is
reported as **NULL**, stated plainly in `REPORT.md`, not reframed as a
near-success or hedged with alternative metrics chosen after the fact.

## Leakage audit

Eight automated checks in `tests/test_leakage_audit.py`, each producing a
pass/fail record written to `docs/badges.json`:

1. Shuffled-label destruction test.
2. Split-boundary assertions (purge/embargo/ordering/disjointness).
3. Point-in-time feature recomputation spot-check (20 random rows).
4. Shift-sensitivity control (extra one-day lag should not materially help).
5. Label-window purity (labels requiring future data must be NaN, not filled).
6. No-full-series-statistics check (nothing fit on train+test combined).
7. Adjusted-price sanity checks (no negative prices, no long zero-volume
   runs, no unflagged >40% daily moves, calendar monotonicity).
8. Determinism check (same seed, same OOS predictions, hash-equal).

## Explainability

SHAP (`TreeExplainer`) on the final-fold (most recent) model only, computed
on that fold's OOS test rows only, never on training rows. Global beeswarm
and mean|SHAP| bar chart, 3 local waterfall plots on hand-picked dates spread
across the OOS period, and a per-fold mean|SHAP| feature-rank stability
heatmap. Feature-rank churn across folds is reported honestly if present, not
hidden.

## Known limitations (stated up front)

- Survivorship bias: the 30-name universe is hand-picked as "liquid since
  2010," which is itself an outcome we can observe today.
- Single asset class (US large-cap equities only); no generalization claim
  beyond this universe.
- Hyperparameters are fixed a priori, not tuned; this avoids an obvious
  leakage vector but also does not represent a best-effort model.
- Flat 10 bps per side transaction cost model; real costs vary by name,
  size, and market regime.
- 30 names is a small cross-section for daily rank IC and decile-spread
  statistics; results should be read with that sample size in mind.

## Data sources

Stooq (`https://stooq.com/q/d/l/?s=TICKER.us&i=d`) as primary source, with a
`yfinance` batch-download fallback for any ticker that fails or if Stooq
quota is hit. A 3-ticker Stooq-vs-Yahoo cross-check (close prices agree
within 1% on overlapping dates) is run and logged to `data/manifest.json` on
every ingest.

## Disclaimer

This is research into a methodology and audit harness, not a trading system
or investment recommendation. Nothing in this repository constitutes
financial advice.
