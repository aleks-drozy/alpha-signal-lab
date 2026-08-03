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

## Addendum: post-hoc supplementary analyses (added 2026-08-03, after the primary verdict)

Everything above this line is the frozen pre-registered spec, computed and
verdict-sealed before this addendum was written. The primary NULL verdict
(honest OOS mean rank IC ~0.0100, Newey-West t-stat ~1.50, below the
0.02 / 2.0 bar) stands exactly as originally computed and is **not** altered,
re-run, or cherry-picked by anything below.

The analyses in this addendum are diagnostic/robustness checks computed on
top of the already-generated OOS predictions (`data/processed/oos_predictions.parquet`)
after the primary result was seen. They exist to answer follow-up questions
a careful reader would ask next ("is this just momentum in disguise?", "does
it work in some regime?", "is one fixed walk-forward path a fluke of that
particular split?"), not to search for a way to turn the NULL into a PASS.
If any of them had come back positive, that would not overturn the primary
verdict; it would only be reported as a secondary finding worth a proper
pre-registered follow-up study. In fact all of them reinforce the primary
NULL.

**Erratum (noted here, primary section above left untouched):** the primary
section's prose above says "30 US large/mega-cap liquid names"; the actual
frozen `UNIVERSE` list in `src/alpha_lab/config.py` has always had 31 names
and every result in this project (folds, IC, the addendum analyses) was
computed on the real 31-name list, not on 30. This is a descriptive miscount
in the spec's prose written before the pipeline ran, not a change to the
universe itself, and not a results-driven edit: the actual ticker list is
unchanged and was never edited. `REPORT.md` and the dashboard now read the
count from `config.py` directly (`len(UNIVERSE)`) rather than a hardcoded
number, so this class of error cannot silently recur.

1. **Feature neutralization vs momentum** (`src/alpha_lab/neutralize.py`,
   `data/processed/neutralization.json`). Cross-sectionally residualizes the
   model's OOS predictions against the 12-1 momentum feature per date (simple
   per-date OLS, prediction regressed on momentum, residual taken), then
   recomputes rank IC on the residual. Tests whether the model contributes
   anything orthogonal to plain momentum.

2. **Regime / slice breakdown** (`src/alpha_lab/regime.py`,
   `data/processed/regime_breakdown.json`). Median-splits the OOS period into
   high- and low-volatility regimes using the trailing 21-day realized
   volatility of the equal-weight universe mean daily return (computed from
   raw, pre rank-transform returns, not the rank-transformed feature), and
   reports rank IC separately per regime. Also reports per-ticker
   time-series rank IC across all 30 names, honestly, including if it is
   indistinguishable from noise.

3. **Rolling per-fold SHAP importance over time**
   (`src/alpha_lab/explain.py`, `data/processed/shap_importance_over_time.json`).
   The per-fold mean |SHAP| values that were already computed for the
   stability-ranking heatmap are reshaped into a fold-indexed time series (one
   row per fold, one column per feature, with each fold's test date range) so
   a rolling-importance chart can be drawn on the dashboard.

4. **Reproducibility manifest** (`data/provenance.json`, written by
   `run_all.py` at the end of a full run). Python version, pinned versions of
   the key packages actually installed (lightgbm, pandas, numpy,
   scikit-learn, shap), the frozen random seed from `config.py`, and a single
   "run fingerprint" sha256 over the concatenated raw-input file hashes from
   `data/manifest.json`.

5. **CPCV + simplified single-config PBO** (`src/alpha_lab/cpcv.py`,
   `data/processed/cpcv.json`). Combinatorial Purged Cross-Validation:
   instead of the one fixed walk-forward path, multiple train/test path
   combinations are drawn from the same ~21 six-month test blocks used by the
   primary spec (choose 2 blocks as test per combination, purge + embargo
   applied symmetrically around each held-out block since a CPCV test block
   is not always at the trailing edge of the training window), LightGBM
   refit per combination using the *same frozen hyperparameters* (no
   tuning), and rank IC computed per path. The result is a distribution of
   OOS rank IC (mean, std, 95% CI by both normal approximation and
   percentile) instead of one point estimate, plus a **simplified,
   single-config "logit-PBO"**: since there is only one frozen candidate
   config here (not a searched candidate set), this is not the full
   multi-strategy Bailey et al. CSCV/PBO framework. It is a diagnostic proxy
   -- the fraction of CPCV paths where in-sample rank IC is above the
   cross-path in-sample median while out-of-sample rank IC is below the
   cross-path out-of-sample median, log-odds transformed -- reported
   honestly as such, not as a full PBO estimate.

If the CPCV 95% CI includes zero, that is read as a stronger form of the same
NULL finding (the single walk-forward path was not an unlucky low draw from
a real positive distribution), not as a different or contradicting result.
Likewise, a still-null neutralized IC is reported plainly as "still null after
removing momentum overlap," not reframed.
