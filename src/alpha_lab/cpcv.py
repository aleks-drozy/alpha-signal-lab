"""Post-hoc addendum: Combinatorial Purged Cross-Validation (CPCV) and a
simplified single-config PBO (probability of backtest overfitting).

The primary spec evaluates one fixed walk-forward path. This asks: how much
does the honest OOS rank IC vary across other, equally valid train/test
splits drawn from the same purged/embargoed six-month blocks? A modest
sample of block-pair combinations is refit with the SAME frozen
hyperparameters (no tuning, no search), giving a distribution of OOS rank IC
instead of one point estimate.

Because a CPCV test block need not sit at the trailing edge of the training
window (unlike the primary walk-forward path), purge + embargo are applied
symmetrically on both sides of each held-out block, not just before it.

PBO here is a simplified, single-config diagnostic, not the full
multi-strategy Bailey et al. CSCV/PBO framework (that framework ranks a
*candidate set* of competing strategies per split; this project has exactly
one frozen config, so there is nothing to rank against). It is reported
honestly as a proxy: the fraction of CPCV paths where in-sample rank IC is
above the cross-path in-sample median while out-of-sample rank IC is below
the cross-path out-of-sample median, log-odds transformed.

Diagnostic only, computed after the primary pre-registered verdict; see the
addendum in specs/methodology.md. Run as: python -m alpha_lab.cpcv
"""

from __future__ import annotations

import itertools
import json
import os
import random
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from alpha_lab.config import CV, DATA_PROCESSED_DIR, FEATURE_COLUMNS, LGBM, REQUIRED_ENCODING
from alpha_lab.evaluate import daily_rank_ic, ic_summary
from alpha_lab.features import run_features
from alpha_lab.util import atomic_write_json

CPCV_PATH = DATA_PROCESSED_DIR / "cpcv.json"
CPCV_SEED = 42
CPCV_N_PATHS = 20
CPCV_N_TEST_BLOCKS_PER_PATH = 2


def six_month_blocks(unique_dates: pd.DatetimeIndex, config=CV) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Same block partition the primary walk-forward path's test blocks are
    drawn from (test_start onward, test_block_months wide), so CPCV paths are
    combinatorial rearrangements of the same ~21 blocks, not a different
    partition."""
    test_start_cfg = pd.Timestamp(config.test_start)
    last_date = unique_dates[-1]
    block_starts = pd.date_range(start=test_start_cfg, end=last_date, freq=f"{config.test_block_months}MS")

    blocks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for block_start in block_starts:
        block_end_excl = block_start + pd.DateOffset(months=config.test_block_months)
        block_end = block_end_excl - pd.Timedelta(days=1)
        if block_end > last_date:
            break
        mask = (unique_dates >= block_start) & (unique_dates <= block_end)
        block_dates = unique_dates[mask]
        if len(block_dates) == 0:
            continue
        blocks.append((block_dates[0], block_dates[-1]))
    return blocks


def train_mask_excluding_blocks(
    unique_dates: pd.DatetimeIndex,
    test_blocks: list[tuple[pd.Timestamp, pd.Timestamp]],
    purge_days: int,
    embargo_days: int,
    train_start: pd.Timestamp,
) -> np.ndarray:
    """Excludes each held-out block plus purge_days + embargo_days on BOTH
    sides (a CPCV test block can be interior to the training span, unlike
    the primary trailing-edge walk-forward path, so a one-sided purge is not
    enough to prevent leakage from either direction)."""
    n = len(unique_dates)
    values = unique_dates.values
    exclude = np.zeros(n, dtype=bool)
    for b_start, b_end in test_blocks:
        idx_start = int(np.searchsorted(values, b_start.to_datetime64()))
        idx_end = int(np.searchsorted(values, b_end.to_datetime64(), side="right")) - 1
        lo = max(idx_start - purge_days - embargo_days, 0)
        hi = min(idx_end + purge_days + embargo_days, n - 1)
        exclude[lo : hi + 1] = True
    return (unique_dates >= train_start) & (~exclude)


def _path_ic(panel: pd.DataFrame, dates: pd.DatetimeIndex, model: lgb.LGBMRegressor) -> dict:
    subset = panel[panel["date"].isin(dates)]
    preds = model.predict(subset[FEATURE_COLUMNS])
    scored = subset[["date", "label"]].copy()
    scored["y_pred"] = preds
    ic = daily_rank_ic(scored, "y_pred")
    return ic_summary(ic)


def run_cpcv_paths(
    panel: pd.DataFrame,
    n_paths: int = CPCV_N_PATHS,
    n_test_blocks: int = CPCV_N_TEST_BLOCKS_PER_PATH,
    seed: int = CPCV_SEED,
) -> tuple[list[dict], int, int]:
    unique_dates = pd.DatetimeIndex(sorted(pd.unique(panel["date"])))
    n_dates = len(unique_dates)
    embargo_days = int(round(CV.embargo_fraction * n_dates))
    train_start = pd.Timestamp(CV.train_start)

    blocks = six_month_blocks(unique_dates, CV)
    all_combos = list(itertools.combinations(range(len(blocks)), n_test_blocks))
    n_combinations_total = len(all_combos)

    rng = random.Random(seed)
    sampled_combos = (
        all_combos if n_combinations_total <= n_paths else rng.sample(all_combos, n_paths)
    )

    paths = []
    for path_id, combo in enumerate(sampled_combos):
        test_blocks = [blocks[i] for i in combo]
        train_mask = train_mask_excluding_blocks(unique_dates, test_blocks, CV.purge_days, embargo_days, train_start)
        train_dates = unique_dates[train_mask]
        test_dates = pd.DatetimeIndex(
            sorted(
                d
                for d in unique_dates
                if any(b_start <= d <= b_end for b_start, b_end in test_blocks)
            )
        )
        if len(train_dates) == 0 or len(test_dates) == 0:
            continue

        train_df = panel[panel["date"].isin(train_dates)]
        model = lgb.LGBMRegressor(**LGBM.as_params())
        model.fit(train_df[FEATURE_COLUMNS], train_df["label"])

        oos_ic = _path_ic(panel, test_dates, model)
        is_ic = _path_ic(panel, train_dates, model)

        paths.append(
            {
                "path_id": path_id,
                "test_block_indices": list(combo),
                "test_block_ranges": [
                    [str(b_start.date()), str(b_end.date())] for b_start, b_end in test_blocks
                ],
                "n_train_days": int(len(train_dates)),
                "n_test_days": int(len(test_dates)),
                "oos_rank_ic": oos_ic,
                "is_rank_ic": is_ic,
            }
        )
    return paths, n_combinations_total, len(blocks)


def distribution_stats(values: list[float]) -> dict:
    arr = np.array([v for v in values if v == v])
    if len(arr) == 0:
        return {"mean": None, "std": None, "n": 0, "ci_95_normal": [None, None], "ci_95_percentile": [None, None]}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    se = std / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
    ci_normal = [mean - 1.96 * se, mean + 1.96 * se]
    ci_pct = [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))] if len(arr) > 1 else [mean, mean]
    return {
        "mean": mean,
        "std": std,
        "n": int(len(arr)),
        "ci_95_normal": ci_normal,
        "ci_95_percentile": ci_pct,
    }


def compute_pbo(is_ic: list[float], oos_ic: list[float]) -> dict:
    is_arr = np.array(is_ic, dtype=float)
    oos_arr = np.array(oos_ic, dtype=float)
    valid = ~(np.isnan(is_arr) | np.isnan(oos_arr))
    is_arr, oos_arr = is_arr[valid], oos_arr[valid]
    if len(is_arr) == 0:
        return {"n_paths": 0, "pbo": None, "logit_pbo": None, "note": "no valid paths"}

    is_median = float(np.median(is_arr))
    oos_median = float(np.median(oos_arr))
    overfit_flags = (is_arr > is_median) & (oos_arr < oos_median)
    pbo = float(np.mean(overfit_flags))
    eps = 1e-6
    pbo_clamped = min(max(pbo, eps), 1 - eps)
    logit_pbo = float(np.log(pbo_clamped / (1 - pbo_clamped)))

    return {
        "n_paths": int(len(is_arr)),
        "is_median_ic": is_median,
        "oos_median_ic": oos_median,
        "pbo": pbo,
        "logit_pbo": logit_pbo,
        "note": (
            "Simplified single-config logit-PBO: fraction of CPCV paths where "
            "in-sample rank IC is above the cross-path in-sample median while "
            "out-of-sample rank IC is below the cross-path out-of-sample "
            "median, log-odds transformed. This is a diagnostic proxy for the "
            "one frozen config in this project, not the full multi-strategy "
            "Bailey et al. CSCV/PBO framework (which requires a searched "
            "candidate set of competing strategies to rank; this project's "
            "hyperparameters are fixed a priori, never tuned or searched)."
        ),
    }


def run_cpcv(force: bool = False) -> dict:
    if not force and os.path.exists(CPCV_PATH):
        with open(CPCV_PATH, encoding=REQUIRED_ENCODING) as f:
            return json.load(f)

    start = time.time()
    panel = run_features(force=False)
    paths, n_combinations_total, n_blocks = run_cpcv_paths(panel)
    elapsed = time.time() - start

    oos_means = [p["oos_rank_ic"]["mean"] for p in paths]
    is_means = [p["is_rank_ic"]["mean"] for p in paths]

    result = {
        "description": (
            "Post-hoc addendum: Combinatorial Purged Cross-Validation. "
            "Multiple train/test path combinations drawn from the same "
            "purged/embargoed six-month blocks as the primary walk-forward "
            "path, LightGBM refit per path with the SAME frozen "
            "hyperparameters (no tuning), rank IC computed per path. "
            "Diagnostic only, does not alter the primary pre-registered "
            "verdict."
        ),
        "n_blocks_available": n_blocks,
        "n_test_blocks_per_path": CPCV_N_TEST_BLOCKS_PER_PATH,
        "n_combinations_total": n_combinations_total,
        "n_paths_sampled": len(paths),
        "seed": CPCV_SEED,
        "purge_days": CV.purge_days,
        "lgbm_params": LGBM.as_params(),
        "paths": paths,
        "oos_ic_distribution": distribution_stats(oos_means),
        "is_ic_distribution": distribution_stats(is_means),
        "pbo": compute_pbo(is_means, oos_means),
        "runtime_seconds": round(elapsed, 1),
    }
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(CPCV_PATH, result, encoding=REQUIRED_ENCODING, indent=2)
    print(f"wrote CPCV results to {CPCV_PATH}")
    dist = result["oos_ic_distribution"]
    if dist["mean"] is not None:
        print(
            f"CPCV OOS rank IC: mean={dist['mean']:.4f} std={dist['std']:.4f} "
            f"95% CI (normal)=[{dist['ci_95_normal'][0]:.4f}, {dist['ci_95_normal'][1]:.4f}] "
            f"over {len(paths)} paths in {elapsed:.1f}s"
        )
    return result


def main() -> None:
    run_cpcv(force=True)


if __name__ == "__main__":
    main()
