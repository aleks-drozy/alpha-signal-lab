"""Purged, embargoed, expanding-window walk-forward cross-validation.

Purge removes `purge_days` trading days immediately before a test block so no
training label window overlaps it. Embargo adds a further fixed number of
trading days (1% of total sample length) after the purge gap. The training
window always starts at `train_start` and expands fold to fold because each
fold's train_end is defined relative to that fold's own test_start.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_lab.config import CV, CVConfig


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex


class PurgedWalkForward:
    def __init__(self, config: CVConfig = CV):
        self.config = config

    def embargo_days(self, n_dates: int) -> int:
        return int(round(self.config.embargo_fraction * n_dates))

    def split(self, dates: pd.DatetimeIndex | pd.Series) -> list[Fold]:
        unique_dates = pd.DatetimeIndex(sorted(pd.unique(pd.DatetimeIndex(dates))))
        n = len(unique_dates)
        embargo = self.embargo_days(n)
        gap = self.config.purge_days + embargo

        train_start = pd.Timestamp(self.config.train_start)
        test_start_cfg = pd.Timestamp(self.config.test_start)
        last_date = unique_dates[-1]

        block_starts = pd.date_range(
            start=test_start_cfg, end=last_date, freq=f"{self.config.test_block_months}MS"
        )

        folds: list[Fold] = []
        for block_start in block_starts:
            block_end_excl = block_start + pd.DateOffset(months=self.config.test_block_months)
            block_end = block_end_excl - pd.Timedelta(days=1)
            if block_end > last_date:
                break

            test_mask = (unique_dates >= block_start) & (unique_dates <= block_end)
            test_dates = unique_dates[test_mask]
            if len(test_dates) == 0:
                continue

            idx_test_start = int(np.searchsorted(unique_dates.values, test_dates[0].to_datetime64()))
            train_end_idx = idx_test_start - gap - 1
            if train_end_idx < 0:
                continue

            train_mask = (unique_dates >= train_start) & (unique_dates <= unique_dates[train_end_idx])
            train_dates = unique_dates[train_mask]
            if len(train_dates) == 0:
                continue

            folds.append(
                Fold(
                    fold_id=len(folds),
                    train_start=train_dates[0],
                    train_end=train_dates[-1],
                    test_start=test_dates[0],
                    test_end=test_dates[-1],
                    train_dates=train_dates,
                    test_dates=test_dates,
                )
            )
        return folds


def main() -> None:
    from alpha_lab.features import run_features

    panel = run_features(force=False)
    splitter = PurgedWalkForward()
    folds = splitter.split(panel["date"])
    for fold in folds:
        gap_days = (fold.test_start - fold.train_end).days
        print(
            f"fold {fold.fold_id}: train {fold.train_start.date()}..{fold.train_end.date()} "
            f"({len(fold.train_dates)} days) -> test {fold.test_start.date()}..{fold.test_end.date()} "
            f"({len(fold.test_dates)} days), calendar gap {gap_days}d"
        )


if __name__ == "__main__":
    main()
