"""Post-hoc addendum: cross-sectional neutralization of the ML model's OOS
predictions against 12-1 momentum. Regresses predictions on the momentum
feature per date (simple per-date OLS) and recomputes rank IC on the
residual, to see whether the model contributes anything orthogonal to plain
momentum or is just noisy momentum. Diagnostic only, computed after the
primary pre-registered verdict; see the addendum in specs/methodology.md.
Run as: python -m alpha_lab.neutralize
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from alpha_lab.config import DATA_PROCESSED_DIR, REQUIRED_ENCODING
from alpha_lab.evaluate import daily_rank_ic, ic_summary
from alpha_lab.features import run_features
from alpha_lab.model import run_model
from alpha_lab.util import atomic_write_json

NEUTRALIZATION_PATH = DATA_PROCESSED_DIR / "neutralization.json"
MOMENTUM_COL = "ret_252d_skip21"


def per_date_residual(group: pd.DataFrame, pred_col: str, momentum_col: str) -> pd.Series:
    """Simple per-date OLS: regress pred_col on momentum_col, return the
    residual. Needs >= 3 points and non-degenerate momentum spread to fit a
    line; otherwise the residual is undefined (NaN, not zero, since a zero
    would silently claim "fully explained by momentum")."""
    x = group[momentum_col].to_numpy(dtype=float)
    y = group[pred_col].to_numpy(dtype=float)
    if len(group) < 3 or np.std(x) == 0.0:
        return pd.Series(np.nan, index=group.index)
    slope, intercept = np.polyfit(x, y, deg=1)
    fitted = intercept + slope * x
    return pd.Series(y - fitted, index=group.index)


def neutralize_predictions(
    df: pd.DataFrame, pred_col: str = "y_pred", momentum_col: str = MOMENTUM_COL
) -> pd.DataFrame:
    df = df.copy()
    df["y_pred_neutralized"] = df.groupby("date", group_keys=False).apply(
        lambda g: per_date_residual(g, pred_col, momentum_col)
    )
    return df


def run_neutralize(force: bool = False) -> dict:
    if not force and os.path.exists(NEUTRALIZATION_PATH):
        with open(NEUTRALIZATION_PATH, encoding=REQUIRED_ENCODING) as f:
            return json.load(f)

    oos = run_model(force=False)
    features = run_features(force=False)
    joined = oos.merge(features[["date", "ticker", MOMENTUM_COL]], on=["date", "ticker"], how="left")

    neutralized = neutralize_predictions(joined, pred_col="y_pred", momentum_col=MOMENTUM_COL)

    raw_ic = daily_rank_ic(neutralized, "y_pred")
    neut_input = neutralized.dropna(subset=["y_pred_neutralized"])
    neut_ic = daily_rank_ic(neut_input, "y_pred_neutralized")

    result = {
        "description": (
            "Post-hoc addendum: rank IC of raw OOS predictions vs predictions "
            "cross-sectionally residualized against 12-1 momentum per date. "
            "Diagnostic only, does not alter the primary pre-registered verdict."
        ),
        "momentum_feature": MOMENTUM_COL,
        "raw": ic_summary(raw_ic),
        "neutralized": ic_summary(neut_ic),
        "n_oos_rows": int(len(joined)),
        "n_rows_dropped_degenerate_date": int(len(joined) - len(neut_input)),
    }
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(NEUTRALIZATION_PATH, result, encoding=REQUIRED_ENCODING, indent=2)
    print(f"wrote neutralization results to {NEUTRALIZATION_PATH}")
    print(
        f"raw mean IC: {result['raw']['mean']:.4f}  "
        f"neutralized mean IC: {result['neutralized']['mean']:.4f}"
    )
    return result


def main() -> None:
    run_neutralize(force=True)


if __name__ == "__main__":
    main()
