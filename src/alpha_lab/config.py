"""Single source of truth for all frozen parameters (see specs/methodology.md).

Every other module imports from here. Nothing frozen is hardcoded elsewhere.
This file is not to be edited based on results once the spec is registered;
only genuine bugs in transcribing the frozen spec are fixed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DOCS_DIR = PROJECT_ROOT / "docs"
DOCS_IMG_DIR = DOCS_DIR / "img"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.json"

UNIVERSE: list[str] = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "JPM", "JNJ", "XOM", "PG", "KO", "WMT",
    "META", "NVDA", "V", "MA", "UNH", "HD", "DIS", "BAC", "PFE", "CSCO",
    "INTC", "VZ", "ADBE", "CRM", "NFLX", "PEP", "T", "CVX", "ABT", "MRK",
    "ORCL",
]
BENCHMARK_TICKER = "^SPX"
STOOQ_BENCHMARK_SYMBOL = "^spx"

INGEST_START_DATE = "2009-01-01"
EVAL_START_DATE = "2016-01-01"

STOOQ_URL_TEMPLATE = "https://stooq.com/q/d/l/?s={symbol}.us&i=d"
STOOQ_REQUEST_DELAY_SECONDS = 1.5
CROSS_CHECK_TICKERS: list[str] = ["AAPL", "MSFT", "JPM"]
CROSS_CHECK_TOLERANCE_PCT = 0.01

LABEL_HORIZON_DAYS = 5

FEATURE_COLUMNS: list[str] = [
    "ret_5d",
    "ret_21d",
    "ret_63d",
    "ret_252d_skip21",
    "ret_1d",
    "vol_21d",
    "vol_63d",
    "vol_ratio_21_63",
    "dollar_vol_21d_z",
    "vol_surge_5_63",
    "rsi_14",
    "dist_from_sma200",
]

RSI_WINDOW = 14
SMA_WINDOW = 200
VOL_WINDOW_SHORT = 21
VOL_WINDOW_LONG = 63
DOLLAR_VOL_WINDOW = 21
DOLLAR_VOL_ZSCORE_WINDOW = 63
VOLUME_SURGE_SHORT = 5
VOLUME_SURGE_LONG = 63
MOMENTUM_SKIP_DAYS = 21
MOMENTUM_LOOKBACK_DAYS = 252


@dataclass(frozen=True)
class CVConfig:
    horizon_days: int = LABEL_HORIZON_DAYS
    purge_days: int = LABEL_HORIZON_DAYS
    embargo_fraction: float = 0.01
    train_start: str = "2010-01-01"
    train_end: str = "2015-12-31"
    test_start: str = EVAL_START_DATE
    test_block_months: int = 6


CV = CVConfig()


@dataclass(frozen=True)
class LightGBMConfig:
    objective: str = "regression"
    num_leaves: int = 31
    max_depth: int = 5
    learning_rate: float = 0.05
    n_estimators: int = 400
    min_child_samples: int = 50
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 1
    seed: int = 42

    def as_params(self) -> dict:
        return {
            "objective": self.objective,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "min_child_samples": self.min_child_samples,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "random_state": self.seed,
        }


LGBM = LightGBMConfig()

NEWEY_WEST_LAG = 5
DECILE_LONG_SHORT_N = 3
TRANSACTION_COST_BPS_PER_SIDE = 10
REBALANCE_DAYS = LABEL_HORIZON_DAYS

IC_SUCCESS_THRESHOLD = 0.02
IC_TSTAT_SUCCESS_THRESHOLD = 2.0

LEAKY_TWIN_N_FOLDS = 5
LEAKY_TWIN_SEED = 42

SHAP_N_LOCAL_EXAMPLES = 3

REQUIRED_ENCODING = "utf-8"
