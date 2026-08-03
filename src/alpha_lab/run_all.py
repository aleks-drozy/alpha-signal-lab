"""CLI orchestrator: ingest -> features -> model -> evaluate -> leaky_twin
-> explain -> report, in order, with progress printing.

Run as: python -m alpha_lab.run_all [--refresh]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version

from alpha_lab.config import LGBM, MANIFEST_PATH, PROJECT_ROOT, REQUIRED_ENCODING
from alpha_lab.cpcv import run_cpcv
from alpha_lab.evaluate import METRICS_PATH, run_evaluate
from alpha_lab.explain import run_explain
from alpha_lab.features import FEATURES_PATH, run_features
from alpha_lab.ingest import run_ingest
from alpha_lab.leaky_twin import run_leaky_twin
from alpha_lab.model import OOS_PREDICTIONS_FINGERPRINT_PATH, OOS_PREDICTIONS_PATH, run_model
from alpha_lab.neutralize import run_neutralize
from alpha_lab.regime import run_regime
from alpha_lab.report import run_report
from alpha_lab.util import atomic_write_json, combined_hash, warn_if_hash_mismatch

STEPS = [
    "ingest",
    "features",
    "model",
    "evaluate",
    "leaky_twin",
    "explain",
    "neutralize (supplementary)",
    "regime (supplementary)",
    "cpcv (supplementary)",
    "report",
]

LOCK_PATH = PROJECT_ROOT / "data" / ".run_all.lock"
PROVENANCE_PATH = PROJECT_ROOT / "data" / "provenance.json"
PROVENANCE_PACKAGES = ["lightgbm", "pandas", "numpy", "scikit-learn", "shap"]


def _acquire_lock() -> None:
    """A concurrent run_all overwriting the same cached artifacts mid-write
    is a real race that has already caused an incident during verification.
    os.O_CREAT | os.O_EXCL is an atomic create-if-absent on both POSIX and
    Windows, so there is no check-then-create window to race."""
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(
            f"lock file {LOCK_PATH} already exists: another run_all is in progress, or a previous "
            "run crashed/was killed without cleaning up. If you are sure no run_all process is "
            f"currently running, delete {LOCK_PATH} and retry."
        ) from exc
    with os.fdopen(fd, "w", encoding=REQUIRED_ENCODING) as f:
        f.write(str(os.getpid()))


def _release_lock() -> None:
    try:
        os.remove(LOCK_PATH)
    except FileNotFoundError:
        pass


def _print_step(name: str, index: int) -> None:
    print(f"\n[{index}/{len(STEPS)}] {name}")
    print("-" * 60)


def _check_stage_fingerprints() -> None:
    """WARN-only consistency check: does each downstream artifact's recorded
    upstream hash still match the upstream file that is actually on disk
    right now? A mismatch means a stale, partial-pipeline run is being
    treated as valid; this must be visible, not silent."""
    if os.path.exists(OOS_PREDICTIONS_FINGERPRINT_PATH):
        with open(OOS_PREDICTIONS_FINGERPRINT_PATH, encoding=REQUIRED_ENCODING) as f:
            oos_fp = json.load(f)
        warn_if_hash_mismatch(oos_fp.get("features_parquet_sha256"), FEATURES_PATH, "oos_predictions.parquet -> features.parquet")

    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, encoding=REQUIRED_ENCODING) as f:
            metrics = json.load(f)
        recorded = (metrics.get("upstream_hashes") or {}).get("oos_predictions_parquet_sha256")
        warn_if_hash_mismatch(recorded, OOS_PREDICTIONS_PATH, "metrics.json -> oos_predictions.parquet")


def _package_versions() -> dict:
    versions = {}
    for pkg in PROVENANCE_PACKAGES:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = None
    return versions


def write_provenance() -> None:
    """Reproducibility manifest for the addendum: python + key package
    versions actually installed, the frozen random seed, and a single
    run-fingerprint sha256 over the concatenated raw-input file hashes
    already recorded in data/manifest.json by ingest.py."""
    input_hashes = {}
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, encoding=REQUIRED_ENCODING) as f:
            manifest = json.load(f)
        input_hashes = {t: info["sha256"] for t, info in manifest.get("tickers", {}).items()}

    provenance = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "package_versions": _package_versions(),
        "random_seed": LGBM.seed,
        "raw_input_file_count": len(input_hashes),
        "run_fingerprint_sha256": combined_hash(input_hashes) if input_hashes else None,
    }
    atomic_write_json(PROVENANCE_PATH, provenance, encoding=REQUIRED_ENCODING, indent=2)
    print(f"wrote provenance manifest to {PROVENANCE_PATH}")


def run_all(refresh: bool = False) -> None:
    _acquire_lock()
    try:
        start = time.time()

        _print_step("ingest", 1)
        run_ingest(refresh=refresh)

        _print_step("features", 2)
        run_features(force=True)

        _print_step("model", 3)
        run_model(force=True)

        _print_step("evaluate", 4)
        run_evaluate(force=True)

        _print_step("leaky_twin", 5)
        run_leaky_twin(force=True)

        _print_step("explain", 6)
        run_explain(force=True)

        _print_step("neutralize (supplementary)", 7)
        run_neutralize(force=True)

        _print_step("regime (supplementary)", 8)
        run_regime(force=True)

        _print_step("cpcv (supplementary)", 9)
        run_cpcv(force=True)

        _print_step("report", 10)
        run_report(force=True)

        _check_stage_fingerprints()
        write_provenance()

        elapsed = time.time() - start
        print(f"\nrun_all complete in {elapsed:.1f}s")
    finally:
        _release_lock()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full alpha-signal-lab pipeline end to end.")
    parser.add_argument("--refresh", action="store_true", help="re-download raw data even if cached")
    args = parser.parse_args()
    run_all(refresh=args.refresh)


if __name__ == "__main__":
    main()
