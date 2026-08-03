"""CLI orchestrator: ingest -> features -> model -> evaluate -> leaky_twin
-> explain -> report, in order, with progress printing.

Run as: python -m alpha_lab.run_all [--refresh]
"""

from __future__ import annotations

import argparse
import time

from alpha_lab.evaluate import run_evaluate
from alpha_lab.explain import run_explain
from alpha_lab.features import run_features
from alpha_lab.ingest import run_ingest
from alpha_lab.leaky_twin import run_leaky_twin
from alpha_lab.model import run_model
from alpha_lab.report import run_report

STEPS = [
    "ingest",
    "features",
    "model",
    "evaluate",
    "leaky_twin",
    "explain",
    "report",
]


def _print_step(name: str, index: int) -> None:
    print(f"\n[{index}/{len(STEPS)}] {name}")
    print("-" * 60)


def run_all(refresh: bool = False) -> None:
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

    _print_step("report", 7)
    run_report(force=True)

    elapsed = time.time() - start
    print(f"\nrun_all complete in {elapsed:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full alpha-signal-lab pipeline end to end.")
    parser.add_argument("--refresh", action="store_true", help="re-download raw data even if cached")
    args = parser.parse_args()
    run_all(refresh=args.refresh)


if __name__ == "__main__":
    main()
