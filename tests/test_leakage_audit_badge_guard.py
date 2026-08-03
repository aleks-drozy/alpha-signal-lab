"""Regression test for the badges.json partial-overwrite bug: running only a
subset of tests\\test_leakage_audit.py (e.g. `pytest -k shuffled`) must not
silently thin docs/badges.json down to whatever subset ran, since the
dashboard treats badges.json as a full audit receipt. Isolated from the real
audit-module import path so this never contaminates the actual badges.json
that the real leakage-audit run writes.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "leakage_audit_module_under_test", Path(__file__).resolve().parent / "test_leakage_audit.py"
)
audit_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit_mod)


@pytest.fixture()
def isolated_badges(tmp_path, monkeypatch):
    badges_path = tmp_path / "badges.json"
    monkeypatch.setattr(audit_mod, "BADGES_PATH", badges_path)
    # Isolate the module-level results accumulator too, so this test's fake
    # entries never leak into (or get overwritten by) the real audit run's
    # bookkeeping in the same process.
    saved = list(audit_mod._RESULTS)
    audit_mod._RESULTS.clear()
    yield badges_path
    audit_mod._RESULTS.clear()
    audit_mod._RESULTS.extend(saved)


def test_partial_run_does_not_write_badges_json(isolated_badges):
    audit_mod._RESULTS.append({"test": "shuffled_label_destruction", "passed": True, "description": "d"})

    audit_mod._maybe_write_badges()

    assert not isolated_badges.exists()


def test_partial_run_does_not_clobber_existing_full_badges_json(isolated_badges):
    previous_full_run = [{"test": name, "passed": True, "description": "d"} for name in sorted(audit_mod.EXPECTED_CHECKS)]
    isolated_badges.parent.mkdir(parents=True, exist_ok=True)
    with open(isolated_badges, "w", encoding="utf-8") as f:
        json.dump(previous_full_run, f)

    audit_mod._RESULTS.append({"test": "determinism", "passed": True, "description": "d"})
    audit_mod._maybe_write_badges()

    with open(isolated_badges, encoding="utf-8") as f:
        on_disk = json.load(f)
    assert on_disk == previous_full_run


def test_full_run_writes_badges_json(isolated_badges):
    for name in sorted(audit_mod.EXPECTED_CHECKS):
        audit_mod._RESULTS.append({"test": name, "passed": True, "description": "d"})

    audit_mod._maybe_write_badges()

    assert isolated_badges.exists()
    with open(isolated_badges, encoding="utf-8") as f:
        on_disk = json.load(f)
    assert {r["test"] for r in on_disk} == audit_mod.EXPECTED_CHECKS
