"""Shared low-level helpers: atomic writes and file fingerprinting.

Atomic writes prevent a killed process from leaving a truncated file that a
later run would trust as valid cache; fingerprinting lets a downstream stage
detect when it was built from a now-stale upstream artifact.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd


def _tmp_path_for(path: Path) -> str:
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    return tmp_path


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _tmp_path_for(path)
    try:
        with open(tmp_path, "wb") as f:
            f.write(data)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path: str | Path, obj, encoding: str = "utf-8", **json_kwargs) -> None:
    atomic_write_text(path, json.dumps(obj, **json_kwargs), encoding=encoding)


def atomic_to_csv(df: pd.DataFrame, path: str | Path, encoding: str = "utf-8", index: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _tmp_path_for(path)
    try:
        df.to_csv(tmp_path, index=index, encoding=encoding)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def atomic_to_parquet(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _tmp_path_for(path)
    try:
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def sha256_of_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def combined_hash(hashes: dict) -> str:
    payload = json.dumps(hashes, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def warn_if_hash_mismatch(recorded_hash: str | None, actual_path: str | Path, label: str) -> None:
    """Loud stderr WARNING (never a hard failure): a downstream artifact's
    recorded upstream hash not matching the upstream file's current contents
    means a stale partial-pipeline run, which must be visible, not silently
    accepted as valid."""
    if recorded_hash is None or not os.path.exists(actual_path):
        return
    actual_hash = sha256_of_file(actual_path)
    if actual_hash != recorded_hash:
        print(
            f"WARNING: {label} fingerprint mismatch: recorded {recorded_hash[:12]}... "
            f"but {actual_path} is currently {actual_hash[:12]}...; this artifact looks stale, "
            "rerun the pipeline from this stage onward.",
            file=sys.stderr,
        )
