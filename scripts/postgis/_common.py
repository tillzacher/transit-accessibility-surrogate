"""Shared helpers for the PostGIS ingest scripts."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import numbers
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SQL_DIR = Path(__file__).resolve().parent / "sql"


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    return logging.getLogger(name)


def file_signature(path: Path) -> str:
    stat = path.stat()
    payload = f"{path.name}|{stat.st_size}|{int(stat.st_mtime)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def jsonable(value):
    """Convert a value to something JSONB accepts (non-finite floats become null)."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        value_f = float(value)
        return value_f if math.isfinite(value_f) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def id_payload(value) -> dict:
    """Typed id record that modules/postgis_tile_source.py restores via `_source_*` keys."""
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return {"value": int(value), "type": "int"}
    if isinstance(value, (np.floating, float)):
        return {"value": float(value), "type": "float"}
    return {"value": str(value), "type": "str"}


def read_run_meta(run_dir: Path) -> tuple[dict, Path]:
    """Return the run metadata and the file it came from (`run_manifest.json` preferred)."""
    for name in ("run_manifest.json", "meta.json"):
        path = run_dir / name
        if path.exists():
            payload = json.loads(path.read_text())
            return payload.get("meta", payload), path
    raise FileNotFoundError(f"Missing run_manifest.json or meta.json in {run_dir}")


def run_column_names(run_dir: Path) -> list[str]:
    import pyarrow.parquet as pq

    path = run_dir / "cells.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing cells.parquet in {run_dir}")
    return list(pq.ParquetFile(path).schema.names)
