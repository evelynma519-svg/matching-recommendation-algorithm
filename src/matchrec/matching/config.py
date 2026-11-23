"""Where builds and caches live. Nothing points at a personal path; set MATCHREC_DATA_DIR or pass --out."""
from __future__ import annotations

import os
from pathlib import Path


def data_dir() -> Path:
    return Path(os.environ.get("MATCHREC_DATA_DIR", "data")).expanduser()


def cache_dir() -> Path:
    return data_dir() / "cache"
