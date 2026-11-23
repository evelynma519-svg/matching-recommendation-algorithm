"""The model-side data door.

Models read `<build>/inputs/` through `load_inputs` and, for training, dev labels through
`train_labels` (which always holds out a fold). Nothing here opens `sealed/` or `eval/`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Outcome fields and de-identification inputs. Any of these in a model input is a leak.
FORBIDDEN_KEYS = {
    "grade", "ended", "end_reason", "end_note", "stages", "status", "in_flight", "excluded", "label",
    "matured", "qrels", "job_ids", "closed_at", "identity", "related_people", "emails", "phones",
}


def assert_clean(obj, path: str = "") -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in FORBIDDEN_KEYS:
                raise AssertionError(f"outcome field '{k}' in model input at {path or '<root>'}")
            assert_clean(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for x in obj:
            assert_clean(x, path + "[]")


def read_jsonl(p: Path) -> list[dict]:
    with Path(p).open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


@dataclass
class Inputs:
    queries: list[dict]
    resumes: dict[str, dict]       # query_id -> as-of resume
    jobs: dict[str, dict]          # job_id -> job view


def load_inputs(build_dir: Path) -> Inputs:
    d = Path(build_dir) / "inputs"
    q, r, j = (read_jsonl(d / f) for f in ("queries.jsonl", "resumes.jsonl", "jobs.jsonl"))
    for rows in (q, r, j):
        for x in rows:
            assert_clean(x)
    return Inputs(q, {x["query_id"]: x for x in r}, {x["job_id"]: x for x in j})


def train_labels(build_dir: Path, holdout_fold: int) -> list[dict]:
    """Dev labels excluding every candidate in `holdout_fold`. Refuses on test builds."""
    p = Path(build_dir) / "labels" / "pairs.jsonl"
    if not p.exists():
        raise PermissionError("no dev labels in this build (test builds expose no labels to models)")
    return [r for r in read_jsonl(p) if r["fold"] != holdout_fold and not r["excluded"]]
