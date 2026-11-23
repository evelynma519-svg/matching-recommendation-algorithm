"""Source adapters and the on-disk snapshot format.

A snapshot directory holds four files, all UTF-8:

    manifest.json         Snapshot
    candidates.jsonl      one Candidate per line
    jobs.jsonl            one Job per line
    applications.jsonl    one Application per line

`SourceAdapter` is the only seam between a real system or data source and this matching system. 
To use real data, write an adapter that reads the system's export and yields the neutral records; 
keep that adapter and its field mapping outside this repository.
"""
from __future__ import annotations

import dataclasses
import json
from enum import Enum
from pathlib import Path
from typing import Iterable, Iterator, Protocol

from .funnel import FunnelSpec, get_funnel
from .model import (Application, Candidate, Compensation, Identity, Job, JobNote, ResumeVersion,
                    Section, Snapshot, StageEvent, StageStatus)

FILES = {"candidates": "candidates.jsonl", "jobs": "jobs.jsonl", "applications": "applications.jsonl"}


class SourceAdapter(Protocol):
    def snapshot(self) -> Snapshot: ...
    def candidates(self) -> Iterable[Candidate]: ...
    def jobs(self) -> Iterable[Job]: ...
    def applications(self) -> Iterable[Application]: ...


# ---------------------------------------------------------------- (de)serialization

def to_dict(obj) -> dict:
    def conv(x):
        if isinstance(x, Enum):
            return x.value
        if isinstance(x, list):
            return [conv(i) for i in x]
        if isinstance(x, dict):
            return {k: conv(v) for k, v in x.items()}
        return x
    return conv(dataclasses.asdict(obj))


def candidate_from(d: dict) -> Candidate:
    return Candidate(
        candidate_id=str(d["candidate_id"]),
        resumes=[ResumeVersion(r["resume_id"], r.get("captured_at"), r.get("text", ""),
                               [Section(**s) for s in r.get("sections", [])], r.get("language"))
                 for r in d.get("resumes", [])],
        skills=list(d.get("skills", [])),
        identity=Identity(**d.get("identity", {})),
    )


def job_from(d: dict) -> Job:
    d = dict(d)
    d["job_id"], d["org_id"] = str(d["job_id"]), str(d["org_id"])
    d["compensation"] = Compensation(**d.get("compensation", {}))
    d["notes"] = [JobNote(**n) for n in d.get("notes", [])]
    return Job(**d)


def application_from(d: dict) -> Application:
    d = dict(d)
    d["stages"] = [StageEvent(e["stage"], StageStatus(e["status"]), e.get("at")) for e in d.get("stages", [])]
    return Application(**d)


def _read_jsonl(p: Path) -> Iterator[dict]:
    with p.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _write_jsonl(p: Path, rows: Iterable) -> int:
    n = 0
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(to_dict(r), ensure_ascii=False) + "\n")
            n += 1
    return n


class JsonlSnapshot:
    """Reference adapter: reads a snapshot directory in the neutral format."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def snapshot(self) -> Snapshot:
        return Snapshot(**json.loads((self.root / "manifest.json").read_text(encoding="utf-8")))

    def candidates(self) -> Iterator[Candidate]:
        return map(candidate_from, _read_jsonl(self.root / FILES["candidates"]))

    def jobs(self) -> Iterator[Job]:
        return map(job_from, _read_jsonl(self.root / FILES["jobs"]))

    def applications(self) -> Iterator[Application]:
        return map(application_from, _read_jsonl(self.root / FILES["applications"]))


def write_snapshot(root: Path, src: SourceAdapter) -> dict:
    """Materialize any adapter into the neutral on-disk format. Returns per-file record counts."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(to_dict(src.snapshot()), indent=1), encoding="utf-8")
    return {k: _write_jsonl(root / f, getattr(src, k)()) for k, f in FILES.items()}


# ---------------------------------------------------------------- validation

def validate(src: SourceAdapter, funnel: FunnelSpec | None = None) -> list[str]:
    """Referential and ordering checks against the snapshot's funnel. Empty list means valid."""
    funnel = funnel or get_funnel(src.snapshot().funnel)
    problems: list[str] = []
    cands = {c.candidate_id: c for c in src.candidates()}
    jobs = {j.job_id: j for j in src.jobs()}
    seen: set[str] = set()
    for a in src.applications():
        where = f"application {a.application_id}"
        if a.application_id in seen:
            problems.append(f"{where}: duplicate id")
        seen.add(a.application_id)
        if a.candidate_id not in cands:
            problems.append(f"{where}: unknown candidate")
        if a.job_id not in jobs:
            problems.append(f"{where}: unknown job")
        if a.resume_id and a.candidate_id in cands and \
                a.resume_id not in {r.resume_id for r in cands[a.candidate_id].resumes}:
            problems.append(f"{where}: resume_id not among the candidate's resumes")
        unknown = [e.stage for e in a.stages if e.stage not in funnel.stages]
        if unknown:
            problems.append(f"{where}: stages not in funnel '{funnel.name}'")
            continue
        ranks = [funnel.rank(e.stage) for e in a.stages]
        if ranks != sorted(ranks):
            problems.append(f"{where}: stages out of order")
        if any(e.status is StageStatus.DROPPED for e in a.stages[:-1]):
            problems.append(f"{where}: a dropped stage is followed by later stages")
    return problems
