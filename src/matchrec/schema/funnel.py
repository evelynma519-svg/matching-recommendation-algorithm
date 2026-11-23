"""Configurable hiring funnels and graded relevance.

A funnel names the ordered stages a process can reach and maps them onto relevance levels:

    L1  put forward          the candidate was judged a plausible fit for the job
    L2  employer review      the employer looked at the candidate
    L3  interview            the employer invested time in the candidate
    L4  offer / hire         the employer wanted the candidate

The deeper a process got, the stronger the evidence of a match. Which stages earn which level
is configuration, not code, so the same pipeline serves a staffing-agency funnel, a direct-hire
funnel, or a source that only records "applied" and "hired".

A funnel that cannot observe some levels (for example `minimal`) still works: those levels are
reported as unobservable, and evaluation falls back to the levels the source can see. Explicit
feedback and stage-transition priors (see docs/funnel.md) are the intended way to fill the gap.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .model import Application, EndReason, StageStatus

REACHED = frozenset({StageStatus.IN_PROGRESS, StageStatus.PASSED})
LEVEL_NAMES = {1: "L1_put_forward", 2: "L2_employer_review", 3: "L3_interview", 4: "L4_offer_or_hire"}


@dataclass(frozen=True)
class Level:
    grade: int
    stages: tuple[str, ...]              # reaching any of these stages earns the grade
    dropped_counts: bool = False         # True: ending *at* the stage still counts (the interview happened)
    end_reasons: tuple[str, ...] = ()    # an end reason that proves this level was reached


@dataclass(frozen=True)
class Graded:
    grade: int
    excluded: bool                       # bookkeeping record, kept out of labels
    ended: bool                          # process ended before a hire; NOT a negative label
    in_flight: bool                      # still moving at snapshot time
    end_reason: str | None


@dataclass(frozen=True)
class FunnelSpec:
    name: str
    stages: tuple[str, ...]
    levels: tuple[Level, ...]
    excluded_reasons: frozenset[str] = frozenset({EndReason.DUPLICATE, EndReason.ENTRY_ERROR})
    description: str = ""
    terminal_stage: str | None = None    # reaching it (in progress or passed) means the process succeeded

    def __post_init__(self):
        known = set(self.stages)
        for lv in self.levels:
            if not set(lv.stages) <= known:
                raise ValueError(f"funnel {self.name}: level {lv.grade} names unknown stages")
        if self.terminal_stage and self.terminal_stage not in known:
            raise ValueError(f"funnel {self.name}: unknown terminal stage")

    def rank(self, stage: str) -> int:
        return self.stages.index(stage)

    @property
    def observable_grades(self) -> frozenset[int]:
        return frozenset(lv.grade for lv in self.levels)

    def grade(self, app: Application) -> Graded:
        status = {e.stage: e.status for e in app.stages}
        g = 0
        for lv in self.levels:
            hit = any(status.get(s) in REACHED or (lv.dropped_counts and status.get(s) is StageStatus.DROPPED)
                      for s in lv.stages)
            if hit or (app.end_reason in lv.end_reasons):
                g = max(g, lv.grade)
        succeeded = self.terminal_stage is not None and status.get(self.terminal_stage) in REACHED
        ended = any(st is StageStatus.DROPPED for st in status.values()) and not succeeded
        last = app.stages[-1].status if app.stages else None
        in_flight = not ended and not succeeded and last is StageStatus.IN_PROGRESS
        return Graded(g, app.end_reason in self.excluded_reasons, ended, in_flight, app.end_reason)

    # ---- custom funnels from JSON, e.g. {"name": ..., "stages": [...], "levels": [{"grade": 1, "stages": [...]}]}
    @classmethod
    def from_dict(cls, d: dict) -> "FunnelSpec":
        return cls(d["name"], tuple(d["stages"]),
                   tuple(Level(lv["grade"], tuple(lv["stages"]), lv.get("dropped_counts", False),
                               tuple(lv.get("end_reasons", ()))) for lv in d["levels"]),
                   frozenset(d.get("excluded_reasons", (EndReason.DUPLICATE, EndReason.ENTRY_ERROR))),
                   d.get("description", ""), d.get("terminal_stage"))

    @classmethod
    def from_json(cls, path: Path) -> "FunnelSpec":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


STAFFING = FunnelSpec(
    "staffing",
    ("submitted", "client_review", "interview", "offer", "offer_accepted", "hired"),
    (Level(1, ("submitted",), dropped_counts=True),
     Level(2, ("client_review",), end_reasons=(EndReason.EMPLOYER_DECLINED,)),
     Level(3, ("interview",), dropped_counts=True),
     Level(4, ("offer", "offer_accepted", "hired"))),
    description="Agency recruiter submits a candidate, then forwards the profile to the hiring client.",
    terminal_stage="hired",
)

DIRECT = FunnelSpec(
    "direct",
    ("applied", "screen", "interview", "offer", "hired"),
    (Level(1, ("applied",), dropped_counts=True),
     Level(2, ("screen",), end_reasons=(EndReason.EMPLOYER_DECLINED,)),
     Level(3, ("interview",), dropped_counts=True),
     Level(4, ("offer", "hired"))),
    description="Candidate applies directly; the employer screens, interviews and makes offers.",
    terminal_stage="hired",
)

MINIMAL = FunnelSpec(
    "minimal",
    ("applied", "hired"),
    (Level(1, ("applied",), dropped_counts=True),
     Level(4, ("hired",))),
    description="Only applications and hires are recorded; L2 and L3 are unobservable.",
    terminal_stage="hired",
)

FUNNELS: dict[str, FunnelSpec] = {f.name: f for f in (STAFFING, DIRECT, MINIMAL)}


def get_funnel(name_or_path: str) -> FunnelSpec:
    if name_or_path in FUNNELS:
        return FUNNELS[name_or_path]
    p = Path(name_or_path)
    if p.suffix == ".json" and p.exists():
        return FunnelSpec.from_json(p)
    raise KeyError(f"unknown funnel {name_or_path!r}; built-ins: {sorted(FUNNELS)}")
