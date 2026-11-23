"""Pair labels from a funnel: graded relevance (L1-L4), maturity, and bookkeeping exclusions.

An ended process is unknown, not negative. Fit negatives need explicit feedback (a rated gold set)
and are not produced here.
"""
from __future__ import annotations

from datetime import date

from ..schema import Application, FunnelSpec


def parse_day(s) -> date | None:
    if not s or not isinstance(s, str):
        return None
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return None


def matured(app: Application, ref_date: date, days: int = 30) -> bool:
    """True when the submission is old enough for its outcome to have had time to show."""
    d = parse_day(app.submitted_at)
    return d is not None and (ref_date - d).days >= days


def label_row(app: Application, funnel: FunnelSpec, ref_date: date, maturity_days: int, fold: int,
              org_id: str | None) -> dict:
    g = funnel.grade(app)
    return {"application_id": app.application_id, "candidate_id": app.candidate_id, "job_id": app.job_id,
            "org_id": org_id, "submitted_at": app.submitted_at, "grade": g.grade, "ended": g.ended,
            "end_reason": g.end_reason, "in_flight": g.in_flight, "excluded": g.excluded,
            "matured": matured(app, ref_date, maturity_days), "fold": fold}
