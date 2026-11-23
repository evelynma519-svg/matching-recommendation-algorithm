"""As-of resume text.

A resume captured after a submission can contain jobs that started later, including the job the
candidate was placed in, which leaks the answer. Version choice, best first:

1. the resume version bound to the submission (`Application.resume_id`),
2. the latest version captured on or before the as-of date,
3. otherwise the latest version, with experience entries starting on or after as-of cut out.

The text cut runs in every case; on a correctly chosen version it removes nothing.

Heuristic: an entry starts at the line holding a date range, plus the short header line just
above it (company / title). It runs until the next entry. Month-precision starts are dropped if
>= as-of month; year-only starts only if the year is later than the as-of year.
"""
from __future__ import annotations

import re
from datetime import date

from ..schema import Candidate, ResumeVersion, Section

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_MON = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?,?"
TOK = (rf"(?:{_MON}\s*(?:19|20)\d{{2}}"
       r"|(?:19|20)\d{2}\s*(?:[./-]|年)\s*\d{1,2}\s*月?"
       r"|\d{1,2}\s*/\s*(?:19|20)\d{2}"
       r"|(?:19|20)\d{2})")
END = TOK + r"|present|current|now|today|至今|现在|今"
RANGE = re.compile(rf"({TOK})\s*(?:-|–|—|~|to|至|until)\s*(?:{END})", re.I)
CUT_SECTIONS = {"experience", "projects", "other", "summary"}


def parse_token(t: str) -> tuple[int, int | None] | None:
    """-> (year, month or None)."""
    t = t.strip().lower()
    m = re.match(rf"({_MON})\s*((?:19|20)\d{{2}})", t)
    if m:
        return int(m.group(2)), MONTHS[m.group(1)[:3]]
    m = re.match(r"((?:19|20)\d{2})\s*(?:[./-]|年)\s*(\d{1,2})", t)
    if m and 1 <= int(m.group(2)) <= 12:
        return int(m.group(1)), int(m.group(2))
    m = re.match(r"(\d{1,2})\s*/\s*((?:19|20)\d{2})", t)
    if m and 1 <= int(m.group(1)) <= 12:
        return int(m.group(2)), int(m.group(1))
    m = re.match(r"((?:19|20)\d{2})", t)
    return (int(m.group(1)), None) if m else None


def starts_after(tok: tuple[int, int | None], as_of: date) -> bool:
    y, mth = tok
    if mth is None:
        return y > as_of.year
    return (y, mth) >= (as_of.year, as_of.month)


def cut_text(text: str, as_of: date) -> tuple[str, dict]:
    lines = text.splitlines()
    date_idx = []
    for i, ln in enumerate(lines):
        m = RANGE.search(ln)
        if m and (tok := parse_token(m.group(1))):
            date_idx.append((i, starts_after(tok, as_of)))
    starts = []
    for i, future in date_idx:
        s = i
        if i > 0 and lines[i - 1].strip() and len(lines[i - 1]) < 120 and not RANGE.search(lines[i - 1]):
            s = i - 1
        starts.append((s, future))
    drop: set[int] = set()
    for k, (s, future) in enumerate(starts):
        if future:
            e = starts[k + 1][0] if k + 1 < len(starts) else len(lines)
            drop.update(range(s, max(e, s + 1)))
    kept = [ln for i, ln in enumerate(lines) if i not in drop]
    return "\n".join(kept), {"entries": len(starts), "dropped_entries": sum(f for _, f in starts),
                             "dropped_lines": len(drop)}


def choose_version(cand: Candidate, bound_ids: set[str], as_of: date) -> tuple[ResumeVersion | None, str]:
    """Pick the resume version a model may see for a query. Returns (version, method)."""
    if not cand.resumes:
        return None, "none"
    by_id = {r.resume_id: r for r in cand.resumes}
    bound = [by_id[i] for i in sorted(bound_ids) if i in by_id]
    if bound:
        return bound[0], "bound"
    before = [r for r in cand.resumes if r.captured_at and r.captured_at[:10] <= as_of.isoformat()]
    if before:
        return max(before, key=lambda r: r.captured_at), "captured_before"
    return max(cand.resumes, key=lambda r: r.captured_at or ""), "text_cut_only"


def cut_resume(resume: ResumeVersion, as_of: date) -> tuple[str, dict]:
    """Model-visible resume text as of `as_of`, plus cut statistics."""
    stats = {"entries": 0, "dropped_entries": 0, "dropped_lines": 0}
    parts = []
    sections = resume.sections or [Section("other", resume.text)]
    for sec in sections:
        t = sec.text or ""
        if sec.kind in CUT_SECTIONS:
            t, st = cut_text(t, as_of)
            for k in stats:
                stats[k] += st[k]
        parts.append(f"## {sec.kind}\n{t}")
    return "\n\n".join(parts), stats
