"""Resume chunking: one vector per experience entry, project, or section instead of one per resume.

A whole resume squeezed into one vector dilutes specific skills, and truncation silently drops the
rest. Here a resume becomes short chunks:

- each experience or project entry, with its title / organization / date lines kept as context
- skills, summary, education, publications: one chunk each (split if very long)
- header and reference sections are skipped (after de-identification they carry no content)

Experience chunks carry start/end dates and a recency rank (0 = most recent), used for recency
weighting and for estimating years of experience. Input is the model-visible resume text produced by
the firewall, i.e. already de-identified and cut at the as-of date.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from .asof import END, RANGE, parse_token

SECTION = re.compile(r"^## (\S+)\n", re.M)
ENTRY_KINDS = {"experience", "projects"}
SKIP_KINDS = {"references"}
_END_TOKEN = re.compile(rf"(?:-|–|—|~|to|至|until)\s*({END})", re.I)
_PRESENT = re.compile(r"(?i)present|current|now|today|至今|现在|今")
_PLACEHOLDER = re.compile(r"\[[A-Z_]{2,}\]")


@dataclass
class Chunk:
    text: str
    kind: str
    start: tuple[int, int] | None = None      # (year, month)
    end: tuple[int, int] | None = None
    rank: int | None = None                   # recency rank among experience entries


def _ym(tok: str | None, as_of: date | None) -> tuple[int, int] | None:
    if not tok:
        return None
    if _PRESENT.fullmatch(tok.strip()):
        return (as_of.year, as_of.month) if as_of else None
    p = parse_token(tok)
    return (p[0], p[1] or 6) if p else None


def _entries(body: str, as_of: date | None) -> list[Chunk]:
    lines = body.splitlines()
    starts = []
    for i, ln in enumerate(lines):
        if RANGE.search(ln):
            s = i - 1 if i > 0 and lines[i - 1].strip() and len(lines[i - 1]) < 120 and not RANGE.search(lines[i - 1]) else i
            starts.append((s, i))
    if not starts:
        return [Chunk(body.strip(), "experience")] if body.strip() else []
    out = []
    for k, (s, di) in enumerate(starts):
        e = starts[k + 1][0] if k + 1 < len(starts) else len(lines)
        m, me = RANGE.search(lines[di]), _END_TOKEN.search(lines[di])
        out.append(Chunk("\n".join(lines[s:e]).strip(), "experience",
                         _ym(m.group(1), as_of), _ym(me.group(1) if me else None, as_of)))
    return out


def _split_long(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts, cur = [], ""
    for piece in re.split(r"(?<=[\n,;、，；])", text):
        if cur and len(cur) + len(piece) > max_chars:
            parts.append(cur.strip())
            cur = ""
        cur += piece
    return parts + ([cur.strip()] if cur.strip() else [])


def resume_chunks(text: str, as_of: date | None = None, max_chars: int = 1200, min_letters: int = 20) -> list[Chunk]:
    pieces = SECTION.split(text)
    sections = [("other", pieces[0])] + list(zip(pieces[1::2], pieces[2::2]))
    chunks: list[Chunk] = []
    for kind, body in sections:
        if kind in SKIP_KINDS or not body.strip():
            continue
        if kind in ENTRY_KINDS:
            chunks += _entries(body, as_of)
        else:
            chunks += [Chunk(t, kind) for t in _split_long(body.strip(), max_chars)]
    chunks = [c for c in chunks if sum(ch.isalpha() for ch in _PLACEHOLDER.sub("", c.text)) >= min_letters]
    dated = sorted((c for c in chunks if c.kind == "experience"), key=lambda c: c.end or (0, 0), reverse=True)
    for r, c in enumerate(dated):
        c.rank = r
    return chunks


def experience_years(chunks: list[Chunk]) -> float | None:
    """Years covered by dated experience entries, overlaps merged."""
    spans = sorted((c.start[0] * 12 + c.start[1], c.end[0] * 12 + c.end[1])
                   for c in chunks if c.kind == "experience" and c.start and c.end and c.end >= c.start)
    if not spans:
        return None
    total, (a, b) = 0, spans[0]
    for s, e in spans[1:]:
        if s > b:
            total += b - a
            a, b = s, e
        else:
            b = max(b, e)
    return round((total + b - a) / 12, 1)
