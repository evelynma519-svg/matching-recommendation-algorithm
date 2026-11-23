"""Model-side text views and a bilingual tokenizer.

Baseline job text = title + summary + responsibilities + requirements + required skills + location.
Excluded by default:
  - notes: unless every note is time-verified, a note may carry feedback written after as-of
  - org_name: a resume that still mentions the placed organization would leak through it

The tokenizer keeps technical tokens such as C++, C#, Node.js intact, and splits CJK runs into
character bigrams so mixed Chinese/English resumes index sensibly.
"""
from __future__ import annotations

import re

PLACEHOLDER = re.compile(r"\[[A-Z_]{2,}\]")      # de-identification placeholders such as [PERSON]
_TOKEN = re.compile(r"[a-z0-9][a-z0-9+#.\-]*[a-z0-9+#]|[a-z0-9]|[㐀-鿿]+")
STOP = set("""a an and are as at be by for from has have in into is it its of on or our that the their this
to was we were will with you your who what when where which while within years year experience work working
team teams ability strong including etc using use used new must plus""".split())


def job_text(job: dict, include_notes: bool = False) -> str:
    parts = [job.get("title"), job.get("summary"), job.get("responsibilities"), job.get("requirements")]
    if job.get("required_skills"):
        parts.append("Skills: " + ", ".join(job["required_skills"]))
    if job.get("locations"):
        parts.append("Location: " + ", ".join(job["locations"]))
    if include_notes:
        parts += [n["text"] for n in job.get("notes") or [] if n.get("time_verified")]
    return PLACEHOLDER.sub(" ", "\n".join(p for p in parts if p)).strip()


def resume_text(resume: dict) -> str:
    return PLACEHOLDER.sub(" ", resume.get("text") or "").strip()


def tokenize(s: str) -> list[str]:
    out = []
    for t in _TOKEN.findall(PLACEHOLDER.sub(" ", s).lower()):
        if "㐀" <= t[0] <= "鿿":
            out += [t] if len(t) == 1 else [t[i:i + 2] for i in range(len(t) - 1)]
            continue
        t = t.rstrip(".-")
        if not t or t in STOP or t.isdigit():
            continue
        out.append(t)
    return out
