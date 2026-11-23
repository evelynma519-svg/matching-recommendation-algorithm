"""Human term decisions (layer 6).

Reviewers settle recurring ambiguous terms once, instead of re-reading them in every document.
A decision file is JSON:

    {"version": "2024-05-01", "entries": [
        {"term": "Kepler", "decision": "allow", "scope": "tech_context", "note": "GPU architecture"},
        {"term": "Grace",  "decision": "mask",  "scope": "global"}]}

decision  mask | allow | conditional (conditional behaves as allow inside its scope)
scope     global | skills_section | tech_context | et_al

When the same term has both mask and allow entries in a matching scope, mask wins. Decision files
are built from real review work and therefore stay local; the repository ships only a synthetic
example. Each run records the file's SHA-256 so a result can be traced to the exact decisions used.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .lexicons import PERSON_CONTEXT, TECH_CONTEXT
from .text import norm, window

DECISIONS = {"mask", "allow", "conditional"}
SCOPES = {"global", "skills_section", "tech_context", "et_al"}


@dataclass(frozen=True)
class TermDecision:
    term: str
    decision: str
    scope: str = "global"
    note: str = ""

    def __post_init__(self):
        if self.decision not in DECISIONS:
            raise ValueError(f"decision must be one of {sorted(DECISIONS)}")
        if self.scope not in SCOPES:
            raise ValueError(f"scope must be one of {sorted(SCOPES)}")

    def applies(self, text: str, a: int, b: int, section: str) -> bool:
        if self.scope == "global":
            return True
        if self.scope == "skills_section":
            return section == "skills"
        if self.scope == "tech_context":
            near = window(text, a, b, 65)
            return bool(TECH_CONTEXT.search(near)) and not PERSON_CONTEXT.search(near)
        return bool(re.search(r"(?i)et\s*$", text[max(0, a - 4):a]))         # et_al


class DecisionBook:
    def __init__(self, entries=(), version: str = "none", sha256: str | None = None):
        self.version, self.sha256 = version, sha256
        self.by_term: dict[str, list[TermDecision]] = defaultdict(list)
        for e in entries:
            self.by_term[norm(e.term)].append(e)

    @classmethod
    def load(cls, path: Path) -> "DecisionBook":
        raw = Path(path).read_bytes()
        doc = json.loads(raw)
        entries = [TermDecision(e["term"], e["decision"], e.get("scope", "global"), e.get("note", ""))
                   for e in doc.get("entries", [])]
        return cls(entries, doc.get("version", "unversioned"), hashlib.sha256(raw).hexdigest())

    def terms(self) -> set[str]:
        return set(self.by_term)

    def lookup(self, term: str, text: str, a: int, b: int, section: str) -> str | None:
        """'mask', 'allow', or None when no decision applies here."""
        hits = [d for d in self.by_term.get(norm(term), []) if d.applies(text, a, b, section)]
        if any(d.decision == "mask" for d in hits):
            return "mask"
        return "allow" if hits else None
