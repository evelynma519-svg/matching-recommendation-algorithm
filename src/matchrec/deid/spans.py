"""Span helpers: known-name index, token boundaries, interval tests. All offsets are source offsets."""
from __future__ import annotations

import re

from .lexicons import MONTHS, SYMBOL_TOKENS
from .text import norm, plain

HAN = re.compile(r"[㐀-鿿豈-﫿]")
SYMBOL_PATTERN = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(t) for t in SYMBOL_TOKENS) + r")(?!\w|[+#])", re.I)
TOKEN = re.compile(r"[^\W_]+(?:[.\-+#/][^\W_]+|[+#])*", re.UNICODE)


def overlaps(spans, a: int, b: int) -> bool:
    return any(x < b and y > a for x, y in spans)


def containing(spans, a: int, b: int) -> bool:
    return any(x <= a and y >= b for x, y in spans)


def whole_token(text: str, a: int, b: int) -> tuple[int, int]:
    """Grow a span to the full technical token around it, so `Node` in `Node.js` is judged as `Node.js`."""
    for m in SYMBOL_PATTERN.finditer(text):
        if m.start() < b and m.end() > a:
            return m.span()
    while a > 0 and re.match(r"[A-Za-z0-9_+#.]", text[a - 1]):
        a -= 1
    while b < len(text) and re.match(r"[A-Za-z0-9_+#.]", text[b]):
        b += 1
    while b > a and text[b - 1] == ".":
        b -= 1
    return a, b


def _name_boundary(text: str, a: int, b: int) -> bool:
    """A name match must be a whole word, and never part of a punctuation-linked technical token."""
    if HAN.search(text[a:b]):
        return b - a >= 2
    if a and (text[a - 1].isalnum() or text[a - 1] == "_"):
        return False
    if b < len(text) and (text[b].isalnum() or text[b] == "_" or text[b] in "+#"):
        return False
    if b + 1 < len(text) and text[b] == "." and text[b + 1].isalnum():
        return False
    if a and text[a - 1] in ".+#" and (a == 1 or text[a - 2].isalnum() or text[a - 1] in "+#"):
        return False
    return True


class NameIndex:
    """Exact, case-insensitive matching of a set of full names. No substring matching of names."""

    def __init__(self, names):
        self.names = frozenset(n for n in (norm(plain(x)) for x in names if x) if n)
        alts = sorted(self.names, key=len, reverse=True)
        self.pattern = (re.compile("|".join(re.escape(n).replace(r"\ ", r"[ \-·]") for n in alts), re.I)
                        if alts else None)

    def spans(self, text: str, protected=()) -> list[tuple[int, int]]:
        if not self.pattern:
            return []
        out = []
        for m in self.pattern.finditer(text):
            a, b = m.span()
            if not _name_boundary(text, a, b):
                continue
            if HAN.search(text[a:b]) and containing(protected, a, b):
                continue                      # a CJK name inside a protected organization name
            out.append((a, b))
        return out

    def search(self, text: str) -> bool:
        return bool(self.spans(text))


def aliases(names) -> set[str]:
    """A person's full names plus each name part of three or more letters (not month names)."""
    out = {plain(n) for n in names if n}
    out |= {w for n in names if n for w in re.findall(r"[^\W\d_]+", plain(n))
            if len(w) >= 3 and w.casefold() not in MONTHS}
    return out


def phrase_pattern(terms, min_len: int = 1) -> re.Pattern | None:
    """Whole-word alternation for allow-list phrases (organization names, skills)."""
    terms = sorted({norm(t) for t in terms if t and len(norm(t)) >= min_len}, key=len, reverse=True)
    if not terms:
        return None
    alt = "|".join(re.escape(t).replace(r"\ ", r"[ \t]+") for t in terms)
    return re.compile(r"(?<![\w])(?:" + alt + r")(?![\w])", re.I)
