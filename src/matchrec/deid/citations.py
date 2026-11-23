"""Resume sections and a conservative bibliography parser (layers 3 and 4).

Publication lists are the densest source of third-party names in technical resumes. The parser
finds citation records, then validates the *whole* author block before accepting it:

- author block validated  -> each author span is masked as PERSON; title and venue stay readable
- author block not parsable -> the whole record is removed (CITATION_REMOVED), since a partial
  author list could leak a name
"""
from __future__ import annotations

import re

from .lexicons import SECTION_HEADINGS

YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
VENUE = re.compile(r"(?i)\b(?:proc\.?|proceedings|conference|journal|trans\.?|transactions|arxiv|neurips|icml|"
                   r"cvpr|iccv|eccv|acl|emnlp|ieee|acm|pmlr|icra|iros|corl|patent)\b|期刊|会议|专利")
QUOTE = re.compile(r"[\"“](.+?)[\"”]", re.S)
LETTER = r"[^\W\d_]"
WORD = LETTER + r"+(?:[-'’]" + LETTER + r"+)*"
INITIAL = LETTER + r"\."
PARTICLES = {"van", "von", "de", "del", "der", "den", "da", "di", "al", "el", "la", "le", "du", "dos"}
_NOT_NAME = {"in", "the", "a", "an", "for", "on", "with", "using", "learning", "neural", "networks", "conference",
             "journal", "proceedings"}
_BULLET = re.compile(r"\s*(?:[•●▪*]|\[\d+\]|\d+[.)])\s*")
PUBLICATION_KINDS = {"publications", "patents"}


def sections(text: str) -> list[tuple[int, int, str]]:
    """(start, end, kind) from explicit heading lines only."""
    out, start, kind = [], 0, "other"
    for m in re.finditer(r"[^\n]+", text):
        key = m.group().strip(" \t•:：").casefold()
        if re.match(r"(?i)^(?:technical skills|skills|programming languages)\s*:", m.group().strip()):
            key = "skills"
        if key in SECTION_HEADINGS:
            if m.start() > start:
                out.append((start, m.start(), kind))
            start, kind = m.start(), SECTION_HEADINGS[key]
    if start < len(text):
        out.append((start, len(text), kind))
    return out


def person_atom(value: str) -> bool:
    """Does this comma-free chunk look like one author name (or an already-masked one)?"""
    s = re.sub(r"^\(?\d{4}\)?[.\s:;-]*", "", value.strip(" ,;:&•\t\n"))
    if s == "[PERSON]":
        return True
    toks = re.findall(r"\[PERSON\]|" + WORD + r"\.?", s)
    if not 1 <= len(toks) <= 5:
        return False
    if any(not t.endswith(".") and t.casefold() in _NOT_NAME for t in toks):
        return False
    if len(toks) == 1 and toks[0] != "[PERSON]":
        return False
    return all(t == "[PERSON]" or t[0].isupper() or t.casefold() in PARTICLES for t in toks)


def author_spans(text: str, start: int, end: int) -> list[tuple[int, int]] | None:
    """Split an author block on ; / and / & / 、 / et al. and commas. None if any chunk fails."""
    s, out = text[start:end], []
    seps = re.compile(r"\s*(?:;|\band\b|&|、|\bet\s+al\.)\s*", re.I)
    stops = [(0, 0)] + [m.span() for m in seps.finditer(s)] + [(len(s), len(s))]
    for left, right in zip(stops, stops[1:]):
        lo = start + left[1]
        value = s[left[1]:right[0]]
        lead = len(value) - len(value.lstrip(" \t\n,•:.-"))
        lo += lead
        value = value[lead:].rstrip(" \t\n,:;-")
        if not value:
            continue
        chunks = list(re.finditer(r"[^,]+", value))
        i = 0
        while i < len(chunks):
            c = chunks[i]
            v = c.group().strip()
            a = lo + c.start() + len(c.group()) - len(c.group().lstrip())
            b = lo + c.end()
            # "Family, J." -> one author across the comma
            if i + 1 < len(chunks) and re.fullmatch(WORD, v) and re.fullmatch(r"\s*(?:" + INITIAL + r"\s*){1,3}", chunks[i + 1].group()):
                b = lo + chunks[i + 1].end()
                v = text[a:b]
                i += 1
            if not person_atom(re.sub(r"(?<=\.)(?=[^\W\d_])", " ", v)):
                return None
            while b > a and text[b - 1].isspace():
                b -= 1
            out.append((a, b))
            i += 1
    return out or None


def records(text: str) -> list[tuple[int, int, str]]:
    out = []
    for start, end, kind in sections(text):
        current: list[tuple[int, int]] = []

        def flush():
            if current:
                out.append((current[0][0], current[-1][1], kind))
                current.clear()

        for line in re.finditer(r"[^\n]+", text[start:end]):
            a, b, s = start + line.start(), start + line.end(), line.group().strip()
            if s.casefold().strip(":：") in SECTION_HEADINGS:
                continue
            explicit = kind in PUBLICATION_KINDS
            probable = bool(YEAR.search(s) and (VENUE.search(s) or QUOTE.search(s))
                            and (person_atom(re.split(r"[\"“(]", s)[0]) or re.match(r"^(?:\d{4}\s+)?" + INITIAL, s)))
            if not explicit and not probable:
                continue
            if current:
                block = text[current[0][0]:current[-1][1]]
                if _BULLET.match(line.group()) or (YEAR.search(block) and (QUOTE.search(block) or VENUE.search(block))):
                    flush()
            current.append((a, b))
        flush()
    return out


def parse(text: str) -> list[dict]:
    result = []
    for a, b, kind in records(text):
        cs = a
        if (m := _BULLET.match(text[a:b])):
            cs += m.end()
        if (m := re.match(r"\s*\(?\d{4}\)?[.\s:;-]+", text[cs:b])):
            cs += m.end()
        authors, title = None, None
        q = QUOTE.search(text[cs:b])
        if q:
            qa, qb = cs + q.start(), cs + q.end()
            title = (qa, qb)
            prefix_end = qa
            if (ym := YEAR.search(text[cs:qa])):
                prefix_end = cs + ym.start()
                prefix_end -= int(prefix_end > cs and text[prefix_end - 1] == "(")
            if text[cs:prefix_end].strip(" ,:;.\n"):
                authors = author_spans(text, cs, prefix_end)
            else:                                   # title first: authors run until venue or year
                tail = qb
                while tail < b and text[tail] in " ,.:;\n":
                    tail += 1
                stops = [m.start() for m in (VENUE.search(text, tail, b), YEAR.search(text, tail, b)) if m]
                authors = author_spans(text, tail, min(stops + [b]))
        else:
            m = re.search(r"\((?:19|20)\d{2}\)\s*[.:]?\s*", text[cs:b])
            if m and text[cs + m.end():b].strip():
                authors = author_spans(text, cs, cs + m.start())
                title = (cs + m.end(), b)
        reason = None if authors else ("author_block_unparsed" if title else "no_title_delimiter")
        result.append({"start": a, "end": b, "kind": kind, "authors": authors or [], "title": title,
                       "reason": reason, "remove": not authors})
    return result
