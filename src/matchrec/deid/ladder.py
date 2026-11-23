"""The precedence ladder: the one function that decides whether a span is masked.

Every candidate span, whatever proposed it, goes through `Ladder.decide`. Layers are tried in
order; the first that claims the span decides. Lower layers can never enlarge or override a
higher layer's mask.

    1  contact          email, phone, URL, social handle, ID number, address, date of birth
    2  known person     exact full-span match of a name from the snapshot's identity records
    3  references       a "References" section is removed whole
    4  citation         validated author spans -> PERSON; unparsable citation records removed whole
    5  identity zone    resume header lines near the candidate's own name or contacts: mask by default
    6  human decision   reviewed term decisions (mask wins over allow)
    7  symbol token     indivisible technical tokens such as C++, C#, Node.js
    8  lexicon          public technical vocabulary and protected organization/skill terms
    9  format rule      months, date tokens, title and degree abbreviations, code identifiers, ...
    10 default          anything a detector proposed that no layer allowed: mask

Layers 1-4 are fixed spans computed before detection. Layers 5-10 judge detector proposals.
The same function serves redaction, simulation and audit, so an audit result describes exactly
what redaction does.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import IntEnum

from .decisions import DecisionBook
from .lexicons import DEGREES, JD_PHRASES, MONTHS, PERSON_CONTEXT, SHORT_DEGREES, TECH, TITLE_ABBR
from .spans import TOKEN, containing, overlaps
from .text import norm, window


class Layer(IntEnum):
    CONTACT = 1
    KNOWN_PERSON = 2
    REFERENCES = 3
    CITATION = 4
    IDENTITY_ZONE = 5
    HUMAN_DECISION = 6
    SYMBOL_TOKEN = 7
    LEXICON = 8
    FORMAT_RULE = 9
    DEFAULT = 10


@dataclass
class Span:
    start: int
    end: int
    layer: int
    rule: str
    label: str


@dataclass(frozen=True)
class Verdict:
    layer: int
    rule: str
    label: str | None          # placeholder to insert, or None to keep the text
    identity: bool
    section: str

    @property
    def action(self) -> str:
        return "mask" if self.label else "allow"


@dataclass
class Context:
    text: str
    kind: str                                   # resume | free_text | org_name
    fixed: list[Span] = field(default_factory=list)
    identity: list[tuple[int, int]] = field(default_factory=list)
    sections: list[tuple[int, int, str]] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    symbols: list[tuple[int, int]] = field(default_factory=list)
    protected: list[tuple[int, int]] = field(default_factory=list)
    lexical: list[tuple[int, int]] = field(default_factory=list)


def choose_layer(spans: list[Span]) -> list[Span]:
    """Disjoint slices; where spans overlap, the lower layer number keeps the characters."""
    chosen: list[Span] = []
    for e in sorted(spans, key=lambda e: (e.layer, e.start, -e.end)):
        pieces = [(e.start, e.end)]
        for old in chosen:
            nxt = []
            for a, b in pieces:
                if old.end <= a or old.start >= b:
                    nxt.append((a, b))
                    continue
                if a < old.start:
                    nxt.append((a, old.start))
                if old.end < b:
                    nxt.append((old.end, b))
            pieces = nxt
        chosen += [Span(a, b, e.layer, e.rule, e.label) for a, b in pieces if a < b]
    return sorted(chosen, key=lambda e: e.start)


_DATE_TOKEN = re.compile(r"(?i)(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
                         r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\d{4}|(?:19|20)\d{2}[./-]\d{1,2}|"
                         r"\d{1,2}[./-](?:19|20)\d{2}")
_INSTITUTION = r"(?:University|College|Institute|School|Academy|Polytechnic|大学|学院)"
_INSTITUTION_NEAR = re.compile(rf"^[ \t]*(?:[A-Z][\w&'.-]*[ \t]+){{0,3}}{_INSTITUTION}\b|^[ \t]*of[ \t]+(?:Technology|Science|Sciences)\b")
_INSTITUTION_AFTER = re.compile(rf"\b{_INSTITUTION}[ \t]+of[ \t]+(?:[A-Z][\w&'.-]*[ \t]+){{0,2}}$")
_BARE_C_TECH = re.compile(r"(?i)\bC\s*(?:/|,|and|&|、)\s*(?:C\+\+|C#|Objective-C)|(?:C\+\+|C#|Objective-C)\s*(?:/|,|and|&|、)\s*C\b|"
                          r"\bC\s+(?:programming|language)\b|\bin\s+C\b|C\s*语言")


class Ladder:
    def __init__(self, allow_terms: frozenset[str] = TECH, decisions: DecisionBook | None = None,
                 dictionary: frozenset[str] = frozenset(), disabled_layers: frozenset[int] = frozenset()):
        self.allow_terms = frozenset(allow_terms)
        self.disabled = frozenset(disabled_layers)     # ablation: allow verdicts from these layers fall to default
        self.decisions = decisions or DecisionBook()
        self.dictionary = dictionary

    # ---------------------------------------------------------------- layer 9
    def format_rule(self, ctx: Context, a: int, b: int, partial: bool, identity: bool, section: str) -> str | None:
        text, word = ctx.text, ctx.text[a:b]
        n = norm(word).strip("•.,;:()[]")
        if n in MONTHS:
            return "month"
        if _DATE_TOKEN.fullmatch(n):
            return "date_token"
        if n in TITLE_ABBR:
            return "title_abbr"
        if identity:
            return None                                   # header lines get only the three rules above
        if any(not c["remove"] and c["start"] <= a < b <= c["end"] and not overlaps(c["authors"], a, b)
               for c in ctx.citations):
            return "citation_metadata"                    # title / venue words inside a parsed citation
        if n in DEGREES:
            return "degree"
        if n.replace(".", "") in SHORT_DEGREES and (section == "education" or "." in word):
            return "degree_short_scoped"
        if any(m.start() <= a and m.end() >= b for p in JD_PHRASES for m in re.finditer(re.escape(p), text, re.I)):
            return "jd_phrase"
        if n == "c":
            return self._bare_c(ctx, a, b, section)
        if word[:1].isupper() and (_INSTITUTION_NEAR.match(text[b:b + 60]) or _INSTITUTION_AFTER.search(text[max(0, a - 60):a])
                                   or re.match(rf"{_INSTITUTION}\b", word)):
            return "institution_name"                     # "Cortek Institute of Technology", "University of X"
        if re.search("[A-Za-z]", word) and (re.search("[0-9_]", word) or re.search("[a-z][A-Z]", word)) \
                or re.fullmatch(r"[a-z]+(?:\.[a-z]+)+", word.strip(".")):
            return "code_identifier"
        if partial and word.islower() and word in self.dictionary:
            return "dictionary_subtoken"
        return None

    def _bare_c(self, ctx: Context, a: int, b: int, section: str) -> str | None:
        """'C' is a language and an initial. Keep it only with clear technical evidence."""
        text = ctx.text
        near = window(text, a, b, 30)
        if any(c["start"] <= a < c["end"] for c in ctx.citations) or re.match(r"\.\s*[A-Z]", text[b:]) \
                or PERSON_CONTEXT.search(near):
            return None
        lo = text.rfind("\n", 0, a) + 1
        hi = text.find("\n", b)
        line = text[lo:len(text) if hi < 0 else hi]
        toks = list(TOKEN.finditer(line))
        idx = next((i for i, m in enumerate(toks) if m.start() <= a - lo < m.end()), None)
        neighbours = [] if idx is None else [norm(m.group()) for i, m in enumerate(toks) if i != idx and abs(i - idx) <= 3]
        if section == "skills" or _BARE_C_TECH.search(line) or sum(t in self.allow_terms - {"c"} for t in neighbours) >= 2:
            return "bare_c_technical"
        return None

    # ---------------------------------------------------------------- the decision
    def decide(self, ctx: Context, a: int, b: int, ambiguous: bool = False, partial: bool = False) -> Verdict:
        text = ctx.text
        n = norm(text[a:b]).strip("•.,;:()[]")
        section = next((k for x, y, k in ctx.sections if x <= a < y), "other")
        identity = overlaps(ctx.identity, a, b)
        mask_label = "AMBIG" if ambiguous else "PERSON"

        def v(layer, rule, label=None):
            if label is None and int(layer) in self.disabled:
                return Verdict(int(Layer.DEFAULT), f"default_after_disabled_{rule}", mask_label, identity, section)
            return Verdict(int(layer), rule, label, identity, section)

        fixed = next((e for e in ctx.fixed if e.start < b and e.end > a), None)
        if fixed:                                                             # layers 1-4
            return v(fixed.layer, fixed.rule, fixed.label)
        symbol = containing(ctx.symbols, a, b)
        person_ctx = bool(PERSON_CONTEXT.search(window(text, a, b, 65)))
        in_lexicon = (n in self.allow_terms or containing(ctx.protected, a, b)) and n != "c"
        fmt = self.format_rule(ctx, a, b, partial, identity, section)

        if identity:                                                          # layer 5
            if symbol:
                return v(Layer.SYMBOL_TOKEN, "symbol_token")
            if n in self.allow_terms and len(n) >= 3 and not person_ctx:
                return v(Layer.LEXICON, "identity_technical_token")
            if fmt in {"month", "date_token", "title_abbr"}:
                return v(Layer.FORMAT_RULE, fmt)
            return v(Layer.IDENTITY_ZONE, "identity_default", mask_label)
        human = self.decisions.lookup(n, text, a, b, section)                 # layer 6
        if human == "mask":
            return v(Layer.HUMAN_DECISION, "human_mask", "AMBIG")
        if human == "allow":
            return v(Layer.HUMAN_DECISION, "human_allow")
        if symbol:                                                            # layer 7
            return v(Layer.SYMBOL_TOKEN, "symbol_token")
        if in_lexicon and n not in SHORT_DEGREES and not person_ctx:          # layer 8
            return v(Layer.LEXICON, "lexicon")
        if fmt:                                                               # layer 9
            return v(Layer.FORMAT_RULE, fmt)
        return v(Layer.DEFAULT, "default_ambiguous" if ambiguous else "default", mask_label)   # layer 10
