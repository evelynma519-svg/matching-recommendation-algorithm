"""Span-level redaction driven by the precedence ladder.

    source text
      -> normalize (HTML, NFKC, glued sentences)
      -> fixed spans: contacts, known names, references, citations           (layers 1-4)
      -> detector proposals on the text with fixed spans blanked
      -> each proposal grown to whole tokens, split at technical/protected boundaries,
         and each piece judged by Ladder.decide                              (layers 5-10)
      -> render placeholders; re-run the detector on the output until nothing new appears
      -> re-check contact patterns that a replacement may have exposed
      -> residual checks decide the status

Only the ambiguous span is masked, never the whole document. Every decision is recorded with source
offsets, layer and rule, and a SHA-256 of the term rather than the term itself, so diagnostics can be
aggregated and audited without carrying the sensitive text.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field, replace

from .citations import parse as parse_citations
from .citations import sections as find_sections
from .decisions import DecisionBook
from .detectors import DetectorUnavailable, PersonDetector
from .ladder import Context, Ladder, Layer, Span, choose_layer
from .lexicons import DUAL_USE_NAMES, QUALIFICATION, RESUME_VOCAB, TECH
from .patterns import contact_spans
from .spans import SYMBOL_PATTERN, NameIndex, aliases, overlaps, phrase_pattern, whole_token
from .text import norm, plain, split_glued_sentences

MARKERS = re.compile(r"\[(?:PERSON|AMBIG|EMAIL|PHONE|URL|SOCIAL|ID|DOB|ADDRESS|REFERENCES_REMOVED|CITATION_REMOVED)\]")
KINDS = ("resume", "free_text", "org_name")
PASSED, REVIEW = "passed_pending_sampling", "review_required"


@dataclass
class Result:
    text: str
    status: str
    issues: list[str]
    redactions: dict[str, int]
    diagnostics: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


class Redactor:
    def __init__(self, known_names=(), detector: PersonDetector | None = None, protected_terms=(),
                 decisions: DecisionBook | None = None, extra_lexicon=(), dictionary=frozenset(),
                 max_rechecks: int = 10, max_density: float = 0.35):
        self.known = NameIndex(known_names)
        self.detector = detector
        self.protected_terms = frozenset(norm(t) for t in protected_terms if t)
        self.protected = phrase_pattern(self.protected_terms, min_len=2)
        self.decisions = decisions or DecisionBook()
        allow = TECH | RESUME_VOCAB | QUALIFICATION | frozenset(norm(t) for t in extra_lexicon)
        self.ladder = Ladder(allow, self.decisions, frozenset(dictionary))
        self.lexical = phrase_pattern(allow | self.decisions.terms())
        self.keep_glued = lambda t: t in allow or t in self.protected_terms
        self.max_rechecks, self.max_density = max_rechecks, max_density

    # ---------------------------------------------------------------- fixed spans and zones
    def context(self, text: str, own_names, kind: str) -> Context:
        ctx = Context(text, kind)
        ctx.protected = [m.span() for m in self.protected.finditer(text)] if self.protected else []
        own = NameIndex(aliases(own_names))
        if kind == "org_name":            # a typed organization field: masked only if it IS a person's name
            if norm(text) in self.known.names:
                ctx.fixed = [Span(0, len(text), Layer.KNOWN_PERSON, "org_field_equals_known_person", "PERSON")]
            return ctx
        fixed = [Span(a, b, Layer.CONTACT, "contact_" + lab, lab) for a, b, lab in contact_spans(text)]
        for a, b in self.known.spans(text, ctx.protected) + own.spans(text, ctx.protected):
            fixed.append(Span(a, b, Layer.KNOWN_PERSON, "known_person", "PERSON"))
        if kind == "resume":
            ctx.sections = find_sections(text)
            ctx.citations = parse_citations(text)
            for a, b, k in ctx.sections:
                if k == "references":
                    fixed.append(Span(a, b, Layer.REFERENCES, "references_section", "REFERENCES_REMOVED"))
            for c in ctx.citations:
                if c["remove"]:
                    fixed.append(Span(c["start"], c["end"], Layer.CITATION, "citation_" + c["reason"], "CITATION_REMOVED"))
                else:
                    fixed += [Span(a, b, Layer.CITATION, "citation_author", "PERSON") for a, b in c["authors"]]
            ctx.identity = self._identity_zone(text, ctx.sections, own, fixed)
        ctx.fixed = choose_layer(fixed)
        ctx.symbols = [m.span() for m in SYMBOL_PATTERN.finditer(text)]
        lex = [m.span() for m in self.lexical.finditer(text)] if self.lexical else []
        ctx.lexical = [(a, b) for a, b in lex if not overlaps(ctx.symbols, a, b)]
        return ctx

    @staticmethod
    def _identity_zone(text, secs, own: NameIndex, fixed) -> list[tuple[int, int]]:
        """Header lines before the first real section: the line with the candidate's own name or a
        contact, plus one line either side. Without an anchor, the first three header lines."""
        lines = list(re.finditer(r"[^\n]*\n|[^\n]+$", text))
        first_heading = next((a for a, b, k in secs if k != "other"), len(text))
        eligible = [i for i, m in enumerate(lines[:25]) if m.start() < first_heading]
        own_spans = own.spans(text)
        anchors = [i for i in eligible
                   if overlaps(own_spans, *lines[i].span())
                   or any(e.layer == Layer.CONTACT and e.start < lines[i].end() and e.end > lines[i].start() for e in fixed)]
        chosen = {j for i in anchors for j in (i - 1, i, i + 1) if j in eligible} if anchors else set(eligible[:3])
        return [lines[i].span() for i in sorted(chosen)]

    # ---------------------------------------------------------------- main entry
    def clean(self, text, own_names=(), kind: str = "free_text", recheck: bool = True) -> Result:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        source, glued = split_glued_sentences(plain(text), self.keep_glued)
        ctx = self.context(source, own_names, kind)
        edits: list[Span] = list(ctx.fixed)
        diags: list[dict] = []
        issues: set[str] = set()
        seen: set[tuple[int, int, bool]] = set()
        passes = 0

        def record(a, b, verdict, origin):
            diags.append({"layer": verdict.layer, "rule": verdict.rule, "label": verdict.label,
                          "action": verdict.action, "identity": verdict.identity, "section": verdict.section,
                          "kind": kind, "origin": origin, "source_start": a, "source_end": b,
                          "term_sha256": hashlib.sha256(source[a:b].encode()).hexdigest()})
            if verdict.label:
                edits.append(Span(a, b, verdict.layer, verdict.rule, verdict.label))

        for e in ctx.fixed:
            record(e.start, e.end, self.ladder.decide(ctx, e.start, e.end), "fixed")

        def render():
            picked = []
            for e in choose_layer(edits):          # touching spans with the same placeholder render once
                if picked and picked[-1].end == e.start and picked[-1].label == e.label:
                    picked[-1] = Span(picked[-1].start, e.end, picked[-1].layer, picked[-1].rule, e.label)
                else:
                    picked.append(e)
            parts, mapping, cur = [], [], 0
            for e in picked:
                parts.append(source[cur:e.start])
                mapping += [(i, i + 1) for i in range(cur, e.start)]
                rep = f"[{e.label}]"
                parts.append(rep)
                mapping += [(e.start, e.end)] * len(rep)
                cur = e.end
            parts.append(source[cur:])
            mapping += [(i, i + 1) for i in range(cur, len(source))]
            return "".join(parts), mapping, picked

        def propose(view: str, mapping) -> bool:
            """Judge detector proposals on `view` (mapped back to source). True if a new mask appeared."""
            nonlocal passes
            passes += 1
            if not any(c.isalpha() for c in view):
                return False
            before = len(edits)
            for cand in self.detector.detect(view):
                x, y = cand.start, cand.end
                if not 0 <= x < y <= len(mapping) or not view[x:y].strip():
                    continue
                a, b = mapping[x][0], mapping[y - 1][1]
                wa, wb = whole_token(source, a, b)
                partial = wa < a or wb > b
                a, b = (wa, wb) if partial else (a, b)
                cuts = {a, b}
                for lo, hi in ctx.symbols + ctx.protected + ctx.lexical + [(f.start, f.end) for f in ctx.fixed]:
                    if lo < b and hi > a:
                        cuts.update((max(a, lo), min(b, hi)))
                cuts = sorted(cuts)
                judged = []
                for lo, hi in zip(cuts, cuts[1:]):
                    while lo < hi and source[lo].isspace():
                        lo += 1
                    while hi > lo and source[hi - 1].isspace():
                        hi -= 1
                    if lo >= hi or not any(c.isalpha() for c in source[lo:hi]) or (lo, hi, cand.ambiguous) in seen:
                        continue
                    seen.add((lo, hi, cand.ambiguous))
                    judged.append((lo, hi, self.ladder.decide(ctx, lo, hi, cand.ambiguous, partial)))
                # "Julia Venrid": a dual-use word kept by the lexicon, inside a proposal whose other part
                # was masked as a person, is that person's name too.
                if any(v.label in ("PERSON", "AMBIG") for _, _, v in judged):
                    judged = [(lo, hi, replace(v, layer=int(Layer.DEFAULT), rule="dual_use_name_in_person_run",
                                               label="PERSON"))
                              if v.label is None and norm(source[lo:hi]) in DUAL_USE_NAMES and source[lo].isupper()
                              and v.layer in (Layer.LEXICON, Layer.FORMAT_RULE) else (lo, hi, v)
                              for lo, hi, v in judged]
                for lo, hi, verdict in judged:
                    if verdict.label and any(z.start <= lo and z.end >= hi for z in edits):
                        continue
                    record(lo, hi, verdict, self.detector.name)
            return len(edits) > before

        def detect_until_stable(limit):
            for _ in range(limit):
                out, mapping, _ = render()
                scrub = MARKERS.sub(lambda m: " " * len(m.group()), out)
                if not propose(scrub, mapping):
                    return
            issues.add("person_residual_after_rechecks")

        if kind != "org_name":
            if self.detector is None:
                issues.add("no_person_detector")
            else:
                try:
                    detect_until_stable(self.max_rechecks if recheck else 1)
                except DetectorUnavailable:
                    issues.add("person_detector_failed")

        # A replacement can join characters into a new contact pattern; mask it on the source and re-detect.
        output, mapping, picked = render()
        for _ in range(3):
            added = False
            for a, b, lab in contact_spans(MARKERS.sub(lambda m: " " * len(m.group()), output)):
                sa, sb = mapping[a][0], mapping[b - 1][1]
                if any(e.start <= sa and e.end >= sb and e.layer == Layer.CONTACT for e in edits):
                    continue
                ctx.fixed.append(Span(sa, sb, Layer.CONTACT, "contact_recheck_" + lab, lab))
                record(sa, sb, self.ladder.decide(ctx, sa, sb), "contact_recheck")
                added = True
            if not added:
                break
            if self.detector is not None and recheck and kind != "org_name":
                detect_until_stable(3)
            output, mapping, picked = render()

        # ---- residual checks
        visible = MARKERS.sub(" ", output)
        if contact_spans(visible):
            issues.add("contact_pattern_remaining")
        if kind != "org_name" and (self.known.search(output) or NameIndex(aliases(own_names)).search(output)):
            issues.add("known_name_remaining")
        density = sum(len(m.group()) for m in MARKERS.finditer(output)) / max(1, len(output))
        if len(source) >= 100 and density > self.max_density:
            issues.add("high_redaction_density")
        counts = Counter(e.label for e in picked)
        stats = {"source_chars": len(source), "output_chars": len(output), "placeholder_char_ratio": round(density, 4),
                 "detector_passes": passes, "glued_sentences_split": glued,
                 "allowed_by_rule": dict(Counter(d["rule"] for d in diags if d["action"] == "allow")),
                 "masked_by_rule": dict(Counter(e.rule for e in picked)),
                 "tech_terms_masked": sum(norm(source[e.start:e.end]) in self.ladder.allow_terms for e in picked),
                 "decisions_sha256": self.decisions.sha256}
        return Result(output, REVIEW if issues else PASSED, sorted(issues), dict(counts), diags, stats)
