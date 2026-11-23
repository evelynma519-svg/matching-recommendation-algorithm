"""Auditing a de-identification run. Two different failure modes need two different tools:

- **Over-masking** (technical words or ordinary text hidden as [PERSON]) hurts matching quality.
  It is found by *rule audit*: aggregate which layer and rule produced each mask, list masked terms
  that recur across many documents (a real person's name rarely does), and ablate one allow layer at
  a time to see what it protects.
- **Leaks** (a real name left visible) hurt privacy. Aggregates cannot find them; only a human reading
  outputs can. *Leak sampling* draws a stratified sample of outputs for that review.

Reports carry rule names, layers, shape flags and term hashes. Plain terms appear only when
`include_terms=True`, which is meant for a reviewer working locally, never for a shared report.
"""
from __future__ import annotations

import random
import re
from collections import Counter, defaultdict

from .ladder import Layer
from .redactor import Redactor, Result
from .spans import HAN

JUDGED_MASK_LAYERS = {int(Layer.IDENTITY_ZONE), int(Layer.DEFAULT)}


def shape_flags(term: str) -> list[str]:
    """Coarse shape of a term, safe to share: it says what kind of token it is, not what it says."""
    tags = []
    if re.search(r"(?i)\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b|\b(?:19|20)\d{2}\b", term):
        tags.append("date_like")
    if re.search(r"\b[vV]?\d+(?:\.\d+)+\b|[_+#]|::|\.[A-Za-z]+", term):
        tags.append("code_or_version_like")
    if re.fullmatch(r"[A-Z][A-Z0-9/&+.-]*", term) and len(term) > 1:
        tags.append("all_caps")
    if re.search(r"[a-z][A-Z]", term):
        tags.append("camel_case")
    if any(c.isdigit() for c in term):
        tags.append("has_digit")
    if HAN.search(term):
        tags.append("cjk")
    if len(term.split()) >= 2:
        tags.append("multi_word")
    return tags or ["plain_word"]


def rule_report(results: list[Result]) -> dict:
    """Shares by layer and rule, document status and issues. Ratios only, plus the document count."""
    n_docs = max(len(results), 1)
    masked, allowed = Counter(), Counter()
    status, issues = Counter(r.status for r in results), Counter(i for r in results for i in r.issues)
    for r in results:
        for d in r.diagnostics:
            (masked if d["action"] == "mask" else allowed)[(d["layer"], d["rule"])] += 1
    tot_m, tot_a = max(sum(masked.values()), 1), max(sum(allowed.values()), 1)
    return {
        "documents": len(results),
        "status_share": {k: round(v / n_docs, 4) for k, v in sorted(status.items())},
        "issue_share": {k: round(v / n_docs, 4) for k, v in sorted(issues.items())},
        "mask_share_by_rule": {f"L{l}:{rule}": round(v / tot_m, 4) for (l, rule), v in sorted(masked.items())},
        "allow_share_by_rule": {f"L{l}:{rule}": round(v / tot_a, 4) for (l, rule), v in sorted(allowed.items())},
        "tech_terms_masked_per_doc": round(sum(r.stats.get("tech_terms_masked", 0) for r in results) / n_docs, 4),
        "mean_placeholder_char_ratio": round(sum(r.stats.get("placeholder_char_ratio", 0) for r in results) / n_docs, 4),
    }


def overmask_candidates(results: list[Result], sources: list[str], min_docs: int = 3,
                        include_terms: bool = False, limit: int = 200) -> list[dict]:
    """Terms masked by judgement layers (identity zone, default) in at least `min_docs` documents.
    These are the first candidates for a human allow decision.

    `sources` are the normalized source texts the diagnostics' offsets refer to (`Result.stats` does
    not keep them, so the caller passes them; they stay in local memory)."""
    docs: dict[str, set[int]] = defaultdict(set)
    rules: dict[str, Counter] = defaultdict(Counter)
    example: dict[str, str] = {}
    for i, (r, src) in enumerate(zip(results, sources)):
        for d in r.diagnostics:
            if d["action"] == "mask" and d["layer"] in JUDGED_MASK_LAYERS:
                h = d["term_sha256"]
                docs[h].add(i)
                rules[h][d["rule"]] += 1
                example.setdefault(h, src[d["source_start"]:d["source_end"]])
    rows = []
    for h, ds in docs.items():
        if len(ds) < min_docs:
            continue
        row = {"term_sha256": h, "documents": len(ds), "rules": dict(rules[h]), "shape": shape_flags(example[h])}
        if include_terms:
            row["term"] = example[h]
        rows.append(row)
    return sorted(rows, key=lambda r: -r["documents"])[:limit]


def ablate(redactor: Redactor, docs: list[tuple[str, list[str], str]], layers=(7, 8, 9)) -> dict:
    """Re-run with one allow layer switched off at a time. The rise in masked spans per document is
    what that layer protects. `docs` = [(text, own_names, kind)]."""
    def masks(ladder_disabled) -> float:
        old = redactor.ladder.disabled
        redactor.ladder.disabled = frozenset(ladder_disabled)
        try:
            return sum(len([d for d in redactor.clean(t, o, k).diagnostics if d["action"] == "mask"])
                       for t, o, k in docs) / max(len(docs), 1)
        finally:
            redactor.ladder.disabled = old

    base = masks(())
    return {"masks_per_doc": round(base, 3),
            **{f"without_layer_{l}": {"masks_per_doc": round(m := masks({l}), 3),
                                      "relative_increase": round((m - base) / base, 4) if base else None}
               for l in layers}}


def leak_sample(results: list[Result], n: int = 50, seed: int = 0) -> list[int]:
    """Indices of documents for human leak review, stratified by status and by placeholder density
    (low-density outputs are where a missed name hides), so small strata are not skipped."""
    strata: dict[tuple, list[int]] = defaultdict(list)
    for i, r in enumerate(results):
        dens = r.stats.get("placeholder_char_ratio", 0)
        band = "none" if dens == 0 else "low" if dens < 0.02 else "mid" if dens < 0.1 else "high"
        strata[(r.status, band)].append(i)
    rng = random.Random(seed)
    keys = sorted(strata)
    picked: list[int] = []
    for k in keys:                                    # at least one from each stratum
        if len(picked) < n:
            picked.append(rng.choice(strata[k]))
    rest = [i for k in keys for i in strata[k] if i not in picked]
    rng.shuffle(rest)
    return sorted(picked + rest[:max(0, n - len(picked))])


def main(argv=None) -> int:
    """python -m matchrec.deid.audit --snapshot <raw snapshot> --out audit.json [--detector spacy]"""
    import argparse
    import json
    from pathlib import Path

    from ..schema import JsonlSnapshot
    from .decisions import DecisionBook
    from .detectors import HeuristicDetector, SpacyDetector
    from .lexicons import dictionary_words, load_terms
    from .pipeline import known_names, protected_terms
    from .text import plain, split_glued_sentences

    ap = argparse.ArgumentParser(description="Rule audit, ablation and leak sample for resume de-identification.")
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--detector", choices=["heuristic", "spacy"], default="spacy")
    ap.add_argument("--decisions", type=Path, default=None)
    ap.add_argument("--extra-lexicon", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=500, help="resumes to audit")
    ap.add_argument("--sample", type=int, default=50, help="documents to draw for human leak review")
    ap.add_argument("--min-docs", type=int, default=3)
    a = ap.parse_args(argv)

    src = JsonlSnapshot(a.snapshot)
    cands, jobs = list(src.candidates()), list(src.jobs())
    red = Redactor(known_names(cands, jobs), SpacyDetector() if a.detector == "spacy" else HeuristicDetector(),
                   protected_terms(cands, jobs), DecisionBook.load(a.decisions) if a.decisions else None,
                   load_terms(a.extra_lexicon) if a.extra_lexicon else (), dictionary_words())
    docs = [(r.text, [n for n in (c.identity.full_name, *c.identity.other_names) if n], "resume")
            for c in cands for r in c.resumes][:a.limit]
    results = [red.clean(*d) for d in docs]
    sources = [split_glued_sentences(plain(t), red.keep_glued)[0] for t, _, _ in docs]
    report = {"rules": rule_report(results), "overmask_candidates": overmask_candidates(results, sources, a.min_docs),
              "ablation": ablate(red, docs[:min(len(docs), 200)]),
              "leak_review_sample": leak_sample(results, a.sample)}
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"audited {len(docs)} resumes -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
