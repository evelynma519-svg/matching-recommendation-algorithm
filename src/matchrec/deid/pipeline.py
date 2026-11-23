"""De-identify a neutral snapshot. Output is a snapshot of the same shape that the matching pipeline reads.

    python -m matchrec.deid.pipeline --snapshot data/synthetic --out data/synthetic-deid --detector heuristic

What happens to each field:

    resume text              resume rules (identity zone, citations, references), own names known
    job title / description  free-text rules, the job's related people known
    job notes, end notes     free-text rules
    org_name                 typed-field rule: masked only when it equals a known person's name
    identity, related_people removed after use

The name directory is every person name in the snapshot's identity records. Protected terms come
only from typed fields (organization names, skill lists), never from free text, so a person's name
in a sentence cannot be whitelisted by accident.

The report holds ratios, rule shares and hashes. It never holds text.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from ..schema import Identity, JsonlSnapshot, Section, write_snapshot
from .audit import rule_report
from .citations import sections as find_sections
from .decisions import DecisionBook
from .detectors import HeuristicDetector, SpacyDetector
from .lexicons import dictionary_words, load_terms
from .redactor import Redactor

POLICY_VERSION = "deid-1.0"


def known_names(cands, jobs) -> set[str]:
    out = set()
    for c in cands:
        i = c.identity
        out |= {n for n in (i.full_name, *i.other_names) if n}
        if i.given_name and i.family_name:
            out.add(f"{i.given_name} {i.family_name}")
    for j in jobs:
        out |= set(j.related_people)
    return out


def protected_terms(cands, jobs) -> set[str]:
    out = {j.org_name for j in jobs if j.org_name}
    out |= {s for j in jobs for s in j.required_skills} | {s for c in cands for s in c.skills}
    return out


class _Out:
    def __init__(self, snap, cands, jobs, apps):
        self.s, self.c, self.j, self.a = snap, cands, jobs, apps

    def snapshot(self): return self.s
    def candidates(self): return iter(self.c)
    def jobs(self): return iter(self.j)
    def applications(self): return iter(self.a)


def deidentify(src, out_dir: Path, redactor_factory) -> dict:
    snap = src.snapshot()
    cands, jobs, apps = list(src.candidates()), list(src.jobs()), list(src.applications())
    names = known_names(cands, jobs)
    red: Redactor = redactor_factory(names, protected_terms(cands, jobs))
    results = {"resume": [], "job_text": [], "note": [], "org_name": []}

    def run(text, own, kind, bucket):
        if not text:
            return text
        r = red.clean(text, own, kind)
        results[bucket].append(r)
        return r.text

    new_cands = []
    for c in cands:
        own = [n for n in (c.identity.full_name, c.identity.given_name, c.identity.family_name, *c.identity.other_names) if n]
        resumes = []
        for v in c.resumes:
            text = run(v.text, own, "resume", "resume")
            secs = [Section(k, text[a:b].strip()) for a, b, k in find_sections(text)]
            resumes.append(replace(v, text=text, sections=secs))
        new_cands.append(replace(c, resumes=resumes, identity=Identity()))

    new_jobs = []
    for j in jobs:
        own = list(j.related_people)
        new_jobs.append(replace(
            j, org_name=run(j.org_name, [], "org_name", "org_name"), title=run(j.title, own, "free_text", "job_text"),
            summary=run(j.summary, own, "free_text", "job_text"),
            responsibilities=run(j.responsibilities, own, "free_text", "job_text"),
            requirements=run(j.requirements, own, "free_text", "job_text"),
            notes=[replace(n, text=run(n.text, own, "free_text", "note")) for n in j.notes],
            related_people=[]))

    new_apps = [replace(a, end_note=run(a.end_note, [], "free_text", "note")) for a in apps]
    out_snap = replace(snap, snapshot_id=snap.snapshot_id + "__deid", source=f"{snap.source}+{POLICY_VERSION}")
    write_snapshot(out_dir, _Out(out_snap, new_cands, new_jobs, new_apps))

    report = {"policy_version": POLICY_VERSION, "built_at": datetime.now(timezone.utc).isoformat(),
              "detector": getattr(red.detector, "name", None), "decisions_sha256": red.decisions.sha256,
              "decisions_version": red.decisions.version, "lexicon_size": len(red.ladder.allow_terms),
              "fields": {k: rule_report(v) for k, v in results.items() if v}}
    (Path(out_dir) / "deid_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="De-identify a neutral snapshot.")
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--detector", choices=["heuristic", "spacy"], default="spacy")
    ap.add_argument("--decisions", type=Path, default=None, help="local term-decision JSON")
    ap.add_argument("--extra-lexicon", type=Path, default=None, help="public vocabulary, one term per line")
    a = ap.parse_args(argv)
    detector = SpacyDetector() if a.detector == "spacy" else HeuristicDetector()
    decisions = DecisionBook.load(a.decisions) if a.decisions else None
    extra = load_terms(a.extra_lexicon) if a.extra_lexicon else ()
    dictionary = dictionary_words()

    def factory(names, protected):
        return Redactor(names, detector, protected, decisions, extra, dictionary)

    rep = deidentify(JsonlSnapshot(a.snapshot), a.out, factory)
    print(json.dumps({k: {"documents": v["documents"], "status_share": v["status_share"]}
                      for k, v in rep["fields"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
