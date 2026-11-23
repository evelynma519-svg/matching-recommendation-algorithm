"""De-identification: layer behaviour, audit tools, and an end-to-end leak check on synthetic data."""
import json
import re

import pytest

from matchrec.deid.audit import ablate, leak_sample, overmask_candidates, rule_report, shape_flags
from matchrec.deid.decisions import DecisionBook, TermDecision
from matchrec.deid.detectors import HeuristicDetector
from matchrec.deid.ladder import Layer, Span, choose_layer
from matchrec.deid.patterns import phone_like
from matchrec.deid.pipeline import deidentify
from matchrec.deid.redactor import PASSED, Redactor
from matchrec.schema import JsonlSnapshot, write_snapshot
from matchrec.synthetic.generate import in_memory

OWN = ["Kalo Venrid"]
RESUME = """Kalo Venrid
kalo.venrid@example.test | +1-415-555-0142 | Metro A
github.com/kvenrid

Summary
Senior Backend Engineer with Python, C++, C# and Node.js. Mentored by Mira Tesloq at Zorbel Robotics.
Worked with Jordan Pell on Kafka pipelines. Programming in C and C++.

Experience
Staff Software Engineer
Zorbel Robotics | Jun 2021 - Present
Docker, Kubernetes, PostgreSQL.

Publications
Venrid, K., Pell, J. and Okafor, T. "Fast joins on streams." Proc. VLDB 2022.
Some garbled line Zeta Quorn 2021 "Untitled" with no structure ;; IEEE

References
Ana Quell, former manager, ana.quell@example.test
"""


@pytest.fixture()
def red():
    return Redactor(known_names=["Kalo Venrid", "Mira Tesloq"], detector=HeuristicDetector(),
                    protected_terms=["Zorbel Robotics"])


def test_contacts_and_known_names_are_masked(red):
    out = red.clean(RESUME, OWN, "resume").text
    for leak in ("kalo.venrid@example.test", "555-0142", "github.com/kvenrid", "Kalo", "Venrid", "Mira Tesloq"):
        assert leak not in out


def test_unknown_third_party_names_are_masked(red):
    out = red.clean(RESUME, OWN, "resume").text
    assert "Jordan Pell" not in out and "Okafor" not in out and "Ana Quell" not in out


def test_technical_tokens_and_protected_orgs_survive(red):
    out = red.clean(RESUME, OWN, "resume").text
    for keep in ("Python", "C++", "C#", "Node.js", "Kafka", "Docker", "Kubernetes", "PostgreSQL",
                 "Zorbel Robotics", "Senior Backend Engineer", "Staff Software Engineer", "Jun 2021"):
        assert keep in out, keep
    assert "Programming in C and C++" in out                         # bare C kept with technical evidence


def test_citations_keep_titles_and_remove_unparsable_records(red):
    res = red.clean(RESUME, OWN, "resume")
    assert '"Fast joins on streams."' in res.text and "Proc. VLDB 2022" in res.text
    rules = res.stats["masked_by_rule"]
    assert rules.get("citation_author", 0) >= 1
    assert "Zeta Quorn" not in res.text


def test_references_section_removed_whole(red):
    res = red.clean(RESUME, OWN, "resume")
    assert "[REFERENCES_REMOVED]" in res.text and "former manager" not in res.text


def test_identity_zone_masks_header_but_not_dates_or_tech():
    r = Redactor(detector=HeuristicDetector())
    out = r.clean("Lumo Fenrik\nPython Developer | Metro B\n\nExperience\nWork", ["Lumo Fenrik"], "resume").text
    assert "Lumo" not in out and "Python" in out and "Developer" in out


def test_human_decisions_scope_and_mask_wins():
    book = DecisionBook([TermDecision("Kepler", "allow", "tech_context"),
                         TermDecision("Grace", "allow", "global"), TermDecision("Grace", "mask", "global")])
    r = Redactor(detector=HeuristicDetector(), decisions=book)
    tech = r.clean("Optimized kernels using Kepler architecture GPUs.").text
    social = r.clean("We met Kepler yesterday; he joined later.").text
    assert "Kepler" in tech and "Kepler" not in social
    assert "Grace" not in r.clean("Notes from talking to Grace about the role.").text


def test_org_name_field_is_masked_only_when_it_is_a_person():
    r = Redactor(known_names=["Tavi Ormand"], detector=HeuristicDetector())
    assert r.clean("Tavi Ormand", kind="org_name").text == "[PERSON]"
    assert r.clean("Ormand Robotics", kind="org_name").text == "Ormand Robotics"


def test_lower_layer_never_enlarges_a_higher_layer():
    picked = choose_layer([Span(0, 20, Layer.REFERENCES, "references_section", "REFERENCES_REMOVED"),
                           Span(5, 10, Layer.CONTACT, "contact_EMAIL", "EMAIL")])
    assert [(p.start, p.end, p.label) for p in picked] == [(0, 5, "REFERENCES_REMOVED"), (5, 10, "EMAIL"),
                                                          (10, 20, "REFERENCES_REMOVED")]


@pytest.mark.parametrize("value,phone", [("+1-415-555-0142", True), ("2019-2023", False),
                                         ("2024-03-01", False), ("555-0142", True), ("12345", False)])
def test_phone_like(value, phone):
    assert phone_like(value) is phone


def test_no_detector_is_flagged_not_silently_passed():
    res = Redactor().clean("Talked with Some Person about it.")
    assert res.status != PASSED and "no_person_detector" in res.issues


def test_audit_tools(red):
    docs = [(RESUME, OWN, "resume")] * 3 + [("Worked with Jordan Pell on Kafka.", [], "free_text")]
    results = [red.clean(t, o, k) for t, o, k in docs]
    rep = rule_report(results)
    assert rep["documents"] == 4 and abs(sum(rep["mask_share_by_rule"].values()) - 1) < 1e-3
    cands = overmask_candidates(results, [red.clean(t, o, k) and t for t, o, k in docs], min_docs=3)
    assert all("term" not in c and len(c["term_sha256"]) == 64 for c in cands)
    abl = ablate(red, docs, layers=(8,))
    assert abl["without_layer_8"]["masks_per_doc"] > abl["masks_per_doc"]
    assert set(leak_sample(results, n=2)) <= set(range(4))
    assert "code_or_version_like" in shape_flags("C++") and "cjk" in shape_flags("工程师")


def test_end_to_end_snapshot_has_no_identity_leaks(tmp_path):
    src = in_memory(n_candidates=200, n_jobs=40, seed=21)
    write_snapshot(tmp_path / "raw", src)
    raw = JsonlSnapshot(tmp_path / "raw")
    secrets = set()
    for c in raw.candidates():
        i = c.identity
        secrets |= {i.full_name, *i.emails, *i.phones}
        secrets |= {n for n in (i.given_name, i.family_name) if n and len(n) >= 3}
    for j in raw.jobs():
        secrets |= set(j.related_people)
    rep = deidentify(raw, tmp_path / "deid", lambda names, prot: Redactor(names, HeuristicDetector(), prot))
    blob = "".join((tmp_path / "deid" / f).read_text() for f in ("candidates.jsonl", "jobs.jsonl", "applications.jsonl"))
    visible = lambda s: re.search(rf"(?<![\w\u4e00-\u9fff]){re.escape(s)}(?![\w\u4e00-\u9fff])", blob)
    assert [s for s in secrets if visible(s)] == []
    # Third-party names (colleagues, co-authors, referees) are not in any identity record: the
    # detector and the ladder must find them. Ground truth comes from the generator.
    third = src.third_party_names()
    leaked = [p for p in third if visible(p)]
    assert len(third) > 50 and len(leaked) / len(third) < 0.02
    assert rep["fields"]["resume"]["documents"] == 200


def test_skill_mentions_survive_deidentification(tmp_path):
    """Over-masking check: skill phrases written in the raw resume are still readable afterwards."""
    from matchrec.synthetic.vocab import SURFACE, ZH_SKILL
    src = in_memory(n_candidates=200, n_jobs=40, seed=22)
    write_snapshot(tmp_path / "raw", src)
    deidentify(JsonlSnapshot(tmp_path / "raw"), tmp_path / "deid", lambda n, p: Redactor(n, HeuristicDetector(), p))
    forms = {f for fs in SURFACE.values() for f in fs} | set(ZH_SKILL.values())
    pat = lambda f: re.compile(rf"(?<![\w+#.]){re.escape(f)}(?![\w+#])")
    before = after = 0
    for r, d in zip(JsonlSnapshot(tmp_path / "raw").candidates(), JsonlSnapshot(tmp_path / "deid").candidates()):
        for f in forms:
            n = len(pat(f).findall(r.resumes[0].text))
            before += n
            after += min(n, len(pat(f).findall(d.resumes[0].text)))
    assert before > 500 and after / before > 0.97


def test_dual_use_given_names_and_chinese_cue_names():
    r = Redactor(detector=HeuristicDetector())
    out = r.clean("Worked with Julia Venrid on Kafka. Prototyped models in Julia and Python.").text
    assert "Julia Venrid" not in out and "Venrid" not in out
    assert "models in Julia and Python" in out
    zh = r.clean("负责核心系统的开发，使用 Python。与王小明合作完成平台迁移。").text
    assert "王小明" not in zh and "Python" in zh and "合作完成平台迁移" in zh


def test_institution_names_are_kept():
    r = Redactor(detector=HeuristicDetector())
    out = r.clean("B.S. Engineering, Cortek Institute of Technology; M.S., University of Vexdyn.").text
    assert "Cortek Institute of Technology" in out and "University of Vexdyn" in out
    assert "Rilo Venmar" not in r.clean("Studied with Rilo Venmar at the lab.").text


def test_context_window_does_not_cut_words():
    """Regression: a window starting inside "the" produced "he", a fake person cue."""
    r = Redactor(detector=HeuristicDetector())
    pad = "Maintained the core services with image recognition, model deployment."
    out = r.clean(f"Experience\n{pad}\nStaff Machine Learning Engineer\nCortek Labs | Apr 2023 - Jan 2024",
                  kind="resume").text
    assert "Staff Machine Learning Engineer" in out
