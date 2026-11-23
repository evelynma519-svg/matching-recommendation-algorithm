"""Seeded synthetic snapshot generator.

Everything is invented. Names and organizations are random syllables (or random combinations of
common Chinese characters), contacts use reserved test ranges (example.test, 555-01xx numbers).

What makes it useful for testing a matcher:

- **Hidden fit.** Outcomes come from a hidden score: skill overlap, role family, seniority gap, an
  organization taste per role family, and a latent candidate-organization affinity that never
  appears in any text. Text models can learn part of the signal, not all of it.
- **Vocabulary mismatch.** Job ads use canonical skill names; resumes use aliases, paraphrases and
  Chinese translations ("k8s", "container orchestration", "深度学习"), and mention only part of a
  candidate's skills. Exact keyword matching misses some true matches.
- **Bilingual resumes.** About 30% are Chinese or mixed: Chinese names, headings and date formats.
- **De-identification stress.** Headers with contacts, colleagues in prose (some with given names that
  are also technical terms, like Julia or Ruby), publications with co-authors, reference sections.
- **Leakage traps.** About a third of resumes contain a job that started after the submission.

    python -m matchrec.synthetic.generate --out data/synthetic --candidates 300 --jobs 120 --seed 7
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from ..schema import (STAFFING, Application, Candidate, Compensation, EndReason, Identity, Job, JobNote,
                      ResumeVersion, Section, Snapshot, StageEvent, StageStatus, write_snapshot)
from .vocab import (CITIES, DUAL_USE_GIVEN, FAMILIES, LEVELS, OBJECTS, ORG_SUFFIX, ORG_SYL, RESPONSIBILITIES,
                    SURFACE, SYL, TITLES, VERBS, ZH_GIVEN, ZH_LEVELS, ZH_SKILL, ZH_SURNAMES, ZH_TITLES, ZH_VERBS)

ALL_SKILLS = sorted({s for v in FAMILIES.values() for s in v})


def _word(rng: random.Random, n: int, pool=SYL) -> str:
    return "".join(rng.choice(pool) for _ in range(n)).capitalize()


def _person(rng, dual_use_share: float = 0.0) -> str:
    given = rng.choice(DUAL_USE_GIVEN) if rng.random() < dual_use_share else _word(rng, 2)
    return f"{given} {_word(rng, 3)}"


def _zh_person(rng) -> str:
    return rng.choice(ZH_SURNAMES) + "".join(rng.choice(ZH_GIVEN) for _ in range(rng.choice([1, 2])))


@dataclass
class _Hidden:
    family: str
    level: int
    skills: set[str]
    affinity: tuple[float, float] = (0.0, 0.0)
    zh: bool = False
    third_parties: list[str] = field(default_factory=list)


def _surface(rng, skill: str, zh: bool) -> str:
    if zh and skill in ZH_SKILL and rng.random() < 0.6:
        return ZH_SKILL[skill]
    forms = SURFACE[skill]
    return forms[0] if rng.random() < 0.5 else rng.choice(forms)


def _mention(rng, hid: _Hidden, k: int) -> list[str]:
    return [_surface(rng, s, hid.zh) for s in rng.sample(sorted(hid.skills), k=min(k, len(hid.skills)))]


def _en_resume(rng, hid, ident, orgs, start, as_of_hint) -> tuple[str, list[Section]]:
    header = f"{ident.full_name}\n{ident.emails[0]} | {ident.phones[0]} | {rng.choice(CITIES)}"
    if rng.random() < 0.4:
        header += f"\nlinkedin.com/in/{ident.given_name.lower()}{ident.family_name.lower()}"
    lines, cursor = [], start
    for k in range(rng.randint(2, 4)):
        end = cursor + timedelta(days=rng.randint(400, 1400))
        body = f"{rng.choice(VERBS)} {rng.choice(OBJECTS)} with {', '.join(_mention(rng, hid, rng.randint(2, 4)))}."
        if rng.random() < 0.3:
            p = _person(rng, dual_use_share=0.15)
            hid.third_parties.append(p)
            body += f" Worked with {p} on the platform migration."
        lines.append(f"{LEVELS[min(k, hid.level)]}{TITLES[hid.family]}\n{rng.choice(orgs)} | {cursor:%b %Y} - {end:%b %Y}\n{body}")
        cursor = end + timedelta(days=rng.randint(0, 90))
    if rng.random() < 0.35:                 # captured after a later move; the as-of cut must remove it
        lines.append(f"{LEVELS[hid.level]}{TITLES[hid.family]}\n{rng.choice(orgs)} | "
                     f"{as_of_hint + timedelta(days=60):%b %Y} - Present\nJoined after the submission.")
    secs = [Section("other", header), Section("experience", "\n".join(lines))]
    if rng.random() < 0.7:                  # many resumes have no skills section at all
        secs.append(Section("skills", ", ".join(_mention(rng, hid, len(hid.skills)))))
    secs.append(Section("education", f"B.S. Engineering, {_word(rng, 2, ORG_SYL)} Institute of Technology"))
    if rng.random() < 0.25:
        pubs = []
        for _ in range(rng.randint(1, 3)):
            co = [f"{_word(rng, 3)}, {_word(rng, 1)[0]}." for _ in range(rng.randint(1, 3))]
            hid.third_parties += [c.split(",")[0] for c in co]
            authors = [f"{ident.family_name}, {ident.given_name[0]}."] + co
            pubs.append(f"{', '.join(authors[:-1])} and {authors[-1]} \"{rng.choice(['Scalable', 'Robust', 'Efficient'])} "
                        f"{rng.choice(['stream joins', 'model serving', 'sensor fusion'])}.\" "
                        f"Proc. {rng.choice(['IEEE', 'ACM'])} Conference {rng.randint(2015, 2023)}.")
        secs.append(Section("publications", "\n".join(pubs)))
    if rng.random() < 0.2:
        refs = [_person(rng), _person(rng)]
        hid.third_parties += refs
        secs.append(Section("references", f"{refs[0]}, former manager\n{refs[1]}, colleague"))
    heading = {"experience": "Experience", "skills": "Skills", "education": "Education",
               "publications": "Publications", "references": "References"}
    return "\n\n".join(s.text if s.kind == "other" else f"{heading[s.kind]}\n{s.text}" for s in secs), secs


def _zh_resume(rng, hid, ident, orgs, start, as_of_hint) -> tuple[str, list[Section]]:
    header = f"{ident.full_name}\n邮箱：{ident.emails[0]} | 电话：{ident.phones[0]} | {rng.choice(CITIES)}"
    lines, cursor = [], start
    for k in range(rng.randint(2, 4)):
        end = cursor + timedelta(days=rng.randint(400, 1400))
        body = f"{rng.choice(ZH_VERBS)}核心系统的开发，使用 {'、'.join(_mention(rng, hid, rng.randint(2, 4)))}。"
        if rng.random() < 0.3:
            p = _zh_person(rng)
            hid.third_parties.append(p)
            body += f"与{p}合作完成平台迁移。"
        lines.append(f"{ZH_LEVELS[min(k, hid.level)]}{ZH_TITLES[hid.family]}\n{rng.choice(orgs)} | "
                     f"{cursor.year}.{cursor.month:02d} - {end.year}.{end.month:02d}\n{body}")
        cursor = end + timedelta(days=rng.randint(0, 90))
    if rng.random() < 0.35:
        t = as_of_hint + timedelta(days=60)
        lines.append(f"{ZH_LEVELS[hid.level]}{ZH_TITLES[hid.family]}\n{rng.choice(orgs)} | {t.year}年{t.month}月 - 至今\n"
                     f"投递之后入职。")
    secs = [Section("other", header), Section("experience", "\n".join(lines))]
    if rng.random() < 0.7:
        secs.append(Section("skills", "、".join(_mention(rng, hid, len(hid.skills)))))
    secs.append(Section("education", f"{_word(rng, 2, ORG_SYL)} Institute of Technology，工学学士"))
    heading = {"experience": "工作经历", "skills": "专业技能", "education": "教育背景"}
    return "\n\n".join(s.text if s.kind == "other" else f"{heading[s.kind]}\n{s.text}" for s in secs), secs


def _candidate(rng, i, orgs, start, span):
    cid = f"c{i:05d}"
    fam = rng.choice(list(FAMILIES))
    hid = _Hidden(fam, rng.randint(0, 2), set(rng.sample(FAMILIES[fam], k=min(5, len(FAMILIES[fam]))))
                  | set(rng.sample(ALL_SKILLS, k=2)), (rng.gauss(0, 1), rng.gauss(0, 1)), rng.random() < 0.3)
    phone = f"+1-{rng.choice([212, 415, 617, 206])}-555-01{rng.randint(0, 99):02d}"
    if hid.zh:
        family, given = rng.choice(ZH_SURNAMES), "".join(rng.choice(ZH_GIVEN) for _ in range(rng.choice([1, 2])))
        ident = Identity(family + given, given, family, [], [f"{_word(rng, 2).lower()}{rng.randint(10, 99)}@example.test"], [phone])
    else:
        given, family = _word(rng, 2), _word(rng, 3)
        ident = Identity(f"{given} {family}", given, family, [], [f"{given.lower()}.{family.lower()}@example.test"], [phone])
    first_sub = start + timedelta(days=rng.randint(0, span))
    career_start = start - timedelta(days=rng.randint(1500, 4000))
    text, secs = (_zh_resume if hid.zh else _en_resume)(rng, hid, ident, orgs, career_start, first_sub)
    res = ResumeVersion(f"{cid}-r1", None, text, secs, "zh" if hid.zh else "en")
    return Candidate(cid, [res], sorted(hid.skills), ident), hid, first_sub


def generate(n_candidates: int = 300, n_jobs: int = 120, seed: int = 7,
             start: date = date(2024, 1, 1), months: int = 18):
    rng = random.Random(seed)
    org_names = sorted({f"{_word(rng, 2, ORG_SYL)} {rng.choice(ORG_SUFFIX)}" for _ in range(max(8, n_jobs // 6))})
    taste = {o: {f: rng.gauss(0, 0.6) for f in FAMILIES} for o in org_names}           # visible-ish: by role family
    affinity = {o: (rng.gauss(0, 1), rng.gauss(0, 1)) for o in org_names}              # invisible in any text
    span = months * 30
    ref = start + timedelta(days=span + 45)

    jobs, jhid = [], {}
    for i in range(n_jobs):
        fam, lvl, org = rng.choice(list(FAMILIES)), rng.randint(0, 2), rng.choice(org_names)
        req = set(rng.sample(FAMILIES[fam], k=4))
        extra = rng.choice(sorted(set(FAMILIES[fam]) - req) or ALL_SKILLS)
        opened = start + timedelta(days=rng.randint(0, span - 60))
        jid, owner = f"j{i:04d}", _person(rng)
        jobs.append(Job(
            job_id=jid, org_id="o" + str(org_names.index(org)), org_name=org,
            title=LEVELS[lvl] + TITLES[fam], employment_type="full_time", locations=[rng.choice(CITIES)],
            compensation=Compensation("yearly", 90000 + 30000 * lvl, 130000 + 40000 * lvl, "USD"),
            summary=f"{org} is hiring a {LEVELS[lvl]}{TITLES[fam]}.",
            responsibilities=RESPONSIBILITIES[fam],
            requirements=f"Experience with {', '.join(SURFACE[s][0] for s in sorted(req))}; {2 + 3 * lvl}+ years. "
                         f"Nice to have: {SURFACE[extra][0]}.",
            required_skills=sorted(req), opened_at=opened.isoformat(),
            closed_at=(opened + timedelta(days=rng.randint(45, 200))).isoformat(),
            notes=[JobNote(f"{jid}-n1", f"Hiring manager prefers hands-on builders; check with {owner} before submitting.")],
            related_people=[owner]))
        jhid[jid] = _Hidden(fam, lvl, req)

    cands, hidden, apps = [], {}, []
    for i in range(n_candidates):
        cand, hid, first_sub = _candidate(rng, i, org_names, start, span)
        cands.append(cand)
        hidden[cand.candidate_id] = hid
        for ep in range(rng.choice([1, 1, 2])):        # one or two submission episodes
            t = first_sub + timedelta(days=ep * rng.randint(90, 200))
            if t > start + timedelta(days=span):
                break
            open_jobs = [j for j in jobs if j.opened_at <= t.isoformat() <= j.closed_at]
            if not open_jobs:
                continue
            # recruiters mostly submit to plausible roles: weight same-family openings up
            weights = [4.0 if jhid[j.job_id].family == hid.family else 1.0 for j in open_jobs]
            picked = {j.job_id: j for j in rng.choices(open_jobs, weights, k=rng.randint(1, 4))}
            for j in picked.values():
                aff = sum(x * y for x, y in zip(hid.affinity, affinity[j.org_name]))
                apps.append(_application(rng, f"a{len(apps):06d}", cand.candidate_id, j, jhid[j.job_id], hid,
                                         taste[j.org_name][hid.family], aff, t + timedelta(days=rng.randint(0, 10))))
    snap = Snapshot(f"synthetic-seed{seed}", ref.isoformat(), "synthetic", STAFFING.name)
    return snap, cands, jobs, apps, hidden


def _application(rng, aid, cid, job: Job, jh: _Hidden, ch: _Hidden, taste: float, affinity: float, t: date) -> Application:
    fit = (len(jh.skills & ch.skills) / len(jh.skills) + (0.3 if jh.family == ch.family else -0.5)
           - 0.25 * abs(jh.level - ch.level) + 0.3 * taste + 0.25 * affinity + rng.gauss(0, 0.25))
    order = list(STAFFING.stages)
    if rng.random() < 0.02:                     # bookkeeping noise that labels must exclude
        return Application(aid, cid, job.job_id, t.isoformat(), [StageEvent(order[0], StageStatus.DROPPED, t.isoformat())],
                           EndReason.DUPLICATE, "", f"{cid}-r1", t.isoformat())
    depth = 0                                   # index of the furthest stage reached
    for thr in (0.2, 0.55, 0.8, 0.95, 1.05):
        if fit > thr + rng.gauss(0, 0.15):
            depth += 1
        else:
            break
    hired = depth == len(order) - 1
    events = []
    for k in range(depth + 1):
        last = k == depth
        status = StageStatus.IN_PROGRESS if (hired and last) else (StageStatus.DROPPED if last else StageStatus.PASSED)
        events.append(StageEvent(order[k], status, (t + timedelta(days=7 * k)).isoformat()))
    reason = None
    if not hired:
        reason = (EndReason.INTERNAL_SCREEN_OUT if depth == 0 else
                  rng.choice([EndReason.EMPLOYER_DECLINED, EndReason.EMPLOYER_DECLINED, EndReason.CANDIDATE_WITHDREW,
                              EndReason.POSITION_CLOSED]))
    return Application(aid, cid, job.job_id, t.isoformat(), events, reason, "", f"{cid}-r1", events[-1].at)


class _Mem:
    def __init__(self, snap, cands, jobs, apps, hidden=None):
        self._s, self._c, self._j, self._a, self.hidden = snap, cands, jobs, apps, hidden or {}

    def snapshot(self): return self._s
    def candidates(self): return iter(self._c)
    def jobs(self): return iter(self._j)
    def applications(self): return iter(self._a)

    def third_party_names(self) -> set[str]:
        """Ground truth for leak tests: every non-candidate person name written into a resume."""
        return {p for h in self.hidden.values() for p in h.third_parties}


def in_memory(**kw) -> _Mem:
    return _Mem(*generate(**kw))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--candidates", type=int, default=300)
    ap.add_argument("--jobs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    counts = write_snapshot(a.out, in_memory(n_candidates=a.candidates, n_jobs=a.jobs, seed=a.seed))
    print(f"wrote {a.out}: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
