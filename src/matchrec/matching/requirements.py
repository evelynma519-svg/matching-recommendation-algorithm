"""Requirement-level matching: score a job by how many of its requirements the resume gives evidence for.

    job      -> requirement units: required skills, other must-have clauses, nice-to-haves, years
    resume   -> chunks (see chunks.py)
    evidence -> matrix requirement x chunk:
                  lexical   normalized skill present in the chunk, or token overlap for free-text clauses
                  semantic  cosine(requirement, chunk), optional, corrected for hubness and
                            rescaled on this build's own distribution (no labels involved)
                  evidence = max(lexical, semantic)
    score    -> weighted coverage: must 1.0, nice-to-have 0.4, years 0.6

Every score keeps, per requirement, the best chunk and how it was matched. `explain` turns that into a
plain-language reason for the recommendation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from .chunks import Chunk, experience_years, resume_chunks
from .skills import find_skills
from .text import tokenize

WEIGHTS = {"must": 1.0, "nice": 0.4, "years": 0.6}
_NICE = re.compile(r"(?i)\b(?:nice to have|preferred|bonus|a plus|plus)\b\s*[:：]?|加分项|优先")
_YEARS = re.compile(r"(?i)(\d{1,2})\s*\+?\s*(?:years?|yrs?|年)")
_CLAUSE = re.compile(r"[;\n•●▪]|(?<=[a-z0-9)])\.\s+|。|；")


@dataclass
class Requirement:
    text: str
    kind: str                      # must | nice | years
    skill: str | None = None       # canonical skill, when the unit is a single skill
    years: int | None = None


@dataclass
class Match:
    requirement: Requirement
    score: float
    chunk: int | None = None
    via: str | None = None         # lexical | semantic | years


@dataclass
class JobMatch:
    score: float
    matches: list[Match] = field(default_factory=list)
    candidate_years: float | None = None


def job_requirements(job: dict) -> list[Requirement]:
    text = job.get("requirements") or ""
    m = _NICE.search(text)
    segments = [(text[:m.start()], "must"), (text[m.end():], "nice")] if m else [(text, "must")]
    units: list[Requirement] = []
    seen: set[str] = set()

    def add_skill(s, kind):
        if s not in seen:
            seen.add(s)
            units.append(Requirement(s, kind, skill=s))

    for s in job.get("required_skills") or []:
        add_skill(s.lower(), "must")
    for seg, kind in segments:
        for clause in filter(None, (c.strip(" ,.:") for c in _CLAUSE.split(seg))):
            if (y := _YEARS.search(clause)):
                units.append(Requirement(clause, "years", years=int(y.group(1))))
                clause = _YEARS.sub(" ", clause).strip(" ,.;+")
            skills = find_skills(clause)
            if skills:
                for s in sorted(skills):
                    add_skill(s, kind)
            elif len(tokenize(clause)) >= 2:
                units.append(Requirement(clause, kind))
    return units


def _lexical(req: Requirement, chunk_skills: set[str], chunk_tokens: set[str]) -> float:
    if req.skill:
        return 1.0 if req.skill in chunk_skills else 0.0
    toks = set(tokenize(req.text))
    return len(toks & chunk_tokens) / len(toks) if toks else 0.0


def match_job(reqs: list[Requirement], chunks: list[Chunk], sem=None) -> JobMatch:
    """sem: optional (requirement indices, matrix) from build_semantic; matrix[a, ci] is the rescaled
    semantic evidence in [0, 1] of requirement indices[a] in chunk ci."""
    row = {ri: a for a, ri in enumerate(sem[0])} if sem is not None else {}
    c_skills = [find_skills(c.text) for c in chunks]
    c_tokens = [set(tokenize(c.text)) for c in chunks]
    years = experience_years(chunks)
    out, num, den = [], 0.0, 0.0
    for ri, r in enumerate(reqs):
        if r.kind == "years":
            s = min(1.0, years / r.years) if (years is not None and r.years) else 0.0
            out.append(Match(r, s, None, "years"))
        else:
            best = Match(r, 0.0)
            for ci in range(len(chunks)):
                lex = _lexical(r, c_skills[ci], c_tokens[ci])
                se = float(sem[1][row[ri], ci]) if ri in row else 0.0
                s, via = (lex, "lexical") if lex >= se else (se, "semantic")
                if s > best.score:
                    best = Match(r, s, ci, via)
            out.append(best)
        num += WEIGHTS[r.kind] * out[-1].score
        den += WEIGHTS[r.kind]
    return JobMatch(num / den if den else 0.0, out, years)


def explain(jm: JobMatch, chunks: list[Chunk], threshold: float = 0.5, max_items: int = 4) -> str:
    """Plain-language reason, e.g. 'Meets 3 of 4 required skills (Kafka, ...); missing: gRPC. ...'"""
    must = [m for m in jm.matches if m.requirement.kind == "must"]
    nice = [m for m in jm.matches if m.requirement.kind == "nice"]
    met = [m for m in must if m.score >= threshold]
    miss = [m for m in must if m.score < threshold]
    name = lambda m: m.requirement.skill or m.requirement.text[:40]
    parts = [f"Meets {len(met)} of {len(must)} required items" +
             (f" ({', '.join(name(m) for m in met[:max_items])})" if met else "")]
    if miss:
        parts.append("missing: " + ", ".join(name(m) for m in miss[:max_items]))
    nice_met = [m for m in nice if m.score >= threshold]
    if nice_met:
        parts.append("also has " + ", ".join(name(m) for m in nice_met[:max_items]))
    yr = next((m for m in jm.matches if m.requirement.kind == "years"), None)
    if yr and jm.candidate_years is not None:
        parts.append(f"about {jm.candidate_years:g} years of experience vs {yr.requirement.years}+ asked")
    sem = [m for m in met if m.via == "semantic" and m.chunk is not None]
    if sem:
        first = chunks[sem[0].chunk].text.splitlines()[0][:60]
        parts.append(f"evidence for {name(sem[0])} found in \"{first}\"")
    return "; ".join(parts) + "."


def build_semantic(reqs_by_job: dict[str, list[Requirement]], chunks_by_q: dict[str, list[Chunk]],
                   pools: dict[str, list[str]], encoder, cache_root=None, lo_q: float = 0.9, hi_q: float = 0.995):
    """Per (query, job): (requirement indices, evidence matrix in [0, 1]).

    Raw cosine similarities are not usable as evidence directly. Some chunks (short, generic ones such
    as an education line) are similar to almost every requirement, and some requirements to almost
    every chunk: the hubness problem. Following CSLS-style local scaling, each similarity is corrected
    by subtracting the chunk's mean similarity to all requirements it is compared with and the
    requirement's mean similarity to all chunks it is compared with. The corrected values are then
    rescaled on the build's own distribution: below the 90th percentile -> 0, the 99.5th -> 1. No labels
    are involved."""
    import numpy as np

    from .encoders import embed
    rtext = {f"{j}#{i}": r.text for j, rs in reqs_by_job.items() for i, r in enumerate(rs) if r.kind != "years"}
    ctext = {f"{q}#{i}": c.text for q, cs in chunks_by_q.items() for i, c in enumerate(cs)}
    rv = embed(rtext, encoder, role="query", task="requirement", cache_root=cache_root)
    cv = embed(ctext, encoder, role="passage", task="requirement", cache_root=cache_root)
    sims = {}
    c_sum, c_n, r_sum, r_n = {}, {}, {}, {}
    for q, pool in pools.items():
        if not chunks_by_q.get(q):
            continue
        C = np.stack([cv[f"{q}#{i}"] for i in range(len(chunks_by_q[q]))])
        for j in pool:
            idx = [i for i, r in enumerate(reqs_by_job.get(j, [])) if r.kind != "years"]
            if not idx:
                continue
            M = np.stack([rv[f"{j}#{i}"] for i in idx]) @ C.T
            sims[(q, j)] = (idx, M)
            c_sum[q] = c_sum.get(q, 0) + M.sum(axis=0)
            c_n[q] = c_n.get(q, 0) + M.shape[0]
            r_sum[j] = r_sum.get(j, 0) + M.sum(axis=1)
            r_n[j] = r_n.get(j, 0) + M.shape[1]
    if not sims:
        return {}
    glob = float(np.mean([m.mean() for _, m in sims.values()]))
    adj = {k: (idx, m - (r_sum[k[1]] / r_n[k[1]])[:, None] - (c_sum[k[0]] / c_n[k[0]])[None, :] + glob)
           for k, (idx, m) in sims.items()}
    flat = np.concatenate([m.ravel() for _, m in adj.values()])
    lo, hi = np.quantile(flat, lo_q), np.quantile(flat, hi_q)
    return {k: (idx, np.clip((m - lo) / max(hi - lo, 1e-6), 0, 1)) for k, (idx, m) in adj.items()}


def chunks_for_queries(resumes: dict[str, dict]) -> dict[str, list[Chunk]]:
    return {q: resume_chunks(r["text"], date.fromisoformat(r["as_of"])) for q, r in resumes.items()}
