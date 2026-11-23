"""Scorers over each query's job pool. Query = the as-of resume, document = the job view (text.py).

    bm25          bilingual BM25 over the whole resume
    dense         one vector per resume (the baseline the chunked scorer is compared with)
    dense_chunks  one vector per resume chunk; aggregated by max, mean of top-k, or recency-weighted max
    requirements  requirement-level coverage (requirements.py); lexical, or lexical + semantic evidence
    hybrid*       reciprocal rank fusion of BM25 with a dense scorer

Everything runs locally unless a remote encoder is passed explicitly (encoders.py).
"""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

from .chunks import resume_chunks
from .encoders import DEFAULT_REQUIREMENT_MODEL, DEFAULT_RETRIEVAL_MODEL, Encoder, STEncoder, embed
from .inputs import load_inputs, read_jsonl
from .text import job_text, resume_text, tokenize

AGGREGATIONS = ("max", "topk", "recency")


def _pools(build_dir: Path) -> dict[str, list[str]]:
    return {p["query_id"]: p["job_ids"] for p in read_jsonl(Path(build_dir) / "eval" / "pools.jsonl")}


def _encoder(encoder, model, **kw):
    if encoder is not None:
        return encoder
    return STEncoder(model, **{k: v for k, v in kw.items() if k in ("max_len", "batch_size")})


# ---------------------------------------------------------------- BM25
class BM25:
    def __init__(self, docs: dict[str, list[str]], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = {d: Counter(toks) for d, toks in docs.items()}
        self.len = {d: len(toks) for d, toks in docs.items()}
        self.avg = sum(self.len.values()) / max(len(docs), 1)
        df = Counter(t for c in self.tf.values() for t in c)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query: list[str], doc_ids) -> dict[str, float]:
        q = {t: 1 + math.log(c) for t, c in Counter(query).items() if t in self.idf}
        out = {}
        for d in doc_ids:
            tf, L = self.tf.get(d), self.len.get(d, 0)
            if tf is None:
                continue
            norm = self.k1 * (1 - self.b + self.b * L / self.avg)
            out[d] = sum(w * self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t, w in q.items() if t in tf)
        return out


def bm25_scores(build_dir: Path, include_notes: bool = False, **_) -> dict[str, dict[str, float]]:
    inp = load_inputs(build_dir)
    index = BM25({j: tokenize(job_text(v, include_notes)) for j, v in inp.jobs.items()})
    pools = _pools(build_dir)
    return {q["query_id"]: index.score(tokenize(resume_text(inp.resumes[q["query_id"]])), pools[q["query_id"]])
            for q in inp.queries}


# ---------------------------------------------------------------- dense, one vector per resume
def dense_scores(build_dir: Path, encoder: Encoder | None = None, include_notes: bool = False,
                 cache_root: Path | None = None, model: str = DEFAULT_RETRIEVAL_MODEL, **kw) -> dict[str, dict[str, float]]:
    import numpy as np
    inp = load_inputs(build_dir)
    enc = _encoder(encoder, model, **kw)
    jv = embed({j: job_text(v, include_notes) for j, v in inp.jobs.items()}, enc, "passage", cache_root=cache_root)
    qv = embed({q["query_id"]: resume_text(inp.resumes[q["query_id"]]) for q in inp.queries}, enc, "query",
               cache_root=cache_root)
    pools = _pools(build_dir)
    return {q: {j: float(np.dot(qv[q], jv[j])) for j in pools[q] if j in jv} for q in qv}


# ---------------------------------------------------------------- dense, one vector per chunk
def dense_chunks_scores(build_dir: Path, encoder: Encoder | None = None, agg: str = "topk", k: int = 3,
                        decay: float = 0.9, include_notes: bool = False, cache_root: Path | None = None,
                        model: str = DEFAULT_RETRIEVAL_MODEL, **kw) -> dict[str, dict[str, float]]:
    """agg: max = best chunk; topk = mean of the k best chunks (rewards breadth);
    recency = max over chunks of similarity x decay^rank for experience entries (recent work counts more)."""
    import numpy as np
    from datetime import date
    if agg not in AGGREGATIONS:
        raise ValueError(f"agg must be one of {AGGREGATIONS}")
    inp = load_inputs(build_dir)
    enc = _encoder(encoder, model, **kw)
    chunks = {q: resume_chunks(r["text"], date.fromisoformat(r["as_of"])) or
              [type("C", (), {"text": resume_text(r), "kind": "other", "rank": None})()]
              for q, r in inp.resumes.items()}
    jv = embed({j: job_text(v, include_notes) for j, v in inp.jobs.items()}, enc, "passage", cache_root=cache_root)
    cv = embed({f"{q}#{i}": c.text for q, cs in chunks.items() for i, c in enumerate(cs)}, enc, "query",
               cache_root=cache_root)
    pools = _pools(build_dir)
    out = {}
    for q, cs in chunks.items():
        C = np.stack([cv[f"{q}#{i}"] for i in range(len(cs))])
        w = np.array([decay ** c.rank if (agg == "recency" and c.rank is not None) else 1.0 for c in cs])
        jobs = [j for j in pools[q] if j in jv]
        if not jobs:
            out[q] = {}
            continue
        S = np.stack([jv[j] for j in jobs]) @ C.T                       # jobs x chunks
        if agg == "max":
            s = S.max(axis=1)
        elif agg == "topk":
            s = np.sort(S, axis=1)[:, ::-1][:, :k].mean(axis=1)
        else:
            s = (S * w).max(axis=1)
        out[q] = dict(zip(jobs, map(float, s)))
    return out


# ---------------------------------------------------------------- requirement-level coverage
def requirements_scores(build_dir: Path, encoder: Encoder | None = None, semantic: bool = False,
                        cache_root: Path | None = None, model: str = DEFAULT_REQUIREMENT_MODEL,
                        return_matches: bool = False, **kw):
    from .requirements import build_semantic, chunks_for_queries, job_requirements, match_job
    inp = load_inputs(build_dir)
    pools = _pools(build_dir)
    reqs = {j: job_requirements(v) for j, v in inp.jobs.items()}
    chunks = chunks_for_queries(inp.resumes)
    sem = build_semantic(reqs, chunks, pools, _encoder(encoder, model, **kw), cache_root) if semantic else {}
    scores, matches = {}, {}
    for q, pool in pools.items():
        scores[q] = {}
        for j in pool:
            if j not in reqs:
                continue
            jm = match_job(reqs[j], chunks.get(q, []), sem.get((q, j)))
            scores[q][j] = jm.score
            if return_matches:
                matches[(q, j)] = jm
    return (scores, matches, chunks) if return_matches else scores


# ---------------------------------------------------------------- fusion
def rrf(*runs: dict[str, dict[str, float]], k: int = 60) -> dict[str, dict[str, float]]:
    """Reciprocal rank fusion: sum over runs of 1 / (k + rank), rank starting at 1."""
    out: dict[str, dict[str, float]] = {}
    for run in runs:
        for q, scores in run.items():
            acc = out.setdefault(q, {})
            for r, j in enumerate(sorted(scores, key=lambda x: (-scores[x], x)), 1):
                acc[j] = acc.get(j, 0.0) + 1 / (k + r)
    return out


def hybrid_scores(build_dir: Path, encoder: Encoder | None = None, **kw) -> dict[str, dict[str, float]]:
    return rrf(bm25_scores(build_dir, **kw), dense_scores(build_dir, encoder=encoder, **kw))


def hybrid_chunks_scores(build_dir: Path, encoder: Encoder | None = None, **kw) -> dict[str, dict[str, float]]:
    return rrf(bm25_scores(build_dir, **kw), dense_chunks_scores(build_dir, encoder=encoder, **kw))
