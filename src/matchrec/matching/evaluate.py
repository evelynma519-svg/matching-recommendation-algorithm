"""Backtest evaluator: graded-relevance metrics with bootstrap confidence intervals.

Scores are {query_id: {job_id: score}} over the query's pool. Jobs without a score rank last.
Metrics per query, then averaged over queries:
  recall@K at each level the funnel can observe   (queries without a positive at a level are skipped)
  ndcg@K with gain 2^grade - 1
  mrr (first L1+ positive)
Jobs nobody was submitted to are unlabelled, not negative, so every number is a lower bound.

Usage: python -m matchrec.matching.evaluate --build <build_dir> --scorer random|popularity|bm25|dense|hybrid
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from datetime import date
from pathlib import Path

from .inputs import load_inputs, read_jsonl as _jsonl, train_labels

KS = (10, 20, 50)
LEVELS = (1, 2, 3, 4)


def _tiebreak(qid: str, jid: str) -> int:
    return int(hashlib.sha256(f"{qid}|{jid}".encode()).hexdigest()[:12], 16)


def rank(qid: str, pool: list[str], scores: dict[str, float]) -> list[str]:
    return sorted(pool, key=lambda j: (-scores.get(j, -math.inf), _tiebreak(qid, j)))


def query_metrics(ranked: list[str], rel: dict[str, int], ks=KS, levels=LEVELS) -> dict[str, float]:
    m: dict[str, float] = {}
    pos = {j: i for i, j in enumerate(ranked)}
    for t in levels:
        R = [j for j, g in rel.items() if g >= t]
        if not R:
            continue
        for k in ks:
            m[f"recall@{k}_L{t}+"] = sum(pos.get(j, 10**9) < k for j in R) / len(R)
    ideal = sorted(rel.values(), reverse=True)
    for k in ks:
        dcg = sum((2 ** rel.get(j, 0) - 1) / math.log2(i + 2) for i, j in enumerate(ranked[:k]))
        idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal[:k]))
        m[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0
    first = min((pos.get(j, 10**9) for j in rel), default=10**9)
    m["mrr"] = 1 / (first + 1) if first < 10**9 else 0.0
    return m


def bootstrap_ci(vals: list[float], n: int = 1000, seed: int = 0) -> tuple[float, float]:
    if not vals:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(vals) for _ in vals) / len(vals) for _ in range(n))
    return means[int(0.025 * n)], means[min(int(0.975 * n), n - 1)]


def fingerprint(build_dir: Path) -> str:
    """Identifies the exact query set, pools and labels; paired comparisons require it to match."""
    build_dir = Path(build_dir)
    lab = build_dir / "labels" if (build_dir / "labels").exists() else build_dir / "sealed"
    h = hashlib.sha256()
    for p in (build_dir / "eval" / "pools.jsonl", lab / "qrels.jsonl"):
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def observable_levels(build_dir: Path) -> tuple[int, ...]:
    meta = Path(build_dir) / "build.json"
    if meta.exists():
        return tuple(json.loads(meta.read_text()).get("observable_grades", LEVELS))
    return LEVELS


def per_query(build_dir: Path, scores: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    build_dir = Path(build_dir)
    levels = observable_levels(build_dir)
    lab = build_dir / "labels" if (build_dir / "labels").exists() else build_dir / "sealed"
    rel: dict[str, dict[str, int]] = defaultdict(dict)
    for r in _jsonl(lab / "qrels.jsonl"):
        rel[r["query_id"]][r["job_id"]] = r["grade"]
    pools = {p["query_id"]: p["job_ids"] for p in _jsonl(build_dir / "eval" / "pools.jsonl")}
    return {qid: query_metrics(rank(qid, pool, scores.get(qid, {})), rel[qid], levels=levels)
            for qid, pool in pools.items()}


def summarize(pq: dict[str, dict[str, float]], n_boot: int = 1000, seed: int = 0) -> dict:
    per: dict[str, list[float]] = defaultdict(list)
    for m in pq.values():
        for k, v in m.items():
            per[k].append(v)
    out = {}
    for k, vals in sorted(per.items()):
        lo, hi = bootstrap_ci(vals, n_boot, seed)
        out[k] = {"mean": round(sum(vals) / len(vals), 4), "ci95": [round(lo, 4), round(hi, 4)], "n_queries": len(vals)}
    return out


def evaluate(build_dir: Path, scores: dict[str, dict[str, float]], n_boot: int = 1000, seed: int = 0,
             return_per_query: bool = False) -> dict:
    build_dir = Path(build_dir)
    pools = {p["query_id"]: p["job_ids"] for p in _jsonl(build_dir / "eval" / "pools.jsonl")}
    pq = per_query(build_dir, scores)
    rep = {"n_queries": len(pools), "avg_pool": round(sum(map(len, pools.values())) / max(len(pools), 1), 1),
           "fingerprint": fingerprint(build_dir), "metrics": summarize(pq, n_boot, seed)}
    if return_per_query:
        rep["per_query"] = pq
    return rep


# ---------------------------------------------------------------- baselines
def random_scores(build_dir: Path, seed: int = 0) -> dict[str, dict[str, float]]:
    pools = _jsonl(Path(build_dir) / "eval" / "pools.jsonl")
    rng = random.Random(seed)
    return {p["query_id"]: {j: rng.random() for j in p["job_ids"]} for p in pools}


def popularity_scores(build_dir: Path) -> dict[str, dict[str, float]]:
    """Job popularity = submissions by candidates in OTHER folds, strictly before the query's as_of."""
    inp = load_inputs(build_dir)
    pools = {p["query_id"]: p["job_ids"] for p in _jsonl(Path(build_dir) / "eval" / "pools.jsonl")}
    by_fold: dict[int, list[dict]] = {}
    out = {}
    for q in inp.queries:
        f = q["fold"]
        if f not in by_fold:
            by_fold[f] = sorted(train_labels(build_dir, holdout_fold=f), key=lambda r: r["submitted_at"])
        as_of = q["as_of"]
        cnt: dict[str, int] = defaultdict(int)
        for r in by_fold[f]:
            if r["submitted_at"][:10] >= as_of:
                break
            cnt[r["job_id"]] += 1
        out[q["query_id"]] = {j: float(cnt.get(j, 0)) for j in pools[q["query_id"]]}
    return out


def _retrieval(name):
    def run(build_dir, **kw):
        from . import retrieval
        return getattr(retrieval, f"{name}_scores")(build_dir, **kw)
    return run


SCORERS = {"random": random_scores, "popularity": popularity_scores,
           "bm25": _retrieval("bm25"), "dense": _retrieval("dense"), "hybrid": _retrieval("hybrid"),
           "dense_chunks": _retrieval("dense_chunks"), "hybrid_chunks": _retrieval("hybrid_chunks"),
           "requirements": _retrieval("requirements")}
ENCODER_SCORERS = {"dense", "hybrid", "dense_chunks", "hybrid_chunks"}


def main(argv=None) -> int:
    from .encoders import DEFAULT_REQUIREMENT_MODEL, DEFAULT_RETRIEVAL_MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--scorer", choices=sorted(SCORERS), required=True)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--model", default=None, help=f"encoder (default {DEFAULT_RETRIEVAL_MODEL}; "
                                                   f"{DEFAULT_REQUIREMENT_MODEL} for --semantic requirements)")
    ap.add_argument("--max-len", type=int, default=None, help="encoder max tokens (default 1024 whole resume, 512 chunks)")
    ap.add_argument("--agg", choices=["max", "topk", "recency"], default="topk", help="chunk aggregation")
    ap.add_argument("--k", type=int, default=3, help="chunks averaged by --agg topk")
    ap.add_argument("--semantic", action="store_true", help="requirements: add semantic evidence")
    ap.add_argument("--explain", type=int, default=0, help="requirements: write reasons for the top N jobs per query")
    ap.add_argument("--tag", default=None, help="report name (default derived from the options)")
    a = ap.parse_args(argv)

    kw, tag = {}, a.scorer
    if a.scorer in ENCODER_SCORERS:
        kw = {"model": a.model or DEFAULT_RETRIEVAL_MODEL, "max_len": a.max_len or (1024 if a.scorer in ("dense", "hybrid") else 512)}
        if "chunks" in a.scorer:
            kw.update(agg=a.agg, k=a.k)
            tag += f"__{a.agg}" + (str(a.k) if a.agg == "topk" else "")
    if a.scorer == "requirements" and a.semantic:
        kw = {"semantic": True, "model": a.model or DEFAULT_REQUIREMENT_MODEL, "max_len": a.max_len or 512}
        tag += "__semantic"
    if a.model and kw.get("model") not in (DEFAULT_RETRIEVAL_MODEL, DEFAULT_REQUIREMENT_MODEL):
        tag += "__" + a.model.replace("/", "__")
    tag = a.tag or tag

    explanations = None
    if a.scorer == "requirements" and a.explain:
        from . import retrieval
        from .requirements import explain
        scores, matches, chunks = retrieval.requirements_scores(a.build, return_matches=True, **kw)
        pools = {p["query_id"]: p["job_ids"] for p in _jsonl(a.build / "eval" / "pools.jsonl")}
        explanations = [{"query_id": q, "job_id": j, "score": round(scores[q][j], 4),
                         "reason": explain(matches[(q, j)], chunks.get(q, []))}
                        for q in sorted(scores) for j in rank(q, pools[q], scores[q])[:a.explain] if j in scores[q]]
    else:
        scores = SCORERS[a.scorer](a.build, **kw)
    rep = evaluate(a.build, scores, n_boot=a.boot, return_per_query=True)
    pq = rep.pop("per_query")
    rep.update({"scorer": a.scorer, "params": kw, "evaluated_on": date.today().isoformat()})
    d = a.build / "eval" / "reports"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{tag}.json").write_text(json.dumps(rep, indent=1))
    (d / f"{tag}.perquery.json").write_text(json.dumps({"fingerprint": rep["fingerprint"], "per_query": pq}))
    if explanations is not None:
        with (d / f"{tag}.explanations.jsonl").open("w", encoding="utf-8") as f:
            for e in explanations:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"{tag}: {rep['n_queries']} queries, avg pool {rep['avg_pool']}")
    for k, v in rep["metrics"].items():
        print(f"  {k:<18} {v['mean']:.3f}  [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]  n={v['n_queries']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
