"""End to end on synthetic data: snapshot -> de-identification -> firewall -> scorers -> evaluation -> comparison."""
import json
from datetime import date

import pytest

from matchrec.matching import compare, evaluate, retrieval
from matchrec.matching.asof import choose_version, cut_text
from matchrec.matching.firewall import build
from matchrec.matching.inputs import FORBIDDEN_KEYS, assert_clean, load_inputs, read_jsonl, train_labels
from matchrec.matching.split import fold_of, role_of
from matchrec.deid.detectors import HeuristicDetector
from matchrec.deid.pipeline import deidentify
from matchrec.deid.redactor import Redactor
from matchrec.schema import Candidate, JsonlSnapshot, ResumeVersion, write_snapshot
from matchrec.synthetic.generate import in_memory


@pytest.fixture(scope="module")
def builds(tmp_path_factory):
    root = tmp_path_factory.mktemp("mr")
    write_snapshot(root / "raw", in_memory(n_candidates=400, n_jobs=120, seed=11))
    deidentify(JsonlSnapshot(root / "raw"), root / "snap", lambda n, p: Redactor(n, HeuristicDetector(), p))
    return {r: build(root / "snap", root / "out", role=r) for r in ("dev", "test")}


def test_dev_and_test_candidates_are_disjoint(builds):
    dev = {q["candidate_id"] for q in read_jsonl(builds["dev"] / "inputs" / "queries.jsonl")}
    test = {q["candidate_id"] for q in read_jsonl(builds["test"] / "inputs" / "queries.jsonl")}
    assert dev and test and not dev & test
    assert all(role_of(c) == "test" for c in test)


def test_test_build_exposes_no_labels_to_models(builds):
    assert (builds["test"] / "sealed" / "qrels.jsonl").exists()
    assert not (builds["test"] / "labels").exists()
    with pytest.raises(PermissionError):
        train_labels(builds["test"], holdout_fold=0)


def test_model_inputs_carry_no_outcomes_or_identity(builds):
    for f in ("queries", "resumes", "jobs"):
        for row in read_jsonl(builds["dev"] / "inputs" / f"{f}.jsonl"):
            assert_clean(row)
    raw = (builds["dev"] / "inputs" / "resumes.jsonl").read_text() + (builds["dev"] / "inputs" / "jobs.jsonl").read_text()
    assert "@example.test" not in raw and "555-01" not in raw
    with pytest.raises(AssertionError):
        assert_clean({"job": {"end_reason": "x"}})


def test_asof_cut_removes_post_submission_entries(builds):
    meta = json.loads((builds["dev"] / "build.json").read_text())
    assert meta["asof_cut"]["dropped_entries"] > 0
    # Every experience entry still visible must start before the query's as-of month.
    from matchrec.matching.asof import RANGE, parse_token, starts_after
    for r in read_jsonl(builds["dev"] / "inputs" / "resumes.jsonl"):
        as_of = date.fromisoformat(r["as_of"])
        for m in RANGE.finditer(r["text"]):
            assert not starts_after(parse_token(m.group(1)), as_of)


def test_pools_contain_every_relevant_job(builds):
    pools = {p["query_id"]: set(p["job_ids"]) for p in read_jsonl(builds["dev"] / "eval" / "pools.jsonl")}
    for q in read_jsonl(builds["dev"] / "labels" / "qrels.jsonl"):
        assert q["job_id"] in pools[q["query_id"]]
    meta = json.loads((builds["dev"] / "build.json").read_text())
    assert meta["shares"]["pool_method"] == {"posting_dates": 120}


def test_train_labels_hold_out_a_fold(builds):
    rows = train_labels(builds["dev"], holdout_fold=2)
    assert rows and all(r["fold"] != 2 for r in rows) and not any(r["excluded"] for r in rows)


def test_bm25_beats_random_and_compare_runs(builds):
    b = builds["dev"]
    reps = {}
    for name, scores in (("random", evaluate.random_scores(b)), ("bm25", retrieval.bm25_scores(b)),
                         ("popularity", evaluate.popularity_scores(b))):
        rep = evaluate.evaluate(b, scores, n_boot=200, return_per_query=True)
        pq = rep.pop("per_query")
        d = b / "eval" / "reports"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.json").write_text(json.dumps(rep))
        (d / f"{name}.perquery.json").write_text(json.dumps({"fingerprint": rep["fingerprint"], "per_query": pq}))
        reps[name] = rep
    assert reps["bm25"]["metrics"]["ndcg@10"]["mean"] > reps["random"]["metrics"]["ndcg@10"]["mean"]
    res = compare.compare(b, ref="random", n_boot=300)
    assert res["paired"]["bm25"]["ndcg@10"]["mean"] > 0
    assert "Paired difference" in compare.to_markdown(res)


class HashEncoder:
    """Stand-in for a sentence encoder: hashed bag of tokens, L2-normalised."""
    name = "hash-test"

    def encode(self, texts, role="passage", task="retrieval"):
        np = pytest.importorskip("numpy")
        from matchrec.matching.text import tokenize
        out = np.zeros((len(texts), 256))
        for i, t in enumerate(texts):
            for tok in tokenize(t):
                out[i, hash(tok) % 256] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def test_dense_and_hybrid_paths(builds, tmp_path):
    pytest.importorskip("numpy")
    d = retrieval.dense_scores(builds["dev"], encoder=HashEncoder(), cache_root=tmp_path)
    h = retrieval.rrf(retrieval.bm25_scores(builds["dev"]), d)
    assert d.keys() == h.keys() and all(h.values())


@pytest.mark.parametrize("agg", ["max", "topk", "recency"])
def test_chunked_dense_runs_and_beats_random(builds, tmp_path, agg):
    pytest.importorskip("numpy")
    b = builds["dev"]
    s = retrieval.dense_chunks_scores(b, encoder=HashEncoder(), agg=agg, cache_root=tmp_path)
    r = evaluate.evaluate(b, s, n_boot=50)["metrics"]["ndcg@10"]["mean"]
    assert r > evaluate.evaluate(b, evaluate.random_scores(b), n_boot=50)["metrics"]["ndcg@10"]["mean"]


def test_requirement_scorer_lexical_and_semantic(builds, tmp_path):
    pytest.importorskip("numpy")
    b = builds["dev"]
    rnd = evaluate.evaluate(b, evaluate.random_scores(b), n_boot=50)["metrics"]["ndcg@10"]["mean"]
    lex = retrieval.requirements_scores(b)
    assert evaluate.evaluate(b, lex, n_boot=50)["metrics"]["ndcg@10"]["mean"] > rnd
    sem, matches, chunks = retrieval.requirements_scores(b, encoder=HashEncoder(), semantic=True,
                                                         cache_root=tmp_path, return_matches=True)
    assert sem.keys() == lex.keys()
    assert all(0.0 <= v <= 1.0 for qs in sem.values() for v in qs.values())
    assert any(m.via == "semantic" for jm in matches.values() for m in jm.matches)


def test_minimal_funnel_reports_only_observable_levels(tmp_path):
    snap = tmp_path / "snap"
    write_snapshot(snap, in_memory(n_candidates=150, n_jobs=50, seed=3))
    # Re-grade the same synthetic data as if only "applied" and "hired" were recorded.
    import matchrec.schema.funnel as F
    staffing_to_minimal = F.FunnelSpec("min-view", F.STAFFING.stages,
                                       (F.Level(1, ("submitted",), dropped_counts=True), F.Level(4, ("hired",))),
                                       terminal_stage="hired")
    F.FUNNELS["min-view"] = staffing_to_minimal
    try:
        b = build(snap, tmp_path / "out", funnel="min-view")
    finally:
        del F.FUNNELS["min-view"]
    rep = evaluate.evaluate(b, evaluate.random_scores(b), n_boot=50)
    keys = set(rep["metrics"])
    assert not any("_L2+" in k or "_L3+" in k for k in keys)
    assert any("_L4+" in k for k in keys)


def test_choose_version_prefers_bound_then_dated():
    c = Candidate("c", [ResumeVersion("r1", "2024-01-05", "old"), ResumeVersion("r2", "2024-06-01", "new")])
    assert choose_version(c, {"r2"}, date(2024, 2, 1)) == (c.resumes[1], "bound")
    assert choose_version(c, set(), date(2024, 2, 1)) == (c.resumes[0], "captured_before")
    assert choose_version(c, set(), date(2023, 1, 1)) == (c.resumes[1], "text_cut_only")


def test_cut_text_handles_bilingual_dates():
    text = "Engineer\nAcme | 2019.03 - 2023.05\nwork\n工程师\n某公司 | 2024年7月 - 至今\nlater"
    kept, st = cut_text(text, date(2024, 3, 1))
    assert "later" not in kept and "work" in kept and st["dropped_entries"] == 1


def test_split_is_stable_and_roughly_proportional():
    ids = [f"c{i}" for i in range(4000)]
    share = sum(role_of(i) == "test" for i in ids) / len(ids)
    assert 0.17 < share < 0.23
    assert [fold_of(i) for i in ids[:50]] == [fold_of(i) for i in ids[:50]]
