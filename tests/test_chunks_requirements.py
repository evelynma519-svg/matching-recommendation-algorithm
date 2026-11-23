from datetime import date

import pytest

from matchrec.matching.chunks import experience_years, resume_chunks
from matchrec.matching.encoders import OpenAICompatibleEncoder, family_of, prepare
from matchrec.matching.requirements import explain, job_requirements, match_job
from matchrec.matching.skills import canonical, find_skills

RESUME = """## other
[PERSON]
[EMAIL] | [PHONE]

## experience
Experience
Senior Backend Engineer
Tekvex Labs | Mar 2021 - Present
Scaled the core services with Kafka, k8s and Postgres.
Backend Engineer
Novadyn Cloud | Jan 2017 - Feb 2021
Built RPC services in Golang.

## skills
Skills
Python, Redis, Docker

## references
[REFERENCES_REMOVED]
"""


def test_resume_chunks_split_entries_with_context_and_recency():
    cs = resume_chunks(RESUME, date(2024, 6, 1))
    kinds = [c.kind for c in cs]
    assert kinds.count("experience") == 2 and "skills" in kinds
    assert "other" not in kinds and "references" not in kinds          # placeholders only -> dropped
    recent = next(c for c in cs if c.rank == 0)
    assert recent.text.startswith("Senior Backend Engineer") and "Tekvex Labs" in recent.text
    assert recent.end == (2024, 6)
    assert experience_years(cs) == pytest.approx(7.4, abs=0.2)


def test_chinese_entries_are_split():
    zh = "## experience\n工作经历\n高级后端工程师\nTekvex Labs | 2020.03 - 2023.05\n负责核心系统。\n后端工程师\nNovadyn Cloud | 2017.01 - 2020.02\n参与开发。"
    cs = resume_chunks(zh, date(2024, 1, 1))
    assert len(cs) == 2 and cs[0].text.startswith("高级后端工程师") and cs[0].start == (2020, 3)


def test_skill_normalization():
    assert find_skills("k8s, Postgres and PySpark; 深度学习") == {"kubernetes", "postgresql", "spark", "deep learning"}
    assert "c" not in find_skills("C++ and C#") and find_skills("C++ and C#") == {"c++", "c#"}
    assert canonical("Golang") == "go"
    assert find_skills("embedded Linux and Yocto") == {"embedded linux"}
    assert find_skills("Linux servers; embedded Linux boards") == {"linux", "embedded linux"}


def test_job_requirements_parse_must_nice_and_years():
    job = {"requirements": "Experience with Kafka, Kubernetes, PostgreSQL; 5+ years. Ownership of on-call rotations. "
                           "Nice to have: Redis.", "required_skills": ["grpc"]}
    reqs = job_requirements(job)
    must = {r.skill for r in reqs if r.kind == "must" and r.skill}
    assert must == {"grpc", "kafka", "kubernetes", "postgresql"}
    assert [r.skill for r in reqs if r.kind == "nice"] == ["redis"]
    assert [r.years for r in reqs if r.kind == "years"] == [5]
    assert any(r.kind == "must" and r.skill is None and "on-call" in r.text for r in reqs)


def test_match_and_explain():
    cs = resume_chunks(RESUME, date(2024, 6, 1))
    job = {"requirements": "Experience with Kafka, Kubernetes, PostgreSQL, CUDA; 5+ years. Nice to have: Redis."}
    jm = match_job(job_requirements(job), cs)
    reason = explain(jm, cs)
    assert 0.5 < jm.score < 1.0
    assert "Meets 3 of 4 required items" in reason and "missing: cuda" in reason
    assert "also has redis" in reason and "years of experience vs 5+ asked" in reason


def test_encoder_family_conventions():
    assert prepare(["x"], "intfloat/multilingual-e5-large", "query") == ["query: x"]
    assert prepare(["x"], "intfloat/multilingual-e5-large", "passage") == ["passage: x"]
    assert prepare(["x"], "BAAI/bge-m3", "query") == ["x"]
    q = prepare(["x"], "Qwen/Qwen3-Embedding-0.6B", "query", task="requirement")[0]
    assert q.startswith("Instruct: ") and q.endswith("Query: x")
    assert family_of("some/unknown-model").query_prefix == ""


def test_remote_encoder_is_off_by_default():
    with pytest.raises(PermissionError):
        OpenAICompatibleEncoder("any-model", "https://example.invalid/v1")
