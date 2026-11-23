# Retrieval and matching

Code: [`src/matchrec/matching/`](../src/matchrec/matching/)

Every scorer ranks the jobs in one query's pool (the jobs open on the candidate's as-of date). All of
them read only the firewall's model inputs: de-identified resumes cut at the as-of date, and job views
without outcomes.

| Scorer | Unit of comparison | Notes |
|---|---|---|
| `bm25` | whole resume vs job text | bilingual tokenizer; keeps `C++`, `C#`, `Node.js`; CJK bigrams |
| `dense` | one vector per resume | baseline for the chunked scorer |
| `dense_chunks` | one vector per resume chunk | aggregation: `max`, `topk` (mean of the k best), `recency` |
| `requirements` | each job requirement vs each resume chunk | weighted coverage; lexical, optionally plus semantic evidence; produces reasons |
| `hybrid`, `hybrid_chunks` | reciprocal rank fusion of BM25 and a dense scorer | |

## Why chunk resumes

One vector per resume dilutes specific skills into an average, and encoders truncate long input, so
the end of a long resume is never seen. `chunks.py` splits a resume into short pieces instead:

- each experience or project entry, keeping its title, organization and date lines as context
- skills, summary, education and publication sections, one chunk each (long ones split)
- header and reference sections are dropped; after de-identification they carry no content

The as-of cut runs first, so chunking never sees post-submission work. Experience chunks carry dates
and a recency rank. A resume then scores against a job by aggregating chunk similarities:

- `max`: the single best chunk. Rewards one strongly matching role; blind to breadth.
- `topk`: mean of the k best chunks (default 3). Rewards several relevant roles.
- `recency`: best chunk after weighting experience by `decay^rank` (default 0.9), so recent work counts more.

## Requirement-level matching

A recruiter reads a job requirement by requirement and looks for evidence in the resume.
`requirements.py` does the same:

1. **Split the job into requirement units.** Required skills, other must-have clauses, nice-to-haves
   (after "nice to have", "preferred", "加分项", ...), and years of experience.
2. **Build an evidence matrix** of requirement × resume chunk:
   - lexical evidence: the requirement's normalized skill appears in the chunk (`k8s` = `Kubernetes`,
     `深度学习` = `deep learning`, via `skills.py`), or token overlap for free-text clauses;
   - semantic evidence (optional): cosine similarity between requirement and chunk. Raw similarity is
     not usable as evidence: short, generic chunks (an education line) are close to almost every
     requirement, the hubness problem. Following CSLS-style local scaling, each similarity is corrected
     by the chunk's mean similarity to all requirements it meets and the requirement's mean similarity
     to all chunks it meets, then rescaled on the build's own distribution (below the 90th percentile
     → 0, 99.5th → 1). No labels are used.
   - evidence = the larger of the two.
3. **Score coverage:** must-haves weigh 1.0, nice-to-haves 0.4, years 0.6 (candidate years estimated
   from dated experience entries, overlaps merged).

Each score keeps the best chunk per requirement and how it matched, so the recommendation can be
explained plainly:

> Meets 3 of 4 required items (kafka, kubernetes, postgresql); missing: cuda; also has redis; about 7
> years of experience vs 5+ asked.

`python -m matchrec.matching.evaluate --scorer requirements --explain 3` writes such reasons for each
query's top three jobs. The output contains de-identified resume snippets and stays local, like all
reports.

Planned next steps on the same matrix: a cross-encoder that scores each requirement–chunk pair
directly, and a local LLM for requirement extraction and judgement (see the roadmap).

## Encoders

`encoders.py` makes the model a configuration choice:

| Use | Default model | Why |
|---|---|---|
| Resume and job retrieval (`dense`, `dense_chunks`) | `BAAI/bge-m3` | Multilingual with strong Chinese–English alignment, so a Chinese resume and an English job ad land in one space; mature and widely used for multilingual retrieval; open weights, runs locally |
| Requirement evidence (`requirements --semantic`) | `intfloat/multilingual-e5-large` | A standard multilingual retrieval baseline that does best on short text, which is what requirements and resume chunks are |
| Optional | `Qwen/Qwen3-Embedding-0.6B` and other sentence-transformers models | Newer instruction-aware models; pass `--model` |

Model families mark their inputs differently (e5 wants `query: ` / `passage: `; Qwen3 embeddings want a
task instruction on queries). `FAMILIES` holds these conventions, so changing the model never changes
calling code. Embeddings are cached per model, role and task, so vectors from different models are
never mixed, and scores are only ever compared within one model.

For public datasets, `OpenAICompatibleEncoder` can call a remote embeddings API. It refuses to run
unless `allow_remote=True` is passed. Do not use it on sensitive text. The default path keeps every
byte local.
