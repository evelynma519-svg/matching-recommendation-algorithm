# Roadmap

Status: `done`, `next`, `planned`.

## Data model
- `done` Neutral schema, source-adapter seam, JSONL snapshot format, validation.
- `done` Configurable funnels with graded relevance (staffing, direct, minimal, custom JSON).
- `done` Seeded synthetic generator with hidden fit, organization taste, and post-submission resume entries.
- `planned` Candidate preferences recorded by the recruiter (location, compensation, role direction, timing) as a dated model input, cut at the as-of date like resumes.
- `done` Harder synthetic text: skill aliases and paraphrases, partial skill mentions, about 30% Chinese or mixed resumes, a latent organization affinity that no text reveals, dual-use names.

## Matching pipeline
- `done` Label builder on top of `FunnelSpec`; leakage firewall (model inputs / labels / sealed test physically separated).
- `done` As-of resume cut; query episodes; open-at-as-of job pool using `opened_at`/`closed_at`, with a submission-window fallback.
- `done` Candidate-level split with a hashed sealed hold-out that needs no future data batch; folds inside dev.
- `next` Temporal check: re-run the comparison on the latest period only, to see whether results hold forward in time.
- `done` Retrieval baselines: bilingual BM25, dense bi-encoder, reciprocal rank fusion.
- `done` Evaluation: Recall@K by level, NDCG with graded gain, MRR, bootstrap CIs, paired bootstrap comparisons.
- `done` Resume chunking (experience entries and sections with context); chunked dense retrieval with max, top-k and recency aggregation.
- `done` Requirement-level matching: requirement units (must, nice-to-have, years), requirement x chunk evidence matrix (normalized skills, token overlap, optional semantic evidence), weighted coverage, plain-language reasons.
- `done` Swappable encoders: per-family input conventions, local default, remote API off by default.
- `done` Requirement evidence with `intfloat/multilingual-e5-large`, with CSLS-style hubness correction.
- `planned` Seniority fit in requirement coverage: today years are capped at the requirement, so over-qualification is not penalized.

## Recruiter support
- `planned` Effective requirements per job: combine the description, time-valid notes and past progress into one requirement list, each item tagged with its source.
- `planned` Uncertainty flags: requirements that conflict across sources or appear only once; candidate evidence that is thin, undated or inconsistent.
- `planned` Clarifying questions for the AM or client, generated from uncertain requirements.
- `planned` Questions for the recruiter to check with the candidate, generated from weak or missing evidence. The system flags; it never rules on whether a resume is truthful.

## Feedback and transfer
- `planned` **Explicit feedback**: a gold set rated against a rubric (fit per requirement, overall fit, confidence), recruiter thumbs-up/down on recommendations, and agreement checks between raters.
- `planned` Stage-transition priors and score-to-outcome calibration estimated from full-funnel data.
- `planned` Partial-funnel support: estimate unobservable levels from priors; report metrics only on observable levels.
- `planned` Rubric validation: agreement between explicit labels and funnel outcomes.
- `planned` Learned level weights replacing the fixed 2^g − 1 gain.

## Models
- `planned` Cross-encoder scoring each requirement–chunk pair on the same evidence matrix.
- `planned` Graph model on the candidate–job–organization graph (GNN baseline), compared against the text models with the same splits and paired tests.

## De-identification pipeline
- `done` Ten-layer precedence ladder, span-level masking, citation authors, references, identity zone, human term decisions with scopes.
- `done` Local detectors (spaCy, heuristic); detector re-runs on the output until stable; contact re-check.
- `done` Audit: rule shares, over-masking candidates (hashed), layer ablation, stratified leak-review sample.
- `done` Leak rate measured against synthetic ground truth (every generated third-party name is known), English and Chinese.

## Local LLM interface
- `planned` A thin interface with swappable local backends and a cache keyed by backend, model, prompt version and input, so results are reproducible and traceable to the prompt. Uses: extracting requirements from job descriptions and requirement-level matching against resume evidence, classifying end notes, drafting rubric scores for explicit feedback (confirmed by a person), and plain-language explanations of why a job is recommended.

## Extension directions (recorded, not being built)
- Internal mobility: internal role recommendations for current employees through a new adapter (HR records, performance, 360 feedback, project history). Requires employee notice and consent, HR governance and a compliance review of sensitive data.
- Company-level talent map.
- Fine-tuning the embedding model on explicit feedback.
- A trial inside a real recruiting workflow.

## Public benchmark
- `planned` Run retrieval on a public resume–job dataset where licensing allows; report the levels that dataset can observe.
