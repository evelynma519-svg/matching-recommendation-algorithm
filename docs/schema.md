# Data schema

The project works on a neutral data model, which reflects the common recruiting process, not on any recruiting system's native export.
A **source adapter** maps a data source's export into these records. The field names below are this project's vocabulary. 
A real source uses its own identifiers, and the adapter's mapping changesn with the source. 
Adapters for real systems, and the data they read, stay outside this repository.

Code: [`src/matchrec/schema/model.py`](../src/matchrec/schema/model.py)

## Records

| Record | Key | Purpose |
|---|---|---|
| `Snapshot` | `snapshot_id` | One export. `reference_date` decides which outcomes are mature enough to label. |
| `Candidate` | `candidate_id` | A person, with one `ResumeVersion`s and a skill list, identified through a unique ID in datasets. |
| `Job` | `job_id` | An opening at a hiring organization (`org_id`): description, required skills, open/close dates, and notes, identified through a unique ID in datasets. |
| `Application` | `application_id` | One candidate submitted to one job, with the `StageEvent`s that followed. |

```
Candidate 1 ── * Application * ── 1 Job * ── 1 Organization
    │                │
    └── * ResumeVersion ◄── resume_id (the version sent with this submission, if recorded)
```

## Process stages

Stage names are not fixed. Each snapshot names a funnel (`Snapshot.funnel`) that lists its stages
in order and maps them to relevance levels L1–L4. See [funnel.md](funnel.md). In practice, the default or common
`staffing` funnel is: `submitted` → `client_review` → `interview` → `offer` → `offer_accepted` → `hired`.

Each `StageEvent` has a status: `not_started`, `in_progress`, `passed`, or `dropped`.
`dropped` can only appear on the last event, and `Application.end_reason` says why. The default
vocabulary is `employer_declined`, `internal_screen_out`, `candidate_withdrew`, `position_closed`,
`auto_expired`, `other`, plus two bookkeeping reasons, `duplicate` and `entry_error`, which the funnel
excludes from labels. 

An ended process is *unknown*, not negative.

## Who sees what

| Field group | De-identification | Label builder | Models |
|---|---|---|---|
| Resume text, job description, required skills | reads and masks | — | reads (after masking and the as-of cut) |
| `Job.notes` | reads and masks | — | reads, flagged when `written_at` is unknown |
| `Candidate.identity`, `Job.related_people` | reads (to know what to mask) | — | **never** |
| `stages`, `end_reason`, `end_note`, `Job.status` | masks `end_note` | reads | **never** (enforced by a key check on every model input) |
| `opened_at` / `closed_at` | — | builds the "open at as-of" job pool | — |

## On-disk format

A snapshot directory holds `manifest.json`, `candidates.jsonl`, `jobs.jsonl` and
`applications.jsonl`. `JsonlSnapshot` reads it, `write_snapshot` writes any adapter into it, and
`validate` checks references, unique ids and stage order.

## Writing an adapter

Implement four methods and yield the neutral records:

```python
class MySourceAdapter:
    def snapshot(self) -> Snapshot: ...
    def candidates(self) -> Iterable[Candidate]: ...
    def jobs(self) -> Iterable[Job]: ...
    def applications(self) -> Iterable[Application]: ...
```

Then call `write_snapshot(out_dir, MySourceAdapter(...))` and run `validate` on the result. Every
step after this point reads only the neutral format.

## Synthetic data

`python -m matchrec.synthetic.generate --out data/synthetic` builds a seeded snapshot. Names are random
syllables or random combinations of common Chinese characters; organizations use a separate syllable
set; contacts use reserved test ranges (`example.test`, `555-01xx` numbers). The generator is built to
make each part of the pipeline earn its keep:

| Property | What it tests |
|---|---|
| Outcomes from a hidden fit: skill overlap, role family, seniority gap, organization taste, and a latent candidate–organization affinity that appears in no text | Text models can learn part of the signal, never all; room for graph models |
| Job ads use canonical skill names; resumes use aliases, paraphrases and Chinese translations, and mention only some skills | Keyword matching misses true matches; semantic retrieval should help |
| For Chinese or mixed resumes, with Chinese headings and date formats | Bilingual tokenization, as-of cut, de-identification |
| Headers with contacts, colleagues in prose (some named Julia or Ruby), co-authors, referees | De-identification; leak rate measured against the generator's list of every third-party name |
| For resumes include a job that started after the submission or application | The as-of cut |
