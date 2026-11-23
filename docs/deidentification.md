# De-identification

Resumes, job notes and outcome notes contain people: the candidate, their colleagues, co-authors,
references, recruiters, hiring managers. The matching models need the professional content and none
of the people. This pipeline removes the people and keeps as much of the professional text as
possible. Everything runs locally; no text is sent to any service.

Code: [`src/matchrec/deid/`](../src/matchrec/deid/)

## Design goals

1. **Mask spans, not documents.** Rejecting a whole document because one span is uncertain throws
   away the training data the system exists to use. Only the uncertain span is replaced.
2. **One decision function.** Many rules are needed, and they conflict: `Julia` is a language and a
   name; `C` is a language and an initial; a co-author list looks like a skills list. If rules run one
   after another, a later rule silently undoes an earlier one. Every candidate span is decided by a
   single ordered ladder, the same code path for redaction, simulation and audit.
3. **Two kinds of error, two tools.** Over-masking hides skills and hurts matching; aggregate audits
   find it. Leaks expose people; only human review of sampled outputs finds them. Neither replaces the
   other.
4. **Public vocabulary only.** Allow lists are hand-written public technical vocabulary plus terms
   from typed structured fields (organization names, skill lists) of the snapshot being processed.
   Nothing is learned from free text, so a person's name in a sentence cannot be whitelisted by accident.

## The precedence ladder

Layers are tried in order. The first that claims a span decides, and a lower layer can never enlarge
or override a higher layer's mask.

| # | Layer | Decides | Result |
|---|---|---|---|
| 1 | Contact | email, phone, URL, social handle, ID number, street address, date of birth | typed placeholder |
| 2 | Known person | exact full-span match of names in the snapshot's identity records | `[PERSON]` |
| 3 | References | a "References" section | removed whole |
| 4 | Citation | validated author spans; unparsable citation records | authors `[PERSON]`; record removed |
| 5 | Identity zone | resume header lines near the candidate's own name or contacts | masked unless a date, title abbreviation or technical token |
| 6 | Human decision | reviewed decisions per term and scope; mask wins conflicts | mask or keep |
| 7 | Symbol token | indivisible technical tokens: `C++`, `C#`, `Node.js`, `.NET` | keep |
| 8 | Lexicon | public technical and resume vocabulary; protected organization and skill terms | keep, unless a person cue is nearby |
| 9 | Format rule | months, date tokens, degree and title abbreviations, code identifiers, institution names, a bare `C` with technical evidence | keep |
| 10 | Default | any detector proposal no layer kept | `[PERSON]`, or `[AMBIG]` if the detector was unsure |

Layers 1–4 are deterministic spans found before any name detector runs. Layers 5–10 judge the spans a
detector proposes. A proposal is first grown to whole tokens (so `Node` inside `Node.js` is judged as
`Node.js`) and split at technical and protected boundaries, so that "Mentored by *Mira Tesloq* at
*Zorbel Robotics*" masks the person and keeps the company.

After rendering, the detector runs again on the output with placeholders blanked, until no new mask
appears. Contact patterns are re-checked because a replacement can join characters into a new one.

## Detectors

A detector only proposes spans; the ladder decides. `SpacyDetector` uses local spaCy pipelines
(English, optionally Chinese) and is the recommended choice for real text. `HeuristicDetector`
proposes capitalized word runs, plus Chinese names (a common surname and one or two characters)
between relational cue words such as 与…合作 collaborate with sb.. It is meant for tests and demos 
and deliberately over-proposes, which makes it a useful stress test of the allow layers.

Some technical terms are also given names (`Julia`, `Ruby`, `Ada`). The lexicon keeps them, except when
the same detector proposal contains a part already masked as a person: in "worked with Julia Venrid",
`Julia` is masked with `Venrid`; in "prototyped models in Julia", it stays.

## Human decisions

Reviewers settle a recurring ambiguous term once (`examples/term_decisions.example.json`). A decision
has a scope: `global`, `skills_section`, `tech_context` (technical words nearby and no person cue), or
`et_al`. Each run records the decision file's SHA-256, so any output can be traced to the exact
decisions that produced it. Real decision files are review artifacts and stay out of version control.

## Audit and sampling

`python -m matchrec.deid.audit --snapshot <raw> --out audit.json` produces:

- **Rule shares.** Which layer and rule produced each mask and each keep, as shares.
- **Over-masking candidates.** Terms masked by judgement layers (identity zone, default) in many
  different documents. A real person's name rarely recurs across unrelated resumes; a recurring masked
  term is usually a missing vocabulary entry. Reported as hashes and shape flags (`all_caps`,
  `code_or_version_like`, `cjk`, ...); plain terms only with an explicit local flag.
- **Ablation.** Re-run with one allow layer switched off. The relative rise in masks is what that
  layer protects.
- **Leak review sample.** A sample stratified by status and placeholder density. Low-density outputs
  are where a missed name hides, so every stratum is represented.

On synthetic data this loop already paid off. The first audit flagged a term masked in every resume:
"Technology" in "*X* Institute of Technology", proposed by the heuristic detector and kept by no layer.
The fix was a format rule for institution names. After it, the candidate list was empty.

## Known limits

- Name detection quality depends on the detector. The ladder limits the damage of over-proposal; it
  cannot recover a name the detector never proposes outside layers 1–5.
- A dual-use word standing alone as a name ("thanks to Julia") is kept unless a person cue is nearby
  or a human decision masks it.
- The heuristic detector finds Chinese names only next to cue words; real Chinese text needs the
  spaCy Chinese model.
- Placeholders keep sentence structure but drop who-did-what-with-whom. For matching this is the
  intended trade-off.
