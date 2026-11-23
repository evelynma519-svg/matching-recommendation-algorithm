# Funnels, feedback, and what to do when outcomes are missing

## Why a funnel

The system recommends jobs to a candidate. To learn and to evaluate, it needs evidence of which
recommendations were good. The cheapest evidence is what already happened: candidates put forward
for jobs, and how far each process went. This is **learning from implicit feedback**. Nobody
labelled "good match"; the funnel depth stands in for it.

Depth is graded because later stages are stronger evidence:

| Level | Evidence | Meaning |
|---|---|---|
| L1 | put forward | a recruiter or the candidate judged it plausible |
| L2 | employer review | the employer looked at the profile |
| L3 | interview | the employer spent time on the candidate |
| L4 | offer or hire | the employer wanted the candidate |

A process that ended is **unknown, not negative**. Candidates withdraw, roles close, budgets
move. Treating every ended process as a mismatch teaches the model the wrong thing, so the labels are
positive-unlabelled: graded positives and an unlabelled rest.

Which stages map to which level is configuration (`src/matchrec/schema/funnel.py`). Built-ins:

| Funnel | Stages | Observable levels |
|---|---|---|
| `staffing` | submitted, client_review, interview, offer, offer_accepted, hired | L1–L4 |
| `direct` | applied, screen, interview, offer, hired | L1–L4 |
| `minimal` | applied, hired | L1, L4 |

A custom funnel is a JSON file (see `tests/test_funnel.py`).

## Matching the method to the data you actually have

Full multi-stage outcomes at scale are rare. Most teams have one of three situations, and the
system is meant to degrade gracefully across them rather than require the first.

| Situation | Typical source | What the system does |
|---|---|---|
| **A. Full funnel, hundreds of candidates or more** | an agency or in-house tracking system with stage history | Graded implicit labels for training and evaluation. The funnel also yields the transferable priors below. |
| **B. Partial funnel** | only applied and hired, or only the final status | Observed levels are used directly. Missing levels are estimated from stage-transition priors (from situation A or published rates) instead of being treated as zero. Metrics are reported only for observable levels. |
| **C. No outcomes, or a few dozen candidates** | a single recruiter's own records, a new team | Explicit feedback carries the load: a small gold set rated against a rubric, plus thumbs-up/down on recommendations. Retrieval runs without training (lexical + semantic). Priors give a starting calibration. |

Even with a full funnel, explicit feedback stays necessary. Funnel outcomes mix fit with timing,
candidate interest, and employer politics. A rated gold set is the only way to check that what the
model learned from outcomes is actually *fit*.

## Turning one rich dataset into things others can use

A full-funnel dataset is unusual. Its value outside its own setting is not the raw outcomes. It is
what can be distilled from them and reused where outcomes are missing:

1. **Stage-transition priors.** P(reach stage k+1 | reached k), broken down by role family and
   seniority gap. Situation B uses them to impute missing levels; situation C uses them as a prior.
   These are ratios, not counts.
2. **Score-to-outcome calibration.** A mapping from a model's match score to the probability of
   reaching each level. Anyone running the same retrieval stack can read a score as "estimated chance
   of an interview" before they have outcomes of their own, then re-calibrate with a few dozen local
   labels.
3. **Which match signals predict depth.** For example, requirement coverage versus title similarity
   versus seniority gap. These become the rubric for explicit feedback and the features for a
   training-free scorer.
4. **Learned level weights.** Instead of fixing the gain per level, estimate how much more an
   interview is worth than a submission from how well each level predicts the next.
5. **A validated rubric.** Check that rubric-based explicit labels agree with funnel outcomes where
   both exist. If they do, the rubric can be used without a funnel.

Priors transfer imperfectly across industries and markets. They are a starting point to be
re-calibrated with local explicit feedback, not ground truth.

## What this repository ships

- The funnel abstraction, the grading rules, and the three built-in funnels.
- A synthetic generator that produces full-funnel data, so every situation can be simulated by
  hiding stages (situation B) or outcomes (situation C) from the same snapshot.
- Code that estimates priors and calibrations from any full-funnel snapshot. The default priors
  shipped here come from synthetic data. Results on a real proprietary funnel are reported only as
  relative comparisons.
