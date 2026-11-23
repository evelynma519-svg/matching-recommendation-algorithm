"""Leakage firewall: neutral snapshot -> model inputs | labels | eval pools, physically separated.

<build>/inputs/   the ONLY thing models may read (see inputs.py)
    queries.jsonl   query_id, candidate_id, as_of, fold          (no outcomes)
    resumes.jsonl   query_id, as-of resume text, version method, cut stats
    jobs.jsonl      job model view: description fields + notes; no status, no outcomes
<build>/labels/   dev builds: graded label per application + qrels
<build>/sealed/   test builds: same files, read only by the evaluator
<build>/eval/     pools.jsonl (candidate job pool per query), reports/
<build>/build.json  provenance, parameters, aggregate diagnostics

Query = one candidate's submission episode: applications grouped while within `gap_days` of the
episode's first submission. as_of = first submission date. Relevant jobs = the episode's jobs,
graded by the funnel.

Job pool for a query = jobs open at as_of. With `Job.opened_at`/`closed_at` this is exact; for
jobs without dates it falls back to a window around the job's observed submissions.

    python -m matchrec.matching.firewall --snapshot data/synthetic --role dev
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from ..schema import Application, Job, JsonlSnapshot, get_funnel, to_dict, validate
from . import config
from .asof import choose_version, cut_resume
from .labels import label_row, parse_day
from .split import ROLES, Registry, fold_of, role_of

JOB_VIEW_FIELDS = ("job_id", "org_id", "org_name", "title", "employment_type", "locations",
                   "compensation", "summary", "responsibilities", "requirements", "required_skills")


def _write_jsonl(p: Path, rows) -> int:
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def episodes(rows: list[dict], gap_days: int) -> list[list[dict]]:
    rows = sorted(rows, key=lambda r: r["submitted_at"])
    out: list[list[dict]] = []
    for r in rows:
        d = parse_day(r["submitted_at"])
        if out and (d - parse_day(out[-1][0]["submitted_at"])).days <= gap_days:
            out[-1].append(r)
        else:
            out.append([r])
    return out


_STOP = {"inc", "llc", "ltd", "corp", "corporation", "company", "technologies", "technology", "group",
         "labs", "the", "global", "international", "systems", "analytics", "cloud"}


def _org_key(name: str | None) -> str | None:
    name = re.sub(r"\[[A-Z_]+\]", " ", name or "")            # drop de-identification placeholders
    toks = [t for t in re.findall(r"[a-z][a-z0-9&.-]{2,}", name.lower()) if t not in _STOP]
    return toks[0] if toks else None


def _mentions(text: str, name: str | None) -> bool | None:
    k = _org_key(name)
    if not k:
        return None
    return re.search(rf"(?<![a-z]){re.escape(k)}(?![a-z])", text.lower()) is not None


def job_windows(jobs: dict[str, Job], apps: list[Application], pad_days: int) -> tuple[dict, dict]:
    """Open interval per job. Exact dates when present, else submission window +/- pad."""
    seen: dict[str, list[date]] = defaultdict(list)
    for a in apps:
        if (d := parse_day(a.submitted_at)):
            seen[a.job_id].append(d)
    pad = timedelta(days=pad_days)
    out, method = {}, defaultdict(int)
    for jid, j in jobs.items():
        o, c = parse_day(j.opened_at), parse_day(j.closed_at)
        if o:
            out[jid] = (o, c or date.max)
            method["posting_dates"] += 1
        elif seen.get(jid):
            out[jid] = (min(seen[jid]) - pad, max(seen[jid]) + pad)
            method["submission_window"] += 1
    return out, dict(method)


def build(snapshot_dir: Path, out_root: Path, role: str = "dev", funnel: str | None = None,
          test_share: float = 0.2, gap_days: int = 60, maturity_days: int = 30, window_pad_days: int = 45,
          registry_path: Path | None = None) -> Path:
    if role not in ROLES:
        raise ValueError(role)
    src = JsonlSnapshot(snapshot_dir)
    snap = src.snapshot()
    fun = get_funnel(funnel or snap.funnel)
    problems = validate(src, fun)
    if problems:
        raise ValueError(f"snapshot failed validation ({len(problems)} problems), first: {problems[0]}")
    ref = date.fromisoformat(snap.reference_date)
    cands = {c.candidate_id: c for c in src.candidates()}
    jobs = {j.job_id: j for j in src.jobs()}
    apps = list(src.applications())

    reg = Registry(registry_path or (Path(out_root) / "registry.json"))
    reg_report = reg.register(snap.snapshot_id, candidates=cands, jobs=jobs,
                              orgs={j.org_id for j in jobs.values()}, applications={a.application_id for a in apps})
    reg.save()

    out = Path(out_root) / "builds" / f"{snap.snapshot_id}__{role}"
    lab_dir = out / ("labels" if role == "dev" else "sealed")

    # ---- model view of jobs: description only, no status / dates / people
    job_view = {}
    for jid, j in jobs.items():
        v = {k: v for k, v in to_dict(j).items() if k in JOB_VIEW_FIELDS}
        v["notes"] = [{"text": n.text, "time_verified": n.written_at is not None} for n in j.notes]
        job_view[jid] = v

    # ---- labels for this role's candidates only
    lab = [label_row(a, fun, ref, maturity_days, fold_of(a.candidate_id), jobs[a.job_id].org_id)
           for a in apps if role_of(a.candidate_id, test_share) == role]
    bound: dict[str, str | None] = {a.application_id: a.resume_id for a in apps}
    _write_jsonl(lab_dir / "pairs.jsonl", lab)

    skipped = defaultdict(int)
    by_cand: dict[str, list[dict]] = defaultdict(list)
    for r in lab:
        if r["excluded"]:
            skipped["excluded_reason"] += 1
        elif not r["matured"]:
            skipped["not_matured"] += 1
        else:
            by_cand[r["candidate_id"]].append(r)

    windows, pool_method = job_windows(jobs, apps, window_pad_days)

    queries, resumes, qrels, pools = [], [], [], []
    diag = defaultdict(lambda: [0, 0, 0])          # grade -> [n, mentioned_before_cut, mentioned_after_cut]
    cut_totals, version_method = defaultdict(int), defaultdict(int)
    for cid, rows in sorted(by_cand.items()):
        cand = cands[cid]
        for ep in episodes(rows, gap_days):
            as_of = parse_day(ep[0]["submitted_at"])
            ver, how = choose_version(cand, {bound[r["application_id"]] for r in ep if bound[r["application_id"]]}, as_of)
            if ver is None:
                skipped["no_resume"] += len(ep)
                continue
            version_method[how] += 1
            qid = f"{cid}@{as_of.isoformat()}"
            text, st = cut_resume(ver, as_of)
            for k, v in st.items():
                cut_totals[k] += v
            queries.append({"query_id": qid, "candidate_id": cid, "as_of": as_of.isoformat(), "fold": fold_of(cid)})
            resumes.append({"query_id": qid, "candidate_id": cid, "as_of": as_of.isoformat(), "text": text,
                            "version_method": how, "asof_stats": st})
            best: dict[str, int] = {}
            for r in ep:
                best[r["job_id"]] = max(best.get(r["job_id"], 0), r["grade"])
                before = _mentions(ver.text, jobs[r["job_id"]].org_name)
                if before is not None:
                    dg = diag[r["grade"]]
                    dg[0] += 1
                    dg[1] += before
                    dg[2] += bool(_mentions(text, jobs[r["job_id"]].org_name))
            qrels += [{"query_id": qid, "job_id": j, "grade": g} for j, g in sorted(best.items())]
            pool = {j for j, (a, b) in windows.items() if a <= as_of <= b} | set(best)
            pools.append({"query_id": qid, "job_ids": sorted(pool)})

    _write_jsonl(out / "inputs" / "queries.jsonl", queries)
    _write_jsonl(out / "inputs" / "resumes.jsonl", resumes)
    _write_jsonl(out / "inputs" / "jobs.jsonl", job_view.values())
    _write_jsonl(lab_dir / "qrels.jsonl", qrels)
    _write_jsonl(out / "eval" / "pools.jsonl", pools)

    n_q = max(len(queries), 1)
    grade_share = defaultdict(int)
    for q in qrels:
        grade_share[q["grade"]] += 1
    meta = {
        "snapshot_id": snap.snapshot_id, "source": snap.source, "reference_date": snap.reference_date,
        "role": role, "funnel": fun.name, "observable_grades": sorted(fun.observable_grades),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "params": {"test_share": test_share, "gap_days": gap_days, "maturity_days": maturity_days,
                   "window_pad_days": window_pad_days},
        "registry": reg_report,
        "counts": {"queries": len(queries), "qrels": len(qrels), "jobs_in_view": len(job_view),
                   "skipped_applications": dict(skipped),
                   "avg_pool_size": round(sum(len(p["job_ids"]) for p in pools) / n_q, 1)},
        "shares": {"qrels_by_grade": {g: round(c / max(len(qrels), 1), 3) for g, c in sorted(grade_share.items())},
                   "resume_version_method": {k: round(v / n_q, 3) for k, v in sorted(version_method.items())},
                   "pool_method": pool_method},
        "asof_cut": dict(cut_totals),
        "leak_diagnostic_target_org_mentioned": {
            f"grade_{g}": {"n": v[0], "before_cut": round(v[1] / v[0], 3), "after_cut": round(v[2] / v[0], 3)}
            for g, v in sorted(diag.items()) if v[0]},
    }
    (out / "build.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build model inputs, labels and eval pools from a snapshot.")
    ap.add_argument("--snapshot", type=Path, required=True, help="neutral snapshot directory")
    ap.add_argument("--role", choices=ROLES, default="dev")
    ap.add_argument("--out", type=Path, default=None, help="output root (default: $MATCHREC_DATA_DIR or ./data)")
    ap.add_argument("--funnel", default=None, help="built-in funnel name or JSON path (default: from manifest)")
    ap.add_argument("--test-share", type=float, default=0.2)
    a = ap.parse_args(argv)
    out = build(a.snapshot, a.out or config.data_dir(), a.role, a.funnel, a.test_share)
    meta = json.loads((out / "build.json").read_text())
    print("build:", out)
    print(json.dumps({k: meta[k] for k in ("counts", "shares", "asof_cut", "leak_diagnostic_target_org_mentioned")},
                     indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
