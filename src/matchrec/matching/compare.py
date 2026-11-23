"""Compare evaluation reports in a build: summary table + paired bootstrap against a reference.

Each scorer run by `matchrec.matching.evaluate` leaves <tag>.json (summary) and <tag>.perquery.json
(metrics per query) in <build>/eval/reports/. This command:

1. Summary table: mean and 95% CI per scorer.
2. Paired comparison: for every other scorer, per-query difference (scorer - reference) on the
   same queries, bootstrapped over queries. ✓ = 95% CI excludes 0 (significant), ↑ better, ↓ worse.
   The reference defaults to the scorer with the highest mean on --ref-metric.

Reading the paired table:
- "+0.026 [+0.010, +0.042] ✓↑": the scorer beats the reference by 2.6 points on average; the
  interval excludes 0, so the gain is unlikely to be luck.
- an interval that contains 0 means the data cannot yet tell the two apart; it does not mean they
  are equal. Many metrics x scorers are tested at once, so an isolated borderline ✓ deserves caution.

Usage:
  python -m matchrec.matching.compare --build <build_dir> [--ref hybrid] [--against bm25,dense]
                             [--metrics ...] [--ref-metric ndcg@10] [--boot 2000]
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import date
from pathlib import Path

PRIMARY = ["recall@10_L3+", "recall@10_L4+", "ndcg@10", "mrr", "recall@10_L1+"]
SUMMARY = ["recall@10_L1+", "recall@10_L2+", "recall@10_L3+", "recall@10_L4+", "recall@20_L4+", "ndcg@10", "mrr"]


def load_reports(build_dir: Path) -> dict[str, dict]:
    d = Path(build_dir) / "eval" / "reports"
    out = {}
    for p in sorted(d.glob("*.json")):
        if p.name.endswith(".perquery.json") or p.name.startswith("compare__") or ".explanations" in p.name:
            continue
        pq = p.with_name(p.stem + ".perquery.json")
        out[p.stem] = {"summary": json.loads(p.read_text()),
                       "per_query": json.loads(pq.read_text()) if pq.exists() else None}
    return out


def paired_diff(a: dict[str, dict[str, float]], b: dict[str, dict[str, float]], metric: str,
                n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean of per-query (a - b) on queries where both have the metric, with a bootstrap 95% CI."""
    qs = sorted(q for q in a if metric in a[q] and q in b and metric in b[q])
    d = [a[q][metric] - b[q][metric] for q in qs]
    if not d:
        return {"mean": None, "ci95": [None, None], "n": 0, "significant": False}
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n_boot))
    lo, hi = boots[int(0.025 * n_boot)], boots[min(int(0.975 * n_boot), n_boot - 1)]
    mean = sum(d) / len(d)
    return {"mean": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)], "n": len(d),
            "significant": lo > 0 or hi < 0}


def resolve(name: str, tags) -> str:
    if name in tags:
        return name
    hits = [t for t in tags if t.startswith(name)]
    if len(hits) != 1:
        raise SystemExit(f"'{name}' matches {hits or 'no report'}; available: {sorted(tags)}")
    return hits[0]


def compare(build_dir: Path, ref: str | None = None, against: list[str] | None = None,
            metrics: list[str] = PRIMARY, ref_metric: str = "ndcg@10", n_boot: int = 2000, seed: int = 0) -> dict:
    reps = load_reports(build_dir)
    paired_ok = {t: r for t, r in reps.items() if r["per_query"] is not None}
    if not paired_ok:
        raise SystemExit("no per-query reports; rerun matchrec.matching.evaluate for each scorer")
    fps = {r["per_query"]["fingerprint"] for r in paired_ok.values()}
    if len(fps) != 1:
        raise SystemExit(f"reports come from different query sets / labels {sorted(fps)}; re-evaluate them on one build")
    if ref:
        ref = resolve(ref, paired_ok)
    else:
        ref = max(paired_ok, key=lambda t: paired_ok[t]["summary"]["metrics"].get(ref_metric, {}).get("mean", -1))
    others = [resolve(x, paired_ok) for x in against] if against else [t for t in paired_ok if t != ref]
    R = paired_ok[ref]["per_query"]["per_query"]
    rows = {t: {m: paired_diff(paired_ok[t]["per_query"]["per_query"], R, m, n_boot, seed) for m in metrics}
            for t in others}
    return {"build": str(build_dir), "fingerprint": fps.pop(), "reference": ref, "ref_metric": ref_metric,
            "metrics": metrics, "n_boot": n_boot, "seed": seed, "compared_on": date.today().isoformat(),
            "paired": rows, "summary": {t: r["summary"]["metrics"] for t, r in reps.items()}}


def _cell(v: dict | None) -> str:
    return f"{v['mean']:.3f} [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}]" if v else "–"


def _diff_cell(v: dict) -> str:
    if v["mean"] is None:
        return "–"
    mark = (" ✓↑" if v["mean"] > 0 else " ✓↓") if v["significant"] else ""
    return f"{v['mean']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]{mark}"


def to_markdown(res: dict, summary_metrics: list[str] = SUMMARY) -> str:
    out = [f"## Summary (mean [95% CI])", "",
           "| scorer | " + " | ".join(summary_metrics) + " |", "|" + "---|" * (len(summary_metrics) + 1)]
    for t, m in res["summary"].items():
        out.append(f"| {t} | " + " | ".join(_cell(m.get(k)) for k in summary_metrics) + " |")
    out += ["", f"## Paired difference vs reference `{res['reference']}` (scorer − reference)", "",
            "| scorer | " + " | ".join(res["metrics"]) + " |", "|" + "---|" * (len(res["metrics"]) + 1)]
    for t, row in res["paired"].items():
        out.append(f"| {t} | " + " | ".join(_diff_cell(row[m]) for m in res["metrics"]) + " |")
    out += ["", "✓ = 95% CI excludes 0 (↑ better / ↓ worse than reference). "
                "A CI containing 0 means the data cannot tell them apart yet, not that they are equal. "
                f"Bootstrap over queries, n_boot={res['n_boot']}, seed={res['seed']}, fingerprint={res['fingerprint']}."]
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--ref", default=None, help="reference scorer tag (prefix ok); default: best on --ref-metric")
    ap.add_argument("--against", default=None, help="comma-separated scorer tags; default: all others")
    ap.add_argument("--metrics", default=",".join(PRIMARY))
    ap.add_argument("--ref-metric", default="ndcg@10")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args(argv)
    res = compare(a.build, a.ref, a.against.split(",") if a.against else None,
                  a.metrics.split(","), a.ref_metric, a.boot)
    md = to_markdown(res)
    d = a.build / "eval" / "reports"
    (d / f"compare__ref-{res['reference']}.json").write_text(json.dumps(res, indent=1))
    (d / f"compare__ref-{res['reference']}.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
