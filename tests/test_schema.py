from collections import Counter

from matchrec.schema import Application, JsonlSnapshot, StageEvent, StageStatus, validate, write_snapshot
from matchrec.synthetic.generate import in_memory


def test_synthetic_snapshot_round_trips_and_validates(tmp_path):
    src = in_memory(n_candidates=60, n_jobs=30, seed=1)
    counts = write_snapshot(tmp_path, src)
    assert counts["candidates"] == 60 and counts["jobs"] == 30 and counts["applications"] > 0
    disk = JsonlSnapshot(tmp_path)
    assert validate(disk) == []
    a0 = next(src.applications())
    b0 = next(disk.applications())
    assert a0 == b0                                    # enums and nested records survive the trip
    assert disk.snapshot().source == "synthetic"


def test_generator_is_deterministic():
    a = [x.application_id + x.job_id for x in in_memory(n_candidates=40, n_jobs=20, seed=3).applications()]
    b = [x.application_id + x.job_id for x in in_memory(n_candidates=40, n_jobs=20, seed=3).applications()]
    assert a == b


def test_outcomes_span_every_stage():
    deepest = Counter(a.stages[-1].stage for a in in_memory(n_candidates=400, n_jobs=120, seed=7).applications())
    assert {"submitted", "client_review", "interview"} <= set(deepest)
    assert deepest["hired"] > 0


def test_contacts_use_reserved_test_ranges():
    for c in in_memory(n_candidates=50, n_jobs=10, seed=2).candidates():
        assert all(e.endswith("@example.test") for e in c.identity.emails)
        assert all(p[7:13] == "555-01" for p in c.identity.phones)


def test_validate_catches_order_and_references():
    src = in_memory(n_candidates=5, n_jobs=5, seed=4)
    bad = Application("x1", "nobody", "nojob", "2024-01-01",
                      [StageEvent("interview", StageStatus.PASSED),
                       StageEvent("submitted", StageStatus.DROPPED)])
    src._a = [bad]
    problems = validate(src)
    assert any("unknown candidate" in p for p in problems)
    assert any("unknown job" in p for p in problems)
    assert any("out of order" in p for p in problems)


def test_validate_rejects_stages_outside_the_funnel():
    src = in_memory(n_candidates=3, n_jobs=3, seed=5)
    a = next(src.applications())
    a.stages = [StageEvent("phone_call", StageStatus.PASSED)]
    src._a = [a]
    assert any("not in funnel" in p for p in validate(src))
