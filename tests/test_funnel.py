import pytest

from matchrec.schema import (DIRECT, MINIMAL, STAFFING, Application, EndReason, FunnelSpec, StageEvent,
                             StageStatus, get_funnel)

P, IP, D = StageStatus.PASSED, StageStatus.IN_PROGRESS, StageStatus.DROPPED


def app(*events, reason=None):
    return Application("a", "c", "j", "2024-03-01", [StageEvent(s, st) for s, st in events], reason)


@pytest.mark.parametrize("events,reason,grade", [
    ([("submitted", D)], EndReason.INTERNAL_SCREEN_OUT, 1),
    ([("submitted", IP)], None, 1),
    ([("submitted", P), ("client_review", IP)], None, 2),
    ([("submitted", D)], EndReason.EMPLOYER_DECLINED, 2),           # reason proves the employer saw it
    ([("submitted", P), ("client_review", D)], EndReason.CANDIDATE_WITHDREW, 1),
    ([("submitted", P), ("client_review", P), ("interview", D)], EndReason.EMPLOYER_DECLINED, 3),
    ([("submitted", P), ("client_review", P), ("interview", P), ("offer", D)], EndReason.CANDIDATE_WITHDREW, 3),
    ([("submitted", P), ("client_review", P), ("interview", P), ("offer", IP)], None, 4),
    ([("submitted", P), ("client_review", P), ("interview", P), ("offer", P), ("offer_accepted", P),
      ("hired", IP)], None, 4),
])
def test_staffing_ladder(events, reason, grade):
    assert STAFFING.grade(app(*events, reason=reason)).grade == grade


def test_ended_is_not_negative_and_bookkeeping_is_excluded():
    g = STAFFING.grade(app(("submitted", P), ("client_review", D), reason=EndReason.EMPLOYER_DECLINED))
    assert g.ended and not g.excluded and g.grade == 2
    assert STAFFING.grade(app(("submitted", D), reason=EndReason.DUPLICATE)).excluded


def test_hired_is_success_not_ended():
    g = STAFFING.grade(app(("submitted", P), ("client_review", P), ("interview", P), ("offer", P),
                           ("offer_accepted", P), ("hired", IP)))
    assert g.grade == 4 and not g.ended and not g.in_flight


def test_direct_and_minimal_funnels():
    assert DIRECT.grade(app(("applied", P), ("screen", P), ("interview", IP))).grade == 3
    assert MINIMAL.observable_grades == {1, 4}
    assert MINIMAL.grade(app(("applied", P), ("hired", IP))).grade == 4
    assert MINIMAL.grade(app(("applied", D), reason=EndReason.EMPLOYER_DECLINED)).grade == 1


def test_custom_funnel_from_json(tmp_path):
    p = tmp_path / "f.json"
    p.write_text('{"name": "campus", "stages": ["applied", "assessment", "interview", "offer"],'
                 ' "levels": [{"grade": 1, "stages": ["applied"], "dropped_counts": true},'
                 ' {"grade": 2, "stages": ["assessment"], "dropped_counts": true},'
                 ' {"grade": 3, "stages": ["interview"], "dropped_counts": true},'
                 ' {"grade": 4, "stages": ["offer"]}], "terminal_stage": "offer"}')
    f = get_funnel(str(p))
    assert f.grade(app(("applied", P), ("assessment", D))).grade == 2


def test_bad_funnel_is_rejected():
    with pytest.raises(ValueError):
        FunnelSpec.from_dict({"name": "x", "stages": ["a"], "levels": [{"grade": 1, "stages": ["b"]}]})
