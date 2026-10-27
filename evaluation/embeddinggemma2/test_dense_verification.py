import pytest
from .validate_dense_verification import halves, decision, metrics


def row(present=False, status="verified", bounds=(0., 3.)):
    return dict(bounds=bounds, reply=dict(status=status, verdict=dict(event_present=present) if status == "verified" else None))


def test_halves_cover_tail_without_gap():
    assert halves(9., 10.) == [(9., 9.5), (9.5, 10.)]
    for a, b in [(1., 1.), (2., 1.), (0., float("inf"))]:
        with pytest.raises(ValueError):
            halves(a, b)


def test_positive_survives_and_failure_remains_pending():
    assert decision([row(True)]) == "positive"
    assert decision([row(), row(), row(status="unverified")]) == "pending"
    assert decision([row(), row()]) == "pending"
    assert decision([row(), row(), row()]) == "negative"


def test_dense_metric_cannot_claim_original_six_seconds():
    source = dict(annotations={"13": [dict(bounds=[4., 5.], class_id=4, official_selected=True)]})
    attempts = {"dense-ut-13-0": [row(False, bounds=(0., 6.)), row(True, bounds=(0., 3.))]}
    assert metrics(source, attempts, (13,), True)["covered_events"] == 0
