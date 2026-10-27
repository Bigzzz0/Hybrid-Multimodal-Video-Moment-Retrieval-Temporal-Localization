import numpy as np
import pytest

from .continuous_screening import calibrate, center, interval_iou, measure, reference_bank


def event(start=2., end=3., class_id=4, selected=True):
    return dict(id="event", bounds=[start, end], class_id=class_id, official_selected=selected)


def record(seq=13):
    feature = dict(bounds=np.array([[0., 6.], [3., 9.], [6., 12.]]),
                   video=np.array([[1., 0.], [0., 1.], [1., 0.]]))
    return seq, feature, dict(prompt_margin=np.array([.5, .8, .2]))


def test_coverage_and_iou_are_distinct_for_brief_events():
    r = measure([record()], "prompt_margin", .5, {13: [event()]})
    assert r["coverage_recall"] == 1
    assert r["temporal_iou_recall"] == 0
    assert r["selected_union_seconds"] == 9
    assert r["selected_windows"] == 2


def test_boundary_touch_is_not_overlap():
    assert interval_iou([0, 3], [3, 4]) == 0


def test_adaptive_view_union_is_independent_of_submission_order():
    feature = dict(bounds=np.array([[4., 6.], [0., 2.], [1., 5.]]))
    result = measure([(13, feature, dict(positive=np.ones(3)))], "positive", 0., {13: []})
    assert result["selected_union_seconds"] == 6.
    assert result["total_seconds"] == 6.


def test_no_candidate_views_still_report_missed_targets():
    result = measure([(13, dict(bounds=np.empty((0, 2))), dict(positive=np.array([])))],
                     "positive", 0., {13: [event()]})
    assert result["covered_events"] == 0 and result["events"] == 1
    assert result["selected_union_seconds"] == result["total_seconds"] == 0.


def test_calibration_preserves_event_coverage_then_minimizes_candidates():
    threshold, result = calibrate([record()], "prompt_margin", {13: [event()]})
    assert threshold == .5
    assert result["covered_events"] == 1
    assert result["selected_windows"] == 2


def test_test_recordings_cannot_select_thresholds():
    with pytest.raises(ValueError, match="validation only"):
        calibrate([record(17)], "prompt_margin", {17: [event()]})


def test_reference_bank_never_accepts_validation_features():
    with pytest.raises(ValueError, match="leakage"):
        reference_bank({13: record()[1]}, {13: [event()]})


def test_kicks_and_unscored_pushes_cannot_be_normal_references():
    feature = dict(bounds=np.array([[0., 1.], [2., 3.], [4., 5.], [6., 7.]]),
                   video=np.eye(4))
    labels = {1: [event(0, 1), event(2, 3, 2), event(4, 5, 5, False)]}
    bank, counts = reference_bank({1: feature}, labels)
    assert counts[1] == dict(positive=1, normal=1)
    assert np.allclose(bank[1], [[0., 0., 0., 1.]])


def test_center_constant_video_is_finite_and_has_no_direction():
    assert np.array_equal(center(np.ones((3, 8))), np.zeros((3, 8)))


def test_labels_other_than_scored_events_do_not_count_as_missed_targets():
    r = measure([record()], "prompt_margin", 2., {13: [event(selected=False), event(class_id=2)]})
    assert r["events"] == 0
    assert r["missed_events"] == []


def test_threshold_above_float32_maximum_selects_nothing():
    seq, feature, _ = record()
    scores = dict(prompt_margin=np.array([.5, .8, .2], dtype=np.float32))
    threshold = float(np.nextafter(float(scores["prompt_margin"].max()), np.inf))
    assert measure([(seq, feature, scores)], "prompt_margin", threshold, {13: [event()]})["selected_windows"] == 0


def test_cannot_calibrate_recall_without_target_events():
    with pytest.raises(ValueError, match="target events"):
        calibrate([record()], "prompt_margin", {13: []})


def test_crossfit_excludes_query_recording_from_both_reference_classes(monkeypatch):
    from . import crossfit_screening
    feature = dict(bounds=np.array([[0., 1.], [2., 3.]]), video=np.eye(2),
                   image=np.eye(2), image_windows=np.array([0, 1]))
    keys = []
    original = crossfit_screening.reference_bank
    def checked(features, labels):
        keys.append(set(features))
        return original(features, labels)
    monkeypatch.setattr(crossfit_screening, "reference_bank", checked)
    records = crossfit_screening.crossfit({1: feature, 2: feature}, {1: [event(0, 1)], 2: [event(0, 1)]},
                                        (np.array([[1., 0.]]), np.array([[0., 1.]])))
    assert keys == [{2}, {1}]
    assert [seq for seq, _, _ in records] == [1, 2]


def test_crossfit_refuses_test_features():
    from .crossfit_screening import crossfit
    with pytest.raises(ValueError, match="development recordings only"):
        crossfit({1: record()[1], 17: record()[1]}, {}, None)


def test_temporal_zoom_cannot_claim_an_event_outside_frames_shown():
    from .verify_continuous_candidates import summarize_policy
    source = dict(scores={"17": {"bounds": [[0., 6.]]}}, annotations={"17": [event(0., 1.)]})
    reply = dict(id="one", status="verified", verdict={"event_present": True}, calls=1)
    result = summarize_policy(source, [reply], {"one": dict(seq=17, window=0, bounds=[0., 6.])}, ["one"], "central", sequences=(17,))
    assert result["gemma_coverage"]["covered_events"] == 1
    assert result["positive_candidate_coverage"]["covered_events"] == 1
    assert result["positive_qwen_coverage"]["covered_events"] == 0


def test_failed_or_missing_qwen_replies_remain_unverified_not_negative():
    from .verify_continuous_candidates import summarize_policy
    source = dict(scores={"17": {"bounds": [[0., 6.], [3., 9.]]}}, annotations={"17": [event()]})
    locations = {"one": dict(seq=17, window=0, bounds=[0., 6.]), "two": dict(seq=17, window=1, bounds=[3., 9.])}
    result = summarize_policy(source, [dict(id="one", status="unverified", verdict=None)], locations, list(locations), "uniform", sequences=(17,))
    assert result["unverified"] == 2
    assert result["positive_qwen_coverage"]["covered_events"] == 0


def test_cascade_keeps_original_positive_without_counting_second_call():
    from .analyse_continuous_sampling import cascade
    yes = dict(status="verified", verdict={"event_present": True}, calls=2)
    no = dict(status="verified", verdict={"event_present": False}, calls=1)
    decisions, intervals, calls = cascade({"one": yes}, {"one": no}, ["one"], {"one": {"bounds": [0., 6.]}})
    assert decisions == {"one": "positive"}
    assert intervals == {"one": [0., 6.]}
    assert calls == 2


def test_cascade_does_not_erase_unverified_evidence_with_a_negative_retry():
    from .analyse_continuous_sampling import cascade
    pending = dict(status="unverified", verdict=None, calls=1)
    no = dict(status="verified", verdict={"event_present": False}, calls=1)
    decisions, intervals, calls = cascade({"one": pending}, {"one": no}, ["one"], {"one": {"bounds": [0., 6.]}})
    assert decisions == {"one": "pending"}
    assert intervals == {}
    assert calls == 2
