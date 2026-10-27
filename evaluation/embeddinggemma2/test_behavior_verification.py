import json

import pytest

from app.inference.contracts import VerifyRequest
from inference_worker.vlm_service import VariantVLMService
from . import worker
from .behavior_factorial import spans, choose_arm
from .analyse_behavior_factorial import localized_attempts, evidence_hits


def verdict(**changes):
    return dict(event_present=True, start_frame_index=0, end_frame_index=1, confidence=.6,
        reason="Arm swings toward person between frames 0 and 1", action_motion="punch",
        contact_observed="unclear", receiver_response="steps backward", uncertainty="Contact occluded", **changes)


def test_motion_evidence_does_not_require_invented_contact():
    raw = verdict()
    result = worker.checked_verdict({"raw_text": json.dumps(raw)}, 2, "behavior-v2")
    assert result["event_present"] and result["contact_observed"] == "unclear"
    assert not result["confidence_calibrated"]


@pytest.mark.parametrize("change", [dict(action_motion="none"), dict(contact_observed="probably"),
                                    dict(receiver_response=""), dict(uncertainty=None)])
def test_invalid_or_contradictory_evidence_is_unknown(change):
    with pytest.raises(ValueError, match="behavior evidence"):
        worker.checked_verdict({"raw_text": json.dumps({**verdict(), **change})}, 2, "behavior-v2")


def test_behavior_repair_retains_fact_questions_and_semantics(monkeypatch):
    service = VariantVLMService(); prompts = []
    def generate(request, prompt, budget):
        prompts.append(prompt)
        return "bad" if len(prompts) == 1 else json.dumps(verdict())
    monkeypatch.setattr(service, "_generate", generate)
    result = service.verify(VerifyRequest(query="person punch motion", frame_paths=["a", "b"], timestamps=[0, 1],
        semantic_requirements=["exclude pointing"], verification_mode="behavior-v2"))
    assert result.event_present and len(prompts) == 2
    assert all("contact_observed" in text and "exclude pointing" in text and "not intent" in text for text in prompts)


def test_behavior_uses_timestamp_labels(monkeypatch):
    service = VariantVLMService(); seen = []
    monkeypatch.setattr(service, "_generate_frames", lambda paths, prompt, budget, frame_labels: seen.extend(frame_labels) or "{}")
    service._generate(VerifyRequest(query="motion", timestamps=[2, 2.25], frame_paths=["a", "b"],
                                  verification_mode="behavior-v2"), "prompt", 256)
    assert "2.000" in seen[0] and "2.250" in seen[1]


def test_grid_covers_full_window_and_short_tail():
    assert spans(0, 6) == [[0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]
    assert spans(10, 11) == [[10, 11]]
    assert spans(0, 5.5)[-1] == [3.5, 5.5]
    with pytest.raises(ValueError):
        spans(0, float("nan"))


def test_selection_rejects_false_positives_and_prefers_lower_cost():
    def arm(hits, extra, calls):
        return dict(metrics=dict(covered_events=hits, non_target_windows=extra), private_requests=calls, pending=0)
    arms = {"temporal-v1/uniform": arm(2, 0, 41), "temporal-v1/grid": arm(3, 0, 200),
            "behavior-v2/uniform": arm(4, 1, 41), "behavior-v2/grid": arm(4, 0, 150)}
    assert choose_arm(arms) == "behavior-v2/grid"
    arms["behavior-v2/uniform"] = arm(4, 0, 41)
    assert choose_arm(arms) == "behavior-v2/uniform"
    arms["behavior-v2/grid"] = arm(3, 0, 150)
    arms["behavior-v2/uniform"] = arm(3, 0, 41)
    assert choose_arm(arms) is None


def test_localization_does_not_claim_the_entire_input_window_as_frame_evidence():
    row = dict(bounds=[0, 6], payload=dict(timestamps=[0, 2, 4, 6]),
               reply=dict(status="verified", verdict=dict(event_present=True, start_frame_index=1, end_frame_index=2)))
    assert localized_attempts({"e": [row]}, False)["e"][0]["bounds"] == [2, 4]
    row["reply"]["verdict"]["end_frame_index"] = 99
    with pytest.raises(ValueError):
        localized_attempts({"e": [row]}, False)


def test_evidence_hit_requires_actual_frame_span_overlap():
    source = dict(annotations={"1": [dict(class_id=4, official_selected=True, bounds=[1, 3])]})
    rows = {"candidate-1-0": [dict(bounds=[4, 5], reply=dict(status="verified", verdict=dict(event_present=True)))]}
    assert evidence_hits(source, rows, [1])["events_with_evidence_overlap"] == 0
    rows["candidate-1-0"][0]["bounds"] = [2, 2.25]
    assert evidence_hits(source, rows, [1])["events_with_evidence_overlap"] == 1


@pytest.mark.parametrize("motion,positive,expected", [("punch", True, "verified"), ("uncertain", False, "unverified")])
def test_broker_separates_unknown_and_skips_caption_only_when_requested(monkeypatch, motion, positive, expected):
    from types import SimpleNamespace
    calls = []
    raw = {**verdict(), "action_motion": motion, "event_present": positive}
    def post(url, **kwargs):
        calls.append(kwargs["json"])
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"raw_text": json.dumps(raw)})
    monkeypatch.setattr(worker.requests, "post", post)
    monkeypatch.setattr(worker, "unload", lambda: None)
    monkeypatch.setattr(worker, "release_qwen", lambda: None)
    monkeypatch.setattr(worker, "telemetry", lambda: {})
    monkeypatch.setattr(worker, "ResourceSampler", lambda: SimpleNamespace(
        thread=SimpleNamespace(start=lambda: None), finish=lambda: {}))
    event = dict(id="e", kind="pushing_punching", frame_paths=["a", "b"], timestamps=[0, 1],
                 verification_mode="behavior-v2", generate_caption=False)
    result = worker.verify_batch(worker.VerifyBatch(events=[event]))["results"][0]
    assert result["status"] == expected and len(calls) == 1
    assert result["caption_status"] == "disabled_for_evaluation"
    assert calls[0]["semantic_requirements"] == worker.BEHAVIOR_REQUIREMENTS
