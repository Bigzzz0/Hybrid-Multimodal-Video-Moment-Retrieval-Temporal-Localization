"""Verify routing, truncation, OOM, and malformed Qwen responses without weights."""
import json
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from .runtime_encoder import RuntimeEncoder
from . import worker

def bare_encoder():
    encoder = object.__new__(RuntimeEncoder)
    encoder.output_dimension = 256
    encoder.inputs = []
    encoder.expected = {}
    return encoder

def test_prompts_image_options_and_mrl_normalization():
    encoder = bare_encoder()
    seen = []
    class Model:
        def encode(self, payload, **kwargs):
            seen.append(kwargs)
            assert kwargs["processing_kwargs"]["text"]["truncation"] is False
            return np.ones((len(payload), 768), dtype=np.float32)
    encoder.model = Model()
    assert encoder.encode_queries(["คำค้น"] ).shape == (1, 256)
    encoder.encode_captions(["caption"])
    values = encoder.encode_images([object()], 560)
    assert seen[0]["prompt_name"] == "SearchQuery"
    assert seen[1]["prompt_name"] == "Document"
    assert seen[2]["processing_kwargs"]["image"]["max_soft_tokens"] == 560
    np.testing.assert_allclose(np.linalg.norm(values, axis=1), 1, atol=1e-6)

def test_input_hook_rejects_dropped_frames_and_audio():
    encoder = bare_encoder()
    encoder.expected = {"frames": 3}
    with pytest.raises(ValueError, match="frames"):
        encoder._inspect(None, ({"input_ids": torch.zeros(1, 5), "pixel_values_videos": torch.zeros(2, 1)},))
    encoder.expected = {"audio": True}
    with pytest.raises(ValueError, match="Audio"):
        encoder._inspect(None, ({"input_ids": torch.zeros(1, 5)},))
    encoder.expected = {"image_tokens": 280}
    with pytest.raises(ValueError, match="Image"):
        encoder._inspect(None, ({"input_ids": torch.zeros(1, 5)},))
    with pytest.raises(ValueError, match="overflow"):
        encoder._inspect(None, ({"input_ids": torch.zeros(1, 8193)},))

@pytest.mark.parametrize("audio", [[], [[1, 2]], [np.nan], np.ones(100000)])
def test_audio_is_not_silently_truncated(audio):
    with pytest.raises(ValueError):
        RuntimeEncoder.check_audio(audio)

def test_legacy_valid_flag_cannot_accept_truncated_caption():
    with pytest.raises(json.JSONDecodeError):
        worker.checked_caption({"json_valid": True, "text": '{"summary":"real text", "actions":['})

def test_oom_unloads_and_returns_507(monkeypatch):
    cleared = []
    def out_of_memory(*args):
        raise torch.cuda.OutOfMemoryError("test allocation")
    monkeypatch.setattr(worker, "get_encoder", out_of_memory)
    monkeypatch.setattr(worker, "unload", lambda: cleared.append(True))
    monkeypatch.setattr(worker, "telemetry", lambda: {})
    result = TestClient(worker.app).post("/embed", json={"modality": "query", "texts": ["fall"]})
    assert result.status_code == 507 and result.json()["detail"]["code"] == "oom"
    assert cleared

def test_qwen_failure_is_unverified_and_releases_gpu(monkeypatch):
    released = []
    class Sampler:
        thread = type("Thread", (), {"start": lambda self: None})()
        def finish(self):
            return {}
    monkeypatch.setattr(worker, "ResourceSampler", Sampler)
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {"json_valid": True, "summary": "fallback", "text": '{"summary":"truncated"'}
    monkeypatch.setattr(worker.requests, "post", lambda *a, **k: Response())
    monkeypatch.setattr(worker, "unload", lambda: None)
    monkeypatch.setattr(worker, "release_qwen", lambda: released.append(True))
    monkeypatch.setattr(worker, "telemetry", lambda: {})
    result = TestClient(worker.app).post("/verify-batch", json={"events": [{"id": "e", "kind": "falling", "label": "fall", "frame_paths": ["a", "b"], "timestamps": [0, 1]}]})
    assert result.json()["results"][0]["status"] == "unverified"
    assert result.json()["results"][0]["verification_attempts"][0]["text"] == '{"summary":"truncated"'
    assert released

@pytest.mark.parametrize("change", [{"event_present": "false"}, {"start_frame_index": -1},
                                    {"end_frame_index": 9}, {"start_frame_index": 1, "end_frame_index": 1},
                                    {"confidence": float("nan")}, {"reason": ""}])
def test_partial_or_invalid_action_evidence_is_not_a_valid_verdict(change):
    raw = {"event_present": True, "start_frame_index": 0, "end_frame_index": 1, "confidence": .8, "reason": "Visible transition"}
    with pytest.raises(ValueError):
        worker.checked_verdict({"event_present": False, "raw_text": json.dumps({**raw, **change})}, 2)

def exercise_verification(monkeypatch, positive, captions, mode="legacy"):
    calls, released = [], []
    class Sampler:
        thread = type("Thread", (), {"start": lambda self: None})()
        def finish(self): return {}
    class Response:
        def __init__(self, data): self.data = data
        def raise_for_status(self): pass
        def json(self): return self.data
    def post(url, **kwargs):
        calls.append((url, kwargs["json"]))
        if url.endswith("/verify"):
            return Response({"raw_text": json.dumps({"event_present": positive, "start_frame_index": 0,
                            "end_frame_index": 1, "confidence": .8, "reason": "Observed motion"})})
        return Response(captions.pop(0))
    monkeypatch.setattr(worker, "ResourceSampler", Sampler)
    monkeypatch.setattr(worker.requests, "post", post)
    monkeypatch.setattr(worker, "unload", lambda: None)
    monkeypatch.setattr(worker, "release_qwen", lambda: released.append(True))
    monkeypatch.setattr(worker, "telemetry", lambda: {})
    result = worker.verify_batch(worker.VerifyBatch(events=[{"id": "e", "kind": "falling", "label": "fall",
                                   "frame_paths": ["a", "b"], "timestamps": [0, 1], "verification_mode": mode}]))["results"][0]
    assert released == [True]
    return result, calls

def test_negative_decision_does_not_generate_caption(monkeypatch):
    result, calls = exercise_verification(monkeypatch, False, [])
    assert result["status"] == "verified" and result["verdict"]["event_present"] is False
    assert result["calls"] == len(calls) == 1 and result["caption_status"] == "not_requested_negative"
    assert "semantic_requirements" in calls[0][1] and "caption_hint" not in calls[0][1]


def test_temporal_verification_mode_reaches_private_worker(monkeypatch):
    result, calls = exercise_verification(monkeypatch, False, [], mode="temporal-v1")
    assert result["status"] == "verified"
    assert calls[0][1]["verification_mode"] == "temporal-v1"


def test_unknown_verification_mode_does_not_call_model(monkeypatch):
    result, calls = exercise_verification(monkeypatch, False, [], mode="unknown")
    assert result["status"] == "unverified" and not calls

def test_push_requirements_distinguish_another_person_from_objects():
    assert any("another person's body" in text and "gate" in text for text in worker.ACTION_REQUIREMENTS["pushing_punching"])

def test_caption_retry_keeps_raw_attempts_and_counts_requests(monkeypatch):
    complete = {"summary": "Person falls", **{key: [] for key in ("objects", "attributes", "actions", "relations", "temporal_events", "uncertainty")}}
    result, calls = exercise_verification(monkeypatch, True, [{"text": '{"summary":"cut"'}, {"text": json.dumps(complete)}])
    assert result["status"] == "verified" and result["caption_status"] == "complete"
    assert result["calls"] == len(calls) == 3
    assert [item[1].get("max_new_tokens") for item in calls] == [256, 384, 768]
    assert result["caption_attempts"][0]["raw_text"] == '{"summary":"cut"'

def test_caption_failure_does_not_erase_valid_action_verification(monkeypatch):
    result, calls = exercise_verification(monkeypatch, True, [{"text": "bad"}, {"text": "still bad"}])
    assert result["status"] == "verified" and result["verdict"]["event_present"] is True
    assert result["caption_status"] == "failed" and result["caption"] == {} and len(calls) == 3
