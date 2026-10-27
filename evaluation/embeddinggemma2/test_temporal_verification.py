import json
from types import SimpleNamespace

import pytest

from app.inference.contracts import VerifyRequest
from inference_worker.vlm_service import VariantVLMService, VLMBackendError


def test_labels_are_interleaved_with_their_exact_images(tmp_path, monkeypatch):
    import httpx
    service = VariantVLMService()
    service.server_process = SimpleNamespace(poll=lambda: None)
    service.server_port = 9999
    paths = []
    for i in range(2):
        path = tmp_path/f"{i}.jpg"; path.write_bytes(bytes([i])); paths.append(str(path))
    seen = []
    def post(url, **kwargs):
        seen.append(kwargs["json"])
        return SimpleNamespace(status_code=200, json=lambda: {"choices": [{"message": {"content": "{}"}}]})
    monkeypatch.setattr(httpx, "post", post)
    request = VerifyRequest(query="push", frame_paths=paths, timestamps=[2., 2.5], verification_mode="temporal-v1")
    service._generate(request, "decision", 256)
    content = seen[0]["messages"][0]["content"]
    assert [c["type"] for c in content] == ["text", "image_url", "text", "image_url", "text"]
    assert "index 0" in content[0]["text"] and "2.000" in content[0]["text"]
    assert "index 1" in content[2]["text"] and "2.500" in content[2]["text"]
    assert content[1]["image_url"] != content[3]["image_url"]
    service._generate(request.model_copy(update={"verification_mode": "legacy"}), "decision", 256)
    assert [c["type"] for c in seen[1]["messages"][0]["content"]] == ["image_url", "image_url", "text"]


def test_missing_temporal_frame_is_not_silently_dropped(tmp_path):
    service = VariantVLMService()
    with pytest.raises(VLMBackendError, match="every input image"):
        service._generate_frames([str(tmp_path/"missing.jpg")], "prompt", 256, frame_labels=["frame0"])
    with pytest.raises(VLMBackendError, match="aligned chronological"):
        service._generate(VerifyRequest(query="push", frame_paths=["a", "b"], timestamps=[1, 1],
                                       verification_mode="temporal-v1"), "prompt", 256)


def test_repair_retains_the_original_event_and_semantic_requirements(monkeypatch):
    service = VariantVLMService(); seen = []
    valid = json.dumps(dict(event_present=False, start_frame_index=0, end_frame_index=1, confidence=.7, reason="No contact"))
    def generate(request, prompt, budget):
        seen.append(prompt)
        return "bad" if len(seen) == 1 else valid
    monkeypatch.setattr(service, "_generate", generate)
    result = service.verify(VerifyRequest(query="unique-action", timestamps=[0, 1], frame_paths=["a", "b"],
        semantic_requirements=["another person only"], verification_mode="temporal-v1"))
    assert not result.event_present
    assert all("unique-action" in text and "another person only" in text for text in seen)
    assert "two distinct evidence frames" in seen[0]
