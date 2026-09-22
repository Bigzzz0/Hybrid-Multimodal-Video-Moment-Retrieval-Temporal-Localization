import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.inference.contracts import CaptionRequest
from app.inference.vlm_registry import get_variant
from app.retrieval.vlm_artifacts import VLMArtifactStore
from inference_worker.model_manager import ModelManager


def test_production_registry_only_exposes_q6():
    assert get_variant("caprl_qwen3vl_4b_q6").quantization == "Q6_K"
    with pytest.raises(ValueError):
        get_variant("qwen3_vl_2b")
    with pytest.raises(ValueError):
        get_variant("caprl_video_4b")


def test_caption_request_defaults_to_production_frame_contract():
    request = CaptionRequest(frame_paths=[r"C:\frames\scene.jpg"], timestamps=[2.0])
    assert request.vlm_backend == "caprl_qwen3vl_4b_q6"
    assert request.release_after is False
    assert request.max_new_tokens == 256


def test_partial_caption_rows_are_hidden_until_metadata_is_ready(monkeypatch):
    monkeypatch.setattr(
        VLMArtifactStore,
        "metadata",
        classmethod(lambda cls, video_id, backend: {"status": "running"}),
    )
    assert VLMArtifactStore.caption_rows("video-1", "caprl_qwen3vl_4b_q6") == []


def test_heavy_vlm_evicts_siglip_before_loading(monkeypatch):
    manager = ModelManager()

    class FakeService:
        def load(self):
            return None

        def unload(self):
            return None

    manager.services["siglip"] = FakeService()
    manager.states["siglip"] = "ready"
    monkeypatch.setattr(manager, "_free_vram_mb", lambda: 0.0)
    monkeypatch.setattr(manager, "_used_vram_mb", lambda: 0.0)

    manager.get("q6", FakeService)

    assert "siglip" not in manager.services
    assert manager.states["siglip"] == "unloaded"
    assert manager.states["q6"] == "ready"
