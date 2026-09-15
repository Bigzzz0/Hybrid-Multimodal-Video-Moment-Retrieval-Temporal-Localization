import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.retrieval.boundary_extractor import TemporalBoundaryExtractor
from app.retrieval.vlm_verifier import QwenVisualReranker


def test_boundary_refinement_clamps_to_observed_timeline():
    axis = np.linspace(0.0, 10.0, 21)
    scores = np.zeros_like(axis)
    scores[8:13] = 1.0
    proposals = TemporalBoundaryExtractor().extract_multiscale_proposals(axis, scores)
    assert proposals
    assert all(0.0 <= item["t_start"] <= item["t_end"] <= 10.0 for item in proposals)


def test_energy_quantile_refinement_is_robust_and_named():
    low, high = TemporalBoundaryExtractor.energy_quantile_refinement([0.0, 0.1, 0.2, 1.0])
    assert 0.0 <= low <= high <= 1.0


def test_qwen_invalid_json_falls_back_without_fake_boundary():
    assert QwenVisualReranker._parse_response("not json", [1.0, 2.0]) is None


def test_qwen_invalid_json_uses_fast_candidate(tmp_path):
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    Image.new("RGB", (8, 8), "black").save(first)
    Image.new("RGB", (8, 8), "white").save(second)

    class FakeCaptioner:
        def generate_scene_caption(self, *args, **kwargs):
            return "not-json"

    reranker = QwenVisualReranker()
    reranker._captioner = FakeCaptioner()
    candidate = {"t_start": 1.0, "t_end": 2.0, "score": 0.7, "rank_score": 0.7}
    frames = [
        {"timestamp": 1.0, "frame_path": str(first)},
        {"timestamp": 2.0, "frame_path": str(second)},
    ]
    result = reranker.verify([candidate], frames, "person walking")
    assert result[0]["t_start"] == 1.0
    assert "verifier_invalid_json_fallback" in reranker.last_warnings


def test_qwen_timeout_and_oom_fallbacks_do_not_raise(tmp_path):
    frame_path = tmp_path / "frame.jpg"
    Image.new("RGB", (8, 8), "black").save(frame_path)
    candidate = {"t_start": 1.0, "t_end": 2.0, "score": 0.7, "rank_score": 0.7}
    frames = [
        {"timestamp": 1.0, "frame_path": str(frame_path)},
        {"timestamp": 2.0, "frame_path": str(frame_path)},
    ]

    timeout_reranker = QwenVisualReranker()
    timeout_reranker.max_seconds = 0.0
    assert timeout_reranker.verify([candidate], frames, "walking") == [candidate]
    assert "verifier_timeout_fallback" in timeout_reranker.last_warnings

    class OutOfMemoryError(RuntimeError):
        pass

    class OOMCaptioner:
        def generate_scene_caption(self, *args, **kwargs):
            raise OutOfMemoryError("fake CUDA OOM")

    oom_reranker = QwenVisualReranker()
    oom_reranker._captioner = OOMCaptioner()
    assert oom_reranker.verify([candidate], frames, "walking") == [candidate]
    assert "verifier_oom_fallback" in oom_reranker.last_warnings
