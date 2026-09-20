import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.retrieval.boundary_extractor import TemporalBoundaryExtractor
from app.retrieval.vlm_verifier import QwenVisualReranker
from app.retrieval.search_engine import HybridMomentSearchEngine
from app.retrieval.query_expander import VisualQueryDecomposer


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


def test_qwen_parser_accepts_markdown_and_optional_explanation():
    raw = "Result:\n```json\n{\"action_present\": true, \"start_frame_index\": \"0\", \"end_frame_index\": 1, \"confidence\": \"0.8\", \"reason\": \"visible\"}\n```"
    parsed = QwenVisualReranker._parse_response(raw, [1.0, 2.0])
    assert parsed == {
        "action_present": True,
        "start_frame_index": 0,
        "end_frame_index": 1,
        "confidence": 0.8,
        "reason": "visible",
    }


def test_qwen_parser_maps_exact_timestamp_indices_from_model_output():
    raw = "`json {\"action_present\": false, \"start_frame_index\": 24.5, \"end_frame_index\": 29.0, \"confidence\": 0.8}`"
    parsed = QwenVisualReranker._parse_response(raw, [21.5, 24.5, 27.0, 29.0])
    assert parsed["start_frame_index"] == 1
    assert parsed["end_frame_index"] == 3


def test_vehicle_passing_query_expands_to_vehicle_motion():
    expanded = VisualQueryDecomposer().expand_query("รถยนต์วิ่งผ่านถนน")
    assert "car driving past" in expanded["visual_keywords"]
    assert "road" in expanded["visual_keywords"]


def test_qwen_invalid_json_uses_fast_candidate(tmp_path):
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    Image.new("RGB", (8, 8), "black").save(first)
    Image.new("RGB", (8, 8), "white").save(second)

    class FakeCaptioner:
        kwargs = None

        def generate_scene_caption(self, *args, **kwargs):
            self.kwargs = kwargs
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
    assert reranker._captioner.kwargs["max_new_tokens"] == 96


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
    class TimeoutCaptioner:
        def generate_scene_caption(self, *args, **kwargs):
            return "{}"
    timeout_reranker._captioner = TimeoutCaptioner()
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


def test_multiscale_nested_proposals_collapse_and_keep_context():
    scored = [
        (0.82, {"t_start": 23.5, "t_end": 32.0, "score": 0.9}),
        (0.84, {"t_start": 26.5, "t_end": 29.0, "score": 0.8}),
        (0.71, {"t_start": 45.0, "t_end": 48.0, "score": 0.7}),
    ]
    grouped = HybridMomentSearchEngine._group_candidates(scored)
    assert len(grouped) == 2
    primary = max(grouped, key=lambda item: item[0])[1]
    assert (primary["t_start"], primary["t_end"]) == (26.5, 29.0)
    assert (primary["context_t_start"], primary["context_t_end"]) == (23.5, 32.0)


def test_disjoint_repeated_events_remain_separate_and_deterministic():
    scored = [
        (0.8, {"t_start": 20.0, "t_end": 22.0}),
        (0.8, {"t_start": 40.0, "t_end": 42.0}),
    ]
    grouped = HybridMomentSearchEngine._group_candidates(scored)
    assert [(item[1]["t_start"], item[1]["t_end"]) for item in grouped] == [(20.0, 22.0), (40.0, 42.0)]
