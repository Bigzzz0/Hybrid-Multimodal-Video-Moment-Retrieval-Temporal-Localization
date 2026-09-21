from app.db.schemas import SearchQueryRequest
from app.inference.contracts import CaptionRequest, VerifyRequest
from app.inference.vlm_registry import all_variants, get_variant
from app.retrieval.vlm_verifier import QwenVisualReranker
from inference_worker.vlm_service import VariantVLMService


def test_registry_has_pinned_a_to_e_variants():
    variants = all_variants()
    assert [variant.backend for variant in variants] == [
        "qwen3_vl_2b", "qwen3_vl_2b_vise", "caprl_qwen3vl_2b",
        "caprl_qwen3vl_4b_q4", "caprl_qwen3vl_4b_q6", "caprl_video_4b",
    ]
    assert get_variant("qwen3_vl_2b").revision
    assert get_variant("qwen3_vl_2b_vise").adapter_id == "shravvvv/VISE"
    assert get_variant("caprl_video_4b").input_mode == "video_chunk"


def test_request_defaults_are_backward_compatible_and_variant_aware():
    request = SearchQueryRequest(query="cat", video_id="video")
    assert request.vlm_backend == "qwen3_vl_2b"
    caption = CaptionRequest(frame_paths=["frame.jpg"], vlm_backend="caprl_qwen3vl_2b", release_after=True)
    assert caption.vlm_backend == "caprl_qwen3vl_2b"
    assert caption.release_after is True
    verify = VerifyRequest(query="person walking", vlm_backend="caprl_video_4b", video_path="video.mp4", max_frames=16)
    assert verify.max_frames == 16


def test_verification_cache_key_isolated_by_backend():
    candidate = {"t_start": 1.0, "t_end": 2.0}
    first = QwenVisualReranker._cache_key("video", candidate, "cat", [1.0, 1.5], "qwen3_vl_2b")
    second = QwenVisualReranker._cache_key("video", candidate, "cat", [1.0, 1.5], "caprl_qwen3vl_2b")
    assert first != second


def test_caption_parser_handles_fenced_and_truncated_json():
    fence = chr(96) * 3
    fenced = fence + 'json\n{"summary":"a cat","objects":["cat"]}\n' + fence
    assert VariantVLMService._extract_json(fenced)["objects"] == ["cat"]

    truncated = '{"summary":"a cat","objects":["cat","statue"'
    recovered = VariantVLMService._extract_partial_caption(truncated)
    assert recovered["summary"] == "a cat"
    assert recovered["objects"] == ["cat", "statue"]
