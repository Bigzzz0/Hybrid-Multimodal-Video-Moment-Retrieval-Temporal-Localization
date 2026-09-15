import sys
from pathlib import Path

import numpy as np
import pytest

backend_dir = Path(__file__).resolve().parent.parent
workspace_dir = backend_dir.parent
sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(workspace_dir))

from app.db.schemas import SearchQueryRequest
from app.retrieval.boundary_extractor import TemporalBoundaryExtractor
from app.retrieval.query_expander import VisualQueryDecomposer
from app.retrieval.rank_fusion import ReciprocalRankFusion
from app.retrieval.temporal_smoother import TemporalSmoother
from app.retrieval.vlm_verifier import QwenVisualReranker
from evaluation.compute_metrics import compute_temporal_iou, evaluate_moment_retrieval, temporal_f1


def test_rrf_visual_caption_only_and_deterministic_tie_break():
    fused = ReciprocalRankFusion(k=60).fuse(
        {"visual": [{"id": "f1", "timestamp": 10.0}, {"id": "f2", "timestamp": 20.0}],
         "caption": [{"id": "f1", "timestamp": 10.0}]},
        {"visual": 0.8, "caption": 0.2},
    )
    assert fused["f1"]["fused_score"] > fused["f2"]["fused_score"]


def test_temporal_smoother_preserves_raw_scale():
    axis, scores = TemporalSmoother().smooth_timeline(30.0, [(10.0, 0.9), (10.5, 0.95), (11.0, 0.85)], resolution_hz=2)
    assert len(axis) == len(scores)
    assert float(np.max(scores)) < 1.0
    assert 9.0 <= float(axis[np.argmax(scores)]) <= 12.0


def test_multiscale_proposals_keep_repeated_disjoint_events():
    axis = np.linspace(0.0, 100.0, 201)
    scores = np.zeros_like(axis)
    scores[(axis >= 20) & (axis <= 30)] = 1.0
    scores[(axis >= 70) & (axis <= 80)] = 0.95
    proposals = TemporalBoundaryExtractor().extract_multiscale_proposals(axis, scores)
    assert len(proposals) >= 2
    assert any(18 <= p["t_start"] <= 22 for p in proposals)
    assert any(68 <= p["t_start"] <= 72 for p in proposals)


def test_temporal_metrics_support_multiple_ground_truth_intervals():
    preds = [[(1.0, 3.0), (10.0, 12.0), (30.0, 31.0)]]
    gts = [[(1.0, 3.0), (10.0, 12.0)]]
    results = evaluate_moment_retrieval(preds, gts, iou_thresholds=[0.5], top_ks=[1, 5])
    assert temporal_f1(preds[0], gts[0], iou_threshold=0.5) == pytest.approx(0.8)
    assert results["tF1@IoU=0.5"] == 80.0
    assert results["count_accuracy"] == 0.0


def test_query_expansion_is_deterministic_and_weighted():
    decomposer = VisualQueryDecomposer()
    first = decomposer.get_query_variants("คนเดินไปหยิบแก้วน้ำ")
    second = decomposer.get_query_variants("คนเดินไปหยิบแก้วน้ำ")
    assert first == second
    assert first[0][1] == pytest.approx(0.60)
    assert sum(weight for _, weight in first) == pytest.approx(1.0)


def test_api_contract_requires_video_and_bounds_top_k():
    request = SearchQueryRequest(query="person walking", video_id="video-1", top_k=20, profile="accurate")
    assert request.video_id == "video-1"
    with pytest.raises(ValueError):
        SearchQueryRequest(query="x", video_id="v", top_k=21)


def test_qwen_json_validation_uses_frame_indices_only():
    parsed = QwenVisualReranker._parse_response(
        '{"action_present": true, "start_frame_index": 0, "end_frame_index": 1, "confidence": 0.8}',
        [1.0, 2.0],
    )
    assert parsed is not None
    assert QwenVisualReranker._parse_response(
        '{"action_present": true, "start_frame_index": 0, "end_frame_index": 9, "confidence": 0.8}',
        [1.0, 2.0],
    ) is None


def test_temporal_iou():
    assert compute_temporal_iou((10.0, 20.0), (10.0, 20.0)) == 1.0
    assert compute_temporal_iou((10.0, 20.0), (30.0, 40.0)) == 0.0


def test_no_match_and_count_metrics_are_reported():
    results = evaluate_moment_retrieval(
        predictions=[[], [(1.0, 2.0)]],
        ground_truths=[[], [(1.0, 2.0), (4.0, 5.0)]],
        iou_thresholds=[0.5],
        top_ks=[1, 5],
    )
    assert results["no_match_precision"] == 1.0
    assert results["no_match_recall"] == 1.0
    assert "count_accuracy" in results
