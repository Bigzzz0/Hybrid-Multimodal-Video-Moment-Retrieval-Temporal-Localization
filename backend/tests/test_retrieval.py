import os
import sys
from pathlib import Path

# Add backend and workspace root to path
backend_dir = Path(__file__).resolve().parent.parent
workspace_dir = backend_dir.parent
sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(workspace_dir))

import numpy as np
import pytest
from app.retrieval.rank_fusion import ReciprocalRankFusion
from app.retrieval.temporal_smoother import TemporalSmoother
from app.retrieval.boundary_extractor import TemporalBoundaryExtractor
from app.pipeline.dense_captioner import QwenVLDenseCaptioner
from evaluation.compute_metrics import compute_temporal_iou, evaluate_moment_retrieval

def test_reciprocal_rank_fusion():
    rrf = ReciprocalRankFusion(k=60)
    ranked_lists = {
        "visual": [{"id": "frame_1", "timestamp": 10.0}, {"id": "frame_2", "timestamp": 20.0}],
        "caption": [{"id": "frame_1", "timestamp": 10.0}],
        "audio": [{"id": "aud_1", "timestamp": 10.0}]
    }
    weights = {"visual": 0.5, "caption": 0.3, "audio": 0.2}
    fused = rrf.fuse(ranked_lists, weights)
    
    assert "frame_1" in fused
    assert fused["frame_1"]["fused_score"] > fused["frame_2"]["fused_score"]

def test_temporal_smoother():
    smoother = TemporalSmoother(default_sigma=1.5)
    duration = 30.0
    timestamp_scores = [(10.0, 0.9), (10.5, 0.95), (11.0, 0.85)]
    
    time_axis, smoothed = smoother.smooth_timeline(duration, timestamp_scores, sigma=1.5, resolution_hz=2)
    
    assert len(time_axis) == len(smoothed)
    assert np.max(smoothed) == 1.0 # normalized
    peak_time = time_axis[np.argmax(smoothed)]
    assert 9.0 <= peak_time <= 12.0

def test_temporal_boundary_extractor():
    extractor = TemporalBoundaryExtractor(threshold_factor=0.8)
    time_axis = np.linspace(0.0, 60.0, 120)
    scores = np.zeros_like(time_axis)
    # Simulate an event between 20s and 30s
    scores[40:60] = 1.0
    
    moments = extractor.extract_moments(time_axis, scores)
    assert len(moments) >= 1
    top_m = moments[0]
    assert top_m["t_start"] <= 21.0
    assert top_m["t_end"] >= 29.0

def test_temporal_iou_calculation():
    # Exact match
    assert compute_temporal_iou((10.0, 20.0), (10.0, 20.0)) == 1.0
    # Half overlap
    assert compute_temporal_iou((10.0, 20.0), (15.0, 25.0)) == pytest.approx(5.0 / 15.0)
    # Disjoint
    assert compute_temporal_iou((10.0, 20.0), (30.0, 40.0)) == 0.0

def test_evaluation_benchmark_metrics():
    preds = [[(10.0, 20.0)]]
    gts = [(10.0, 20.0)]
    results = evaluate_moment_retrieval(preds, gts, iou_thresholds=[0.5], top_ks=[1])
    assert results["R@1@IoU=0.5"] == 100.0
    assert results["mIoU"] == 1.0
    assert results["mean_delta_t_start_sec"] == 0.0

def test_qwen_vl_dense_captioner_init():
    captioner = QwenVLDenseCaptioner(device="cpu")
    assert captioner.model_id == "Qwen/Qwen2.5-VL-7B-Instruct"
    # Empty frames should safely return empty string
    empty_cap = captioner.generate_scene_caption([])
    assert empty_cap == ""
    # With dummy keyframe on cpu (disabled), should return fallback caption
    from PIL import Image
    dummy_img = Image.new("RGB", (64, 64), color="blue")
    cap = captioner.generate_scene_caption([dummy_img])
    assert "Scene showing keyframe visuals" in cap

def test_visual_query_decomposer_and_tta():
    from app.retrieval.query_expander import VisualQueryDecomposer
    decomposer = VisualQueryDecomposer()
    res = decomposer.expand_query("คนเดินไปหยิบแก้วน้ำ")
    assert len(res["visual_keywords"]) > 0
    assert len(res["action_keywords"]) > 0
    assert res["audio_keywords"] == [] # Audio explicitly removed

    tta_list = decomposer.get_tta_queries("คนเดินไปหยิบแก้วน้ำ")
    assert len(tta_list) >= 2
    assert tta_list[0][1] == 0.50 # Primary query weight

def test_wasserstein_and_omtg():
    extractor = TemporalBoundaryExtractor(threshold_factor=0.5)
    time_axis = np.linspace(0.0, 100.0, 200) # 0.5 sec resolution
    scores = np.zeros_like(time_axis)
    
    # Event 1 at 20s-30s
    scores[40:60] = 1.0
    # Event 2 at 70s-80s (Disjoint)
    scores[140:160] = 0.95
    
    moments = extractor.extract_moments(
        time_axis, scores, enable_wasserstein=True, nms_iou_threshold=0.25
    )
    # Should detect both disjoint events under OMTG
    assert len(moments) >= 2
    # Verify moments are ordered by score
    assert moments[0]["score"] >= moments[1]["score"]
    # Check boundaries match the synthetic events
    assert any(18.0 <= m["t_start"] <= 22.0 for m in moments)
    assert any(68.0 <= m["t_start"] <= 72.0 for m in moments)

