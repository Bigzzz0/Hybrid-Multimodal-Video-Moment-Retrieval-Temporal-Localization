import numpy as np
import pytest
from app.retrieval.rank_fusion import ReciprocalRankFusion
from app.retrieval.temporal_smoother import TemporalSmoother
from app.retrieval.boundary_extractor import TemporalBoundaryExtractor
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
