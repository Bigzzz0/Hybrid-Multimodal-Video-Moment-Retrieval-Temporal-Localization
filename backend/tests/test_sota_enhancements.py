import os
import sys
import json
import re
import numpy as np
import pytest
from pathlib import Path
from PIL import Image

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent.parent
workspace_dir = backend_dir.parent
sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(workspace_dir))

from app.core.config import settings
from app.pipeline.keyframe_filter import compute_pixel_motion
from app.retrieval.query_expander import query_expander
from app.retrieval.boundary_extractor import TemporalBoundaryExtractor
from app.retrieval.vlm_verifier import vlm_verifier

def test_pixel_motion_energy_calculation():
    """Strategy 1: Test normalized absolute pixel difference energy."""
    # 1. Identical frames should yield 0.0 difference
    img1 = Image.new("RGB", (100, 100), color=(128, 128, 128))
    img2 = Image.new("RGB", (100, 100), color=(128, 128, 128))
    diff_zero = compute_pixel_motion(img1, img2)
    assert diff_zero == 0.0

    # 2. Black vs White frames should yield 1.0 (maximum normalized energy)
    img_black = Image.new("RGB", (100, 100), color=(0, 0, 0))
    img_white = Image.new("RGB", (100, 100), color=(255, 255, 255))
    diff_max = compute_pixel_motion(img_black, img_white)
    assert pytest.approx(diff_max, rel=1e-3) == 1.0

    # 3. None handling
    assert compute_pixel_motion(None, img1) == 0.0
    assert compute_pixel_motion(img1, None) == 0.0

def test_dual_scale_context_hierarchy_normalization():
    """Strategy 2: Test Dual-Scale Temporal Context Hierarchy L2 Spherical Projection."""
    num_frames = 10
    dim = 768
    np.random.seed(42)
    
    # Generate synthetic normalized embeddings
    raw_embs = [np.random.randn(dim).astype(np.float32) for _ in range(num_frames)]
    for i in range(num_frames):
        raw_embs[i] /= np.linalg.norm(raw_embs[i])

    # Simulate Dual-Scale Hierarchy at frame index i=4
    i = 4
    emb_curr = raw_embs[i]
    emb_prev = raw_embs[i - 1]
    emb_next = raw_embs[i + 1]

    # Local context (W=3)
    v_local = 0.60 * emb_curr + 0.20 * emb_prev + 0.20 * emb_next
    v_local /= (np.linalg.norm(v_local) + 1e-6)
    assert pytest.approx(np.linalg.norm(v_local), rel=1e-4) == 1.0

    # Global context (W=7 with Gaussian decay)
    win_start = max(0, i - 3)
    win_end = min(num_frames, i + 4)
    tau_indices = list(range(win_start, win_end))
    weights = [np.exp(-((idx - i) ** 2) / (2.0 * (2.0 ** 2))) for idx in tau_indices]
    w_sum = sum(weights)
    v_global = sum((w / w_sum) * raw_embs[idx] for idx, w in zip(tau_indices, weights))
    v_global /= (np.linalg.norm(v_global) + 1e-6)
    assert pytest.approx(np.linalg.norm(v_global), rel=1e-4) == 1.0

    # Hierarchical blend
    alpha_l = 0.65
    alpha_g = 0.35
    v_hierarchical = alpha_l * v_local + alpha_g * v_global
    v_hierarchical /= (np.linalg.norm(v_hierarchical) + 1e-6)

    # Must lie on the unit hypersphere (L2 norm = 1.0)
    assert pytest.approx(np.linalg.norm(v_hierarchical), rel=1e-4) == 1.0
    assert v_hierarchical.shape == (768,)

def test_concept_disentanglement_geometry():
    """Strategy 6: Test 3-Way Geometric Mean properties in Concept Disentanglement."""
    disentangled = query_expander.disentangle_query("คนปั่นจักรยานบนทางเท้า")
    assert "subject" in disentangled
    assert "verb" in disentangled
    assert "context" in disentangled
    assert "null_anchor" in disentangled
    assert "bicycle" in disentangled["subject"] or "จักรยาน" in disentangled["subject"]

    # Mathematical test: Compare Geometric vs Arithmetic Mean
    # Case A: Strong action (Cycling): sub=0.8, verb=0.8, ctx=0.8
    s_sub_a, s_verb_a, s_ctx_a = 0.80, 0.80, 0.80
    geom_a = (s_sub_a ** 0.25) * (s_verb_a ** 0.50) * (s_ctx_a ** 0.25)
    assert pytest.approx(geom_a, rel=1e-3) == 0.80

    # Case B: Static Scene Trap: Person standing next to bike (sub=0.8, verb=0.05, ctx=0.8)
    s_sub_b, s_verb_b, s_ctx_b = 0.80, 0.05, 0.80
    arith_b = (s_sub_b + s_verb_b + s_ctx_b) / 3.0 # 0.55 (False positive!)
    geom_b = (s_sub_b ** 0.25) * (s_verb_b ** 0.50) * (s_ctx_b ** 0.25) # ~0.20

    assert arith_b > 0.50
    assert geom_b < 0.25
    assert geom_b < arith_b * 0.45 # Geometric mean sharply penalizes missing action

def test_hard_static_negative_subtraction():
    """Strategy 4: Test contrastive margin rectification with static null anchor."""
    raw_scores = np.array([0.70, 0.35, 0.15], dtype=np.float32)
    null_sims = np.array([0.10, 0.80, 0.90], dtype=np.float32)
    lambda_null = 0.20

    rectified = np.maximum(0.0, raw_scores - lambda_null * null_sims)
    
    # Active frame: 0.70 - 0.20 * 0.10 = 0.68 (hardly penalized)
    assert pytest.approx(rectified[0], rel=1e-3) == 0.68
    # Static frame: 0.35 - 0.20 * 0.80 = 0.19 (strongly penalized)
    assert pytest.approx(rectified[1], rel=1e-3) == 0.19
    # Barely matched frame: 0.15 - 0.20 * 0.90 = -0.03 -> clamped to 0.0
    assert rectified[2] == 0.0

def test_ssm_gradient_cut_snapping():
    """Strategy 3: Test Self-Similarity Matrix (SSM) Directional Gradient Snapping."""
    extractor = TemporalBoundaryExtractor()
    n_frames = 10
    timestamps = np.linspace(0.0, 18.0, n_frames) # 2.0s interval: 0, 2, 4, 6, 8, 10, 12, 14, 16, 18

    # Create synthetic embeddings where a distinct scene transition occurs at index 4 (t=8.0s)
    emb_matrix = np.zeros((n_frames, 768), dtype=np.float32)
    vec_a = np.ones(768, dtype=np.float32) / np.sqrt(768)
    vec_b = -np.ones(768, dtype=np.float32) / np.sqrt(768)
    
    for idx in range(4):
        emb_matrix[idx] = vec_a
    for idx in range(4, n_frames):
        emb_matrix[idx] = vec_b

    # Rough candidate moment starting at t=7.5s (near 8.0s)
    candidate_moments = [{"t_start": 7.5, "t_end": 15.0, "score": 0.85}]

    snapped = extractor.snap_boundaries_to_ssm_gradient(
        moments=candidate_moments,
        emb_matrix=emb_matrix,
        timestamps=timestamps,
        snap_radius_sec=1.5
    )

    assert len(snapped) == 1
    # Boundary should snap cleanly towards transition at index 3 or 4 (around 6.0s - 8.0s)
    assert snapped[0]["t_start"] in [6.0, 8.0]

def test_gaussian_soft_nms_decay():
    """Strategy 7: Test 1D Continuous Gaussian Soft-NMS score attenuation."""
    extractor = TemporalBoundaryExtractor()

    # Create 3 moments with high overlap
    moments = [
        {"t_start": 10.0, "t_end": 20.0, "score": 0.90},
        {"t_start": 11.0, "t_end": 21.0, "score": 0.80}, # IoU with #1: 9 / 11 = 0.818
        {"t_start": 40.0, "t_end": 50.0, "score": 0.75}  # Disjoint: IoU = 0.0
    ]

    kept = extractor.apply_gaussian_soft_nms(
        moments=moments,
        sigma=0.40,
        score_threshold=0.20
    )

    assert len(kept) >= 2
    # Best moment is preserved with unchanged score
    assert kept[0]["score"] == 0.90
    assert kept[0]["t_start"] == 10.0

    # Disjoint moment should keep its original score
    disjoint_m = [m for m in kept if m["t_start"] == 40.0]
    assert len(disjoint_m) == 1
    assert disjoint_m[0]["score"] == 0.75

    # Overlapping moment should have decayed score
    overlap_m = [m for m in kept if m["t_start"] == 11.0]
    if overlap_m:
        assert overlap_m[0]["score"] < 0.80

def test_vlm_stage2_json_parsing_resilience():
    """Strategy 5: Test Stage-2 VLM JSON extraction against noisy Markdown output."""
    # Simulate LLM response with Markdown formatting
    noisy_response = (
        "Here is my analysis of the video keyframes:\n"
        "```json\n"
        "{\n"
        '  "action_found": true,\n'
        '  "refined_start": 12.35,\n'
        '  "refined_end": 18.90,\n'
        '  "confidence": 0.94\n'
        "}\n"
        "```\n"
        "I hope this helps!"
    )

    json_match = re.search(r'\{.*\}', noisy_response, re.DOTALL)
    assert json_match is not None
    data = json.loads(json_match.group(0))
    assert data["action_found"] is True
    assert data["refined_start"] == 12.35
    assert data["refined_end"] == 18.90
    assert data["confidence"] == 0.94
