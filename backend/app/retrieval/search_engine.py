import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from app.core.config import settings
from app.core.logger import logger
from app.db.connection import db_manager
from app.db.schemas import MomentItem, SearchResponse
from app.pipeline.visual_encoder import SigLIP2VisualEncoder
from app.retrieval.rank_fusion import ReciprocalRankFusion
from app.retrieval.temporal_smoother import TemporalSmoother
from app.retrieval.boundary_extractor import TemporalBoundaryExtractor
from app.retrieval.query_expander import query_expander
from app.retrieval.vlm_verifier import vlm_verifier

class HybridMomentSearchEngine:
    """
    SOTA Unified Moment Retrieval Engine with Cross-Modal Query Expansion,
    Multi-Scale Gaussian Temporal Pyramid, and Adaptive Valley Boundary Extraction.
    """

    def __init__(self):
        self.text_encoder = SigLIP2VisualEncoder()
        self.rrf = ReciprocalRankFusion(k=settings.DEFAULT_RRF_K)
        self.smoother = TemporalSmoother(default_sigma=settings.TEMPORAL_GAUSSIAN_SIGMA)
        self.boundary_extractor = TemporalBoundaryExtractor()

    def search_moments(
        self,
        query: str,
        video_id: Optional[str] = None,
        top_k: int = 5,
        weight_visual: float = settings.DEFAULT_WEIGHT_VISUAL,
        weight_caption: float = settings.DEFAULT_WEIGHT_CAPTION,
        weight_audio: float = settings.DEFAULT_WEIGHT_AUDIO,
        gaussian_sigma: float = settings.TEMPORAL_GAUSSIAN_SIGMA,
        threshold_factor: float = settings.DYNAMIC_THRESHOLD_FACTOR
    ) -> SearchResponse:
        """
        Executes multi-modal SOTA retrieval and returns timestamped moments with dynamic density heatmap.
        """
        t0 = time.time()
        logger.info(f"Executing SOTA Moment Search for query: '{query}' (video_id: {video_id})")

        # 1. Fetch Video Metadata from LanceDB
        tbl_videos = db_manager.get_table("videos")
        try:
            if video_id:
                video_records = tbl_videos.search().where(f"id = '{video_id}'").limit(1).to_list()
            else:
                video_records = tbl_videos.to_arrow().to_pylist()
        except Exception:
            video_records = [r for r in tbl_videos.to_arrow().to_pylist() if not video_id or r.get("id") == video_id]

        if not video_records:
            logger.warning(f"No video found for search with video_id: {video_id}")
            return SearchResponse(
                query=query,
                video_id=video_id,
                moments=[],
                timeline_heatmap=[],
                total_duration=0.0,
                latency_ms=0.0,
                top_k=top_k
            )

        target_video = video_records[0]
        actual_video_id = target_video.get("id")
        duration_sec = float(target_video.get("duration_sec", 10.0))

        # 2. Fetch all frames for this video
        tbl_frames = db_manager.get_table("video_frames")
        try:
            video_frames = tbl_frames.search().where(f"video_id = '{actual_video_id}'").limit(5000).to_list()
        except Exception:
            video_frames = [r for r in tbl_frames.to_arrow().to_pylist() if r.get("video_id") == actual_video_id]

        # 3. Disentangle Query into Subject, Verb, Context, and Static Null Anchor
        expanded = query_expander.expand_query(query)
        visual_keywords = expanded["visual_keywords"]
        audio_keywords = expanded["audio_keywords"]

        if getattr(settings, "ENABLE_CONCEPT_DISENTANGLEMENT", True):
            disentangled = query_expander.disentangle_query(query)
            vec_sub = np.array(self.text_encoder.encode_text(disentangled["subject"]), dtype=np.float32)
            vec_verb = np.array(self.text_encoder.encode_text(disentangled["verb"]), dtype=np.float32)
            vec_ctx = np.array(self.text_encoder.encode_text(disentangled["context"]), dtype=np.float32)
            vec_null = np.array(self.text_encoder.encode_text(disentangled["null_anchor"]), dtype=np.float32)

            for v in [vec_sub, vec_verb, vec_ctx, vec_null]:
                nrm = np.linalg.norm(v)
                if nrm > 0:
                    v /= nrm
        else:
            vec_sub = vec_verb = vec_ctx = vec_null = None

        # 4. Standard/TTA Query Encoding (for fallback or general query representation)
        if getattr(settings, "ENABLE_TTA_ENSEMBLE", True):
            tta_items = query_expander.get_tta_queries(query)  # List of (q_str, weight)
            accum_vec = np.zeros(768, dtype=np.float32)
            total_weight = 0.0
            for q_str, w in tta_items:
                v = np.array(self.text_encoder.encode_text(q_str), dtype=np.float32)
                vn = np.linalg.norm(v)
                if vn > 0:
                    v = v / vn
                accum_vec += w * v
                total_weight += w
            if total_weight > 0:
                accum_vec /= total_weight
            q_norm = np.linalg.norm(accum_vec)
            query_vec = accum_vec / q_norm if q_norm > 0 else accum_vec
        else:
            query_vec = np.array(self.text_encoder.encode_text(expanded["expanded_search_str"]), dtype=np.float32)
            q_norm = np.linalg.norm(query_vec)
            if q_norm > 0:
                query_vec = query_vec / q_norm

        # 5. Extract Frame Embeddings, Timestamps, and Pixel Motion
        frame_embs = []
        frame_meta = []
        for f in video_frames:
            emb = f.get("siglip2_vector")
            if emb is not None and len(emb) == 768:
                frame_embs.append(emb)
                frame_meta.append(f)

        timestamp_score_map: Dict[float, float] = {}

        if frame_embs:
            emb_matrix = np.array(frame_embs, dtype=np.float32)  # [N, 768]
            timestamps = np.array([float(f.get("timestamp", 0.0)) for f in frame_meta], dtype=np.float32)
            pixel_motions = np.array([
                float(f.get("pixel_motion", 0.0) if "pixel_motion" in f and f["pixel_motion"] is not None else 0.0)
                for f in frame_meta
            ], dtype=np.float32)

            sim_full = np.maximum(0.0, np.dot(emb_matrix, query_vec))

            # Strategy 6: Concept Disentanglement Scoring (3-Way Geometric Mean)
            if getattr(settings, "ENABLE_CONCEPT_DISENTANGLEMENT", True) and vec_sub is not None:
                sim_sub = np.maximum(0.0, np.dot(emb_matrix, vec_sub))
                sim_verb = np.maximum(0.0, np.dot(emb_matrix, vec_verb))
                sim_ctx = np.maximum(0.0, np.dot(emb_matrix, vec_ctx))
                w_sub = getattr(settings, "DISENTANGLE_WEIGHT_SUB", 0.25)
                w_verb = getattr(settings, "DISENTANGLE_WEIGHT_VERB", 0.50)
                w_ctx = getattr(settings, "DISENTANGLE_WEIGHT_CTX", 0.25)
                disentangled_scores = ((sim_sub + 1e-6) ** w_sub) * ((sim_verb + 1e-6) ** w_verb) * ((sim_ctx + 1e-6) ** w_ctx)
                blended_scores = 0.50 * sim_full + 0.50 * disentangled_scores
            else:
                blended_scores = sim_full

            # Strategy 4: Hard Static Negative Anchor Subtraction
            if getattr(settings, "ENABLE_STATIC_NEGATIVE", True) and vec_null is not None:
                sim_null = np.maximum(0.0, np.dot(emb_matrix, vec_null))
                lambda_null = getattr(settings, "STATIC_NEGATIVE_LAMBDA", 0.20)
                rectified_scores = np.maximum(0.0, blended_scores - lambda_null * sim_null)
            else:
                rectified_scores = blended_scores

            # Strategy 1: Pixel Motion Energy Gating Multiplier
            if getattr(settings, "ENABLE_PIXEL_MOTION_GATE", True) and len(pixel_motions) == len(rectified_scores):
                gamma_motion = getattr(settings, "PIXEL_MOTION_WEIGHT", 0.35)
                beta_slope = getattr(settings, "PIXEL_MOTION_TANH_BETA", 2.5)
                motion_multiplier = 1.0 + gamma_motion * np.tanh(beta_slope * pixel_motions)
                gated_scores = rectified_scores * motion_multiplier
            else:
                gated_scores = rectified_scores

            for f_data, s in zip(frame_meta, gated_scores):
                ts = float(f_data.get("timestamp", 0.0))
                timestamp_score_map[ts] = float(s) * weight_visual

            # Action Caption Keyword & Semantic Boosting
            action_keywords = expanded.get("action_keywords", [])
            if weight_caption > 0:
                for f_data in video_frames:
                    caption = (f_data.get("vlm_caption") or "").lower()
                    ts = float(f_data.get("timestamp", 0.0))
                    if caption and visual_keywords:
                        match_count = sum(1 for kw in visual_keywords if kw.lower() in caption)
                        if match_count > 0:
                            boost = (match_count / max(1, len(visual_keywords))) * weight_caption
                            if action_keywords and any(act.lower() in caption for act in action_keywords):
                                boost *= 1.25
                            timestamp_score_map[ts] = timestamp_score_map.get(ts, 0.0) + boost

        # 6. Build (Timestamp, Score) List
        timestamp_scores: List[Tuple[float, float]] = list(timestamp_score_map.items())
        if not timestamp_scores:
            timestamp_scores = [(0.0, 0.1)]

        # 7. SOTA Multi-Scale 1D Gaussian Temporal Pyramid
        time_axis, smoothed_scores = self.smoother.smooth_timeline(
            duration_sec=duration_sec,
            timestamp_scores=timestamp_scores,
            sigma=gaussian_sigma,
            resolution_hz=2,
            use_multiscale=True
        )

        # 8. Absolute Confidence Floor & Dynamic Contrast Calibration
        min_s, max_s = float(np.min(smoothed_scores)), float(np.max(smoothed_scores))
        
        if max_s < 0.12:
            logger.info(f"Query '{query}' max relevance ({max_s:.3f}) below confidence floor (0.12). No match.")
            return SearchResponse(
                query=query,
                video_id=actual_video_id,
                moments=[],
                timeline_heatmap=[round(float(s), 3) for s in smoothed_scores[::2]],
                total_duration=round(duration_sec, 2),
                latency_ms=round((time.time() - t0) * 1000.0, 2),
                top_k=top_k
            )

        if max_s > min_s:
            contrast_smoothed = (smoothed_scores - min_s) / (max_s - min_s)
            raw_peak_factor = min(1.0, max(0.40, max_s / 0.35))
        else:
            contrast_smoothed = smoothed_scores
            raw_peak_factor = min(1.0, max(0.40, max_s / 0.35))

        # 9. Adaptive Valley Boundary Extraction with 1D Wasserstein Snapping & OMTG
        candidate_moments = self.boundary_extractor.extract_moments(
            time_axis=time_axis,
            smoothed_scores=contrast_smoothed,
            threshold_factor=threshold_factor,
            enable_wasserstein=True,
            nms_iou_threshold=0.25
        )

        # Strategy 3: SSM Gradient Boundary Snapping
        if getattr(settings, "ENABLE_SSM_BOUNDARY_SNAP", True) and frame_embs and len(emb_matrix) >= 4:
            snapped_moments = self.boundary_extractor.snap_boundaries_to_ssm_gradient(
                moments=candidate_moments,
                emb_matrix=emb_matrix,
                timestamps=timestamps,
                snap_radius_sec=getattr(settings, "SSM_SNAP_WINDOW_SEC", 1.5)
            )
        else:
            snapped_moments = candidate_moments

        # Strategy 7: 1D Continuous Gaussian Soft-NMS
        if getattr(settings, "ENABLE_GAUSSIAN_SOFT_NMS", True) and snapped_moments:
            nms_moments = self.boundary_extractor.apply_gaussian_soft_nms(
                moments=snapped_moments,
                sigma=getattr(settings, "GAUSSIAN_SOFT_NMS_SIGMA", 0.40),
                score_threshold=getattr(settings, "GAUSSIAN_SOFT_NMS_FLOOR", 0.20)
            )
        else:
            nms_moments = snapped_moments

        # Strategy 5: Two-Stage VLM Temporal Verification & Endpoint Snapping (TimeLens CVPR 2026)
        if getattr(settings, "ENABLE_VLM_STAGE2_VERIFY", False) and nms_moments:
            extracted_moments = vlm_verifier.verify_and_refine(
                candidate_moments=nms_moments,
                video_frames=video_frames,
                query=query,
                top_k_verify=getattr(settings, "VLM_VERIFY_TOP_K", 3)
            )
        else:
            extracted_moments = nms_moments

        # 10. Hydrate Moment Items with Previews
        moments_response: List[MomentItem] = []
        for m in extracted_moments[:top_k]:
            t_mid = (m["t_start"] + m["t_end"]) / 2.0
            
            # Find closest keyframe
            closest_frame = None
            closest_caption = None
            min_dist = 999.0
            for f in video_frames:
                dist = abs(float(f.get("timestamp", 0.0)) - t_mid)
                if dist < min_dist:
                    min_dist = dist
                    closest_frame = f.get("frame_path")
                    closest_caption = f.get("vlm_caption")

            # Clean and sanitize caption preview
            clean_caption = closest_caption
            refusal_check_list = [
                "sorry", "cannot browse", "can't browse", "unable to browse", 
                "large language model", "training data", "cutoff date",
                "对不起", "抱歉", "语言模型", "无法访问", "没有访问", "作为ai",
                "你好", "提供帮助", "javascript", "const numbers"
            ]
            if clean_caption:
                c_low = clean_caption.lower()
                if any(w in c_low or w in clean_caption for w in refusal_check_list):
                    clean_caption = "Visual keyframe capturing scene activity and subjects."
            else:
                clean_caption = "Visual keyframe capturing scene activity and subjects."

            calibrated_score = round(float(m["score"] * raw_peak_factor), 3)

            moments_response.append(MomentItem(
                t_start=m["t_start"],
                t_end=m["t_end"],
                score=calibrated_score,
                preview_frame_path=closest_frame,
                caption_preview=clean_caption,
                transcript_preview=None
            ))

        # 11. Construct 1-Hz Heatmap array for Frontend Canvas
        heatmap_1hz = []
        total_seconds = int(np.ceil(duration_sec))
        for sec in range(total_seconds):
            idx = min(len(contrast_smoothed) - 1, int(sec * 2))
            heatmap_1hz.append(round(float(contrast_smoothed[idx]), 3))

        latency_ms = round((time.time() - t0) * 1000.0, 2)
        logger.info(f"SOTA Search for '{query}' completed in {latency_ms} ms.")

        return SearchResponse(
            query=query,
            video_id=actual_video_id,
            moments=moments_response,
            timeline_heatmap=heatmap_1hz,
            total_duration=round(duration_sec, 2),
            latency_ms=latency_ms,
            top_k=top_k
        )

search_engine = HybridMomentSearchEngine()
