"""Hybrid pure-visual retrieval engine (frame embeddings + scene captions)."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.core.config import settings
from app.db.connection import db_manager
from app.db.schemas import MomentItem, SearchResponse
from app.pipeline.visual_encoder import SigLIP2VisualEncoder
from app.retrieval.boundary_extractor import TemporalBoundaryExtractor
from app.retrieval.query_expander import query_expander
from app.retrieval.rank_fusion import ReciprocalRankFusion
from app.retrieval.temporal_smoother import TemporalSmoother
from app.retrieval.vlm_verifier import vlm_verifier


def _lexical_score(text: str, query: str) -> float:
    tokens = {token.casefold() for token in query.split() if token.strip()}
    if not text or not tokens:
        return 0.0
    haystack = text.casefold()
    return sum(1 for token in tokens if token in haystack) / len(tokens)


class HybridMomentSearchEngine:
    """Search every frame of one video and produce calibrated proposals."""

    def __init__(self) -> None:
        self.text_encoder = SigLIP2VisualEncoder()
        self.rrf = ReciprocalRankFusion(k=settings.DEFAULT_RRF_K)
        self.smoother = TemporalSmoother(default_sigma=settings.TEMPORAL_GAUSSIAN_SIGMA)
        self.boundary_extractor = TemporalBoundaryExtractor()
        self._calibration_cache: Optional[Dict[str, Any]] = None
        self._fusion_cache: Optional[Dict[str, float]] = None

    @staticmethod
    def _rows(table: Any, where: Optional[str] = None, limit: int = 5000) -> List[Dict[str, Any]]:
        try:
            search = table.search()
            if where:
                search = search.where(where)
            return search.limit(limit).to_list()
        except Exception:
            try:
                rows = table.to_arrow().to_pylist()
            except Exception:
                return []
            if where and " = '" in where:
                key, value = where.split(" = '", 1)
                rows = [row for row in rows if str(row.get(key)) == value.rstrip("'")]
            return rows[:limit]

    def _caption_rows(self, query: str, video_id: str) -> List[Dict[str, Any]]:
        """Query the scene FTS index; lexical scoring is only a safe fallback."""
        table = db_manager.get_table("scenes_v2")
        rows: List[Dict[str, Any]] = []
        try:
            rows = table.search(query, query_type="fts").where(
                f"video_id = '{video_id}' AND caption_status = 'generated'"
            ).limit(200).to_list()
        except Exception:
            all_rows = self._rows(table, f"video_id = '{video_id}'", 5000)
            rows = [row for row in all_rows
                    if row.get("caption_status") == "generated" and _lexical_score(str(row.get("caption", "")), query) > 0]
            for row in rows:
                row["_score"] = _lexical_score(str(row.get("caption", "")), query)
        return rows

    def _load_calibration(self) -> Optional[Dict[str, Any]]:
        if self._calibration_cache is not None:
            return self._calibration_cache
        path = Path(settings.CALIBRATION_ARTIFACT_PATH)
        if not path.exists():
            self._calibration_cache = None
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("index_version") != settings.VISUAL_INDEX_VERSION:
                return None
            if data.get("model_id") != settings.SIGLIP2_MODEL_ID:
                return None
            self._calibration_cache = data
            return data
        except Exception:
            return None

    def _fusion_weights(self) -> Tuple[float, float]:
        if self._fusion_cache is None:
            try:
                data = json.loads(Path(settings.FUSION_ARTIFACT_PATH).read_text(encoding="utf-8"))
                if data.get("index_version") not in {None, settings.VISUAL_INDEX_VERSION}:
                    raise ValueError("fusion artifact index version mismatch")
                if data.get("model_id") not in {None, "", settings.SIGLIP2_MODEL_ID}:
                    raise ValueError("fusion artifact model mismatch")
                visual = float(data.get("visual_weight", settings.DEFAULT_WEIGHT_VISUAL))
                caption = float(data.get("caption_weight", settings.DEFAULT_WEIGHT_CAPTION))
                total = max(1e-8, visual + caption)
                self._fusion_cache = {"visual": visual / total, "caption": caption / total}
            except Exception:
                self._fusion_cache = {"visual": settings.DEFAULT_WEIGHT_VISUAL, "caption": settings.DEFAULT_WEIGHT_CAPTION}
        return self._fusion_cache["visual"], self._fusion_cache["caption"]
    def _probability(self, raw_score: float, display_score: float) -> Tuple[float, bool, Optional[float]]:
        artifact = self._load_calibration()
        if not artifact:
            # This is explicitly uncalibrated; display score keeps the API
            # useful for development without pretending it is a probability.
            return float(np.clip(display_score, 0.0, 1.0)), False, None
        slope = float(artifact.get("slope", 1.0))
        intercept = float(artifact.get("intercept", -0.5))
        value = slope * float(raw_score) + intercept
        probability = float(1.0 / (1.0 + np.exp(-np.clip(value, -60.0, 60.0))))
        return probability, True, float(artifact.get("no_match_threshold", settings.NO_MATCH_THRESHOLD))

    @staticmethod
    def _normalize_display(signal: np.ndarray) -> np.ndarray:
        if len(signal) == 0:
            return signal
        lo, hi = float(np.min(signal)), float(np.max(signal))
        if hi <= lo + 1e-8:
            return np.zeros_like(signal, dtype=np.float32)
        return np.clip((signal - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

    def search_moments(self, query: str, video_id: str, top_k: int = 5, profile: str = "fast") -> SearchResponse:
        started = time.monotonic()
        profile = profile if profile in {"fast", "accurate"} else "fast"
        top_k = max(1, min(int(top_k), 20))
        warnings: List[str] = []
        videos = self._rows(db_manager.get_table("videos"), f"id = '{video_id}'", 1)
        if not videos:
            warnings.append("video_not_found")
            return SearchResponse(query=query, video_id=video_id, moments=[], timeline_heatmap=[], total_duration=0.0,
                                  latency_ms=round((time.monotonic() - started) * 1000, 2), top_k=top_k,
                                  profile=profile, calibrated=False, index_version=settings.VISUAL_INDEX_VERSION, warnings=warnings)
        target = videos[0]
        duration = max(0.0, float(target.get("duration_sec", 0.0)))
        target_index_version = str(target.get("visual_index_version") or "")
        if target_index_version and target_index_version != settings.VISUAL_INDEX_VERSION:
            warnings.append("reindex_required")
            return SearchResponse(query=query, video_id=video_id, moments=[], timeline_heatmap=[], total_duration=round(duration, 2),
                                  latency_ms=round((time.monotonic() - started) * 1000, 2), top_k=top_k,
                                  profile=profile, calibrated=False, index_version=settings.VISUAL_INDEX_VERSION, warnings=warnings)
        # Older ``videos`` tables may not have the v2 metadata columns.  A
        # newly ingested video is still valid when its v2 metadata/frames are
        # present; a legacy-only video must explicitly reindex.
        if not target_index_version:
            legacy_rows = self._rows(db_manager.get_table("video_frames"), f"video_id = '{video_id}'", 1)
            if legacy_rows:
                warnings.append("reindex_required")
                return SearchResponse(query=query, video_id=video_id, moments=[], timeline_heatmap=[], total_duration=round(duration, 2),
                                      latency_ms=round((time.monotonic() - started) * 1000, 2), top_k=top_k,
                                      profile=profile, calibrated=False, index_version=settings.VISUAL_INDEX_VERSION, warnings=warnings)
        compatibility = db_manager.visual_index_compatibility(video_id)
        if not compatibility.get("compatible", False):
            warnings.append("reindex_required")
            return SearchResponse(query=query, video_id=video_id, moments=[], timeline_heatmap=[], total_duration=round(duration, 2),
                                  latency_ms=round((time.monotonic() - started) * 1000, 2), top_k=top_k,
                                  profile=profile, calibrated=False, index_version=settings.VISUAL_INDEX_VERSION, warnings=warnings)
        frames = self._rows(db_manager.get_table("video_frames_v2"), f"video_id = '{video_id}'", 200000)
        dim = int(settings.SIGLIP2_EMBEDDING_DIM)
        frames = [frame for frame in frames if frame.get("siglip2_vector") is not None and len(frame.get("siglip2_vector")) == dim]
        if not frames:
            warnings.append("index_empty")
            return SearchResponse(query=query, video_id=video_id, moments=[], timeline_heatmap=[], total_duration=round(duration, 2),
                                  latency_ms=round((time.monotonic() - started) * 1000, 2), top_k=top_k,
                                  profile=profile, calibrated=False, index_version=settings.VISUAL_INDEX_VERSION, warnings=warnings)

        variants = query_expander.get_query_variants(query)
        vectors: List[np.ndarray] = []
        weights: List[float] = []
        for text, weight in variants:
            vector = np.asarray(self.text_encoder.encode_text(text), dtype=np.float32)
            vector /= max(1e-8, float(np.linalg.norm(vector)))
            vectors.append(vector)
            weights.append(float(weight))
        qvec = np.average(np.stack(vectors), axis=0, weights=np.asarray(weights))
        qvec /= max(1e-8, float(np.linalg.norm(qvec)))
        matrix = np.asarray([frame["siglip2_vector"] for frame in frames], dtype=np.float32)
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)
        cosine = matrix @ qvec
        visual_relevance = np.clip((cosine + 1.0) / 2.0, 0.0, 1.0)

        # Caption BM25/FTS results are distributed only to their own scene.
        scenes = self._rows(db_manager.get_table("scenes_v2"), f"video_id = '{video_id}'", 10000)
        scene_by_id = {str(scene.get("id")): scene for scene in scenes}
        caption_rows = self._caption_rows(" ".join(text for text, _ in variants), video_id)
        caption_by_scene: Dict[str, float] = {}
        for row in caption_rows:
            scene_id = str(row.get("id"))
            score = float(row.get("_score", row.get("score", 0.0)) or 0.0)
            if score <= 0.0:
                score = _lexical_score(str(row.get("caption", "")), query)
            caption_by_scene[scene_id] = max(caption_by_scene.get(scene_id, 0.0), score)
        caption_relevance = np.asarray([caption_by_scene.get(str(frame.get("scene_id")), 0.0) for frame in frames], dtype=np.float32)

        visual_rank = [{"id": str(frame.get("id")), "timestamp": float(frame.get("timestamp", 0.0)), "score": float(score), "frame": frame}
                       for frame, score in zip(frames, visual_relevance)]
        caption_rank = [{"id": str(frame.get("id")), "timestamp": float(frame.get("timestamp", 0.0)), "score": float(score), "frame": frame}
                        for frame, score in zip(frames, caption_relevance) if score > 0]
        visual_rank.sort(key=lambda item: (-item["score"], item["timestamp"], item["id"]))
        caption_rank.sort(key=lambda item: (-item["score"], item["timestamp"], item["id"]))
        tuned_visual, tuned_caption = self._fusion_weights()
        caption_weight = tuned_caption if caption_rank else 0.0
        visual_weight = 1.0 if not caption_rank else tuned_visual
        if caption_rank:
            total_weight = visual_weight + caption_weight
            visual_weight, caption_weight = visual_weight / total_weight, caption_weight / total_weight
        fused = self.rrf.fuse({"visual": visual_rank, "caption": caption_rank}, {"visual": visual_weight, "caption": caption_weight})

        raw_by_timestamp: Dict[float, float] = {}
        transition_by_timestamp: Dict[float, float] = {}
        visual_by_timestamp: Dict[float, float] = {}
        caption_by_timestamp: Dict[float, float] = {}
        frame_by_timestamp: Dict[float, Dict[str, Any]] = {}
        for frame, visual in zip(frames, visual_relevance):
            ts = float(frame.get("timestamp", 0.0))
            visual_by_timestamp[ts] = float(visual)
            caption_by_timestamp[ts] = float(caption_by_scene.get(str(frame.get("scene_id")), 0.0))
            transition_by_timestamp[ts] = float(frame.get("transition_energy", 0.0) or 0.0)
            frame_by_timestamp[ts] = frame
        for result in fused.values():
            item = result["item_data"]
            ts = float(item.get("timestamp", 0.0))
            raw_by_timestamp[ts] = max(raw_by_timestamp.get(ts, -1.0), float(result["fused_score"]))
        axis, raw_timeline = self.smoother.smooth_timeline(duration, list(raw_by_timestamp.items()), sigma=settings.TEMPORAL_GAUSSIAN_SIGMA,
                                                           resolution_hz=2, use_multiscale=(profile == "accurate"))
        display_timeline = self._normalize_display(raw_timeline)
        transition_timeline = np.interp(axis, sorted(transition_by_timestamp), [transition_by_timestamp[t] for t in sorted(transition_by_timestamp)]) if transition_by_timestamp else np.zeros_like(axis)
        boundaries = [0.0, duration]
        for scene in scenes:
            boundaries.extend([float(scene.get("t_start", 0.0)), float(scene.get("t_end", duration))])
        candidates = self.boundary_extractor.extract_multiscale_proposals(axis, display_timeline, raw_scores=raw_timeline,
                                                                          transition_energy=transition_timeline, scene_boundaries=sorted(set(boundaries)))
        if not candidates:
            warnings.append("no_candidates")
        calibration_available = self._load_calibration() is not None
        if not calibration_available:
            warnings.append("calibration_missing")
        if profile == "accurate" and settings.ENABLE_VLM_STAGE2_VERIFY and candidates:
            verify_limit = max(1, int(settings.VLM_VERIFY_TOP_K))
            # Pass the source path as ephemeral metadata so Accurate mode can
            # decode a fresh 2fps window around each candidate.  It is never
            # persisted into LanceDB and is ignored by Fast mode.
            verifier_frames = [{"_video_path": str(target.get("filepath", ""))}] + frames
            candidates = vlm_verifier.verify(candidates[:verify_limit], verifier_frames, query) + candidates[verify_limit:]
            warnings.extend(vlm_verifier.last_warnings)

        scored: List[Tuple[float, Dict[str, Any]]] = []
        threshold: Optional[float] = None
        for candidate in candidates:
            rank_score = float(candidate.get("rank_score", candidate.get("score", 0.0)))
            display_score = float(np.clip(candidate.get("score", 0.0), 0.0, 1.0))
            probability, calibrated, artifact_threshold = self._probability(rank_score, display_score)
            threshold = artifact_threshold if artifact_threshold is not None else threshold
            candidate["probability"] = probability
            candidate["calibrated"] = calibrated
            scored.append((probability, candidate))
        if threshold is not None and scored and max(score for score, _ in scored) < threshold:
            warnings.append("no_match")
            scored = []
        scored.sort(key=lambda pair: (-pair[0], float(pair[1].get("t_start", 0.0))))
        moments: List[MomentItem] = []
        for occurrence_index, (probability, candidate) in enumerate(scored[:top_k], start=1):
            start, end = float(candidate["t_start"]), float(candidate["t_end"])
            timestamps = [ts for ts in frame_by_timestamp if start <= ts <= end]
            nearest = min(frames, key=lambda frame: abs(float(frame.get("timestamp", 0.0)) - (start + end) / 2.0), default={})
            visual = max((visual_by_timestamp.get(ts, 0.0) for ts in timestamps), default=0.0)
            caption = max((caption_by_timestamp.get(ts, 0.0) for ts in timestamps), default=0.0)
            verifier = float(candidate.get("verifier_confidence", 0.0))
            breakdown = {"visual": round(float(visual), 4), "caption": round(float(caption), 4),
                         "temporal": round(float(candidate.get("score", 0.0)), 4), "verifier": round(verifier, 4)}
            scene_caption = ""
            if nearest.get("scene_id") in scene_by_id:
                scene_caption = str(scene_by_id[nearest["scene_id"]].get("caption", "") or "")
            moments.append(MomentItem(t_start=start, t_end=end, score=round(float(probability), 4),
                                      raw_score=round(rank_score, 6), display_score=round(float(candidate.get("score", 0.0)), 4),
                                      preview_frame_path=nearest.get("frame_path"), caption_preview=scene_caption or None,
                                      modality_breakdown=breakdown, occurrence_index=occurrence_index))
        # one heatmap value per second, derived only from display timeline
        heatmap = [round(float(display_timeline[min(len(display_timeline) - 1, int(sec * 2))]), 4) for sec in range(max(0, int(np.ceil(duration))))] if len(display_timeline) else []
        return SearchResponse(query=query, video_id=video_id, moments=moments, timeline_heatmap=heatmap,
                              total_duration=round(duration, 2), latency_ms=round((time.monotonic() - started) * 1000, 2),
                              top_k=top_k, profile=profile, calibrated=calibration_available,
                              index_version=settings.VISUAL_INDEX_VERSION, warnings=warnings)


search_engine = HybridMomentSearchEngine()
