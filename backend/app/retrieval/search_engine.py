"""Hybrid pure-visual retrieval engine (frame embeddings + scene captions)."""

import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.core.config import settings
from app.db.connection import db_manager
from app.db.schemas import MomentItem, SearchResponse
from app.inference.client import inference_client, pe_worker_client
from app.inference.contracts import PEEmbedTextRequest
from app.inference.errors import InferenceWorkerError, InferenceWorkerOOM, InferenceWorkerTimeout, InferenceWorkerUnavailable
from app.pipeline.visual_encoder import SigLIP2VisualEncoder
from app.retrieval.boundary_extractor import TemporalBoundaryExtractor
from app.retrieval.query_expander import query_expander
from app.retrieval.rank_fusion import ReciprocalRankFusion
from app.retrieval.temporal_smoother import TemporalSmoother
from app.retrieval.vlm_verifier import vlm_verifier
from app.retrieval.query_router import query_router
from app.retrieval.grounding import grounding_orchestrator


def _lexical_score(text: str, query: str) -> float:
    tokens = {token.casefold() for token in query.split() if token.strip()}
    if not text or not tokens:
        return 0.0
    haystack = text.casefold()
    return sum(1 for token in tokens if token in haystack) / len(tokens)


_GPU_SEARCH_LOCK = threading.RLock()


def _serialize_gpu_search(func):
    """Keep the two local workers from loading heavy models concurrently."""
    def wrapped(*args, **kwargs):
        with _GPU_SEARCH_LOCK:
            return func(*args, **kwargs)
    return wrapped


class HybridMomentSearchEngine:
    """Search every frame of one video and produce calibrated proposals."""

    def __init__(self) -> None:
        self.text_encoder = SigLIP2VisualEncoder()
        self.rrf = ReciprocalRankFusion(k=settings.DEFAULT_RRF_K)
        self.smoother = TemporalSmoother(default_sigma=settings.TEMPORAL_GAUSSIAN_SIGMA)
        self.boundary_extractor = TemporalBoundaryExtractor()
        self._calibration_cache: Dict[str, Optional[Dict[str, Any]]] = {}
        self._fusion_cache: Optional[Dict[str, float]] = None
        self._active_retrieval_backend = "siglip2"
        self._active_retrieval_model_id = settings.SIGLIP2_MODEL_ID
        self._active_retrieval_embedding_version = settings.SIGLIP2_EMBEDDING_VERSION

    @staticmethod
    def _retrieval_spec(backend: str) -> Dict[str, str]:
        if backend == "pe_core_b16":
            return {
                "backend": backend,
                "alias": "pe_core_b16",
                "model_id": "PE-Core-B16-224",
                "revision": settings.PE_CORE_B16_REVISION,
                "vector_key": "pe_core_vector",
                "embedding_version": f"{settings.PE_CORE_INDEX_VERSION}:PE-Core-B16-224:{settings.PE_CORE_B16_REVISION[:12]}",
            }
        if backend == "pe_core_l14":
            return {
                "backend": backend,
                "alias": "pe_core_l14",
                "model_id": "PE-Core-L14-336",
                "revision": settings.PE_CORE_L14_REVISION,
                "vector_key": "pe_core_vector",
                "embedding_version": f"{settings.PE_CORE_INDEX_VERSION}:PE-Core-L14-336:{settings.PE_CORE_L14_REVISION[:12]}",
            }
        return {
            "backend": "siglip2",
            "alias": "siglip",
            "model_id": settings.SIGLIP2_MODEL_ID,
            "revision": "",
            "vector_key": "siglip2_vector",
            "embedding_version": settings.SIGLIP2_EMBEDDING_VERSION,
        }

    def _pe_index_ready(self, video_id: str, spec: Dict[str, str]) -> bool:
        try:
            rows = self._rows(
                db_manager.get_table("pe_core_index_metadata_v1"),
                f"video_id = '{video_id}' AND model_id = '{spec['model_id']}' AND embedding_version = '{spec['embedding_version']}'",
                10,
            )
            return bool(rows and rows[0].get("status") == "ready" and
                        int(rows[0].get("embedding_dim", 0)) == int(settings.PE_CORE_EMBEDDING_DIM) and
                        int(rows[0].get("indexed_frame_count", 0)) == int(rows[0].get("source_frame_count", -1)))
        except Exception:
            return False

    def _retrieval_frames(self, video_id: str, spec: Dict[str, str]) -> List[Dict[str, Any]]:
        if spec["backend"] == "siglip2":
            rows = self._rows(db_manager.get_table("video_frames_v2"), f"video_id = '{video_id}'", 200000)
            dim = int(settings.SIGLIP2_EMBEDDING_DIM)
            return [row for row in rows if row.get(spec["vector_key"]) is not None and len(row.get(spec["vector_key"])) == dim]

        rows = self._rows(
            db_manager.get_table("video_frames_pe_core_v1"),
            f"video_id = '{video_id}' AND model_id = '{spec['model_id']}' AND embedding_version = '{spec['embedding_version']}'",
            200000,
        )
        source_rows = self._rows(db_manager.get_table("video_frames_v2"), f"video_id = '{video_id}'", 200000)
        transition_by_id = {str(row.get("id")): float(row.get("transition_energy", 0.0) or 0.0) for row in source_rows}
        dim = int(settings.PE_CORE_EMBEDDING_DIM)
        for row in rows:
            row["transition_energy"] = transition_by_id.get(str(row.get("source_frame_id")), 0.0)
        return [row for row in rows if row.get(spec["vector_key"]) is not None and len(row.get(spec["vector_key"])) == dim]

    def _encode_retrieval_text(self, variants: List[Tuple[str, float]], spec: Dict[str, str], release_after: bool) -> List[List[float]]:
        if spec["backend"] == "siglip2":
            if hasattr(self.text_encoder, "encode_texts"):
                return self.text_encoder.encode_texts([text for text, _ in variants], release_after=release_after)
            encoded = [self.text_encoder.encode_text(text) for text, _ in variants]
            if release_after and callable(getattr(self.text_encoder, "unload", None)):
                self.text_encoder.unload()
            return encoded
        response = pe_worker_client.embed_text(
            PEEmbedTextRequest(
                model_id=spec["model_id"],
                texts=[text for text, _ in variants],
                release_after=release_after,
            )
        )
        if response.embedding_dim != int(settings.PE_CORE_EMBEDDING_DIM):
            raise ValueError(f"PE-Core returned unexpected embedding dimension {response.embedding_dim}")
        return response.embeddings

    @staticmethod
    def _fuse_accurate_score(base: float, sam_score: float = 0.0, qwen_confidence: Optional[float] = None, qwen_present: bool = True) -> float:
        if qwen_confidence is not None and not qwen_present:
            return base * 0.20
        if sam_score > 0.0 and qwen_confidence is not None:
            return 0.45 * base + 0.25 * sam_score + 0.30 * qwen_confidence
        if qwen_confidence is not None:
            return 0.60 * base + 0.40 * qwen_confidence
        if sam_score > 0.0:
            return 0.55 * base + 0.45 * sam_score
        return base

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
            if where:
                predicates = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*'([^']*)'", where)
                for key, value in predicates:
                    rows = [row for row in rows if str(row.get(key)) == value]
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
        cache_key = f"{self._active_retrieval_model_id}:{self._active_retrieval_embedding_version}"
        if cache_key in self._calibration_cache:
            return self._calibration_cache[cache_key]
        path = Path(settings.CALIBRATION_ARTIFACT_PATH)
        if not path.exists():
            self._calibration_cache[cache_key] = None
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("index_version") != settings.VISUAL_INDEX_VERSION:
                self._calibration_cache[cache_key] = None
                return None
            if self._active_retrieval_model_id != settings.SIGLIP2_MODEL_ID or data.get("model_id") != self._active_retrieval_model_id:
                self._calibration_cache[cache_key] = None
                return None
            artifact_version = str(data.get("embedding_version") or settings.SIGLIP2_EMBEDDING_VERSION)
            if artifact_version != self._active_retrieval_embedding_version:
                self._calibration_cache[cache_key] = None
                return None
            self._calibration_cache[cache_key] = data
            return data
        except Exception:
            self._calibration_cache[cache_key] = None
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
        if self._active_retrieval_backend != "siglip2":
            return settings.DEFAULT_WEIGHT_VISUAL, settings.DEFAULT_WEIGHT_CAPTION
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
    def _interval_overlap_ratio(left: Dict[str, Any], right: Dict[str, Any]) -> float:
        """Return intersection divided by the shorter interval length."""
        left_start, left_end = float(left.get("t_start", 0.0)), float(left.get("t_end", 0.0))
        right_start, right_end = float(right.get("t_start", 0.0)), float(right.get("t_end", 0.0))
        intersection = max(0.0, min(left_end, right_end) - max(left_start, right_start))
        shorter = max(1e-6, min(left_end - left_start, right_end - right_start))
        return intersection / shorter

    @classmethod
    def _same_event(cls, left: Dict[str, Any], right: Dict[str, Any]) -> bool:
        overlap = cls._interval_overlap_ratio(left, right)
        left_start, left_end = float(left.get("t_start", 0.0)), float(left.get("t_end", 0.0))
        right_start, right_end = float(right.get("t_start", 0.0)), float(right.get("t_end", 0.0))
        contains = (left_start <= right_start and left_end >= right_end) or (right_start <= left_start and right_end >= left_end)
        left_center = (left_start + left_end) / 2.0
        right_center = (right_start + right_end) / 2.0
        return overlap >= 0.70 or (contains and abs(left_center - right_center) <= 2.0)

    @classmethod
    def _group_candidates(cls, scored: List[Tuple[float, Dict[str, Any]]]) -> List[Tuple[float, Dict[str, Any]]]:
        """Collapse multi-scale proposals that describe one event.

        Candidates are first grouped by interval containment/overlap. The
        winning interval remains the primary moment while the union is exposed
        as optional context bounds for the UI. Disjoint events are untouched.
        """
        groups: List[List[Tuple[float, Dict[str, Any]]]] = []
        for item in scored:
            matches = [group for group in groups if any(cls._same_event(item[1], member[1]) for member in group)]
            if not matches:
                groups.append([item])
                continue
            target = matches[0]
            target.append(item)
            for other in matches[1:]:
                target.extend(other)
                groups.remove(other)

        grouped: List[Tuple[float, Dict[str, Any]]] = []
        for group in groups:
            # Highest probability wins; when effectively tied prefer the
            # narrower interval because it is more useful for precise seeking.
            best_probability = max(probability for probability, _ in group)
            near_best = [item for item in group if best_probability - item[0] <= 0.03]
            probability, primary = min(
                near_best,
                key=lambda item: (float(item[1].get("t_end", 0.0)) - float(item[1].get("t_start", 0.0)), -item[0], float(item[1].get("t_start", 0.0))),
            )
            context_start = min(float(candidate.get("t_start", 0.0)) for _, candidate in group)
            context_end = max(float(candidate.get("t_end", 0.0)) for _, candidate in group)
            primary = dict(primary)
            primary["context_t_start"] = context_start
            primary["context_t_end"] = context_end
            grouped.append((probability, primary))
        return grouped

    @staticmethod
    def _normalize_display(signal: np.ndarray) -> np.ndarray:
        if len(signal) == 0:
            return signal
        lo, hi = float(np.min(signal)), float(np.max(signal))
        if hi <= lo + 1e-8:
            return np.zeros_like(signal, dtype=np.float32)
        return np.clip((signal - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

    @_serialize_gpu_search
    def search_moments(
        self,
        query: str,
        video_id: str,
        top_k: int = 5,
        profile: str = "fast",
        retrieval_backend: str = "siglip2",
    ) -> SearchResponse:
        started = time.monotonic()
        profile = profile if profile in {"fast", "accurate"} else "fast"
        top_k = max(1, min(int(top_k), 20))
        warnings: List[str] = []
        if retrieval_backend not in {"siglip2", "pe_core_b16", "pe_core_l14"}:
            warnings.append("retrieval_backend_invalid")
            retrieval_backend = "siglip2"
        requested_spec = self._retrieval_spec(retrieval_backend)
        active_spec = dict(requested_spec)
        retrieval_backend_requested = requested_spec["backend"]
        retrieval_backend_used = active_spec["backend"]
        retrieval_model_id = active_spec["model_id"]
        retrieval_embedding_version = active_spec["embedding_version"]
        self._active_retrieval_backend = retrieval_backend_used
        self._active_retrieval_model_id = retrieval_model_id
        self._active_retrieval_embedding_version = retrieval_embedding_version
        stage_latency_ms: Dict[str, float] = {}
        models_used: List[str] = []
        models_attempted: List[str] = []
        strategy_used = "fast"
        cache_hits: Dict[str, bool] = {}
        cascade_path: List[str] = []
        stage_status: Dict[str, str] = {}
        planner_version = ""
        accurate_deadline = started + float(settings.ACCURATE_MAX_SECONDS) if profile == "accurate" else None

        def empty_response(total_duration: float) -> SearchResponse:
            return SearchResponse(
                query=query,
                video_id=video_id,
                moments=[],
                timeline_heatmap=[],
                total_duration=round(total_duration, 2),
                latency_ms=round((time.monotonic() - started) * 1000, 2),
                top_k=top_k,
                profile=profile,
                calibrated=False,
                index_version=settings.VISUAL_INDEX_VERSION,
                warnings=warnings,
                models_attempted=models_attempted,
                stage_status=stage_status,
                cascade_path=cascade_path,
                retrieval_backend_requested=retrieval_backend_requested,
                retrieval_backend_used=retrieval_backend_used,
                retrieval_model_id=retrieval_model_id,
                retrieval_embedding_version=retrieval_embedding_version,
            )

        videos = self._rows(db_manager.get_table("videos"), f"id = '{video_id}'", 1)
        if not videos:
            warnings.append("video_not_found")
            return empty_response(0.0)
        target = videos[0]
        duration = max(0.0, float(target.get("duration_sec", 0.0)))
        target_index_version = str(target.get("visual_index_version") or "")
        if target_index_version and target_index_version != settings.VISUAL_INDEX_VERSION:
            warnings.append("reindex_required")
            return empty_response(duration)
        # Older ``videos`` tables may not have the v2 metadata columns.  A
        # newly ingested video is still valid when its v2 metadata/frames are
        # present; a legacy-only video must explicitly reindex.
        if not target_index_version:
            legacy_rows = self._rows(db_manager.get_table("video_frames"), f"video_id = '{video_id}'", 1)
            if legacy_rows:
                warnings.append("reindex_required")
                return empty_response(duration)
        compatibility = db_manager.visual_index_compatibility(video_id)
        if not compatibility.get("compatible", False):
            warnings.append("reindex_required")
            return empty_response(duration)
        if active_spec["backend"] != "siglip2" and not self._pe_index_ready(video_id, active_spec):
            warnings.append("pe_index_missing")
            active_spec = self._retrieval_spec("siglip2")
            retrieval_backend_used = active_spec["backend"]
            retrieval_model_id = active_spec["model_id"]
            retrieval_embedding_version = active_spec["embedding_version"]
            self._active_retrieval_backend = retrieval_backend_used
            self._active_retrieval_model_id = retrieval_model_id
            self._active_retrieval_embedding_version = retrieval_embedding_version
            warnings.append("pe_fallback_siglip")
        frames = self._retrieval_frames(video_id, active_spec)
        if not frames:
            warnings.append("index_empty")
            return empty_response(duration)

        variants = query_expander.get_query_variants(query)
        retrieval_started = time.monotonic()
        vectors: List[np.ndarray] = []
        weights: List[float] = []
        models_attempted.append(active_spec["model_id"])
        try:
            if active_spec["backend"] != "siglip2" and inference_client.enabled:
                # The legacy worker owns SigLIP/SAM/Qwen. Evict SigLIP before
                # the isolated PE process allocates its own checkpoint.
                inference_client.unload("siglip")
            elif active_spec["backend"] == "siglip2" and pe_worker_client.enabled:
                # Switching back from the experiment must release PE first.
                pe_worker_client.unload()
            encoded_variants = self._encode_retrieval_text(
                variants, active_spec, release_after=profile == "accurate"
            )
        except (InferenceWorkerError, RuntimeError, ValueError) as exc:
            if active_spec["backend"] == "siglip2":
                raise
            try:
                pe_worker_client.unload()
            except Exception:
                pass
            warnings.append("pe_oom_fallback" if isinstance(exc, InferenceWorkerOOM) else
                            "pe_timeout_fallback" if isinstance(exc, InferenceWorkerTimeout) else
                            "pe_worker_unavailable")
            warnings.append("pe_fallback_siglip")
            active_spec = self._retrieval_spec("siglip2")
            retrieval_backend_used = active_spec["backend"]
            retrieval_model_id = active_spec["model_id"]
            retrieval_embedding_version = active_spec["embedding_version"]
            self._active_retrieval_backend = retrieval_backend_used
            self._active_retrieval_model_id = retrieval_model_id
            self._active_retrieval_embedding_version = retrieval_embedding_version
            frames = self._retrieval_frames(video_id, active_spec)
            models_attempted.append(active_spec["model_id"])
            encoded_variants = self._encode_retrieval_text(
                variants, active_spec, release_after=profile == "accurate"
            )
        models_used.append(active_spec["model_id"])
        for (_, weight), encoded in zip(variants, encoded_variants):
            vector = np.asarray(encoded, dtype=np.float32)
            vector /= max(1e-8, float(np.linalg.norm(vector)))
            vectors.append(vector)
            weights.append(float(weight))
        qvec = np.average(np.stack(vectors), axis=0, weights=np.asarray(weights))
        qvec /= max(1e-8, float(np.linalg.norm(qvec)))
        matrix = np.asarray([frame[active_spec["vector_key"]] for frame in frames], dtype=np.float32)
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)
        cosine = matrix @ qvec
        visual_relevance = np.clip((cosine + 1.0) / 2.0, 0.0, 1.0)
        retrieval_stage = active_spec["alias"]
        stage_latency_ms[retrieval_stage] = round((time.monotonic() - retrieval_started) * 1000, 2)
        stage_latency_ms["retrieval"] = stage_latency_ms[retrieval_stage]
        stage_status[retrieval_stage] = "success"
        stage_status["retrieval"] = "success"
        if profile == "accurate":
            cascade_path.extend([f"{retrieval_stage}:success", f"{retrieval_stage}:unloaded"])

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
        # Keep the proposal generator identical between profiles. Accurate
        # gets its extra signal from SAM/Qwen; it must not discard a good Fast
        # proposal merely because a second-stage verifier is unavailable.
        axis, raw_timeline = self.smoother.smooth_timeline(duration, list(raw_by_timestamp.items()), sigma=settings.TEMPORAL_GAUSSIAN_SIGMA,
                                                           resolution_hz=2, use_multiscale=False)
        display_timeline = self._normalize_display(raw_timeline)
        transition_timeline = np.interp(axis, sorted(transition_by_timestamp), [transition_by_timestamp[t] for t in sorted(transition_by_timestamp)]) if transition_by_timestamp else np.zeros_like(axis)
        boundaries = [0.0, duration]
        for scene in scenes:
            boundaries.extend([float(scene.get("t_start", 0.0)), float(scene.get("t_end", duration))])
        candidates = self.boundary_extractor.extract_multiscale_proposals(axis, display_timeline, raw_scores=raw_timeline,
                                                                          transition_energy=transition_timeline, scene_boundaries=sorted(set(boundaries)))
        if not candidates:
            warnings.append("no_candidates")
        # Snapshot the profile-independent retrieval result. If a heavy model
        # is disabled, times out, or returns no valid evidence, Accurate must
        # return this exact baseline instead of silently degrading ranking.
        baseline_candidates = [dict(candidate) for candidate in candidates]
        calibration_available = self._load_calibration() is not None
        if not calibration_available:
            warnings.append("calibration_missing")
        route_info = query_router.route(query) if profile == "accurate" else {"route": "fast"}
        route = str(route_info.get("route", "fast"))
        if profile == "accurate":
            planner_version = str(route_info.get("planner_version", "cascade-planner-v2"))
            for candidate in candidates:
                candidate["fast_score"] = float(candidate.get("score", 0.0))

            grounding_limit = max(1, int(settings.SAM_SEARCH_TOP_K))
            grounding_started = time.monotonic()
            if settings.ENABLE_SAM_GROUNDING and candidates:
                models_attempted.append(settings.SAM_MODEL_ID)
                candidates, grounding_warnings, grounding_cache_hits = grounding_orchestrator.ground_candidates(
                    video_id=video_id,
                    candidates=candidates,
                    frames=frames,
                    prompts=[str(prompt) for prompt in route_info.get("sam_prompts", [])],
                    limit=grounding_limit,
                    deadline=accurate_deadline,
                    release_after=True,
                )
                warnings.extend(grounding_warnings)
                cache_hits.update({f"sam:{key}": value for key, value in grounding_cache_hits.items()})
                sam_checked = candidates[:grounding_limit]
                has_sam_evidence = any(float(item.get("sam_score", 0.0)) > 0.0 for item in sam_checked)
                has_sam_error = any(item.get("sam_error") for item in sam_checked) or any(
                    code in grounding_warnings for code in ("sam_timeout_fallback", "sam_oom_fallback", "sam_worker_unavailable")
                )
                if has_sam_evidence:
                    stage_status["sam"] = "detected"
                    cascade_path.append("sam:detected")
                    models_used.append(settings.SAM_MODEL_ID)
                elif has_sam_error:
                    stage_status["sam"] = "unavailable"
                    cascade_path.append("sam:unavailable")
                else:
                    stage_status["sam"] = "no_detection"
                    cascade_path.append("sam:no_detection")
                    warnings.append("sam_no_detection_qwen_fallback")
                cascade_path.append("sam:unloaded")
            else:
                has_sam_evidence = False
                stage_status["sam"] = "disabled"
                cascade_path.append("sam:disabled")
                warnings.append("sam_worker_unavailable")
            stage_latency_ms["sam"] = round((time.monotonic() - grounding_started) * 1000, 2)

            sam_scores = [float(item.get("sam_score", 0.0)) for item in candidates[:grounding_limit]]
            qwen_required = (
                not sam_scores
                or max(sam_scores, default=0.0) < float(settings.SAM_STRONG_SCORE_THRESHOLD)
                or bool(route_info.get("ambiguous"))
                or bool(route_info.get("semantic_requirements"))
                or len({e.get("concept") for item in candidates[:grounding_limit] for e in item.get("grounding_evidence", [])}) > 1
            )
            if route_info.get("ambiguous"):
                warnings.append("sam_ambiguous_qwen_fallback")

            remaining_budget = max(0.0, accurate_deadline - time.monotonic()) if accurate_deadline else 0.0
            should_verify_with_qwen = (
                qwen_required
                and settings.ENABLE_VLM_STAGE2_VERIFY
                and bool(candidates)
                and remaining_budget >= float(settings.QWEN_MIN_REMAINING_SECONDS)
            )
            if qwen_required and not should_verify_with_qwen:
                warnings.append("accurate_partial_budget" if remaining_budget < float(settings.QWEN_MIN_REMAINING_SECONDS) else "accurate_fallback_fast")
                stage_status["qwen"] = "skipped"

            if should_verify_with_qwen:
                verify_limit = max(1, int(settings.QWEN_FALLBACK_TOP_K))
                for candidate in candidates[:verify_limit]:
                    midpoint = (float(candidate.get("t_start", 0.0)) + float(candidate.get("t_end", 0.0))) / 2.0
                    scene = next((item for item in scenes if float(item.get("t_start", 0.0)) <= midpoint <= float(item.get("t_end", duration))), None)
                    if scene:
                        candidate["caption_preview"] = str(scene.get("caption", "") or "")
                verifier_started = time.monotonic()
                models_attempted.append(settings.QWEN_VL_MODEL_ID)
                verifier_frames = [{"_video_path": str(target.get("filepath", ""))}] + frames
                video_fingerprint = grounding_orchestrator._video_fingerprint(video_id)
                verified_head = vlm_verifier.verify(
                    candidates[:verify_limit],
                    verifier_frames,
                    query,
                    budget_seconds=remaining_budget,
                    semantic_requirements=list(route_info.get("semantic_requirements", [])),
                    release_after=True,
                    video_id=video_id,
                    video_fingerprint=video_fingerprint,
                    limit_override=verify_limit,
                )
                candidates = verified_head + candidates[verify_limit:]
                warnings.extend(vlm_verifier.last_warnings)
                cache_hits.update({f"qwen:{key}": value for key, value in vlm_verifier.last_cache_hits.items()})
                stage_latency_ms["qwen"] = round((time.monotonic() - verifier_started) * 1000, 2)
                has_qwen_evidence = any("verifier_confidence" in item for item in candidates[:verify_limit])
                stage_status["qwen"] = "verified" if has_qwen_evidence else "unavailable"
                cascade_path.extend(["qwen:verified" if has_qwen_evidence else "qwen:unavailable", "qwen:unloaded"])
                if has_qwen_evidence:
                    models_used.append(settings.QWEN_VL_MODEL_ID)
                    if any(bool(item.get("action_present")) for item in candidates[:verify_limit]):
                        warnings.append("qwen_fallback_verified")
                    else:
                        warnings.append("qwen_fallback_rejected")

            for candidate in candidates:
                base = float(candidate.get("fast_score", candidate.get("score", 0.0)))
                sam_score = float(candidate.get("sam_score", 0.0))
                has_sam = sam_score > 0.0
                has_qwen = "verifier_confidence" in candidate
                fused_score = self._fuse_accurate_score(
                    base,
                    sam_score,
                    float(candidate["verifier_confidence"]) if has_qwen else None,
                    bool(candidate.get("action_present", False)),
                )
                candidate["grounding_verified"] = has_sam
                candidate["score"] = fused_score
                candidate["rank_score"] = fused_score

            has_qwen_evidence = any("verifier_confidence" in item for item in candidates)
            has_sam_evidence = any(float(item.get("sam_score", 0.0)) > 0.0 for item in candidates)
            strategy_used = "sam_qwen" if has_sam_evidence and has_qwen_evidence else "qwen" if has_qwen_evidence else "sam" if has_sam_evidence else "fast"
            if strategy_used == "fast":
                warnings.append("accurate_fallback_fast")

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
        scored = self._group_candidates(scored)
        # Occurrence numbering is semantic/time-based, while API ranking stays
        # probability-based. This prevents nested proposals from becoming
        # fake occurrence 1/2 results.
        occurrence_by_key: Dict[Tuple[float, float], int] = {}
        for occurrence_index, (_, candidate) in enumerate(sorted(scored, key=lambda pair: (float(pair[1].get("t_start", 0.0)), float(pair[1].get("t_end", 0.0)))), start=1):
            occurrence_by_key[(float(candidate.get("t_start", 0.0)), float(candidate.get("t_end", 0.0)))] = occurrence_index
        # For Accurate object searches, a positive SAM track is stronger
        # evidence than an ungrounded SigLIP-only proposal. Promote verified
        # moments so the UI opens on a frame where the mask is actually
        # available instead of showing SAM=0 on the first card while hiding
        # the useful evidence lower in the list.
        scored.sort(key=lambda pair: (
            0 if pair[1].get("action_present") is True else 1 if pair[1].get("grounding_verified") else 2,
            -pair[0],
            float(pair[1].get("t_start", 0.0)),
        ))
        moments: List[MomentItem] = []
        for probability, candidate in scored[:top_k]:
            start, end = float(candidate["t_start"]), float(candidate["t_end"])
            timestamps = [ts for ts in frame_by_timestamp if start <= ts <= end]
            nearest = min(frames, key=lambda frame: abs(float(frame.get("timestamp", 0.0)) - (start + end) / 2.0), default={})
            visual = max((visual_by_timestamp.get(ts, 0.0) for ts in timestamps), default=0.0)
            caption = max((caption_by_timestamp.get(ts, 0.0) for ts in timestamps), default=0.0)
            verifier = float(candidate.get("verifier_confidence", 0.0))
            breakdown = {"visual": round(float(visual), 4), "caption": round(float(caption), 4),
                         "temporal": round(float(candidate.get("score", 0.0)), 4), "verifier": round(verifier, 4),
                         "sam": round(float(candidate.get("sam_score", 0.0)), 4)}
            scene_caption = ""
            if nearest.get("scene_id") in scene_by_id:
                scene_caption = str(scene_by_id[nearest["scene_id"]].get("caption", "") or "")
            moments.append(MomentItem(t_start=start, t_end=end, score=round(float(probability), 4),
                                      raw_score=round(float(candidate.get("rank_score", candidate.get("score", 0.0))), 6), display_score=round(float(candidate.get("score", 0.0)), 4),
                                      preview_frame_path=nearest.get("frame_path"), caption_preview=scene_caption or None,
                                      modality_breakdown=breakdown,
                                      occurrence_index=occurrence_by_key.get((start, end), 0),
                                      context_t_start=round(float(candidate.get("context_t_start", start)), 3),
                                      context_t_end=round(float(candidate.get("context_t_end", end)), 3),
                                      grounding_evidence=list(candidate.get("grounding_evidence", [])),
                                      verifier_evidence=(
                                          {"event_present": bool(candidate.get("action_present", False)),
                                           "confidence": round(verifier, 4),
                                           "reason": str(candidate.get("verifier_reason", "")),
                                           "model_id": settings.QWEN_VL_MODEL_ID}
                                          if "verifier_confidence" in candidate else None
                                      )))
        # one heatmap value per second, derived only from display timeline
        heatmap = [round(float(display_timeline[min(len(display_timeline) - 1, int(sec * 2))]), 4) for sec in range(max(0, int(np.ceil(duration))))] if len(display_timeline) else []
        return SearchResponse(query=query, video_id=video_id, moments=moments, timeline_heatmap=heatmap,
                              total_duration=round(duration, 2), latency_ms=round((time.monotonic() - started) * 1000, 2),
                              top_k=top_k, profile=profile, calibrated=calibration_available,
                              index_version=settings.VISUAL_INDEX_VERSION, warnings=warnings,
                              strategy_used=strategy_used, models_used=models_used,
                              stage_latency_ms=stage_latency_ms, cache_hits=cache_hits,
                              cascade_path=cascade_path, models_attempted=models_attempted,
                              stage_status=stage_status, planner_version=planner_version,
                              retrieval_backend_requested=retrieval_backend_requested,
                              retrieval_backend_used=retrieval_backend_used,
                              retrieval_model_id=retrieval_model_id,
                              retrieval_embedding_version=retrieval_embedding_version)


search_engine = HybridMomentSearchEngine()
