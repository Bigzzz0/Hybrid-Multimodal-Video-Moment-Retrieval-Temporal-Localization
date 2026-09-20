from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from app.core.config import settings
from app.core.logger import logger
from app.db.connection import db_manager
from app.inference.client import inference_client
from app.inference.contracts import GroundRequest
from app.inference.errors import InferenceWorkerError


class GroundingOrchestrator:
    """Candidate-level SAM orchestration with additive LanceDB persistence."""

    @staticmethod
    def _cache_key(video_id: str, start: float, end: float, prompts: List[str], fps: float, max_frames: int) -> str:
        raw = "|".join([
            video_id,
            f"{start:.3f}",
            f"{end:.3f}",
            ",".join(sorted(prompts)),
            settings.SAM_MODEL_ID,
            settings.GROUNDING_VERSION,
            str(fps),
            str(max_frames),
        ])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @classmethod
    def _video_fingerprint(cls, video_id: str) -> str:
        """Use a stable local-file fingerprint without hashing a full CCTV file."""
        rows = cls._rows(db_manager.get_table("videos"), f"id = '{video_id}'", 1)
        path = Path(str(rows[0].get("filepath", ""))) if rows else Path()
        try:
            stat = path.stat()
            return hashlib.sha256(f"{video_id}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")).hexdigest()
        except OSError:
            return hashlib.sha256(video_id.encode("utf-8")).hexdigest()

    @staticmethod
    def _rows(table: Any, where: str, limit: int = 5000) -> List[Dict[str, Any]]:
        try:
            return table.search().where(where).limit(limit).to_list()
        except Exception:
            try:
                rows = table.to_arrow().to_pylist()
                if " = '" in where:
                    key, value = where.split(" = '", 1)
                    rows = [row for row in rows if str(row.get(key)) == value.rstrip("'")]
                return rows[:limit]
            except Exception:
                return []

    def _cache_hit(self, key: str) -> List[Dict[str, Any]] | None:
        rows = self._rows(db_manager.get_table("model_artifact_cache"), f"id = '{key}'", 1)
        if not rows or rows[0].get("status") != "complete":
            return None
        cache = rows[0]
        video_id = str(cache.get("video_id", ""))
        prompts = {value.strip() for value in str(cache.get("normalized_prompt", "")).split(",") if value.strip()}
        start = float(cache.get("t_start", 0.0))
        end = float(cache.get("t_end", start))
        tracks = self._rows(db_manager.get_table("sam_tracks_v1"), f"video_id = '{video_id}'", 10000)
        tracks = [track for track in tracks
                  if str(track.get("grounding_version", "")) == settings.GROUNDING_VERSION
                  and (not prompts or str(track.get("normalized_prompt", "")) in prompts)
                  and float(track.get("t_end", 0.0)) >= start
                  and float(track.get("t_start", 0.0)) <= end]
        try:
            db_manager.get_table("model_artifact_cache").update(
                where=f"id = '{key}'",
                values={"last_accessed_at": datetime.now().isoformat()},
            )
        except Exception:
            pass
        return tracks

    @staticmethod
    def _persist(
        video_id: str,
        video_fingerprint: str,
        key: str,
        response: Any,
        source: str,
        requested_start: float,
        requested_end: float,
        prompts: List[str],
    ) -> List[Dict[str, Any]]:
        now = datetime.now().isoformat()
        track_rows: List[Dict[str, Any]] = []
        observation_rows: List[Dict[str, Any]] = []
        artifact_id = str(uuid.uuid4())
        artifact_dir = settings.GROUNDING_ARTIFACTS_DIR / video_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        for track in response.tracks:
            track_id = str(track.track_id)
            track_rows.append({
                "id": track_id,
                "video_id": video_id,
                "video_fingerprint": video_fingerprint,
                "concept": track.concept,
                "normalized_prompt": track.concept,
                "t_start": float(track.t_start),
                "t_end": float(track.t_end),
                "mean_score": float(track.mean_score),
                "max_score": float(track.max_score),
                "source": source,
                "model_id": response.model_id,
                "grounding_version": response.grounding_version,
                "created_at": now,
            })
            for observation in track.observations:
                observation_id = str(uuid.uuid4())
                mask_path = artifact_dir / f"{observation_id}.json"
                mask_path.write_text(json.dumps(observation.mask_rle, ensure_ascii=False), encoding="utf-8")
                observation_rows.append({
                    "id": observation_id,
                    "track_id": track_id,
                    "video_id": video_id,
                    "concept": track.concept,
                    "timestamp": float(observation.timestamp),
                    "bbox_xyxy": [float(value) for value in observation.bbox_xyxy[:4]],
                    "score": float(observation.score),
                    "mask_artifact_path": str(mask_path),
                    "frame_width": int(observation.frame_width),
                    "frame_height": int(observation.frame_height),
                })
        if track_rows:
            db_manager.get_table("sam_tracks_v1").add(track_rows)
        if observation_rows:
            db_manager.get_table("sam_observations_v1").add(observation_rows)
        cache_row = {
            "id": key,
            "video_id": video_id,
            "task_type": "sam_ground",
            "t_start": float(requested_start),
            "t_end": float(requested_end),
            "normalized_prompt": ",".join(sorted(prompts)),
            "model_id": response.model_id,
            "artifact_version": response.grounding_version,
            "artifact_id": track_rows[0]["id"] if track_rows else artifact_id,
            "status": "complete",
            "created_at": now,
            "last_accessed_at": now,
        }
        try:
            db_manager.get_table("model_artifact_cache").add([cache_row])
        except Exception:
            logger.debug("SAM artifact cache write failed", exc_info=True)
        return track_rows

    def ground_candidates(
        self,
        video_id: str,
        candidates: List[Dict[str, Any]],
        frames: List[Dict[str, Any]],
        prompts: List[str],
        limit: int,
        deadline: float | None = None,
        fps: float | None = None,
        release_after: bool = False,
    ) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, bool]]:
        warnings: List[str] = []
        cache_hits: Dict[str, bool] = {}
        if not candidates or not prompts:
            return candidates, warnings, cache_hits
        if not settings.ENABLE_SAM_GROUNDING:
            warnings.append("sam_paused")
            return candidates, warnings, cache_hits
        if not inference_client.enabled:
            warnings.append("sam_worker_disabled")
            return candidates, warnings, cache_hits
        sampling_fps = float(fps if fps is not None else settings.SAM_SEARCH_FPS)
        for candidate in candidates[:max(1, limit)]:
            candidate["sam_attempted"] = True
            if deadline is not None and time.monotonic() >= deadline:
                warnings.append("accurate_budget_exhausted")
                break
            start = float(candidate.get("t_start", 0.0))
            end = min(float(candidate.get("t_end", start + settings.SAM_MAX_WINDOW_SEC)), start + settings.SAM_MAX_WINDOW_SEC)
            selected = [frame for frame in frames if start <= float(frame.get("timestamp", 0.0)) <= end and frame.get("frame_path")]
            selected.sort(key=lambda row: float(row.get("timestamp", 0.0)))
            cap = max(1, int(settings.SAM_MAX_FRAMES_PER_WINDOW))
            if len(selected) > cap:
                positions = [round(index * (len(selected) - 1) / (cap - 1)) for index in range(cap)] if cap > 1 else [len(selected) // 2]
                selected = [selected[index] for index in positions]
            if not selected:
                warnings.append("sam_insufficient_frames")
                continue
            frame_paths = [str(frame["frame_path"]) for frame in selected]
            timestamps = [float(frame.get("timestamp", 0.0)) for frame in selected]
            video_fingerprint = self._video_fingerprint(video_id)
            key = self._cache_key(video_fingerprint, start, end, prompts, sampling_fps, len(frame_paths))
            cached = self._cache_hit(key)
            if cached is not None:
                cache_hits[key] = True
                tracks = cached
            else:
                try:
                    remaining = None if deadline is None else max(0.1, deadline - time.monotonic())
                    request = GroundRequest(
                        video_fingerprint=video_fingerprint,
                        frame_paths=frame_paths,
                        timestamps=timestamps,
                        prompts=prompts,
                        fps=sampling_fps,
                        max_frames=cap,
                        request_source="search",
                        release_after=release_after,
                    )
                    try:
                        response = inference_client.ground(request, timeout_sec=remaining)
                    except InferenceWorkerError as exc:
                        if "oom" not in str(exc).lower() or len(frame_paths) <= 1:
                            raise
                        reduced_count = max(1, len(frame_paths) // 2)
                        request.frame_paths = frame_paths[:reduced_count]
                        request.timestamps = timestamps[:reduced_count]
                        request.max_frames = reduced_count
                        response = inference_client.ground(request, timeout_sec=remaining)
                    tracks = self._persist(
                        video_id, video_fingerprint, key, response, "on_demand", start, end, prompts
                    )
                    cache_hits[key] = bool(response.cache_hit)
                except InferenceWorkerError as exc:
                    error_text = str(exc).lower()
                    warning = "sam_timeout_fallback" if "timeout" in error_text else "sam_oom_fallback" if "oom" in error_text else "sam_worker_unavailable"
                    warnings.append(warning)
                    candidate["sam_error"] = warning
                    continue
            evidence = []
            scores = []
            for track in tracks:
                score = float(track.get("max_score", 0.0))
                scores.append(score)
                evidence.append({
                    "track_id": str(track.get("id", "")),
                    "concept": str(track.get("concept", "")),
                    "t_start": float(track.get("t_start", start)),
                    "t_end": float(track.get("t_end", end)),
                    "confidence": score,
                })
            if scores:
                coverage = max(0.0, min(1.0, max((item["t_end"] - item["t_start"]) / max(0.001, end - start) for item in evidence)))
                candidate["sam_confidence"] = max(scores)
                candidate["sam_score"] = 0.5 * (sum(scores) / len(scores)) + 0.5 * coverage
                candidate["grounding_evidence"] = evidence
                candidate["mask_artifact_ids"] = [item["track_id"] for item in evidence]
            else:
                candidate["sam_no_detection"] = True
        return candidates, warnings, cache_hits


grounding_orchestrator = GroundingOrchestrator()
