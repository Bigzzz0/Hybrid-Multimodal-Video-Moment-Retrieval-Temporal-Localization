"""Bounded visual temporal reranking for Accurate search."""

import json
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Protocol

from PIL import Image

from app.core.config import settings
from app.core.logger import logger


class TemporalReranker(Protocol):
    def verify(self, candidates: List[Dict[str, Any]], frames: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        ...


class QwenVisualReranker:
    """Verify top candidates using only the sampled frame images."""

    def __init__(self) -> None:
        self._captioner = None
        self.max_seconds = float(getattr(settings, "VLM_RERANK_MAX_SECONDS", 15.0))
        self.last_warnings: List[str] = []

    @property
    def captioner(self):
        if self._captioner is None:
            from app.pipeline.dense_captioner import dense_captioner
            self._captioner = dense_captioner
        return self._captioner

    @staticmethod
    def _parse_response(raw: str, timestamps: List[float]) -> Optional[Dict[str, Any]]:
        match = re.search(r"\{.*\}", raw or "", re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
            required = {"action_present", "start_frame_index", "end_frame_index", "confidence"}
            if set(data) != required or not isinstance(data["action_present"], bool):
                return None
            start, end = data["start_frame_index"], data["end_frame_index"]
            if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int):
                return None
            if not (0 <= start <= end < len(timestamps)):
                return None
            confidence = float(data["confidence"])
            if not 0.0 <= confidence <= 1.0:
                return None
            return data
        except (TypeError, ValueError, json.JSONDecodeError):
            return None

    @staticmethod
    def _sample_frames(
        candidate: Dict[str, Any],
        frames: List[Dict[str, Any]],
        video_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Decode a real 2fps window when the source video is available.

        Indexed frame paths remain a deterministic fallback for CPU tests and
        older callers, but Accurate production verification is deliberately
        based on fresh video decoding rather than the sparse index alone.
        """
        source_path = video_path or next(
            (str(frame.get("_video_path")) for frame in frames if frame.get("_video_path")),
            None,
        )
        if source_path and os.path.exists(source_path):
            try:
                from app.pipeline.video_decoder import GPUVideoDecoder

                # Keep decoding on CPU so the verifier's bounded GPU budget is
                # reserved for Qwen/SigLIP and remains below the 8GB target.
                decoder = GPUVideoDecoder(source_path, use_gpu=False)
                start = max(0.0, float(candidate["t_start"]) - 2.0)
                end = min(float(decoder.duration_sec), float(candidate["t_end"]) + 2.0)
                if end >= start:
                    sample_count = max(2, int(math.floor((end - start) * 2.0)) + 1)
                    raw_timestamps = [min(end, start + (idx * 0.5)) for idx in range(sample_count)]
                    timestamps: List[float] = []
                    for timestamp in raw_timestamps:
                        rounded = round(float(timestamp), 3)
                        if not timestamps or rounded > timestamps[-1]:
                            timestamps.append(rounded)
                    frame_cap = max(1, int(settings.VLM_VERIFY_FRAMES_PER_MOMENT))
                    if len(timestamps) > frame_cap:
                        if frame_cap == 1:
                            timestamps = [timestamps[len(timestamps) // 2]]
                        else:
                            positions = [round(idx * (len(timestamps) - 1) / (frame_cap - 1)) for idx in range(frame_cap)]
                            timestamps = [timestamps[idx] for idx in positions]
                    images = decoder.get_batch_frames([int(timestamp * decoder.fps) for timestamp in timestamps])
                    if len(images) == len(timestamps):
                        return [{"timestamp": timestamp, "image": image} for timestamp, image in zip(timestamps, images)]
            except Exception as exc:
                logger.debug("Accurate verifier source decode fallback: %s", exc)

        start = float(candidate["t_start"]) - 2.0
        end = float(candidate["t_end"]) + 2.0
        selected = [f for f in frames if start <= float(f.get("timestamp", 0.0)) <= end and f.get("frame_path")]
        selected = [f for f in selected if os.path.exists(str(f.get("frame_path")))]
        selected.sort(key=lambda f: float(f.get("timestamp", 0.0)))
        # Approximate 2 fps from the indexed frames while respecting the
        # configured per-candidate cap (8 by default).
        frame_cap = max(1, int(settings.VLM_VERIFY_FRAMES_PER_MOMENT))
        if len(selected) > frame_cap:
            step = max(1, len(selected) // frame_cap)
            selected = selected[::step][:frame_cap]
        return selected

    def verify(self, candidates: List[Dict[str, Any]], frames: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        self.last_warnings = []
        if not candidates:
            return []
        started = time.monotonic()
        verified: List[Dict[str, Any]] = []
        limit = min(max(1, int(settings.VLM_VERIFY_TOP_K)), len(candidates))
        for candidate in candidates[:limit]:
            if time.monotonic() - started >= self.max_seconds:
                self.last_warnings.append("verifier_timeout_fallback")
                verified.append(candidate)
                continue
            sampled = self._sample_frames(candidate, frames)
            if len(sampled) < 2:
                self.last_warnings.append("verifier_insufficient_frames")
                verified.append(candidate)
                continue
            timestamps = [round(float(f["timestamp"]), 3) for f in sampled]
            images = [
                item["image"] if item.get("image") is not None
                else Image.open(str(item["frame_path"])).convert("RGB")
                for item in sampled
            ]
            prompt = (
                "Analyze only visible objects, physical actions, movement and state changes. "
                f"Determine whether the action in query '{query}' is present in the supplied frame sequence. "
                f"Frame timestamps are metadata only: {timestamps}. Use frame indices, never invent times. "
                "Return exactly JSON with keys action_present, start_frame_index, end_frame_index, confidence."
            )
            try:
                response = self.captioner.generate_scene_caption(
                    images,
                    prompt_override=prompt,
                    max_frames=len(images),
                )
                if time.monotonic() - started >= self.max_seconds:
                    self.last_warnings.append("verifier_timeout_fallback")
                    verified.append(candidate)
                    continue
                parsed = self._parse_response(response, timestamps)
                if parsed is None:
                    self.last_warnings.append("verifier_invalid_json_fallback")
                    verified.append(candidate)
                    continue
                refined = dict(candidate)
                # The fusion contract uses the Fast profile's display score
                # (0..1) as the verifier prior.  ``rank_score`` is an internal
                # raw value reserved for calibration and must not make the
                # 0.35 verifier term dominate simply because RRF is small.
                base = float(candidate.get("fast_score", candidate.get("score", 0.0)))
                conf = float(parsed["confidence"])
                refined["verifier_confidence"] = conf
                refined["action_present"] = bool(parsed["action_present"])
                refined["fast_score"] = base
                refined["score"] = 0.65 * base + 0.35 * conf if parsed["action_present"] else base * 0.20
                refined["rank_score"] = refined["score"]
                start_index = parsed["start_frame_index"]
                end_index = parsed["end_frame_index"]
                refined["t_start"] = timestamps[start_index]
                refined["t_end"] = timestamps[end_index]
                if refined["t_end"] <= refined["t_start"]:
                    # Never invent a floating timestamp.  Expand to an
                    # adjacent sampled frame if Qwen returned a single index.
                    if end_index + 1 < len(timestamps):
                        refined["t_end"] = timestamps[end_index + 1]
                    elif start_index > 0:
                        refined["t_start"] = timestamps[start_index - 1]
                verified.append(refined)
            except Exception as exc:
                is_oom = exc.__class__.__name__ in {"OutOfMemoryError", "CUDAOutOfMemoryError"}
                logger.warning("Qwen verifier fallback: %s", exc)
                self.last_warnings.append("verifier_oom_fallback" if is_oom else "verifier_error_fallback")
                if is_oom:
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                verified.append(candidate)
        verified.extend(candidates[limit:])
        verified.sort(key=lambda item: (-float(item.get("score", 0.0)), float(item.get("t_start", 0.0))))
        return verified


class VLMStage2Verifier(QwenVisualReranker):
    """Compatibility alias for existing imports."""

    def verify_and_refine(self, candidate_moments: List[Dict[str, Any]], video_frames: List[Dict[str, Any]], query: str, top_k_verify: int = 3) -> List[Dict[str, Any]]:
        limit = max(1, int(top_k_verify))
        return self.verify(candidate_moments[:limit], video_frames, query) + candidate_moments[limit:]


vlm_verifier = VLMStage2Verifier()
