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
    def verify(
        self,
        candidates: List[Dict[str, Any]],
        frames: List[Dict[str, Any]],
        query: str,
        budget_seconds: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        ...


class QwenVisualReranker:
    """Verify top candidates using only the sampled frame images."""

    def __init__(self) -> None:
        self._captioner = None
        self.max_seconds = float(getattr(settings, "VLM_RERANK_MAX_SECONDS", settings.ACCURATE_MAX_SECONDS))
        self.last_warnings: List[str] = []

    @property
    def captioner(self):
        if self._captioner is None:
            from app.pipeline.dense_captioner import dense_captioner
            self._captioner = dense_captioner
        return self._captioner

    @staticmethod
    def _parse_response(raw: str, timestamps: List[float]) -> Optional[Dict[str, Any]]:
        """Extract the first valid verifier object from natural model output.

        Qwen occasionally wraps the object in a markdown fence or adds a short
        explanation. We accept that harmless formatting while keeping the
        semantic contract strict and ignoring optional extra keys.
        """
        text = raw or ""
        decoder = json.JSONDecoder()
        required = {"action_present", "start_frame_index", "end_frame_index", "confidence"}
        for match in re.finditer(r"\{", text):
            try:
                data, _ = decoder.raw_decode(text[match.start():])
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict) or not required.issubset(data):
                continue
            if not isinstance(data.get("action_present"), bool):
                continue
            start, end = data.get("start_frame_index"), data.get("end_frame_index")
            if isinstance(start, bool) or isinstance(end, bool):
                continue
            try:
                start_num = QwenVisualReranker._coerce_frame_index(start, timestamps)
                end_num = QwenVisualReranker._coerce_frame_index(end, timestamps)
                confidence = float(data.get("confidence"))
            except (TypeError, ValueError, OverflowError):
                continue
            if start_num is None or end_num is None:
                continue
            if not (0 <= start_num <= end_num < len(timestamps)):
                continue
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                continue
            return {
                "action_present": data["action_present"],
                "start_frame_index": start_num,
                "end_frame_index": end_num,
                "confidence": confidence,
                "reason": str(data.get("reason", "")).strip(),
            }
        return None

    @staticmethod
    def _coerce_frame_index(value: Any, timestamps: List[float]) -> Optional[int]:
        """Return a valid frame index, accepting an exact sampled timestamp.

        Qwen sometimes follows the timestamp metadata despite being asked for
        indices. We only map a value when it exactly matches one of the frames
        sent to the model; arbitrary/floating boundaries remain invalid.
        """
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(number):
            return None
        if number.is_integer() and 0 <= int(number) < len(timestamps):
            return int(number)
        for index, timestamp in enumerate(timestamps):
            if math.isclose(number, float(timestamp), rel_tol=0.0, abs_tol=1e-3):
                return index
        return None

    @staticmethod
    def _sample_frames(
        candidate: Dict[str, Any],
        frames: List[Dict[str, Any]],
        video_path: Optional[str] = None,
        decoder: Any = None,
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
        try:
            from app.inference.client import inference_client
            if inference_client.enabled:
                # The worker contract is path-only.  Use indexed local frames
                # so verification never serializes decoded PIL images.
                source_path = None
        except Exception:
            pass
        if source_path and os.path.exists(source_path):
            try:
                from app.pipeline.video_decoder import GPUVideoDecoder

                # Keep decoding on CPU so the verifier's bounded GPU budget is
                # reserved for Qwen/SigLIP and remains below the configured budget.
                decoder = decoder or GPUVideoDecoder(source_path, use_gpu=False)
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
                logger.debug("Accurate verifier source decode fallback: {}", exc)

        start = float(candidate["t_start"]) - 2.0
        end = float(candidate["t_end"]) + 2.0
        selected = [f for f in frames if start <= float(f.get("timestamp", 0.0)) <= end and f.get("frame_path")]
        selected = [f for f in selected if os.path.exists(str(f.get("frame_path")))]
        selected.sort(key=lambda f: float(f.get("timestamp", 0.0)))
        # Approximate 2 fps from the indexed frames while respecting the
        # configured per-candidate cap (4 by default).
        frame_cap = max(1, int(settings.VLM_VERIFY_FRAMES_PER_MOMENT))
        if len(selected) > frame_cap:
            step = max(1, len(selected) // frame_cap)
            selected = selected[::step][:frame_cap]
        return selected

    def verify(
        self,
        candidates: List[Dict[str, Any]],
        frames: List[Dict[str, Any]],
        query: str,
        budget_seconds: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        self.last_warnings = []
        if not candidates:
            return []
        # Model loading is a one-time startup cost. Warm it before starting the
        # bounded verification budget so a cold GPU does not consume the
        # entire budget before the first candidate is analyzed.
        try:
            from app.inference.client import inference_client
            worker_enabled = inference_client.enabled
        except Exception:
            worker_enabled = False
        try:
            if not worker_enabled:
                captioner = self.captioner
                warmup = getattr(captioner, "_lazy_load", None)
                if callable(warmup):
                    warmup()
        except Exception as exc:
            logger.warning("Qwen verifier warm-up failed: {}", exc)
            self.last_warnings.append("verifier_error_fallback")
        started = time.monotonic()
        budget = self.max_seconds if budget_seconds is None else max(0.0, float(budget_seconds))
        try:
            from app.retrieval.query_expander import query_expander
            expanded_query = query_expander.expand_query(query)
            visual_query = expanded_query.get("expanded_search_str", query)
            action_keywords = expanded_query.get("action_keywords", [])
        except Exception:
            visual_query = query
            action_keywords = []
        motion_terms = ("drive", "driving", "drove", "walk", "walking", "run", "running", "turn", "turning", "enter", "entering", "fall", "falling", "move", "moving", "pass", "passing")
        object_only_query = not action_keywords and not any(term in query.lower() for term in motion_terms)
        verified: List[Dict[str, Any]] = []
        limit = min(max(1, int(settings.VLM_VERIFY_TOP_K)), len(candidates))
        source_path = next((str(frame.get("_video_path")) for frame in frames if frame.get("_video_path")), None)
        decoder = None
        if source_path and os.path.exists(source_path):
            try:
                from app.pipeline.video_decoder import GPUVideoDecoder
                decoder = GPUVideoDecoder(source_path, use_gpu=False)
            except Exception as exc:
                logger.debug("Accurate verifier decoder setup fallback: {}", exc)
        for candidate_index, candidate in enumerate(candidates[:limit]):
            if time.monotonic() - started >= budget:
                self.last_warnings.append("verifier_timeout_fallback")
                verified.append(candidate)
                continue
            sampled = self._sample_frames(candidate, frames, video_path=source_path, decoder=decoder)
            if len(sampled) < 2:
                self.last_warnings.append("verifier_insufficient_frames")
                verified.append(candidate)
                continue
            timestamps = [round(float(f["timestamp"]), 3) for f in sampled]
            prompt = (
                "Analyze only visible objects, physical actions, movement and state changes; do not use audio, OCR, "
                "subtitles or readable on-screen text as evidence. "
                f"Determine whether the visual subject or event in query '{query}' is present in the supplied frame sequence. "
                f"Useful visual wording for this query is: '{visual_query}'. "
                f"This is an object-only query: {object_only_query}. "
                "When object-only is true (for example, the query is simply 'car'), answer true if the object is visible "
                "in at least one frame, even when it is parked or stationary. "
                "For a static-object or appearance query, presence in any supplied frame is sufficient; movement is not required. "
                "Only require motion or a state change when the query explicitly asks for it (for example walking, driving past, "
                "turning, entering, or falling). For a vehicle-motion query, a car visibly moving across/along a road counts "
                "as driving or passing; do not require obvious motion blur. If the queried subject/event is visible in any "
                "frame, set action_present true and use confidence as confidence that the query is present. "
                f"Frame index mapping is: {list(enumerate(timestamps))}. "
                "start_frame_index and end_frame_index MUST be integer indices from this mapping (0..N-1), not seconds. "
                "Return exactly one JSON object with keys action_present, start_frame_index, end_frame_index, confidence. "
                "Do not use markdown fences, explanations, or extra text."
            )
            candidate_caption = str(
                candidate.get("caption_preview") or candidate.get("caption") or ""
            ).strip()
            if candidate_caption:
                prompt += (
                    f" Indexed dense visual caption (a weak hint, verify it against the frames): "
                    f"'{candidate_caption}'."
                )
            worker_response = None
            try:
                from app.inference.client import inference_client
                from app.inference.contracts import VerifyRequest
                # Prefer an explicitly injected/local captioner.  This keeps
                # deterministic unit doubles and an already-loaded fallback
                # usable even when the worker is enabled globally.
                if (
                    self._captioner is None
                    and inference_client.enabled
                    and all(item.get("frame_path") for item in sampled)
                ):
                    worker = inference_client.verify(VerifyRequest(
                        query=query,
                        frame_paths=[str(item["frame_path"]) for item in sampled],
                        timestamps=timestamps,
                        caption_hint=str(candidate.get("caption_preview") or candidate.get("caption") or ""),
                        max_new_tokens=settings.QWEN_VERIFY_MAX_NEW_TOKENS,
                    ))
                    worker_response = {
                        "action_present": worker.event_present,
                        "start_frame_index": worker.start_frame_index,
                        "end_frame_index": worker.end_frame_index,
                        "confidence": worker.confidence,
                        "reason": worker.reason,
                    }
            except Exception as exc:
                self.last_warnings.append("qwen_timeout_fallback" if "timeout" in str(exc).lower() else "qwen_worker_unavailable")
                logger.warning("Qwen worker verifier fallback: {}", exc)

            images = []
            generation_seconds = 0.0
            try:
                if worker_response is not None:
                    response = json.dumps(worker_response)
                else:
                    images = [
                        item["image"] if item.get("image") is not None
                        else Image.open(str(item["frame_path"])).convert("RGB")
                        for item in sampled
                    ]
                    generation_started = time.monotonic()
                    response = self.captioner.generate_scene_caption(
                        images,
                        prompt_override=prompt,
                        max_frames=len(images),
                        # The verifier emits a small JSON object. A short output
                        # budget reduces latency and discourages explanations.
                        max_new_tokens=settings.QWEN_VERIFY_MAX_NEW_TOKENS,
                    )
                    generation_seconds = time.monotonic() - generation_started
            except Exception as exc:
                is_oom = exc.__class__.__name__ in {"OutOfMemoryError", "CUDAOutOfMemoryError"}
                logger.warning("Qwen verifier generation fallback: {}", exc)
                self.last_warnings.append("verifier_oom_fallback" if is_oom else "verifier_error_fallback")
                verified.append(candidate)
                continue
            try:
                total_seconds = time.monotonic() - started
                logger.debug(
                    "Qwen verifier candidate {}/{}: {} frames, generation {:.2f}s, "
                    "total {:.2f}s, response: {}",
                    candidate_index + 1,
                    limit,
                    len(sampled),
                    generation_seconds,
                    total_seconds,
                    (response or "")[:500].replace("\n", " "),
                )
                if total_seconds >= budget:
                    self.last_warnings.append("verifier_timeout_fallback")
                    verified.append(candidate)
                    continue
                parsed = self._parse_response(response, timestamps)
                if parsed is None:
                    self.last_warnings.append("verifier_invalid_json_fallback")
                    verified.append(candidate)
                    continue
                # Qwen can conservatively reject a static object even when
                # the indexed dense visual caption independently names it.
                # For object-only queries, use that visual caption as a narrow
                # fallback signal; motion queries never take this shortcut.
                candidate_caption = str(
                    candidate.get("caption_preview") or candidate.get("caption") or ""
                ).lower()
                query_terms = [term for term in re.findall(r"[\w]+", query.lower()) if len(term) > 1]
                if (
                    not parsed["action_present"]
                    and object_only_query
                    and candidate_caption
                    and query_terms
                    and any(term in candidate_caption for term in query_terms)
                ):
                    parsed = dict(parsed)
                    parsed["action_present"] = True
                    parsed["confidence"] = max(0.55, min(float(parsed["confidence"]), 0.75))
                    self.last_warnings.append("verifier_caption_support_fallback")
                refined = dict(candidate)
                # The fusion contract uses the Fast profile's display score
                # (0..1) as the verifier prior.  ``rank_score`` is an internal
                # raw value reserved for calibration and must not make the
                # 0.35 verifier term dominate simply because RRF is small.
                base = float(candidate.get("fast_score", candidate.get("score", 0.0)))
                conf = float(parsed["confidence"])
                refined["verifier_confidence"] = conf
                refined["action_present"] = bool(parsed["action_present"])
                refined["verifier_reason"] = str(parsed.get("reason", "")).strip()
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
                logger.warning("Qwen verifier fallback: {}", exc)
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
