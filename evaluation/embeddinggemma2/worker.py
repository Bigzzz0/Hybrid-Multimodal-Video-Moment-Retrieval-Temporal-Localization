"""Local GPU broker. One queue owns Gemma/SigLIP and all branch Qwen work.

Run in the isolated pinned Gemma environment, bound to 127.0.0.1:8012.
The branch Qwen worker is private on 8013. Its models are unloaded before
loading Gemma; Gemma is unloaded before a batch of <=8 Qwen verifications.
"""
import gc
import json
import math
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Literal

import numpy as np
import psutil
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image

MODEL_ID = "google/embeddinggemma-2"
REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
BASE_REVISION = "b53b807d3a2d5e2b3911292f2d69e5341cdc064c"
MODEL_ROOT = Path(os.environ.get("GEMMA_MODEL_ROOT", "scratch/embeddinggemma2/models"))
QWEN_URL = os.environ.get("GEMMA_QWEN_URL", "http://127.0.0.1:8013")
DEVICE = os.environ.get("GEMMA_DEVICE", "cuda")
app = FastAPI(title="EmbeddingGemma 2 GPU broker")
lock = threading.RLock()
encoder = None
role = None
dimension = None
timings = {}

class ResourceSampler:
    """Sample whole-card VRAM and the private Qwen process tree RSS."""
    def __init__(self):
        self.stop = threading.Event()
        self.peak_gpu = 0
        self.peak_rss = 0
        self.samples = 0
        self.process = None
        try:
            data = requests.get(QWEN_URL + "/health", timeout=5).json()
            if data.get("process_id"):
                self.process = psutil.Process(data["process_id"])
        except (requests.RequestException, psutil.Error):
            pass
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop.is_set():
            try:
                value = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True, timeout=3)
                self.peak_gpu = max(self.peak_gpu, int(value.strip().splitlines()[0]) * 1024**2)
                self.samples += 1
                if self.process:
                    rss = sum(process.memory_info().rss for process in [self.process, *self.process.children(recursive=True)])
                    self.peak_rss = max(self.peak_rss, rss)
            except (OSError, subprocess.SubprocessError, ValueError, psutil.Error):
                pass
            self.stop.wait(.5)

    def finish(self):
        self.stop.set()
        self.thread.join(timeout=4)
        return {"sampled_whole_gpu_peak_bytes": self.peak_gpu or None,
                "qwen_process_tree_peak_rss_bytes": self.peak_rss or None,
                "samples": self.samples, "interval_seconds": .5,
                "scope": "Whole-card GPU includes other applications; Qwen uses llama.cpp, not broker PyTorch allocations"}

class EmbedRequest(BaseModel):
    modality: Literal["query", "caption", "image", "video", "audio", "joint"]
    texts: list[str] = Field(default_factory=list, max_length=64)
    frame_paths: list[str] = Field(default_factory=list, max_length=12)
    audio_path: str | None = None
    dimension: Literal[768, 512, 256] = 768
    video_tokens: Literal[140, 280, 560, 1120] = 140
    image_tokens: Literal[140, 280, 560, 1120] = 280
    fps: float = Field(default=2.0, gt=0)
    backend: Literal["gemma2", "siglip2"] = "gemma2"

class VerifyBatch(BaseModel):
    events: list[dict] = Field(min_length=1, max_length=8)

def checked_caption(response):
    """The legacy worker can label a fallback summary as valid JSON. Check raw."""
    raw = response.get("text", "")
    left, right = raw.find("{"), raw.rfind("}")
    parsed = json.loads(raw[left:right+1])
    if not isinstance(parsed, dict) or not isinstance(parsed.get("summary"), str) or not parsed["summary"].strip():
        raise ValueError("Incomplete Qwen caption JSON")
    for key in ("objects", "attributes", "actions", "relations", "temporal_events", "uncertainty"):
        if not isinstance(parsed.get(key), list):
            raise ValueError("Missing caption evidence field: " + key)
    return {**response, **parsed, "raw_text": raw, "json_valid": True}

ACTION_REQUIREMENTS = {
    "pushing_punching": ["Visible hand/body contact followed by a forceful push or punch across consecutive frames.",
                        "The contact must be directed at another person's body, not a gate, wall, door, or other object.",
                        "Exclude standing close, walking, greetings, resting, and gently holding or helping someone.",
                        "Use only observable motion; do not infer intent or sound."],
    "falling": ["Visible transition from upright/raised position to falling toward the ground across consecutive frames.",
                "Exclude a person already lying down, sitting down normally, bending, or intentionally resting.",
                "If the transition is not visible, do not claim a fall from posture alone."],
}

BEHAVIOR_REQUIREMENTS = [
    "Recognize an observable punch/push movement directed toward another person from chronological evidence.",
    "Report contact visibility separately; do not infer physical injury, force magnitude, intent or sound.",
    "Exclude isolated poses, pointing, greetings, hugging, assisting, and actions directed at objects.",
    "If ambiguous, report uncertainty rather than assuming either an assault or a normal interaction.",
]

def checked_verdict(response, frame_count, verification_mode="legacy"):
    raw = response.get("raw_text", "")
    left, right = raw.find("{"), raw.rfind("}")
    parsed = json.loads(raw[left:right+1])
    if not isinstance(parsed, dict) or not isinstance(parsed.get("event_present"), bool):
        raise ValueError("Unparseable Qwen verification")
    start, end = parsed.get("start_frame_index"), parsed.get("end_frame_index")
    if (type(start) is not int or type(end) is not int or not 0 <= start <= end < frame_count
            or (parsed["event_present"] and start == end)):
        raise ValueError("Qwen action evidence requires valid chronological frame indices")
    confidence = parsed.get("confidence")
    if (type(confidence) not in (int, float) or not math.isfinite(confidence)
            or not 0 <= confidence <= 1 or not isinstance(parsed.get("reason"), str) or not parsed["reason"].strip()):
        raise ValueError("Incomplete Qwen verification evidence")
    # Never adopt the private worker's partial-JSON fallback or clamped indices.
    if verification_mode == "behavior-v2":
        motion = parsed.get("action_motion")
        if (motion not in ("punch", "push", "none", "uncertain") or
                parsed.get("contact_observed") not in ("yes", "no", "unclear") or
                not isinstance(parsed.get("receiver_response"), str) or not parsed["receiver_response"].strip() or
                not isinstance(parsed.get("uncertainty"), str) or
                parsed["event_present"] != (motion in ("punch", "push"))):
            raise ValueError("Incomplete or contradictory behavior evidence")
    if verification_mode == "open-anomaly-v1":
        assessment = parsed.get("assessment")
        if assessment not in ("normal", "suspicious", "uncertain") or parsed["event_present"] != (assessment == "suspicious"):
            raise ValueError("Contradictory anomaly assessment")
        for key in ("observations", "deviation", "alternative_explanation", "uncertainty"):
            if not isinstance(parsed.get(key), str):
                raise ValueError("Missing contextual evidence: " + key)
        if not parsed["observations"].strip() or (assessment == "suspicious" and not parsed["deviation"].strip()):
            raise ValueError("Empty contextual evidence")
        if assessment == "uncertain":
            parsed["event_present"] = None
    return {**response, **parsed, "raw_text": raw, "confidence_calibrated": False}

def telemetry():
    import torch
    result = {"resident": role, "dimension": dimension, "rss_bytes": psutil.Process().memory_info().rss,
              "dtype": ("fp16" if role == "siglip2" else "bf16") if DEVICE.startswith("cuda") else "fp32", "timings_ms": timings.copy()}
    if torch.cuda.is_available():
        result.update(allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
        try:
            report = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True, timeout=3)
            result["whole_gpu_used_bytes"] = int(report.strip().splitlines()[0]) * 1024**2
        except (OSError, subprocess.SubprocessError, ValueError):
            result["whole_gpu_used_bytes"] = None
    return result

def unload():
    global encoder, role, dimension
    if encoder is not None:
        encoder.close()
    encoder, role, dimension = None, None, None
    gc.collect()

def release_qwen():
    try:
        response = requests.post(QWEN_URL + "/v1/models/unload", timeout=30)
        response.raise_for_status()
    except requests.ConnectionError:
        # An absent private worker cannot have a resident model.
        return

def get_encoder(backend, dims):
    global encoder, role, dimension
    if encoder is not None and (role, dimension) == (backend, dims):
        return encoder
    unload()
    release_qwen()
    folder = MODEL_ROOT / ("embeddinggemma2" if backend == "gemma2" else "siglip2")
    marker = json.loads((folder / "experiment_revision.json").read_text())
    wanted = REVISION if backend == "gemma2" else BASE_REVISION
    if marker.get("revision") != wanted:
        raise ValueError("Pinned model revision does not match local snapshot")
    started = time.perf_counter()
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    if backend == "gemma2":
        from .runtime_encoder import RuntimeEncoder
        encoder = RuntimeEncoder(folder, DEVICE, dims)
    else:
        if dims != 768:
            raise ValueError("SigLIP2 baseline uses its native 768 dimensions")
        from .encoders import SigLIPEncoder
        encoder = SigLIPEncoder(folder, DEVICE, dims)
    role, dimension = backend, dims
    timings["load"] = (time.perf_counter() - started) * 1000
    return encoder

@app.get("/health")
def health():
    return {"status": "healthy", "model_id": MODEL_ID, "revision": REVISION, **telemetry()}

@app.post("/unload")
def unload_endpoint():
    with lock:
        unload()
    return {"status": "unloaded", **telemetry()}

@app.post("/embed")
def embed(request: EmbedRequest):
    with lock:
        started = time.perf_counter()
        images = []
        try:
            model = get_encoder(request.backend, request.dimension)
            for path in request.frame_paths:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            audio = None
            if request.audio_path:
                import soundfile as sf
                audio, rate = sf.read(request.audio_path, dtype="float32")
                if rate != 16000:
                    raise ValueError("Audio must be 16 kHz")
            kind = request.modality
            if request.backend == "siglip2" and kind not in ("query", "image"):
                raise ValueError("SigLIP2 supports only the image/text baseline")
            if kind in ("query", "caption"):
                if not request.texts or any(not text.strip() for text in request.texts):
                    raise ValueError("Non-empty text required")
                result = model.encode_queries(request.texts) if kind == "query" else model.encode_captions(request.texts)
            elif kind == "image":
                if not images:
                    raise ValueError("Image required")
                result = model.encode_images(images, request.image_tokens) if request.backend == "gemma2" else model.encode_images(images)
            elif kind == "audio":
                if audio is None:
                    raise ValueError("Audio required")
                result = model.encode_audio(audio)
            else:
                if not images or (kind == "joint" and audio is None):
                    raise ValueError("Missing joint/video input")
                result = model.encode_clip(images, audio if kind == "joint" else None,
                                           vision_tokens=request.video_tokens, sample_hz=request.fps)
            timings["encode"] = (time.perf_counter() - started) * 1000
            return {"embeddings": result.tolist(), "inputs": getattr(model, "inputs", []),
                    "model_id": MODEL_ID if request.backend == "gemma2" else "google/siglip2-base-patch16-naflex",
                    "revision": REVISION if request.backend == "gemma2" else BASE_REVISION,
                    "telemetry": telemetry()}
        except Exception as exc:
            import torch
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                unload()
                raise HTTPException(507, detail={"code": "oom", "message": str(exc)}) from exc
            raise HTTPException(422, detail={"code": "encode_failed", "message": str(exc)}) from exc
        finally:
            for image in images:
                image.close()

@app.post("/verify-batch")
def verify_batch(request: VerifyBatch):
    with lock:
        unload()
        results = []
        sampler = ResourceSampler()
        sampler.thread.start()
        started = time.perf_counter()
        try:
            for event in request.events:
                calls = 0
                verdict, caption = None, None
                attempts = []
                verification_attempts = []
                try:
                    stamps = event["timestamps"]
                    verification_mode = event.get("verification_mode", "legacy")
                    if verification_mode not in ("legacy", "temporal-v1", "behavior-v2", "open-anomaly-v1"):
                        raise ValueError("Unknown verification mode")
                    if verification_mode == "behavior-v2" and event["kind"] != "pushing_punching":
                        raise ValueError("Behavior experiment supports person punch/push only")
                    if (verification_mode == "open-anomaly-v1") != (event["kind"] == "open_anomaly"):
                        raise ValueError("Open anomaly kind requires matching contextual verification mode")
                    generate_caption = event.get("generate_caption", True)
                    if type(generate_caption) is not bool:
                        raise ValueError("generate_caption must be boolean")
                    if (not stamps or len(stamps) != len(event["frame_paths"]) or len(stamps) > (12 if verification_mode == "open-anomaly-v1" else 8)
                            or any(not math.isfinite(value) for value in stamps)
                            or any(a >= b for a, b in zip(stamps, stamps[1:]))):
                        raise ValueError("Invalid chronological verification frames")
                    if event["kind"] != "shouting":
                        if len(stamps) < 2:
                            raise ValueError("Insufficient frames to verify an action transition")
                        calls += 1
                        response = requests.post(QWEN_URL + "/v1/vlm/verify", json={
                            "query": ("Any observable abnormal activity in this scene" if verification_mode == "open-anomaly-v1" else "A visible punch or push movement directed toward another person"
                                      if verification_mode == "behavior-v2" else
                                      "A person forcefully pushing or punching another person" if event["kind"] == "pushing_punching" else "A person losing balance and falling toward the ground"),
                            "frame_paths": event["frame_paths"], "timestamps": event["timestamps"],
                            "semantic_requirements": BEHAVIOR_REQUIREMENTS if verification_mode == "behavior-v2" else ACTION_REQUIREMENTS.get(event["kind"], []),
                            "verification_mode": verification_mode,
                            "scene_context": event.get("scene_context", ""),
                            "reference_frames": event.get("reference_frames", [])[:2],
                            "max_new_tokens": 512 if verification_mode == "open-anomaly-v1" else 256, "release_after": False,
                            "vlm_backend": "caprl_qwen3vl_4b_q6"}, timeout=240)
                        response.raise_for_status()
                        raw_verdict = response.json()
                        verification_attempts.append(raw_verdict)
                        verdict = checked_verdict(raw_verdict, len(stamps), verification_mode)
                    # Negative decisions already carry a full reason. Generate
                    # rich captions for positive units and acoustic context only.
                    caption_status = "disabled_for_evaluation" if not generate_caption else "not_requested_negative"
                    if generate_caption and (verdict is None or verdict["event_present"]):
                        caption_status = "failed"
                        for budget in (384, 768):
                            calls += 1
                            attempt = {"max_new_tokens": budget}
                            attempts.append(attempt)
                            try:
                                response = requests.post(QWEN_URL + "/v1/vlm/caption", json={
                                    "scene_id": event["id"], "frame_paths": event["frame_paths"], "timestamps": stamps,
                                    "vlm_backend": "caprl_qwen3vl_4b_q6", "release_after": False,
                                    "prompt_version": "observable-events-v2", "max_new_tokens": budget,
                                    "prompt": "Describe only visible actions in these chronological frames. Return one compact JSON object: "
                                    "summary (under 20 words), objects, attributes, actions, relations, temporal_events, uncertainty. "
                                    "Use at most two short strings per array, empty arrays if absent. "
                                    "Cite frame indices for transitions. Do not infer intent, identities, sound or speech. "
                                    "Finish all seven fields; no markdown."}, timeout=240)
                                response.raise_for_status()
                                raw = response.json()
                                attempt["raw_text"] = raw.get("text", "")
                                caption = checked_caption(raw)
                                caption_status = "complete"
                                break
                            except (ValueError, requests.RequestException) as exc:
                                attempt["error"] = str(exc)
                    results.append({"id": event["id"], "status": ("visual_context_only" if caption else "unverified") if verdict is None else ("unverified" if verdict.get("action_motion") == "uncertain" or verdict.get("assessment") == "uncertain" else "verified"),
                                    "caption": caption or {}, "caption_status": caption_status, "caption_attempts": attempts,
                                    "verification_attempts": verification_attempts,
                                    "verdict": verdict, "audio_verified": False, "calls": calls,
                                    "calls_scope": "private_worker_http_requests_internal_repairs_excluded"})
                except Exception as exc:
                    results.append({"id": event.get("id"), "status": "unverified", "error": str(exc),
                                    "verdict": verdict, "caption_attempts": attempts, "audio_verified": False, "calls": calls,
                                    "verification_attempts": verification_attempts,
                                    "calls_scope": "private_worker_http_requests_internal_repairs_excluded"})
        finally:
            try:
                release_qwen()
            finally:
                resources = sampler.finish()
        return {"results": results, "telemetry": {**telemetry(), "qwen_resources": resources,
                "verification_batch_seconds": time.perf_counter()-started}}
