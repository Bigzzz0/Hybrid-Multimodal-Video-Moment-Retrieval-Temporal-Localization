from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict

from fastapi import Depends, FastAPI, Header, HTTPException, status

from app.core.config import settings
from app.core.logger import logger
from app.inference.contracts import (
    AnswerRequest,
    CaptionRequest,
    EmbedImagesRequest,
    EmbedTextRequest,
    GroundRequest,
    VerifyRequest,
    VLMUnloadRequest,
)
from inference_worker.model_manager import model_manager
from inference_worker.vlm_service import VariantVLMService
from inference_worker.sam_service import SAM31Grounder
from inference_worker.scheduler import inference_scheduler
from inference_worker.siglip_service import SigLIPService
from app.inference.vlm_registry import get_variant


def _authorize(token: str | None = Header(default=None, alias="X-Inference-Token")) -> None:
    configured = settings.INFERENCE_WORKER_TOKEN
    if configured and token != configured:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid inference worker token")


async def _run(name: str, factory: Callable[[], Any], method: str, request: Any, priority: str = "vqa"):
    release_after = bool(getattr(request, "release_after", False))
    try:
        def operation():
            model_manager.current_model = name
            model_manager.current_task = method
            model_manager.cascade_stage = name
            try:
                service = model_manager.get(name, factory)
                model_manager.states[name] = "busy"
                inference_started = time.perf_counter()
                result = getattr(service, method)(request)
                model_manager.timings.setdefault(name, {})["inference_ms"] = round(
                    (time.perf_counter() - inference_started) * 1000, 1
                )
                return result
            finally:
                if release_after:
                    model_manager.unload(name)
                elif name in model_manager.services:
                    model_manager.states[name] = "ready"
                elif model_manager.states.get(name) == "busy":
                    model_manager.states[name] = "unloaded"
                model_manager.current_model = ""
                model_manager.current_task = ""
                model_manager.cascade_stage = "idle"
        return await inference_scheduler.submit(priority, operation)
    except torch_oom_error() as exc:
        model_manager.unload(name)
        raise HTTPException(status_code=507, detail=f"{name} CUDA OOM: {exc}") from exc
    except Exception as exc:
        if release_after:
            model_manager.unload(name)
        elif model_manager.states.get(name) == "busy":
            model_manager.states[name] = "error"
        logger.exception("%s inference failed", name)
        raise HTTPException(status_code=503, detail=f"{name} inference unavailable: {exc}") from exc


def torch_oom_error():
    try:
        import torch
        return (torch.cuda.OutOfMemoryError,)
    except Exception:
        return (RuntimeError,)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await inference_scheduler.start()
    if settings.INFERENCE_WARMUP_QWEN:
        try:
            # Pay the one-time model load before Accurate starts its bounded
            # verification budget. The model manager still enforces VRAM.
            await asyncio.to_thread(model_manager.get, "qwen", lambda: VariantVLMService("qwen3_vl_2b"))
        except Exception as exc:
            # Keep Fast search available if the optional warm-up fails.
            model_manager.states["qwen"] = "error"
            print(f"Qwen warm-up failed: {exc}")
    yield
    await inference_scheduler.stop()
    model_manager.unload_inactive()


app = FastAPI(title="Video Event Retrieval Local Inference Worker", lifespan=lifespan)


@app.get("/health")
def health(_: None = Depends(_authorize)) -> Dict[str, Any]:
    telemetry = model_manager.telemetry()
    telemetry["queue_depth"] = inference_scheduler.queue.qsize()
    return {"status": "healthy", "worker": "local", "models": telemetry}


@app.post("/v1/siglip/embed-text")
async def embed_text(request: EmbedTextRequest, _: None = Depends(_authorize)):
    return await _run("siglip", SigLIPService, "embed_text", request, priority="interactive_search")


@app.post("/v1/siglip/embed-images")
async def embed_images(request: EmbedImagesRequest, _: None = Depends(_authorize)):
    return await _run("siglip", SigLIPService, "embed_images", request, priority="ingestion")


@app.post("/v1/qwen/caption")
async def caption(request: CaptionRequest, _: None = Depends(_authorize)):
    request.vlm_backend = "qwen3_vl_2b"
    return await _run("qwen", lambda: VariantVLMService("qwen3_vl_2b"), "caption", request, priority="ingestion")


@app.post("/v1/qwen/verify")
async def verify(request: VerifyRequest, _: None = Depends(_authorize)):
    request.vlm_backend = "qwen3_vl_2b"
    return await _run("qwen", lambda: VariantVLMService("qwen3_vl_2b"), "verify", request, priority="interactive_search")


@app.post("/v1/qwen/answer")
async def answer(request: AnswerRequest, _: None = Depends(_authorize)):
    request.vlm_backend = "qwen3_vl_2b"
    return await _run("qwen", lambda: VariantVLMService("qwen3_vl_2b"), "answer", request, priority="vqa")


def _vlm_worker_name(backend: str) -> str:
    try:
        variant = get_variant(backend)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return "q6" if variant.backend == "caprl_qwen3vl_4b_q6" else "qwen"


@app.post("/v1/vlm/caption")
async def vlm_caption(request: CaptionRequest, _: None = Depends(_authorize)):
    name = _vlm_worker_name(request.vlm_backend)
    return await _run(name, lambda: VariantVLMService(request.vlm_backend), "caption", request, priority="ingestion")


@app.post("/v1/vlm/verify")
async def vlm_verify(request: VerifyRequest, _: None = Depends(_authorize)):
    name = _vlm_worker_name(request.vlm_backend)
    return await _run(name, lambda: VariantVLMService(request.vlm_backend), "verify", request, priority="interactive_search")


@app.post("/v1/vlm/unload")
async def vlm_unload(request: VLMUnloadRequest, _: None = Depends(_authorize)):
    name = _vlm_worker_name(request.vlm_backend)
    model_manager.unload(name)
    return {"status": "unloaded", "backend": request.vlm_backend, "model_state": model_manager.states.get(name, "unloaded")}


@app.post("/v1/sam/ground-video")
async def ground_video(request: GroundRequest, _: None = Depends(_authorize)):
    priority = "interactive_search" if request.request_source == "search" else "ingestion"
    return await _run("sam", SAM31Grounder, "ground", request, priority=priority)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("inference_worker.main:app", host="127.0.0.1", port=8011, reload=False)
