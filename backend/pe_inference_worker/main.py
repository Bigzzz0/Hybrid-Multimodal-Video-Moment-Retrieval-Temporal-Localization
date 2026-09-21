from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import Depends, FastAPI, Header, HTTPException, status

from app.core.config import settings
from app.inference.contracts import PEEmbedImagesRequest, PEEmbedTextRequest
from pe_inference_worker.model_manager import model_manager


ALLOWED_MODELS = {"PE-Core-B16-224", "PE-Core-L14-336"}
_gpu_lock = asyncio.Lock()


def _authorize(token: str | None = Header(default=None, alias="X-Inference-Token")) -> None:
    if settings.PE_WORKER_TOKEN and token != settings.PE_WORKER_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid PE worker token")


def _validate_model(model_id: str) -> None:
    if model_id not in ALLOWED_MODELS:
        raise HTTPException(status_code=400, detail=f"unsupported PE-Core model: {model_id}")


async def _run(model_id: str, method: str, request: Any):
    _validate_model(model_id)
    release_after = bool(getattr(request, "release_after", False))
    async with _gpu_lock:
        model_manager.current_task = method
        try:
            service = await asyncio.to_thread(model_manager.get, model_id)
            model_manager.state = "busy"
            started = time.perf_counter()
            result = None
            attempts = 2 if method == "embed_images" else 1
            current_request = request
            for attempt in range(attempts):
                try:
                    result = await asyncio.to_thread(getattr(service, method), current_request)
                    break
                except torch_oom_error():
                    model_manager.unload()
                    if attempt + 1 >= attempts:
                        raise
                    batch_size = max(1, int(getattr(current_request, "batch_size", 1)) // 2)
                    current_request = current_request.model_copy(update={"batch_size": batch_size})
                    service = await asyncio.to_thread(model_manager.get, model_id)
            model_manager.timings["inference_ms"] = round((time.perf_counter() - started) * 1000, 1)
            return result
        except torch_oom_error() as exc:
            model_manager.unload()
            raise HTTPException(status_code=507, detail=f"PE-Core CUDA OOM: {exc}") from exc
        except Exception as exc:
            model_manager.unload()
            raise HTTPException(status_code=503, detail=f"PE-Core inference unavailable: {exc}") from exc
        finally:
            if release_after:
                model_manager.unload()
            elif model_manager.service is not None:
                model_manager.state = "ready"
            model_manager.current_task = ""


def torch_oom_error():
    try:
        import torch
        return (torch.cuda.OutOfMemoryError,)
    except Exception:
        return (RuntimeError,)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    model_manager.unload()


app = FastAPI(title="Video Event Retrieval PE-Core Worker", lifespan=lifespan)


@app.get("/health")
def health(_: None = Depends(_authorize)) -> Dict[str, Any]:
    return {"status": "healthy", "worker": "pe_core", "models": model_manager.telemetry()}


@app.post("/v1/pe/embed-text")
async def embed_text(request: PEEmbedTextRequest, _: None = Depends(_authorize)):
    return await _run(request.model_id, "embed_text", request)


@app.post("/v1/pe/embed-images")
async def embed_images(request: PEEmbedImagesRequest, _: None = Depends(_authorize)):
    return await _run(request.model_id, "embed_images", request)


@app.post("/v1/models/pe_core/unload")
async def unload(_: None = Depends(_authorize)):
    async with _gpu_lock:
        await asyncio.to_thread(model_manager.unload)
    return {"status": "unloaded", "model": "pe_core"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("pe_inference_worker.main:app", host="127.0.0.1", port=8012, reload=False)
