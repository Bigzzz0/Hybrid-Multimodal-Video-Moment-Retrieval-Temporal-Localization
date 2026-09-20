from __future__ import annotations

from typing import Any, Dict, Optional

import httpx

from app.core.config import settings
from app.inference.contracts import (
    AnswerRequest,
    AnswerResponse,
    CaptionRequest,
    CaptionResponse,
    EmbedImagesRequest,
    EmbedTextRequest,
    EmbeddingResponse,
    GroundRequest,
    GroundResponse,
    VerifyRequest,
    VerifyResponse,
)
from app.inference.errors import (
    InferenceWorkerError,
    InferenceWorkerOOM,
    InferenceWorkerTimeout,
    InferenceWorkerUnavailable,
)


class InferenceWorkerClient:
    """Synchronous localhost client used by the existing sync pipeline."""

    def __init__(self) -> None:
        self.base_url = settings.INFERENCE_WORKER_URL.rstrip("/")

    @property
    def enabled(self) -> bool:
        return bool(settings.INFERENCE_WORKER_ENABLED)

    def _post(self, path: str, payload: Dict[str, Any], model: Any, timeout_sec: Optional[float] = None) -> Any:
        if not self.enabled:
            raise InferenceWorkerUnavailable("inference worker is disabled")
        headers = {}
        if settings.INFERENCE_WORKER_TOKEN:
            headers["X-Inference-Token"] = settings.INFERENCE_WORKER_TOKEN
        try:
            timeout = float(timeout_sec if timeout_sec is not None else settings.INFERENCE_REQUEST_TIMEOUT_SEC)
            with httpx.Client(timeout=max(0.1, timeout)) as client:
                response = client.post(f"{self.base_url}{path}", json=payload, headers=headers)
            if response.status_code == 503:
                raise InferenceWorkerUnavailable(response.text)
            if response.status_code == 408 or response.status_code == 504:
                raise InferenceWorkerTimeout(response.text)
            if response.status_code == 507:
                raise InferenceWorkerOOM(response.text)
            response.raise_for_status()
            return model.model_validate(response.json())
        except httpx.TimeoutException as exc:
            raise InferenceWorkerTimeout(str(exc)) from exc
        except httpx.ConnectError as exc:
            raise InferenceWorkerUnavailable(str(exc)) from exc
        except InferenceWorkerError:
            raise
        except httpx.HTTPError as exc:
            raise InferenceWorkerError(str(exc)) from exc

    def health(self) -> Dict[str, Any]:
        if not self.enabled:
            return {"status": "disabled"}
        headers = {}
        if settings.INFERENCE_WORKER_TOKEN:
            headers["X-Inference-Token"] = settings.INFERENCE_WORKER_TOKEN
        try:
            with httpx.Client(timeout=3.0) as client:
                response = client.get(f"{self.base_url}/health", headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            return {"status": "unavailable", "error": str(exc)}

    def caption(self, request: CaptionRequest) -> CaptionResponse:
        return self._post("/v1/qwen/caption", request.model_dump(), CaptionResponse)

    def embed_text(self, request: EmbedTextRequest) -> EmbeddingResponse:
        return self._post("/v1/siglip/embed-text", request.model_dump(), EmbeddingResponse)

    def embed_images(self, request: EmbedImagesRequest) -> EmbeddingResponse:
        return self._post("/v1/siglip/embed-images", request.model_dump(), EmbeddingResponse)

    def verify(self, request: VerifyRequest) -> VerifyResponse:
        return self._post("/v1/qwen/verify", request.model_dump(), VerifyResponse)

    def answer(self, request: AnswerRequest) -> AnswerResponse:
        return self._post("/v1/qwen/answer", request.model_dump(), AnswerResponse)

    def ground(self, request: GroundRequest, timeout_sec: Optional[float] = None) -> GroundResponse:
        return self._post("/v1/sam/ground-video", request.model_dump(), GroundResponse, timeout_sec=timeout_sec)


inference_client = InferenceWorkerClient()
