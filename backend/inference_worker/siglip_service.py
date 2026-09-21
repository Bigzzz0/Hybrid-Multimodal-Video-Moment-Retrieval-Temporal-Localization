from __future__ import annotations

from typing import List

from PIL import Image

from app.core.config import settings
from app.inference.contracts import EmbedImagesRequest, EmbedTextRequest, EmbeddingResponse
from app.pipeline.visual_encoder import SigLIP2VisualEncoder


class SigLIPService:
    def __init__(self) -> None:
        self.encoder = SigLIP2VisualEncoder(use_worker=True)

    def load(self) -> None:
        self.encoder._lazy_load()

    def unload(self) -> None:
        self.encoder.model = None
        self.encoder.processor = None

    def embed_text(self, request: EmbedTextRequest) -> EmbeddingResponse:
        return EmbeddingResponse(
            embeddings=[self.encoder.encode_text(text) for text in request.texts],
            model_id=settings.SIGLIP2_MODEL_ID,
            embedding_dim=settings.SIGLIP2_EMBEDDING_DIM,
            status="generated",
        )

    def embed_images(self, request: EmbedImagesRequest) -> EmbeddingResponse:
        images = [Image.open(path).convert("RGB") for path in request.frame_paths]
        try:
            embeddings = self.encoder.encode_images(images, batch_size=max(1, int(request.batch_size)))
        finally:
            for image in images:
                image.close()
        return EmbeddingResponse(
            embeddings=embeddings,
            model_id=settings.SIGLIP2_MODEL_ID,
            embedding_dim=settings.SIGLIP2_EMBEDDING_DIM,
            status="generated",
        )
