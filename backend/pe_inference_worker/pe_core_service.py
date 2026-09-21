from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Iterable, List

import torch
import torch.nn.functional as F
from PIL import Image
from huggingface_hub import hf_hub_download

from app.core.config import settings
from app.inference.contracts import EmbeddingResponse, PEEmbedImagesRequest, PEEmbedTextRequest


ALLOWED_MODELS = {"PE-Core-B16-224", "PE-Core-L14-336"}
MODEL_REVISIONS = {
    "PE-Core-B16-224": settings.PE_CORE_B16_REVISION,
    "PE-Core-L14-336": settings.PE_CORE_L14_REVISION,
}


def _extract_embedding(output: Any, preferred_keys: Iterable[str]) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        for key in preferred_keys:
            if key in output and output[key] is not None:
                return _extract_embedding(output[key], preferred_keys)
        if output:
            return _extract_embedding(next(iter(output.values())), preferred_keys)
    for key in preferred_keys:
        value = getattr(output, key, None)
        if value is not None:
            return _extract_embedding(value, preferred_keys)
    if isinstance(output, (tuple, list)) and output:
        return _extract_embedding(output[0], preferred_keys)
    raise TypeError(f"PE-Core returned unsupported output type: {type(output)!r}")


class PECoreService:
    def __init__(self, model_id: str) -> None:
        if model_id not in ALLOWED_MODELS:
            raise ValueError(f"unsupported PE-Core model: {model_id}")
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.preprocess = None
        self.tokenizer = None

    def load(self) -> None:
        if self.model is not None:
            return
        # Import lazily so the main API and the existing SAM/Qwen worker do not
        # need PE-Core's package tree at import time.
        import core.vision_encoder.pe as pe
        import core.vision_encoder.transforms as transforms

        revision = MODEL_REVISIONS[self.model_id]
        checkpoint_path = hf_hub_download(
            repo_id=f"facebook/{self.model_id}",
            filename=f"{self.model_id}.pt",
            revision=revision,
        )
        self.model = pe.CLIP.from_config(
            self.model_id,
            pretrained=True,
            checkpoint_path=checkpoint_path,
        )
        self.model = self.model.to(self.device).eval()
        self.preprocess = transforms.get_image_transform(self.model.image_size)
        self.tokenizer = transforms.get_text_tokenizer(self.model.context_length)

    def _autocast(self):
        if self.device == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return torch.autocast(device_type="cpu", enabled=False)

    def _validate(self, tensor: torch.Tensor) -> List[List[float]]:
        if tensor.ndim != 2 or tensor.shape[-1] != settings.PE_CORE_EMBEDDING_DIM:
            raise ValueError(
                f"PE-Core returned dimension {tuple(tensor.shape)}; "
                f"expected {settings.PE_CORE_EMBEDDING_DIM}"
            )
        if not torch.isfinite(tensor).all():
            raise ValueError("PE-Core returned NaN or infinite values")
        return F.normalize(tensor.float(), p=2, dim=-1).cpu().tolist()

    @torch.inference_mode()
    def embed_text(self, request: PEEmbedTextRequest) -> EmbeddingResponse:
        self.load()
        tokens = self.tokenizer(request.texts).to(self.device)
        with self._autocast():
            output = self.model.encode_text(tokens)
        return EmbeddingResponse(
            embeddings=self._validate(_extract_embedding(output, ("z_text", "text_features", "feats_text"))),
            model_id=self.model_id,
            embedding_dim=settings.PE_CORE_EMBEDDING_DIM,
            status="generated",
        )

    @torch.inference_mode()
    def embed_images(self, request: PEEmbedImagesRequest) -> EmbeddingResponse:
        self.load()
        if not request.frame_paths:
            return EmbeddingResponse(
                embeddings=[], model_id=self.model_id,
                embedding_dim=settings.PE_CORE_EMBEDDING_DIM, status="generated"
            )

        all_embeddings: List[List[float]] = []
        batch_size = max(1, int(request.batch_size))
        for start in range(0, len(request.frame_paths), batch_size):
            paths = request.frame_paths[start:start + batch_size]
            images: List[Image.Image] = []
            try:
                images = [Image.open(Path(path)).convert("RGB") for path in paths]
                batch = torch.stack([self.preprocess(image) for image in images]).to(self.device)
                with self._autocast():
                    output = self.model.encode_image(batch)
                all_embeddings.extend(
                    self._validate(_extract_embedding(output, ("z_image", "image_features", "feats_image")))
                )
            finally:
                for image in images:
                    image.close()
        return EmbeddingResponse(
            embeddings=all_embeddings,
            model_id=self.model_id,
            embedding_dim=settings.PE_CORE_EMBEDDING_DIM,
            status="generated",
        )

    def unload(self) -> None:
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
