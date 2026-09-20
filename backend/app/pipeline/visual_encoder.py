import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Any
from transformers import AutoModel, AutoProcessor
from app.core.config import settings
from app.core.logger import logger

def _extract_tensor(out: Any) -> torch.Tensor:
    """Safely extracts torch.Tensor from ModelOutput or BaseModelOutputWithPooling objects."""
    if isinstance(out, torch.Tensor):
        return out
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        return out.last_hidden_state[:, 0, :] if out.last_hidden_state.ndim == 3 else out.last_hidden_state
    if hasattr(out, "logits") and out.logits is not None:
        return out.logits
    if isinstance(out, (tuple, list)):
        return _extract_tensor(out[0])
    return out

class SigLIP2VisualEncoder:
    """Vision-Language Multimodal Feature Extractor using SigLIP 2 (Google DeepMind)."""

    def __init__(self, model_id: str = settings.SIGLIP2_MODEL_ID, device: str = settings.DEVICE, use_worker: bool = False):
        self.model_id = model_id
        self.device = device
        self.use_worker = use_worker
        self.model = None
        self.processor = None

    def _lazy_load(self):
        if self.model is None:
            logger.info(f"Loading SigLIP 2 model: {self.model_id} on {self.device}...")
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.model = AutoModel.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                device_map="auto" if self.device == "cuda" else None
            ).eval()
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            logger.info(f"SigLIP 2 ({self.model_id}) successfully loaded.")

    def encode_images(self, images: List[Image.Image], batch_size: int = 16, progress_callback=None) -> List[List[float]]:
        """
        Encode list of PIL images into normalized raw SigLIP2 NaFlex vectors.
        Temporal context is intentionally not baked into the stored vector.
        """
        if not images:
            return []

        if not self.use_worker:
            try:
                from app.inference.client import inference_client
                from app.inference.contracts import EmbedImagesRequest
                if inference_client.enabled:
                    # The worker API is path-based; PIL-only callers (tests and
                    # legacy code) continue through the local implementation.
                    paths = [getattr(image, "filename", "") for image in images]
                    if all(paths):
                        response = inference_client.embed_images(EmbedImagesRequest(frame_paths=paths, batch_size=batch_size))
                        if progress_callback:
                            progress_callback(94, f"Generating SigLIP 2 Embeddings: {len(images)}/{len(images)} frames (100%)", "siglip2_embedding", {"sub_percent": 100})
                        return response.embeddings
            except Exception as exc:
                logger.warning(f"SigLIP worker fallback to local encoder: {exc}")

        self._lazy_load()

        all_embeddings = []
        total_images = len(images)
        total_batches = max(1, (total_images + batch_size - 1) // batch_size)

        for b_idx, i in enumerate(range(0, total_images, batch_size)):
            batch = images[i : i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
                tensor_features = _extract_tensor(features)
                if tensor_features.ndim != 2 or tensor_features.shape[-1] != settings.SIGLIP2_EMBEDDING_DIM:
                    raise ValueError(
                        f"SigLIP2 returned dimension {tuple(tensor_features.shape)}; "
                        f"expected {settings.SIGLIP2_EMBEDDING_DIM} for visual index v2"
                    )
                norm_features = F.normalize(tensor_features, p=2, dim=-1)
                all_embeddings.extend(norm_features.cpu().to(torch.float32).tolist())

            if progress_callback:
                processed = min(total_images, i + len(batch))
                sub_pct = int((processed / total_images) * 100)
                macro_pct = min(94, 78 + int((processed / total_images) * 16))
                msg = f"Generating SigLIP 2 Embeddings: {processed}/{total_images} frames ({sub_pct}%)"
                progress_callback(
                    macro_pct,
                    msg,
                    "siglip2_embedding",
                    {
                        "sub_percent": sub_pct,
                        "processed_frames": processed,
                        "total_frames": total_images,
                        "batch": b_idx + 1,
                        "total_batches": total_batches
                    }
                )

        return all_embeddings

    def encode_text(self, text_query: str) -> List[float]:
        """
        Encode natural language query string into normalized 768-dim text vector.
        """
        if not self.use_worker:
            try:
                from app.inference.client import inference_client
                from app.inference.contracts import EmbedTextRequest
                if inference_client.enabled:
                    response = inference_client.embed_text(EmbedTextRequest(texts=[text_query]))
                    if response.embeddings:
                        return response.embeddings[0]
            except Exception as exc:
                logger.warning(f"SigLIP worker text fallback to local encoder: {exc}")

        self._lazy_load()
        inputs = self.processor(text=[text_query], padding="max_length", return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
            tensor_features = _extract_tensor(features)
            norm_features = F.normalize(tensor_features, p=2, dim=-1)
            return norm_features[0].cpu().to(torch.float32).tolist()
