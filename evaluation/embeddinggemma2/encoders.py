"""Local-only encoders for explicitly requested benchmark runs and UI searches."""
from __future__ import annotations

import gc
from pathlib import Path

import numpy as np


def normalized_vectors(values, expected_count: int, dimension: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (expected_count, dimension):
        raise ValueError(f"Embedding shape {array.shape}; expected {(expected_count, dimension)}")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if not np.isfinite(array).all() or np.any(norms <= 1e-8):
        raise ValueError("Non-finite or zero embedding; refusing to produce a misleading ranking")
    return array / norms


class SigLIPEncoder:
    def __init__(self, folder: Path, device: str, dimension: int):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.device = device
        self.dimension = dimension
        self.dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = AutoModel.from_pretrained(
            str(folder), local_files_only=True, trust_remote_code=False, dtype=self.dtype,
        ).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(str(folder), local_files_only=True)

    def _encode(self, *, images=None, texts=None) -> np.ndarray:
        import torch
        from app.pipeline.visual_encoder import _extract_tensor

        if images is not None:
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(text=texts, padding="max_length", return_tensors="pt").to(self.device)
        # Cast floating image inputs only; token ids and masks remain integers.
        for key, tensor in inputs.items():
            if tensor.is_floating_point():
                inputs[key] = tensor.to(self.dtype)
        with torch.inference_mode():
            result = self.model.get_image_features(**inputs) if images is not None else self.model.get_text_features(**inputs)
        vectors = _extract_tensor(result).float().cpu().numpy()
        return normalized_vectors(vectors, len(images if images is not None else texts), self.dimension)

    def encode_images(self, images) -> np.ndarray:
        return self._encode(images=images)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts=texts)

    def close(self) -> None:
        del self.model, self.processor
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class EmbeddingGemmaEncoder:
    def __init__(self, folder: Path, device: str, dimension: int, *, full_multimodal: bool = False):
        import torch
        from sentence_transformers import SentenceTransformer

        self.dimension = dimension
        dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float32
        self.model = SentenceTransformer(
            str(folder), device=device, local_files_only=True, trust_remote_code=False,
            config_kwargs={} if full_multimodal else {"audio_config": None},
            model_kwargs={"torch_dtype": dtype},
        )

    def encode_clip(self, frames, audio, *, vision_tokens: int = 1120, sample_hz: float = 1.0) -> np.ndarray:
        """One joint video+audio embedding; frames are chronological, audio is 16kHz mono."""
        from transformers.video_utils import VideoMetadata
        payload = {"video": frames}
        if audio is not None:
            payload["audio"] = audio
        vectors = self.model.encode(
            [payload], prompt="", batch_size=1,
            processing_kwargs={"text": {"truncation": False}, "video": {"max_soft_tokens": vision_tokens,
                "max_frames": len(frames), "do_sample_frames": False, "add_timestamps": True,
                "video_metadata": [VideoMetadata(total_num_frames=len(frames), fps=sample_hz,
                    frames_indices=list(range(len(frames))))]}},
            normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
        )
        return normalized_vectors(vectors, 1, self.dimension)

    def encode_images(self, images) -> np.ndarray:
        vectors = self.model.encode(
            [{"image": image} for image in images], prompt="", batch_size=len(images),
            normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
        )
        return normalized_vectors(vectors, len(images), self.dimension)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts, prompt_name="SearchQuery", batch_size=len(texts),
            normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
        )
        return normalized_vectors(vectors, len(texts), self.dimension)

    def encode_captions(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts, prompt_name="Document", normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )
        return normalized_vectors(vectors, len(texts), self.dimension)

    def close(self) -> None:
        del self.model
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
