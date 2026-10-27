"""Production adapter reusing the experiment's encoder, with checked input routing."""
import numpy as np
from .encoders import EmbeddingGemmaEncoder, normalized_vectors

class RuntimeEncoder(EmbeddingGemmaEncoder):
    def __init__(self, folder, device, dimension=768):
        if dimension not in (768, 512, 256):
            raise ValueError("Unsupported MRL dimension")
        # The checkpoint projects to 768; truncate and normalize only after encoding.
        super().__init__(folder, device, 768, full_multimodal=True)
        self.output_dimension = dimension
        self.expected = {}
        self.inputs = []
        self.hook = self.model[0].register_forward_pre_hook(self._inspect)

    def _inspect(self, module, args):
        features = args[0]
        shapes = {key: list(value.shape) for key, value in features.items() if hasattr(value, "shape")}
        tokens = features["input_ids"].shape[-1]
        if tokens > 8192:
            raise ValueError(f"Context overflow: {tokens} > 8192")
        expected = self.expected
        if expected.get("image_tokens") and "pixel_values" not in features:
            raise ValueError("Image was dropped before encoding")
        if expected.get("frames"):
            actual = features.get("pixel_values_videos")
            if actual is None or actual.shape[0] != expected["frames"]:
                raise ValueError("Video frames were dropped before encoding")
        if expected.get("audio") and "input_features" not in features:
            raise ValueError("Audio was dropped before encoding")
        self.inputs.append({"shapes": shapes, "tokens": tokens, **expected})

    def _encode(self, payload, *, prompt="", prompt_name=None, processing=None, expected=None):
        self.expected = expected or {}
        self.inputs = []
        options = {"prompt_name": prompt_name} if prompt_name else {"prompt": prompt}
        values = self.model.encode(payload, **options, batch_size=1,
            processing_kwargs={"text": {"truncation": False}, **(processing or {})},
            normalize_embeddings=False, convert_to_numpy=True, show_progress_bar=False)
        native = normalized_vectors(values, len(payload), 768)
        return normalized_vectors(native[:, :self.output_dimension], len(payload), self.output_dimension)

    def encode_queries(self, texts):
        return self._encode(texts, prompt_name="SearchQuery")

    def encode_captions(self, texts):
        return self._encode(texts, prompt_name="Document")

    def encode_images(self, images, vision_tokens=280):
        return self._encode([{"image": image} for image in images],
                            processing={"image": {"max_soft_tokens": vision_tokens}}, expected={"image_tokens": vision_tokens})

    def encode_audio(self, audio):
        audio = self.check_audio(audio)
        return self._encode([{"audio": audio}], expected={"audio": True, "audio_samples": len(audio)})

    @staticmethod
    def check_audio(audio):
        array = np.asarray(audio, dtype=np.float32)
        if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
            raise ValueError("Expected finite non-empty mono 16 kHz audio")
        if len(array) > 6 * 16000 + 1600:
            raise ValueError("Window audio exceeds six-second budget")
        return array

    def encode_clip(self, frames, audio=None, *, vision_tokens=140, sample_hz=2.0):
        from transformers.video_utils import VideoMetadata
        if not 0 < len(frames) <= 12 or len(frames) * vision_tokens + 300 > 8192:
            raise ValueError("Video frame/context budget exceeded")
        payload = {"video": frames}
        if audio is not None:
            payload["audio"] = self.check_audio(audio)
        return self._encode([payload], processing={"video": {
            "max_soft_tokens": vision_tokens, "max_frames": len(frames),
            "do_sample_frames": False, "add_timestamps": True,
            "video_metadata": [VideoMetadata(total_num_frames=len(frames), fps=sample_hz,
                                             frames_indices=list(range(len(frames))))]}},
            expected={"frames": len(frames), "audio": audio is not None,
                      "audio_samples": 0 if audio is None else len(audio), "vision_tokens": vision_tokens})

    def close(self):
        self.hook.remove()
        super().close()
