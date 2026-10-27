"""Check native input routing without loading the model or GPU weights."""
import numpy as np
import pytest
from .encoders import EmbeddingGemmaEncoder


@pytest.mark.parametrize("has_audio", [False, True])
def test_clip_preserves_order_timestamps_audio_and_precision_budget(has_audio):
    encoder = object.__new__(EmbeddingGemmaEncoder)
    encoder.dimension = 768
    frames = [object() for _ in range(6)]
    audio = np.ones(96000, dtype=np.float32) if has_audio else None

    class CaptureModel:
        def encode(self, inputs, **kwargs):
            payload = inputs[0]
            assert payload["video"] is frames
            assert ("audio" in payload) == has_audio
            if has_audio:
                assert payload["audio"] is audio
            assert kwargs["prompt"] == ""
            assert kwargs["processing_kwargs"]["text"]["truncation"] is False
            video = kwargs["processing_kwargs"]["video"]
            assert video["max_soft_tokens"] == 1120
            assert video["do_sample_frames"] is False
            assert video["add_timestamps"] is True
            assert video["video_metadata"][0].frames_indices == list(range(6))
            assert video["video_metadata"][0].fps == 0.999
            return np.ones((1, 768), dtype=np.float32)

    encoder.model = CaptureModel()
    result = encoder.encode_clip(frames, audio, sample_hz=0.999)
    assert result.shape == (1, 768)
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1, atol=1e-6)
