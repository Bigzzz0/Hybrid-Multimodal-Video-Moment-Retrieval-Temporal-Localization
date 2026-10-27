"""Prove that video/audio reach the model and audio changes its representation."""
import json
import sys

import numpy as np
import soundfile as sf
from PIL import Image

from .encoders import EmbeddingGemmaEncoder
from .kamakura import WORK
from .prepare import ROOT, load_config

def main():
    config = load_config()
    media = json.loads((WORK / "media.json").read_text(encoding="utf-8"))
    encoder = EmbeddingGemmaEncoder(ROOT / config["candidate"]["local_dir"], "cuda", 768, full_multimodal=True)
    captured = []
    def inspect(module, args):
        features = args[0]
        captured.append({key: list(value.shape) for key, value in features.items() if hasattr(value, "shape")})
    hook = encoder.model[0].register_forward_pre_hook(inspect)
    images = []
    try:
        # Choose an actual train segment, without using it to tune or train the model.
        records = media["frames"][24:30]
        for frame in records:
            with Image.open(frame["path"]) as image:
                images.append(image.convert("RGB"))
        wave, sr = sf.read(media["audio_path"], dtype="float32")
        audio = wave[round(records[0]["timestamp"]*sr):round(media["frames"][30]["timestamp"]*sr)]
        actual = encoder.encode_clip(images, audio, sample_hz=media["sample_hz"])[0]
        silent = encoder.encode_clip(images, np.zeros_like(audio), sample_hz=media["sample_hz"])[0]
        assert not np.allclose(actual, silent, atol=1e-5), "Audio appears to have been discarded"
        assert all(row["input_ids"][-1] <= 8192 for row in captured)
        assert all("input_features" in row and "pixel_values_videos" in row for row in captured)
        result = {"passed": True, "parameter_count": sum(p.numel() for p in encoder.model.parameters()),
            "input_shapes": captured, "audio_rms": float(np.sqrt(np.mean(audio**2))),
            "audio_vs_silence_cosine": float(actual @ silent), "audio_vs_silence_l2": float(np.linalg.norm(actual-silent)),
            "scope": "Input-routing and non-truncation sanity check; does not establish audio recognition accuracy."}
        (WORK / "input-verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
    finally:
        hook.remove()
        for image in images:
            image.close()
        encoder.close()

if __name__ == "__main__":
    main()
