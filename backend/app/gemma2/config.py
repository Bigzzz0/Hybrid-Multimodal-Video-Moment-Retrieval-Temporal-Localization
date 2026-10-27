from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path

@dataclass(frozen=True)
class IndexConfig:
    model_id: str = "google/embeddinggemma-2"
    revision: str = "914f7f89142e33e77833254d9c9b90c3cef7303b"
    dimension: int = 768
    window_seconds: float = 6.0
    stride_seconds: float = 3.0
    image_fps: float = 1.0
    video_fps: float = 2.0
    max_frames: int = 12
    video_tokens: int = 140
    image_tokens: int = 280
    audio_sample_rate: int = 16000
    modalities: tuple = ("image", "video", "audio", "caption")
    version: str = "gemma2-events-v1"

    def __post_init__(self):
        if self.dimension not in (768, 512, 256):
            raise ValueError("Use 768, 512 or 256 dimensions")
        if self.video_tokens not in (140, 280, 560, 1120):
            raise ValueError("Unsupported vision budget")
        if self.image_tokens not in (140, 280, 560, 1120) or not 1 <= self.max_frames <= 12:
            raise ValueError("Unsupported image/frame budget")
        if self.audio_sample_rate != 16000 or self.window_seconds > 6:
            raise ValueError("Use mono 16 kHz audio in windows no longer than six seconds")
        if min(self.window_seconds, self.stride_seconds, self.image_fps, self.video_fps) <= 0:
            raise ValueError("Sampling values must be positive")
        ratio = self.video_fps/self.image_fps
        if ratio < 1 or abs(ratio-round(ratio)) > 1e-6:
            raise ValueError("Image sampling must align with the chronological video frames")
        if not self.modalities or set(self.modalities)-{"image", "video", "audio", "caption"}:
            raise ValueError("Unsupported modality configuration")
        if self.window_seconds * self.video_fps > self.max_frames or self.max_frames * self.video_tokens + 300 > 8192:
            raise ValueError("Frame/context budget would silently discard input; reduce window/fps/tokens")

    def namespace(self, fingerprint: str, captions_hash: str = "") -> str:
        payload = {**asdict(self), "video_sha256": fingerprint, "captions_sha256": captions_hash}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def json(self):
        return asdict(self)

ROOT = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get("GEMMA_DATA_DIR", ROOT / "data/gemma2"))
WORKER_URL = os.environ.get("GEMMA_WORKER_URL", "http://127.0.0.1:8012")

def configured_index():
    return IndexConfig(dimension=int(os.environ.get("GEMMA_DIMENSION", "768")),
                       video_tokens=int(os.environ.get("GEMMA_VIDEO_TOKENS", "140")))
