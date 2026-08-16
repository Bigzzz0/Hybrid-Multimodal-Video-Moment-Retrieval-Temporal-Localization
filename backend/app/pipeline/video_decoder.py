import os
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
from app.core.logger import logger
from app.core.config import settings

class GPUVideoDecoder:
    """Hardware-accelerated video reader using Decord (NVDEC on GPU or CPU fallback)."""

    def __init__(self, video_path: str, use_gpu: bool = True):
        self.video_path = video_path
        self.use_gpu = use_gpu and (settings.DEVICE == "cuda")
        self.vr = None
        self._init_reader()

    def _init_reader(self):
        try:
            import decord
            if self.use_gpu:
                try:
                    self.ctx = decord.gpu(0)
                    self.vr = decord.VideoReader(self.video_path, ctx=self.ctx)
                    logger.info(f"Initialized Decord with GPU NVDEC for {os.path.basename(self.video_path)}")
                except Exception as e:
                    logger.warning(f"Decord GPU init failed ({e}), falling back to CPU.")
                    self.ctx = decord.cpu(0)
                    self.vr = decord.VideoReader(self.video_path, ctx=self.ctx)
            else:
                self.ctx = decord.cpu(0)
                self.vr = decord.VideoReader(self.video_path, ctx=self.ctx)
        except Exception as ex:
            logger.error(f"Failed to initialize Decord VideoReader: {ex}")
            raise ex

    @property
    def fps(self) -> float:
        return float(self.vr.get_avg_fps()) if self.vr else 30.0

    @property
    def total_frames(self) -> int:
        return len(self.vr) if self.vr else 0

    @property
    def duration_sec(self) -> float:
        return self.total_frames / self.fps if self.fps > 0 else 0.0

    @property
    def resolution(self) -> str:
        if self.vr and len(self.vr) > 0:
            sample = self.vr[0]
            return f"{sample.shape[1]}x{sample.shape[0]}"
        return "Unknown"

    def get_batch_frames(self, frame_indices: List[int]) -> List[Image.Image]:
        """Fetch multiple frames by index as PIL Images."""
        if not frame_indices or not self.vr:
            return []
        
        # Clamp indices
        max_idx = self.total_frames - 1
        valid_indices = [min(max(0, idx), max_idx) for idx in frame_indices]
        
        batch = self.vr.get_batch(valid_indices).asnumpy()
        return [Image.fromarray(f) for f in batch]

    def get_frame_at_timestamp(self, timestamp_sec: float) -> Optional[Image.Image]:
        frame_idx = int(timestamp_sec * self.fps)
        frames = self.get_batch_frames([frame_idx])
        return frames[0] if frames else None
