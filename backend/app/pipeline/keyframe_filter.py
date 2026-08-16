import numpy as np
from PIL import Image
from typing import List, Tuple
from app.core.config import settings
from app.core.logger import logger

class SSIMKeyframeFilter:
    """Adaptive Structural Similarity Index (SSIM) & Difference Filter for Keyframe Sampling."""

    def __init__(self, ssim_threshold: float = settings.SSIM_THRESHOLD):
        self.ssim_threshold = ssim_threshold

    @staticmethod
    def _compute_fast_difference(img1: Image.Image, img2: Image.Image) -> float:
        """Fast normalized luminance difference approximation of SSIM on 64x64 thumbnails."""
        t1 = np.array(img1.convert('L').resize((64, 64)), dtype=np.float32)
        t2 = np.array(img2.convert('L').resize((64, 64)), dtype=np.float32)
        
        # Mean squared difference normalized
        diff = np.mean(np.abs(t1 - t2)) / 255.0
        return float(diff)

    def filter_keyframes(
        self,
        frames: List[Image.Image],
        timestamps: List[float]
    ) -> Tuple[List[Image.Image], List[float]]:
        """
        Filters out visually redundant frames.
        Returns: (filtered_frames, filtered_timestamps)
        """
        if not frames:
            return [], []

        filtered_frames = [frames[0]]
        filtered_timestamps = [timestamps[0]]
        last_kept_frame = frames[0]

        for i in range(1, len(frames)):
            diff = self._compute_fast_difference(last_kept_frame, frames[i])
            # If difference exceeds threshold, keep it
            if diff >= (1.0 - self.ssim_threshold) * 0.15:
                filtered_frames.append(frames[i])
                filtered_timestamps.append(timestamps[i])
                last_kept_frame = frames[i]

        logger.debug(f"Filtered keyframes from {len(frames)} -> {len(filtered_frames)} (kept {len(filtered_frames)/max(1, len(frames))*100:.1f}%)")
        return filtered_frames, filtered_timestamps
