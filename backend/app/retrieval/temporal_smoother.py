import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import List, Tuple
from app.core.config import settings

class TemporalSmoother:
    """1D Gaussian Temporal Convolution for smoothing similarity timeline and removing false spikes."""

    def __init__(self, default_sigma: float = settings.TEMPORAL_GAUSSIAN_SIGMA):
        self.default_sigma = default_sigma

    def smooth_timeline(
        self,
        duration_sec: float,
        timestamp_scores: List[Tuple[float, float]],
        sigma: float = None,
        resolution_hz: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Builds a continuous discrete timeline signal S(t) and applies 1D Gaussian filtering.
        
        Args:
            duration_sec: total video duration
            timestamp_scores: list of (timestamp_sec, score)
            sigma: Gaussian standard deviation
            resolution_hz: samples per second (default: 2 Hz, i.e. 0.5s step)
            
        Returns:
            (time_axis, smoothed_scores)
        """
        if sigma is None:
            sigma = self.default_sigma

        total_steps = max(1, int(np.ceil(duration_sec * resolution_hz)))
        time_axis = np.linspace(0.0, duration_sec, total_steps)
        raw_signal = np.zeros(total_steps, dtype=np.float32)

        # Map discrete timestamp scores to timeline array
        for ts, score in timestamp_scores:
            idx = int(min(total_steps - 1, max(0, round(ts * resolution_hz))))
            raw_signal[idx] = max(raw_signal[idx], float(score))

        # Apply 1D Gaussian Convolution: S_smooth(t) = S(t) * G_sigma(t)
        sigma_steps = sigma * resolution_hz
        smoothed = gaussian_filter1d(raw_signal, sigma=max(0.5, sigma_steps), mode="nearest")

        # Normalize smoothed signal to [0.0, 1.0] for consistent rendering & thresholding
        max_val = np.max(smoothed)
        if max_val > 0:
            norm_smoothed = smoothed / max_val
        else:
            norm_smoothed = smoothed

        return time_axis, norm_smoothed
