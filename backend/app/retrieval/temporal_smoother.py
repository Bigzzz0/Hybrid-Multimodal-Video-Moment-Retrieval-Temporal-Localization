import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import List, Tuple, Optional
from app.core.config import settings

class TemporalSmoother:
    """
    SOTA Multi-Scale 1D Gaussian Temporal Pyramid Smoother.
    Fuses micro-actions (0.5s), normal actions (1.5s), and macro-activities (3.5s).
    """

    def __init__(self, default_sigma: float = settings.TEMPORAL_GAUSSIAN_SIGMA):
        self.default_sigma = default_sigma
        # Multi-scale Gaussian standard deviations
        self.scale_sigmas = [0.5, 1.5, 3.5]
        self.scale_weights = [0.35, 0.45, 0.20]

    def smooth_timeline(
        self,
        duration_sec: float,
        timestamp_scores: List[Tuple[float, float]],
        sigma: Optional[float] = None,
        resolution_hz: int = 2,
        use_multiscale: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Builds discrete timeline signal S(t) and applies Multi-Scale 1D Gaussian filtering.
        
        Args:
            duration_sec: total video duration
            timestamp_scores: list of (timestamp_sec, score)
            sigma: optional single Gaussian sigma (if use_multiscale is False)
            resolution_hz: samples per second (default: 2 Hz)
            use_multiscale: whether to use 3-scale Gaussian pyramid
            
        Returns:
            (time_axis, smoothed_scores)
        """
        total_steps = max(1, int(np.ceil(duration_sec * resolution_hz)))
        time_axis = np.linspace(0.0, duration_sec, total_steps)
        raw_signal = np.zeros(total_steps, dtype=np.float32)

        # Map discrete timestamp scores to timeline array
        for ts, score in timestamp_scores:
            idx = int(min(total_steps - 1, max(0, round(ts * resolution_hz))))
            raw_signal[idx] = max(raw_signal[idx], float(score))

        if use_multiscale:
            # Multi-scale Gaussian Pyramid Convolution
            smoothed_accum = np.zeros(total_steps, dtype=np.float32)
            for s_val, weight in zip(self.scale_sigmas, self.scale_weights):
                sigma_steps = max(0.5, s_val * resolution_hz)
                layer = gaussian_filter1d(raw_signal, sigma=sigma_steps, mode="nearest")
                smoothed_accum += layer * weight
            smoothed = smoothed_accum
        else:
            actual_sigma = sigma if sigma is not None else self.default_sigma
            sigma_steps = max(0.5, actual_sigma * resolution_hz)
            smoothed = gaussian_filter1d(raw_signal, sigma=sigma_steps, mode="nearest")

        # Normalize smoothed signal to [0.0, 1.0]
        max_val = np.max(smoothed)
        min_val = np.min(smoothed)
        if max_val > min_val:
            norm_smoothed = (smoothed - min_val) / (max_val - min_val)
        else:
            norm_smoothed = smoothed

        return time_axis, norm_smoothed
