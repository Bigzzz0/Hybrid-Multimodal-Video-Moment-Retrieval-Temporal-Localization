import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import List, Tuple, Optional
from app.core.config import settings

class TemporalSmoother:
    """
    Display-only multi-scale temporal smoother; raw scores remain untouched.
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
        # Include both endpoints on a true resolution_hz grid.  A final short
        # interval is added only when the duration is not an exact grid point.
        step_sec = 1.0 / max(1, int(resolution_hz))
        if duration_sec <= 0.0:
            time_axis = np.asarray([0.0], dtype=np.float32)
        else:
            time_axis = np.arange(0.0, duration_sec, step_sec, dtype=np.float32)
            if len(time_axis) == 0 or time_axis[-1] < duration_sec - 1e-6:
                time_axis = np.append(time_axis, np.float32(duration_sec))
            else:
                time_axis[-1] = np.float32(duration_sec)
        total_steps = len(time_axis)
        # Sort discrete timestamp scores chronologically and interpolate continuous signal
        if not timestamp_scores:
            return time_axis, np.zeros(total_steps, dtype=np.float32)

        sorted_scores = sorted(timestamp_scores, key=lambda x: x[0])
        ts_list = [x[0] for x in sorted_scores]
        sc_list = [float(x[1]) for x in sorted_scores]

        if len(ts_list) == 1:
            raw_signal = np.full(total_steps, sc_list[0], dtype=np.float32)
        else:
            # Handle boundary padding: if discrete scores don't cover endpoints, pad with 0.0
            left_val = sc_list[0] if ts_list[0] <= 1.0 / resolution_hz else 0.0
            right_val = sc_list[-1] if ts_list[-1] >= (duration_sec - 1.0 / resolution_hz) else 0.0
            raw_signal = np.interp(time_axis, ts_list, sc_list, left=left_val, right=right_val).astype(np.float32)

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

        # Keep raw score scale for calibration. The caller may derive a
        # display-only normalized copy for the heatmap/proposal visualization.
        return time_axis, smoothed.astype(np.float32)
