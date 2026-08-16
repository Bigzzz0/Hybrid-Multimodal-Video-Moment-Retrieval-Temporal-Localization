import numpy as np
from typing import List, Dict, Any
from app.core.config import settings

class TemporalBoundaryExtractor:
    """Extracts continuous temporal moment intervals [t_start, t_end] from smoothed score timeline."""

    def __init__(self, threshold_factor: float = settings.DYNAMIC_THRESHOLD_FACTOR):
        self.threshold_factor = threshold_factor

    def extract_moments(
        self,
        time_axis: np.ndarray,
        smoothed_scores: np.ndarray,
        threshold_factor: float = None,
        min_duration_sec: float = 1.5,
        max_duration_sec: float = 60.0
    ) -> List[Dict[str, Any]]:
        """
        Calculates dynamic threshold theta = mu + lambda * sigma,
        and groups contiguous indices into moment intervals [t_start, t_end].
        """
        if len(smoothed_scores) == 0:
            return []

        if threshold_factor is None:
            threshold_factor = self.threshold_factor

        mu = float(np.mean(smoothed_scores))
        std = float(np.std(smoothed_scores))
        threshold = mu + threshold_factor * std

        above_indices = np.where(smoothed_scores >= threshold)[0]
        if len(above_indices) == 0:
            # Fallback: Top peak point
            best_idx = int(np.argmax(smoothed_scores))
            t_s = float(time_axis[best_idx])
            return [{
                "t_start": t_s,
                "t_end": min(float(time_axis[-1]), t_s + 3.0),
                "score": float(smoothed_scores[best_idx])
            }]

        # Group contiguous index clusters
        clusters: List[List[int]] = []
        current_cluster = [above_indices[0]]

        for idx in above_indices[1:]:
            # If gap between indices is <= 2 steps (allows small 1-step dip)
            if idx - current_cluster[-1] <= 2:
                current_cluster.append(idx)
            else:
                clusters.append(current_cluster)
                current_cluster = [idx]
        clusters.append(current_cluster)

        moments = []
        for grp in clusters:
            t_start = float(time_axis[grp[0]])
            t_end = float(time_axis[grp[-1]])
            
            # Ensure minimum duration padding
            if (t_end - t_start) < min_duration_sec:
                t_end = min(float(time_axis[-1]), t_start + min_duration_sec)

            peak_score = float(np.max(smoothed_scores[grp]))
            moments.append({
                "t_start": round(t_start, 2),
                "t_end": round(t_end, 2),
                "score": round(peak_score, 4)
            })

        # Sort moments by score descending
        moments.sort(key=lambda m: m["score"], reverse=True)
        return moments
