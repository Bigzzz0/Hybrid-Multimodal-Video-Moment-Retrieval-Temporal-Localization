import numpy as np
from typing import List, Dict, Any, Optional
from app.core.config import settings

class TemporalBoundaryExtractor:
    """
    SOTA Adaptive Valley Boundary Extractor.
    Extracts continuous temporal moment intervals [t_start, t_end] by detecting
    local peaks and snapping boundaries to local valley minima.
    """

    def __init__(self, threshold_factor: float = settings.DYNAMIC_THRESHOLD_FACTOR):
        self.threshold_factor = threshold_factor

    def _find_valley_boundaries(
        self,
        scores: np.ndarray,
        peak_idx: int,
        floor_threshold: float
    ) -> (int, int):
        """Expand outward from peak_idx to the nearest local minima (valleys)."""
        n = len(scores)
        
        # Expand backwards (start boundary)
        left = peak_idx
        while left > 0:
            if scores[left - 1] > scores[left] and scores[left] <= floor_threshold:
                break
            if scores[left - 1] < floor_threshold * 0.7:
                left -= 1
                break
            left -= 1

        # Expand forwards (end boundary)
        right = peak_idx
        while right < n - 1:
            if scores[right + 1] > scores[right] and scores[right] <= floor_threshold:
                break
            if scores[right + 1] < floor_threshold * 0.7:
                right += 1
                break
            right += 1

        return max(0, left), min(n - 1, right)

    def extract_moments(
        self,
        time_axis: np.ndarray,
        smoothed_scores: np.ndarray,
        threshold_factor: Optional[float] = None,
        min_duration_sec: float = 1.5,
        max_duration_sec: float = 60.0
    ) -> List[Dict[str, Any]]:
        """
        Calculates dynamic threshold and uses Valley Detection to extract precise boundaries.
        """
        if len(smoothed_scores) == 0 or len(time_axis) == 0:
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

        # Group contiguous clusters
        clusters: List[List[int]] = []
        current_cluster = [above_indices[0]]

        for idx in above_indices[1:]:
            if idx - current_cluster[-1] <= 3:
                current_cluster.append(idx)
            else:
                clusters.append(current_cluster)
                current_cluster = [idx]
        clusters.append(current_cluster)

        moments = []
        for grp in clusters:
            peak_local_idx = grp[int(np.argmax(smoothed_scores[grp]))]
            left_valley, right_valley = self._find_valley_boundaries(
                smoothed_scores, peak_local_idx, floor_threshold=threshold
            )
            
            t_start = float(time_axis[left_valley])
            t_end = float(time_axis[right_valley])

            # Clamp durations
            if (t_end - t_start) < min_duration_sec:
                t_end = min(float(time_axis[-1]), t_start + min_duration_sec)
            if (t_end - t_start) > max_duration_sec:
                t_end = t_start + max_duration_sec

            peak_score = float(smoothed_scores[peak_local_idx])
            moments.append({
                "t_start": round(t_start, 2),
                "t_end": round(t_end, 2),
                "score": round(peak_score, 4)
            })

        # Remove overlapping duplicate intervals and sort by score
        moments.sort(key=lambda m: m["score"], reverse=True)
        unique_moments = []
        for m in moments:
            overlap = False
            for u in unique_moments:
                # Check IoU or substantial overlap
                inter_s = max(m["t_start"], u["t_start"])
                inter_e = min(m["t_end"], u["t_end"])
                if inter_e > inter_s:
                    overlap_len = inter_e - inter_s
                    m_len = m["t_end"] - m["t_start"]
                    if overlap_len / max(1e-5, m_len) > 0.6:
                        overlap = True
                        break
            if not overlap:
                unique_moments.append(m)

        return unique_moments
