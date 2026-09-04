import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from app.core.config import settings
from app.core.logger import logger

class TemporalBoundaryExtractor:
    """
    SOTA Adaptive Valley & 1D Wasserstein Boundary Extractor with OMTG Support.
    (Visual-Centric SOTA 2025-2026 / TimeLens2 & ICML OMTG Principles).
    
    Features:
    1. Multi-Scale Peak Detection with dynamic standard deviation floor.
    2. Local Valley Seeking: expands outward to local minima.
    3. 1D Wasserstein Optimal Transport Snapping: refines [t_start, t_end]
       against the Cumulative Distribution Function (CDF) for sub-second precision.
    4. Disjoint One-to-Many Temporal Grounding (OMTG) with Temporal NMS.
    """

    def __init__(self, threshold_factor: float = settings.DYNAMIC_THRESHOLD_FACTOR):
        self.threshold_factor = threshold_factor

    def _find_valley_boundaries(
        self,
        scores: np.ndarray,
        peak_idx: int,
        floor_threshold: float
    ) -> Tuple[int, int]:
        """Expand outward from peak_idx to the nearest local minima (valleys)."""
        n = len(scores)
        
        # Expand backwards (start boundary)
        left = peak_idx
        while left > 0:
            if scores[left - 1] > scores[left] and scores[left] <= floor_threshold:
                break
            if scores[left - 1] < floor_threshold * 0.70:
                left -= 1
                break
            left -= 1

        # Expand forwards (end boundary)
        right = peak_idx
        while right < n - 1:
            if scores[right + 1] > scores[right] and scores[right] <= floor_threshold:
                break
            if scores[right + 1] < floor_threshold * 0.70:
                right += 1
                break
            right += 1

        return max(0, left), min(n - 1, right)

    def _wasserstein_refine_boundaries(
        self,
        time_axis: np.ndarray,
        scores: np.ndarray,
        left_idx: int,
        right_idx: int
    ) -> Tuple[float, float]:
        """
        1D Wasserstein Optimal Transport Snapping (TimeLens2 CVPR 2026).
        Refines boundaries by fitting an empirical distribution to the signal energy
        and finding the 10th and 90th energy percentiles within the valley window.
        """
        if right_idx <= left_idx:
            return float(time_axis[left_idx]), float(time_axis[right_idx])

        segment_scores = scores[left_idx:right_idx + 1]
        segment_times = time_axis[left_idx:right_idx + 1]

        # Shift to positive energy
        min_seg = np.min(segment_scores)
        energy = np.maximum(0.0, segment_scores - min_seg)
        total_energy = np.sum(energy)

        if total_energy <= 1e-6:
            return float(segment_times[0]), float(segment_times[-1])

        # Compute empirical Cumulative Distribution Function (CDF)
        cdf = np.cumsum(energy) / total_energy

        # 10% and 90% energy thresholds define tight onset and offset
        p_start_idx = np.searchsorted(cdf, 0.08)
        p_end_idx = min(len(segment_times) - 1, np.searchsorted(cdf, 0.92))

        t_start = float(segment_times[p_start_idx])
        t_end = float(segment_times[p_end_idx])

        return t_start, t_end

    def extract_moments(
        self,
        time_axis: np.ndarray,
        smoothed_scores: np.ndarray,
        threshold_factor: Optional[float] = None,
        min_duration_sec: float = 1.5,
        max_duration_sec: float = 60.0,
        enable_wasserstein: bool = True,
        nms_iou_threshold: float = 0.25
    ) -> List[Dict[str, Any]]:
        """
        Extracts continuous temporal moment intervals [t_start, t_end]
        using Valley Detection, 1D Wasserstein Transport Refinement, and OMTG Temporal NMS.
        """
        if len(smoothed_scores) == 0 or len(time_axis) == 0:
            return []

        if threshold_factor is None:
            threshold_factor = self.threshold_factor

        mu = float(np.mean(smoothed_scores))
        std = float(np.std(smoothed_scores))
        max_s = float(np.max(smoothed_scores))

        raw_threshold = mu + threshold_factor * std
        if raw_threshold >= max_s * 0.95:
            threshold = max(0.15, min(mu, max_s * 0.80))
        else:
            threshold = raw_threshold

        above_indices = np.where(smoothed_scores >= threshold)[0]
        if len(above_indices) == 0:
            return []

        # Group contiguous clusters of active frames
        clusters: List[List[int]] = []
        current_cluster = [above_indices[0]]

        for idx in above_indices[1:]:
            if idx - current_cluster[-1] <= 3:
                current_cluster.append(idx)
            else:
                clusters.append(current_cluster)
                current_cluster = [idx]
        clusters.append(current_cluster)

        candidate_moments: List[Dict[str, Any]] = []

        for grp in clusters:
            peak_local_idx = grp[int(np.argmax(smoothed_scores[grp]))]
            left_valley, right_valley = self._find_valley_boundaries(
                smoothed_scores, peak_local_idx, floor_threshold=threshold
            )
            
            if enable_wasserstein and (right_valley - left_valley) >= 3:
                t_start, t_end = self._wasserstein_refine_boundaries(
                    time_axis, smoothed_scores, left_valley, right_valley
                )
            else:
                t_start = float(time_axis[left_valley])
                t_end = float(time_axis[right_valley])

            # Clamp durations
            if (t_end - t_start) < min_duration_sec:
                t_end = min(float(time_axis[-1]), t_start + min_duration_sec)
            if (t_end - t_start) > max_duration_sec:
                t_end = t_start + max_duration_sec

            peak_score = float(smoothed_scores[peak_local_idx])
            candidate_moments.append({
                "t_start": round(t_start, 2),
                "t_end": round(t_end, 2),
                "score": round(peak_score, 4)
            })

        # Disjoint One-to-Many Temporal Grounding (OMTG) with Temporal NMS
        candidate_moments.sort(key=lambda m: m["score"], reverse=True)
        final_moments: List[Dict[str, Any]] = []

        for m in candidate_moments:
            suppress = False
            for f in final_moments:
                # Calculate 1D Temporal IoU
                inter_s = max(m["t_start"], f["t_start"])
                inter_e = min(m["t_end"], f["t_end"])
                if inter_e > inter_s:
                    intersection = inter_e - inter_s
                    union = (m["t_end"] - m["t_start"]) + (f["t_end"] - f["t_start"]) - intersection
                    iou = intersection / max(1e-5, union)
                    if iou > nms_iou_threshold:
                        suppress = True
                        break
            if not suppress:
                final_moments.append(m)

        return final_moments
