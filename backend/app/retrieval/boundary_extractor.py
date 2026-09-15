"""Pure-visual temporal proposal generation and overlap suppression."""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


class TemporalBoundaryExtractor:
    """Generate multi-scale proposals without treating motion as relevance."""

    @staticmethod
    def energy_quantile_refinement(
        transition_energy: Sequence[float], lower: float = 0.08, upper: float = 0.92
    ) -> Tuple[float, float]:
        """Return robust transition-energy bounds for boundary refinement.

        This is deliberately an energy-quantile operation, not a semantic
        relevance score and not a transport distance.  Quantiles reduce the
        effect of isolated decoder spikes while keeping real scene cuts
        available to the boundary snapper.
        """
        values = np.asarray(list(transition_energy), dtype=np.float32)
        if values.size == 0:
            return 0.0, 0.0
        return float(np.quantile(values, lower)), float(np.quantile(values, upper))

    @staticmethod
    def _temporal_iou(a: Dict[str, Any], b: Dict[str, Any]) -> float:
        inter = max(0.0, min(float(a["t_end"]), float(b["t_end"])) - max(float(a["t_start"]), float(b["t_start"])))
        union = max(1e-8, (float(a["t_end"]) - float(a["t_start"])) +
                    (float(b["t_end"]) - float(b["t_start"])) - inter)
        return inter / union

    def _refine_interval(
        self, axis: np.ndarray, scores: np.ndarray, center: int,
        half_window: int, transition_energy: Optional[np.ndarray] = None,
        scene_boundaries: Optional[Sequence[float]] = None,
    ) -> Tuple[float, float]:
        peak = max(0.0, float(scores[center]))
        cutoff = 0.60 * peak
        lower_bound = max(0, center - half_window)
        upper_bound = min(len(scores) - 1, center + half_window)
        left = center
        while left > lower_bound and float(scores[left - 1]) >= cutoff:
            left -= 1
        right = center
        while right < upper_bound and float(scores[right + 1]) >= cutoff:
            right += 1
        if transition_energy is not None and len(transition_energy) == len(axis):
            radius = max(1, int(round(0.5 / max(1e-6, float(axis[1] - axis[0]))))) if len(axis) > 1 else 1
            energy_low, energy_high = self.energy_quantile_refinement(transition_energy)
            for edge, direction in ((left, -1), (right, 1)):
                lo = max(0, edge - radius)
                hi = min(len(axis), edge + radius + 1)
                if hi > lo:
                    local_energy = np.asarray(transition_energy[lo:hi], dtype=np.float32)
                    # Clip extreme outliers to the robust 8/92% range before
                    # selecting the nearest observed transition peak.
                    local_energy = np.clip(local_energy, energy_low, energy_high)
                    best = lo + int(np.argmax(local_energy))
                    if direction < 0:
                        left = min(left, best)
                    else:
                        right = max(right, best)
        start, end = float(axis[max(0, left)]), float(axis[min(len(axis) - 1, right)])
        if scene_boundaries:
            nearby_start = [b for b in scene_boundaries if abs(float(b) - start) <= 1.0]
            nearby_end = [b for b in scene_boundaries if abs(float(b) - end) <= 1.0]
            if nearby_start:
                start = min(nearby_start, key=lambda value: abs(float(value) - start))
            if nearby_end:
                end = min(nearby_end, key=lambda value: abs(float(value) - end))
        if end <= start:
            end = min(float(axis[-1]), start + max(0.5, float(axis[1] - axis[0]) if len(axis) > 1 else 0.5))
        return max(0.0, start), max(start, end)

    def extract_multiscale_proposals(
        self,
        time_axis: np.ndarray,
        scores: np.ndarray,
        raw_scores: Optional[np.ndarray] = None,
        transition_energy: Optional[np.ndarray] = None,
        scene_boundaries: Optional[Sequence[float]] = None,
        min_duration_sec: float = 0.5,
        max_duration_sec: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Return local maxima from 2/4/8/... second rolling windows.

        ``scores`` is display-normalized for peak finding while ``raw_scores``
        is preserved as ``rank_score`` for calibration.
        """
        axis = np.asarray(time_axis, dtype=np.float32)
        signal = np.asarray(scores, dtype=np.float32)
        if len(axis) == 0 or len(signal) == 0:
            return []
        if len(axis) != len(signal):
            raise ValueError("time_axis and scores must have equal length")
        raw = np.asarray(raw_scores if raw_scores is not None else signal, dtype=np.float32)
        if len(raw) != len(signal):
            raw = signal
        hz = 1.0 / max(1e-6, float(axis[1] - axis[0])) if len(axis) > 1 else 1.0
        duration = float(axis[-1])
        # Use 2, 4, 8, ... second windows and cap the final window at the
        # actual video duration so very long videos do not create an
        # out-of-range proposal scale.
        scales = [min(2.0, max(0.5, duration))]
        while scales[-1] < duration:
            next_scale = min(duration, scales[-1] * 2.0)
            if next_scale <= scales[-1] + 1e-6:
                break
            scales.append(next_scale)
        if max_duration_sec is not None:
            scales = [s for s in scales if s <= max_duration_sec] or [max_duration_sec]
        proposals: List[Dict[str, Any]] = []
        for scale in scales:
            width = max(1, int(round(scale * hz)))
            kernel = np.ones(width, dtype=np.float32) / float(width)
            mean_signal = np.convolve(signal, kernel, mode="same")
            radius = max(1, width // 2)
            for center in range(len(signal)):
                lo, hi = max(0, center - radius), min(len(signal), center + radius + 1)
                if float(signal[center]) <= 1e-6:
                    continue
                if float(signal[center]) < float(np.max(signal[lo:hi])) - 1e-7:
                    continue
                if center > 0 and signal[center] < signal[center - 1] - 1e-7:
                    continue
                if center + 1 < len(signal) and signal[center] < signal[center + 1] - 1e-7:
                    continue
                start, end = self._refine_interval(axis, signal, center, radius, transition_energy, scene_boundaries)
                if end - start < min_duration_sec:
                    end = min(float(axis[-1]), start + min_duration_sec)
                if max_duration_sec is not None and end - start > max_duration_sec:
                    end = min(float(axis[-1]), start + max_duration_sec)
                local_mean = float(mean_signal[center])
                contrast = float(signal[center] - np.mean(signal[lo:hi]))
                display_score = float(np.clip(0.60 * signal[center] + 0.30 * local_mean + 0.10 * max(0.0, contrast), 0.0, 1.0))
                mask = (axis >= start) & (axis <= end)
                raw_values = raw[mask] if np.any(mask) else np.asarray([raw[center]], dtype=np.float32)
                raw_peak = float(np.max(raw_values))
                raw_mean = float(np.mean(raw_values))
                raw_contrast = float(raw_peak - np.mean(raw[lo:hi]))
                # Keep an unnormalised score for calibration.  The display
                # score above is intentionally normalized only for the heatmap
                # and peak visualisation.
                raw_rank_score = 0.60 * raw_peak + 0.30 * raw_mean + 0.10 * max(0.0, raw_contrast)
                proposals.append({
                    "t_start": round(start, 3), "t_end": round(end, 3),
                    "score": display_score, "rank_score": float(raw_rank_score), "scale_sec": scale,
                })
        proposals.sort(key=lambda item: (-float(item["score"]), float(item["t_start"]), float(item["t_end"])))
        compact: List[Dict[str, Any]] = []
        for item in proposals:
            if all(self._temporal_iou(item, other) < 0.95 for other in compact):
                compact.append(item)
        return self.apply_gaussian_soft_nms(compact, sigma=0.40, iou_threshold=0.5, score_threshold=0.05)

    def extract_moments(self, time_axis: np.ndarray, scores: np.ndarray, **kwargs: Any) -> List[Dict[str, Any]]:
        """Compatibility wrapper for legacy callers."""
        return self.extract_multiscale_proposals(time_axis, scores, max_duration_sec=kwargs.get("max_duration_sec"))

    def apply_gaussian_soft_nms(
        self, moments: List[Dict[str, Any]], sigma: float = 0.40,
        iou_threshold: float = 0.5, score_threshold: float = 0.05,
    ) -> List[Dict[str, Any]]:
        """Decay overlap scores while preserving disjoint events."""
        pending = [dict(m) for m in moments]
        kept: List[Dict[str, Any]] = []
        sigma = max(1e-6, float(sigma))
        while pending:
            pending.sort(key=lambda item: (-float(item.get("score", 0.0)), float(item.get("t_start", 0.0))))
            best = pending.pop(0)
            kept.append(best)
            updated: List[Dict[str, Any]] = []
            for item in pending:
                iou = self._temporal_iou(best, item)
                if iou > iou_threshold:
                    item["score"] = float(item.get("score", 0.0)) * float(np.exp(-(iou * iou) / sigma))
                if float(item.get("score", 0.0)) >= score_threshold:
                    updated.append(item)
            pending = updated
        kept.sort(key=lambda item: (-float(item.get("score", 0.0)), float(item.get("t_start", 0.0))))
        return kept
