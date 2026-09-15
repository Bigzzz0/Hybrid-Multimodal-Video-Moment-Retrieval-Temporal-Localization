import numpy as np
from typing import Iterable, List, Tuple, Dict, Any, Union

Interval = Tuple[float, float]
GroundTruth = Union[Interval, List[Interval]]

def compute_temporal_iou(pred_interval: Tuple[float, float], gt_interval: Tuple[float, float]) -> float:
    """
    Computes 1D Temporal Intersection over Union (IoU):
    IoU = (Intersection length) / (Union length)
    """
    p_start, p_end = pred_interval
    g_start, g_end = gt_interval

    inter_start = max(p_start, g_start)
    inter_end = min(p_end, g_end)
    intersection = max(0.0, inter_end - inter_start)

    union_start = min(p_start, g_start)
    union_end = max(p_end, g_end)
    union = max(1e-6, union_end - union_start)

    return float(intersection / union)


def _normalize_ground_truth(gt: GroundTruth) -> List[Interval]:
    """Accept the legacy single interval and the new one-to-many format."""
    if isinstance(gt, tuple) and len(gt) == 2:
        return [(float(gt[0]), float(gt[1]))]
    if isinstance(gt, list) and len(gt) == 2 and all(isinstance(v, (int, float)) for v in gt):
        return [(float(gt[0]), float(gt[1]))]
    return [(float(start), float(end)) for start, end in (gt or [])]


def _best_iou(pred: Interval, gts: Iterable[Interval]) -> float:
    return max((compute_temporal_iou(pred, gt) for gt in gts), default=0.0)


def temporal_f1(
    predictions: List[Interval],
    ground_truths: GroundTruth,
    iou_threshold: float = 0.5,
) -> float:
    """Greedy one-to-one temporal F1 for one-to-many ground truth intervals."""
    gts = _normalize_ground_truth(ground_truths)
    if not predictions and not gts:
        return 1.0
    if not predictions or not gts:
        return 0.0

    pairs = sorted(
        ((compute_temporal_iou(pred, gt), pi, gi)
         for pi, pred in enumerate(predictions)
         for gi, gt in enumerate(gts)),
        reverse=True,
    )
    matched_pred, matched_gt = set(), set()
    true_positive = 0
    for iou, pi, gi in pairs:
        if iou < iou_threshold or pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)
        true_positive += 1

    precision = true_positive / max(1, len(predictions))
    recall = true_positive / max(1, len(gts))
    return float(2 * precision * recall / max(1e-8, precision + recall))

def evaluate_moment_retrieval(
    predictions: List[List[Interval]],
    ground_truths: List[GroundTruth],
    iou_thresholds: List[float] | None = None,
    top_ks: List[int] | None = None,
) -> Dict[str, float]:
    """Calculate single-target recall plus one-to-many coverage metrics."""
    iou_thresholds = iou_thresholds or [0.3, 0.5, 0.7]
    top_ks = top_ks or [1, 5]
    num_queries = len(ground_truths)
    if num_queries == 0:
        return {}

    results = {}
    ious_top1 = []
    delta_t_starts = []

    for k in top_ks:
        for thresh in iou_thresholds:
            correct_count = 0
            for preds, gt in zip(predictions, ground_truths):
                gt_intervals = _normalize_ground_truth(gt)
                top_k_preds = preds[:k]
                max_iou = max((_best_iou(p, gt_intervals) for p in top_k_preds), default=0.0)
                if max_iou >= thresh:
                    correct_count += 1
            results[f"R@{k}@IoU={thresh}"] = round((correct_count / num_queries) * 100.0, 2)

    # Compute top-1 mIoU and nearest-GT temporal start error.
    for preds, gt in zip(predictions, ground_truths):
        gt_intervals = _normalize_ground_truth(gt)
        if preds:
            iou_top1 = _best_iou(preds[0], gt_intervals)
            delta_t = min(abs(preds[0][0] - target[0]) for target in gt_intervals)
        else:
            iou_top1 = 0.0
            delta_t = 100.0
        ious_top1.append(iou_top1)
        delta_t_starts.append(delta_t)

    results["mIoU"] = round(float(np.mean(ious_top1)), 4)
    results["mean_delta_t_start_sec"] = round(float(np.mean(delta_t_starts)), 2)

    for thresh in iou_thresholds:
        f1_values = [
            temporal_f1(preds, gt, iou_threshold=thresh)
            for preds, gt in zip(predictions, ground_truths)
        ]
        results[f"tF1@IoU={thresh}"] = round(float(np.mean(f1_values) * 100.0), 2)

    count_matches = []
    no_match_tp = no_match_fp = no_match_fn = 0
    for preds, gt in zip(predictions, ground_truths):
        gt_count = len(_normalize_ground_truth(gt))
        count_matches.append(float(len(preds) == gt_count))
        if gt_count == 0 and not preds:
            no_match_tp += 1
        elif gt_count == 0 and preds:
            no_match_fp += 1
        elif gt_count > 0 and not preds:
            no_match_fn += 1
    results["count_accuracy"] = round(float(np.mean(count_matches) * 100.0), 2)
    results["no_match_precision"] = round(no_match_tp / max(1, no_match_tp + no_match_fp), 4)
    results["no_match_recall"] = round(no_match_tp / max(1, no_match_tp + no_match_fn), 4)
    return results
