import numpy as np
from typing import List, Tuple, Dict, Any

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

def evaluate_moment_retrieval(
    predictions: List[List[Tuple[float, float]]], # Top-K predicted [t_s, t_e] for each query
    ground_truths: List[Tuple[float, float]],     # Single ground truth [t_s, t_e] for each query
    iou_thresholds: List[float] = [0.3, 0.5, 0.7],
    top_ks: List[int] = [1, 5]
) -> Dict[str, float]:
    """
    Calculates R@K@IoU and Mean IoU across all evaluation samples.
    """
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
                # Check top-k predictions
                top_k_preds = preds[:k]
                max_iou = max([compute_temporal_iou(p, gt) for p in top_k_preds]) if top_k_preds else 0.0
                if max_iou >= thresh:
                    correct_count += 1
            results[f"R@{k}@IoU={thresh}"] = round((correct_count / num_queries) * 100.0, 2)

    # Compute mIoU and mean temporal start error for top-1
    for preds, gt in zip(predictions, ground_truths):
        if preds:
            iou_top1 = compute_temporal_iou(preds[0], gt)
            delta_t = abs(preds[0][0] - gt[0])
        else:
            iou_top1 = 0.0
            delta_t = 100.0
        ious_top1.append(iou_top1)
        delta_t_starts.append(delta_t)

    results["mIoU"] = round(float(np.mean(ious_top1)), 4)
    results["mean_delta_t_start_sec"] = round(float(np.mean(delta_t_starts)), 2)
    return results
