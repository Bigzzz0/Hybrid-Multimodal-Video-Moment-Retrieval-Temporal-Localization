import json
import time
import sys
import os
import statistics
import platform
import subprocess
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure backend and workspace are on path
workspace_dir = Path(__file__).resolve().parent.parent
backend_dir = workspace_dir / "backend"
sys.path.insert(0, str(workspace_dir))
sys.path.insert(0, str(backend_dir))

from app.retrieval.search_engine import search_engine
from app.core.config import settings
from evaluation.compute_metrics import compute_temporal_iou, evaluate_moment_retrieval
from evaluation.validate_dataset import validate_dataset

try:
    import torch
except Exception:  # pragma: no cover - backend already requires torch in production
    torch = None

def run_real_video_benchmark(
    dataset_json_path: str = "evaluation/datasets/real_video_benchmark.json",
    output_results_path: str = "evaluation/benchmark_real_results.json",
    profile: str = "fast",
    acceptance: bool = False,
):
    """
    Executes automated benchmark evaluation on real-world video dataset downloaded from internet.
    Evaluates pure visual temporal moment retrieval accuracy against ground-truth intervals.
    """
    print("=" * 80)
    print(f"    PURE VISUAL VIDEO MOMENT RETRIEVAL - {profile.upper()} BENCHMARK")
    print("=" * 80)

    dataset_path = Path(dataset_json_path)
    if not dataset_path.exists():
        print(f"Dataset file not found: {dataset_json_path}")
        return

    if acceptance:
        contract = validate_dataset(dataset_path)
        print(f"Acceptance dataset validated: {contract['query_count']} queries / {contract['video_count']} videos")

    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_predictions: List[List[Tuple[float, float]]] = []
    all_ground_truths: List[Any] = []
    query_details: List[Dict[str, Any]] = []
    latencies: List[float] = []

    total_videos = len(data.get("videos", []))
    total_query_count = sum(len(e.get("queries", [])) for v in data.get("videos", []) for e in v.get("events", []))
    print(f"Loaded {total_videos} real benchmark videos | {total_query_count} total evaluation queries.")
    print("-" * 80)

    for v_idx, v_data in enumerate(data.get("videos", [])):
        video_id = v_data["video_id"]
        v_title = v_data.get("filename", video_id)
        v_duration = v_data.get("duration_sec", 0.0)
        print(f"\n[Video {v_idx+1}/{total_videos}] {v_title} (ID: {video_id}, {v_duration:.1f}s)")

        for event in v_data.get("events", []):
            event_id = event["event_id"]
            gt_values = event.get("ground_truths")
            if gt_values is not None:
                gt_interval = [(float(pair[0]), float(pair[1])) for pair in gt_values]
            else:
                gt_interval = (float(event["t_start"]), float(event["t_end"]))
            event_desc = event.get("description", "")

            for q_text in event.get("queries", []):
                t_start = time.time()
                resp = search_engine.search_moments(
                    query=q_text,
                    video_id=video_id,
                    top_k=5,
                    profile=profile,
                )
                if acceptance and not resp.calibrated:
                    raise RuntimeError(
                        "acceptance benchmark requires a matching calibration artifact; "
                        "runtime returned calibrated=false"
                    )
                latency_ms = (time.time() - t_start) * 1000.0
                latencies.append(latency_ms)

                preds: List[Tuple[float, float]] = [
                    (float(m.t_start), float(m.t_end)) for m in resp.moments
                ]
                all_predictions.append(preds)
                all_ground_truths.append(gt_interval)

                # Compute IoU for Top-1
                if preds:
                    gt_list = gt_interval if isinstance(gt_interval, list) else [gt_interval]
                    top1_iou = max((compute_temporal_iou(preds[0], target) for target in gt_list), default=0.0)
                    delta_t = min((abs(preds[0][0] - target[0]) for target in gt_list), default=999.0)
                    top1_str = f"[{preds[0][0]:.1f}s - {preds[0][1]:.1f}s]"
                    score_val = resp.moments[0].score
                else:
                    top1_iou = 0.0
                    delta_t = 999.0
                    top1_str = "None"
                    score_val = 0.0

                query_info = {
                    "video_id": video_id,
                    "event_id": event_id,
                    "query": q_text,
                    "gt_interval": [list(pair) for pair in gt_interval] if isinstance(gt_interval, list) else list(gt_interval),
                    "is_no_match": isinstance(gt_interval, list) and len(gt_interval) == 0,
                    "pred_top1": list(preds[0]) if preds else None,
                    "top1_iou": round(top1_iou, 4),
                    "delta_t_start": round(delta_t, 2),
                    "confidence_score": round(score_val, 3),
                    "latency_ms": round(latency_ms, 2),
                    "all_predictions": [list(p) for p in preds]
                }
                query_details.append(query_info)

                status_mark = "[PASS]" if top1_iou >= 0.3 else "[MISS]"
                print(f"  {status_mark} Query: '{q_text}'")
                print(f"         GT: {gt_interval} | Pred: {top1_str} | IoU: {top1_iou:.3f} | Latency: {latency_ms:.1f}ms")

    # Compute overall benchmark metrics
    metrics = evaluate_moment_retrieval(
        predictions=all_predictions,
        ground_truths=all_ground_truths,
        iou_thresholds=[0.3, 0.5, 0.7],
        top_ks=[1, 5]
    )

    avg_latency = float(sum(latencies) / max(1, len(latencies)))
    metrics["avg_query_latency_ms"] = round(avg_latency, 2)
    metrics["total_evaluated_queries"] = len(all_ground_truths)
    if latencies:
        metrics["p50_query_latency_ms"] = round(float(statistics.median(latencies)), 2)
        metrics["p95_query_latency_ms"] = round(float(np.percentile(latencies, 95)), 2)
    else:
        metrics["p50_query_latency_ms"] = 0.0
        metrics["p95_query_latency_ms"] = 0.0
    peak_vram_mb = 0.0
    if torch is not None and torch.cuda.is_available():
        try:
            peak_vram_mb = max(
                float(torch.cuda.max_memory_allocated()) / (1024 ** 2),
                float(torch.cuda.max_memory_reserved()) / (1024 ** 2),
            )
        except Exception:
            pass
    metrics["peak_vram_mb"] = round(peak_vram_mb, 1)

    # Display Academic Evaluation Summary Table
    print("\n" + "=" * 80)
    print("           REAL VIDEO MOMENT RETRIEVAL BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"  Total Real Videos Tested    : {total_videos}")
    print(f"  Total Queries Evaluated     : {len(all_ground_truths)} (Thai & English)")
    print(f"  Average Query Latency       : {metrics['avg_query_latency_ms']:.2f} ms")
    print("-" * 80)
    print(f"  Metric                      | Score")
    print("-" * 80)
    print(f"  R@1 @ IoU=0.3               | {metrics.get('R@1@IoU=0.3', 0.0):.2f}%")
    print(f"  R@1 @ IoU=0.5               | {metrics.get('R@1@IoU=0.5', 0.0):.2f}%")
    print(f"  R@1 @ IoU=0.7               | {metrics.get('R@1@IoU=0.7', 0.0):.2f}%")
    print(f"  R@5 @ IoU=0.3               | {metrics.get('R@5@IoU=0.3', 0.0):.2f}%")
    print(f"  R@5 @ IoU=0.5               | {metrics.get('R@5@IoU=0.5', 0.0):.2f}%")
    print(f"  R@5 @ IoU=0.7               | {metrics.get('R@5@IoU=0.7', 0.0):.2f}%")
    print(f"  Mean IoU (mIoU)             | {metrics.get('mIoU', 0.0):.4f}")
    print(f"  Mean Delta t_start Error    | {metrics.get('mean_delta_t_start_sec', 0.0):.2f} sec")
    print(f"  tF1 @ IoU=0.5              | {metrics.get('tF1@IoU=0.5', 0.0):.2f}%")
    print(f"  Count Accuracy             | {metrics.get('count_accuracy', 0.0):.2f}%")
    print(f"  No-match Precision/Recall  | {metrics.get('no_match_precision', 0.0):.4f} / {metrics.get('no_match_recall', 0.0):.4f}")
    if latencies:
        print(f"  p50 / p95 Query Latency    | {statistics.median(latencies):.2f} / {np.percentile(latencies, 95):.2f} ms")
    print("=" * 80)

    # Save output json
    out_path = Path(output_results_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        commit_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=workspace_dir,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except Exception:
        commit_sha = "unknown"
    report_data = {
        "benchmark_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "profile": profile,
        "run_metadata": {
            "commit_sha": commit_sha,
            "hardware": platform.platform(),
            "device": settings.DEVICE,
            "embedding_model": settings.SIGLIP2_MODEL_ID,
            "caption_model": settings.QWEN_VL_MODEL_ID,
            "index_version": settings.VISUAL_INDEX_VERSION,
            "sampling_hz": 2,
        },
        "metrics": metrics,
        "queries": query_details
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] Benchmark report written to: {output_results_path}\n")
    return metrics

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the pure-visual moment retrieval benchmark.")
    parser.add_argument("--dataset", default="evaluation/datasets/real_video_benchmark.json")
    parser.add_argument("--output", default="evaluation/benchmark_real_results.json")
    parser.add_argument("--profile", choices=("fast", "accurate"), default="fast")
    parser.add_argument("--acceptance", action="store_true", help="Fail unless held-out dataset contract is satisfied")
    args = parser.parse_args()
    run_real_video_benchmark(args.dataset, args.output, args.profile, args.acceptance)
