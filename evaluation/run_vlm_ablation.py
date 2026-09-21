"""Run one selected VLM backend against the same SigLIP2 candidates.

The small real-video JSON bundled in this repository is a smoke/regression
set, not a held-out AIRC evaluation. The runner rejects silent backend
fallbacks so A--E results cannot be accidentally mixed.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "backend"))

from app.core.config import settings
from app.inference.vlm_registry import get_variant
from app.retrieval.search_engine import search_engine
from evaluation.compute_metrics import compute_temporal_iou, evaluate_moment_retrieval


def run(backend: str, dataset: str, output: str, profile: str = "accurate") -> dict:
    variant = get_variant(backend)
    data = json.loads(Path(dataset).read_text(encoding="utf-8"))
    predictions, truths, details, latencies = [], [], [], []
    fallback_count = 0
    for video in data.get("videos", []):
        for event in video.get("events", []):
            gt_values = event.get("ground_truths")
            truth = [(float(a), float(b)) for a, b in gt_values] if gt_values is not None else (float(event["t_start"]), float(event["t_end"]))
            for query in event.get("queries", []):
                started = time.perf_counter()
                response = search_engine.search_moments(query, video["video_id"], top_k=5, profile=profile, vlm_backend=backend)
                latency = (time.perf_counter() - started) * 1000.0
                latencies.append(latency)
                if response.vlm_backend_used != backend:
                    fallback_count += 1
                predicted = [(float(moment.t_start), float(moment.t_end)) for moment in response.moments]
                predictions.append(predicted)
                truths.append(truth)
                targets = truth if isinstance(truth, list) else [truth]
                top_iou = max((compute_temporal_iou(predicted[0], target) for target in targets), default=0.0) if predicted else 0.0
                details.append({
                    "video_id": video["video_id"], "event_id": event.get("event_id", ""), "query": query,
                    "backend_requested": backend, "backend_used": response.vlm_backend_used,
                    "top1_iou": round(top_iou, 4), "latency_ms": round(latency, 2),
                    "warnings": response.warnings, "cascade_path": response.cascade_path,
                    "pred_top1": list(predicted[0]) if predicted else None,
                })
    if fallback_count:
        raise RuntimeError(f"benchmark rejected: {fallback_count} queries silently fell back from {backend}")
    metrics = evaluate_moment_retrieval(predictions, truths, iou_thresholds=[0.3, 0.5, 0.7], top_ks=[1, 5])
    metrics.update({
        "backend": backend,
        "model_id": variant.model_id,
        "model_revision": variant.revision,
        "quantization": variant.quantization,
        "input_mode": variant.input_mode,
        "profile": profile,
        "smoke_only": True,
        "query_count": len(details),
        "avg_latency_ms": round(float(statistics.mean(latencies)) if latencies else 0.0, 2),
        "p50_latency_ms": round(float(statistics.median(latencies)) if latencies else 0.0, 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)) if latencies else 0.0, 2),
        "fallback_count": fallback_count,
        "siglip_model_id": settings.SIGLIP2_MODEL_ID,
    })
    report = {"metadata": metrics, "queries": details}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=[
        "qwen3_vl_2b", "qwen3_vl_2b_vise", "caprl_qwen3vl_2b",
        "caprl_qwen3vl_4b_q4", "caprl_qwen3vl_4b_q6", "caprl_video_4b",
    ])
    parser.add_argument("--dataset", default="evaluation/datasets/real_video_benchmark.json")
    parser.add_argument("--output", default="evaluation/vlm_ablation_result.json")
    parser.add_argument("--profile", choices=("fast", "accurate"), default="accurate")
    args = parser.parse_args()
    run(args.backend, args.dataset, args.output, args.profile)

