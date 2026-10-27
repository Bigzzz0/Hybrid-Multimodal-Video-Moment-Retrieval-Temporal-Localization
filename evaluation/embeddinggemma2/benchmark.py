"""Controlled visual retrieval comparison; no inference without explicit --run.

Both encoders use identical sampled frames, raw bilingual queries, rank fusion,
smoothing and temporal proposal code. This isolates visual embedding quality;
it is not the full production caption/VLM/grounding pipeline.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .prepare import DEFAULT_CONFIG, ROOT, check_prerequisites, file_hash, load_config


def sync_gpu(device: str) -> None:
    if device.startswith("cuda"):
        import torch
        torch.cuda.synchronize()


def sample_frames(path: Path, video_id: str, sample_hz: float, output: Path, on_progress=None) -> dict:
    """Sample without consulting event boundaries or descriptions."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or fps <= 0 or count <= 0:
            raise ValueError(f"Unreadable video: {path}")
        step = max(1, int(round(fps / sample_hz)))
        folder = output / "frames" / video_id
        folder.mkdir(parents=True, exist_ok=True)
        frames = []
        for index in range(0, count, step):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"Failed to decode {video_id}, frame {index}")
            frame_path = folder / f"frame-{index:08d}.png"
            if not cv2.imwrite(str(frame_path), image):
                raise RuntimeError(f"Failed to write sample frame: {frame_path}")
            frames.append({"id": f"{video_id}:{index}", "timestamp": index / fps, "path": str(frame_path)})
            if on_progress:
                on_progress(len(frames), (count + step - 1) // step)
        return {"video_id": video_id, "duration_sec": count / fps, "frames": frames,
                "media_sha256": file_hash(path), "actual_sample_hz": fps / step}
    finally:
        capture.release()


def make_queries(dataset: dict) -> list[dict]:
    queries = []
    for video in dataset["videos"]:
        for event in video["events"]:
            ground_truth = event.get("ground_truths")
            if ground_truth is None:
                ground_truth = [[float(event["t_start"]), float(event["t_end"])]]
            for query in event["queries"]:
                queries.append({"video_id": video["video_id"], "event_id": event["event_id"],
                                "query": query, "ground_truths": ground_truth,
                                "language": "th" if any("\u0e00" <= c <= "\u0e7f" for c in query) else "en"})
    return queries


def retrieve_moments(video: dict, image_vectors: np.ndarray, query_vector: np.ndarray, config: dict) -> tuple[list, list]:
    from app.retrieval.rank_fusion import ReciprocalRankFusion
    from app.retrieval.temporal_smoother import TemporalSmoother
    from app.retrieval.boundary_extractor import TemporalBoundaryExtractor

    frames = video["frames"]
    cosine = image_vectors @ query_vector
    ranking = [{"id": f["id"], "timestamp": f["timestamp"], "score": float(s)} for f, s in zip(frames, cosine)]
    ranking.sort(key=lambda item: (-item["score"], item["timestamp"], item["id"]))
    fused = ReciprocalRankFusion(k=config["rrf_k"]).fuse({"visual": ranking}, {"visual": 1.0})
    signal = [(v["item_data"]["timestamp"], v["fused_score"]) for v in fused.values()]
    axis, raw = TemporalSmoother().smooth_timeline(
        video["duration_sec"], signal, sigma=config["smoothing_sigma_sec"],
        resolution_hz=config["timeline_hz"], use_multiscale=False,
    )
    span = float(np.max(raw) - np.min(raw))
    normalized = (raw - np.min(raw)) / span if span > 1e-8 else np.zeros_like(raw)
    proposals = TemporalBoundaryExtractor().extract_multiscale_proposals(
        axis, normalized, raw_scores=raw, scene_boundaries=[0.0, video["duration_sec"]],
    )
    intervals = [[float(p["t_start"]), float(p["t_end"])] for p in proposals[:config["top_k"]]]
    best_frames = [{"timestamp": r["timestamp"], "cosine": r["score"]} for r in ranking[:config["top_k"]]]
    return intervals, best_frames


def evaluate_rows(rows: list[dict]) -> dict:
    from evaluation.compute_metrics import evaluate_moment_retrieval

    predictions = [[tuple(p) for p in r["all_predictions"]] for r in rows]
    ground_truths = [[tuple(p) for p in r["ground_truths"]] for r in rows]
    metrics = evaluate_moment_retrieval(predictions, ground_truths, [0.3, 0.5, 0.7], [1, 5])
    latencies = [r["latency_ms"] for r in rows]
    metrics.update(total_evaluated_queries=len(rows),
                   avg_query_latency_ms=round(float(np.mean(latencies)), 2),
                   p50_query_latency_ms=round(float(np.percentile(latencies, 50)), 2),
                   p95_query_latency_ms=round(float(np.percentile(latencies, 95)), 2))
    for k in (1, 5):
        hits = [any(any(start <= f["timestamp"] <= end for start, end in r["ground_truths"])
                    for f in r["top_frames"][:k]) for r in rows]
        metrics[f"FrameHit@{k}"] = round(float(np.mean(hits)) * 100, 2)
    return metrics


def run_role(role: str, videos: dict, queries: list[dict], config: dict, output: Path, readiness: dict) -> Path:
    import torch
    from PIL import Image
    from evaluation.compute_metrics import compute_temporal_iou
    from .encoders import SigLIPEncoder, EmbeddingGemmaEncoder

    spec = config[role]
    encoder_type = SigLIPEncoder if role == "baseline" else EmbeddingGemmaEncoder
    if config["device"].startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    encoder = encoder_type(ROOT / spec["local_dir"], config["device"], config["embedding_dim"])
    sync_gpu(config["device"])
    load_ms = (time.perf_counter() - started) * 1000
    vectors_by_video = {}
    index_started = time.perf_counter()
    try:
        for video_id, video in videos.items():
            batches = []
            for offset in range(0, len(video["frames"]), config["image_batch_size"]):
                records = video["frames"][offset:offset + config["image_batch_size"]]
                images = []
                try:
                    for record in records:
                        with Image.open(record["path"]) as image:
                            images.append(image.convert("RGB"))
                    batches.append(encoder.encode_images(images))
                finally:
                    for image in images:
                        image.close()
            vectors_by_video[video_id] = np.concatenate(batches)
            # Each model gets its own artifact; vectors never enter LanceDB.
            np.savez_compressed(output / f"{role}-{video_id}-vectors.npz",
                                vectors=vectors_by_video[video_id],
                                timestamps=np.asarray([f["timestamp"] for f in video["frames"]]),
                                model_id=spec["model_id"], revision=spec["revision"],
                                media_sha256=video["media_sha256"], sample_hz=config["sample_hz"])
            print(f"{role}: indexed {video_id}, {len(video['frames'])} frames", flush=True)
        sync_gpu(config["device"])
        index_seconds = time.perf_counter() - index_started
        # Same warmup policy for both roles; excluded from measured query time.
        warmup = queries[0]
        warmup_vector = encoder.encode_queries([warmup["query"]])[0]
        retrieve_moments(videos[warmup["video_id"]], vectors_by_video[warmup["video_id"]], warmup_vector, config)
        sync_gpu(config["device"])
        rows = []
        for query in queries:
            started = time.perf_counter()
            qvec = encoder.encode_queries([query["query"]])[0]
            sync_gpu(config["device"])
            predictions, frames = retrieve_moments(videos[query["video_id"]], vectors_by_video[query["video_id"]], qvec, config)
            latency_ms = (time.perf_counter() - started) * 1000
            top_iou = max((compute_temporal_iou(tuple(predictions[0]), tuple(gt)) for gt in query["ground_truths"]), default=0.0) if predictions else 0.0
            start_error = min((abs(predictions[0][0] - gt[0]) for gt in query["ground_truths"]), default=100.0) if predictions else 100.0
            rows.append({**query, "gt_interval": query["ground_truths"], "all_predictions": predictions,
                         "pred_top1": predictions[0] if predictions else None, "top1_iou": top_iou,
                         "delta_t_start": start_error, "top_frames": frames, "latency_ms": latency_ms})
            print(f"{role}: {len(rows)}/{len(queries)} IoU={top_iou:.3f} {query['query']}", flush=True)
        metrics = evaluate_rows(rows)
        metrics["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1) if config["device"].startswith("cuda") else 0.0
        language_metrics = {language: evaluate_rows([r for r in rows if r["language"] == language])
                            for language in ("th", "en") if any(r["language"] == language for r in rows)}
        report = {"benchmark_timestamp": datetime.now(timezone.utc).isoformat(), "profile": "visual-only-smoke",
                  "scope": "Controlled frame retrieval; excludes production captions, query expansion, scene cuts, calibration, candidate grouping, SAM and VLM verification",
                  "run_metadata": {"commit_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                                   "hardware": platform.platform(), "gpu": readiness["gpu"],
                                   "versions": readiness["versions"], "model": spec, "config": config,
                                   "dataset_sha256": readiness["dataset_sha256"],
                                   "videos": [{k: v for k, v in video.items() if k != "frames"} for video in videos.values()],
                                   "model_load_ms": load_ms, "frame_index_seconds": index_seconds,
                                   "query_warmup_count": 1, "precomputed_query_vectors": False},
                  "metrics": metrics, "language_metrics": language_metrics, "queries": rows}
        report_path = output / f"{role}.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report_path
    finally:
        encoder.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run", action="store_true", help="Explicitly start model inference and benchmark")
    args = parser.parse_args()
    config = load_config(args.config)
    readiness = check_prerequisites(config)
    if not readiness["ready"]:
        raise SystemExit("Prerequisites failed: " + "; ".join(readiness["errors"]))
    if not args.run:
        print("Ready. No model loaded and no inference/benchmark executed. Add --run only when authorized.")
        return
    # No model instantiation or frame extraction is reachable before this gate.
    sys.path.insert(0, str(ROOT / "backend"))
    import torch
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    dataset = json.loads((ROOT / config["dataset"]).read_text(encoding="utf-8"))
    output = ROOT / config["output_dir"] / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    queries = make_queries(dataset)
    videos = {v["video_id"]: sample_frames(ROOT / config["media_dir"] / v["filename"], v["video_id"], config["sample_hz"], output)
              for v in dataset["videos"]}
    baseline = run_role("baseline", videos, queries, config, output, readiness)
    candidate = run_role("candidate", videos, queries, config, output, readiness)
    from evaluation.compare_benchmarks import compare_reports
    comparison = compare_reports(baseline, candidate)
    comparison["scope"] = "Smoke comparison only; not production acceptance"
    (output / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    reports = {role: json.loads((output / f"{role}.json").read_text(encoding="utf-8")) for role in ("baseline", "candidate")}
    lines = ["# EmbeddingGemma 2 visual retrieval smoke experiment", "",
             "Same frames and queries; separate model vectors. Small dataset, no production acceptance claim.", "",
             "| Metric | SigLIP2 | EmbeddingGemma 2 |", "| --- | ---: | ---: |"]
    for metric in ("FrameHit@1", "FrameHit@5", "R@1@IoU=0.5", "R@5@IoU=0.5", "mIoU", "p50_query_latency_ms", "p95_query_latency_ms", "peak_vram_mb"):
        lines.append(f"| {metric} | {reports['baseline']['metrics'][metric]} | {reports['candidate']['metrics'][metric]} |")
    lines += ["", "Language breakdown and every query's predictions are in baseline.json and candidate.json.",
              "Model-load time and frame-index time are recorded separately from warm query latency.",
              "The temporal proposal stage reuses the project's RRF, TemporalSmoother and TemporalBoundaryExtractor."]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Reports: {output}")


if __name__ == "__main__":
    main()
