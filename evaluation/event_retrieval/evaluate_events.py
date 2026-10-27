"""Run a source-disjoint event split and report actual screening/final evidence.

An incomplete dataset is rejected. A held-out test cannot run before a validation
lock exists. Qwen failures are counted as unverified, never relabelled negatives.
"""
import argparse
import json
import time
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.gemma2.config import IndexConfig
from app.gemma2.service import EventService, WorkerClient
from app.gemma2.store import Store
from .dataset import validate
from .metrics import event_metrics, gates

def evaluate(manifest, split, output, threshold, margin, validation_lock=None, base=None):
    audit = validate(manifest)
    if not audit["ready"]:
        raise ValueError("Dataset incomplete: " + "; ".join(audit["errors"]))
    if split == "test" and validation_lock is None:
        raise ValueError("Lock validation thresholds/configuration before opening test")
    config = base or IndexConfig()
    if validation_lock:
        lock = json.loads(Path(validation_lock).read_text())
        if lock.get("selection_split") != "validation" or not lock.get("validation_passed"):
            raise ValueError("Invalid validation lock")
        config = IndexConfig(**{**lock["config"], "modalities": tuple(lock["config"]["modalities"])})
        threshold, margin = lock["threshold"], lock["margin"]
    output.mkdir(parents=True, exist_ok=True)
    service = EventService(Store(output / "store"), WorkerClient(), config)
    # The validation runner may evaluate a proposed threshold; it is not deployed as
    # validated until select_validation verifies all gates and writes the final lock.
    service.thresholds = lambda: (threshold, margin, "validation_candidate" if split != "test" else "validation_locked")
    truth, proposals, final, timings, telemetry, failures = [], [], [], [], [], []
    qwen_calls, baseline_calls, normal_hours = 0, 0, 0.
    fast_seconds = []
    unknown_qwen_calls = 0
    for video in [row for row in manifest["videos"] if row["split"] == split]:
        service.register(video["path"], video["id"], video.get("captions"))
        index_job, screen_job = "index:" + video["id"], "screen:" + video["id"]
        for key in (index_job, screen_job):
            service.store.put("jobs", key, {"id": key, "video_id": video["id"], "status": "running", "cancel_requested": False})
        started = time.perf_counter()
        service.index(index_job, video["id"])
        for query in ("คนตะโกน", "คนผลักหรือชกกัน", "คนล้มลงกับพื้น"):
            service.search(video["id"], query)
            for _ in range(5):
                query_started = time.perf_counter()
                service.search(video["id"], query)
                fast_seconds.append(time.perf_counter()-query_started)
        service.screen_video(screen_job, video["id"], ["shouting", "pushing_punching", "falling"])
        timings.append(time.perf_counter()-started)
        rows = service.store.list("events", video["id"])
        for row in rows:
            row["normal_recording"] = bool(video.get("continuous_normal"))
            proposals.append({**row, "t_start": row.get("screening_t_start", row["t_start"]), "t_end": row.get("screening_t_end", row["t_end"])})
            result = row.get("verification", {})
            if result.get("status") == "unverified":
                failures.append(row["id"])
            elif row["kind"] == "shouting" or result.get("verdict", {}).get("event_present"):
                final.append(row)
        truth.extend({**event, "video_id": video["id"]} for event in video.get("events", []))
        if video.get("continuous_normal"):
            normal_hours += video["duration"]/3600
        job = service.store.get("jobs", screen_job)
        qwen_calls += job.get("qwen_calls", 0)
        baseline_calls += job.get("qwen_baseline_calls", 0)
        unknown_qwen_calls += job.get("qwen_calls_unknown", 0)
        telemetry.append(service.client.last_telemetry.copy())
    screening_metrics = event_metrics(proposals, truth, normal_hours)
    final_metrics = event_metrics(final, truth, normal_hours)
    allocated = [row["peak_allocated_bytes"] for row in telemetry if row.get("peak_allocated_bytes") is not None]
    performance = {"processing_seconds": sum(timings),
                   "allocated_gib": max(allocated)/1024**3 if allocated else None,
                   "qwen_call_reduction": 1-qwen_calls/baseline_calls if baseline_calls and not unknown_qwen_calls else None,
                   "fast_p95_seconds": float(np.percentile(fast_seconds, 95)) if fast_seconds else None}
    report = {"split": split, "config": config.json(), "threshold": threshold, "margin": margin,
              "dataset_audit": audit, "screening": screening_metrics, "final": final_metrics,
              "performance": performance, "unverified_events": failures, "telemetry": telemetry,
              "qwen_calls": qwen_calls, "qwen_baseline_calls_estimated": baseline_calls,
              "unknown_qwen_call_events": unknown_qwen_calls,
              "qwen_baseline_timing": "Not measured; use qwen_comparison for a matched verification comparison",
              "gates": gates(screening_metrics, final_metrics, performance)}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--split", choices=("development", "validation", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=.35)
    parser.add_argument("--margin", type=float, default=.02)
    parser.add_argument("--validation-lock", type=Path)
    parser.add_argument("--video-tokens", type=int, default=140)
    parser.add_argument("--video-fps", type=float, default=2)
    parser.add_argument("--max-frames", type=int, default=12)
    parser.add_argument("--dimension", type=int, default=768)
    args = parser.parse_args()
    base = IndexConfig(video_tokens=args.video_tokens, video_fps=args.video_fps, max_frames=args.max_frames, dimension=args.dimension)
    evaluate(json.loads(args.manifest.read_text(encoding="utf-8")), args.split, args.output, args.threshold, args.margin, args.validation_lock, base)

if __name__ == "__main__":
    main()
