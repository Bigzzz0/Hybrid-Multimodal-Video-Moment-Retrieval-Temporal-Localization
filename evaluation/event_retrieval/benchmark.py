"""Ablation runner. Outputs stay under scratch; missing modality results stay missing.

Each configuration decodes identical windows. Token budgets overflowing 8K are
rejected, rather than silently sampling fewer frames and claiming a fair comparison.
"""
import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.gemma2.config import IndexConfig
from app.gemma2.service import EventService, WorkerClient
from app.gemma2.store import Store
from .metrics import retrieval

def configurations(base=None):
    base = base or IndexConfig()
    yield "siglip2_image", replace(base, dimension=768, model_id="google/siglip2-base-patch16-naflex", revision="b53b807d3a2d5e2b3911292f2d69e5341cdc064c", modalities=("image",))
    for name, kinds in (("image", ("image",)), ("video", ("video",)), ("audio", ("audio",)), ("caption", ("caption",)),
                        ("video_audio", ("video", "audio")), ("all", ("image", "video", "audio", "caption"))):
        yield "gemma2_" + name, replace(base, modalities=kinds)

def run(manifest, output, names=None, repeats=5, base=None):
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for name, config in configurations(base):
        if names and name not in names:
            continue
        service = EventService(Store(output / "indexes"), WorkerClient(), config)
        records, timings, index_seconds, joint_seconds, telemetry = [], [], [], [], []
        index_stages, unavailable = [], []
        for video in manifest["videos"]:
            registered = service.register(video["path"], video["id"], video.get("captions"))
            if config.modalities == ("caption",) and not registered["captions"]:
                unavailable.append({"video_id": video["id"], "modality": "caption"})
                continue
            if config.modalities == ("audio",) and not registered["has_audio"]:
                unavailable.append({"video_id": video["id"], "modality": "audio"})
                continue
            job_id = f"{name}:{video['id']}"
            service.store.put("jobs", job_id, {"id": job_id, "video_id": video["id"], "cancel_requested": False})
            started = time.perf_counter()
            with service.gpu_lock:
                service.index(job_id, video["id"])
            index_seconds.append(time.perf_counter()-started)
            index_stages.append({"video_id": video["id"], **service.store.get("jobs", job_id)})
            for query in video.get("queries", []):
                result = service.search(video["id"], query["text"])
                records.append({"language": query["language"], "truth": query.get("truth", []), "moments": result["moments"], "query": query["text"],
                                "first_query_latency_ms": result["latency_ms"], "query_cache_hit": result["cache_hits"]["query"]})
                for repeat in range(repeats):
                    timings.append(service.search(video["id"], query["text"])["latency_ms"])
                if name == "gemma2_video_audio" and registered["has_audio"]:
                    started = time.perf_counter()
                    enhanced = service.search(video["id"], query["text"], profile="accurate")
                    joint_seconds.append({"query": query["text"], "seconds": time.perf_counter()-started,
                                          "status": enhanced["stage_status"], "moments": enhanced["moments"]})
                telemetry.append(service.client.last_telemetry.copy())
        report = {"name": name, "config": config.json(), "protocol": "temporal_iou_0.3_overlapping_6s_stride_3s",
                  "metrics": retrieval(records), "records": records, "index_seconds": index_seconds, "index_stages": index_stages,
                  "joint_runs": joint_seconds, "unavailable_modalities": unavailable,
                  "warm_query_p50_ms": float(np.percentile(timings, 50)) if timings else None,
                  "warm_query_p95_ms": float(np.percentile(timings, 95)) if timings else None,
                  "first_query_p95_ms": float(np.percentile([row["first_query_latency_ms"] for row in records], 95)) if records else None,
                  "telemetry": telemetry, "accuracy_scope": "Only supplied human retrieval annotations; no event screening claims"}
        (output / f"{name}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        reports.append(report)
        service.client.post("/unload", {})
    return reports

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "scratch/event-retrieval/benchmark")
    parser.add_argument("--configurations", nargs="*")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--video-tokens", type=int, default=140)
    parser.add_argument("--dimension", type=int, default=768)
    parser.add_argument("--video-fps", type=float, default=2)
    parser.add_argument("--max-frames", type=int, default=12)
    args = parser.parse_args()
    base = IndexConfig(dimension=args.dimension, video_tokens=args.video_tokens, video_fps=args.video_fps, max_frames=args.max_frames)
    run(json.loads(args.manifest.read_text(encoding="utf-8")), args.output, args.configurations, args.repeats, base)

if __name__ == "__main__":
    main()
