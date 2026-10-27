"""Matched verification units for all-windows versus screened-windows.

Both arms use identical prompts and eight equally spaced frames for each six-
second window. Persisted observations are reused; no synthetic Qwen results.
Run when the demo's index/screen queue is idle, through the same GPU broker.
"""
import argparse
import json
import time
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.gemma2.config import DATA
from app.gemma2.service import EventService
from app.gemma2.store import Store
from app.gemma2.ranking import EVENTS, screen

def compare(service, video_id, output, limit=None):
    video = service.store.get("videos", video_id)
    namespace = service.namespace(video)
    rows = service.store.require(video_id, namespace)
    if limit:
        rows = rows[:limit]
    threshold, margin, status = service.thresholds()
    kinds = list(EVENTS)
    with service.gpu_lock:
        candidates = screen(service.store.embeddings(namespace, service.config.dimension), service.prompt_vectors(kinds), kinds, threshold, margin)
        selected = {(row["kind"], row["window_id"]) for row in candidates}
        units = []
        for row in rows:
            slots = np.linspace(0, len(row["frame_paths"])-1, min(8, len(row["frame_paths"]))).astype(int)
            for kind in kinds:
                # No sound cannot be a shouting candidate or acoustic baseline unit.
                if kind == "shouting" and row["audio_status"] != "available":
                    continue
                units.append({"id": f"{namespace}:{kind}:{row['id']}", "kind": kind, "label": EVENTS[kind]["label"],
                              "frame_paths": [row["frame_paths"][slot] for slot in slots], "timestamps": [row["timestamps"][slot] for slot in slots],
                              "window_id": row["id"], "selected": (kind, row["id"]) in selected})
        observations = []
        for arm, arm_units in (("all", units), ("screened", [unit for unit in units if unit["selected"]])):
            for offset in range(0, len(arm_units), 8):
                batch = arm_units[offset:offset+8]
                started = time.perf_counter()
                result = service.client.post("/verify-batch", {"events": batch}, timeout=8*480+30)
                observations.append({"arm": arm, "units": batch, "results": result["results"], "batch_seconds": time.perf_counter()-started,
                                     "telemetry": result["telemetry"]})
    all_calls = sum(1 + (unit["kind"] != "shouting") for unit in units)
    candidate_calls = sum(1 + (unit["kind"] != "shouting") for unit in units if unit["selected"])
    report = {"video_id": video_id, "config": service.config.json(), "threshold_status": status,
              "windows": len(rows), "all_calls": all_calls, "candidate_calls": candidate_calls,
              "call_reduction": 1-candidate_calls/all_calls if all_calls else None,
              "observations": observations, "arm_seconds": {arm: sum(row["batch_seconds"] for row in observations if row["arm"] == arm) for arm in ("all", "screened")},
              "scope": "Matched units; accuracy still requires independent human event annotations"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_id")
    parser.add_argument("--store", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    compare(EventService(Store(args.store)), args.video_id, args.output, args.limit)
