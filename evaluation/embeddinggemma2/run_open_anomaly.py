"""Four-arm frozen-encoder development/validation run, isolated from demo data.

No test rows accepted here. Threshold search uses validation only, then freezes
all four arms before Qwen. All recordings and code hashes accompany raw results.
"""
import argparse
import copy
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import requests

from app.gemma2 import normality as n
from app.gemma2.config import IndexConfig
from app.gemma2.open_screening import screen_open, snapshot_policy
from app.gemma2.service import EventService, WorkerClient
from app.gemma2.store import Store, file_hash
from .open_anomaly_eval import audit, bootstrap, freeze, metrics, stage_diagnostics


def prediction(units):
    return [{"t_start": u["t_start"], "t_end": u["t_end"], "assessment": "suspicious"} for u in units]


def arm_thresholds(novelty, margin, arm):
    return {m: {"novelty": 3. if arm == "text" else novelty,
                "margin": 3. if arm == "normal" else margin, "similarity": .35} for m in n.DEFAULT_THRESHOLDS}


def reuse_indexes(store, source_path, video_ids):
    """Copy only media/index data; predictions, profiles, labels and reviews stay isolated."""
    source = sqlite3.connect(f"file:{Path(source_path).resolve().as_posix()}?mode=ro", uri=True)
    try:
        ids = sorted(set(video_ids))
        if not ids:
            return
        marks = ",".join("?" for _ in ids)
        namespaces = [r[0] for r in source.execute(f"SELECT namespace FROM indexes WHERE video_id IN ({marks})", ids)]
        with store.db() as destination:
            for table, key, values in (("videos", "id", ids), ("indexes", "video_id", ids),
                                       ("active", "video_id", ids), ("windows", "namespace", namespaces),
                                       ("embeddings", "namespace", namespaces)):
                if not values:
                    continue
                placeholders = ",".join("?" for _ in values)
                for row in source.execute(f"SELECT * FROM {table} WHERE {key} IN ({placeholders})", values):
                    destination.execute(f"INSERT OR IGNORE INTO {table} VALUES ({','.join('?' for _ in row)})", tuple(row))
    finally:
        source.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reuse-index-store", help="Read-only copy of compatible index cache; never changes prior run")
    args = parser.parse_args()
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    result_path = root / "results.json"
    if (root / "protocol.json").exists():
        raise ValueError("Use a fresh experiment output; existing locks/results are immutable")
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8")); rows = manifest["videos"]
    audit(rows)
    if any(j["status"] in ("running", "queued", "cancelling") for j in requests.get("http://127.0.0.1:8002/api/v1/events/jobs", timeout=10).json()):
        raise ValueError("Demo jobs active")
    config = IndexConfig()
    protocol = {"manifest_hash": n.digest(manifest), "config": config.json(), "prompts": n.PROMPTS,
                "code_hashes": {str(p): file_hash(p) for p in [Path(__file__), Path("backend/app/gemma2/normality.py"),
                    Path("backend/app/gemma2/open_screening.py"), Path("evaluation/embeddinggemma2/worker.py"),
                    Path("backend/inference_worker/vlm_service.py"), Path("backend/app/gemma2/service.py"),
                    Path("backend/app/inference/contracts.py"), Path("evaluation/embeddinggemma2/open_anomaly_eval.py")]},
                "grid": {"novelty": [.05, .1, .2, .3, .5], "margin": [0., .03, .1]},
                "selection": "screening recall >= .95 first, then recall, fewer false positives, less candidate duration, stable thresholds",
                "test_opened": False, "index_cache_reused": args.reuse_index_store,
                "subset": manifest.get("subset", False), "started_at": time.time()}
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    store = Store(root / "store")
    if args.reuse_index_store:
        reuse_indexes(store, args.reuse_index_store, [r["id"] for r in rows])
    service = EventService(store, WorkerClient(), config)
    profiles, prepared, encode_telemetry = {}, {}, []
    started = time.perf_counter()
    for row in rows:
        service.register(row["path"], row["id"], name=row["id"])
        job_id = "index-" + row["id"]
        service.store.put("jobs", job_id, {"id": job_id, "video_id": row["id"], "status": "running", "cancel_requested": False})
        service.index(job_id, row["id"])
        encode_telemetry.append(service.client.last_telemetry)
        print("INDEXED", row["id"], flush=True)
    for scene in sorted({r["scene_id"] for r in rows}):
        profile = n.create_profile(service.store, scene, "A fixed surveillance camera observing a pedestrian area. Routine walking and waiting are normal; evaluate activity in context.")
        refs = [r for r in rows if r["scene_id"] == scene and r["split"] == "development" and not r["events"]]
        if not refs:
            raise ValueError("No independent normal development recordings for scene " + scene)
        n.add_references(service, profile["id"], [{"video_id": r["id"], "t_start": 0., "t_end": r["duration"],
            "split": "development", "verified_normal": True} for r in refs])
        profiles[scene] = profile["id"]
    validation = [r for r in rows if r["split"] == "validation"]
    for row in validation:
        video = service.store.get("videos", row["id"]); ns = service.namespace(video)
        p = profiles[row["scene_id"]]; policy = snapshot_policy(service, p)
        prompts = tuple(np.stack([service.query_vector(t + " Scene context: " + policy["context"])[0]
                                 for t in n.PROMPTS[role]]) for role in ("suspicious", "normal"))
        prepared[row["id"]] = (service.store.require(row["id"], ns), service.store.embeddings(ns, config.dimension), n.bank(service.store, p), prompts)
    selected, predictions, arm_metrics = {}, {}, {}
    for arm in ("normal", "text", "combined"):
        options = []
        for novelty in protocol["grid"]["novelty"]:
            for margin in protocol["grid"]["margin"]:
                thresholds = arm_thresholds(novelty, margin, arm)
                pred = {}
                for row in validation:
                    timeline, channels, bank, prompts = prepared[row["id"]]
                    scores = n.score_channels(channels, bank, prompts, thresholds, config.dimension)
                    pred[row["id"]] = prediction(n.segments(timeline, scores))
                measured = metrics(validation, pred)
                recall = measured["recall"] or 0.
                key = (recall >= .95, recall, -measured["fp"], -(measured["positive_video_fraction"] or 0), novelty, margin)
                options.append((key, thresholds, pred, measured))
        _, selected[arm], predictions[arm], arm_metrics[arm] = max(options, key=lambda x: x[0])
    lock = freeze(manifest, selected, config.json(), [r for r in rows if r["split"] == "development" and not r["events"]], arm_metrics)
    lock["reference_versions"] = {scene: n.profile(service.store, p)["current_version"] for scene, p in profiles.items()}
    (root / "validation-lock.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
    predictions["combined_qwen"] = {}
    result = {"protocol": protocol, "lock": lock, "predictions": predictions, "metrics": arm_metrics,
              "encode_telemetry": encode_telemetry, "verification_jobs": [], "status": "verifying"}
    def save():
        result["elapsed_seconds"] = time.perf_counter()-started
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    save()
    for row in validation:
        p = profiles[row["scene_id"]]; policy = snapshot_policy(service, p)
        policy["thresholds"] = selected["combined"]
        policy["threshold_status"] = "validation_selected_subset_experimental"
        job_id = "verify-" + row["id"]
        service.store.put("jobs", job_id, {"id": job_id, "video_id": row["id"], "open_policy": policy,
            "status": "running", "mode": "open_anomaly", "cancel_requested": False})
        screen_open(service, job_id)
        events = [e for e in service.store.list("events", row["id"]) if e["screening_job_id"] == job_id]
        predictions["combined_qwen"][row["id"]] = [{"t_start": e["t_start"], "t_end": e["t_end"], "assessment": e["assessment"]} for e in events]
        result["verification_jobs"].append(service.store.get("jobs", job_id))
        save(); print("VERIFIED", row["id"], flush=True)
    result["metrics"]["combined_qwen"] = metrics(validation, predictions["combined_qwen"])
    result["stage_diagnostics"] = stage_diagnostics(validation, predictions["combined"], predictions["combined_qwen"])
    result["confidence_intervals"] = {arm: bootstrap(validation, pred) for arm, pred in predictions.items()}
    result["status"] = "complete"; result["production_activation"] = False
    result["code_unchanged"] = all(file_hash(Path(p)) == sha for p, sha in protocol["code_hashes"].items())
    save()
    service.executor.shutdown(wait=True)
    print(json.dumps(result["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
