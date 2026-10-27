"""Frozen test inference: no calibration, isolated copy of development references."""
import argparse
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
from app.gemma2.store import Store
from .open_anomaly_eval import audit, bootstrap, metrics
from .run_open_anomaly import prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--reference-store", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output)
    if root.exists():
        raise ValueError("Test output must be new; preserve earlier outcomes")
    lock = json.loads(Path(args.lock).read_text(encoding="utf-8"))
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if (lock["policy"] != n.POLICY or lock["prompt_hash"] != n.digest(n.PROMPTS) or lock["split"] != "validation"
            or manifest.get("development_lock_hash") != n.digest(lock)):
        raise ValueError("Test manifest/validation lock mismatch")
    rows = manifest["videos"]
    if not rows or any(r["split"] != "test" or r["official_split"] != "test" for r in rows):
        raise ValueError("Only official held-out test recordings accepted")
    audit(rows, allow_test=True)
    if not lock.get("source_ledger"):
        raise ValueError("Lock has no development/validation source ledger; create a new complete validation lock")
    audit(lock["source_ledger"] + rows, allow_test=True, check_files=False)
    jobs = requests.get("http://127.0.0.1:8002/api/v1/events/jobs", timeout=10).json()
    if any(j["status"] in ("running", "queued", "cancelling") for j in jobs):
        raise ValueError("Demo jobs active")
    root.mkdir(parents=True)
    store = Store(root / "store")
    source = sqlite3.connect(f"file:{Path(args.reference_store).resolve().as_posix()}?mode=ro", uri=True)
    try:
        with sqlite3.connect(store.path) as destination:
            source.backup(destination)
    finally:
        source.close()
    config = IndexConfig(**{**lock["config"], "modalities": tuple(lock["config"]["modalities"])})
    service = EventService(store, WorkerClient(), config)
    profiles = {p["name"]: p for p in n.profiles(store)}
    reference_hashes = set()
    for scene, version in lock["reference_versions"].items():
        p = profiles[scene]
        if p["current_version"] != version:
            raise ValueError("Reference bank changed after lock")
        reference_hashes.update(r["fingerprint"] for r in n.bank(store, p["id"], version)["references"])
    if reference_hashes & {r["sha256"] for r in rows}:
        raise ValueError("Test source exists in reference bank")
    report = {"validation_lock_hash": n.digest(lock), "test_manifest_hash": n.digest(manifest),
              "predictions": {a: {} for a in ("normal", "text", "combined", "combined_qwen")},
              "jobs": [], "status": "running", "production_activation": False}
    (root / "protocol.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    started = time.perf_counter()
    for row in rows:
        if row["scene_id"] not in profiles:
            raise ValueError("No locked reference profile for test scene")
        service.register(row["path"], row["id"])
        job_id = "test-index-" + row["id"]
        store.put("jobs", job_id, {"id": job_id, "video_id": row["id"], "status": "running", "cancel_requested": False})
        service.index(job_id, row["id"])
        p = profiles[row["scene_id"]]; bank = n.bank(store, p["id"], p["current_version"])
        ns = service.namespace(store.get("videos", row["id"]))
        timeline, channels = store.require(row["id"], ns), store.embeddings(ns, config.dimension)
        policy = snapshot_policy(service, p["id"])
        prompts = tuple(np.stack([service.query_vector(t + " Scene context: " + policy["context"])[0]
                                 for t in n.PROMPTS[role]]) for role in ("suspicious", "normal"))
        for arm in ("normal", "text", "combined"):
            scores = n.score_channels(channels, bank, prompts, lock["thresholds"][arm], config.dimension)
            report["predictions"][arm][row["id"]] = prediction(n.segments(timeline, scores))
        policy["thresholds"] = lock["thresholds"]["combined"]
        policy["threshold_status"] = "frozen_test_experimental"
        job_id = "test-verify-" + row["id"]
        store.put("jobs", job_id, {"id": job_id, "video_id": row["id"], "status": "running", "open_policy": policy, "cancel_requested": False})
        screen_open(service, job_id)
        report["predictions"]["combined_qwen"][row["id"]] = [{"t_start": e["t_start"], "t_end": e["t_end"], "assessment": e["assessment"]}
            for e in store.list("events", row["id"]) if e.get("screening_job_id") == job_id]
        report["jobs"].append(store.get("jobs", job_id))
        (root / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    report.update(status="complete", elapsed_seconds=time.perf_counter()-started,
                  metrics={a: {"metrics": metrics(rows, pred), "95ci": bootstrap(rows, pred)} for a, pred in report["predictions"].items()})
    (root / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    service.executor.shutdown(wait=True)
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
