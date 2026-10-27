"""Frozen motion retry on validation, with an actually measured matched control."""
import hashlib
import json
import time
from pathlib import Path

import requests

from app.gemma2.service import WorkerClient
from .compare_motion_zoom import run
from .validate_dense_verification import metrics


def main():
    root = Path("scratch/event-retrieval")
    development_path = root/"motion-zoom-dev/results.json"
    development = json.loads(development_path.read_text(encoding="utf-8"))
    if not development["complete"] or not development["promote_to_validation"] or development["sequences"] != [1, 2]:
        raise ValueError("Development promotion gate failed")
    output = root/"motion-zoom-validation"; output.mkdir(parents=True, exist_ok=True)
    if any((output/name).exists() for name in ("protocol-lock.json", "control.json", "results.json")):
        raise ValueError("Preserve validation protocol and measured run")
    jobs = requests.get("http://127.0.0.1:8002/api/v1/events/jobs", timeout=10).json()
    if any(j["status"] in ("running", "queued", "cancelling") for j in jobs):
        raise RuntimeError("Active screening jobs")
    frozen = json.loads((root/"dense-validation/results.json").read_text(encoding="utf-8"))
    if not frozen["complete"] or frozen["sequences"] != [13, 14, 15, 16]:
        raise ValueError("Needs immutable candidate control")
    lock = dict(created_ns=time.time_ns(), sequences=[13, 14, 15, 16],
        development_sha256=hashlib.sha256(development_path.read_bytes()).hexdigest(),
        algorithm_sha256=hashlib.sha256(Path(__file__).with_name("motion_zoom.py").read_bytes()).hexdigest(),
        gemma_threshold=frozen["threshold"], candidate_ids=list(frozen["attempts"]),
        mode="temporal-v1", motion_seconds=2., frames=8, sampled_fps_target=8., quiet_peak_cutoff=1.,
        rule="strictly higher coverage; no increase in non-target positive views; no automatic production acceptance",
        scope="Repeated validation regression, not a fresh blind test")
    (output/"protocol-lock.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
    source = json.loads((root/"continuous-ut/results.json").read_text(encoding="utf-8"))
    payload = [dict(rows[0]["payload"], verification_mode="temporal-v1") for rows in frozen["attempts"].values()]
    locations = {identifier: rows[0]["bounds"] for identifier, rows in frozen["attempts"].items()}
    attempts, telemetry = {}, []
    measured = dict(attempts=attempts, telemetry=telemetry, complete=False)
    control = dict(split="validation", sequences=lock["sequences"], payload=payload,
        locations=locations, runs={"temporal-v1": measured}, complete=False)
    def save():
        (output/"control.json").write_text(json.dumps(control, ensure_ascii=False, indent=2), encoding="utf-8")
    save(); client = WorkerClient(); started = time.perf_counter()
    for offset in range(0, len(payload), 8):
        batch = payload[offset:offset+8]
        response = client.post("/verify-batch", {"events": batch}, timeout=8*480+30)
        replies = {r["id"]: r for r in response["results"]}
        for event in batch:
            identifier = event["id"]
            reply = replies.get(identifier, dict(status="unverified", verdict=None, calls=0))
            attempts[identifier] = [dict(bounds=locations[identifier], reply=reply)]
        telemetry.append(response.get("telemetry", {})); save()
        print("VALIDATION CONTROL", min(offset+8, len(payload)), "/", len(payload), flush=True)
    measured.update(complete=True, elapsed_seconds=time.perf_counter()-started,
        metrics=metrics(source, attempts, tuple(lock["sequences"]), False),
        private_requests=sum(rows[0]["reply"].get("calls", 0) for rows in attempts.values()))
    control["complete"] = True; save()
    run(control, source, output)


if __name__ == "__main__":
    main()
