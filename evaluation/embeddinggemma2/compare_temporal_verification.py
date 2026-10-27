"""Paired development-only verification ablation, identical full-window frames.
All windows in seq1/2 are evaluated, including normal windows. No label-driven
sampling, candidate selection, prompt adjustment or test access.
"""
import json
import time
from pathlib import Path

import numpy as np
import requests

from app.gemma2.service import WorkerClient, verification_frames
from .validate_dense_verification import metrics


def main():
    root = Path("scratch/event-retrieval/continuous-ut")
    output = Path("scratch/event-retrieval/temporal-dev"); output.mkdir(parents=True, exist_ok=True)
    target = output/"results.json"
    if target.exists():
        raise ValueError("Preserve measured run")
    jobs = requests.get("http://127.0.0.1:8002/api/v1/events/jobs", timeout=10).json()
    if any(j["status"] in ("running", "queued", "cancelling") for j in jobs):
        raise RuntimeError("Active screening jobs")
    source = json.loads((root/"results.json").read_text(encoding="utf-8"))
    payload, locations = [], {}
    for seq in (1, 2):
        sidecar = next(p for p in (root/"features").glob("*.json") if json.loads(p.read_text())["seq"] == seq)
        with np.load(sidecar.with_suffix(".npz"), allow_pickle=False) as data:
            bounds = data["bounds"].copy()
        for wi, (start, end) in enumerate(bounds):
            trace = source["provenance"][str(seq)]["traces"][wi]
            paths = [str((root/"frames"/sidecar.stem/str(wi)/f"frame-{i:02}.jpg").resolve()) for i in range(trace["frame_count"])]
            frames = verification_frames([wi], {wi: dict(timestamps=trace["timestamps"], frame_paths=paths)}, start, end)
            identifier = f"aligned-ut-{seq}-{wi}"
            payload.append(dict(id=identifier, kind="pushing_punching", label="ผลัก / ชก",
                timestamps=[t for t, _ in frames], frame_paths=[p for _, p in frames]))
            locations[identifier] = list(map(float, (start, end)))
    result = dict(split="development", sequences=[1, 2], scope="All source windows; verifier-only, not a Gemma candidate benchmark",
                  payload=payload, locations=locations, runs={}, complete=False)
    client = WorkerClient()
    def save():
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    save()
    for mode in ("legacy", "temporal-v1"):
        started = time.perf_counter(); attempts = {}; telemetry = []
        result["runs"][mode] = dict(attempts=attempts, telemetry=telemetry, complete=False)
        for offset in range(0, len(payload), 8):
            response = client.post("/verify-batch", {"events": [dict(p, verification_mode=mode) for p in payload[offset:offset+8]]}, timeout=8*480+30)
            for reply in response["results"]:
                attempts[reply["id"]] = [dict(bounds=locations[reply["id"]], reply=reply)]
            telemetry.append(response.get("telemetry", {})); save()
            print(mode, min(offset+8, len(payload)), "/", len(payload), flush=True)
        if len(attempts) != len(payload):
            raise ValueError("Incomplete response set; retain partial artifact")
        result["runs"][mode].update(complete=True, elapsed_seconds=time.perf_counter()-started,
            metrics=metrics(source, attempts, (1, 2), False),
            private_requests=sum(rows[0]["reply"].get("calls", 0) for rows in attempts.values()))
        save()
    baseline, alternative = [result["runs"][mode]["metrics"] for mode in ("legacy", "temporal-v1")]
    result["promote_to_validation"] = (alternative["covered_events"] > baseline["covered_events"] and
        alternative["non_target_windows"] <= baseline["non_target_windows"])
    result["complete"] = True; save()
    print("COMPLETE", {mode: run["metrics"] for mode, run in result["runs"].items()}, flush=True)


if __name__ == "__main__":
    main()
