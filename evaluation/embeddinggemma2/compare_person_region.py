"""Paired spatial-only development experiment, using saved full-frame control."""
import argparse
import json
import time
from pathlib import Path

import requests

from app.gemma2.service import WorkerClient
from .person_region import prepare_region, SSDLitePersonDetector
from .validate_dense_verification import metrics


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--detector", choices=("hog", "ssdlite"), default="hog")
    args = parser.parse_args()
    root = Path("scratch/event-retrieval")
    control = json.loads((root/"temporal-dev/results.json").read_text(encoding="utf-8"))
    if not control["complete"] or control["sequences"] != [1, 2]:
        raise ValueError("Needs complete development control")
    source = json.loads((root/"continuous-ut/results.json").read_text(encoding="utf-8"))
    output = root/("person-region-dev" if args.detector == "hog" else "person-region-ssdlite-dev")
    output.mkdir(parents=True, exist_ok=True)
    target = output/"results.json"
    if target.exists():
        raise ValueError("Preserve measured run")
    jobs = requests.get("http://127.0.0.1:8002/api/v1/events/jobs", timeout=10).json()
    if any(j["status"] in ("running", "queued", "cancelling") for j in jobs):
        raise RuntimeError("Active screening jobs")
    started = time.perf_counter()
    detector = SSDLitePersonDetector(output/"model-cache") if args.detector == "ssdlite" else None
    result = dict(split="development", sequences=[1, 2], mode="temporal-v1", complete=False,
        detector=args.detector,
        control_file=str((root/"temporal-dev/results.json").resolve()),
        baseline=control["runs"]["temporal-v1"], regions={}, payload=[], attempts={}, telemetry=[])
    def save():
        result["elapsed_seconds"] = time.perf_counter()-started
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    save()
    for payload in control["payload"]:
        paths, provenance = prepare_region(payload["frame_paths"], output/"frames"/payload["id"], detector)
        result["regions"][payload["id"]] = provenance
        result["payload"].append(dict(payload, frame_paths=paths, verification_mode="temporal-v1"))
        save()
    client = WorkerClient(); verification_started = time.perf_counter()
    for offset in range(0, len(result["payload"]), 8):
        response = client.post("/verify-batch", {"events": result["payload"][offset:offset+8]}, timeout=8*480+30)
        for reply in response["results"]:
            result["attempts"][reply["id"]] = [dict(bounds=control["locations"][reply["id"]], reply=reply)]
        result["telemetry"].append(response.get("telemetry", {})); save()
        print("REGION", min(offset+8, len(result["payload"])), "/", len(result["payload"]), flush=True)
    if set(result["attempts"]) != set(control["locations"]):
        raise ValueError("Incomplete measured run")
    measured = metrics(source, result["attempts"], (1, 2), False)
    baseline = result["baseline"]["metrics"]
    result.update(complete=True, metrics=measured,
        verification_seconds=time.perf_counter()-verification_started,
        private_requests=sum(rows[0]["reply"].get("calls", 0) for rows in result["attempts"].values()),
        promote_to_validation=measured["covered_events"] > baseline["covered_events"] and
                              measured["non_target_windows"] <= baseline["non_target_windows"])
    save(); print("COMPLETE", measured, "PROMOTE", result["promote_to_validation"], flush=True)


if __name__ == "__main__":
    main()
