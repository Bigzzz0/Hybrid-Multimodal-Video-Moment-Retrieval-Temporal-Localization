"""Repeat screening on existing indexes; preserve every run's raw evidence.

No threshold tuning or human reviews are performed. Results are diagnostic,
not accuracy estimates without separately annotated temporal ground truth.
"""
import argparse
import json
import time
from pathlib import Path
import requests


def run(api, video_ids, output, timeout_seconds=1800):
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    def save():
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(output)
    for video_id in video_ids:
        response = requests.post(api + "/events/jobs", json={"video_id": video_id, "kind": "screen",
                                 "event_types": ["pushing_punching", "falling"]}, timeout=30)
        response.raise_for_status()
        job = response.json()
        record = {"video_id": video_id, "job_id": job["id"], "started_at": time.time()}
        records.append(record)
        started = time.perf_counter()
        previous = None
        while True:
            response = requests.get(api + "/events/jobs/" + job["id"], timeout=30)
            response.raise_for_status()
            job = response.json()
            record.update(job=job, http_wall_seconds=time.perf_counter()-started)
            save()
            state = (job["status"], job.get("phase"), job.get("completed"), job.get("total"))
            if state != previous:
                print(video_id, state, flush=True)
                previous = state
            if job["status"] in ("complete", "failed", "cancelled", "interrupted"):
                break
            if time.perf_counter()-started >= timeout_seconds:
                response = requests.post(api + "/events/jobs/" + job["id"] + "/cancel", timeout=30)
                response.raise_for_status()
                record["watchdog_cancel_requested"] = True
                save()
                raise TimeoutError("Screening watchdog reached; cancellation requested, evidence retained")
            time.sleep(2)
        response = requests.get(api + "/events/list/" + video_id, timeout=30)
        response.raise_for_status()
        record["events"] = response.json()
        save()
        if job["status"] != "complete":
            raise RuntimeError(f"Screening did not complete: {job['status']} {job.get('error')}")
        rows = record["events"]
        verdicts = [row.get("verification", {}).get("verdict") or {} for row in rows]
        print(video_id, "SUMMARY", {"units": len(rows), "positive": sum(v.get("event_present") is True for v in verdicts),
              "negative": sum(v.get("event_present") is False for v in verdicts),
              "unverified": sum(row["model_status"] == "unverified" for row in rows),
              "worker_requests": job.get("qwen_calls"), "seconds": job.get("elapsed_seconds")}, flush=True)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8002/api/v1")
    parser.add_argument("--videos", nargs="+", default=["ut-punch-demo", "ur-fall-demo", "ucf-long-normal634", "ucf-long-fighting011"])
    parser.add_argument("--output", type=Path, default=Path("scratch/event-retrieval/iterations") / f"screening-{time.time_ns()}.json")
    args = parser.parse_args()
    run(args.api.rstrip("/"), args.videos, args.output)
