"""Actually run Qwen on every twelve-second unit with the same context/frame prompt.

Matches merged verification unit budget, not six-second overlapping index windows.
No reference/threshold changes, no annotations supplied to Qwen.
"""
import argparse
import json
import time
from pathlib import Path

import requests

from app.gemma2 import normality as n
from app.gemma2.open_screening import reference_frames
from app.gemma2.service import WorkerClient
from app.gemma2.store import Store, file_hash
from .open_anomaly_eval import audit, metrics


def main():
    from types import SimpleNamespace
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output)
    if root.exists():
        raise ValueError("Baseline output must be new")
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    rows = manifest["videos"]
    audit(rows)
    experiment = Path(args.experiment)
    frozen = json.loads((experiment / "results.json").read_text(encoding="utf-8"))
    if frozen["status"] != "complete" or not frozen["code_unchanged"]:
        raise ValueError("Comparison requires completed frozen experiment")
    if n.digest(manifest) != frozen["protocol"]["manifest_hash"]:
        raise ValueError("Baseline manifest differs from screened experiment")
    jobs = requests.get("http://127.0.0.1:8002/api/v1/events/jobs", timeout=10).json()
    if any(j["status"] in ("running", "queued", "cancelling") for j in jobs):
        raise ValueError("Demo jobs active")
    root.mkdir(parents=True)
    store = Store(experiment / "store")
    payload, lookups = [], {}
    # reference_frames uses only these three read-only operations from EventService.
    service = SimpleNamespace(store=store, namespace=lambda v: frozen["protocol"]["config"] and
        __import__("app.gemma2.config", fromlist=["IndexConfig"]).IndexConfig(**frozen["protocol"]["config"]).namespace(v["fingerprint"], v["captions_hash"]))
    def validate(video):
        if file_hash(Path(video["path"])) != video["fingerprint"]:
            raise ValueError("Reference source changed")
    service.validate_source = validate
    for row in rows:
        if row["split"] != "validation":
            continue
        prior_job = next(j for j in frozen["verification_jobs"] if j["video_id"] == row["id"])
        policy = prior_job["open_policy"]
        bank = n.bank(store, policy["profile_id"], policy["reference_version"])
        # Same nearest-window evidence ordering as screened units, ignoring selection gates.
        for index, start in enumerate(range(0, int(row["duration"])+1, 12)):
            if start >= row["duration"]:
                continue
            end = min(start+12., row["duration"])
            found = {}
            for window_id, evidence in prior_job["screening_scores"].items():
                windows = store.windows(service.namespace(store.get("videos", row["id"])))
                w = next(w for w in windows if w["id"] == int(window_id))
                if w["t_start"] >= end or w["t_end"] <= start:
                    continue
                for entries in evidence.values():
                    for score in entries:
                        for ref in score["references"]:
                            found.setdefault(ref["key"], ref)
            frames = n.sample_frames(row["path"], start, end, row["duration"], root / row["id"] / str(index))
            identifier = row["id"] + f"-{index}"
            lookups[identifier] = {"video_id": row["id"], "t_start": start, "t_end": end}
            payload.append({"id": identifier, "kind": "open_anomaly", "verification_mode": n.POLICY,
                "generate_caption": False, "scene_context": bank["context"], "frame_paths": [f["path"] for f in frames],
                "timestamps": [f["timestamp"] for f in frames], "reference_frames": reference_frames(service, list(found.values())[:2])})
    protocol = {"source_results_hash": file_hash(experiment / "results.json"), "unit_seconds": 12,
                "frame_budget": "8 event + up to 2 before + 2 after + 2 normal reference", "payload": payload,
                "test_opened": False, "index_baseline": False}
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    client = WorkerClient(); started = time.perf_counter(); replies, telemetry = [], []
    for offset in range(0, len(payload), 8):
        response = client.post("/verify-batch", {"events": payload[offset:offset+8]}, timeout=8*480+30)
        replies.extend(response["results"]); telemetry.append(response.get("telemetry", {}))
    predictions = {r["id"]: [] for r in rows if r["split"] == "validation"}
    for reply in replies:
        interval = lookups[reply["id"]]
        predictions[interval["video_id"]].append({"t_start": interval["t_start"], "t_end": interval["t_end"],
            "assessment": (reply.get("verdict") or {}).get("assessment", "uncertain")})
    calls = sum(r.get("calls", 0) for r in replies)
    screened_calls = sum(j.get("qwen_calls", 0) for j in frozen["verification_jobs"])
    report = {"protocol": protocol, "replies": replies, "predictions": predictions,
              "elapsed_seconds": time.perf_counter()-started, "private_http_requests": calls,
              "screened_private_http_requests": screened_calls,
              "request_reduction": 1-screened_calls/calls if calls else None,
              "telemetry": telemetry, "metrics": metrics([r for r in rows if r["split"] == "validation"], predictions),
              "limitations": "Tiny synthetic validation subset; normal rows have no candidates, no retry equality claim outside this run"}
    (root / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("elapsed_seconds", "private_http_requests", "screened_private_http_requests", "request_reduction", "metrics")}, indent=2))


if __name__ == "__main__":
    main()
