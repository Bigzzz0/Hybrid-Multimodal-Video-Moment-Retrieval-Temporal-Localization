"""Matched-frame Qwen context ablation; no encoder/threshold/default changes.

Reuses every request from a completed all-unit baseline. The only input change is
a generic camera-normality decision rule appended to scene_context. Annotations
are used only by the evaluator, never to choose frames or construct the rule.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import requests

from app.gemma2.normality import digest
from app.gemma2.service import WorkerClient
from app.gemma2.store import file_hash
from .open_anomaly_eval import audit, metrics, bootstrap

RULE = (
    "Camera normality policy: assess observable deviation from this camera's routine, "
    "not whether the activity is dangerous, violent or harmful. Routine examples are not exhaustive. "
    "Compare the current activity with the stated routine and the separate normal references. "
    "Describe a visible behavioral deviation even if it could be harmless. "
    "A plausible harmless explanation alone does not establish that an unusual behavior is routine here. "
    "If the activity or its normality cannot be determined from the available temporal evidence, "
    "use uncertain rather than a definitive normal judgment. Novel clothing or appearance alone is insufficient."
)


def contextualize(payload):
    changed = copy.deepcopy(payload)
    for event in changed:
        event["scene_context"] += " " + RULE
    return changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output)
    if root.exists():
        raise ValueError("Use a fresh output to preserve prior results")
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    rows = manifest["videos"]
    audit(rows)
    baseline_path = Path(args.baseline) / "results.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    protocol = baseline["protocol"]
    source_path = Path(args.baseline) / "protocol.json"
    if json.loads(source_path.read_text(encoding="utf-8")) != protocol:
        raise ValueError("Baseline protocol changed")
    payload = protocol["payload"]
    validation = [r for r in rows if r["split"] == "validation"]
    ids = {r["id"] for r in validation}
    if {p["id"] for p in payload} != {r["id"] for r in baseline["replies"]}:
        raise ValueError("Baseline replies incomplete")
    # The baseline carries source results; verify the supplied manifest matches it.
    candidate = Path(args.experiment) / "results.json"
    if file_hash(candidate) != protocol["source_results_hash"]:
        raise ValueError("Source experiment hash differs from baseline")
    source_result = json.loads(candidate.read_text(encoding="utf-8"))
    if digest(manifest) != source_result["protocol"]["manifest_hash"]:
        raise ValueError("Cannot establish matching source experiment and manifest")
    jobs = requests.get("http://127.0.0.1:8002/api/v1/events/jobs", timeout=10).json()
    if any(j["status"] in ("running", "queued", "cancelling") for j in jobs):
        raise ValueError("Demo jobs active")
    changed = contextualize(payload)
    lookups = {}
    for row in validation:
        for index, start in enumerate(range(0, int(row["duration"])+1, 12)):
            if start < row["duration"]:
                lookups[row["id"] + f"-{index}"] = (row["id"], start, min(start+12., row["duration"]))
    if set(lookups) != {p["id"] for p in payload}:
        raise ValueError("Baseline does not cover the same validation units")
    frame_hashes = {}
    for event in payload:
        for frame in event["frame_paths"] + [r["path"] for r in event.get("reference_frames", [])]:
            if not Path(frame).is_file():
                raise ValueError("Missing frozen frame")
            frame_hashes[frame] = file_hash(Path(frame))
    root.mkdir(parents=True)
    evidence = {"baseline_hash": file_hash(baseline_path), "manifest_hash": digest(manifest),
                "rule": RULE, "rule_hash": digest(RULE), "payload": changed, "frame_hashes": frame_hashes,
                "only_changed_field": "scene_context", "test_opened": False,
                "selection": "all baseline units, not selected by error or label", "code_hash": file_hash(Path(__file__))}
    (root / "protocol.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    client = WorkerClient(); started = time.perf_counter()
    replies, telemetry = [], []
    for offset in range(0, len(changed), 8):
        response = client.post("/verify-batch", {"events": changed[offset:offset+8]}, timeout=8*480+30)
        replies.extend(response["results"]); telemetry.append(response.get("telemetry", {}))
        (root / "partial.json").write_text(json.dumps({"replies": replies, "telemetry": telemetry}, indent=2), encoding="utf-8")
        print("VERIFIED UNITS", len(replies), "/", len(changed), flush=True)
    if len(replies) != len(lookups) or {r["id"] for r in replies} != set(lookups):
        raise ValueError("Missing or duplicated ablation results")
    predictions = {identifier: [] for identifier in ids}
    transitions = []
    old = {r["id"]: (r.get("verdict") or {}).get("assessment", "uncertain") for r in baseline["replies"]}
    for reply in replies:
        video_id, start, end = lookups[reply["id"]]
        assessment = (reply.get("verdict") or {}).get("assessment", "uncertain")
        predictions[video_id].append({"t_start": start, "t_end": end, "assessment": assessment})
        if assessment != old[reply["id"]]:
            transitions.append({"id": reply["id"], "before": old[reply["id"]], "after": assessment})
    report = {"status": "complete", "protocol": evidence, "predictions": predictions,
              "replies": replies, "telemetry": telemetry, "transitions": transitions,
              "elapsed_seconds": time.perf_counter()-started,
              "private_http_requests": sum(r.get("calls", 0) for r in replies),
              "unknown_request_counts": sum(bool(r.get("calls_unknown")) for r in replies),
              "metrics": metrics(validation, predictions), "baseline_metrics": baseline["metrics"],
              "confidence_intervals": bootstrap(validation, predictions),
              "code_unchanged": file_hash(Path(__file__)) == evidence["code_hash"],
              "frames_unchanged": all(file_hash(Path(p)) == sha for p, sha in frame_hashes.items()),
              "production_activation": False, "test_opened": False}
    (root / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("metrics", "baseline_metrics", "transitions", "elapsed_seconds")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
