"""Development-only motion-focused retry after nonpositive full-frame Qwen.
Cached control cost is reported separately from actually measured retry cost.
"""
import json
import time
from pathlib import Path

import requests

from app.gemma2.config import IndexConfig
from app.gemma2.media import extract
from app.gemma2.service import WorkerClient, verification_frames
from .analyse_continuous_sampling import is_positive
from .motion_zoom import motion_trace, choose_motion_span
from .validate_dense_verification import metrics


def run(control, source, output):
    if not control["complete"] or control["sequences"] not in ([1, 2], [13, 14, 15, 16]):
        raise ValueError("Needs complete approved control split")
    sequences = tuple(control["sequences"])
    expected = "development" if sequences == (1, 2) else "validation"
    if control.get("split") != expected:
        raise ValueError("Mismatched evaluation split")
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    target = output/"results.json"
    if target.exists():
        raise ValueError("Preserve measured run")
    jobs = requests.get("http://127.0.0.1:8002/api/v1/events/jobs", timeout=10).json()
    if any(j["status"] in ("running", "queued", "cancelling") for j in jobs):
        raise RuntimeError("Active screening jobs")
    started = time.perf_counter()
    baseline = control["runs"]["temporal-v1"]
    attempts = json.loads(json.dumps(baseline["attempts"]))
    result = dict(split=control["split"], sequences=list(sequences), mode="temporal-v1", complete=False,
        baseline=baseline, attempts=attempts, payload=[], traces={}, telemetry=[],
        scope="Cached control plus measured retries; combined end-to-end cascade time not measured")
    def save():
        result["retry_elapsed_seconds"] = time.perf_counter()-started
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    save()
    for payload in control["payload"]:
        identifier = payload["id"]
        if is_positive(attempts[identifier][0]["reply"]):
            continue
        seq = int(identifier.split("-")[-2]); start, end = control["locations"][identifier]
        if len(payload["timestamps"]) < 2:
            continue
        path = source["provenance"][str(seq)]["path"]
        trace = motion_trace(path, start, end)
        span = choose_motion_span(trace["timestamps"], trace["energies"], start, end)
        result["traces"][identifier] = dict(**trace, selected_span=span)
        if span is None:
            continue
        left, right = span; duration = right-left
        config = IndexConfig(window_seconds=duration, stride_seconds=duration,
            video_fps=8/duration, image_fps=8/duration, max_frames=8)
        frames = extract(path, left, right, output/"frames"/identifier, config, False)
        selected = verification_frames([0], {0: frames}, left, right)
        result["payload"].append(dict(payload, frame_paths=[p for _, p in selected],
            timestamps=[t for t, _ in selected], verification_mode="temporal-v1"))
        save()
    client = WorkerClient(); verification_started = time.perf_counter()
    for offset in range(0, len(result["payload"]), 8):
        batch = result["payload"][offset:offset+8]
        response = client.post("/verify-batch", {"events": batch}, timeout=8*480+30)
        replies = {r["id"]: r for r in response["results"]}
        for payload in batch:
            identifier = payload["id"]
            reply = replies.get(identifier, dict(status="unverified", verdict=None, calls=0))
            attempts[identifier].append(dict(bounds=result["traces"][identifier]["selected_span"], reply=reply))
        result["telemetry"].append(response.get("telemetry", {})); save()
        print("MOTION", min(offset+8, len(result["payload"])), "/", len(result["payload"]), flush=True)
    measured = metrics(source, attempts, sequences, True)
    result.update(complete=True, metrics=measured,
        retry_verification_seconds=time.perf_counter()-verification_started,
        derived_total_requests=sum(row["reply"].get("calls", 0) for rows in attempts.values() for row in rows),
        actual_retry_requests=sum(rows[1]["reply"].get("calls", 0) for rows in attempts.values() if len(rows)>1),
        pending=sum(not any(is_positive(row["reply"]) for row in rows) and
                    any(row["reply"].get("status") != "verified" or row["reply"].get("verdict") is None for row in rows)
                    for rows in attempts.values()),
        promote_to_validation=measured["covered_events"] > baseline["metrics"]["covered_events"] and
                              measured["non_target_windows"] <= baseline["metrics"]["non_target_windows"])
    save(); print("COMPLETE", measured, "PROMOTE", result["promote_to_validation"], flush=True)


def main():
    root = Path("scratch/event-retrieval")
    control = json.loads((root/"temporal-dev/results.json").read_text(encoding="utf-8"))
    if control["sequences"] != [1, 2]:
        raise ValueError("Development recordings only")
    source = json.loads((root/"continuous-ut/results.json").read_text(encoding="utf-8"))
    run(control, source, root/"motion-zoom-dev")


if __name__ == "__main__":
    main()
