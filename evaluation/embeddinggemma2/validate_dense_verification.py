"""Validation-only adaptive dense verification; no labels in model requests.

Keep uniform positives. Retry other candidates in two chronological halves,
stopping after a positive. Never turn failed verification into a negative.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import requests

from app.gemma2.config import IndexConfig
from app.gemma2.media import extract
from app.gemma2.service import WorkerClient, verification_frames
from .analyse_continuous_sampling import is_positive
from .continuous_screening import measure


def halves(start, end):
    if not np.isfinite([start, end]).all() or end <= start:
        raise ValueError("Invalid interval")
    middle = (start+end)/2
    return [(start, middle), (middle, end)]


def decision(attempts):
    if any(is_positive(row["reply"]) for row in attempts):
        return "positive"
    if len(attempts) < 3 or any(row["reply"].get("status") != "verified" or
                              row["reply"].get("verdict") is None for row in attempts):
        return "pending"
    return "negative"


def metrics(source, attempts, sequences, dense):
    records = []
    for seq in sequences:
        intervals, values = [], []
        for identifier, rows in attempts.items():
            if int(identifier.split("-")[-2]) != seq:
                continue
            for row in rows if dense else rows[:1]:
                intervals.append(row["bounds"])
                values.append(1. if is_positive(row["reply"]) else -1.)
        records.append((seq, {"bounds": np.asarray(intervals).reshape(-1, 2)},
                        {"positive": np.asarray(values)}))
    labels = {int(k): v for k, v in source["annotations"].items()}
    return measure(records, "positive", 0., labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="scratch/event-retrieval/continuous-ut")
    parser.add_argument("--output", default="scratch/event-retrieval/dense-validation")
    args = parser.parse_args(); root = Path(args.source); output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True); target = output/"results.json"
    if target.exists():
        raise ValueError("Preserve measured run; choose a new output directory")
    jobs = requests.get("http://127.0.0.1:8002/api/v1/events/jobs", timeout=10).json()
    if any(j["status"] in ("running", "queued", "cancelling") for j in jobs):
        raise RuntimeError("Screening jobs active")
    source = json.loads((root/"results.json").read_text(encoding="utf-8"))
    lock = json.loads((root/"crossfit-lock.json").read_text(encoding="utf-8"))
    threshold = lock["arms"]["prompt_margin"]["threshold"]
    sequences = (13, 14, 15, 16)
    candidates, attempts = {}, {}
    for seq in sequences:
        feature = source["scores"][str(seq)]
        for wi in np.flatnonzero(np.asarray(feature["prompt_margin"], dtype=np.float64) >= threshold):
            identifier = f"dense-ut-{seq}-{wi}"
            candidates[identifier] = dict(seq=seq, window=int(wi), bounds=feature["bounds"][int(wi)])
            attempts[identifier] = []
    result = dict(split="validation", sequences=sequences, threshold=threshold,
        protocol="uniform eight frames; two dense halves only after nonpositive; stop at first positive",
        candidates=candidates, attempts=attempts, telemetry=[], complete=False)
    client = WorkerClient(); started = time.perf_counter()
    def save():
        result["elapsed_seconds"] = time.perf_counter()-started
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    save()
    for stage in range(3):
        tasks = []
        for identifier, where in candidates.items():
            if any(is_positive(row["reply"]) for row in attempts[identifier]):
                continue
            start, end = where["bounds"]; seq = where["seq"]; wi = where["window"]
            provenance = source["provenance"][str(seq)]
            if stage == 0:
                sidecar = next(p for p in (root/"features").glob("*.json") if json.loads(p.read_text())["seq"] == seq)
                trace = provenance["traces"][wi]
                paths = [str((root/"frames"/sidecar.stem/str(wi)/f"frame-{i:02}.jpg").resolve()) for i in range(trace["frame_count"])]
                selected = verification_frames([wi], {wi: dict(timestamps=trace["timestamps"], frame_paths=paths)}, start, end)
            else:
                start, end = halves(start, end)[stage-1]; span = end-start
                config = IndexConfig(window_seconds=span, stride_seconds=span,
                    video_fps=8/span, image_fps=8/span, max_frames=8)
                frames = extract(provenance["path"], start, end, output/"frames"/identifier/str(stage), config, False)
                selected = verification_frames([0], {0: frames}, start, end)
            if not selected or not all(Path(p).is_file() for _, p in selected):
                raise ValueError("Missing verification frames")
            payload = dict(id=identifier, kind="pushing_punching", label="ผลัก / ชก",
                timestamps=[t for t, _ in selected], frame_paths=[p for _, p in selected])
            tasks.append((payload, [start, end]))
        for offset in range(0, len(tasks), 8):
            if time.perf_counter()-started > 1200:
                raise TimeoutError("Twenty-minute budget; partial results retained")
            batch = tasks[offset:offset+8]
            response = client.post("/verify-batch", {"events": [p for p, _ in batch]}, timeout=8*480+30)
            replies = {r["id"]: r for r in response["results"]}
            for payload, bounds in batch:
                reply = replies.get(payload["id"], dict(id=payload["id"], status="unverified", verdict=None, calls=0))
                attempts[payload["id"]].append(dict(stage=stage, bounds=bounds, payload=payload, reply=reply))
            result["telemetry"].append(response.get("telemetry", {})); save()
            print("STAGE", stage, min(offset+8, len(tasks)), "/", len(tasks), flush=True)
    result.update(complete=True, uniform=metrics(source, attempts, sequences, False),
        dense=metrics(source, attempts, sequences, True),
        decisions={identifier: decision(rows) for identifier, rows in attempts.items()},
        private_requests=sum(row["reply"].get("calls", 0) for rows in attempts.values() for row in rows))
    save(); print("COMPLETE", result["uniform"], result["dense"], flush=True)


if __name__ == "__main__":
    main()
