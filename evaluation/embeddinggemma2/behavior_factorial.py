"""Frozen development 2x2 prompt/sampling comparison with conditional retries.

Same Qwen weights, full frames and verification budgets in all arms. Grid arms
include the full-window control then up to five overlapping 2s/1s-step retries,
stopping per window at a positive. Caption generation is disabled for every arm.
No annotations enter requests or sampling. Inference is through the GPU broker.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import requests

from app.gemma2.config import IndexConfig
from app.gemma2.media import extract
from app.gemma2.service import WorkerClient, verification_frames
from .analyse_continuous_sampling import is_positive
from .validate_dense_verification import metrics

MODES = ("temporal-v1", "behavior-v2")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spans(start, end):
    if not np.isfinite([start, end]).all() or start < 0 or end <= start:
        raise ValueError("Invalid grid interval")
    if end-start <= 2.:
        return [[start, end]]
    starts = list(np.arange(start, end-2.+1e-8, 1.))
    if not np.isclose(starts[-1]+2., end):
        starts.append(end-2.)
    return [[float(s), float(min(s+2., end))] for s in starts]


def summarize(source, attempts, sequences, grid):
    views = {key: rows if grid else rows[:1] for key, rows in attempts.items()}
    return dict(metrics=metrics(source, views, sequences, True),
        private_requests=sum(row["reply"].get("calls", 0) for rows in views.values() for row in rows),
        pending=sum(not any(is_positive(row["reply"]) for row in rows) and
                    any(row["reply"].get("status") != "verified" for row in rows) for rows in views.values()))


def choose_arm(arms):
    # Four targets, zero out-of-target development windows allowed. Must beat
    # historical motion dev 3/4 as well as the current matched uniform control.
    baseline = arms["temporal-v1/uniform"]["metrics"]
    eligible = [(name, arm) for name, arm in arms.items()
        if arm["metrics"]["non_target_windows"] == 0 and
        arm["metrics"]["covered_events"] > max(3, baseline["covered_events"])]
    if not eligible:
        return None
    return min(eligible, key=lambda pair: (-pair[1]["metrics"]["covered_events"],
        pair[1]["pending"], pair[1]["private_requests"], pair[0]))[0]


def check_jobs():
    jobs = requests.get("http://127.0.0.1:8002/api/v1/events/jobs", timeout=10).json()
    if any(job["status"] in ("running", "queued", "cancelling") for job in jobs):
        raise RuntimeError("Active application job; evaluation stopped without erasing progress")


def run(split, root):
    development = split == "development"
    sequences = [1, 2] if development else [13, 14, 15, 16]
    control_path = root/("temporal-dev/results.json" if development else "motion-zoom-validation/control.json")
    control = json.loads(control_path.read_text(encoding="utf-8"))
    if not control["complete"] or control["sequences"] != sequences or control["split"] != split:
        raise ValueError("Wrong or incomplete control")
    source_path = root/"continuous-ut/results.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    output = root/("behavior-factorial-dev" if development else "behavior-factorial-validation")
    output.mkdir(parents=True, exist_ok=True); target = output/"results.json"
    if target.exists():
        raise ValueError("Preserve measured run; choose a separate output directory")
    code_paths = [Path(__file__), Path("evaluation/embeddinggemma2/worker.py"),
                  Path("backend/inference_worker/vlm_service.py"), Path("backend/app/inference/contracts.py")]
    protocol = dict(split=split, sequences=sequences, code_sha256={str(p): sha(p) for p in code_paths},
        control_sha256=sha(control_path), source_sha256=sha(source_path),
        sampling="uniform full window then conditional 2-second/1-second-step grid, eight frames, stop at positive",
        selection="zero dev out-of-target; coverage > matched uniform and historical motion 3/4; then pending/calls/name",
        captions=False, model="caprl_qwen3vl_4b_q6", scope="Observable motion, not confirmed harmful contact")
    modes = list(MODES); grid_modes = list(MODES)
    if not development:
        lock_path = root/"behavior-factorial-dev/selection-lock.json"
        locked = json.loads(lock_path.read_text(encoding="utf-8"))
        if not locked["selected_arm"] or locked["code_sha256"] != protocol["code_sha256"]:
            raise ValueError("No passing frozen development configuration or code changed")
        selected_mode, sampling = locked["selected_arm"].split("/")
        modes = list(dict.fromkeys(["temporal-v1", selected_mode]))
        grid_modes = [selected_mode] if sampling == "grid" else []
        protocol.update(selected_arm=locked["selected_arm"], selection_lock_sha256=sha(lock_path))
    (output/"protocol-lock.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    # Verify media identity before inference; no GT-dependent frame sampling.
    for seq in sequences:
        provenance = source["provenance"][str(seq)]
        if sha(provenance["path"]) != provenance["sha256"]:
            raise ValueError("Source recording fingerprint changed")
    check_jobs(); started = time.perf_counter()
    result = dict(protocol=protocol, complete=False, locations=control["locations"],
        attempts={mode: {} for mode in modes}, batches=[], arms={})
    client = WorkerClient()
    def save():
        result["elapsed_seconds"] = time.perf_counter()-started
        target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    save()
    # Stage zero is shared by the uniform and conditional-grid arms, no duplicated
    # inference or fabricated arm wall time. Two modes interleaved within batches.
    for stage in range(6):
        tasks = []
        for payload in control["payload"]:
            identifier = payload["id"]; bounds = control["locations"][identifier]
            for mode in modes:
                rows = result["attempts"][mode].setdefault(identifier, [])
                if stage and (mode not in grid_modes or any(is_positive(r["reply"]) for r in rows)):
                    continue
                where = bounds
                frames_payload = payload
                if stage:
                    grid = spans(*bounds)
                    if stage > len(grid):
                        continue
                    where = grid[stage-1]; duration = where[1]-where[0]
                    seq = int(identifier.split("-")[-2]); path = source["provenance"][str(seq)]["path"]
                    config = IndexConfig(window_seconds=duration, stride_seconds=duration,
                        video_fps=8/duration, image_fps=8/duration, max_frames=8)
                    frames = extract(path, *where, output/"frames"/identifier/str(stage), config, False)
                    selected = verification_frames([0], {0: frames}, *where)
                    frames_payload = dict(payload, frame_paths=[p for _, p in selected], timestamps=[t for t, _ in selected])
                task = dict(frames_payload, id=f"{mode}:{identifier}", verification_mode=mode, generate_caption=False)
                tasks.append((mode, identifier, where, task))
        for offset in range(0, len(tasks), 8):
            batch = tasks[offset:offset+8]; check_jobs(); batch_started = time.perf_counter()
            response = client.post("/verify-batch", {"events": [task[3] for task in batch]}, timeout=8*480+30)
            replies = {reply["id"]: reply for reply in response["results"]}
            if len(replies) != len(batch) or set(replies) != {task[3]["id"] for task in batch}:
                raise ValueError("Missing/duplicated Qwen responses; retain partial run")
            for mode, identifier, where, task in batch:
                result["attempts"][mode][identifier].append(dict(bounds=where, stage=stage,
                    payload=task, reply=replies[task["id"]]))
            result["batches"].append(dict(stage=stage, events=len(batch),
                elapsed_seconds=time.perf_counter()-batch_started, telemetry=response.get("telemetry", {})))
            save(); print(split, "STAGE", stage, min(offset+8, len(tasks)), "/", len(tasks), flush=True)
        for mode in modes:
            result["arms"][mode+"/uniform"] = summarize(source, result["attempts"][mode], sequences, False)
            if mode in grid_modes:
                result["arms"][mode+"/grid"] = summarize(source, result["attempts"][mode], sequences, True)
        save()
    result["complete"] = True
    if development:
        result["selected_arm"] = choose_arm(result["arms"])
    save()
    if development:
        (output/"selection-lock.json").write_text(json.dumps(dict(selected_arm=result["selected_arm"],
            development_sha256=sha(target), code_sha256=protocol["code_sha256"],
            arms=result["arms"]), indent=2), encoding="utf-8")
    print("COMPLETE", json.dumps(result["arms"]), "SELECTED", result.get("selected_arm"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["development", "validation"], default="development")
    args = parser.parse_args(); run(args.split, Path("scratch/event-retrieval"))
