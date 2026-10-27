"""Development calibration then frozen validation of optional action triage.

This measures candidate coverage, NOT final Qwen-confirmed event accuracy.
No labels, annotation boxes or action boundaries enter media selection/inference.
"""
import argparse
import hashlib
import json
import threading
import time
from pathlib import Path

import numpy as np
import psutil

from .action_expert import ActionExpert
from .motion_zoom import motion_trace, choose_motion_span
from .validate_dense_verification import metrics


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def attempts(rows, threshold, use_selected_span=False):
    result = {}
    for identifier, row in rows.items():
        evidence = row["evidence"]
        scored = evidence["status"] == "scored"
        positive = scored and evidence["scores"]["target"] >= threshold and evidence["scores"]["margin"] > 0
        result[identifier] = [dict(bounds=evidence["bounds"] if use_selected_span else row["window_bounds"],
            reply=dict(status="verified" if scored else "unverified",
                       verdict={"event_present": bool(positive)} if scored else None))]
    return result


def calibrate(source, rows):
    # Predeclared rule: zero out-of-target development windows, maximize event
    # coverage; ties prefer fewer positive windows, then higher threshold.
    values = [r["evidence"]["scores"]["target"] for r in rows.values() if r["evidence"]["status"] == "scored"]
    candidates = sorted(set(values + [float(np.nextafter(max(values, default=0.), np.inf))]))
    trials = []
    for threshold in candidates:
        selected = attempts(rows, threshold)
        measured = metrics(source, selected, (1, 2), True)
        positives = sum(r[0]["reply"].get("verdict", {}).get("event_present", False)
                        for r in selected.values() if r[0]["reply"]["verdict"] is not None)
        trials.append(dict(threshold=threshold, metrics=measured, positive_windows=positives))
    eligible = [t for t in trials if t["metrics"]["non_target_windows"] == 0]
    chosen = max(eligible, key=lambda t: (t["metrics"]["covered_events"], -t["positive_windows"], t["threshold"]))
    return chosen, trials


def run(split, root):
    development = split == "development"
    sequences = [1, 2] if development else [13, 14, 15, 16]
    control_path = root/("temporal-dev/results.json" if development else "motion-zoom-validation/control.json")
    control = json.loads(control_path.read_text(encoding="utf-8"))
    if not control["complete"] or control["sequences"] != sequences or control["split"] != split:
        raise ValueError("Wrong/unfinished control")
    source = json.loads((root/"continuous-ut/results.json").read_text(encoding="utf-8"))
    output = root/("action-expert-dev" if development else "action-expert-validation")
    output.mkdir(parents=True, exist_ok=True)
    target = output/"results.json"
    if target.exists():
        raise ValueError("Preserve existing measured run")
    protocol = dict(split=split, sequences=sequences, control_sha256=digest(control_path),
        algorithm_sha256=digest(__file__), adapter_sha256=digest(Path(__file__).with_name("action_expert.py")),
        sampling="highest-motion 2s per existing window; quiet fallback central 2s; 32 unique frames",
        score="max AVA person fight/hit and push; target-normal(handshake,hug) > 0",
        selection="zero development out-of-target windows; maximize coverage, fewer positives, higher threshold",
        scope="Action evidence for review/triage; not a final event confirmation")
    if not development:
        dev_path = root/"action-expert-dev/results.json"
        dev = json.loads(dev_path.read_text(encoding="utf-8"))
        if not dev["complete"] or dev["protocol"]["adapter_sha256"] != protocol["adapter_sha256"] or dev["protocol"]["algorithm_sha256"] != protocol["algorithm_sha256"]:
            raise ValueError("Development must finish with identical frozen code")
        protocol.update(threshold=dev["selection"]["threshold"], development_sha256=digest(dev_path))
    (output/"protocol-lock.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    stop = threading.Event(); peak = [0]
    process = psutil.Process()
    def monitor():
        while not stop.is_set():
            peak[0] = max(peak[0], process.memory_info().rss)
            stop.wait(.1)
    sampler = threading.Thread(target=monitor, daemon=True); sampler.start()
    started = time.perf_counter()
    result = dict(protocol=protocol, complete=False, rows={})
    def save():
        result.update(elapsed_seconds=time.perf_counter()-started, peak_process_rss_bytes=peak[0])
        target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    try:
        expert = ActionExpert(root/"action-expert/SLOWFAST_8x8_R50_DETECTION.pyth",
                              root/"person-region-ssdlite-dev/model-cache")
        result["model"] = expert.metadata; save()
        for payload in control["payload"]:
            identifier = payload["id"]; seq = int(identifier.split("-")[-2])
            bounds = control["locations"][identifier]; left, right = bounds
            path = source["provenance"][str(seq)]["path"]
            trace = motion_trace(path, left, right)
            span = choose_motion_span(trace["timestamps"], trace["energies"], left, right)
            if span is None:
                duration = min(2., right-left); center = (left+right)/2
                span = [center-duration/2, center+duration/2]
            try:
                evidence = expert.predict(path, span)
            except ValueError as error:
                evidence = dict(status="unverified", reason=str(error), bounds=span)
            result["rows"][identifier] = dict(window_bounds=bounds, evidence=evidence)
            save()
            print(split, len(result["rows"]), "/", len(control["payload"]), flush=True)
        if development:
            result["selection"], result["calibration_trials"] = calibrate(source, result["rows"])
            threshold = result["selection"]["threshold"]
        else:
            threshold = protocol["threshold"]
        result.update(threshold=threshold,
            candidate_metrics=metrics(source, attempts(result["rows"], threshold), sequences, True),
            selected_span_metrics=metrics(source, attempts(result["rows"], threshold, True), sequences, True),
            unverified=sum(r["evidence"]["status"] != "scored" for r in result["rows"].values()), complete=True)
        save(); print("COMPLETE", result["candidate_metrics"], "SPAN", result["selected_span_metrics"], flush=True)
    finally:
        stop.set(); sampler.join(timeout=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["development", "validation"], default="development")
    args = parser.parse_args()
    run(args.split, Path("scratch/event-retrieval"))
