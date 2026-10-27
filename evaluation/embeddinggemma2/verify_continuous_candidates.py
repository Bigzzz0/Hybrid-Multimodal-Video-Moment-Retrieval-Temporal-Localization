"""Verify the already-frozen candidate list with the production Qwen broker.
No labels enter the Qwen request. Raw failures stay unverified in the artifact.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import requests

from app.gemma2.service import WorkerClient, verification_frames
from app.gemma2.config import IndexConfig
from app.gemma2.media import extract
from .continuous_screening import measure


def summarize_policy(source, replies, locations, ids, sampling, sequences=(17, 18, 19, 20)):
    labels = {int(seq): rows for seq, rows in source["annotations"].items()}
    wanted = set(ids)
    candidate = {seq: np.full(len(source["scores"][str(seq)]["bounds"]), -1.) for seq in sequences}
    confirmed = {seq: values.copy() for seq, values in candidate.items()}
    view_bounds = {seq: np.array(source["scores"][str(seq)]["bounds"], dtype=float) for seq in sequences}
    for identifier in wanted:
        where = locations[identifier]; seq, wi = where["seq"], where["window"]
        candidate[seq][wi] = 1.
        if sampling == "central":
            start, end = where["bounds"]; span = min(3., end-start)
            left = (start+end-span)/2
            view_bounds[seq][wi] = [left, left+span]
    subset = [reply for reply in replies if reply["id"] in wanted]
    for reply in subset:
        if reply["status"] == "verified" and (reply.get("verdict") or {}).get("event_present"):
            where = locations[reply["id"]]; confirmed[where["seq"]][where["window"]] = 1.
    def evaluate(values, visible=False):
        return measure([(seq, {"bounds": view_bounds[seq] if visible else np.array(source["scores"][str(seq)]["bounds"])},
                         {"positive": values[seq]}) for seq in sequences], "positive", 0., labels)
    return dict(gemma_coverage=evaluate(candidate), positive_candidate_coverage=evaluate(confirmed),
        positive_qwen_coverage=evaluate(confirmed, visible=True),
        unverified=sum(reply["status"] != "verified" or reply.get("verdict") is None for reply in subset)+len(wanted-{reply["id"] for reply in subset}),
        private_worker_http_requests=sum(reply.get("calls", 0) for reply in subset))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--compare", action="store_true")
    parser.add_argument("--sampling", choices=("uniform", "central"), default="uniform")
    parser.add_argument("--output", default="scratch/event-retrieval/continuous-ut")
    args = parser.parse_args()
    root = Path(args.output)
    source = json.loads((root/"results.json").read_text(encoding="utf-8"))
    lock = json.loads((root/"crossfit-lock.json").read_text(encoding="utf-8"))
    method = lock["selected"]; threshold = lock["arms"][method]["threshold"]
    policies = {method: threshold}
    if args.compare:
        policies["prompt_margin"] = lock["arms"]["prompt_margin"]["threshold"]
    policy_ids = {name: [] for name in policies}
    output_path = root/("qwen-results.json" if args.sampling == "uniform" else "qwen-central-results.json")
    if output_path.exists():
        raise ValueError("Qwen result already exists; preserve the measured run")
    jobs = requests.get("http://127.0.0.1:8002/api/v1/events/jobs", timeout=10).json()
    if any(j["status"] in ("running", "queued", "cancelling") for j in jobs):
        raise RuntimeError("Screening jobs active")
    payload, locations = [], {}
    for seq in range(17, 21):
        feature = source["scores"][str(seq)]; provenance = source["provenance"][str(seq)]
        sidecar = next(p for p in (root/"features").glob("*.json") if json.loads(p.read_text())["seq"] == seq)
        selected_ids = {name: set(map(int, np.flatnonzero(np.asarray(feature[name], dtype=np.float64) >= value))) for name, value in policies.items()}
        for wi in sorted(set().union(*selected_ids.values())):
            start, end = feature["bounds"][int(wi)]
            trace = provenance["traces"][int(wi)]
            paths = [str((root/"frames"/sidecar.stem/str(wi)/f"frame-{i:02}.jpg").resolve()) for i in range(trace["frame_count"])]
            if not all(Path(p).is_file() for p in paths):
                raise ValueError("Verification frame cache missing")
            selected = verification_frames([int(wi)], {int(wi): dict(timestamps=trace["timestamps"], frame_paths=paths)}, start, end)
            if args.sampling == "central":
                span = min(3., end-start); left = (start+end-span)/2
                zoom_config = IndexConfig(window_seconds=span, stride_seconds=span,
                    video_fps=8/span, image_fps=8/span, max_frames=8)
                zoom = extract(provenance["path"], left, left+span,
                    root/"central-frames"/f"seq{seq}"/str(wi), zoom_config, False)
                selected = verification_frames([0], {0: zoom}, left, left+span)
            identifier = f"continuous-ut-{seq}-{wi}"
            for name, ids in selected_ids.items():
                if wi in ids:
                    policy_ids[name].append(identifier)
            payload.append(dict(id=identifier, kind="pushing_punching", label="ผลัก / ชก",
                timestamps=[stamp for stamp, _ in selected], frame_paths=[path for _, path in selected]))
            locations[identifier] = dict(seq=seq, window=int(wi), bounds=[start, end])
    started = time.perf_counter(); client = WorkerClient(); batches = []; replies = []
    output = dict(method=method, threshold=threshold, total_candidates=len(payload),
                  locations=locations, payload=payload, batches=batches, results=replies,
                  policies=policies, policy_ids=policy_ids, complete=False,
                  sampling=args.sampling,
                  limitation="Qwen decisions are compared to official labels, not used as ground truth; coverage is not final localization precision")
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    for offset in range(0, len(payload), 8):
        if time.perf_counter()-started > 900:
            raise TimeoutError("Fifteen-minute verification budget exceeded; partial artifact retained")
        batch = payload[offset:offset+8]
        response = client.post("/verify-batch", {"events": batch}, timeout=8*480+30)
        replies.extend(response["results"]); batches.append(response.get("telemetry", {}))
        output["elapsed_seconds"] = time.perf_counter()-started
        output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print("QWEN", len(replies), "/", len(payload), flush=True)
    labels = {int(seq): rows for seq, rows in source["annotations"].items()}
    positive = {seq: np.full(len(source["scores"][str(seq)]["bounds"]), -1.) for seq in range(17, 21)}
    unknown = 0
    for reply in replies:
        if reply["status"] != "verified" or reply.get("verdict") is None:
            unknown += 1
        elif reply["verdict"]["event_present"]:
            where = locations[reply["id"]]; positive[where["seq"]][where["window"]] = 1.
    records = [(seq, {"bounds": np.array(source["scores"][str(seq)]["bounds"])}, {"positive": positive[seq]}) for seq in range(17, 21)]
    output.update(positive_candidate_coverage=measure(records, "positive", 0., labels),
        unverified=unknown, private_worker_http_requests=sum(r.get("calls", 0) for r in replies),
        complete=True, elapsed_seconds=time.perf_counter()-started)
    output["per_policy"] = {name: summarize_policy(source, replies, locations, ids, args.sampling) for name, ids in policy_ids.items()}
    output["comparison_scope"] = "Shared Qwen responses for overlapping candidates; per-policy request counts are subsets of one union run, not independent timed runs. Control arm added for diagnostic comparison after Gemma test inspection; thresholds remain frozen."
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print("COMPLETE QWEN", output["positive_candidate_coverage"], flush=True)


if __name__ == "__main__":
    main()
