"""Exploratory recall-first triage arm, frozen before validation inference.

Raw original development protocol remains immutable. No claim of preregistration
for this extra arm: it was motivated by development-only error analysis.
"""
import argparse
import json
from pathlib import Path

from .benchmark_action_expert import attempts, digest
from .validate_dense_verification import metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["freeze", "report"])
    args = parser.parse_args()
    root = Path("scratch/event-retrieval")
    lock_path = root/"action-expert-recall-lock.json"
    dev_path = root/"action-expert-dev/results.json"
    dev = json.loads(dev_path.read_text(encoding="utf-8"))
    if not dev["complete"]:
        raise ValueError("Incomplete development run")
    if args.mode == "freeze":
        if lock_path.exists() or (root/"action-expert-validation/results.json").exists():
            raise ValueError("Freeze once before validation inference")
        chosen = max(dev["calibration_trials"], key=lambda t: (t["metrics"]["covered_events"],
            -t["metrics"]["non_target_windows"], -t["positive_windows"], t["threshold"]))
        lock = dict(development_sha256=digest(dev_path), threshold=chosen["threshold"],
                    development=chosen, exploratory=True,
                    rule="maximize coverage; fewer out-of-target windows, fewer positives, higher threshold")
        lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
        print("FROZEN", chosen, flush=True)
        return
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["development_sha256"] != digest(dev_path):
        raise ValueError("Development artifact changed")
    source = json.loads((root/"continuous-ut/results.json").read_text(encoding="utf-8"))
    report = dict(lock=lock, arms={})
    for split, path, sequences in [("development", dev_path, (1, 2)),
            ("validation", root/"action-expert-validation/results.json", (13, 14, 15, 16))]:
        run = json.loads(path.read_text(encoding="utf-8"))
        if not run["complete"]:
            raise ValueError("Incomplete measured run")
        threshold = lock["threshold"]
        report["arms"][split] = dict(strict_candidate=run["candidate_metrics"],
            recall_first_candidate=metrics(source, attempts(run["rows"], threshold), sequences, True),
            recall_first_span=metrics(source, attempts(run["rows"], threshold, True), sequences, True),
            elapsed_seconds=run["elapsed_seconds"], peak_process_rss_bytes=run["peak_process_rss_bytes"],
            unverified=run["unverified"], raw_sha256=digest(path))
    target = root/"action-expert-analysis.json"
    if target.exists():
        raise ValueError("Preserve existing analysis")
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
