"""Read-only measured-artifact analysis; never changes frozen selection rules."""
import json
from pathlib import Path

from .analyse_continuous_sampling import is_positive
from .behavior_factorial import sha
from .validate_dense_verification import metrics
from .continuous_screening import selected_targets, overlap


def localized_attempts(attempts, grid):
    result = {}
    for identifier, rows in attempts.items():
        result[identifier] = []
        for row in rows if grid else rows[:1]:
            bounds = row["bounds"]
            if is_positive(row["reply"]):
                verdict = row["reply"]["verdict"]; stamps = row["payload"]["timestamps"]
                start, end = verdict["start_frame_index"], verdict["end_frame_index"]
                if not 0 <= start < end < len(stamps):
                    raise ValueError("Invalid measured positive frame evidence")
                bounds = [stamps[start], stamps[end]]
            result[identifier].append(dict(bounds=bounds, reply=row["reply"]))
    return result


def evidence_hits(source, attempts, sequences):
    hits, total = 0, 0
    for seq in sequences:
        evidence = [row["bounds"] for identifier, rows in attempts.items()
                    if int(identifier.split("-")[-2]) == seq for row in rows if is_positive(row["reply"])]
        for event in selected_targets(source["annotations"][str(seq)]):
            total += 1
            hits += any(overlap(bounds, event["bounds"]) > 0 for bounds in evidence)
    return dict(events=total, events_with_evidence_overlap=hits,
                scope="Any overlap only, not duration coverage/IoU or a correct complete event boundary")


def analyse(root):
    source = json.loads((root/"continuous-ut/results.json").read_text(encoding="utf-8"))
    result = {}
    for split, folder in [("development", "behavior-factorial-dev"), ("validation", "behavior-factorial-validation")]:
        path = root/folder/"results.json"
        if not path.exists():
            continue
        run = json.loads(path.read_text(encoding="utf-8"))
        if not run["complete"]:
            raise ValueError("Measured run still incomplete")
        resources = [b["telemetry"].get("qwen_resources", {}) for b in run["batches"]]
        def peak(key):
            values = [r[key] for r in resources if isinstance(r.get(key), (int, float))]
            return max(values) if values else None
        arms = {}
        for name, arm in run["arms"].items():
            mode, sampling = name.split("/")
            localized = localized_attempts(run["attempts"][mode], sampling == "grid")
            arms[name] = dict(arm, localized_frame_span_metrics=metrics(source,
                localized, run["protocol"]["sequences"], True),
                frame_evidence_any_overlap=evidence_hits(source, localized, run["protocol"]["sequences"]))
        result[split] = dict(raw_sha256=sha(path), complete=True, arms=arms,
            elapsed_seconds=run["elapsed_seconds"],
            actual_private_requests=sum(row["reply"].get("calls", 0)
                for attempts in run["attempts"].values() for rows in attempts.values() for row in rows),
            sampled_whole_gpu_peak_bytes=peak("sampled_whole_gpu_peak_bytes"),
            qwen_process_tree_peak_rss_bytes=peak("qwen_process_tree_peak_rss_bytes"),
            scope="Combined interleaved experiment wall time, not independent per-arm wall times; whole GPU includes other apps")
    return result


if __name__ == "__main__":
    root = Path("scratch/event-retrieval"); target = root/"behavior-factorial-analysis.json"
    if target.exists():
        raise ValueError("Preserve existing analysis")
    report = analyse(root)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
