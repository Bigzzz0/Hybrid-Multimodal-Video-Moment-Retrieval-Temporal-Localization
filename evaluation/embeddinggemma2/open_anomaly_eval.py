"""Auditable continuous-video metrics and sealed open-anomaly evaluation protocol.

Manifest rows require recording_id, scene_id, dataset, split, official_split,
path, sha256, duration, events [{t_start,t_end}], source_url and license.
Use `python -m evaluation.embeddinggemma2.open_anomaly_eval --manifest ...`
to audit before encoding. Test labels are read only with --test --lock.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from app.gemma2.normality import POLICY, PROMPTS, digest
from app.gemma2.store import file_hash


def audit(rows, allow_test=False, check_files=True):
    ids, recordings, hashes = set(), {}, {}
    for row in rows:
        if row["id"] in ids:
            raise ValueError("Duplicate sample")
        ids.add(row["id"])
        split = row["split"]
        if split not in ("development", "validation", "test"):
            raise ValueError("Unknown split")
        if row["official_split"] == "test" and split != "test":
            raise ValueError("Official test cannot become development/validation")
        if split == "test" and not allow_test:
            raise ValueError("Test access requires a frozen validation lock")
        for mapping, key in ((recordings, (row["dataset"], row["recording_id"])), (hashes, row["sha256"])):
            if key in mapping and mapping[key] != split:
                raise ValueError("Recording/hash leakage across splits")
            mapping[key] = split
        if not row["scene_id"] or not row["source_url"] or not row["license"]:
            raise ValueError("Missing scene/provenance/license")
        duration = row["duration"]
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Invalid duration")
        for event in row["events"]:
            if not 0 <= event["t_start"] < event["t_end"] <= duration:
                raise ValueError("Invalid annotation")
        if check_files and file_hash(Path(row["path"])) != row["sha256"]:
            raise ValueError("Source hash mismatch")


def union(intervals):
    merged = []
    for a, b in sorted(intervals):
        if not all(math.isfinite(x) for x in (a, b)) or a >= b:
            raise ValueError("Invalid predicted interval")
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(b, merged[-1][1])
        else:
            merged.append([a, b])
    return merged


def iou(a, b):
    overlap = max(0, min(a[1], b[1])-max(a[0], b[0]))
    return overlap/(a[1]-a[0]+b[1]-b[0]-overlap)


def match_events(pred, gt):
    """Maximum one-to-one matching; map ground-truth indices to prediction indices."""
    edges = [[j for j, event in enumerate(gt) if iou(p, event) >= .3] for p in pred]
    matched = {}
    def augment(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in matched or augment(matched[j], seen):
                matched[j] = i
                return True
        return False
    for i in range(len(pred)):
        augment(i, set())
    return matched


def stage_diagnostics(rows, screened, verified):
    """Attribute event losses without sending labels to either inference stage."""
    metrics(rows, screened); metrics(rows, verified)  # Require complete coverage.
    report = []
    for row in rows:
        gt = [(e["t_start"], e["t_end"]) for e in row["events"]]
        stages = [match_events(union([(p["t_start"], p["t_end"]) for p in predictions[row["id"]]
                  if p.get("assessment") == "suspicious"]), gt) for predictions in (screened, verified)]
        cases = []
        for index, event in enumerate(row["events"]):
            before, after = index in stages[0], index in stages[1]
            cause = ("retained" if after and before else "boundary_recovered" if after
                     else "verification_loss" if before else "screening_miss")
            cases.append({**event, "screening_matched": before, "final_matched": after, "outcome": cause})
        report.append({"video_id": row["id"], "scene_id": row["scene_id"], "events": cases,
                       "screening": metrics([row], screened), "final": metrics([row], verified)})
    return {"recordings": report,
            "counts": {cause: sum(e["outcome"] == cause for r in report for e in r["events"])
                       for cause in ("retained", "boundary_recovered", "verification_loss", "screening_miss")},
            "interpretation": "Temporal IoU >= .3 with one-to-one matching; boundary changes can change matches. Not semantic error labels."}


def metrics(rows, predictions):
    """Maximum bipartite event matching, no duplicate credit. FA/h on normal-only clips."""
    missing = {row["id"] for row in rows} - predictions.keys()
    if missing:
        raise ValueError("Missing predictions for recordings: " + ", ".join(sorted(missing)))
    tp = fp = fn = normal_alerts = uncertain = 0
    normal_seconds = selected_seconds = duration = 0.
    for row in rows:
        records = predictions[row["id"]]
        uncertain += sum(p.get("assessment") == "uncertain" for p in records)
        pred = union([(p["t_start"], p["t_end"]) for p in records if p.get("assessment") == "suspicious"])
        if any(a < -1e-6 or b > row["duration"]+1e-6 for a, b in pred):
            raise ValueError("Prediction outside video")
        pred = [(max(0., a), min(row["duration"], b)) for a, b in pred]
        gt = [(e["t_start"], e["t_end"]) for e in row["events"]]
        hits = len(match_events(pred, gt))
        tp += hits; fp += len(pred)-hits; fn += len(gt)-hits
        duration += row["duration"]
        selected_seconds += sum(b-a for a, b in pred)
        if not gt:
            normal_alerts += len(pred); normal_seconds += row["duration"]
    return {"tp": tp, "fp": fp, "fn": fn, "recall": tp/(tp+fn) if tp+fn else None,
            "precision": tp/(tp+fp) if tp+fp else None, "uncertain_units": uncertain,
            "normal_only_hours": normal_seconds/3600,
            "false_alerts_per_hour": normal_alerts/(normal_seconds/3600) if normal_seconds else None,
            "positive_video_fraction": selected_seconds/duration if duration else None}


def bootstrap(rows, predictions, seed=73, repetitions=1000):
    rng = np.random.default_rng(seed)
    groups = {}
    for row in rows:
        groups.setdefault((row["dataset"], row["recording_id"]), []).append(row)
    units = list(groups.values())
    intervals = {k: [] for k in ("recall", "precision", "false_alerts_per_hour")}
    for _ in range(repetitions):
        sample = [row for i in rng.integers(0, len(units), len(units)) for row in units[i]] if units else []
        result = metrics(sample, predictions)
        for key in intervals:
            if result[key] is not None:
                intervals[key].append(result[key])
    return {k: np.quantile(v, [.025, .975]).tolist() if v else None for k, v in intervals.items()}


def freeze(manifest, thresholds, config, references, metrics_by_arm):
    rows = manifest["videos"]
    audit(rows)
    if not any(r["split"] == "validation" and r["events"] for r in rows):
        raise ValueError("No anomalous validation labels; cannot select thresholds")
    if any(r["split"] != "development" or r["events"] for r in references):
        raise ValueError("Only verified normal development recordings may be references")
    return {"policy": POLICY, "prompt_hash": digest(PROMPTS), "manifest_hash": digest(manifest),
            "reference_manifest_hash": digest(references), "config": config, "thresholds": thresholds,
            "source_ledger": rows,
            "split": "validation", "metrics_by_arm": metrics_by_arm, "test_opened": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--lock")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if args.test:
        if not args.lock:
            raise ValueError("Test requires a frozen lock")
        lock = json.loads(Path(args.lock).read_text(encoding="utf-8"))
        if lock["policy"] != POLICY or lock["prompt_hash"] != digest(PROMPTS) or lock["split"] != "validation":
            raise ValueError("Validation lock mismatch")
        if manifest.get("development_lock_hash") != digest(lock):
            raise ValueError("Test manifest is not bound to the frozen development lock")
        if not lock.get("source_ledger"):
            raise ValueError("Validation source ledger required")
        audit(lock["source_ledger"] + manifest["videos"], allow_test=True, check_files=False)
    audit(manifest["videos"], allow_test=args.test)
    report = {"recordings": len(manifest["videos"]), "manifest_hash": digest(manifest), "test_opened": args.test}
    if args.predictions:
        predictions = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
        rows = [r for r in manifest["videos"] if r["split"] == ("test" if args.test else "validation")]
        report["arms"] = {arm: {"metrics": metrics(rows, results), "video_bootstrap_95ci": bootstrap(rows, results)}
                          for arm, results in predictions.items()}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
