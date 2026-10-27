"""Audit annotated source videos; never infer labels from generated captions."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

KINDS = ("shouting", "pushing_punching", "falling")

def validate(manifest):
    errors = []
    splits = defaultdict(set)
    hashes = defaultdict(set)
    counts = {split: {kind: {"positive": 0, "hard_negative": 0} for kind in KINDS} for split in ("development", "validation", "test")}
    normal_minutes = {"validation": 0., "test": 0.}
    ids = set()
    for video in manifest["videos"]:
        identifier, split = video["id"], video["split"]
        if identifier in ids:
            errors.append(f"Duplicate video id: {identifier}")
        ids.add(identifier)
        if split not in counts:
            errors.append(f"Invalid split for {identifier}")
            continue
        splits[video["recording_id"]].add(split)
        if video.get("sha256"):
            hashes[video["sha256"]].add(split)
        if not video.get("source_url") and not video.get("source_description"):
            errors.append(f"Missing provenance for {identifier}")
        if not Path(video["path"]).is_file():
            errors.append(f"Media unavailable: {identifier}")
        if video.get("annotation_review") == "pending":
            errors.append(f"Annotation review pending: {identifier}")
        if video.get("continuous_normal") and split in normal_minutes:
            normal_minutes[split] += video["duration"] / 60
        for event in video.get("events", []):
            kind = event["kind"]
            if kind not in KINDS or not 0 <= event["t_start"] < event["t_end"] <= video["duration"]:
                errors.append(f"Invalid event annotation for {identifier}")
                continue
            counts[split][kind]["positive"] += 1
        for kind in video.get("hard_negatives", []):
            counts[split][kind]["hard_negative"] += 1
        if video.get("audio_visual_pairing") == "unrelated":
            errors.append(f"Unrelated audio is not AV ground truth: {identifier}")
    for key, values in {**splits, **hashes}.items():
        if len(values) > 1:
            errors.append(f"Source leakage across splits: {key}")
    for split, minimum in (("development", 18), ("validation", 6), ("test", 6)):
        for kind in KINDS:
            for label in ("positive", "hard_negative"):
                if counts[split][kind][label] < minimum:
                    errors.append(f"Need {minimum-counts[split][kind][label]} more {split}/{kind}/{label}")
    for split, minimum in (("validation", 30), ("test", 60)):
        if normal_minutes[split] < minimum:
            errors.append(f"Need {minimum-normal_minutes[split]:.1f} more continuous normal minutes in {split}")
    return {"ready": not errors, "errors": errors, "counts": counts, "normal_minutes": normal_minutes, "videos": len(ids)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    result = validate(json.loads(args.manifest.read_text(encoding="utf-8")))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["ready"] else 2)

if __name__ == "__main__":
    main()
