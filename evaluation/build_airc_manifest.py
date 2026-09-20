"""Build a deterministic, video-disjoint AIRC-SMARTCLASS manifest.

The script only records file metadata and split membership.  It intentionally
does not inspect or emit face, identity, or Re-ID labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".webm"}


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _media_metadata(path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {"duration": 0.0, "fps": 0.0, "resolution": ""}
    try:
        import cv2
        capture = cv2.VideoCapture(str(path))
        metadata["fps"] = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        if metadata["fps"] > 0:
            metadata["duration"] = frames / metadata["fps"]
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        metadata["resolution"] = f"{width}x{height}" if width and height else ""
        capture.release()
    except Exception:
        # Metadata can be filled later with ffprobe without changing split IDs.
        pass
    return metadata


def build_manifest(source_dir: Path, output: Path, seed: int = 20260920) -> dict[str, Any]:
    files = sorted(
        [path for path in source_dir.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS],
        key=lambda path: str(path.relative_to(source_dir)).casefold(),
    )
    if len(files) != 45:
        raise ValueError(f"AIRC-SMARTCLASS split requires exactly 45 videos; found {len(files)}")

    # Stable pseudo-shuffle independent of platform locale and filesystem order.
    files.sort(key=lambda path: hashlib.sha256(f"{seed}|{path.name}".encode("utf-8")).hexdigest())
    rows = []
    for index, path in enumerate(files):
        split = "development" if index < 27 else "validation" if index < 36 else "test"
        metadata = _media_metadata(path)
        rows.append({
            "dataset_id": "AIRC-SMARTCLASS-part2",
            "video_id": f"airc-{index + 1:03d}",
            "source_path": str(path.resolve()),
            "sha256": _sha256(path),
            "duration": round(float(metadata["duration"]), 3),
            "fps": round(float(metadata["fps"]), 3),
            "resolution": metadata["resolution"],
            "split": split,
            "license": "see AIRC-SMARTCLASS source terms",
            "doi": "10.17632/fw5hs57z78.1",
        })
    manifest = {
        "dataset_id": "AIRC-SMARTCLASS-part2",
        "source": "https://data.mendeley.com/datasets/fw5hs57z78/1",
        "seed": seed,
        "splits": {
            "development": [row["video_id"] for row in rows if row["split"] == "development"],
            "validation": [row["video_id"] for row in rows if row["split"] == "validation"],
            "test": [row["video_id"] for row in rows if row["split"] == "test"],
        },
        "videos": rows,
        "annotation_policy": {
            "identity_labels": False,
            "face_recognition": False,
            "reid": False,
            "person_boxes": "full-body only",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    result = build_manifest(args.source_dir, args.output, args.seed)
    print(json.dumps({"output": str(args.output), "videos": len(result["videos"]), "splits": {key: len(value) for key, value in result["splits"].items()}}, ensure_ascii=False))
