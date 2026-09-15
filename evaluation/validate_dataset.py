"""Validate the video-disjoint held-out benchmark contract.

The checked-in two-video fixture is intentionally a smoke/regression set.  Use
this command against the acceptance annotation file before reporting metrics;
it fails closed when the required query count, language balance, or split
separation is missing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def _iter_queries(data: dict[str, Any]) -> Iterable[tuple[dict[str, Any], dict[str, Any], str]]:
    for video in data.get("videos", []):
        for event in video.get("events", []):
            for query in event.get("queries", []):
                yield video, event, str(query)


def validate_dataset(path: Path, min_queries: int = 100, min_videos: int = 10) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    videos = data.get("videos", [])
    if not isinstance(videos, list):
        raise ValueError("videos must be a list")

    video_ids = {str(video.get("video_id")) for video in videos}
    if len(video_ids) < min_videos:
        raise ValueError(f"acceptance set needs at least {min_videos} unique videos (found {len(video_ids)})")

    query_rows = list(_iter_queries(data))
    if len(query_rows) < min_queries:
        raise ValueError(f"acceptance set needs at least {min_queries} queries (found {len(query_rows)})")

    thai = sum(any("\u0e00" <= char <= "\u0e7f" for char in query) for _, _, query in query_rows)
    english = len(query_rows) - thai
    if thai == 0 or english == 0:
        raise ValueError("acceptance set must contain both Thai and English queries")
    language_ratio = thai / len(query_rows)
    if not 0.30 <= language_ratio <= 0.70:
        raise ValueError(f"Thai/English balance is outside the 30/70 tolerance (Thai ratio={language_ratio:.3f})")

    # A split can be set per video or at the root as {train/dev/test: [...]};
    # the validator accepts either representation and rejects video overlap.
    split_by_video: dict[str, str] = {}
    for video in videos:
        video_id = str(video.get("video_id"))
        split = video.get("split")
        if split:
            split_by_video[video_id] = str(split)
    for split, ids in (data.get("splits") or {}).items():
        for video_id in ids:
            video_id = str(video_id)
            previous = split_by_video.setdefault(video_id, str(split))
            if previous != str(split):
                raise ValueError(f"video {video_id} appears in multiple splits")
    if not split_by_video:
        raise ValueError("acceptance set must label every video with a train/dev/test split")
    if set(split_by_video) != video_ids:
        missing = sorted(video_ids - set(split_by_video))
        raise ValueError(f"every video needs a split label (missing: {missing})")
    if split_by_video:
        split_sets: dict[str, set[str]] = {}
        for video_id, split in split_by_video.items():
            split_sets.setdefault(split, set()).add(video_id)
        split_names = sorted(split_sets)
        for index, left in enumerate(split_names):
            for right in split_names[index + 1 :]:
                overlap = split_sets[left] & split_sets[right]
                if overlap:
                    raise ValueError(f"video overlap between {left} and {right}: {sorted(overlap)}")

    no_match = 0
    multi_event = 0
    for video, event, _ in query_rows:
        truths = event.get("ground_truths")
        if truths is not None:
            if len(truths) == 0:
                no_match += 1
            if len(truths) > 1:
                multi_event += 1
        elif float(event.get("t_end", 0.0)) <= float(event.get("t_start", 0.0)):
            no_match += 1

    if no_match == 0:
        raise ValueError("acceptance set must include negative/no-match queries")
    return {
        "path": str(path),
        "query_count": len(query_rows),
        "video_count": len(video_ids),
        "thai_queries": thai,
        "english_queries": english,
        "no_match_events": no_match,
        "multi_interval_events": multi_event,
        "video_disjoint_splits": bool(split_by_video),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a held-out pure-visual benchmark dataset")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--min-queries", type=int, default=100)
    parser.add_argument("--min-videos", type=int, default=10)
    args = parser.parse_args()
    result = validate_dataset(args.dataset, args.min_queries, args.min_videos)
    print(json.dumps(result, indent=2, ensure_ascii=False))
