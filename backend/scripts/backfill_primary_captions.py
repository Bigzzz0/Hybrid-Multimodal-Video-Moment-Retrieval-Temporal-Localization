"""Resume-safe production CapRL Q6 scene-caption backfill.

Run from backend:
  python -m scripts.backfill_primary_captions --all-videos --resume
  python -m scripts.backfill_primary_captions --video-id <ID> --resume
"""

from __future__ import annotations

import argparse

from app.db.connection import db_manager
from app.pipeline.ingestion_manager import ingestion_manager


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-videos", action="store_true")
    parser.add_argument("--video-id", default="")
    parser.add_argument("--resume", action="store_true", help="Keep completed artifacts and continue missing scenes")
    args = parser.parse_args()

    video_ids = [args.video_id] if args.video_id else []
    if args.all_videos:
        video_ids = [str(row.get("id")) for row in db_manager.get_table("videos").to_arrow().to_pylist()]
    if not video_ids:
        parser.error("provide --video-id or --all-videos")

    for video_id in video_ids:
        ingestion_manager.process_video_primary_captions(video_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
