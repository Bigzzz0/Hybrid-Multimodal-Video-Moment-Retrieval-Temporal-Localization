"""Resumable VLM A--E caption artifact backfill.

Examples (run from backend):
  python -m scripts.backfill_vlm_ablation --all-videos --all-backends --resume
  python -m scripts.backfill_vlm_ablation --video-id <ID> --backend caprl_qwen3vl_2b --resume
"""

from __future__ import annotations

import argparse

from app.db.connection import db_manager
from app.inference.vlm_registry import get_variant
from app.pipeline.ingestion_manager import ingestion_manager


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-videos", action="store_true")
    parser.add_argument("--video-id", default="")
    parser.add_argument("--all-backends", action="store_true")
    parser.add_argument("--backend", default="")
    parser.add_argument("--resume", action="store_true", help="Keep completed rows and continue missing artifacts")
    args = parser.parse_args()
    if args.backend:
        get_variant(args.backend)
    video_ids = [args.video_id] if args.video_id else []
    if args.all_videos:
        video_ids = [str(row.get("id")) for row in db_manager.get_table("videos").to_arrow().to_pylist()]
    if not video_ids:
        parser.error("provide --video-id or --all-videos")
    # The ingestion method is resumable by artifact id. A single backend is
    # useful for rerunning a failed variant; --all-backends processes A--E.
    backend_filter = args.backend if args.backend and not args.all_backends else None
    for video_id in video_ids:
        ingestion_manager.process_video_vlm_ablation_background(video_id, backend_filter=backend_filter)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

