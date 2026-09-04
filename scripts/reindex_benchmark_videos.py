import sys
import os
from pathlib import Path

# Set up paths
root_dir = Path(__file__).resolve().parent.parent
backend_dir = root_dir / "backend"
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(backend_dir))

from app.core.config import settings
from app.core.logger import logger
from app.db.connection import db_manager
from app.pipeline.ingestion_manager import ingestion_manager

def reindex_benchmark_videos():
    benchmark_videos = [
        {
            "video_id": "benchmark_person_bicycle_car",
            "filename": "benchmark_person_bicycle_car.mp4",
            "video_path": str(settings.RAW_VIDEOS_DIR / "benchmark_person_bicycle_car.mp4")
        },
        {
            "video_id": "benchmark_classroom",
            "filename": "benchmark_classroom.mp4",
            "video_path": str(settings.RAW_VIDEOS_DIR / "benchmark_classroom.mp4")
        }
    ]

    tbl_videos = db_manager.get_table("videos")
    tbl_scenes = db_manager.get_table("scenes")
    tbl_frames = db_manager.get_table("video_frames")

    for bv in benchmark_videos:
        vid = bv["video_id"]
        vpath = bv["video_path"]
        fname = bv["filename"]

        print(f"\n--- Re-indexing Benchmark Video: {vid} ---")
        if not Path(vpath).exists():
            print(f"Error: Video file {vpath} does not exist!")
            continue

        # Delete existing entries
        try:
            tbl_videos.delete(f"id = '{vid}'")
            tbl_scenes.delete(f"video_id = '{vid}'")
            tbl_frames.delete(f"video_id = '{vid}'")
            print(f"Purged old entries for {vid}")
        except Exception as e:
            print(f"Warning during purge: {e}")

        # Ingest with new SOTA Dual-Scale & Pixel Motion Pipeline
        ingestion_manager.process_video_phase1(
            video_id=vid,
            video_path=vpath,
            filename=fname
        )
        print(f"Successfully re-indexed {vid} with SOTA Dual-Scale Hierarchy and Pixel Motion Energy!")

    print("\nRe-indexing Complete! Refreshing database indices...")
    db_manager.create_indices()
    print("Database indices successfully created!")

if __name__ == "__main__":
    reindex_benchmark_videos()
