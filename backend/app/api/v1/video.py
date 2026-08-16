import os
import subprocess
from pathlib import Path
from typing import List
from fastapi import APIRouter, Request, HTTPException, status, Query
from fastapi.responses import FileResponse
from app.core.config import settings
from app.core.logger import logger
from app.db.connection import db_manager
from app.db.schemas import VideoMetadata
from app.utils.video_stream import range_requests_response

router = APIRouter()

@router.get("/list", response_model=List[VideoMetadata])
async def list_videos():
    """Returns list of all indexed videos with metadata and ingestion phase."""
    tbl_videos = db_manager.get_table("videos")
    try:
        records_list = tbl_videos.to_arrow().to_pylist()
    except Exception:
        records_list = []

    if not records_list:
        return []

    records = []
    for row in records_list:
        records.append(VideoMetadata(
            id=str(row["id"]),
            filename=str(row["filename"]),
            filepath=str(row["filepath"]),
            duration_sec=float(row["duration_sec"]),
            fps=float(row["fps"]),
            resolution=str(row["resolution"]),
            total_frames=int(row["total_frames"]),
            ingestion_phase=str(row["ingestion_phase"]),
            created_at=str(row["created_at"])
        ))
    return records

@router.get("/{video_id}/stream")
async def stream_video(video_id: str, request: Request):
    """Streams video with HTTP 206 Byte-Range requests for instant HTML5 seek."""
    tbl_videos = db_manager.get_table("videos")
    try:
        matches = tbl_videos.search().where(f"id = '{video_id}'").limit(1).to_list()
    except Exception:
        matches = [r for r in tbl_videos.to_arrow().to_pylist() if r.get("id") == video_id]

    if not matches:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found.")

    video_path = matches[0]["filepath"]
    return range_requests_response(request, video_path)

@router.get("/frame-preview")
async def get_frame_preview(path: str = Query(...)):
    """Serves extracted keyframe image thumbnails."""
    if not os.path.exists(path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Keyframe image not found.")
    return FileResponse(path, media_type="image/jpeg")

@router.post("/{video_id}/cut-clip")
async def cut_video_clip(video_id: str, t_start: float = Query(...), t_end: float = Query(...)):
    """Fast stream-copy video clipping using FFmpeg."""
    tbl_videos = db_manager.get_table("videos")
    try:
        matches = tbl_videos.search().where(f"id = '{video_id}'").limit(1).to_list()
    except Exception:
        matches = [r for r in tbl_videos.to_arrow().to_pylist() if r.get("id") == video_id]

    if not matches:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found.")

    src_path = matches[0]["filepath"]
    clip_filename = f"clip_{video_id}_{t_start:.1f}_{t_end:.1f}.mp4"
    out_dir = settings.DATA_DIR / "clips"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / clip_filename

    duration = max(0.5, t_end - t_start)
    cmd = [
        "ffmpeg", "-y", "-ss", str(t_start), "-i", src_path,
        "-t", str(duration), "-c", "copy", str(out_path)
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return {"status": "success", "clip_path": str(out_path), "download_url": f"/api/v1/videos/download-clip/{clip_filename}"}
    except Exception as e:
        logger.error(f"FFmpeg cutting failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cut clip: {str(e)}")
