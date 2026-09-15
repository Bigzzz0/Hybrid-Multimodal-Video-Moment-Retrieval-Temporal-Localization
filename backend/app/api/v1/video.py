import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Request, HTTPException, status, Query
from fastapi.responses import FileResponse
from app.core.config import settings
from app.core.logger import logger
from app.db.connection import db_manager
from app.db.schemas import VideoMetadata
from app.utils.video_stream import range_requests_response

router = APIRouter()

def _get_ffmpeg_bin() -> str:
    """Finds FFmpeg binary from PATH or fallback to imageio_ffmpeg bundled binary."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"

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
            visual_index_version=row.get("visual_index_version"),
            embedding_model=row.get("embedding_model"),
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

@router.get("/{video_id}/keyframes")
async def get_video_keyframes(video_id: str):
    """
    Returns sorted list of extracted keyframes with timestamps for UI timeline scrubbing.
    """
    tbl_frames = db_manager.get_table("video_frames_v2")
    try:
        matches = tbl_frames.search().where(f"video_id = '{video_id}'").limit(500).to_list()
    except Exception:
        matches = [r for r in tbl_frames.to_arrow().to_pylist() if r.get("video_id") == video_id]

    keyframes = []
    for r in sorted(matches, key=lambda x: float(x.get("timestamp", 0.0))):
        frame_path = r.get("frame_path", "")
        keyframes.append({
            "timestamp": float(r.get("timestamp", 0.0)),
            "frame_path": frame_path,
            "thumbnail_url": f"/api/v1/videos/frame-preview?path={frame_path}"
        })
    return keyframes


@router.get("/download-clip/{clip_filename}")
async def download_video_clip(clip_filename: str):
    """Serves clipped video file for direct download."""
    clip_path = settings.DATA_DIR / "clips" / clip_filename
    if not clip_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Clip file not found.")
    return FileResponse(str(clip_path), media_type="video/mp4", filename=clip_filename)

@router.post("/{video_id}/cut-clip")
async def cut_video_clip(
    video_id: str,
    t_start: float = Query(...),
    t_end: float = Query(...)
):
    """Cut a visual moment using a fast stream copy."""
    tbl_videos = db_manager.get_table("videos")
    try:
        matches = tbl_videos.search().where(f"id = '{video_id}'").limit(1).to_list()
    except Exception:
        matches = [r for r in tbl_videos.to_arrow().to_pylist() if r.get("id") == video_id]

    if not matches:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found.")

    src_path = matches[0]["filepath"]
    clip_filename = f"moment_{video_id[:8]}_{t_start:.1f}_{t_end:.1f}.mp4"
    out_dir = settings.DATA_DIR / "clips"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / clip_filename
    duration = max(0.5, t_end - t_start)

    # Execute FFmpeg stream copy.
    try:
        ffmpeg_bin = _get_ffmpeg_bin()
        cmd = [ffmpeg_bin, "-y", "-ss", str(t_start), "-i", src_path,
               "-t", str(duration), "-c", "copy", str(out_path)]

        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        return {
            "status": "success",
            "clip_path": str(out_path),
            "clip_filename": clip_filename,
            "download_url": f"/api/v1/videos/download-clip/{clip_filename}",
        }
    except Exception as e:
        logger.error(f"FFmpeg clipping failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export clip: {str(e)}")

@router.delete("/{video_id}")
async def delete_video(video_id: str):
    """
    Cascade deletes video metadata, visual embeddings, scene records, and physical files.
    """
    tbl_videos = db_manager.get_table("videos")
    try:
        matches = tbl_videos.search().where(f"id = '{video_id}'").limit(1).to_list()
    except Exception:
        matches = [r for r in tbl_videos.to_arrow().to_pylist() if r.get("id") == video_id]

    if not matches:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found.")

    video_record = matches[0]
    filename = str(video_record.get("filename", "video"))
    filepath = str(video_record.get("filepath", ""))

    # 1. Delete from LanceDB tables
    tables_to_clean = [
        ("videos", f"id = '{video_id}'"),
        ("video_frames_v2", f"video_id = '{video_id}'"),
        ("scenes_v2", f"video_id = '{video_id}'"),
        ("search_logs", f"video_id = '{video_id}'"),
    ]

    for tbl_name, filter_expr in tables_to_clean:
        try:
            tbl = db_manager.get_table(tbl_name)
            tbl.delete(filter_expr)
        except Exception as e:
            logger.warning(f"Notice while deleting from {tbl_name}: {e}")

    try:
        db_manager.get_index_metadata_table().delete(f"id = '{video_id}'")
    except Exception as e:
        logger.warning(f"Notice while deleting index metadata: {e}")

    # 2. Delete raw video file
    if filepath and os.path.exists(filepath):
        try:
            os.remove(filepath)
            logger.info(f"Removed raw video file: {filepath}")
        except Exception as e:
            logger.warning(f"Could not remove raw video {filepath}: {e}")
    else:
        for ext in [".mp4", ".mkv", ".mov", ".webm", ".avi"]:
            possible_path = settings.RAW_VIDEOS_DIR / f"{video_id}{ext}"
            if possible_path.exists():
                try:
                    os.remove(possible_path)
                except Exception:
                    pass

    # 3. Delete keyframe directory
    keyframe_dir = settings.KEYFRAMES_DIR / video_id
    if keyframe_dir.exists():
        try:
            shutil.rmtree(keyframe_dir, ignore_errors=True)
            logger.info(f"Removed keyframes folder: {keyframe_dir}")
        except Exception as e:
            logger.warning(f"Could not remove keyframes {keyframe_dir}: {e}")

    # 4. Clean up any exported clips for this video
    clips_dir = settings.DATA_DIR / "clips"
    if clips_dir.exists():
        prefix = f"moment_{video_id[:8]}_"
        for clip_file in clips_dir.glob(f"{prefix}*"):
            try:
                os.remove(clip_file)
            except Exception:
                pass

    logger.info(f"Successfully deleted video {video_id} ('{filename}')")
    return {
        "status": "success",
        "message": f"Video '{filename}' and all associated indexed data removed successfully.",
        "deleted_id": video_id
    }

