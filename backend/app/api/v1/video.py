import os
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

def _format_srt_time(seconds: float) -> str:
    """Format seconds into SRT timestamp HH:MM:SS,mmm"""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"

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
    t_end: float = Query(...),
    burn_subtitles: bool = Query(default=False)
):
    """
    SOTA Video Exporter: Cuts highlight interval with optional hardcoded Thai/English subtitles.
    """
    tbl_videos = db_manager.get_table("videos")
    try:
        matches = tbl_videos.search().where(f"id = '{video_id}'").limit(1).to_list()
    except Exception:
        matches = [r for r in tbl_videos.to_arrow().to_pylist() if r.get("id") == video_id]

    if not matches:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found.")

    src_path = matches[0]["filepath"]
    sub_tag = "_sub" if burn_subtitles else ""
    clip_filename = f"moment_{video_id[:8]}_{t_start:.1f}_{t_end:.1f}{sub_tag}.mp4"
    out_dir = settings.DATA_DIR / "clips"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / clip_filename
    duration = max(0.5, t_end - t_start)

    # 1. Check Subtitle Burning
    srt_path = None
    if burn_subtitles:
        try:
            tbl_transcripts = db_manager.get_table("transcripts")
            transcripts = [r for r in tbl_transcripts.to_arrow().to_pylist() if r.get("video_id") == video_id]
            matched_subs = []
            for tr in transcripts:
                tr_s = float(tr.get("t_start", 0.0))
                tr_e = float(tr.get("t_end", 0.0))
                if tr_e >= t_start and tr_s <= t_end:
                    rel_s = max(0.0, tr_s - t_start)
                    rel_e = min(duration, tr_e - t_start)
                    matched_subs.append((rel_s, rel_e, tr.get("spoken_text", "")))

            if matched_subs:
                srt_path = out_dir / f"temp_{video_id[:8]}.srt"
                with open(srt_path, "w", encoding="utf-8") as f_srt:
                    for i, (s_s, s_e, text) in enumerate(matched_subs, 1):
                        f_srt.write(f"{i}\n")
                        f_srt.write(f"{_format_srt_time(s_s)} --> {_format_srt_time(s_e)}\n")
                        f_srt.write(f"{text.strip()}\n\n")
        except Exception as e:
            logger.debug(f"Subtitle extraction notice: {e}")

    # 2. Execute FFmpeg
    try:
        if burn_subtitles and srt_path and srt_path.exists():
            # Burn subtitles into video stream
            srt_escaped = str(srt_path).replace("\\", "/").replace(":", "\\:")
            vf_filter = f"subtitles='{srt_escaped}'"
            cmd = [
                "ffmpeg", "-y", "-ss", str(t_start), "-i", src_path,
                "-t", str(duration), "-vf", vf_filter,
                "-c:v", "libx264", "-c:a", "aac", "-preset", "fast", str(out_path)
            ]
        else:
            # Fast stream copy without re-encoding
            cmd = [
                "ffmpeg", "-y", "-ss", str(t_start), "-i", src_path,
                "-t", str(duration), "-c", "copy", str(out_path)
            ]

        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Cleanup temp srt
        if srt_path and srt_path.exists():
            try:
                os.remove(srt_path)
            except Exception:
                pass

        return {
            "status": "success",
            "clip_path": str(out_path),
            "clip_filename": clip_filename,
            "download_url": f"/api/v1/videos/download-clip/{clip_filename}",
            "burned_subtitles": burn_subtitles and (srt_path is not None)
        }
    except Exception as e:
        logger.error(f"FFmpeg clipping failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export clip: {str(e)}")
