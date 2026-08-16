import uuid
import shutil
import threading
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from app.core.config import settings
from app.core.logger import logger
from app.pipeline.ingestion_manager import ingestion_manager
from app.api.v1.websocket import ws_manager

router = APIRouter()

def sync_progress_adapter(video_id: str, percent: int, msg: str, stage: str = "decoding", phase: str = "phase1", details: dict = None):
    """Adapter to record and broadcast progress from pipeline worker thread."""
    logger.info(f"[{video_id}] {percent}% | Stage: {stage} | {msg}")
    ws_manager.update_progress(
        video_id=video_id,
        progress_percent=percent,
        message=msg,
        stage=stage,
        phase=phase,
        stage_details=details
    )

def run_progressive_pipeline(video_id: str, file_path: str, filename: str):
    """Dedicated Background Worker Thread for Progressive Video Ingestion."""
    try:
        # Initial broadcast
        sync_progress_adapter(video_id, 10, "Starting Video Decoding & Scene Analysis...", "decoding", "phase1")

        # Phase 1: Fast Ingestion (~45s)
        ingestion_manager.process_video_phase1(
            video_id=video_id,
            video_path=file_path,
            filename=filename,
            progress_callback=lambda vid, pct, msg, stg, det: sync_progress_adapter(vid, pct, msg, stg, "phase1", det)
        )

        # Phase 2: Deep Action Captioning (Background)
        ingestion_manager.process_video_phase2_background(
            video_id=video_id,
            progress_callback=lambda vid, pct, msg, stg, det: sync_progress_adapter(vid, pct, msg, stg, "phase2", det)
        )
    except Exception as e:
        logger.error(f"Error in progressive pipeline for {video_id}: {e}")
        sync_progress_adapter(video_id, 100, f"Error: {str(e)}", "error", "error")

@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Uploads a video file (.mp4, .mkv, .mov, .webm) and launches background ingestion in a worker thread.
    """
    allowed_extensions = {".mp4", ".mkv", ".mov", ".webm", ".avi"}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{file_ext}'. Allowed formats: {allowed_extensions}"
        )

    video_id = str(uuid.uuid4())
    saved_filename = f"{video_id}{file_ext}"
    saved_path = settings.RAW_VIDEOS_DIR / saved_filename

    logger.info(f"Receiving upload: {file.filename} -> {saved_path}")

    # Set initial progress state
    ws_manager.update_progress(video_id, 5, "Saving video file to server...", "decoding", "phase1")

    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Launch dedicated daemon thread to prevent blocking asyncio event loop
    worker_thread = threading.Thread(
        target=run_progressive_pipeline,
        args=(video_id, str(saved_path), file.filename),
        daemon=True
    )
    worker_thread.start()

    return {
        "status": "upload_success",
        "video_id": video_id,
        "filename": file.filename,
        "message": "Video uploaded successfully. Progressive ingestion started.",
        "websocket_url": f"/ws/progress/{video_id}",
        "status_url": f"/api/v1/ws/progress/{video_id}/status"
    }
