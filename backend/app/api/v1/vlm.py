from __future__ import annotations

import threading
from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.db.connection import db_manager
from app.inference.vlm_registry import get_variant
from app.pipeline.ingestion_manager import ingestion_manager

router = APIRouter()


@router.post("/videos/{video_id}/vlm-artifacts/rebuild")
def rebuild_vlm_artifacts(video_id: str):
    rows = db_manager.get_table("videos").search().where(f"id = '{video_id}'").limit(1).to_list()
    if not rows:
        raise HTTPException(status_code=404, detail="video not found")
    thread = threading.Thread(
        target=ingestion_manager.process_video_vlm_ablation_background,
        kwargs={"video_id": video_id},
        daemon=True,
    )
    thread.start()
    return {"status": "started", "video_id": video_id, "message": "VLM A–E artifact rebuild started in background."}

