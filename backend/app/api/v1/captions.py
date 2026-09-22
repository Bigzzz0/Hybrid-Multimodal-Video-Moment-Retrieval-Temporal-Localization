from __future__ import annotations

import threading
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Query

from app.core.config import settings
from app.db.connection import db_manager
from app.inference.client import inference_client
from app.inference.vlm_registry import get_variant
from app.pipeline.ingestion_manager import ingestion_manager
from app.retrieval.vlm_artifacts import vlm_artifact_store

router = APIRouter()


def _video_exists(video_id: str) -> bool:
    rows = db_manager.get_table("videos").search().where(f"id = '{video_id}'").limit(1).to_list()
    return bool(rows)


@router.get("/{video_id}/caption-status")
def caption_status(video_id: str) -> Dict[str, Any]:
    if not _video_exists(video_id):
        raise HTTPException(status_code=404, detail="video not found")
    primary = get_variant(settings.CAPTION_PRIMARY_BACKEND)
    fallback = get_variant(settings.CAPTION_FALLBACK_BACKEND)
    metadata = vlm_artifact_store.metadata(video_id, primary.backend) or {}
    return {
        "video_id": video_id,
        "status": metadata.get("build_status", metadata.get("status", "unavailable")),
        "expected_count": int(metadata.get("expected_count", 0) or 0),
        "completed_count": int(metadata.get("completed_count", 0) or 0),
        "fallback_count": int(metadata.get("fallback_count", 0) or 0),
        "active_backend": primary.backend,
        "active_model_id": primary.model_id,
        "fallback_backend": fallback.backend,
        "artifact_version": settings.CAPTION_ARTIFACT_VERSION,
        "worker": inference_client.health(),
        "warnings": [metadata["error_message"]] if metadata.get("error_message") else [],
    }


@router.post("/{video_id}/captions/rebuild")
def rebuild_captions(video_id: str, force: bool = Query(default=False)) -> Dict[str, Any]:
    if not _video_exists(video_id):
        raise HTTPException(status_code=404, detail="video not found")
    # Rebuild is resume-safe by default. force is accepted for API
    # compatibility but still preserves the active artifact version until the
    # new version is complete.
    thread = threading.Thread(
        target=ingestion_manager.process_video_primary_captions,
        kwargs={"video_id": video_id},
        daemon=True,
    )
    thread.start()
    return {
        "status": "started",
        "video_id": video_id,
        "resume": not force,
        "message": "Primary Q6 caption rebuild started in background.",
    }
