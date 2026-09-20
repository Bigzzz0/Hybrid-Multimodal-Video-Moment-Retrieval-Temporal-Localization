from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, status

from app.core.config import settings
from app.db.connection import db_manager

router = APIRouter()


def _nearest_observation(track_id: str, timestamp: float | None):
    table = db_manager.get_table("sam_observations_v1")
    try:
        rows = table.search().where(f"track_id = '{track_id}'").limit(1000).to_list()
    except Exception:
        rows = [row for row in table.to_arrow().to_pylist() if str(row.get("track_id")) == track_id]
    if not rows:
        return None
    return min(rows, key=lambda row: abs(float(row.get("timestamp", 0.0)) - float(timestamp or row.get("timestamp", 0.0))))


def _serialize_observation(row):
    path = Path(str(row.get("mask_artifact_path", ""))).resolve()
    root = settings.GROUNDING_ARTIFACTS_DIR.resolve()
    if root not in path.parents or not path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mask artifact not found")
    try:
        mask = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read mask artifact: {exc}") from exc
    return {
        "timestamp": float(row.get("timestamp", 0.0)),
        "concept": str(row.get("concept", "")),
        "bbox_xyxy": [float(value) for value in row.get("bbox_xyxy", [])],
        "score": float(row.get("score", 0.0)),
        "mask_rle": mask,
        "frame_width": int(row.get("frame_width", 0)),
        "frame_height": int(row.get("frame_height", 0)),
    }


@router.get("/track/{track_id}")
async def get_grounding_track(track_id: str, timestamp: float | None = Query(default=None)):
    row = _nearest_observation(track_id, timestamp)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Grounding track not found")
    return _serialize_observation(row)


@router.get("/{artifact_id}")
async def get_grounding_artifact(artifact_id: str, timestamp: float | None = Query(default=None)):
    """Return the nearest persisted SAM observation for a grounding artifact."""
    table = db_manager.get_table("sam_observations_v1")
    try:
        rows = table.search().where(f"id = '{artifact_id}'").limit(1).to_list()
    except Exception:
        rows = [row for row in table.to_arrow().to_pylist() if str(row.get("id")) == artifact_id]
    if not rows:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Grounding artifact not found")
    row = rows[0]
    return _serialize_observation(row)
