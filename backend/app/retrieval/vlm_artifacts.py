"""Additive storage helpers for the VLM A--E ablation artifacts."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.db.connection import db_manager
from app.inference.vlm_registry import VLMVariant, get_variant


def _now() -> str:
    return datetime.now().isoformat()


class VLMArtifactStore:
    @staticmethod
    def metadata_id(video_id: str, backend: str) -> str:
        variant = get_variant(backend)
        return f"{video_id}:{variant.backend}:{variant.revision}:{settings.VLM_ARTIFACT_VERSION}"

    @staticmethod
    def _rows(table_name: str, where: Optional[str] = None, limit: int = 10000) -> List[Dict[str, Any]]:
        try:
            table = db_manager.get_table(table_name)
            search = table.search()
            if where:
                search = search.where(where)
            return search.limit(limit).to_list()
        except Exception:
            try:
                rows = db_manager.get_table(table_name).to_arrow().to_pylist()
                if where and " = '" in where:
                    key, value = where.split(" = '", 1)
                    rows = [row for row in rows if str(row.get(key)) == value.rstrip("'")]
                return rows[:limit]
            except Exception:
                return []

    @classmethod
    def metadata(cls, video_id: str, backend: str) -> Optional[Dict[str, Any]]:
        key = cls.metadata_id(video_id, backend)
        rows = cls._rows("vlm_artifact_metadata_v1", f"id = '{key}'", 1)
        return rows[0] if rows else None

    @classmethod
    def is_ready(cls, video_id: str, backend: str) -> bool:
        row = cls.metadata(video_id, backend)
        return bool(row and row.get("status") == "ready")

    @classmethod
    def set_metadata(cls, video_id: str, variant: VLMVariant, expected_count: int, completed_count: int, status: str, error_message: str = "", started_at: str = "", completed_at: str = "") -> None:
        table = db_manager.get_table("vlm_artifact_metadata_v1")
        key = cls.metadata_id(video_id, variant.backend)
        try:
            table.delete(f"id = '{key}'")
        except Exception:
            pass
        table.add([{
            "id": key,
            "video_id": video_id,
            "vlm_backend": variant.backend,
            "model_id": variant.model_id,
            "model_revision": variant.revision,
            "expected_count": int(expected_count),
            "completed_count": int(completed_count),
            "status": status,
            "error_message": error_message,
            "started_at": started_at or _now(),
            "completed_at": completed_at,
        }])

    @classmethod
    def save_caption(cls, video_id: str, variant: VLMVariant, source_kind: str, source_id: str, t_start: float, t_end: float, timestamps: List[float], response: Any, prompt_version: str) -> str:
        key = ":".join([video_id, variant.backend, source_kind, source_id, settings.VLM_ARTIFACT_VERSION])
        table = db_manager.get_table("vlm_caption_artifacts_v1")
        try:
            table.delete(f"id = '{key}'")
        except Exception:
            pass
        structured = {
            "summary": getattr(response, "summary", ""),
            "objects": list(getattr(response, "objects", []) or []),
            "attributes": list(getattr(response, "attributes", []) or []),
            "actions": list(getattr(response, "actions", []) or []),
            "relations": list(getattr(response, "relations", []) or []),
            "temporal_events": list(getattr(response, "temporal_events", []) or []),
            "uncertainty": list(getattr(response, "uncertainty", []) or []),
        }
        now = _now()
        table.add([{
            "id": key, "video_id": video_id, "source_kind": source_kind, "source_id": source_id,
            "t_start": float(t_start), "t_end": float(t_end),
            "sampled_timestamps": [float(x) for x in timestamps],
            "caption_text": str(getattr(response, "text", "") or getattr(response, "summary", "")),
            "structured_json": json.dumps(structured, ensure_ascii=False),
            "vlm_backend": variant.backend, "model_id": variant.model_id, "model_revision": variant.revision,
            "quantization": variant.quantization, "input_mode": variant.input_mode,
            "artifact_version": settings.VLM_ARTIFACT_VERSION, "prompt_version": prompt_version,
            "status": "ready", "load_ms": float(getattr(response, "load_ms", 0.0) or 0.0),
            "inference_ms": float(getattr(response, "inference_ms", 0.0) or 0.0),
            "created_at": now, "last_accessed_at": now,
        }])
        return key

    @classmethod
    def caption_rows(cls, video_id: str, backend: str) -> List[Dict[str, Any]]:
        variant = get_variant(backend)
        rows = cls._rows("vlm_caption_artifacts_v1", f"video_id = '{video_id}'", 200000)
        output = []
        for row in rows:
            if row.get("vlm_backend") != variant.backend or row.get("model_revision") != variant.revision or row.get("artifact_version") != settings.VLM_ARTIFACT_VERSION or row.get("status") != "ready":
                continue
            output.append({
                "id": str(row.get("source_id", row.get("id", ""))),
                "video_id": video_id,
                "caption": str(row.get("caption_text", "")),
                "caption_status": "generated",
                "source_kind": row.get("source_kind", "scene"),
                "t_start": float(row.get("t_start", 0.0) or 0.0),
                "t_end": float(row.get("t_end", 0.0) or 0.0),
            })
        return output


vlm_artifact_store = VLMArtifactStore()
