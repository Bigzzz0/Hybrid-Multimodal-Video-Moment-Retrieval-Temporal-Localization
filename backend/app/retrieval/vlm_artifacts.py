"""Versioned storage helpers for production CapRL Q6 caption artifacts."""

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
    def _add(table: Any, row: Dict[str, Any]) -> None:
        """Write only columns supported by an older additive table schema."""
        try:
            names = set(table.schema.names)
            row = {key: value for key, value in row.items() if key in names}
        except Exception:
            pass
        table.add([row])

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
    def set_metadata(
        cls,
        video_id: str,
        variant: VLMVariant,
        expected_count: int,
        completed_count: int,
        status: str,
        error_message: str = "",
        started_at: str = "",
        completed_at: str = "",
        fallback_count: int = 0,
        active_artifact_version: Optional[str] = None,
        pending_artifact_version: Optional[str] = None,
        build_status: Optional[str] = None,
    ) -> None:
        table = db_manager.get_table("vlm_artifact_metadata_v1")
        key = cls.metadata_id(video_id, variant.backend)
        try:
            table.delete(f"id = '{key}'")
        except Exception:
            pass
        active_version = active_artifact_version
        if active_version is None:
            active_version = settings.VLM_ARTIFACT_VERSION if status == "ready" else ""
        cls._add(table, {
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
            "active_artifact_version": active_version,
            "pending_artifact_version": pending_artifact_version or "",
            "build_status": build_status or status,
            "fallback_count": int(fallback_count),
        })

    @classmethod
    def save_caption(cls, video_id: str, variant: VLMVariant, source_kind: str, source_id: str, t_start: float, t_end: float, timestamps: List[float], response: Any, prompt_version: str, artifact_version: Optional[str] = None) -> str:
        version = artifact_version or settings.VLM_ARTIFACT_VERSION
        key = ":".join([video_id, variant.backend, source_kind, source_id, version])
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
        cls._add(table, {
            "id": key, "video_id": video_id, "source_kind": source_kind, "source_id": source_id,
            "t_start": float(t_start), "t_end": float(t_end),
            "sampled_timestamps": [float(x) for x in timestamps],
            "caption_text": str(getattr(response, "text", "") or getattr(response, "summary", "")),
            "structured_json": json.dumps(structured, ensure_ascii=False),
            "vlm_backend": variant.backend, "model_id": variant.model_id, "model_revision": variant.revision,
            "quantization": variant.quantization, "input_mode": variant.input_mode,
            "artifact_version": version, "prompt_version": prompt_version,
            "status": "ready", "load_ms": float(getattr(response, "load_ms", 0.0) or 0.0),
            "inference_ms": float(getattr(response, "inference_ms", 0.0) or 0.0),
            "json_valid": bool(getattr(response, "json_valid", False)),
            "fallback_used": bool(settings.CAPTION_FALLBACK_BACKEND) and variant.backend == settings.CAPTION_FALLBACK_BACKEND,
            "created_at": now, "last_accessed_at": now,
        })
        return key

    @classmethod
    def caption_rows(cls, video_id: str, backend: str, require_ready: bool = True, artifact_version: Optional[str] = None) -> List[Dict[str, Any]]:
        variant = get_variant(backend)
        # Rows are written scene-by-scene so interrupted jobs can resume, but
        # retrieval must only see a version after the complete set has been
        # atomically marked ready.  Otherwise a partial caption run could
        # silently change Fast ranking while Phase 2 is still running.
        metadata = cls.metadata(video_id, variant.backend)
        active_version = ""
        if metadata:
            active_version = str(metadata.get("active_artifact_version") or "")
            if not active_version and metadata.get("status") == "ready":
                # Compatibility with the first additive schema revision.
                active_version = settings.VLM_ARTIFACT_VERSION
        if require_ready and not active_version:
            return []
        selected_version = artifact_version or active_version
        if not require_ready and not selected_version and metadata:
            selected_version = str(metadata.get("pending_artifact_version") or "")
        if not selected_version:
            selected_version = settings.VLM_ARTIFACT_VERSION
        fallback_backend = str(settings.CAPTION_FALLBACK_BACKEND or "").strip()
        rows = cls._rows("vlm_caption_artifacts_v1", f"video_id = '{video_id}'", 200000)
        output = []
        for row in rows:
            row_backend = str(row.get("vlm_backend", ""))
            allowed = {variant.backend}
            if fallback_backend and fallback_backend != variant.backend:
                allowed.add(fallback_backend)
            if row_backend not in allowed or row.get("artifact_version") != selected_version or row.get("status") != "ready":
                continue
            if row_backend == variant.backend and row.get("model_revision") != variant.revision:
                continue
            if fallback_backend and row_backend == fallback_backend and row.get("model_id") != settings.QWEN_VL_MODEL_ID:
                continue
            output.append({
                "id": str(row.get("source_id", row.get("id", ""))),
                "video_id": video_id,
                "caption": str(row.get("caption_text", "")),
                "caption_status": "generated",
                "source_kind": row.get("source_kind", "scene"),
                "t_start": float(row.get("t_start", 0.0) or 0.0),
                "t_end": float(row.get("t_end", 0.0) or 0.0),
                "vlm_backend": row_backend,
            })
        return output


vlm_artifact_store = VLMArtifactStore()
