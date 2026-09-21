"""Build the additive PE-Core frame index from existing v2 keyframes.

This command never rewrites ``video_frames_v2``. It is intentionally resumable:
rows already present for the selected model/version are skipped and metadata is
marked ready only after all source frames have been embedded.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List

from app.core.config import settings
from app.db.connection import db_manager
from app.inference.client import pe_worker_client
from app.inference.contracts import PEEmbedImagesRequest


MODEL_REVISIONS = {
    "PE-Core-B16-224": settings.PE_CORE_B16_REVISION,
    "PE-Core-L14-336": settings.PE_CORE_L14_REVISION,
}
MODEL_BATCHES = {
    "PE-Core-B16-224": settings.PE_CORE_BATCH_B16,
    "PE-Core-L14-336": settings.PE_CORE_BATCH_L14,
}


def _rows(table: Any, where: str | None = None, limit: int = 200000) -> List[Dict[str, Any]]:
    try:
        query = table.search()
        if where:
            query = query.where(where)
        return query.limit(limit).to_list()
    except Exception:
        rows = table.to_arrow().to_pylist()
        if where:
            predicates = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*'([^']*)'", where)
            for key, value in predicates:
                rows = [row for row in rows if str(row.get(key)) == value]
        return rows[:limit]


def _embedding_version(model_id: str) -> str:
    revision = MODEL_REVISIONS[model_id]
    return f"{settings.PE_CORE_INDEX_VERSION}:{model_id}:{revision[:12]}"


def _metadata_id(video_id: str, model_id: str, embedding_version: str) -> str:
    return f"{video_id}:{model_id}:{embedding_version}"


def _write_metadata(table: Any, row: Dict[str, Any]) -> None:
    try:
        table.delete(f"id = '{row['id']}'")
    except Exception:
        pass
    table.add([row])


def _source_frames(video_id: str) -> List[Dict[str, Any]]:
    rows = _rows(db_manager.get_table("video_frames_v2"), f"video_id = '{video_id}'")
    return [
        row for row in rows
        if row.get("frame_path") and Path(str(row["frame_path"])).exists()
    ]


def backfill_video(video_id: str, model_id: str) -> Dict[str, Any]:
    if model_id not in MODEL_REVISIONS:
        raise ValueError(f"unsupported model: {model_id}")
    source_rows = _source_frames(video_id)
    if not source_rows:
        raise RuntimeError(f"no existing v2 keyframes found for video {video_id}")

    frame_table = db_manager.get_table("video_frames_pe_core_v1")
    metadata_table = db_manager.get_table("pe_core_index_metadata_v1")
    version = _embedding_version(model_id)
    metadata_id = _metadata_id(video_id, model_id, version)
    existing = _rows(
        frame_table,
        f"video_id = '{video_id}' AND model_id = '{model_id}' AND embedding_version = '{version}'",
    )
    existing_ids = {str(row.get("id")) for row in existing}
    expected_ids = {f"{row['id']}:{version}" for row in source_rows}

    metadata = {
        "id": metadata_id,
        "video_id": video_id,
        "model_id": model_id,
        "model_revision": MODEL_REVISIONS[model_id],
        "embedding_dim": settings.PE_CORE_EMBEDDING_DIM,
        "embedding_version": version,
        "source_index_version": settings.VISUAL_INDEX_VERSION,
        "source_frame_count": len(source_rows),
        "indexed_frame_count": len(existing_ids & expected_ids),
        "status": "pending",
        "indexed_at": dt.datetime.now().isoformat(),
        "error_message": "",
    }
    _write_metadata(metadata_table, metadata)

    missing = [row for row in source_rows if f"{row['id']}:{version}" not in existing_ids]
    print(f"[{model_id}] {video_id}: {len(source_rows)} source frames, {len(missing)} missing")
    try:
        batch_size = max(1, int(MODEL_BATCHES[model_id]))
        for start in range(0, len(missing), batch_size):
            batch = missing[start:start + batch_size]
            response = pe_worker_client.embed_images(
                PEEmbedImagesRequest(
                    model_id=model_id,
                    frame_paths=[str(row["frame_path"]) for row in batch],
                    batch_size=batch_size,
                    release_after=False,
                )
            )
            if len(response.embeddings) != len(batch):
                raise RuntimeError(
                    f"PE-Core returned {len(response.embeddings)} embeddings for {len(batch)} frames"
                )
            rows_to_add = []
            now = dt.datetime.now().isoformat()
            for source, vector in zip(batch, response.embeddings):
                if len(vector) != settings.PE_CORE_EMBEDDING_DIM:
                    raise RuntimeError(f"invalid PE-Core dimension for frame {source['id']}")
                rows_to_add.append({
                    "id": f"{source['id']}:{version}",
                    "source_frame_id": str(source["id"]),
                    "video_id": video_id,
                    "scene_id": str(source.get("scene_id") or ""),
                    "timestamp": float(source.get("timestamp", 0.0)),
                    "frame_path": str(source["frame_path"]),
                    "pe_core_vector": [float(value) for value in vector],
                    "model_id": model_id,
                    "model_revision": MODEL_REVISIONS[model_id],
                    "embedding_version": version,
                    "created_at": now,
                })
            frame_table.add(rows_to_add)
            completed = len(existing_ids & expected_ids) + start + len(batch)
            print(f"  embedded {completed}/{len(source_rows)}")

        metadata["indexed_frame_count"] = len(source_rows)
        metadata["status"] = "ready"
        _write_metadata(metadata_table, metadata)
        return metadata
    except Exception as exc:
        completed_rows = _rows(
            frame_table,
            f"video_id = '{video_id}' AND model_id = '{model_id}' AND embedding_version = '{version}'",
        )
        metadata["indexed_frame_count"] = len({str(row.get("source_frame_id")) for row in completed_rows} & {str(row["id"]) for row in source_rows})
        metadata["status"] = "error"
        metadata["error_message"] = str(exc)[:1000]
        _write_metadata(metadata_table, metadata)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill additive PE-Core frame embeddings")
    parser.add_argument("--model", choices=tuple(MODEL_REVISIONS), required=True)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--all-videos", action="store_true")
    selector.add_argument("--video-id")
    parser.add_argument("--resume", action="store_true", help="skip existing frame embeddings")
    args = parser.parse_args()

    if not pe_worker_client.enabled:
        print("PE_WORKER_ENABLED=true is required", file=sys.stderr)
        return 2
    if args.all_videos:
        videos = _rows(db_manager.get_table("videos"), limit=200000)
        video_ids = [str(row["id"]) for row in videos]
    else:
        video_ids = [str(args.video_id)]

    failures = 0
    try:
        for video_id in video_ids:
            try:
                backfill_video(video_id, args.model)
            except Exception as exc:
                failures += 1
                print(f"ERROR {video_id}: {exc}", file=sys.stderr)
    finally:
        try:
            pe_worker_client.unload()
        except Exception as exc:
            print(f"WARNING: PE-Core unload failed: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
