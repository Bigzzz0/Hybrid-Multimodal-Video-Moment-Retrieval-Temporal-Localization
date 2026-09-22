import lancedb
import datetime
from typing import Optional
from app.core.config import settings
from app.core.logger import logger
from app.db.schemas import (
    VIDEO_SCHEMA,
    SCENE_SCHEMA,
    VIDEO_FRAME_SCHEMA,
    SCENE_V2_SCHEMA,
    VIDEO_FRAME_V2_SCHEMA,
    SEARCH_LOG_SCHEMA,
    VISUAL_INDEX_METADATA_SCHEMA,
    SAM_TRACK_SCHEMA,
    SAM_OBSERVATION_SCHEMA,
    MODEL_ARTIFACT_CACHE_SCHEMA,
    SCENE_ANALYSIS_SCHEMA,
    QWEN_VERIFICATION_SCHEMA,
    QWEN_VERIFICATION_V2_SCHEMA,
    VLM_CAPTION_ARTIFACT_SCHEMA,
    VLM_ARTIFACT_METADATA_SCHEMA,
)

class LanceDBManager:
    _instance: Optional["LanceDBManager"] = None
    
    def __init__(self):
        self.db_path = str(settings.LANCEDB_DIR)
        self.index_metadata_table_name = "index_metadata"
        logger.info(f"Initializing LanceDB connection at: {self.db_path}")
        self.db = lancedb.connect(self.db_path)
        self._init_tables()
        
    @classmethod
    def get_instance(cls) -> "LanceDBManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _init_tables(self):
        existing_tables = self.db.table_names()

        def ensure_table(name, schema):
            if name in existing_tables:
                return
            try:
                self.db.create_table(name, schema=schema)
                logger.info(f"Created LanceDB table: {name}")
            except Exception as exc:
                # LanceDB can return a stale table_names snapshot while its
                # background loop is completing another create_table call.
                # Treat an already-existing table as success so API/worker
                # processes can start concurrently.
                if "already exists" not in str(exc).lower() and "exists" not in str(exc).lower():
                    raise
                logger.debug(f"LanceDB table already exists during startup: {name}")
        
        # 1. Table: videos
        ensure_table("videos", VIDEO_SCHEMA)
            
        # Legacy tables are never overwritten. They remain available only for
        # migration/backup while all new ingestion goes to *_v2 tables.
        ensure_table("scenes", SCENE_SCHEMA)
            
        # 3. Table: video_frames
        ensure_table("video_frames", VIDEO_FRAME_SCHEMA)

        ensure_table("scenes_v2", SCENE_V2_SCHEMA)
        ensure_table("video_frames_v2", VIDEO_FRAME_V2_SCHEMA)

        ensure_table("sam_tracks_v1", SAM_TRACK_SCHEMA)
        ensure_table("sam_observations_v1", SAM_OBSERVATION_SCHEMA)
        ensure_table("model_artifact_cache", MODEL_ARTIFACT_CACHE_SCHEMA)
        ensure_table("scene_analysis_v1", SCENE_ANALYSIS_SCHEMA)
        ensure_table("qwen_verifications_v1", QWEN_VERIFICATION_SCHEMA)
        ensure_table("qwen_verifications_v2", QWEN_VERIFICATION_V2_SCHEMA)
        ensure_table("vlm_caption_artifacts_v1", VLM_CAPTION_ARTIFACT_SCHEMA)
        ensure_table("vlm_artifact_metadata_v1", VLM_ARTIFACT_METADATA_SCHEMA)
            
        # 4. Table: search_logs
        ensure_table("search_logs", SEARCH_LOG_SCHEMA)

        ensure_table("visual_index_metadata", VISUAL_INDEX_METADATA_SCHEMA)
        ensure_table("index_metadata", VISUAL_INDEX_METADATA_SCHEMA)
        # A pre-v2 database may already have an incompatible table.  Never
        # alter it in place during startup; isolate the new metadata table so
        # an interrupted migration cannot corrupt the active index.
        try:
            existing_schema = set(self.db.open_table("index_metadata").schema.names)
            required_schema = set(VISUAL_INDEX_METADATA_SCHEMA.names)
            if not required_schema.issubset(existing_schema):
                self.index_metadata_table_name = "index_metadata_v2"
        except Exception:
            self.index_metadata_table_name = "index_metadata_v2"
        if self.index_metadata_table_name == "index_metadata_v2" and "index_metadata_v2" not in existing_tables:
            ensure_table("index_metadata_v2", VISUAL_INDEX_METADATA_SCHEMA)

    def get_table(self, table_name: str):
        return self.db.open_table(table_name)

    def get_index_metadata_table(self):
        return self.db.open_table(self.index_metadata_table_name)

    def create_indices(self):
        """Create v2 visual IVF-PQ and caption full-text indices when useful."""
        try:
            tbl_frames = self.get_table("video_frames_v2")
            if len(tbl_frames) >= 128:
                logger.info("Building IVF-PQ index on video_frames_v2 (siglip2_vector)...")
                tbl_frames.create_index(
                    metric="cosine",
                    vector_column_name="siglip2_vector",
                    num_partitions=min(64, len(tbl_frames) // 10),
                    num_sub_vectors=16,
                    replace=True
                )
            tbl_scenes = self.get_table("scenes_v2")
            if len(tbl_scenes) >= 32:
                tbl_scenes.create_fts_index("caption", replace=True)
        except Exception as e:
            logger.warning(f"Index creation deferred or failed: {e}")

    def visual_index_compatibility(self, video_id: Optional[str] = None) -> dict:
        """Return a non-destructive compatibility report for the visual index.

        Old databases are intentionally not silently mixed with v2 vectors.
        Callers can show the reason and trigger a re-ingestion/migration.
        """
        expected_dim = int(settings.SIGLIP2_EMBEDDING_DIM)
        expected_model = settings.SIGLIP2_MODEL_ID
        expected_version = settings.VISUAL_INDEX_VERSION
        try:
            frames_table = self.get_table("video_frames_v2")
            if video_id:
                try:
                    rows = frames_table.search().where(f"video_id = '{video_id}'").limit(200000).to_list()
                except Exception:
                    rows = [r for r in frames_table.to_arrow().to_pylist() if str(r.get("video_id")) == str(video_id)]
            else:
                rows = frames_table.to_arrow().to_pylist()
        except Exception as exc:
            return {"compatible": False, "reason": f"cannot read visual index: {exc}"}
        if not rows:
            return {"compatible": True, "reason": "empty visual index"}
        try:
            metadata = self.get_index_metadata_table().to_arrow().to_pylist()
            if video_id:
                current = next((row for row in metadata if row.get("id") == video_id), None)
            else:
                indexed_videos = {str(row.get("video_id")) for row in rows}
                valid_metadata = {str(row.get("id")) for row in metadata
                                  if row.get("schema_version") == expected_version
                                  and row.get("index_version") == expected_version
                                  and row.get("model_id") == expected_model}
                current = True if indexed_videos.issubset(valid_metadata) else None
        except Exception:
            current = None
        if not current or (video_id and (
            current.get("schema_version") != expected_version
            or current.get("index_version") != expected_version
            or current.get("model_id") != expected_model
        )):
            return {"compatible": False, "reason": "visual index metadata is missing or stale; re-ingest for v2"}
        for row in rows:
            vector = row.get("siglip2_vector")
            if vector is None or len(vector) != expected_dim:
                return {"compatible": False, "reason": "embedding dimension is not v2/768"}
            if row.get("embedding_version") != settings.SIGLIP2_EMBEDDING_VERSION:
                return {"compatible": False, "reason": "embedding version does not match v2"}
            if row.get("embedding_model") != expected_model:
                return {"compatible": False, "reason": "embedding model does not match configured SigLIP2 NaFlex"}
        return {"compatible": True, "reason": "visual vectors match configured v2 contract"}

    # Public alias used by migration/admin callers.
    def validate_visual_index(self) -> dict:
        return self.visual_index_compatibility()

    def mark_visual_index_v2(self, video_id: Optional[str] = None) -> None:
        """Record the exact vector contract after a successful v2 ingestion."""
        if video_id:
            frames = self.get_table("video_frames_v2")
            try:
                rows = frames.search().where(f"video_id = '{video_id}'").limit(200000).to_list()
            except Exception:
                rows = [r for r in frames.to_arrow().to_pylist() if str(r.get("video_id")) == str(video_id)]
            if not rows:
                raise ValueError("cannot activate v2 metadata without frame records")
            for row in rows:
                vector = row.get("siglip2_vector")
                if vector is None or len(vector) != int(settings.SIGLIP2_EMBEDDING_DIM):
                    raise ValueError("cannot activate v2 metadata with invalid embedding dimension")
                if row.get("embedding_model") != settings.SIGLIP2_MODEL_ID or row.get("embedding_version") != settings.SIGLIP2_EMBEDDING_VERSION:
                    raise ValueError("cannot activate v2 metadata with stale embedding model/version")
        table = self.get_index_metadata_table()
        metadata_id = str(video_id or "visual")
        try:
            table.delete(f"id = '{metadata_id}'")
        except Exception:
            pass
        table.add([{
            "id": metadata_id,
            "schema_version": settings.VISUAL_INDEX_VERSION,
            "index_version": settings.VISUAL_INDEX_VERSION,
            "model_id": settings.SIGLIP2_MODEL_ID,
            "embedding_dim": int(settings.SIGLIP2_EMBEDDING_DIM),
            "caption_model": settings.QWEN_VL_MODEL_ID,
            "indexed_at": datetime.datetime.now().isoformat(),
            "created_at": datetime.datetime.now().isoformat(),
        }])

db_manager = LanceDBManager.get_instance()
