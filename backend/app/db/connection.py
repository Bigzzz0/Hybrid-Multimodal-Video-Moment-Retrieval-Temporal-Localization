import lancedb
from typing import Optional
from app.core.config import settings
from app.core.logger import logger
from app.db.schemas import (
    VIDEO_SCHEMA,
    SCENE_SCHEMA,
    VIDEO_FRAME_SCHEMA,
    TRANSCRIPT_SCHEMA,
    SEARCH_LOG_SCHEMA
)

class LanceDBManager:
    _instance: Optional["LanceDBManager"] = None
    
    def __init__(self):
        self.db_path = str(settings.LANCEDB_DIR)
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
        
        # 1. Table: videos
        if "videos" not in existing_tables:
            self.db.create_table("videos", schema=VIDEO_SCHEMA)
            logger.info("Created LanceDB table: videos")
            
        # 2. Table: scenes
        if "scenes" not in existing_tables:
            self.db.create_table("scenes", schema=SCENE_SCHEMA)
            logger.info("Created LanceDB table: scenes")
            
        # 3. Table: video_frames
        if "video_frames" not in existing_tables:
            self.db.create_table("video_frames", schema=VIDEO_FRAME_SCHEMA)
            logger.info("Created LanceDB table: video_frames")
            
        # 4. Table: transcripts
        if "transcripts" not in existing_tables:
            self.db.create_table("transcripts", schema=TRANSCRIPT_SCHEMA)
            logger.info("Created LanceDB table: transcripts")
            
        # 5. Table: search_logs
        if "search_logs" not in existing_tables:
            self.db.create_table("search_logs", schema=SEARCH_LOG_SCHEMA)
            logger.info("Created LanceDB table: search_logs")

    def get_table(self, table_name: str):
        return self.db.open_table(table_name)

    def create_indices(self):
        """Creates IVF-PQ and Full-Text Search (Tantivy) indices if data is present."""
        try:
            tbl_frames = self.get_table("video_frames")
            if len(tbl_frames) >= 128:
                logger.info("Building IVF-PQ index on video_frames (siglip2_vector)...")
                tbl_frames.create_index(
                    metric="cosine",
                    vector_column_name="siglip2_vector",
                    num_partitions=min(64, len(tbl_frames) // 10),
                    num_sub_vectors=16,
                    replace=True
                )
                tbl_frames.create_fts_index("vlm_caption", replace=True)
                
            tbl_transcripts = self.get_table("transcripts")
            if len(tbl_transcripts) > 0:
                tbl_transcripts.create_fts_index("spoken_text", replace=True)
        except Exception as e:
            logger.warning(f"Index creation deferred or failed: {e}")

db_manager = LanceDBManager.get_instance()
