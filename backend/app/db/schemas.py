import pyarrow as pa
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal

# ======================= Apache Arrow Schemas for LanceDB =======================

# Table: videos
VIDEO_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("filename", pa.string()),
    pa.field("filepath", pa.string()),
    pa.field("duration_sec", pa.float32()),
    pa.field("fps", pa.float32()),
    pa.field("resolution", pa.string()),
    pa.field("total_frames", pa.int64()),
    pa.field("ingestion_phase", pa.string()),  # "phase1_ready", "phase2_complete", "error"
    pa.field("visual_index_version", pa.string()),
    pa.field("embedding_model", pa.string()),
    pa.field("created_at", pa.string())
])

# Legacy tables are retained read-only for migration/backup. Runtime v2 writes
# only to *_v2 tables so an interrupted reindex cannot corrupt the active index.
SCENE_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("video_id", pa.string()),
    pa.field("scene_index", pa.int32()),
    pa.field("t_start", pa.float32()),
    pa.field("t_end", pa.float32()),
    pa.field("keyframe_count", pa.int32())
])

# Legacy frame schema (v1 compatibility only).
VIDEO_FRAME_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("video_id", pa.string()),
    pa.field("scene_id", pa.string()),
    pa.field("timestamp", pa.float32()),
    pa.field("frame_path", pa.string()),
    pa.field("siglip2_vector", pa.list_(pa.float32(), 768)),
    pa.field("embedding_model", pa.string()),
    pa.field("embedding_version", pa.string()),
    pa.field("pixel_motion", pa.float32()),  # Inter-frame absolute pixel motion energy
])

SCENE_V2_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("video_id", pa.string()),
    pa.field("scene_index", pa.int32()),
    pa.field("t_start", pa.float32()),
    pa.field("t_end", pa.float32()),
    pa.field("keyframe_count", pa.int32()),
    pa.field("caption", pa.string()),
    pa.field("caption_status", pa.string()),
    pa.field("transition_energy", pa.float32()),
    pa.field("embedding_model", pa.string()),
    pa.field("caption_model", pa.string()),
])

VIDEO_FRAME_V2_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("video_id", pa.string()),
    pa.field("scene_id", pa.string()),
    pa.field("timestamp", pa.float32()),
    pa.field("frame_path", pa.string()),
    pa.field("siglip2_vector", pa.list_(pa.float32(), 768)),
    pa.field("embedding_model", pa.string()),
    pa.field("embedding_version", pa.string()),
    pa.field("transition_energy", pa.float32()),
])

# Table: search_logs
SEARCH_LOG_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("query_text", pa.string()),
    pa.field("video_id", pa.string()),
    pa.field("latency_ms", pa.float32()),
    pa.field("retrieved_moments_count", pa.int32()),
    pa.field("created_at", pa.string())
])

VISUAL_INDEX_METADATA_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("schema_version", pa.string()),
    pa.field("index_version", pa.string()),
    pa.field("model_id", pa.string()),
    pa.field("embedding_dim", pa.int32()),
    pa.field("caption_model", pa.string()),
    pa.field("indexed_at", pa.string()),
    pa.field("created_at", pa.string()),
])

# Additive grounding/caption artifacts.  These tables are intentionally
# separate from the v2 visual index so changing Qwen/SAM versions never
# invalidates or overwrites SigLIP2 vectors.
SAM_TRACK_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("video_id", pa.string()),
    pa.field("video_fingerprint", pa.string()),
    pa.field("concept", pa.string()),
    pa.field("normalized_prompt", pa.string()),
    pa.field("t_start", pa.float32()),
    pa.field("t_end", pa.float32()),
    pa.field("mean_score", pa.float32()),
    pa.field("max_score", pa.float32()),
    pa.field("source", pa.string()),
    pa.field("model_id", pa.string()),
    pa.field("grounding_version", pa.string()),
    pa.field("created_at", pa.string()),
])

SAM_OBSERVATION_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("track_id", pa.string()),
    pa.field("video_id", pa.string()),
    pa.field("concept", pa.string()),
    pa.field("timestamp", pa.float32()),
    pa.field("bbox_xyxy", pa.list_(pa.float32(), 4)),
    pa.field("score", pa.float32()),
    pa.field("mask_artifact_path", pa.string()),
    pa.field("frame_width", pa.int32()),
    pa.field("frame_height", pa.int32()),
])

MODEL_ARTIFACT_CACHE_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("video_id", pa.string()),
    pa.field("task_type", pa.string()),
    pa.field("t_start", pa.float32()),
    pa.field("t_end", pa.float32()),
    pa.field("normalized_prompt", pa.string()),
    pa.field("model_id", pa.string()),
    pa.field("artifact_version", pa.string()),
    pa.field("artifact_id", pa.string()),
    pa.field("status", pa.string()),
    pa.field("created_at", pa.string()),
    pa.field("last_accessed_at", pa.string()),
])

SCENE_ANALYSIS_SCHEMA = pa.schema([
    pa.field("scene_id", pa.string()),
    pa.field("video_id", pa.string()),
    pa.field("caption_text", pa.string()),
    pa.field("structured_json", pa.string()),
    pa.field("model_id", pa.string()),
    pa.field("caption_version", pa.string()),
    pa.field("prompt_version", pa.string()),
    pa.field("status", pa.string()),
    pa.field("created_at", pa.string()),
])

# ======================= Pydantic Models for REST API =======================

class VideoMetadata(BaseModel):
    id: str
    filename: str
    filepath: str
    duration_sec: float
    fps: float
    resolution: str
    total_frames: int
    ingestion_phase: str
    created_at: str
    visual_index_version: Optional[str] = None
    embedding_model: Optional[str] = None

class VideoFrameItem(BaseModel):
    id: str
    video_id: str
    scene_id: str
    timestamp: float
    frame_path: str
    siglip2_vector: List[float]
    embedding_model: Optional[str] = None
    embedding_version: Optional[str] = None
    transition_energy: float = 0.0

class MomentItem(BaseModel):
    t_start: float
    t_end: float
    score: float
    raw_score: Optional[float] = None
    display_score: Optional[float] = None
    preview_frame_path: Optional[str] = None
    caption_preview: Optional[str] = None
    modality_breakdown: Optional[Dict[str, float]] = None
    occurrence_index: int = 0
    context_t_start: Optional[float] = None
    context_t_end: Optional[float] = None
    grounding_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    verifier_evidence: Optional[Dict[str, Any]] = None

class SearchResponse(BaseModel):
    query: str
    video_id: str
    moments: List[MomentItem]
    timeline_heatmap: List[float] # Normalized density scores sampled per second
    total_duration: float
    latency_ms: float
    top_k: int
    profile: Literal["fast", "accurate"] = "fast"
    calibrated: bool = False
    index_version: str = "v2"
    warnings: List[str] = Field(default_factory=list)
    strategy_used: str = "fast"
    models_used: List[str] = Field(default_factory=list)
    stage_latency_ms: Dict[str, float] = Field(default_factory=dict)
    cache_hits: Dict[str, bool] = Field(default_factory=dict)

class SearchQueryRequest(BaseModel):
    query: str
    video_id: str
    top_k: int = Field(default=5, ge=1, le=20)
    profile: Literal["fast", "accurate"] = "fast"

    class Config:
        extra = "ignore"
