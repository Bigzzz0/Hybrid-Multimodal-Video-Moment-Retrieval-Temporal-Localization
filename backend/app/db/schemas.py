import pyarrow as pa
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

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
    pa.field("created_at", pa.string())
])

# Table: scenes
SCENE_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("video_id", pa.string()),
    pa.field("scene_index", pa.int32()),
    pa.field("t_start", pa.float32()),
    pa.field("t_end", pa.float32()),
    pa.field("keyframe_count", pa.int32())
])

# Table: video_frames (Vector Dimension: 768 for SigLIP 2)
VIDEO_FRAME_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("video_id", pa.string()),
    pa.field("scene_id", pa.string()),
    pa.field("timestamp", pa.float32()),
    pa.field("frame_path", pa.string()),
    pa.field("siglip2_vector", pa.list_(pa.float32(), 768)),
    pa.field("vlm_caption", pa.string()),
    pa.field("has_dense_caption", pa.bool_())
])

# Table: transcripts (Timestamped Speech Segments)
TRANSCRIPT_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("video_id", pa.string()),
    pa.field("t_start", pa.float32()),
    pa.field("t_end", pa.float32()),
    pa.field("speaker_tag", pa.string()),
    pa.field("spoken_text", pa.string())
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

class MomentItem(BaseModel):
    t_start: float
    t_end: float
    score: float
    preview_frame_path: Optional[str] = None
    caption_preview: Optional[str] = None
    transcript_preview: Optional[str] = None
    modality_breakdown: Optional[Dict[str, float]] = None

class SearchResponse(BaseModel):
    query: str
    video_id: Optional[str] = None
    moments: List[MomentItem]
    timeline_heatmap: List[float] # Normalized density scores sampled per second
    total_duration: float
    latency_ms: float
    top_k: int

class SearchQueryRequest(BaseModel):
    query: str
    video_id: Optional[str] = None
    top_k: int = 5
    weight_visual: float = 0.45
    weight_caption: float = 0.35
    weight_audio: float = 0.20
    gaussian_sigma: float = 1.5
    threshold_factor: float = 0.8
