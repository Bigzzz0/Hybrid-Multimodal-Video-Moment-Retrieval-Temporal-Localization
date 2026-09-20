from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EmbedTextRequest(BaseModel):
    texts: List[str] = Field(default_factory=list)


class EmbedImagesRequest(BaseModel):
    frame_paths: List[str] = Field(default_factory=list)
    batch_size: int = 16


class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]] = Field(default_factory=list)
    model_id: str = ""
    status: str = "unavailable"


class CaptionRequest(BaseModel):
    scene_id: str = ""
    frame_paths: List[str] = Field(default_factory=list)
    timestamps: List[float] = Field(default_factory=list)
    prompt: str = ""
    prompt_version: str = "classroom-caption-v1"
    max_new_tokens: int = 128


class CaptionResponse(BaseModel):
    text: str = ""
    summary: str = ""
    people_count: Optional[int] = None
    objects: List[str] = Field(default_factory=list)
    actions: List[str] = Field(default_factory=list)
    relations: List[str] = Field(default_factory=list)
    uncertainty: List[str] = Field(default_factory=list)
    model_id: str = ""
    status: str = "unavailable"


class VerifyRequest(BaseModel):
    query: str
    frame_paths: List[str] = Field(default_factory=list)
    timestamps: List[float] = Field(default_factory=list)
    caption_hint: str = ""
    max_new_tokens: int = 96


class VerifyResponse(BaseModel):
    event_present: bool = False
    start_frame_index: int = 0
    end_frame_index: int = 0
    confidence: float = 0.0
    reason: str = ""
    raw_text: str = ""
    model_id: str = ""


class AnswerRequest(BaseModel):
    question: str
    frame_paths: List[str] = Field(default_factory=list)
    timestamps: List[float] = Field(default_factory=list)
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    max_new_tokens: int = 256


class AnswerResponse(BaseModel):
    answer: str = ""
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    model_id: str = ""
    status: str = "unavailable"


class GroundRequest(BaseModel):
    video_path: str = ""
    video_fingerprint: str = ""
    frame_paths: List[str] = Field(default_factory=list)
    timestamps: List[float] = Field(default_factory=list)
    prompts: List[str] = Field(default_factory=list)
    fps: float = 4.0
    max_frames: int = 48
    request_source: str = "search"


class GroundObservation(BaseModel):
    timestamp: float
    bbox_xyxy: List[float] = Field(default_factory=list)
    score: float = 0.0
    mask_rle: Dict[str, Any] = Field(default_factory=dict)
    frame_width: int = 0
    frame_height: int = 0


class GroundTrack(BaseModel):
    track_id: str
    concept: str
    t_start: float
    t_end: float
    mean_score: float = 0.0
    max_score: float = 0.0
    observations: List[GroundObservation] = Field(default_factory=list)


class GroundResponse(BaseModel):
    tracks: List[GroundTrack] = Field(default_factory=list)
    model_id: str = ""
    grounding_version: str = ""
    cache_hit: bool = False
    status: str = "unavailable"
