export interface VideoMetadata {
  id: string;
  filename: string;
  filepath: string;
  duration_sec: number;
  fps: number;
  resolution: string;
  total_frames: number;
  ingestion_phase: string;
  created_at: string;
  visual_index_version?: string | null;
  embedding_model?: string | null;
}

export interface VideoKeyframeItem {
  timestamp: number;
  frame_path: string;
  thumbnail_url: string;
}

export interface ScrubberHoverState {
  isActive: boolean;
  clientX: number;
  relativeRatio: number;
  hoverTime: number;
  scoreAtTime: number;
  nearestFrameUrl: string | null;
}

export interface ActionSuggestionCategory {
  categoryName: string;
  iconName: string;
  badgeColor: string;
  items: {
    labelTh: string;
    labelEn: string;
    query: string;
  }[];
}


export interface MomentItem {
  t_start: number;
  t_end: number;
  score: number;
  preview_frame_path?: string | null;
  caption_preview?: string | null;
  modality_breakdown?: Record<string, number> | null;
  occurrence_index?: number;
  context_t_start?: number | null;
  context_t_end?: number | null;
  grounding_evidence?: GroundingEvidence[];
  verifier_evidence?: VerifierEvidence | null;
}

export interface GroundingEvidence {
  artifact_id?: string;
  track_id: string;
  concept: string;
  t_start: number;
  t_end: number;
  confidence: number;
}

export interface VerifierEvidence {
  event_present: boolean;
  confidence: number;
  reason?: string;
  model_id?: string;
}

export interface DragHandleState {
  isDragging: boolean;
  handleType: "start" | "end" | "middle" | null;
  initialX: number;
  initialInterval: [number, number];
}

export interface TimelineZoomState {
  scale: number; // 1.0, 2.0, 4.0
  scrollOffsetSec: number;
}

export interface FilterCriteria {
  minConfidence: number; // 0.0 - 1.0
  motionFilter: "all" | "high" | "static";
  minDurationSec: number;
  maxDurationSec: number;
}

export interface SearchResponse {
  query: string;
  video_id: string;
  moments: MomentItem[];
  timeline_heatmap: number[];
  total_duration: number;
  latency_ms: number;
  top_k: number;
  profile: "fast" | "accurate";
  calibrated: boolean;
  index_version: string;
  warnings: string[];
  strategy_used?: string;
  models_used?: string[];
  stage_latency_ms?: Record<string, number>;
  cache_hits?: Record<string, boolean>;
  cascade_path?: string[];
  models_attempted?: string[];
  stage_status?: Record<string, string>;
  planner_version?: string;
  retrieval_backend_requested?: RetrievalBackend;
  retrieval_backend_used?: RetrievalBackend;
  retrieval_model_id?: string;
  retrieval_embedding_version?: string;
}

export type RetrievalBackend = "siglip2" | "pe_core_b16" | "pe_core_l14";

export interface RetrievalBackendInfo {
  id: RetrievalBackend;
  label: string;
  model_id: string;
  embedding_version: string;
  model_revision?: string;
  ready: boolean;
  indexed_frame_count?: number;
  source_frame_count?: number;
  experimental: boolean;
  worker?: Record<string, any>;
}

export interface GroundedMoment {
  t_start: number;
  t_end: number;
  citation_text: string;
  thumbnail_path?: string | null;
}

export interface VideoQAResult {
  question: string;
  answer: string;
  grounded_moments: GroundedMoment[];
  latency_ms: number;
}

export interface UploadResponse {
  status: string;
  video_id: string;
  filename: string;
  message: string;
  websocket_url: string;
  status_url: string;
}

export interface ProgressStatus {
  video_id: string;
  progress: number;
  message: string;
  stage: string;
  phase: string;
  details?: Record<string, any>;
}

export interface ClipExportResponse {
  status: string;
  clip_path: string;
  clip_filename: string;
  download_url: string;
}

export interface SystemTelemetry {
  status: string;
  gpu: {
    available: boolean;
    device_name: string;
    device_count: number;
    allocated_vram_mb: number;
    reserved_vram_mb: number;
    total_vram_mb: number;
    compute_capability: string;
  };
  system: {
    cpu_percent: number;
    cpu_count_logical: number;
    cpu_count_physical: number;
    ram_total_gb: number;
    ram_used_gb: number;
    ram_percent: number;
    platform: string;
    python_version: string;
  };
  lancedb: {
    tables: Record<string, number>;
    total_records: number;
    storage_path: string;
  };
  models: Record<string, any>;
  inference_worker?: Record<string, any>;
  pe_worker?: Record<string, any>;
  recent_logs: string[];
}

