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
  recent_logs: string[];
}

