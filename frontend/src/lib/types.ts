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
}

export interface MomentItem {
  t_start: number;
  t_end: number;
  score: number;
  preview_frame_path?: string | null;
  caption_preview?: string | null;
  transcript_preview?: string | null;
  modality_breakdown?: Record<string, number> | null;
}

export interface SearchResponse {
  query: string;
  video_id?: string | null;
  moments: MomentItem[];
  timeline_heatmap: number[];
  total_duration: number;
  latency_ms: number;
  top_k: number;
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
  burned_subtitles: boolean;
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

