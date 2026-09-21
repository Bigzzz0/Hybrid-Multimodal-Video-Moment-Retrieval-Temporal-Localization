import axios from "axios";
import {
  VideoMetadata,
  VideoKeyframeItem,
  SearchResponse,
  VideoQAResult,
  UploadResponse,
  ProgressStatus,
  ClipExportResponse,
  SystemTelemetry,
  RetrievalBackend,
  RetrievalBackendInfo,
} from "./types";

export type { VideoMetadata, VideoKeyframeItem, VideoQAResult, UploadResponse, ProgressStatus, ClipExportResponse, SystemTelemetry, RetrievalBackend, RetrievalBackendInfo };


const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const api = axios.create({
  baseURL: `${API_BASE_URL}/api/v1`,
  headers: {
    "Content-Type": "application/json",
  },
});

export const apiClient = {
  // 1. Video List
  async getVideos(): Promise<VideoMetadata[]> {
    const res = await api.get<VideoMetadata[]>("/videos/list");
    return res.data;
  },

  // 2. Upload Video
  async uploadVideo(file: File): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append("file", file);
    const res = await api.post<UploadResponse>("/videos/upload", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });
    return res.data;
  },

  // 3. Progress Status (HTTP Polling fallback)
  async getProgressStatus(videoId: string): Promise<ProgressStatus> {
    const res = await api.get<ProgressStatus>(`/progress/${videoId}/status`);
    return res.data;
  },

  // 4. Moment Search
  async searchMoments(
    query: string,
    videoId: string,
    topK: number = 5,
    profile: "fast" | "accurate" = "fast",
    retrievalBackend: RetrievalBackend = "siglip2",
    signal?: AbortSignal
  ): Promise<SearchResponse> {
    const res = await api.post<SearchResponse>("/search/moment", {
      query,
      video_id: videoId,
      top_k: topK,
      profile,
      retrieval_backend: retrievalBackend,
    }, { signal });
    return res.data;
  },

  // 5. Video-RAG QA
  async chatWithVideo(
    videoId: string,
    question: string
  ): Promise<VideoQAResult> {
    const res = await api.post<VideoQAResult>("/rag/chat", {
      video_id: videoId,
      question,
    });
    return res.data;
  },

  // 6. Export / Cut Highlight Clip
  async exportClip(
    videoId: string,
    tStart: number,
    tEnd: number
  ): Promise<ClipExportResponse> {
    const res = await api.post<ClipExportResponse>(
      `/videos/${videoId}/cut-clip`,
      null,
      {
        params: {
          t_start: tStart,
          t_end: tEnd,
        },
      }
    );
    return res.data;
  },

  // 7. Video Stream URL helper
  getVideoStreamUrl(videoId: string): string {
    return `${API_BASE_URL}/api/v1/videos/${videoId}/stream`;
  },

  getDownloadUrl(downloadUrl: string): string {
    return downloadUrl.startsWith("http") ? downloadUrl : `${API_BASE_URL}${downloadUrl}`;
  },

  // 8. Keyframe Preview URL helper
  getFramePreviewUrl(framePath: string): string {
    return `${API_BASE_URL}/api/v1/videos/frame-preview?path=${encodeURIComponent(
      framePath
    )}`;
  },

  // 9. Delete Video & Data
  async deleteVideo(videoId: string): Promise<{ status: string; message: string; deleted_id: string }> {
    const res = await api.delete<{ status: string; message: string; deleted_id: string }>(`/videos/${videoId}`);
    return res.data;
  },

  // 10. Developer Panel System Telemetry
  async getSystemTelemetry(): Promise<SystemTelemetry> {
    const res = await api.get<SystemTelemetry>("/system/telemetry");
    return res.data;
  },

  async getRetrievalBackends(videoId?: string): Promise<RetrievalBackendInfo[]> {
    const res = await api.get<RetrievalBackendInfo[]>("/system/retrieval-backends", {
      params: videoId ? { video_id: videoId } : undefined,
    });
    return res.data;
  },

  // 11. Keyframes List for Timeline Scrubber Tooltip
  async getVideoKeyframes(videoId: string): Promise<VideoKeyframeItem[]> {
    const res = await api.get<VideoKeyframeItem[]>(`/videos/${videoId}/keyframes`);
    return res.data;
  },

  async getGroundingTrack(trackId: string, timestamp?: number): Promise<{
    timestamp: number;
    concept: string;
    bbox_xyxy: number[];
    score: number;
    mask_rle: { size?: number[]; counts?: number[]; order?: string };
    frame_width: number;
    frame_height: number;
  }> {
    const query = timestamp == null ? "" : `?timestamp=${encodeURIComponent(timestamp)}`;
    const res = await api.get(`/grounding/track/${encodeURIComponent(trackId)}${query}`);
    return res.data;
  },
};



