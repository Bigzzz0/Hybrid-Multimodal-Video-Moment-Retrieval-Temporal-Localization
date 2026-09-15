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
} from "./types";

export type { VideoMetadata, VideoKeyframeItem, VideoQAResult, UploadResponse, ProgressStatus, ClipExportResponse, SystemTelemetry };


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
    profile: "fast" | "accurate" = "fast"
  ): Promise<SearchResponse> {
    const res = await api.post<SearchResponse>("/search/moment", {
      query,
      video_id: videoId,
      top_k: topK,
      profile,
    });
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

  // 11. Keyframes List for Timeline Scrubber Tooltip
  async getVideoKeyframes(videoId: string): Promise<VideoKeyframeItem[]> {
    const res = await api.get<VideoKeyframeItem[]>(`/videos/${videoId}/keyframes`);
    return res.data;
  },
};



