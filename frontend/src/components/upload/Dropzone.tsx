"use client";

import React, { useState, useRef, useEffect } from "react";
import {
  UploadCloud,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Film,
  Mic,
  Cpu,
  Database,
  Sparkles,
  Layers
} from "lucide-react";
import { apiClient } from "@/lib/api";

interface DropzoneProps {
  onUploadSuccess: (videoId: string) => void;
}

interface PipelineStage {
  id: string;
  name: string;
  desc: string;
  icon: any;
}

const PIPELINE_STAGES: PipelineStage[] = [
  { id: "decoding", name: "1. Hardware Video Decoding", desc: "Decord GPU NVDEC reading frames", icon: Film },
  { id: "scene_detect", name: "2. Adaptive Scene Cuts", desc: "PySceneDetect boundary segmentation", icon: Layers },
  { id: "asr_whisper", name: "3. Speech Transcription", desc: "Whisper-Large-v3-Turbo on CUDA FP16", icon: Mic },
  { id: "keyframe_ssim", name: "4. Keyframe Sampling", desc: "SSIM structural difference filtering", icon: Film },
  { id: "siglip2_embedding", name: "5. Multimodal Embedding", desc: "SigLIP 2 NaFlex 768-dim vectors", icon: Cpu },
  { id: "lancedb_commit", name: "6. LanceDB Storage & Indexing", desc: "Disk-based IVF-PQ & Tantivy FTS", icon: Database },
  { id: "minicpmv_caption", name: "7. Dense Action Captioning", desc: "MiniCPM-V 2.6 4-bit scene understanding", icon: Sparkles }
];

export const Dropzone: React.FC<DropzoneProps> = ({ onUploadSuccess }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState("");
  const [currentStage, setCurrentStage] = useState<string>("decoding");
  const [completedStages, setCompletedStages] = useState<string[]>([]);
  const [stageDetails, setStageDetails] = useState<Record<string, any>>({});
  const [error, setError] = useState<string | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const pollingRef = useRef<NodeJS.Timeout | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const cleanUpListeners = () => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  };

  useEffect(() => {
    return () => cleanUpListeners();
  }, []);

  const updateProgressState = (data: any, videoId: string) => {
    if (!data) return;

    if (typeof data.progress === "number") {
      setProgress((prev) => Math.max(prev, data.progress));
    }
    if (data.message) {
      setStatusMessage(data.message);
    }
    if (data.details) {
      setStageDetails(data.details);
    }
    if (data.stage) {
      setCurrentStage(data.stage);
      const stageOrder = ["decoding", "scene_detect", "asr_whisper", "keyframe_ssim", "siglip2_embedding", "lancedb_commit", "minicpmv_caption", "complete"];
      const currentIdx = stageOrder.indexOf(data.stage);
      if (currentIdx > 0) {
        const done = stageOrder.slice(0, currentIdx);
        setCompletedStages(done);
      }
    }

    if (data.progress >= 100 || data.stage === "complete") {
      setCompletedStages(PIPELINE_STAGES.map((s) => s.id));
      cleanUpListeners();
      setTimeout(() => {
        setIsUploading(false);
        onUploadSuccess(videoId);
      }, 1500);
    }
  };

  const handleFile = async (file: File) => {
    cleanUpListeners();
    setIsUploading(true);
    setProgress(10);
    setStatusMessage("Uploading video file to server...");
    setCurrentStage("decoding");
    setCompletedStages([]);
    setStageDetails({});
    setError(null);

    try {
      const res = await apiClient.uploadVideo(file);
      const videoId = res.video_id;

      // 1. Start Fast HTTP Status Polling (Every 600ms)
      pollingRef.current = setInterval(async () => {
        try {
          const status = await apiClient.getProgressStatus(videoId);
          updateProgressState(status, videoId);
        } catch (e) {
          console.debug("Status polling tick:", e);
        }
      }, 600);

      // 2. Connect WebSocket for Real-time Push
      const wsUrl = `ws://localhost:8000/api/v1/ws/progress/${videoId}`;
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          updateProgressState(data, videoId);
        } catch (e) {
          console.error("WS Parse error:", e);
        }
      };

      ws.onerror = (e) => {
        console.debug("WS notice (HTTP Polling fallback active):", e);
      };

    } catch (err: any) {
      cleanUpListeners();
      setError(err?.response?.data?.detail || "Upload failed. Please check backend connection.");
      setIsUploading(false);
    }
  };

  return (
    <div className="w-full glass-panel rounded-2xl p-6 space-y-5">
      <div className="flex items-center justify-between">
        <h3 className="text-base font-semibold text-white flex items-center gap-2">
          <UploadCloud className="w-5 h-5 text-indigo-400" />
          Upload & Ingest Video
        </h3>
        <span className="text-xs text-gray-400 font-mono">Progressive Two-Phase Engine</span>
      </div>

      {!isUploading ? (
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setIsDragging(false);
            if (e.dataTransfer.files && e.dataTransfer.files[0]) {
              handleFile(e.dataTransfer.files[0]);
            }
          }}
          onClick={() => fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-200 ${
            isDragging
              ? "border-cyan-500 bg-cyan-500/10 scale-[0.99]"
              : "border-surfaceBorder hover:border-indigo-500 hover:bg-surface/50"
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="video/mp4,video/mkv,video/mov,video/webm"
            className="hidden"
            onChange={(e) => {
              if (e.target.files && e.target.files[0]) {
                handleFile(e.target.files[0]);
              }
            }}
          />

          <div className="space-y-2">
            <UploadCloud className="w-10 h-10 mx-auto text-gray-400 group-hover:text-indigo-400 transition-colors" />
            <p className="text-sm font-medium text-gray-200">
              Drag & Drop your video file here, or <span className="text-cyan-400 underline">browse files</span>
            </p>
            <p className="text-xs text-gray-500">Supports .mp4, .mkv, .mov, .webm (Full HD / 4K)</p>
          </div>
        </div>
      ) : (
        /* Real-Time Live Pipeline Tracker */
        <div className="space-y-4 rounded-xl bg-surface/90 border border-surfaceBorder p-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Loader2 className="w-5 h-5 text-cyan-400 animate-spin" />
              <span className="text-sm font-bold text-white">Live Ingestion Pipeline</span>
            </div>
            <span className="text-xs font-mono font-bold px-2.5 py-1 rounded-full bg-cyan-950/80 border border-cyan-800 text-cyan-300">
              {progress}% Completed
            </span>
          </div>

          {/* Master Progress Bar */}
          <div className="w-full bg-surfaceBorder rounded-full h-2.5 overflow-hidden">
            <div
              className="bg-gradient-to-r from-indigo-500 via-cyan-400 to-emerald-400 h-2.5 rounded-full transition-all duration-300 shadow-lg shadow-cyan-500/20"
              style={{ width: `${Math.max(8, progress)}%` }}
            />
          </div>

          {/* Current Action Banner */}
          <div className="p-3 rounded-lg bg-background/80 border border-surfaceBorder text-xs text-gray-300 flex items-start gap-2.5">
            <span className="w-2 h-2 rounded-full bg-cyan-400 mt-1 animate-ping flex-shrink-0" />
            <div className="space-y-0.5 flex-1">
              <span className="font-semibold text-white">Current Action: </span>
              <span className="text-cyan-300 font-medium">{statusMessage || "Processing video pipeline..."}</span>
            </div>
          </div>

          {/* Multi-Stage Pipeline Step Checklist */}
          <div className="space-y-2 pt-1">
            <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Pipeline Steps</p>
            <div className="grid grid-cols-1 gap-2.5">
              {PIPELINE_STAGES.map((stg) => {
                const isDone = completedStages.includes(stg.id) || progress >= 100;
                const isRunning = currentStage === stg.id && progress < 100;
                const IconComponent = stg.icon;
                const subPct = isRunning && stageDetails?.sub_percent !== undefined ? stageDetails.sub_percent : (isDone ? 100 : 0);

                return (
                  <div
                    key={stg.id}
                    className={`flex flex-col p-3 rounded-xl border text-xs transition-all ${
                      isDone
                        ? "bg-emerald-950/20 border-emerald-800/40 text-emerald-300"
                        : isRunning
                        ? "bg-cyan-950/40 border-cyan-500/60 text-white shadow-lg shadow-cyan-950/50 ring-1 ring-cyan-500/30"
                        : "bg-surface/40 border-surfaceBorder/60 text-gray-500"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2.5">
                        <div
                          className={`w-7 h-7 rounded-lg flex items-center justify-center transition-colors ${
                            isDone
                              ? "bg-emerald-900/50 text-emerald-400"
                              : isRunning
                              ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                              : "bg-surfaceBorder text-gray-600"
                          }`}
                        >
                          <IconComponent className="w-4 h-4" />
                        </div>
                        <div>
                          <p className={`font-semibold ${isRunning ? "text-cyan-200" : ""}`}>{stg.name}</p>
                          <p className="text-[11px] text-gray-400">{stg.desc}</p>
                        </div>
                      </div>

                      <div>
                        {isDone ? (
                          <span className="flex items-center gap-1 text-[11px] text-emerald-400 font-semibold bg-emerald-950/60 px-2 py-0.5 rounded-full border border-emerald-800/60">
                            <CheckCircle2 className="w-3.5 h-3.5" /> 100% Ready
                          </span>
                        ) : isRunning ? (
                          <span className="flex items-center gap-1.5 text-[11px] text-cyan-300 font-mono font-bold bg-cyan-900/60 px-2.5 py-0.5 rounded-full border border-cyan-600 shadow-sm animate-pulse">
                            <Loader2 className="w-3 h-3 animate-spin" /> {subPct}% Active
                          </span>
                        ) : (
                          <span className="text-[11px] text-gray-600">Pending</span>
                        )}
                      </div>
                    </div>

                    {/* Real-Time Sub-Progress Bar & Detail Badges for Active Stage */}
                    {isRunning && (
                      <div className="mt-2.5 pt-2 border-t border-cyan-800/30 space-y-1.5 animate-in fade-in duration-200">
                        <div className="flex items-center justify-between text-[11px] font-mono">
                          <span className="text-gray-300 flex items-center gap-1.5">
                            {stg.id === "asr_whisper" && stageDetails?.current_sec !== undefined && (
                              <>⏱️ Transcribed: <b className="text-cyan-300">{stageDetails.current_sec}s</b> / {stageDetails.total_sec}s ({stageDetails.segment_count || 0} segments)</>
                            )}
                            {stg.id === "keyframe_ssim" && stageDetails?.scene_idx !== undefined && (
                              <>🖼️ Keyframes: Scene <b className="text-cyan-300">{stageDetails.scene_idx}</b> / {stageDetails.total_scenes} ({stageDetails.frame_count || 0} frames)</>
                            )}
                            {stg.id === "siglip2_embedding" && stageDetails?.processed_frames !== undefined && (
                              <>⚡ Encoded: <b className="text-cyan-300">{stageDetails.processed_frames}</b> / {stageDetails.total_frames} frames (Batch {stageDetails.batch}/{stageDetails.total_batches})</>
                            )}
                            {stg.id === "minicpmv_caption" && stageDetails?.scene_idx !== undefined && (
                              <>🤖 Captioned: Scene <b className="text-cyan-300">{stageDetails.scene_idx}</b> / {stageDetails.total_scenes}</>
                            )}
                            {stg.id === "scene_detect" && stageDetails?.duration_sec !== undefined && (
                              <>✂️ Length: {stageDetails.duration_sec?.toFixed(1)}s • {stageDetails.resolution} @ {stageDetails.fps}fps</>
                            )}
                            {stg.id === "lancedb_commit" && stageDetails?.frame_count !== undefined && (
                              <>💾 Indexing: {stageDetails.frame_count} visual frames • {stageDetails.transcript_count} speech rows</>
                            )}
                          </span>
                          <span className="text-cyan-400 font-bold">{subPct}%</span>
                        </div>

                        <div className="w-full bg-cyan-950/80 rounded-full h-1.5 overflow-hidden border border-cyan-800/50">
                          <div
                            className="bg-gradient-to-r from-cyan-500 to-indigo-400 h-1.5 rounded-full transition-all duration-200 shadow-sm"
                            style={{ width: `${Math.max(4, subPct)}%` }}
                          />
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {error && (
        <div className="flex items-center gap-2 p-3 bg-red-950/50 border border-red-800/60 rounded-xl text-xs text-red-300">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
};
