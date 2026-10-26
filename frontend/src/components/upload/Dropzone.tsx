"use client";

import React, { useState, useRef, useEffect } from "react";
import {
  UploadCloud,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Film,
  Cpu,
  Database,
  Sparkles,
  Layers
} from "lucide-react";
import { apiClient } from "@/lib/api";
import { useLanguage } from "@/lib/language";

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
  { id: "keyframe_ssim", name: "3. Keyframe Sampling", desc: "SSIM structural difference filtering", icon: Film },
  { id: "siglip2_embedding", name: "4. Temporal & Visual Embedding", desc: "SigLIP 2 NaFlex frame representations", icon: Cpu },
  { id: "dense_visual_caption", name: "5. Background Visual Captions", desc: "CapRL 4B Q6 scene captions and verification", icon: Sparkles }
];

export const Dropzone: React.FC<DropzoneProps> = ({ onUploadSuccess }) => {
  const { language, t } = useLanguage();
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
      const normalizedStage = data.stage;
      setCurrentStage(normalizedStage);
      const stageOrder = ["decoding", "scene_detect", "keyframe_ssim", "siglip2_embedding", "lancedb_commit", "dense_visual_caption", "complete"];
      const currentIdx = stageOrder.indexOf(normalizedStage);
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
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
      const wsProtocol = apiUrl.startsWith("https") ? "wss" : "ws";
      const wsUrl = `${wsProtocol}://${apiUrl.replace(/^https?:\/\//, "")}/api/v1/ws/progress/${videoId}`;
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
      const serverMsg = err?.response?.data?.detail;
      const isConnectionIssue = !err?.response || err?.code === "ERR_NETWORK" || err?.message?.includes("Network Error");
      setError(
        serverMsg ||
        (isConnectionIssue
          ? "Backend is still initializing or unreachable. Please wait a few seconds and try uploading again."
          : "Upload failed. Please check backend connection.")
      );
      setIsUploading(false);
    }
  };

  return (
    <div className="w-full bg-[var(--panel)] border border-[var(--line)] rounded-2xl p-6 space-y-5 text-[var(--text)]">
      <div className="flex items-center justify-between">
        <h3 className="text-base font-semibold text-[var(--text)] flex items-center gap-2">
          <UploadCloud className="w-5 h-5 text-[var(--accent)]" />
          {t("modal.uploadTitle")}
        </h3>
        <span className="text-xs text-[var(--muted)] font-mono">Progressive Two-Phase Engine</span>
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
          className={`border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all duration-200 ${
            isDragging
              ? "border-[var(--accent)] bg-[var(--accent)]/10 scale-[0.99]"
              : "border-[var(--line)] hover:border-[var(--accent)] hover:bg-[var(--line)]/30"
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
            <UploadCloud className="w-10 h-10 mx-auto text-[var(--muted)] group-hover:text-[var(--accent)] transition-colors" />
            <p className="text-sm font-semibold text-[var(--text)]">
              {language === "th" ? (
                <>ลากและวางไฟล์วิดีโอที่นี่ หรือ <span className="text-[var(--accent)] underline">เลือกไฟล์จากเครื่อง</span></>
              ) : (
                <>Drag and drop video file here, or <span className="text-[var(--accent)] underline">browse files</span></>
              )}
            </p>
            <p className="text-xs text-[var(--muted)]">
              {language === "th" ? "รองรับ .mp4, .mkv, .mov, .webm (Full HD / 4K)" : "Supports .mp4, .mkv, .mov, .webm (Full HD / 4K)"}
            </p>
          </div>
        </div>
      ) : (
        /* Real-Time Live Pipeline Tracker */
        <div className="space-y-4 rounded-2xl bg-[var(--panel)] border border-[var(--line)] p-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Loader2 className="w-5 h-5 text-[var(--accent)] animate-spin" />
              <span className="text-sm font-bold text-[var(--text)]">Live Ingestion Pipeline</span>
            </div>
            <span className="text-xs font-mono font-bold px-2.5 py-1 rounded-full bg-[var(--accent)]/10 border border-[var(--accent)]/30 text-[var(--accent)]">
              {progress}% Completed
            </span>
          </div>

          {/* Master Progress Bar */}
          <div className="w-full bg-[var(--line)] rounded-full h-2.5 overflow-hidden">
            <div
              className="bg-[var(--accent)] h-2.5 rounded-full transition-all duration-300 shadow-sm"
              style={{ width: `${Math.max(8, progress)}%` }}
            />
          </div>

          {/* Current Action Banner */}
          <div className="p-3 rounded-xl bg-[var(--bg)] border border-[var(--line)] text-xs text-[var(--text)] flex items-start gap-2.5">
            <span className="w-2 h-2 rounded-full bg-[var(--accent)] mt-1 animate-pulse flex-shrink-0" />
            <div className="space-y-0.5 flex-1">
              <span className="font-semibold text-[var(--muted)]">
                {language === "th" ? "ขั้นตอนปัจจุบัน: " : "Current Step: "}
              </span>
              <span className="text-[var(--accent)] font-medium">
                {statusMessage || (language === "th" ? "กำลังประมวลผลไฟล์วิดีโอ..." : "Processing video file...")}
              </span>
            </div>
          </div>

          {/* Multi-Stage Pipeline Step Checklist */}
          <div className="space-y-2 pt-1">
            <p className="text-xs font-semibold text-[var(--muted)] uppercase tracking-wider">Pipeline Steps</p>
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
                        ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-700 dark:text-emerald-300"
                        : isRunning
                        ? "bg-[var(--accent)]/10 border-[var(--accent)] text-[var(--text)] shadow-sm"
                        : "bg-[var(--bg)] border-[var(--line)] text-[var(--muted)]"
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
                            {stg.id === "keyframe_ssim" && stageDetails?.scene_idx !== undefined && (
                              <>🖼️ Keyframes: Scene <b className="text-cyan-300">{stageDetails.scene_idx}</b> / {stageDetails.total_scenes} ({stageDetails.frame_count || 0} frames)</>
                            )}
                            {stg.id === "siglip2_embedding" && stageDetails?.processed_frames !== undefined && (
                              <>⚡ Encoded: <b className="text-cyan-300">{stageDetails.processed_frames}</b> / {stageDetails.total_frames} frames (Batch {stageDetails.batch}/{stageDetails.total_batches})</>
                            )}
                            {stg.id === "dense_visual_caption" && stageDetails?.scene_idx !== undefined && (
                              <>🤖 Captioned: Scene <b className="text-cyan-300">{stageDetails.scene_idx}</b> / {stageDetails.total_scenes}</>
                            )}
                            {stg.id === "scene_detect" && stageDetails?.duration_sec !== undefined && (
                              <>✂️ Length: {stageDetails.duration_sec?.toFixed(1)}s • {stageDetails.resolution} @ {stageDetails.fps}fps</>
                            )}
                            {stg.id === "lancedb_commit" && stageDetails?.frame_count !== undefined && (
                              <>💾 Indexing: {stageDetails.frame_count || 0} visual frames</>
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
