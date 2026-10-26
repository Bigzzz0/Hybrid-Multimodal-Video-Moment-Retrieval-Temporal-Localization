"use client";

import React, { useRef, useEffect, useState } from "react";
import {
  Play,
  Pause,
  RotateCcw,
  Volume2,
  VolumeX,
  Maximize,
  Repeat,
  ChevronLeft,
  ChevronRight,
  Download,
  Loader2,
  Sparkles,
} from "lucide-react";
import { TimelineHeatmap } from "./TimelineHeatmap";
import { GroundingEvidence, MomentItem, VideoKeyframeItem, VideoMetadata } from "@/lib/types";
import { apiClient } from "@/lib/api";
import { useLanguage } from "@/lib/language";

interface VideoPlayerProps {
  streamUrl: string;
  duration: number;
  seekTime?: number | null;
  highlightInterval?: [number, number] | null;
  heatmapData: number[];
  moments?: MomentItem[];
  keyframeRecords?: VideoKeyframeItem[];
  fps?: number;
  isCinemaMode?: boolean;
  onToggleCinemaMode?: () => void;
  onBoundaryChange?: (newStart: number, newEnd: number) => void;
  boundaryAdjusted?: boolean;
  contextExpanded?: boolean;
  onResetBoundary?: () => void;
  groundingEvidence?: GroundingEvidence[];
  videoMetadata?: VideoMetadata | null;
  activeMoment?: MomentItem | null;
  searchQuery?: string;
}

export const VideoPlayer: React.FC<VideoPlayerProps> = ({
  streamUrl,
  duration,
  seekTime,
  highlightInterval,
  heatmapData,
  moments = [],
  keyframeRecords = [],
  fps = 25,
  isCinemaMode = false,
  onToggleCinemaMode,
  onBoundaryChange,
  boundaryAdjusted = false,
  contextExpanded = false,
  onResetBoundary,
  groundingEvidence = [],
  videoMetadata,
  activeMoment,
  searchQuery,
}) => {
  const { language, t } = useLanguage();
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [isMuted, setIsMuted] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1.0);
  const [isLoopActive, setIsLoopActive] = useState(false);
  const [maskEnabled, setMaskEnabled] = useState(true);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [isExportingCurrent, setIsExportingCurrent] = useState(false);
  const [groundingArtifact, setGroundingArtifact] = useState<any>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const lastTrackRef = useRef<string | null>(null);

  // Grounding track sync
  useEffect(() => {
    const evidence = groundingEvidence.find(
      (item) => currentTime >= item.t_start - 0.25 && currentTime <= item.t_end + 0.25
    );
    if (!evidence || !maskEnabled) {
      setGroundingArtifact(null);
      lastTrackRef.current = null;
      return;
    }
    if (lastTrackRef.current === `${evidence.track_id}:${Math.floor(currentTime * 2)}`) return;
    lastTrackRef.current = `${evidence.track_id}:${Math.floor(currentTime * 2)}`;
    apiClient
      .getGroundingTrack(evidence.track_id, currentTime)
      .then(setGroundingArtifact)
      .catch(() => setGroundingArtifact(null));
  }, [currentTime, groundingEvidence, maskEnabled]);

  // Grounding SAM Mask Canvas Drawing
  useEffect(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;
    const parent = canvas.parentElement;
    if (!parent) return;
    const width = parent.clientWidth;
    const height = parent.clientHeight;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);

    // Note: Bounding box drawing temporarily disabled per user instruction
    if (!groundingArtifact || !maskEnabled) return;
  }, [groundingArtifact, maskEnabled]);

  // Handle external seek requests
  useEffect(() => {
    if (seekTime !== null && seekTime !== undefined && videoRef.current) {
      videoRef.current.currentTime = seekTime;
      videoRef.current.play().catch(() => {});
      setIsPlaying(true);
    }
  }, [seekTime]);

  // Playback rate
  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.playbackRate = playbackSpeed;
    }
  }, [playbackSpeed]);

  const togglePlay = () => {
    if (!videoRef.current) return;
    if (isPlaying) {
      videoRef.current.pause();
      setIsPlaying(false);
    } else {
      videoRef.current.play();
      setIsPlaying(true);
    }
  };

  const handleTimeUpdate = () => {
    if (!videoRef.current) return;
    const curr = videoRef.current.currentTime;
    setCurrentTime(curr);

    // Loop moment interval if enabled
    if (isLoopActive && highlightInterval) {
      const [ts, te] = highlightInterval;
      if (curr >= te - 0.06 || curr < ts - 0.2) {
        videoRef.current.currentTime = ts;
        if (!isPlaying) {
          videoRef.current.play().catch(() => {});
          setIsPlaying(true);
        }
      }
    }
  };

  const handleSeek = (time: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = time;
      setCurrentTime(time);
    }
  };

  const handleStepFrame = (frames: number) => {
    if (!videoRef.current) return;
    const frameDuration = 1.0 / Math.max(10, fps);
    const target = Math.max(0, Math.min(duration, videoRef.current.currentTime + frames * frameDuration));
    videoRef.current.currentTime = target;
    setCurrentTime(target);
  };

  const formatTime = (secs: number) => {
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
  };

  const getCameraTime = (elapsedSec: number) => {
    const totalSec = 14 * 3600 + 31 * 60 + Math.floor(elapsedSec);
    const h = Math.floor((totalSec / 3600) % 24);
    const m = Math.floor((totalSec % 3600) / 60);
    const s = Math.floor(totalSec % 60);
    return `${h.toString().padStart(2, "0")}:${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
  };

  const handleExportCurrentClip = async () => {
    if (!videoMetadata?.id) return;
    setIsExportingCurrent(true);

    const tStart = highlightInterval ? highlightInterval[0] : Math.max(0, currentTime - 3);
    const tEnd = highlightInterval ? highlightInterval[1] : Math.min(duration, currentTime + 3);

    try {
      const res = await apiClient.exportClip(videoMetadata.id, tStart, tEnd);
      if (res && res.download_url) {
        const a = document.createElement("a");
        a.href = apiClient.getDownloadUrl(res.download_url);
        a.download = res.clip_filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
      }
    } catch (err) {
      console.error("Export current clip failed:", err);
    } finally {
      setIsExportingCurrent(false);
    }
  };

  // Detect category from filename
  const getCategoryLabel = () => {
    const name = (videoMetadata?.filename || "").toLowerCase();
    if (name.includes("street") || name.includes("ถนน") || name.includes("traffic")) return t("sidebar.cat.street");
    if (name.includes("gate") || name.includes("ประตู")) return t("sidebar.cat.gate");
    if (name.includes("parking") || name.includes("จอดรถ")) return t("sidebar.cat.parking");
    if (name.includes("class") || name.includes("ห้องเรียน") || name.includes("people")) return t("sidebar.cat.classroom");
    return language === "th" ? "กล้องวงจรปิด" : "CCTV Stream";
  };

  return (
    <div className="w-full app-panel rounded-2xl p-4 sm:p-5 flex flex-col gap-3.5 border border-[var(--line)] shadow-sm">
      {/* Video Title & Specs Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2.5">
        <div>
          <h2 className="text-base font-bold text-[var(--text)] tracking-tight">
            {videoMetadata?.filename || "street.mp4"}
          </h2>
          <div className="flex flex-wrap items-center gap-1.5 text-xs text-[var(--muted)] font-mono">
            <span>{videoMetadata?.resolution || "1920×1080"}</span>
            <span>·</span>
            <span>{fps} fps</span>
            <span>·</span>
            <span>{formatTime(duration)}</span>
            <span>·</span>
            <span className="font-sans font-medium">{getCategoryLabel()}</span>
          </div>
        </div>

        {/* Header Right Controls: Heatmap Toggle & Speed Pills */}
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setShowHeatmap(!showHeatmap)}
            className={`px-3 py-1.5 rounded-xl text-xs font-semibold border transition-all ${
              showHeatmap
                ? "bg-[var(--line)]/50 border-[var(--accent)] text-[var(--accent)]"
                : "border-[var(--line)] text-[var(--muted)] hover:text-[var(--text)]"
            }`}
          >
            {t("player.heatmap")}
          </button>

          {/* Segmented Speed Pills [ 1x ] [ 2x ] [ 4x ] */}
          <div className="flex items-center bg-[var(--bg)] border border-[var(--line)] rounded-xl p-0.5 font-mono text-xs">
            {[1.0, 2.0, 4.0].map((rate) => (
              <button
                key={rate}
                type="button"
                onClick={() => setPlaybackSpeed(rate)}
                className={`px-2.5 py-1 rounded-lg font-bold transition-all ${
                  playbackSpeed === rate
                    ? "bg-[var(--panel)] text-[var(--text)] shadow-sm"
                    : "text-[var(--muted)] hover:text-[var(--text)]"
                }`}
              >
                {rate}x
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* 16:9 Video Canvas Viewport */}
      <div className="relative aspect-video rounded-xl overflow-hidden bg-black border border-[var(--line)] group shadow-inner">
        <video
          ref={videoRef}
          src={streamUrl}
          onTimeUpdate={handleTimeUpdate}
          onPlay={() => setIsPlaying(true)}
          onPause={() => setIsPlaying(false)}
          className="w-full h-full object-contain"
        />

        <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none" aria-hidden="true" />

        {/* OSD Top-Left: Camera ID / Preview Label */}
        <div className="absolute top-3 left-3 flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-black/70 backdrop-blur-sm border border-white/10 text-white font-mono text-[11px]">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
          <span>{t("player.camPreview")}</span>
        </div>

        {/* OSD Top-Right: Camera Timestamp */}
        <div className="absolute top-3 right-3 px-2.5 py-1 rounded-md bg-black/70 backdrop-blur-sm border border-white/10 text-white font-mono text-[11px]">
          <span>26/10/2026 {getCameraTime(currentTime)}</span>
        </div>

        {/* SAM Mask Toggle Button */}
        {groundingEvidence.length > 0 && (
          <button
            type="button"
            onClick={() => setMaskEnabled((value) => !value)}
            className="absolute bottom-3 left-3 z-10 rounded-lg border border-white/20 bg-black/75 px-2.5 py-1 text-[10px] font-mono text-white hover:bg-black"
          >
            {maskEnabled ? t("player.hideMask") : t("player.showMask")}
          </button>
        )}

        {/* Video Overlay Play/Pause Action */}
        <div
          onClick={togglePlay}
          className="absolute inset-0 flex items-center justify-center bg-black/10 opacity-0 group-hover:opacity-100 transition-opacity cursor-pointer"
        >
          <div className="w-14 h-14 rounded-full bg-[var(--accent)] text-white flex items-center justify-center shadow-lg transform group-hover:scale-105 transition-transform">
            {isPlaying ? <Pause className="w-6 h-6 fill-current" /> : <Play className="w-6 h-6 ml-0.5 fill-current" />}
          </div>
        </div>
      </div>

      {/* Studio Control Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 px-1 text-xs text-[var(--muted)]">
        {/* Left: Round Play Button, Frame Stepper, Video Time & Camera Time */}
        <div className="flex items-center gap-2 sm:gap-3">
          {/* Round Blue Play Button matching Mockup */}
          <button
            type="button"
            onClick={togglePlay}
            className="w-9 h-9 rounded-full bg-[var(--accent)] text-white hover:opacity-95 flex items-center justify-center shadow-sm transition-all"
            title={isPlaying ? (language === "th" ? "หยุดชั่วคราว (Space)" : "Pause (Space)") : (language === "th" ? "เล่น (Space)" : "Play (Space)")}
          >
            {isPlaying ? <Pause className="w-4 h-4 fill-current" /> : <Play className="w-4 h-4 ml-0.5 fill-current" />}
          </button>

          {/* Time Display with Real Camera Time */}
          <div className="flex items-center gap-2 font-mono text-xs text-[var(--text)] font-semibold">
            <span>
              {formatTime(currentTime)} / {formatTime(duration)}
            </span>
            <span className="text-[var(--muted)] font-normal text-[11px]">
              {t("player.cameraTime")} {getCameraTime(currentTime)}
            </span>
          </div>

          {/* Frame Steppers */}
          <div className="hidden sm:flex items-center gap-0.5 bg-[var(--bg)] border border-[var(--line)] rounded-xl p-0.5">
            <button
              type="button"
              onClick={() => handleStepFrame(-1)}
              title={t("player.stepBack")}
              className="p-1 hover:bg-[var(--line)] text-[var(--muted)] hover:text-[var(--text)] rounded-lg transition-colors"
            >
              <ChevronLeft className="w-3.5 h-3.5" />
            </button>
            <span className="text-[10px] font-mono text-[var(--muted)] px-1">
              F:{Math.floor(currentTime * Math.max(10, fps))}
            </span>
            <button
              type="button"
              onClick={() => handleStepFrame(1)}
              title={t("player.stepForward")}
              className="p-1 hover:bg-[var(--line)] text-[var(--muted)] hover:text-[var(--text)] rounded-lg transition-colors"
            >
              <ChevronRight className="w-3.5 h-3.5" />
            </button>
          </div>

          {/* Rewind 5s */}
          <button
            type="button"
            onClick={() => handleSeek(Math.max(0, currentTime - 5))}
            title={t("player.rewind")}
            className="p-1.5 rounded-xl hover:bg-[var(--line)] text-[var(--muted)] hover:text-[var(--text)] transition-colors"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>

          {/* Mute Toggle */}
          <button
            type="button"
            onClick={() => {
              if (videoRef.current) {
                videoRef.current.muted = !isMuted;
                setIsMuted(!isMuted);
              }
            }}
            title={isMuted ? t("player.unmute") : t("player.mute")}
            className="p-1.5 rounded-xl hover:bg-[var(--line)] text-[var(--muted)] hover:text-[var(--text)] transition-colors"
          >
            {isMuted ? <VolumeX className="w-3.5 h-3.5" /> : <Volume2 className="w-3.5 h-3.5" />}
          </button>

          {/* A-B Loop Toggle */}
          {highlightInterval && (
            <button
              type="button"
              onClick={() => setIsLoopActive(!isLoopActive)}
              className={`px-2 py-1 rounded-xl text-xs font-mono font-semibold flex items-center gap-1 border transition-all ${
                isLoopActive
                  ? "bg-[var(--accent)] text-white border-[var(--accent)] shadow-sm"
                  : "border-[var(--line)] text-[var(--muted)] hover:text-[var(--text)]"
              }`}
              title={t("player.loop")}
            >
              <Repeat className="w-3 h-3" />
              <span>{isLoopActive ? "Loop ON" : "Loop"}</span>
            </button>
          )}
        </div>

        {/* Right: Export Clip Button & Fullscreen */}
        <div className="flex items-center gap-2">
          {/* Export Current Clip Button matching Mockup */}
          <button
            type="button"
            onClick={handleExportCurrentClip}
            disabled={isExportingCurrent}
            className="px-3 py-1.5 rounded-xl border border-[var(--line)] hover:border-[var(--accent)] bg-[var(--panel)] text-[var(--text)] text-xs font-semibold flex items-center gap-1.5 transition-all shadow-sm cursor-pointer disabled:opacity-50"
          >
            {isExportingCurrent ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin text-[var(--accent)]" />
            ) : (
              <Download className="w-3.5 h-3.5 text-[var(--muted)]" />
            )}
            <span>{isExportingCurrent ? t("player.exporting") : t("player.exportClip")}</span>
          </button>

          {/* Fullscreen */}
          <button
            type="button"
            onClick={() => {
              if (videoRef.current) {
                if (document.fullscreenElement) {
                  document.exitFullscreen();
                } else {
                  videoRef.current.requestFullscreen();
                }
              }
            }}
            title={t("player.fullscreen")}
            className="p-2 rounded-xl hover:bg-[var(--line)] text-[var(--muted)] hover:text-[var(--text)] transition-colors"
          >
            <Maximize className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Visual Relevance Heatmap Timeline */}
      {showHeatmap && (
        <TimelineHeatmap
          heatmapData={heatmapData}
          duration={duration}
          currentTime={currentTime}
          highlightInterval={highlightInterval}
          moments={moments}
          keyframeRecords={keyframeRecords}
          onSeek={handleSeek}
          onBoundaryChange={onBoundaryChange}
        />
      )}

      {/* Boundary Adjusted / Context Info */}
      {highlightInterval && (boundaryAdjusted || contextExpanded) && (
        <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-[var(--line)] bg-[var(--bg)] px-3 py-2 text-xs text-[var(--text)]">
          <span>{contextExpanded ? (language === "th" ? "กำลังดูช่วงบริบทขยาย" : "Viewing Expanded Context") : t("player.customBoundary")}</span>
          {onResetBoundary && (
            <button
              type="button"
              onClick={onResetBoundary}
              className="rounded-lg border border-[var(--line)] px-2 py-0.5 text-[11px] font-semibold hover:border-[var(--accent)] transition-colors"
            >
              {t("player.resetBoundary")}
            </button>
          )}
        </div>
      )}
    </div>
  );
};
