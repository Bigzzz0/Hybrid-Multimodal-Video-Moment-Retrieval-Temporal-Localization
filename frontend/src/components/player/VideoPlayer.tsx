"use client";

import React, { useRef, useEffect, useState, useCallback } from "react";
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
  Gauge,
  Sparkles,
  Tv,
} from "lucide-react";
import { TimelineHeatmap } from "./TimelineHeatmap";
import { GroundingEvidence, MomentItem, VideoKeyframeItem } from "@/lib/types";
import { apiClient } from "@/lib/api";

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
}) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [isMuted, setIsMuted] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1.0);
  const [isLoopActive, setIsLoopActive] = useState(false);
  const [maskEnabled, setMaskEnabled] = useState(true);
  const [groundingArtifact, setGroundingArtifact] = useState<any>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const lastTrackRef = useRef<string | null>(null);

  useEffect(() => {
    const evidence = groundingEvidence.find((item) => currentTime >= item.t_start - 0.25 && currentTime <= item.t_end + 0.25);
    if (!evidence || !maskEnabled) {
      setGroundingArtifact(null);
      lastTrackRef.current = null;
      return;
    }
    if (lastTrackRef.current === `${evidence.track_id}:${Math.floor(currentTime * 2)}`) return;
    lastTrackRef.current = `${evidence.track_id}:${Math.floor(currentTime * 2)}`;
    apiClient.getGroundingTrack(evidence.track_id, currentTime)
      .then(setGroundingArtifact)
      .catch(() => setGroundingArtifact(null));
  }, [currentTime, groundingEvidence, maskEnabled]);

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
    if (!groundingArtifact || !maskEnabled) return;
    const sourceWidth = Math.max(1, Number(groundingArtifact.frame_width || width));
    const sourceHeight = Math.max(1, Number(groundingArtifact.frame_height || height));
    const bbox = groundingArtifact.bbox_xyxy || [];
    if (bbox.length >= 4) {
      ctx.strokeStyle = "rgba(232,121,249,0.95)";
      ctx.lineWidth = 2;
      ctx.strokeRect((bbox[0] / sourceWidth) * width, (bbox[1] / sourceHeight) * height, ((bbox[2] - bbox[0]) / sourceWidth) * width, ((bbox[3] - bbox[1]) / sourceHeight) * height);
    }
    const rle = groundingArtifact.mask_rle || {};
    const size = rle.size || [];
    const counts = rle.counts || [];
    if (size.length < 2 || counts.length === 0) return;
    const maskWidth = Number(size[1]);
    const maskHeight = Number(size[0]);
    const image = ctx.createImageData(width, height);
    let cursor = 0;
    let filled = false;
    counts.forEach((run: number) => {
      const end = Math.min(maskWidth * maskHeight, cursor + Math.max(0, Number(run)));
      if (filled) {
        for (let index = cursor; index < end; index += 1) {
          const x = index % maskWidth;
          const y = Math.floor(index / maskWidth);
          const dx = Math.min(width - 1, Math.floor((x / maskWidth) * width));
          const dy = Math.min(height - 1, Math.floor((y / maskHeight) * height));
          const offset = (dy * width + dx) * 4;
          image.data[offset] = 217;
          image.data[offset + 1] = 70;
          image.data[offset + 2] = 239;
          image.data[offset + 3] = 70;
        }
      }
      cursor = end;
      filled = !filled;
    });
    ctx.putImageData(image, 0, 0);
  }, [groundingArtifact, maskEnabled]);

  // Handle external seek requests
  useEffect(() => {
    if (seekTime !== null && seekTime !== undefined && videoRef.current) {
      videoRef.current.currentTime = seekTime;
      videoRef.current.play().catch(() => {});
      setIsPlaying(true);
    }
  }, [seekTime]);

  // Apply playback rate
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

  // Time update listener with A-B Moment Loop Enforcement
  const handleTimeUpdate = () => {
    if (!videoRef.current) return;
    const curr = videoRef.current.currentTime;
    setCurrentTime(curr);

    // A-B Moment Loop Clamping
    if (isLoopActive && highlightInterval) {
      const [ts, te] = highlightInterval;
      // When reaching near end of highlighted moment, loop back to start
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

  // Step exact single frame forward/backward
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

  return (
    <div className="w-full glass-panel rounded-2xl p-4 space-y-3.5 shadow-2xl border border-surfaceBorder/80">
      {/* 16:9 Video Canvas Viewport */}
      <div className="relative aspect-video rounded-xl overflow-hidden bg-black border border-surfaceBorder group shadow-inner">
        <video
          ref={videoRef}
          src={streamUrl}
          onTimeUpdate={handleTimeUpdate}
          onPlay={() => setIsPlaying(true)}
          onPause={() => setIsPlaying(false)}
          className="w-full h-full object-contain"
        />

        <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none" aria-hidden="true" />

        {groundingEvidence.length > 0 && (
          <button
            type="button"
            onClick={() => setMaskEnabled((value) => !value)}
            className="absolute bottom-3 left-3 z-10 rounded-lg border border-fuchsia-400/50 bg-black/75 px-2.5 py-1 text-[10px] font-mono text-fuchsia-200"
          >
            {maskEnabled ? "ซ่อน SAM mask" : "แสดง SAM mask"}
          </button>
        )}

        {/* Video Overlay Play/Pause Action */}
        <div
          onClick={togglePlay}
          className="absolute inset-0 flex items-center justify-center bg-black/20 opacity-0 group-hover:opacity-100 transition-opacity cursor-pointer"
        >
          <div className="w-14 h-14 rounded-full bg-cyan-500/90 text-black flex items-center justify-center shadow-lg shadow-cyan-500/30 transform group-hover:scale-110 transition-transform">
            {isPlaying ? <Pause className="w-6 h-6 fill-current" /> : <Play className="w-6 h-6 ml-0.5 fill-current" />}
          </div>
        </div>

        {/* Active Highlight Event Banner */}
        {highlightInterval && (
          <div className="absolute top-3 left-3 right-3 flex items-center justify-between pointer-events-none">
            <span className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-black/80 backdrop-blur-md border border-cyan-500/60 text-xs font-mono font-semibold text-cyan-300 shadow-md">
              <Sparkles className="w-3.5 h-3.5 text-cyan-400 animate-pulse" />
              {contextExpanded ? "Context Window" : "Active Event"}: {highlightInterval[0].toFixed(1)}s - {highlightInterval[1].toFixed(1)}s
              ({(highlightInterval[1] - highlightInterval[0]).toFixed(1)}s)
            </span>

            {isLoopActive && (
              <span className="flex items-center gap-1 px-2.5 py-1 rounded-full bg-cyan-950/90 border border-cyan-500 text-[10px] font-mono font-bold text-cyan-200 animate-pulse">
                <Repeat className="w-3 h-3" /> A-B LOOP ACTIVE
              </span>
            )}
          </div>
        )}
      </div>

      {/* Visual relevance timeline with live hover scrubbing & drag handles */}
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

      {highlightInterval && (boundaryAdjusted || contextExpanded) && (
        <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-indigo-800/50 bg-indigo-950/20 px-3 py-2 text-[11px] text-indigo-200">
          <span>{contextExpanded ? "กำลังดูช่วงบริบท" : "Adjusted locally — ยังไม่เปลี่ยนผลค้นหา"}</span>
          {onResetBoundary && <button type="button" onClick={onResetBoundary} className="min-h-9 rounded-lg border border-indigo-700/60 px-2.5 py-1 font-semibold hover:bg-indigo-900/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400">Reset boundary</button>}
        </div>
      )}

      {/* Studio Control Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 px-1 pt-1 text-gray-300 text-xs">
        {/* Left Side: Playback, Frame Stepping, Time */}
        <div className="flex items-center gap-2 sm:gap-2.5">
          <button
            type="button"
            onClick={togglePlay}
            className="p-2 rounded-xl bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-300 border border-cyan-500/40 transition-colors shadow-sm"
            title={isPlaying ? "Pause (Space)" : "Play (Space)"}
          >
            {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 ml-0.5" />}
          </button>

          {/* Frame Steppers */}
          <div className="flex items-center gap-1 bg-surface border border-surfaceBorder rounded-xl p-0.5">
            <button
              type="button"
              onClick={() => handleStepFrame(-1)}
              title="Step Backward 1 Frame (,)"
              className="p-1.5 hover:bg-surfaceBorder text-gray-400 hover:text-white rounded-lg transition-colors"
            >
              <ChevronLeft className="w-3.5 h-3.5" />
            </button>
            <span className="text-[10px] font-mono text-gray-500 px-1">
              F:{Math.floor(currentTime * Math.max(10, fps))}
            </span>
            <button
              type="button"
              onClick={() => handleStepFrame(1)}
              title="Step Forward 1 Frame (.)"
              className="p-1.5 hover:bg-surfaceBorder text-gray-400 hover:text-white rounded-lg transition-colors"
            >
              <ChevronRight className="w-3.5 h-3.5" />
            </button>
          </div>

          <button
            type="button"
            onClick={() => handleSeek(Math.max(0, currentTime - 5))}
            title="Rewind 5s (J)"
            className="p-2 rounded-xl hover:bg-surface text-gray-400 hover:text-white transition-colors border border-transparent hover:border-surfaceBorder"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>

          <button
            type="button"
            onClick={() => {
              if (videoRef.current) {
                videoRef.current.muted = !isMuted;
                setIsMuted(!isMuted);
              }
            }}
            title={isMuted ? "Unmute" : "Mute"}
            className="p-2 rounded-xl hover:bg-surface text-gray-400 hover:text-white transition-colors border border-transparent hover:border-surfaceBorder"
          >
            {isMuted ? <VolumeX className="w-3.5 h-3.5" /> : <Volume2 className="w-3.5 h-3.5" />}
          </button>

          <span className="text-xs font-mono text-gray-400 ml-1">
            {formatTime(currentTime)} / {formatTime(duration)}
          </span>
        </div>

        {/* Right Side: A-B Loop Toggle, Speed Selector, Cinema Mode, Fullscreen */}
        <div className="flex items-center gap-2">
          {/* A-B Moment Loop Toggle */}
          {highlightInterval && (
            <button
              type="button"
              onClick={() => setIsLoopActive(!isLoopActive)}
              className={`px-2.5 py-1.5 rounded-xl text-xs font-mono font-semibold flex items-center gap-1.5 transition-all border ${
                isLoopActive
                  ? "bg-cyan-950 border-cyan-500 text-cyan-300 shadow-md shadow-cyan-500/20"
                  : "bg-surface hover:bg-surfaceBorder text-gray-400 hover:text-white border-surfaceBorder"
              }`}
              title="Continuously loop localized moment (R)"
            >
              <Repeat className={`w-3.5 h-3.5 ${isLoopActive ? "text-cyan-400 animate-spin" : ""}`} />
              <span>{isLoopActive ? "Loop ON" : "Loop Moment"}</span>
            </button>
          )}

          {/* Speed Selector */}
          <div className="flex items-center gap-1 bg-surface border border-surfaceBorder rounded-xl px-2 py-1 text-[11px] font-mono">
            <Gauge className="w-3 h-3 text-gray-400" />
            <select
              value={playbackSpeed}
              onChange={(e) => setPlaybackSpeed(parseFloat(e.target.value))}
              className="bg-transparent text-gray-300 font-semibold focus:outline-none cursor-pointer"
            >
              <option value="0.5" className="bg-surface text-white">0.5x</option>
              <option value="0.75" className="bg-surface text-white">0.75x</option>
              <option value="1.0" className="bg-surface text-white">1.0x</option>
              <option value="1.25" className="bg-surface text-white">1.25x</option>
              <option value="1.5" className="bg-surface text-white">1.5x</option>
              <option value="2.0" className="bg-surface text-white">2.0x</option>
            </select>
          </div>

          {/* Cinema Theater Mode Button */}
          {onToggleCinemaMode && (
            <button
              type="button"
              onClick={onToggleCinemaMode}
              className={`p-2 rounded-xl border transition-all ${
                isCinemaMode
                  ? "bg-cyan-950 border-cyan-500 text-cyan-300 shadow-md shadow-cyan-500/20"
                  : "hover:bg-surface text-gray-400 hover:text-white border-transparent hover:border-surfaceBorder"
              }`}
              title={isCinemaMode ? "Exit Cinema Mode (T)" : "Cinema Theater Mode (T)"}
            >
              <Tv className="w-3.5 h-3.5" />
            </button>
          )}

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
            title="Fullscreen"
            className="p-2 rounded-xl hover:bg-surface text-gray-400 hover:text-white transition-colors border border-transparent hover:border-surfaceBorder"
          >
            <Maximize className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </div>
  );
};
