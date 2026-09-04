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
import { MomentItem, VideoKeyframeItem } from "@/lib/types";

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
}) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [isMuted, setIsMuted] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1.0);
  const [isLoopActive, setIsLoopActive] = useState(false);

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
              Event Boundary: {highlightInterval[0].toFixed(1)}s - {highlightInterval[1].toFixed(1)}s
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

      {/* SOTA Spline Timeline Heatmap with Live Hover Scrubbing & Drag Handles */}
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
