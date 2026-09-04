"use client";

import React, { useRef, useEffect, useState, useCallback } from "react";
import { MomentItem, VideoKeyframeItem } from "@/lib/types";
import { apiClient } from "@/lib/api";
import { Flame, Sparkles } from "lucide-react";

interface TimelineHeatmapProps {
  heatmapData: number[]; // 1 value per second [0.0 - 1.0]
  duration: number;
  currentTime: number;
  highlightInterval?: [number, number] | null;
  moments?: MomentItem[];
  keyframeRecords?: VideoKeyframeItem[];
  onSeek: (time: number) => void;
}

interface HoverState {
  isActive: boolean;
  x: number;
  hoverTime: number;
  score: number;
  frameUrl: string | null;
}

export const TimelineHeatmap: React.FC<TimelineHeatmapProps> = ({
  heatmapData,
  duration,
  currentTime,
  highlightInterval,
  moments = [],
  keyframeRecords = [],
  onSeek,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [hoverState, setHoverState] = useState<HoverState>({
    isActive: false,
    x: 0,
    hoverTime: 0,
    score: 0,
    frameUrl: null,
  });

  // Render High-DPI Smooth Continuous Spline Waveform
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const width = rect.width;
    const height = 44; // height in CSS pixels

    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    // Clear background
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#090d18";
    ctx.fillRect(0, 0, width, height);

    // Baseline grid rule
    ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, height - 1);
    ctx.lineTo(width, height - 1);
    ctx.stroke();

    if (!heatmapData || heatmapData.length === 0 || duration <= 0) {
      // Empty state dotted guide
      ctx.strokeStyle = "rgba(99, 102, 241, 0.2)";
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(0, height / 2);
      ctx.lineTo(width, height / 2);
      ctx.stroke();
      ctx.setLineDash([]);
      return;
    }

    const n = heatmapData.length;
    const stepX = width / Math.max(1, n - 1);

    // 1. Calculate Waveform Coordinates (x, y)
    const points: { x: number; y: number; val: number }[] = [];
    for (let i = 0; i < n; i++) {
      const rawVal = heatmapData[i] || 0;
      const val = Math.max(0.04, Math.min(1.0, rawVal));
      const x = i * stepX;
      const y = height - (val * (height - 6));
      points.push({ x, y, val });
    }

    // 2. Draw Area Gradient with Cubic Bezier Spline
    const areaGradient = ctx.createLinearGradient(0, 0, 0, height);
    areaGradient.addColorStop(0, "rgba(0, 240, 255, 0.55)");
    areaGradient.addColorStop(0.5, "rgba(99, 102, 241, 0.25)");
    areaGradient.addColorStop(1, "rgba(99, 102, 241, 0.02)");

    ctx.beginPath();
    ctx.moveTo(0, height);
    ctx.lineTo(points[0].x, points[0].y);

    for (let i = 0; i < points.length - 1; i++) {
      const p0 = points[i];
      const p1 = points[i + 1];
      const cpx1 = p0.x + (p1.x - p0.x) / 2.2;
      const cpy1 = p0.y;
      const cpx2 = p0.x + (p1.x - p0.x) / 2.2;
      const cpy2 = p1.y;
      ctx.bezierCurveTo(cpx1, cpy1, cpx2, cpy2, p1.x, p1.y);
    }

    ctx.lineTo(width, height);
    ctx.closePath();
    ctx.fillStyle = areaGradient;
    ctx.fill();

    // 3. Draw Neon Glowing Spline Stroke
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 0; i < points.length - 1; i++) {
      const p0 = points[i];
      const p1 = points[i + 1];
      const cpx1 = p0.x + (p1.x - p0.x) / 2.2;
      const cpy1 = p0.y;
      const cpx2 = p0.x + (p1.x - p0.x) / 2.2;
      const cpy2 = p1.y;
      ctx.bezierCurveTo(cpx1, cpy1, cpx2, cpy2, p1.x, p1.y);
    }
    ctx.strokeStyle = "#00f0ff";
    ctx.lineWidth = 2;
    ctx.shadowColor = "#00f0ff";
    ctx.shadowBlur = 6;
    ctx.stroke();
    ctx.shadowBlur = 0; // reset

    // 4. Draw Highlight Interval [ts, te] with Glowing Bracket
    if (highlightInterval && duration > 0) {
      const [ts, te] = highlightInterval;
      const x1 = Math.max(0, (ts / duration) * width);
      const x2 = Math.min(width, (te / duration) * width);
      const w = Math.max(8, x2 - x1);

      // Interval background
      ctx.fillStyle = "rgba(0, 240, 255, 0.18)";
      ctx.fillRect(x1, 0, w, height);

      // Top & bottom border
      ctx.strokeStyle = "rgba(0, 240, 255, 0.9)";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(x1, 0, w, height);

      // Accent start and end boundary notches
      ctx.fillStyle = "#00f0ff";
      ctx.fillRect(x1 - 1.5, 0, 3, height);
      ctx.fillRect(x2 - 1.5, 0, 3, height);
    }

    // 5. Draw Playhead Needle
    if (duration > 0) {
      const playheadX = (currentTime / duration) * width;
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.shadowColor = "rgba(255, 255, 255, 0.8)";
      ctx.shadowBlur = 5;
      ctx.beginPath();
      ctx.moveTo(playheadX, 0);
      ctx.lineTo(playheadX, height);
      ctx.stroke();
      ctx.shadowBlur = 0;

      // Top handle circle
      ctx.fillStyle = "#ffffff";
      ctx.beginPath();
      ctx.arc(playheadX, 3, 3.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }, [heatmapData, duration, currentTime, highlightInterval]);

  // Handle Hover Scrubbing & Find Nearest Keyframe Thumbnail
  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      const canvas = canvasRef.current;
      if (!canvas || duration <= 0) return;

      const rect = canvas.getBoundingClientRect();
      const clientX = e.clientX - rect.left;
      const ratio = Math.max(0, Math.min(1, clientX / rect.width));
      const hoverTime = ratio * duration;

      // Get relevance score at hover time
      const secIdx = Math.floor(hoverTime);
      const score = heatmapData[secIdx] || 0;

      // Find nearest keyframe image
      let nearestFrameUrl: string | null = null;
      if (keyframeRecords.length > 0) {
        let bestDiff = Infinity;
        for (const kf of keyframeRecords) {
          const diff = Math.abs(kf.timestamp - hoverTime);
          if (diff < bestDiff) {
            bestDiff = diff;
            nearestFrameUrl = kf.thumbnail_url || (kf.frame_path ? apiClient.getFramePreviewUrl(kf.frame_path) : null);
          }
        }
      }

      setHoverState({
        isActive: true,
        x: clientX,
        hoverTime,
        score,
        frameUrl: nearestFrameUrl,
      });
    },
    [duration, heatmapData, keyframeRecords]
  );

  const handleMouseLeave = () => {
    setHoverState((prev) => ({ ...prev, isActive: false }));
  };

  const handleClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || duration <= 0) return;
    const rect = canvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const ratio = Math.max(0, Math.min(1, clickX / rect.width));
    onSeek(ratio * duration);
  };

  const peakScore = heatmapData.length > 0 ? Math.max(...heatmapData) : 0;

  const formatSeconds = (sec: number) => {
    const m = Math.floor(sec / 60);
    const s = (sec % 60).toFixed(1);
    return `${m.toString().padStart(2, "0")}:${Number(s) < 10 ? "0" : ""}${s}s`;
  };

  return (
    <div className="w-full space-y-1.5 select-none relative" ref={containerRef}>
      {/* Header Info Bar */}
      <div className="flex items-center justify-between text-xs text-gray-400 font-medium px-1">
        <div className="flex items-center gap-2">
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
            <span className="text-gray-200 font-semibold">Continuous Spline Waveform</span>
          </span>
          {heatmapData.length > 0 && (
            <span className="text-[10px] font-mono font-bold px-2 py-0.5 rounded-full bg-cyan-950 text-cyan-300 border border-cyan-800">
              Peak: {(peakScore * 100).toFixed(0)}%
            </span>
          )}
        </div>

        <span className="font-mono text-gray-300 text-xs">
          <b className="text-cyan-400 font-bold">{currentTime.toFixed(1)}s</b> / {duration.toFixed(1)}s
        </span>
      </div>

      {/* Top-K Moment Anchor Pins Layer */}
      {moments.length > 0 && duration > 0 && (
        <div className="relative w-full h-5">
          {moments.slice(0, 3).map((m, idx) => {
            const peakT = (m.t_start + m.t_end) / 2;
            const pct = Math.max(2, Math.min(96, (peakT / duration) * 100));

            return (
              <button
                key={idx}
                type="button"
                onClick={() => onSeek(m.t_start)}
                style={{ left: `${pct}%` }}
                title={`Jump to Moment #${idx + 1} (${m.t_start.toFixed(1)}s - ${m.t_end.toFixed(1)}s)`}
                className="absolute -translate-x-1/2 top-0 px-2 py-0.5 rounded-full bg-cyan-950/90 hover:bg-cyan-900 border border-cyan-500/80 text-[10px] font-mono font-bold text-cyan-300 shadow-md shadow-cyan-500/20 hover:scale-110 transition-all flex items-center gap-1 z-20"
              >
                <Sparkles className="w-2.5 h-2.5 text-cyan-400" />
                <span>#{idx + 1}</span>
              </button>
            );
          })}
        </div>
      )}

      {/* Canvas Timeline Viewport with Hover Engine */}
      <div
        className="relative w-full cursor-pointer rounded-xl overflow-visible group"
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        onClick={handleClick}
      >
        <canvas
          ref={canvasRef}
          className="w-full h-11 rounded-xl border border-surfaceBorder group-hover:border-cyan-500/50 transition-colors shadow-inner"
        />

        {/* Live Hover Scrubber Tooltip */}
        {hoverState.isActive && (
          <div
            style={{ left: `${hoverState.x}px` }}
            className="absolute bottom-14 -translate-x-1/2 pointer-events-none z-30 animate-in fade-in zoom-in-95 duration-100"
          >
            <div className="glass-tooltip p-2 rounded-xl text-center space-y-1.5 min-w-[130px] border border-cyan-500/50">
              {/* Miniature Keyframe Image Thumbnail */}
              {hoverState.frameUrl ? (
                <div className="w-28 h-16 mx-auto rounded-lg overflow-hidden bg-black/80 border border-surfaceBorder/60">
                  <img
                    src={hoverState.frameUrl}
                    alt="Scrubbing Keyframe"
                    className="w-full h-full object-cover"
                  />
                </div>
              ) : null}

              {/* Timestamp & Score */}
              <div className="font-mono text-[11px] font-bold text-white">
                {formatSeconds(hoverState.hoverTime)}
              </div>

              {hoverState.score > 0 && (
                <div className="text-[10px] font-mono text-cyan-300 bg-cyan-950/80 px-1.5 py-0.5 rounded border border-cyan-800">
                  Match: {(hoverState.score * 100).toFixed(1)}%
                </div>
              )}
            </div>

            {/* Downward Pointer Triangle */}
            <div className="w-2 h-2 bg-[#060911] border-r border-b border-cyan-500/50 rotate-45 mx-auto -mt-1" />
          </div>
        )}
      </div>
    </div>
  );
};
