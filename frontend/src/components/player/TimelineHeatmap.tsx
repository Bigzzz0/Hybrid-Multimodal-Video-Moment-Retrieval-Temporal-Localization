"use client";

import React, { useRef, useEffect, useState, useCallback } from "react";
import { MomentItem, VideoKeyframeItem, DragHandleState, TimelineZoomState } from "@/lib/types";
import { apiClient } from "@/lib/api";
import { Sparkles, ZoomIn, ZoomOut, Activity, Flame } from "lucide-react";

interface TimelineHeatmapProps {
  heatmapData: number[]; // 1 value per second [0.0 - 1.0]
  motionData?: number[]; // pixel motion values
  duration: number;
  currentTime: number;
  highlightInterval?: [number, number] | null;
  moments?: MomentItem[];
  keyframeRecords?: VideoKeyframeItem[];
  onSeek: (time: number) => void;
  onBoundaryChange?: (newStart: number, newEnd: number) => void;
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
  motionData = [],
  duration,
  currentTime,
  highlightInterval,
  moments = [],
  keyframeRecords = [],
  onSeek,
  onBoundaryChange,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  const [zoom, setZoom] = useState<TimelineZoomState>({ scale: 1.0, scrollOffsetSec: 0 });
  const [showMotionLayer, setShowMotionLayer] = useState(true);

  const [hoverState, setHoverState] = useState<HoverState>({
    isActive: false,
    x: 0,
    hoverTime: 0,
    score: 0,
    frameUrl: null,
  });

  const [dragState, setDragState] = useState<DragHandleState>({
    isDragging: false,
    handleType: null,
    initialX: 0,
    initialInterval: [0, 0],
  });

  // Convert Time (seconds) to Canvas Coordinate X
  const timeToX = useCallback(
    (timeSec: number, width: number): number => {
      if (duration <= 0) return 0;
      const visibleDuration = duration / zoom.scale;
      const relSec = timeSec - zoom.scrollOffsetSec;
      return (relSec / visibleDuration) * width;
    },
    [duration, zoom]
  );

  // Convert Canvas Coordinate X to Time (seconds)
  const xToTime = useCallback(
    (x: number, width: number): number => {
      if (width <= 0 || duration <= 0) return 0;
      const visibleDuration = duration / zoom.scale;
      const rawTime = zoom.scrollOffsetSec + (x / width) * visibleDuration;
      return Math.max(0, Math.min(duration, rawTime));
    },
    [duration, zoom]
  );

  // Render High-DPI Smooth Continuous Spline Waveform & Dual-Layers
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const width = rect.width;
    const height = 48; // Canvas height in CSS pixels

    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    // 1. Clear background
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#080c16";
    ctx.fillRect(0, 0, width, height);

    // 2. Subtle Baseline Grid Rule
    ctx.strokeStyle = "rgba(255, 255, 255, 0.06)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, height - 1);
    ctx.lineTo(width, height - 1);
    ctx.stroke();

    if (!heatmapData || heatmapData.length === 0 || duration <= 0) {
      // Empty state guide
      ctx.strokeStyle = "rgba(99, 102, 241, 0.25)";
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

    // 3. Layer 2: Optional Pixel Motion Energy Curve (Amber dashed stroke)
    if (showMotionLayer && motionData && motionData.length > 0) {
      const motStep = width / Math.max(1, motionData.length - 1);
      ctx.beginPath();
      for (let i = 0; i < motionData.length; i++) {
        const motVal = Math.max(0, Math.min(1.0, motionData[i] || 0));
        const mx = i * motStep;
        const my = height - (motVal * (height - 8));
        if (i === 0) ctx.moveTo(mx, my);
        else ctx.lineTo(mx, my);
      }
      ctx.strokeStyle = "rgba(245, 158, 11, 0.55)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 3]);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // 4. Layer 1: Semantic Relevance Spline Points
    const points: { x: number; y: number; val: number }[] = [];
    for (let i = 0; i < n; i++) {
      const rawVal = heatmapData[i] || 0;
      const val = Math.max(0.04, Math.min(1.0, rawVal));
      const x = i * stepX;
      const y = height - (val * (height - 8));
      points.push({ x, y, val });
    }

    // 5. Draw Area Gradient with Cubic Bezier Spline
    const areaGradient = ctx.createLinearGradient(0, 0, 0, height);
    areaGradient.addColorStop(0, "rgba(0, 240, 255, 0.50)");
    areaGradient.addColorStop(0.5, "rgba(99, 102, 241, 0.20)");
    areaGradient.addColorStop(1, "rgba(99, 102, 241, 0.01)");

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

    // 6. Draw Glowing Spline Stroke
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
    ctx.lineWidth = 2.2;
    ctx.shadowColor = "#00f0ff";
    ctx.shadowBlur = 6;
    ctx.stroke();
    ctx.shadowBlur = 0; // reset glow

    // 7. Keyframe Position Ticks (Bottom Notch Marks)
    if (keyframeRecords && keyframeRecords.length > 0) {
      ctx.fillStyle = "rgba(255, 255, 255, 0.25)";
      for (const kf of keyframeRecords) {
        const kx = timeToX(kf.timestamp, width);
        if (kx >= 0 && kx <= width) {
          ctx.fillRect(kx - 0.5, height - 4, 1, 4);
        }
      }
    }

    // 8. Highlight Interval with Interactive Drag Handles
    if (highlightInterval && duration > 0) {
      const [ts, te] = highlightInterval;
      const x1 = timeToX(ts, width);
      const x2 = timeToX(te, width);
      const w = Math.max(6, x2 - x1);

      // Translucent event fill
      ctx.fillStyle = "rgba(0, 240, 255, 0.18)";
      ctx.fillRect(x1, 0, w, height);

      // Border outline
      ctx.strokeStyle = "rgba(0, 240, 255, 0.95)";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(x1, 0, w, height);

      // Start Handle (Left Bracket)
      ctx.fillStyle = "#00f0ff";
      ctx.fillRect(x1 - 2, 0, 4, height);
      // Grip Handle Dot
      ctx.fillStyle = "#ffffff";
      ctx.beginPath();
      ctx.arc(x1, height / 2, 3, 0, Math.PI * 2);
      ctx.fill();

      // End Handle (Right Bracket)
      ctx.fillStyle = "#00f0ff";
      ctx.fillRect(x2 - 2, 0, 4, height);
      // Grip Handle Dot
      ctx.fillStyle = "#ffffff";
      ctx.beginPath();
      ctx.arc(x2, height / 2, 3, 0, Math.PI * 2);
      ctx.fill();
    }

    // 9. Playhead Needle
    if (duration > 0) {
      const playheadX = timeToX(currentTime, width);
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.shadowColor = "rgba(255, 255, 255, 0.9)";
      ctx.shadowBlur = 6;
      ctx.beginPath();
      ctx.moveTo(playheadX, 0);
      ctx.lineTo(playheadX, height);
      ctx.stroke();
      ctx.shadowBlur = 0;

      // Handle Head
      ctx.fillStyle = "#ffffff";
      ctx.beginPath();
      ctx.arc(playheadX, 3.5, 3.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }, [heatmapData, motionData, duration, currentTime, highlightInterval, zoom, showMotionLayer, keyframeRecords, timeToX]);

  // Mouse Move Listener (Hover tooltip + Drag execution)
  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      const canvas = canvasRef.current;
      if (!canvas || duration <= 0) return;

      const rect = canvas.getBoundingClientRect();
      const clientX = e.clientX - rect.left;
      const hoverTime = xToTime(clientX, rect.width);

      // If actively dragging boundary handles
      if (dragState.isDragging && onBoundaryChange && highlightInterval) {
        const deltaSec = (e.clientX - dragState.initialX) / rect.width * (duration / zoom.scale);
        const [initStart, initEnd] = dragState.initialInterval;

        let newStart = initStart;
        let newEnd = initEnd;

        if (dragState.handleType === "start") {
          newStart = Math.max(0, Math.min(initEnd - 0.5, initStart + deltaSec));
        } else if (dragState.handleType === "end") {
          newEnd = Math.min(duration, Math.max(initStart + 0.5, initEnd + deltaSec));
        } else if (dragState.handleType === "middle") {
          const span = initEnd - initStart;
          newStart = Math.max(0, Math.min(duration - span, initStart + deltaSec));
          newEnd = newStart + span;
        }

        // Magnetic Snap to keyframes if within 0.3s
        for (const kf of keyframeRecords) {
          if (dragState.handleType === "start" && Math.abs(kf.timestamp - newStart) <= 0.3) {
            newStart = kf.timestamp;
          }
          if (dragState.handleType === "end" && Math.abs(kf.timestamp - newEnd) <= 0.3) {
            newEnd = kf.timestamp;
          }
        }

        onBoundaryChange(roundTo(newStart, 2), roundTo(newEnd, 2));
        return;
      }

      // Update Cursor based on proximity to drag handles
      if (highlightInterval && containerRef.current) {
        const [ts, te] = highlightInterval;
        const x1 = timeToX(ts, rect.width);
        const x2 = timeToX(te, rect.width);

        if (Math.abs(clientX - x1) <= 7 || Math.abs(clientX - x2) <= 7) {
          containerRef.current.style.cursor = "ew-resize";
        } else if (clientX > x1 + 7 && clientX < x2 - 7) {
          containerRef.current.style.cursor = "grab";
        } else {
          containerRef.current.style.cursor = "pointer";
        }
      }

      // Relevance score at hover time
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
    [duration, zoom, dragState, highlightInterval, onBoundaryChange, xToTime, timeToX, heatmapData, keyframeRecords]
  );

  const handleMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !highlightInterval || duration <= 0) return;

    const rect = canvas.getBoundingClientRect();
    const clientX = e.clientX - rect.left;
    const [ts, te] = highlightInterval;
    const x1 = timeToX(ts, rect.width);
    const x2 = timeToX(te, rect.width);

    if (Math.abs(clientX - x1) <= 7) {
      // Dragging Start Handle
      setDragState({
        isDragging: true,
        handleType: "start",
        initialX: e.clientX,
        initialInterval: [ts, te],
      });
    } else if (Math.abs(clientX - x2) <= 7) {
      // Dragging End Handle
      setDragState({
        isDragging: true,
        handleType: "end",
        initialX: e.clientX,
        initialInterval: [ts, te],
      });
    } else if (clientX > x1 + 7 && clientX < x2 - 7) {
      // Sliding the entire interval
      setDragState({
        isDragging: true,
        handleType: "middle",
        initialX: e.clientX,
        initialInterval: [ts, te],
      });
    } else {
      // Simple seek click
      const clickTime = xToTime(clientX, rect.width);
      onSeek(clickTime);
    }
  };

  const handleMouseUp = () => {
    if (dragState.isDragging) {
      setDragState((prev) => ({ ...prev, isDragging: false, handleType: null }));
    }
  };

  // Window mouseup listener for drag release outside canvas
  useEffect(() => {
    const handleGlobalMouseUp = () => {
      if (dragState.isDragging) {
        setDragState((prev) => ({ ...prev, isDragging: false, handleType: null }));
      }
    };
    window.addEventListener("mouseup", handleGlobalMouseUp);
    return () => window.removeEventListener("mouseup", handleGlobalMouseUp);
  }, [dragState.isDragging]);

  const handleMouseLeave = () => {
    if (!dragState.isDragging) {
      setHoverState((prev) => ({ ...prev, isActive: false }));
    }
  };

  const peakScore = heatmapData.length > 0 ? Math.max(...heatmapData) : 0;

  const formatSeconds = (sec: number) => {
    const m = Math.floor(sec / 60);
    const s = (sec % 60).toFixed(2);
    return `${m.toString().padStart(2, "0")}:${Number(s) < 10 ? "0" : ""}${s}`;
  };

  const handleZoom = (level: number) => {
    setZoom((prev) => ({
      scale: level,
      scrollOffsetSec: level === 1.0 ? 0 : Math.max(0, Math.min(duration - duration / level, currentTime - (duration / level) / 2)),
    }));
  };

  return (
    <div className="w-full space-y-1.5 select-none relative" ref={containerRef}>
      {/* Header Info Bar */}
      <div className="flex items-center justify-between text-xs text-gray-400 font-medium px-1">
        <div className="flex items-center gap-2">
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
            <span className="text-gray-200 font-semibold text-xs">Visual Relevance Heatmap</span>
          </span>

          {/* Toggle Pixel Motion Curve */}
          <button
            type="button"
            onClick={() => setShowMotionLayer(!showMotionLayer)}
            className={`px-2 py-0.5 rounded text-[10px] font-mono flex items-center gap-1 border transition-colors ${
              showMotionLayer
                ? "bg-amber-950/60 border-amber-600/50 text-amber-300"
                : "bg-surface border-surfaceBorder text-gray-500 hover:text-gray-300"
            }`}
            title="Toggle Pixel Motion Energy Layer ΔP(t)"
          >
            <Activity className="w-2.5 h-2.5" />
            <span>Motion ΔP</span>
          </button>

          {/* Zoom Buttons */}
          <div className="flex items-center gap-1 bg-surface border border-surfaceBorder rounded-lg p-0.5">
            <button
              type="button"
              onClick={() => handleZoom(1.0)}
              className={`px-1.5 py-0.5 rounded text-[10px] font-mono font-bold transition-colors ${
                zoom.scale === 1.0 ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40" : "text-gray-500 hover:text-gray-300"
              }`}
            >
              1x
            </button>
            <button
              type="button"
              onClick={() => handleZoom(2.0)}
              className={`px-1.5 py-0.5 rounded text-[10px] font-mono font-bold transition-colors ${
                zoom.scale === 2.0 ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40" : "text-gray-500 hover:text-gray-300"
              }`}
            >
              2x
            </button>
            <button
              type="button"
              onClick={() => handleZoom(4.0)}
              className={`px-1.5 py-0.5 rounded text-[10px] font-mono font-bold transition-colors ${
                zoom.scale === 4.0 ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40" : "text-gray-500 hover:text-gray-300"
              }`}
            >
              4x
            </button>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {peakScore > 0 && (
            <span className="text-[10px] font-mono font-bold px-2 py-0.5 rounded-full bg-cyan-950 text-cyan-300 border border-cyan-800">
              Peak: {(peakScore * 100).toFixed(0)}%
            </span>
          )}
          <span className="font-mono text-gray-300 text-xs">
            <b className="text-cyan-400 font-bold">{currentTime.toFixed(1)}s</b> / {duration.toFixed(1)}s
          </span>
        </div>
      </div>

      {/* Top-K Moment Anchor Pins Layer */}
      {moments.length > 0 && duration > 0 && (
        <div className="relative w-full h-5">
          {moments.slice(0, 3).map((m, idx) => {
            const peakT = (m.t_start + m.t_end) / 2;
            const px = timeToX(peakT, containerRef.current?.clientWidth || 500);
            const wWidth = containerRef.current?.clientWidth || 500;
            const pct = Math.max(2, Math.min(96, (px / wWidth) * 100));

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

      {/* Canvas Timeline Viewport with Hover Engine & Drag Handles */}
      <div
        className="relative w-full rounded-xl overflow-visible group"
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
      >
        <canvas
          ref={canvasRef}
          className="w-full h-12 rounded-xl border border-surfaceBorder group-hover:border-cyan-500/50 transition-colors shadow-inner"
        />

        {/* Live Hover Scrubbing Tooltip with Frame Thumbnail Preview */}
        {hoverState.isActive && (
          <div
            className="absolute -top-28 pointer-events-none z-30 transition-opacity duration-150 animate-in fade-in"
            style={{
              left: `${Math.max(64, Math.min((containerRef.current?.clientWidth || 600) - 64, hoverState.x))}px`,
              transform: "translateX(-50%)",
            }}
          >
            <div className="glass-tooltip p-1.5 rounded-xl flex flex-col items-center gap-1 shadow-2xl border border-cyan-500/40 w-28 text-center">
              {hoverState.frameUrl ? (
                <div className="w-full h-14 rounded-lg overflow-hidden bg-black border border-white/10 relative">
                  <img
                    src={hoverState.frameUrl}
                    alt="Keyframe Preview"
                    className="w-full h-full object-cover"
                    loading="eager"
                  />
                  {hoverState.score > 0 && (
                    <span className="absolute bottom-0.5 right-0.5 text-[9px] font-mono font-bold bg-black/80 px-1 py-0.2 rounded text-cyan-300">
                      {(hoverState.score * 100).toFixed(0)}%
                    </span>
                  )}
                </div>
              ) : (
                <div className="w-full h-14 rounded-lg bg-surface/80 flex items-center justify-center text-[10px] text-gray-500 font-mono">
                  No Thumbnail
                </div>
              )}
              <span className="font-mono text-[11px] font-bold text-white tracking-wider">
                {formatSeconds(hoverState.hoverTime)}
              </span>
            </div>
            {/* Tooltip caret arrow */}
            <div className="w-2.5 h-2.5 bg-[#060911] border-r border-b border-cyan-500/40 rotate-45 mx-auto -mt-1" />
          </div>
        )}
      </div>

      {/* Drag Instruction Cue */}
      {highlightInterval && (
        <div className="flex items-center justify-between text-[10px] font-mono text-gray-500 px-1">
          <span className="flex items-center gap-1">
            <span className="text-cyan-400">◄ Drag Brackets ►</span> to fine-tune event boundary
          </span>
          <span className="text-gray-400 font-medium">
            Active Event: {highlightInterval[0].toFixed(1)}s - {highlightInterval[1].toFixed(1)}s ({(highlightInterval[1] - highlightInterval[0]).toFixed(1)}s)
          </span>
        </div>
      )}
    </div>
  );
};

function roundTo(val: number, decimals: number = 2): number {
  const factor = Math.pow(10, decimals);
  return Math.round(val * factor) / factor;
}
