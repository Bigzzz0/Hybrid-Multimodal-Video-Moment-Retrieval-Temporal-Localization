"use client";

import React, { useRef, useEffect, useState, useCallback } from "react";
import { MomentItem, VideoKeyframeItem, DragHandleState, TimelineZoomState } from "@/lib/types";
import { apiClient } from "@/lib/api";
import { Sparkles, Activity } from "lucide-react";

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
  const [showMotionLayer, setShowMotionLayer] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(false);

  // Detect and observe theme switching (Light / Dark)
  useEffect(() => {
    const updateTheme = () => {
      setIsDarkMode(document.documentElement.classList.contains("dark"));
    };
    updateTheme();

    const observer = new MutationObserver(() => {
      updateTheme();
    });
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    return () => observer.disconnect();
  }, []);

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

  // Render Theme-Adaptive Spline Waveform Heatmap
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

    // Color theme values
    const canvasBg = isDarkMode ? "#141A22" : "#F4F6F9";
    const gridLine = isDarkMode ? "rgba(255, 255, 255, 0.08)" : "rgba(0, 0, 0, 0.06)";
    const splineColor = isDarkMode ? "#5CA9FF" : "#0A63C9";
    const playheadColor = isDarkMode ? "#FFFFFF" : "#12171F";
    const kfTickColor = isDarkMode ? "rgba(255, 255, 255, 0.25)" : "rgba(0, 0, 0, 0.2)";

    // 1. Clear & fill background according to current theme
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = canvasBg;
    ctx.fillRect(0, 0, width, height);

    // 2. Subtle Baseline Grid Rule
    ctx.strokeStyle = gridLine;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, height - 1);
    ctx.lineTo(width, height - 1);
    ctx.stroke();

    if (!heatmapData || heatmapData.length === 0 || duration <= 0) {
      ctx.strokeStyle = isDarkMode ? "rgba(92, 169, 255, 0.3)" : "rgba(10, 99, 201, 0.3)";
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

    // 3. Optional Pixel Motion Energy Curve
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
      ctx.strokeStyle = "rgba(245, 158, 11, 0.7)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 3]);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // 4. Semantic Relevance Spline Points
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
    if (isDarkMode) {
      areaGradient.addColorStop(0, "rgba(92, 169, 255, 0.40)");
      areaGradient.addColorStop(0.5, "rgba(92, 169, 255, 0.15)");
      areaGradient.addColorStop(1, "rgba(92, 169, 255, 0.01)");
    } else {
      areaGradient.addColorStop(0, "rgba(10, 99, 201, 0.35)");
      areaGradient.addColorStop(0.5, "rgba(10, 99, 201, 0.12)");
      areaGradient.addColorStop(1, "rgba(10, 99, 201, 0.01)");
    }

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

    // 6. Draw Spline Stroke
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
    ctx.strokeStyle = splineColor;
    ctx.lineWidth = 2.2;
    ctx.shadowColor = splineColor;
    ctx.shadowBlur = isDarkMode ? 6 : 2;
    ctx.stroke();
    ctx.shadowBlur = 0; // reset glow

    // 7. Keyframe Position Ticks (Bottom Notch Marks)
    if (keyframeRecords && keyframeRecords.length > 0) {
      ctx.fillStyle = kfTickColor;
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
      ctx.fillStyle = isDarkMode ? "rgba(92, 169, 255, 0.22)" : "rgba(10, 99, 201, 0.16)";
      ctx.fillRect(x1, 0, w, height);

      // Border outline
      ctx.strokeStyle = splineColor;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(x1, 0, w, height);

      // Start Handle (Left Bracket)
      ctx.fillStyle = splineColor;
      ctx.fillRect(x1 - 2, 0, 4, height);
      ctx.fillStyle = "#ffffff";
      ctx.beginPath();
      ctx.arc(x1, height / 2, 3, 0, Math.PI * 2);
      ctx.fill();

      // End Handle (Right Bracket)
      ctx.fillStyle = splineColor;
      ctx.fillRect(x2 - 2, 0, 4, height);
      ctx.fillStyle = "#ffffff";
      ctx.beginPath();
      ctx.arc(x2, height / 2, 3, 0, Math.PI * 2);
      ctx.fill();
    }

    // 9. Playhead Needle
    if (duration > 0) {
      const playheadX = timeToX(currentTime, width);
      ctx.strokeStyle = playheadColor;
      ctx.lineWidth = 2;
      ctx.shadowColor = isDarkMode ? "rgba(255, 255, 255, 0.8)" : "rgba(0, 0, 0, 0.3)";
      ctx.shadowBlur = 4;
      ctx.beginPath();
      ctx.moveTo(playheadX, 0);
      ctx.lineTo(playheadX, height);
      ctx.stroke();
      ctx.shadowBlur = 0;

      // Handle Head
      ctx.fillStyle = playheadColor;
      ctx.beginPath();
      ctx.arc(playheadX, 3.5, 3.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }, [
    heatmapData,
    motionData,
    duration,
    currentTime,
    highlightInterval,
    zoom,
    showMotionLayer,
    keyframeRecords,
    timeToX,
    isDarkMode,
  ]);

  // Mouse Move Listener
  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      const canvas = canvasRef.current;
      if (!canvas || duration <= 0) return;

      const rect = canvas.getBoundingClientRect();
      const clientX = e.clientX - rect.left;
      const hoverTime = xToTime(clientX, rect.width);

      if (dragState.isDragging && onBoundaryChange && highlightInterval) {
        const deltaSec = ((e.clientX - dragState.initialX) / rect.width) * (duration / zoom.scale);
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

      const secIdx = Math.floor(hoverTime);
      const score = heatmapData[secIdx] || 0;

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
    if (!canvas || duration <= 0) return;

    const rect = canvas.getBoundingClientRect();
    const clientX = e.clientX - rect.left;
    if (!highlightInterval) {
      onSeek(xToTime(clientX, rect.width));
      return;
    }
    const [ts, te] = highlightInterval;
    const x1 = timeToX(ts, rect.width);
    const x2 = timeToX(te, rect.width);

    if (Math.abs(clientX - x1) <= 7) {
      setDragState({
        isDragging: true,
        handleType: "start",
        initialX: e.clientX,
        initialInterval: [ts, te],
      });
    } else if (Math.abs(clientX - x2) <= 7) {
      setDragState({
        isDragging: true,
        handleType: "end",
        initialX: e.clientX,
        initialInterval: [ts, te],
      });
    } else if (clientX > x1 + 7 && clientX < x2 - 7) {
      setDragState({
        isDragging: true,
        handleType: "middle",
        initialX: e.clientX,
        initialInterval: [ts, te],
      });
    } else {
      const clickTime = xToTime(clientX, rect.width);
      onSeek(clickTime);
    }
  };

  const handleMouseUp = () => {
    if (dragState.isDragging) {
      setDragState((prev) => ({ ...prev, isDragging: false, handleType: null }));
    }
  };

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
      <div className="flex items-center justify-between text-xs font-medium px-1">
        <div className="flex items-center gap-2">
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-[var(--accent)] animate-pulse" />
            <span className="text-[var(--text)] font-semibold text-xs">Visual Relevance Heatmap</span>
          </span>

          {/* Toggle Pixel Motion Curve */}
          <button
            type="button"
            onClick={() => setShowMotionLayer(!showMotionLayer)}
            aria-pressed={showMotionLayer}
            aria-label="Toggle motion energy layer"
            className={`px-2 py-0.5 rounded-lg text-[10px] font-mono flex items-center gap-1 border transition-colors ${
              showMotionLayer
                ? "bg-amber-500/15 border-amber-500/40 text-amber-600 dark:text-amber-400 font-semibold"
                : "bg-[var(--bg)] border-[var(--line)] text-[var(--muted)] hover:text-[var(--text)]"
            }`}
            title="Motion energy helps refine boundaries; it is not semantic relevance"
          >
            <Activity className="w-2.5 h-2.5" />
            <span>Motion ΔP</span>
          </button>

          {/* Zoom Buttons */}
          <div className="flex items-center gap-0.5 bg-[var(--bg)] border border-[var(--line)] rounded-lg p-0.5 font-mono">
            {[1.0, 2.0, 4.0].map((lvl) => (
              <button
                key={lvl}
                type="button"
                onClick={() => handleZoom(lvl)}
                aria-label={`Timeline zoom ${lvl}x`}
                className={`px-2 py-0.5 rounded text-[10px] font-bold transition-all ${
                  zoom.scale === lvl
                    ? "bg-[var(--panel)] text-[var(--accent)] shadow-sm"
                    : "text-[var(--muted)] hover:text-[var(--text)]"
                }`}
              >
                {lvl}x
              </button>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-2">
          {peakScore > 0 && (
            <span
              title="Display-normalized heatmap value, not calibrated confidence"
              className="text-[10px] font-mono font-bold px-2 py-0.5 rounded-full bg-[var(--accent)]/10 text-[var(--accent)] border border-[var(--accent)]/30"
            >
              Normalized peak: {peakScore.toFixed(2)}
            </span>
          )}
          <span className="font-mono text-xs text-[var(--muted)]">
            <b className="text-[var(--accent)] font-bold">{currentTime.toFixed(1)}s</b> / {duration.toFixed(1)}s
          </span>
        </div>
      </div>

      {/* Top-K Moment Anchor Pins Layer */}
      {moments.length > 0 && duration > 0 && (
        <div className="relative w-full h-5">
          {moments.map((m, idx) => {
            const wWidth = containerRef.current?.clientWidth || 500;
            const leftPct = Math.max(0, Math.min(100, (timeToX(m.t_start, wWidth) / wWidth) * 100));
            const rightPct = Math.max(leftPct, Math.min(100, (timeToX(m.t_end, wWidth) / wWidth) * 100));
            const pct = Math.max(2, Math.min(98, (leftPct + rightPct) / 2));

            const key = `${m.occurrence_index ?? 0}:${m.t_start}:${m.t_end}`;
            return (
              <React.Fragment key={key}>
                <span
                  aria-hidden="true"
                  className="absolute top-3 h-1 rounded-full bg-[var(--accent)]/60"
                  style={{ left: `${leftPct}%`, width: `${Math.max(0.5, rightPct - leftPct)}%` }}
                />
                <button
                  type="button"
                  onClick={() => onSeek(m.t_start)}
                  style={{ left: `${pct}%` }}
                  title={`Jump to event ${m.occurrence_index ?? idx + 1} (${m.t_start.toFixed(1)}s - ${m.t_end.toFixed(1)}s)`}
                  className="absolute -translate-x-1/2 top-0 px-2 py-0.5 rounded-full bg-[var(--panel)] hover:bg-[var(--bg)] border border-[var(--accent)] text-[10px] font-mono font-bold text-[var(--accent)] shadow-sm hover:scale-105 transition-all flex items-center gap-1 z-20"
                >
                  <Sparkles className="w-2.5 h-2.5 text-[var(--accent)]" />
                  <span>#{idx + 1}</span>
                </button>
              </React.Fragment>
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
          className="w-full h-12 rounded-xl border border-[var(--line)] shadow-inner transition-colors"
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
            <div className="p-1.5 rounded-xl flex flex-col items-center gap-1 shadow-xl bg-[var(--panel)] border border-[var(--line)] w-28 text-center">
              {hoverState.frameUrl ? (
                <div className="w-full h-14 rounded-lg overflow-hidden bg-black border border-[var(--line)] relative">
                  <img
                    src={hoverState.frameUrl}
                    alt="Keyframe Preview"
                    className="w-full h-full object-cover"
                    loading="eager"
                  />
                  {hoverState.score > 0 && (
                    <span className="absolute bottom-0.5 right-0.5 text-[9px] font-mono font-bold bg-black/80 px-1 py-0.2 rounded text-white">
                      {(hoverState.score * 100).toFixed(0)}%
                    </span>
                  )}
                </div>
              ) : (
                <div className="w-full h-14 rounded-lg bg-[var(--bg)] flex items-center justify-center text-[10px] text-[var(--muted)] font-mono">
                  No Thumbnail
                </div>
              )}
              <span className="font-mono text-[11px] font-bold text-[var(--text)] tracking-wider">
                {formatSeconds(hoverState.hoverTime)}
              </span>
            </div>
            {/* Tooltip caret arrow */}
            <div className="w-2.5 h-2.5 bg-[var(--panel)] border-r border-b border-[var(--line)] rotate-45 mx-auto -mt-1" />
          </div>
        )}
      </div>

      {/* Drag Instruction Cue */}
      {highlightInterval && (
        <div className="flex items-center justify-between text-[10px] font-mono text-[var(--muted)] px-1">
          <span className="flex items-center gap-1">
            <span className="text-[var(--accent)] font-semibold">◄ Drag Brackets ►</span> to fine-tune event boundary
          </span>
          <span className="text-[var(--text)] font-medium">
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
