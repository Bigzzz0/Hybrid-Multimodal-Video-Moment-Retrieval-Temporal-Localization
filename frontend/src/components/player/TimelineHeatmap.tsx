"use client";

import React, { useRef, useEffect } from "react";

interface TimelineHeatmapProps {
  heatmapData: number[]; // 1 value per second [0.0 - 1.0]
  duration: number;
  currentTime: number;
  highlightInterval?: [number, number] | null;
  onSeek: (time: number) => void;
}

export const TimelineHeatmap: React.FC<TimelineHeatmapProps> = ({
  heatmapData,
  duration,
  currentTime,
  highlightInterval,
  onSeek,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Deep modern dark background
    ctx.fillStyle = "#0a0f1d";
    ctx.fillRect(0, 0, width, height);

    if (!heatmapData || heatmapData.length === 0 || duration <= 0) {
      // Empty grid pattern
      ctx.strokeStyle = "#1e293b";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, height / 2);
      ctx.lineTo(width, height / 2);
      ctx.stroke();
      return;
    }

    const n = heatmapData.length;
    const barWidth = width / n;

    // 1. Draw Background Area Gradient
    const gradient = ctx.createLinearGradient(0, height, 0, 0);
    gradient.addColorStop(0, "rgba(99, 102, 241, 0.1)");
    gradient.addColorStop(0.5, "rgba(6, 182, 212, 0.4)");
    gradient.addColorStop(1, "rgba(16, 185, 129, 0.85)");

    // 2. Draw Distinct Vertical Density Bars
    heatmapData.forEach((rawVal, idx) => {
      const val = Math.max(0.05, Math.min(1.0, rawVal));
      const x = idx * barWidth;
      const barHeight = Math.max(3, val * (height - 4));
      const y = height - barHeight;

      // Color coding based on exact relevance level
      if (val >= 0.75) {
        ctx.fillStyle = "#10b981"; // Vibrant Emerald (Top Match)
      } else if (val >= 0.5) {
        ctx.fillStyle = "#06b6d4"; // Electric Cyan (Strong Match)
      } else if (val >= 0.3) {
        ctx.fillStyle = "#6366f1"; // Indigo (Moderate Match)
      } else {
        ctx.fillStyle = "#1e293b"; // Dim Slate (Low Relevance)
      }

      ctx.fillRect(x, y, Math.max(1.5, barWidth - 1), barHeight);
    });

    // 3. Draw Continuous Smooth Top Edge Line
    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    heatmapData.forEach((rawVal, idx) => {
      const val = Math.max(0.05, Math.min(1.0, rawVal));
      const x = (idx + 0.5) * barWidth;
      const y = height - (val * (height - 4));
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // 4. Draw Highlight Interval [t_start, t_end] with Glow
    if (highlightInterval && duration > 0) {
      const [ts, te] = highlightInterval;
      const x1 = Math.max(0, (ts / duration) * width);
      const x2 = Math.min(width, (te / duration) * width);
      const w = Math.max(6, x2 - x1);

      ctx.fillStyle = "rgba(6, 182, 212, 0.3)";
      ctx.fillRect(x1, 0, w, height);

      ctx.strokeStyle = "#22d3ee";
      ctx.lineWidth = 2;
      ctx.strokeRect(x1, 0, w, height);
    }

    // 5. Draw Playhead Position Cursor
    if (duration > 0) {
      const playheadX = (currentTime / duration) * width;
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.shadowColor = "#ffffff";
      ctx.shadowBlur = 4;
      ctx.beginPath();
      ctx.moveTo(playheadX, 0);
      ctx.lineTo(playheadX, height);
      ctx.stroke();
      ctx.shadowBlur = 0; // reset
    }
  }, [heatmapData, duration, currentTime, highlightInterval]);

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || duration <= 0) return;
    const rect = canvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const ratio = Math.max(0, Math.min(1, clickX / rect.width));
    const targetTime = ratio * duration;
    onSeek(targetTime);
  };

  const peakScore = heatmapData.length > 0 ? Math.max(...heatmapData) : 0;

  return (
    <div className="w-full space-y-1.5">
      <div className="flex items-center justify-between text-xs text-gray-400 font-medium px-1">
        <span className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
          <span className="text-gray-200 font-semibold">Relevance Density Heatmap</span>
          {heatmapData.length > 0 && (
            <span className="text-[10px] px-2 py-0.5 rounded-full bg-cyan-950 text-cyan-300 border border-cyan-800 font-mono">
              Peak: {(peakScore * 100).toFixed(0)}%
            </span>
          )}
        </span>
        <span className="font-mono text-gray-300">
          {Math.floor(currentTime)}s / {Math.floor(duration)}s
        </span>
      </div>
      <canvas
        ref={canvasRef}
        width={800}
        height={36}
        onClick={handleClick}
        className="w-full h-9 rounded-xl cursor-pointer border border-surfaceBorder hover:border-cyan-500/60 transition-all duration-200 shadow-lg shadow-black/40"
      />
    </div>
  );
};
