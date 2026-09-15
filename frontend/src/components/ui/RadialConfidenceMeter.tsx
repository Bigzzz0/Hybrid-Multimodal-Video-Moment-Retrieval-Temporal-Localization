"use client";

import React from "react";

interface RadialConfidenceMeterProps {
  score: number; // 0.0 to 1.0
  calibrated?: boolean;
  size?: number; // default 40px
  strokeWidth?: number; // default 3.5px
  showLabel?: boolean;
}

export const RadialConfidenceMeter: React.FC<RadialConfidenceMeterProps> = ({
  score,
  calibrated = true,
  size = 40,
  strokeWidth = 3.5,
  showLabel = true,
}) => {
  const clampedScore = Math.max(0, Math.min(1.0, score));
  const percentage = Math.round(clampedScore * 100);
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - clampedScore * circumference;

  // Determine dynamic color scheme based on confidence bands
  let strokeColor = "#f59e0b"; // amber for < 60%
  let glowColor = "rgba(245, 158, 11, 0.4)";
  let textColor = "text-amber-400";
  let bgFill = "bg-amber-950/20";

  if (clampedScore >= 0.8 && calibrated) {
    strokeColor = "#10b981"; // emerald for >= 80%
    glowColor = "rgba(16, 185, 129, 0.45)";
    textColor = "text-emerald-400";
    bgFill = "bg-emerald-950/20";
  } else if (clampedScore >= 0.6 && calibrated) {
    strokeColor = "#00f0ff"; // cyan for >= 60%
    glowColor = "rgba(0, 240, 255, 0.45)";
    textColor = "text-cyan-400";
    bgFill = "bg-cyan-950/20";
  }

  return (
    <div
      className={`relative inline-flex items-center justify-center rounded-full ${bgFill}`}
      style={{ width: size, height: size }}
      title={calibrated ? `Calibrated confidence: ${percentage}%` : "Uncalibrated rank score"}
    >
      <svg
        width={size}
        height={size}
        className="transform -rotate-90"
        style={{ filter: `drop-shadow(0 0 4px ${glowColor})` }}
      >
        {/* Background Track Ring */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke="rgba(255, 255, 255, 0.08)"
          strokeWidth={strokeWidth}
          fill="transparent"
        />
        {/* Animated Fill Gauge */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke={strokeColor}
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          strokeLinecap="round"
          fill="transparent"
          style={{ transition: "stroke-dashoffset 0.6s cubic-bezier(0.16, 1, 0.3, 1)" }}
        />
      </svg>
      {showLabel && calibrated && (
        <span className={`absolute text-[10px] font-mono font-extrabold ${textColor}`}>
          {percentage}%
        </span>
      )}
    </div>
  );
};
