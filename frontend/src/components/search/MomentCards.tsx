"use client";

import React, { useState } from "react";
import { MomentItem } from "@/lib/types";
import {
  Play,
  Sparkles,
  Download,
  Check,
  Loader2,
  Subtitles,
  Repeat,
  Copy,
  Clock,
  Tag,
} from "lucide-react";
import { apiClient } from "@/lib/api";

interface MomentCardsProps {
  moments: MomentItem[];
  videoId?: string;
  onSelectMoment: (moment: MomentItem, autoLoop?: boolean) => void;
  activeMoment?: MomentItem | null;
}

export const MomentCards: React.FC<MomentCardsProps> = ({
  moments,
  videoId,
  onSelectMoment,
  activeMoment,
}) => {
  const [exportingIndex, setExportingIndex] = useState<number | null>(null);
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);

  const handleExport = async (e: React.MouseEvent, m: MomentItem, idx: number, withSubtitles: boolean) => {
    e.stopPropagation();
    if (!videoId) return;

    setExportingIndex(idx);
    try {
      const res = await apiClient.exportClip(videoId, m.t_start, m.t_end, withSubtitles);
      if (res && res.download_url) {
        const a = document.createElement("a");
        a.href = `http://localhost:8000${res.download_url}`;
        a.download = res.clip_filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
      }
    } catch (err) {
      console.error("Failed to export clip:", err);
    } finally {
      setExportingIndex(null);
    }
  };

  const handleCopyTimestamp = (e: React.MouseEvent, m: MomentItem, idx: number) => {
    e.stopPropagation();
    const text = `${m.t_start.toFixed(1)}s - ${m.t_end.toFixed(1)}s`;
    navigator.clipboard.writeText(text);
    setCopiedIndex(idx);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  // Helper to extract action keywords for semantic micro-tags
  const extractActionTags = (caption?: string | null): string[] => {
    if (!caption) return [];
    const keywords = ["walking", "turns", "left", "right", "bicycle", "car", "standing", "hand", "water", "student", "teacher", "raising", "lecture", "crossing"];
    const lower = caption.toLowerCase();
    const found: string[] = [];
    for (const kw of keywords) {
      if (lower.includes(kw)) {
        found.push(`#${kw.charAt(0).toUpperCase() + kw.slice(1)}`);
      }
    }
    return found.slice(0, 3);
  };

  if (!moments || moments.length === 0) {
    return (
      <div className="glass-panel rounded-2xl p-8 text-center text-gray-400 space-y-2 border border-surfaceBorder/80">
        <Sparkles className="w-8 h-8 mx-auto text-cyan-400 opacity-60 animate-pulse" />
        <p className="font-semibold text-gray-200">ยังไม่มีช่วงเวลาที่ค้นพบ (No Moments)</p>
        <p className="text-xs text-gray-500">
          พิมพ์คำค้นหาภาษาไทยหรืออังกฤษในช่องด้านบน หรือคลิกชิปหมวดการเคลื่อนไหวเพื่อเริ่มค้นหา
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {/* Header Bar */}
      <div className="flex items-center justify-between px-1">
        <h3 className="text-xs font-bold text-gray-300 uppercase tracking-wider flex items-center gap-2">
          <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
          Retrieved Moments ({moments.length})
        </h3>
        <span className="text-[11px] font-mono text-cyan-400/80 bg-cyan-950/60 px-2 py-0.5 rounded-full border border-cyan-800/40">
          Ranked by SOTA Multi-Scale IoU
        </span>
      </div>

      {/* Cards List */}
      <div className="grid grid-cols-1 gap-3 max-h-[580px] overflow-y-auto pr-1">
        {moments.map((m, idx) => {
          const isActive =
            activeMoment?.t_start === m.t_start && activeMoment?.t_end === m.t_end;

          const duration = Math.max(0.1, m.t_end - m.t_start);
          const scorePercent = (m.score * 100).toFixed(1);
          const actionTags = extractActionTags(m.caption_preview);

          // Radial SVG stroke calculations (r = 16)
          const radius = 16;
          const circumference = 2 * Math.PI * radius;
          const strokeDashoffset = circumference * (1 - Math.min(1.0, Math.max(0.05, m.score)));
          const ringColor = m.score >= 0.75 ? "#10b981" : m.score >= 0.5 ? "#00f0ff" : "#6366f1";

          return (
            <div
              key={idx}
              onClick={() => onSelectMoment(m, false)}
              className={`glass-panel p-3.5 rounded-2xl cursor-pointer transition-all duration-200 flex flex-col gap-2.5 border group ${
                isActive
                  ? "border-cyan-500 bg-surface/90 shadow-xl shadow-cyan-500/10 ring-1 ring-cyan-500/40"
                  : "border-surfaceBorder hover:border-gray-500 hover:bg-surface/70"
              }`}
            >
              {/* Top Row: Thumbnail + Info */}
              <div className="flex items-start gap-3.5">
                {/* Visual Thumbnail & Duration Badge */}
                <div className="relative w-32 h-20 rounded-xl overflow-hidden bg-black/90 flex-shrink-0 border border-surfaceBorder group-hover:border-cyan-500/50 transition-colors">
                  {m.preview_frame_path ? (
                    <img
                      src={apiClient.getFramePreviewUrl(m.preview_frame_path)}
                      alt={`Preview at ${m.t_start}s`}
                      className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-gray-600 font-mono text-[10px]">
                      No Frame
                    </div>
                  )}

                  {/* Gradient Overlay & Controls */}
                  <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent flex items-end justify-between p-1.5">
                    <span className="text-[10px] font-mono font-bold text-white bg-black/70 px-1.5 py-0.5 rounded border border-white/10">
                      {duration.toFixed(1)}s
                    </span>

                    <div className="w-6 h-6 rounded-full bg-cyan-500 text-black flex items-center justify-center shadow-md group-hover:scale-110 transition-transform">
                      <Play className="w-3 h-3 ml-0.5 fill-current" />
                    </div>
                  </div>
                </div>

                {/* Details & Confidence Radial Meter */}
                <div className="flex-1 min-w-0 space-y-1.5">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-1.5">
                      <span className="text-xs font-bold text-cyan-300 bg-cyan-950 px-2 py-0.5 rounded-md border border-cyan-800">
                        #{idx + 1}
                      </span>
                      <span className="text-xs font-mono font-semibold text-gray-200">
                        {m.t_start.toFixed(1)}s - {m.t_end.toFixed(1)}s
                      </span>
                    </div>

                    {/* Radial Confidence Meter */}
                    <div className="flex items-center gap-1.5">
                      <div className="relative w-9 h-9 flex items-center justify-center">
                        <svg className="w-9 h-9 transform -rotate-90">
                          <circle
                            cx="18"
                            cy="18"
                            r={radius}
                            stroke="rgba(255, 255, 255, 0.1)"
                            strokeWidth="3"
                            fill="transparent"
                          />
                          <circle
                            cx="18"
                            cy="18"
                            r={radius}
                            stroke={ringColor}
                            strokeWidth="3"
                            fill="transparent"
                            strokeDasharray={circumference}
                            strokeDashoffset={strokeDashoffset}
                            strokeLinecap="round"
                            className="transition-all duration-500"
                          />
                        </svg>
                        <span className="absolute text-[9px] font-mono font-bold text-white">
                          {Math.round(m.score * 100)}%
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Dense Action Caption Preview */}
                  {m.caption_preview ? (
                    <p className="text-xs text-gray-300 line-clamp-2 leading-relaxed">
                      <b className="text-gray-400">Action:</b> {m.caption_preview}
                    </p>
                  ) : (
                    <p className="text-xs text-gray-500 italic">Visual spatiotemporal match</p>
                  )}

                  {/* Action Micro-Tags */}
                  {actionTags.length > 0 && (
                    <div className="flex items-center gap-1.5 pt-0.5">
                      {actionTags.map((tag, tIdx) => (
                        <span
                          key={tIdx}
                          className="text-[10px] font-mono text-cyan-300 bg-cyan-950/60 px-1.5 py-0.2 rounded border border-cyan-800/40"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              {/* Action Toolbar Row */}
              <div className="flex items-center justify-between pt-1 border-t border-surfaceBorder/60 text-xs">
                <div className="flex items-center gap-1.5">
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      onSelectMoment(m, true);
                    }}
                    title="Play and loop this exact moment"
                    className="px-2.5 py-1 rounded-lg bg-cyan-950/80 hover:bg-cyan-900 text-cyan-300 border border-cyan-700/60 text-[11px] font-medium flex items-center gap-1 transition-colors"
                  >
                    <Repeat className="w-3 h-3" />
                    <span>Loop</span>
                  </button>

                  <button
                    type="button"
                    onClick={(e) => handleCopyTimestamp(e, m, idx)}
                    title="Copy timestamp interval to clipboard"
                    className="px-2 py-1 rounded-lg bg-surface hover:bg-surfaceBorder text-gray-300 hover:text-white border border-surfaceBorder text-[11px] flex items-center gap-1 transition-colors"
                  >
                    {copiedIndex === idx ? (
                      <Check className="w-3 h-3 text-emerald-400" />
                    ) : (
                      <Copy className="w-3 h-3" />
                    )}
                    <span>{copiedIndex === idx ? "Copied" : "Copy Time"}</span>
                  </button>
                </div>

                <div className="flex items-center gap-1.5">
                  <button
                    type="button"
                    disabled={exportingIndex === idx}
                    onClick={(e) => handleExport(e, m, idx, false)}
                    className="px-2.5 py-1 rounded-lg bg-surface hover:bg-surfaceBorder text-gray-300 hover:text-white border border-surfaceBorder text-[11px] flex items-center gap-1 transition-colors disabled:opacity-50"
                  >
                    {exportingIndex === idx ? (
                      <Loader2 className="w-3 h-3 animate-spin text-cyan-400" />
                    ) : (
                      <Download className="w-3 h-3 text-cyan-400" />
                    )}
                    <span>Cut MP4</span>
                  </button>

                  <button
                    type="button"
                    disabled={exportingIndex === idx}
                    onClick={(e) => handleExport(e, m, idx, true)}
                    className="px-2.5 py-1 rounded-lg bg-indigo-950/60 hover:bg-indigo-900 text-indigo-300 border border-indigo-800/60 text-[11px] flex items-center gap-1 transition-colors disabled:opacity-50"
                  >
                    <Subtitles className="w-3 h-3 text-indigo-400" />
                    <span>Subtitles</span>
                  </button>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
