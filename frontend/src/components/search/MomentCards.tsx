"use client";

import React, { useState } from "react";
import { MomentItem } from "@/lib/types";
import { Play, Sparkles, MessageSquare, Image as ImageIcon, Download, Check, Loader2, Subtitles } from "lucide-react";
import { apiClient } from "@/lib/api";

interface MomentCardsProps {
  moments: MomentItem[];
  videoId?: string;
  onSelectMoment: (moment: MomentItem) => void;
  activeMoment?: MomentItem | null;
}

export const MomentCards: React.FC<MomentCardsProps> = ({
  moments,
  videoId,
  onSelectMoment,
  activeMoment,
}) => {
  const [exportingIndex, setExportingIndex] = useState<number | null>(null);
  const [exportedMap, setExportedMap] = useState<Record<number, string>>({});

  const handleExport = async (e: React.MouseEvent, m: MomentItem, idx: number, withSubtitles: boolean) => {
    e.stopPropagation();
    if (!videoId) return;

    setExportingIndex(idx);
    try {
      const res = await apiClient.exportClip(videoId, m.t_start, m.t_end, withSubtitles);
      if (res && res.download_url) {
        setExportedMap((prev) => ({ ...prev, [idx]: res.download_url }));
        // Trigger direct browser download
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

  if (!moments || moments.length === 0) {
    return (
      <div className="glass-panel rounded-2xl p-8 text-center text-gray-400 space-y-2">
        <Sparkles className="w-8 h-8 mx-auto text-primary-500 opacity-60" />
        <p className="font-medium text-gray-300">No moments retrieved yet</p>
        <p className="text-xs text-gray-500">
          Enter a natural language description above to find specific events and timestamps.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between px-1">
        <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wider flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-cyan-400" />
          Retrieved Moments ({moments.length})
        </h3>
        <span className="text-xs text-gray-500">Multi-Scale SOTA Ranked</span>
      </div>

      <div className="grid grid-cols-1 gap-3">
        {moments.map((m, idx) => {
          const isActive =
            activeMoment?.t_start === m.t_start && activeMoment?.t_end === m.t_end;

          return (
            <div
              key={idx}
              onClick={() => onSelectMoment(m)}
              className={`glass-panel p-3.5 rounded-xl cursor-pointer transition-all duration-200 flex items-start gap-4 border ${
                isActive
                  ? "border-cyan-500 bg-surface/90 shadow-lg shadow-cyan-500/10"
                  : "border-surfaceBorder hover:border-gray-600 hover:bg-surface/60"
              }`}
            >
              {/* Thumbnail / Timestamp Box */}
              <div className="relative w-28 h-20 rounded-lg overflow-hidden bg-black/80 flex-shrink-0 border border-surfaceBorder">
                {m.preview_frame_path ? (
                  <img
                    src={apiClient.getFramePreviewUrl(m.preview_frame_path)}
                    alt={`Preview at ${m.t_start}s`}
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center text-gray-600">
                    <ImageIcon className="w-6 h-6" />
                  </div>
                )}
                <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent flex items-end p-1.5 justify-between">
                  <span className="text-[10px] font-mono font-bold text-white bg-black/60 px-1 py-0.5 rounded">
                    {Math.floor(m.t_start)}s - {Math.floor(m.t_end)}s
                  </span>
                  <div className="w-5 h-5 rounded-full bg-primary-600 text-white flex items-center justify-center">
                    <Play className="w-2.5 h-2.5 ml-0.5" />
                  </div>
                </div>
              </div>

              {/* Text Snippets & Score */}
              <div className="flex-1 min-w-0 space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-cyan-400 bg-cyan-950/60 px-2 py-0.5 rounded-full border border-cyan-800/40">
                    Match #{idx + 1}
                  </span>
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-mono font-medium text-emerald-400">
                      {(m.score * 100).toFixed(1)}% Match
                    </span>
                  </div>
                </div>

                {m.caption_preview && (
                  <p className="text-xs text-gray-300 line-clamp-2 leading-relaxed">
                    <span className="font-semibold text-gray-400">Action: </span>
                    {m.caption_preview}
                  </p>
                )}

                {/* Clip Exporter Action Buttons */}
                <div className="flex items-center gap-2 pt-1">
                  <button
                    type="button"
                    disabled={exportingIndex === idx}
                    onClick={(e) => handleExport(e, m, idx, false)}
                    className="px-2 py-1 rounded-md bg-surfaceBorder hover:bg-gray-700 text-[11px] text-gray-300 flex items-center gap-1 transition-colors"
                  >
                    {exportingIndex === idx ? (
                      <Loader2 className="w-3 h-3 animate-spin text-cyan-400" />
                    ) : (
                      <Download className="w-3 h-3 text-cyan-400" />
                    )}
                    Cut Clip
                  </button>

                  <button
                    type="button"
                    disabled={exportingIndex === idx}
                    onClick={(e) => handleExport(e, m, idx, true)}
                    className="px-2 py-1 rounded-md bg-indigo-950/80 hover:bg-indigo-900 text-[11px] text-indigo-200 border border-indigo-700/50 flex items-center gap-1 transition-colors"
                  >
                    <Subtitles className="w-3 h-3 text-indigo-400" />
                    Burn Subtitles
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
