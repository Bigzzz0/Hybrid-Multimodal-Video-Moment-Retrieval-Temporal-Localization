"use client";

import React from "react";
import { MomentItem } from "@/lib/types";
import { Play, Sparkles, MessageSquare, Image as ImageIcon } from "lucide-react";
import { apiClient } from "@/lib/api";

interface MomentCardsProps {
  moments: MomentItem[];
  onSelectMoment: (moment: MomentItem) => void;
  activeMoment?: MomentItem | null;
}

export const MomentCards: React.FC<MomentCardsProps> = ({
  moments,
  onSelectMoment,
  activeMoment,
}) => {
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
        <span className="text-xs text-gray-500">Ranked by Hybrid Confidence</span>
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
                  <span className="text-xs font-mono font-medium text-amber-400">
                    Score: {(m.score * 100).toFixed(1)}%
                  </span>
                </div>

                {m.caption_preview && (
                  <p className="text-xs text-gray-300 line-clamp-2 leading-relaxed">
                    <span className="font-semibold text-gray-400">Action: </span>
                    {m.caption_preview}
                  </p>
                )}

                {m.transcript_preview && (
                  <p className="text-xs text-gray-400 flex items-start gap-1 line-clamp-1 italic">
                    <MessageSquare className="w-3 h-3 mt-0.5 text-indigo-400 flex-shrink-0" />
                    "{m.transcript_preview}"
                  </p>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
