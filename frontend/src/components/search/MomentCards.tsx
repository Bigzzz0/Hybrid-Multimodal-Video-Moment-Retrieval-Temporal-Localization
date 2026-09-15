"use client";

import React, { useEffect, useRef, useState } from "react";
import { MomentItem } from "@/lib/types";
import {
  Sparkles,
  Download,
  Check,
  Loader2,
  Repeat,
  Copy,
  LayoutList,
  LayoutGrid,
  CheckSquare,
  Square,
  Activity,
} from "lucide-react";
import { apiClient } from "@/lib/api";
import { MomentEvidenceBreakdown } from "@/components/search/MomentEvidenceBreakdown";
import { MomentScore } from "@/components/search/MomentScore";
import { momentIdentity, formatMomentTime } from "@/lib/ui";
import { SearchEmptyState } from "@/components/search/SearchEmptyState";

interface MomentCardsProps {
  moments: MomentItem[];
  videoId?: string;
  calibrated?: boolean;
  warnings?: string[];
  profile?: "fast" | "accurate";
  onSelectMoment: (moment: MomentItem, autoLoop?: boolean) => void;
  onExpandContext?: (moment: MomentItem) => void;
  emptyState?: "initial" | "no_match" | "reindex";
  onReindex?: () => void;
  activeMoment?: MomentItem | null;
}

export const MomentCards: React.FC<MomentCardsProps> = ({
  moments,
  videoId,
  calibrated = false,
  warnings = [],
  profile = "fast",
  onSelectMoment,
  onExpandContext,
  emptyState = "initial",
  onReindex,
  activeMoment,
}) => {
  const [layoutMode, setLayoutMode] = useState<"list" | "grid">("list");
  const [selectedMoments, setSelectedMoments] = useState<Set<number>>(new Set());
  const [exportingIndex, setExportingIndex] = useState<number | null>(null);
  const [isBatchExporting, setIsBatchExporting] = useState(false);
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);
  const headingRef = useRef<HTMLHeadingElement | null>(null);

  useEffect(() => {
    if (moments.length > 0) headingRef.current?.focus();
  }, [moments]);

  useEffect(() => {
    setSelectedMoments(new Set());
    setCopiedIndex(null);
  }, [moments]);

  const toggleSelectMoment = (e: React.MouseEvent, idx: number) => {
    e.stopPropagation();
    const updated = new Set(selectedMoments);
    if (updated.has(idx)) {
      updated.delete(idx);
    } else {
      updated.add(idx);
    }
    setSelectedMoments(updated);
  };

  const selectAllMoments = () => {
    if (selectedMoments.size === moments.length) {
      setSelectedMoments(new Set());
    } else {
      setSelectedMoments(new Set(moments.map((_, i) => i)));
    }
  };

  const handleExport = async (e: React.MouseEvent, m: MomentItem, idx: number) => {
    e.stopPropagation();
    if (!videoId) return;

    setExportingIndex(idx);
    try {
      const res = await apiClient.exportClip(videoId, m.t_start, m.t_end);
      if (res && res.download_url) {
        const a = document.createElement("a");
        a.href = apiClient.getDownloadUrl(res.download_url);
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

  const handleBatchExport = async () => {
    if (!videoId || selectedMoments.size === 0) return;
    setIsBatchExporting(true);

    const targetIndices = Array.from(selectedMoments).sort((a, b) => a - b);
    for (const idx of targetIndices) {
      const m = moments[idx];
      try {
          const res = await apiClient.exportClip(videoId, m.t_start, m.t_end);
        if (res && res.download_url) {
          const a = document.createElement("a");
          a.href = apiClient.getDownloadUrl(res.download_url);
          a.download = res.clip_filename;
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
        }
      } catch (err) {
        console.error("Error exporting clip in batch:", err);
      }
    }
    setIsBatchExporting(false);
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
    const keywords = [
      "walking", "turns", "left", "right", "bicycle", "cyclist", "car", "driving",
      "standing", "hand", "water", "student", "teacher", "raising", "lecture", "crossing", "riding"
    ];
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
    return <SearchEmptyState state={emptyState} onReindex={onReindex} />;
  }

  return (
    <div className="space-y-3 relative">
      {/* Header Bar with View Switcher & Counter */}
      <div className="flex items-center justify-between px-1">
        <div className="flex items-center gap-2">
          <h3 ref={headingRef} id="search-results-heading" className="text-xs font-bold text-gray-300 uppercase tracking-wider flex items-center gap-1.5" tabIndex={-1}>
            <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
            Moments ({moments.length})
          </h3>

          <button
            type="button"
            onClick={selectAllMoments}
            className="text-[11px] font-mono text-gray-400 hover:text-cyan-300 flex items-center gap-1 transition-colors ml-2"
          >
            {selectedMoments.size === moments.length ? (
              <CheckSquare className="w-3.5 h-3.5 text-cyan-400" />
            ) : (
              <Square className="w-3.5 h-3.5 text-gray-500" />
            )}
            <span>Select All</span>
          </button>
        </div>

        <div className="flex items-center gap-2">
          {/* Detailed List vs Storyboard Grid Toggle */}
          <div className="flex items-center bg-surface border border-surfaceBorder rounded-lg p-0.5">
            <button
              type="button"
              onClick={() => setLayoutMode("list")}
              aria-label="Detailed list view"
              className={`p-1 rounded transition-colors ${
                layoutMode === "list"
                  ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                  : "text-gray-500 hover:text-gray-300"
              }`}
              title="Detailed List View"
            >
              <LayoutList className="w-3.5 h-3.5" />
            </button>
            <button
              type="button"
              onClick={() => setLayoutMode("grid")}
              aria-label="Storyboard grid view"
              className={`p-1 rounded transition-colors ${
                layoutMode === "grid"
                  ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                  : "text-gray-500 hover:text-gray-300"
              }`}
              title="Storyboard Grid View"
            >
              <LayoutGrid className="w-3.5 h-3.5" />
            </button>
          </div>

          <span className="text-[11px] font-mono text-cyan-400/80 bg-cyan-950/60 px-2 py-0.5 rounded-full border border-cyan-800/40">
            {calibrated ? "Calibrated visual ranking" : "Visual ranking (calibration pending)"}
          </span>
        </div>
      </div>

      {/* Cards Viewport */}
      {layoutMode === "list" ? (
        /* Detailed List View */
        <div className="grid grid-cols-1 gap-3 max-h-[580px] overflow-y-auto pr-1">
          {moments.map((m, idx) => {
            const isActive =
              activeMoment?.t_start === m.t_start && activeMoment?.t_end === m.t_end;
            const isSelected = selectedMoments.has(idx);
            const duration = Math.max(0.1, m.t_end - m.t_start);
            const actionTags = extractActionTags(m.caption_preview);

            const breakdown = m.modality_breakdown || {};

            return (
              <div
                key={momentIdentity(m)}
                onClick={() => onSelectMoment(m, false)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSelectMoment(m, false); } }}
                className={`glass-panel p-3.5 rounded-2xl cursor-pointer transition-all duration-200 flex flex-col gap-2.5 border group ${
                  isActive
                    ? "border-cyan-500 bg-surface/90 shadow-xl shadow-cyan-500/10 ring-1 ring-cyan-500/40"
                    : isSelected
                    ? "border-indigo-500/80 bg-surface/80 shadow-md"
                    : "border-surfaceBorder hover:border-gray-500 hover:bg-surface/70"
                }`}
              >
                {/* Top Row: Select Checkbox, Thumbnail, Timestamps, and Gauge */}
                <div className="flex items-start gap-3">
                  {/* Select Checkbox */}
                  <button
                    type="button"
                    onClick={(e) => toggleSelectMoment(e, idx)}
                    className="mt-1 text-gray-500 hover:text-cyan-400 transition-colors"
                  >
                    {isSelected ? (
                      <CheckSquare className="w-4 h-4 text-cyan-400" />
                    ) : (
                      <Square className="w-4 h-4 text-gray-600 group-hover:text-gray-400" />
                    )}
                  </button>

                  {/* Thumbnail */}
                  <div className="relative w-28 h-18 rounded-xl overflow-hidden bg-black/90 flex-shrink-0 border border-surfaceBorder group-hover:border-cyan-500/50 transition-colors">
                    {m.preview_frame_path ? (
                      <img
                        src={apiClient.getFramePreviewUrl(m.preview_frame_path)}
                        alt={`Preview at ${m.t_start}s`}
                        className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center text-gray-600 text-xs font-mono">
                        No Frame
                      </div>
                    )}

                    <span className="absolute bottom-1 right-1 px-1.5 py-0.5 rounded bg-black/85 backdrop-blur-sm text-[10px] font-mono font-bold text-white border border-white/10">
                      {duration.toFixed(1)}s
                    </span>

                    {/* Rank Badge */}
                    <span className="absolute top-1 left-1 px-1.5 py-0.2 rounded-md bg-cyan-950/90 border border-cyan-500/60 text-[9px] font-mono font-bold text-cyan-300">
                      #{idx + 1}
                    </span>
                  </div>

                  {/* Moment Metadata Info */}
                  <div className="flex-1 min-w-0 space-y-1">
                    <div className="flex items-center justify-between">
                      <div className="flex flex-wrap items-center gap-1.5">
                        <span className="text-sm font-mono font-bold text-white tracking-wide">
                          {formatMomentTime(m.t_start)} – {formatMomentTime(m.t_end)}
                        </span>
                        {typeof m.occurrence_index === "number" && (
                          <span className="text-[10px] font-mono text-indigo-300 bg-indigo-950/60 border border-indigo-800/60 rounded px-1.5 py-0.5">
                            เหตุการณ์ครั้งที่ {m.occurrence_index}
                          </span>
                        )}
                        <button
                          type="button"
                          onClick={(e) => handleCopyTimestamp(e, m, idx)}
                          aria-label={`Copy timecode ${formatMomentTime(m.t_start)} to ${formatMomentTime(m.t_end)}`}
                          title="Copy interval timecode"
                          className="p-1 rounded hover:bg-surfaceBorder text-gray-400 hover:text-white transition-colors"
                        >
                          {copiedIndex === idx ? (
                            <Check className="w-3 h-3 text-emerald-400" />
                          ) : (
                            <Copy className="w-3 h-3" />
                          )}
                        </button>
                      </div>

                      <MomentScore score={m.score} calibrated={calibrated} size={36} />
                    </div>

                    {/* Dense visual caption preview */}
                    <p className="text-xs text-gray-300 line-clamp-2 leading-relaxed">
                      {m.caption_preview || "Caption unavailable — ranking uses frame embeddings only."}
                    </p>

                    {/* Action Micro-Tags */}
                    {actionTags.length > 0 && (
                      <div className="flex flex-wrap gap-1 pt-0.5">
                        {actionTags.map((tag, tIdx) => (
                          <span
                            key={tIdx}
                            className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-surface border border-surfaceBorder text-cyan-300/90"
                          >
                            {tag}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>

                <MomentEvidenceBreakdown breakdown={breakdown} showVerifier={profile === "accurate"} />

                {/* Bottom Row Actions Toolbar */}
                <div className="flex items-center justify-between pt-1 border-t border-surfaceBorder/40 text-xs">
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      onSelectMoment(m, true);
                    }}
                    className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[11px] font-medium transition-all ${
                      isActive
                        ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                        : "hover:bg-surface text-gray-400 hover:text-white"
                    }`}
                  >
                    <Repeat className="w-3 h-3" />
                    <span>A-B Loop</span>
                  </button>

                  <div className="flex items-center gap-1.5">
                    {onExpandContext && (m.context_t_start != null && m.context_t_end != null && (m.context_t_start < m.t_start - 0.01 || m.context_t_end > m.t_end + 0.01)) && (
                      <button
                        type="button"
                        onClick={(e) => { e.stopPropagation(); onExpandContext(m); }}
                        className="flex items-center gap-1 rounded-lg border border-indigo-800/60 bg-indigo-950/40 px-2.5 py-1 text-[11px] font-medium text-indigo-200 hover:bg-indigo-900/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400"
                        title={`แสดงบริบท ${formatMomentTime(m.context_t_start)} – ${formatMomentTime(m.context_t_end)}`}
                      >
                        <Activity className="h-3 w-3" /> บริบท
                      </button>
                    )}
                    <button
                      type="button"
                      disabled={exportingIndex === idx}
                      onClick={(e) => handleExport(e, m, idx)}
                      className="flex items-center gap-1 px-2.5 py-1 rounded-lg bg-surface hover:bg-surfaceBorder text-gray-300 hover:text-white border border-surfaceBorder text-[11px] font-medium transition-colors"
                      title="Download MP4 clip"
                    >
                      {exportingIndex === idx ? (
                        <Loader2 className="w-3 h-3 animate-spin text-cyan-400" />
                      ) : (
                        <Download className="w-3 h-3 text-cyan-400" />
                      )}
                      <span>Clip</span>
                    </button>

                  </div>
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        /* Storyboard Grid View (2 Columns) */
        <div className="grid grid-cols-2 gap-3 max-h-[580px] overflow-y-auto pr-1">
          {moments.map((m, idx) => {
            const isActive =
              activeMoment?.t_start === m.t_start && activeMoment?.t_end === m.t_end;
            const isSelected = selectedMoments.has(idx);
            const duration = Math.max(0.1, m.t_end - m.t_start);

            return (
              <div
                key={momentIdentity(m)}
                onClick={() => onSelectMoment(m, false)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSelectMoment(m, false); } }}
                className={`glass-panel p-2.5 rounded-xl cursor-pointer transition-all duration-200 flex flex-col gap-2 border group ${
                  isActive
                    ? "border-cyan-500 bg-surface/90 shadow-lg shadow-cyan-500/10 ring-1 ring-cyan-500/40"
                    : isSelected
                    ? "border-indigo-500/80 bg-surface/80"
                    : "border-surfaceBorder hover:border-gray-500"
                }`}
              >
                {/* Large Visual Thumbnail */}
                <div className="relative aspect-video rounded-lg overflow-hidden bg-black border border-surfaceBorder">
                  {m.preview_frame_path ? (
                    <img
                      src={apiClient.getFramePreviewUrl(m.preview_frame_path)}
                      alt={`Preview at ${m.t_start}s`}
                      className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-gray-600 text-xs font-mono">
                      No Frame
                    </div>
                  )}

                  <span className="absolute top-1 left-1 px-1.5 py-0.2 rounded bg-cyan-950/90 border border-cyan-500/60 text-[9px] font-mono font-bold text-cyan-300">
                    #{idx + 1}
                  </span>

                  <span className="absolute bottom-1 right-1 px-1.5 py-0.2 rounded bg-black/85 text-[9px] font-mono font-bold text-white border border-white/10">
                    {duration.toFixed(1)}s
                  </span>
                </div>

                <div className="flex items-center justify-between px-0.5">
                  <span className="text-xs font-mono font-bold text-white">
                    {formatMomentTime(m.t_start)} – {formatMomentTime(m.t_end)}
                    <span className="ml-1.5 text-[10px] font-normal text-indigo-300">event {m.occurrence_index ?? idx + 1}</span>
                  </span>
                  <MomentScore score={m.score} calibrated={calibrated} size={28} />
                </div>

                <p className="text-[11px] text-gray-400 line-clamp-1 leading-snug">
                  {m.caption_preview || "Caption unavailable — frame embeddings only"}
                </p>
                {onExpandContext && m.context_t_start != null && m.context_t_end != null && (m.context_t_start < m.t_start - 0.01 || m.context_t_end > m.t_end + 0.01) && (
                  <button type="button" onClick={(e) => { e.stopPropagation(); onExpandContext(m); }} className="min-h-9 rounded-lg border border-indigo-800/60 px-2 py-1 text-[10px] font-semibold text-indigo-200 hover:bg-indigo-900/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400">ดูบริบท {formatMomentTime(m.context_t_start)}–{formatMomentTime(m.context_t_end)}</button>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Floating Multi-Moment Batch Action Toolbar */}
      {selectedMoments.size > 0 && (
        <div className="sticky bottom-2 left-0 right-0 glass-panel-glow p-2.5 rounded-xl flex items-center justify-between gap-3 shadow-2xl border border-cyan-500/60 z-30 animate-in slide-in-from-bottom-3 duration-200">
          <div className="flex items-center gap-2 text-xs font-mono text-cyan-300">
            <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
            <span>
              {selectedMoments.size} Moment{selectedMoments.size > 1 ? "s" : ""} Selected
            </span>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setSelectedMoments(new Set())}
              className="px-2 py-1 rounded-lg text-[11px] font-mono text-gray-400 hover:text-white"
            >
              Deselect
            </button>
            <button
              type="button"
              disabled={isBatchExporting}
              onClick={handleBatchExport}
              className="px-3 py-1.5 rounded-lg bg-gradient-to-r from-cyan-500 to-indigo-600 text-white text-xs font-bold shadow-md shadow-cyan-500/20 hover:opacity-95 transition-opacity flex items-center gap-1.5 disabled:opacity-50"
            >
              {isBatchExporting ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  <span>Exporting...</span>
                </>
              ) : (
                <>
                  <Download className="w-3.5 h-3.5" />
                  <span>Export Batch ({selectedMoments.size})</span>
                </>
              )}
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
