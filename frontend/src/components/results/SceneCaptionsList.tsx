"use client";

import React, { useState, useEffect, useMemo } from "react";
import {
  Film,
  Clock,
  Play,
  Sparkles,
  Search,
  ChevronDown,
  ChevronUp,
  AlertCircle,
  RefreshCw,
  X,
  Maximize2,
  ExternalLink,
  Layers,
} from "lucide-react";
import { apiClient } from "@/lib/api";
import { SceneCaptionItem } from "@/lib/types";

interface SceneCaptionsListProps {
  videoId?: string;
  onSeekToTimestamp?: (time: number) => void;
  onRebuildRequested?: () => void;
  initialSelectedSceneId?: string | null;
  scenes?: SceneCaptionItem[];
  loading?: boolean;
  onRefreshScenes?: () => void;
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  const ms = Math.floor((seconds % 1) * 10);
  return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}.${ms}`;
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}m ${s > 0 ? `${s}s` : ""}`;
}

export const SceneCaptionsList: React.FC<SceneCaptionsListProps> = ({
  videoId,
  onSeekToTimestamp,
  onRebuildRequested,
  initialSelectedSceneId,
  scenes: propScenes,
  loading: propLoading,
  onRefreshScenes,
}) => {
  const [internalScenes, setInternalScenes] = useState<SceneCaptionItem[]>([]);
  const [internalLoading, setInternalLoading] = useState<boolean>(false);
  const [filterQuery, setFilterQuery] = useState<string>("");
  const [selectedSceneId, setSelectedSceneId] = useState<string | null>(initialSelectedSceneId || null);
  const [brokenImageIds, setBrokenImageIds] = useState<Set<string>>(new Set());

  const scenes = propScenes !== undefined ? propScenes : internalScenes;
  const loading = propLoading !== undefined ? propLoading : internalLoading;

  const fetchScenes = async () => {
    if (onRefreshScenes) {
      onRefreshScenes();
      return;
    }
    if (!videoId) return;
    setInternalLoading(true);
    try {
      const data = await apiClient.getVideoScenes(videoId);
      setInternalScenes(data);
    } catch (err) {
      console.error("Failed to load scenes:", err);
    } finally {
      setInternalLoading(false);
    }
  };

  useEffect(() => {
    if (propScenes === undefined) {
      void fetchScenes();
    }
    if (!initialSelectedSceneId) {
      setSelectedSceneId(null);
    }
  }, [videoId]);

  useEffect(() => {
    if (initialSelectedSceneId) {
      setSelectedSceneId(initialSelectedSceneId);
      const timer = setTimeout(() => {
        const el = document.getElementById(`scene-item-${initialSelectedSceneId}`);
        if (el) {
          el.scrollIntoView({ behavior: "smooth", block: "center" });
        }
      }, 150);
      return () => clearTimeout(timer);
    }
  }, [initialSelectedSceneId]);

  const getThumbSrc = (scene: SceneCaptionItem): string => {
    if (scene.frame_path) {
      return apiClient.getFramePreviewUrl(scene.frame_path);
    }
    if (scene.thumbnail_url) {
      return apiClient.getDownloadUrl(scene.thumbnail_url);
    }
    return "";
  };

  const handleImageError = (sceneId: string) => {
    setBrokenImageIds((prev) => new Set(prev).add(sceneId));
  };

  const filteredScenes = useMemo(() => {
    if (!filterQuery.trim()) return scenes;
    const q = filterQuery.toLowerCase();
    return scenes.filter((s) => {
      const text = `${s.summary} ${s.structured?.objects?.join(" ") || ""} ${s.structured?.actions?.join(" ") || ""}`.toLowerCase();
      return text.includes(q);
    });
  }, [scenes, filterQuery]);

  const selectedScene = useMemo(() => {
    return scenes.find((s) => s.scene_id === selectedSceneId) || null;
  }, [scenes, selectedSceneId]);

  if (!videoId) {
    return (
      <div className="py-12 text-center text-xs text-[var(--text-muted)] space-y-2">
        <Film className="w-8 h-8 mx-auto opacity-40" />
        <p>กรุณาเลือกวิดีโอเพื่อดูฉากและแคปชัน</p>
      </div>
    );
  }

  if (loading && scenes.length === 0) {
    return (
      <div className="py-12 text-center text-xs text-[var(--text-muted)] space-y-2">
        <RefreshCw className="w-6 h-6 mx-auto animate-spin text-[var(--accent)]" />
        <p>กำลังโหลดรายการฉากและแคปชัน...</p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {/* Search & Refresh Controls */}
      <div className="flex items-center gap-2">
        <div className="relative flex-1">
          <Search className="w-3.5 h-3.5 absolute left-2.5 top-1/2 -translate-y-1/2 text-[var(--text-muted)]" />
          <input
            type="text"
            value={filterQuery}
            onChange={(e) => setFilterQuery(e.target.value)}
            placeholder="ค้นหาวัตถุหรือการกระทำในฉาก..."
            className="w-full pl-8 pr-3 py-1.5 rounded-xl text-xs bg-[var(--card-bg)] border border-[var(--line)] text-[var(--text)] placeholder:text-[var(--text-muted)] focus:outline-none focus:border-[var(--accent)] transition-colors shadow-xs"
          />
        </div>
        <button
          type="button"
          onClick={fetchScenes}
          title="รีเฟรชข้อมูลฉาก"
          className="p-1.5 rounded-xl border border-[var(--line)] bg-[var(--card-bg)] hover:bg-[var(--card-hover)] text-[var(--text-muted)] hover:text-[var(--text)] transition-colors shadow-xs"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
        </button>
      </div>

      {/* Summary Count Bar */}
      <div className="flex items-center justify-between text-[11px] text-[var(--text-muted)] px-1">
        <span>
          พบทั้งหมด <strong className="text-[var(--text)]">{scenes.length}</strong> ฉาก
          {filterQuery && ` (ตรง ${filteredScenes.length})`}
        </span>
        <span className="flex items-center gap-1 font-mono text-[10px]">
          <Sparkles className="w-3 h-3 text-[var(--accent)]" /> CapRL Q6 Multi-frame
        </span>
      </div>

      {/* Main Content Area */}
      <div className="space-y-2.5 max-h-[520px] overflow-y-auto pr-1">
        {filteredScenes.length === 0 ? (
          <div className="app-panel p-6 rounded-2xl border border-[var(--line)] text-center text-xs text-[var(--text-muted)] space-y-2">
            <AlertCircle className="w-7 h-7 mx-auto text-[var(--text-muted)] opacity-60" />
            <p className="font-semibold text-[var(--text)]">ไม่พบฉากที่ค้นหา</p>
            <p className="text-[11px]">ลองค้นหาด้วยคำอื่น หรือล้างคำค้นหา</p>
          </div>
        ) : (
          filteredScenes.map((scene) => {
            const isSelected = selectedSceneId === scene.scene_id;
            const thumbSrc = getThumbSrc(scene);
            const isBroken = brokenImageIds.has(scene.scene_id);
            const objects = scene.structured?.objects || [];
            const actions = scene.structured?.actions || [];
            const attributes = scene.structured?.attributes || [];
            const temporalEvents = scene.structured?.temporal_events || [];

            return (
              <div
                key={scene.scene_id}
                id={`scene-item-${scene.scene_id}`}
                className={`rounded-2xl border transition-all overflow-hidden ${
                  isSelected
                    ? "border-[var(--accent)] bg-[var(--accent-subtle)]/40 shadow-sm ring-2 ring-[var(--accent)]/50"
                    : "border-[var(--line)] bg-[var(--card-bg)] hover:border-[var(--line-strong)] hover:bg-[var(--card-hover)]"
                }`}
              >
                {/* 1. Compact List Row (Always Visible) */}
                <div
                  onClick={() => setSelectedSceneId(isSelected ? null : scene.scene_id)}
                  className="flex items-start gap-2.5 p-2.5 cursor-pointer select-none"
                >
                  {/* Small 16:9 Thumbnail Box */}
                  <div className="relative w-24 sm:w-28 aspect-video rounded-xl overflow-hidden bg-slate-900/90 flex-shrink-0 border border-[var(--line)] shadow-2xs group">
                    {thumbSrc && !isBroken ? (
                      <img
                        src={thumbSrc}
                        alt={`ฉากที่ ${scene.scene_index + 1}`}
                        onError={() => handleImageError(scene.scene_id)}
                        className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-200"
                        loading="lazy"
                      />
                    ) : (
                      <div className="w-full h-full flex flex-col items-center justify-center text-[var(--text-muted)] bg-[var(--panel)]">
                        <Film className="w-4 h-4 opacity-40 mb-0.5" />
                        <span className="text-[9px] font-mono">ฉาก {scene.scene_index + 1}</span>
                      </div>
                    )}

                    {/* Small Play Button Hover Indicator */}
                    <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity">
                      <div className="w-6 h-6 rounded-full bg-[var(--accent)] text-white flex items-center justify-center shadow-md">
                        <Play className="w-2.5 h-2.5 ml-0.5 fill-current" />
                      </div>
                    </div>

                    {/* Duration Notch on Thumbnail */}
                    <div className="absolute bottom-1 right-1 px-1 py-0.2 rounded bg-black/75 backdrop-blur-xs text-[9px] font-mono text-white/90">
                      {formatDuration(scene.duration_sec)}
                    </div>
                  </div>

                  {/* Scene Info & Caption Snippet Beside Thumbnail */}
                  <div className="flex-1 min-w-0 flex flex-col gap-1">
                    {/* Scene Number & Time Range Pill */}
                    <div className="flex items-center justify-between gap-1">
                      <div className="flex items-center gap-1.5 flex-wrap">
                        <span className="font-bold text-xs text-[var(--text)]">
                          ฉากที่ {scene.scene_index + 1}
                        </span>
                        <span className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-md bg-[var(--panel)] border border-[var(--line)] font-mono text-[10px] text-[var(--accent)] font-medium">
                          <Clock className="w-2.5 h-2.5" />
                          {formatTime(scene.t_start)} - {formatTime(scene.t_end)}
                        </span>
                      </div>

                      {/* Expand / Collapse Indicator */}
                      <div className="text-[var(--text-muted)] p-0.5">
                        {isSelected ? (
                          <ChevronUp className="w-3.5 h-3.5 text-[var(--accent)]" />
                        ) : (
                          <ChevronDown className="w-3.5 h-3.5" />
                        )}
                      </div>
                    </div>

                    {/* Caption Summary Snippet (Clamped to 2 lines) */}
                    <p className="text-[11px] text-[var(--text)] line-clamp-2 leading-snug">
                      {scene.summary || (
                        <span className="text-[var(--text-muted)] italic">
                          ยังไม่มีคำบรรยายสำหรับฉากนี้
                        </span>
                      )}
                    </p>

                    {/* Top Object Chips */}
                    {objects.length > 0 && (
                      <div className="flex items-center gap-1 flex-wrap pt-0.5">
                        {objects.slice(0, 3).map((obj, idx) => (
                          <span
                            key={idx}
                            className="px-1.5 py-0.2 rounded bg-[var(--panel)] border border-[var(--line)] text-[9px] text-[var(--text-muted)] font-medium"
                          >
                            {obj}
                          </span>
                        ))}
                        {objects.length > 3 && (
                          <span className="text-[9px] text-[var(--text-muted)]">
                            +{objects.length - 3}
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                </div>

                {/* 2. Expanded Detail View (Appears When Clicked) */}
                {isSelected && (
                  <div className="border-t border-[var(--line)] p-3 space-y-3 bg-[var(--panel)]/70 animate-in fade-in duration-200">
                    {/* Larger Image Banner with Quick Seek */}
                    {thumbSrc && !isBroken && (
                      <div
                        onClick={() => onSeekToTimestamp?.(scene.t_start)}
                        className="relative w-full aspect-video rounded-xl overflow-hidden bg-slate-900 cursor-pointer group shadow-sm"
                        title="คลิกเพื่อเล่นวิดีโอที่จุดเริ่มต้นของฉากนี้"
                      >
                        <img
                          src={thumbSrc}
                          alt={`ภาพขยายฉากที่ ${scene.scene_index + 1}`}
                          className="w-full h-full object-cover group-hover:scale-102 transition-transform duration-300"
                        />
                        <div className="absolute inset-0 bg-gradient-to-t from-black/75 via-transparent to-transparent" />

                        {/* Play Center Overlay */}
                        <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity bg-black/30">
                          <div className="px-3 py-1.5 rounded-full bg-[var(--accent)] text-white font-medium text-xs flex items-center gap-1.5 shadow-lg">
                            <Play className="w-3.5 h-3.5 fill-current" />
                            <span>เล่นฉากนี้ ({formatTime(scene.t_start)})</span>
                          </div>
                        </div>

                        {/* Top Badge */}
                        <div className="absolute top-2 left-2 flex items-center gap-1.5">
                          <span className="px-2 py-0.5 rounded-lg bg-black/70 backdrop-blur-md text-white font-bold text-[10px] border border-white/20">
                            ฉากที่ {scene.scene_index + 1}
                          </span>
                          <span className="px-2 py-0.5 rounded-lg bg-black/70 backdrop-blur-md text-cyan-300 font-mono text-[10px] border border-cyan-500/30">
                            {formatTime(scene.t_start)} - {formatTime(scene.t_end)}
                          </span>
                        </div>

                        {/* Bottom Stats Badge */}
                        <div className="absolute bottom-2 right-2 flex items-center gap-1.5 font-mono text-[9px] text-gray-300">
                          <span className="px-1.5 py-0.5 rounded bg-black/60 backdrop-blur-md border border-white/10">
                            ความยาว: {formatDuration(scene.duration_sec)}
                          </span>
                          <span className="px-1.5 py-0.5 rounded bg-black/60 backdrop-blur-md border border-white/10">
                            {scene.keyframe_count} เฟรม
                          </span>
                        </div>
                      </div>
                    )}

                    {/* Full Action Caption */}
                    <div className="space-y-1">
                      <h4 className="text-[10px] font-bold text-[var(--text-muted)] uppercase tracking-wider flex items-center gap-1">
                        <Sparkles className="w-3 h-3 text-[var(--accent)]" /> คำบรรยายฉากฉบับเต็ม (CapRL Q6)
                      </h4>
                      <p className="text-xs text-[var(--text)] leading-relaxed">
                        {scene.summary}
                      </p>
                    </div>

                    {/* Objects & Actions Breakdown */}
                    {(objects.length > 0 || actions.length > 0) && (
                      <div className="space-y-2 pt-1 border-t border-[var(--line)]">
                        {objects.length > 0 && (
                          <div>
                            <span className="text-[9px] font-bold text-[var(--text-muted)] uppercase tracking-wider block mb-1">
                              สิ่งที่พบในฉาก (Objects):
                            </span>
                            <div className="flex flex-wrap gap-1">
                              {objects.map((obj, idx) => (
                                <span
                                  key={idx}
                                  className="px-1.5 py-0.5 rounded-md bg-[var(--card-bg)] text-[var(--text)] border border-[var(--line)] text-[10px] font-medium"
                                >
                                  {obj}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}

                        {actions.length > 0 && (
                          <div>
                            <span className="text-[9px] font-bold text-[var(--text-muted)] uppercase tracking-wider block mb-1">
                              การเคลื่อนไหว/กริยา (Actions):
                            </span>
                            <div className="flex flex-wrap gap-1">
                              {actions.map((act, idx) => (
                                <span
                                  key={idx}
                                  className="px-1.5 py-0.5 rounded-md bg-emerald-50 dark:bg-emerald-950/50 text-emerald-800 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-800/60 text-[10px] font-medium"
                                >
                                  {act}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Attributes & Temporal Events */}
                    {(attributes.length > 0 || temporalEvents.length > 0) && (
                      <div className="space-y-2 pt-1 border-t border-[var(--line)]">
                        {attributes.length > 0 && (
                          <div>
                            <span className="text-[9px] font-bold text-[var(--text-muted)] uppercase tracking-wider block mb-1">
                              คุณลักษณะเด่น (Attributes):
                            </span>
                            <div className="flex flex-wrap gap-1">
                              {attributes.map((attr, idx) => (
                                <span
                                  key={idx}
                                  className="px-1.5 py-0.5 rounded bg-[var(--card-bg)] text-[var(--text-muted)] border border-[var(--line)] text-[10px]"
                                >
                                  {attr}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}

                        {temporalEvents.length > 0 && (
                          <div>
                            <span className="text-[9px] font-bold text-[var(--text-muted)] uppercase tracking-wider block mb-1">
                              ลำดับเหตุการณ์ย่อยตามเวลา (Temporal Events):
                            </span>
                            <ul className="list-disc list-inside space-y-0.5 text-[10px] text-[var(--text)]">
                              {temporalEvents.map((evt, idx) => (
                                <li key={idx} className="leading-snug">{evt}</li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Quick Seek Action Bar */}
                    <div className="pt-2 border-t border-[var(--line)] flex items-center justify-between">
                      <span className="text-[10px] font-mono text-[var(--text-muted)]">
                        ID: {scene.scene_id.slice(0, 8)}
                      </span>
                      <button
                        type="button"
                        onClick={() => onSeekToTimestamp?.(scene.t_start)}
                        className="inline-flex items-center gap-1.5 px-3 py-1 rounded-xl bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white text-xs font-semibold shadow-xs transition-colors"
                      >
                        <Play className="w-3 h-3 fill-current" />
                        <span>เล่นฉากนี้ทันที</span>
                      </button>
                    </div>
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
