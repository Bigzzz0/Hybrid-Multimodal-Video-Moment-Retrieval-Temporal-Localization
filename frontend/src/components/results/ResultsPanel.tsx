"use client";

import React, { useState, useEffect } from "react";
import {
  ListFilter,
  MessageSquare,
  Sparkles,
  Download,
  Bookmark,
  BookmarkCheck,
  Check,
  Copy,
  Clock,
  Loader2,
  Trash2,
  ChevronDown,
  ChevronUp,
  FileText,
  Bot,
  Maximize2,
  Film,
  Layers,
  ArrowRight,
} from "lucide-react";
import { MomentItem, VideoMetadata, SceneCaptionItem } from "@/lib/types";
import { apiClient } from "@/lib/api";
import { VideoQAPanel } from "@/components/rag/VideoQAPanel";
import { SceneCaptionsList } from "./SceneCaptionsList";
import { useLanguage } from "@/lib/language";

interface ResultsPanelProps {
  moments: MomentItem[];
  selectedVideo: VideoMetadata | null;
  searchQuery: string;
  activeMoment: MomentItem | null;
  onSelectMoment: (moment: MomentItem) => void;
  onExpandContext?: (moment: MomentItem) => void;
  onSeekToTimestamp: (time: number) => void;
  onSearchWithQuery?: (query: string) => void;
}

interface SavedSearch {
  id: string;
  query: string;
  tag?: string;
  date: string;
}

const DEFAULT_SAVED_SEARCHES: SavedSearch[] = [
  { id: "1", query: "คนปั่นจักรยาน", tag: "ถนน/ทางแยก", date: "26/10/2026" },
  { id: "2", query: "รถจอดขวางทางเข้า", tag: "ประตู", date: "26/10/2026" },
  { id: "3", query: "คนเข้า-ออก 22:00-06:00", tag: "ทุกวิดีโอ", date: "26/10/2026" },
];

export const ResultsPanel: React.FC<ResultsPanelProps> = ({
  moments,
  selectedVideo,
  searchQuery,
  activeMoment,
  onSelectMoment,
  onExpandContext,
  onSeekToTimestamp,
  onSearchWithQuery,
}) => {
  const { language, t } = useLanguage();
  const [activeTab, setActiveTab] = useState<"moments" | "scenes" | "rag">("moments");
  const [savedSearches, setSavedSearches] = useState<SavedSearch[]>([]);
  const [isExportingAll, setIsExportingAll] = useState(false);
  const [exportingIndex, setExportingIndex] = useState<number | null>(null);
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);
  const [isSavedCurrent, setIsSavedCurrent] = useState(false);
  const [expandedDetails, setExpandedDetails] = useState<Set<number>>(new Set());
  const [expandedContextIndices, setExpandedContextIndices] = useState<Set<number>>(new Set());
  const [scenes, setScenes] = useState<SceneCaptionItem[]>([]);
  const [scenesLoading, setScenesLoading] = useState(false);
  const [targetSceneId, setTargetSceneId] = useState<string | null>(null);

  const fetchScenes = async () => {
    if (!selectedVideo?.id) {
      setScenes([]);
      return;
    }
    setScenesLoading(true);
    try {
      const data = await apiClient.getVideoScenes(selectedVideo.id);
      setScenes(data || []);
    } catch (err) {
      console.error("Failed to load scenes in ResultsPanel:", err);
    } finally {
      setScenesLoading(false);
    }
  };

  useEffect(() => {
    void fetchScenes();
    setTargetSceneId(null);
    setExpandedContextIndices(new Set());
  }, [selectedVideo?.id]);

  const findMatchingScene = (m: MomentItem): SceneCaptionItem | null => {
    if (!scenes || scenes.length === 0) return null;
    const mid = (m.t_start + m.t_end) / 2;
    const enclosing = scenes.find((s) => s.t_start <= mid && mid <= s.t_end);
    if (enclosing) return enclosing;
    const overlapping = scenes.find((s) => s.t_start < m.t_end && s.t_end > m.t_start);
    if (overlapping) return overlapping;
    let best = scenes[0];
    let bestDist = Math.abs(scenes[0].t_start - m.t_start);
    for (let i = 1; i < scenes.length; i++) {
      const s = scenes[i];
      const dist = Math.min(Math.abs(s.t_start - m.t_start), Math.abs(s.t_end - m.t_end));
      if (dist < bestDist) {
        bestDist = dist;
        best = s;
      }
    }
    return best;
  };

  const handleToggleContext = (e: React.MouseEvent, m: MomentItem, idx: number) => {
    e.stopPropagation();
    const updated = new Set(expandedContextIndices);
    if (updated.has(idx)) {
      updated.delete(idx);
    } else {
      updated.add(idx);
    }
    setExpandedContextIndices(updated);
    if (onExpandContext) {
      onExpandContext(m);
    }
  };

  const handleJumpToScene = (e: React.MouseEvent, sceneId: string) => {
    e.stopPropagation();
    setTargetSceneId(sceneId);
    setActiveTab("scenes");
  };

  // Load saved searches from localStorage
  useEffect(() => {
    try {
      const stored = localStorage.getItem("videomoment_saved_searches");
      if (stored) {
        setSavedSearches(JSON.parse(stored));
      } else {
        const initialSearches: SavedSearch[] = language === "en" ? [
          { id: "1", query: "cyclist riding", tag: "Street", date: "26/10/2026" },
          { id: "2", query: "car parking by entrance", tag: "Gate", date: "26/10/2026" },
          { id: "3", query: "people walking 22:00-06:00", tag: "All Videos", date: "26/10/2026" },
        ] : DEFAULT_SAVED_SEARCHES;
        setSavedSearches(initialSearches);
        localStorage.setItem("videomoment_saved_searches", JSON.stringify(initialSearches));
      }
    } catch {
      setSavedSearches(DEFAULT_SAVED_SEARCHES);
    }
  }, [language]);

  const saveCurrentSearch = () => {
    if (!searchQuery.trim()) return;
    const tag = selectedVideo?.filename?.includes("street")
      ? (language === "th" ? "ถนน/ทางแยก" : "Street")
      : (language === "th" ? "วิดีโอนี้" : "This Video");
    const newEntry: SavedSearch = {
      id: Date.now().toString(),
      query: searchQuery.trim(),
      tag,
      date: new Date().toLocaleDateString(language === "th" ? "th-TH" : "en-US"),
    };
    const updated = [newEntry, ...savedSearches.filter((s) => s.query !== newEntry.query)].slice(0, 10);
    setSavedSearches(updated);
    try {
      localStorage.setItem("videomoment_saved_searches", JSON.stringify(updated));
    } catch {}
    setIsSavedCurrent(true);
    setTimeout(() => setIsSavedCurrent(false), 2500);
  };

  const removeSavedSearch = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const updated = savedSearches.filter((s) => s.id !== id);
    setSavedSearches(updated);
    try {
      localStorage.setItem("videomoment_saved_searches", JSON.stringify(updated));
    } catch {}
  };

  const formatSeconds = (sec: number) => {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
  };

  // Convert video time to real camera clock (base: 14:31:00)
  const getCameraTime = (elapsedSec: number) => {
    const totalSec = 14 * 3600 + 31 * 60 + Math.floor(elapsedSec);
    const h = Math.floor((totalSec / 3600) % 24);
    const m = Math.floor((totalSec % 3600) / 60);
    const s = Math.floor(totalSec % 60);
    return `${h.toString().padStart(2, "0")}:${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
  };

  const handleExportSingle = async (e: React.MouseEvent, m: MomentItem, idx: number) => {
    e.stopPropagation();
    if (!selectedVideo?.id) return;
    setExportingIndex(idx);
    try {
      const res = await apiClient.exportClip(selectedVideo.id, m.t_start, m.t_end);
      if (res && res.download_url) {
        const a = document.createElement("a");
        a.href = apiClient.getDownloadUrl(res.download_url);
        a.download = res.clip_filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
      }
    } catch (err) {
      console.error("Export clip failed:", err);
    } finally {
      setExportingIndex(null);
    }
  };

  const handleExportAll = async () => {
    if (!selectedVideo?.id || moments.length === 0) return;
    setIsExportingAll(true);
    for (let i = 0; i < moments.length; i++) {
      const m = moments[i];
      try {
        const res = await apiClient.exportClip(selectedVideo.id, m.t_start, m.t_end);
        if (res && res.download_url) {
          const a = document.createElement("a");
          a.href = apiClient.getDownloadUrl(res.download_url);
          a.download = res.clip_filename;
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
        }
      } catch (err) {
        console.error("Batch export failed for moment:", i, err);
      }
    }
    setIsExportingAll(false);
  };

  const handleCopyTime = (e: React.MouseEvent, m: MomentItem, idx: number) => {
    e.stopPropagation();
    navigator.clipboard.writeText(`${formatSeconds(m.t_start)} - ${formatSeconds(m.t_end)}`);
    setCopiedIndex(idx);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  const toggleDetails = (e: React.MouseEvent, idx: number) => {
    e.stopPropagation();
    const updated = new Set(expandedDetails);
    if (updated.has(idx)) updated.delete(idx);
    else updated.add(idx);
    setExpandedDetails(updated);
  };

  return (
    <aside className="w-full lg:w-80 xl:w-96 flex-shrink-0 flex flex-col gap-3.5">
      {/* Top Mode Navigation Tabs */}
      <div className="app-panel p-1 rounded-2xl flex items-center gap-1 border border-[var(--line)]">
        <button
          type="button"
          onClick={() => setActiveTab("moments")}
          className={`flex-1 py-2 px-2 rounded-xl text-xs font-semibold flex items-center justify-center gap-1 transition-all ${
            activeTab === "moments"
              ? "bg-[var(--accent)] text-white shadow-sm"
              : "text-[var(--muted)] hover:text-[var(--text)] hover:bg-[var(--line)]/30"
          }`}
          title="รายการช่วงเวลาที่ค้นพบ"
        >
          <ListFilter className="w-3.5 h-3.5 flex-shrink-0" />
          <span className="truncate">{t("results.foundMoments")} ({moments.length})</span>
        </button>

        <button
          type="button"
          onClick={() => setActiveTab("scenes")}
          className={`flex-1 py-2 px-2 rounded-xl text-xs font-semibold flex items-center justify-center gap-1 transition-all ${
            activeTab === "scenes"
              ? "bg-[var(--accent)] text-white shadow-sm"
              : "text-[var(--muted)] hover:text-[var(--text)] hover:bg-[var(--line)]/30"
          }`}
          title="พรีวิวและแคปชันของแต่ละฉาก"
        >
          <Film className="w-3.5 h-3.5 flex-shrink-0" />
          <span className="truncate">{t("results.scenes")}</span>
        </button>

        <button
          type="button"
          onClick={() => setActiveTab("rag")}
          className={`flex-1 py-2 px-2 rounded-xl text-xs font-semibold flex items-center justify-center gap-1 transition-all ${
            activeTab === "rag"
              ? "bg-[var(--accent)] text-white shadow-sm"
              : "text-[var(--muted)] hover:text-[var(--text)] hover:bg-[var(--line)]/30"
          }`}
          title="ถามตอบเกี่ยวกับวิดีโอ"
        >
          <MessageSquare className="w-3.5 h-3.5 flex-shrink-0" />
          <span className="truncate">{t("results.ragAI")}</span>
        </button>
      </div>


      {/* Main Tab Content */}
      {activeTab === "moments" ? (
        <div className="flex flex-col gap-3.5">
          {/* Header Summary */}
          <div className="px-1">
            <h3 className="text-sm font-bold text-[var(--text)]">
              {t("results.found")} {moments.length} {t("results.momentsCount")}
            </h3>
            <p className="text-xs text-[var(--muted)] line-clamp-1">
              {t("results.searchIn")} "{searchQuery || t("results.allActions")}" {t("results.inVideo")} {selectedVideo?.filename || t("results.videoFallback")} · {t("results.rankedBy")}
            </p>
          </div>

          {/* Moment Cards List */}
          <div className="flex flex-col gap-2.5 max-h-[520px] overflow-y-auto pr-1">
            {moments.length === 0 ? (
              <div className="app-panel p-6 rounded-2xl border border-[var(--line)] text-center text-xs text-[var(--muted)] space-y-2">
                <Sparkles className="w-8 h-8 mx-auto text-[var(--muted)] opacity-60" />
                <p className="font-semibold">{t("results.noMoments")}</p>
                <p className="text-[11px]">{t("results.noMomentsPrompt")}</p>
              </div>
            ) : (
              moments.map((m, idx) => {
                const isActive =
                  activeMoment?.t_start === m.t_start && activeMoment?.t_end === m.t_end;
                const scorePct = Math.round((m.score || 0.85) * 100);
                const isExpanded = expandedDetails.has(idx);
                const isContextExpanded = expandedContextIndices.has(idx);
                const matchedScene = findMatchingScene(m);

                return (
                  <div
                    key={`${m.t_start}-${m.t_end}-${idx}`}
                    onClick={() => onSelectMoment(m)}
                    className={`app-panel p-3.5 rounded-2xl border transition-all cursor-pointer group flex flex-col gap-2 ${
                      isActive
                        ? "border-[var(--accent)] shadow-md ring-1 ring-[var(--accent)]/50"
                        : "border-[var(--line)] hover:border-[var(--accent)]/60 hover:shadow-sm"
                    }`}
                  >
                    {/* Top Section: Frame Thumbnail + Details */}
                    <div className="flex items-start gap-3">
                      {/* Frame Thumbnail Preview */}
                      <div className="relative w-28 h-18 rounded-xl overflow-hidden bg-black/85 flex-shrink-0 border border-[var(--line)] group-hover:border-[var(--accent)] transition-all">
                        {m.preview_frame_path ? (
                          <img
                            src={apiClient.getFramePreviewUrl(m.preview_frame_path)}
                            alt={`Preview at ${m.t_start}s`}
                            className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                          />
                        ) : (
                          <div className="w-full h-full flex items-center justify-center text-[var(--muted)] text-[10px] font-mono">
                            No Frame
                          </div>
                        )}
                        <span className="absolute bottom-1 right-1 px-1.5 py-0.5 rounded bg-black/80 backdrop-blur-sm text-[9px] font-mono font-bold text-white border border-white/10">
                          {Math.max(0.1, m.t_end - m.t_start).toFixed(1)}s
                        </span>
                      </div>

                      {/* Moment Information */}
                      <div className="flex-1 min-w-0 flex flex-col gap-1.5">
                        <div className="flex items-center justify-between">
                          <span className="font-mono font-bold text-sm text-[var(--text)]">
                            {formatSeconds(m.t_start)} – {formatSeconds(m.t_end)}
                          </span>
                          <span className="font-mono font-bold text-sm text-[var(--accent)]">
                            {scorePct}%
                          </span>
                        </div>

                        <div className="text-xs text-[var(--muted)] line-clamp-1">
                          {m.caption_preview || matchedScene?.summary || searchQuery || t("results.detectedMoment")}
                        </div>

                        <div className="font-mono text-[10px] text-[var(--muted)]">
                          {t("results.cameraTime")} {getCameraTime(m.t_start)}
                        </div>

                        {/* Blue Match Progress Bar */}
                        <div className="w-full h-1.5 rounded-full bg-[var(--line)] overflow-hidden mt-0.5">
                          <div
                            className="h-full bg-[var(--accent)] rounded-full transition-all duration-300"
                            style={{ width: `${Math.max(5, Math.min(100, scorePct))}%` }}
                          />
                        </div>
                      </div>
                    </div>

                    {/* Row 4: Hover / Action Buttons */}
                    <div className="flex items-center justify-between pt-1 text-[11px] text-[var(--muted)] border-t border-[var(--line)]/50 mt-1 opacity-80 group-hover:opacity-100 transition-opacity">
                      <div className="flex items-center gap-2">
                        <button
                          type="button"
                          onClick={(e) => handleCopyTime(e, m, idx)}
                          title={t("results.copyTooltip")}
                          className="hover:text-[var(--accent)] flex items-center gap-1 font-mono transition-colors"
                        >
                          {copiedIndex === idx ? (
                            <>
                              <Check className="w-3 h-3 text-emerald-500" />
                              <span className="text-emerald-500 text-[10px]">{t("results.copied")}</span>
                            </>
                          ) : (
                            <>
                              <Copy className="w-3 h-3" />
                              <span className="text-[10px]">{t("results.copy")}</span>
                            </>
                          )}
                        </button>

                        <button
                          type="button"
                          onClick={(e) => handleToggleContext(e, m, idx)}
                          title={t("results.contextTooltip")}
                          className={`flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded-md transition-colors ${
                            isContextExpanded
                              ? "bg-[var(--accent)] text-white font-medium shadow-xs"
                              : "hover:text-[var(--accent)] hover:bg-[var(--line)]/40"
                          }`}
                        >
                          <Maximize2 className="w-3 h-3" />
                          <span>{t("results.context")}</span>
                        </button>
                      </div>

                      <div className="flex items-center gap-1.5">
                        {(m.caption_full || m.verifier_evidence || matchedScene?.summary) && (
                          <button
                            type="button"
                            onClick={(e) => toggleDetails(e, idx)}
                            className="hover:text-[var(--accent)] text-[10px] flex items-center gap-0.5"
                          >
                            <span>{t("results.details")}</span>
                            {isExpanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                          </button>
                        )}

                        <button
                          type="button"
                          onClick={(e) => handleExportSingle(e, m, idx)}
                          disabled={exportingIndex === idx}
                          className="hover:text-[var(--accent)] flex items-center gap-1 text-[10px] transition-colors"
                          title={language === "th" ? "ตัดและดาวน์โหลดคลิปช่วงนี้" : "Export clip"}
                        >
                          {exportingIndex === idx ? (
                            <Loader2 className="w-3 h-3 animate-spin text-[var(--accent)]" />
                          ) : (
                            <Download className="w-3 h-3" />
                          )}
                          <span>{t("results.export")}</span>
                        </button>
                      </div>
                    </div>

                    {/* Expandable Context & Scene Drawer */}
                    {isContextExpanded && (
                      <div
                        onClick={(e) => e.stopPropagation()}
                        className="p-3 rounded-xl bg-[var(--bg)] border border-[var(--line)] text-[11px] space-y-2 mt-1 text-[var(--text)] animate-in fade-in duration-150 shadow-xs"
                      >
                        {/* Context Window Header */}
                        <div className="flex items-center justify-between border-b border-[var(--line)]/60 pb-1.5">
                          <div className="flex items-center gap-1.5 text-[var(--accent)] font-semibold text-[11px]">
                            <Layers className="w-3.5 h-3.5" />
                            <span>{t("results.contextWindow")}</span>
                          </div>
                          <span className="font-mono text-[10px] text-[var(--text)] bg-[var(--panel)] px-2 py-0.5 rounded-md border border-[var(--line)] font-medium">
                            {formatSeconds(m.context_t_start ?? Math.max(0, m.t_start - 2))} – {formatSeconds(m.context_t_end ?? Math.min(selectedVideo?.duration_sec || m.t_end + 2, m.t_end + 2))}
                          </span>
                        </div>

                        {/* Matched Scene Card */}
                        {matchedScene ? (
                          <div className="space-y-2">
                            <div className="flex items-center justify-between gap-2">
                              <div className="flex items-center gap-1.5">
                                <Film className="w-3.5 h-3.5 text-[var(--accent)]" />
                                <span className="font-bold text-[var(--text)] text-xs">
                                  {language === "th" ? `ฉากที่ ${matchedScene.scene_index + 1}` : `Scene ${matchedScene.scene_index + 1}`}
                                </span>
                                <span className="text-[10px] font-mono text-[var(--muted)]">
                                  ({formatSeconds(matchedScene.t_start)} - {formatSeconds(matchedScene.t_end)} · {matchedScene.duration_sec.toFixed(1)}s)
                                </span>
                              </div>

                              <button
                                type="button"
                                onClick={(e) => handleJumpToScene(e, matchedScene.scene_id)}
                                className="flex items-center gap-1 text-[10px] text-[var(--accent)] hover:underline font-semibold bg-[var(--accent)]/10 hover:bg-[var(--accent)]/20 px-2 py-0.5 rounded-md transition-colors"
                                title={language === "th" ? "เปิดดูฉากนี้ในแท็บฉาก & แคปชัน" : "Open in Scenes & Captions tab"}
                              >
                                <span>{t("results.openInScenes")}</span>
                                <ArrowRight className="w-3 h-3" />
                              </button>
                            </div>

                            {/* Scene Summary / Caption */}
                            <div className="p-2 rounded-lg bg-[var(--panel)] border border-[var(--line)] space-y-1">
                              <div className="text-[10px] uppercase font-semibold text-[var(--muted)] flex items-center gap-1">
                                <Sparkles className="w-3 h-3 text-[var(--accent)]" />
                                <span>{t("results.sceneCaption")}</span>
                              </div>
                              <p className="text-xs leading-relaxed text-[var(--text)] font-normal">
                                {matchedScene.summary || m.caption_full || (
                                  <span className="text-[var(--muted)] italic">
                                    {t("results.noSceneCaption")}
                                  </span>
                                )}
                              </p>
                            </div>

                            {/* Action & Object Tags */}
                            {((matchedScene.structured?.actions && matchedScene.structured.actions.length > 0) ||
                              (matchedScene.structured?.objects && matchedScene.structured.objects.length > 0)) && (
                              <div className="flex flex-wrap items-center gap-1 pt-0.5">
                                {matchedScene.structured?.actions?.map((act, i) => (
                                  <span
                                    key={`act-${i}`}
                                    className="px-1.5 py-0.5 rounded text-[9px] bg-[var(--accent)]/15 text-[var(--accent)] border border-[var(--accent)]/30 font-medium"
                                  >
                                    {act}
                                  </span>
                                ))}
                                {matchedScene.structured?.objects?.map((obj, i) => (
                                  <span
                                    key={`obj-${i}`}
                                    className="px-1.5 py-0.5 rounded text-[9px] bg-[var(--line)]/50 text-[var(--text)] border border-[var(--line)]"
                                  >
                                    {obj}
                                  </span>
                                ))}
                              </div>
                            )}
                          </div>
                        ) : (
                          <div className="text-[11px] text-[var(--muted)] italic py-1">
                            {scenesLoading ? t("results.loadingContext") : (m.caption_full || t("results.noSceneCaption"))}
                          </div>
                        )}
                      </div>
                    )}

                    {/* Expandable Caption / Evidence Drawer */}
                    {isExpanded && !isContextExpanded && (
                      <div
                        onClick={(e) => e.stopPropagation()}
                        className="p-2.5 rounded-xl bg-[var(--bg)] border border-[var(--line)] text-[11px] space-y-2 mt-1 text-[var(--text)] animate-in fade-in duration-150"
                      >
                        {(m.caption_full || matchedScene?.summary) && (
                          <div>
                            <div className="font-semibold text-[10px] uppercase tracking-wider text-[var(--muted)] mb-0.5">
                              {language === "th" ? "Caption คำบรรยาย" : "Generated Caption"}
                            </div>
                            <p className="font-mono text-[10px] leading-relaxed break-words whitespace-pre-wrap text-[var(--muted)]">
                              {m.caption_full || matchedScene?.summary}
                            </p>
                          </div>
                        )}
                        {m.verifier_evidence && (
                          <div>
                            <div className="font-semibold text-[10px] uppercase tracking-wider text-[var(--accent)] mb-0.5">
                              Accurate Verifier
                            </div>
                            <div className="flex flex-wrap gap-2 text-[10px] font-mono text-[var(--muted)]">
                              <span>Event: {String(m.verifier_evidence.event_present)}</span>
                              <span>Conf: {m.verifier_evidence.confidence.toFixed(2)}</span>
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })
            )}
          </div>

          {/* Action Buttons */}
          <div className="flex flex-col gap-2 pt-1">
            <button
              type="button"
              disabled={isExportingAll || moments.length === 0}
              onClick={handleExportAll}
              className="w-full py-2.5 px-4 rounded-xl bg-[var(--accent)] hover:opacity-95 text-white font-semibold text-xs flex items-center justify-center gap-2 shadow-sm transition-all disabled:opacity-50 cursor-pointer"
            >
              {isExportingAll ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>{t("results.exportingAll")}</span>
                </>
              ) : (
                <>
                  <Download className="w-4 h-4" />
                  <span>{t("results.exportAll")}</span>
                </>
              )}
            </button>

            <button
              type="button"
              onClick={saveCurrentSearch}
              className="w-full py-2.5 px-4 rounded-xl border border-[var(--line)] hover:border-[var(--accent)] bg-[var(--panel)] text-[var(--text)] font-semibold text-xs flex items-center justify-center gap-2 transition-all cursor-pointer"
            >
              {isSavedCurrent ? (
                <>
                  <BookmarkCheck className="w-4 h-4 text-emerald-500" />
                  <span className="text-emerald-500">{t("results.savedSearch")}</span>
                </>
              ) : (
                <>
                  <Bookmark className="w-4 h-4 text-[var(--muted)]" />
                  <span>{t("results.saveSearch")}</span>
                </>
              )}
            </button>
          </div>

          {/* Saved Searches Section */}
          <div className="pt-2 border-t border-[var(--line)] space-y-2">
            <h4 className="text-xs font-semibold text-[var(--muted)] px-1">
              {t("results.savedSearches")}
            </h4>

            <div className="flex flex-col gap-1.5">
              {savedSearches.map((item) => (
                <div
                  key={item.id}
                  onClick={() => onSearchWithQuery?.(item.query)}
                  className="flex items-center justify-between p-2 rounded-xl border border-transparent hover:border-[var(--line)] hover:bg-[var(--panel)] text-xs text-[var(--text)] cursor-pointer group transition-all"
                >
                  <div className="flex items-center gap-2 line-clamp-1 pr-2">
                    <Bookmark className="w-3.5 h-3.5 text-[var(--muted)] group-hover:text-[var(--accent)] flex-shrink-0" />
                    <span className="font-medium group-hover:text-[var(--accent)] transition-colors">
                      {item.query}
                    </span>
                    {item.tag && (
                      <span className="text-[10px] text-[var(--muted)]">
                        · {item.tag}
                      </span>
                    )}
                  </div>

                  <button
                    type="button"
                    onClick={(e) => removeSavedSearch(item.id, e)}
                    className="opacity-0 group-hover:opacity-100 p-1 text-[var(--muted)] hover:text-red-500 transition-opacity"
                    title={t("results.deleteSearchTooltip")}
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : activeTab === "scenes" ? (
        /* Scene Captions Explorer Tab */
        <div className="app-panel p-3.5 rounded-2xl border border-[var(--line)] min-h-[500px]">
          <SceneCaptionsList
            videoId={selectedVideo?.id}
            onSeekToTimestamp={onSeekToTimestamp}
            scenes={scenes}
            loading={scenesLoading}
            onRefreshScenes={fetchScenes}
            initialSelectedSceneId={targetSceneId}
          />
        </div>
      ) : (
        /* Video-RAG AI Tab */
        <div className="app-panel p-3.5 rounded-2xl border border-[var(--line)] min-h-[500px]">
          <VideoQAPanel
            videoId={selectedVideo?.id}
            onSeekToTimestamp={onSeekToTimestamp}
          />
        </div>
      )}

    </aside>
  );
};
