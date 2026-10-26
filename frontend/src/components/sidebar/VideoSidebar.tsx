"use client";

import React, { useState, useMemo } from "react";
import { Plus, Search, Folder, Video, Trash2, CheckCircle2, Clock } from "lucide-react";
import { VideoMetadata } from "@/lib/types";
import { useLanguage } from "@/lib/language";

interface VideoSidebarProps {
  videos: VideoMetadata[];
  selectedVideo: VideoMetadata | null;
  onSelectVideo: (video: VideoMetadata) => void;
  onOpenUpload: () => void;
  onRequestDelete: (video: VideoMetadata) => void;
}

type CategoryType = "all" | "gate" | "street" | "parking" | "classroom";

const CATEGORIES: { id: CategoryType; label: string; keywords: string[] }[] = [
  { id: "all", label: "ทั้งหมด", keywords: [] },
  { id: "gate", label: "ประตู / ทางเข้า", keywords: ["ประตู", "ทางเข้า", "gate", "door", "entrance", "cửa"] },
  { id: "street", label: "ถนน / ทางแยก", keywords: ["ถนน", "ทางแยก", "street", "road", "intersection", "traffic", "cctv", "ฝ่าน้ำท่วม"] },
  { id: "parking", label: "ลานจอดรถ", keywords: ["ลานจอด", "จอดรถ", "parking", "garage"] },
  { id: "classroom", label: "ห้องเรียน", keywords: ["ห้องเรียน", "ห้อง", "classroom", "class", "people-detection", "room", "hub"] },
];

export const VideoSidebar: React.FC<VideoSidebarProps> = ({
  videos,
  selectedVideo,
  onSelectVideo,
  onOpenUpload,
  onRequestDelete,
}) => {
  const [filterQuery, setFilterQuery] = useState("");
  const [selectedCategory, setSelectedCategory] = useState<CategoryType>("all");
  const { t } = useLanguage();

  const formatDuration = (secs?: number) => {
    if (!secs) return "00:00";
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
  };

  // Category counts
  const categoryCounts = useMemo(() => {
    const counts: Record<CategoryType, number> = {
      all: videos.length,
      gate: 0,
      street: 0,
      parking: 0,
      classroom: 0,
    };

    videos.forEach((vid) => {
      const name = (vid.filename || "").toLowerCase();
      CATEGORIES.forEach((cat) => {
        if (cat.id === "all") return;
        if (cat.keywords.some((k) => name.includes(k.toLowerCase()))) {
          counts[cat.id] += 1;
        }
      });
    });

    return counts;
  }, [videos]);

  // Status counts
  const indexedCount = useMemo(
    () => videos.filter((v) => v.ingestion_phase === "phase1_ready" || v.ingestion_phase === "caption_ready" || !v.ingestion_phase).length,
    [videos]
  );
  const processingCount = useMemo(
    () => videos.filter((v) => v.ingestion_phase === "processing" || v.ingestion_phase === "pending").length,
    [videos]
  );

  // Filtered videos
  const filteredVideos = useMemo(() => {
    return videos.filter((vid) => {
      const name = (vid.filename || "").toLowerCase();
      // 1. Text filter
      if (filterQuery.trim()) {
        const q = filterQuery.toLowerCase().trim();
        if (!name.includes(q)) return false;
      }
      // 2. Category filter
      if (selectedCategory !== "all") {
        const cat = CATEGORIES.find((c) => c.id === selectedCategory);
        if (cat && !cat.keywords.some((k) => name.includes(k.toLowerCase()))) {
          return false;
        }
      }
      return true;
    });
  }, [videos, filterQuery, selectedCategory]);

  return (
    <aside className="flex h-full w-full flex-col gap-3.5 rounded-2xl border border-[var(--line)] bg-[var(--panel)] p-3.5 shadow-sm transition-colors lg:w-[280px] lg:flex-shrink-0">
      {/* 1. Header & Add Video Button */}
      <div className="flex items-center justify-between gap-2">
        <h2 className="text-sm font-bold text-[var(--text)]">{t("sidebar.title")}</h2>
        <button
          type="button"
          onClick={onOpenUpload}
          className="inline-flex items-center gap-1 rounded-lg bg-[var(--accent)] px-2.5 py-1.5 text-xs font-semibold text-[var(--accent-text)] shadow-sm transition-opacity hover:opacity-90"
        >
          <Plus className="h-3.5 w-3.5" />
          <span>{t("sidebar.import")}</span>
        </button>
      </div>

      {/* 2. Search / Filter Input */}
      <div className="space-y-1">
        <label htmlFor="sidebar-filter-input" className="text-[11px] font-medium text-[var(--text-muted)]">
          {t("sidebar.filterLabel")}
        </label>
        <div className="relative">
          <input
            id="sidebar-filter-input"
            type="text"
            value={filterQuery}
            onChange={(e) => setFilterQuery(e.target.value)}
            placeholder={t("sidebar.searchPlaceholder")}
            className="w-full rounded-lg border border-[var(--line)] bg-[var(--bg)] py-1.5 pl-8 pr-2.5 text-xs text-[var(--text)] placeholder-[var(--text-muted)] focus:border-[var(--accent)] focus:outline-none"
          />
          <Search className="absolute left-2.5 top-2 h-3.5 w-3.5 text-[var(--text-muted)]" />
        </div>
      </div>

      {/* 3. Folders / Cameras List */}
      <div className="space-y-1">
        <p className="text-[11px] font-medium text-[var(--text-muted)]">{t("sidebar.folders")}</p>
        <div className="space-y-0.5">
          {CATEGORIES.map((cat) => {
            const count = categoryCounts[cat.id];
            const isActive = selectedCategory === cat.id;
            return (
              <button
                key={cat.id}
                type="button"
                onClick={() => setSelectedCategory(cat.id)}
                className={`flex w-full items-center justify-between rounded-lg px-2.5 py-1.5 text-xs transition-colors ${
                  isActive
                    ? "bg-[var(--accent-subtle)] font-semibold text-[var(--accent)]"
                    : "text-[var(--text)] hover:bg-[var(--card-hover)]"
                }`}
              >
                <div className="flex items-center gap-2 truncate">
                  <Folder className={`h-3.5 w-3.5 ${isActive ? "text-[var(--accent)] fill-current/20" : "text-[var(--text-muted)]"}`} />
                  <span className="truncate">{t(`sidebar.cat.${cat.id}`, cat.label)}</span>
                </div>
                <span className={`text-[10px] font-mono ${isActive ? "text-[var(--accent)]" : "text-[var(--text-muted)]"}`}>
                  {count}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {/* 4. Status Indicator Summary */}
      <div className="space-y-1">
        <p className="text-[11px] font-medium text-[var(--text-muted)]">{t("sidebar.status")}</p>
        <div className="flex flex-wrap items-center gap-2 text-[11px]">
          <span className="inline-flex items-center gap-1 text-emerald-600 dark:text-emerald-400">
            <span className="h-2 w-2 rounded-full bg-emerald-500" />
            <span>{t("sidebar.indexed")} {indexedCount}</span>
          </span>
          {processingCount > 0 && (
            <span className="inline-flex items-center gap-1 text-amber-600 dark:text-amber-400">
              <span className="h-2 w-2 rounded-full bg-amber-500 animate-pulse" />
              <span>{t("sidebar.processing")} {processingCount}</span>
            </span>
          )}
        </div>
      </div>

      {/* 5. Video Items List */}
      <div className="flex min-h-0 flex-1 flex-col space-y-1">
        <div className="flex items-center justify-between">
          <p className="text-[11px] font-medium text-[var(--text-muted)]">
            {t("sidebar.videos")} ({filteredVideos.length})
          </p>
        </div>

        <div className="flex-1 space-y-1.5 overflow-y-auto pr-1">
          {filteredVideos.length === 0 ? (
            <div className="rounded-xl border border-dashed border-[var(--line)] p-4 text-center text-xs text-[var(--text-muted)]">
              {t("sidebar.noVideos")}
            </div>
          ) : (
            filteredVideos.map((vid) => {
              const isSelected = selectedVideo?.id === vid.id;
              const isProcessing = vid.ingestion_phase === "processing" || vid.ingestion_phase === "pending";

              return (
                <div
                  key={vid.id}
                  onClick={() => onSelectVideo(vid)}
                  className={`group relative flex cursor-pointer items-start gap-2.5 rounded-xl border p-2.5 transition-all ${
                    isSelected
                      ? "border-[var(--accent)] bg-[var(--accent-subtle)] shadow-sm"
                      : "border-[var(--line)] bg-[var(--card-bg)] hover:bg-[var(--card-hover)]"
                  }`}
                >
                  <div
                    className={`mt-0.5 flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-lg border ${
                      isSelected
                        ? "border-[var(--accent)] bg-[var(--panel)] text-[var(--accent)]"
                        : "border-[var(--line)] bg-[var(--bg)] text-[var(--text-muted)]"
                    }`}
                  >
                    <Video className="h-3.5 w-3.5" />
                  </div>

                  <div className="min-w-0 flex-1">
                    <p
                      className={`truncate text-xs font-medium ${
                        isSelected ? "font-semibold text-[var(--accent)]" : "text-[var(--text)]"
                      }`}
                      title={vid.filename}
                    >
                      {vid.filename}
                    </p>
                    <p className="mt-0.5 text-[10px] font-mono text-[var(--text-muted)]">
                      {formatDuration(vid.duration_sec)} · {Math.round(vid.fps || 25)} fps · {vid.resolution || "1080p"}
                    </p>
                    <div className="mt-1 flex items-center gap-1.5 text-[10px]">
                      {isProcessing ? (
                        <span className="inline-flex items-center gap-1 font-mono text-amber-600 dark:text-amber-400">
                          <span className="h-1.5 w-1.5 rounded-full bg-amber-500 animate-pulse" />
                          <span>{t("sidebar.processing")}</span>
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 font-mono text-emerald-600 dark:text-emerald-400">
                          <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
                          <span>{t("sidebar.indexed")}</span>
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Delete Button */}
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      onRequestDelete(vid);
                    }}
                    className="absolute right-2 top-2 rounded p-1 text-[var(--text-muted)] opacity-0 transition-opacity hover:bg-rose-500/20 hover:text-rose-500 group-hover:opacity-100"
                    title={`${t("sidebar.delete")} "${vid.filename}"`}
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </div>
              );
            })
          )}
        </div>
      </div>
    </aside>
  );
};
