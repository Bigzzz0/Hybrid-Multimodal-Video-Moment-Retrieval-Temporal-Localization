"use client";

import React, { useState, useEffect, useRef } from "react";
import { Search, Sparkles, X, Clock, AlertTriangle, Loader2, RefreshCw } from "lucide-react";
import { apiClient } from "@/lib/api";
import { CaptionStatus } from "@/lib/types";
import { useLanguage, Language } from "@/lib/language";

interface SearchSectionProps {
  searchQuery: string;
  onSearchQueryChange: (q: string) => void;
  onSearch: (q?: string) => void;
  isSearching: boolean;
  searchProfile: "fast" | "accurate";
  onSearchProfileChange: (p: "fast" | "accurate") => void;
  videoId?: string;
  searchInputRef: React.RefObject<HTMLInputElement>;
}

type ExampleCategory = "transit" | "classroom" | "vehicle" | "action";

const EXAMPLE_TABS_CONFIG: Record<Language, { id: ExampleCategory; label: string; queries: string[] }[]> = {
  th: [
    {
      id: "transit",
      label: "การสัญจร",
      queries: ["คนปั่นจักรยาน", "คนเดินริมถนน", "รถยนต์ขับผ่าน", "คนข้ามถนน"],
    },
    {
      id: "classroom",
      label: "ห้องเรียน",
      queries: ["ยกมือถาม", "เดินเข้าห้อง", "ครูเขียนกระดาน", "นักเรียนลุกขึ้น"],
    },
    {
      id: "vehicle",
      label: "ยานพาหนะ",
      queries: ["รถเลี้ยวซ้าย", "รถพยาบาลผ่าน", "มอเตอร์ไซค์ขับเร็ว", "รถจอดริมทาง"],
    },
    {
      id: "action",
      label: "ท่าทาง",
      queries: ["คนกำลังวิ่ง", "คนถือกระเป๋า", "คนยืนคุยกัน", "คนใส่หมวก"],
    },
  ],
  en: [
    {
      id: "transit",
      label: "Transit",
      queries: ["cyclist riding", "pedestrian walking", "car driving by", "crossing the street"],
    },
    {
      id: "classroom",
      label: "Classroom",
      queries: ["raising hand", "walking into room", "writing on whiteboard", "standing up"],
    },
    {
      id: "vehicle",
      label: "Vehicles",
      queries: ["car turning left", "ambulance passing", "speeding motorcycle", "car parking"],
    },
    {
      id: "action",
      label: "Actions",
      queries: ["person running", "carrying a backpack", "people talking", "person wearing hat"],
    },
  ],
};

export const SearchSection: React.FC<SearchSectionProps> = ({
  searchQuery,
  onSearchQueryChange,
  onSearch,
  isSearching,
  searchProfile,
  onSearchProfileChange,
  videoId,
  searchInputRef,
}) => {
  const { language, t } = useLanguage();
  const currentExampleTabs = EXAMPLE_TABS_CONFIG[language] || EXAMPLE_TABS_CONFIG.th;
  const [scope, setScope] = useState<"video" | "folder" | "all">("video");
  const [activeTab, setActiveTab] = useState<ExampleCategory>("transit");
  const [recentQueries, setRecentQueries] = useState<string[]>([]);
  const [captionStatus, setCaptionStatus] = useState<CaptionStatus | null>(null);
  const [selectedDensity, setSelectedDensity] = useState<number>(8);
  const [isRebuilding, setIsRebuilding] = useState<boolean>(false);
  const [isCancelling, setIsCancelling] = useState<boolean>(false);

  // Load recent queries from localStorage
  useEffect(() => {
    try {
      const saved = JSON.parse(localStorage.getItem("videomoment_recent_queries") || "[]");
      if (Array.isArray(saved) && saved.length > 0) {
        setRecentQueries(saved.slice(0, 6));
      } else {
        setRecentQueries(["blue shirt", "black shirt guy", "yellow hat person"]);
      }
    } catch (_) {
      setRecentQueries(["blue shirt", "black shirt guy", "yellow hat person"]);
    }
  }, [searchQuery]);

  // Load and poll caption status
  useEffect(() => {
    if (!videoId) {
      setCaptionStatus(null);
      return;
    }
    let active = true;
    const fetchStatus = async () => {
      try {
        const next = await apiClient.getCaptionStatus(videoId);
        if (active) setCaptionStatus(next);
      } catch (_) {}
    };
    void fetchStatus();
    const interval = window.setInterval(fetchStatus, 3000);
    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, [videoId]);

  const handleRebuild = async (density: number) => {
    if (!videoId || isRebuilding) return;
    setIsRebuilding(true);
    try {
      await apiClient.rebuildCaptions(videoId, density, true);
      const next = await apiClient.getCaptionStatus(videoId);
      setCaptionStatus(next);
    } catch (_) {
    } finally {
      setIsRebuilding(false);
    }
  };

  const handleCancelCaption = async () => {
    if (!videoId || isCancelling) return;
    setIsCancelling(true);
    try {
      await apiClient.cancelCaptions(videoId);
      const next = await apiClient.getCaptionStatus(videoId);
      setCaptionStatus(next);
    } catch (_) {
    } finally {
      setIsCancelling(false);
    }
  };

  const isRunning = captionStatus?.status === "running" || captionStatus?.status === "pending";
  const isErrorOrUnavailable =
    captionStatus?.status === "error" ||
    captionStatus?.status === "unavailable" ||
    captionStatus?.status === "cancelled";

  return (
    <div className="flex flex-col gap-3 rounded-2xl border border-[var(--line)] bg-[var(--panel)] p-4 shadow-sm transition-colors">
      {/* 1. Main Search Bar */}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          onSearch();
        }}
        className="flex items-center gap-2.5"
      >
        <div className="relative flex-1">
          <input
            ref={searchInputRef}
            type="text"
            value={searchQuery}
            onChange={(e) => onSearchQueryChange(e.target.value)}
            placeholder={t("search.placeholder")}
            className="w-full rounded-xl border border-[var(--line)] bg-[var(--bg)] py-3 pl-10 pr-24 text-sm text-[var(--text)] placeholder-[var(--text-muted)] focus:border-[var(--accent)] focus:outline-none focus:ring-1 focus:ring-[var(--accent)]"
          />
          <Search className="absolute left-3.5 top-3.5 h-4 w-4 text-[var(--text-muted)]" />

          {/* Quick Clear and Ctrl K indicator */}
          <div className="absolute right-3 top-2.5 flex items-center gap-1.5">
            {searchQuery && (
              <button
                type="button"
                onClick={() => onSearchQueryChange("")}
                className="p-1 text-[var(--text-muted)] hover:text-[var(--text)]"
                title={t("search.clearTooltip")}
              >
                <X className="h-3.5 w-3.5" />
              </button>
            )}
            <kbd className="hidden rounded border border-[var(--line-strong)] bg-[var(--panel)] px-2 py-0.5 font-mono text-[10px] text-[var(--text-muted)] sm:inline-block">
              Ctrl K
            </kbd>
          </div>
        </div>

        {/* Search Submit Button */}
        <button
          type="submit"
          disabled={isSearching}
          className="inline-flex flex-shrink-0 items-center gap-2 rounded-xl bg-[var(--accent)] px-5 py-3 text-xs font-semibold text-[var(--accent-text)] shadow-sm transition-opacity hover:opacity-95 disabled:opacity-50 cursor-pointer"
        >
          {isSearching ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              <span>{t("search.searching")}</span>
            </>
          ) : (
            <>
              <Sparkles className="h-4 w-4 fill-current/20" />
              <span>{t("search.button")}</span>
            </>
          )}
        </button>
      </form>

      {/* 2. Controls Row: Scope, Mode, Density, Time Filter */}
      <div className="flex flex-wrap items-center justify-between gap-3 text-xs">
        {/* Left Side: Scope, Mode, Frame Density */}
        <div className="flex flex-wrap items-center gap-3">
          {/* Scope Selector */}
          <div className="flex items-center gap-1.5">
            <span className="text-[11px] text-[var(--text-muted)]">{t("search.scope")}</span>
            <div className="flex items-center rounded-lg border border-[var(--line)] bg-[var(--bg)] p-0.5">
              {(
                [
                  { id: "video", label: t("search.scope.video") },
                  { id: "folder", label: t("search.scope.folder") },
                  { id: "all", label: t("search.scope.all") },
                ] as const
              ).map((item) => (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => setScope(item.id)}
                  className={`rounded-md px-2.5 py-1 text-xs transition-colors ${
                    scope === item.id
                      ? "bg-[var(--panel)] font-semibold text-[var(--text)] shadow-sm"
                      : "text-[var(--text-muted)] hover:text-[var(--text)]"
                  }`}
                >
                  {item.label}
                </button>
              ))}
            </div>
          </div>

          {/* Mode Selector */}
          <div className="flex items-center gap-1.5">
            <span className="text-[11px] text-[var(--text-muted)]">{t("search.mode")}</span>
            <div className="flex items-center rounded-lg border border-[var(--line)] bg-[var(--bg)] p-0.5">
              <button
                type="button"
                onClick={() => onSearchProfileChange("fast")}
                className={`rounded-md px-2.5 py-1 text-xs transition-colors ${
                  searchProfile === "fast"
                    ? "bg-[var(--panel)] font-semibold text-[var(--text)] shadow-sm"
                    : "text-[var(--text-muted)] hover:text-[var(--text)]"
                }`}
              >
                Fast
              </button>
              <button
                type="button"
                onClick={() => onSearchProfileChange("accurate")}
                className={`rounded-md px-2.5 py-1 text-xs transition-colors ${
                  searchProfile === "accurate"
                    ? "bg-[var(--panel)] font-semibold text-[var(--text)] shadow-sm"
                    : "text-[var(--text-muted)] hover:text-[var(--text)]"
                }`}
              >
                Accurate
              </button>
            </div>
          </div>

          {/* Frame Density Selector & Explicit Rebuild Button */}
          <div className="flex items-center gap-1.5 flex-wrap">
            <span className="text-[11px] text-[var(--text-muted)]">{t("search.density")}</span>
            <select
              value={selectedDensity}
              onChange={(e) => setSelectedDensity(Number(e.target.value))}
              disabled={isRunning || isRebuilding}
              className="rounded-lg border border-[var(--line)] bg-[var(--panel)] px-2.5 py-1 text-xs text-[var(--text)] focus:border-[var(--accent)] focus:outline-none disabled:opacity-50"
            >
              <option value={4}>{t("search.density.4")}</option>
              <option value={8}>{t("search.density.8")}</option>
              <option value={12}>{t("search.density.12")}</option>
              <option value={16}>{t("search.density.16")}</option>
            </select>

            <button
              type="button"
              onClick={() => handleRebuild(selectedDensity)}
              disabled={isRunning || isRebuilding || !videoId}
              className="inline-flex items-center gap-1.5 px-3 py-1 rounded-lg bg-[var(--accent)] text-white hover:opacity-95 text-xs font-semibold shadow-sm transition-all disabled:opacity-50 cursor-pointer"
              title={t("search.rebuildTooltip")}
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isRebuilding || isRunning ? "animate-spin" : ""}`} />
              <span>{isRebuilding || isRunning ? t("search.rebuilding") : t("search.rebuildCaptions")}</span>
            </button>

            {isRunning && (
              <button
                type="button"
                onClick={handleCancelCaption}
                disabled={isCancelling}
                className="inline-flex items-center gap-1 rounded-lg border border-rose-600/40 bg-rose-500/10 px-2.5 py-1 text-xs font-semibold text-rose-500 hover:bg-rose-500/20 transition-all disabled:opacity-50 cursor-pointer"
                title={t("search.cancelTooltip")}
              >
                <span>{isCancelling ? t("search.cancelling") : t("search.cancel")}</span>
              </button>
            )}
          </div>
        </div>

        {/* Right Side: Camera Time Filter */}
        <div className="inline-flex items-center gap-1.5 rounded-lg border border-[var(--line)] bg-[var(--bg)] px-2.5 py-1 text-xs text-[var(--text-muted)]">
          <Clock className="h-3.5 w-3.5 text-[var(--text-muted)]" />
          <span>{t("search.cameraTime")}</span>
        </div>
      </div>

      {/* 3. Caption Warning / Status Pill */}
      {isErrorOrUnavailable && (
        <div className="flex items-center gap-2 rounded-lg border border-amber-600/30 bg-amber-500/10 px-3 py-1.5 text-xs text-amber-700 dark:text-amber-300">
          <AlertTriangle className="h-3.5 w-3.5 flex-shrink-0" />
          <span>
            {captionStatus?.status === "cancelled"
              ? t("search.captionCancelled")
              : t("search.captionWarning")}
          </span>
        </div>
      )}

      {isRunning && (
        <div className="flex items-center gap-2 rounded-lg border border-[var(--accent)]/30 bg-[var(--accent-subtle)] px-3 py-1.5 text-xs text-[var(--accent)]">
          <Loader2 className="h-3.5 w-3.5 animate-spin flex-shrink-0" />
          <span>
            {language === "th" ? "กำลังสร้าง Caption" : "Generating Captions"}: {language === "th" ? "ฉากที่" : "Scene"} {captionStatus?.completed_count || 0}/{captionStatus?.expected_count || 1} ({captionStatus?.progress_percent || 0}%)
          </span>
        </div>
      )}

      {/* 4. Example Query Categories & Chips */}
      <div className="space-y-2 border-t border-[var(--line)] pt-2.5">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-[11px] font-medium text-[var(--text-muted)]">{t("search.examples")}</span>
          <div className="flex items-center rounded-lg border border-[var(--line)] bg-[var(--bg)] p-0.5">
            {currentExampleTabs.map((tab) => (
              <button
                key={tab.id}
                type="button"
                onClick={() => setActiveTab(tab.id)}
                className={`rounded-md px-2.5 py-0.5 text-xs transition-colors ${
                  activeTab === tab.id
                    ? "bg-[var(--panel)] font-semibold text-[var(--text)] shadow-sm"
                    : "text-[var(--text-muted)] hover:text-[var(--text)]"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        {/* Chips for Active Category */}
        <div className="flex flex-wrap items-center gap-1.5">
          {currentExampleTabs.find((tItem) => tItem.id === activeTab)?.queries.map((q) => (
            <button
              key={q}
              type="button"
              onClick={() => {
                onSearchQueryChange(q);
                onSearch(q);
              }}
              className="rounded-lg border border-[var(--line)] bg-[var(--card-bg)] px-2.5 py-1 text-xs text-[var(--text)] transition-colors hover:border-[var(--line-strong)] hover:bg-[var(--card-hover)] cursor-pointer"
            >
              {q}
            </button>
          ))}
        </div>
      </div>

      {/* 5. Recent Search History Chips */}
      {recentQueries.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5 pt-1 text-xs">
          <span className="text-[11px] text-[var(--text-muted)]">{t("search.recent")}</span>
          {recentQueries.map((item) => (
            <button
              key={item}
              type="button"
              onClick={() => {
                onSearchQueryChange(item);
                onSearch(item);
              }}
              className="rounded-md border border-[var(--line)] bg-[var(--tag-bg)] px-2 py-0.5 text-[11px] font-mono text-[var(--text-muted)] transition-colors hover:border-[var(--line-strong)] hover:text-[var(--text)] cursor-pointer"
            >
              {item}
            </button>
          ))}
        </div>
      )}
    </div>
  );
};
