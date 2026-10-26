"use client";

import React, { useState, useEffect, useRef } from "react";
import { X, AlertTriangle, Loader2, Trash2 } from "lucide-react";
import { VideoMetadata, MomentItem, SearchResponse, VideoKeyframeItem } from "@/lib/types";
import { apiClient } from "@/lib/api";
import { AppHeader } from "@/components/layout/AppHeader";
import { VideoSidebar } from "@/components/sidebar/VideoSidebar";
import { SearchSection } from "@/components/search/SearchSection";
import { VideoPlayer } from "@/components/player/VideoPlayer";
import { ResultsPanel } from "@/components/results/ResultsPanel";
import { Dropzone } from "@/components/upload/Dropzone";
import { DevPanel } from "@/components/dev/DevPanel";
import { ShortcutModal } from "@/components/ui/ShortcutModal";
import { momentIdentity } from "@/lib/ui";
import { useLanguage } from "@/lib/language";

export default function DashboardPage() {
  const { language, t } = useLanguage();
  const [videos, setVideos] = useState<VideoMetadata[]>([]);
  const [selectedVideo, setSelectedVideo] = useState<VideoMetadata | null>(null);
  const [keyframeRecords, setKeyframeRecords] = useState<VideoKeyframeItem[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchProfile, setSearchProfile] = useState<"fast" | "accurate">("fast");
  const [isSearching, setIsSearching] = useState(false);
  const [searchResult, setSearchResult] = useState<SearchResponse | null>(null);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [seekTime, setSeekTime] = useState<number | null>(null);
  const [highlightInterval, setHighlightInterval] = useState<[number, number] | null>(null);
  const [activeMoment, setActiveMoment] = useState<MomentItem | null>(null);
  const [activeMomentKey, setActiveMomentKey] = useState<string | null>(null);

  // Modals & Panels
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [showDevPanel, setShowDevPanel] = useState(false);
  const [showShortcutModal, setShowShortcutModal] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [videoToDelete, setVideoToDelete] = useState<VideoMetadata | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  // Player enhancements
  const [isCinemaMode, setIsCinemaMode] = useState(false);
  const [isBoundaryAdjusted, setIsBoundaryAdjusted] = useState(false);
  const [isContextExpanded, setIsContextExpanded] = useState(false);

  const searchAbortRef = useRef<AbortController | null>(null);
  const searchRequestIdRef = useRef(0);
  const searchInputRef = useRef<HTMLInputElement | null>(null);

  // Load videos on mount
  const fetchVideos = async () => {
    try {
      const data = await apiClient.getVideos();
      setVideos(data);
      if (data.length > 0 && !selectedVideo) {
        setSelectedVideo(data[0]);
      }
    } catch (e) {
      console.error("Failed to load videos:", e);
    }
  };

  useEffect(() => {
    fetchVideos();
  }, []);

  // Fetch keyframe records when selected video changes
  useEffect(() => {
    if (selectedVideo?.id) {
      apiClient
        .getVideoKeyframes(selectedVideo.id)
        .then((kfs) => setKeyframeRecords(kfs))
        .catch(() => setKeyframeRecords([]));
    } else {
      setKeyframeRecords([]);
    }
  }, [selectedVideo]);

  // Global Keyboard Shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const activeEl = document.activeElement;
      const isInput =
        activeEl?.tagName === "INPUT" ||
        activeEl?.tagName === "TEXTAREA" ||
        (activeEl as HTMLElement)?.isContentEditable;

      if ((e.ctrlKey && e.key === "k") || (e.key === "/" && !isInput)) {
        e.preventDefault();
        searchInputRef.current?.focus();
      } else if (!isInput) {
        if (e.key === "t" || e.key === "T") {
          e.preventDefault();
          setIsCinemaMode((prev) => !prev);
        } else if (e.key === "?") {
          e.preventDefault();
          setShowShortcutModal((prev) => !prev);
        } else if (e.key === "[" && activeMoment) {
          e.preventDefault();
          setSeekTime(activeMoment.t_start);
        } else if (e.key === "]" && activeMoment) {
          e.preventDefault();
          setSeekTime(activeMoment.t_end);
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [activeMoment]);

  const handleSearch = async (queryToSearch?: string) => {
    const q = (queryToSearch !== undefined ? queryToSearch : searchQuery).trim();
    if (!q) return;

    if (queryToSearch !== undefined) {
      setSearchQuery(queryToSearch);
    }

    if (!selectedVideo?.id) {
      setSearchError("กรุณาเลือกวิดีโอก่อนค้นหา");
      return;
    }

    searchAbortRef.current?.abort();
    const requestId = ++searchRequestIdRef.current;
    const controller = new AbortController();
    searchAbortRef.current = controller;
    setIsSearching(true);
    setSearchError(null);
    setSearchResult(null);
    setActiveMoment(null);
    setActiveMomentKey(null);
    setHighlightInterval(null);
    setSeekTime(null);
    setIsBoundaryAdjusted(false);
    setIsContextExpanded(false);

    try {
      const res = await apiClient.searchMoments(q, selectedVideo.id, 5, searchProfile, controller.signal);
      if (requestId !== searchRequestIdRef.current) return;
      setSearchResult(res);

      if (res.moments && res.moments.length > 0) {
        const first = res.moments[0];
        setActiveMoment(first);
        setActiveMomentKey(momentIdentity(first));
        setSeekTime(first.t_start);
        setHighlightInterval([first.t_start, first.t_end]);
      }
    } catch (err: any) {
      if (err?.code === "ERR_CANCELED" || err?.name === "CanceledError" || controller.signal.aborted) return;
      console.error("Search failed:", err);
      const detail = err?.response?.data?.detail;
      const message = typeof detail === "string" ? detail : detail?.message;
      setSearchError(message || "ค้นหาไม่สำเร็จ กรุณาลองใหม่อีกครั้ง");
    } finally {
      if (requestId === searchRequestIdRef.current) setIsSearching(false);
    }
  };

  const handleSelectMoment = (m: MomentItem) => {
    setActiveMoment(m);
    setActiveMomentKey(momentIdentity(m));
    setSeekTime(m.t_start);
    setHighlightInterval([m.t_start, m.t_end]);
    setIsContextExpanded(false);
    setIsBoundaryAdjusted(false);
  };

  const handleExpandContext = (m: MomentItem) => {
    setActiveMoment(m);
    setActiveMomentKey(momentIdentity(m));
    const start = m.context_t_start ?? Math.max(0, m.t_start - 2);
    const end = m.context_t_end ?? Math.min(selectedVideo?.duration_sec || m.t_end + 2, m.t_end + 2);
    setSeekTime(start);
    setHighlightInterval([start, end]);
    setIsContextExpanded(true);
    setIsBoundaryAdjusted(false);
  };

  const handleResetBoundary = () => {
    if (!activeMoment || !searchResult) return;
    const original = searchResult.moments.find((m) => momentIdentity(m) === activeMomentKey);
    if (!original) return;
    setActiveMoment(original);
    setHighlightInterval([original.t_start, original.t_end]);
    setIsContextExpanded(false);
    setIsBoundaryAdjusted(false);
  };

  const handleBoundaryChange = (newStart: number, newEnd: number) => {
    setHighlightInterval([newStart, newEnd]);
    setIsBoundaryAdjusted(true);
  };

  const handleSeekFromQA = (time: number) => {
    setActiveMoment(null);
    setActiveMomentKey(null);
    setIsContextExpanded(false);
    setIsBoundaryAdjusted(false);
    setSeekTime(time);
    setHighlightInterval([time, Math.min(selectedVideo?.duration_sec || time + 4, time + 4.0)]);
  };

  const handleRequestDelete = (video: VideoMetadata) => {
    setVideoToDelete(video);
    setShowDeleteConfirm(true);
  };

  const handleDeleteVideo = async () => {
    const target = videoToDelete || selectedVideo;
    if (!target) return;

    setIsDeleting(true);
    setDeleteError(null);
    try {
      await apiClient.deleteVideo(target.id);
      setShowDeleteConfirm(false);
      setVideoToDelete(null);

      if (selectedVideo?.id === target.id) {
        setSearchResult(null);
        setHighlightInterval(null);
        setActiveMoment(null);
        setActiveMomentKey(null);
        setSeekTime(null);
      }

      const remainingVideos = await apiClient.getVideos();
      setVideos(remainingVideos);
      if (remainingVideos.length > 0) {
        if (selectedVideo?.id === target.id) {
          setSelectedVideo(remainingVideos[0]);
        }
      } else {
        setSelectedVideo(null);
      }
    } catch (err: any) {
      console.error("Delete video failed:", err);
      setDeleteError(err?.response?.data?.detail || "ลบวิดีโอไม่สำเร็จ กรุณาลองใหม่อีกครั้ง");
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <div className="min-h-screen bg-[var(--bg)] flex flex-col font-sans transition-colors duration-200">
      {/* 1. App Header */}
      <AppHeader
        isCinemaMode={isCinemaMode}
        onToggleCinemaMode={() => setIsCinemaMode((prev) => !prev)}
        onOpenHotkeys={() => setShowShortcutModal(true)}
        onOpenSettings={() => setShowDevPanel(true)}
      />

      {/* 2. Main 3-Column Responsive Layout */}
      <main className="flex-1 w-full max-w-[1680px] mx-auto p-3 sm:p-4 md:p-5 flex flex-col lg:flex-row gap-4 items-start">
        {/* Left Column: Video Library Sidebar (hidden in cinema mode on smaller screens or toggleable) */}
        {!isCinemaMode && (
          <VideoSidebar
            videos={videos}
            selectedVideo={selectedVideo}
            onSelectVideo={(vid) => {
              searchAbortRef.current?.abort();
              searchRequestIdRef.current += 1;
              setSelectedVideo(vid);
              setSearchResult(null);
              setSearchError(null);
              setHighlightInterval(null);
              setActiveMoment(null);
              setActiveMomentKey(null);
              setIsBoundaryAdjusted(false);
              setIsContextExpanded(false);
            }}
            onOpenUpload={() => setShowUploadModal(true)}
            onRequestDelete={handleRequestDelete}
          />
        )}

        {/* Center Column: Search Section + Video Studio Player */}
        <div className="flex-1 flex flex-col gap-4 min-w-0 w-full">
          {/* Natural Language Search Suite */}
          <SearchSection
            searchQuery={searchQuery}
            onSearchQueryChange={setSearchQuery}
            onSearch={handleSearch}
            isSearching={isSearching}
            searchProfile={searchProfile}
            onSearchProfileChange={setSearchProfile}
            videoId={selectedVideo?.id}
            searchInputRef={searchInputRef}
          />

          {searchError && (
            <div className="text-xs text-amber-600 dark:text-amber-300 bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-700/40 rounded-xl px-4 py-3">
              {searchError}
            </div>
          )}

          {/* Video Studio Player */}
          {selectedVideo ? (
            <VideoPlayer
              streamUrl={apiClient.getVideoStreamUrl(selectedVideo.id)}
              duration={selectedVideo.duration_sec}
              seekTime={seekTime}
              highlightInterval={highlightInterval}
              heatmapData={searchResult?.timeline_heatmap || []}
              moments={searchResult?.moments || []}
              keyframeRecords={keyframeRecords}
              fps={selectedVideo.fps || 25}
              isCinemaMode={isCinemaMode}
              onToggleCinemaMode={() => setIsCinemaMode(!isCinemaMode)}
              onBoundaryChange={handleBoundaryChange}
              boundaryAdjusted={isBoundaryAdjusted}
              contextExpanded={isContextExpanded}
              onResetBoundary={handleResetBoundary}
              groundingEvidence={activeMoment?.grounding_evidence || []}
              videoMetadata={selectedVideo}
              activeMoment={activeMoment}
              searchQuery={searchQuery}
            />
          ) : (
            <div className="aspect-video app-panel rounded-2xl flex items-center justify-center text-[var(--muted)] border border-[var(--line)]">
              <div className="text-center space-y-2 p-6">
                <p className="text-sm font-semibold text-[var(--text)]">ยังไม่ได้เลือกวิดีโอ</p>
                <p className="text-xs">เลือกวิดีโอจากคลังทางซ้าย หรือคลิก "+ นำเข้าวิดีโอ" เพื่อเริ่มต้น</p>
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Results Panel (Moments Cards & Video-RAG AI) */}
        {!isCinemaMode && (
          <ResultsPanel
            moments={searchResult?.moments || []}
            selectedVideo={selectedVideo}
            searchQuery={searchQuery}
            activeMoment={activeMoment}
            onSelectMoment={handleSelectMoment}
            onExpandContext={handleExpandContext}
            onSeekToTimestamp={handleSeekFromQA}
            onSearchWithQuery={(q) => {
              setSearchQuery(q);
              handleSearch(q);
            }}
          />
        )}
      </main>

      {/* Video Upload Modal */}
      {showUploadModal && (
        <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="app-panel border border-[var(--line)] rounded-2xl max-w-2xl w-full p-6 shadow-2xl space-y-4 animate-in fade-in zoom-in-95 duration-200">
            <div className="flex items-center justify-between pb-2 border-b border-[var(--line)]">
              <h3 className="text-base font-bold text-[var(--text)]">
                นำเข้าวิดีโอเข้าสู่คลัง
              </h3>
              <button
                type="button"
                onClick={() => setShowUploadModal(false)}
                className="p-1.5 rounded-lg text-[var(--muted)] hover:text-[var(--text)] hover:bg-[var(--line)]/50 transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <Dropzone
              onUploadSuccess={(vid) => {
                fetchVideos();
                setShowUploadModal(false);
              }}
            />
          </div>
        </div>
      )}

      {/* Delete Confirmation Modal */}
      {showDeleteConfirm && (videoToDelete || selectedVideo) && (
        <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="app-panel border border-red-500/30 rounded-2xl max-w-md w-full p-6 shadow-2xl space-y-4 animate-in fade-in zoom-in-95 duration-200">
            <div className="flex items-center gap-3 text-red-500">
              <div className="p-2.5 rounded-xl bg-red-500/10 border border-red-500/20">
                <AlertTriangle className="w-6 h-6 text-red-500" />
              </div>
              <div>
                <h3 className="text-base font-bold text-[var(--text)]">
                  {t("modal.deleteTitle")}
                </h3>
                <p className="text-xs text-[var(--muted)]">{t("modal.deleteDesc")}</p>
              </div>
            </div>

            <p className="text-sm text-[var(--text)]">
              {t("modal.deleteConfirm")}{" "}
              <span className="font-semibold text-[var(--accent)]">
                "{(videoToDelete || selectedVideo)?.filename}"
              </span>
              ?
            </p>

            <p className="text-xs text-red-700 dark:text-red-300 bg-red-50 dark:bg-red-950/40 p-3 rounded-xl border border-red-200 dark:border-red-900/50 leading-relaxed">
              {t("modal.deleteWarning")}
            </p>

            {deleteError && (
              <p className="text-xs text-red-500 font-medium">{deleteError}</p>
            )}

            <div className="flex items-center justify-end gap-3 pt-2">
              <button
                type="button"
                disabled={isDeleting}
                onClick={() => {
                  setShowDeleteConfirm(false);
                  setVideoToDelete(null);
                  setDeleteError(null);
                }}
                className="px-4 py-2 rounded-xl bg-[var(--panel)] hover:bg-[var(--line)]/50 text-[var(--text)] text-xs font-semibold border border-[var(--line)] transition-colors cursor-pointer"
              >
                {t("modal.cancel")}
              </button>
              <button
                type="button"
                disabled={isDeleting}
                onClick={handleDeleteVideo}
                className="px-4 py-2 rounded-xl bg-red-600 hover:bg-red-500 text-white text-xs font-semibold shadow-sm transition-colors flex items-center gap-2 disabled:opacity-50 cursor-pointer"
              >
                {isDeleting ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" /> {t("modal.deleting")}
                  </>
                ) : (
                  <>
                    <Trash2 className="w-4 h-4" /> {t("modal.delete")}
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Developer & System Telemetry Panel Modal */}
      <DevPanel isOpen={showDevPanel} onClose={() => setShowDevPanel(false)} />

      {/* Keyboard Shortcuts Modal */}
      <ShortcutModal isOpen={showShortcutModal} onClose={() => setShowShortcutModal(false)} />
    </div>
  );
}
