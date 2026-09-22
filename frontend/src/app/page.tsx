"use client";

import React, { useState, useEffect, useRef } from "react";
import {
  Search,
  Sparkles,
  Film,
  Loader2,
  MessageSquare,
  ListFilter,
  Trash2,
  AlertTriangle,
  Terminal,
  Cpu,
  X,
  UploadCloud,
  Command,
  HelpCircle,
  Tv,
} from "lucide-react";
import { VideoMetadata, MomentItem, SearchResponse, VideoKeyframeItem } from "@/lib/types";
import { apiClient } from "@/lib/api";
import { VideoLibraryReel } from "@/components/library/VideoLibraryReel";
import { VideoPlayer } from "@/components/player/VideoPlayer";
import { QueryAssistant } from "@/components/search/QueryAssistant";
import { MomentCards } from "@/components/search/MomentCards";
import { VideoQAPanel } from "@/components/rag/VideoQAPanel";
import { Dropzone } from "@/components/upload/Dropzone";
import { DevPanel } from "@/components/dev/DevPanel";
import { ShortcutModal } from "@/components/ui/ShortcutModal";
import { SearchProfileControl } from "@/components/search/SearchProfileControl";
import { SearchStatus } from "@/components/search/SearchStatus";
import { CaptionStatusBadge } from "@/components/search/CaptionStatusBadge";
import { SearchWarningBanner } from "@/components/search/SearchWarningBanner";
import { SearchResultSummary } from "@/components/search/SearchResultSummary";
import { uniqueWarnings, momentIdentity } from "@/lib/ui";

export default function DashboardPage() {
  const [videos, setVideos] = useState<VideoMetadata[]>([]);
  const [selectedVideo, setSelectedVideo] = useState<VideoMetadata | null>(null);
  const [keyframeRecords, setKeyframeRecords] = useState<VideoKeyframeItem[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchProfile, setSearchProfile] = useState<"fast" | "accurate">("fast");
  const [isSearching, setIsSearching] = useState(false);
  const [searchResult, setSearchResult] = useState<SearchResponse | null>(null);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [needsReindex, setNeedsReindex] = useState(false);
  const [seekTime, setSeekTime] = useState<number | null>(null);
  const [highlightInterval, setHighlightInterval] = useState<[number, number] | null>(null);
  const [activeMoment, setActiveMoment] = useState<MomentItem | null>(null);
  const [activeMomentKey, setActiveMomentKey] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"moments" | "rag">("moments");
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [videoToDelete, setVideoToDelete] = useState<VideoMetadata | null>(null);
  const [showDevPanel, setShowDevPanel] = useState(false);
  const [showShortcutModal, setShowShortcutModal] = useState(false);
  const [showUploader, setShowUploader] = useState(false);
  const [isCinemaMode, setIsCinemaMode] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
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

  // Fetch keyframes cache whenever selected video changes
  useEffect(() => {
    if (selectedVideo?.id) {
      apiClient
        .getVideoKeyframes(selectedVideo.id)
        .then((kfs) => setKeyframeRecords(kfs))
        .catch((e) => {
          console.debug("Failed to fetch keyframes for scrubber:", e);
          setKeyframeRecords([]);
        });
    } else {
      setKeyframeRecords([]);
    }
  }, [selectedVideo]);

  // Global Pro Hotkeys Engine
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

  const saveRecentQuery = (q: string) => {
    try {
      const saved = JSON.parse(localStorage.getItem("videomoment_recent_queries") || "[]");
      const updated = [q, ...saved.filter((x: string) => x !== q)].slice(0, 8);
      localStorage.setItem("videomoment_recent_queries", JSON.stringify(updated));
    } catch (e) {}
  };

  const handleSearch = async (queryToSearch?: string) => {
    const q = (queryToSearch || searchQuery).trim();
    if (!q) return;

    if (queryToSearch) {
      setSearchQuery(queryToSearch);
    }

    saveRecentQuery(q);
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
    setNeedsReindex(false);
    setSearchResult(null);
    setActiveMoment(null);
    setActiveMomentKey(null);
    setHighlightInterval(null);
    setSeekTime(null);
    setIsBoundaryAdjusted(false);
    setIsContextExpanded(false);
    setActiveTab("moments");

    try {
      const res = await apiClient.searchMoments(q, selectedVideo.id, 5, searchProfile, controller.signal);
      if (requestId !== searchRequestIdRef.current) return;
      setSearchResult(res);
      // Accurate is VLM-only in the production flow; open the top ranked moment.
      if (searchProfile === "accurate") {
        const first = res.moments[0];
        if (first) {
          setActiveMoment(first);
          setActiveMomentKey(momentIdentity(first));
          setSeekTime(first.t_start);
          setHighlightInterval([first.t_start, first.t_end]);
        }
      }
      if (res.warnings?.includes("reindex_required")) {
        setNeedsReindex(true);
      }
    } catch (err: any) {
      if (err?.code === "ERR_CANCELED" || err?.name === "CanceledError" || controller.signal.aborted) return;
      console.error("Search failed:", err);
      const detail = err?.response?.data?.detail;
      const message = typeof detail === "string" ? detail : detail?.message;
      if (detail?.code === "reindex_required") {
        setNeedsReindex(true);
        setSearchError(null);
        return;
      }
      setSearchError(message || "ค้นหาไม่สำเร็จ กรุณาตรวจสอบวิดีโอหรือทำดัชนีใหม่");
    } finally {
      if (requestId === searchRequestIdRef.current) setIsSearching(false);
    }
  };

  const handleSelectMoment = (m: MomentItem, autoLoop: boolean = false) => {
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
    setSeekTime(m.context_t_start ?? m.t_start);
    setHighlightInterval([m.context_t_start ?? m.t_start, m.context_t_end ?? m.t_end]);
    setIsContextExpanded(true);
    setIsBoundaryAdjusted(false);
  };

  const handleResetBoundary = () => {
    if (!activeMoment || !searchResult) return;
    const original = searchResult.moments.find((moment) => momentIdentity(moment) === activeMomentKey);
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

      // Clear search and playback states if active video was deleted
      if (selectedVideo?.id === target.id) {
        setSearchResult(null);
        setHighlightInterval(null);
        setActiveMoment(null);
        setActiveMomentKey(null);
        setSeekTime(null);
      }

      // Refresh list and select next video
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
      setDeleteError(err?.response?.data?.detail || "Failed to delete video. Please try again.");
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <div className="space-y-5">
      {/* 1. Video Library Reel (Visual Horizontal Shelf) */}
      <VideoLibraryReel
        videos={videos}
        selectedVideo={selectedVideo}
        onSelectVideo={(vid) => {
          searchAbortRef.current?.abort();
          searchRequestIdRef.current += 1;
          setSelectedVideo(vid);
          setSearchResult(null);
          setSearchError(null);
          setNeedsReindex(false);
          setHighlightInterval(null);
          setActiveMoment(null);
          setActiveMomentKey(null);
          setIsBoundaryAdjusted(false);
          setIsContextExpanded(false);
        }}
        onRequestDelete={handleRequestDelete}
        onToggleUpload={() => setShowUploader(!showUploader)}
        isCinemaMode={isCinemaMode}
        onToggleCinemaMode={() => setIsCinemaMode(!isCinemaMode)}
      />

      {/* 2. Natural Language Search Suite with Pro Shortcut Pill */}
      <div className="glass-panel rounded-2xl p-4 shadow-xl space-y-3 border border-surfaceBorder/80">
        <div className="flex items-center justify-between gap-3 border-b border-surfaceBorder/60 pb-2">
          <div>
            <p className="text-xs font-semibold text-gray-200">Pure-Visual Moment Search</p>
            <div className="flex items-center gap-2">
              <p className="text-[10px] font-mono text-gray-500">SigLIP2 retrieval + precomputed CapRL Q6 visual captions</p>
              <CaptionStatusBadge videoId={selectedVideo?.id} />
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button type="button" onClick={() => setShowShortcutModal(true)} aria-label="Open keyboard shortcuts" className="min-h-9 rounded-xl border border-surfaceBorder bg-surface px-3 text-xs font-mono text-gray-300 hover:border-cyan-500/40 hover:text-cyan-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400" title="Pro Keyboard Shortcuts (?)">
              <Command className="inline h-4 w-4 text-cyan-400" /> <span className="hidden sm:inline">Hotkeys</span>
            </button>
            <button type="button" onClick={() => setShowDevPanel(true)} aria-label="Open developer panel" className="min-h-9 rounded-xl border border-surfaceBorder bg-surface px-3 text-xs font-mono font-semibold text-gray-300 hover:border-cyan-500/40 hover:text-cyan-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400" title="Open Developer & System Telemetry Panel">
              <Terminal className="inline h-4 w-4 text-cyan-400" /> <span className="hidden sm:inline">Dev Panel</span>
            </button>
          </div>
        </div>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSearch();
          }}
          className="flex flex-col gap-3 lg:flex-row lg:items-center"
        >
          {/* Query Input */}
          <div className="relative flex-1">
            <input
              ref={searchInputRef}
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="ค้นหาการกระทำทางกายภาพ (เช่น 'คนปั่นจักรยาน', 'รถยนต์เลี้ยวซ้าย', 'student raising hand')..."
              className="w-full bg-surface/90 border border-surfaceBorder rounded-xl pl-11 pr-24 py-3 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 transition-all shadow-inner"
            />
            <Search className="w-5 h-5 absolute left-3.5 top-3.5 text-cyan-400" />

            {/* Quick Clear & Keyboard Shortcut Badge */}
            <div className="absolute right-3 top-2.5 flex items-center gap-1.5">
              {searchQuery && (
                <button
                  type="button"
                  onClick={() => setSearchQuery("")}
                  className="p-1 text-gray-500 hover:text-gray-300"
                >
                  <X className="w-4 h-4" />
                </button>
              )}
              <kbd className="hidden sm:inline-block px-2 py-0.5 rounded bg-surfaceBorder/60 border border-white/10 text-[10px] font-mono text-gray-400">
                Ctrl K
              </kbd>
            </div>
          </div>

          {/* Search Button */}
          <button
            type="submit"
            disabled={isSearching}
            className="px-6 py-3 rounded-xl bg-gradient-to-r from-cyan-500 via-indigo-600 to-cyan-600 text-white font-bold text-xs tracking-wide shadow-lg shadow-cyan-500/20 hover:opacity-95 transition-all flex items-center gap-2 disabled:opacity-50 flex-shrink-0"
          >
            {isSearching ? (
              <><Loader2 className="w-4 h-4 animate-spin" /> กำลังค้นหา...</>
            ) : (
              <>
                <Sparkles className="w-4 h-4 text-cyan-200" /> สกัดช่วงเวลา (VMR)
              </>
            )}
          </button>

          <SearchProfileControl value={searchProfile} disabled={isSearching} onChange={setSearchProfile} />

        </form>

        {isSearching && <SearchStatus profile={searchProfile} />}

        {/* Categorized Action Suggestions & Recent Searches */}
        <QueryAssistant
          onSelectQuery={(q) => {
            setSearchQuery(q);
            handleSearch(q);
          }}
          currentQuery={searchQuery}
        />
        {searchError && (
          <div className="text-xs text-amber-300 bg-amber-950/30 border border-amber-700/40 rounded-lg px-3 py-2 flex items-center justify-between gap-3">
            <span>{searchError}</span>
          </div>
        )}
      </div>

      {/* 3. Collapsible Video Upload Drawer */}
      {showUploader && (
        <div className="animate-in fade-in slide-in-from-top-4 duration-200">
          <Dropzone
            onUploadSuccess={(vid) => {
              fetchVideos();
              setShowUploader(false);
            }}
          />
        </div>
      )}

      {/* 4. Studio Layout Grid: Player & Multi-modal Inspector */}
      <div
        className={`grid grid-cols-1 ${
          isCinemaMode ? "lg:grid-cols-12 gap-8" : "lg:grid-cols-12 gap-6"
        } items-start studio-cinema-transition`}
      >
        {/* Left Column: Video Studio Player & Scrubber */}
        <div
          className={`${
            isCinemaMode ? "lg:col-span-12" : "lg:col-span-7"
          } space-y-5 transition-all duration-300`}
        >
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
            />
          ) : (
            <div className="aspect-video glass-panel rounded-2xl flex items-center justify-center text-gray-500 border border-surfaceBorder">
              <div className="text-center space-y-2">
                <Film className="w-12 h-12 mx-auto text-gray-600" />
                <p>เลือกหรืออัปโหลดวิดีโอเพื่อเริ่มต้นการสืบค้นช่วงเวลา</p>
              </div>
            </div>
          )}

          {/* Quick Ingest Button if uploader is closed */}
          {!showUploader && (
            <div
              onClick={() => setShowUploader(true)}
              className="glass-panel p-3 rounded-xl border border-dashed border-surfaceBorder hover:border-cyan-500/50 cursor-pointer text-center text-xs text-gray-400 hover:text-cyan-300 transition-colors flex items-center justify-center gap-2"
            >
              <UploadCloud className="w-4 h-4 text-cyan-400" />
              <span>ต้องการนำเข้าวิดีโอใหม่? คลิกเพื่อเปิดกล่องอัปโหลดวิดีโอ</span>
            </div>
          )}
        </div>

        {/* Right Column: Tab Navigation (Retrieved Moments vs Video-RAG QA) */}
        <div
          className={`${
            isCinemaMode ? "lg:col-span-12" : "lg:col-span-5"
          } space-y-3.5 transition-all duration-300`}
        >
          {/* Mode Switcher Tabs */}
          <div className="flex items-center gap-2 p-1 bg-surface border border-surfaceBorder rounded-xl">
            <button
              type="button"
              onClick={() => setActiveTab("moments")}
              className={`flex-1 py-2 px-3 rounded-lg text-xs font-semibold flex items-center justify-center gap-1.5 transition-all ${
                activeTab === "moments"
                  ? "bg-gradient-to-r from-indigo-600 to-cyan-600 text-white shadow-md"
                  : "text-gray-400 hover:text-gray-200"
              }`}
            >
              <ListFilter className="w-3.5 h-3.5" /> ช่วงเวลาที่พบ ({searchResult?.moments?.length || 0})
            </button>
            <button
              type="button"
              onClick={() => setActiveTab("rag")}
              className={`flex-1 py-2 px-3 rounded-lg text-xs font-semibold flex items-center justify-center gap-1.5 transition-all ${
                activeTab === "rag"
                  ? "bg-gradient-to-r from-indigo-600 to-cyan-600 text-white shadow-md"
                  : "text-gray-400 hover:text-gray-200"
              }`}
            >
              <MessageSquare className="w-3.5 h-3.5" /> Video-RAG AI
            </button>
          </div>

          {/* Latency & Telemetry Metric Pill */}
          {searchResult && activeTab === "moments" && <SearchResultSummary query={searchResult.query} count={searchResult.moments.length} profile={searchResult.profile} latencyMs={searchResult.latency_ms} calibrated={searchResult.calibrated} indexVersion={searchResult.index_version} strategyUsed={searchResult.strategy_used} modelsUsed={searchResult.models_used} modelsAttempted={searchResult.models_attempted} cascadePath={searchResult.cascade_path} captionStatus={searchResult.caption_status} captionModelId={searchResult.caption_model_id} onlineVerifierUsed={searchResult.online_verifier_used} />}
          {searchResult && activeTab === "moments" && <SearchWarningBanner warnings={uniqueWarnings(searchResult.warnings)} onReindex={() => setShowUploader(true)} />}

          {/* Tab Content */}
          {activeTab === "moments" ? (
            <MomentCards
              moments={searchResult?.moments || []}
              videoId={selectedVideo?.id}
              calibrated={searchResult?.calibrated || false}
              warnings={searchResult?.warnings || []}
              profile={searchResult?.profile || "fast"}
              emptyState={needsReindex || searchResult?.warnings?.includes("reindex_required") ? "reindex" : !searchResult ? "initial" : "no_match"}
              onReindex={() => setShowUploader(true)}
              onSelectMoment={handleSelectMoment}
              onExpandContext={handleExpandContext}
              activeMoment={activeMoment}
            />
          ) : (
            <VideoQAPanel
              videoId={selectedVideo?.id}
              onSeekToTimestamp={handleSeekFromQA}
            />
          )}
        </div>
      </div>

      {/* Delete Confirmation Modal */}
      {showDeleteConfirm && (videoToDelete || selectedVideo) && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-md flex items-center justify-center p-4">
          <div className="glass-panel border border-red-500/40 rounded-2xl max-w-md w-full p-6 shadow-2xl space-y-4 animate-in fade-in zoom-in-95 duration-200">
            <div className="flex items-center gap-3 text-red-400">
              <div className="p-2.5 rounded-xl bg-red-500/10 border border-red-500/20">
                <AlertTriangle className="w-6 h-6 text-red-400" />
              </div>
              <div>
                <h3 className="text-base font-bold text-white">ลบวิดีโอและเวกเตอร์ในระบบ</h3>
                <p className="text-xs text-gray-400">Permanently remove video and LanceDB vector indices</p>
              </div>
            </div>

            <p className="text-sm text-gray-300">
              คุณแน่ใจหรือไม่ว่าต้องการลบวิดีโอ{" "}
              <span className="font-semibold text-white">
                "{(videoToDelete || selectedVideo)?.filename}"
              </span>
              ?
            </p>

            <p className="text-xs text-red-300/90 bg-red-950/40 p-3 rounded-xl border border-red-900/50 leading-relaxed">
              ⚠️ การลบนี้จะลบเวกเตอร์ภาพ SigLIP 2 (768 มิติ), คำบรรยาย Qwen3-VL, หลักฐาน SAM 3.1, ตลอดจนไฟล์คีย์เฟรมและไฟล์วิดีโอต้นฉบับออกจากระบบอย่างถาวร
            </p>

            {deleteError && (
              <p className="text-xs text-red-400 font-medium">{deleteError}</p>
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
                className="px-4 py-2 rounded-xl bg-surface hover:bg-surfaceBorder text-gray-300 text-sm font-medium border border-surfaceBorder transition-colors"
              >
                ยกเลิก
              </button>
              <button
                type="button"
                disabled={isDeleting}
                onClick={handleDeleteVideo}
                className="px-4 py-2 rounded-xl bg-red-600 hover:bg-red-500 text-white text-sm font-semibold shadow-lg shadow-red-600/30 transition-colors flex items-center gap-2 disabled:opacity-50"
              >
                {isDeleting ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" /> กำลังลบ...
                  </>
                ) : (
                  <>
                    <Trash2 className="w-4 h-4" /> ยืนยันการลบ
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Developer & System Telemetry Panel Modal */}
      <DevPanel isOpen={showDevPanel} onClose={() => setShowDevPanel(false)} />

      {/* Pro Studio Keyboard Shortcuts Modal */}
      <ShortcutModal isOpen={showShortcutModal} onClose={() => setShowShortcutModal(false)} />
    </div>
  );
}
