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
  ChevronDown,
  ChevronUp,
  X,
  UploadCloud,
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

export default function DashboardPage() {
  const [videos, setVideos] = useState<VideoMetadata[]>([]);
  const [selectedVideo, setSelectedVideo] = useState<VideoMetadata | null>(null);
  const [keyframeRecords, setKeyframeRecords] = useState<VideoKeyframeItem[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [isSearching, setIsSearching] = useState(false);
  const [searchResult, setSearchResult] = useState<SearchResponse | null>(null);
  const [seekTime, setSeekTime] = useState<number | null>(null);
  const [highlightInterval, setHighlightInterval] = useState<[number, number] | null>(null);
  const [activeMoment, setActiveMoment] = useState<MomentItem | null>(null);
  const [activeTab, setActiveTab] = useState<"moments" | "rag">("moments");
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [videoToDelete, setVideoToDelete] = useState<VideoMetadata | null>(null);
  const [showDevPanel, setShowDevPanel] = useState(false);
  const [showUploader, setShowUploader] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

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

  // Global hotkey listener (Ctrl + K or / to focus search)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey && e.key === "k") || (e.key === "/" && document.activeElement?.tagName !== "INPUT")) {
        e.preventDefault();
        searchInputRef.current?.focus();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

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
    setIsSearching(true);
    setActiveTab("moments");

    try {
      const res = await apiClient.searchMoments(q, selectedVideo?.id);
      setSearchResult(res);

      if (res.moments && res.moments.length > 0) {
        const topMoment = res.moments[0];
        handleSelectMoment(topMoment);
      }
    } catch (err) {
      console.error("Search failed:", err);
    } finally {
      setIsSearching(false);
    }
  };

  const handleSelectMoment = (m: MomentItem, autoLoop: boolean = false) => {
    setActiveMoment(m);
    setSeekTime(m.t_start);
    setHighlightInterval([m.t_start, m.t_end]);
  };

  const handleSeekFromQA = (time: number) => {
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
      {/* 1. Video Library Reel (Visual Horizontal Filmstrip) */}
      <VideoLibraryReel
        videos={videos}
        selectedVideo={selectedVideo}
        onSelectVideo={(vid) => {
          setSelectedVideo(vid);
          setSearchResult(null);
          setHighlightInterval(null);
          setActiveMoment(null);
        }}
        onRequestDelete={handleRequestDelete}
        onToggleUpload={() => setShowUploader(!showUploader)}
      />

      {/* 2. Top Natural Language Search Suite */}
      <div className="glass-panel rounded-2xl p-4 shadow-xl space-y-3 border border-surfaceBorder/80">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSearch();
          }}
          className="flex items-center gap-3"
        >
          {/* Query Input */}
          <div className="relative flex-1">
            <input
              ref={searchInputRef}
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="ค้นหาการกระทำทางกายภาพ (เช่น 'คนปั่นจักรยาน', 'รถยนต์เลี้ยวซ้าย', 'student raising hand')..."
              className="w-full bg-surface/90 border border-surfaceBorder rounded-xl pl-11 pr-10 py-3 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 transition-all shadow-inner"
            />
            <Search className="w-5 h-5 absolute left-3.5 top-3.5 text-cyan-400" />
            {searchQuery && (
              <button
                type="button"
                onClick={() => setSearchQuery("")}
                className="absolute right-3.5 top-3.5 text-gray-500 hover:text-gray-300"
              >
                <X className="w-4 h-4" />
              </button>
            )}
          </div>

          {/* Search Button */}
          <button
            type="submit"
            disabled={isSearching}
            className="px-6 py-3 rounded-xl bg-gradient-to-r from-cyan-500 via-indigo-600 to-cyan-600 text-white font-bold text-xs tracking-wide shadow-lg shadow-cyan-500/20 hover:opacity-95 transition-all flex items-center gap-2 disabled:opacity-50 flex-shrink-0"
          >
            {isSearching ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" /> ค้นหา...
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4 text-cyan-200" /> สกัดช่วงเวลา (VMR)
              </>
            )}
          </button>

          {/* Dev & Telemetry Panel Button */}
          <button
            type="button"
            onClick={() => setShowDevPanel(true)}
            className="px-3 py-3 rounded-xl bg-surface hover:bg-surfaceBorder text-gray-300 hover:text-cyan-300 border border-surfaceBorder hover:border-cyan-500/40 text-xs font-mono font-semibold transition-all flex items-center gap-1.5 flex-shrink-0 shadow-sm"
            title="Open Developer & System Telemetry Panel"
          >
            <Terminal className="w-4 h-4 text-cyan-400" />
            <span className="hidden sm:inline">Dev Panel</span>
          </button>
        </form>

        {/* Categorized Action Suggestions & Recent Searches */}
        <QueryAssistant
          onSelectQuery={(q) => {
            setSearchQuery(q);
            handleSearch(q);
          }}
          currentQuery={searchQuery}
        />
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

      {/* 4. Studio Layout Grid: Player (7 cols) & Multi-modal Inspector (5 cols) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column: Video Studio Player & Scrubber (7 cols) */}
        <div className="lg:col-span-7 space-y-5">
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

        {/* Right Column: Tab Navigation (Retrieved Moments vs Video-RAG QA) (5 cols) */}
        <div className="lg:col-span-5 space-y-3.5">
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
          {searchResult && activeTab === "moments" && (
            <div className="glass-panel p-3 rounded-xl border border-surfaceBorder flex items-center justify-between text-xs text-gray-400">
              <span className="truncate pr-2">
                คำค้นหา: <b className="text-white font-medium">"{searchResult.query}"</b>
              </span>
              <span className="font-mono text-cyan-400 font-bold bg-cyan-950 px-2 py-0.5 rounded border border-cyan-800 flex-shrink-0">
                ⚡ {searchResult.latency_ms} ms
              </span>
            </div>
          )}

          {/* Tab Content */}
          {activeTab === "moments" ? (
            <MomentCards
              moments={searchResult?.moments || []}
              videoId={selectedVideo?.id}
              onSelectMoment={handleSelectMoment}
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
              ⚠️ การลบนี้จะลบเวกเตอร์ภาพ SigLIP 2 (768 มิติ), คำบรรยายการกระทำ Qwen2.5-VL, ตลอดจนไฟล์คีย์เฟรมและไฟล์วิดีโอต้นฉบับออกจากระบบอย่างถาวร
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
    </div>
  );
}
