"use client";

import React, { useState, useEffect } from "react";
import { Search, Sparkles, Film, Loader2, MessageSquare, ListFilter, Trash2, AlertTriangle } from "lucide-react";
import { VideoMetadata, MomentItem, SearchResponse } from "@/lib/types";
import { apiClient } from "@/lib/api";
import { VideoPlayer } from "@/components/player/VideoPlayer";
import { MomentCards } from "@/components/search/MomentCards";
import { VideoQAPanel } from "@/components/rag/VideoQAPanel";
import { Dropzone } from "@/components/upload/Dropzone";

export default function DashboardPage() {
  const [videos, setVideos] = useState<VideoMetadata[]>([]);
  const [selectedVideo, setSelectedVideo] = useState<VideoMetadata | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [isSearching, setIsSearching] = useState(false);
  const [searchResult, setSearchResult] = useState<SearchResponse | null>(null);
  const [seekTime, setSeekTime] = useState<number | null>(null);
  const [highlightInterval, setHighlightInterval] = useState<[number, number] | null>(null);
  const [activeMoment, setActiveMoment] = useState<MomentItem | null>(null);
  const [activeTab, setActiveTab] = useState<"moments" | "rag">("moments");
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  // Load videos on mount
  const fetchVideos = async () => {
    try {
      const data = await apiClient.getVideos();
      setVideos(data);
      if (data.length > 0 && !selectedVideo) {
        setSelectedVideo(data[0]);
      }
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    fetchVideos();
  }, []);

  const handleSearch = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!searchQuery.trim()) return;

    setIsSearching(true);
    setActiveTab("moments");
    try {
      const res = await apiClient.searchMoments(searchQuery, selectedVideo?.id);
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

  const handleSelectMoment = (m: MomentItem) => {
    setActiveMoment(m);
    setSeekTime(m.t_start);
    setHighlightInterval([m.t_start, m.t_end]);
  };

  const handleSeekFromQA = (time: number) => {
    setSeekTime(time);
    setHighlightInterval([time, time + 4.0]);
  };

  const handleDeleteVideo = async () => {
    if (!selectedVideo) return;

    setIsDeleting(true);
    setDeleteError(null);
    try {
      await apiClient.deleteVideo(selectedVideo.id);
      setShowDeleteConfirm(false);

      // Clear search and playback states
      setSearchResult(null);
      setHighlightInterval(null);
      setActiveMoment(null);
      setSeekTime(null);

      // Refresh list and select next video
      const remainingVideos = await apiClient.getVideos();
      setVideos(remainingVideos);
      if (remainingVideos.length > 0) {
        setSelectedVideo(remainingVideos[0]);
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
    <div className="space-y-6">
      {/* Top Search Bar & Video Selector */}
      <div className="glass-panel rounded-2xl p-4 shadow-xl space-y-3 border border-surfaceBorder">
        <form onSubmit={handleSearch} className="flex items-center gap-3">
          {/* Video Selector Dropdown & Delete Action */}
          <div className="flex items-center gap-2">
            <div className="relative min-w-[200px]">
              <select
                value={selectedVideo?.id || ""}
                onChange={(e) => {
                  const vid = videos.find((v) => v.id === e.target.value);
                  if (vid) {
                    setSelectedVideo(vid);
                    setSearchResult(null);
                    setHighlightInterval(null);
                  }
                }}
                className="w-full bg-surface border border-surfaceBorder rounded-xl px-3.5 py-3 text-sm text-white font-medium focus:outline-none focus:border-cyan-500 transition-colors"
              >
                {videos.length === 0 && <option value="">No videos uploaded</option>}
                {videos.map((v) => (
                  <option key={v.id} value={v.id}>
                    📹 {v.filename.length > 25 ? v.filename.substring(0, 22) + "..." : v.filename}
                  </option>
                ))}
              </select>
            </div>

            {selectedVideo && (
              <button
                type="button"
                onClick={() => setShowDeleteConfirm(true)}
                title="Delete this video and indexed data"
                className="p-3 rounded-xl bg-red-500/10 hover:bg-red-500/20 text-red-400 hover:text-red-300 border border-red-500/30 transition-all flex items-center justify-center flex-shrink-0"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            )}
          </div>

          {/* Natural Language Query Input */}
          <div className="relative flex-1">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search actions, speech, or scene events (e.g. 'ฉากที่อธิบายกราฟแท่ง', 'person drinking water')..."
              className="w-full bg-surface/80 border border-surfaceBorder rounded-xl pl-11 pr-4 py-3 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 transition-all"
            />
            <Search className="w-5 h-5 absolute left-3.5 top-3.5 text-gray-400" />
          </div>

          {/* Search Button */}
          <button
            type="submit"
            disabled={isSearching}
            className="px-6 py-3 rounded-xl bg-gradient-to-r from-indigo-600 to-cyan-500 text-white font-semibold text-sm shadow-lg shadow-cyan-500/20 hover:opacity-95 transition-opacity flex items-center gap-2 disabled:opacity-50"
          >
            {isSearching ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" /> Searching...
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4" /> Retrieve Moments
              </>
            )}
          </button>
        </form>

        {/* Query Suggestions */}
        <div className="flex items-center gap-2 text-xs text-gray-400 overflow-x-auto pt-1">
          <span className="text-gray-500 flex-shrink-0">Suggestions:</span>
          {[
            "person shares slide presentation",
            "car turns left at intersection",
            "speaker explains system architecture",
            "hand reaches for water bottle",
          ].map((suggestion, i) => (
            <button
              key={i}
              type="button"
              onClick={() => {
                setSearchQuery(suggestion);
              }}
              className="px-2.5 py-1 rounded-full bg-surface hover:bg-surfaceBorder text-gray-300 text-[11px] border border-surfaceBorder transition-colors flex-shrink-0"
            >
              {suggestion}
            </button>
          ))}
        </div>
      </div>

      {/* Main Content Layout: Player & Tabs */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column: Video Player & Uploader (7 cols) */}
        <div className="lg:col-span-7 space-y-6">
          {selectedVideo ? (
            <VideoPlayer
              streamUrl={apiClient.getVideoStreamUrl(selectedVideo.id)}
              duration={selectedVideo.duration_sec}
              seekTime={seekTime}
              highlightInterval={highlightInterval}
              heatmapData={searchResult?.timeline_heatmap || []}
            />
          ) : (
            <div className="aspect-video glass-panel rounded-2xl flex items-center justify-center text-gray-500 border border-surfaceBorder">
              <div className="text-center space-y-2">
                <Film className="w-12 h-12 mx-auto text-gray-600" />
                <p>Upload or select a video to begin retrieval</p>
              </div>
            </div>
          )}

          {/* Upload Dropzone */}
          <Dropzone
            onUploadSuccess={(vid) => {
              fetchVideos();
            }}
          />
        </div>

        {/* Right Column: Tab Navigation (Moments vs Video-RAG QA) (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
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
              <ListFilter className="w-3.5 h-3.5" /> Retrieved Moments
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
              <MessageSquare className="w-3.5 h-3.5" /> Video-RAG QA
            </button>
          </div>

          {/* Search Result Latency Badge */}
          {searchResult && activeTab === "moments" && (
            <div className="glass-panel p-3 rounded-xl border border-surfaceBorder flex items-center justify-between text-xs text-gray-400">
              <span>
                Query: <span className="text-white font-medium">"{searchResult.query}"</span>
              </span>
              <span className="font-mono text-cyan-400 font-semibold">{searchResult.latency_ms} ms</span>
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
      {showDeleteConfirm && selectedVideo && (
        <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="glass-panel border border-red-500/30 rounded-2xl max-w-md w-full p-6 shadow-2xl space-y-4 animate-in fade-in zoom-in-95 duration-200">
            <div className="flex items-center gap-3 text-red-400">
              <div className="p-2.5 rounded-xl bg-red-500/10 border border-red-500/20">
                <AlertTriangle className="w-6 h-6 text-red-400" />
              </div>
              <div>
                <h3 className="text-base font-bold text-white">Delete Video</h3>
                <p className="text-xs text-gray-400">Permanently delete video & indexed vectors</p>
              </div>
            </div>

            <p className="text-sm text-gray-300">
              Are you sure you want to delete <span className="font-semibold text-white">"{selectedVideo.filename}"</span>?
            </p>

            <p className="text-xs text-red-400/90 bg-red-950/40 p-3 rounded-xl border border-red-900/50">
              ⚠️ This will permanently remove all SigLIP 2 visual vectors, Whisper speech transcripts, MiniCPM-V captions, and video files from LanceDB.
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
                  setDeleteError(null);
                }}
                className="px-4 py-2 rounded-xl bg-surface hover:bg-surfaceBorder text-gray-300 text-sm font-medium border border-surfaceBorder transition-colors"
              >
                Cancel
              </button>
              <button
                type="button"
                disabled={isDeleting}
                onClick={handleDeleteVideo}
                className="px-4 py-2 rounded-xl bg-red-600 hover:bg-red-500 text-white text-sm font-semibold shadow-lg shadow-red-600/30 transition-colors flex items-center gap-2 disabled:opacity-50"
              >
                {isDeleting ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" /> Deleting...
                  </>
                ) : (
                  <>
                    <Trash2 className="w-4 h-4" /> Confirm Delete
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
