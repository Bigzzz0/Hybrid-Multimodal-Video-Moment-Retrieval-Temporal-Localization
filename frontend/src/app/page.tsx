"use client";

import React, { useState, useEffect } from "react";
import { Search, Sparkles, Film, Loader2, ArrowRight } from "lucide-react";
import { VideoMetadata, MomentItem, SearchResponse } from "@/lib/types";
import { apiClient } from "@/lib/api";
import { VideoPlayer } from "@/components/player/VideoPlayer";
import { MomentCards } from "@/components/search/MomentCards";
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

  return (
    <div className="space-y-6">
      {/* Top Search Bar & Video Selector */}
      <div className="glass-panel rounded-2xl p-4 shadow-xl space-y-3 border border-surfaceBorder">
        <form onSubmit={handleSearch} className="flex items-center gap-3">
          {/* Video Selector Dropdown */}
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

      {/* Main Content Layout: Player & Moment List */}
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

        {/* Right Column: Ranked Moments & Insights (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          {searchResult && (
            <div className="glass-panel p-3.5 rounded-xl border border-surfaceBorder flex items-center justify-between text-xs text-gray-400">
              <span>
                Query: <span className="text-white font-medium">"{searchResult.query}"</span>
              </span>
              <span className="font-mono text-cyan-400 font-semibold">{searchResult.latency_ms} ms</span>
            </div>
          )}

          <MomentCards
            moments={searchResult?.moments || []}
            onSelectMoment={handleSelectMoment}
            activeMoment={activeMoment}
          />
        </div>
      </div>
    </div>
  );
}
