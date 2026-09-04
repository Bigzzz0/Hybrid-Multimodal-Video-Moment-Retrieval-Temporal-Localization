"use client";

import React, { useRef } from "react";
import { VideoMetadata } from "@/lib/types";
import {
  Film,
  Clock,
  Layers,
  Trash2,
  ChevronLeft,
  ChevronRight,
  CheckCircle,
  UploadCloud,
  Tv,
  Cpu,
  Sparkles,
} from "lucide-react";

interface VideoLibraryReelProps {
  videos: VideoMetadata[];
  selectedVideo: VideoMetadata | null;
  onSelectVideo: (video: VideoMetadata) => void;
  onRequestDelete: (video: VideoMetadata) => void;
  onToggleUpload?: () => void;
  isCinemaMode?: boolean;
  onToggleCinemaMode?: () => void;
}

export const VideoLibraryReel: React.FC<VideoLibraryReelProps> = ({
  videos,
  selectedVideo,
  onSelectVideo,
  onRequestDelete,
  onToggleUpload,
  isCinemaMode = false,
  onToggleCinemaMode,
}) => {
  const scrollRef = useRef<HTMLDivElement | null>(null);

  const scroll = (direction: "left" | "right") => {
    if (scrollRef.current) {
      const amount = direction === "left" ? -280 : 280;
      scrollRef.current.scrollBy({ left: amount, behavior: "smooth" });
    }
  };

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  };

  return (
    <div className="w-full glass-panel rounded-2xl p-3.5 space-y-2.5 border border-surfaceBorder/80 shadow-xl">
      {/* Shelf Header */}
      <div className="flex items-center justify-between px-1">
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 rounded-lg bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
            <Film className="w-3.5 h-3.5" />
          </div>
          <span className="text-xs font-bold text-white tracking-wide uppercase">
            Indexed Video Catalog
          </span>
          <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-surface text-cyan-300 border border-surfaceBorder">
            {videos.length} {videos.length === 1 ? "Video" : "Videos"}
          </span>
        </div>

        {/* Shelf Toolbar Controls */}
        <div className="flex items-center gap-1.5">
          {/* Cinema Theater Mode Toggle */}
          {onToggleCinemaMode && (
            <button
              type="button"
              onClick={onToggleCinemaMode}
              className={`px-2.5 py-1 rounded-lg text-[11px] font-medium flex items-center gap-1.5 transition-all border ${
                isCinemaMode
                  ? "bg-cyan-950 border-cyan-500 text-cyan-300 shadow-md shadow-cyan-500/20"
                  : "bg-surface hover:bg-surfaceBorder text-gray-300 border-surfaceBorder"
              }`}
              title="Toggle Cinema Theater Mode (T)"
            >
              <Tv className="w-3.5 h-3.5 text-cyan-400" />
              <span>{isCinemaMode ? "Exit Cinema" : "Cinema Mode"}</span>
            </button>
          )}

          {onToggleUpload && (
            <button
              type="button"
              onClick={onToggleUpload}
              className="px-2.5 py-1 rounded-lg bg-indigo-600/20 hover:bg-indigo-600/30 text-indigo-300 border border-indigo-500/30 text-[11px] font-medium flex items-center gap-1.5 transition-all"
            >
              <UploadCloud className="w-3.5 h-3.5" />
              <span>+ Ingest Video</span>
            </button>
          )}

          <div className="flex items-center gap-1 ml-1">
            <button
              type="button"
              onClick={() => scroll("left")}
              className="p-1 rounded-lg bg-surface hover:bg-surfaceBorder text-gray-400 hover:text-white border border-surfaceBorder transition-colors"
              title="Scroll left"
            >
              <ChevronLeft className="w-3.5 h-3.5" />
            </button>
            <button
              type="button"
              onClick={() => scroll("right")}
              className="p-1 rounded-lg bg-surface hover:bg-surfaceBorder text-gray-400 hover:text-white border border-surfaceBorder transition-colors"
              title="Scroll right"
            >
              <ChevronRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>

      {/* Horizontal Scrollable Reel */}
      <div
        ref={scrollRef}
        className="flex items-center gap-3 overflow-x-auto pb-1 scrollbar-none snap-x"
        style={{ scrollbarWidth: "none", msOverflowStyle: "none" }}
      >
        {videos.length === 0 ? (
          <div className="w-full py-4 text-center text-gray-400 text-xs">
            No indexed videos yet. Upload a video below to get started.
          </div>
        ) : (
          videos.map((vid) => {
            const isSelected = selectedVideo?.id === vid.id;

            return (
              <div
                key={vid.id}
                onClick={() => onSelectVideo(vid)}
                className={`group relative flex-shrink-0 w-64 p-3 rounded-xl cursor-pointer transition-all duration-200 border snap-start ${
                  isSelected
                    ? "bg-gradient-to-b from-cyan-950/40 to-surface border-cyan-500/80 shadow-lg shadow-cyan-500/10 ring-1 ring-cyan-500/30"
                    : "bg-surface/60 hover:bg-surface border-surfaceBorder hover:border-gray-600"
                }`}
              >
                {/* Active Indicator & Delete Trigger */}
                <div className="flex items-start justify-between gap-2 mb-1.5">
                  <div className="flex items-center gap-1.5 min-w-0">
                    {isSelected ? (
                      <span className="flex items-center gap-1 text-[10px] font-semibold text-cyan-300 bg-cyan-950 px-1.5 py-0.5 rounded border border-cyan-800">
                        <CheckCircle className="w-2.5 h-2.5 text-cyan-400" /> Active Video
                      </span>
                    ) : (
                      <span className="text-[10px] font-mono text-gray-500 bg-black/40 px-1.5 py-0.5 rounded">
                        ID: {vid.id.substring(0, 8)}
                      </span>
                    )}

                    <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-indigo-950/70 border border-indigo-800/40 text-indigo-300">
                      768d Dual-Scale
                    </span>
                  </div>

                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      onRequestDelete(vid);
                    }}
                    title="Delete video & indexed vectors"
                    className="p-1 rounded-md text-gray-500 hover:text-red-400 hover:bg-red-500/10 transition-colors opacity-0 group-hover:opacity-100"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>

                {/* Video Title */}
                <h4
                  className={`text-xs font-semibold truncate transition-colors ${
                    isSelected ? "text-white" : "text-gray-300 group-hover:text-white"
                  }`}
                  title={vid.filename}
                >
                  {vid.filename}
                </h4>

                {/* Technical Metadata Badges */}
                <div className="flex items-center gap-1.5 mt-2 text-[10px] font-mono text-gray-400">
                  <span className="flex items-center gap-1 bg-black/40 px-1.5 py-0.5 rounded text-gray-300">
                    <Clock className="w-2.5 h-2.5 text-cyan-400" />
                    {formatDuration(vid.duration_sec)}
                  </span>

                  <span className="flex items-center gap-1 bg-black/40 px-1.5 py-0.5 rounded text-gray-300">
                    <Layers className="w-2.5 h-2.5 text-indigo-400" />
                    {vid.total_frames} kfs
                  </span>

                  {vid.fps && (
                    <span className="bg-black/40 px-1.5 py-0.5 rounded text-gray-400">
                      {vid.fps}fps
                    </span>
                  )}

                  {vid.resolution && (
                    <span className="text-[9px] text-gray-500 ml-auto uppercase">
                      {vid.resolution}
                    </span>
                  )}
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
