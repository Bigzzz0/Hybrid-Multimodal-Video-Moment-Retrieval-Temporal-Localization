"use client";

import React, { useRef, useEffect, useState } from "react";
import { Play, Pause, RotateCcw, Volume2, VolumeX, Maximize } from "lucide-react";
import { TimelineHeatmap } from "./TimelineHeatmap";

interface VideoPlayerProps {
  streamUrl: string;
  duration: number;
  seekTime?: number | null;
  highlightInterval?: [number, number] | null;
  heatmapData: number[];
}

export const VideoPlayer: React.FC<VideoPlayerProps> = ({
  streamUrl,
  duration,
  seekTime,
  highlightInterval,
  heatmapData,
}) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [isMuted, setIsMuted] = useState(false);

  useEffect(() => {
    if (seekTime !== null && seekTime !== undefined && videoRef.current) {
      videoRef.current.currentTime = seekTime;
      videoRef.current.play().catch(() => {});
      setIsPlaying(true);
    }
  }, [seekTime]);

  const togglePlay = () => {
    if (!videoRef.current) return;
    if (isPlaying) {
      videoRef.current.pause();
      setIsPlaying(false);
    } else {
      videoRef.current.play();
      setIsPlaying(true);
    }
  };

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  };

  const handleSeek = (time: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = time;
      setCurrentTime(time);
    }
  };

  return (
    <div className="w-full glass-panel rounded-2xl p-4 space-y-3 shadow-2xl">
      <div className="relative aspect-video rounded-xl overflow-hidden bg-black border border-surfaceBorder group">
        <video
          ref={videoRef}
          src={streamUrl}
          onTimeUpdate={handleTimeUpdate}
          onPlay={() => setIsPlaying(true)}
          onPause={() => setIsPlaying(false)}
          className="w-full h-full object-contain"
        />

        {/* Video Overlay Play/Pause Action */}
        <div
          onClick={togglePlay}
          className="absolute inset-0 flex items-center justify-center bg-black/20 opacity-0 group-hover:opacity-100 transition-opacity cursor-pointer"
        >
          <div className="w-14 h-14 rounded-full bg-primary-600/90 text-white flex items-center justify-center shadow-lg transform group-hover:scale-110 transition-transform">
            {isPlaying ? <Pause className="w-7 h-7" /> : <Play className="w-7 h-7 ml-1" />}
          </div>
        </div>
      </div>

      {/* Dynamic Heatmap Timeline */}
      <TimelineHeatmap
        heatmapData={heatmapData}
        duration={duration}
        currentTime={currentTime}
        highlightInterval={highlightInterval}
        onSeek={handleSeek}
      />

      {/* Control Bar */}
      <div className="flex items-center justify-between px-2 pt-1 text-gray-300">
        <div className="flex items-center gap-3">
          <button
            onClick={togglePlay}
            className="p-2 rounded-lg bg-surface hover:bg-surfaceBorder text-white transition-colors"
          >
            {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
          </button>
          <button
            onClick={() => handleSeek(Math.max(0, currentTime - 5))}
            className="p-2 rounded-lg hover:bg-surface text-gray-400 hover:text-white transition-colors"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
          <button
            onClick={() => {
              if (videoRef.current) {
                videoRef.current.muted = !isMuted;
                setIsMuted(!isMuted);
              }
            }}
            className="p-2 rounded-lg hover:bg-surface text-gray-400 hover:text-white transition-colors"
          >
            {isMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
          </button>
          <span className="text-sm font-mono text-gray-400">
            {new Date(currentTime * 1000).toISOString().substring(14, 19)} /{" "}
            {new Date(duration * 1000).toISOString().substring(14, 19)}
          </span>
        </div>

        <button
          onClick={() => {
            if (videoRef.current) {
              if (document.fullscreenElement) {
                document.exitFullscreen();
              } else {
                videoRef.current.requestFullscreen();
              }
            }
          }}
          className="p-2 rounded-lg hover:bg-surface text-gray-400 hover:text-white transition-colors"
        >
          <Maximize className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
};
