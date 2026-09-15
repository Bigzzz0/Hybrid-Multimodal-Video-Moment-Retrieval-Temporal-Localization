"use client";

import React from "react";
import { Database, Gauge, Layers, Timer } from "lucide-react";

interface SearchResultSummaryProps {
  query: string;
  count: number;
  profile: "fast" | "accurate";
  latencyMs: number;
  calibrated: boolean;
  indexVersion: string;
}

export const SearchResultSummary: React.FC<SearchResultSummaryProps> = ({ query, count, profile, latencyMs, calibrated, indexVersion }) => (
  <div className="glass-panel rounded-xl border border-surfaceBorder p-3" aria-label="Search result summary">
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-[11px] text-gray-400">
      <span className="min-w-0 flex-1 truncate">คำค้นหา: <b className="text-white font-medium">“{query}”</b></span>
      <span className="inline-flex items-center gap-1"><Layers className="h-3 w-3 text-cyan-400" /> {count} events</span>
      <span className="inline-flex items-center gap-1"><Gauge className="h-3 w-3 text-indigo-400" /> {profile}</span>
      <span className="inline-flex items-center gap-1 font-mono"><Timer className="h-3 w-3 text-cyan-400" /> {latencyMs.toFixed(0)} ms</span>
      <span className={`inline-flex items-center gap-1 font-mono ${calibrated ? "text-emerald-300" : "text-amber-300"}`}><Database className="h-3 w-3" /> {calibrated ? "calibrated" : "uncalibrated"}</span>
      <span className="font-mono text-gray-500">index {indexVersion}</span>
    </div>
  </div>
);
