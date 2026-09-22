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
  strategyUsed?: string;
  modelsUsed?: string[];
  modelsAttempted?: string[];
  cascadePath?: string[];
  captionStatus?: string;
  captionModelId?: string | null;
  onlineVerifierUsed?: boolean;
}

export const SearchResultSummary: React.FC<SearchResultSummaryProps> = ({ query, count, profile, latencyMs, calibrated, indexVersion, strategyUsed = "fast", modelsUsed = [], modelsAttempted = [], cascadePath = [], captionStatus, captionModelId, onlineVerifierUsed = false }) => (
  <div className="glass-panel rounded-xl border border-surfaceBorder p-3" aria-label="Search result summary">
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-[11px] text-gray-400">
      <span className="min-w-0 flex-1 truncate">คำค้นหา: <b className="text-white font-medium">“{query}”</b></span>
      <span className="inline-flex items-center gap-1"><Layers className="h-3 w-3 text-cyan-400" /> {count} events</span>
      <span className="inline-flex items-center gap-1"><Gauge className="h-3 w-3 text-indigo-400" /> {profile}</span>
      <span className="inline-flex items-center gap-1 font-mono"><Timer className="h-3 w-3 text-cyan-400" /> {latencyMs.toFixed(0)} ms</span>
      <span className={`inline-flex items-center gap-1 font-mono ${calibrated ? "text-emerald-300" : "text-amber-300"}`}><Database className="h-3 w-3" /> {calibrated ? "calibrated" : "uncalibrated"}</span>
      <span className="font-mono text-gray-500">index {indexVersion}</span>
      <span className="font-mono text-fuchsia-300">strategy {strategyUsed}</span>
      {modelsUsed.length > 0 && <span className="truncate max-w-full text-gray-500" title={modelsUsed.join(", ")}>{modelsUsed.join(" + ")}</span>}
      {captionStatus && <span className="font-mono text-indigo-300">caption {captionStatus}</span>}
      {captionModelId && <span className="truncate max-w-full text-gray-500" title={captionModelId}>{captionModelId}</span>}
      {onlineVerifierUsed && <span className="font-mono text-emerald-300">CapRL Q6 verified</span>}
    </div>
    {profile === "accurate" && cascadePath.length > 0 && (
      <div className="mt-2 flex flex-wrap gap-1.5 text-[10px] font-mono text-cyan-300" aria-label="Accurate cascade path">
        {cascadePath.map((stage, index) => <React.Fragment key={`${stage}-${index}`}><span className="rounded border border-cyan-900/70 bg-cyan-950/30 px-1.5 py-0.5">{stage}</span>{index < cascadePath.length - 1 && <span>→</span>}</React.Fragment>)}
      </div>
    )}
    {profile === "accurate" && modelsAttempted.length > 0 && <p className="mt-1.5 truncate text-[10px] text-gray-500" title={modelsAttempted.join(", ")}>attempted: {modelsAttempted.join(" → ")}</p>}
  </div>
);
