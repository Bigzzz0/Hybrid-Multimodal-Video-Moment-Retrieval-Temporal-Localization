"use client";

import React from "react";
import { Activity, Box, Sparkles, User, Zap } from "lucide-react";

const EVIDENCE = [
  ["visual", "VIS", "Visual match", "text-cyan-300", "bg-cyan-400", User],
  ["caption", "CAP", "Scene caption match", "text-indigo-300", "bg-indigo-400", Sparkles],
  ["temporal", "TMP", "Temporal proposal quality", "text-amber-300", "bg-amber-400", Activity],
  ["verifier", "VLM", "Qwen visual verification", "text-emerald-300", "bg-emerald-400", Zap],
  ["sam", "SAM", "SAM 3.1 grounding confidence", "text-fuchsia-300", "bg-fuchsia-400", Box],
] as const;

export const MomentEvidenceBreakdown: React.FC<{ breakdown?: Record<string, number> | null; showVerifier?: boolean; showSam?: boolean }> = ({ breakdown = {}, showVerifier = true, showSam = false }) => (
  <div className="grid grid-cols-2 gap-1.5 pt-2 border-t border-surfaceBorder/60 text-[10px] font-mono sm:grid-cols-5">
    {EVIDENCE.filter(([key]) => key !== "sam" || showSam).map(([key, label, title, textColor, barColor, Icon]) => {
      const available = key !== "verifier" || showVerifier;
      const value = Number(breakdown?.[key] ?? 0);
      return (
        <div key={key} className="rounded-lg border border-surfaceBorder/60 bg-surface/90 p-1.5" title={title}>
          <div className="flex items-center justify-between text-gray-400">
            <span className="flex items-center gap-1"><Icon className={`h-2.5 w-2.5 ${textColor}`} /> {label}</span>
            <span className={`font-bold ${available ? textColor : "text-gray-500"}`}>{available ? value.toFixed(3) : "—"}</span>
          </div>
          <div className="mt-1 h-1 w-full overflow-hidden rounded-full bg-gray-800">
            <div className={`h-full rounded-full ${barColor}`} style={{ width: `${available ? Math.max(0, Math.min(1, value)) * 100 : 0}%` }} />
          </div>
        </div>
      );
    })}
  </div>
);
