"use client";

import React from "react";
import { RadialConfidenceMeter } from "@/components/ui/RadialConfidenceMeter";
import { formatScore } from "@/lib/ui";

export const MomentScore: React.FC<{ score: number; calibrated: boolean; size?: number }> = ({ score, calibrated, size = 36 }) => (
  <div className="flex items-center gap-2" title={calibrated ? "Calibrated probability of relevance" : "Uncalibrated rank score; do not interpret as probability"}>
    {calibrated ? <RadialConfidenceMeter score={score} calibrated size={size} strokeWidth={3} /> : <span className="rounded-lg border border-amber-700/50 bg-amber-950/20 px-2 py-1 text-right font-mono text-[10px] font-semibold text-amber-300">{formatScore(score, false)}</span>}
  </div>
);
