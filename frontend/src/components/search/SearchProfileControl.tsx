"use client";

import React from "react";
import { Gauge, ShieldCheck, Zap } from "lucide-react";

interface SearchProfileControlProps {
  value: "fast" | "accurate";
  disabled?: boolean;
  onChange: (value: "fast" | "accurate") => void;
}

export const SearchProfileControl: React.FC<SearchProfileControlProps> = ({ value, disabled = false, onChange }) => (
  <div className="flex items-center gap-1 rounded-xl bg-surface border border-surfaceBorder p-1" role="group" aria-label="Retrieval profile">
    {(["fast", "accurate"] as const).map((profile) => {
      const selected = value === profile;
      return (
        <button
          key={profile}
          type="button"
          disabled={disabled}
          aria-pressed={selected}
          onClick={() => onChange(profile)}
          className={`min-h-9 px-3 py-1.5 rounded-lg text-[10px] font-mono font-semibold transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400 disabled:cursor-not-allowed disabled:opacity-50 ${selected ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40" : "text-gray-500 hover:text-gray-300"}`}
          title={profile === "accurate" ? "SigLIP2 → selected VLM; budget ไม่เกิน 60 วินาที" : "ค้นหาเร็วจาก visual index; warm target ไม่เกิน 1 วินาที"}
        >
          <span className="inline-flex items-center gap-1.5">
            {profile === "accurate" ? <ShieldCheck className="w-3 h-3" /> : <Zap className="w-3 h-3" />}
            {profile === "accurate" ? "Accurate" : "Fast"}
          </span>
        </button>
      );
    })}
    <span className="sr-only"><Gauge /> {value === "accurate" ? "SigLIP2 and selected VLM verification enabled" : "Visual index retrieval enabled"}</span>
  </div>
);
