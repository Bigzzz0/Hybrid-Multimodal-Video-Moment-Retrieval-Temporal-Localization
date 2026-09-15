"use client";

import React from "react";
import { AlertTriangle, Info, XCircle } from "lucide-react";
import { UiWarning } from "@/lib/ui";

interface SearchWarningBannerProps {
  warnings: UiWarning[];
  onReindex?: () => void;
}

export const SearchWarningBanner: React.FC<SearchWarningBannerProps> = ({ warnings, onReindex }) => {
  if (warnings.length === 0) return null;
  const blocking = warnings.some((warning) => warning.severity === "blocking");
  const Icon = blocking ? XCircle : warnings.some((warning) => warning.severity === "warning") ? AlertTriangle : Info;
  return (
    <div className={`flex items-start gap-2.5 rounded-xl border px-3 py-2.5 text-xs ${blocking ? "border-red-700/60 bg-red-950/30 text-red-200" : warnings.some((w) => w.severity === "warning") ? "border-amber-700/50 bg-amber-950/25 text-amber-200" : "border-cyan-800/60 bg-cyan-950/20 text-cyan-200"}`} role="status">
      <Icon className="mt-0.5 h-4 w-4 flex-shrink-0" aria-hidden="true" />
      <div className="min-w-0 flex-1 space-y-1">
        {warnings.map((warning) => <p key={warning.code}><span className="font-mono text-[10px] opacity-70">[{warning.code}]</span> {warning.message}</p>)}
      </div>
      {warnings.some((warning) => warning.action === "reindex") && onReindex && (
        <button type="button" onClick={onReindex} className="min-h-9 shrink-0 rounded-lg border border-amber-500/50 px-2.5 py-1 text-[10px] font-semibold text-amber-100 hover:bg-amber-500/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400">สร้างดัชนีใหม่</button>
      )}
    </div>
  );
};
