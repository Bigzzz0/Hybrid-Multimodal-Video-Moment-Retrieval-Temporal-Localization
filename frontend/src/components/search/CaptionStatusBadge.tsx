"use client";

import React, { useEffect, useState } from "react";
import { Loader2, CheckCircle2, AlertTriangle } from "lucide-react";
import { apiClient } from "@/lib/api";
import { CaptionStatus } from "@/lib/types";

export const CaptionStatusBadge: React.FC<{ videoId?: string }> = ({ videoId }) => {
  const [status, setStatus] = useState<CaptionStatus | null>(null);

  useEffect(() => {
    if (!videoId) {
      setStatus(null);
      return;
    }
    let active = true;
    const load = async () => {
      try {
        const next = await apiClient.getCaptionStatus(videoId);
        if (active) setStatus(next);
      } catch {
        if (active) setStatus(null);
      }
    };
    void load();
    const timer = window.setInterval(() => void load(), 3000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [videoId]);

  if (!status) return null;
  const ready = status.status === "ready";
  const error = status.status === "error";
  const unavailable = status.status === "unavailable";
  const fallback = Number(status.fallback_count ?? 0) > 0;
  const expected = Number(status.expected_count ?? 0);
  const completed = Number(status.completed_count ?? 0);
  const percent = Math.min(100, Math.max(0, Number(status.progress_percent ?? (expected ? (completed / expected) * 100 : 0))));
  const currentModel = status.current_model || status.worker?.models?.current_model || "";
  const currentTask = status.current_task || status.worker?.models?.current_task || "";
  const running = status.status === "running" || status.status === "pending";
  return (
    <span className={`inline-flex min-w-[210px] flex-col gap-1 rounded-lg border px-2 py-1 text-[10px] font-mono ${ready ? "border-emerald-700/50 text-emerald-300" : error ? "border-amber-700/50 text-amber-300" : "border-cyan-800/60 text-cyan-300"}`} role="status" title={running ? `CapRL Q6 progress ${completed}/${expected}` : undefined}>
      <span className="inline-flex items-center gap-1.5">
        {ready ? <CheckCircle2 className="h-3 w-3" /> : error || unavailable ? <AlertTriangle className="h-3 w-3" /> : <Loader2 className="h-3 w-3 animate-spin" />}
        {ready ? "Caption CapRL Q6 พร้อม" : error ? "Caption error — ใช้ Fast ได้" : unavailable ? "Caption ยังไม่เริ่ม — ใช้ Fast ได้" : `กำลังสร้าง Caption ${completed}/${expected} (${Math.round(percent)}%)`}
      </span>
      {running && (
        <>
          <span className="h-1.5 w-full overflow-hidden rounded-full bg-slate-800">
            <span className="block h-full rounded-full bg-cyan-400 transition-all duration-500" style={{ width: `${Math.max(2, percent)}%` }} />
          </span>
          <span className="text-[9px] text-cyan-400/80">
            {currentModel === "q6" || currentModel.includes("q6") ? "GPU: CapRL Q6 กำลังประมวลผล" : "กำลังเตรียม worker"}
            {currentTask ? ` · ${currentTask}` : ""}
          </span>
        </>
      )}
    </span>
  );
};
