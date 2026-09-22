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
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-lg border px-2 py-1 text-[10px] font-mono ${ready ? "border-emerald-700/50 text-emerald-300" : error ? "border-amber-700/50 text-amber-300" : "border-cyan-800/60 text-cyan-300"}`} role="status">
      {ready ? <CheckCircle2 className="h-3 w-3" /> : error || unavailable ? <AlertTriangle className="h-3 w-3" /> : <Loader2 className="h-3 w-3 animate-spin" />}
      {ready ? (fallback ? "Caption พร้อม — Qwen สำรองบางฉาก" : "Caption CapRL Q6 พร้อม") : error ? "Caption error — ใช้ Fast ได้" : unavailable ? "Caption ยังไม่เริ่ม — ใช้ Fast ได้" : `กำลังสร้าง Caption ${status.completed_count}/${status.expected_count}`}
    </span>
  );
};
