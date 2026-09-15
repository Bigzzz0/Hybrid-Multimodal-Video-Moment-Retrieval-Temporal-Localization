"use client";

import React, { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";

export const SearchStatus: React.FC<{ profile: "fast" | "accurate" }> = ({ profile }) => {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    const started = performance.now();
    const timer = window.setInterval(() => setElapsed((performance.now() - started) / 1000), 100);
    return () => window.clearInterval(timer);
  }, []);
  return (
    <div className="flex items-center gap-2 text-xs text-cyan-200" role="status" aria-live="polite">
      <Loader2 className="w-4 h-4 animate-spin" aria-hidden="true" />
      <span>{profile === "accurate" ? "กำลังค้นหาและตรวจสอบภาพด้วย Qwen" : "กำลังค้นหาจาก visual index"}</span>
      <span className="font-mono text-cyan-400">{elapsed.toFixed(1)}s</span>
    </div>
  );
};
