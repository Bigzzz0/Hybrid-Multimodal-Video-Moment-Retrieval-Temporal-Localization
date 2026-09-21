"use client";

import React from "react";
import { FlaskConical } from "lucide-react";
import { VlmBackend, VlmBackendStatus } from "@/lib/types";

const FALLBACK_OPTIONS: Array<{ backend: VlmBackend; label: string; research?: boolean }> = [
  { backend: "qwen3_vl_2b", label: "A  Qwen3-VL-2B" },
  { backend: "qwen3_vl_2b_vise", label: "B  Qwen3-VL-2B + VISE" },
  { backend: "caprl_qwen3vl_2b", label: "C  CapRL-Qwen3VL-2B", research: true },
  { backend: "caprl_qwen3vl_4b_q4", label: "D  CapRL-Qwen3VL-4B Q4", research: true },
  { backend: "caprl_qwen3vl_4b_q6", label: "D  CapRL-Qwen3VL-4B Q6", research: true },
  { backend: "caprl_video_4b", label: "E  CapRL-Video-4B", research: true },
];

interface Props {
  value: VlmBackend;
  statuses?: VlmBackendStatus[];
  disabled?: boolean;
  onChange: (backend: VlmBackend) => void;
}

export const VlmBackendSelector: React.FC<Props> = ({ value, statuses = [], disabled = false, onChange }) => (
  <label className="flex items-center gap-2 rounded-xl border border-surfaceBorder bg-surface px-2.5 py-1.5 text-[10px] font-mono text-gray-400" title="Experimental VLM backend; research/demo only">
    <FlaskConical className="h-3.5 w-3.5 text-fuchsia-300" />
    <span className="hidden xl:inline">VLM</span>
    <select
      value={value}
      disabled={disabled}
      onChange={(event) => onChange(event.target.value as VlmBackend)}
      className="max-w-[210px] bg-transparent text-cyan-200 outline-none disabled:opacity-50"
      aria-label="VLM backend"
    >
      {FALLBACK_OPTIONS.map((option) => {
        const status = statuses.find((item) => item.backend === option.backend);
        const ready = status ? status.ready : true;
        return <option key={option.backend} value={option.backend} disabled={status ? !ready : false}>
          {option.label}{option.research ? " · research" : ""}{status && !ready ? ` · ${status.status}` : ""}
        </option>;
      })}
    </select>
  </label>
);

