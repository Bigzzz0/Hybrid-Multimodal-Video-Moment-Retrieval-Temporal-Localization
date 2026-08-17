"use client";

import React, { useState, useEffect, useRef } from "react";
import {
  Terminal,
  Cpu,
  Database,
  Layers,
  Activity,
  RefreshCw,
  X,
  Server,
  Zap,
  HardDrive,
  Copy,
  Check,
  Code2,
  Sparkles,
  ShieldCheck,
  Gauge
} from "lucide-react";
import { SystemTelemetry } from "@/lib/types";
import { apiClient } from "@/lib/api";

interface DevPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

export const DevPanel: React.FC<DevPanelProps> = ({ isOpen, onClose }) => {
  const [telemetry, setTelemetry] = useState<SystemTelemetry | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<"hardware" | "storage" | "models" | "logs">("hardware");
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [copiedLogs, setCopiedLogs] = useState(false);
  const [logFilter, setLogFilter] = useState("");
  const logEndRef = useRef<HTMLDivElement | null>(null);

  const fetchTelemetry = async () => {
    try {
      setIsLoading(true);
      const data = await apiClient.getSystemTelemetry();
      setTelemetry(data);
    } catch (err) {
      console.debug("Failed to fetch dev telemetry:", err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (isOpen) {
      fetchTelemetry();
    }
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen || !autoRefresh) return;
    const timer = setInterval(() => {
      fetchTelemetry();
    }, 3000);
    return () => clearInterval(timer);
  }, [isOpen, autoRefresh]);

  if (!isOpen) return null;

  const handleCopyLogs = () => {
    if (telemetry?.recent_logs) {
      navigator.clipboard.writeText(telemetry.recent_logs.join("\n"));
      setCopiedLogs(true);
      setTimeout(() => setCopiedLogs(false), 2000);
    }
  };

  const filteredLogs = telemetry?.recent_logs?.filter((l) =>
    l.toLowerCase().includes(logFilter.toLowerCase())
  ) || [];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 bg-black/80 backdrop-blur-md animate-in fade-in duration-200">
      <div className="w-full max-w-5xl h-[85vh] bg-surface/95 border border-surfaceBorder rounded-2xl shadow-2xl flex flex-col overflow-hidden text-gray-200">
        
        {/* Top Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-surfaceBorder bg-background/60">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-cyan-500/20 border border-cyan-500/40 flex items-center justify-center text-cyan-400">
              <Terminal className="w-4 h-4" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-base font-bold text-white tracking-wide">Developer & System Telemetry Panel</h2>
                <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded-full bg-emerald-950/80 border border-emerald-800 text-emerald-300 flex items-center gap-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping" /> Live Telemetry
                </span>
              </div>
              <p className="text-xs text-gray-400">Hardware metrics, Vector Storage stats, AI Model registry & execution logs</p>
            </div>
          </div>

          <div className="flex items-center gap-2.5">
            <button
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={`text-xs px-3 py-1.5 rounded-lg border font-mono flex items-center gap-1.5 transition-all ${
                autoRefresh
                  ? "bg-cyan-950/60 border-cyan-700 text-cyan-300"
                  : "bg-surface border-surfaceBorder text-gray-400"
              }`}
            >
              <RefreshCw className={`w-3.5 h-3.5 ${autoRefresh && isLoading ? "animate-spin" : ""}`} />
              {autoRefresh ? "Auto (3s)" : "Paused"}
            </button>

            <button
              onClick={fetchTelemetry}
              disabled={isLoading}
              className="p-1.5 text-gray-400 hover:text-white rounded-lg hover:bg-surfaceBorder/60 transition-colors"
              title="Manual Refresh"
            >
              <RefreshCw className={`w-4 h-4 ${isLoading ? "animate-spin" : ""}`} />
            </button>

            <button
              onClick={onClose}
              className="p-1.5 text-gray-400 hover:text-white rounded-lg hover:bg-surfaceBorder/60 transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex items-center gap-1 px-6 pt-3 border-b border-surfaceBorder bg-background/40">
          <button
            onClick={() => setActiveTab("hardware")}
            className={`flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-t-lg border-b-2 transition-all ${
              activeTab === "hardware"
                ? "border-cyan-400 text-cyan-300 bg-surface/60"
                : "border-transparent text-gray-400 hover:text-gray-200"
            }`}
          >
            <Cpu className="w-4 h-4" /> Hardware & GPU Acceleration
          </button>

          <button
            onClick={() => setActiveTab("storage")}
            className={`flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-t-lg border-b-2 transition-all ${
              activeTab === "storage"
                ? "border-cyan-400 text-cyan-300 bg-surface/60"
                : "border-transparent text-gray-400 hover:text-gray-200"
            }`}
          >
            <Database className="w-4 h-4" /> LanceDB Vector Storage
          </button>

          <button
            onClick={() => setActiveTab("models")}
            className={`flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-t-lg border-b-2 transition-all ${
              activeTab === "models"
                ? "border-cyan-400 text-cyan-300 bg-surface/60"
                : "border-transparent text-gray-400 hover:text-gray-200"
            }`}
          >
            <Layers className="w-4 h-4" /> AI Models & Pipeline Config
          </button>

          <button
            onClick={() => setActiveTab("logs")}
            className={`flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-t-lg border-b-2 transition-all ${
              activeTab === "logs"
                ? "border-cyan-400 text-cyan-300 bg-surface/60"
                : "border-transparent text-gray-400 hover:text-gray-200"
            }`}
          >
            <Terminal className="w-4 h-4" /> Live Backend Logs
          </button>
        </div>

        {/* Tab Contents */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          
          {/* TAB 1: Hardware & GPU */}
          {activeTab === "hardware" && (
            <div className="space-y-6">
              {/* GPU Metric Card */}
              <div className="p-5 rounded-xl bg-surface border border-surfaceBorder space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2.5">
                    <div className="w-8 h-8 rounded-lg bg-emerald-500/20 text-emerald-400 flex items-center justify-center">
                      <Zap className="w-4 h-4" />
                    </div>
                    <div>
                      <h3 className="text-sm font-bold text-white">NVIDIA CUDA GPU Engine</h3>
                      <p className="text-xs text-gray-400">Direct Tensor Core & FP16 hardware acceleration</p>
                    </div>
                  </div>
                  <span className={`text-xs font-mono font-bold px-2.5 py-1 rounded-full border ${
                    telemetry?.gpu?.available
                      ? "bg-emerald-950 border-emerald-700 text-emerald-300"
                      : "bg-red-950 border-red-700 text-red-300"
                  }`}>
                    {telemetry?.gpu?.available ? "GPU Acceleration Active" : "CPU Fallback"}
                  </span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
                  <div className="p-3.5 rounded-lg bg-background/70 border border-surfaceBorder/80">
                    <span className="text-[11px] text-gray-400 uppercase tracking-wider block">GPU Model</span>
                    <span className="text-sm font-bold text-cyan-300 font-mono block mt-1">
                      {telemetry?.gpu?.device_name || "Detecting..."}
                    </span>
                  </div>

                  <div className="p-3.5 rounded-lg bg-background/70 border border-surfaceBorder/80">
                    <span className="text-[11px] text-gray-400 uppercase tracking-wider block">VRAM in Use (Hardware)</span>
                    <span className="text-sm font-bold text-white font-mono block mt-1">
                      {telemetry?.gpu?.allocated_vram_mb || 0} MB <span className="text-xs text-gray-500 font-normal">/ {telemetry?.gpu?.total_vram_mb || 0} MB</span>
                    </span>
                  </div>

                  <div className="p-3.5 rounded-lg bg-background/70 border border-surfaceBorder/80">
                    <span className="text-[11px] text-gray-400 uppercase tracking-wider block">Free VRAM</span>
                    <span className="text-sm font-bold text-emerald-300 font-mono block mt-1">
                      {Math.max(0, (telemetry?.gpu?.total_vram_mb || 0) - (telemetry?.gpu?.allocated_vram_mb || 0)).toFixed(1)} MB
                    </span>
                  </div>

                  <div className="p-3.5 rounded-lg bg-background/70 border border-surfaceBorder/80">
                    <span className="text-[11px] text-gray-400 uppercase tracking-wider block">Compute Capability</span>
                    <span className="text-sm font-bold text-indigo-300 font-mono block mt-1">
                      {telemetry?.gpu?.compute_capability || "N/A"}
                    </span>
                  </div>
                </div>

                {/* VRAM Utilization Bar */}
                {telemetry?.gpu?.total_vram_mb && (
                  <div className="space-y-1.5 pt-1">
                    <div className="flex justify-between text-xs font-mono text-gray-400">
                      <span>Live VRAM Consumption</span>
                      <span className="text-cyan-300 font-bold">{((telemetry.gpu.allocated_vram_mb / telemetry.gpu.total_vram_mb) * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-background rounded-full h-2 overflow-hidden border border-surfaceBorder">
                      <div
                        className="bg-gradient-to-r from-cyan-500 via-indigo-500 to-emerald-400 h-2 rounded-full transition-all duration-300 shadow-sm"
                        style={{ width: `${Math.min(100, Math.max(1, (telemetry.gpu.allocated_vram_mb / telemetry.gpu.total_vram_mb) * 100))}%` }}
                      />
                    </div>
                  </div>
                )}
              </div>

              {/* Host CPU & RAM Card */}
              <div className="p-5 rounded-xl bg-surface border border-surfaceBorder space-y-4">
                <div className="flex items-center gap-2.5">
                  <div className="w-8 h-8 rounded-lg bg-indigo-500/20 text-indigo-400 flex items-center justify-center">
                    <Server className="w-4 h-4" />
                  </div>
                  <div>
                    <h3 className="text-sm font-bold text-white">Host System Resource Monitor</h3>
                    <p className="text-xs text-gray-400">CPU multi-threading and system memory telemetry</p>
                  </div>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
                  <div className="p-3.5 rounded-lg bg-background/70 border border-surfaceBorder/80">
                    <span className="text-[11px] text-gray-400 uppercase tracking-wider block">CPU Usage</span>
                    <span className="text-sm font-bold text-white font-mono block mt-1">
                      {telemetry?.system?.cpu_percent || 0}%
                    </span>
                  </div>

                  <div className="p-3.5 rounded-lg bg-background/70 border border-surfaceBorder/80">
                    <span className="text-[11px] text-gray-400 uppercase tracking-wider block">CPU Cores</span>
                    <span className="text-sm font-bold text-white font-mono block mt-1">
                      {telemetry?.system?.cpu_count_physical} Physical / {telemetry?.system?.cpu_count_logical} Logical
                    </span>
                  </div>

                  <div className="p-3.5 rounded-lg bg-background/70 border border-surfaceBorder/80">
                    <span className="text-[11px] text-gray-400 uppercase tracking-wider block">System RAM</span>
                    <span className="text-sm font-bold text-white font-mono block mt-1">
                      {telemetry?.system?.ram_used_gb || 0} GB <span className="text-xs text-gray-500 font-normal">/ {telemetry?.system?.ram_total_gb || 0} GB ({telemetry?.system?.ram_percent}%)</span>
                    </span>
                  </div>

                  <div className="p-3.5 rounded-lg bg-background/70 border border-surfaceBorder/80">
                    <span className="text-[11px] text-gray-400 uppercase tracking-wider block">Environment</span>
                    <span className="text-sm font-bold text-white font-mono block mt-1">
                      Python {telemetry?.system?.python_version} ({telemetry?.system?.platform})
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* TAB 2: LanceDB Storage */}
          {activeTab === "storage" && (
            <div className="space-y-6">
              <div className="p-5 rounded-xl bg-surface border border-surfaceBorder space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2.5">
                    <div className="w-8 h-8 rounded-lg bg-amber-500/20 text-amber-400 flex items-center justify-center">
                      <Database className="w-4 h-4" />
                    </div>
                    <div>
                      <h3 className="text-sm font-bold text-white">LanceDB Serverless Vector Database</h3>
                      <p className="text-xs text-gray-400">Zero-copy PyArrow & Disk-based IVF-PQ ANN indexing</p>
                    </div>
                  </div>
                  <span className="text-xs font-mono font-bold px-2.5 py-1 rounded-full bg-amber-950 border border-amber-700 text-amber-300">
                    {telemetry?.lancedb?.total_records || 0} Total Records Indexed
                  </span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                  {Object.entries(telemetry?.lancedb?.tables || {}).map(([tbl, count]) => (
                    <div key={tbl} className="p-3.5 rounded-lg bg-background/70 border border-surfaceBorder/80 flex items-center justify-between">
                      <div>
                        <span className="text-xs font-mono font-semibold text-white">table_{tbl}</span>
                        <p className="text-[11px] text-gray-400">
                          {tbl === "videos" && "Video metadata catalog"}
                          {tbl === "video_frames" && "768-dim SigLIP 2 visual vectors"}
                          {tbl === "scenes" && "Temporal boundary cuts"}
                          {tbl === "transcripts" && "Timestamped Whisper ASR text"}
                          {tbl === "search_logs" && "Query latency & evaluation logs"}
                        </p>
                      </div>
                      <span className="text-sm font-mono font-bold text-cyan-300 bg-cyan-950 px-2 py-0.5 rounded border border-cyan-800">
                        {count}
                      </span>
                    </div>
                  ))}
                </div>

                <div className="p-3 rounded-lg bg-background/90 border border-surfaceBorder text-xs text-gray-400 flex items-center gap-2 font-mono">
                  <HardDrive className="w-4 h-4 text-gray-500 flex-shrink-0" />
                  <span className="truncate">Storage Location: {telemetry?.lancedb?.storage_path}</span>
                </div>
              </div>
            </div>
          )}

          {/* TAB 3: AI Models */}
          {activeTab === "models" && (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {/* Faster-Whisper */}
              <div className="p-5 rounded-xl bg-surface border border-surfaceBorder space-y-3">
                <div className="flex items-center gap-2.5">
                  <div className="w-8 h-8 rounded-lg bg-purple-500/20 text-purple-400 flex items-center justify-center">
                    <Sparkles className="w-4 h-4" />
                  </div>
                  <div>
                    <h4 className="text-sm font-bold text-white">Speech ASR: Faster-Whisper</h4>
                    <p className="text-xs text-gray-400">Timestamped Speech-to-Text</p>
                  </div>
                </div>
                <div className="p-3 rounded-lg bg-background/80 border border-surfaceBorder space-y-1 text-xs font-mono">
                  <p className="text-gray-300">Model: <span className="text-purple-300">Whisper-Large-v3-Turbo</span></p>
                  <p className="text-gray-300">Engine: <span className="text-emerald-300">CTranslate2 (CUDA FP16)</span></p>
                  <p className="text-gray-300">Beam Size: <span className="text-white">5</span> | VAD: <span className="text-white">Active</span></p>
                </div>
              </div>

              {/* SigLIP 2 */}
              <div className="p-5 rounded-xl bg-surface border border-surfaceBorder space-y-3">
                <div className="flex items-center gap-2.5">
                  <div className="w-8 h-8 rounded-lg bg-cyan-500/20 text-cyan-400 flex items-center justify-center">
                    <Gauge className="w-4 h-4" />
                  </div>
                  <div>
                    <h4 className="text-sm font-bold text-white">Vision-Text: SigLIP 2 (NaFlex)</h4>
                    <p className="text-xs text-gray-400">Zero-Shot Joint Visual Embedding</p>
                  </div>
                </div>
                <div className="p-3 rounded-lg bg-background/80 border border-surfaceBorder space-y-1 text-xs font-mono">
                  <p className="text-gray-300">Model ID: <span className="text-cyan-300">google/siglip2-base-patch16-256</span></p>
                  <p className="text-gray-300">Vector Dimension: <span className="text-emerald-300">768-dim</span></p>
                  <p className="text-gray-300">Distance Metric: <span className="text-white">Cosine Similarity</span></p>
                </div>
              </div>

              {/* MiniCPM-V 2.6 */}
              <div className="p-5 rounded-xl bg-surface border border-surfaceBorder space-y-3">
                <div className="flex items-center gap-2.5">
                  <div className="w-8 h-8 rounded-lg bg-blue-500/20 text-blue-400 flex items-center justify-center">
                    <Code2 className="w-4 h-4" />
                  </div>
                  <div>
                    <h4 className="text-sm font-bold text-white">Dense Captioner: MiniCPM-V 2.6</h4>
                    <p className="text-xs text-gray-400">Phase 2 Background Action Reasoner</p>
                  </div>
                </div>
                <div className="p-3 rounded-lg bg-background/80 border border-surfaceBorder space-y-1 text-xs font-mono">
                  <p className="text-gray-300">Model ID: <span className="text-blue-300">openbmb/MiniCPM-V-2_6</span></p>
                  <p className="text-gray-300">Quantization: <span className="text-emerald-300">4-bit NF4 (BitsAndBytes)</span></p>
                  <p className="text-gray-300">Execution: <span className="text-white">Background Worker</span></p>
                </div>
              </div>

              {/* Temporal Boundary Localizer */}
              <div className="p-5 rounded-xl bg-surface border border-surfaceBorder space-y-3">
                <div className="flex items-center gap-2.5">
                  <div className="w-8 h-8 rounded-lg bg-amber-500/20 text-amber-400 flex items-center justify-center">
                    <ShieldCheck className="w-4 h-4" />
                  </div>
                  <div>
                    <h4 className="text-sm font-bold text-white">Boundary Localizer: 2D-TAN + NMS</h4>
                    <p className="text-xs text-gray-400">Multi-Scale Temporal IoU + 1D Wasserstein NMS</p>
                  </div>
                </div>
                <div className="p-3 rounded-lg bg-background/80 border border-surfaceBorder space-y-1 text-xs font-mono">
                  <p className="text-gray-300">Algorithm: <span className="text-amber-300">2D-TAN Anchor Ranking</span></p>
                  <p className="text-gray-300">NMS Suppress: <span className="text-emerald-300">IoU threshold = 0.5</span></p>
                  <p className="text-gray-300">Top-K Selection: <span className="text-white">K = 5 Moments</span></p>
                </div>
              </div>
            </div>
          )}

          {/* TAB 4: Live Backend Logs */}
          {activeTab === "logs" && (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 flex-1 max-w-sm">
                  <input
                    type="text"
                    placeholder="Filter logs (e.g. CUDA, ERROR, SigLIP)..."
                    value={logFilter}
                    onChange={(e) => setLogFilter(e.target.value)}
                    className="w-full bg-background border border-surfaceBorder rounded-lg px-3 py-1.5 text-xs text-white placeholder-gray-500 focus:outline-none focus:border-cyan-500 font-mono"
                  />
                </div>

                <div className="flex items-center gap-2">
                  <button
                    onClick={handleCopyLogs}
                    className="flex items-center gap-1 px-3 py-1.5 rounded-lg border border-surfaceBorder bg-background/80 hover:bg-surface text-xs text-gray-300 transition-colors"
                  >
                    {copiedLogs ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                    {copiedLogs ? "Copied!" : "Copy Logs"}
                  </button>
                </div>
              </div>

              <div className="bg-black/90 border border-surfaceBorder rounded-xl p-4 font-mono text-[11px] h-96 overflow-y-auto space-y-1 select-text">
                {filteredLogs.length === 0 ? (
                  <p className="text-gray-500 italic">No logs matched filter or no recent logs recorded.</p>
                ) : (
                  filteredLogs.map((line, idx) => {
                    const isError = line.includes("ERROR") || line.includes("Exception");
                    const isWarn = line.includes("WARNING");
                    const isInfo = line.includes("INFO");
                    const isSuccess = line.includes("successfully") || line.includes("Complete");

                    return (
                      <div
                        key={idx}
                        className={`leading-relaxed whitespace-pre-wrap ${
                          isError
                            ? "text-red-400 bg-red-950/30 px-1 rounded"
                            : isWarn
                            ? "text-amber-300"
                            : isSuccess
                            ? "text-emerald-300"
                            : isInfo
                            ? "text-gray-300"
                            : "text-gray-400"
                        }`}
                      >
                        {line}
                      </div>
                    );
                  })
                )}
                <div ref={logEndRef} />
              </div>
            </div>
          )}

        </div>

        {/* Footer */}
        <div className="px-6 py-3 border-t border-surfaceBorder bg-background/60 flex items-center justify-between text-xs text-gray-400 font-mono">
          <span>Status: <b className="text-emerald-400">System Healthy</b></span>
          <span>FastAPI Backend + Next.js 14 Web Stack</span>
        </div>

      </div>
    </div>
  );
};
