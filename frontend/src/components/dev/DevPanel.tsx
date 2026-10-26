"use client";

import React, { useState, useEffect, useRef } from "react";
import {
  Terminal,
  Cpu,
  Database,
  Layers,
  RefreshCw,
  X,
  Server,
  Zap,
  HardDrive,
  Copy,
  Check,
  Code2,
  ShieldCheck,
  Gauge,
  Globe,
  Sun,
  Moon,
  Sparkles,
  Sliders,
} from "lucide-react";
import { SystemTelemetry } from "@/lib/types";
import { apiClient } from "@/lib/api";
import { useLanguage } from "@/lib/language";

interface DevPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

export const DevPanel: React.FC<DevPanelProps> = ({ isOpen, onClose }) => {
  const [telemetry, setTelemetry] = useState<SystemTelemetry | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<"preferences" | "hardware" | "storage" | "models" | "logs">("preferences");
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [copiedLogs, setCopiedLogs] = useState(false);
  const [logFilter, setLogFilter] = useState("");
  const logEndRef = useRef<HTMLDivElement | null>(null);

  const { language, setLanguage, t } = useLanguage();
  const [isDark, setIsDark] = useState<boolean>(true);

  useEffect(() => {
    try {
      const isCurrentlyDark = document.documentElement.classList.contains("dark");
      setIsDark(isCurrentlyDark);
    } catch (_) {}
  }, []);

  const toggleTheme = (dark: boolean) => {
    setIsDark(dark);
    if (dark) {
      document.documentElement.classList.add("dark");
      localStorage.setItem("theme", "dark");
    } else {
      document.documentElement.classList.remove("dark");
      localStorage.setItem("theme", "light");
    }
  };

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

  const filteredLogs =
    telemetry?.recent_logs?.filter((l) =>
      l.toLowerCase().includes(logFilter.toLowerCase())
    ) || [];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 bg-black/60 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="w-full max-w-5xl h-[85vh] bg-[var(--panel)] border border-[var(--line)] rounded-2xl shadow-2xl flex flex-col overflow-hidden text-[var(--text)]">
        {/* Top Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-[var(--line)] bg-[var(--panel)]">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-xl bg-[var(--accent)]/15 border border-[var(--accent)]/30 flex items-center justify-center text-[var(--accent)]">
              <Sliders className="w-4 h-4" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-base font-bold text-[var(--text)] tracking-wide">
                  {t("settings.title")}
                </h2>
                <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded-full border border-emerald-500/40 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 flex items-center gap-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                  {t("settings.liveTelemetry")}
                </span>
              </div>
              <p className="text-xs text-[var(--muted)]">
                {t("settings.subtitle")}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {/* Quick Header Language Pill */}
            <div className="flex items-center rounded-xl border border-[var(--line)] bg-[var(--bg)] p-0.5">
              <button
                type="button"
                onClick={() => setLanguage("th")}
                className={`px-2.5 py-1 text-xs rounded-lg font-medium transition-all ${
                  language === "th"
                    ? "bg-[var(--panel)] text-[var(--accent)] font-bold shadow-sm"
                    : "text-[var(--muted)] hover:text-[var(--text)]"
                }`}
              >
                TH
              </button>
              <button
                type="button"
                onClick={() => setLanguage("en")}
                className={`px-2.5 py-1 text-xs rounded-lg font-medium transition-all ${
                  language === "en"
                    ? "bg-[var(--panel)] text-[var(--accent)] font-bold shadow-sm"
                    : "text-[var(--muted)] hover:text-[var(--text)]"
                }`}
              >
                EN
              </button>
            </div>

            <button
              type="button"
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={`text-xs px-3 py-1.5 rounded-xl border font-mono font-medium flex items-center gap-1.5 transition-all ${
                autoRefresh
                  ? "bg-[var(--accent)]/10 border-[var(--accent)]/40 text-[var(--accent)]"
                  : "bg-[var(--bg)] border-[var(--line)] text-[var(--muted)]"
              }`}
            >
              <RefreshCw className={`w-3.5 h-3.5 ${autoRefresh && isLoading ? "animate-spin" : ""}`} />
              <span>{autoRefresh ? t("settings.autoRefresh") : t("settings.paused")}</span>
            </button>

            <button
              type="button"
              onClick={fetchTelemetry}
              disabled={isLoading}
              className="p-2 text-[var(--muted)] hover:text-[var(--text)] rounded-xl hover:bg-[var(--line)]/50 transition-colors"
              title={t("settings.manualRefresh")}
            >
              <RefreshCw className={`w-4 h-4 ${isLoading ? "animate-spin" : ""}`} />
            </button>

            <button
              type="button"
              onClick={onClose}
              className="p-2 text-[var(--muted)] hover:text-[var(--text)] rounded-xl hover:bg-[var(--line)]/50 transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex items-center gap-1 px-6 pt-2 border-b border-[var(--line)] bg-[var(--panel)] overflow-x-auto">
          <button
            type="button"
            onClick={() => setActiveTab("preferences")}
            className={`flex items-center gap-2 px-4 py-2.5 text-xs font-semibold rounded-t-xl border-b-2 transition-all whitespace-nowrap ${
              activeTab === "preferences"
                ? "border-[var(--accent)] text-[var(--accent)] bg-[var(--bg)]"
                : "border-transparent text-[var(--muted)] hover:text-[var(--text)]"
            }`}
          >
            <Globe className="w-4 h-4" />
            <span>{t("settings.tab.preferences")}</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveTab("hardware")}
            className={`flex items-center gap-2 px-4 py-2.5 text-xs font-semibold rounded-t-xl border-b-2 transition-all whitespace-nowrap ${
              activeTab === "hardware"
                ? "border-[var(--accent)] text-[var(--accent)] bg-[var(--bg)]"
                : "border-transparent text-[var(--muted)] hover:text-[var(--text)]"
            }`}
          >
            <Cpu className="w-4 h-4" />
            <span>{t("settings.tab.hardware")}</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveTab("storage")}
            className={`flex items-center gap-2 px-4 py-2.5 text-xs font-semibold rounded-t-xl border-b-2 transition-all whitespace-nowrap ${
              activeTab === "storage"
                ? "border-[var(--accent)] text-[var(--accent)] bg-[var(--bg)]"
                : "border-transparent text-[var(--muted)] hover:text-[var(--text)]"
            }`}
          >
            <Database className="w-4 h-4" />
            <span>{t("settings.tab.storage")}</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveTab("models")}
            className={`flex items-center gap-2 px-4 py-2.5 text-xs font-semibold rounded-t-xl border-b-2 transition-all whitespace-nowrap ${
              activeTab === "models"
                ? "border-[var(--accent)] text-[var(--accent)] bg-[var(--bg)]"
                : "border-transparent text-[var(--muted)] hover:text-[var(--text)]"
            }`}
          >
            <Layers className="w-4 h-4" />
            <span>{t("settings.tab.models")}</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveTab("logs")}
            className={`flex items-center gap-2 px-4 py-2.5 text-xs font-semibold rounded-t-xl border-b-2 transition-all whitespace-nowrap ${
              activeTab === "logs"
                ? "border-[var(--accent)] text-[var(--accent)] bg-[var(--bg)]"
                : "border-transparent text-[var(--muted)] hover:text-[var(--text)]"
            }`}
          >
            <Terminal className="w-4 h-4" />
            <span>{t("settings.tab.logs")}</span>
          </button>
        </div>

        {/* Tab Contents */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6 bg-[var(--bg)]">
          {/* TAB 0: Preferences & Language */}
          {activeTab === "preferences" && (
            <div className="space-y-5 max-w-3xl">
              {/* Language Selection Card */}
              <div className="p-5 rounded-2xl bg-[var(--panel)] border border-[var(--line)] shadow-sm space-y-4">
                <div className="flex items-center gap-2.5">
                  <div className="w-8 h-8 rounded-xl bg-[var(--accent)]/15 text-[var(--accent)] flex items-center justify-center">
                    <Globe className="w-4 h-4" />
                  </div>
                  <div>
                    <h3 className="text-sm font-bold text-[var(--text)]">
                      {t("settings.languageCard.title")}
                    </h3>
                    <p className="text-xs text-[var(--muted)]">
                      {t("settings.languageCard.desc")}
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
                  <button
                    type="button"
                    onClick={() => setLanguage("th")}
                    className={`flex items-center justify-between p-4 rounded-xl border text-left transition-all ${
                      language === "th"
                        ? "border-[var(--accent)] bg-[var(--accent)]/10 ring-1 ring-[var(--accent)]"
                        : "border-[var(--line)] hover:border-[var(--line-strong)] bg-[var(--panel)]"
                    }`}
                  >
                    <div className="space-y-0.5">
                      <div className="flex items-center gap-2">
                        <span className="text-base">🇹🇭</span>
                        <span className="text-xs font-bold text-[var(--text)]">ภาษาไทย (TH)</span>
                      </div>
                      <p className="text-[11px] text-[var(--muted)]">
                        เมนูและการแจ้งเตือนภาษาไทย (คงคำศัพท์เทคนิคในโหมดอังกฤษ)
                      </p>
                    </div>
                    {language === "th" && <Check className="w-4 h-4 text-[var(--accent)]" />}
                  </button>

                  <button
                    type="button"
                    onClick={() => setLanguage("en")}
                    className={`flex items-center justify-between p-4 rounded-xl border text-left transition-all ${
                      language === "en"
                        ? "border-[var(--accent)] bg-[var(--accent)]/10 ring-1 ring-[var(--accent)]"
                        : "border-[var(--line)] hover:border-[var(--line-strong)] bg-[var(--panel)]"
                    }`}
                  >
                    <div className="space-y-0.5">
                      <div className="flex items-center gap-2">
                        <span className="text-base">🇺🇸</span>
                        <span className="text-xs font-bold text-[var(--text)]">English (EN)</span>
                      </div>
                      <p className="text-[11px] text-[var(--muted)]">
                        Full international English interface for all labels and telemetry
                      </p>
                    </div>
                    {language === "en" && <Check className="w-4 h-4 text-[var(--accent)]" />}
                  </button>
                </div>

                <div className="p-3 rounded-xl bg-[var(--bg)] border border-[var(--line)] text-xs text-[var(--muted)] leading-relaxed">
                  {language === "th" ? (
                    <span>
                      💡 <b>คำแนะนำ:</b> ในโหมดภาษาไทย ระบบแปลเฉพาะข้อความอินเทอร์เฟซทั่วไป ส่วนคำศัพท์เฉพาะทางคอมพิวเตอร์วิทัศน์ เช่น <i>SigLIP 2, CapRL Q6, Fast/Accurate, LanceDB, CUDA, Frame SSIM</i> จะยังคงเป็นภาษาอังกฤษเพื่อให้เข้าใจง่ายและสอดคล้องกับมาตรฐานทางเทคนิค
                    </span>
                  ) : (
                    <span>
                      💡 <b>Note:</b> In English mode, all application surfaces, buttons, tooltips, telemetry, and moment results are formatted entirely in English.
                    </span>
                  )}
                </div>
              </div>

              {/* Theme Preference Card */}
              <div className="p-5 rounded-2xl bg-[var(--panel)] border border-[var(--line)] shadow-sm space-y-4">
                <div className="flex items-center gap-2.5">
                  <div className="w-8 h-8 rounded-xl bg-amber-500/15 text-amber-500 flex items-center justify-center">
                    <Sun className="w-4 h-4" />
                  </div>
                  <div>
                    <h3 className="text-sm font-bold text-[var(--text)]">
                      {t("settings.themeCard.title")}
                    </h3>
                    <p className="text-xs text-[var(--muted)]">
                      {t("settings.themeCard.desc")}
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3 pt-1">
                  <button
                    type="button"
                    onClick={() => toggleTheme(false)}
                    className={`flex items-center justify-between p-3.5 rounded-xl border text-left transition-all ${
                      !isDark
                        ? "border-[var(--accent)] bg-[var(--accent)]/10 ring-1 ring-[var(--accent)] font-semibold"
                        : "border-[var(--line)] hover:border-[var(--line-strong)] bg-[var(--panel)]"
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <Sun className="w-4 h-4 text-amber-500" />
                      <span className="text-xs font-semibold text-[var(--text)]">Light Mode ({t("header.theme.light")})</span>
                    </div>
                    {!isDark && <Check className="w-4 h-4 text-[var(--accent)]" />}
                  </button>

                  <button
                    type="button"
                    onClick={() => toggleTheme(true)}
                    className={`flex items-center justify-between p-3.5 rounded-xl border text-left transition-all ${
                      isDark
                        ? "border-[var(--accent)] bg-[var(--accent)]/10 ring-1 ring-[var(--accent)] font-semibold"
                        : "border-[var(--line)] hover:border-[var(--line-strong)] bg-[var(--panel)]"
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <Moon className="w-4 h-4 text-cyan-400" />
                      <span className="text-xs font-semibold text-[var(--text)]">Dark Mode ({t("header.theme.dark")})</span>
                    </div>
                    {isDark && <Check className="w-4 h-4 text-[var(--accent)]" />}
                  </button>
                </div>
              </div>
            </div>
          )}
          {/* TAB 1: Hardware & GPU */}
          {activeTab === "hardware" && (
            <div className="space-y-6">
              {/* GPU Metric Card */}
              <div className="p-5 rounded-2xl bg-[var(--panel)] border border-[var(--line)] shadow-sm space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2.5">
                    <div className="w-8 h-8 rounded-xl bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 flex items-center justify-center">
                      <Zap className="w-4 h-4" />
                    </div>
                    <div>
                      <h3 className="text-sm font-bold text-[var(--text)]">NVIDIA CUDA GPU Engine</h3>
                      <p className="text-xs text-[var(--muted)]">Direct Tensor Core & FP16 hardware acceleration</p>
                    </div>
                  </div>
                  <span
                    className={`text-xs font-mono font-bold px-3 py-1 rounded-full border ${
                      telemetry?.gpu?.available
                        ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400"
                        : "border-red-500/40 bg-red-500/10 text-red-600 dark:text-red-400"
                    }`}
                  >
                    {telemetry?.gpu?.available ? "GPU Acceleration Active" : "CPU Fallback"}
                  </span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
                  <div className="p-3.5 rounded-xl bg-[var(--bg)] border border-[var(--line)]">
                    <span className="text-[11px] font-semibold text-[var(--muted)] uppercase tracking-wider block">
                      GPU Model
                    </span>
                    <span className="text-sm font-bold text-[var(--accent)] font-mono block mt-1">
                      {telemetry?.gpu?.device_name || "Detecting..."}
                    </span>
                  </div>

                  <div className="p-3.5 rounded-xl bg-[var(--bg)] border border-[var(--line)]">
                    <span className="text-[11px] font-semibold text-[var(--muted)] uppercase tracking-wider block">
                      VRAM in Use (Hardware)
                    </span>
                    <span className="text-sm font-bold text-[var(--text)] font-mono block mt-1">
                      {telemetry?.gpu?.allocated_vram_mb || 0} MB{" "}
                      <span className="text-xs text-[var(--muted)] font-normal">
                        / {telemetry?.gpu?.total_vram_mb || 0} MB
                      </span>
                    </span>
                  </div>

                  <div className="p-3.5 rounded-xl bg-[var(--bg)] border border-[var(--line)]">
                    <span className="text-[11px] font-semibold text-[var(--muted)] uppercase tracking-wider block">
                      Free VRAM
                    </span>
                    <span className="text-sm font-bold text-emerald-600 dark:text-emerald-400 font-mono block mt-1">
                      {Math.max(
                        0,
                        (telemetry?.gpu?.total_vram_mb || 0) - (telemetry?.gpu?.allocated_vram_mb || 0)
                      ).toFixed(1)}{" "}
                      MB
                    </span>
                  </div>

                  <div className="p-3.5 rounded-xl bg-[var(--bg)] border border-[var(--line)]">
                    <span className="text-[11px] font-semibold text-[var(--muted)] uppercase tracking-wider block">
                      Compute Capability
                    </span>
                    <span className="text-sm font-bold text-[var(--accent)] font-mono block mt-1">
                      {telemetry?.gpu?.compute_capability || "N/A"}
                    </span>
                  </div>
                </div>

                {/* VRAM Utilization Bar */}
                {telemetry?.gpu?.total_vram_mb && (
                  <div className="space-y-1.5 pt-1">
                    <div className="flex justify-between text-xs font-mono text-[var(--muted)]">
                      <span>Live VRAM Consumption</span>
                      <span className="text-[var(--accent)] font-bold">
                        {((telemetry.gpu.allocated_vram_mb / telemetry.gpu.total_vram_mb) * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full bg-[var(--line)] rounded-full h-2 overflow-hidden">
                      <div
                        className="bg-[var(--accent)] h-2 rounded-full transition-all duration-300"
                        style={{
                          width: `${Math.min(
                            100,
                            Math.max(1, (telemetry.gpu.allocated_vram_mb / telemetry.gpu.total_vram_mb) * 100)
                          )}%`,
                        }}
                      />
                    </div>
                  </div>
                )}
              </div>

              {/* Host CPU & RAM Card */}
              <div className="p-5 rounded-2xl bg-[var(--panel)] border border-[var(--line)] shadow-sm space-y-4">
                <div className="flex items-center gap-2.5">
                  <div className="w-8 h-8 rounded-xl bg-[var(--accent)]/15 text-[var(--accent)] flex items-center justify-center">
                    <Server className="w-4 h-4" />
                  </div>
                  <div>
                    <h3 className="text-sm font-bold text-[var(--text)]">Host System Resource Monitor</h3>
                    <p className="text-xs text-[var(--muted)]">CPU multi-threading and system memory telemetry</p>
                  </div>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
                  <div className="p-3.5 rounded-xl bg-[var(--bg)] border border-[var(--line)]">
                    <span className="text-[11px] font-semibold text-[var(--muted)] uppercase tracking-wider block">
                      CPU Usage
                    </span>
                    <span className="text-sm font-bold text-[var(--text)] font-mono block mt-1">
                      {telemetry?.system?.cpu_percent || 0}%
                    </span>
                  </div>

                  <div className="p-3.5 rounded-xl bg-[var(--bg)] border border-[var(--line)]">
                    <span className="text-[11px] font-semibold text-[var(--muted)] uppercase tracking-wider block">
                      CPU Cores
                    </span>
                    <span className="text-sm font-bold text-[var(--text)] font-mono block mt-1">
                      {telemetry?.system?.cpu_count_physical} Physical / {telemetry?.system?.cpu_count_logical} Logical
                    </span>
                  </div>

                  <div className="p-3.5 rounded-xl bg-[var(--bg)] border border-[var(--line)]">
                    <span className="text-[11px] font-semibold text-[var(--muted)] uppercase tracking-wider block">
                      System RAM
                    </span>
                    <span className="text-sm font-bold text-[var(--text)] font-mono block mt-1">
                      {telemetry?.system?.ram_used_gb || 0} GB{" "}
                      <span className="text-xs text-[var(--muted)] font-normal">
                        / {telemetry?.system?.ram_total_gb || 0} GB ({telemetry?.system?.ram_percent}%)
                      </span>
                    </span>
                  </div>

                  <div className="p-3.5 rounded-xl bg-[var(--bg)] border border-[var(--line)]">
                    <span className="text-[11px] font-semibold text-[var(--muted)] uppercase tracking-wider block">
                      Environment
                    </span>
                    <span className="text-sm font-bold text-[var(--text)] font-mono block mt-1 truncate">
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
              <div className="p-5 rounded-2xl bg-[var(--panel)] border border-[var(--line)] shadow-sm space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2.5">
                    <div className="w-8 h-8 rounded-xl bg-amber-500/15 text-amber-600 dark:text-amber-400 flex items-center justify-center">
                      <Database className="w-4 h-4" />
                    </div>
                    <div>
                      <h3 className="text-sm font-bold text-[var(--text)]">LanceDB Serverless Vector Database</h3>
                      <p className="text-xs text-[var(--muted)]">Zero-copy PyArrow & Disk-based IVF-PQ ANN indexing</p>
                    </div>
                  </div>
                  <span className="text-xs font-mono font-bold px-3 py-1 rounded-full border border-amber-500/40 bg-amber-500/10 text-amber-600 dark:text-amber-400">
                    {telemetry?.lancedb?.total_records || 0} Total Records Indexed
                  </span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                  {Object.entries(telemetry?.lancedb?.tables || {}).map(([tbl, count]) => (
                    <div
                      key={tbl}
                      className="p-3.5 rounded-xl bg-[var(--bg)] border border-[var(--line)] flex items-center justify-between"
                    >
                      <div>
                        <span className="text-xs font-mono font-bold text-[var(--text)]">table_{tbl}</span>
                        <p className="text-[11px] text-[var(--muted)]">
                          {tbl === "videos" && "Video metadata catalog"}
                          {tbl === "video_frames" && "768-dim SigLIP 2 visual vectors"}
                          {tbl === "scenes" && "Temporal boundary cuts"}
                          {tbl === "search_logs" && "Query latency & evaluation logs"}
                        </p>
                      </div>
                      <span className="text-sm font-mono font-bold text-[var(--accent)] bg-[var(--panel)] px-2.5 py-1 rounded-lg border border-[var(--line)]">
                        {count}
                      </span>
                    </div>
                  ))}
                </div>

                <div className="p-3 rounded-xl bg-[var(--bg)] border border-[var(--line)] text-xs text-[var(--muted)] flex items-center gap-2 font-mono">
                  <HardDrive className="w-4 h-4 text-[var(--muted)] flex-shrink-0" />
                  <span className="truncate">Storage Location: {telemetry?.lancedb?.storage_path}</span>
                </div>
              </div>
            </div>
          )}

          {/* TAB 3: AI Models */}
          {activeTab === "models" && (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {/* SigLIP 2 */}
              <div className="p-5 rounded-2xl bg-[var(--panel)] border border-[var(--line)] shadow-sm space-y-3">
                <div className="flex items-center gap-2.5">
                  <div className="w-8 h-8 rounded-xl bg-[var(--accent)]/15 text-[var(--accent)] flex items-center justify-center">
                    <Gauge className="w-4 h-4" />
                  </div>
                  <div>
                    <h4 className="text-sm font-bold text-[var(--text)]">Vision-Text: SigLIP 2 (NaFlex)</h4>
                    <p className="text-xs text-[var(--muted)]">Zero-Shot Joint Visual Embedding</p>
                  </div>
                </div>
                <div className="p-3 rounded-xl bg-[var(--bg)] border border-[var(--line)] space-y-1.5 text-xs font-mono">
                  <p className="text-[var(--muted)]">
                    Model ID: <span className="text-[var(--accent)] font-semibold">google/siglip2-base-patch16-naflex</span>
                  </p>
                  <p className="text-[var(--muted)]">
                    Vector Dimension: <span className="text-emerald-600 dark:text-emerald-400 font-semibold">768-dim</span>
                  </p>
                  <p className="text-[var(--muted)]">
                    Distance Metric: <span className="text-[var(--text)] font-semibold">Cosine Similarity</span>
                  </p>
                </div>
              </div>

              {/* CapRL 4B Q6 */}
              <div className="p-5 rounded-2xl bg-[var(--panel)] border border-[var(--line)] shadow-sm space-y-3">
                <div className="flex items-center gap-2.5">
                  <div className="w-8 h-8 rounded-xl bg-blue-500/15 text-blue-600 dark:text-blue-400 flex items-center justify-center">
                    <Code2 className="w-4 h-4" />
                  </div>
                  <div>
                    <h4 className="text-sm font-bold text-[var(--text)]">Dense Captioner: CapRL 4B Q6</h4>
                    <p className="text-xs text-[var(--muted)]">Scene captioning and Accurate verification with CapRL Q6</p>
                  </div>
                </div>
                <div className="p-3 rounded-xl bg-[var(--bg)] border border-[var(--line)] space-y-1.5 text-xs font-mono">
                  <p className="text-[var(--muted)]">
                    Model ID: <span className="text-[var(--accent)] font-semibold">internlm/CapRL-Qwen3VL-4B-GGUF</span>
                  </p>
                  <p className="text-[var(--muted)]">
                    Quantization: <span className="text-emerald-600 dark:text-emerald-400 font-semibold">Q6_K (llama.cpp CUDA)</span>
                  </p>
                  <p className="text-[var(--muted)]">
                    Execution: <span className="text-[var(--text)] font-semibold">Background Worker</span>
                  </p>
                </div>
              </div>

              {/* Local inference worker */}
              <div className="p-5 rounded-2xl bg-[var(--panel)] border border-[var(--line)] shadow-sm space-y-3">
                <div className="flex items-center gap-2.5">
                  <div className="w-8 h-8 rounded-xl bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 flex items-center justify-center">
                    <Server className="w-4 h-4" />
                  </div>
                  <div>
                    <h4 className="text-sm font-bold text-[var(--text)]">Local Inference Worker</h4>
                    <p className="text-xs text-[var(--muted)]">GPU queue, residency and cache boundary</p>
                  </div>
                </div>
                <div className="p-3 rounded-xl bg-[var(--bg)] border border-[var(--line)] space-y-1.5 text-xs font-mono">
                  <p className="text-[var(--muted)]">
                    Status: <span className="text-emerald-600 dark:text-emerald-400 font-semibold">{telemetry?.inference_worker?.status || "unknown"}</span>
                  </p>
                  <p className="text-[var(--muted)]">
                    Queue: <span className="text-[var(--accent)] font-semibold">{telemetry?.inference_worker?.models?.queue_depth ?? "—"}</span>
                  </p>
                  <p className="text-[var(--muted)]">
                    VRAM budget: <span className="text-amber-600 dark:text-amber-400 font-semibold">{telemetry?.inference_worker?.models?.budget_mb ?? "—"} MB</span>
                  </p>
                </div>
              </div>

              {/* Temporal Boundary Localizer */}
              <div className="p-5 rounded-2xl bg-[var(--panel)] border border-[var(--line)] shadow-sm space-y-3">
                <div className="flex items-center gap-2.5">
                  <div className="w-8 h-8 rounded-xl bg-amber-500/15 text-amber-600 dark:text-amber-400 flex items-center justify-center">
                    <ShieldCheck className="w-4 h-4" />
                  </div>
                  <div>
                    <h4 className="text-sm font-bold text-[var(--text)]">Boundary Localizer: Multi-Scale Proposals</h4>
                    <p className="text-xs text-[var(--muted)]">Multi-scale proposals + calibrated temporal ranking</p>
                  </div>
                </div>
                <div className="p-3 rounded-xl bg-[var(--bg)] border border-[var(--line)] space-y-1.5 text-xs font-mono">
                  <p className="text-[var(--muted)]">
                    Algorithm: <span className="text-amber-600 dark:text-amber-400 font-semibold">Visual proposal ranking</span>
                  </p>
                  <p className="text-[var(--muted)]">
                    NMS Suppress: <span className="text-emerald-600 dark:text-emerald-400 font-semibold">IoU threshold = 0.5</span>
                  </p>
                  <p className="text-[var(--muted)]">
                    Top-K Selection: <span className="text-[var(--text)] font-semibold">K = 5 Moments</span>
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* TAB 4: Live Backend Logs */}
          {activeTab === "logs" && (
            <div className="space-y-3">
              <div className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-2 flex-1 max-w-sm">
                  <input
                    type="text"
                    placeholder="Filter logs (e.g. CUDA, ERROR, SigLIP)..."
                    value={logFilter}
                    onChange={(e) => setLogFilter(e.target.value)}
                    className="w-full bg-[var(--panel)] border border-[var(--line)] rounded-xl px-3 py-2 text-xs text-[var(--text)] placeholder-[var(--muted)] focus:outline-none focus:border-[var(--accent)] font-mono"
                  />
                </div>

                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={handleCopyLogs}
                    className="flex items-center gap-1.5 px-3 py-2 rounded-xl border border-[var(--line)] bg-[var(--panel)] hover:bg-[var(--line)]/50 text-xs font-semibold text-[var(--text)] transition-colors"
                  >
                    {copiedLogs ? <Check className="w-3.5 h-3.5 text-emerald-500" /> : <Copy className="w-3.5 h-3.5" />}
                    <span>{copiedLogs ? "Copied!" : "Copy Logs"}</span>
                  </button>
                </div>
              </div>

              <div className="bg-[var(--panel)] border border-[var(--line)] rounded-2xl p-4 font-mono text-[11px] h-96 overflow-y-auto space-y-1 select-text shadow-inner">
                {filteredLogs.length === 0 ? (
                  <p className="text-[var(--muted)] italic">No logs matched filter or no recent logs recorded.</p>
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
                            ? "text-red-600 dark:text-red-400 bg-red-500/10 px-1.5 py-0.5 rounded"
                            : isWarn
                            ? "text-amber-600 dark:text-amber-400"
                            : isSuccess
                            ? "text-emerald-600 dark:text-emerald-400"
                            : isInfo
                            ? "text-[var(--text)]"
                            : "text-[var(--muted)]"
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
        <div className="px-6 py-3 border-t border-[var(--line)] bg-[var(--panel)] flex items-center justify-between text-xs text-[var(--muted)] font-mono">
          <span>
            Status: <b className="text-emerald-600 dark:text-emerald-400">System Healthy</b>
          </span>
          <span>FastAPI Backend + Next.js 14 Web Stack</span>
        </div>
      </div>
    </div>
  );
};
