"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import NormalityPanel from "./NormalityPanel";
import { AppHeader } from "@/components/layout/AppHeader";
import { DevPanel } from "@/components/dev/DevPanel";
import { ShortcutModal } from "@/components/ui/ShortcutModal";
import {
  Check,
  X,
  Search,
  Upload,
  Play,
  Activity,
  Loader2,
  Volume2,
  AlertTriangle,
  ShieldAlert,
  Clock,
  Film,
  Sparkles,
  ChevronRight,
  Info,
  CheckCircle2,
  XCircle,
  HelpCircle,
  Eye,
  SlidersHorizontal,
} from "lucide-react";
import type { MomentItem, SearchResponse } from "@/lib/types";

const BASE = "";
const API = `${BASE}/api/v1`;

const kinds = {
  pushing_punching: { label: "ผลัก / ชกต่อย", icon: "🥊", desc: "การใช้ความรุนแรง ทะเลาะวิวาท ผลักกัน" },
  falling: { label: "ล้มลงกับพื้น", icon: "⚠️", desc: "คนเสียหลักล้ม สะดุด หรือหมดสติ" },
};
type Kind = keyof typeof kinds;

type Video = {
  id: string;
  name: string;
  duration: number;
  has_audio: boolean;
  caption_count: number;
  index_status: string;
  completed_windows: number;
  total_windows: number;
};

type Job = {
  id: string;
  kind: string;
  video_id: string;
  status: string;
  phase: string;
  completed: number;
  total: number;
  error: string;
  telemetry?: Record<string, unknown>;
  qwen_calls?: number;
  qwen_baseline_calls?: number;
  mode?: string;
  profile_id?: string;
};

type Reference = { video_id: string; window_id: number; t_start: number; t_end: number };

type Event = {
  id: string;
  label: string;
  kind: Kind;
  t_start: number;
  t_end: number;
  context_t_start: number;
  context_t_end: number;
  score?: number;
  model_status: string;
  review: string;
  assessment?: string;
  reference_status?: string;
  references?: Reference[];
  preview_frame_path: string;
  caption_full: string;
  evidence: unknown;
  audio_status: string;
  threshold_status: string;
  verification?: {
    verdict?: { event_present?: boolean | null; reason?: string };
    error?: string;
    audio_verified?: boolean;
    caption_status?: string;
    verification_attempts?: unknown[];
    caption_attempts?: unknown[];
  };
  refinement_status: string;
  verification_frames?: { timestamp: number; path: string }[];
};

type Selection = { t_start: number; t_end: number; context_t_start?: number | null; context_t_end?: number | null };

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(API + path, options);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail || data));
  }
  return data;
}

const post = <T,>(path: string, data: unknown) =>
  request<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });

const clock = (value: number) => {
  const m = Math.floor(value / 60).toString().padStart(2, "0");
  const s = (value % 60).toFixed(1).padStart(4, "0");
  return `${m}:${s}`;
};

const visualDecision = (event: Event, compact = false) => {
  if (event.assessment) {
    const map: Record<string, string> = {
      normal: "ปกติตามบริบท",
      suspicious: "พบสิ่งน่าสงสัย",
      uncertain: "หลักฐานยังไม่พอ",
    };
    return map[event.assessment] || event.assessment;
  }
  if (event.verification?.verdict?.event_present === true) {
    return "ตรวจพบพฤติกรรมจากภาพจริง";
  }
  if (event.verification?.verdict?.event_present === false) {
    return "Qwen ไม่พบการกระทำจากภาพ";
  }
  return compact ? "ยังไม่มีข้อสรุป" : "ยังไม่มีข้อสรุปจากภาพ";
};

const isModelNegative = (event: Event) => event.verification?.verdict?.event_present === false;

export default function EventsPage() {
  const [videos, setVideos] = useState<Video[]>([]);
  const [videoId, setVideoId] = useState("");
  const [jobs, setJobs] = useState<Job[]>([]);
  const [events, setEvents] = useState<Event[]>([]);
  const [eventFilter, setEventFilter] = useState("all");
  const [screeningFilter, setScreeningFilter] = useState<"active" | "negative" | "all">("active");
  const [isCinemaMode, setIsCinemaMode] = useState(false);
  const [isHotkeysOpen, setIsHotkeysOpen] = useState(false);
  const [isDevOpen, setIsDevOpen] = useState(false);

  const [screenMode, setScreenMode] = useState<"specific" | "open_anomaly">("specific");
  const [bankRefresh, setBankRefresh] = useState(0);
  const [cameraProfile, setCameraProfile] = useState("");
  const [selectedKinds, setSelectedKinds] = useState<Kind[]>(["pushing_punching", "falling"]);
  const [query, setQuery] = useState("");
  const [profile, setProfile] = useState<"fast" | "accurate">("fast");
  const [selection, setSelection] = useState<Selection | null>(null);
  const [selectedEventId, setSelectedEventId] = useState<string | null>(null);
  const selectedEvent = events.find((event) => event.id === selectedEventId) || null;

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [connectionError, setConnectionError] = useState("");
  const [health, setHealth] = useState<Record<string, unknown> | null>(null);
  const [reviewNote, setReviewNote] = useState("");
  const [showTelemetry, setShowTelemetry] = useState(false);

  const player = useRef<HTMLVideoElement>(null);
  const currentVideoId = useRef(videoId);
  currentVideoId.current = videoId;
  const video = videos.find((item) => item.id === videoId);

  const activeJob = jobs.find(
    (job) => job.video_id === videoId && ["queued", "running", "cancelling"].includes(job.status)
  );

  const refresh = useCallback(async () => {
    try {
      const [nextVideos, nextJobs, nextHealth] = await Promise.all([
        request<Video[]>("/events/videos"),
        request<Job[]>("/events/jobs"),
        request<Record<string, unknown>>("/events/health").catch(() => null),
      ]);
      setVideos(nextVideos);
      setJobs(nextJobs.reverse());
      if (nextHealth) {
        setHealth(nextHealth);
      }
      if (videoId) {
        const nextEvents = await request<Event[]>(`/events/list/${videoId}`);
        if (currentVideoId.current === videoId) {
          setEvents(nextEvents.filter((ev) => (ev.kind as string) !== "shouting"));
        }
      }
      setConnectionError("");
    } catch (err: unknown) {
      setConnectionError(err instanceof Error ? err.message : String(err));
    }
  }, [videoId]);

  useEffect(() => {
    let active = true;
    const run = () => refresh().catch((err) => active && setConnectionError(err.message));
    run();
    const timer = setInterval(run, 3000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, [refresh]);

  useEffect(() => {
    if (!videoId && videos.length) setVideoId(videos[0].id);
  }, [videos, videoId]);

  useEffect(() => {
    setSelection(null);
    setSelectedEventId(null);
    setEvents([]);
    setReviewNote("");
  }, [videoId]);

  useEffect(() => {
    if (player.current && selection) {
      player.current.currentTime = selection.context_t_start ?? selection.t_start;
    }
  }, [selection]);

  async function action(operation: () => Promise<void>) {
    setBusy(true);
    setError("");
    try {
      await operation();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function start(kind: "index" | "screen", resume?: Job) {
    await post<Job>("/events/jobs", {
      video_id: videoId,
      kind,
      event_types: selectedKinds,
      resume_id: resume?.id,
      mode: kind === "index" ? "specific" : resume?.mode || screenMode,
      profile_id: resume?.profile_id || cameraProfile || null,
    });
    await refresh();
  }

  function selectEvent(event: Event) {
    setSelectedEventId(event.id);
    setSelection(event);
    setReviewNote("");
  }

  async function review(event: Event, decision: string) {
    const updated = await post<Event>(`/events/${event.id}/review`, { decision, note: reviewNote });
    setEvents((current) => current.map((item) => (item.id === updated.id ? updated : item)));
    await refresh();
  }

  // Filtered lists
  const visibleEvents = events.filter((event) => {
    const negative = isModelNegative(event);
    if (screeningFilter === "active" && negative) return false;
    if (screeningFilter === "negative" && !negative) return false;
    if (eventFilter === "all") return true;
    if (eventFilter === "pending") return event.review === "pending";
    if (eventFilter === "confirmed") return event.review === "confirmed";
    if (eventFilter === "rejected") return event.review === "rejected";
    return true;
  });

  const confirmedCount = events.filter((e) => e.review === "confirmed").length;
  const rejectedCount = events.filter((e) => e.review === "rejected").length;
  const pendingCount = events.filter((e) => e.review === "pending").length;

  return (
    <div className="min-h-screen bg-[var(--bg)] flex flex-col font-sans transition-colors duration-200">
      <AppHeader
        isCinemaMode={isCinemaMode}
        onToggleCinemaMode={() => setIsCinemaMode((v) => !v)}
        onOpenHotkeys={() => setIsHotkeysOpen(true)}
        onOpenSettings={() => setIsDevOpen(true)}
      />

      <main className="flex-1 w-full max-w-[1680px] mx-auto p-4 sm:p-6 space-y-6">
        {/* Page Title & Breadcrumb */}
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="flex h-2 w-2 rounded-full bg-amber-500 animate-pulse" />
              <p className="text-[11px] font-mono tracking-widest uppercase font-semibold text-[var(--accent)]">
                AI CONTEXT-AWARE SCREENING & HUMAN REVIEW
              </p>
            </div>
            <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-[var(--text)] mt-1 flex items-center gap-2.5">
              <ShieldAlert className="h-6 w-6 text-amber-500" />
              ระบบตรวจจับและคัดกรองเหตุการณ์ผิดปกติ (Anomaly Screening)
            </h1>
            <p className="text-xs sm:text-sm text-[var(--text-muted)] mt-1">
              สแกนค้นหาพฤติกรรมเสี่ยง (ผลัก/ชกต่อย, ล้มลงกับพื้น) และคัดกรองความผิดปกติเชิงบริบทในระดับวินาที
            </p>
          </div>

          <Link
            href="/"
            className="flex items-center gap-2 rounded-xl border border-[var(--line)] bg-[var(--card-bg)] px-3.5 py-2 text-xs font-medium text-[var(--text)] hover:border-[var(--line-strong)] hover:bg-[var(--card-hover)] transition-all shadow-sm"
          >
            ← กลับหน้าค้นหาโมเมนต์
          </Link>
        </div>

        {/* Global Error Banner */}
        {(error || connectionError) && (
          <div
            role="alert"
            className="rounded-xl bg-red-500/10 border border-red-500/30 p-4 text-xs sm:text-sm text-red-500 flex items-start gap-3"
          >
            <AlertTriangle className="h-5 w-5 shrink-0 mt-0.5" />
            <div className="whitespace-pre-wrap">{error || connectionError}</div>
          </div>
        )}

        {/* Control Toolbar Card */}
        <section className="app-panel rounded-2xl p-4 sm:p-5 space-y-4 shadow-sm">
          <div className="flex flex-wrap gap-3 items-center justify-between">
            {/* Video Selector */}
            <div className="flex flex-wrap items-center gap-3 flex-1 min-w-[280px]">
              <div className="relative flex-1 min-w-[220px]">
                <select
                  aria-label="เลือกวิดีโอ"
                  value={videoId}
                  onChange={(e) => setVideoId(e.target.value)}
                  className="w-full rounded-xl bg-[var(--bg)] border border-[var(--line)] px-3.5 py-2.5 text-xs sm:text-sm text-[var(--text)] font-medium focus:border-[var(--accent)] focus:outline-none transition-colors"
                >
                  <option value="">เลือกวิดีโอที่ต้องการตรวจสอบ...</option>
                  {videos.map((item) => (
                    <option key={item.id} value={item.id}>
                      🎬 {item.name} ({clock(item.duration)})
                    </option>
                  ))}
                </select>
              </div>

              {/* Upload Video Button */}
              <label
                className={`flex items-center gap-1.5 rounded-xl border border-[var(--line)] bg-[var(--card-bg)] hover:bg-[var(--card-hover)] px-3.5 py-2.5 text-xs font-semibold text-[var(--text)] cursor-pointer transition-all shadow-sm ${
                  busy ? "opacity-50 cursor-not-allowed" : ""
                }`}
              >
                <Upload className="h-4 w-4 text-[var(--accent)]" />
                <span>+ เพิ่มวิดีโอใหม่</span>
                <input
                  aria-label="อัปโหลดวิดีโอ"
                  type="file"
                  accept="video/*"
                  className="sr-only"
                  disabled={busy}
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) {
                      void action(async () => {
                        const body = new FormData();
                        body.append("file", file);
                        const record = await request<Video>("/events/videos/upload", { method: "POST", body });
                        await refresh();
                        setVideoId(record.id);
                      });
                    }
                  }}
                />
              </label>
            </div>

            {/* Screening Mode Selector */}
            <div className="flex items-center gap-2 bg-[var(--bg)] p-1 rounded-xl border border-[var(--line)]">
              <button
                type="button"
                onClick={() => setScreenMode("specific")}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                  screenMode === "specific"
                    ? "bg-[var(--panel)] text-[var(--accent)] shadow-sm font-semibold"
                    : "text-[var(--text-muted)] hover:text-[var(--text)]"
                }`}
              >
                🎯 เหตุการณ์จำเพาะ
              </button>
              <button
                type="button"
                onClick={() => setScreenMode("open_anomaly")}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                  screenMode === "open_anomaly"
                    ? "bg-[var(--panel)] text-[var(--accent)] shadow-sm font-semibold"
                    : "text-[var(--text-muted)] hover:text-[var(--text)]"
                }`}
              >
                🌐 ความผิดปกติทั่วไป
              </button>
            </div>
          </div>

          {/* Video Metadata Pill Bar */}
          {video && (
            <div className="flex flex-wrap items-center gap-2 text-xs text-[var(--text-muted)] pt-1">
              <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-[var(--bg)] border border-[var(--line)]">
                <Clock className="h-3.5 w-3.5 text-[var(--accent)]" />
                ความยาว: {clock(video.duration)}
              </span>
              <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-[var(--bg)] border border-[var(--line)]">
                <Volume2 className="h-3.5 w-3.5 text-[var(--accent)]" />
                {video.has_audio ? "มีสัญญาณเสียง" : "ไม่มีเสียง (Pure-Visual)"}
              </span>
              <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-[var(--bg)] border border-[var(--line)]">
                <Film className="h-3.5 w-3.5 text-[var(--accent)]" />
                คำบรรยายภาพ {video.caption_count} ฉาก
              </span>
            </div>
          )}

          {/* Anomaly Category Pills & Start Action */}
          <div className="flex flex-wrap items-center justify-between gap-4 pt-2 border-t border-[var(--line)]">
            {screenMode === "specific" ? (
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-xs font-semibold text-[var(--text)] mr-1">หมวดหมู่ตรวจจับ:</span>
                {(Object.entries(kinds) as [Kind, { label: string; icon: string; desc: string }][]).map(
                  ([kind, item]) => {
                    const isChecked = selectedKinds.includes(kind);
                    return (
                      <button
                        type="button"
                        key={kind}
                        onClick={() =>
                          setSelectedKinds((cur) =>
                            cur.includes(kind) ? cur.filter((k) => k !== kind) : [...cur, kind]
                          )
                        }
                        className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold transition-all border ${
                          isChecked
                            ? "bg-[var(--accent-subtle)] border-[var(--accent)] text-[var(--accent)] shadow-sm"
                            : "bg-[var(--card-bg)] border-[var(--line)] text-[var(--text-muted)] hover:border-[var(--line-strong)]"
                        }`}
                      >
                        <span>{item.icon}</span>
                        <span>{item.label}</span>
                      </button>
                    );
                  }
                )}
              </div>
            ) : (
              <p className="text-xs text-[var(--text-muted)]">
                โหมด Open-Category: สแกนความผิดปกติทุกประเภทโดยเทียบกับบริบทสถานที่
              </p>
            )}

            {/* Start Screening Action */}
            <button
              disabled={!videoId || !!activeJob || busy || (screenMode === "specific" && !selectedKinds.length)}
              onClick={() => void action(() => start("screen"))}
              className="inline-flex items-center gap-2 rounded-xl bg-[var(--accent)] text-[var(--accent-text)] px-4 py-2.5 text-xs sm:text-sm font-semibold hover:bg-[var(--accent-hover)] disabled:opacity-40 transition-all shadow-sm ml-auto"
            >
              {activeJob ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>กำลังคัดกรองเหตุการณ์...</span>
                </>
              ) : (
                <>
                  <Activity className="h-4 w-4" />
                  <span>เริ่มคัดกรองเหตุการณ์</span>
                </>
              )}
            </button>
          </div>

          {/* Normality Panel for Open Anomaly */}
          {screenMode === "open_anomaly" && (
            <NormalityPanel
              key={bankRefresh}
              videoId={videoId}
              ready={video?.index_status === "ready"}
              busy={busy || !!activeJob}
              profileId={cameraProfile}
              onProfile={setCameraProfile}
            />
          )}

          {/* Running Job Progress Bar */}
          {activeJob && (
            <div className="bg-[var(--accent-subtle)] border border-[var(--line)] rounded-xl p-3 text-xs space-y-2 animate-in fade-in">
              <div className="flex justify-between items-center text-[var(--text)] font-medium">
                <span className="flex items-center gap-2">
                  <Loader2 className="h-3.5 w-3.5 animate-spin text-[var(--accent)]" />
                  สถานะการทำงาน: {activeJob.phase} ({activeJob.completed}/{activeJob.total || 1})
                </span>
                <button
                  onClick={() =>
                    void action(async () => {
                      await post(`/events/jobs/${activeJob.id}/cancel`, {});
                      await refresh();
                    })
                  }
                  className="text-xs text-rose-500 hover:underline font-semibold"
                >
                  ยกเลิกงาน
                </button>
              </div>
              <div className="w-full h-1.5 bg-[var(--line)] rounded-full overflow-hidden">
                <div
                  className="h-full bg-[var(--accent)] transition-all duration-300"
                  style={{
                    width: `${Math.round(((activeJob.completed || 0) / (activeJob.total || 1)) * 100)}%`,
                  }}
                />
              </div>
            </div>
          )}
        </section>

        {/* Free-text Anomaly Search Bar */}
        <form
          className="app-panel rounded-2xl p-2.5 sm:p-3 flex flex-wrap gap-2.5 items-center shadow-sm"
          onSubmit={(e) => {
            e.preventDefault();
            if (!query.trim() || !videoId) return;
            void action(async () => {
              const found = await post<SearchResponse>("/search/moment", {
                video_id: videoId,
                query,
                profile,
                top_k: 5,
              });
              if (found.moments && found.moments[0]) {
                setSelection(found.moments[0]);
              }
            });
          }}
        >
          <div className="relative flex-1 min-w-[240px]">
            <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 h-4 w-4 text-[var(--text-muted)]" />
            <input
              aria-label="คำค้นหาความผิดปกติ"
              placeholder="พิมพ์ค้นหาเหตุการณ์เฉพาะ เช่น 'คนวิ่งหนีแตกตื่น', 'ไฟไหม้ / ควัน', 'คนปีนข้ามรั้ว'..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="w-full rounded-xl bg-[var(--bg)] border border-[var(--line)] pl-10 pr-3.5 py-2.5 text-xs sm:text-sm text-[var(--text)] focus:border-[var(--accent)] focus:outline-none transition-colors"
            />
          </div>

          <button
            type="submit"
            disabled={busy || !query.trim() || !videoId}
            className="inline-flex items-center gap-1.5 rounded-xl bg-[var(--card-bg)] border border-[var(--line)] hover:border-[var(--line-strong)] px-4 py-2.5 text-xs font-semibold text-[var(--text)] disabled:opacity-40 transition-all shadow-sm"
          >
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4 text-[var(--accent)]" />}
            <span>ค้นหาเฉพาะจุด</span>
          </button>
        </form>

        {/* Studio Grid Layout (Matching Main Page) */}
        <div className={`grid gap-6 items-start ${isCinemaMode ? "grid-cols-1" : "lg:grid-cols-[1.3fr_1fr]"}`}>
          {/* Left Column: Video Studio & Evidence Review */}
          <section className="space-y-4">
            {/* HTML5 Video Player Container */}
            <div className="app-panel rounded-2xl border border-[var(--line)] p-3 overflow-hidden shadow-sm">
              {videoId ? (
                <div className="space-y-2">
                  <video
                    ref={player}
                    key={videoId}
                    src={`${API}/events/videos/${videoId}/stream`}
                    controls
                    preload="metadata"
                    className="w-full aspect-video rounded-xl bg-black object-contain shadow-inner"
                    onLoadedMetadata={() => {
                      if (selection && player.current) {
                        player.current.currentTime = selection.context_t_start ?? selection.t_start;
                      }
                    }}
                    onTimeUpdate={() => {
                      if (
                        player.current &&
                        selection &&
                        player.current.currentTime > (selection.context_t_end ?? selection.t_end)
                      ) {
                        // Pause when exceeding window to focus user
                        player.current.pause();
                      }
                    }}
                  />

                  {/* Playback Controls & Time Indicator */}
                  {selection ? (
                    <div className="flex items-center justify-between text-xs text-[var(--text)] bg-[var(--bg)] p-2.5 rounded-xl border border-[var(--line)]">
                      <div className="flex items-center gap-2">
                        <span className="flex h-2 w-2 rounded-full bg-amber-500 animate-pulse" />
                        <span className="font-semibold text-[var(--accent)] font-mono">
                          [{clock(selection.t_start)} – {clock(selection.t_end)}]
                        </span>
                        <span className="text-[var(--text-muted)] hidden sm:inline">
                          (รวมบริบทก่อน–หลัง 3 วินาที)
                        </span>
                      </div>
                      <button
                        type="button"
                        onClick={() => {
                          if (player.current) {
                            player.current.currentTime = selection.context_t_start ?? selection.t_start;
                            void player.current.play();
                          }
                        }}
                        className="inline-flex items-center gap-1.5 text-xs font-semibold text-[var(--accent)] hover:underline"
                      >
                        <Play className="h-3.5 w-3.5 fill-current" />
                        เล่นช่วงนี้ซ้ำ
                      </button>
                    </div>
                  ) : (
                    <div className="text-center py-2 text-xs text-[var(--text-muted)]">
                      คลิกการ์ดเหตุการณ์ทางขวาเพื่อ Seek ไปยังวินาทีที่ตรวจพบ
                    </div>
                  )}
                </div>
              ) : (
                <div className="aspect-video grid place-items-center text-center p-6 text-[var(--text-muted)] bg-[var(--bg)] rounded-xl">
                  <div>
                    <Film className="h-10 w-10 mx-auto text-[var(--text-muted)] opacity-40 mb-2" />
                    <p className="text-sm font-semibold text-[var(--text)]">ยังไม่ได้เลือกวิดีโอ</p>
                    <p className="text-xs">กรุณาเลือกหรืออัปโหลดวิดีโอด้านบนเพื่อเริ่มการตรวจสอบ</p>
                  </div>
                </div>
              )}
            </div>

            {/* Active Selected Event: Full Review & Evidence Panel */}
            {selectedEvent ? (
              <div className="app-panel rounded-2xl border border-[var(--line)] p-5 space-y-4 shadow-sm animate-in fade-in">
                <div className="flex items-start justify-between gap-3 border-b border-[var(--line)] pb-3">
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-lg">
                        {kinds[selectedEvent.kind]?.icon || "⚠️"}
                      </span>
                      <h2 className="text-base font-bold text-[var(--text)]">
                        {selectedEvent.label}
                      </h2>
                      <span
                        className={`text-[11px] font-semibold px-2.5 py-0.5 rounded-full border ${
                          selectedEvent.review === "confirmed"
                            ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-500"
                            : selectedEvent.review === "rejected"
                            ? "bg-rose-500/10 border-rose-500/30 text-rose-500"
                            : "bg-amber-500/10 border-amber-500/30 text-amber-500"
                        }`}
                      >
                        {selectedEvent.review === "confirmed"
                          ? "ยืนยันเหตุแล้ว"
                          : selectedEvent.review === "rejected"
                          ? "ปฏิเสธเหตุ (ปกติ)"
                          : "รอเจ้าหน้าที่ตรวจยืนยัน"}
                      </span>
                    </div>
                    <p className="text-xs text-[var(--text-muted)] mt-1 font-mono">
                      ช่วงเวลา: {clock(selectedEvent.t_start)} – {clock(selectedEvent.t_end)} (ความยาว{" "}
                      {(selectedEvent.t_end - selectedEvent.t_start).toFixed(1)} วินาที)
                    </p>
                  </div>

                  {selectedEvent.score != null && (
                    <div className="text-right">
                      <div className="text-xs text-[var(--text-muted)]">ความมั่นใจ AI</div>
                      <div className="text-base font-bold text-[var(--accent)] font-mono">
                        {(selectedEvent.score * 100).toFixed(1)}%
                      </div>
                    </div>
                  )}
                </div>

                {/* Visual Verdict */}
                <div className="rounded-xl bg-[var(--accent-subtle)] border border-[var(--line)] p-3 text-xs space-y-1">
                  <div className="font-semibold text-[var(--accent)] flex items-center justify-between gap-1.5">
                    <span className="flex items-center gap-1.5">
                      <Sparkles className="h-4 w-4" />
                      <span>ข้อสรุปจาก AI:</span>
                      <span>{visualDecision(selectedEvent)}</span>
                    </span>
                    {selectedEvent.model_status === "pending" && (
                      <span className="text-[11px] font-medium text-amber-500">รอ Qwen ตรวจภาพ</span>
                    )}
                  </div>
                  {selectedEvent.verification?.verdict?.reason && (
                    <p className="text-[var(--text-muted)] leading-relaxed">
                      {selectedEvent.verification.verdict.reason}
                    </p>
                  )}
                  {selectedEvent.verification?.caption_status === "not_requested_negative" && (
                    <p className="text-[11px] text-[var(--text-muted)]">
                      ไม่พบเหตุการณ์ จึงไม่สร้าง Caption เพิ่ม
                    </p>
                  )}
                </div>

                {/* Keyframe Evidence Grid */}
                {selectedEvent.verification_frames && selectedEvent.verification_frames.length > 0 && (
                  <div>
                    <h3 className="text-xs font-bold text-[var(--text)] mb-2 flex items-center gap-1.5">
                      <Eye className="h-3.5 w-3.5 text-[var(--accent)]" />
                      ภาพหลักฐานเชิงประจักษ์ (Ground Truth Frames)
                    </h3>
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                      {selectedEvent.verification_frames.map((frame, slot) => (
                        <figure key={slot} className="rounded-lg overflow-hidden border border-[var(--line)] bg-[var(--bg)]">
                          <img
                            src={`${API}/events/${selectedEvent.id}/frames/${slot}`}
                            alt={`Frame ${clock(frame.timestamp)}`}
                            className="w-full aspect-video object-cover"
                          />
                          <figcaption className="text-[10px] text-[var(--text-muted)] text-center py-1 font-mono">
                            {clock(frame.timestamp)}
                          </figcaption>
                        </figure>
                      ))}
                    </div>
                  </div>
                )}

                {/* Human Review Form */}
                <div className="space-y-3 pt-3 border-t border-[var(--line)]">
                  <label className="text-xs font-semibold text-[var(--text)] block">
                    บันทึกผลการตรวจสอบโดยมนุษย์ (Human Audit Log):
                  </label>
                  <textarea
                    aria-label="หมายเหตุการตรวจ"
                    placeholder="พิมพ์ข้อสังเกต เช่น 'ยืนยันเหตุการณ์ทะเลาะวิวาทจริง แจ้ง รปภ. ตรวจสอบแล้ว' หรือ 'เป็นมุมกล้องหลอกตา คนก้มเก็บของตามปกติ'..."
                    value={reviewNote}
                    onChange={(e) => setReviewNote(e.target.value)}
                    className="w-full bg-[var(--bg)] border border-[var(--line)] rounded-xl p-3 text-xs text-[var(--text)] focus:border-[var(--accent)] focus:outline-none min-h-[70px]"
                  />
                  <div className="flex flex-wrap gap-2.5">
                    <button
                      type="button"
                      disabled={busy}
                      onClick={() => void action(() => review(selectedEvent, "confirmed"))}
                      className="inline-flex items-center gap-1.5 bg-emerald-600 hover:bg-emerald-700 text-white rounded-xl px-4 py-2.5 text-xs font-semibold transition-all shadow-sm"
                    >
                      <Check className="h-4 w-4" />
                      ยืนยันเป็นเหตุการณ์ผิดปกติ (Confirmed)
                    </button>
                    <button
                      type="button"
                      disabled={busy}
                      onClick={() => void action(() => review(selectedEvent, "rejected"))}
                      className="inline-flex items-center gap-1.5 bg-rose-600 hover:bg-rose-700 text-white rounded-xl px-4 py-2.5 text-xs font-semibold transition-all shadow-sm"
                    >
                      <X className="h-4 w-4" />
                      ปฏิเสธ / เป็นเหตุการณ์ปกติ (Rejected)
                    </button>
                    <button
                      type="button"
                      disabled={busy}
                      onClick={() => void action(() => review(selectedEvent, "pending"))}
                      className="inline-flex items-center gap-1.5 border border-[var(--line)] bg-[var(--card-bg)] hover:bg-[var(--card-hover)] text-[var(--text)] rounded-xl px-3.5 py-2.5 text-xs font-medium transition-all"
                    >
                      รอตรวจสอบเพิ่มเติม
                    </button>
                  </div>
                </div>
              </div>
            ) : null}
          </section>

          {/* Right Column: Detected Anomalies List */}
          <section className="space-y-4">
            <div className="app-panel rounded-2xl border border-[var(--line)] p-4 sm:p-5 space-y-4 shadow-sm">
              <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[var(--line)] pb-3">
                <div>
                  <h2 className="text-base font-bold text-[var(--text)] flex items-center gap-2">
                    <span>รายการเหตุการณ์ที่ตรวจพบ</span>
                    <span className="text-xs font-mono font-bold px-2 py-0.5 rounded-full bg-[var(--accent-subtle)] text-[var(--accent)]">
                      {events.length}
                    </span>
                  </h2>
                  <p className="text-xs text-[var(--text-muted)] mt-0.5">
                    จัดอันดับตามระดับความรุนแรงและความมั่นใจของ AI
                  </p>
                  {events.length > 0 && (
                    <p className="text-[11px] text-amber-500 mt-0.5">
                      หมายเหตุ: จำนวนนี้ไม่ใช่จำนวนเหตุจริง (ผลคัดกรองเบื้องต้นต้องรอตรวจยืนยัน)
                    </p>
                  )}
                </div>
                <select
                  aria-label="กรองผลคัดกรอง"
                  value={screeningFilter}
                  onChange={(e) => setScreeningFilter(e.target.value as "active" | "negative" | "all")}
                  className="rounded-lg bg-[var(--bg)] border border-[var(--line)] px-2.5 py-1.5 text-xs text-[var(--text)] focus:border-[var(--accent)] focus:outline-none"
                >
                  <option value="active">ผลคัดกรองที่ต้องตรวจ</option>
                  <option value="negative">ที่ Qwen ตัดออก</option>
                  <option value="all">ทั้งหมด</option>
                </select>
              </div>

              {/* Status Filter Badges */}
              <div className="flex items-center gap-1.5 bg-[var(--bg)] p-1 rounded-xl border border-[var(--line)]">
                <button
                  type="button"
                  onClick={() => setEventFilter("all")}
                  className={`flex-1 py-1.5 text-xs font-medium rounded-lg transition-all ${
                    eventFilter === "all"
                      ? "bg-[var(--panel)] text-[var(--text)] font-semibold shadow-sm"
                      : "text-[var(--text-muted)] hover:text-[var(--text)]"
                  }`}
                >
                  ทั้งหมด ({events.length})
                </button>
                <button
                  type="button"
                  onClick={() => setEventFilter("pending")}
                  className={`flex-1 py-1.5 text-xs font-medium rounded-lg transition-all ${
                    eventFilter === "pending"
                      ? "bg-[var(--panel)] text-amber-500 font-semibold shadow-sm"
                      : "text-[var(--text-muted)] hover:text-[var(--text)]"
                  }`}
                >
                  รอตรวจ ({pendingCount})
                </button>
                <button
                  type="button"
                  onClick={() => setEventFilter("confirmed")}
                  className={`flex-1 py-1.5 text-xs font-medium rounded-lg transition-all ${
                    eventFilter === "confirmed"
                      ? "bg-[var(--panel)] text-emerald-500 font-semibold shadow-sm"
                      : "text-[var(--text-muted)] hover:text-[var(--text)]"
                  }`}
                >
                  ยืนยัน ({confirmedCount})
                </button>
                <button
                  type="button"
                  onClick={() => setEventFilter("rejected")}
                  className={`flex-1 py-1.5 text-xs font-medium rounded-lg transition-all ${
                    eventFilter === "rejected"
                      ? "bg-[var(--panel)] text-rose-500 font-semibold shadow-sm"
                      : "text-[var(--text-muted)] hover:text-[var(--text)]"
                  }`}
                >
                  ปฏิเสธ ({rejectedCount})
                </button>
              </div>

              {/* Events List */}
              <div className="space-y-2.5 max-h-[640px] overflow-y-auto pr-1">
                {visibleEvents.length === 0 ? (
                  <div className="text-center py-12 px-4 text-[var(--text-muted)] bg-[var(--bg)] rounded-xl border border-[var(--line)]">
                    <HelpCircle className="h-8 w-8 mx-auto opacity-40 mb-2" />
                    <p className="text-xs sm:text-sm font-semibold text-[var(--text)]">ยังไม่พบรายการเหตุการณ์</p>
                    <p className="text-xs mt-1">
                      คลิกปุ่ม &quot;เริ่มคัดกรองเหตุการณ์&quot; ด้านบนเพื่อสแกนวิดีโอนี้
                    </p>
                  </div>
                ) : (
                  visibleEvents.map((event) => {
                    const isSelected = selectedEvent?.id === event.id;
                    const itemKind = kinds[event.kind];
                    return (
                      <button
                        type="button"
                        key={event.id}
                        aria-label={`${event.label} ${event.review} ${clock(event.t_start)}–${clock(event.t_end)}`}
                        onClick={() => selectEvent(event)}
                        className={`w-full text-left rounded-xl p-3 border transition-all cursor-pointer ${
                          isSelected
                            ? "bg-[var(--accent-subtle)] border-[var(--accent)] shadow-sm"
                            : "app-card border-[var(--line)] hover:border-[var(--line-strong)]"
                        }`}
                      >
                        <div className="flex items-center justify-between gap-2 mb-2">
                          <div className="flex items-center gap-1.5">
                            <span className="text-base">{itemKind?.icon || "⚠️"}</span>
                            <span className="text-xs font-bold text-[var(--text)]">{event.label}</span>
                          </div>
                          <span
                            className={`text-[10px] font-semibold px-2 py-0.5 rounded-full border ${
                              event.review === "confirmed"
                                ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-500"
                                : event.review === "rejected"
                                ? "bg-rose-500/10 border-rose-500/30 text-rose-500"
                                : "bg-amber-500/10 border-amber-500/30 text-amber-500"
                            }`}
                          >
                            {event.review === "confirmed"
                              ? "ยืนยันแล้ว"
                              : event.review === "rejected"
                              ? "ปฏิเสธแล้ว"
                              : "รอตรวจ"}
                          </span>
                        </div>

                        {/* Event Thumbnail Preview */}
                        <div className="relative rounded-lg overflow-hidden border border-[var(--line)] aspect-video mb-2 bg-black">
                          <img
                            src={`${API}/events/${event.id}/preview`}
                            alt={event.label}
                            loading="lazy"
                            className="w-full h-full object-cover"
                            onError={(e) => {
                              (e.target as HTMLElement).style.opacity = "0";
                            }}
                          />
                          <div className="absolute bottom-1 right-1 bg-black/70 px-1.5 py-0.5 rounded text-[10px] text-white font-mono pointer-events-none">
                            {clock(event.t_start)} – {clock(event.t_end)}
                          </div>
                        </div>

                        <div className="flex items-center justify-between text-xs text-[var(--text-muted)] mt-1">
                          <span className="text-[11px] font-mono text-[var(--accent)] font-semibold">
                            ⏱️ {clock(event.t_start)} – {clock(event.t_end)}
                          </span>
                          {event.score != null && (
                            <span className="text-[11px] font-bold text-[var(--text)]">
                              {(event.score * 100).toFixed(0)}% Match
                            </span>
                          )}
                        </div>

                        <p className="text-[11px] text-[var(--text-muted)] mt-1 line-clamp-1">
                          {visualDecision(event, true)}
                        </p>
                      </button>
                    );
                  })
                )}
              </div>
            </div>
          </section>
        </div>

        {/* Developer Telemetry Drawer (Cleanly Hidden at Bottom) */}
        <section className="app-panel rounded-2xl border border-[var(--line)] p-4 text-xs">
          <details
            open={showTelemetry}
            onToggle={(e) => setShowTelemetry((e.target as HTMLDetailsElement).open)}
            className="group"
          >
            <summary className="cursor-pointer font-semibold text-[var(--text-muted)] hover:text-[var(--text)] flex items-center justify-between">
              <span className="flex items-center gap-1.5">
                <SlidersHorizontal className="h-4 w-4 text-[var(--accent)]" />
                สถานะบริการและทรัพยากรระบบ (System & Inference Telemetry)
              </span>
              <span className="text-[11px]">{showTelemetry ? "ซ่อน" : "แสดงรายละเอียด"}</span>
            </summary>

            <div className="pt-3 mt-3 border-t border-[var(--line)] space-y-3">
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
                <div className="p-2.5 rounded-xl bg-[var(--bg)] border border-[var(--line)]">
                  <p className="text-[10px] text-[var(--text-muted)]">Backend Service</p>
                  <p className="text-xs font-bold text-[var(--text)] mt-0.5">
                    {typeof health?.backend === "string" ? health.backend : "FastAPI"}
                  </p>
                </div>
                <div className="p-2.5 rounded-xl bg-[var(--bg)] border border-[var(--line)]">
                  <p className="text-[10px] text-[var(--text-muted)]">Gemma2 Worker</p>
                  <p
                    className={`text-xs font-bold mt-0.5 ${
                      (health?.worker as { status?: string } | undefined)?.status === "ok"
                        ? "text-emerald-500"
                        : "text-amber-500"
                    }`}
                  >
                    {String((health?.worker as { status?: string } | undefined)?.status || "fallback/unknown")}
                  </p>
                </div>
                <div className="p-2.5 rounded-xl bg-[var(--bg)] border border-[var(--line)]">
                  <p className="text-[10px] text-[var(--text-muted)]">Verified Events</p>
                  <p className="text-xs font-bold text-[var(--accent)] mt-0.5">
                    {confirmedCount} / {events.length}
                  </p>
                </div>
                <div className="p-2.5 rounded-xl bg-[var(--bg)] border border-[var(--line)]">
                  <p className="text-[10px] text-[var(--text-muted)]">API Status</p>
                  <p
                    className={`text-xs font-bold mt-0.5 ${
                      connectionError ? "text-rose-500" : health ? "text-emerald-500" : "text-[var(--text-muted)]"
                    }`}
                  >
                    {connectionError ? "Disconnected" : health ? "Connected" : "Checking..."}
                  </p>
                </div>
              </div>

              {health && (
                <details className="text-[11px] text-[var(--text-muted)]">
                  <summary className="cursor-pointer hover:underline">ข้อมูล JSON สดจาก /api/v1/events/health</summary>
                  <pre className="mt-2 p-3 bg-[var(--bg)] rounded-xl border border-[var(--line)] overflow-x-auto text-[10px] font-mono">
                    {JSON.stringify(health, null, 2)}
                  </pre>
                </details>
              )}
            </div>
          </details>
        </section>
      </main>
      <DevPanel isOpen={isDevOpen} onClose={() => setIsDevOpen(false)} />
      <ShortcutModal isOpen={isHotkeysOpen} onClose={() => setIsHotkeysOpen(false)} />
    </div>
  );
}
