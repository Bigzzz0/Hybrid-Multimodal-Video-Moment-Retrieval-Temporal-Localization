"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import NormalityPanel from "./NormalityPanel";
import { AppHeader } from "@/components/layout/AppHeader";
import { Check, X, Search, Upload, Play, Activity, Loader2, Volume2 } from "lucide-react";
import type { MomentItem, SearchResponse } from "@/lib/types";

// Keep browser requests on the page's origin; Next forwards them to the local API.
const BASE = "";
const API = `${BASE}/api/v1`;
const kinds = { shouting: "ตะโกน / กรีดร้อง", pushing_punching: "ผลัก / ชก", falling: "ล้ม" };
type Kind = keyof typeof kinds;
type Video = { id: string; name: string; duration: number; has_audio: boolean; caption_count: number; index_status: string; completed_windows: number; total_windows: number };
type Job = { id: string; kind: string; video_id: string; status: string; phase: string; completed: number; total: number; error: string; telemetry?: Record<string, unknown>; qwen_calls?: number; qwen_baseline_calls?: number; mode?: string; profile_id?: string };
type Reference = { video_id: string; window_id: number; t_start: number; t_end: number };
type Event = { assessment?: string; reference_status?: string; references?: Reference[]; id: string; label: string; kind: Kind; t_start: number; t_end: number; context_t_start: number; context_t_end: number; model_status: string; review: string; preview_frame_path: string; caption_full: string; evidence: unknown; audio_status: string; threshold_status: string; verification?: { verdict?: { event_present?: boolean | null; reason?: string }; error?: string; audio_verified?: boolean; caption_status?: string; verification_attempts?: unknown[]; caption_attempts?: unknown[] }; refinement_status: string; verification_frames?: { timestamp: number }[] };
type Selection = { t_start: number; t_end: number; context_t_start?: number | null; context_t_end?: number | null };

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(API + path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail || data));
  return data;
}
const post = <T,>(path: string, data: unknown) => request<T>(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
const clock = (value: number) => `${Math.floor(value / 60).toString().padStart(2, "0")}:${(value % 60).toFixed(1).padStart(4, "0")}`;
const visualDecision = (event: Event, compact = false) => event.assessment
  ? ({ normal: "ปกติตามบริบท", suspicious: "พบสิ่งน่าสงสัย", uncertain: "หลักฐานยังไม่พอ" } as Record<string, string>)[event.assessment]
  : event.verification?.verdict?.event_present === true ? "Qwen พบการกระทำจากภาพ"
  : event.verification?.verdict?.event_present === false ? "Qwen ไม่พบการกระทำจากภาพ" : compact ? "ยังไม่มีข้อสรุป" : "ยังไม่มีข้อสรุปจากภาพ";


export default function EventsPage() {
  const [videos, setVideos] = useState<Video[]>([]);
  const [videoId, setVideoId] = useState("");
  const [jobs, setJobs] = useState<Job[]>([]);
  const [events, setEvents] = useState<Event[]>([]);
  const [eventFilter, setEventFilter] = useState("reviewable");
  const positiveUnits = events.filter(event => event.verification?.verdict?.event_present === true);
  const negativeUnits = events.filter(event => event.verification?.verdict?.event_present === false);
  const undecidedUnits = events.filter(event => event.verification?.verdict?.event_present == null);
  const visibleEvents = events.filter(event => eventFilter === "all"
    || (eventFilter === "positive" && event.verification?.verdict?.event_present === true)
    || (eventFilter === "negative" && event.verification?.verdict?.event_present === false)
    || (eventFilter === "undecided" && event.verification?.verdict?.event_present == null)
    || (eventFilter === "reviewable" && event.verification?.verdict?.event_present !== false));
  const [screenMode, setScreenMode] = useState("specific");
  const [bankRefresh, setBankRefresh] = useState(0);
  const [cameraProfile, setCameraProfile] = useState("");
  const [selectedKinds, setSelectedKinds] = useState<Kind[]>(Object.keys(kinds) as Kind[]);
  const [query, setQuery] = useState("");
  const [profile, setProfile] = useState<"fast" | "accurate">("fast");
  const [result, setResult] = useState<SearchResponse | null>(null);
  const [selection, setSelection] = useState<Selection | null>(null);
  const [selectedEventId, setSelectedEventId] = useState<string | null>(null);
  const selectedEvent = events.find(event => event.id === selectedEventId) || null;
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [connectionError, setConnectionError] = useState("");
  const [health, setHealth] = useState<Record<string, unknown> | null>(null);
  const [reviewNote, setReviewNote] = useState("");
  const player = useRef<HTMLVideoElement>(null);
  const currentVideoId = useRef(videoId);
  currentVideoId.current = videoId;
  const video = videos.find(item => item.id === videoId);
  const worker = (health?.worker || {}) as Record<string, unknown>;
  const qwenJob = jobs.find(job => job.video_id === videoId && job.qwen_calls != null);
  const qwenResources = (qwenJob?.telemetry?.qwen_resources || {}) as Record<string, unknown>;
  const gib = (value: unknown) => typeof value === "number" ? `${(value / 1024 ** 3).toFixed(2)} GiB` : "—";
  const activeJob = jobs.find(job => job.video_id === videoId && ["queued", "running", "cancelling"].includes(job.status));

  const refresh = useCallback(async () => {
    const [nextVideos, nextJobs] = await Promise.all([request<Video[]>("/events/videos"), request<Job[]>("/events/jobs")]);
    setVideos(nextVideos); setJobs(nextJobs.reverse());
    if (videoId) {
      const nextEvents = await request<Event[]>(`/events/list/${videoId}`);
      if (currentVideoId.current === videoId) setEvents(nextEvents);
    }
    setConnectionError("");
  }, [videoId]);
  useEffect(() => {
    let active = true;
    const run = () => refresh().catch(err => active && setConnectionError(err.message));
    run();
    const timer = setInterval(run, 3000);
    return () => { active = false; clearInterval(timer); };
  }, [refresh]);
  useEffect(() => {
    if (!videoId && videos.length) setVideoId(videos[0].id);
  }, [videos, videoId]);
  useEffect(() => {
    setResult(null); setSelection(null); setSelectedEventId(null); setEvents([]); setReviewNote("");
  }, [videoId]);
  useEffect(() => {
    const timer = setInterval(() => request<Record<string, unknown>>("/events/health").then(setHealth).catch(() => {}), 10000);
    request<Record<string, unknown>>("/events/health").then(setHealth).catch(() => {});
    return () => clearInterval(timer);
  }, []);
  useEffect(() => {
    if (player.current && selection) player.current.currentTime = selection.context_t_start ?? selection.t_start;
  }, [selection]);

  async function action(operation: () => Promise<void>) {
    setBusy(true); setError("");
    try { await operation(); } catch (err) { setError(err instanceof Error ? err.message : String(err)); } finally { setBusy(false); }
  }
  async function start(kind: "index" | "screen", resume?: Job) {
    await post<Job>("/events/jobs", { video_id: videoId, kind, event_types: selectedKinds, resume_id: resume?.id, mode: kind === "index" ? "specific" : resume?.mode || screenMode, profile_id: resume?.profile_id || cameraProfile || null });
    await refresh();
  }
  function selectEvent(event: Event) { setSelectedEventId(event.id); setSelection(event); setReviewNote(""); }
  async function review(event: Event, decision: string) {
    const updated = await post<Event>(`/events/${event.id}/review`, { decision, note: reviewNote });
    setEvents(current => current.map(item => item.id === updated.id ? updated : item)); await refresh();
  }

  return (
    <div className="min-h-screen bg-[var(--bg)] flex flex-col font-sans transition-colors duration-200">
      <AppHeader
        isCinemaMode={false}
        onToggleCinemaMode={() => {}}
        onOpenHotkeys={() => {}}
        onOpenSettings={() => {}}
      />
      <main className="flex-1 w-full max-w-[1680px] mx-auto p-4 sm:p-6 space-y-6">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-cyan-400 text-xs font-mono tracking-widest uppercase">RECORDED VIDEO · HUMAN REVIEW</p>
            <h1 className="text-2xl sm:text-3xl font-semibold mt-1">ค้นหาและตรวจเหตุการณ์ผิดปกติ (Anomaly Screening)</h1>
            <p className="text-slate-400 text-sm mt-1">EmbeddingGemma 2 ค้นจากภาพ วิดีโอ เสียง และ Caption • Qwen ตรวจภาพของเหตุที่ผ่านคัดกรอง</p>
          </div>
          <Link href="/" className="rounded-xl border border-[var(--line)] bg-[var(--card-bg)] px-4 py-2 text-sm text-[var(--text)] hover:border-[var(--line-strong)] transition-colors">
            ← กลับหน้าค้นหาโมเมนต์
          </Link>
        </div>
    <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-4 text-sm text-amber-200">ระบบทดลอง: เกณฑ์คัดกรองยังไม่ได้ผ่านชุด validation คะแนนความคล้ายไม่ใช่ความน่าจะเป็น กรุณาตรวจหลักฐานก่อนยืนยันเหตุ</div>
    {(error || connectionError) && <div role="alert" className="rounded-xl bg-red-500/15 border border-red-500/40 p-4 text-red-200 whitespace-pre-wrap">{error || connectionError}</div>}
    <section className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5 space-y-4">
      <div className="flex flex-wrap gap-3 items-center">
        <select aria-label="เลือกวิดีโอ" className="flex-1 min-w-52 rounded-xl bg-slate-950 border border-slate-700 p-3" value={videoId} onChange={e => setVideoId(e.target.value)}><option value="">เลือกวิดีโอที่บันทึกไว้</option>{videos.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}</select>
        <label className={`rounded-xl border border-slate-700 px-4 py-3 flex gap-2 cursor-pointer ${busy ? "opacity-50" : ""}`}><Upload size={18} /> เพิ่มวิดีโอ<input aria-label="อัปโหลดวิดีโอ" type="file" accept="video/*" className="sr-only" disabled={busy} onChange={e => { const file = e.target.files?.[0]; if (file) void action(async () => { const body = new FormData(); body.append("file", file); const record = await request<Video>("/events/videos/upload", { method: "POST", body }); await refresh(); setVideoId(record.id); }); }} /></label>
        <button disabled={!videoId || busy || !!activeJob} onClick={() => void action(() => start("index"))} className="rounded-xl bg-cyan-600 px-4 py-3 disabled:opacity-40">สร้าง / ทำต่อดัชนี</button>
      </div>
      {video && <p className="text-sm text-slate-400">{clock(video.duration)} · {video.has_audio ? "มีเสียง" : "ไม่มีหลักฐานเสียง"} · Caption {video.caption_count} รายการ · ดัชนี {video.index_status} ({video.completed_windows}/{video.total_windows} ช่วง)</p>}
      <div className="flex gap-3 items-center"><label htmlFor="screen-mode">วิธีคัดกรอง</label><select id="screen-mode" value={screenMode} onChange={e => setScreenMode(e.target.value)} className="rounded-xl bg-slate-950 p-3"><option value="specific">เหตุที่กำหนดประเภท</option><option value="open_anomaly">ความผิดปกติทั่วไป · ทดลอง</option></select></div>
      {screenMode === "open_anomaly" && <NormalityPanel key={bankRefresh} videoId={videoId} ready={video?.index_status === "ready"} busy={busy || !!activeJob} profileId={cameraProfile} onProfile={setCameraProfile} />}
      <div className="flex flex-wrap gap-4 items-center">{screenMode === "specific" && (Object.entries(kinds) as [Kind, string][]).map(([kind, label]) => <label key={kind} className="flex gap-2 items-center"><input type="checkbox" checked={selectedKinds.includes(kind)} onChange={e => setSelectedKinds(current => e.target.checked ? [...current, kind] : current.filter(item => item !== kind))} />{label}</label>)}<button disabled={video?.index_status !== "ready" || !!activeJob || busy || (screenMode === "specific" ? !selectedKinds.length : !cameraProfile)} className="ml-auto rounded-xl bg-indigo-600 px-4 py-3 disabled:opacity-40" onClick={() => void action(() => start("screen"))}><Activity className="inline mr-2" size={17} />เริ่มคัดกรองเหตุ</button></div>
      {jobs.filter(job => job.video_id === videoId).slice(0, 3).map(job => <div key={job.id} className="bg-slate-950 rounded-xl p-3 text-sm space-y-2"><div className="flex justify-between gap-2"><span>{job.kind} · {job.phase} · {job.status} {job.completed}/{job.total}</span>{["queued", "running", "cancelling"].includes(job.status) ? <button onClick={() => void action(async () => { await post(`/events/jobs/${job.id}/cancel`, {}); await refresh(); })}>ยกเลิก</button> : ["failed", "cancelled", "interrupted"].includes(job.status) && <button disabled={busy || !!activeJob} onClick={() => void action(() => start(job.kind as "index" | "screen", job))}>ทำต่อ</button>}</div><progress aria-label="ความคืบหน้า" className="w-full h-2 accent-cyan-400" value={job.completed} max={job.total || 1} />{job.error && <p className="text-red-300">{job.error}</p>}{job.qwen_calls != null && <p className="text-slate-400">คำขอตรวจ Qwen: {job.qwen_calls} · baseline สมมติ {job.qwen_baseline_calls} (ไม่ได้รันเปรียบเทียบ)</p>}</div>)}
    </section>
    <form className="flex flex-wrap gap-3" onSubmit={e => { e.preventDefault(); void action(async () => { const found = await post<SearchResponse>("/search/moment", { video_id: videoId, query, profile, backend: "gemma2", top_k: 5 }); setResult(found); setSelectedEventId(null); if (found.moments[0]) setSelection(found.moments[0]); }); }}>
      <input aria-label="คำค้น" placeholder="เช่น คนล้มลงบนพื้น หรือ เสียงคนตะโกน" value={query} onChange={e => setQuery(e.target.value)} className="flex-1 min-w-52 rounded-xl bg-slate-900 border border-slate-700 p-3" />
      <select aria-label="โหมดค้นหา" value={profile} onChange={e => setProfile(e.target.value as "fast" | "accurate")} className="rounded-xl bg-slate-900 border border-slate-700 p-3"><option value="fast">Fast · ดัชนีที่เก็บไว้</option><option value="accurate">Accurate · ตรวจภาพและเสียงเพิ่ม</option></select><button aria-label="ค้นหาช่วงเวลา" disabled={busy || !query.trim() || video?.index_status !== "ready" || !!activeJob} className="rounded-xl bg-cyan-600 px-5 py-3 disabled:opacity-40">{busy ? <Loader2 className="animate-spin" /> : <Search />}</button>
    </form>
    <div className="grid lg:grid-cols-[1.3fr_1fr] gap-5">
      <section className="space-y-4">
        <div className="rounded-2xl bg-slate-950 border border-slate-800 p-3">{videoId ? <video ref={player} key={videoId} src={`${API}/events/videos/${videoId}/stream`} controls preload="metadata" className="w-full aspect-video rounded-xl" onLoadedMetadata={() => { if (selection && player.current) player.current.currentTime = selection.context_t_start ?? selection.t_start; }} onTimeUpdate={() => { if (player.current && selection && player.current.currentTime > (selection.context_t_end ?? selection.t_end)) player.current.pause(); }} /> : <div className="aspect-video grid place-items-center text-slate-500">เลือกหรือเพิ่มวิดีโอ</div>}{selection && <div className="flex justify-between text-sm p-3"><span>เหตุ {clock(selection.t_start)} – {clock(selection.t_end)} · ดูก่อน–หลังอย่างละ 5 วินาที</span><button onClick={() => { if (player.current) { player.current.currentTime = selection.context_t_start ?? selection.t_start; void player.current.play(); } }}><Play className="inline" size={16} /> เล่นช่วงนี้</button></div>}</div>
        {selectedEvent && <div className="rounded-2xl border border-slate-800 bg-slate-900 p-5 space-y-4"><h2 className="text-xl">{selectedEvent.label}</h2>{selectedEvent.assessment && <p className="text-amber-200">ผลตรวจบริบท: {({ normal: "ปกติ", suspicious: "น่าสงสัย", uncertain: "หลักฐานไม่พอ" } as Record<string, string>)[selectedEvent.assessment]} · {selectedEvent.reference_status}</p>}{selectedEvent.references?.map((ref, i) => <figure key={i}><figcaption className="text-sm text-cyan-300">ตัวอย่างปกติที่ใกล้เคียง · {clock(ref.t_start)} – {clock(ref.t_end)}</figcaption><video controls preload="metadata" src={`${API}/events/videos/${ref.video_id}/stream#t=${ref.t_start},${ref.t_end}`} className="w-full aspect-video rounded-lg" /></figure>)}<p className="text-cyan-200">{visualDecision(selectedEvent)}</p><p className="text-sm text-slate-400">ผลตรวจภาพ: {selectedEvent.model_status} · ขอบเขต: {selectedEvent.refinement_status} · ผู้ใช้: {selectedEvent.review}</p>{selectedEvent.audio_status === "available" ? <div><p className="mb-2 text-sm"><Volume2 className="inline mr-2" size={16} />เสียงตรงเหตุและบริบท (Qwen ไม่ได้ยืนยันเสียง)</p><audio controls key={selectedEvent.id} src={`${API}/events/${selectedEvent.id}/audio`} className="w-full" /></div> : <p>คลิปนี้ไม่มีหลักฐานเสียง</p>}<div className="grid grid-cols-2 gap-3">{selectedEvent.verification_frames?.map((frame, slot) => <figure key={slot}><img src={`${API}/events/${selectedEvent.id}/frames/${slot}`} alt={`ภาพที่ Qwen ตรวจเวลา ${clock(frame.timestamp)}`} className="w-full aspect-video object-contain rounded-lg" /><figcaption className="text-xs text-slate-400 mt-1">{clock(frame.timestamp)}</figcaption></figure>)}</div><details open><summary className="cursor-pointer text-cyan-300">Caption และผลตรวจภาพฉบับเต็ม</summary><pre className="mt-3 whitespace-pre-wrap break-words text-sm bg-slate-950 rounded-xl p-4">{selectedEvent.caption_full || (selectedEvent.verification?.caption_status === "not_requested_negative" ? "Qwen ไม่พบการกระทำ จึงไม่สร้าง Caption เพิ่ม อ่านเหตุผลฉบับเต็มด้านล่าง" : selectedEvent.verification?.caption_status === "failed" ? "สร้าง Caption ไม่สำเร็จ แต่ผลตรวจการกระทำยังอยู่ด้านล่าง" : selectedEvent.model_status === "unverified" ? "Qwen ตรวจไม่สำเร็จ กรุณาตรวจคลิปและหลักฐานเอง" : "รอ Qwen ตรวจภาพ")}</pre><pre className="mt-3 whitespace-pre-wrap break-words text-xs text-slate-400">{JSON.stringify(selectedEvent.verification?.verdict || { status: selectedEvent.model_status, error: selectedEvent.verification?.error, audio_verified: false }, null, 2)}</pre></details><details><summary>คำตอบดิบของ Qwen ทุกครั้ง</summary><pre className="whitespace-pre-wrap break-words text-xs mt-3">{JSON.stringify({ verification: selectedEvent.verification?.verification_attempts || [], caption: selectedEvent.verification?.caption_attempts || [] }, null, 2)}</pre></details><details><summary>หลักฐานคัดกรองรายช่องทาง</summary><pre className="whitespace-pre-wrap break-words text-xs mt-3">{JSON.stringify(selectedEvent.evidence, null, 2)}</pre></details>{selectedEvent.review === "rejected" && screenMode === "open_anomaly" && cameraProfile && <button disabled={busy || !!activeJob} onClick={() => void action(async () => { await post(`/events/profiles/${cameraProfile}/normal-events/${selectedEvent.id}`, { verified_normal: true, split: "development" }); setBankRefresh(v => v + 1); await refresh(); })} className="rounded-xl border border-cyan-700 p-3">ยืนยันว่าปกติและเป็น development · เพิ่มเข้าคลัง</button>}<textarea aria-label="หมายเหตุการตรวจ" placeholder="หมายเหตุจากการตรวจหลักฐาน" value={reviewNote} onChange={e => setReviewNote(e.target.value)} className="w-full bg-slate-950 border border-slate-700 rounded-xl p-3" /><div className="flex gap-3"><button disabled={busy} onClick={() => void action(() => review(selectedEvent, "confirmed"))} className="bg-emerald-600 rounded-xl px-4 py-3"><Check className="inline mr-2" size={16} />ยืนยันเหตุ</button><button disabled={busy} onClick={() => void action(() => review(selectedEvent, "rejected"))} className="bg-rose-600 rounded-xl px-4 py-3"><X className="inline mr-2" size={16} />ปฏิเสธเหตุ</button><button disabled={busy} onClick={() => void action(() => review(selectedEvent, "pending"))} className="border border-slate-700 rounded-xl px-4 py-3">รอตรวจ</button></div></div>}
        <section className="rounded-2xl border border-slate-800 p-5 space-y-4"><h2 className="text-lg">ทรัพยากรที่ใช้</h2><div className="grid grid-cols-2 gap-3">{[["Gemma / broker VRAM ที่จัดสรร", worker.allocated_bytes], ["VRAM สำรองของ broker", worker.reserved_bytes], ["VRAM รวมทั้งการ์ดตอนนี้", worker.whole_gpu_used_bytes], ["RAM ของ broker", worker.rss_bytes]].map(([label, value]) => <div key={String(label)} className="rounded-xl bg-slate-950 p-3"><p className="text-xs text-slate-400">{String(label)}</p><p className="text-lg mt-2 text-cyan-200">{gib(value)}</p></div>)}</div>{qwenJob && <p className="text-sm text-slate-300">ช่วงที่ Qwen ตรวจ: VRAM รวมทั้งการ์ดสูงสุดที่สุ่มวัด {gib(qwenResources.sampled_whole_gpu_peak_bytes)} · RAM ของ Qwen และโปรเซสลูก {gib(qwenResources.qwen_process_tree_peak_rss_bytes)}</p>}<p className="text-xs text-slate-500">จำนวนคำขอ Qwen ไม่นับการซ่อมคำตอบภายใน worker • VRAM รวมทั้งการ์ดรวมโปรแกรมอื่นด้วย Qwen ใช้ llama.cpp จึงแยกจากค่าจัดสรร PyTorch ของ broker</p><details><summary>สถานะบริการโดยละเอียด</summary><pre className="mt-3 whitespace-pre-wrap break-words text-xs text-slate-400">{JSON.stringify(health, null, 2)}</pre></details></section>
      </section>
      <section className="space-y-4">
        <div className="rounded-2xl border border-slate-800 bg-slate-900 p-4 space-y-3">
          <h2 className="text-lg">ผลคัดกรอง {events.length} ช่วง</h2>
          <p className="text-sm text-slate-300">
            Qwen พบจากภาพ {positiveUnits.length} · ไม่พบ {negativeUnits.length} · ยังไม่มีข้อสรุป {undecidedUnits.length}
          </p>
          <p className="text-xs text-slate-400">ช่วงเวลาอาจซ้อนกัน จำนวนนี้ไม่ใช่จำนวนเหตุจริง ผู้ใช้ยังต้องตรวจยืนยัน</p>
          <select aria-label="กรองผลคัดกรอง" value={eventFilter} onChange={e => setEventFilter(e.target.value)} className="w-full rounded-xl bg-slate-950 border border-slate-700 p-3">
            <option value="reviewable">ควรตรวจต่อ</option>
            <option value="positive">Qwen พบจากภาพ</option>
            <option value="negative">Qwen ไม่พบจากภาพ</option>
            <option value="undecided">ยังไม่มีข้อสรุป</option>
            <option value="all">ทั้งหมด</option>
          </select>
          {!events.length && <p className="text-slate-400 text-sm">ยังไม่มีรายการเหตุ เริ่มคัดกรองหลังสร้างดัชนีเสร็จ หากงานล้มเหลวให้ดูสถานะงานด้านบน</p>}
          {visibleEvents.map(event => (
            <button key={event.id} onClick={() => selectEvent(event)} className={`w-full text-left rounded-xl p-3 border ${selectedEvent?.id === event.id ? "border-cyan-400 bg-cyan-950/30" : "border-slate-800 bg-slate-950"}`}>
              <img src={BASE + event.preview_frame_path} alt={event.label} loading="lazy" className="w-full aspect-video object-contain rounded-lg mb-3" />
              <div className="flex justify-between">
                <strong>{event.label}</strong>
                <span className="text-xs text-cyan-300">{event.review}</span>
              </div>
              <p className="text-sm mt-2">{clock(event.t_start)} – {clock(event.t_end)}</p>
              <p className="text-sm mt-2 text-cyan-200">{visualDecision(event, true)}</p>
              <p className="text-xs text-slate-400 mt-2">ภาพ: {event.model_status} · เสียง: {event.audio_status} · {event.threshold_status}</p>
            </button>
          ))}
        </div>
      </section>
    </div>
  </main>
</div>
  );
}
