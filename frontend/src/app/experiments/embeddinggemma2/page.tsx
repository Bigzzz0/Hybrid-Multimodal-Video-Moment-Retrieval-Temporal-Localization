"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { ArrowLeft, CheckCircle2, Clock3, Film, GitCompareArrows, ImageIcon, Loader2, Play, Search, UploadCloud } from "lucide-react";

const API = process.env.NEXT_PUBLIC_EMBEDDING_EXPERIMENT_URL || "http://localhost:8012";
type Role = "baseline" | "candidate" | "candidate_av";
type Mode = "both" | Role;
type Video = { id: string; name: string; duration_sec: number; source: string; examples: string[]; indexed_models: Role[]; frame_count: number };
type Hit = { timestamp: number; cosine: number; frame_index: number };
type Moment = { start: number; end: number; frame_index: number };
type ModelResult = { name: string; query_ms: number; load_ms: number; index_ms: number; frames: Hit[]; moments: Moment[] };
type Result = { query: string; video_id: string; roles: Partial<Record<Role, ModelResult>>; elapsed_ms: number; frame_count: number; sample_hz: number; has_audio?: boolean; window_count?: number };
type KamakuraReport = { query_p50_ms: number; torch_peak_allocated_mib: number; index_resources: {wall_seconds:number; peaks:{rss_mib:number}}; accuracy: Record<string, number> };
type Job = { status: "queued" | "running" | "done" | "error"; progress: number; message: string; result: Result | null; error: string | null };
type Benchmark = { query_count: number; roles: Record<Role, Record<string, number>> };
const roleNames: Record<Role, string> = { baseline: "SigLIP2", candidate: "EmbeddingGemma 2 · ภาพ", candidate_av: "EmbeddingGemma 2 · Video + Audio" };
const formatTime = (seconds: number) => `${Math.floor(seconds / 60).toString().padStart(2, "0")}:${Math.floor(seconds % 60).toString().padStart(2, "0")}`;

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, init);
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(typeof body?.detail === "string" ? body.detail : `ไม่สามารถดำเนินการได้ (${response.status})`);
  }
  return response.json();
}

export default function EmbeddingExperimentPage() {
  const [videos, setVideos] = useState<Video[]>([]);
  const [videoId, setVideoId] = useState("");
  const [query, setQuery] = useState("รถไฟวิ่งริมทะเล");
  const [mode, setMode] = useState<Mode>("candidate_av");
  const [online, setOnline] = useState(false);
  const [connecting, setConnecting] = useState(true);
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [job, setJob] = useState<Job | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState("");
  const [view, setView] = useState<"frames" | "moments">("frames");
  const [selectedHit, setSelectedHit] = useState("");
  const [currentTime, setCurrentTime] = useState(0);
  const [benchmark, setBenchmark] = useState<Benchmark | null>(null);
  const [kamakura, setKamakura] = useState<Record<string, KamakuraReport>>({});
  const player = useRef<HTMLVideoElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const controller = useRef<AbortController | null>(null);
  const selectedVideo = videos.find((video) => video.id === videoId);

  async function connect() {
    setConnecting(true);
    setError("");
    try {
      await api("/health", { signal: AbortSignal.timeout(8000) });
      const data = await api<Video[]>("/videos");
      setVideos(data);
      setVideoId((current) => current || data.find(video => video.name.includes("Kamakura"))?.id || data[0]?.id || "");
      setOnline(true);
      setBenchmark(await api<Benchmark | null>("/benchmark"));
      setKamakura(await api<Record<string, KamakuraReport>>("/benchmark/kamakura"));
    } catch {
      setOnline(false);
      setError("ยังเชื่อมต่อหน้าทดลองไม่ได้ กรุณาตรวจว่าบริการทดลองเปิดอยู่ แล้วลองเชื่อมต่ออีกครั้ง");
    } finally {
      setConnecting(false);
    }
  }

  useEffect(() => {
    void connect();
    return () => controller.current?.abort();
    // The first connection is intentionally made only on mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function search() {
    if (!query.trim() || !videoId || busy) return;
    setBusy(true);
    setError("");
    setResult(null);
    setSelectedHit("");
    setJob({ status: "queued", progress: 0, message: "ส่งคำค้น…", result: null, error: null });
    const abort = new AbortController();
    controller.current = abort;
    try {
      const { job_id } = await api<{ job_id: string }>("/search", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video_id: videoId, query: query.trim(), mode }), signal: abort.signal,
      });
      while (!abort.signal.aborted) {
        const next = await api<Job>(`/jobs/${job_id}`, { signal: abort.signal });
        setJob(next);
        if (next.status === "error") throw new Error(next.error || "ค้นหาไม่สำเร็จ");
        if (next.status === "done") {
          setResult(next.result);
          setVideos(await api<Video[]>("/videos", { signal: abort.signal }));
          break;
        }
        await new Promise((resolve) => setTimeout(resolve, 700));
      }
    } catch (caught) {
      if (!abort.signal.aborted) setError(caught instanceof Error ? caught.message : "ค้นหาไม่สำเร็จ");
    } finally {
      if (!abort.signal.aborted) setBusy(false);
    }
  }

  async function upload(file: File) {
    if (file.size > 1024 ** 3) { setError("กรุณาเลือกไฟล์ขนาดไม่เกิน 1 GB"); return; }
    setUploading(true);
    setError("");
    try {
      const body = new FormData();
      body.append("file", file);
      const uploaded = await api<Video>("/videos/upload", { method: "POST", body });
      setVideos((current) => [...current, uploaded]);
      setVideoId(uploaded.id);
      setResult(null);
      setJob(null);
      setCurrentTime(0);
      setQuery("");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "อัปโหลดไม่สำเร็จ");
    } finally {
      setUploading(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  }

  function seek(seconds: number, key: string) {
    if (!player.current) return;
    player.current.currentTime = seconds;
    setCurrentTime(seconds);
    setSelectedHit(key);
    void player.current.play().catch(() => {});
  }

  return (
    <div className="space-y-6 pb-10" lang="th">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <Link href="/" className="mb-3 inline-flex items-center gap-2 text-sm text-slate-400 hover:text-white"><ArrowLeft size={15} /> กลับหน้าหลัก</Link>
          <div className="flex items-center gap-3"><span className="rounded-xl border border-cyan-400/20 bg-cyan-400/10 p-2.5 text-cyan-300"><GitCompareArrows size={24} /></span><h2 className="text-2xl font-semibold text-white sm:text-3xl">ทดลองค้นหาวิดีโอ</h2></div>
          <p className="mt-3 text-sm leading-relaxed text-slate-400">พิมพ์เหตุการณ์ที่อยากหา แล้วเปรียบเทียบ SigLIP2 กับ EmbeddingGemma 2 ด้วยวิดีโอเดียวกัน</p>
        </div>
        <div className={`mt-7 flex items-center gap-2 rounded-full border px-3 py-2 text-xs ${online ? "border-emerald-500/20 bg-emerald-500/10 text-emerald-300" : "border-slate-700 bg-slate-800 text-slate-400"}`}>
          {connecting ? <Loader2 size={13} className="animate-spin" /> : <span className={`h-2 w-2 rounded-full ${online ? "bg-emerald-400" : "bg-slate-500"}`} />}{connecting ? "กำลังเชื่อมต่อ" : online ? "พร้อมทดลอง" : "ยังไม่เชื่อมต่อ"}
        </div>
      </div>

      {error && <div role="alert" className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-red-400/20 bg-red-400/10 px-4 py-3 text-sm text-red-200"><span>{error}</span>{!online && <button onClick={() => void connect()} disabled={connecting} className="rounded-lg border border-red-300/30 px-3 py-1.5 disabled:opacity-50">เชื่อมต่ออีกครั้ง</button>}</div>}

      <div className="grid gap-5 lg:grid-cols-[minmax(0,1.65fr)_minmax(300px,1fr)]">
        <section className="overflow-hidden rounded-2xl border border-white/10 bg-[#0c1220]">
          <div className="flex items-center justify-between border-b border-white/5 px-5 py-4"><span className="flex items-center gap-2 text-sm font-medium"><Film size={17} className="text-cyan-300" /> วิดีโอที่ใช้ทดลอง</span><span className="text-xs text-slate-500">{selectedVideo ? formatTime(selectedVideo.duration_sec) : "เลือกวิดีโอ"}</span></div>
          <div className="aspect-video bg-black">{selectedVideo ? <video key={videoId} ref={player} controls playsInline preload="metadata" src={`${API}/videos/${encodeURIComponent(videoId)}/stream`} className="h-full w-full" onTimeUpdate={() => setCurrentTime(player.current?.currentTime || 0)} aria-label={`วิดีโอ ${selectedVideo.name}`} /> : <div className="flex h-full items-center justify-center text-slate-500"><Film size={40} /></div>}</div>
          <div className="px-5 py-4"><div className="flex items-center justify-between gap-3 text-sm"><p className="truncate font-medium text-slate-200">{selectedVideo?.name || "ยังไม่มีวิดีโอ"}</p><span className="shrink-0 font-mono text-xs text-cyan-300">{formatTime(currentTime)}</span></div><p className="mt-2 text-xs leading-relaxed text-slate-500">กดภาพที่ค้นพบเพื่อไปยังเวลานั้นในวิดีโอ การค้นหาใช้ภาพตัวอย่างทุก 1 วินาที</p></div>
        </section>

        <section className="space-y-5 rounded-2xl border border-white/10 bg-[#0c1220] p-5">
          <div><label htmlFor="experiment-video" className="mb-2 block text-sm font-medium text-slate-200">เลือกวิดีโอ</label><select id="experiment-video" value={videoId} disabled={busy || uploading || !online} onChange={(event) => { setVideoId(event.target.value); setResult(null); setJob(null); setCurrentTime(0); }} className="w-full rounded-xl border border-slate-700 bg-[#080f1c] px-3 py-3 text-sm text-slate-200 disabled:opacity-50">{videos.map((video) => <option key={video.id} value={video.id}>{video.name}</option>)}</select></div>
          <div><input ref={fileInput} type="file" accept=".mp4,.mov,.avi,.webm,.m4v" className="hidden" onChange={(event) => { const file = event.target.files?.[0]; if (file) void upload(file); }} aria-label="เลือกไฟล์วิดีโอ" /><button disabled={busy || uploading || !online} onClick={() => fileInput.current?.click()} className="flex w-full items-center justify-center gap-2 rounded-xl border border-dashed border-slate-600 px-4 py-3 text-sm text-slate-300 transition hover:border-cyan-400 hover:text-cyan-200 disabled:opacity-40">{uploading ? <Loader2 size={17} className="animate-spin" /> : <UploadCloud size={17} />}{uploading ? "กำลังอัปโหลด…" : "อัปโหลดวิดีโอของคุณ"}</button><p className="mt-2 text-xs text-slate-500">MP4 MOV AVI WEBM หรือ M4V ขนาดไม่เกิน 1 GB</p></div>
          <fieldset disabled={busy || uploading}><legend className="mb-2 text-sm font-medium text-slate-200">โมเดลที่ต้องการทดสอบ</legend><div className="grid grid-cols-2 gap-1.5 rounded-xl bg-[#080f1c] p-1.5">{(["both", "baseline", "candidate", "candidate_av"] as Mode[]).map((value) => <button key={value} type="button" aria-pressed={mode === value} onClick={() => setMode(value)} className={`rounded-lg px-2 py-2.5 text-xs font-medium transition ${mode === value ? "bg-cyan-400/15 text-cyan-200 ring-1 ring-cyan-400/30" : "text-slate-400 hover:text-white"}`}>{value === "both" ? "เทียบทั้งสอง" : value === "baseline" ? "SigLIP2" : value === "candidate" ? "Gemma · ภาพ" : "Gemma · Video + Audio"}</button>)}</div></fieldset>
          <form onSubmit={(event) => { event.preventDefault(); void search(); }} className="space-y-3"><label htmlFor="experiment-query" className="block text-sm font-medium text-slate-200">อยากหาเหตุการณ์อะไร</label><textarea id="experiment-query" value={query} onChange={(event) => setQuery(event.target.value)} maxLength={300} disabled={busy || uploading} placeholder="เช่น คนปั่นจักรยาน หรือ person riding a bicycle" rows={3} className="w-full resize-none rounded-xl border border-slate-700 bg-[#080f1c] px-3 py-3 text-sm leading-relaxed text-slate-200 placeholder:text-slate-600" /><button disabled={!online || !query.trim() || !videoId || busy || uploading} className="flex w-full items-center justify-center gap-2 rounded-xl bg-cyan-300 px-4 py-3 font-semibold text-[#04212b] transition hover:bg-cyan-200 disabled:cursor-not-allowed disabled:opacity-40">{busy ? <Loader2 size={18} className="animate-spin" /> : <Search size={18} />}{busy ? "กำลังค้นหา" : "ค้นหาเหตุการณ์"}</button></form>
          {selectedVideo && <div className="flex flex-wrap gap-2">{selectedVideo.examples.filter((example) => /[\u0e00-\u0e7f]/.test(example)).slice(0, 4).map((example) => <button key={example} disabled={busy || uploading} onClick={() => setQuery(example)} className="rounded-full border border-slate-700/80 px-2.5 py-1.5 text-xs text-slate-400 hover:border-cyan-400/40 hover:text-cyan-200">{example}</button>)}</div>}
        </section>
      </div>

      {busy && job && <div role="status" aria-live="polite" className="rounded-xl border border-cyan-400/15 bg-cyan-400/5 px-5 py-4"><div className="mb-3 flex items-center justify-between gap-3 text-sm"><span className="flex items-center gap-2 text-cyan-200"><Loader2 size={16} className="animate-spin" />{job.message}</span><span className="font-mono text-xs text-slate-400">{job.progress}%</span></div><div role="progressbar" aria-label="ความคืบหน้าการค้นหา" aria-valuemin={0} aria-valuemax={100} aria-valuenow={job.progress} className="h-1.5 overflow-hidden rounded-full bg-slate-800"><div className="h-full rounded-full bg-cyan-300 transition-all" style={{ width: `${job.progress}%` }} /></div><p className="mt-2 text-xs text-slate-500">ครั้งแรกอาจใช้เวลาเตรียมภาพและโหลดโมเดล คำค้นครั้งถัดไปใช้ดัชนีภาพเดิมได้</p></div>}

      {result ? <section className="space-y-4"><div className="flex flex-wrap items-end justify-between gap-3"><div><h3 className="flex items-center gap-2 text-lg font-semibold"><CheckCircle2 size={19} className="text-emerald-400" /> ผลค้นหา</h3><p className="mt-1 text-sm text-slate-400">“{result.query}” <span className="text-slate-600">· {result.frame_count} ภาพ · รอรวม {(result.elapsed_ms / 1000).toFixed(1)} วินาที</span></p></div><div className="flex rounded-lg border border-slate-700 bg-[#0c1220] p-1">{(["frames", "moments"] as const).map((tab) => <button key={tab} onClick={() => setView(tab)} aria-pressed={view === tab} className={`rounded-md px-3 py-2 text-xs ${view === tab ? "bg-slate-700 text-white" : "text-slate-400"}`}>{tab === "frames" ? "เฟรมที่ตรงที่สุด" : "ช่วงเวลาที่แนะนำ"}</button>)}</div></div>
        <div className={`grid gap-5 ${Object.keys(result.roles).length > 1 ? "lg:grid-cols-2" : ""}`}>{(["baseline", "candidate", "candidate_av"] as Role[]).map((role) => {
          const data = result.roles[role];
          if (!data) return null;
          const hits = view === "frames" ? data.frames : data.moments;
          return <div key={role} className="overflow-hidden rounded-2xl border border-white/10 bg-[#0c1220]"><div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/5 px-5 py-4"><h4 className="flex items-center gap-2 font-semibold"><span className={`h-2 w-2 rounded-full ${role === "baseline" ? "bg-cyan-300" : "bg-violet-400"}`} />{role === "candidate_av" ? data.name : roleNames[role]}</h4><span className="flex items-center gap-1.5 rounded-full bg-white/5 px-2.5 py-1 text-xs text-slate-300"><Clock3 size={12} /> ค้นหา {data.query_ms.toFixed(0)} ms</span></div><div className="space-y-2 p-3">{hits.length ? hits.map((hit, i) => {
            const isFrame = "timestamp" in hit;
            const timestamp = isFrame ? hit.timestamp : hit.start;
            const key = `${role}-${view}-${i}`;
            return <button key={key} onClick={() => seek(timestamp, key)} className={`group flex w-full items-center gap-3 rounded-xl border p-2 text-left transition ${selectedHit === key ? "border-cyan-300/50 bg-cyan-300/10" : "border-transparent hover:border-slate-600 hover:bg-white/5"}`} aria-label={`ดูผลอันดับ ${i+1} ของ ${roleNames[role]} ที่เวลา ${formatTime(timestamp)}`}><div className="relative w-28 shrink-0 overflow-hidden rounded-lg bg-black"><img src={`${API}/videos/${encodeURIComponent(result.video_id)}/frames/${hit.frame_index}`} alt={`ภาพผลค้นหาอันดับ ${i + 1}`} className="aspect-video w-full object-cover" /><span className="absolute bottom-1 left-1 rounded bg-black/80 px-1.5 py-0.5 text-[10px] text-white">#{i+1}</span><Play size={18} className="absolute right-2 top-2 text-white opacity-0 transition group-hover:opacity-100" /></div><div className="min-w-0 flex-1"><p className="font-mono text-sm font-medium text-slate-200">{formatTime(timestamp)}{!isFrame && <> — {formatTime(hit.end)}</>}</p><p className="mt-1 text-xs text-slate-500">{isFrame ? `ความคล้าย ${hit.cosine.toFixed(3)}` : `ช่วงเวลา ${(hit.end - hit.start).toFixed(1)} วินาที`}</p></div><Play size={14} className="mr-2 shrink-0 text-slate-500" /></button>;
          }) : <p className="px-3 py-6 text-sm text-slate-500">ยังไม่พบช่วงเวลาที่แนะนำ ลองดูผลในแท็บเฟรมหรือเปลี่ยนคำค้น</p>}</div><p className="border-t border-white/5 px-5 py-3 text-xs leading-relaxed text-slate-500">{data.index_ms > 0 ? `เตรียมดัชนี ${(data.index_ms / 1000).toFixed(1)} วินาที` : "ใช้ดัชนีภาพที่เตรียมไว้"}{data.load_ms > 0 ? ` · โหลดโมเดล ${(data.load_ms / 1000).toFixed(1)} วินาที` : " · โมเดลพร้อมอยู่แล้ว"}</p></div>;
        })}</div><p className="text-xs leading-relaxed text-slate-500">เวลาค้นหาข้างต้นไม่รวมเวลาโหลดโมเดลและเตรียมดัชนี คะแนนความคล้ายใช้จัดอันดับภายในแต่ละโมเดล ไม่ใช่เปอร์เซ็นต์ความถูกต้อง</p></section> : !busy && <div className="flex flex-col items-center rounded-2xl border border-dashed border-slate-700/60 py-10 text-center"><ImageIcon size={28} className="mb-3 text-slate-600" /><p className="text-sm text-slate-400">ผลค้นหาของแต่ละโมเดลจะแสดงตรงนี้</p><p className="mt-2 text-xs text-slate-600">โหมด Video + Audio ใช้ภาพหลายเฟรมกับเสียงจริงของช่วงนั้น คลิปไม่มีเสียงจะใช้วิดีโออย่างเดียว</p></div>}

      {Object.keys(kamakura).length > 0 && <details open className="rounded-xl border border-violet-400/20 bg-[#0c1220] px-5 py-4"><summary className="cursor-pointer text-sm text-violet-200">ผลทดสอบคลิป Kamakura · ความแม่นยำและทรัพยากร</summary><div className="mt-4 overflow-x-auto"><table className="w-full text-left text-sm"><thead><tr className="border-b border-slate-700 text-slate-400"><th className="p-2">โหมด</th><th className="p-2">Hit@1</th><th className="p-2">Hit@5</th><th className="p-2">ค้นหา p50</th><th className="p-2">GPU allocated สูงสุด</th><th className="p-2">RAM สูงสุดขณะทำดัชนี</th><th className="p-2">ทำดัชนี</th></tr></thead><tbody>{Object.entries(kamakura).map(([name, report]) => <tr key={name} className="border-b border-slate-800 text-slate-300"><td className="p-2">{name === "siglip" ? "SigLIP2 · เฉลี่ยเฟรม" : name === "gemma_frames" ? "Gemma · เฉลี่ยเฟรม" : "Gemma · Video + Audio"}</td><td className="p-2">{report.accuracy["CenterFrameHit@1"].toFixed(1)}%</td><td className="p-2">{report.accuracy["CenterFrameHit@5"].toFixed(1)}%</td><td className="p-2">{report.query_p50_ms.toFixed(1)} ms</td><td className="p-2">{report.torch_peak_allocated_mib.toFixed(0)} MiB</td><td className="p-2">{report.index_resources.peaks.rss_mib.toFixed(0)} MiB</td><td className="p-2">{report.index_resources.wall_seconds.toFixed(1)} s</td></tr>)}</tbody></table></div><p className="mt-3 text-xs leading-relaxed text-slate-500">คลิปเต็ม 19:43 นาที · 198 ช่วง · 20 คำค้น (10 เหตุการณ์ ไทย–อังกฤษ) Hit วัดความตรงกับภาพกลางช่วงที่ตรวจไว้ ไม่ใช่ความแม่นของเสียงหรือขอบเขตเหตุการณ์ โหมดเต็มใช้ 1,120 vision tokens ต่อเฟรมและเปิด audio encoder เวลา benchmark ไม่รวม warmup</p></details>}
      {benchmark && <details className="rounded-xl border border-white/10 bg-[#0c1220] px-5 py-4"><summary className="cursor-pointer text-sm text-slate-300">ผล benchmark ล่าสุด · {benchmark.query_count} คำถาม</summary><div className="mt-4 overflow-x-auto"><table className="w-full text-left text-sm"><thead><tr className="border-b border-slate-700 text-slate-400"><th className="pb-3 font-medium">ตัวชี้วัด</th><th className="pb-3 font-medium">SigLIP2</th><th className="pb-3 font-medium">EmbeddingGemma 2</th></tr></thead><tbody>{[["FrameHit@1", "เฟรมอันดับแรกตรงเหตุการณ์", "%"], ["FrameHit@5", "มีเฟรมตรงใน 5 อันดับแรก", "%"], ["p50_query_latency_ms", "เวลาค้นหาค่ากลาง", " ms"]].map(([key, label, unit]) => <tr key={key} className="border-b border-slate-800 text-slate-300"><td className="py-3">{label}</td><td>{benchmark.roles.baseline[key].toFixed(2)}{unit}</td><td>{benchmark.roles.candidate[key].toFixed(2)}{unit}</td></tr>)}</tbody></table></div><p className="mt-3 text-xs leading-relaxed text-slate-500">ทดสอบจากภาพในวิดีโอ 2 ไฟล์ ยังไม่รวมคำบรรยาย เสียง หรือขั้นตรวจสอบเหตุการณ์ของระบบเต็ม</p></details>}
    </div>
  );
}
