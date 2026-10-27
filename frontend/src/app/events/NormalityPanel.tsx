"use client";
import { useEffect, useState } from "react";

type Profile = { id: string; name: string; context: string; current_version: string | null };
type Props = { videoId: string; ready: boolean; busy: boolean; profileId: string; onProfile: (id: string) => void };
async function api(path: string, body?: unknown) {
  const response = await fetch(`/api/v1/events${path}`, body === undefined ? undefined : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail));
  return data;
}

export default function NormalityPanel({ videoId, ready, busy, profileId, onProfile }: Props) {
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [name, setName] = useState("");
  const [context, setContext] = useState("");
  const [start, setStart] = useState("0");
  const [end, setEnd] = useState("6");
  const [confirmed, setConfirmed] = useState(false);
  const [working, setWorking] = useState(false);
  const [message, setMessage] = useState("");
  const selected = profiles.find(p => p.id === profileId);
  async function refresh() {
    const data = await api("/profiles");
    setProfiles(Array.isArray(data) ? data.filter(p => typeof p.context === "string") : []);
  }
  useEffect(() => { void refresh().catch(e => setMessage(e.message)); }, []);
  useEffect(() => { setConfirmed(false); }, [videoId, profileId, start, end]);
  async function run(operation: () => Promise<void>) {
    setWorking(true); setMessage("");
    try { await operation(); await refresh(); } catch (e) { setMessage(e instanceof Error ? e.message : String(e)); }
    finally { setWorking(false); }
  }
  return <div className="rounded-xl border border-cyan-800/50 bg-cyan-950/20 p-4 space-y-3">
    <h2 className="text-lg">บริบทกล้องและตัวอย่างปกติ</h2>
    <select aria-label="โปรไฟล์กล้อง" value={profileId} onChange={e => onProfile(e.target.value)} className="w-full rounded-xl bg-slate-950 p-3">
      <option value="">เลือกโปรไฟล์พื้นที่</option>{profiles.map(p => <option key={p.id} value={p.id}>{p.name}</option>)}
    </select>
    {selected && <p className="text-sm text-slate-300">{selected.context} · {selected.current_version ? `คลังปกติรุ่น ${selected.current_version.slice(0, 10)}` : "reference_missing: ยังไม่มีคลังปกติ ใช้คัดจากข้อความเท่านั้น"}</p>}
    <details><summary className="cursor-pointer">สร้างโปรไฟล์ใหม่</summary><div className="space-y-2 mt-3">
      <input aria-label="ชื่อโปรไฟล์" value={name} onChange={e => setName(e.target.value)} placeholder="เช่น ทางเดินอาคาร A" className="w-full rounded-lg bg-slate-950 p-3" />
      <textarea aria-label="บริบทพื้นที่" value={context} onChange={e => setContext(e.target.value)} placeholder="อธิบายพื้นที่ กิจกรรมปกติ และข้อจำกัดของพื้นที่" className="w-full rounded-lg bg-slate-950 p-3" />
      <button disabled={busy || working || !name.trim() || !context.trim()} onClick={() => void run(async () => { const p = await api("/profiles", { name, context }); onProfile(p.id); setName(""); setContext(""); })} className="rounded-lg bg-cyan-700 px-4 py-2 disabled:opacity-40">สร้างโปรไฟล์</button>
    </div></details>
    <details><summary className="cursor-pointer">เพิ่มช่วงปกติที่ตรวจแล้วจากวิดีโอนี้</summary><div className="space-y-3 mt-3">
      <p className="text-sm text-slate-400">ใช้เฉพาะข้อมูล development ที่ตรวจว่าปกติแล้ว เก็บเฉพาะหน้าต่างดัชนีที่อยู่ภายในช่วงนี้ทั้งหมด ห้ามใช้คลิป validation หรือ test</p>
      <div className="flex gap-3"><label>เริ่ม (วินาที)<input aria-label="ช่วงปกติเริ่ม" type="number" min="0" step="0.1" value={start} onChange={e => setStart(e.target.value)} className="w-full bg-slate-950 p-2 rounded-lg" /></label><label>สิ้นสุด (วินาที)<input aria-label="ช่วงปกติสิ้นสุด" type="number" min="0" step="0.1" value={end} onChange={e => setEnd(e.target.value)} className="w-full bg-slate-950 p-2 rounded-lg" /></label></div>
      <label className="flex gap-2 text-sm"><input type="checkbox" checked={confirmed} onChange={e => setConfirmed(e.target.checked)} />ตรวจดูช่วงนี้แล้วว่าเป็นกิจกรรมปกติ และเป็นข้อมูล development</label>
      <button disabled={busy || working || !ready || !profileId || !confirmed || !(Number(end) > Number(start))} onClick={() => void run(async () => {
        const data = await api(`/profiles/${profileId}/references`, { expected_version: selected?.current_version, ranges: [{ video_id: videoId, t_start: Number(start), t_end: Number(end), split: "development", verified_normal: true }] });
        setMessage(`บันทึกคลังรุ่น ${data.version.slice(0, 10)} · ${data.reference_count} ตัวอย่าง`); setConfirmed(false);
      })} className="rounded-lg bg-cyan-700 px-4 py-2 disabled:opacity-40">เพิ่มตัวอย่างปกติ</button>
    </div></details>
    {message && <p role="status" className="text-sm text-amber-200">{message}</p>}
  </div>;
}
