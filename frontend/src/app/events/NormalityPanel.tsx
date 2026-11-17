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
  return (
    <div className="rounded-2xl border border-[var(--line)] bg-[var(--card-bg)] p-5 space-y-4 shadow-sm">
      <div className="flex items-center justify-between border-b border-[var(--line)] pb-3">
        <div>
          <h2 className="text-sm font-bold text-[var(--text)]">บริบทกล้องและคลังพฤติกรรมปกติ (Camera Profile & Normality Bank)</h2>
          <p className="text-xs text-[var(--text-muted)] mt-0.5">ใช้เปรียบเทียบพฤติกรรมปกติของแต่ละสถานที่เพื่อลดผลการแจ้งเตือนหลอกตา</p>
        </div>
      </div>

      <select
        aria-label="โปรไฟล์กล้อง"
        value={profileId}
        onChange={e => onProfile(e.target.value)}
        className="w-full rounded-xl bg-[var(--bg)] border border-[var(--line)] p-2.5 text-sm text-[var(--text)] focus:border-[var(--accent)] focus:outline-none transition-colors"
      >
        <option value="">เลือกโปรไฟล์พื้นที่ / กล้อง</option>
        {profiles.map(p => <option key={p.id} value={p.id}>{p.name}</option>)}
      </select>

      {selected && (
        <div className="rounded-xl bg-[var(--accent-subtle)] border border-[var(--line)] p-3 text-xs text-[var(--text)] space-y-1">
          <p className="font-semibold text-[var(--accent)]">{selected.name}</p>
          <p className="text-[var(--text-muted)]">{selected.context}</p>
          <p className="text-[11px] text-[var(--text-muted)]">
            {selected.current_version
              ? `คลังปกติรุ่น ${selected.current_version.slice(0, 10)}`
              : "reference_missing: ยังไม่มีคลังตัวอย่างปกติ (ใช้เกณฑ์ข้อความเท่านั้น)"}
          </p>
        </div>
      )}

      <details className="group">
        <summary className="cursor-pointer text-xs font-semibold text-[var(--accent)] hover:underline">
          + สร้างโปรไฟล์พื้นที่ใหม่
        </summary>
        <div className="space-y-3 mt-3 pt-3 border-t border-[var(--line)]">
          <input
            aria-label="ชื่อโปรไฟล์"
            value={name}
            onChange={e => setName(e.target.value)}
            placeholder="เช่น ทางเดินอาคารเรียน 1 หรือ ลานจอดรถหน้าทางเข้า"
            className="w-full rounded-xl bg-[var(--bg)] border border-[var(--line)] p-2.5 text-xs text-[var(--text)] focus:border-[var(--accent)] focus:outline-none"
          />
          <textarea
            aria-label="บริบทพื้นที่"
            value={context}
            onChange={e => setContext(e.target.value)}
            placeholder="อธิบายกิจกรรมปกติในพื้นที่ (เช่น คนเดินผ่านปกติ ห้ามวิ่ง ห้ามมีรถจอดขวาง)"
            className="w-full rounded-xl bg-[var(--bg)] border border-[var(--line)] p-2.5 text-xs text-[var(--text)] focus:border-[var(--accent)] focus:outline-none min-h-[70px]"
          />
          <button
            disabled={busy || working || !name.trim() || !context.trim()}
            onClick={() => void run(async () => {
              const p = await api("/profiles", { name, context });
              onProfile(p.id);
              setName("");
              setContext("");
            })}
            className="rounded-xl bg-[var(--accent)] text-[var(--accent-text)] px-4 py-2 text-xs font-semibold disabled:opacity-40 hover:bg-[var(--accent-hover)] transition-all shadow-sm"
          >
            บันทึกโปรไฟล์
          </button>
        </div>
      </details>

      <details className="group">
        <summary className="cursor-pointer text-xs font-semibold text-[var(--text-muted)] hover:text-[var(--text)]">
          + เพิ่มช่วงเวลาปกติเข้าสู่คลังอ้างอิงจากคลิปนี้
        </summary>
        <div className="space-y-3 mt-3 pt-3 border-t border-[var(--line)]">
          <p className="text-xs text-[var(--text-muted)]">
            ระบุช่วงเวลา development ที่ตรวจสอบแล้วว่าเป็นเหตุการณ์ปกติ (ห้ามใช้คลิป validation หรือ test)
          </p>
          <div className="flex gap-3">
            <div className="flex-1">
              <label className="text-[11px] text-[var(--text-muted)] block mb-1">วินาทีเริ่มต้น</label>
              <input
                aria-label="ช่วงปกติเริ่ม"
                type="number"
                min="0"
                step="0.1"
                value={start}
                onChange={e => setStart(e.target.value)}
                className="w-full bg-[var(--bg)] border border-[var(--line)] p-2 rounded-xl text-xs text-[var(--text)]"
              />
            </div>
            <div className="flex-1">
              <label className="text-[11px] text-[var(--text-muted)] block mb-1">วินาทีสิ้นสุด</label>
              <input
                aria-label="ช่วงปกติสิ้นสุด"
                type="number"
                min="0"
                step="0.1"
                value={end}
                onChange={e => setEnd(e.target.value)}
                className="w-full bg-[var(--bg)] border border-[var(--line)] p-2 rounded-xl text-xs text-[var(--text)]"
              />
            </div>
          </div>
          <label className="flex items-center gap-2 text-xs text-[var(--text)] cursor-pointer">
            <input
              type="checkbox"
              checked={confirmed}
              onChange={e => setConfirmed(e.target.checked)}
              className="rounded accent-[var(--accent)]"
            />
            ยืนยันว่าช่วงเวลานี้เป็นกิจกรรมปกติที่เกิดขึ้นจริง
          </label>
          <button
            disabled={busy || working || !ready || !profileId || !confirmed || !(Number(end) > Number(start))}
            onClick={() => void run(async () => {
              const data = await api(`/profiles/${profileId}/references`, {
                expected_version: selected?.current_version,
                ranges: [{ video_id: videoId, t_start: Number(start), t_end: Number(end), split: "development", verified_normal: true }]
              });
              setMessage(`บันทึกคลังรุ่น ${data.version.slice(0, 10)} เรียบร้อย (${data.reference_count} ตัวอย่าง)`);
              setConfirmed(false);
            })}
            className="rounded-xl bg-[var(--accent)] text-[var(--accent-text)] px-4 py-2 text-xs font-semibold disabled:opacity-40 hover:bg-[var(--accent-hover)] transition-all shadow-sm"
          >
            เพิ่มเข้าคลังตัวอย่างปกติ
          </button>
        </div>
      </details>

      {message && <p role="status" className="text-xs text-amber-500 font-medium">{message}</p>}
    </div>
  );
}
