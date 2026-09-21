import { MomentItem } from "./types";

export type WarningSeverity = "info" | "warning" | "blocking";

export interface UiWarning {
  code: string;
  message: string;
  severity: WarningSeverity;
  action?: "reindex";
}

const WARNING_COPY: Record<string, Omit<UiWarning, "code">> = {
  calibration_missing: {
    message: "ยังไม่ได้ calibrate คะแนน — ลำดับผลใช้เปรียบเทียบกันได้ แต่คะแนนไม่ใช่ probability",
    severity: "info",
  },
  reindex_required: {
    message: "วิดีโอนี้ต้องสร้าง visual index v2 ใหม่ก่อนค้นหา",
    severity: "blocking",
    action: "reindex",
  },
  verifier_timeout_fallback: {
    message: "Accurate ใช้ผล Fast เดิม เพราะการตรวจภาพเกินเวลา",
    severity: "warning",
  },
  verifier_oom_fallback: {
    message: "Accurate ใช้ผล Fast เดิม เพราะหน่วยความจำ GPU ไม่พอ",
    severity: "warning",
  },
  verifier_invalid_json_fallback: {
    message: "Accurate ใช้ผล Fast เดิม เพราะผลตรวจภาพจากโมเดลไม่ถูกต้อง",
    severity: "warning",
  },
  no_match: {
    message: "ไม่พบช่วงเวลาที่ตรงกับคำค้นนี้ตาม threshold ที่ calibrate ไว้",
    severity: "info",
  },
  sam_worker_disabled: {
    message: "SAM 3.1 worker ยังไม่เปิดใช้ จึงแสดงผลจากการค้นหาหลักแทน",
    severity: "warning",
  },
  sam_worker_unavailable: {
    message: "เรียก SAM 3.1 ไม่สำเร็จ จึงใช้ผลค้นหาหลักแทน",
    severity: "warning",
  },
  sam_timeout_fallback: {
    message: "SAM 3.1 ใช้เวลาเกินกำหนด จึงใช้ผลค้นหาหลักแทน",
    severity: "warning",
  },
  sam_paused: {
    message: "พัก SAM 3.1 ชั่วคราว ระบบใช้ผลค้นหาหลักต่อ",
    severity: "info",
  },
  sam_paused_vlm_fallback: {
    message: "พัก SAM 3.1 ชั่วคราว จึงใช้ Qwen VLM ตรวจแทน",
    severity: "info",
  },
  accurate_budget_exhausted: {
    message: "หมดเวลา Accurate budget แล้ว แสดงเฉพาะ candidate ที่ตรวจเสร็จ",
    severity: "warning",
  },
  accurate_partial_budget: {
    message: "เวลาที่เหลือไม่พอโหลด Qwen จึงแสดงผล SAM/Fast ที่ตรวจเสร็จแล้ว",
    severity: "warning",
  },
  sam_no_detection_qwen_fallback: {
    message: "SAM 3.1 ทำงานสำเร็จแต่ไม่พบ object ระบบจึงให้ Qwen ตรวจความหมายต่อ",
    severity: "info",
  },
  sam_ambiguous_qwen_fallback: {
    message: "คำค้นมีหลายความหมาย ระบบจึงให้ Qwen ช่วยยืนยันบริบท",
    severity: "info",
  },
  qwen_fallback_verified: {
    message: "Qwen ยืนยันว่าพบเหตุการณ์ตามความหมายของคำค้น",
    severity: "info",
  },
  qwen_fallback_rejected: {
    message: "Qwen ตรวจภาพแล้วไม่พบเหตุการณ์ตามคำค้น จึงลดอันดับ candidate นี้",
    severity: "info",
  },
  sam_oom_fallback: {
    message: "หน่วยความจำ GPU ไม่พอสำหรับ SAM 3.1 หลังลองลดจำนวนเฟรมแล้ว ระบบคืนผลที่มีอยู่",
    severity: "warning",
  },
  accurate_fallback_fast: {
    message: "โมเดลตรวจยืนยันยังไม่มีหลักฐาน จึงคงผล Fast เดิมไว้เพื่อไม่ให้ผลค้นหาแย่ลง",
    severity: "warning",
  },
  inference_worker_unavailable: {
    message: "Local inference worker ไม่พร้อมใช้งาน",
    severity: "warning",
  },
  qwen_worker_unavailable: {
    message: "Qwen3-VL worker ไม่พร้อมใช้งาน จึงใช้หลักฐาน retrieval ที่มีอยู่",
    severity: "warning",
  },
  qwen_vqa_fallback: {
    message: "ตอบ Video VQA จากหลักฐาน retrieval แทน เพราะ Qwen3-VL ใช้งานไม่ได้",
    severity: "warning",
  },
  sam_insufficient_frames: {
    message: "ช่วงเวลานี้มีเฟรมไม่พอสำหรับ SAM 3.1",
    severity: "info",
  },
  sam_checkpoint_unavailable: {
    message: "ยังไม่พบ SAM 3.1 checkpoint หรือยังไม่ได้รับสิทธิ์ดาวน์โหลด",
    severity: "warning",
  },
  caption_backfill_pending: {
    message: "กำลังสร้าง caption รุ่นใหม่ จึงอาจใช้ caption เดิมบางฉากชั่วคราว",
    severity: "info",
  },
  vlm_artifact_not_ready: {
    message: "VLM รุ่นที่เลือกยังสร้าง artifact ไม่ครบ จึงใช้ caption เดิมของระบบค้นหา",
    severity: "info",
  },
  vlm_worker_unavailable: {
    message: "VLM worker ไม่พร้อมใช้งาน จึงคงผล Fast เดิมไว้",
    severity: "warning",
  },
  vlm_timeout_fallback: {
    message: "VLM ใช้เวลาเกินกำหนด จึงคงผล Fast เดิมไว้",
    severity: "warning",
  },
  vlm_verified: {
    message: "VLM ยืนยันเหตุการณ์จากภาพแล้ว",
    severity: "info",
  },
  vlm_rejected: {
    message: "VLM ไม่พบเหตุการณ์ตามคำค้น จึงลดอันดับ candidate",
    severity: "info",
  },
  vlm_fallback_fast: {
    message: "VLM ตรวจยืนยันไม่สำเร็จ จึงใช้ผล Fast เดิม",
    severity: "warning",
  },
  vlm_backend_invalid_fallback: {
    message: "VLM backend ไม่ถูกต้อง จึงกลับไปใช้ A: Qwen3-VL-2B",
    severity: "warning",
  },
};

export function mapWarning(code: string): UiWarning {
  const copy = WARNING_COPY[code];
  return copy
    ? { code, ...copy }
    : { code, message: `ระบบแจ้งเตือน: ${code}`, severity: "warning" };
}

export function uniqueWarnings(codes: string[] = []): UiWarning[] {
  return Array.from(new Set(codes.filter(Boolean))).map(mapWarning);
}

export function formatMomentTime(seconds: number): string {
  const safe = Math.max(0, Number.isFinite(seconds) ? seconds : 0);
  const minutes = Math.floor(safe / 60);
  const secs = safe - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${secs.toFixed(1).padStart(4, "0")}`;
}

export function momentIdentity(moment: MomentItem): string {
  return `${moment.occurrence_index ?? 0}:${moment.t_start.toFixed(3)}:${moment.t_end.toFixed(3)}`;
}

export function formatScore(score: number, calibrated: boolean): string {
  if (!calibrated) return `Rank score ${Number(score || 0).toFixed(3)}`;
  return `Confidence ${Math.round(Math.max(0, Math.min(1, score || 0)) * 100)}%`;
}
