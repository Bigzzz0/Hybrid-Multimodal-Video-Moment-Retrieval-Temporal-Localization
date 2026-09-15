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
