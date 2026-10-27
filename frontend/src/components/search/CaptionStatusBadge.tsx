"use client";

import React, { useEffect, useState, useCallback } from "react";
import { Loader2, CheckCircle2, AlertTriangle, RefreshCw, Layers, XCircle } from "lucide-react";
import { apiClient } from "@/lib/api";
import { CaptionStatus } from "@/lib/types";

interface CaptionStatusBadgeProps {
  videoId?: string;
  onRebuildTriggered?: () => void;
}

export const CaptionStatusBadge: React.FC<CaptionStatusBadgeProps> = ({ videoId, onRebuildTriggered }) => {
  const [status, setStatus] = useState<CaptionStatus | null>(null);
  const [selectedDensity, setSelectedDensity] = useState<number>(8);
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [isCancelling, setIsCancelling] = useState<boolean>(false);
  const [feedbackMsg, setFeedbackMsg] = useState<string>("");

  const loadStatus = useCallback(async () => {
    if (!videoId) return;
    try {
      const next = await apiClient.getCaptionStatus(videoId);
      setStatus(next);
    } catch {
      // keep prior or clear
    }
  }, [videoId]);

  useEffect(() => {
    if (!videoId) {
      setStatus(null);
      return;
    }
    void loadStatus();
    const timer = window.setInterval(() => {
      void loadStatus();
    }, 2500);
    return () => {
      window.clearInterval(timer);
    };
  }, [videoId, loadStatus]);

  const handleRebuild = async (force: boolean = true) => {
    if (!videoId || isSubmitting) return;
    setIsSubmitting(true);
    setFeedbackMsg(force ? "กำลังส่งคำขอ..." : "กำลังดำเนินการต่อ (Resume)...");
    try {
      await apiClient.rebuildCaptions(videoId, selectedDensity, force);
      setFeedbackMsg(force ? `เริ่มวิเคราะห์ใหม่ด้วย ${selectedDensity} เฟรม/ฉาก` : `กำลังทำต่อจากจุดเดิม (${completed}/${expected} ฉาก)`);
      if (status) {
        setStatus({
          ...status,
          status: "running",
          progress_percent: force ? 0 : status.progress_percent,
          completed_count: force ? 0 : status.completed_count,
        });
      }
      if (onRebuildTriggered) onRebuildTriggered();
      setTimeout(() => void loadStatus(), 800);
    } catch (err: any) {
      setFeedbackMsg(err?.response?.data?.detail || "เกิดข้อผิดพลาดในการคำนวณใหม่");
    } finally {
      setIsSubmitting(false);
      setTimeout(() => setFeedbackMsg(""), 4000);
    }
  };

  const handleCancel = async () => {
    if (!videoId || isCancelling) return;
    setIsCancelling(true);
    setFeedbackMsg("กำลังส่งคำขอยกเลิก...");
    try {
      await apiClient.cancelCaptions(videoId);
      setFeedbackMsg("ยกเลิกการคำนวณเรียบร้อยแล้ว");
      if (status) {
        setStatus({
          ...status,
          status: "cancelled",
        });
      }
      setTimeout(() => void loadStatus(), 600);
    } catch (err: any) {
      setFeedbackMsg(err?.response?.data?.detail || "เกิดข้อผิดพลาดในการยกเลิก");
    } finally {
      setIsCancelling(false);
      setTimeout(() => setFeedbackMsg(""), 4000);
    }
  };

  if (!videoId) return null;

  const currentStatusStr = status?.status ?? "unavailable";
  const ready = currentStatusStr === "ready";
  const error = currentStatusStr === "error";
  const cancelled = currentStatusStr === "cancelled";
  const unavailable = currentStatusStr === "unavailable";
  const running = currentStatusStr === "running" || currentStatusStr === "pending";
  const interrupted = currentStatusStr === "interrupted";

  const expected = Number(status?.expected_count ?? 0);
  const completed = Number(status?.completed_count ?? 0);
  const percent = Math.min(
    100,
    Math.max(0, Number(status?.progress_percent ?? (expected ? (completed / expected) * 100 : 0)))
  );
  const currentModel = status?.current_model || status?.worker?.models?.current_model || "";
  const currentTask = status?.current_task || status?.worker?.models?.current_task || "";

  return (
    <div className="flex flex-wrap items-center gap-2 font-mono text-[11px]">
      {/* 1. Status Indicator Pill */}
      <div
        className={`inline-flex flex-col gap-1 rounded-lg border px-2.5 py-1.5 transition-all ${
          ready
            ? "border-emerald-700/60 bg-emerald-950/30 text-emerald-300"
            : cancelled
            ? "border-slate-700/70 bg-slate-900/80 text-slate-300"
            : interrupted
            ? "border-amber-600/70 bg-amber-950/40 text-amber-200 shadow-sm"
            : error
            ? "border-amber-700/60 bg-amber-950/30 text-amber-300"
            : running
            ? "border-cyan-600/70 bg-cyan-950/40 text-cyan-200 shadow-sm shadow-cyan-950/50"
            : "border-slate-800 bg-slate-900/60 text-slate-400"
        }`}
        role="status"
        title={running ? `CapRL Q6 progress: ${completed}/${expected}` : undefined}
      >
        <div className="inline-flex items-center gap-1.5">
          {ready ? (
            <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
          ) : cancelled ? (
            <XCircle className="h-3.5 w-3.5 text-slate-400" />
          ) : interrupted ? (
            <AlertTriangle className="h-3.5 w-3.5 text-amber-400" />
          ) : error ? (
            <AlertTriangle className="h-3.5 w-3.5 text-amber-400" />
          ) : running ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin text-cyan-400" />
          ) : (
            <AlertTriangle className="h-3.5 w-3.5 text-slate-400" />
          )}

          <span className="font-medium">
            {ready
              ? `Caption CapRL Q6 พร้อม (${expected} ฉาก · ${selectedDensity} เฟรม/ฉาก)`
              : cancelled
              ? `ยกเลิกการคำนวณแล้ว — ใช้ Fast Search ได้`
              : interrupted
              ? `การทำงานถูกขัดจังหวะ (${completed}/${expected} ฉาก) — กดทำต่อได้`
              : error
              ? "Caption error — สลับใช้ Fast Search ได้"
              : unavailable
              ? "ยังไม่มี Caption (ใช้ Fast Search ได้)"
              : `กำลังสร้าง Caption: ฉากที่ ${completed}/${expected} (${selectedDensity} เฟรม/ฉาก)`}
          </span>
        </div>

        {running && (
          <div className="mt-0.5 space-y-1">
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-800/80">
              <div
                className="h-full rounded-full bg-gradient-to-r from-cyan-500 to-blue-500 transition-all duration-500"
                style={{ width: `${Math.max(4, percent)}%` }}
              />
            </div>
            <div className="flex items-center justify-between text-[9px] text-cyan-300/80">
              <span>{currentModel.includes("q6") ? "CapRL Q6 GPU Resident" : "Inference Worker"}</span>
              <span>{`ฉากที่ ${completed}/${expected} · วิเคราะห์ ${selectedDensity} เฟรม`}</span>
            </div>
          </div>
        )}
      </div>

      {/* 2. Detail Level Selector & Actions */}
      <div className="inline-flex items-center gap-1.5 rounded-lg border border-surfaceBorder bg-surface/90 px-2 py-1 shadow-sm backdrop-blur-sm">
        <Layers className="h-3 w-3 text-cyan-400" />
        <label htmlFor="caption-density-select" className="text-[10px] text-gray-400">
          ความละเอียด:
        </label>
        <select
          id="caption-density-select"
          value={selectedDensity}
          disabled={running || isSubmitting}
          onChange={(e) => setSelectedDensity(Number(e.target.value))}
          className="rounded border border-surfaceBorder/80 bg-slate-900 px-1.5 py-0.5 text-[10px] text-gray-200 focus:border-cyan-500 focus:outline-none focus:ring-1 focus:ring-cyan-500 disabled:opacity-50"
          title="จำนวนเฟรมที่สุ่มมาให้ CapRL Q6 ทำความเข้าใจฉาก (ยิ่งเยอะยิ่งเห็นคน/การเคลื่อนไหวครบ)"
        >
          <option value={4}>4 เฟรม (เร็ว - พื้นฐาน)</option>
          <option value={8}>8 เฟรม (แนะนำ - ละเอียด)</option>
          <option value={12}>12 เฟรม (ละเอียดพิเศษ)</option>
          <option value={16}>16 เฟรม (หนาแน่นสูงสุด)</option>
        </select>

        {/* Resume Button if Interrupted */}
        {interrupted && (
          <button
            type="button"
            onClick={() => handleRebuild(false)}
            disabled={running || isSubmitting}
            className="inline-flex items-center gap-1 rounded bg-amber-600/30 px-2 py-0.5 text-[10px] font-semibold text-amber-300 transition-colors hover:bg-amber-600/50 hover:text-amber-100 focus:outline-none focus:ring-1 focus:ring-amber-400 disabled:cursor-not-allowed disabled:opacity-40"
            title="ดำเนินการวิเคราะห์ต่อจากฉากที่ค้างไว้โดยไม่ต้องเริ่มใหม่ทั้งหมด"
          >
            <RefreshCw className={`h-2.5 w-2.5 ${isSubmitting ? "animate-spin" : ""}`} />
            <span>ทำต่อ (Resume)</span>
          </button>
        )}

        {/* Recompute Button */}
        <button
          type="button"
          onClick={() => handleRebuild(true)}
          disabled={running || isSubmitting}
          className="inline-flex items-center gap-1 rounded bg-cyan-600/20 px-2 py-0.5 text-[10px] font-semibold text-cyan-300 transition-colors hover:bg-cyan-600/30 hover:text-cyan-100 focus:outline-none focus:ring-1 focus:ring-cyan-400 disabled:cursor-not-allowed disabled:opacity-40"
          title="ประมวลผลโมเดล CapRL Q6 ใหม่ด้วยระดับความละเอียดที่เลือก"
        >
          <RefreshCw className={`h-2.5 w-2.5 ${running || isSubmitting ? "animate-spin" : ""}`} />
          <span>{running ? "กำลังคำนวณ..." : interrupted ? "เริ่มใหม่ทั้งหมด" : "คำนวณแคปชันใหม่"}</span>
        </button>

        {/* Cancel Button (appears when calculating) */}
        {running && (
          <button
            type="button"
            onClick={handleCancel}
            disabled={isCancelling}
            className="inline-flex items-center gap-1 rounded border border-rose-600/60 bg-rose-950/60 px-2 py-0.5 text-[10px] font-semibold text-rose-300 transition-colors hover:bg-rose-900/80 hover:text-rose-100 focus:outline-none focus:ring-1 focus:ring-rose-400 disabled:cursor-not-allowed disabled:opacity-50"
            title="ยกเลิกการสร้างแคปชันสำหรับวิดีโอนี้ทันที"
          >
            <XCircle className="h-2.5 w-2.5 text-rose-400" />
            <span>{isCancelling ? "กำลังยกเลิก..." : "ยกเลิก"}</span>
          </button>
        )}
      </div>


      {feedbackMsg && (
        <span className="text-[10px] text-cyan-400 animate-pulse">
          {feedbackMsg}
        </span>
      )}
    </div>
  );
};
