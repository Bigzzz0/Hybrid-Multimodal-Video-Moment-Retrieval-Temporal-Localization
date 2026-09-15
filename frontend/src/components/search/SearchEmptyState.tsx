"use client";

import React from "react";
import { Film, SearchX, Sparkles } from "lucide-react";

export const SearchEmptyState: React.FC<{ state: "initial" | "no_match" | "reindex"; onReindex?: () => void }> = ({ state, onReindex }) => {
  const copy = state === "initial"
    ? { Icon: Sparkles, title: "พร้อมค้นหาช่วงเวลา", body: "พิมพ์คำค้นภาษาไทยหรืออังกฤษ แล้วเลือกผลลัพธ์เพื่อ seek ไปยังเหตุการณ์" }
    : state === "reindex"
    ? { Icon: Film, title: "ต้องสร้าง visual index v2 ใหม่", body: "วิดีโอนี้ยังใช้ดัชนีรุ่นเก่า จึงยังค้นหาอย่างถูกต้องไม่ได้" }
    : { Icon: SearchX, title: "ไม่พบช่วงเวลาที่ตรงกัน", body: "ลองใช้คำกว้างขึ้น หรือตรวจสอบคำค้นและวิดีโอที่เลือก" };
  const Icon = copy.Icon;
  return (
    <div className="glass-panel rounded-2xl border border-surfaceBorder/80 p-8 text-center text-gray-400">
      <Icon className="mx-auto mb-2 h-8 w-8 text-cyan-400 opacity-70" aria-hidden="true" />
      <p className="font-semibold text-gray-200">{copy.title}</p>
      <p className="mt-1 text-xs text-gray-500">{copy.body}</p>
      {state === "reindex" && onReindex && <button type="button" onClick={onReindex} className="mt-4 min-h-9 rounded-lg border border-cyan-600/60 px-3 py-1.5 text-xs font-semibold text-cyan-200 hover:bg-cyan-900/30 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400">เปิด uploader</button>}
    </div>
  );
};
