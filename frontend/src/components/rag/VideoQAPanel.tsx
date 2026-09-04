"use client";

import React, { useState } from "react";
import {
  MessageSquare,
  Send,
  Sparkles,
  Loader2,
  Clock,
  CheckCircle2,
  Bot,
  User,
  ExternalLink,
  HelpCircle,
} from "lucide-react";
import { apiClient, VideoQAResult } from "@/lib/api";

interface VideoQAPanelProps {
  videoId?: string;
  onSeekToTimestamp: (time: number) => void;
}

const SUGGESTED_PROMPTS = [
  "เกิดเหตุการณ์สำคัญอะไรขึ้นบ้างในคลิปนี้?",
  "มีคนทำกิจกรรมหรือขี่จักรยานหรือไม่?",
  "สรุปเหตุการณ์ทั้งหมดตั้งแต่ต้นจนจบ",
  "มีคนลุกขึ้นยืนหรือยกมือในห้องหรือไม่?",
];

export const VideoQAPanel: React.FC<VideoQAPanelProps> = ({
  videoId,
  onSeekToTimestamp,
}) => {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [chatLog, setChatLog] = useState<Array<{ q: string; res: VideoQAResult }>>([]);

  const handleAsk = async (promptText?: string) => {
    const q = promptText || question;
    if (!q.trim() || !videoId) return;

    setQuestion("");
    setLoading(true);

    try {
      const result = await apiClient.chatWithVideo(videoId, q);
      setChatLog((prev) => [...prev, { q, res: result }]);
    } catch (err) {
      console.error("Video-RAG QA failed:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="glass-panel rounded-2xl p-4 space-y-3.5 border border-surfaceBorder/80">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 rounded-lg bg-indigo-500/20 border border-indigo-500/30 flex items-center justify-center text-indigo-400">
            <Bot className="w-3.5 h-3.5" />
          </div>
          <h3 className="text-xs font-bold text-gray-200 uppercase tracking-wider">
            Video-RAG Grounded Intelligence
          </h3>
        </div>

        <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-cyan-950 text-cyan-300 border border-cyan-800">
          Qwen2.5-VL 7B Vision Grounded
        </span>
      </div>

      {/* Suggested Prompts Pill Tray */}
      <div className="flex items-center gap-1.5 overflow-x-auto pb-1 text-[11px]">
        <span className="text-gray-500 text-[10px] flex items-center gap-1 flex-shrink-0">
          <HelpCircle className="w-3 h-3 text-cyan-400" /> ลองถาม:
        </span>
        {SUGGESTED_PROMPTS.map((prompt, pIdx) => (
          <button
            key={pIdx}
            type="button"
            disabled={loading || !videoId}
            onClick={() => handleAsk(prompt)}
            className="px-2.5 py-1 rounded-full bg-surface/80 hover:bg-surfaceBorder text-gray-300 hover:text-cyan-200 text-[10px] border border-surfaceBorder whitespace-nowrap transition-colors disabled:opacity-50 flex-shrink-0"
          >
            {prompt}
          </button>
        ))}
      </div>

      {/* Conversational Messages Log */}
      <div className="space-y-3.5 max-h-[360px] overflow-y-auto pr-1">
        {chatLog.length === 0 && (
          <div className="py-8 text-center text-gray-400 space-y-2">
            <MessageSquare className="w-8 h-8 mx-auto text-indigo-400 opacity-50 animate-pulse" />
            <p className="text-xs text-gray-300 font-medium">ถามตอบอัจฉริยะเกี่ยวกับเหตุการณ์ในวิดีโอ</p>
            <p className="text-[11px] text-gray-500 max-w-xs mx-auto">
              ระบบใช้โมเดล Vision-Language ดึงภาพและจับคู่เหตุการณ์จริงพร้อมแสดงหลักฐานอ้างอิงช่วงเวลา
            </p>
          </div>
        )}

        {chatLog.map((item, idx) => (
          <div key={idx} className="space-y-2 text-xs">
            {/* User Message Bubble */}
            <div className="flex items-start gap-2 justify-end">
              <div className="bg-indigo-600/30 border border-indigo-500/40 rounded-2xl rounded-tr-none px-3.5 py-2 text-gray-100 max-w-[85%] shadow-sm">
                <p className="leading-relaxed">{item.q}</p>
              </div>
              <div className="w-6 h-6 rounded-full bg-indigo-600 text-white flex items-center justify-center flex-shrink-0 mt-0.5">
                <User className="w-3 h-3" />
              </div>
            </div>

            {/* AI Assistant Grounded Response */}
            <div className="flex items-start gap-2 justify-start">
              <div className="w-6 h-6 rounded-full bg-cyan-500/20 border border-cyan-500/40 text-cyan-300 flex items-center justify-center flex-shrink-0 mt-0.5">
                <Sparkles className="w-3 h-3" />
              </div>

              <div className="bg-surface/90 border border-surfaceBorder rounded-2xl rounded-tl-none p-3.5 text-gray-200 max-w-[90%] space-y-2.5 shadow-lg">
                <div className="flex items-center justify-between text-[10px] text-gray-400 border-b border-surfaceBorder/60 pb-1.5">
                  <span className="flex items-center gap-1 text-cyan-300 font-semibold">
                    <CheckCircle2 className="w-3 h-3 text-cyan-400" /> Grounded Synthesis
                  </span>
                  <span className="font-mono text-gray-500">{item.res.latency_ms} ms</span>
                </div>

                <p className="whitespace-pre-line leading-relaxed text-gray-200">
                  {item.res.answer}
                </p>

                {/* Visual Evidence Cards */}
                {item.res.grounded_moments && item.res.grounded_moments.length > 0 && (
                  <div className="pt-2 border-t border-surfaceBorder/60 space-y-1.5">
                    <span className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider block">
                      Evidence Moments:
                    </span>

                    <div className="flex flex-wrap gap-2">
                      {item.res.grounded_moments.map((g, gIdx) => (
                        <button
                          key={gIdx}
                          type="button"
                          onClick={() => onSeekToTimestamp(g.t_start)}
                          className="px-2.5 py-1.5 rounded-xl bg-cyan-950/80 hover:bg-cyan-900 text-cyan-200 border border-cyan-700/60 text-[11px] font-mono flex items-center gap-1.5 transition-all shadow-sm hover:scale-[1.02]"
                          title="Click to jump to this video timestamp"
                        >
                          <Clock className="w-3 h-3 text-cyan-400" />
                          <span>{g.t_start.toFixed(1)}s - {g.t_end.toFixed(1)}s</span>
                          <ExternalLink className="w-2.5 h-2.5 text-cyan-400 ml-0.5" />
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Input Form */}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          handleAsk();
        }}
        className="flex items-center gap-2 pt-2 border-t border-surfaceBorder"
      >
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="พิมพ์คำถามเกี่ยวกับภาพและเหตุการณ์ในคลิป..."
          disabled={loading || !videoId}
          className="flex-1 bg-surface border border-surfaceBorder rounded-xl px-3.5 py-2.5 text-xs text-white placeholder-gray-500 focus:outline-none focus:border-cyan-500 transition-colors"
        />
        <button
          type="submit"
          disabled={loading || !question.trim() || !videoId}
          className="px-4 py-2.5 rounded-xl bg-gradient-to-r from-indigo-600 to-cyan-600 hover:opacity-95 text-white font-semibold text-xs flex items-center gap-1.5 disabled:opacity-50 transition-all shadow-lg shadow-cyan-600/20"
        >
          {loading ? (
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
          ) : (
            <Send className="w-3.5 h-3.5" />
          )}
          <span>ส่ง</span>
        </button>
      </form>
    </div>
  );
};
