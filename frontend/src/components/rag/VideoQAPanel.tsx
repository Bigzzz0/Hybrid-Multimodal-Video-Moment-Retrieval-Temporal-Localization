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
  Film,
  Repeat,
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

  // Helper to render text with interactive clickable timecode pins
  const renderInteractiveAnswer = (text: string) => {
    const timecodeRegex = /(\[\d+\.?\d*\s*s?\s*-\s*\d+\.?\d*\s*s?\]|\b\d+\.?\d*\s*s\b)/gi;
    const parts = text.split(timecodeRegex);

    return parts.map((part, pIdx) => {
      const matchRange = part.match(/\[(\d+\.?\d*)\s*s?\s*-\s*(\d+\.?\d*)\s*s?\]/i);
      const matchSingle = part.match(/\b(\d+\.?\d*)\s*s\b/i);

      if (matchRange) {
        const tStart = parseFloat(matchRange[1]);
        const tEnd = parseFloat(matchRange[2]);
        return (
          <button
            key={pIdx}
            type="button"
            onClick={() => onSeekToTimestamp(tStart)}
            className="inline-flex items-center gap-1 px-1.5 py-0.5 mx-0.5 rounded-md bg-cyan-100/80 hover:bg-cyan-200/90 dark:bg-cyan-950/80 dark:hover:bg-cyan-900 border border-cyan-300 dark:border-cyan-500/50 text-cyan-800 dark:text-cyan-300 font-mono font-bold text-[11px] shadow-xs hover:scale-105 transition-transform align-baseline"
            title={`Seek to ${tStart}s - ${tEnd}s`}
          >
            <Clock className="w-2.5 h-2.5 text-cyan-600 dark:text-cyan-400" />
            <span>{tStart.toFixed(1)}s-{tEnd.toFixed(1)}s</span>
          </button>
        );
      } else if (matchSingle) {
        const t = parseFloat(matchSingle[1]);
        if (t > 0) {
          return (
            <button
              key={pIdx}
              type="button"
              onClick={() => onSeekToTimestamp(t)}
              className="inline-flex items-center gap-1 px-1.5 py-0.5 mx-0.5 rounded-md bg-indigo-100/80 hover:bg-indigo-200/90 dark:bg-indigo-950/80 dark:hover:bg-indigo-900 border border-indigo-300 dark:border-indigo-500/50 text-indigo-800 dark:text-indigo-300 font-mono font-bold text-[11px] shadow-xs hover:scale-105 transition-transform align-baseline"
              title={`Seek to ${t}s`}
            >
              <Clock className="w-2.5 h-2.5 text-indigo-600 dark:text-indigo-400" />
              <span>{t.toFixed(1)}s</span>
            </button>
          );
        }
      }
      return <span key={pIdx}>{part}</span>;
    });
  };

  return (
    <div className="rounded-2xl p-4 space-y-3.5 bg-[var(--card-bg)] border border-[var(--line)] shadow-xs transition-colors">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 rounded-lg bg-indigo-500/15 border border-indigo-500/30 flex items-center justify-center text-indigo-600 dark:text-indigo-400">
            <Bot className="w-3.5 h-3.5" />
          </div>
          <h3 className="text-xs font-bold text-[var(--text)] uppercase tracking-wider">
            Video-RAG Grounded Intelligence
          </h3>
        </div>

        <span className="text-[10px] font-mono font-medium px-2 py-0.5 rounded-full bg-cyan-100 dark:bg-cyan-950/80 text-cyan-800 dark:text-cyan-300 border border-cyan-300/80 dark:border-cyan-800">
          CapRL Q6 Vision Grounded
        </span>
      </div>

      {/* Suggested Prompts Pill Tray */}
      <div className="flex items-center gap-1.5 overflow-x-auto pb-1 text-[11px]">
        <span className="text-[var(--text-muted)] text-[10px] font-semibold flex items-center gap-1 flex-shrink-0">
          <HelpCircle className="w-3 h-3 text-[var(--accent)]" /> ลองถาม:
        </span>
        {SUGGESTED_PROMPTS.map((prompt, pIdx) => (
          <button
            key={pIdx}
            type="button"
            disabled={loading || !videoId}
            onClick={() => handleAsk(prompt)}
            className="px-2.5 py-1 rounded-full bg-[var(--panel)] hover:bg-[var(--card-hover)] text-[var(--text)] hover:text-[var(--accent)] text-[10px] font-medium border border-[var(--line)] whitespace-nowrap transition-colors disabled:opacity-40 flex-shrink-0 shadow-xs"
          >
            {prompt}
          </button>
        ))}
      </div>

      {/* Conversational Messages Log */}
      <div className="space-y-3.5 max-h-[360px] overflow-y-auto pr-1">
        {chatLog.length === 0 && (
          <div className="py-8 text-center space-y-2">
            <MessageSquare className="w-8 h-8 mx-auto text-[var(--accent)] opacity-60 animate-pulse" />
            <p className="text-xs text-[var(--text)] font-semibold">ถามตอบอัจฉริยะเกี่ยวกับเหตุการณ์ในวิดีโอ</p>
            <p className="text-[11px] text-[var(--text-muted)] max-w-xs mx-auto leading-relaxed">
              ระบบใช้โมเดล Vision-Language ดึงภาพและจับคู่เหตุการณ์จริงพร้อมแสดงหลักฐานอ้างอิงช่วงเวลา
            </p>
          </div>
        )}

        {chatLog.map((item, idx) => (
          <div key={idx} className="space-y-2 text-xs">
            {/* User Message Bubble */}
            <div className="flex items-start gap-2 justify-end">
              <div className="bg-[var(--accent)] text-white rounded-2xl rounded-tr-none px-3.5 py-2 max-w-[85%] shadow-sm">
                <p className="leading-relaxed font-normal">{item.q}</p>
              </div>
              <div className="w-6 h-6 rounded-full bg-[var(--accent)] text-white flex items-center justify-center flex-shrink-0 mt-0.5 shadow-xs">
                <User className="w-3 h-3" />
              </div>
            </div>

            {/* AI Assistant Grounded Response */}
            <div className="flex items-start gap-2 justify-start">
              <div className="w-6 h-6 rounded-full bg-cyan-500/20 border border-cyan-500/40 text-cyan-600 dark:text-cyan-300 flex items-center justify-center flex-shrink-0 mt-0.5 shadow-xs">
                <Sparkles className="w-3 h-3" />
              </div>

              <div className="bg-[var(--panel)] border border-[var(--line)] rounded-2xl rounded-tl-none p-3.5 text-[var(--text)] max-w-[92%] space-y-2.5 shadow-md">
                <div className="flex items-center justify-between text-[10px] text-[var(--text-muted)] border-b border-[var(--line)] pb-1.5">
                  <span className="flex items-center gap-1 text-cyan-700 dark:text-cyan-300 font-bold">
                    <CheckCircle2 className="w-3 h-3 text-cyan-600 dark:text-cyan-400" /> Grounded Synthesis
                  </span>
                  <span className="font-mono text-[var(--text-muted)]">{item.res.latency_ms} ms</span>
                </div>

                <div className="whitespace-pre-line leading-relaxed text-[var(--text)]">
                  {renderInteractiveAnswer(item.res.answer)}
                </div>

                {/* Visual Evidence Cards */}
                {item.res.grounded_moments && item.res.grounded_moments.length > 0 && (
                  <div className="pt-2 border-t border-[var(--line)] space-y-1.5">
                    <span className="text-[10px] font-semibold text-[var(--text-muted)] uppercase tracking-wider block">
                      Evidence Moments:
                    </span>

                    <div className="flex flex-wrap gap-2">
                      {item.res.grounded_moments.map((g, gIdx) => (
                        <button
                          key={gIdx}
                          type="button"
                          onClick={() => onSeekToTimestamp(g.t_start)}
                          className="px-2.5 py-1.5 rounded-xl bg-cyan-100/90 hover:bg-cyan-200 dark:bg-cyan-950/80 dark:hover:bg-cyan-900 text-cyan-900 dark:text-cyan-200 border border-cyan-300 dark:border-cyan-700/60 text-[11px] font-mono font-medium flex items-center gap-1.5 transition-all shadow-xs hover:scale-[1.02]"
                          title="Click to jump to this video timestamp"
                        >
                          <Clock className="w-3 h-3 text-cyan-600 dark:text-cyan-400" />
                          <span>{g.t_start.toFixed(1)}s - {g.t_end.toFixed(1)}s</span>
                          <ExternalLink className="w-2.5 h-2.5 text-cyan-600 dark:text-cyan-400 ml-0.5" />
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
        className="flex items-center gap-2 pt-2 border-t border-[var(--line)]"
      >
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="พิมพ์คำถามเกี่ยวกับภาพและเหตุการณ์ในคลิป..."
          disabled={loading || !videoId}
          className="flex-1 bg-[var(--panel)] border border-[var(--line)] rounded-xl px-3.5 py-2.5 text-xs text-[var(--text)] placeholder:text-[var(--text-muted)] focus:outline-none focus:border-[var(--accent)] focus:ring-1 focus:ring-[var(--accent)] shadow-xs transition-colors"
        />
        <button
          type="submit"
          disabled={loading || !question.trim() || !videoId}
          className="px-4 py-2.5 rounded-xl bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white font-semibold text-xs flex items-center gap-1.5 disabled:opacity-40 transition-all shadow-sm"
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

