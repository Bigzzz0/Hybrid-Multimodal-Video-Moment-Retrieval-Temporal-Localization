"use client";

import React, { useState } from "react";
import { MessageSquare, Send, Sparkles, Loader2, Clock, CheckCircle2 } from "lucide-react";
import { apiClient, VideoQAResult } from "@/lib/api";

interface VideoQAPanelProps {
  videoId?: string;
  onSeekToTimestamp: (time: number) => void;
}

export const VideoQAPanel: React.FC<VideoQAPanelProps> = ({
  videoId,
  onSeekToTimestamp,
}) => {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [chatLog, setChatLog] = useState<Array<{ q: string; res: VideoQAResult }>>([]);

  const handleAsk = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim() || !videoId) return;

    const currentQ = question;
    setQuestion("");
    setLoading(true);

    try {
      const result = await apiClient.chatWithVideo(videoId, currentQ);
      setChatLog((prev) => [...prev, { q: currentQ, res: result }]);
    } catch (err) {
      console.error("Video-RAG QA failed:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="glass-panel rounded-2xl p-4 space-y-4 border border-surfaceBorder">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-200 uppercase tracking-wider flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-cyan-400" />
          Video-RAG Intelligence QA
        </h3>
        <span className="text-[11px] px-2 py-0.5 rounded-full bg-cyan-950 text-cyan-300 border border-cyan-800 font-mono">
          MiniCPM-V 2.6 Grounded
        </span>
      </div>

      {/* Chat Messages Log */}
      <div className="space-y-3 max-h-[350px] overflow-y-auto pr-1">
        {chatLog.length === 0 && (
          <div className="p-6 text-center text-gray-400 space-y-2">
            <MessageSquare className="w-8 h-8 mx-auto text-indigo-400 opacity-60" />
            <p className="text-xs text-gray-300">
              Ask anything about this video (e.g. "ผู้บรรยายพูดถึงหัวข้ออะไรบ้าง?", "สรุปเนื้อหาสำคัญ")
            </p>
          </div>
        )}

        {chatLog.map((item, idx) => (
          <div key={idx} className="space-y-2 text-xs">
            {/* User Question */}
            <div className="bg-surface/80 p-2.5 rounded-xl border border-surfaceBorder text-gray-200 font-medium">
              <span className="text-cyan-400 font-bold">Q: </span> {item.q}
            </div>

            {/* AI Grounded Answer */}
            <div className="bg-indigo-950/40 p-3 rounded-xl border border-indigo-800/40 text-gray-300 space-y-2 leading-relaxed">
              <div className="flex items-center justify-between text-[10px] text-gray-400">
                <span className="flex items-center gap-1 text-cyan-300 font-semibold">
                  <CheckCircle2 className="w-3 h-3 text-cyan-400" /> Grounded Synthesis
                </span>
                <span className="font-mono text-gray-500">{item.res.latency_ms} ms</span>
              </div>
              <p className="whitespace-pre-line">{item.res.answer}</p>

              {/* Citations / Timestamps */}
              {item.res.grounded_moments && item.res.grounded_moments.length > 0 && (
                <div className="pt-2 border-t border-indigo-900/60 flex flex-wrap items-center gap-2">
                  <span className="text-[10px] text-gray-400">Jump to Evidence:</span>
                  {item.res.grounded_moments.map((g, gIdx) => (
                    <button
                      key={gIdx}
                      type="button"
                      onClick={() => onSeekToTimestamp(g.t_start)}
                      className="px-2 py-0.5 rounded-md bg-cyan-900/60 hover:bg-cyan-800 text-cyan-200 border border-cyan-700/60 text-[10px] font-mono flex items-center gap-1 transition-colors"
                    >
                      <Clock className="w-2.5 h-2.5" />
                      {g.t_start}s - {g.t_end}s
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Question Input Form */}
      <form onSubmit={handleAsk} className="flex items-center gap-2 pt-2 border-t border-surfaceBorder">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a question about the video..."
          className="flex-1 bg-surface border border-surfaceBorder rounded-xl px-3.5 py-2.5 text-xs text-white placeholder-gray-500 focus:outline-none focus:border-cyan-500"
        />
        <button
          type="submit"
          disabled={loading || !question.trim() || !videoId}
          className="px-4 py-2.5 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white font-medium text-xs flex items-center gap-1.5 disabled:opacity-50 transition-colors shadow-lg shadow-cyan-600/20"
        >
          {loading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Send className="w-3.5 h-3.5" />}
          Ask
        </button>
      </form>
    </div>
  );
};
