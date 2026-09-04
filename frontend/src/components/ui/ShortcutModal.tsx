"use client";

import React, { useEffect } from "react";
import { X, Command, Play, Repeat, Search, Film, Download, FastForward } from "lucide-react";

interface ShortcutModalProps {
  isOpen: boolean;
  onClose: () => void;
}

interface ShortcutItem {
  keys: string[];
  description: string;
  category: "playback" | "navigation" | "studio";
}

const SHORTCUTS: ShortcutItem[] = [
  { keys: ["Space"], description: "Play / Pause playback", category: "playback" },
  { keys: ["J"], description: "Rewind 5 seconds", category: "playback" },
  { keys: ["K"], description: "Pause playback", category: "playback" },
  { keys: ["L"], description: "Fast-forward 5 seconds", category: "playback" },
  { keys: [","], description: "Step backward 1 frame", category: "playback" },
  { keys: ["."], description: "Step forward 1 frame", category: "playback" },
  { keys: ["R"], description: "Toggle A-B Moment Looping", category: "playback" },
  { keys: ["["], description: "Jump to start of current moment (t_start)", category: "navigation" },
  { keys: ["]"], description: "Jump to end of current moment (t_end)", category: "navigation" },
  { keys: ["Ctrl", "K"], description: "Focus Spotlight Natural Language Search", category: "navigation" },
  { keys: ["/"], description: "Quick search focus", category: "navigation" },
  { keys: ["T"], description: "Toggle Cinema Theater Mode (Full Width)", category: "studio" },
  { keys: ["M"], description: "Export current active moment clip", category: "studio" },
  { keys: ["?"], description: "Open keyboard shortcuts cheat sheet", category: "studio" },
];

export const ShortcutModal: React.FC<ShortcutModalProps> = ({ isOpen, onClose }) => {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isOpen) {
        onClose();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-md flex items-center justify-center p-4">
      <div className="glass-panel border border-surfaceBorder/80 rounded-2xl max-w-lg w-full p-6 shadow-2xl space-y-5 animate-in fade-in zoom-in-95 duration-200">
        {/* Modal Header */}
        <div className="flex items-center justify-between border-b border-surfaceBorder/60 pb-3">
          <div className="flex items-center gap-2 text-cyan-400">
            <Command className="w-5 h-5" />
            <h3 className="text-sm font-bold text-white tracking-wide">
              Pro Studio Keyboard Shortcuts
            </h3>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-surface text-gray-400 hover:text-white transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Shortcuts Grid */}
        <div className="space-y-4 max-h-[60vh] overflow-y-auto pr-1">
          {/* Playback Controls */}
          <div>
            <h4 className="text-[11px] font-mono font-bold text-cyan-400/80 uppercase tracking-wider mb-2 flex items-center gap-1.5">
              <Play className="w-3 h-3" /> Playback & Shuttle
            </h4>
            <div className="space-y-1.5">
              {SHORTCUTS.filter((s) => s.category === "playback").map((s, idx) => (
                <div
                  key={idx}
                  className="flex items-center justify-between p-2 rounded-xl bg-surface/60 border border-surfaceBorder/40 text-xs"
                >
                  <span className="text-gray-300">{s.description}</span>
                  <div className="flex items-center gap-1">
                    {s.keys.map((k, kIdx) => (
                      <kbd
                        key={kIdx}
                        className="px-2 py-0.5 rounded-md bg-surfaceBorder/80 text-[11px] font-mono font-bold text-white border border-white/10 shadow-sm"
                      >
                        {k}
                      </kbd>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Navigation Controls */}
          <div>
            <h4 className="text-[11px] font-mono font-bold text-indigo-400/80 uppercase tracking-wider mb-2 flex items-center gap-1.5">
              <FastForward className="w-3 h-3" /> Moment & Timeline Navigation
            </h4>
            <div className="space-y-1.5">
              {SHORTCUTS.filter((s) => s.category === "navigation").map((s, idx) => (
                <div
                  key={idx}
                  className="flex items-center justify-between p-2 rounded-xl bg-surface/60 border border-surfaceBorder/40 text-xs"
                >
                  <span className="text-gray-300">{s.description}</span>
                  <div className="flex items-center gap-1">
                    {s.keys.map((k, kIdx) => (
                      <kbd
                        key={kIdx}
                        className="px-2 py-0.5 rounded-md bg-surfaceBorder/80 text-[11px] font-mono font-bold text-white border border-white/10 shadow-sm"
                      >
                        {k}
                      </kbd>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Studio Controls */}
          <div>
            <h4 className="text-[11px] font-mono font-bold text-emerald-400/80 uppercase tracking-wider mb-2 flex items-center gap-1.5">
              <Film className="w-3 h-3" /> Studio Modes & Export
            </h4>
            <div className="space-y-1.5">
              {SHORTCUTS.filter((s) => s.category === "studio").map((s, idx) => (
                <div
                  key={idx}
                  className="flex items-center justify-between p-2 rounded-xl bg-surface/60 border border-surfaceBorder/40 text-xs"
                >
                  <span className="text-gray-300">{s.description}</span>
                  <div className="flex items-center gap-1">
                    {s.keys.map((k, kIdx) => (
                      <kbd
                        key={kIdx}
                        className="px-2 py-0.5 rounded-md bg-surfaceBorder/80 text-[11px] font-mono font-bold text-white border border-white/10 shadow-sm"
                      >
                        {k}
                      </kbd>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="pt-2 text-center text-[11px] text-gray-500 font-mono">
          Press <kbd className="px-1.5 py-0.5 bg-surface rounded border border-white/10">ESC</kbd> to close
        </div>
      </div>
    </div>
  );
};
