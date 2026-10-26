"use client";

import React, { useEffect } from "react";
import { X, Command, Play, Film, FastForward } from "lucide-react";
import { useLanguage, Language } from "@/lib/language";

interface ShortcutModalProps {
  isOpen: boolean;
  onClose: () => void;
}

interface ShortcutItem {
  keys: string[];
  description: string;
  category: "playback" | "navigation" | "studio";
}

const SHORTCUTS_DATA: Record<Language, ShortcutItem[]> = {
  th: [
    { keys: ["Space"], description: "เล่น / หยุดเล่นชั่วคราว (Play / Pause)", category: "playback" },
    { keys: ["J"], description: "ย้อนหลัง 5 วินาที", category: "playback" },
    { keys: ["K"], description: "หยุดเล่นชั่วคราว", category: "playback" },
    { keys: ["L"], description: "เดินหน้า 5 วินาที", category: "playback" },
    { keys: [","], description: "ถอยหลัง 1 เฟรม", category: "playback" },
    { keys: ["."], description: "เดินหน้า 1 เฟรม", category: "playback" },
    { keys: ["R"], description: "สลับโหมดวนซ้ำช่วงเหตุการณ์ (A-B Moment Loop)", category: "playback" },
    { keys: ["["], description: "กระโดดไปจุดเริ่มต้นเหตุการณ์ (t_start)", category: "navigation" },
    { keys: ["]"], description: "กระโดดไปจุดสิ้นสุดเหตุการณ์ (t_end)", category: "navigation" },
    { keys: ["Ctrl", "K"], description: "โฟกัสช่องค้นหาเหตุการณ์ด้วยภาษาธรรมชาติ", category: "navigation" },
    { keys: ["/"], description: "โฟกัสช่องค้นหาด่วน", category: "navigation" },
    { keys: ["T"], description: "สลับโหมด Cinema Theater Mode (เต็มหน้าต่าง)", category: "studio" },
    { keys: ["?"], description: "เปิดหน้าต่างสรุปปุ่มลัดคีย์บอร์ดทั้งหมด", category: "studio" },
  ],
  en: [
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
    { keys: ["?"], description: "Open keyboard shortcuts cheat sheet", category: "studio" },
  ],
};

export const ShortcutModal: React.FC<ShortcutModalProps> = ({ isOpen, onClose }) => {
  const { language, t } = useLanguage();
  const shortcuts = SHORTCUTS_DATA[language] || SHORTCUTS_DATA.th;

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
    <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="app-panel border border-[var(--line)] rounded-2xl max-w-lg w-full p-6 shadow-2xl space-y-5 animate-in fade-in zoom-in-95 duration-200 bg-[var(--panel)] text-[var(--text)]">
        {/* Modal Header */}
        <div className="flex items-center justify-between border-b border-[var(--line)] pb-3">
          <div className="flex items-center gap-2 text-[var(--accent)]">
            <Command className="w-5 h-5" />
            <h3 className="text-sm font-bold text-[var(--text)] tracking-wide">
              {t("modal.shortcutsTitle")}
            </h3>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-[var(--line)]/50 text-[var(--muted)] hover:text-[var(--text)] transition-colors cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Shortcuts Grid */}
        <div className="space-y-4 max-h-[60vh] overflow-y-auto pr-1">
          {/* Playback Controls */}
          <div>
            <h4 className="text-[11px] font-mono font-bold text-[var(--accent)] uppercase tracking-wider mb-2 flex items-center gap-1.5">
              <Play className="w-3 h-3" /> {t("modal.playbackSection")}
            </h4>
            <div className="space-y-1.5">
              {shortcuts.filter((s) => s.category === "playback").map((s, idx) => (
                <div
                  key={idx}
                  className="flex items-center justify-between p-2 rounded-xl bg-[var(--bg)] border border-[var(--line)] text-xs"
                >
                  <span className="text-[var(--text)]">{s.description}</span>
                  <div className="flex items-center gap-1">
                    {s.keys.map((k, kIdx) => (
                      <kbd
                        key={kIdx}
                        className="px-2 py-0.5 rounded-md bg-[var(--panel)] text-[11px] font-mono font-bold text-[var(--text)] border border-[var(--line)] shadow-sm"
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
            <h4 className="text-[11px] font-mono font-bold text-[var(--accent)] uppercase tracking-wider mb-2 flex items-center gap-1.5">
              <FastForward className="w-3 h-3" /> {t("modal.navSection")}
            </h4>
            <div className="space-y-1.5">
              {shortcuts.filter((s) => s.category === "navigation").map((s, idx) => (
                <div
                  key={idx}
                  className="flex items-center justify-between p-2 rounded-xl bg-[var(--bg)] border border-[var(--line)] text-xs"
                >
                  <span className="text-[var(--text)]">{s.description}</span>
                  <div className="flex items-center gap-1">
                    {s.keys.map((k, kIdx) => (
                      <kbd
                        key={kIdx}
                        className="px-2 py-0.5 rounded-md bg-[var(--panel)] text-[11px] font-mono font-bold text-[var(--text)] border border-[var(--line)] shadow-sm"
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
            <h4 className="text-[11px] font-mono font-bold text-emerald-600 dark:text-emerald-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
              <Film className="w-3 h-3" /> {t("modal.studioSection")}
            </h4>
            <div className="space-y-1.5">
              {shortcuts.filter((s) => s.category === "studio").map((s, idx) => (
                <div
                  key={idx}
                  className="flex items-center justify-between p-2 rounded-xl bg-[var(--bg)] border border-[var(--line)] text-xs"
                >
                  <span className="text-[var(--text)]">{s.description}</span>
                  <div className="flex items-center gap-1">
                    {s.keys.map((k, kIdx) => (
                      <kbd
                        key={kIdx}
                        className="px-2 py-0.5 rounded-md bg-[var(--panel)] text-[11px] font-mono font-bold text-[var(--text)] border border-[var(--line)] shadow-sm"
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
        <div className="pt-2 text-center text-[11px] text-[var(--muted)] font-mono">
          {language === "th" ? (
            <>กด <kbd className="px-1.5 py-0.5 bg-[var(--bg)] rounded border border-[var(--line)]">ESC</kbd> เพื่อปิดหน้าต่าง</>
          ) : (
            <>Press <kbd className="px-1.5 py-0.5 bg-[var(--bg)] rounded border border-[var(--line)]">ESC</kbd> to close</>
          )}
        </div>
      </div>
    </div>
  );
};
