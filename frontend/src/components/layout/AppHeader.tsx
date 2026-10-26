"use client";

import React, { useEffect, useState } from "react";
import { Video, Sun, Moon, Sliders, Command, Tv, Globe } from "lucide-react";
import { useLanguage } from "@/lib/language";

interface AppHeaderProps {
  isCinemaMode: boolean;
  onToggleCinemaMode: () => void;
  onOpenHotkeys: () => void;
  onOpenSettings: () => void;
}

export const AppHeader: React.FC<AppHeaderProps> = ({
  isCinemaMode,
  onToggleCinemaMode,
  onOpenHotkeys,
  onOpenSettings,
}) => {
  const [isDark, setIsDark] = useState<boolean>(true);
  const { language, setLanguage, t } = useLanguage();

  useEffect(() => {
    try {
      const isCurrentlyDark = document.documentElement.classList.contains("dark");
      setIsDark(isCurrentlyDark);
    } catch (_) {}
  }, []);

  const toggleTheme = (dark: boolean) => {
    setIsDark(dark);
    if (dark) {
      document.documentElement.classList.add("dark");
      localStorage.setItem("theme", "dark");
    } else {
      document.documentElement.classList.remove("dark");
      localStorage.setItem("theme", "light");
    }
  };

  return (
    <header className="sticky top-0 z-40 w-full border-b border-[var(--line)] bg-[var(--panel)] px-4 py-2.5 transition-colors">
      <div className="mx-auto flex max-w-[1680px] items-center justify-between gap-4">
        {/* Brand Logo & Title */}
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-[var(--accent)] text-white shadow-sm">
            <Video className="h-5 w-5 fill-current" />
          </div>
          <div>
            <h1 className="text-sm font-semibold tracking-tight text-[var(--text)] sm:text-base">
              {t("header.title")}
            </h1>
            <p className="text-[11px] text-[var(--text-muted)]">
              {t("header.subtitle")}
            </p>
          </div>
        </div>

        {/* Header Right Actions */}
        <div className="flex items-center gap-2 text-xs">
          {/* Cinema Mode */}
          <button
            type="button"
            onClick={onToggleCinemaMode}
            className={`flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 font-medium transition-colors ${
              isCinemaMode
                ? "border-[var(--accent)] bg-[var(--accent-subtle)] text-[var(--accent)]"
                : "border-[var(--line)] bg-[var(--card-bg)] text-[var(--text)] hover:border-[var(--line-strong)]"
            }`}
            title={t("header.cinemaModeTooltip")}
          >
            <Tv className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">{t("header.cinemaMode")}</span>
          </button>

          {/* Hotkeys */}
          <button
            type="button"
            onClick={onOpenHotkeys}
            className="flex items-center gap-1.5 rounded-lg border border-[var(--line)] bg-[var(--card-bg)] px-2.5 py-1.5 font-medium text-[var(--text)] transition-colors hover:border-[var(--line-strong)]"
            title={t("header.hotkeysTooltip")}
          >
            <Command className="h-3.5 w-3.5 text-[var(--accent)]" />
            <span className="hidden sm:inline">{t("header.hotkeys")}</span>
          </button>

          {/* Settings / Dev Panel */}
          <button
            type="button"
            onClick={onOpenSettings}
            className="flex items-center gap-1.5 rounded-lg border border-[var(--line)] bg-[var(--card-bg)] px-2.5 py-1.5 font-medium text-[var(--text)] transition-colors hover:border-[var(--line-strong)]"
            title={t("header.settingsTooltip")}
          >
            <Sliders className="h-3.5 w-3.5 text-[var(--accent)]" />
            <span>{t("header.settings")}</span>
          </button>

          {/* Language Switcher Pill [ TH | EN ] */}
          <div className="flex items-center rounded-lg border border-[var(--line)] bg-[var(--bg)] p-0.5">
            <button
              type="button"
              onClick={() => setLanguage("th")}
              className={`flex items-center gap-1 rounded-md px-2 py-1 text-[11px] font-medium transition-all ${
                language === "th"
                  ? "bg-[var(--panel)] text-[var(--accent)] shadow-sm font-bold"
                  : "text-[var(--text-muted)] hover:text-[var(--text)]"
              }`}
              title="สลับเป็นภาษาไทย"
            >
              <span>TH</span>
            </button>
            <button
              type="button"
              onClick={() => setLanguage("en")}
              className={`flex items-center gap-1 rounded-md px-2 py-1 text-[11px] font-medium transition-all ${
                language === "en"
                  ? "bg-[var(--panel)] text-[var(--accent)] shadow-sm font-bold"
                  : "text-[var(--text-muted)] hover:text-[var(--text)]"
              }`}
              title="Switch to English"
            >
              <span>EN</span>
            </button>
          </div>

          {/* Theme Toggle (Light / Dark Pill) */}
          <div className="flex items-center rounded-lg border border-[var(--line)] bg-[var(--bg)] p-0.5">
            <button
              type="button"
              onClick={() => toggleTheme(false)}
              className={`flex items-center gap-1 rounded-md px-2 py-1 text-[11px] font-medium transition-all ${
                !isDark
                  ? "bg-[var(--panel)] text-[var(--text)] shadow-sm font-semibold"
                  : "text-[var(--text-muted)] hover:text-[var(--text)]"
              }`}
              title={t("header.theme.lightTooltip")}
            >
              <Sun className="h-3 w-3 text-amber-500" />
              <span>{t("header.theme.light")}</span>
            </button>
            <button
              type="button"
              onClick={() => toggleTheme(true)}
              className={`flex items-center gap-1 rounded-md px-2 py-1 text-[11px] font-medium transition-all ${
                isDark
                  ? "bg-[var(--panel)] text-[var(--text)] shadow-sm font-semibold"
                  : "text-[var(--text-muted)] hover:text-[var(--text)]"
              }`}
              title={t("header.theme.darkTooltip")}
            >
              <Moon className="h-3 w-3 text-cyan-400" />
              <span>{t("header.theme.dark")}</span>
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};
