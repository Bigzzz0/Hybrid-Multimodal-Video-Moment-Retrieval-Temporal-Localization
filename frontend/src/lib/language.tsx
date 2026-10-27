"use client";

import React, { createContext, useContext, useState, useEffect } from "react";

export type Language = "th" | "en";

interface LanguageContextType {
  language: Language;
  setLanguage: (lang: Language) => void;
  t: (key: string, defaultText?: string) => string;
}

const translations: Record<Language, Record<string, string>> = {
  th: {
    // Header
    "header.title": "Moment Search",
    "header.subtitle": "ค้นหาเหตุการณ์ในวิดีโอ CCTV ด้วยภาษาธรรมชาติ",
    "header.cinemaMode": "Cinema Mode",
    "header.cinemaModeTooltip": "ขยายพื้นที่เครื่องเล่นวิดีโอ (T)",
    "header.hotkeys": "Hotkeys",
    "header.hotkeysTooltip": "ดูปุ่มลัดคีย์บอร์ดทั้งหมด (?)",
    "header.settings": "ตั้งค่า",
    "header.settingsTooltip": "ตั้งค่าระบบและดู Dev Telemetry",
    "header.theme.light": "สว่าง",
    "header.theme.dark": "มืด",
    "header.theme.lightTooltip": "เปลี่ยนเป็นธีมสว่าง",
    "header.theme.darkTooltip": "เปลี่ยนเป็นธีมมืด",
    "header.language": "ภาษา",

    // Sidebar
    "sidebar.title": "คลังวิดีโอ",
    "sidebar.import": "นำเข้าวิดีโอ",
    "sidebar.filterLabel": "กรองชื่อไฟล์หรือแท็ก",
    "sidebar.searchPlaceholder": "ค้นหาชื่อไฟล์...",
    "sidebar.folders": "โฟลเดอร์ / กล้อง",
    "sidebar.cat.all": "ทั้งหมด",
    "sidebar.cat.gate": "ประตู / ทางเข้า",
    "sidebar.cat.street": "ถนน / ทางแยก",
    "sidebar.cat.parking": "ลานจอดรถ",
    "sidebar.cat.classroom": "ห้องเรียน",
    "sidebar.status": "สถานะ",
    "sidebar.indexed": "Indexed",
    "sidebar.processing": "กำลังประมวลผล",
    "sidebar.videos": "วิดีโอ",
    "sidebar.noVideos": "ไม่พบวิดีโอที่ค้นหา",
    "sidebar.delete": "ลบวิดีโอ",

    // Search Section
    "search.placeholder": "ค้นหาการกระทำทางกายภาพ เช่น คนปั่นจักรยาน, รถยนต์เลี้ยวซ้าย...",
    "search.clearTooltip": "ล้างข้อความ",
    "search.button": "ค้นหาช่วงเวลา",
    "search.searching": "กำลังค้นหา...",
    "search.scope": "ขอบเขต",
    "search.scope.video": "วิดีโอนี้",
    "search.scope.folder": "โฟลเดอร์นี้",
    "search.scope.all": "ทุกวิดีโอ",
    "search.mode": "โหมด",
    "search.density": "ความละเอียด",
    "search.density.4": "4 เฟรม (Fast - Base)",
    "search.density.8": "8 เฟรม (แนะนำ - ละเอียด)",
    "search.density.12": "12 เฟรม (ละเอียดพิเศษ)",
    "search.density.16": "16 เฟรม (หนาแน่นสูงสุด)",
    "search.rebuildCaptions": "สร้างแคปชั่นใหม่",
    "search.rebuilding": "กำลังสร้าง...",
    "search.rebuildTooltip": "ประมวลผลโมเดล CapRL Q6 ใหม่ด้วยระดับความละเอียดที่เลือก",
    "search.cancel": "✕ ยกเลิก",
    "search.cancelling": "กำลังยกเลิก...",
    "search.cancelTooltip": "ยกเลิกการสร้างแคปชั่น",
    "search.cameraTime": "เวลากล้อง: ทั้งวัน",
    "search.captionWarning": "Caption ใช้ไม่ได้ในวิดีโอนี้ ระบบสลับไปใช้ Fast Search ให้อัตโนมัติ",
    "search.captionCancelled": "ยกเลิกการสร้าง Caption ในวิดีโอนี้แล้ว ระบบสลับไปใช้ Fast Search ให้โดยอัตโนมัติ",
    "search.examples": "ตัวอย่างคำค้น",
    "search.recent": "ค้นหาล่าสุด",
    "search.tab.transit": "การสัญจร",
    "search.tab.classroom": "ห้องเรียน",
    "search.tab.vehicle": "ยานพาหนะ",
    "search.tab.action": "ท่าทาง",

    // Player
    "player.heatmap": "Visual Relevance Heatmap",
    "player.camPreview": "CAM-03 · ภาพตัวอย่าง",
    "player.cameraTime": "เวลากล้อง",
    "player.exportClip": "ส่งออกคลิปช่วงนี้",
    "player.exporting": "กำลังส่งออก...",
    "player.stepBack": "ถอยหลัง 1 เฟรม (,)",
    "player.stepForward": "เดินหน้า 1 เฟรม (.)",
    "player.rewind": "ถอย 5 วินาที (J)",
    "player.mute": "ปิดเสียง",
    "player.unmute": "เปิดเสียง",
    "player.loop": "วนซ้ำช่วงเหตุการณ์ (R)",
    "player.fullscreen": "เต็มจอ",
    "player.customBoundary": "ปรับขอบเขตช่วงเวลาแบบกำหนดเอง",
    "player.resetBoundary": "รีเซ็ตขอบเขตเดิม",
    "player.noVideo": "ยังไม่ได้เลือกวิดีโอ",
    "player.selectPrompt": "เลือกวิดีโอจากคลังทางซ้าย หรือคลิก '+ นำเข้าวิดีโอ' เพื่อเริ่มต้น",
    "player.hideMask": "ซ่อน SAM Mask",
    "player.showMask": "แสดง SAM Mask",

    // Results Panel
    "results.foundMoments": "ช่วงเวลาที่พบ",
    "results.ragAI": "Video-RAG AI",
    "results.scenes": "ฉาก & แคปชัน",
    "results.found": "พบ",
    "results.momentsCount": "ช่วงเวลา",
    "results.searchIn": "ค้นหา",
    "results.inVideo": "ใน",
    "results.allActions": "ทุกการกระทำ",
    "results.videoFallback": "วิดีโอ",
    "results.rankedBy": "เรียงตามความตรง",
    "results.noMoments": "ยังไม่พบช่วงเวลา",
    "results.noMomentsPrompt": "พิมพ์คำค้นหาในช่องค้นหา แล้วกด 'ค้นหาช่วงเวลา'",
    "results.cameraTime": "เวลากล้อง",
    "results.detectedMoment": "ช่วงเหตุการณ์ที่ตรวจพบ",
    "results.copy": "คัดลอก",
    "results.copied": "คัดลอกแล้ว",
    "results.copyTooltip": "คัดลอกเวลา",
    "results.context": "ดูบริบท",
    "results.contextTooltip": "ขยายช่วงและดูบริบทของฉาก",
    "results.contextWindow": "ช่วงเวลาบริบท",
    "results.associatedScene": "ฉากที่ตรวจพบ",
    "results.openInScenes": "เปิดในแท็บฉาก & แคปชัน",
    "results.sceneCaption": "คำบรรยายบริบทของฉาก",
    "results.noSceneCaption": "ยังไม่มีคำบรรยายสำหรับฉากนี้",
    "results.loadingContext": "กำลังโหลดข้อมูลบริบท...",
    "results.details": "รายละเอียด",
    "results.export": "ส่งออก",
    "results.exporting": "กำลังส่งออก...",
    "results.exportAll": "ส่งออกทุกช่วง",
    "results.exportingAll": "กำลังส่งออกทุกช่วง...",
    "results.saveSearch": "บันทึกคำค้น",
    "results.savedSearch": "บันทึกเรียบร้อย",
    "results.savedSearches": "คำค้นที่บันทึกไว้",
    "results.deleteSearchTooltip": "ลบคำค้นนี้",

    // Settings Modal
    "settings.title": "Developer & System Telemetry Panel",
    "settings.subtitle": "Hardware metrics, Vector Storage stats, AI Model registry & execution logs",
    "settings.tab.preferences": "การตั้งค่าทั่วไป & ภาษา",
    "settings.tab.hardware": "Hardware & GPU Acceleration",
    "settings.tab.storage": "LanceDB Vector Storage",
    "settings.tab.models": "AI Models & Pipeline Config",
    "settings.tab.logs": "Live Backend Logs",
    "settings.languageCard.title": "ภาษาของระบบ (Interface Language)",
    "settings.languageCard.desc": "เลือกระหว่างภาษาไทย (คงคำศัพท์เทคนิคเป็นภาษาอังกฤษ) หรือภาษาอังกฤษเต็มรูปแบบ",
    "settings.themeCard.title": "ธีมการแสดงผล (Display Theme)",
    "settings.themeCard.desc": "สลับโหมดสว่างหรือโหมดมืดที่สบายตา",
    "settings.searchModeCard.title": "โหมดการค้นหาเริ่มต้น (Default Search Mode)",
    "settings.searchModeCard.desc": "Fast (SigLIP2 Pure Visual) หรือ Accurate (SigLIP2 + CapRL Q6 Captions)",
    "settings.densityCard.title": "ความหนาแน่นเฟรมเริ่มต้น (Keyframe Density)",
    "settings.densityCard.desc": "จำนวนเฟรมเฉลี่ยต่อฉากสำหรับสร้างภาพตัวอย่างและคำบรรยาย",
    "settings.langThai": "ภาษาไทย (TH)",
    "settings.langEnglish": "English (EN)",
    "settings.liveTelemetry": "Live Telemetry",
    "settings.autoRefresh": "อัตโนมัติ (3วิ)",
    "settings.paused": "หยุดชั่วคราว",
    "settings.manualRefresh": "รีเฟรชข้อมูล",

    // Modals
    "modal.deleteTitle": "ลบวิดีโอออกจากระบบ",
    "modal.deleteDesc": "Permanently remove video and vector indices",
    "modal.deleteConfirm": "คุณแน่ใจหรือไม่ว่าต้องการลบวิดีโอ",
    "modal.deleteWarning": "⚠️ การลบนี้จะลบเวกเตอร์ภาพ SigLIP 2, คำบรรยาย CapRL Q6, ไฟล์คีย์เฟรมและไฟล์วิดีโอต้นฉบับออกจากระบบอย่างถาวร",
    "modal.cancel": "ยกเลิก",
    "modal.delete": "ยืนยันการลบ",
    "modal.deleting": "กำลังลบ...",
    "modal.uploadTitle": "นำเข้าวิดีโอเข้าสู่คลัง",
    "modal.shortcutsTitle": "Pro Studio Keyboard Shortcuts",
    "modal.playbackSection": "Playback & Shuttle",
    "modal.navSection": "Moment & Timeline Navigation",
    "modal.studioSection": "Studio Modes & Export",
    "modal.pressEsc": "กด ESC เพื่อปิดหน้าต่าง",
  },

  en: {
    // Header
    "header.title": "Moment Search",
    "header.subtitle": "Natural Language CCTV Video Moment Retrieval",
    "header.cinemaMode": "Cinema Mode",
    "header.cinemaModeTooltip": "Expand video player viewport (T)",
    "header.hotkeys": "Hotkeys",
    "header.hotkeysTooltip": "View all keyboard shortcuts (?)",
    "header.settings": "Settings",
    "header.settingsTooltip": "System settings and Dev Telemetry",
    "header.theme.light": "Light",
    "header.theme.dark": "Dark",
    "header.theme.lightTooltip": "Switch to Light mode",
    "header.theme.darkTooltip": "Switch to Dark mode",
    "header.language": "Language",

    // Sidebar
    "sidebar.title": "Video Library",
    "sidebar.import": "Ingest Video",
    "sidebar.filterLabel": "Filter filename or tag",
    "sidebar.searchPlaceholder": "Search filename...",
    "sidebar.folders": "Folders / Cameras",
    "sidebar.cat.all": "All",
    "sidebar.cat.gate": "Gate / Entrance",
    "sidebar.cat.street": "Street / Intersection",
    "sidebar.cat.parking": "Parking Lot",
    "sidebar.cat.classroom": "Classroom",
    "sidebar.status": "Status",
    "sidebar.indexed": "Indexed",
    "sidebar.processing": "Processing",
    "sidebar.videos": "Videos",
    "sidebar.noVideos": "No matching videos found",
    "sidebar.delete": "Delete video",

    // Search Section
    "search.placeholder": "Search physical actions (e.g. 'cyclist riding', 'car turning left')...",
    "search.clearTooltip": "Clear query",
    "search.button": "Search Moments",
    "search.searching": "Searching...",
    "search.scope": "Scope",
    "search.scope.video": "This Video",
    "search.scope.folder": "This Folder",
    "search.scope.all": "All Videos",
    "search.mode": "Mode",
    "search.density": "Density",
    "search.density.4": "4 Frames (Fast - Base)",
    "search.density.8": "8 Frames (Recommended)",
    "search.density.12": "12 Frames (High Detail)",
    "search.density.16": "16 Frames (Dense Max)",
    "search.rebuildCaptions": "Rebuild Captions",
    "search.rebuilding": "Building...",
    "search.rebuildTooltip": "Re-run CapRL Q6 model with selected frame density",
    "search.cancel": "✕ Cancel",
    "search.cancelling": "Cancelling...",
    "search.cancelTooltip": "Cancel caption generation",
    "search.cameraTime": "Camera Time: All Day",
    "search.captionWarning": "Captions unavailable for this video. Automatically fell back to Fast Search.",
    "search.captionCancelled": "Caption generation cancelled. Automatically fell back to Fast Search.",
    "search.examples": "Example Queries",
    "search.recent": "Recent Searches",
    "search.tab.transit": "Transit",
    "search.tab.classroom": "Classroom",
    "search.tab.vehicle": "Vehicles",
    "search.tab.action": "Actions",

    // Player
    "player.heatmap": "Visual Relevance Heatmap",
    "player.camPreview": "CAM-03 · Live Feed",
    "player.cameraTime": "Camera Time",
    "player.exportClip": "Export This Clip",
    "player.exporting": "Exporting...",
    "player.stepBack": "Step backward 1 frame (,)",
    "player.stepForward": "Step forward 1 frame (.)",
    "player.rewind": "Rewind 5s (J)",
    "player.mute": "Mute",
    "player.unmute": "Unmute",
    "player.loop": "Loop localized moment (R)",
    "player.fullscreen": "Fullscreen",
    "player.customBoundary": "Adjusted locally (Custom boundary)",
    "player.resetBoundary": "Reset Boundary",
    "player.noVideo": "No Video Selected",
    "player.selectPrompt": "Select a video from the library or click '+ Ingest Video' to begin",
    "player.hideMask": "Hide SAM Mask",
    "player.showMask": "Show SAM Mask",

    // Results Panel
    "results.foundMoments": "Found Moments",
    "results.ragAI": "Video-RAG AI",
    "results.scenes": "Scene Captions",
    "results.found": "Found",
    "results.momentsCount": "moments",
    "results.searchIn": "Search",
    "results.inVideo": "in",
    "results.allActions": "all physical actions",
    "results.videoFallback": "video",
    "results.rankedBy": "Ranked by relevance",
    "results.noMoments": "No Moments Found",
    "results.noMomentsPrompt": "Type a natural language query above and click 'Search Moments'",
    "results.cameraTime": "Camera Time",
    "results.detectedMoment": "Detected moment",
    "results.copy": "Copy",
    "results.copied": "Copied!",
    "results.copyTooltip": "Copy timestamp",
    "results.context": "Context",
    "results.contextTooltip": "Expand window & view scene context",
    "results.contextWindow": "Context Window",
    "results.associatedScene": "Associated Scene",
    "results.openInScenes": "View in Scenes Tab",
    "results.sceneCaption": "Scene Action Caption",
    "results.noSceneCaption": "No caption available for this scene",
    "results.loadingContext": "Loading context...",
    "results.details": "Details",
    "results.export": "Export",
    "results.exporting": "Exporting...",
    "results.exportAll": "Export All Moments",
    "results.exportingAll": "Exporting all moments...",
    "results.saveSearch": "Save Search",
    "results.savedSearch": "Search Saved!",
    "results.savedSearches": "Saved Searches",
    "results.deleteSearchTooltip": "Delete saved search",

    // Settings Modal
    "settings.title": "Developer & System Telemetry Panel",
    "settings.subtitle": "Hardware metrics, Vector Storage stats, AI Model registry & execution logs",
    "settings.tab.preferences": "Preferences & Language",
    "settings.tab.hardware": "Hardware & GPU Acceleration",
    "settings.tab.storage": "LanceDB Vector Storage",
    "settings.tab.models": "AI Models & Pipeline Config",
    "settings.tab.logs": "Live Backend Logs",
    "settings.languageCard.title": "System Interface Language",
    "settings.languageCard.desc": "Choose between Thai (with technical terms preserved in English) or Full English",
    "settings.themeCard.title": "Display Theme",
    "settings.themeCard.desc": "Switch between comfortable Light and Dark interface modes",
    "settings.searchModeCard.title": "Default Search Mode",
    "settings.searchModeCard.desc": "Fast (SigLIP2 Pure Visual) or Accurate (SigLIP2 + CapRL Q6 Captions)",
    "settings.densityCard.title": "Default Keyframe Density",
    "settings.densityCard.desc": "Average keyframe count per scene for preview and captions",
    "settings.langThai": "Thai (TH)",
    "settings.langEnglish": "English (EN)",
    "settings.liveTelemetry": "Live Telemetry",
    "settings.autoRefresh": "Auto (3s)",
    "settings.paused": "Paused",
    "settings.manualRefresh": "Manual Refresh",

    // Modals
    "modal.deleteTitle": "Delete Video from System",
    "modal.deleteDesc": "Permanently remove video and vector indices",
    "modal.deleteConfirm": "Are you sure you want to delete video",
    "modal.deleteWarning": "⚠️ This will permanently remove SigLIP 2 visual vectors, CapRL Q6 captions, and raw video files from the system.",
    "modal.cancel": "Cancel",
    "modal.delete": "Confirm Delete",
    "modal.deleting": "Deleting...",
    "modal.uploadTitle": "Ingest Video into Library",
    "modal.shortcutsTitle": "Pro Studio Keyboard Shortcuts",
    "modal.playbackSection": "Playback & Shuttle",
    "modal.navSection": "Moment & Timeline Navigation",
    "modal.studioSection": "Studio Modes & Export",
    "modal.pressEsc": "Press ESC to close",
  },
};

const LanguageContext = createContext<LanguageContextType>({
  language: "th",
  setLanguage: () => {},
  t: (key: string, defaultText?: string) => defaultText || key,
});

export const LanguageProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [language, setLanguageState] = useState<Language>("th");

  useEffect(() => {
    try {
      const saved = localStorage.getItem("videomoment_lang") as Language;
      if (saved === "en" || saved === "th") {
        setLanguageState(saved);
      }
    } catch (_) {}
  }, []);

  const setLanguage = (lang: Language) => {
    setLanguageState(lang);
    try {
      localStorage.setItem("videomoment_lang", lang);
    } catch (_) {}
  };

  const t = (key: string, defaultText?: string): string => {
    return translations[language]?.[key] || defaultText || key;
  };

  return (
    <LanguageContext.Provider value={{ language, setLanguage, t }}>
      {children}
    </LanguageContext.Provider>
  );
};

export const useLanguage = () => useContext(LanguageContext);
