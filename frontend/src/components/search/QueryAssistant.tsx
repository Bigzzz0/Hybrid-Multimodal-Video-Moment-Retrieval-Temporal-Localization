"use client";

import React, { useState, useEffect } from "react";
import { Sparkles, History, Compass, ArrowRight, X } from "lucide-react";

interface QueryAssistantProps {
  onSelectQuery: (query: string) => void;
  currentQuery: string;
}

interface ActionCategory {
  id: string;
  name: string;
  icon: string;
  color: string;
  queries: { th: string; en: string }[];
}

const ACTION_CATEGORIES: ActionCategory[] = [
  {
    id: "motion",
    name: "การเคลื่อนไหว (Physical Actions)",
    icon: "🏃‍♂️",
    color: "from-cyan-500/20 to-indigo-500/20 text-cyan-300 border-cyan-500/30",
    queries: [
      { th: "คนเดินเลี้ยวซ้าย", en: "person walking left" },
      { th: "คนปั่นจักรยานบนทางเท้า", en: "person riding bicycle" },
      { th: "คนลุกขึ้นยืน", en: "person standing up" },
      { th: "คนข้ามถนน", en: "pedestrian crossing street" },
    ],
  },
  {
    id: "vehicles",
    name: "ยานพาหนะ (Vehicles & Roads)",
    icon: "🚗",
    color: "from-blue-500/20 to-cyan-500/20 text-blue-300 border-blue-500/30",
    queries: [
      { th: "รถยนต์เลี้ยวซ้ายตรงสี่แยก", en: "car turning left at intersection" },
      { th: "รถสีขาววิ่งผ่าน", en: "white car driving past" },
      { th: "จักรยานแซงรถยนต์", en: "bicycle passing car" },
      { th: "รถชะลอความเร็ว", en: "vehicle slowing down" },
    ],
  },
  {
    id: "classroom",
    name: "กิจกรรมในห้อง (Classroom & Social)",
    icon: "🏫",
    color: "from-purple-500/20 to-indigo-500/20 text-purple-300 border-purple-500/30",
    queries: [
      { th: "นักเรียนยกมือถามคำถาม", en: "student raising hand" },
      { th: "คนยืนบรรยายหน้าห้อง", en: "teacher lecturing at blackboard" },
      { th: "นักเรียนนั่งฟังในห้อง", en: "students sitting in classroom" },
      { th: "คนเขียนกระดาน", en: "person writing on board" },
    ],
  },
  {
    id: "objects",
    name: "การหยิบจับวัตถุ (Object Actions)",
    icon: "📦",
    color: "from-emerald-500/20 to-teal-500/20 text-emerald-300 border-emerald-500/30",
    queries: [
      { th: "คนหยิบขวดน้ำดื่ม", en: "person drinking water" },
      { th: "คนเปิดประตูเดินเข้าห้อง", en: "person opening door" },
      { th: "มือเอื้อมหยิบสิ่งของ", en: "hand reaching for object" },
      { th: "คนวางของลงบนโต๊ะ", en: "person placing item on table" },
    ],
  },
];

export const QueryAssistant: React.FC<QueryAssistantProps> = ({
  onSelectQuery,
  currentQuery,
}) => {
  const [activeTab, setActiveTab] = useState<string>("motion");
  const [recentSearches, setRecentSearches] = useState<string[]>([]);

  // Load recent searches from localStorage
  useEffect(() => {
    try {
      const saved = localStorage.getItem("videomoment_recent_queries");
      if (saved) {
        setRecentSearches(JSON.parse(saved).slice(0, 5));
      }
    } catch (e) {
      console.debug("Could not load recent searches:", e);
    }
  }, []);

  const clearRecents = () => {
    try {
      localStorage.removeItem("videomoment_recent_queries");
      setRecentSearches([]);
    } catch (e) {}
  };

  const activeCategory = ACTION_CATEGORIES.find((c) => c.id === activeTab) || ACTION_CATEGORIES[0];

  return (
    <div className="space-y-2.5 pt-1">
      {/* Category Pills & Recent Searches Header */}
      <div className="flex items-center justify-between gap-2 overflow-x-auto pb-0.5">
        <div className="flex items-center gap-1.5 flex-shrink-0">
          <span className="text-[11px] font-semibold text-gray-400 flex items-center gap-1">
            <Compass className="w-3.5 h-3.5 text-cyan-400" /> หมวดคำค้นหา:
          </span>
          {ACTION_CATEGORIES.map((cat) => (
            <button
              key={cat.id}
              type="button"
              onClick={() => setActiveTab(cat.id)}
              className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-all flex items-center gap-1 border ${
                activeTab === cat.id
                  ? "bg-cyan-950/80 border-cyan-500 text-cyan-300 shadow-sm"
                  : "bg-surface hover:bg-surfaceBorder text-gray-400 hover:text-gray-200 border-surfaceBorder"
              }`}
            >
              <span>{cat.icon}</span>
              <span className="hidden sm:inline">{cat.name.split(" ")[0]}</span>
            </button>
          ))}
        </div>

        {/* Hotkey Hint */}
        <span className="hidden md:flex items-center gap-1 text-[10px] font-mono text-gray-500 bg-surface px-2 py-0.5 rounded border border-surfaceBorder">
          <kbd className="text-gray-300 font-bold">Ctrl + K</kbd> ค้นหาทันที
        </span>
      </div>

      {/* Action Chips for Active Category */}
      <div className="flex flex-wrap items-center gap-2">
        {activeCategory.queries.map((item, idx) => (
          <button
            key={idx}
            type="button"
            onClick={() => onSelectQuery(item.th)}
            className="px-3 py-1 rounded-full bg-surface/80 hover:bg-surface border border-surfaceBorder hover:border-cyan-500/60 text-xs text-gray-300 hover:text-cyan-200 transition-all flex items-center gap-1.5 group shadow-sm"
          >
            <span>{item.th}</span>
            <span className="text-[10px] font-mono text-gray-500 group-hover:text-cyan-400 transition-colors">
              ({item.en})
            </span>
          </button>
        ))}
      </div>

      {/* Recent Searches Row */}
      {recentSearches.length > 0 && (
        <div className="flex items-center gap-2 pt-1 overflow-x-auto text-[11px] text-gray-400">
          <span className="flex items-center gap-1 text-gray-500 flex-shrink-0">
            <History className="w-3 h-3 text-gray-400" /> ค้นหาล่าสุด:
          </span>

          <div className="flex items-center gap-1.5 flex-1 min-w-0 overflow-x-auto">
            {recentSearches.map((rec, i) => (
              <button
                key={i}
                type="button"
                onClick={() => onSelectQuery(rec)}
                className="px-2 py-0.5 rounded bg-black/40 hover:bg-surface text-gray-300 text-[10px] font-mono border border-surfaceBorder hover:border-gray-500 transition-colors flex-shrink-0"
              >
                "{rec}"
              </button>
            ))}
          </div>

          <button
            type="button"
            onClick={clearRecents}
            title="ล้างประวัติการค้นหา"
            className="text-gray-500 hover:text-gray-300 p-0.5 rounded transition-colors flex-shrink-0"
          >
            <X className="w-3 h-3" />
          </button>
        </div>
      )}
    </div>
  );
};
