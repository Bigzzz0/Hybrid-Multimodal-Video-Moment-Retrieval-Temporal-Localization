"use client";

import React, { useState, useEffect } from "react";
import {
  Sparkles,
  History,
  Compass,
  ArrowRight,
  X,
  SlidersHorizontal,
  Bookmark,
  Activity,
  Check,
  ChevronDown,
} from "lucide-react";
import { FilterCriteria } from "@/lib/types";

interface QueryAssistantProps {
  onSelectQuery: (query: string) => void;
  currentQuery: string;
  onFilterChange?: (filters: FilterCriteria) => void;
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
    name: "การสัญจร/เดินทาง (Transit & Movement)",
    icon: "🚴‍♂️",
    color: "from-cyan-500/20 to-indigo-500/20 text-cyan-300 border-cyan-500/30",
    queries: [
      { th: "คนปั่นจักรยาน", en: "bicycle cyclist riding" },
      { th: "คนเดินริมถนน", en: "pedestrian walking along sidewalk" },
      { th: "รถยนต์ขับผ่าน", en: "car driving on road" },
      { th: "คนข้ามถนน", en: "pedestrian crossing street" },
    ],
  },
  {
    id: "classroom",
    name: "ห้องเรียน/การสอน (Classroom & Lecture)",
    icon: "🏫",
    color: "from-purple-500/20 to-indigo-500/20 text-purple-300 border-purple-500/30",
    queries: [
      { th: "อาจารย์ยืนหน้าห้อง", en: "instructor standing front" },
      { th: "นักเรียนนั่งในห้องเรียน", en: "students sitting at desks" },
      { th: "ผู้บรรยายยืนสอน", en: "teacher lecturing at blackboard" },
      { th: "เด็กนักเรียนนั่งเรียน", en: "pupils seated at desks" },
    ],
  },
  {
    id: "vehicles",
    name: "รถยนต์/การเลี้ยว (Vehicles & Turns)",
    icon: "🚗",
    color: "from-blue-500/20 to-cyan-500/20 text-blue-300 border-blue-500/30",
    queries: [
      { th: "รถยนต์วิ่งผ่านถนน", en: "car vehicle moving on road" },
      { th: "รถยนต์แล่นผ่าน", en: "automobile driving past" },
      { th: "รถเลี้ยวตรงสี่แยก", en: "vehicle turning at corner" },
      { th: "รถยนต์สีขาวขับผ่าน", en: "white car moving on roadway" },
    ],
  },
  {
    id: "objects",
    name: "อากัปกิริยาบุคคล (Human Gestures)",
    icon: "🚶",
    color: "from-emerald-500/20 to-teal-500/20 text-emerald-300 border-emerald-500/30",
    queries: [
      { th: "คนกำลังเดิน", en: "person walking on street" },
      { th: "คนยืนหยุดนิ่ง", en: "person standing motionless" },
      { th: "คนยกมือ", en: "person raising hand" },
      { th: "คนหันหลัง", en: "person turning around" },
    ],
  },
];

export const QueryAssistant: React.FC<QueryAssistantProps> = ({
  onSelectQuery,
  currentQuery,
  onFilterChange,
}) => {
  const [activeTab, setActiveTab] = useState<string>("motion");
  const [recentSearches, setRecentSearches] = useState<string[]>([]);
  const [showFilterPopover, setShowFilterPopover] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(true);

  const [filters, setFilters] = useState<FilterCriteria>({
    minConfidence: 0.35,
    motionFilter: "all",
    minDurationSec: 1.0,
    maxDurationSec: 30.0,
  });

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

  const handleFilterUpdate = (partial: Partial<FilterCriteria>) => {
    const updated = { ...filters, ...partial };
    setFilters(updated);
    if (onFilterChange) {
      onFilterChange(updated);
    }
  };

  const activeCategory =
    ACTION_CATEGORIES.find((c) => c.id === activeTab) || ACTION_CATEGORIES[0];

  return (
    <div className="space-y-2.5 pt-1 relative">
      <button
        type="button"
        aria-expanded={showSuggestions}
        onClick={() => setShowSuggestions((previous) => !previous)}
        className="flex min-h-9 items-center gap-1.5 rounded-lg px-1 text-[11px] font-mono text-gray-400 hover:text-cyan-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400"
      >
        <ChevronDown className={`h-3.5 w-3.5 transition-transform ${showSuggestions ? "rotate-0" : "-rotate-90"}`} />
        คำค้นแนะนำและประวัติ
      </button>
      {showSuggestions && <div className="space-y-2.5">
      {/* Category Pills & Recent Searches Header */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        {/* Category Tabs */}
        <div className="flex items-center gap-1.5 overflow-x-auto pb-1 max-w-full">
          {ACTION_CATEGORIES.map((cat) => {
            const isCatActive = activeTab === cat.id;
            return (
              <button
                key={cat.id}
                type="button"
                onClick={() => setActiveTab(cat.id)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold transition-all flex-shrink-0 border ${
                  isCatActive
                    ? "bg-surface border-cyan-500 text-cyan-300 shadow-md shadow-cyan-500/10"
                    : "bg-surface/50 border-surfaceBorder text-gray-400 hover:text-gray-200 hover:border-gray-600"
                }`}
              >
                <span>{cat.icon}</span>
                <span>{cat.name}</span>
              </button>
            );
          })}
        </div>

        {/* Filter Popover Toggle Button */}
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setShowFilterPopover(!showFilterPopover)}
            className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl text-xs font-mono font-semibold transition-all border ${
              showFilterPopover || filters.minConfidence > 0.35 || filters.motionFilter !== "all"
                ? "bg-cyan-950/80 border-cyan-500 text-cyan-300 shadow-md"
                : "bg-surface border-surfaceBorder text-gray-400 hover:text-white"
            }`}
            title="Configure Precision Retrieval Filters"
          >
            <SlidersHorizontal className="w-3.5 h-3.5" />
            <span>Filters</span>
            {filters.motionFilter !== "all" && (
              <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
            )}
          </button>
        </div>
      </div>

      {/* Action Chips for Active Category */}
      <div className="flex flex-wrap gap-2">
        {activeCategory.queries.map((q, idx) => (
          <button
            key={idx}
            type="button"
            onClick={() => onSelectQuery(q.th)}
            className="group flex items-center gap-2 px-3 py-1.5 rounded-xl bg-surface/80 hover:bg-surface border border-surfaceBorder hover:border-cyan-500/40 text-xs text-gray-300 hover:text-white transition-all shadow-sm"
          >
            <Sparkles className="w-3 h-3 text-cyan-400 opacity-60 group-hover:opacity-100 transition-opacity" />
            <span>{q.th}</span>
            <span className="text-[10px] font-mono text-gray-500 group-hover:text-cyan-400/80">
              ({q.en})
            </span>
          </button>
        ))}
      </div>

      {/* Recent Searches Line (if any) */}
      {recentSearches.length > 0 && (
        <div className="flex items-center gap-2 text-xs text-gray-400 pt-1">
          <History className="w-3.5 h-3.5 text-gray-500 flex-shrink-0" />
          <span className="text-[11px] font-mono text-gray-500 flex-shrink-0">ค้นหาล่าสุด:</span>
          <div className="flex items-center gap-1.5 overflow-x-auto">
            {recentSearches.map((sq, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => onSelectQuery(sq)}
                className="px-2 py-0.5 rounded-lg bg-surface/60 hover:bg-surface border border-surfaceBorder text-[11px] text-gray-300 hover:text-cyan-300 transition-colors truncate max-w-[180px]"
              >
                {sq}
              </button>
            ))}
            <button
              type="button"
              onClick={clearRecents}
              className="text-gray-500 hover:text-gray-300 p-0.5 rounded"
              title="Clear Search History"
            >
              <X className="w-3 h-3" />
            </button>
          </div>
        </div>
      )}

      {/* Precision Filters Popover Modal */}
      {showFilterPopover && (
        <div className="absolute top-12 right-0 z-40 w-72 glass-panel-glow p-4 rounded-2xl shadow-2xl border border-cyan-500/50 space-y-3.5 animate-in fade-in zoom-in-95 duration-150">
          <div className="flex items-center justify-between border-b border-surfaceBorder pb-2">
            <span className="text-xs font-mono font-bold text-white flex items-center gap-1.5">
              <SlidersHorizontal className="w-3.5 h-3.5 text-cyan-400" />
              Precision Retrieval Filters
            </span>
            <button
              type="button"
              onClick={() => setShowFilterPopover(false)}
              className="p-1 text-gray-400 hover:text-white"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>

          {/* Min Confidence Slider */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="text-gray-400">Min Confidence Floor:</span>
              <span className="text-cyan-300 font-bold">{(filters.minConfidence * 100).toFixed(0)}%</span>
            </div>
            <input
              type="range"
              min="0.20"
              max="0.85"
              step="0.05"
              value={filters.minConfidence}
              onChange={(e) => handleFilterUpdate({ minConfidence: parseFloat(e.target.value) })}
              className="w-full accent-cyan-400 h-1.5 bg-gray-700 rounded-lg cursor-pointer"
            />
          </div>

          {/* Motion Filter Selector */}
          <div className="space-y-1.5">
            <span className="text-xs font-mono text-gray-400 block">Motion Dynamics Filter:</span>
            <div className="grid grid-cols-3 gap-1 text-[11px] font-mono">
              {(["all", "high", "static"] as const).map((mode) => (
                <button
                  key={mode}
                  type="button"
                  onClick={() => handleFilterUpdate({ motionFilter: mode })}
                  className={`py-1 rounded-lg border text-center transition-all ${
                    filters.motionFilter === mode
                      ? "bg-cyan-500/20 border-cyan-500 text-cyan-300 font-bold shadow-sm"
                      : "bg-surface border-surfaceBorder text-gray-400 hover:text-white"
                  }`}
                >
                  {mode === "all" ? "All" : mode === "high" ? "High Mot" : "Static"}
                </button>
              ))}
            </div>
          </div>

          <div className="pt-1 flex items-center justify-between text-[11px] font-mono">
            <button
              type="button"
              onClick={() => {
                handleFilterUpdate({ minConfidence: 0.35, motionFilter: "all" });
              }}
              className="text-gray-400 hover:text-white"
            >
              Reset Defaults
            </button>
            <button
              type="button"
              onClick={() => setShowFilterPopover(false)}
              className="px-2.5 py-1 rounded-lg bg-cyan-600 text-white font-semibold shadow-md"
            >
              Apply
            </button>
          </div>
        </div>
      )}
      </div>}
    </div>
  );
};
