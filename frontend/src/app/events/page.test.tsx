import { afterEach, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor, cleanup } from "@testing-library/react";
import EventsPage from "./page";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

it("updates the selected evidence when a pending verification completes", async () => {
  let event = { id: "e", label: "ล้ม", kind: "falling", t_start: 0, t_end: 6,
    context_t_start: 0, context_t_end: 11, model_status: "pending", review: "pending",
    caption_full: "", evidence: [], audio_status: "no_audio", threshold_status: "experimental",
    preview_frame_path: "/preview", refinement_status: "bounded_window", verification: {} };
  vi.stubGlobal("fetch", vi.fn(async (url: string) => ({ ok: true, json: async () => {
    if (url.endsWith("/videos")) return [{ id: "v", name: "Test video", duration: 20,
      has_audio: false, caption_count: 0, index_status: "ready", completed_windows: 7, total_windows: 7 }];
    if (url.endsWith("/jobs")) return [];
    if (url.endsWith("/health")) return {};
    return [event];
  }})));
  render(<EventsPage />);
  fireEvent.click(await screen.findByRole("button", { name: /ล้ม pending/ }));
  expect(screen.getByText("ยังไม่มีข้อสรุปจากภาพ")).toBeInTheDocument();
  event = { ...event, model_status: "verified", verification: {
    verdict: { event_present: false, reason: "No upright-to-ground transition" },
    caption_status: "not_requested_negative",
  }};
  await waitFor(() => expect(screen.getByText("Qwen ไม่พบการกระทำจากภาพ")).toBeInTheDocument(), { timeout: 5000 });
  expect(screen.getByText(/จึงไม่สร้าง Caption เพิ่ม/)).toBeInTheDocument();
  expect(screen.queryByText("รอ Qwen ตรวจภาพ")).not.toBeInTheDocument();
  // A model-negative unit stays accessible without filling the review queue.
  expect(screen.queryByRole("button", { name: /ล้ม pending/ })).not.toBeInTheDocument();
  fireEvent.change(screen.getByLabelText("กรองผลคัดกรอง"), { target: { value: "negative" } });
  expect(screen.getByRole("button", { name: /ล้ม pending/ })).toBeInTheDocument();
  expect(screen.getByText(/จำนวนนี้ไม่ใช่จำนวนเหตุจริง/)).toBeInTheDocument();
});
