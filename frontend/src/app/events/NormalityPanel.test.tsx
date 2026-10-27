import { afterEach, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import NormalityPanel from "./NormalityPanel";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

it("requires explicit normal-development confirmation before publishing a bank", async () => {
  const fetch = vi.fn(async (url: string, options?: RequestInit) => ({ ok: true, json: async () => {
    if (!options) return [{ id: "p", name: "Corridor", context: "People walk here", current_version: null }];
    return { version: "new-reference-version", reference_count: 2 };
  }}));
  vi.stubGlobal("fetch", fetch);
  render(<NormalityPanel videoId="v" ready busy={false} profileId="p" onProfile={() => {}} />);
  expect(await screen.findByText(/reference_missing/)).toBeInTheDocument();
  fireEvent.click(screen.getByText("เพิ่มช่วงปกติที่ตรวจแล้วจากวิดีโอนี้"));
  const button = screen.getByRole("button", { name: "เพิ่มตัวอย่างปกติ" });
  expect(button).toBeDisabled();
  fireEvent.click(screen.getByRole("checkbox"));
  expect(button).toBeEnabled(); fireEvent.click(button);
  await waitFor(() => expect(fetch).toHaveBeenCalledWith("/api/v1/events/profiles/p/references", expect.objectContaining({ method: "POST" })));
  const body = JSON.parse(String(fetch.mock.calls.find(c => c[1])?.[1]?.body));
  expect(body.ranges[0]).toMatchObject({ video_id: "v", split: "development", verified_normal: true });
});
