import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { MomentScore } from "./MomentScore";
import { SearchWarningBanner } from "./SearchWarningBanner";
import { MomentEvidenceBreakdown } from "./MomentEvidenceBreakdown";
import { uniqueWarnings } from "@/lib/ui";

describe("search result UI contract", () => {
  it("renders rank score without a misleading confidence percent", () => {
    render(<MomentScore score={0.9} calibrated={false} />);
    expect(screen.getByText("Rank score 0.900")).toBeInTheDocument();
    expect(screen.queryByText(/90%/)).not.toBeInTheDocument();
  });

  it("renders calibrated confidence only when calibration is available", () => {
    render(<MomentScore score={0.9} calibrated />);
    expect(screen.getByText("90%")).toBeInTheDocument();
  });

  it("deduplicates and maps warnings into one actionable banner", () => {
    render(<SearchWarningBanner warnings={uniqueWarnings(["calibration_missing", "calibration_missing"])} />);
    expect(screen.getAllByText(/ยังไม่ได้ calibrate คะแนน/)).toHaveLength(1);
  });

  it("does not fabricate a verifier score in Fast mode", () => {
    render(<MomentEvidenceBreakdown breakdown={{ visual: 0.5, verifier: 0 }} showVerifier={false} />);
    expect(screen.getByText("—")).toBeInTheDocument();
  });
});
