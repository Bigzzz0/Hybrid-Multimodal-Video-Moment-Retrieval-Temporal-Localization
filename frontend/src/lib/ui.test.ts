import { describe, expect, it } from "vitest";
import { formatScore, formatMomentTime, momentIdentity, uniqueWarnings } from "./ui";

describe("pure visual UI formatting", () => {
  it("never presents an uncalibrated score as a probability", () => {
    expect(formatScore(0.9, false)).toBe("Rank score 0.900");
    expect(formatScore(0.9, false)).not.toContain("%");
  });

  it("formats timestamps and stable moment identity deterministically", () => {
    expect(formatMomentTime(65.25)).toBe("01:05.3");
    expect(momentIdentity({ t_start: 1, t_end: 2, score: 0.5, occurrence_index: 2 })).toBe("2:1.000:2.000");
  });

  it("deduplicates warning codes while preserving order", () => {
    const warnings = uniqueWarnings(["calibration_missing", "calibration_missing", "verifier_timeout_fallback"]);
    expect(warnings.map((warning) => warning.code)).toEqual(["calibration_missing", "verifier_timeout_fallback"]);
  });
});
