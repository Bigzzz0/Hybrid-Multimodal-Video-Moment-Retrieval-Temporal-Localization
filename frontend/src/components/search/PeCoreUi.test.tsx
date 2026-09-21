import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { SearchResultSummary } from "./SearchResultSummary";

describe("PE-Core retrieval UI contract", () => {
  it("shows the requested retrieval backend and model", () => {
    render(
      <SearchResultSummary
        query="person"
        count={1}
        profile="fast"
        latencyMs={42}
        calibrated={false}
        indexVersion="v2"
        retrievalBackend="pe_core_b16"
        retrievalModelId="PE-Core-B16-224"
      />
    );
    expect(screen.getByText("retrieval pe_core_b16")).toBeInTheDocument();
    expect(screen.getByText("PE-Core-B16-224")).toBeInTheDocument();
  });
});
