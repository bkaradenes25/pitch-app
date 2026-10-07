import { describe, expect, it } from "vitest";
import { errMsg, pct, pp } from "./format";

describe("formatting helpers", () => {
  it("formats probabilities as percentages", () => expect(pct(0.3456)).toBe("34.6%"));
  it("shows a dash for missing values", () => {
    expect(pp(null)).toBe("–");
    expect(pp(undefined)).toBe("–");
    expect(pp(0.5)).toBe("50.0%");
  });
  it("extracts messages from errors and non-errors", () => {
    expect(errMsg(new Error("boom"))).toBe("boom");
    expect(errMsg("plain")).toBe("plain");
  });
});
