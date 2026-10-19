import { describe, expect, it } from "vitest";
import { batterOptions, errMsg, findBatter, pct, pp, standFor, stuffLine, xwoba } from "./format";
import type { BatterListItem, Stuff } from "./types";

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

describe("batter picker helpers", () => {
  const batters: BatterListItem[] = [
    { batter_id: 1, name: "Aaron Judge", stand: "R", switch: false },
    { batter_id: 9, name: "Will Smith", stand: "R", switch: false },
    { batter_id: 10, name: "Will Smith", stand: "L", switch: false },
    { batter_id: 2, name: "Switch Guy", stand: "R", switch: true },
  ];
  const options = batterOptions(batters);

  it("adds the id to a label only when two batters share a name", () => {
    expect(options.map((o) => o.label)).toEqual(["Aaron Judge", "Will Smith (#9)", "Will Smith (#10)", "Switch Guy"]);
  });
  it("finds a batter by typed text, ignoring case and spaces", () => {
    expect(findBatter(options, "  aaron JUDGE ")?.batter_id).toBe(1);
    expect(findBatter(options, "will smith (#10)")?.batter_id).toBe(10);
  });
  it("finds nothing for empty, partial, or ambiguous text", () => {
    expect(findBatter(options, "")).toBeNull();
    expect(findBatter(options, "Aaron")).toBeNull();
    expect(findBatter(options, "Will Smith")).toBeNull();
  });
  it("picks the side of the plate: fixed for most hitters, opposite the pitcher's hand for switch hitters", () => {
    const [judge, , , sw] = batters;
    expect(standFor(judge ?? null, "L", "L")).toBe("R");
    expect(standFor(sw ?? null, "L", "L")).toBe("R");
    expect(standFor(sw ?? null, "R", "R")).toBe("L");
    expect(standFor(null, "R", "L")).toBe("L"); // no batter: leave the current choice alone
  });
});

describe("number and pitch-quality formatting", () => {
  it("formats xwOBA-style rates without the leading zero", () => {
    expect(xwoba(0.385)).toBe(".385");
    expect(xwoba(0.4)).toBe(".400");
    expect(xwoba(null)).toBe("–");
  });
  it("describes a pitch, skipping unknown parts", () => {
    const full: Stuff = { velo: 94.24, spin: 2310.4, ext: 6.5, ivb_in: 15.6, hb_in: -6, source: "pitcher" };
    expect(stuffLine(full)).toBe("94.2 mph · 2,310 rpm · 15.6 in vertical · -6.0 in horizontal");
    expect(stuffLine({ ...full, spin: null, hb_in: null })).toBe("94.2 mph · 15.6 in vertical");
    expect(stuffLine({ ...full, source: "league" })).toMatch(/\(league average for this pitch\)$/);
    expect(stuffLine({ velo: null, spin: null, ext: null, ivb_in: null, hb_in: null, source: "league" })).toBe("");
  });
});
