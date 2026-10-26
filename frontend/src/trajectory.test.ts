import { describe, expect, it } from "vitest";
import { flightTime, plateCrossing, PLATE_Y, positionAt, samplePath } from "./trajectory";
import type { Trajectory } from "./types";

const FF: Trajectory = { x0: -1.6, y0: 50, z0: 6, vx0: 4, vy0: -135, vz0: -4, ax: -10, ay: 27, az: -17 };

describe("flightTime", () => {
  it("puts the ball at the front of the plate when time is up", () => {
    const t = flightTime(FF)!;
    expect(t).toBeGreaterThan(0.35);
    expect(t).toBeLessThan(0.45);
    expect(positionAt(FF, t)[1]).toBeCloseTo(PLATE_Y, 9);
  });
  it("is null when the ball never gets there", () => {
    expect(flightTime({ ...FF, vy0: -10, ay: 1000 })).toBeNull();
    expect(flightTime({ ...FF, vy0: 0, ay: 0 })).toBeNull();
    expect(flightTime({ ...FF, vy0: 135, ay: 0 })).toBeNull();
  });
  it("handles zero acceleration", () => {
    expect(flightTime({ ...FF, ay: 0, vy0: -100 })).toBeCloseTo((50 - PLATE_Y) / 100, 9);
  });
});

describe("samplePath", () => {
  it("starts at the 50 ft point, ends at the plate, and moves steadily toward the catcher", () => {
    const pts = samplePath(FF, 20);
    expect(pts).toHaveLength(21);
    expect(pts[0]).toEqual([FF.x0, FF.y0, FF.z0]);
    expect(pts[20][1]).toBeCloseTo(PLATE_Y, 6);
    for (let i = 1; i < pts.length; i++) expect(pts[i][1]).toBeLessThan(pts[i - 1][1]);
  });
  it("is empty for numbers that are not a pitch", () => {
    expect(samplePath({ ...FF, vy0: 0, ay: 0 })).toEqual([]);
  });
});

describe("plateCrossing", () => {
  it("matches the last sampled point", () => {
    const pts = samplePath(FF, 10);
    const c = plateCrossing(FF)!;
    expect(c.plate_x).toBeCloseTo(pts[10][0], 9);
    expect(c.plate_z).toBeCloseTo(pts[10][2], 9);
  });
});
