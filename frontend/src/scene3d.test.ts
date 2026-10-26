import { describe, expect, it } from "vitest";
import * as THREE from "three";
import {
  barDepth, buildField, buildHeatmap, buildPitches, buildTarget, buildZoneFrame, pointAlong, probColor, toWorld, ZONE,
  type CellInfo,
} from "./scene3d";
import { PLATE_Y, plateCrossing } from "./trajectory";
import type { GridCell, Trajectory } from "./types";

const FF: Trajectory = { x0: -1.6, y0: 50, z0: 6, vx0: 4, vy0: -135, vz0: -4, ax: -10, ay: 27, az: -17 };
const grid: GridCell[] = [];
for (const x of [-0.6, 0, 0.6]) for (const z of [1.8, 2.5, 3.2]) grid.push({ plate_x: x, plate_z: z, prob: 0.3 + 0.1 * (x + 1) + 0.05 * z });

describe("toWorld", () => {
  it("puts the catcher's right on +X, up on +Y and the pitcher toward -Z", () => {
    expect(toWorld(1, 50, 3)).toEqual([1, 3, -50]);
    expect(toWorld(0, 0, 0)).toEqual([0, 0, -0]);
  });
});

describe("heatmap", () => {
  const group = buildHeatmap(grid);
  const cells = group.children.filter((c: THREE.Object3D) => (c.userData as CellInfo).prob !== undefined) as THREE.Mesh[];
  it("has one box per cell and exactly one outlined best cell", () => {
    expect(cells).toHaveLength(9);
    expect(cells.filter((c: THREE.Mesh) => (c.userData as CellInfo).best)).toHaveLength(1);
    expect(group.children.filter((c: THREE.Object3D) => c instanceof THREE.LineSegments)).toHaveLength(1);
  });
  it("makes better cells deeper and greener", () => {
    const sorted = [...cells].sort((a, b) => (a.userData as CellInfo).prob - (b.userData as CellInfo).prob);
    const depth = (m: THREE.Mesh) => (m.geometry as THREE.BoxGeometry).parameters.depth;
    expect(depth(sorted[8])).toBeGreaterThan(depth(sorted[0]));
    const hue = (m: THREE.Mesh) => ((m.material as THREE.MeshStandardMaterial).color.getHSL({ h: 0, s: 0, l: 0 }).h);
    expect(hue(sorted[8])).toBeGreaterThan(hue(sorted[0]));
  });
  it("sits on the plate plane and grows toward the catcher", () => {
    for (const m of cells) {
      const d = (m.geometry as THREE.BoxGeometry).parameters.depth;
      expect(m.position.z - d / 2).toBeCloseTo(-PLATE_Y, 9);
    }
  });
  it("copes with an empty grid and with all-equal probabilities", () => {
    expect(buildHeatmap([]).children).toHaveLength(0);
    expect(buildHeatmap(grid.map((c) => ({ ...c, prob: 0.4 }))).children.length).toBeGreaterThan(0);
  });
});

describe("colour and depth scales", () => {
  it("run from red to green and clamp", () => {
    expect(probColor(0).getHSL({ h: 0, s: 0, l: 0 }).h).toBeCloseTo(0);
    expect(probColor(1).getHSL({ h: 0, s: 0, l: 0 }).h).toBeCloseTo(140 / 360, 2);
    expect(barDepth(-1)).toBeCloseTo(0.12);
    expect(barDepth(5)).toBeCloseTo(1.42);
  });
});

describe("field", () => {
  it("has a plate on the ground, a mound and a zone frame the size of the strike zone", () => {
    const names = buildField().children.map((c: THREE.Object3D) => c.name);
    expect(names).toEqual(expect.arrayContaining(["plate", "mound", "zone"]));
    const box = new THREE.Box3().setFromObject(buildZoneFrame());
    expect(box.max.x).toBeCloseTo(ZONE.halfWidth);
    expect(box.min.y).toBeCloseTo(ZONE.bottom);
    expect(box.max.y).toBeCloseTo(ZONE.top);
  });
  it("lays the plate flat with its point at the back and its front edge toward the pitcher", () => {
    const plate = buildField().getObjectByName("plate") as THREE.Mesh;
    const box = new THREE.Box3().setFromObject(plate);
    expect(box.max.y - box.min.y).toBeLessThan(0.05);
    expect(box.min.z).toBeCloseTo(-PLATE_Y, 6); // front edge
    expect(box.max.z).toBeCloseTo(0, 6); // back tip
  });
  it("places the target at the recommended location", () => {
    const t = buildTarget(0.2, 2.6);
    expect(t.position.toArray()).toEqual([0.2, 2.6, -PLATE_Y]);
  });
});

describe("pitches", () => {
  it("draws a trail and ball that end where the recorded plate location is", () => {
    const plate = plateCrossing(FF)!;
    const { flights } = buildPitches([{ plate_x: plate.plate_x, plate_z: plate.plate_z, type: "FF", traj: FF }]);
    expect(flights).toHaveLength(1);
    const end = flights[0].path[flights[0].path.length - 1];
    expect(end.x).toBeCloseTo(plate.plate_x, 6);
    expect(end.y).toBeCloseTo(plate.plate_z, 6);
    expect(end.z).toBeCloseTo(-PLATE_Y, 6);
    expect(flights[0].path[0].z).toBeCloseTo(-50, 6);
    expect(flights[0].duration).toBeGreaterThan(0.3);
  });
  it("falls back to a marker at the plate when there is no flight path, and skips pitches with neither", () => {
    const { group, flights } = buildPitches([
      { plate_x: 0.1, plate_z: 2.4, type: "SL" },
      { plate_x: null, plate_z: null },
    ]);
    expect(flights).toHaveLength(0);
    expect(group.children).toHaveLength(1);
  });
  it("interpolates a point along the path", () => {
    const { flights } = buildPitches([{ plate_x: 0, plate_z: 0, traj: FF }]);
    const p = flights[0].path;
    expect(pointAlong(p, 0).equals(p[0])).toBe(true);
    expect(pointAlong(p, 1).distanceTo(p[p.length - 1])).toBeLessThan(1e-9);
    expect(pointAlong(p, 0.5).z).toBeGreaterThan(p[0].z);
  });
});
