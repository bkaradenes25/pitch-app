/**
 * Pure three.js scene builders for the 3D view. Nothing here touches a renderer or the DOM, so the
 * builders can be tested without WebGL. Units are feet.
 *
 * World mapping: (X, Y, Z) = (mlb_x, mlb_z, -mlb_y), so the catcher's camera sits at +Z looking
 * toward -Z, X is the catcher's right, Y is up, and the pitcher is at Z = -60.5.
 */
import * as THREE from "three";
import { flightTime, PLATE_Y, samplePath } from "./trajectory";
import type { GridCell, PlotPitch, Trajectory } from "./types";

export const ZONE = { halfWidth: 0.83, bottom: 1.5, top: 3.5 }; // feet, at the front of the plate
export const MOUND_Y = 60.5;
export const BALL_RADIUS = 0.12; // feet; a real baseball is ~0.12 ft in radius (2.9 in across)

export const toWorld = (x: number, y: number, z: number): [number, number, number] => [x, z, -y];

/** Probability (0..1 within the grid) to a red -> green colour, the same scale as the 2D heatmap. */
export function probColor(t: number): THREE.Color {
  const c = Math.min(1, Math.max(0, t));
  return new THREE.Color().setHSL((140 * c) / 360, 0.7, 0.42);
}

export const barDepth = (t: number): number => 0.12 + 1.3 * Math.min(1, Math.max(0, t));

const PITCH_COLORS: Record<string, number> = {
  FF: 0xe05252, SI: 0xe0884a, FC: 0xd9b84a, SL: 0x7ac74f, ST: 0x4fc7a8, SV: 0x4fa8c7,
  CU: 0x4f7ac7, KC: 0x6a4fc7, CH: 0xa84fc7, FS: 0xc74f9a, KN: 0x999999,
};
export const pitchColor = (type?: string | null): number => (type && PITCH_COLORS[type]) || 0xdddddd;

/** Smallest positive spacing between distinct values (the grid is regular, but do not assume its size). */
function spacing(values: number[], fallback: number): number {
  const u = [...new Set(values.map((v) => Math.round(v * 1000) / 1000))].sort((a, b) => a - b);
  let best = Infinity;
  for (let i = 1; i < u.length; i++) best = Math.min(best, u[i] - u[i - 1]);
  return Number.isFinite(best) && best > 0 ? best : fallback;
}

export interface CellInfo {
  plate_x: number;
  plate_z: number;
  prob: number;
  best: boolean;
}

/** One box per grid cell at the front of the plate, extruded toward the catcher; taller and greener = better. */
export function buildHeatmap(grid: GridCell[]): THREE.Group {
  const group = new THREE.Group();
  group.name = "heatmap";
  if (!grid.length) return group;
  const probs = grid.map((c) => c.prob);
  const lo = Math.min(...probs);
  const hi = Math.max(...probs);
  const norm = (p: number) => (hi > lo ? (p - lo) / (hi - lo) : 0.5);
  const w = spacing(grid.map((c) => c.plate_x), 0.3) * 0.92;
  const h = spacing(grid.map((c) => c.plate_z), 0.3) * 0.92;
  const bestProb = hi;
  for (const cell of grid) {
    const t = norm(cell.prob);
    const d = barDepth(t);
    const best = cell.prob === bestProb;
    const mesh = new THREE.Mesh(
      new THREE.BoxGeometry(w, h, d),
      new THREE.MeshStandardMaterial({ color: probColor(t), roughness: 0.6, metalness: 0.05 }),
    );
    const [x, y, z] = toWorld(cell.plate_x, PLATE_Y, cell.plate_z);
    mesh.position.set(x, y, z + d / 2); // back face on the plate plane, growing toward the catcher
    const info: CellInfo = { plate_x: cell.plate_x, plate_z: cell.plate_z, prob: cell.prob, best };
    mesh.userData = info;
    group.add(mesh);
    if (best) {
      const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(mesh.geometry),
        new THREE.LineBasicMaterial({ color: 0xffffff }),
      );
      edges.position.copy(mesh.position);
      group.add(edges);
    }
  }
  return group;
}

/** Home plate, mound, ground and the strike-zone frame. */
export function buildField(): THREE.Group {
  const group = new THREE.Group();
  group.name = "field";

  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(60, 130),
    new THREE.MeshStandardMaterial({ color: 0x24352a, roughness: 1 }),
  );
  ground.rotation.x = -Math.PI / 2;
  ground.position.set(0, -0.01, -MOUND_Y + 5);
  group.add(ground);

  const half = 8.5 / 12;
  const shape = new THREE.Shape();
  shape.moveTo(0, 0);
  shape.lineTo(half, 8.5 / 12);
  shape.lineTo(half, PLATE_Y);
  shape.lineTo(-half, PLATE_Y);
  shape.lineTo(-half, 8.5 / 12);
  shape.closePath();
  const plateGeo = new THREE.ShapeGeometry(shape);
  plateGeo.rotateX(-Math.PI / 2); // (x, y) -> (x, 0, -y), the same mapping as toWorld
  const plate = new THREE.Mesh(plateGeo, new THREE.MeshStandardMaterial({ color: 0xf2f2f2, roughness: 0.8 }));
  plate.position.y = 0.01;
  plate.name = "plate";
  group.add(plate);

  const mound = new THREE.Mesh(
    new THREE.CylinderGeometry(9, 9, 0.83, 40),
    new THREE.MeshStandardMaterial({ color: 0x8a6a45, roughness: 1 }),
  );
  mound.position.set(0, 0.83 / 2, -MOUND_Y);
  mound.name = "mound";
  group.add(mound);

  group.add(buildZoneFrame());
  return group;
}

export function buildZoneFrame(): THREE.Line {
  const { halfWidth: w, bottom, top } = ZONE;
  const z = -PLATE_Y;
  const pts = [
    new THREE.Vector3(-w, bottom, z),
    new THREE.Vector3(w, bottom, z),
    new THREE.Vector3(w, top, z),
    new THREE.Vector3(-w, top, z),
    new THREE.Vector3(-w, bottom, z),
  ];
  const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: 0xffffff }));
  line.name = "zone";
  return line;
}

export function buildTarget(plate_x: number, plate_z: number): THREE.Mesh {
  const [x, y, z] = toWorld(plate_x, PLATE_Y, plate_z);
  const m = new THREE.Mesh(
    new THREE.SphereGeometry(0.1, 16, 12),
    new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.9 }),
  );
  m.position.set(x, y, z);
  m.name = "target";
  return m;
}

export interface Flight {
  path: THREE.Vector3[]; // y = 50 ft -> plate, world coordinates
  duration: number; // seconds
  ball: THREE.Mesh;
}

/** Trail line plus ball for each pitch. Pitches with only a plate location get a marker at the plate. */
export function buildPitches(pitches: PlotPitch[]): { group: THREE.Group; flights: Flight[] } {
  const group = new THREE.Group();
  group.name = "pitches";
  const flights: Flight[] = [];
  for (const p of pitches) {
    const color = pitchColor(p.type);
    const ballMat = new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.35 });
    const path = p.traj ? pathOf(p.traj) : [];
    if (path.length > 1) {
      const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(path), new THREE.LineBasicMaterial({ color }));
      group.add(line);
      const ball = new THREE.Mesh(new THREE.SphereGeometry(BALL_RADIUS, 16, 12), ballMat);
      ball.position.copy(path[path.length - 1]);
      group.add(ball);
      flights.push({ path, duration: flightTime(p.traj!) ?? 0.4, ball });
    } else if (p.plate_x !== null && p.plate_z !== null) {
      const ball = new THREE.Mesh(new THREE.SphereGeometry(BALL_RADIUS, 16, 12), ballMat);
      const [x, y, z] = toWorld(p.plate_x, PLATE_Y, p.plate_z);
      ball.position.set(x, y, z);
      group.add(ball);
    }
  }
  return { group, flights };
}

function pathOf(t: Trajectory): THREE.Vector3[] {
  return samplePath(t, 40).map(([x, y, z]) => new THREE.Vector3(...toWorld(x, y, z)));
}

/** Position along a flight for progress 0..1 (linear in time, so slow motion is a uniform stretch). */
export function pointAlong(path: THREE.Vector3[], progress: number): THREE.Vector3 {
  const f = Math.min(1, Math.max(0, progress)) * (path.length - 1);
  const i = Math.min(path.length - 2, Math.floor(f));
  return path[i].clone().lerp(path[i + 1], f - i);
}

export const CAMERA_PRESETS = {
  catcher: { position: [0, 2.7, 6], target: [0, 2.4, -PLATE_Y] },
  pitcher: { position: [0, 4.5, -16], target: [0, 2.4, -PLATE_Y] },
  side: { position: [17, 3.5, -8], target: [0, 2.4, -8] },
} as const;
export type CameraPreset = keyof typeof CAMERA_PRESETS;

/** Free the GPU buffers of everything under `obj` (call before dropping a rebuilt group). */
export function disposeTree(obj: THREE.Object3D): void {
  obj.traverse((o: THREE.Object3D) => {
    const m = o as THREE.Mesh;
    m.geometry?.dispose();
    const mat = m.material as THREE.Material | THREE.Material[] | undefined;
    if (Array.isArray(mat)) mat.forEach((x) => x.dispose());
    else mat?.dispose();
  });
}
