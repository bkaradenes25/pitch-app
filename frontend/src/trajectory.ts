/**
 * Pitch flight maths (no three.js, so it is easy to test). Units: feet and seconds.
 * MLB coordinates: x = catcher's right, y = toward the pitcher, z = up, origin at the back tip of the plate.
 * Position(t) = p0 + v0*t + a*t^2/2, with velocity and acceleration given at y = 50 ft.
 */
import type { Trajectory } from "./types";

export const PLATE_Y = 17 / 12; // front of the plate, where plate_x / plate_z are measured

export type Vec3 = [number, number, number];

/** Seconds until the ball reaches `yTarget` (the earlier positive root), or null if it never does. */
export function flightTime(p: Trajectory, yTarget: number = PLATE_Y): number | null {
  const a = 0.5 * p.ay;
  const b = p.vy0;
  const c = p.y0 - yTarget;
  let roots: number[];
  if (Math.abs(a) < 1e-12) {
    roots = Math.abs(b) < 1e-12 ? [] : [-c / b];
  } else {
    const disc = b * b - 4 * a * c;
    if (disc < 0) return null;
    const s = Math.sqrt(disc);
    roots = [(-b - s) / (2 * a), (-b + s) / (2 * a)];
  }
  const ok = roots.filter((t) => t > 1e-9);
  return ok.length ? Math.min(...ok) : null;
}

export function positionAt(p: Trajectory, t: number): Vec3 {
  return [
    p.x0 + p.vx0 * t + 0.5 * p.ax * t * t,
    p.y0 + p.vy0 * t + 0.5 * p.ay * t * t,
    p.z0 + p.vz0 * t + 0.5 * p.az * t * t,
  ];
}

/** `n + 1` points along the flight from y = 50 ft to the plate (empty if the numbers do not describe a pitch). */
export function samplePath(p: Trajectory, n = 40): Vec3[] {
  const T = flightTime(p);
  if (T === null) return [];
  return Array.from({ length: n + 1 }, (_, i) => positionAt(p, (T * i) / n));
}

/** Where the path crosses the front of the plate, as the app plots it. */
export function plateCrossing(p: Trajectory): { plate_x: number; plate_z: number } | null {
  const T = flightTime(p);
  if (T === null) return null;
  const [x, , z] = positionAt(p, T);
  return { plate_x: x, plate_z: z };
}
