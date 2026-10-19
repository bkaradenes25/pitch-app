import type { BatterListItem, Hand, Stuff } from "./types";

export const pct = (p: number): string => `${(p * 100).toFixed(1)}%`;

/** Like pct, but shows a dash for missing values. */
export const pp = (x: number | null | undefined): string => (x == null ? "–" : pct(x));

export const errMsg = (e: unknown): string => (e instanceof Error ? e.message : String(e));

/** xwOBA-style rate without the leading zero: .385 */
export const xwoba = (x: number | null | undefined): string =>
  x == null ? "–" : x.toFixed(3).replace(/^0/, "");

export interface BatterOption {
  label: string;
  batter: BatterListItem;
}

/** Picker labels: the name, plus the id only when two batters share a name (e.g. two Will Smiths). */
export function batterOptions(batters: BatterListItem[]): BatterOption[] {
  const count = new Map<string, number>();
  for (const b of batters) count.set(b.name, (count.get(b.name) ?? 0) + 1);
  return batters.map((b) => ({
    label: (count.get(b.name) ?? 0) > 1 ? `${b.name} (#${b.batter_id})` : b.name,
    batter: b,
  }));
}

/** The batter whose label matches the typed text (ignoring case and surrounding spaces), if any. */
export function findBatter(options: BatterOption[], text: string): BatterListItem | null {
  const t = text.trim().toLowerCase();
  if (!t) return null;
  return options.find((o) => o.label.toLowerCase() === t)?.batter ?? null;
}

/** Side of the plate he hits from against this pitcher: switch hitters take the opposite side. */
export function standFor(batter: BatterListItem | null, pThrows: Hand, current: Hand): Hand {
  if (!batter) return current;
  if (batter.switch) return pThrows === "R" ? "L" : "R";
  return batter.stand;
}

/** "94.2 mph · 2,310 rpm · 15.6 in vertical · -6.0 in horizontal" (skips unknown parts). */
export function stuffLine(s: Stuff): string {
  const parts: string[] = [];
  if (s.velo != null) parts.push(`${s.velo.toFixed(1)} mph`);
  if (s.spin != null) parts.push(`${Math.round(s.spin).toLocaleString("en-US")} rpm`);
  if (s.ivb_in != null) parts.push(`${s.ivb_in.toFixed(1)} in vertical`);
  if (s.hb_in != null) parts.push(`${s.hb_in.toFixed(1)} in horizontal`);
  const line = parts.join(" · ");
  return s.source === "league" && line ? `${line} (league average for this pitch)` : line;
}
