export const pct = (p: number): string => `${(p * 100).toFixed(1)}%`;

/** Like pct, but shows a dash for missing values. */
export const pp = (x: number | null | undefined): string => (x == null ? "–" : pct(x));

export const errMsg = (e: unknown): string => (e instanceof Error ? e.message : String(e));
