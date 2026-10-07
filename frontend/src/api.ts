/** Backend base URL; override with VITE_API_URL at build time. */
export const API: string = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

export async function api<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(API + url, init);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return (await r.json()) as T;
}

export function post<T>(url: string, body: unknown): Promise<T> {
  return api<T>(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}
