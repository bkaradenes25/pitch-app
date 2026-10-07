import { afterEach, describe, expect, it, vi } from "vitest";
import { api, post } from "./api";

const respond = (ok: boolean, body: unknown) =>
  vi.fn(() =>
    Promise.resolve({ ok, status: ok ? 200 : 503, statusText: ok ? "OK" : "Unavailable", json: () => Promise.resolve(body) }),
  );

afterEach(() => vi.unstubAllGlobals());

describe("api helpers", () => {
  it("returns parsed JSON on success", async () => {
    vi.stubGlobal("fetch", respond(true, { ok: 1 }));
    expect(await api<{ ok: number }>("/health")).toEqual({ ok: 1 });
  });

  it("throws a readable error on HTTP failure", async () => {
    vi.stubGlobal("fetch", respond(false, {}));
    await expect(api("/replay/games")).rejects.toThrow("503 Unavailable");
  });

  it("post sends a JSON body with the right header", async () => {
    const fetchMock = respond(true, { run_id: 7 });
    vi.stubGlobal("fetch", fetchMock);
    await post("/runs", { source: "replay" });
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url.endsWith("/runs")).toBe(true);
    expect(init.method).toBe("POST");
    expect(init.body).toBe('{"source":"replay"}');
    expect(init.headers).toEqual({ "Content-Type": "application/json" });
  });
});
