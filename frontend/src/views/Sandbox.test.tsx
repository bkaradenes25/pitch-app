import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { Sandbox } from "./Sandbox";
import type { BatterInfo, BatterListItem, Pitcher, Recommendation } from "../types";

const PITCHERS: Pitcher[] = [
  { pitcher_id: 1, name: "Ace Lefty", pitches: ["FF"], throws: "L" },
  { pitcher_id: 2, name: "Righty Rob", pitches: ["FF"], throws: "R" },
];
const BATTERS: BatterListItem[] = [
  { batter_id: 7, name: "Switch Guy", stand: "R", switch: true },
  { batter_id: 8, name: "Lefty Lou", stand: "L", switch: false },
  { batter_id: 9, name: "Will Smith", stand: "R", switch: false },
  { batter_id: 10, name: "Will Smith", stand: "L", switch: false },
];
const BATTER_INFO: BatterInfo = {
  batter_id: 7,
  name: "Switch Guy",
  known: true,
  stand: "R",
  switch: true,
  vs: { fastballs: { whiff: 0.18, xwoba_con: 0.42, swings: 300 } },
};

const recommendation = (batter: BatterInfo | null): Recommendation => ({
  pitcher_id: 1,
  pitcher_name: "Ace Lefty",
  used_default_arsenal: false,
  pitcher_cluster: 0,
  hitter_cluster: -1,
  count_bucket: "even",
  batter,
  recommendations: [
    {
      pitch_type: "FF",
      success_prob: 0.5,
      n_samples: 30,
      locations_source: "pitcher",
      best_location: { plate_x: 0, plate_z: 2.5 },
      best_cell_prob: 0.5,
      grid: [{ plate_x: 0, plate_z: 2.5, prob: 0.5 }],
      stuff: { velo: 94.2, spin: 2310, ext: 6.5, ivb_in: 15.6, hb_in: -6, source: "pitcher" },
    },
  ],
});

interface Posted {
  pitcher_id: number;
  batter_id?: number;
  state: { stand: string; p_throws: string };
}
let posts: Posted[] = [];
const last = (): Posted | undefined => posts[posts.length - 1];
const ok = (data: unknown) => Promise.resolve({ ok: true, json: () => Promise.resolve(data) });

beforeEach(() => {
  posts = [];
  vi.stubGlobal(
    "fetch",
    vi.fn((url: string, init?: RequestInit) => {
      const u = String(url);
      if (u.endsWith("/pitchers")) return ok(PITCHERS);
      if (u.endsWith("/batters")) return ok(BATTERS);
      if (u.endsWith("/recommend")) {
        const body = JSON.parse(String(init?.body)) as Posted;
        posts.push(body);
        return ok(recommendation(body.batter_id === 7 ? BATTER_INFO : null));
      }
      return ok([]);
    }),
  );
});

const pickBatter = (text: string) => fireEvent.change(screen.getByLabelText("batter"), { target: { value: text } });

describe("Sandbox batter picker", () => {
  it("starts with the first pitcher's throwing hand and no batter", async () => {
    render(<Sandbox />);
    await waitFor(() => expect(last()?.state.p_throws).toBe("L"));
    expect(last()?.batter_id).toBeUndefined();
  });

  it("sends the batter's id and bats a switch hitter from the opposite side of the pitcher", async () => {
    render(<Sandbox />);
    await waitFor(() => expect(posts.length).toBeGreaterThan(0));
    pickBatter("Switch Guy");
    await waitFor(() => expect(last()?.batter_id).toBe(7));
    expect(last()?.state.stand).toBe("R"); // pitcher throws left
  });

  it("shows the batter's profile and the pitch's velocity and spin", async () => {
    render(<Sandbox />);
    await waitFor(() => expect(posts.length).toBeGreaterThan(0));
    pickBatter("Switch Guy");
    expect(await screen.findByText(/against each pitch type/)).toBeInTheDocument();
    expect(screen.getByText("94.2 mph · 2,310 rpm · 15.6 in vertical · -6.0 in horizontal")).toBeInTheDocument();
    expect(screen.getByText("18.0%")).toBeInTheDocument();
    expect(screen.getByText(".420")).toBeInTheDocument();
  });

  it("flips a switch hitter's side when the pitcher changes hands, but not a normal hitter's", async () => {
    render(<Sandbox />);
    await waitFor(() => expect(posts.length).toBeGreaterThan(0));
    pickBatter("Switch Guy");
    await waitFor(() => expect(last()?.batter_id).toBe(7));
    fireEvent.change(screen.getByLabelText("pitcher"), { target: { value: "2" } }); // right-hander
    await waitFor(() => expect(last()?.state).toMatchObject({ p_throws: "R", stand: "L" }));

    pickBatter("Lefty Lou");
    await waitFor(() => expect(last()?.batter_id).toBe(8));
    expect(last()?.state.stand).toBe("L");
    fireEvent.change(screen.getByLabelText("pitcher"), { target: { value: "1" } });
    await waitFor(() => expect(last()?.state).toMatchObject({ p_throws: "L", stand: "L" }));
  });

  it("tells two batters with the same name apart by id, and warns about unmatched text", async () => {
    render(<Sandbox />);
    await waitFor(() => expect(posts.length).toBeGreaterThan(0));
    pickBatter("Will Smith");
    expect(await screen.findByText(/No batter called "Will Smith"/)).toBeInTheDocument();
    expect(last()?.batter_id).toBeUndefined();
    pickBatter("Will Smith (#10)");
    await waitFor(() => expect(last()?.batter_id).toBe(10));
    expect(screen.queryByText(/No batter called/)).not.toBeInTheDocument();
  });

  it("still works when the backend has no batter list", async () => {
    vi.mocked(fetch).mockImplementation((url) => {
      const u = String(url);
      if (u.endsWith("/batters")) return Promise.resolve({ ok: false, status: 404, statusText: "Not Found" } as Response);
      if (u.endsWith("/pitchers")) return ok(PITCHERS) as Promise<Response>;
      return ok(recommendation(null)) as Promise<Response>;
    });
    render(<Sandbox />);
    expect(await screen.findByText(/Recommended:/)).toBeInTheDocument();
    expect(screen.getByLabelText("batter")).toBeInTheDocument();
  });
});
