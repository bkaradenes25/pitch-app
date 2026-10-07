import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import App from "./App";
import { Results, type ResultsData } from "./components/Results";
import { Scoreboard } from "./components/Scoreboard";
import { StrikeZone } from "./components/StrikeZone";
import type { GridCell, PitchRec, ScoreboardStats } from "./types";

const grid: GridCell[] = [
  { plate_x: -0.7, plate_z: 1.8, prob: 0.4 },
  { plate_x: 0, plate_z: 2.5, prob: 0.7 },
  { plate_x: 0.7, plate_z: 3.2, prob: 0.5 },
];
const rec = (pitch_type: string, p: number): PitchRec => ({
  pitch_type,
  success_prob: p,
  n_samples: 30,
  locations_source: "pitcher",
  best_location: { plate_x: 0, plate_z: 2.5 },
  best_cell_prob: 0.7,
  grid,
});
const data: ResultsData = { used_default_arsenal: false, recommendations: [rec("FF", 0.7), rec("CU", 0.6)] };

const sb: ScoreboardStats = {
  n: 113063, top1_pct: 0.2906, top3_pct: 0.7262, avg_rec_prob: 0.6823, avg_actual_prob: 0.6641,
  success_matched: 0.3622, success_unmatched: 0.3784, n_matched: 32860, n_unmatched: 80203,
};

describe("Scoreboard", () => {
  it("shows an empty state", () => {
    render(<Scoreboard sb={{ n: 0 }} title="Test" />);
    expect(screen.getByText("Nothing scored yet.")).toBeInTheDocument();
  });
  it("renders the metrics", () => {
    render(<Scoreboard sb={sb} title="Test" />);
    expect(screen.getByText("29.1%")).toBeInTheDocument();
    expect(screen.getByText("72.6%")).toBeInTheDocument();
    expect(screen.getByText(/Good outcome when pitcher matched/)).toBeInTheDocument();
  });
  it("hides outcome stats when they don't exist (live mode)", () => {
    render(<Scoreboard sb={sb} title="Live" outcomes={false} />);
    expect(screen.queryByText(/Good outcome/)).not.toBeInTheDocument();
  });
});

describe("StrikeZone", () => {
  it("draws one rect per grid cell plus the zone, highlights the best cell, and plots real pitches", () => {
    const { container } = render(
      <StrikeZone rec={rec("FF", 0.7)} pitches={[{ plate_x: 0.1, plate_z: 2.4 }, { plate_x: null, plate_z: null }]} />,
    );
    expect(container.querySelectorAll("rect").length).toBe(4);
    expect(container.querySelectorAll("rect.best").length).toBe(1);
    expect(container.querySelectorAll("circle").length).toBe(1); // the pitch without coordinates is skipped
  });
});

describe("Results", () => {
  it("recommends the top pitch and switches the heatmap when another bar is clicked", () => {
    render(<Results data={data} />);
    expect(screen.getByText("FF", { selector: ".hl" })).toBeInTheDocument();
    expect(screen.getByText(/FF location heatmap/)).toBeInTheDocument();
    fireEvent.click(screen.getByText("CU"));
    expect(screen.getByText(/CU location heatmap/)).toBeInTheDocument();
  });
  it("shows a loading state without data", () => {
    render(<Results data={null} />);
    expect(screen.getByText(/Loading/)).toBeInTheDocument();
  });
});

describe("App", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn((url: string) =>
        Promise.resolve({ ok: true, json: () => Promise.resolve(url.includes("/scoreboard") ? { n: 0 } : []) }),
      ),
    );
  });
  it("switches to the replay tab and loads held-out games", async () => {
    render(<App />);
    expect(screen.getByText(/Pitch Recommender/)).toBeInTheDocument();
    fireEvent.click(screen.getByText("Replay & Scoreboard"));
    expect(await screen.findByText("held-out game")).toBeInTheDocument();
    const calls = vi.mocked(fetch).mock.calls;
    expect(calls.some(([u]) => String(u).includes("/replay/games"))).toBe(true);
  });
});
