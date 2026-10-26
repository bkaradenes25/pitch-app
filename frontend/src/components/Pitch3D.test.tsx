import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import Pitch3D from "./Pitch3D";
import type { PitchRec } from "../types";

const rec: PitchRec = {
  pitch_type: "FF", success_prob: 0.5, n_samples: 10, locations_source: "pitcher",
  best_location: { plate_x: 0, plate_z: 2.5 }, best_cell_prob: 0.6,
  grid: [{ plate_x: 0, plate_z: 2.5, prob: 0.6 }],
};

describe("Pitch3D", () => {
  it("explains itself instead of crashing when WebGL is not available (jsdom has none)", () => {
    render(<Pitch3D rec={rec} />);
    expect(screen.getByText(/needs WebGL/)).toBeInTheDocument();
  });
});
