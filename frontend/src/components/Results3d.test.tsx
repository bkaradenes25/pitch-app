import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { Results, type ResultsData } from "./Results";

vi.mock("./Pitch3D", () => ({ default: () => <div>pitch3d-mock</div> }));

const grid = [{ plate_x: 0, plate_z: 2.5, prob: 0.7 }];
const data: ResultsData = {
  used_default_arsenal: false,
  recommendations: [
    { pitch_type: "FF", success_prob: 0.7, n_samples: 30, locations_source: "pitcher",
      best_location: { plate_x: 0, plate_z: 2.5 }, best_cell_prob: 0.7, grid },
  ],
};

describe("Results 2D/3D toggle", () => {
  it("starts in 2D, loads the 3D view only when asked, and switches back", async () => {
    const { container } = render(<Results data={data} />);
    expect(container.querySelector("svg")).toBeInTheDocument();
    expect(screen.queryByText("pitch3d-mock")).not.toBeInTheDocument();
    fireEvent.click(screen.getByText("3D"));
    expect(await screen.findByText("pitch3d-mock")).toBeInTheDocument();
    expect(container.querySelector("svg")).not.toBeInTheDocument();
    fireEvent.click(screen.getByText("2D"));
    expect(container.querySelector("svg")).toBeInTheDocument();
  });
});
