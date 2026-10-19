import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { BatterCard } from "./BatterCard";
import type { BatterInfo } from "../types";

const known: BatterInfo = {
  batter_id: 1,
  name: "Aaron Judge",
  known: true,
  stand: "R",
  switch: false,
  vs: {
    fastballs: { whiff: 0.2, xwoba_con: 0.45, swings: 410 },
    offspeed: { whiff: null, xwoba_con: null, swings: 0 },
  },
};

describe("BatterCard", () => {
  it("lists his rates for the pitch categories he has data for", () => {
    render(<BatterCard batter={known} />);
    expect(screen.getByText("Fastballs")).toBeInTheDocument();
    expect(screen.getByText("20.0%")).toBeInTheDocument();
    expect(screen.getByText(".450")).toBeInTheDocument();
    expect(screen.getByText("410")).toBeInTheDocument();
    expect(screen.getByText("Offspeed")).toBeInTheDocument();
    expect(screen.queryByText("Breaking balls")).not.toBeInTheDocument(); // no data for that category
  });

  it("shows dashes for rates that could not be computed", () => {
    render(<BatterCard batter={known} />);
    expect(screen.getAllByText("–")).toHaveLength(2);
  });

  it("explains that an unknown batter is treated as league average", () => {
    render(<BatterCard batter={{ ...known, known: false, name: "Batter 99", vs: {} }} />);
    expect(screen.getByText(/Batter 99: no profile/)).toBeInTheDocument();
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
  });
});
