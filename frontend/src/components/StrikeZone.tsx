import { pct } from "../format";
import type { PitchRec, PlotPitch } from "../types";

interface Plotted {
  plate_x: number;
  plate_z: number;
}

interface Props {
  rec: PitchRec;
  pitches?: PlotPitch[];
}

const W = 300;
const H = 360;
const sx = (x: number): number => ((x + 1.9) / 3.8) * W; // x range -1.9..1.9 ft (includes out-of-zone cells)
const sz = (z: number): number => ((4.4 - z) / 4.0) * H; // z range 0.4..4.4 ft
const CELL_W = (0.7 / 3.8) * W;
const CELL_H = (0.7 / 4.0) * H;

/** Heatmap: each cell is the model's success probability if the pitch is located there. */
export function StrikeZone({ rec, pitches = [] }: Props) {
  const probs = rec.grid.map((g) => g.prob);
  const lo = Math.min(...probs);
  const hi = Math.max(...probs);
  const plotted = pitches.filter((p): p is Plotted => p.plate_x != null && p.plate_z != null);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="zone">
      <rect
        className="zone-box"
        x={sx(-0.83)}
        y={sz(3.5)}
        width={sx(0.83) - sx(-0.83)}
        height={sz(1.5) - sz(3.5)}
      />
      {rec.grid.map((g, i) => {
        const t = hi === lo ? 0.5 : (g.prob - lo) / (hi - lo);
        const best = g.plate_x === rec.best_location.plate_x && g.plate_z === rec.best_location.plate_z;
        return (
          <g key={i}>
            <rect
              x={sx(g.plate_x) - CELL_W / 2}
              y={sz(g.plate_z) - CELL_H / 2}
              width={CELL_W}
              height={CELL_H}
              fill={`hsl(${140 * t} 70% 42%)`}
              opacity={0.35 + 0.5 * t}
              className={best ? "best" : ""}
            />
            <text x={sx(g.plate_x)} y={sz(g.plate_z) + 4} textAnchor="middle" className="cell-text">
              {pct(g.prob)}
            </text>
          </g>
        );
      })}
      {plotted.map((p, i) => (
        <g key={"p" + i}>
          <circle cx={sx(p.plate_x)} cy={sz(p.plate_z)} r="10" className="dot" />
          <text x={sx(p.plate_x)} y={sz(p.plate_z) + 4} textAnchor="middle" className="dot-text">
            {i + 1}
          </text>
        </g>
      ))}
    </svg>
  );
}
